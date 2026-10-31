//! TF_DSV41_KV_MIRROR on the device: a mirror tensor a KV source, the fixed mirror table, a restored slot's fill.

const std = @import("std");
const cuda = @import("cuda");
const kv = @import("kv");
const buffers = @import("buffers.zig");
const run = @import("run.zig");
const Kv = @import("cuda_kv.zig").Kv;

/// log2 of a family's rows a page
fn shift(f: kv.sessions.Family, page: u32) u32 {
    return std.math.log2_int(u32, f.pageRows(page));
}

pub const Mirror = struct {
    gpa: std.mem.Allocator,
    positions: u64,
    geo: kv.mirror.Geometry,
    /// a tensor a KV source (the layout's order), zeroed: every slot's run of pages, then the discard page
    tensors: []cuda.DeviceBuffer,
    /// int32 [slots, pts] ("s.kv.mt"), written once at boot
    table: cuda.DeviceBuffer,
    /// a fill's row lists (a source's at i x stride u32s) and one source's send rows
    phys: cuda.DeviceBuffer,
    send: cuda.DeviceBuffer,
    stride: u32,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, k: *Kv, positions: u64) !Mirror {
        const page = k.pool.page;
        const geo = kv.mirror.Geometry.of(positions, page, @intCast(k.slots.len), k.world);
        const fams = k.layout.families();
        const ns = k.layout.nsources;
        const stream = k.runner.stream;
        const tensors = try gpa.alloc(cuda.DeviceBuffer, ns);
        errdefer gpa.free(tensors);
        var made: usize = 0;
        errdefer for (tensors[0..made]) |*b| b.free();
        var stride: u32 = 0;
        var rb: u32 = 0;
        for (k.layout.sources[0..ns], tensors) |L, *t| {
            const f = fams[k.layout.compOf(L).?];
            t.* = try cuda.DeviceBuffer.alloc(d, @intCast(@as(u64, geo.tensorPages()) * f.pageBytes(page)));
            made += 1;
            try t.fill8(0, stream.handle);
            stride = @max(stride, geo.fillRows(shift(f, page)));
            rb = @max(rb, f.row_bytes);
        }
        const host = try gpa.alloc(i32, @as(usize, geo.slots) * k.slot.max_pages);
        defer gpa.free(host);
        geo.table(k.slot.max_pages, host);
        var table = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(host));
        errdefer table.free();
        var phys = try cuda.DeviceBuffer.alloc(d, @max(256, 4 * @as(usize, ns) * stride));
        errdefer phys.free();
        var send = try cuda.DeviceBuffer.alloc(d, @max(256, @as(usize, stride) * rb));
        errdefer send.free();
        // the fills ran on the compute stream (as DevicePool.init): done before any window reads the mirror
        try stream.synchronize();
        return .{ .gpa = gpa, .positions = positions, .geo = geo, .tensors = tensors, .table = table, .phys = phys, .send = send, .stride = stride };
    }

    pub fn deinit(m: *Mirror) void {
        for (m.tensors) |*b| b.free();
        m.gpa.free(m.tensors);
        m.table.free();
        m.phys.free();
        m.send.free();
    }

    /// Device bytes on this rank (the boot log's figure).
    pub fn bytes(m: *const Mirror) u64 {
        var b: u64 = m.table.len + m.phys.len + m.send.len;
        for (m.tensors) |t| b += t.len;
        return b;
    }

    /// The window roles of the mirror ("s.kv.mc.L<src>", "s.kv.mt") at its buffers, as Kv.bind binds the pool's.
    pub fn bind(m: *const Mirror, k: *Kv, r: *run.Runner, plan: *const buffers.Plan) !void {
        var nb: [64]u8 = undefined;
        for (k.layout.sources[0..k.layout.nsources], m.tensors) |L, t| try k.external(r, plan, try std.fmt.bufPrint(&nb, "s.kv.mc.L{d}", .{L}), t);
        try k.external(r, plan, "s.kv.mt", m.table);
    }

    /// The current slot's run from its owners' stored pages (a restore): a gather and all-gather a source, all ranks.
    pub fn fill(m: *Mirror, k: *Kv) !void {
        const r = k.runner;
        const page = k.pool.page;
        const fams = k.layout.families();
        const ns = k.layout.nsources;
        const local = try m.gpa.alloc(u32, m.geo.pages);
        defer m.gpa.free(local);
        for (local, 0..) |*x, p| x.* = k.slot.localTableAt(@intCast(p));
        const host = try m.gpa.alloc(u32, @as(usize, ns) * m.stride);
        defer m.gpa.free(host);
        @memset(host, 0);
        for (k.layout.sources[0..ns], 0..) |L, i| {
            const sh = shift(fams[k.layout.compOf(L).?], page);
            m.geo.fillPhys(k.rank, sh, local, host[i * m.stride ..][0..m.geo.fillRows(sh)]);
        }
        // a restore is outside any round: drained, so no earlier fill's gather still reads the lists
        try r.stream.synchronize();
        try m.phys.upload(0, std.mem.sliceAsBytes(host));
        for (k.layout.sources[0..ns], m.tensors, 0..) |L, t, i| {
            const c = k.layout.compOf(L).?;
            const f = fams[c];
            const at = @as(u64, k.slot.id) * m.geo.pages * f.pageBytes(page);
            try k.x.unionBlock(k.dev.tensors[c].ptr, f.row_bytes, m.phys.ptr + 4 * i * m.stride, m.geo.fillRows(shift(f, page)), m.send.ptr, t.ptr + at, r.stream.handle);
        }
    }
};
