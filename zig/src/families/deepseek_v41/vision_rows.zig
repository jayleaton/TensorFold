//! Rank zero encodes held image spans and splices them before the existing embedding all-sum.
const std = @import("std");
const cuda = @import("cuda");
const held = @import("vision/held.zig");
const tower_mod = @import("vision_tower.zig");
const Store = @import("vision/store.zig").Store;
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.dsv41);
pub const VBASE: u32 = 1 << 24;
pub const isVid = @import("image_opts.zig").isVid;
pub const hasImage = @import("image_opts.zig").hasImage;
pub fn checkWindow(ids: []const u32) !void {
    if (hasImage(ids)) return error.ImageInWindow;
}
const Span = struct { buf: cuda.DeviceBuffer, stamp: u64 };

pub const Rows = struct {
    gpa: Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    tower: *tower_mod.Tower,
    mutex: std.Io.Mutex = .init,
    store: Store,
    spans: std.AutoHashMapUnmanaged([32]u8, Span) = .empty,
    spans_churn: @import("map_churn.zig").Churn = .{},
    span_bytes: u64 = 0,
    cap: u64,
    clock: u64 = 0,
    patches: ?cuda.DeviceBuffer = null,
    pending: ?cuda.DeviceBuffer = null,

    pub fn init(gpa: Allocator, io: std.Io, d: *const cuda.Driver, tower: *tower_mod.Tower) Rows {
        const mb: u64 = if (std.c.getenv("TF_DSV41_VISION_CACHE_MB")) |v|
            std.fmt.parseInt(u64, std.mem.trim(u8, std.mem.span(v), " "), 10) catch 64
        else
            64;
        return .{ .gpa = gpa, .io = io, .d = d, .tower = tower, .cap = @min(mb, 1 << 30) << 20, .store = .{ .gpa = gpa } };
    }

    pub fn deinit(r: *Rows) void {
        var it = r.spans.valueIterator();
        while (it.next()) |s| s.buf.free();
        r.spans.deinit(r.gpa);
        r.store.deinit();
        if (r.patches) |*p| p.free();
        if (r.pending) |*p| p.free();
    }

    pub fn hold(r: *Rows, id: u64, h: *const held.Held) !void {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        try r.store.hold(id, h);
    }

    pub fn release(r: *Rows, id: u64) void {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        r.store.release(id);
    }

    /// The image's span rows on the device (the tower's output), encoded on `s` when not cached.
    fn spanOf(r: *Rows, s: cuda.Stream, img: *const held.HeldImage) !u64 {
        if (r.pending != null) return error.VisionPoisoned;
        r.clock +%= 1;
        if (r.spans.getPtr(img.digest)) |e| {
            e.stamp = r.clock;
            return e.buf.ptr;
        }
        const D: usize = r.tower.shape.out;
        const bytes = img.n_vids * D * 2;
        // the cache's oldest spans out first (the stream drained: a queued copy may still read one)
        if (r.span_bytes + bytes > r.cap and r.spans.count() > 0) try s.synchronize();
        while (r.span_bytes + bytes > r.cap and r.spans.count() > 0) {
            var oldest: ?[32]u8 = null;
            var best: u64 = std.math.maxInt(u64);
            var it = r.spans.iterator();
            while (it.next()) |e| if (e.value_ptr.stamp < best) {
                best = e.value_ptr.stamp;
                oldest = e.key_ptr.*;
            };
            var gone = r.spans.fetchRemove(oldest.?).?.value;
            r.spans_churn.dropped([32]u8, &r.spans);
            r.span_bytes -= gone.buf.len;
            gone.buf.free();
        }
        try s.synchronize();
        const np: usize = @as(usize, img.vit_h) * img.vit_w * 3 * 14 * 14 * 2;
        if (r.patches == null or r.patches.?.len < np) {
            if (r.patches) |*p| {
                try s.synchronize();
                p.free();
            }
            r.patches = null;
            r.patches = try cuda.DeviceBuffer.alloc(r.d, np);
        }
        try r.patches.?.upload(0, std.mem.sliceAsBytes(img.patches[0 .. np / 2]));
        const out = try cuda.DeviceBuffer.alloc(r.d, bytes);
        r.pending = out;
        // Finish before either source ownership or failed output allocation can be released.
        try r.tower.span(s, img, r.patches.?.ptr, out.ptr);
        try s.synchronize();
        try r.spans.put(r.gpa, img.digest, .{ .buf = out, .stamp = r.clock });
        r.pending = null;
        r.span_bytes += bytes;
        return out.ptr;
    }

    /// ``vision.fill``: the window's image positions' rows into `rows` (bf16 [ids.len, D], rank 0's embedding
    pub fn fill(r: *Rows, s: cuda.Stream, ids: []const u32, rows: u64, D: usize) !void {
        if (D != r.tower.shape.out) return error.VisionWidth;
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        var i: usize = 0;
        while (i < ids.len) {
            if (!isVid(ids[i])) {
                i += 1;
                continue;
            }
            const loc = r.store.where.get(ids[i]) orelse {
                log.err("image rows not held for virtual id {d} (position {d})", .{ ids[i], i });
                return error.ImageRowsNotHeld;
            };
            var j = i + 1;
            while (j < ids.len and loc.idx + (j - i) < loc.img.n_vids and ids[j] == loc.img.vids[loc.idx + (j - i)]) j += 1;
            const base = try r.spanOf(s, loc.img);
            const row = D * 2;
            try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(rows + i * row, base + loc.idx * row, (j - i) * row, s.handle), "cuMemcpyDtoDAsync");
            i = j;
        }
    }
};

const fwd = @import("forward.zig");
const run = @import("run.zig");
const calls = @import("block_calls.zig");
fn fillHook(v: *anyopaque, s: cuda.Stream, ids: []const u32, rows: u64, D: usize) !void {
    const r: *Rows = @ptrCast(@alignCast(v));
    try r.fill(s, ids, rows, D);
}
pub const hooks: @import("forward_hooks.zig").Vision = .{
    .fill = fillHook,
    .checkWindow = checkWindow,
    .keep = keep,
    .split = splitRows,
};
/// Glue image_split (x [n, D] -> image rows, text rows) / image_merge (the two partials -> [n, D]): rows by the
pub fn splitRows(f: *fwd.Forward, r: *run.Runner, c: *const calls.Call, merge: bool) !void {
    const whole = if (merge) c.args[2].arg.t else c.args[0].arg.t;
    const img = if (merge) c.args[0].arg.t else c.args[1].arg.t;
    const txt = if (merge) c.args[1].arg.t else c.args[2].arg.t;
    const n = f.ids.len;
    if (@as(usize, @intCast(whole.shape[0])) != n) return error.BadGlue;
    const row: usize = @as(usize, @intCast(whole.shape[1])) * 2;
    const wa = try r.tensorAddr(whole);
    const ia = try r.tensorAddr(img);
    const ta: u64 = if (txt.shape[0] > 0) try r.tensorAddr(txt) else 0;
    var at: usize = 0;
    var ni: usize = 0;
    var nt: usize = 0;
    while (at < n) {
        const image = isVid(f.ids[at]);
        var end = at + 1;
        while (end < n and isVid(f.ids[end]) == image) end += 1;
        const k = end - at;
        const part = if (image) ia + ni * row else ta + nt * row;
        const dst = if (merge) wa + at * row else part;
        const src = if (merge) part else wa + at * row;
        try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(dst, src, k * row, r.stream.handle), "cuMemcpyDtoDAsync");
        if (image) ni += k else nt += k;
        at = end;
    }
    if (ni != @as(usize, @intCast(img.shape[0])) or nt != @as(usize, @intCast(@max(txt.shape[0], 0)))) return error.BadGlue;
}

/// Glue "image_keep" (vision.window's ``keep``): fp32 [n], 0 at image positions, 1 elsewhere.
pub fn keep(f: *fwd.Forward, r: *run.Runner, c: *const calls.Call) !void {
    const t = c.args[0].arg.t;
    const n = f.ids.len;
    if (@as(usize, @intCast(t.shape[0])) != n) return error.BadGlue;
    const host = try f.gpa.alloc(f32, n);
    defer f.gpa.free(host);
    for (f.ids, host) |id, *k| k.* = if (isVid(id)) 0.0 else 1.0;
    try r.stream.synchronize(); // the previous segment's Engram has read the buffer
    try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = try r.tensorAddr(t), .len = 4 * n }, 0, std.mem.sliceAsBytes(host));
}
