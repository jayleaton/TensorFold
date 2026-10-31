//! TF_DSV41_ROT_FOLD bit 1: split_rot_kernel against split_kernel + wo_a's seg_rot_in, byte for byte (synthetic).

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const dense = dsv41.dense;
const ops = dsv41.ops;
const attn = ops.attn;
const ar = dsv41.attn_rot;
const Gpu = check.Gpu;

const Rng = struct {
    s: u64,
    fn next(r: *Rng) u64 {
        r.s ^= r.s << 13;
        r.s ^= r.s >> 7;
        r.s ^= r.s << 17;
        return r.s;
    }
    fn unit(r: *Rng) f32 {
        return @as(f32, @floatFromInt(r.next() >> 40)) / @as(f32, 1 << 24);
    }
    fn normalish(r: *Rng) f32 {
        return (r.unit() + r.unit() + r.unit() - 1.5) * 2;
    }
};

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn upload(d: *const cuda.Driver, data: []const u8) !cuda.DeviceBuffer {
    return cuda.DeviceBuffer.fromHost(d, data);
}

fn zeroed(d: *const cuda.Driver, n: usize) !cuda.DeviceBuffer {
    const b = try cuda.DeviceBuffer.alloc(d, @max(n, 4));
    try b.fill8(0, null);
    return b;
}

fn bytes(gpa: std.mem.Allocator, b: cuda.DeviceBuffer, n: usize) ![]u8 {
    const x = try gpa.alloc(u8, n);
    errdefer gpa.free(x);
    try b.download(0, x);
    return x;
}

const ring = 256; // the SWA ring (rows a slot); row mode's single slot 0
const comp_rows = 1024; // compressed rows the selection points into
const ts = 512; // selection columns a row (index_topk)
const pos0 = 300; // row 0's position: every row's window holds 128 keys
const rope_rows = 1024;

/// KV rows: 448 e4m3 bytes (no NaN encoding), then 64 bf16 rope values; `hot`: rope values far past fp16 once rotated.
fn kvRows(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, rows: usize, hot: bool) ![2]cuda.DeviceBuffer {
    const v = try gpa.alloc(u8, rows * 576);
    defer gpa.free(v);
    const s = try gpa.alloc(u8, rows * 8);
    defer gpa.free(s);
    for (0..rows) |r| {
        for (v[r * 576 ..][0..448]) |*b| b.* = @as(u8, @truncate(rng.next())) & 0xbf;
        for (0..64) |i| std.mem.writeInt(u16, v[r * 576 + 448 + 2 * i ..][0..2], bf16(rng.normalish() * @as(f32, if (hot) 3000 else 0.5)), .little);
        for (s[r * 8 ..][0..8]) |*b| b.* = 120 + @as(u8, @truncate(rng.next() % 8)); // 2^-7 .. 2^0
    }
    return .{ try upload(d, v), try upload(d, s) };
}

fn halves(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, n: usize, lo: f32, hi: f32, hot: bool) !cuda.DeviceBuffer {
    const h = try gpa.alloc(f16, n);
    defer gpa.free(h);
    for (h) |*v| {
        v.* = @floatCast(lo + (hi - lo) * rng.unit());
        if (hot and rng.next() % 97 == 0) v.* = if (rng.next() & 1 == 0) std.math.inf(f16) else -std.math.inf(f16);
    }
    return upload(d, std.mem.sliceAsBytes(h));
}

fn floats(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, n: usize, scale: f32) !cuda.DeviceBuffer {
    const h = try gpa.alloc(f32, n);
    defer gpa.free(h);
    for (h) |*v| v.* = rng.normalish() * scale;
    return upload(d, std.mem.sliceAsBytes(h));
}

/// A rank's attention: `heads` heads, wo_a in `groups` groups; `hot`: fp16 overflow and infinite suh in wo_a's rows.
const Shape = struct { heads: usize, groups: usize, hot: bool = false };

const shapes = [_]Shape{
    .{ .heads = 32, .groups = 4 }, // TP2's rank
    .{ .heads = 64, .groups = 8 }, // TP1
    .{ .heads = 32, .groups = 4, .hot = true },
};

fn runShape(gpu: Gpu, o: ops.Ops, lin: dense.Ops, stream: cuda.Stream, rng: *Rng, sh: Shape, rows: []const usize) !usize {
    const gpa = gpu.gpa;
    const d = gpu.d;
    const H = sh.heads;
    const gk = H * ar.head_dim / sh.groups;
    const max_r: usize = 64;
    var swa = try kvRows(gpa, d, rng, ring, sh.hot);
    defer for (&swa) |*b| b.free();
    var comp = try kvRows(gpa, d, rng, comp_rows, sh.hot);
    defer for (&comp) |*b| b.free();
    var q = blk: {
        const h = try gpa.alloc(u16, max_r * H * 512);
        defer gpa.free(h);
        for (h) |*v| v.* = bf16(rng.normalish() * 0.5);
        break :blk try upload(d, std.mem.sliceAsBytes(h));
    };
    defer q.free();
    var tok = blk: {
        const h = try gpa.alloc(i32, max_r * ts);
        defer gpa.free(h);
        for (h) |*v| v.* = if (rng.next() % 16 == 0) -1 else @intCast(rng.next() % comp_rows);
        break :blk try upload(d, std.mem.sliceAsBytes(h));
    };
    defer tok.free();
    var cnt = blk: {
        var h: [max_r]i32 = undefined;
        for (&h, 0..) |*v, r| v.* = ([_]i32{ 512, 0, 77, 300 })[r % 4]; // 4, 0, 1 and 3 compressed chunks
        break :blk try upload(d, std.mem.sliceAsBytes(h[0..]));
    };
    defer cnt.free();
    const at0 = [_]i32{pos0};
    var pos32 = try upload(d, std.mem.sliceAsBytes(at0[0..]));
    defer pos32.free();
    var pos64 = blk: {
        var h: [max_r]i64 = undefined;
        for (&h, 0..) |*v, r| v.* = @intCast(pos0 + r);
        break :blk try upload(d, std.mem.sliceAsBytes(h[0..]));
    };
    defer pos64.free();
    var slots = try zeroed(d, max_r * 8);
    defer slots.free();
    var lo = try zeroed(d, max_r * 4);
    defer lo.free();
    var sink = try floats(gpa, d, rng, H, 1.0);
    defer sink.free();
    var cs = blk: {
        const h = try gpa.alloc(f32, rope_rows * 64);
        defer gpa.free(h);
        for (h) |*v| v.* = rng.unit() * 2 - 1;
        break :blk try upload(d, std.mem.sliceAsBytes(h));
    };
    defer cs.free();
    var po = try zeroed(d, attn.NCH * max_r * H * 512 * 4);
    defer po.free();
    var pm = try zeroed(d, attn.NCH * max_r * H * 4);
    defer pm.free();
    var pl = try zeroed(d, attn.NCH * max_r * H * 4);
    defer pl.free();
    var ticket = try zeroed(d, max_r * (H / attn.BMQ) * attn.MAXP * 4);
    defer ticket.free();
    var suh: [ar.max_groups]cuda.DeviceBuffer = undefined;
    const lim: f32 = if (sh.hot) 64.0 else 2.0; // hot: wo_a's suh past fp16 once rotated, and some infinities
    for (suh[0..sh.groups]) |*s| s.* = try halves(gpa, d, rng, gk, -lim, lim, sh.hot);
    defer for (suh[0..sh.groups]) |*s| s.free();
    var outs: [4]cuda.DeviceBuffer = undefined; // o and wo_a's rows: H's, then the fold's
    for (&outs) |*b| b.* = try zeroed(d, max_r * H * 512 * 2);
    defer for (&outs) |*b| b.free();
    var name: [160]u8 = undefined;
    var n_cases: usize = 0;
    for (rows) |R| for ([_]u32{ 1, 4 }) |split| for ([_]bool{ false, true }) |row_mode| {
        for (&outs) |*b| try b.fill8(0, null);
        var sp: [ar.max_groups]u64 = undefined;
        var offs: [ar.max_groups]usize = undefined;
        for (0..sh.groups) |i| {
            sp[i] = suh[i].ptr;
            offs[i] = i * R * gk; // fusedRot's L.o.xh layout
        }
        for ([_]bool{ false, true }) |fold| {
            const out = outs[if (fold) 2 else 0];
            const xh = outs[if (fold) 3 else 1];
            const a: attn.Args = .{
                .q = q.ptr,
                .has_comp = 1,
                .cv = comp[0].ptr,
                .csc = comp[1].ptr,
                .cvs = 576,
                .css = 8,
                .tok = tok.ptr,
                .ts = ts,
                .cnt = cnt.ptr,
                .sv = swa[0].ptr,
                .ssc = swa[1].ptr,
                .svs = 576,
                .sss = 8,
                .lo = lo.ptr,
                .pos = if (row_mode) pos64.ptr else pos32.ptr,
                .sl = if (row_mode) slots.ptr else 0,
                .sink = sink.ptr,
                .cs = cs.ptr,
                .csst = 64,
                .po = po.ptr,
                .pm = pm.ptr,
                .pl = pl.ptr,
                .out = out.ptr,
                .ticket = ticket.ptr,
                .R = @intCast(R),
                .H = @intCast(H),
                .ring = ring,
                .qrope = 1,
            };
            if (fold) {
                const ro = try ar.rotIn(H, sp[0..sh.groups], gk, xh.ptr, offs[0..sh.groups]);
                try ar.launch(o, a, ro, 4, row_mode, false, split);
            } else {
                try o.attention(a, 4, row_mode, false, split);
                // H: wo_a's rot_in over the groups' column views of o, one launch, into L.o.xh at fusedRot's offsets
                var t = std.mem.zeroes(dense.RotTable);
                t.n = @intCast(sh.groups);
                for (0..sh.groups) |i| t.s[i] = .{ .x = out.ptr + 2 * @as(u64, i * gk), .suh = sp[i], .xh = xh.ptr + 2 * @as(u64, offs[i]), .ld = @intCast(H * 512), .K = @intCast(gk), .x_dtype = @backingInt(dense.DType.bf16) };
                try lin.segRotIn(t, R, false);
            }
        }
        try stream.synchronize();
        const tag = try std.fmt.bufPrint(&name, "heads {d} groups {d} hot {any} R {d} P {d} rows {any}", .{ H, sh.groups, sh.hot, R, split, row_mode });
        const lens = [_]usize{ R * H * 512 * 2, sh.groups * R * gk * 2 };
        var got: [4][]u8 = undefined;
        for (&got, outs, 0..) |*g, b, i| g.* = try bytes(gpa, b, lens[i % 2]);
        defer for (got) |g| gpa.free(g);
        try check.sameBytes(tag, got[2], got[0]); // the merge's own bf16 rows (L.o)
        try check.sameBytes(tag, got[3], got[1]); // wo_a's rotated fp16 rows (L.o.xh)
        const tk = try bytes(gpa, ticket, max_r * (H / attn.BMQ) * attn.MAXP * 4);
        defer gpa.free(tk);
        for (tk) |b| try check.expect(b == 0, "{s}: merge tickets not back at zero", .{tag});
        n_cases += 1;
    };
    return n_cases;
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels, rows_arg: ?[]const u8) !void {
    var rows_buf: [32]usize = undefined;
    var rows: []const usize = &.{ 1, 2, 3, 5, 16, 33, 64 };
    if (rows_arg) |r| {
        var it = std.mem.tokenizeScalar(u8, r, ',');
        var n: usize = 0;
        while (it.next()) |t| : (n += 1) {
            if (n == rows_buf.len) return error.TooManyRows;
            rows_buf[n] = try std.fmt.parseInt(usize, t, 10);
            if (rows_buf[n] < 1 or rows_buf[n] > 64) return error.BadRows;
        }
        rows = rows_buf[0..n];
    }
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    // the fold's instances from the same attention image as H's split_kernel
    var f = k.ops;
    f.attn_rot = try ar.Rot.resolve(k.mods[7], @intCast(try gpu.ctx.attribute(.max_shared_memory_per_block_optin)));
    const o: ops.Ops = .{ .f = &f, .s = stream };
    var rng: Rng = .{ .s = 0x9e3779b97f4a7c15 };
    var total: usize = 0;
    for (shapes) |sh| total += try runShape(gpu, o, k.linears(stream), stream, &rng, sh, rows);
    check.pass("rot-fold-attn: {d} launches' L.o and L.o.xh equal to split_kernel + seg_rot_in byte for byte", .{total});
}
