//! TF_DSV41_ROT_FOLD: lanes_rot_kernel against lanes_linear_kernel + seg_rot_in, byte for byte (synthetic data).

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const dense = dsv41.dense;
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
};

/// fp16 values in [lo, hi); `hot`: one in 97 an infinity of either sign (NaN and overflow paths downstream).
fn halves(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, n: usize, lo: f32, hi: f32, hot: bool) !cuda.DeviceBuffer {
    const h = try gpa.alloc(f16, n);
    defer gpa.free(h);
    for (h) |*v| {
        v.* = @floatCast(lo + (hi - lo) * rng.unit());
        if (hot and rng.next() % 97 == 0) v.* = if (rng.next() & 1 == 0) std.math.inf(f16) else -std.math.inf(f16);
    }
    return cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(h));
}

fn words(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, n: usize) !cuda.DeviceBuffer {
    const h = try gpa.alloc(u32, n);
    defer gpa.free(h);
    for (h) |*w| w.* = @truncate(rng.next());
    return cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(h));
}

fn zeroed(d: *const cuda.Driver, n: usize) !cuda.DeviceBuffer {
    const b = try cuda.DeviceBuffer.alloc(d, @max(n, 4));
    try b.fill8(0, null);
    return b;
}

/// A wo_a-like group: segments of K rows each, output columns side by side in one bf16 row of kb columns.
const Case = struct { k: usize, ns: []const usize, k2s: []const u32, hot: bool = false };

const cases = [_]Case{
    .{ .k = 4096, .ns = &.{ 1024, 1024, 1024, 1024 }, .k2s = &.{ 10, 10, 12, 8 } }, // the served wo_a: SK 8, WK 4
    .{ .k = 4096, .ns = &.{ 1024, 1024, 1024, 1024 }, .k2s = &.{ 10, 10, 12, 8 }, .hot = true },
    .{ .k = 512, .ns = &.{ 256, 384, 128 }, .k2s = &.{ 6, 4, 6 } }, // SK 1, WK 4
    .{ .k = 1024, .ns = &.{ 8192, 1024 }, .k2s = &.{ 10, 10 } }, // an 8-warp launch with a 4-warp split segment
};

fn minb(wk: usize) u32 {
    return if (wk == 8) 1 else 3;
}

fn bytes(gpa: std.mem.Allocator, b: cuda.DeviceBuffer) ![]u8 {
    const x = try gpa.alloc(u8, b.len);
    errdefer gpa.free(x);
    try b.download(0, x);
    return x;
}

fn runCase(gpu: Gpu, o: dense.Ops, stream: cuda.Stream, rng: *Rng, c: Case, rows: []const usize) !usize {
    const gpa = gpu.gpa;
    const d = gpu.d;
    const max_m: usize = 64;
    var segs: [dense.max_seg]dense.Seg = undefined;
    var kb: usize = 0;
    for (c.ns, c.k2s, 0..) |n, k2, i| {
        const pl = dense.plan(c.k, n);
        segs[i] = .{ .k = c.k, .n = n, .k2 = k2, .sk = pl[0], .wk = pl[1], .out = 0, .col = kb };
        kb += n;
    }
    const ss = segs[0..c.ns.len];
    const big = try dense.layout(ss, max_m);
    var xh_in = try halves(gpa, d, rng, big.xh_total, -1.0, 1.0, c.hot);
    defer xh_in.free();
    var T: [dense.max_seg]cuda.DeviceBuffer = undefined;
    var svh: [dense.max_seg]cuda.DeviceBuffer = undefined;
    // hot: column scales past fp16's range once rotated (infinities in wo_b's rows)
    const lo: f32 = if (c.hot) 4.0 else 0.01;
    const hi: f32 = if (c.hot) 64.0 else 0.03;
    for (ss, 0..) |s, i| {
        T[i] = try words(gpa, d, rng, (s.k / 16) * (s.n / 16) * 4 * s.k2);
        svh[i] = try halves(gpa, d, rng, s.n, lo, hi, false);
    }
    defer for (ss, 0..) |_, i| {
        T[i].free();
        svh[i].free();
    };
    var rsuh = try halves(gpa, d, rng, kb, -2.0, 2.0, false);
    defer rsuh.free();
    var z = try zeroed(d, big.z_total * 4);
    defer z.free();
    var counters = try zeroed(d, big.c_total * 4);
    defer counters.free();
    var outs: [4]cuda.DeviceBuffer = undefined; // oa and wo_b's rows: H's, then the fold's
    for (&outs) |*b| b.* = try zeroed(d, max_m * kb * 2);
    defer for (&outs) |*b| b.free();
    var name: [160]u8 = undefined;
    var n_cases: usize = 0;
    for (rows) |m| {
        const lay = try dense.layout(ss, m);
        for (&outs) |*b| try b.fill8(0, null);
        for ([_]bool{ false, true }) |fold| {
            const oa = outs[if (fold) 2 else 0];
            const xh = outs[if (fold) 3 else 1];
            var bufs: [dense.max_seg]dense.SegBufs = undefined;
            for (ss, 0..) |_, i| bufs[i] = .{ .T = T[i].ptr, .svh = svh[i].ptr, .y = oa.ptr, .ld = kb };
            for (lay.launches[0..lay.n_launches]) |l| {
                const tab = dense.linTable(lay, l, ss, bufs[0..ss.len], .bf16, xh_in.ptr, z.ptr, counters.ptr);
                if (fold) {
                    const cols = dense.launchCols(l, ss);
                    const ro = try dense.rotOut(cols[0..l.count], xh.ptr, rsuh.ptr, kb);
                    try o.lanesRotLinear(tab, ro, .bf16, m, l.programs, l.k2, l.wk, minb(l.wk), false);
                } else try o.lanesLinear(tab, .bf16, m, l, minb(l.wk), false);
            }
            if (!fold) {
                var t = std.mem.zeroes(dense.RotTable);
                t.n = 1;
                t.s[0] = .{ .x = oa.ptr, .suh = rsuh.ptr, .xh = xh.ptr, .ld = @intCast(kb), .K = @intCast(kb), .x_dtype = @backingInt(dense.DType.bf16) };
                try o.segRotIn(t, m, false);
            }
        }
        try stream.synchronize();
        const len = m * kb * 2;
        var got: [4][]u8 = undefined;
        for (&got, outs) |*g, b| g.* = try bytes(gpa, b);
        defer for (got) |g| gpa.free(g);
        const tag = try std.fmt.bufPrint(&name, "K {d} N {any} K2 {any} hot {any} M {d}", .{ c.k, c.ns, c.k2s, c.hot, m });
        try check.sameBytes(tag, got[2][0..len], got[0][0..len]); // the linear's own bf16 rows (L.oa)
        try check.sameBytes(tag, got[3][0..len], got[1][0..len]); // wo_b's rotated fp16 rows (L.wo_b.xh)
        const cnt = try bytes(gpa, counters);
        defer gpa.free(cnt);
        for (cnt) |b| try check.expect(b == 0, "{s}: k-split counters not back at zero", .{tag});
        n_cases += 1;
    }
    return n_cases;
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels, rows_arg: ?[]const u8) !void {
    var rows_buf: [32]usize = undefined;
    var rows: []const usize = &.{ 1, 3, 8, 9, 16, 17, 32, 33, 64 };
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
    // H's launches never take lin2's variants here; the fold's instances from the same dense3 image
    var f = k.dense;
    f.lin2 = null;
    f.rot = try dense.Rot.resolve(k.mods[6]);
    const o: dense.Ops = .{ .f = &f, .s = stream };
    var rng: Rng = .{ .s = 0x9e3779b97f4a7c15 };
    var total: usize = 0;
    for (cases) |c| total += try runCase(gpu, o, stream, &rng, c, rows);
    check.pass("rot-fold: {d} launches' L.oa and L.wo_b.xh equal to lanes_linear + seg_rot_in byte for byte", .{total});
}
