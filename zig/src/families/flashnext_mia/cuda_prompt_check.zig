//! Prompt matmul and indexer checks: both paths on the same inputs, bytes compared, and their timings.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("cuda_triton.zig");
const prompt = @import("cuda_prompt.zig");
const b16 = prompt.b16;
const fp4 = prompt.fp4;
const QsaScores = prompt.QsaScores;
const qsaScores = prompt.qsaScores;
const qsaSelect = prompt.qsaSelect;
const qsaScoresRows = prompt.qsaScoresRows;
const qsaSelectTiles = prompt.qsaSelectTiles;

// -- the check: both paths on the same inputs, bytes compared --------------------------------------------------

const Case = struct { n: usize, k: usize, fp32: bool, fp4: bool, tp: u32 };

/// Every prompt matmul with split K at TP=1 and TP=2 (kernels.json's 128-row `_b16mm` / `_fp4mm` entries with SK > 1).
pub const cases = [_]Case{
    .{ .n = 324, .k = 10240, .fp32 = true, .fp4 = false, .tp = 3 }, // HC down + inject
    .{ .n = 320, .k = 10240, .fp32 = true, .fp4 = false, .tp = 3 }, // mixer down
    .{ .n = 2560, .k = 2560, .fp32 = false, .fp4 = false, .tp = 3 }, // PLE key/value, MTP fc_e / fc_h
    .{ .n = 2560, .k = 6144, .fp32 = false, .fp4 = false, .tp = 1 }, // TP=1 out_proj / o_proj
    .{ .n = 2560, .k = 3072, .fp32 = true, .fp4 = false, .tp = 2 }, // TP=2 out_proj / o_proj partials
    .{ .n = 8240, .k = 2560, .fp32 = false, .fp4 = false, .tp = 2 }, // TP=2 GDN in_proj
    .{ .n = 7296, .k = 2560, .fp32 = false, .fp4 = false, .tp = 2 }, // TP=2 attention proj
    .{ .n = 2560, .k = 640, .fp32 = false, .fp4 = true, .tp = 1 }, // TP=1 shared down
    .{ .n = 2560, .k = 640, .fp32 = true, .fp4 = true, .tp = 1 },
    .{ .n = 2560, .k = 320, .fp32 = false, .fp4 = true, .tp = 2 }, // TP=2 shared down
    .{ .n = 2560, .k = 320, .fp32 = true, .fp4 = true, .tp = 2 },
};

pub const Fill = enum { normal, wide, cancel, edge, special, slices };

fn bf16Bits(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (std.math.isNan(x)) return 0x7FC0;
    const r = b + 0x7FFF + ((b >> 16) & 1);
    return @intCast(r >> 16);
}

/// A bf16 value for `fill` (`edge`: zeros of both signs, subnormals, the largest finite values, a few inf).
pub fn sample(r: std.Random, fill: Fill, i: usize) u16 {
    return switch (fill) {
        .normal => bf16Bits(r.floatNorm(f32)),
        .wide => blk: {
            // sign, exponent across 2^-30 .. 2^30, random mantissa: sums that round at every step
            const e: i32 = r.intRangeAtMost(i32, -30, 30);
            const m: f32 = 1.0 + r.float(f32);
            const v = std.math.ldexp(m, e);
            break :blk bf16Bits(if (r.boolean()) -v else v);
        },
        .cancel => blk: {
            // pairs +a, -a (a few bits off) so sums cancel and the low bits decide
            const a = std.math.ldexp(1.0 + r.float(f32), r.intRangeAtMost(i32, -4, 8));
            const off: f32 = if (r.uintLessThan(u32, 8) == 0) std.math.ldexp(@as(f32, 1.0), r.intRangeAtMost(i32, -12, -2)) else 0;
            break :blk bf16Bits(if (i % 2 == 0) a else -a + off);
        },
        // infinities, quiet and signaling NaNs with payloads, among normal values
        .special => switch (r.uintLessThan(u32, 64)) {
            0 => 0x7F80,
            1 => 0xFF80,
            2 => 0x7FC0 | @as(u16, r.intRangeAtMost(u16, 0, 0x3F)),
            3 => 0xFF81 + @as(u16, r.intRangeAtMost(u16, 0, 0x3E)),
            else => bf16Bits(r.floatNorm(f32)),
        },
        .slices => 0, // filled by slicesFill
        .edge => switch (r.uintLessThan(u32, 16)) {
            0 => 0x0000,
            1 => 0x8000,
            2 => @intCast(r.intRangeAtMost(u16, 1, 0x7F)), // subnormals
            3 => @as(u16, @intCast(r.intRangeAtMost(u16, 1, 0x7F))) | 0x8000,
            4 => 0x7F7F, // the largest finite
            5 => 0xFF7F,
            else => bf16Bits(r.floatNorm(f32) * 1e-3),
        },
    };
}

pub fn fillBuf(gpa: std.mem.Allocator, buf: cuda.DeviceBuffer, count: usize, r: std.Random, fill: Fill) !void {
    const h = try gpa.alloc(u16, count);
    defer gpa.free(h);
    for (h, 0..) |*v, i| v.* = sample(r, fill, i);
    try buf.upload(0, std.mem.sliceAsBytes(h));
}

/// Directed slice totals: rows of 2^24, 1, -2^24 across K slices, so only slice-order fp32 association gives 0.
fn slicesFill(gpa: std.mem.Allocator, x: cuda.DeviceBuffer, w: cuda.DeviceBuffer, rows: usize, stride: usize, n: usize, k: usize, sk: usize, r: std.Random) !void {
    const hx = try gpa.alloc(u16, rows * stride);
    defer gpa.free(hx);
    @memset(hx, 0);
    const ks = k / sk;
    for (0..rows) |i| {
        const row = hx[i * stride ..][0..k];
        const a = (i * 7) % ks;
        const b = (i * 13 + 5) % ks;
        const c = (i * 29 + 11) % ks;
        row[a] = 0x4B80; // 2^24
        if (sk >= 3) {
            row[ks + b] = 0x3F80;
            row[2 * ks + c] = 0xCB80;
        } else {
            row[ks + b] = 0x3F80;
            row[ks + (b + 1 + c % (ks - 1)) % ks] = 0xCB80;
        }
        // inside one slice too: the same three values in one K block of the last slice
        const base = (sk - 1) * ks + (i % (ks / 64)) * 64;
        row[base + 2] = 0x4B80;
        row[base + 3] = 0x3F80;
        row[base + 4] = 0xCB80;
    }
    try x.upload(0, std.mem.sliceAsBytes(hx));
    const hw = try gpa.alloc(u16, n * k);
    defer gpa.free(hw);
    const vals = [_]u16{ 0x3F80, 0xBF80, 0x4000, 0x3F00 };
    for (0..n) |j| {
        const v = vals[r.uintLessThan(usize, vals.len)];
        @memset(hw[j * k ..][0..k], v);
    }
    try w.upload(0, std.mem.sliceAsBytes(hw));
}

/// Both paths of every case at TP `tp` over `rows`, each input kind; true when every output byte is equal.
pub fn check(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri, tp: u32, rows: []const usize) !bool {
    var prng = std.Random.DefaultPrng.init(0x5eed_9f1);
    const r = prng.random();
    var all = true;
    var max_rows: usize = 0;
    for (rows) |m| max_rows = @max(max_rows, m);
    for (cases) |c| {
        if (c.tp & tp == 0) continue;
        const stride_extra: usize = 64;
        const xs_stride = c.k + stride_extra;
        const sk = if (c.fp4) tri.fp4SplitK(c.n, c.k) else tri.b16SplitK(c.n, c.k);
        var x = try cuda.DeviceBuffer.alloc(d, max_rows * xs_stride * 2);
        defer x.free();
        var w = try cuda.DeviceBuffer.alloc(d, c.n * c.k * 2);
        defer w.free();
        const es: usize = if (c.fp32) 4 else 2;
        var o1 = try cuda.DeviceBuffer.alloc(d, max_rows * c.n * es);
        defer o1.free();
        var o2 = try cuda.DeviceBuffer.alloc(d, max_rows * c.n * es);
        defer o2.free();
        var part = try cuda.DeviceBuffer.alloc(d, sk * max_rows * c.n * 4);
        defer part.free();
        var sc = try cuda.DeviceBuffer.alloc(d, (c.k / 16) * c.n * 4 + 256);
        defer sc.free();
        var s2 = try cuda.DeviceBuffer.alloc(d, c.n * 4 + 256);
        defer s2.free();
        try s2.fill32(@bitCast(@as(f32, 1.0)), t.s.handle);
        const h1 = try gpa.alloc(u8, max_rows * c.n * es);
        defer gpa.free(h1);
        const h2 = try gpa.alloc(u8, max_rows * c.n * es);
        defer gpa.free(h2);
        for ([_]Fill{ .normal, .wide, .cancel, .edge, .special, .slices }) |fill| {
            if (fill == .slices) {
                try slicesFill(gpa, x, w, max_rows, xs_stride, c.n, c.k, sk, r);
            } else {
                try fillBuf(gpa, x, max_rows * xs_stride, r, fill);
                // the other operand: random signs against `cancel`'s pairs, edge values against edge values
                try fillBuf(gpa, w, c.n * c.k, r, switch (fill) {
                    .cancel => .wide,
                    .special => .normal,
                    else => fill,
                });
            }
            if (c.fp4) {
                // the shared expert's scales are 1.0; other values check the scale step too
                const hs = try gpa.alloc(f32, (c.k / 16) * c.n);
                defer gpa.free(hs);
                for (hs) |*v| v.* = if (fill == .normal) 1.0 else std.math.ldexp(1.0 + r.float(f32), r.intRangeAtMost(i32, -3, 3));
                try sc.upload(0, std.mem.sliceAsBytes(hs));
            }
            for (rows) |m| {
                const strides = [_]usize{ c.k, xs_stride };
                for (strides) |stride| {
                    if (m * stride > max_rows * xs_stride) continue;
                    try o1.fill8(0xA5, t.s.handle);
                    try o2.fill8(0x5A, t.s.handle);
                    if (c.fp4) {
                        const fp: tri.Fp4 = .{ .weight = w.ptr, .scale = sc.ptr, .scale2 = s2.ptr };
                        try t.fp4mm(x.ptr, stride, fp, o1.ptr, c.fp32, part.ptr, m, c.n, c.k);
                        try fp4(t, x.ptr, stride, fp, o2.ptr, c.fp32, m, c.n, c.k);
                    } else {
                        try t.b16mm(x.ptr, stride, w.ptr, o1.ptr, c.fp32, part.ptr, m, c.n, c.k);
                        try b16(t, x.ptr, stride, w.ptr, o2.ptr, c.fp32, m, c.n, c.k);
                    }
                    try t.s.synchronize();
                    const bytes = m * c.n * es;
                    try o1.download(0, h1[0..bytes]);
                    try o2.download(0, h2[0..bytes]);
                    var differ: usize = 0;
                    var first: ?usize = null;
                    var i: usize = 0;
                    while (i < bytes) : (i += es) {
                        if (!std.mem.eql(u8, h1[i .. i + es], h2[i .. i + es])) {
                            differ += 1;
                            if (first == null) first = i / es;
                        }
                    }
                    const ok = differ == 0;
                    if (!ok) all = false;
                    std.debug.print("{s} {s} N {d} K {d} SK {d} {s} M {d} stride {d} {s}: {s}", .{ if (ok) "EQUAL" else "DIFFER", if (c.fp4) "_fp4mm_ks" else "_b16mm_ks", c.n, c.k, sk, if (c.fp32) "fp32" else "bf16", m, stride, @tagName(fill), if (ok) "" else "" });
                    if (first) |f0| std.debug.print("{d} of {d} values differ, first at row {d} col {d}", .{ differ, bytes / es, f0 / c.n, f0 % c.n });
                    std.debug.print("\n", .{});
                }
            }
        }
    }
    return all;
}

/// Time `reps` launches of each path for one case (ms a call), the speed side of the check.
pub fn bench(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri, c: Case, m: usize, reps: usize) !void {
    _ = gpa;
    const sk = if (c.fp4) tri.fp4SplitK(c.n, c.k) else tri.b16SplitK(c.n, c.k);
    var x = try cuda.DeviceBuffer.alloc(d, m * c.k * 2);
    defer x.free();
    try x.fill8(0x3c, t.s.handle);
    var w = try cuda.DeviceBuffer.alloc(d, c.n * c.k * 2);
    defer w.free();
    try w.fill8(0x3b, t.s.handle);
    var o = try cuda.DeviceBuffer.alloc(d, m * c.n * 4);
    defer o.free();
    var part = try cuda.DeviceBuffer.alloc(d, sk * m * c.n * 4);
    defer part.free();
    var sc = try cuda.DeviceBuffer.alloc(d, (c.k / 16) * c.n * 4 + 256);
    defer sc.free();
    try sc.fill32(@bitCast(@as(f32, 1.0)), t.s.handle);
    const fp: tri.Fp4 = .{ .weight = w.ptr, .scale = sc.ptr, .scale2 = sc.ptr };
    var e0 = try cuda.Event.init(d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(d, true);
    defer e1.deinit();
    var e2 = try cuda.Event.init(d, true);
    defer e2.deinit();
    for (0..2) |_| {
        try e0.record(t.s);
        for (0..reps) |_| if (c.fp4) try t.fp4mm(x.ptr, c.k, fp, o.ptr, c.fp32, part.ptr, m, c.n, c.k) else try t.b16mm(x.ptr, c.k, w.ptr, o.ptr, c.fp32, part.ptr, m, c.n, c.k);
        try e1.record(t.s);
        for (0..reps) |_| if (c.fp4) try fp4(t, x.ptr, c.k, fp, o.ptr, c.fp32, m, c.n, c.k) else try b16(t, x.ptr, c.k, w.ptr, o.ptr, c.fp32, m, c.n, c.k);
        try e2.record(t.s);
        try e2.synchronize();
    }
    const a = try cuda.Event.elapsedMs(e0, e1) / @as(f32, @floatFromInt(reps));
    const b = try cuda.Event.elapsedMs(e1, e2) / @as(f32, @floatFromInt(reps));
    const flop = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(c.n)) * @as(f64, @floatFromInt(c.k));
    std.debug.print("bench {s} N {d} K {d} SK {d} M {d}: split + _reduce {d:.3} ms ({d:.1} TFLOPS), in-program {d:.3} ms ({d:.1} TFLOPS), {d:.2}x\n", .{ if (c.fp4) "fp4" else "b16", c.n, c.k, sk, m, a, flop / a / 1e9, b, flop / b / 1e9, a / b });
}

/// `qsa-check`: `_scores_rows` vs `_scores`, `_select_tiles` vs `_select` on random and tie-heavy keys; true if equal.
pub fn qsaCheck(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri, fast: ?*const QsaScores) !bool {
    const capacity: usize = 262144 + 8;
    const g = tri.AttnGeometry.init(capacity, 2048, 4);
    const dims: tri.Dims = .{};
    const DI = dims.index_dim;
    const HI = dims.index_heads;
    const rows: usize = 256;
    var prng = std.Random.DefaultPrng.init(0x95a_0001);
    const r = prng.random();
    const hp = try gpa.alloc(u16, g.nb * DI);
    defer gpa.free(hp);
    const hq = try gpa.alloc(u16, rows * HI * DI);
    defer gpa.free(hq);
    var pooled = try cuda.DeviceBuffer.alloc(d, hp.len * 2);
    defer pooled.free();
    var iq = try cuda.DeviceBuffer.alloc(d, hq.len * 2);
    defer iq.free();
    var pos = try cuda.DeviceBuffer.alloc(d, 256);
    defer pos.free();
    var bufs: [2][4]cuda.DeviceBuffer = undefined;
    const sz = [4]usize{ rows * g.nb * 4, rows * g.idw * 4, rows * 4, rows * 4 };
    for (&bufs) |*set| for (set, sz) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n + 256);
    };
    defer for (&bufs) |*set| for (set) |*b| b.free();
    const host1 = try gpa.alloc(u8, sz[0]);
    defer gpa.free(host1);
    const host2 = try gpa.alloc(u8, sz[0]);
    defer gpa.free(host2);
    var all = true;
    // 16 B-aligned start positions: dense rows, register select, near and past the tiled switch (131,072 keys)
    const positions = [_]usize{ 1000, 2040, 20000, 70000, 129000, 131000, 200000, 261000 };
    for ([_]u8{ 0, 1, 2 }) |kind| {
        const ties = kind == 1;
        // kind 2: NaN, subnormal, -0 and inf codes among the keys and queries (Grok review P1-grok-1a #3)
        for (hp, 0..) |*v, i| v.* = if (ties) (if (i % 7 == 0) 0x3f80 else 0) else if (kind == 2) sample(r, .special, i) | (if (i % 97 == 3) @as(u16, 0x0001) else 0) else bf16Bits(r.floatNorm(f32));
        for (hq, 0..) |*v, i| v.* = if (kind == 2 and i % 89 == 5) @as(u16, if (i % 2 == 0) 0x8000 else 0x0003) else bf16Bits(r.floatNorm(f32));
        try pooled.upload(0, std.mem.sliceAsBytes(hp));
        try iq.upload(0, std.mem.sliceAsBytes(hq));
        for (positions) |p0| {
            const n = if (p0 + rows > capacity) capacity - p0 else rows;
            try pos.fill32(@intCast(p0), t.s.handle);
            const ends = p0 + n;
            var scs: [2]tri.AttnScratch = undefined;
            for (&scs, bufs) |*sc, set| {
                for (set) |b| try b.fill8(0xA7, t.s.handle);
                sc.* = .{ .po = 0, .pm = 0, .pl = 0, .ids = set[1].ptr, .nk = set[2].ptr, .sparse = set[3].ptr, .scores = set[0].ptr };
            }
            try qsaScores(t, iq.ptr, pooled.ptr, pos.ptr, scs[0], n, g, ends, dims);
            try qsaSelect(t, pos.ptr, scs[0], n, g, ends);
            for ([_]usize{ 8, 16, 0, 1, 2, 3 }) |rt| {
                try bufs[1][0].fill8(0xA7, t.s.handle);
                if (rt < 4) {
                    const q = fast orelse continue;
                    var qq = q.*;
                    qq.pick = rt;
                    try qq.run(t.s, iq.ptr, pooled.ptr, pos.ptr, scs[1], n, g, ends);
                } else try qsaScoresRows(t, iq.ptr, pooled.ptr, pos.ptr, scs[1], n, g, ends, dims, rt);
                try t.s.synchronize();
                try bufs[0][0].download(0, host1[0 .. n * g.nb * 4]);
                try bufs[1][0].download(0, host2[0 .. n * g.nb * 4]);
                const ok = std.mem.eql(u8, host1[0 .. n * g.nb * 4], host2[0 .. n * g.nb * 4]);
                if (!ok) all = false;
                std.debug.print("{s} {s} RT {d} position {d} rows {d}{s}: score bytes\n", .{ if (ok) "EQUAL" else "DIFFER", if (rt < 4) "fn_qsa_scores tile" else "_scores_rows", rt, p0, n, if (ties) " tie-heavy" else if (kind == 2) " special" else "" });
            }
            // the lists from the same scores by the tiled select
            try qsaSelectTiles(t, pos.ptr, scs[1], n, g);
            try t.s.synchronize();
            var ok = true;
            for (1..4) |k| {
                const bytes = if (k == 1) n * g.idw * 4 else n * 4;
                try bufs[0][k].download(0, host1[0..bytes]);
                try bufs[1][k].download(0, host2[0..bytes]);
                ok = ok and std.mem.eql(u8, host1[0..bytes], host2[0..bytes]);
            }
            if (!ok) all = false;
            std.debug.print("{s} _select_tiles = _select position {d} rows {d}{s}: key lists, lengths, flags\n", .{ if (ok) "EQUAL" else "DIFFER", p0, n, if (ties) " tie-heavy" else if (kind == 2) " special" else "" });
        }
    }
    return all;
}

/// Scoring time of 256 rows: `_scores` against fn_qsa_scores at several key counts (keys are random).
pub fn qsaBench(d: *const cuda.Driver, t: tri.Tri, fast: *const QsaScores) !void {
    const capacity: usize = 1048576 + 8;
    const g = tri.AttnGeometry.init(capacity, 2048, 4);
    const dims: tri.Dims = .{};
    const rows: usize = 256;
    var pooled = try cuda.DeviceBuffer.alloc(d, g.nb * dims.index_dim * 2);
    defer pooled.free();
    try pooled.fill8(0x3c, t.s.handle);
    var iq = try cuda.DeviceBuffer.alloc(d, rows * dims.index_heads * dims.index_dim * 2);
    defer iq.free();
    try iq.fill8(0x3b, t.s.handle);
    var pos = try cuda.DeviceBuffer.alloc(d, 256);
    defer pos.free();
    var scores = try cuda.DeviceBuffer.alloc(d, rows * g.nb * 4);
    defer scores.free();
    const sc: tri.AttnScratch = .{ .po = 0, .pm = 0, .pl = 0, .ids = 0, .nk = 0, .sparse = 0, .scores = scores.ptr };
    var e0 = try cuda.Event.init(d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(d, true);
    defer e1.deinit();
    var e2 = try cuda.Event.init(d, true);
    defer e2.deinit();
    for ([_]usize{ 32768, 131072, 524288, 1048000 }) |p0| {
        try pos.fill32(@intCast(p0), t.s.handle);
        const ends = p0 + rows;
        for (0..2) |_| {
            try e0.record(t.s);
            for (0..5) |_| try qsaScores(t, iq.ptr, pooled.ptr, pos.ptr, sc, rows, g, ends, dims);
            try e1.record(t.s);
            try e2.record(t.s);
            try e2.synchronize();
        }
        const a = try cuda.Event.elapsedMs(e0, e1) / 5;
        for (0..4) |pk| {
            var qq = fast.*;
            qq.pick = pk;
            try e1.record(t.s);
            for (0..5) |_| try qq.run(t.s, iq.ptr, pooled.ptr, pos.ptr, sc, rows, g, ends);
            try e2.record(t.s);
            try e2.synchronize();
            const b = try cuda.Event.elapsedMs(e1, e2) / 5;
            std.debug.print("bench scores {d} keys, 256 rows: _scores {d:.3} ms, fn_qsa_scores tile {d} {d:.3} ms ({d:.2}x)\n", .{ ends, a, pk, b, a / b });
        }
    }
}
