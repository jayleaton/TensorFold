//! MiniMax-M3.1-Flash fp8 KV cache (docs/fp8-kv.md), host only: codes, layout, quantizer and attention twins; no GPU.
const std = @import("std");
const core = @import("core");
const geom = @import("mm3_geom");
const rows_geom = @import("mm3_rows_geom");
const twin = @import("minimax_kv8_twin.zig");

const mr = core.minimax_ref;
const bf = core.bf16_round;
const fp8 = core.quant.fp8;
const testing = std.testing;

const scale64 = 0.08838834764831845; // 128^-0.5
const scale32: f32 = @floatCast(scale64);

fn must(ok: bool, src: std.builtin.SourceLocation) !void {
    if (!ok) {
        std.debug.print("control check failed at line {d}\n", .{src.line});
        return error.TestUnexpectedResult;
    }
}

fn bf16v(rng: std.Random, n: usize, sd: f32) ![]f32 {
    const v = try testing.allocator.alloc(f32, n);
    for (v) |*x| x.* = bf.round32(rng.floatNorm(f32) * sd);
    return v;
}

fn widen(x: []const f32) ![]f64 {
    const o = try testing.allocator.alloc(f64, x.len);
    for (o, x) |*d, s| d.* = s;
    return o;
}

test "e4m3 codes: every code round-trips, ties go to the even code, 448 saturates, NaN is 0x7f" {
    for (0..256) |i| {
        const c: u8 = @intCast(i);
        if (c & 0x7f == 0x7f) {
            try testing.expect(std.math.isNan(mr.e4m3Value(c)));
            continue;
        }
        try testing.expectEqual(c, mr.e4m3Code(mr.e4m3Value(c)));
        try testing.expectEqual(fp8.e4m3(c), mr.e4m3Value(c)); // the weights' decoder agrees
    }
    try testing.expectEqual(@as(f64, 448), mr.e4m3Value(0x7e));
    try testing.expectEqual(@as(u8, 0x38), mr.e4m3Code(1.0625)); // between 1 and 1.125: the even code
    try testing.expectEqual(@as(u8, 0x3a), mr.e4m3Code(1.1875)); // between 1.125 and 1.25: the even code
    try testing.expectEqual(@as(u8, 0xb8), mr.e4m3Code(-1.0625));
    try testing.expectEqual(@as(u8, 0x01), mr.e4m3Code(0x1p-9)); // the smallest subnormal
    try testing.expectEqual(@as(u8, 0x00), mr.e4m3Code(0x1p-10)); // half of it: a tie, to zero
    try testing.expectEqual(@as(u8, 0x02), mr.e4m3Code(3 * 0x1p-10)); // 1.5 subnormals: to 2
    try testing.expectEqual(@as(u8, 0x80), mr.e4m3Code(-0.0));
    for ([_]f64{ 448, 449, 463.9, 464, 1e9, std.math.inf(f64) }) |x| {
        try testing.expectEqual(@as(u8, 0x7e), mr.e4m3Code(x));
        try testing.expectEqual(@as(u8, 0xfe), mr.e4m3Code(-x));
    }
    try testing.expectEqual(@as(u8, 0x7f), mr.e4m3Code(std.math.nan(f64)));
    // against the weights' quantizer on random values (ties have measure zero)
    var prng = std.Random.DefaultPrng.init(1);
    const rng = prng.random();
    for (0..20000) |_| {
        const x = rng.floatNorm(f64) * std.math.pow(f64, 2, rng.float(f64) * 20 - 12);
        if (@abs(x) > 440) continue;
        try testing.expectEqual(fp8.toE4m3(x), mr.e4m3Value(mr.e4m3Code(x)));
    }
}

test "page bytes: 47,040 a token a rank with fp8 K/V (76,800 with bf16), 100,352 a layer, 6,021,120 a page" {
    const rank: geom.Pool = .{ .kvh = 2, .layers = 60, .kv8 = true };
    try testing.expectEqual(@as(usize, 100_352), rank.layerBytes());
    try testing.expectEqual(@as(usize, 784), rank.layerBytes() / 128); // K 256 + V 256 + IK 256 + 4 scales of 4 bytes
    try testing.expectEqual(@as(usize, 47_040), rank.tokenBytes());
    try testing.expectEqual(@as(usize, 6_021_120), rank.pageBytes());
    const bf16: geom.Pool = .{ .kvh = 2, .layers = 60 };
    try testing.expectEqual(@as(usize, 76_800), bf16.tokenBytes()); // the default is unchanged
    try testing.expect(rank.tokenBytes() * 1000 / bf16.tokenBytes() == 612); // 0.6125 of bf16
    // before the scales it is 46,080 = 60 * (512 + 256), the "about 46 KB" of the plan
    try testing.expectEqual(@as(usize, 46_080), rank.tokenBytes() - 60 * 2 * 2 * 4);
    const full: geom.Pool = .{ .kvh = 4, .layers = 1, .kv8 = true };
    try testing.expectEqual(@as(usize, 4 * 132 * 2 + 256), full.tokenBytes());
    // the same memory holds 1.63 times the tokens (whole pages)
    try testing.expectEqual(@as(usize, 983 * 128), bf16.tokensFitting(9 << 30));
    try testing.expectEqual(@as(usize, 1604 * 128), rank.tokensFitting(9 << 30)); // 205,312
}

fn mark(seen: []u8, byte: usize, len: usize) void {
    for (seen[byte..][0..len]) |*s| {
        std.debug.assert(s.* == 0);
        s.* = 1;
    }
}

test "fp8 paged addressing: K, V rows, scales and IK of every (page, layer, head, token) tile the pool with no gap and no overlap" {
    const p: geom.Pool = .{ .kvh = 2, .layers = 3, .kv8 = true };
    const pages = 3;
    const seen = try testing.allocator.alloc(u8, pages * p.pageBytes());
    defer testing.allocator.free(seen);
    @memset(seen, 0);
    for (0..pages) |pg| for (0..p.layers) |ly| for (0..128) |t| {
        for (0..2) |w| for (0..p.kvh) |h| {
            mark(seen, p.kv(pg, ly, w, h, t), 128);
            mark(seen, p.scale(pg, ly, w, h, t), 4);
        };
        mark(seen, p.ik(pg, ly, t), 256);
    };
    for (seen) |s| try testing.expectEqual(@as(u8, 1), s);
    // the scale of row t sits 4 bytes after row t - 1's; scales start after the IK plane
    try testing.expectEqual(p.scale(0, 0, 0, 0, 0) + 4, p.scale(0, 0, 0, 0, 1));
    try testing.expectEqual(p.ik(0, 0, 127) + 256, p.scale(0, 0, 0, 0, 0));
    try testing.expectEqual(p.layerBytes(), p.scale(0, 0, 1, 1, 127) + 4);
    // K and V rows are 128-byte aligned in every layer (the kernels load 16 bytes at a time)
    for (0..p.layers) |ly| try testing.expectEqual(@as(usize, 0), p.kv(1, ly, 1, 1, 5) % 16);
    // IK and the bf16 layout agree on where the index keys are read: 2 * kvh * 128 * 128 values in
    try testing.expectEqual(@as(usize, 2 * 2 * 128 * 128), p.ikInLayer());
    const b: geom.Pool = .{ .kvh = 2, .layers = 3 };
    try testing.expectEqual(2 * p.ikInLayer(), b.ikInLayer());
}

test "the launch flag: kv8 is bit 64 of the rows kernel's flags, off by default" {
    try testing.expectEqual(@as(i32, 1 | 2 | 4 | 8), (rows_geom.Flags{}).bits());
    try testing.expectEqual(@as(i32, 1 | 2 | 4 | 8 | 64), (rows_geom.Flags{ .kv8 = true }).bits());
}

fn randomRow(rng: std.Random, sd: f32, out: *[128]f32) void {
    for (out) |*x| x.* = bf.round32(rng.floatNorm(f32) * sd);
}

test "the fp32 twin follows the float64 reference: scales within an ulp, codes within one step, error within half a step" {
    var prng = std.Random.DefaultPrng.init(2);
    const rng = prng.random();
    var total: usize = 0;
    var off: usize = 0;
    var worst_rel: f64 = 0;
    for (0..2000) |trial| {
        var x: [128]f32 = undefined;
        randomRow(rng, std.math.pow(f32, 2, @floatFromInt(@as(i32, @intCast(trial % 24)) - 12)), &x);
        var c32: [128]u8 = undefined;
        var c64: [128]u8 = undefined;
        const s32 = twin.quantRow(&x, .none, 0, &c32);
        var x64: [128]f64 = undefined;
        for (&x64, x) |*d, v| d.* = v;
        const s64 = mr.quantRow(&x64, &c64);
        try testing.expectApproxEqRel(s64, @as(f64, s32), 1e-6);
        // the twin's own function is the reference's fp32 form
        var cf: [128]u8 = undefined;
        try testing.expectEqual(s32, mr.quantRowF32(&x, &cf));
        try testing.expectEqualSlices(u8, &c32, &cf);
        var dec: [128]f64 = undefined;
        mr.dequantRow(&c32, s32, &dec);
        for (c32, c64, x64, dec) |a, b, v, d| {
            total += 1;
            if (a != b) {
                off += 1;
                // neighbouring codes only: the fp32 product sat next to a midpoint
                try testing.expect(@abs(@as(i32, a & 0x7f) - @as(i32, b & 0x7f)) <= 1);
            }
            // half a step: relative 2^-4 above the subnormal range (|y| >= 2^-6), 2^-10 scales below it
            const y = @abs(v / @as(f64, s32));
            const bound = @as(f64, s32) * if (y >= 0x1p-6) y * 0x1p-4 * 1.0001 else 0x1p-10 * 1.0001;
            try testing.expect(@abs(d - v) <= bound);
            if (v != 0 and y >= 1) worst_rel = @max(worst_rel, @abs(d - v) / @abs(v));
        }
    }
    try testing.expect(off * 1000 < total); // near-midpoint disagreements: under 0.1 percent of values
    try testing.expect(worst_rel <= 0x1p-4 * 1.0001);
}

test "rule corners: zero rows, tiny rows, the largest value, one huge value, negative zero" {
    var c: [128]u8 = undefined;
    var x: [128]f32 = @splat(0);
    try testing.expectEqual(@as(f32, 1), twin.quantRow(&x, .none, 0, &c)); // all zero: scale 1, zero codes
    try testing.expectEqualSlices(u8, &@as([128]u8, @splat(0)), &c);
    x[3] = -0.0;
    _ = twin.quantRow(&x, .none, 0, &c);
    try testing.expectEqual(@as(u8, 0x80), c[3]); // the sign of a zero survives
    // the largest magnitude is code 448 exactly, in either sign
    x = @splat(0);
    x[0] = 3.5;
    x[1] = -3.5;
    const s = twin.quantRow(&x, .none, 0, &c);
    try testing.expectEqual(@as(f32, 3.5 / 448.0), s);
    try testing.expectEqual(@as(u8, 0x7e), c[0]);
    try testing.expectEqual(@as(u8, 0xfe), c[1]);
    // a denormal row: the scale is floored at 2^-126 and the codes stay finite
    for ([_]f32{ 1e-40, 1e-37, 4e-36, 5.3e-36 }) |tiny| {
        x = @splat(0);
        x[7] = tiny;
        const st = twin.quantRow(&x, .none, 0, &c);
        try testing.expect(st >= 0x1p-126 and std.math.isFinite(st));
        try testing.expect(c[7] & 0x7f != 0x7f);
        var x64: [128]f64 = undefined;
        for (&x64, x) |*d, v| d.* = v;
        var c64: [128]u8 = undefined;
        try testing.expectApproxEqRel(@as(f64, st), mr.quantRow(&x64, &c64), 1e-6);
    }
    // a huge row: scale near 3e36, no overflow
    x = @splat(1e38);
    const sh = twin.quantRow(&x, .none, 0, &c);
    try testing.expect(std.math.isFinite(sh));
    try testing.expectEqual(@as(u8, 0x7e), c[0]);
    var dec: [128]f64 = undefined;
    mr.dequantRow(&c, sh, &dec);
    for (dec) |d| try testing.expect(std.math.isFinite(d));
}

test "NaN and infinity: the whole row reads back NaN, a clean neighbour is untouched" {
    for ([_]f32{ nan32, inf32, -inf32 }) |bad| {
        var x: [128]f32 = undefined;
        var prng = std.Random.DefaultPrng.init(4);
        randomRow(prng.random(), 1, &x);
        x[77] = bad;
        var c: [128]u8 = undefined;
        const s = twin.quantRow(&x, .none, 0, &c);
        try testing.expect(std.math.isNan(s));
        try testing.expectEqualSlices(u8, &@as([128]u8, @splat(0)), &c);
        var dec: [128]f64 = undefined;
        mr.dequantRow(&c, s, &dec);
        for (dec) |d| try testing.expect(std.math.isNan(d));
        var x64: [128]f64 = undefined;
        for (&x64, x) |*d, v| d.* = v;
        var c64: [128]u8 = undefined;
        try testing.expect(std.math.isNan(mr.quantRow(&x64, &c64)));
        try testing.expectEqualSlices(u8, &c, &c64);
    }
}

const nan32 = std.math.nan(f32);
const inf32 = std.math.inf(f32);

test "a row's bits do not depend on the window: 1, 3 or 16 rows in a call give the same codes and scales" {
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    var rows: [16][128]f32 = undefined;
    for (&rows, 0..) |*r, i| randomRow(rng, 0.25 * @as(f32, @floatFromInt(1 + i * i)), r);
    var alone_c: [16][128]u8 = undefined;
    var alone_s: [16]f32 = undefined;
    for (rows, &alone_c, &alone_s) |r, *c, *s| s.* = twin.quantRow(&r, .none, 0, c);
    for ([_]usize{ 1, 3, 16 }) |w| {
        var at: usize = 0;
        while (at < 16) : (at += w) {
            const n = @min(w, 16 - at);
            var c: [16][128]u8 = undefined;
            var s: [16]f32 = undefined;
            twin.quantWindow(rows[at..][0..n], .none, c[0..n], s[0..n]);
            for (0..n) |i| {
                try testing.expectEqualSlices(u8, &alone_c[at + i], &c[i]);
                try testing.expectEqual(alone_s[at + i], s[i]);
            }
        }
    }
    // control: scaling by the window's amax makes a row depend on its neighbours
    var c: [16][128]u8 = undefined;
    var s: [16]f32 = undefined;
    twin.quantWindow(&rows, .batch_amax, &c, &s);
    try must(s[0] != alone_s[0], @src());
}

test "planted controls on the quantizer: each mutant breaks the rule and the checks see it" {
    var prng = std.Random.DefaultPrng.init(6);
    const rng = prng.random();
    // ties: amax 448 gives scale 1 and inv 1, so 1.0625 and 1.1875 are exact ties
    var x: [128]f32 = @splat(0);
    x[0] = 448;
    x[1] = 1.0625;
    x[2] = 1.1875;
    x[3] = -1.0625;
    x[4] = 2.4;
    var good: [128]u8 = undefined;
    var bad: [128]u8 = undefined;
    _ = twin.quantRow(&x, .none, 0, &good);
    try testing.expectEqual(@as(u8, 0x38), good[1]);
    try testing.expectEqual(@as(u8, 0x3a), good[2]);
    _ = twin.quantRow(&x, .ties_up, 0, &bad);
    try must(bad[1] == 0x39 and bad[2] == good[2], @src()); // the second tie is odd below, even above: both rules agree
    _ = twin.quantRow(&x, .truncate, 0, &bad);
    try must(bad[4] != good[4] or bad[1] != good[1], @src()); // 2.4 sits above 2.25 and below 2.5: truncation keeps 2.25 ...
    try testing.expect(mr.e4m3Value(good[4]) == 2.5); // ... the nearest is 2.5
    try testing.expect(mr.e4m3Value(bad[4]) == 2.25);
    // the reference scale catches a wrong maximum; the poison rule catches an ignored NaN
    var r: [128]f32 = undefined;
    randomRow(rng, 1, &r);
    const s_ok = twin.quantRow(&r, .none, 0, &good);
    try must(twin.quantRow(&r, .wrong_max, 0, &bad) != s_ok, @src());
    r[10] = nan32;
    try must(std.math.isNan(twin.quantRow(&r, .none, 0, &good)), @src());
    try must(!std.math.isNan(twin.quantRow(&r, .nan_ignored, 0, &bad)), @src());
    r[10] = inf32;
    try must(!std.math.isNan(twin.quantRow(&r, .nan_ignored, 0, &bad)), @src());
    // the scale floor: a denormal row without it saturates every nonzero value and turns zeros into NaN codes
    var t: [128]f32 = @splat(0);
    t[7] = 1e-40;
    _ = twin.quantRow(&t, .none, 0, &good);
    _ = twin.quantRow(&t, .no_floor, 0, &bad);
    try must(!std.mem.eql(u8, &good, &bad), @src());
    // k from its bf16 rounding: fp32 inputs that bf16 moves across a rounding midpoint change a code
    var moved = false;
    for (0..40) |_| {
        var o: [128]f32 = undefined;
        for (&o) |*v| v.* = rng.floatNorm(f32);
        _ = twin.quantKey(&o, .none, 0, &good);
        _ = twin.quantKey(&o, .k_unrounded, 0, &bad);
        moved = moved or !std.mem.eql(u8, &good, &bad);
    }
    try must(moved, @src());
}

fn idsFor(pos: usize, out: *[geom.topk]i32) []const i32 {
    @memset(out, -1);
    const nb = geom.blocks(pos);
    const n = @min(nb, geom.topk);
    for (out[0..n], 0..) |*o, i| o.* = @intCast(nb - n + i);
    return out[0..n];
}

const Case = struct {
    q: []f32,
    k: []f32,
    v: []f32,
    kc: []u8,
    vc: []u8,
    ks: []f32,
    vs: []f32,
    n: usize,

    fn init(rng: std.Random, pos: usize) !Case {
        const n = pos + 1;
        const a = testing.allocator;
        var c: Case = .{ .q = try bf16v(rng, 16 * 128, 1), .k = try bf16v(rng, n * 128, 1), .v = try bf16v(rng, n * 128, 1), .kc = try a.alloc(u8, n * 128), .vc = try a.alloc(u8, n * 128), .ks = try a.alloc(f32, n), .vs = try a.alloc(f32, n), .n = n };
        c.requant();
        return c;
    }

    fn requant(c: *Case) void {
        for (0..c.n) |j| {
            c.ks[j] = twin.quantRow(c.k[j * 128 ..][0..128], .none, 0, c.kc[j * 128 ..][0..128]);
            c.vs[j] = twin.quantRow(c.v[j * 128 ..][0..128], .none, 0, c.vc[j * 128 ..][0..128]);
        }
    }

    fn deinit(c: Case) void {
        const a = testing.allocator;
        a.free(c.q);
        a.free(c.k);
        a.free(c.v);
        a.free(c.kc);
        a.free(c.vc);
        a.free(c.ks);
        a.free(c.vs);
    }
};

/// The float64 attention over exactly the values the twin's codes and scales dequantize to.
fn attendDecoded(c: Case, rids: []const usize, pos: usize, out: []f64) !void {
    const a = testing.allocator;
    const q64 = try widen(c.q);
    defer a.free(q64);
    const kd = try a.alloc(f64, c.n * 128);
    defer a.free(kd);
    const vd = try a.alloc(f64, c.n * 128);
    defer a.free(vd);
    for (0..c.n) |j| {
        mr.dequantRow(c.kc[j * 128 ..][0..128], c.ks[j], kd[j * 128 ..][0..128]);
        mr.dequantRow(c.vc[j * 128 ..][0..128], c.vs[j], vd[j * 128 ..][0..128]);
    }
    var keys: [geom.topk * 128]usize = undefined;
    mr.attendBlocks(q64, kd, vd, rids, pos, 128, 128, 16, &keys, out);
}

fn maxAbs(x: []const f64) f64 {
    var m: f64 = 0;
    for (x) |v| m = @max(m, @abs(v));
    return m;
}

fn rmsRel(want: []const f64, got: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |a, b| {
        num += (a - b) * (a - b);
        den += a * a;
    }
    return @sqrt(num / den);
}

fn worstDiff(want: []const f64, got: []const f32) f64 {
    var w: f64 = 0;
    for (want, got) |a, b| w = @max(w, if (std.math.isNan(a) and std.math.isNan(b)) 0 else @abs(a - b));
    return w;
}

test "attention over fp8 K/V: the twin follows the float64 reference, and the fp8 error against bf16 K/V is bounded" {
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    var worst_vs_bf16: f64 = 0;
    var worst_rms: f64 = 0;
    for ([_]usize{ 127, 128, 129, 2047, 2048, 2049 }) |pos| {
        const c = try Case.init(rng, pos);
        defer c.deinit();
        var idb: [geom.topk]i32 = undefined;
        const ids = idsFor(pos, &idb);
        var rids: [geom.topk]usize = undefined;
        for (rids[0..ids.len], ids) |*r, i| r.* = @intCast(i);
        const q64 = try widen(c.q);
        defer testing.allocator.free(q64);
        const k64 = try widen(c.k);
        defer testing.allocator.free(k64);
        const v64 = try widen(c.v);
        defer testing.allocator.free(v64);
        const kq = try testing.allocator.alloc(f64, c.n * 128);
        defer testing.allocator.free(kq);
        const vq = try testing.allocator.alloc(f64, c.n * 128);
        defer testing.allocator.free(vq);
        var keys: [geom.topk * 128]usize = undefined;
        var want8: [16 * 128]f64 = undefined;
        var want16: [16 * 128]f64 = undefined;
        mr.attendBlocksKv8(q64, k64, v64, rids[0..ids.len], pos, 128, 128, 16, &keys, kq, vq, &want8);
        mr.attendBlocks(q64, k64, v64, rids[0..ids.len], pos, 128, 128, 16, &keys, &want16);
        var got: [16 * 128]f32 = undefined;
        twin.attend(c.q, c.kc, c.ks, c.vc, c.vs, ids, pos, scale32, .none, false, &got);
        const big = maxAbs(&want8);
        // over the twin's own codes the float64 attention differs only by fp32 order and where the scale is applied
        var same: [16 * 128]f64 = undefined;
        try attendDecoded(c, rids[0..ids.len], pos, &same);
        try testing.expect(worstDiff(&same, &got) <= 2e-4 * big + 1e-9);
        // against the float64 reference that quantizes by itself: exact ties in x / scale go the other way, 4 in 10,000
        try testing.expect(worstDiff(&want8, &got) <= 1e-2 * big);
        worst_vs_bf16 = @max(worst_vs_bf16, worstDiff(&want16, &got) / maxAbs(&want16));
        worst_rms = @max(worst_rms, rmsRel(&want16, &got));
        // padding with more -1 entries, again: the same bits
        var again: [16 * 128]f32 = undefined;
        twin.attend(c.q, c.kc, c.ks, c.vc, c.vs, &idb, pos, scale32, .none, false, &again);
        try testing.expectEqualSlices(f32, &got, &again);
    }
    std.debug.print("kv8 attention vs bf16 K/V, random N(0,1) rows, 6 positions: worst element error {d:.4} of the output's largest value, worst relative rms {d:.4}\n", .{ worst_vs_bf16, worst_rms });
    try testing.expect(worst_vs_bf16 < 0.1 and worst_rms < 0.06); // 3 mantissa bits: a few percent per element, never more
}

test "planted controls on the attention twin: a missing or misplaced scale, or a window-dependent tile, is caught" {
    var prng = std.Random.DefaultPrng.init(8);
    const rng = prng.random();
    const pos = 4 * 128 + 9;
    const c = try Case.init(rng, pos);
    defer c.deinit();
    var idb: [geom.topk]i32 = undefined;
    const ids = idsFor(pos, &idb);
    var rids: [geom.topk]usize = undefined;
    for (rids[0..ids.len], ids) |*r, i| r.* = @intCast(i);
    var want: [16 * 128]f64 = undefined;
    try attendDecoded(c, rids[0..ids.len], pos, &want);
    const big = maxAbs(&want);
    var out: [16 * 128]f32 = undefined;
    twin.attend(c.q, c.kc, c.ks, c.vc, c.vs, ids, pos, scale32, .none, false, &out);
    try testing.expect(worstDiff(&want, &out) <= 2e-4 * big);
    for ([_]twin.Mutant{ .k_scale_missing, .v_scale_missing, .v_scale_on_sum }) |m| {
        twin.attend(c.q, c.kc, c.ks, c.vc, c.vs, ids, pos, scale32, m, false, &out);
        try must(worstDiff(&want, &out) > 0.05 * big, @src());
    }
    // window_split: the 16-key tile changes fp32 bits for a short row; the fixed 32-key twin repeats exactly
    var a: [16 * 128]f32 = undefined;
    var b: [16 * 128]f32 = undefined;
    var s: [16 * 128]f32 = undefined;
    twin.attend(c.q, c.kc, c.ks, c.vc, c.vs, ids, pos, scale32, .none, false, &a);
    twin.attend(c.q, c.kc, c.ks, c.vc, c.vs, ids, pos, scale32, .none, false, &b);
    twin.attend(c.q, c.kc, c.ks, c.vc, c.vs, ids, pos, scale32, .window_split, false, &s);
    try testing.expectEqualSlices(f32, &a, &b);
    try must(!std.mem.eql(f32, &a, &s), @src());
}

test "a NaN or infinity in a key or value row turns that head's output NaN through its scale; other rows' heads stay finite" {
    var prng = std.Random.DefaultPrng.init(9);
    const rng = prng.random();
    const pos = 300;
    var c = try Case.init(rng, pos);
    defer c.deinit();
    var idb: [geom.topk]i32 = undefined;
    const ids = idsFor(pos, &idb);
    var out: [16 * 128]f32 = undefined;
    twin.attend(c.q, c.kc, c.ks, c.vc, c.vs, ids, pos, scale32, .none, false, &out);
    for (out) |x| try testing.expect(std.math.isFinite(x));
    // a poisoned key: every head's score for it is NaN, so every head is NaN (as with a NaN key in bf16)
    c.k[5 * 128 + 3] = nan32;
    c.requant();
    twin.attend(c.q, c.kc, c.ks, c.vc, c.vs, ids, pos, scale32, .none, false, &out);
    for (out) |x| try testing.expect(std.math.isNan(x));
    // a poisoned value row: its weight times a NaN scale makes every output dim NaN (bf16 would poison only dim 9)
    c.k[5 * 128 + 3] = 0.5;
    c.v[7 * 128 + 9] = inf32;
    c.requant();
    twin.attend(c.q, c.kc, c.ks, c.vc, c.vs, ids, pos, scale32, .none, false, &out);
    for (out) |x| try testing.expect(std.math.isNan(x));
    // a bad key row beyond the row's position is never read
    c.v[7 * 128 + 9] = 0.5;
    c.k[(pos + 1 - 1) * 128] = 1;
    c.requant();
    var dup = try Case.init(rng, pos + 40);
    defer dup.deinit();
    @memcpy(dup.q, c.q);
    @memcpy(dup.kc[0 .. c.n * 128], c.kc);
    @memcpy(dup.vc[0 .. c.n * 128], c.vc);
    @memcpy(dup.ks[0..c.n], c.ks);
    @memcpy(dup.vs[0..c.n], c.vs);
    var a: [16 * 128]f32 = undefined;
    twin.attend(c.q, c.kc, c.ks, c.vc, c.vs, ids, pos, scale32, .none, false, &out);
    twin.attend(dup.q, dup.kc, dup.ks, dup.vc, dup.vs, ids, pos, scale32, .none, false, &a);
    try testing.expectEqualSlices(f32, &out, &a);
}
