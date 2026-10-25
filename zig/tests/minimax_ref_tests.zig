//! MiniMax-M3.1-Flash float64 reference: primitives against hand-computed vectors (layer and model tests: model_tests).
const std = @import("std");
const core = @import("core");
const ref = @import("minimax_ref");
const mr = core.minimax_ref;

const expectEqual = std.testing.expectEqual;
const near = std.testing.expectApproxEqAbs;

test {
    _ = @import("minimax_ref_model_tests.zig");
}

test "gemma norm: (1 + w) scales the unit-rms row, plain w when the offset is off" {
    var out: [4]f64 = undefined;
    mr.gemmaNorm(&.{ 3, 4, 0, 0 }, &.{ 0, 0.5, 9, 9 }, 0, true, &out);
    try near(@as(f64, 1.2), out[0], 1e-15);
    try near(@as(f64, 2.4), out[1], 1e-15);
    try expectEqual(@as(f64, 0), out[2]);
    mr.gemmaNorm(&.{ 3, 4, 0, 0 }, &.{ 0, 0.5, 9, 9 }, 0, false, &out);
    try near(@as(f64, 0), out[0], 1e-15);
    try near(@as(f64, 0.8), out[1], 1e-15);
}

test "gemma norm plus NeoX RoPE equals core/qk_ref.zig with the (1 + w) weight and rotate-half tables" {
    var rng = std.Random.DefaultPrng.init(7);
    var x: [128]f64 = undefined;
    var w: [128]f64 = undefined;
    var w1: [128]f64 = undefined;
    for (&x, &w, &w1) |*a, *b, *c| {
        a.* = rng.random().float(f64) * 4 - 2;
        b.* = rng.random().float(f64) - 0.5;
        c.* = 1 + b.*;
    }
    var cs: [64]f64 = undefined;
    var sn: [64]f64 = undefined;
    for (0..32) |i| {
        const ang = 129.0 * std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(2 * i)) / 64.0);
        cs[i] = @cos(ang);
        cs[i + 32] = @cos(ang);
        sn[i] = @sin(ang);
        sn[i + 32] = @sin(ang);
    }
    var want: [128]f64 = undefined;
    core.qk_ref.head(&x, &w1, 1e-6, &cs, &sn, 64, &want);
    var got: [128]f64 = undefined;
    mr.gemmaNorm(&x, &w, 1e-6, true, &got);
    mr.rope(&got, 64, 129, 1e7, .neox_half);
    for (want, got) |p, q| try near(p, q, 1e-12);
}

test "rope: position 0 and dims past rotary_dim are untouched; NeoX pairs (i, i + 32), GPT-J pairs (2i, 2i + 1)" {
    var x: [128]f64 = @splat(0);
    x[0] = 1;
    x[1] = 1;
    x[100] = 5;
    mr.rope(&x, 64, 0, 1e7, .neox_half);
    try expectEqual(@as(f64, 1), x[0]);
    mr.rope(&x, 64, 1, 1e7, .neox_half);
    try near(@cos(@as(f64, 1)), x[0], 1e-15);
    try near(@sin(@as(f64, 1)), x[32], 1e-15);
    try expectEqual(@as(f64, 5), x[100]);
    const f1 = std.math.pow(f64, 1e7, -2.0 / 64.0);
    var z: [128]f64 = @splat(0);
    z[1] = 1;
    mr.rope(&z, 64, 1, 1e7, .neox_half);
    try near(@cos(f1), z[1], 1e-15);
    try near(@sin(f1), z[33], 1e-15);
    var y: [128]f64 = @splat(0);
    y[0] = 1;
    mr.rope(&y, 64, 1, 1e7, .gptj_interleaved);
    try near(@cos(@as(f64, 1)), y[0], 1e-15);
    try near(@sin(@as(f64, 1)), y[1], 1e-15);
    try expectEqual(@as(f64, 0), y[32]);
}

test "rope: norm preserved and q.k depends only on the position gap, in both layouts" {
    var rng = std.Random.DefaultPrng.init(3);
    var q0: [128]f64 = undefined;
    var k0: [128]f64 = undefined;
    for (&q0, &k0) |*a, *b| {
        a.* = rng.random().float(f64) - 0.5;
        b.* = rng.random().float(f64) - 0.5;
    }
    for ([_]mr.RopeLayout{ .neox_half, .gptj_interleaved }) |layout| {
        var dots: [2]f64 = undefined;
        for ([_][2]usize{ .{ 40, 33 }, .{ 7040, 7033 } }, 0..) |pq, i| {
            var q = q0;
            var k = k0;
            mr.rope(&q, 64, pq[0], 1e7, layout);
            mr.rope(&k, 64, pq[1], 1e7, layout);
            var d: f64 = 0;
            for (q, k) |a, b| d += a * b;
            dots[i] = d;
            var n0: f64 = 0;
            var n1: f64 = 0;
            for (q0, q) |a, b| {
                n0 += a * a;
                n1 += b * b;
            }
            try near(n0, n1, 1e-12);
        }
        try near(dots[0], dots[1], 1e-9);
    }
    var a = q0;
    var b = q0;
    mr.rope(&a, 64, 5, 1e7, .neox_half);
    mr.rope(&b, 64, 5, 1e7, .gptj_interleaved);
    try std.testing.expect(a[1] != b[1]);
}

test "swigluoai: g sigmoid(1.702 g) (u + 1), g capped at 7, u clamped to +-7, NaN propagates" {
    const sw: mr.Switches = .{};
    const s1 = 1.0 / (1.0 + @exp(-1.702));
    try near(s1 * 3, mr.swigluoai(1, 2, 1.702, 7, sw), 1e-15);
    try expectEqual(@as(f64, 0), mr.swigluoai(0, 5, 1.702, 7, sw));
    try expectEqual(@as(f64, 0), mr.swigluoai(3, -1, 1.702, 7, sw));
    try expectEqual(mr.swigluoai(7, 0, 1.702, 7, sw), mr.swigluoai(100, 0, 1.702, 7, sw));
    try expectEqual(mr.swigluoai(1, 7, 1.702, 7, sw), mr.swigluoai(1, 9, 1.702, 7, sw));
    try expectEqual(mr.swigluoai(1, -7, 1.702, 7, sw), mr.swigluoai(1, -9, 1.702, 7, sw));
    try near(s1 * -6, mr.swigluoai(1, -7, 1.702, 7, sw), 1e-15);
    try std.testing.expect(std.math.isNan(mr.swigluoai(std.math.nan(f64), 1, 1.702, 7, sw)));
    try std.testing.expect(std.math.isNan(mr.swigluoai(1, std.math.nan(f64), 1.702, 7, sw)));
}

test "swigluoai switches: up_plus_one off multiplies u, symmetric clamps the gate below too" {
    try near(1.0 / (1.0 + @exp(-1.702)) * 2, mr.swigluoai(1, 2, 1.702, 7, .{ .up_plus_one = false }), 1e-15);
    try expectEqual(@as(f64, 0), mr.swigluoai(3, 0, 1.702, 7, .{ .up_plus_one = false }));
    const upper = mr.swigluoai(-100, 0, 1.702, 7, .{});
    const sym = mr.swigluoai(-100, 0, 1.702, 7, .{ .gate_clamp = .symmetric });
    try std.testing.expect(@abs(upper) < 1e-60);
    try near(-7.0 / (1.0 + @exp(1.702 * 7.0)), sym, 1e-15);
}

test "topk: ties by the lower index, or the higher with the switch; NaN sorts last" {
    const s = [_]f64{ 1, 3, 3, 3, 0, 3 };
    var ids: [3]usize = undefined;
    mr.topk(&s, &ids, .lower_block);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, &ids);
    mr.topk(&s, &ids, .higher_block);
    try std.testing.expectEqualSlices(usize, &.{ 5, 3, 2 }, &ids);
    const n = [_]f64{ std.math.nan(f64), 2, 1 };
    var two: [2]usize = undefined;
    mr.topk(&n, &two, .lower_block);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, &two);
}

test "block scores: max over causal keys, blocks of 128, positions 127, 128, 129, partial last block" {
    var ik: [300 * 2]f64 = undefined;
    for (0..300) |j| {
        const f: f64 = @floatFromInt(j);
        ik[2 * j] = if (j % 3 == 0) f else -f;
        ik[2 * j + 1] = 1e9;
    }
    var bs: [3]f64 = undefined;
    try expectEqual(@as(usize, 1), mr.blockScores(&.{ 1, 0 }, &ik, 127, 2, 1, 128, &bs));
    try expectEqual(@as(f64, 126), bs[0]);
    try expectEqual(@as(usize, 2), mr.blockScores(&.{ 1, 0 }, &ik, 128, 2, 1, 128, &bs));
    try expectEqual(@as(f64, 126), bs[0]);
    try expectEqual(@as(f64, -128), bs[1]);
    try expectEqual(@as(usize, 2), mr.blockScores(&.{ 1, 0 }, &ik, 129, 2, 0.5, 128, &bs));
    try expectEqual(@as(f64, 63), bs[0]);
    try expectEqual(@as(f64, 64.5), bs[1]);
    try expectEqual(@as(usize, 3), mr.blockScores(&.{ 1, 0 }, &ik, 299, 2, 1, 128, &bs));
    try expectEqual(@as(f64, 297), bs[2]);
}

fn pick(bs: []const f64, own: usize, k: usize, sw: mr.Switches, out: []usize) usize {
    var scratch: [16]f64 = undefined;
    var ids: [16]usize = undefined;
    const n = mr.selectBlocks(bs, own, k, sw, &scratch, &ids);
    @memcpy(out[0..n], ids[0..n]);
    return n;
}

test "selection: own block forced, top-k of the rest, ascending; forced init and local variants" {
    const bs = [_]f64{ 5, 1, 9, 2, 8, 7, 3 };
    var o: [16]usize = undefined;
    try expectEqual(@as(usize, 3), pick(&bs, 6, 3, .{}, &o));
    try std.testing.expectEqualSlices(usize, &.{ 2, 4, 6 }, o[0..3]);
    _ = pick(&bs, 6, 3, .{ .sparse_local_block = 0 }, &o);
    try std.testing.expectEqualSlices(usize, &.{ 2, 4, 5 }, o[0..3]);
    _ = pick(&bs, 6, 3, .{ .sparse_init_block = 2 }, &o);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 6 }, o[0..3]);
    _ = pick(&bs, 6, 3, .{ .init_mode = .inclusive }, &o);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 6 }, o[0..3]);
    _ = pick(&bs, 6, 3, .{ .local_mode = .trailing_window, .sparse_local_block = 2 }, &o);
    try std.testing.expectEqualSlices(usize, &.{ 2, 5, 6 }, o[0..3]);
    _ = pick(&bs, 6, 2, .{ .sparse_init_block = 2 }, &o);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, o[0..2]);
    _ = pick(&bs, 6, 2, .{ .sparse_init_block = 2, .tie = .higher_block }, &o);
    try std.testing.expectEqualSlices(usize, &.{ 1, 6 }, o[0..2]);
}

test "selection: nb <= k attends every block; ties between scores go by the switch; own block beats a higher score" {
    var o: [16]usize = undefined;
    try expectEqual(@as(usize, 3), pick(&.{ 1, 2, 3 }, 2, 3, .{}, &o));
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, o[0..3]);
    try expectEqual(@as(usize, 1), pick(&.{7}, 0, 16, .{}, &o));
    try expectEqual(@as(usize, 2), pick(&.{ 1, 1 }, 1, 4, .{ .sparse_local_block = 0 }, &o));
    const tied = [_]f64{ 4, 4, 4, 4, 4 };
    _ = pick(&tied, 4, 3, .{}, &o);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 4 }, o[0..3]);
    _ = pick(&tied, 4, 3, .{ .tie = .higher_block }, &o);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, o[0..3]);
    _ = pick(&.{ 0, 0, 100, 100, -5 }, 4, 2, .{}, &o);
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, o[0..2]);
}

test "selection agrees with a sorted brute force on random tied scores" {
    var rng = std.Random.DefaultPrng.init(11);
    for (0..300) |_| {
        var bs: [12]f64 = undefined;
        const nb = 4 + rng.random().uintLessThan(usize, 9);
        for (bs[0..nb]) |*v| v.* = @floatFromInt(rng.random().uintLessThan(usize, 4));
        const own = nb - 1;
        const k = 1 + rng.random().uintLessThan(usize, 5);
        var o: [16]usize = undefined;
        const n = pick(bs[0..nb], own, k, .{}, &o);
        var order: [12]usize = undefined;
        for (order[0..nb], 0..) |*p, i| p.* = i;
        bs[own] = std.math.inf(f64);
        std.mem.sort(usize, order[0..nb], @as([]const f64, bs[0..nb]), struct {
            fn lt(s: []const f64, a: usize, b: usize) bool {
                return if (s[a] != s[b]) s[a] > s[b] else a < b;
            }
        }.lt);
        if (nb <= k) {
            try expectEqual(nb, n);
            continue;
        }
        std.mem.sort(usize, order[0..k], {}, std.sort.asc(usize));
        try std.testing.expectEqualSlices(usize, order[0..k], o[0..n]);
    }
}

fn naive(q: []const f64, k: []const f64, v: []const f64, keys: []const usize, d: usize, out: []f64) void {
    var s: [512]f64 = undefined;
    var m: f64 = -std.math.inf(f64);
    for (keys, 0..) |j, i| {
        var t: f64 = 0;
        for (0..d) |e| t += q[e] * k[j * d + e];
        s[i] = t / @sqrt(@as(f64, @floatFromInt(d)));
        m = @max(m, s[i]);
    }
    var z: f64 = 0;
    @memset(out, 0);
    for (keys, 0..) |j, i| {
        const p = @exp(s[i] - m);
        z += p;
        for (0..d) |e| out[e] += p * v[j * d + e];
    }
    for (out) |*o| o.* /= z;
}

fn fill(rng: *std.Random.DefaultPrng, x: []f64) void {
    for (x) |*v| v.* = rng.random().float(f64) * 2 - 1;
}

test "selected-block attention: dense when nb <= topk, only selected keys otherwise, causal inside the own block" {
    const d = 8;
    var rng = std.Random.DefaultPrng.init(5);
    var k: [300 * d]f64 = undefined;
    var v: [300 * d]f64 = undefined;
    var q: [2 * d]f64 = undefined;
    fill(&rng, &k);
    fill(&rng, &v);
    fill(&rng, &q);
    var keys: [512]usize = undefined;
    var got: [2 * d]f64 = undefined;
    var want: [d]f64 = undefined;
    var all: [201]usize = undefined;
    for (&all, 0..) |*a, i| a.* = i;
    mr.attendBlocks(&q, &k, &v, &.{ 0, 1 }, 200, 128, d, 2, &keys, &got);
    for (0..2) |h| {
        naive(q[h * d ..][0..d], &k, &v, &all, d, &want);
        for (want, got[h * d ..][0..d]) |p, g| try near(p, g, 1e-12);
    }
    @memset(v[128 * d .. 256 * d], 1e6);
    mr.attendBlocks(&q, &k, &v, &.{ 0, 2 }, 299, 128, d, 2, &keys, &got);
    var sel: [128 + 44]usize = undefined;
    for (0..128) |i| sel[i] = i;
    for (0..44) |i| sel[128 + i] = 256 + i;
    for (0..2) |h| {
        naive(q[h * d ..][0..d], &k, &v, &sel, d, &want);
        for (want, got[h * d ..][0..d]) |p, g| {
            try near(p, g, 1e-12);
            try std.testing.expect(@abs(g) < 10);
        }
    }
}

test "GQA 64 q heads over 4 kv heads: head h reads kv head h / 16 and its index head's own blocks" {
    const d = 128;
    const pos = 300;
    var rng = std.Random.DefaultPrng.init(9);
    const a = std.testing.allocator;
    const q = try a.alloc(f64, 64 * d);
    const k = try a.alloc(f64, 4 * 320 * d);
    const v = try a.alloc(f64, 4 * 320 * d);
    const ik = try a.alloc(f64, 320 * d);
    const iq = try a.alloc(f64, 4 * d);
    defer for ([_][]f64{ q, k, v, ik, iq }) |s| a.free(s);
    for ([_][]f64{ q, k, v, ik, iq }) |s| fill(&rng, s);
    var sets: [4][2]usize = undefined;
    var out: [64 * d]f64 = undefined;
    var keys: [512]usize = undefined;
    for (0..4) |g| {
        var bs: [3]f64 = undefined;
        var scratch: [3]f64 = undefined;
        var ids: [16]usize = undefined;
        const nb = mr.blockScores(iq[g * d ..][0..d], ik, pos, d, std.math.pow(f64, 128, -0.5), 128, &bs);
        try expectEqual(@as(usize, 3), nb);
        const n = mr.selectBlocks(&bs, 2, 2, .{}, &scratch, &ids);
        try expectEqual(@as(usize, 2), n);
        sets[g] = .{ ids[0], ids[1] };
        mr.attendBlocks(q[g * 16 * d ..][0 .. 16 * d], k[g * 320 * d ..][0 .. 320 * d], v[g * 320 * d ..][0 .. 320 * d], ids[0..n], pos, 128, d, 16, &keys, out[g * 16 * d ..][0 .. 16 * d]);
    }
    var differ = false;
    for (sets[1..]) |s| differ = differ or s[0] != sets[0][0];
    try std.testing.expect(differ);
    for ([_]usize{ 0, 15, 16, 47, 63 }) |h| {
        const g = h / 16;
        var list: [300]usize = undefined;
        var n: usize = 0;
        for (sets[g]) |b| for (b * 128..@min(b * 128 + 128, pos + 1)) |j| {
            list[n] = j;
            n += 1;
        };
        var want: [d]f64 = undefined;
        naive(q[h * d ..][0..d], k[g * 320 * d ..], v[g * 320 * d ..], list[0..n], d, &want);
        for (want, out[h * d ..][0..d]) |p, o| try near(p, o, 1e-12);
    }
}

test "router: sigmoid + bias picks, ties to the lower id, bias not in the weights, normalize then x2" {
    const l3 = @log(3.0);
    const logits = [_]f64{ 0, l3, -l3, l3, 0, -l3 };
    var scratch: [6]f64 = undefined;
    var ids: [2]usize = undefined;
    var w: [2]f64 = undefined;
    mr.routerPick(&logits, &@as([6]f64, @splat(0)), 2.0, .{}, &scratch, &ids, &w);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, &ids);
    try near(@as(f64, 1), w[0], 1e-15);
    try near(@as(f64, 1), w[1], 1e-15);
    const bias = [_]f64{ 0.3, 0, 0, 0, 0, 0 };
    mr.routerPick(&logits, &bias, 2.0, .{}, &scratch, &ids, &w);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, &ids);
    try near(@as(f64, 0.8), w[0], 1e-15);
    try near(@as(f64, 1.2), w[1], 1e-15);
    var rid: [2]usize = undefined;
    var rw: [2]f64 = undefined;
    core.route_ref.topk(&.{ 0.8, 0.75, 0.25, 0.75, 0.5, 0.25 }, &rid, &rw);
    try std.testing.expectEqualSlices(usize, &rid, &ids);
}

test "router switches: scale only, normalize only, bias in the weights, scale on routed plus shared" {
    const l3 = @log(3.0);
    const logits = [_]f64{ 0, l3, -l3, l3, 0, -l3 };
    const bias = [_]f64{ 0.3, 0, 0, 0, 0, 0 };
    var scratch: [6]f64 = undefined;
    var ids: [2]usize = undefined;
    var w: [2]f64 = undefined;
    mr.routerPick(&logits, &bias, 2.0, .{ .router_norm = .scale_only }, &scratch, &ids, &w);
    try near(@as(f64, 1.0), w[0], 1e-15);
    try near(@as(f64, 1.5), w[1], 1e-15);
    mr.routerPick(&logits, &bias, 2.0, .{ .router_norm = .normalize_only }, &scratch, &ids, &w);
    try near(@as(f64, 0.4), w[0], 1e-15);
    try near(@as(f64, 0.6), w[1], 1e-15);
    mr.routerPick(&logits, &bias, 2.0, .{ .router_weights_use_bias = true }, &scratch, &ids, &w);
    try near(@as(f64, 1.6 / 1.55), w[0], 1e-15);
    try near(@as(f64, 1.5 / 1.55), w[1], 1e-15);
    const sw: mr.Switches = .{ .scale_on = .routed_plus_shared };
    mr.routerPick(&logits, &bias, 2.0, sw, &scratch, &ids, &w);
    try near(@as(f64, 0.4), w[0], 1e-15);
    try expectEqual(@as(f64, 2), mr.totalScale(sw, 2));
    try expectEqual(@as(f64, 1), mr.totalScale(.{}, 2));
    try expectEqual(@as(f64, 1), mr.totalScale(.{ .scale_on = .routed_plus_shared, .router_norm = .normalize_only }, 2));
}

test "mlp: swigluoai(gate, up) down; w1 and w3 swap only for routed experts" {
    const eye = [_]f64{ 1, 0, 0, 1 };
    const swp = [_]f64{ 0, 1, 1, 0 };
    const m: ref.Mlp = .{ .w1 = .{ .k = 2, .n = 2, .w = &eye }, .w3 = .{ .k = 2, .n = 2, .w = &swp }, .w2 = .{ .k = 2, .n = 2, .w = &eye } };
    var out: [2]f64 = undefined;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const c: ref.Config = .{};
    try m.apply(c, &.{ 1, 2 }, false, arena.allocator(), &out);
    try near(mr.swigluoai(1, 2, 1.702, 7, .{}), out[0], 1e-15);
    try near(mr.swigluoai(2, 1, 1.702, 7, .{}), out[1], 1e-15);
    var c2 = c;
    c2.sw.w1_is_gate = false;
    try m.apply(c2, &.{ 1, 2 }, true, arena.allocator(), &out);
    try near(mr.swigluoai(2, 1, 1.702, 7, .{}), out[0], 1e-15);
    try near(mr.swigluoai(1, 2, 1.702, 7, .{}), out[1], 1e-15);
    try m.apply(c2, &.{ 1, 2 }, false, arena.allocator(), &out);
    try near(mr.swigluoai(1, 2, 1.702, 7, .{}), out[0], 1e-15);
}
