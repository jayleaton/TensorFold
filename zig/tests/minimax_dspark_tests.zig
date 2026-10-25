//! MiniMax-M3.1-Flash dspark draft kernels, host only: twins against float64, planted mutants, launch geometry and
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const twin = @import("minimax_dspark_twin.zig");

const c = core.contract.spec;
const mr = core.minimax_ref;
const ds = cuda.minimax_m3_dspark;
const testing = std.testing;
const registry = core.contract.registry;

test {
    _ = ds;
}

fn bf16v(rng: std.Random, n: usize, sd: f32) ![]f32 {
    const v = try testing.allocator.alloc(f32, n);
    for (v) |*x| x.* = c.fromBf16(c.toBf16(rng.floatNorm(f32) * sd));
    return v;
}

fn differs(a: []const u16, b: []const u16) bool {
    return !std.mem.eql(u16, a, b);
}

test "a tap and an add round once: bf16 is round-to-nearest-even, not truncation" {
    var prng = std.Random.DefaultPrng.init(1);
    const rng = prng.random();
    var cut: usize = 0;
    for (0..2000) |_| {
        const v = rng.floatNorm(f32) * 30;
        const trunc: u16 = @truncate(@as(u32, @bitCast(v)) >> 16);
        cut += @intFromBool(trunc != c.toBf16(v));
        try testing.expect(@abs(c.fromBf16(c.toBf16(v)) - v) <= @abs(v) * 0x1p-8);
    }
    try testing.expect(cut > 500);
}

test "norm: the twin tracks float64 within the chain bound, the sum order is the kernel's, and the planted faults are caught" {
    var prng = std.Random.DefaultPrng.init(2);
    const rng = prng.random();
    const k = 6144;
    const x = try bf16v(rng, k, 1);
    defer testing.allocator.free(x);
    const w = try bf16v(rng, k, 0.25);
    defer testing.allocator.free(w);
    var want: [k]f64 = undefined;
    var xs: [k]f64 = undefined;
    var ws: [k]f64 = undefined;
    for (&xs, &ws, x, w) |*a, *b, p, q| {
        a.* = p;
        b.* = q;
    }
    mr.gemmaNorm(&xs, &ws, 1e-6, true, &want);
    var got: [k]u16 = undefined;
    twin.norm(x, w, 1e-6, true, &got, .none);
    for (got, want) |g, wv| try testing.expect(@abs(c.fromBf16(g) - wv) <= @abs(wv) * (0x1p-8 + 40 * 0x1p-24) + 1e-30);
    // integer inputs: squares exact, so any order gives the same bits (gate 4's premise)
    var ints: [k]f32 = undefined;
    for (&ints) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -32, 32));
    var a: [k]u16 = undefined;
    var b: [k]u16 = undefined;
    twin.norm(&ints, w, 1e-6, true, &a, .none);
    twin.norm(&ints, w, 1e-6, true, &b, .sequential_sum);
    try testing.expectEqualSlices(u16, &a, &b);
    var differ = false;
    for (0..8) |s| {
        var r2 = std.Random.DefaultPrng.init(100 + s);
        const xr = try bf16v(r2.random(), k, 3);
        defer testing.allocator.free(xr);
        twin.norm(xr, w, 1e-6, true, &a, .none);
        twin.norm(xr, w, 1e-6, true, &b, .sequential_sum);
        differ = differ or differs(&a, &b);
    }
    try testing.expect(differ);
    twin.norm(x, w, 1e-6, true, &a, .none);
    twin.norm(x, w, 1e-6, true, &b, .no_offset);
    try testing.expect(differs(&a, &b));
    twin.norm(x, w, 1e-6, false, &b, .none);
    try testing.expect(differs(&a, &b)); // plain w is not Gemma's 1 + w
}

fn headRef(x: *const [128]f32, w: *const [128]f32, gemma: bool, rot: usize, pos: usize, y: *[128]f64) void {
    var xs: [128]f64 = undefined;
    var ws: [128]f64 = undefined;
    for (&xs, &ws, x, w) |*a, *b, p, q| {
        a.* = p;
        b.* = q;
    }
    mr.gemmaNorm(&xs, &ws, 1e-6, gemma, y);
    mr.rope(y, rot, pos, 1e7, .neox_half);
}

test "head: norm and RoPE follow core.minimax_ref, fp32 angles fail at a million positions, the pair layout and the ring slot matter" {
    var prng = std.Random.DefaultPrng.init(3);
    const rng = prng.random();
    var x: [128]f32 = undefined;
    var w: [128]f32 = undefined;
    for (&x, &w) |*a, *b| {
        a.* = c.fromBf16(c.toBf16(rng.floatNorm(f32) * 2));
        b.* = c.fromBf16(c.toBf16(rng.floatNorm(f32) * 0.25));
    }
    for ([_]usize{ 0, 7, 3000, 1_000_003 }) |pos| {
        var want: [128]f64 = undefined;
        headRef(&x, &w, true, 64, pos, &want);
        var got: [128]u16 = undefined;
        twin.head(&x, &w, 1e-6, true, 64, 1e7, pos, &got, .none);
        for (got, want) |g, wv| try testing.expect(@abs(c.fromBf16(g) - wv) <= @abs(wv) * 0x1p-7 + 1e-4);
    }
    var good: [128]u16 = undefined;
    var bad: [128]u16 = undefined;
    twin.head(&x, &w, 1e-6, true, 64, 1e7, 1_000_003, &good, .none);
    twin.head(&x, &w, 1e-6, true, 64, 1e7, 1_000_003, &bad, .angle_f32);
    try testing.expect(differs(&good, &bad));
    var want: [128]f64 = undefined;
    headRef(&x, &w, true, 64, 1_000_003, &want);
    var off: usize = 0;
    for (bad, want) |g, wv| off += @intFromBool(@abs(c.fromBf16(g) - wv) > @abs(wv) * 0x1p-7 + 1e-4);
    try testing.expect(off > 0); // the fp32 angle is outside the float64 bound
    twin.head(&x, &w, 1e-6, true, 64, 1e7, 5, &good, .none);
    twin.head(&x, &w, 1e-6, true, 64, 1e7, 5, &bad, .interleaved);
    try testing.expect(differs(&good, &bad));
    // dims past the rotary width are untouched by RoPE
    var plain: [128]u16 = undefined;
    twin.head(&x, &w, 1e-6, true, 64, 1e7, 0, &plain, .none);
    twin.head(&x, &w, 1e-6, true, 64, 1e7, 999, &good, .none);
    try testing.expectEqualSlices(u16, plain[64..], good[64..]);
    try testing.expectEqual(@as(usize, 7), twin.slot(4119, 4112, .none));
    try testing.expect(twin.slot(4119, 4112, .slot_unwrapped) != 7);
}

test "window start: the row's own window, the ring's first valid position, and the planted off-by-one faults" {
    try testing.expectEqual(@as(usize, 905), twin.windowStart(5000, 0, 4096, 0, .none));
    try testing.expectEqual(@as(usize, 908), twin.windowStart(5000, 3, 4096, 0, .none));
    try testing.expectEqual(@as(usize, 4000), twin.windowStart(5000, 3, 4096, 4000, .none));
    try testing.expectEqual(@as(usize, 100), twin.windowStart(300, 0, 1 << 20, 100, .none));
    try testing.expectEqual(@as(usize, 0), twin.windowStart(10, 6, 4096, 0, .none));
    try testing.expectEqual(@as(usize, 904), twin.windowStart(5000, 0, 4096, 0, .window_early));
    try testing.expect(twin.windowStart(5000, 0, 4096, 4500, .ignores_lo) != twin.windowStart(5000, 0, 4096, 4500, .none));
}

fn attnRef(q: *const [128]f32, ck: []const [128]f32, cv: []const [128]f32, bk: []const [128]f32, bv: []const [128]f32, scale: f64, out: *[128]f64) void {
    const n = ck.len + bk.len;
    var s = testing.allocator.alloc(f64, n) catch unreachable;
    defer testing.allocator.free(s);
    var mx: f64 = -std.math.inf(f64);
    for (0..n) |j| {
        const key = if (j < ck.len) ck[j] else bk[j - ck.len];
        var d: f64 = 0;
        for (0..128) |e| d += @as(f64, q[e]) * key[e];
        s[j] = d * scale;
        mx = @max(mx, s[j]);
    }
    var z: f64 = 0;
    out.* = @splat(0);
    for (0..n) |j| {
        const p = @exp(s[j] - mx);
        z += p;
        const val = if (j < ck.len) cv[j] else bv[j - ck.len];
        for (out, val) |*o, v| o.* += p * v;
    }
    for (out) |*o| o.* /= z;
}

test "attention: chunked fp32 softmax with a merge equals float64, and every planted fault changes the output" {
    var prng = std.Random.DefaultPrng.init(4);
    const rng = prng.random();
    const nctx = 300; // three chunks, the last partial
    const nblk = 7;
    var ck: [nctx][128]f32 = undefined;
    var cv: [nctx][128]f32 = undefined;
    var bk: [nblk][128]f32 = undefined;
    var bv: [nblk][128]f32 = undefined;
    var q: [128]f32 = undefined;
    for (&q) |*x| x.* = c.fromBf16(c.toBf16(rng.floatNorm(f32) * 2));
    for (&ck, &cv) |*a, *b| for (a, b) |*x, *y| {
        x.* = c.fromBf16(c.toBf16(rng.floatNorm(f32)));
        y.* = c.fromBf16(c.toBf16(rng.floatNorm(f32)));
    };
    for (&bk, &bv) |*a, *b| for (a, b) |*x, *y| {
        x.* = c.fromBf16(c.toBf16(rng.floatNorm(f32)));
        y.* = c.fromBf16(c.toBf16(rng.floatNorm(f32)));
    };
    const scale: f32 = @floatCast(std.math.pow(f64, 128, -0.5));
    var want: [128]f64 = undefined;
    attnRef(&q, &ck, &cv, &bk, &bv, scale, &want);
    var got: [128]f32 = undefined;
    twin.attend(&q, &ck, &cv, &bk, &bv, 2, scale, &got, .none);
    for (got, want) |g, wv| if (!(@abs(g - wv) <= 2e-5 + 2e-6 * @abs(wv))) {
        std.debug.print("twin {e} float64 {e}\n", .{ g, wv });
        return error.TestUnexpectedResult;
    };
    for ([_]twin.Mutant{ .merge_unweighted, .merge_drops_block, .drops_last_key, .causal_block, .unscaled }) |bad| {
        var o: [128]f32 = undefined;
        twin.attend(&q, &ck, &cv, &bk, &bv, 2, scale, &o, bad);
        var most: f64 = 0;
        for (o, want) |g, wv| most = @max(most, @abs(g - wv));
        if (!(most > 1e-3)) std.debug.print("mutant {t} moved by {e}\n", .{ bad, most });
        try testing.expect(most > 1e-3);
    }
    // a single context chunk and the block alone still agree
    twin.attend(&q, ck[0..50], cv[0..50], &bk, &bv, 6, scale, &got, .none);
    attnRef(&q, ck[0..50], cv[0..50], &bk, &bv, scale, &want);
    for (got, want) |g, wv| try testing.expect(@abs(g - wv) <= 2e-5 + 2e-6 * @abs(wv));
    twin.attend(&q, ck[0..0], cv[0..0], &bk, &bv, 6, scale, &got, .none);
    attnRef(&q, ck[0..0], cv[0..0], &bk, &bv, scale, &want);
    for (got, want) |g, wv| try testing.expect(@abs(g - wv) <= 2e-5 + 2e-6 * @abs(wv));
}

fn sortedOracle(row: []const f32, n: usize, want: usize, id0: u32, out: []twin.Pair) void {
    var idx: [4096]u32 = undefined;
    for (idx[0..n], 0..) |*x, i| x.* = @intCast(i);
    const Ctx = struct {
        row: []const f32,
        fn before(ctx: @This(), a: u32, b: u32) bool {
            const ka = twin.orderKey(ctx.row[a], .none);
            const kb = twin.orderKey(ctx.row[b], .none);
            return ka > kb or (ka == kb and a < b);
        }
    };
    std.mem.sort(u32, idx[0..n], Ctx{ .row = row }, Ctx.before);
    for (out[0..want], 0..) |*o, j| o.* = if (j < n) .{ .logit = row[idx[j]], .id = @intCast(id0 + idx[j]) } else .{ .logit = -std.math.inf(f32), .id = -1 };
}

fn samePairs(a: []const twin.Pair, b: []const twin.Pair) bool {
    for (a, b) |x, y| {
        if (x.id != y.id) return false;
        if (!(x.logit == y.logit or (std.math.isNan(x.logit) and std.math.isNan(y.logit)))) return false;
        if (!std.math.isNan(x.logit) and @as(u32, @bitCast(x.logit)) != @as(u32, @bitCast(y.logit))) return false;
    }
    return true;
}

test "top-C: the radix selection equals the sorted order on ties, NaN, infinities and signed zeros; the faults are caught" {
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    var row: [3000 + 64]f32 = undefined;
    var want: [64]twin.Pair = undefined;
    var got: [64]twin.Pair = undefined;
    var caught: [std.enums.values(twin.Mutant).len]bool = @splat(false);
    for (0..8) |kind| {
        for (0..3000) |i| row[i] = switch (kind) {
            0 => rng.floatNorm(f32) * 3,
            1 => @floatFromInt(rng.intRangeAtMost(i32, -3, 3)),
            2 => 1.5,
            3 => if (i % 5 == 0) nan else rng.floatNorm(f32),
            4 => nan,
            5 => if (i % 7 == 0) inf else if (i % 2 == 0) -0.0 else 0.0,
            6 => -inf,
            else => if (i % 3 == 0) -0.0 else @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -2, 2))),
        };
        @memset(row[3000..], inf); // decoys
        for ([_]usize{ 3000, 77, 40 }) |n| {
            sortedOracle(&row, n, 64, 128, &want);
            twin.topc(&row, n, 64, 128, &got, .none);
            try testing.expect(samePairs(&got, &want));
            for (std.enums.values(twin.Mutant)) |bad| switch (bad) {
                .ties_high, .nan_high, .signed_zero, .reads_padding => {
                    twin.topc(&row, n, 64, 128, &got, bad);
                    if (!samePairs(&got, &want)) caught[@backingInt(bad)] = true;
                },
                else => {},
            };
        }
    }
    for ([_]twin.Mutant{ .ties_high, .nan_high, .signed_zero, .reads_padding }) |bad| try testing.expect(caught[@backingInt(bad)]);
}

test "chain: the best score wins with ties to the lower id, the previous token feeds the next step, and the faults are caught" {
    var prng = std.Random.DefaultPrng.init(6);
    const rng = prng.random();
    const rank = 256;
    const cand = 8;
    const world = 2;
    var w1: [rank]f32 = undefined;
    var w2: [world * cand][rank]f32 = undefined;
    var w2s: [world * cand][]const f32 = undefined;
    var base: [world * cand]f32 = undefined;
    var ids: [world * cand]i32 = undefined;
    for (&w1) |*x| x.* = c.fromBf16(c.toBf16(rng.floatNorm(f32) * 0.5));
    for (&w2, &w2s, 0..) |*row, *s, i| {
        for (row) |*x| x.* = c.fromBf16(c.toBf16(rng.floatNorm(f32) * 0.5));
        s.* = row;
        base[i] = rng.floatNorm(f32) * 2;
        ids[i] = @intCast(10 + 3 * i);
    }
    // float64 oracle
    var best: usize = 0;
    var bz: f64 = -std.math.inf(f64);
    for (0..world * cand) |i| {
        var z: f64 = base[i];
        for (w1, w2[i]) |a, b| z += @as(f64, a) * b;
        if (z > bz) {
            bz = z;
            best = i;
        }
    }
    try testing.expectEqual(ids[best], twin.step(&base, &ids, &w1, &w2s, cand, .none));
    // an exact tie between two candidates that beat the rest goes to the lower id
    w2[5] = w2[3];
    base[3] = 100;
    base[5] = 100;
    try testing.expectEqual(ids[3], twin.step(&base, &ids, &w1, &w2s, cand, .none));
    try testing.expectEqual(ids[5], twin.step(&base, &ids, &w1, &w2s, cand, .chain_ties_high));
    // a candidate on the second rank wins: dropping that rank changes the token
    base[12] = 1000;
    try testing.expectEqual(ids[12], twin.step(&base, &ids, &w1, &w2s, cand, .none));
    try testing.expect(twin.step(&base, &ids, &w1, &w2s, cand, .chain_first_rank) != ids[12]);
    // NaN loses unless it is planted to win; padding is skipped; nothing valid gives -1
    base[12] = std.math.nan(f32);
    try testing.expect(twin.step(&base, &ids, &w1, &w2s, cand, .none) != ids[12]);
    try testing.expectEqual(ids[12], twin.step(&base, &ids, &w1, &w2s, cand, .chain_nan_wins));
    var none: [world * cand]i32 = @splat(-1);
    try testing.expectEqual(@as(i32, -1), twin.step(&base, &none, &w1, &w2s, cand, .none));
    // the previous token
    try testing.expectEqual(@as(u32, 77), twin.prevAfter(77, 5, .none));
    try testing.expectEqual(@as(u32, 5), twin.prevAfter(-1, 5, .none));
    try testing.expectEqual(@as(u32, 5), twin.prevAfter(77, 5, .chain_prev_anchor));
}

test "gate 5: every launch is registered, matches the geometry, and fits every chip" {
    const want = [_]struct { sym: []const u8, threads: u32, static: u32, dynamic: u32 = 0 }{
        .{ .sym = "tf_mm3_ds_tap", .threads = ds.threads, .static = 0 },
        .{ .sym = "tf_mm3_ds_add_bf16", .threads = ds.threads, .static = 0 },
        .{ .sym = "tf_mm3_ds_add_f32", .threads = ds.threads, .static = 0 },
        .{ .sym = "tf_mm3_ds_norm_bf16", .threads = ds.threads, .static = 32 },
        .{ .sym = "tf_mm3_ds_norm_f32", .threads = ds.threads, .static = 32 },
        .{ .sym = "tf_mm3_ds_block_heads", .threads = ds.threads, .static = 4608 },
        .{ .sym = "tf_mm3_ds_ring_put", .threads = ds.threads, .static = 4608 },
        .{ .sym = "tf_mm3_ds_attn", .threads = ds.threads, .static = 0, .dynamic = ds.attn_shared },
        .{ .sym = "tf_mm3_ds_attn_merge", .threads = ds.head_dim, .static = 0 },
        .{ .sym = "tf_mm3_ds_topc", .threads = ds.threads, .static = 4128 },
        .{ .sym = "tf_mm3_ds_chain", .threads = ds.threads, .static = 6288 },
    };
    var mine: [want.len]registry.CudaRecord = undefined;
    for (want, &mine) |w, *m| {
        const rec = for (registry.cuda) |r| {
            if (std.mem.eql(u8, r.launch.symbol, w.sym)) break r;
        } else return error.RecordMissing;
        try testing.expectEqualStrings("minimax_m3_dspark", rec.launch.image);
        try testing.expectEqual(w.threads, rec.launch.threads);
        try testing.expectEqual(w.dynamic, rec.launch.dynamic);
        try testing.expectEqual(w.dynamic > 48 * 1024, rec.launch.opt_in);
        for (rec.usage) |u| {
            try testing.expectEqual(w.static, u.static);
            try testing.expectEqual(w.threads, u.bound);
            try testing.expect(u.regs * w.threads <= 65536);
        }
        m.* = rec;
    }
    try testing.expectEqual(@as(usize, 0), try registry.checkCuda(testing.allocator, &mine, true));
    var over = mine[9];
    over.launch.threads = 2048;
    try testing.expect(try registry.checkCuda(testing.allocator, &.{over}, true) > 0);
}
