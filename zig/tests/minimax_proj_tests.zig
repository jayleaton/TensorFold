//! MiniMax-M3.1-Flash fp16 index projection and per-rank pick, host only: twins, mutants, merge rule, records.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const tp = @import("mm3_tp");
const twin = @import("minimax_proj_twin.zig");

const c = core.contract.spec;
const px = cuda.minimax_m3_proj;
const testing = std.testing;
const registry = core.contract.registry;

const hidden = 6144;
const iq_w = 256; // this rank's two index heads
const ik_w = 128;

test {
    _ = px;
}

fn bf16v(rng: std.Random, n: usize, sd: f32) ![]f32 {
    const v = try testing.allocator.alloc(f32, n);
    for (v) |*x| x.* = c.fromBf16(c.toBf16(rng.floatNorm(f32) * sd));
    return v;
}

fn fp16v(rng: std.Random, n: usize, sd: f32) ![]f32 {
    const v = try testing.allocator.alloc(f32, n);
    for (v) |*x| {
        const h: f16 = @floatCast(rng.floatNorm(f32) * sd);
        x.* = h;
    }
    return v;
}

fn fromBits(ot: twin.OutType, b: u16) f64 {
    return switch (ot) {
        .f16 => @as(f16, @bitCast(b)),
        .bf16 => c.fromBf16(b),
    };
}

/// The projection of `rows` rows through `n` outputs, as the kernel's call computes it.
fn call(x: []const f32, w: []const f32, rows: usize, n: usize, ot: twin.OutType, m: twin.Mutant, out: []u16) void {
    for (0..rows) |r| for (0..n) |e| {
        out[r * n + e] = twin.proj(x[r * hidden ..][0..hidden], w[e * hidden ..][0..hidden], hidden, rows, ot, m);
    };
}

fn dot(x: []const f32, w: []const f32) struct { sum: f64, mag: f64 } {
    var s: f64 = 0;
    var a: f64 = 0;
    for (x, w) |p, q| {
        s += @as(f64, p) * q;
        a += @abs(@as(f64, p) * q);
    }
    return .{ .sum = s, .mag = a };
}

test "products of bf16 and fp16 values are exact in fp32, so the fma only rounds the sums" {
    var prng = std.Random.DefaultPrng.init(41);
    const rng = prng.random();
    for (0..20000) |_| {
        const a = c.fromBf16(c.toBf16(rng.floatNorm(f32) * 3));
        const hw: f16 = @floatCast(rng.floatNorm(f32) * 0.2);
        const b: f32 = hw;
        try testing.expectEqual(@as(f64, a) * b, @as(f64, a * b));
    }
}

test "fp32 sums track the float64 dot within the chain bound, rows 1 to 17, fp16 and bf16 outputs, both matrices" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const x = try bf16v(rng, 17 * hidden, 1.5);
    defer testing.allocator.free(x);
    const w = try fp16v(rng, iq_w * hidden, 0.03);
    defer testing.allocator.free(w);
    const out = try testing.allocator.alloc(u16, 17 * iq_w);
    defer testing.allocator.free(out);
    inline for (.{ twin.OutType.f16, twin.OutType.bf16 }) |ot| {
        call(x, w, 17, iq_w, ot, .none, out);
        const unit: f64 = if (ot == .f16) 0x1p-11 else 0x1p-8;
        for (0..17) |r| for (0..iq_w) |e| {
            const d = dot(x[r * hidden ..][0..hidden], w[e * hidden ..][0..hidden]);
            // 48 fma steps a lane chain, the 4-chain sum, the 5-level tree, then the one rounding to the output
            const work = c.gamma(48 + 3 + 5, 0x1p-24) * d.mag;
            const got = fromBits(ot, out[r * iq_w + e]);
            try testing.expect(@abs(got - d.sum) <= work + unit * (@abs(d.sum) + work) + (if (ot == .f16) 0x1p-24 else 0x1p-133));
        };
    }
}

test "a row's fp32 sum is the same in a call of 1, 3, 4, 5 or 17 rows and at any row index; the planted faults are caught" {
    var prng = std.Random.DefaultPrng.init(43);
    const rng = prng.random();
    const x = try bf16v(rng, 17 * hidden, 2);
    defer testing.allocator.free(x);
    const w = try fp16v(rng, ik_w * hidden, 0.05);
    defer testing.allocator.free(w);
    // the sums, before the output's rounding hides their last 13 bits
    var full: [17 * ik_w]f32 = undefined;
    var part: [17 * ik_w]f32 = undefined;
    sums(x, w, 17, .none, &full);
    for ([_]usize{ 1, 3, 4, 5 }) |n| {
        sums(x[0 .. n * hidden], w, n, .none, part[0 .. n * ik_w]);
        try testing.expectEqualSlices(f32, full[0 .. n * ik_w], part[0 .. n * ik_w]);
    }
    sums(x[16 * hidden ..], w, 1, .none, part[0..ik_w]);
    try testing.expectEqualSlices(f32, full[16 * ik_w ..][0..ik_w], part[0..ik_w]);
    // a depth that follows the row count: row 0 of 17 differs from row 0 alone
    var one: [ik_w]f32 = undefined;
    sums(x, w, 17, .split_by_rows, &part);
    sums(x[0..hidden], w, 1, .split_by_rows, &one);
    try testing.expect(!std.mem.eql(f32, part[0..ik_w], &one));
    // the other faults change the sums of a call that the real order leaves alone
    for ([_]twin.Mutant{ .reorder_chains, .drops_tail, .half_accum }) |bad| {
        sums(x, w, 17, bad, &part);
        try testing.expect(!std.mem.eql(f32, &full, &part));
    }
    // rounding to the wrong type changes the output bits
    var bits: [17 * ik_w]u16 = undefined;
    var wrong: [17 * ik_w]u16 = undefined;
    call(x, w, 17, ik_w, .f16, .none, &bits);
    call(x, w, 17, ik_w, .f16, .wrong_out, &wrong);
    try testing.expect(!std.mem.eql(u16, &bits, &wrong));
}

fn sums(x: []const f32, w: []const f32, rows: usize, m: twin.Mutant, out: []f32) void {
    for (0..rows) |r| for (0..ik_w) |e| {
        out[r * ik_w + e] = twin.projSum(x[r * hidden ..][0..hidden], w[e * hidden ..][0..hidden], hidden, rows, m);
    };
}

test "integer inputs sum exactly in any order: the kernel's order equals float64, half accumulation and a dropped tail do not" {
    var prng = std.Random.DefaultPrng.init(44);
    const rng = prng.random();
    const x = try testing.allocator.alloc(f32, 3 * hidden);
    defer testing.allocator.free(x);
    const w = try testing.allocator.alloc(f32, 8 * hidden);
    defer testing.allocator.free(w);
    // gate 4's shape: multiples mirrored into the second half, partial sums near 2^20, the result small
    for (0..3) |r| for (0..hidden / 2) |i| {
        const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 32));
        x[r * hidden + i] = v;
        x[r * hidden + hidden / 2 + i] = v + if (i < 8) @as(f32, 1) else 0;
    };
    for (0..8) |e| for (0..hidden / 2) |i| {
        const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
        w[e * hidden + i] = v;
        w[e * hidden + hidden / 2 + i] = -v;
    };
    var out: [3 * 8]u16 = undefined;
    call(x, w, 3, 8, .bf16, .none, &out);
    for (0..3) |r| for (0..8) |e| {
        const d = dot(x[r * hidden ..][0..hidden], w[e * hidden ..][0..hidden]);
        try testing.expectEqual(c.toBf16(@floatCast(d.sum)), out[r * 8 + e]);
    };
    for ([_]twin.Mutant{ .half_accum, .drops_tail }) |bad| {
        var got: [3 * 8]u16 = undefined;
        call(x, w, 3, 8, .bf16, bad, &got);
        try testing.expect(!std.mem.eql(u16, &out, &got));
    }
}

test "NaN and infinities reach the output as float64 arithmetic gives them" {
    var prng = std.Random.DefaultPrng.init(45);
    const rng = prng.random();
    const x = try bf16v(rng, 4 * hidden, 1);
    defer testing.allocator.free(x);
    const w = try fp16v(rng, 16 * hidden, 0.05);
    defer testing.allocator.free(w);
    x[0 * hidden + 77] = std.math.nan(f32);
    x[1 * hidden + 5000] = std.math.inf(f32);
    x[2 * hidden + 6143] = -std.math.inf(f32);
    x[2 * hidden + 3] = std.math.inf(f32);
    var out: [4 * 16]u16 = undefined;
    call(x, w, 4, 16, .f16, .none, &out);
    for (0..4) |r| for (0..16) |e| {
        const d = dot(x[r * hidden ..][0..hidden], w[e * hidden ..][0..hidden]);
        try testing.expectEqual(c.classOf(d.sum), c.classOf(fromBits(.f16, out[r * 16 + e])));
    };
}

test "pick: the slice's best, ties to the lowest id, the vocabulary id is id0 plus the index, padding is never read" {
    var prng = std.Random.DefaultPrng.init(46);
    const rng = prng.random();
    var row: [3000]f32 = undefined;
    for (0..200) |t| {
        const n = 1 + rng.uintLessThan(usize, 2999);
        for (&row) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -4, 4)); // many ties
        for (row[n..]) |*v| v.* = 1000; // padding that would win
        const id0: u32 = 782 * 128 * @as(u32, @intCast(t % 2));
        const got = twin.pick(&row, n, id0, .none);
        var best: usize = 0;
        for (row[0..n], 0..) |v, i| if (v > row[best]) {
            best = i;
        };
        try testing.expectEqual(row[best], got.logit);
        try testing.expectEqual(id0 + @as(u32, @intCast(best)), got.id);
        for ([_]twin.Mutant{ .forgets_id0, .scans_padding }) |bad| {
            const w = twin.pick(&row, n, id0 + 1, bad);
            try testing.expect(w.id != id0 + 1 + @as(u32, @intCast(best)) or w.logit != got.logit);
        }
    }
}

test "pick: two ranks' picks merge to the argmax of the whole vocabulary, as tp.mergePicks gives it" {
    var prng = std.Random.DefaultPrng.init(47);
    const rng = prng.random();
    const split = [2][2]usize{ .{ 0, 384 }, .{ 384, 640 } }; // 5 units of 128 over 2 ranks: 3 and 2 (shard.Share)
    var logits: [640]f32 = undefined;
    for (0..300) |_| {
        for (&logits) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, -3, 3));
        var picks: [2]tp.Pick = undefined;
        for (0..2) |r| {
            const lo = split[r][0];
            const hi = split[r][1];
            var row: [400]f32 = @splat(7777); // padded slot
            @memcpy(row[0 .. hi - lo], logits[lo..hi]);
            const p = twin.pick(&row, hi - lo, @intCast(lo), .none);
            picks[r] = .{ .logit = p.logit, .id = p.id };
        }
        const merged = tp.mergePicks(&picks);
        const mx = std.mem.max(f32, &logits);
        try testing.expectEqual(mx, merged.logit);
        try testing.expectEqual(@as(u32, @intCast(std.mem.indexOfScalar(f32, &logits, mx).?)), merged.id);
    }
}

test "pick: NaN ranks last, an all-NaN row goes to its lowest id, and the planted faults are caught" {
    const nan = std.math.nan(f32);
    var row: [1000]f32 = undefined;
    for (&row, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast(i % 7)) - 3);
    row[0] = nan;
    row[5] = nan;
    const got = twin.pick(&row, 1000, 100, .none);
    try testing.expectEqual(@as(f32, 3), got.logit);
    try testing.expectEqual(@as(u32, 100 + 6), got.id); // the first 3 is at index 6
    // a NaN at the lowest index stays under the older comparison, a NaN-wins pick returns it
    try testing.expect(std.math.isNan(twin.pick(&row, 1000, 100, .nan_wins).logit));
    try testing.expect(std.math.isNan(twin.pick(&row, 1000, 100, .nan_first_stays).logit));
    // the tie rule: the highest id wins under the planted fault
    try testing.expect(twin.pick(&row, 1000, 100, .ties_to_last).id != got.id);
    @memset(&row, nan);
    const all = twin.pick(&row, 700, 5, .none);
    try testing.expect(std.math.isNan(all.logit));
    try testing.expectEqual(@as(u32, 5), all.id);
    // fewer logits than threads: the idle threads never win
    const small = twin.pick(&[_]f32{ -5, -2, -9 }, 3, 40, .none);
    try testing.expectEqual(@as(f32, -2), small.logit);
    try testing.expectEqual(@as(u32, 41), small.id);
    // the host merge ranks NaN last too, in either order
    const fin: tp.Pick = .{ .logit = -8, .id = 700 };
    try testing.expectEqual(fin, tp.mergePicks(&.{ .{ .logit = nan, .id = 3 }, fin }));
    try testing.expectEqual(fin, tp.mergePicks(&.{ fin, .{ .logit = nan, .id = 3 } }));
}

test "gate 5: the three launches are registered, match the geometry, and fit every chip" {
    const want = [_]struct { sym: []const u8, threads: u32, static: u32 }{
        .{ .sym = "tf_mm3_proj_bf16", .threads = px.threads, .static = 0 },
        .{ .sym = "tf_mm3_proj_f32", .threads = px.threads, .static = 0 },
        .{ .sym = "tf_mm3_pick", .threads = px.pick_threads, .static = 2 * (px.pick_threads / 32) * 4 },
    };
    var mine: [3]registry.CudaRecord = undefined;
    for (want, &mine) |w, *m| {
        const rec = for (registry.cuda) |r| {
            if (std.mem.eql(u8, r.launch.symbol, w.sym)) break r;
        } else return error.RecordMissing;
        try testing.expectEqualStrings("minimax_m3_proj", rec.launch.image);
        try testing.expectEqual(w.threads, rec.launch.threads);
        try testing.expectEqual(@as(u32, 0), rec.launch.dynamic);
        for (rec.usage) |u| {
            try testing.expectEqual(w.static, u.static);
            try testing.expectEqual(w.threads, u.bound);
            try testing.expect(u.regs * w.threads <= 65536);
        }
        m.* = rec;
    }
    try testing.expectEqual(@as(usize, 0), try registry.checkCuda(testing.allocator, &mine, true));
    var over = mine[2];
    over.launch.threads = 512;
    try testing.expect(try registry.checkCuda(testing.allocator, &.{over}, true) > 0);
}
