//! MiniMax-M3.1-Flash's two-rank TP plan: each tensor's collective, their geometry and the rank-order fp32 sum (host).
const std = @import("std");
const names = @import("names.zig");
const Config = @import("config.zig").Config;
const shard = @import("shard.zig");

/// Part of ranks.Fingerprint: bump when the order or the width of the exchanged partials changes.
pub const reduction_version: u32 = 1;

/// What a rank does with a tensor's result.
pub const Reduction = enum {
    none, // each rank owns whole output columns (or a replicated tensor): nothing is exchanged
    partial_sum, // each rank holds an fp32 partial over its input rows: gathered and added in rank order
    vocab_gather, // each rank holds its vocabulary rows' logits: gathered into the whole vocabulary
};

/// Where in a layer (or after the last) a collective happens.
pub const Site = enum {
    attn_out, // after o_proj (row cut)
    ffn_out, // after the FFN: dense down (layers before the first MoE layer), or routed + shared together
    head, // after lm_head
};

pub const Collective = struct {
    site: Site,
    kind: Reduction,
    /// fp32 elements a row contributes (hidden for sums, the rank's vocabulary slot for the gather).
    width: usize,
};

/// What the sum of `p` needs; the cut axis of names.Proj decides (row cut means partial sums), plus the head.
pub fn reductionOf(p: names.Proj) Reduction {
    if (p == .head) return .vocab_gather;
    return switch (p.cut()) {
        .row => .partial_sum,
        .col => .none,
    };
}

/// Tensors replicated on every rank with the same kernel, so the same bits, and so needing no collective.
pub const replicated = [_][]const u8{ "index_k_proj", "q_norm", "k_norm", "index_q_norm", "index_k_norm", "router_gate", "embed_tokens", "layernorms" };

/// Layer `layer`'s collectives in order: the attention sum, then one FFN sum (routed + shared added before it).
pub fn layerPlan(c: *const Config, layer: u32) [2]Collective {
    _ = layer; // dense and MoE layers have the same two sites; only the tensors feeding the second differ
    return .{
        .{ .site = .attn_out, .kind = reductionOf(.o), .width = c.hidden },
        .{ .site = .ffn_out, .kind = reductionOf(.w2), .width = c.hidden },
    };
}

/// Exchanges a token costs: two a layer.
pub fn tokenExchanges(c: *const Config, layers: u32) usize {
    var n: usize = 0;
    for (0..layers) |l| for (layerPlan(c, @intCast(l))) |x| {
        if (x.kind == .partial_sum) n += 1;
    };
    return n;
}

/// fp32 bytes a rank sends for one sum of `rows` rows; the all-gather receive buffer has a slot per rank in rank order.
pub fn sumSlotBytes(c: *const Config, rows: usize) usize {
    return rows * c.hidden * 4;
}
pub fn sumGatherBytes(c: *const Config, rows: usize, world: usize) usize {
    return world * sumSlotBytes(c, rows);
}

/// The lm_head's gather: equal slots of the largest share, the shorter ranks' tails padding the merge ignores.
pub const LogitsGather = struct {
    slot: usize, // logits a rank's slot holds per row
    world: usize,
    vocab: usize,

    pub fn of(c: *const Config, world: usize) !LogitsGather {
        var biggest: usize = 0;
        for (0..world) |r| biggest = @max(biggest, (try shard.Share.of(c, r, world)).vocab.len());
        return .{ .slot = biggest, .world = world, .vocab = c.vocab };
    }

    /// Bytes of the receive buffer for `rows` rows of fp32 logits.
    pub fn bytes(g: LogitsGather, rows: usize) usize {
        return g.world * rows * g.slot * 4;
    }

    /// Row `row` of the whole vocabulary from the gathered slots ([rank][row][slot]); `out` holds `vocab` logits.
    pub fn merge(g: LogitsGather, c: *const Config, gathered: []const f32, rows: usize, row: usize, out: []f32) !void {
        std.debug.assert(out.len == g.vocab and gathered.len == g.world * rows * g.slot);
        for (0..g.world) |r| {
            const v = (try shard.Share.of(c, r, g.world)).vocab;
            const from = (r * rows + row) * g.slot;
            @memcpy(out[v.lo..v.hi], gathered[from .. from + v.len()]);
        }
    }
};

/// A greedy pick (logit, id); merging ranks keeps the larger logit, lower id on ties, equal to a whole-vocab argmax.
pub const Pick = struct { logit: f32, id: u32 };

/// A NaN logit ranks last (all NaN: the lower id), as tf_mm3_pick does, so a NaN rank never hides a finite one.
pub fn betterPick(a: Pick, b: Pick) Pick {
    const an = std.math.isNan(a.logit);
    const bn = std.math.isNan(b.logit);
    if (an and bn) return if (b.id < a.id) b else a;
    if (bn) return a;
    if (an) return b;
    if (b.logit > a.logit or (b.logit == a.logit and b.id < a.id)) return b;
    return a;
}

pub fn mergePicks(all: []const Pick) Pick {
    var best = all[0];
    for (all[1..]) |p| best = betterPick(best, p);
    return best;
}

/// The rank-order sum as tf_rank_sum does it: out[i] = ((t0 + t1) + t2) ..., single __fadd_rn adds, own at slot `me`.
pub fn rankSum(peers: []const f32, own: []const f32, world: usize, me: usize, out: []f32) void {
    const n = own.len;
    std.debug.assert(out.len == n and peers.len == world * n and me < world);
    for (0..n) |i| {
        var acc: f32 = if (me == 0) own[i] else peers[i];
        for (1..world) |r| {
            const term = if (r == me) own[i] else peers[r * n + i];
            acc = acc + term;
        }
        out[i] = acc;
    }
}

/// Which collective carries a sum (both give `rankSum`'s bits): gather_sum for decode windows, scatter_gather prompts.
pub const Form = enum { gather_sum, scatter_gather };

/// The reduction form for `rows` rows. Both are bit-identical by construction (tested below on host arithmetic).
pub fn formFor(rows: usize) Form {
    return if (rows <= 128) .gather_sum else .scatter_gather;
}

/// A `scatter_gather` on the host: each rank sums its rows' share, then the shares are concatenated.
pub fn scatterGatherSum(world: usize, parts: []const []const f32, rows: usize, width: usize, out: []f32) void {
    for (0..world) |r| {
        const lo = rowShare(rows, world, r)[0];
        const hi = rowShare(rows, world, r)[1];
        for (lo * width..hi * width) |i| {
            var acc = parts[0][i];
            for (1..world) |k| acc = acc + parts[k][i];
            out[i] = acc;
        }
    }
}

/// rank_scatter.zig's `share`, restated (it is not importable from the host build): whole rows, early ranks one more.
pub fn rowShare(rows: usize, world: usize, rank: usize) [2]usize {
    const base = rows / world;
    const extra = rows % world;
    const lo = rank * base + @min(rank, extra);
    return .{ lo, lo + base + @intFromBool(rank < extra) };
}

// ---------------------------------------------------------------- tests

const flash = &@import("config.zig").flash;

fn spanEq(a: shard.Span, lo: usize, hi: usize) !void {
    try std.testing.expectEqual(shard.Span{ .lo = lo, .hi = hi }, a);
}

test "every projection's reduction follows its cut axis; only the head gathers" {
    for (std.enums.values(names.Proj)) |p| {
        const want: Reduction = if (p == .head) .vocab_gather else if (p.cut() == .row) .partial_sum else .none;
        try std.testing.expectEqual(want, reductionOf(p));
    }
    // the row cuts are exactly o_proj and the three down projections
    var rows: usize = 0;
    for (std.enums.values(names.Proj)) |p| rows += @intFromBool(reductionOf(p) == .partial_sum);
    try std.testing.expectEqual(@as(usize, 4), rows);
    try std.testing.expectEqual(Reduction.partial_sum, reductionOf(.o));
    try std.testing.expectEqual(Reduction.partial_sum, reductionOf(.shared_down));
    try std.testing.expectEqual(Reduction.partial_sum, reductionOf(.w2));
    try std.testing.expectEqual(Reduction.partial_sum, reductionOf(.dense_down));
    try std.testing.expectEqual(Reduction.none, reductionOf(.q));
    try std.testing.expectEqual(Reduction.none, reductionOf(.w1));
}

test "two exchanges a layer, 120 a token, 24 KB a row; routed and shared experts share one" {
    for ([_]u32{ 0, 2, 3, 17, 59 }) |l| {
        const plan = layerPlan(flash, l);
        try std.testing.expectEqual(Site.attn_out, plan[0].site);
        try std.testing.expectEqual(Site.ffn_out, plan[1].site);
        for (plan) |x| {
            try std.testing.expectEqual(Reduction.partial_sum, x.kind);
            try std.testing.expectEqual(@as(usize, 6144), x.width);
        }
    }
    try std.testing.expectEqual(@as(usize, 120), tokenExchanges(flash, 60));
    try std.testing.expectEqual(@as(usize, 8), tokenExchanges(flash, 4));
    try std.testing.expectEqual(@as(usize, 24576), sumSlotBytes(flash, 1));
    try std.testing.expectEqual(@as(usize, 49152), sumGatherBytes(flash, 1, 2));
    // shared expert and routed experts cut the same way, so their rows add into one partial
    try std.testing.expectEqual(reductionOf(.shared_down), reductionOf(.w2));
}

test "Share-based: every axis of every rank of 2 and of 3 covers its whole axis exactly once" {
    for ([_]usize{ 1, 2, 3, 4 }) |world| {
        var kv: usize = 0;
        var q: usize = 0;
        var dense: usize = 0;
        var moe: usize = 0;
        var shared: usize = 0;
        var vocab: usize = 0;
        var idx: usize = 0;
        for (0..world) |r| {
            const s = try shard.Share.of(flash, r, world);
            try std.testing.expectEqual(kv, s.kv_heads.lo);
            try std.testing.expectEqual(q, s.q_heads.lo);
            try std.testing.expectEqual(dense, s.dense.lo);
            try std.testing.expectEqual(moe, s.moe.lo);
            try std.testing.expectEqual(shared, s.shared.lo);
            try std.testing.expectEqual(vocab, s.vocab.lo);
            try std.testing.expectEqual(idx, s.indexRows(flash).lo);
            kv = s.kv_heads.hi;
            q = s.q_heads.hi;
            dense = s.dense.hi;
            moe = s.moe.hi;
            shared = s.shared.hi;
            vocab = s.vocab.hi;
            idx = s.indexRows(flash).hi;
            try std.testing.expectEqual(s.kv_heads.len() * 16, s.q_heads.len()); // a kv group is never split
            try std.testing.expectEqual(s.kv_heads, s.index_heads);
            try std.testing.expectEqual(@as(usize, 0), s.vocab.lo % 128);
        }
        try std.testing.expectEqual(@as(usize, 4), kv);
        try std.testing.expectEqual(@as(usize, 64), q);
        try std.testing.expectEqual(@as(usize, 12288), dense);
        try std.testing.expectEqual(@as(usize, 3072), moe);
        try std.testing.expectEqual(@as(usize, 3072), shared);
        try std.testing.expectEqual(@as(usize, 200064), vocab);
        try std.testing.expectEqual(@as(usize, 512), idx);
    }
}

test "Share-based: the TP2 spans of every projection, rank by rank" {
    const r0 = try shard.Share.of(flash, 0, 2);
    const r1 = try shard.Share.of(flash, 1, 2);
    // columns, no sum
    try spanEq(r0.cut(flash, .q), 0, 4096);
    try spanEq(r1.cut(flash, .q), 4096, 8192);
    try spanEq(r0.cut(flash, .k), 0, 256);
    try spanEq(r1.cut(flash, .v), 256, 512);
    try spanEq(r0.indexRows(flash), 0, 256);
    try spanEq(r1.indexRows(flash), 256, 512);
    try spanEq(r0.cut(flash, .dense_gate), 0, 6144);
    try spanEq(r1.cut(flash, .dense_up), 6144, 12288);
    try spanEq(r0.cut(flash, .shared_gate), 0, 1536);
    try spanEq(r1.cut(flash, .w1), 1536, 3072);
    try spanEq(r1.cut(flash, .w3), 1536, 3072);
    // rows, summed
    try spanEq(r0.cut(flash, .o), 0, 4096);
    try spanEq(r1.cut(flash, .o), 4096, 8192);
    try spanEq(r0.cut(flash, .dense_down), 0, 6144);
    try spanEq(r1.cut(flash, .shared_down), 1536, 3072);
    try spanEq(r0.cut(flash, .w2), 0, 1536);
    // the head: 782 and 781 units of 128
    try spanEq(r0.cut(flash, .head), 0, 782 * 128);
    try spanEq(r1.cut(flash, .head), 782 * 128, 200064);
    // GQA: kv head g's 16 q heads sit on its rank, so there is no attention exchange before o_proj
    for (0..4) |g| {
        const owner: usize = if (g < 2) 0 else 1;
        const s = if (owner == 0) r0 else r1;
        try std.testing.expect(g >= s.kv_heads.lo and g < s.kv_heads.hi);
        try std.testing.expect(g * 16 >= s.q_heads.lo and (g + 1) * 16 <= s.q_heads.hi);
    }
}

test "Share-based: ranks that would be left empty are refused for every axis" {
    try std.testing.expectError(error.EmptyShare, shard.Share.of(flash, 4, 5)); // kv heads
    var c = flash.*;
    c.moe_inter = 128;
    try std.testing.expectError(error.EmptyShare, shard.Share.of(&c, 1, 2));
    c = flash.*;
    c.shared_inter = 128;
    try std.testing.expectError(error.EmptyShare, shard.Share.of(&c, 1, 2));
    c = flash.*;
    c.dense_inter = 128;
    try std.testing.expectError(error.EmptyShare, shard.Share.of(&c, 1, 2));
    c = flash.*;
    c.vocab = 128;
    try std.testing.expectError(error.EmptyShare, shard.Share.of(&c, 1, 2));
    c = flash.*;
    c.vocab = 200000; // not whole units
    try std.testing.expectError(error.BadSplit, shard.Share.of(&c, 0, 2));
}

test "the logits gather pads rank 1's slot by one unit and merges back the whole vocabulary" {
    const g = try LogitsGather.of(flash, 2);
    try std.testing.expectEqual(@as(usize, 782 * 128), g.slot);
    try std.testing.expectEqual(@as(usize, 2 * 782 * 128 * 4), g.bytes(1));
    // a small model with the same 782/781 shape of unevenness: 5 units over 2 ranks -> 3 and 2
    var c = flash.*;
    c.vocab = 5 * 128;
    const small = try LogitsGather.of(&c, 2);
    try std.testing.expectEqual(@as(usize, 384), small.slot);
    const rows = 2;
    var gathered: [2 * rows * 384]f32 = @splat(-1); // -1: padding no merge may copy
    for (0..2) |r| {
        const v = (try shard.Share.of(&c, r, 2)).vocab;
        for (0..rows) |row| for (0..v.len()) |i| {
            gathered[(r * rows + row) * 384 + i] = @floatFromInt(row * 10000 + v.lo + i);
        };
    }
    var out: [640]f32 = undefined;
    for (0..rows) |row| {
        try small.merge(&c, &gathered, rows, row, &out);
        for (out, 0..) |x, i| try std.testing.expectEqual(@as(f32, @floatFromInt(row * 10000 + i)), x);
    }
}

test "per-rank argmax merged lowest id first equals argmax over the whole vocabulary, ties included" {
    var logits: [640]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (0..200) |_| {
        for (&logits) |*x| x.* = @floatFromInt(prng.random().intRangeAtMost(i32, -4, 4)); // many ties
        var whole: Pick = .{ .logit = logits[0], .id = 0 };
        for (logits, 0..) |x, i| whole = betterPick(whole, .{ .logit = x, .id = @intCast(i) });
        var c = flash.*;
        c.vocab = 640;
        var picks: [2]Pick = undefined;
        for (0..2) |r| {
            const v = (try shard.Share.of(&c, r, 2)).vocab;
            var best: Pick = .{ .logit = logits[v.lo], .id = @intCast(v.lo) };
            for (v.lo..v.hi) |i| best = betterPick(best, .{ .logit = logits[i], .id = @intCast(i) });
            picks[r] = best;
        }
        try std.testing.expectEqual(whole, mergePicks(&picks));
        // the lowest id with the maximum, spelled out
        const mx = std.mem.max(f32, &logits);
        try std.testing.expectEqual(@as(u32, @intCast(std.mem.indexOfScalar(f32, &logits, mx).?)), whole.id);
    }
}

test "picks: a NaN logit ranks last whatever the rank order, all-NaN goes to the lower id" {
    const nan = std.math.nan(f32);
    const fin: Pick = .{ .logit = -3, .id = 900 };
    const bad: Pick = .{ .logit = nan, .id = 5 };
    try std.testing.expectEqual(fin, mergePicks(&.{ bad, fin }));
    try std.testing.expectEqual(fin, mergePicks(&.{ fin, bad }));
    const bad2: Pick = .{ .logit = nan, .id = 400 };
    try std.testing.expectEqual(@as(u32, 5), mergePicks(&.{ bad2, bad }).id);
    try std.testing.expectEqual(@as(u32, 5), mergePicks(&.{ bad, bad2 }).id);
}

fn bits(x: f32) u32 {
    return @bitCast(x);
}

test "reduction order: both ranks hold identical bits, and the rank-order sum differs from adding into the residual first" {
    // 2^24: adding 1 or 2 to it rounds, so the order of the three terms shows in the bits
    const h: f32 = 16777216.0;
    const p0 = [_]f32{1};
    const p1 = [_]f32{2};
    // rank-order: gather both partials, then (p0 + p1) on each rank, then the residual
    var s0: [1]f32 = undefined;
    var s1: [1]f32 = undefined;
    const gathered = [2]f32{ p0[0], p1[0] }; // the same buffer on both ranks
    rankSum(&gathered, &p0, 2, 0, &s0);
    rankSum(&gathered, &p1, 2, 1, &s1);
    try std.testing.expectEqual(bits(s0[0]), bits(s1[0]));
    const ordered0 = h + s0[0];
    const ordered1 = h + s1[0];
    try std.testing.expectEqual(bits(ordered0), bits(ordered1));
    try std.testing.expectEqual(@as(f32, 16777220.0), ordered0);
    // unordered alternative: each rank folds its own partial into the residual, then the peer's
    const unordered0 = (h + p0[0]) + p1[0];
    const unordered1 = (h + p1[0]) + p0[0];
    try std.testing.expect(bits(unordered0) != bits(unordered1)); // the ranks disagree
    try std.testing.expect(bits(unordered0) != bits(ordered0) or bits(unordered1) != bits(ordered1));
    try std.testing.expectEqual(@as(f32, 16777218.0), unordered0);
    try std.testing.expectEqual(@as(f32, 16777220.0), unordered1);
}

test "two terms commute, so for TP2 the order that matters is the sum before the residual; three terms need rank order" {
    var prng = std.Random.DefaultPrng.init(11);
    for (0..10_000) |_| {
        const a: f32 = prng.random().floatNorm(f32) * 1e-3;
        const b: f32 = prng.random().floatNorm(f32) * 1e6;
        try std.testing.expectEqual(bits(a + b), bits(b + a));
    }
    // world 3: ranks that start from their own term (own, then the others) disagree; rank order does not
    const t = [3]f32{ 16777216.0, 1.0, 1.0 };
    var mine: [3]f32 = undefined;
    for (0..3) |me| {
        // the gathered buffer is the same on every rank; slot `me` is never read, own comes from the rank itself
        var out: [1]f32 = undefined;
        rankSum(&t, &.{t[me]}, 3, me, &out);
        mine[me] = out[0];
    }
    try std.testing.expectEqual(bits(mine[0]), bits(mine[1]));
    try std.testing.expectEqual(bits(mine[1]), bits(mine[2]));
    // own-first rotation: rank 1 adds 1 + 1 + 2^24 = 16777218, rank 0 adds ((2^24 + 1) + 1) = 2^24
    const rot0 = (t[0] + t[1]) + t[2];
    const rot1 = (t[1] + t[2]) + t[0];
    try std.testing.expect(bits(rot0) != bits(rot1));
    try std.testing.expectEqual(bits(rot0), bits(mine[0]));
}

test "rank-order sums are identical on both ranks over random fp32 partials, in both collective forms" {
    var prng = std.Random.DefaultPrng.init(3);
    const rows = 7; // odd: the scatter shares are 4 and 3 rows
    const width = 16;
    const n = rows * width;
    for (0..50) |_| {
        var a: [n]f32 = undefined;
        var b: [n]f32 = undefined;
        for (&a, &b) |*x, *y| {
            x.* = prng.random().floatNorm(f32) * 1e3;
            y.* = prng.random().floatNorm(f32) * 1e-2 * @as(f32, @floatFromInt(prng.random().intRangeAtMost(u32, 1, 1 << 20)));
        }
        // gather_sum: the gathered buffer is [rank0 | rank1] on both ranks
        var gathered: [2 * n]f32 = undefined;
        @memcpy(gathered[0..n], &a);
        @memcpy(gathered[n..], &b);
        var r0: [n]f32 = undefined;
        var r1: [n]f32 = undefined;
        rankSum(&gathered, &a, 2, 0, &r0);
        rankSum(&gathered, &b, 2, 1, &r1);
        try std.testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.asBytes(&r0)), std.mem.bytesAsSlice(u32, std.mem.asBytes(&r1)));
        // the device kernel with peers = the gathered buffer and own aliasing out gives the same bits
        var inplace = a;
        rankSum(&gathered, &inplace, 2, 0, &inplace);
        try std.testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.asBytes(&r0)), std.mem.bytesAsSlice(u32, std.mem.asBytes(&inplace)));
        // scatter_gather: shares summed apart, then concatenated, equals the whole-row sum
        var sg: [n]f32 = undefined;
        scatterGatherSum(2, &.{ &a, &b }, rows, width, &sg);
        try std.testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.asBytes(&r0)), std.mem.bytesAsSlice(u32, std.mem.asBytes(&sg)));
        // and a plain left-to-right host sum is what both are
        for (0..n) |i| try std.testing.expectEqual(bits(a[i] + b[i]), bits(r0[i]));
    }
}

test "row shares match rank_scatter.zig's and the form switches at the decode window" {
    for ([_]usize{ 0, 1, 7, 64, 2047, 2048 }) |rows| for (1..5) |world| {
        var at: usize = 0;
        for (0..world) |r| {
            const s = rowShare(rows, world, r);
            try std.testing.expectEqual(at, s[0]);
            try std.testing.expectEqual(rows / world + @intFromBool(r < rows % world), s[1] - s[0]);
            at = s[1];
        }
        try std.testing.expectEqual(rows, at);
    };
    try std.testing.expectEqual([2]usize{ 683, 1365 }, rowShare(2047, 3, 1));
    try std.testing.expectEqual(Form.gather_sum, formFor(1));
    try std.testing.expectEqual(Form.gather_sum, formFor(128));
    try std.testing.expectEqual(Form.scatter_gather, formFor(129));
}
