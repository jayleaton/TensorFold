//! A TP rank's share of MiniMax-M3.1-Flash (plan section 5): whole kv groups, EXL3 and vocabulary cuts in 128s.
const std = @import("std");
const canon = @import("canon");
const layout = @import("core").exl3_layout;
const Config = @import("config.zig").Config;
const names = @import("names.zig");

/// Elements of an EXL3 width or of the vocabulary a rank takes whole: the Hadamard block.
pub const unit = 128;

pub const Span = struct {
    lo: usize,
    hi: usize,

    pub fn len(s: Span) usize {
        return s.hi - s.lo;
    }
};

pub const Share = struct {
    rank: usize,
    world: usize,
    kv_heads: Span, // kv heads; index head g selects for kv head g, so the index heads are the same span
    q_heads: Span, // the query heads of those kv groups
    index_heads: Span,
    dense: Span, // dense MLP intermediate rows (layers before the first MoE layer)
    moe: Span, // each routed expert's intermediate rows
    shared: Span, // the shared expert's intermediate rows
    vocab: Span, // the head's rows, in whole 128-row units

    /// Rank `rank` of `world`; refused where any rank would be left without a head, a unit or a row.
    pub fn of(c: *const Config, rank: usize, world: usize) !Share {
        if (world == 0 or rank >= world) return error.BadRank;
        const kv = try span(c.kv_heads, 1, rank, world);
        return .{
            .rank = rank,
            .world = world,
            .kv_heads = kv,
            .q_heads = .{ .lo = kv.lo * c.group(), .hi = kv.hi * c.group() },
            .index_heads = kv,
            .dense = try span(c.dense_inter, unit, rank, world),
            .moe = try span(c.moe_inter, unit, rank, world),
            .shared = try span(c.shared_inter, unit, rank, world),
            .vocab = try span(c.vocab, unit, rank, world),
        };
    }

    /// The rank's slice of matrix `p` along its cut axis, in elements of that axis.
    pub fn cut(s: Share, c: *const Config, p: names.Proj) Span {
        return switch (p) {
            .q, .o => scale(s.q_heads, c.head_dim),
            .k, .v => scale(s.kv_heads, c.head_dim),
            .dense_gate, .dense_up, .dense_down => s.dense,
            .shared_gate, .shared_up, .shared_down => s.shared,
            .w1, .w3, .w2 => s.moe,
            .head => s.vocab,
        };
    }

    /// The rows of the fp16 index query projection this rank scores with; the key projection is replicated.
    pub fn indexRows(s: Share, c: *const Config) Span {
        return scale(s.index_heads, c.index.dim);
    }

    /// The rank's slice of an EXL3 group: whole tiles, with `suh` or `svh` cut to match.
    pub fn part(s: Share, c: *const Config, p: names.Proj, g: layout.Group) !layout.Part {
        return cutGroup(g, p.cut(), s.cut(c, p));
    }
};

fn scale(s: Span, by: usize) Span {
    return .{ .lo = s.lo * by, .hi = s.hi * by };
}

/// Rank `rank`'s [lo, hi) of `total` elements in whole `per`s: largest-remainder shares, ties to the lower rank.
fn span(total: usize, per: usize, rank: usize, world: usize) !Span {
    if (total % per != 0 or world > 256) return error.BadSplit;
    const ones: [256]u64 = @splat(1);
    var out: [256]canon.Range = undefined;
    canon.partition(@intCast(total / per), ones[0..world], out[0..world]);
    const r = out[rank];
    if (r.len() == 0) return error.EmptyShare;
    return .{ .lo = r.begin * per, .hi = r.end * per };
}

/// `g`'s tiles [lo, hi) along `axis` (128-aligned elements): the trellis runs and the scale bytes to read.
pub fn cutGroup(g: layout.Group, axis: names.Cut, s: Span) !layout.Part {
    const whole = if (axis == .col) g.n() else g.k();
    if (s.lo >= s.hi or s.hi > whole or s.lo % unit != 0 or s.hi % unit != 0) return error.SplitNotOnBlocks;
    const lo: u64 = s.lo / 16;
    const per: u64 = s.len() / 16;
    var p: layout.Part = .{
        .group = g,
        .trellis = .{ .offset = 0, .len = g.trellisBytes() },
        .suh = .{ .offset = 0, .len = 2 * @as(u64, g.k()) },
        .svh = .{ .offset = 0, .len = 2 * @as(u64, g.n()) },
    };
    switch (axis) {
        .row => { // K tiles lo..hi are whole rows of N tiles: one run
            const row = @as(u64, g.n_tiles) * g.tileBytes();
            p.group.k_tiles = @intCast(per);
            p.trellis = .{ .offset = lo * row, .len = per * row };
            p.suh = .{ .offset = 2 * 16 * lo, .len = 2 * 16 * per };
        },
        .col => { // this rank's N tiles are a run inside each of the K tile rows
            p.group.n_tiles = @intCast(per);
            p.trellis = .{ .offset = lo * g.tileBytes(), .len = per * g.tileBytes(), .stride = @as(u64, g.n_tiles) * g.tileBytes(), .count = g.k_tiles };
            p.svh = .{ .offset = 2 * 16 * lo, .len = 2 * 16 * per };
        },
    }
    return p;
}

test "TP2 splits the heads, widths and vocabulary as plan section 5 says" {
    const c = &@import("config.zig").flash;
    const r0 = try Share.of(c, 0, 2);
    const r1 = try Share.of(c, 1, 2);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 2 }, r0.kv_heads);
    try std.testing.expectEqual(Span{ .lo = 2, .hi = 4 }, r1.kv_heads);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 32 }, r0.q_heads);
    try std.testing.expectEqual(Span{ .lo = 32, .hi = 64 }, r1.q_heads);
    try std.testing.expectEqual(r1.kv_heads, r1.index_heads);
    try std.testing.expectEqual(Span{ .lo = 256, .hi = 512 }, r1.indexRows(c));
    try std.testing.expectEqual(Span{ .lo = 6144, .hi = 12288 }, r1.dense);
    try std.testing.expectEqual(Span{ .lo = 1536, .hi = 3072 }, r1.moe);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 1536 }, r0.shared);
    // 1563 units of 128 rows: rank 0 takes the odd one
    try std.testing.expectEqual(@as(usize, 782 * 128), r0.vocab.len());
    try std.testing.expectEqual(@as(usize, 781 * 128), r1.vocab.len());
    try std.testing.expectEqual(r0.vocab.hi, r1.vocab.lo);
    try std.testing.expectEqual(c.vocab, r1.vocab.hi);
    // q_proj and o_proj: 4096 of 8192 each; k and v: 256 of 512
    try std.testing.expectEqual(Span{ .lo = 4096, .hi = 8192 }, r1.cut(c, .q));
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 4096 }, r0.cut(c, .o));
    try std.testing.expectEqual(Span{ .lo = 256, .hi = 512 }, r1.cut(c, .v));
}

test "ranks that would be empty and ranks out of range are refused" {
    const c = &@import("config.zig").flash;
    try std.testing.expectError(error.BadRank, Share.of(c, 2, 2));
    try std.testing.expectError(error.BadRank, Share.of(c, 0, 0));
    // 4 kv heads: a fifth rank has none
    try std.testing.expectError(error.EmptyShare, Share.of(c, 4, 5));
    try std.testing.expectError(error.EmptyShare, Share.of(c, 4, 8));
    try std.testing.expectError(error.BadSplit, Share.of(c, 0, 257));
    // three ranks: uneven but whole groups, 2/1/1 kv heads
    const t0 = try Share.of(c, 0, 3);
    const t2 = try Share.of(c, 2, 3);
    try std.testing.expectEqual(@as(usize, 32), t0.q_heads.len());
    try std.testing.expectEqual(@as(usize, 16), t2.q_heads.len());
    // a model whose MoE width has fewer 128-units than ranks
    var small = c.*;
    small.moe_inter = 128;
    try std.testing.expectError(error.EmptyShare, Share.of(&small, 1, 2));
    small.moe_inter = 100;
    try std.testing.expectError(error.BadSplit, Share.of(&small, 0, 2));
}

test "group cuts agree with core's even cuts and keep the scales aligned" {
    const c = &@import("config.zig").flash;
    const q = try layout.Group.fromShape(&.{ 384, 512, 96 }); // q_proj: K 6144, N 8192, 6 bits
    const o = try layout.Group.fromShape(&.{ 512, 384, 96 }); // o_proj: K 8192, N 6144
    for ([_]usize{ 0, 1 }) |r| {
        const s = try Share.of(c, r, 2);
        const mine = try s.part(c, .q, q);
        const theirs = try q.part(.col, @intCast(r), 2);
        try std.testing.expectEqualDeep(theirs, mine);
        try std.testing.expectEqualDeep(try o.part(.row, @intCast(r), 2), try s.part(c, .o, o));
    }
    const r1 = try (try Share.of(c, 1, 2)).part(c, .o, o);
    try std.testing.expectEqual(@as(u64, 4096 * 2), r1.suh.len);
    try std.testing.expectEqual(@as(u64, 4096 * 2), r1.suh.offset);
    try std.testing.expectEqual(@as(u64, 6144 * 2), r1.svh.len); // the row cut keeps svh whole
    try std.testing.expectEqual(@as(u32, 256), r1.group.k_tiles);
}

test "the head's odd unit count cuts at 128s that core's even split refuses" {
    const c = &@import("config.zig").flash;
    const head = try layout.Group.fromShape(&.{ 384, 12504, 96 });
    try std.testing.expectError(error.SplitNotOnBlocks, head.part(.col, 0, 2)); // 1563 units are not even
    const p0 = try (try Share.of(c, 0, 2)).part(c, .head, head);
    const p1 = try (try Share.of(c, 1, 2)).part(c, .head, head);
    try std.testing.expectEqual(@as(u32, 782 * 8), p0.group.n_tiles);
    try std.testing.expectEqual(@as(u32, 781 * 8), p1.group.n_tiles);
    try std.testing.expectEqual(p0.group.n_tiles * 16 * 2, p0.svh.len);
    try std.testing.expectEqual(p0.svh.len, p1.svh.offset);
    try std.testing.expectEqual(@as(u64, 384), p1.trellis.count);
    // together the two runs cover each whole row of N tiles, back to back
    try std.testing.expectEqual(head.tileBytes() * 12504, p0.trellis.len + p1.trellis.len);
    try std.testing.expectEqual(p0.trellis.len, p1.trellis.offset);
}

test "cuts off the 128 grid or out of range are refused" {
    const g = try layout.Group.fromShape(&.{ 16, 16, 64 }); // 256 x 256
    try std.testing.expectError(error.SplitNotOnBlocks, cutGroup(g, .col, .{ .lo = 0, .hi = 100 }));
    try std.testing.expectError(error.SplitNotOnBlocks, cutGroup(g, .row, .{ .lo = 128, .hi = 384 }));
    try std.testing.expectError(error.SplitNotOnBlocks, cutGroup(g, .row, .{ .lo = 128, .hi = 128 }));
    _ = try cutGroup(g, .col, .{ .lo = 128, .hi = 256 });
}
