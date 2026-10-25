//! The fp8 KV cache's family side on the shipped model's numbers (docs/fp8-kv.md): pages, map, budget, fingerprint.
const std = @import("std");
const kv = @import("core").kv_pool;
const config = @import("config.zig");
const shard = @import("shard.zig");
const pool = @import("pool.zig");
const budget = @import("budget.zig");
const ranks = @import("ranks.zig");

const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;
const GiB = budget.GiB;

fn dims(rank: usize, kv8: bool) !pool.Dims {
    const c = config.flash;
    return (try pool.Dims.of(&c, try shard.Share.of(&c, rank, 2))).withKv8(kv8);
}

test "a rank caches 47,040 bytes a token with fp8 K/V and a page is 6,021,120; bf16 stays 76,800 and 9,830,400" {
    for ([_]usize{ 0, 1 }) |r| {
        const d = try dims(r, true);
        try expectEqual(@as(u64, 784), d.layerToken()); // K 256 + V 256 + IK 256 + 4 scales of 4 bytes
        try expectEqual(@as(u64, 47_040), d.tokenBytes());
        try expectEqual(@as(u64, 6_021_120), d.pageBytes());
        const l = d.layout(1024);
        try expectEqual(@as(u64, 47_040), l.token_bytes);
        try expectEqual(@as(u64, 4096), l.fixed_bytes); // the block-table row is the same
        const b = try dims(r, false);
        try expectEqual(@as(u64, 76_800), b.tokenBytes());
        try expectEqual(@as(u64, 9_830_400), b.pageBytes());
    }
    // one rank alone caches all four kv heads
    const c = config.flash;
    const whole = (try pool.Dims.of(&c, try shard.Share.of(&c, 0, 1))).withKv8(true);
    try expectEqual(@as(u64, 60 * (2 * 4 * (128 + 4) + 256)), whole.tokenBytes());
}

test "the fp8 page map tiles the page: K and V codes, IK, then the scales of every layer, no gap and no overlap" {
    const d = try dims(0, true);
    const m = d.map();
    const page: usize = @intCast(d.pageBytes());
    const seen = try std.testing.allocator.alloc(u8, page);
    defer std.testing.allocator.free(seen);
    @memset(seen, 0);
    const head = @as(u64, pool.page_tokens) * d.head_dim; // one byte a value
    const mark = struct {
        fn at(s: []u8, off: u64, len: u64) void {
            for (s[@intCast(off)..][0..@intCast(len)]) |*x| {
                std.debug.assert(x.* == 0);
                x.* = 1;
            }
        }
    }.at;
    for (0..d.layers) |layer| {
        for (0..d.kv_heads) |h| {
            mark(seen, m.k(@intCast(layer), @intCast(h)), head);
            mark(seen, m.v(@intCast(layer), @intCast(h)), head);
            mark(seen, m.kScale(@intCast(layer), @intCast(h)), pool.page_tokens * pool.scale_bytes);
            mark(seen, m.vScale(@intCast(layer), @intCast(h)), pool.page_tokens * pool.scale_bytes);
        }
        mark(seen, m.ik(@intCast(layer)), @as(u64, pool.page_tokens) * d.index_dim * 2);
    }
    for (seen) |s| try expectEqual(@as(u8, 1), s);
    // the order inside a layer: K heads, V heads, IK, K scales, V scales (the kernels' kv8_byte, ik_byte, scale8_byte)
    try expectEqual(@as(u64, 0), m.k(0, 0));
    try expectEqual(@as(u64, 2 * 16384), m.v(0, 0));
    try expectEqual(@as(u64, 2 * 2 * 16384), m.ik(0));
    try expectEqual(m.ik(0) + 32768, m.kScale(0, 0));
    try expectEqual(m.k(1, 0), m.vScale(0, 1) + 512); // the next layer starts right after the last scale
}

test "the same memory holds 1.63 times the pages: the real rank budgets, 7.18 and 7.02 GiB of KV room" {
    const bf = try dims(0, false);
    const f8 = try dims(0, true);
    const Row = struct { room_gib: f64, bf16_pages: usize, fp8_pages: usize };
    const want = [_]Row{ .{ .room_gib = 7.18, .bf16_pages = 784, .fp8_pages = 1280 }, .{ .room_gib = 7.02, .bf16_pages = 766, .fp8_pages = 1251 } };
    for (want) |w| {
        const room: u64 = @intFromFloat(w.room_gib * @as(f64, @floatFromInt(GiB)));
        // a layout without a stream table or cache, the pages that remain: room / page
        const lb: kv.Layout = .{ .fixed_bytes = 0, .token_bytes = bf.tokenBytes(), .page_tokens = pool.page_tokens, .snap_bytes = pool.snap_bytes, .all_paged = true };
        const lf: kv.Layout = .{ .fixed_bytes = 0, .token_bytes = f8.tokenBytes(), .page_tokens = pool.page_tokens, .snap_bytes = pool.snap_bytes, .all_paged = true };
        const pb = try kv.plan(lb, .{ .room = room, .engine = 0, .lanes = 1 });
        const pf = try kv.plan(lf, .{ .room = room, .engine = 0, .lanes = 1 });
        try expectEqual(w.bf16_pages, pb.pages);
        try expectEqual(w.fp8_pages, pf.pages);
        std.debug.print("kv8 room {d:.2} GiB: bf16 {d} pages ({d} tokens), fp8 {d} pages ({d} tokens)\n", .{ w.room_gib, pb.pages, pb.pages * 128, pf.pages, pf.pages * 128 });
    }
    // the 784 pages the first rank was served are exactly 1280 fp8 pages of memory
    try expectEqual(@as(u64, 784) * bf.pageBytes(), @as(u64, 1280) * f8.pageBytes());
}

test "the budget check admits a longer context with fp8 K/V in the same room, and names fp8 bytes in a refusal" {
    const w = @import("pool_test.zig").weights_rank0;
    const bf = try dims(0, false);
    const f8 = try dims(0, true);
    const in: budget.Input = .{ .available = 119 * GiB, .weights = w, .terms = try budget.tensorfold(119 * GiB, null), .lanes = 1, .window = pool.page_tokens, .max_positions = 1 << 20 };
    const v16 = budget.check(bf, in, null).ok;
    const v8 = budget.check(f8, in, null).ok;
    try expect(v8.context > v16.context);
    try expect(v8.pool_bytes <= v8.room and v16.pool_bytes <= v16.room);
    try expect(@as(f64, @floatFromInt(v8.context)) > 1.6 * @as(f64, @floatFromInt(v16.context)) and @as(f64, @floatFromInt(v8.context)) < 1.7 * @as(f64, @floatFromInt(v16.context)));
    std.debug.print("kv8 119 GiB host, TensorFold reserve: largest context bf16 {d}, fp8 {d}\n", .{ v16.context, v8.context });
    // a context bf16 cannot hold fits with fp8; the refusal of a bigger one quotes 47,040 bytes a token
    const over = budget.check(bf, in, v8.context);
    try expect(over == .refused);
    try expect(budget.check(f8, in, v8.context) == .ok);
    const huge = budget.check(f8, in, 1 << 20);
    try expect(huge == .refused);
    try expectEqual(@as(u64, 47_040), huge.refused.token_bytes);
}

test "the ranks' fingerprint carries kv8: one rank on fp8 pages and the other on bf16 refuse each other" {
    try expectEqual(false, (ranks.Settings{}).kv8);
    const h = ranks.hashBytes("program");
    const a = ranks.Fingerprint.of(.{ .rank = 0, .world = 2 }, .{ .kv8 = true }, 131072, ranks.hashBytes("c"), ranks.hashBytes("q"), h, ranks.hashBytes("w"));
    var b = ranks.Fingerprint.of(.{ .rank = 1, .world = 2 }, .{ .kv8 = true }, 131072, ranks.hashBytes("c"), ranks.hashBytes("q"), h, ranks.hashBytes("w"));
    var all = [_]ranks.Fingerprint{ a, b };
    try expect(ranks.mismatch(&all) == null);
    b = ranks.Fingerprint.of(.{ .rank = 1, .world = 2 }, .{}, 131072, ranks.hashBytes("c"), ranks.hashBytes("q"), h, ranks.hashBytes("w"));
    all[1] = b;
    const mm = ranks.mismatch(&all).?;
    try expectEqual(ranks.Fingerprint.Field.kv8, mm.field);
    try expectEqual(@as(usize, 1), mm.rank);
    try expect(std.mem.indexOf(u8, ranks.explain(.kv8), "--kv-cache") != null);
}
