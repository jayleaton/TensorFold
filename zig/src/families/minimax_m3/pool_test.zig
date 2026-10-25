//! Page geometry, block tables and the memory budget on the shipped model's numbers (plan sections 2 and 5).
const std = @import("std");
const kv = @import("core").kv_pool;
const config = @import("config.zig");
const shard = @import("shard.zig");
const pool = @import("pool.zig");
const budget = @import("budget.zig");

const gpa = std.testing.allocator;
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;
const GiB = budget.GiB;

/// The real plan's slotBytes() of rank 0 and 1 (load_test.zig, the full pack's index: 91.0855 GiB and 91.0850 GiB).
pub const weights_rank0: u64 = 97_801_819_136;
pub const weights_rank1: u64 = 97_801_229_056;

pub fn dims(rank: usize) !pool.Dims {
    const c = config.flash;
    return pool.Dims.of(&c, try shard.Share.of(&c, rank, 2));
}

test "a rank caches 76,800 bytes a token and a page is 9,830,400" {
    for ([_]usize{ 0, 1 }) |r| {
        const d = try dims(r);
        try expectEqual(@as(u64, 76_800), d.tokenBytes());
        try expectEqual(@as(u64, 9_830_400), d.pageBytes());
        try expectEqual(@as(u64, 1280), d.layerToken()); // K 512 + V 512 + IK 256
        const l = d.layout(1024);
        try expectEqual(@as(u64, 76_800), l.token_bytes);
        try expectEqual(@as(u32, 128), l.page_tokens);
        try expectEqual(@as(u64, 4096), l.fixed_bytes); // the block-table row
    }
    // one rank alone caches all four kv heads
    const c = config.flash;
    const whole = try pool.Dims.of(&c, try shard.Share.of(&c, 0, 1));
    try expectEqual(@as(u64, 138_240), whole.tokenBytes());
}

test "a selection block that is not a page is refused" {
    var c = config.flash;
    c.index.block = 64;
    try std.testing.expectError(error.BlockIsNotPage, pool.Dims.of(&c, try shard.Share.of(&c, 0, 2)));
}

test "the page map tiles the page: K, V and IK of every layer, head-major, no gap and no overlap" {
    const d = try dims(0);
    const m = d.map();
    const Region = struct { at: u64, len: u64 };
    var regions: std.ArrayList(Region) = .empty;
    defer regions.deinit(gpa);
    const head = @as(u64, pool.page_tokens) * d.head_dim * 2;
    for (0..d.layers) |layer| {
        for (0..d.kv_heads) |h| {
            try regions.append(gpa, .{ .at = m.k(@intCast(layer), @intCast(h)), .len = head });
            try regions.append(gpa, .{ .at = m.v(@intCast(layer), @intCast(h)), .len = head });
        }
        try regions.append(gpa, .{ .at = m.ik(@intCast(layer)), .len = @as(u64, pool.page_tokens) * d.index_dim * 2 });
    }
    try expectEqual(@as(usize, 60 * 5), regions.items.len);
    std.mem.sort(Region, regions.items, {}, struct {
        fn less(_: void, a: Region, b: Region) bool {
            return a.at < b.at;
        }
    }.less);
    var end: u64 = 0;
    for (regions.items) |r| {
        try expectEqual(end, r.at);
        end = r.at + r.len;
    }
    try expectEqual(d.pageBytes(), end);
    // one block's keys of a head are contiguous: row t of head h starts t rows after its first
    try expectEqual(m.k(7, 1) - m.k(7, 0), head);
    try expectEqual(m.page(3), 3 * d.pageBytes());
    const at = pool.locate(129);
    try expectEqual(@as(usize, 1), at.block);
    try expectEqual(@as(u32, 1), at.row);
}

test "block tables: page ids by block, -1 past the stream, a failed upload leaves the row as it was" {
    var t = try pool.Tables.init(gpa, 2, 8);
    defer t.deinit();
    for (t.rows) |e| try expectEqual(@as(i32, -1), e);
    try t.write(1, &.{ 9, 4, 17 }, 0);
    try t.write(1, &.{5}, 3);
    try expectEqual(@as(?u32, 9), t.page(1, 0));
    try expectEqual(@as(?u32, 5), t.page(1, 3));
    try expectEqual(@as(?u32, null), t.page(1, 4));
    try expectEqual(@as(?u32, null), t.page(0, 0));
    try expectEqual(@as(?u32, null), t.page(1, 8)); // past the row
    try std.testing.expectError(error.TableFull, t.write(1, &.{ 1, 2, 3, 4, 5 }, 4));
    try std.testing.expectError(error.BadSlot, t.write(2, &.{1}, 0));
    const Fail = struct {
        fn call(_: *anyopaque, _: u32, _: usize, _: []const i32) anyerror!void {
            return error.UploadFailed;
        }
    };
    var dummy: u8 = 0;
    t.upload = .{ .ctx = &dummy, .call = Fail.call };
    try std.testing.expectError(error.UploadFailed, t.write(1, &.{ 70, 71 }, 1));
    try expectEqual(@as(?u32, 4), t.page(1, 1));
    try expectEqual(@as(?u32, 17), t.page(1, 2));
    t.clear(1);
    try expectEqual(@as(?u32, null), t.page(1, 0));
}

test "the kernel's selection lists index the table: block b of a stream is page row[b]" {
    var p = try kv.Pool.init(gpa, 16, 9_830_400);
    defer p.deinit();
    var t = try pool.Tables.init(gpa, 1, 16);
    defer t.deinit();
    var s = t.sink(0);
    var l: kv.Lease = .{};
    try p.promise(&l, 5);
    try p.cover(&l, 3, s.table());
    try p.cover(&l, 5, s.table()); // the second call writes entries 3 and 4 only
    for (l.ids.items, 0..) |id, b| try expectEqual(@as(?u32, id), t.page(0, b));
    try expectEqual(@as(?u32, null), t.page(0, 5));
    p.finish(&l);
    try expect(p.idle());
}

// ---- the budget ----------------------------------------------------------------------------------------------------

fn input(avail_gib: f64, terms: budget.Terms) budget.Input {
    return .{ .available = @intFromFloat(avail_gib * @as(f64, @floatFromInt(GiB))), .weights = weights_rank0, .terms = terms };
}

test "119 GiB, TensorFold's reserve: the largest context and pool" {
    const d = try dims(0);
    const total: u64 = 119 * GiB;
    const terms = try budget.tensorfold(total, null);
    try expectEqual(@as(u64, 119 * GiB / 10), terms.reserve);
    const in = input(119, terms);
    const v = budget.check(d, in, null);
    const cap = v.ok;
    // 119 - 11.9 - 91.0855 - 4.5 - 1.5 - score chunk (8192 x 2 x 1k blocks x 4 B = 64 MiB) leaves about 9.9 GiB
    try expect(cap.context > 128 * 1024 and cap.context < 140_000);
    try expectEqual(@as(u64, 0), (cap.context + in.window) % 128);
    try expect(cap.pool_tokens >= cap.context + in.window);
    try expect(cap.pool_bytes <= cap.room);
    try expect(cap.room - cap.pool_bytes < 2 * d.pageBytes() + 4096); // nothing left on the table but a page
    // the largest is the largest: one more row needs one more page, which the budget lacks
    try expect(budget.check(d, in, cap.context).ok.context == cap.context);
    const over = budget.check(d, in, cap.context + 1);
    try expectEqual(budget.Reason.does_not_fit, over.refused.reason);
    try expectEqual(cap.context, over.refused.largest);
    try expectEqual(cap.pool_tokens, over.refused.largest_pool_tokens);
    // the whole budget adds up
    const spent = in.weights + terms.reserve + terms.runtime + cap.workspace + cap.pool_bytes;
    try expect(spent <= in.available);
    std.debug.print("\n[119 GiB tensorfold] context {d} pool {d} tokens {d:.3} GiB room {d:.3} GiB workspace {d:.3} GiB\n", .{ cap.context, cap.pool_tokens, @as(f64, @floatFromInt(cap.pool_bytes)) / GiB, @as(f64, @floatFromInt(cap.room)) / GiB, @as(f64, @floatFromInt(cap.workspace)) / GiB });
}

test "119 GiB under the DeepSeek sibling's floor conventions" {
    const d = try dims(0);
    const in = input(119, budget.dsv41());
    const cap = budget.check(d, in, null).ok;
    // 119 - 4 floor - 1 runtime - 4 workspace - 0.25 - 0.125 - 91.0855 leaves 18.5 GiB
    try expect(cap.context > 240_000 and cap.context < 270_000);
    std.debug.print("[119 GiB dsv41] context {d} pool {d} tokens {d:.3} GiB\n", .{ cap.context, cap.pool_tokens, @as(f64, @floatFromInt(cap.pool_bytes)) / GiB });
}

test "rank 1 has the same KV and 590 KiB less weights" {
    const terms = try budget.tensorfold(119 * GiB, null);
    var in = input(119, terms);
    const a = budget.check(try dims(0), in, null).ok;
    in.weights = weights_rank1;
    const b = budget.check(try dims(1), in, null).ok;
    try expect(b.pool_tokens >= a.pool_tokens);
    try expect(b.pool_tokens - a.pool_tokens <= 2 * 128);
}

test "other budgets: the sizes the plan names, 64 and 128 GiB hosts" {
    const d = try dims(0);
    for ([_]f64{ 105, 110, 119, 124 }) |avail| {
        const terms = try budget.tensorfold(@intFromFloat(avail * @as(f64, @floatFromInt(GiB))), null);
        const v = budget.check(d, input(avail, terms), null);
        switch (v) {
            .ok => |c| std.debug.print("[{d} GiB tensorfold] context {d}\n", .{ avail, c.context }),
            .refused => |r| std.debug.print("[{d} GiB tensorfold] refused {s}\n", .{ avail, @tagName(r.reason) }),
        }
    }
}

test "a declared 1,048,576 context cannot fit: the refusal names the largest that does" {
    const d = try dims(0);
    const in = input(119, try budget.tensorfold(119 * GiB, null));
    const best = budget.check(d, in, null).ok;
    const r = budget.check(d, in, 1 << 20).refused;
    try expectEqual(budget.Reason.does_not_fit, r.reason);
    try expectEqual(best.context, r.largest);
    // 1 M rows are 75 GiB of KV (plan section 5): the text says so and what to set
    try expect(r.need_bytes > 74 * GiB and r.need_bytes < 77 * GiB);
    const text = try budget.refusalText(gpa, r);
    defer gpa.free(text);
    var want: [96]u8 = undefined;
    try expect(std.mem.indexOf(u8, text, "CONTEXT=1048576") != null);
    try expect(std.mem.indexOf(u8, text, try std.fmt.bufPrint(&want, "largest context that fits is {d} tokens", .{best.context})) != null);
    try expect(std.mem.indexOf(u8, text, "76800 bytes a token") != null);
    try expect(std.mem.indexOf(u8, text, "set CONTEXT to at most") != null);
    // 65,536 and 131,072 are the contexts the plan advertises: the first fits, the second fits at 119 GiB
    try expect(budget.check(d, in, 65_536) == .ok);
    try expect(budget.check(d, in, 131_072) == .ok);
}

test "refusals: past the model, no room beside the weights, too many streams" {
    const d = try dims(0);
    const in = input(119, try budget.tensorfold(119 * GiB, null));
    const beyond = budget.check(d, in, 2_000_000).refused;
    try expectEqual(budget.Reason.beyond_model, beyond.reason);
    {
        const text = try budget.refusalText(gpa, beyond);
        defer gpa.free(text);
        try expect(std.mem.indexOf(u8, text, "past the model's 1048576 positions") != null);
    }
    var small = input(95, try budget.tensorfold(95 * GiB, null)); // weights 91.1 + reserve 9.5 > 95
    const none = budget.check(d, small, null).refused;
    try expectEqual(budget.Reason.no_room, none.reason);
    try expectEqual(@as(u64, 0), none.largest);
    {
        const text = try budget.refusalText(gpa, budget.check(d, small, 4096).refused);
        defer gpa.free(text);
        try expect(std.mem.indexOf(u8, text, "leaves no KV room") != null);
    }
    small = input(119, try budget.tensorfold(119 * GiB, null));
    small.lanes = 5000;
    small.lanes_fixed = true;
    try expectEqual(budget.Reason.too_many_lanes, budget.check(d, small, 8192).refused.reason);
    // unfixed lanes serve fewer: the plan's pool still holds a page for each granted stream
    small.lanes = 4;
    small.lanes_fixed = false;
    const cap = budget.check(d, small, null).ok;
    try expect(cap.lanes >= 1 and cap.lanes <= 4);
    try expect(cap.pages >= (cap.context + small.window) / 128 + cap.lanes - 1);
}

test "the reserve variable: a tenth and a 4 GiB floor, 2 GiB or more, never past the memory" {
    try expectEqual(@as(u64, 12 * GiB), try budget.reserveBytes("", 120 * GiB));
    try expectEqual(@as(u64, 4 * GiB), try budget.reserveBytes(null, 24 * GiB));
    try expectEqual(@as(u64, 3 * GiB), try budget.reserveBytes("3", 120 * GiB));
    try std.testing.expectError(error.BadReserve, budget.reserveBytes("1", 120 * GiB));
    try std.testing.expectError(error.BadReserve, budget.reserveBytes("200", 120 * GiB));
    try std.testing.expectError(error.BadReserve, budget.reserveBytes("x", 120 * GiB));
    // a smaller reserve buys context
    const d = try dims(0);
    const big = budget.check(d, input(119, try budget.tensorfold(119 * GiB, "4")), null).ok;
    const tenth = budget.check(d, input(119, try budget.tensorfold(119 * GiB, null)), null).ok;
    try expect(big.context > tenth.context);
}
