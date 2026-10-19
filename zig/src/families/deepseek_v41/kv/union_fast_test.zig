//! union_fast.Planner against split.planUnion: the same block bytes on random and edge selections, one reused planner.

const std = @import("std");
const split = @import("split.zig");
const uf = @import("union_fast.zig");
const testing = std.testing;

/// One case through both plans: planUnion's one block, the planner's m, phys and tokens equal (or both several blocks).
fn same(p: *uf.Planner, in: split.UnionIn) !void {
    const gpa = testing.allocator;
    var blocks: std.ArrayList(split.Block) = .empty;
    defer {
        for (blocks.items) |*b| b.deinit(gpa);
        blocks.deinit(gpa);
    }
    try split.planUnion(gpa, in, &blocks);
    const m = try p.measure(gpa, in) orelse return testing.expect(blocks.items.len > 1);
    try testing.expectEqual(@as(usize, 1), blocks.items.len);
    const b = blocks.items[0];
    try testing.expectEqual(b.m, m);
    try testing.expect(m <= @max(1, uf.ownerBound(in.table.len, in.psh, in.world, in.sel.len)));
    const phys = try gpa.alloc(u32, m);
    defer gpa.free(phys);
    const tokens = try gpa.alloc(i32, in.sel.len / in.k * in.k);
    defer gpa.free(tokens);
    @memset(phys, 0xAAAA_AAAA);
    p.write(in, phys, tokens);
    try testing.expectEqualSlices(u32, b.phys, phys);
    try testing.expectEqualSlices(i32, b.tokens, tokens);
    for (p.seen.items) |x| try testing.expectEqual(@as(u32, 0), x);
}

fn table(gpa: std.mem.Allocator, pages: usize, seed: u64) ![]u32 {
    const t = try gpa.alloc(u32, pages);
    var rng = std.Random.DefaultPrng.init(seed);
    for (t) |*x| x.* = rng.random().uintLessThan(u32, 1 << 16);
    return t;
}

test "planner: random selections at world 1, 2, 3 and 8, every rank, page shifts 6 and 1" {
    const gpa = testing.allocator;
    var p: uf.Planner = .{};
    defer p.deinit(gpa);
    var rng = std.Random.DefaultPrng.init(0x5eed);
    const r = rng.random();
    for ([_]u32{ 1, 2, 3, 8 }) |world| for ([_]u32{ 6, 1 }) |psh| for (0..world) |rank| for (0..4) |round| {
        const pages = 1 + r.uintLessThan(usize, 40);
        const tab = try table(gpa, pages, round);
        defer gpa.free(tab);
        const k = 1 + r.uintLessThan(u32, 33);
        const rows = r.uintLessThan(u32, 19);
        const sel = try gpa.alloc(i32, rows * k);
        defer gpa.free(sel);
        // a prefix of the table's rows, as a prompt's selection reaches only the positions before it
        const reach = 1 + r.uintLessThan(usize, pages << @intCast(psh));
        for (sel) |*t| t.* = if (r.uintLessThan(u32, 5) == 0) -1 else @intCast(r.uintLessThan(usize, reach));
        try same(&p, .{ .sel = sel, .k = k, .table = tab, .psh = psh, .world = world, .rank = @intCast(rank), .row_bytes = 584, .cap = 128 << 20 });
    };
}

test "planner: empty, all -1, every row selected, odd rows, a row past the table, several blocks" {
    const gpa = testing.allocator;
    var p: uf.Planner = .{};
    defer p.deinit(gpa);
    const tab = try table(gpa, 5, 7);
    defer gpa.free(tab);
    const base: split.UnionIn = .{ .sel = &.{}, .k = 4, .table = tab, .psh = 6, .world = 2, .rank = 1, .row_bytes = 584, .cap = 128 << 20 };
    try same(&p, base);
    var pad: [7 * 4]i32 = @splat(-1);
    var in = base;
    in.sel = &pad;
    try same(&p, in);
    // every row of the table, descending, then a ragged tail the plan ignores (sel.len not a multiple of k)
    var all: [5 * 64 + 3]i32 = undefined;
    for (all[0 .. 5 * 64], 0..) |*t, i| t.* = @intCast(5 * 64 - 1 - i);
    @memcpy(all[5 * 64 ..], &[_]i32{ 0, 1, 2 });
    for (0..2) |rank| {
        in.sel = &all;
        in.rank = @intCast(rank);
        try same(&p, in);
    }
    // a row past the table: planUnion's error, and the planner's scratch stays clean for the next plan
    var bad = [_]i32{ 3, 9, 5 * 64, 1 };
    in.sel = &bad;
    try testing.expectError(error.RowOutsideTable, p.measure(gpa, in));
    for (p.seen.items) |x| try testing.expectEqual(@as(u32, 0), x);
    try testing.expectError(error.BadWorld, p.measure(gpa, .{ .sel = &bad, .k = 4, .table = tab, .psh = 6, .world = 2, .rank = 2, .row_bytes = 584, .cap = 1 }));
    // a cap one block cannot meet: null, as planUnion halves the rows
    in.sel = all[0 .. 5 * 64];
    in.cap = 3 * 584 * 100;
    try same(&p, in);
    for (p.seen.items) |x| try testing.expectEqual(@as(u32, 0), x);
    // one row at any cap stays one block, as planUnion's
    in.sel = all[0..4];
    in.cap = 1;
    try same(&p, in);
}
