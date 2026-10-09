//! TF_DSV41_INDEX_BOUND (block.Options.index_bound, prod_knobs.indexBound) on the real emitters, host only (`zig build
//! test-dsv41-wide`): with the knob every dense `_scores` of a decode window (materialised or in backend._blocked's row
//! blocks) is `_scores_b` with the same grid and arguments, a Reindex layer's candidate scores (GATHER) stay `_scores`,
//! nothing else changes, and row mode rewrites `_scores_b` as it does `_scores`. The bits are the GPU A/B's
//! (`tf-dsv41-test scores-b`) and the TTGIR / PTX check (tools/zig/dsv41_triton/check_scores_b.py).

const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const check = @import("m1_check.zig");
const rowmode = @import("rowmode.zig");
const pk = @import("prod_knobs.zig");
const Config = @import("config.zig").Config;
const Call = calls.Call;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    return w;
}

fn prodOptions(limit: i64, bound: bool) block.Options {
    var o: block.Options = .{ .limit = limit, .r1 = true, .mhc_defer = true, .expert_topp = 0.85, .index_budget = 64 << 20, .rows = true, .rope_rows = limit + 2048, .taps = true, .index_bound = bound };
    const pages = @divExact(limit, block.Pool.page);
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
    return o;
}

fn eq(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

fn named(c: Call, name: []const u8) ?calls.Arg {
    for (c.args) |x| if (eq(x.name, name)) return x.arg;
    return null;
}

/// `on` is `off` with every dense `_scores` renamed `_scores_b`; returns (renamed, gather kept).
fn expectRenamed(off: []const Call, on: []const Call) ![2]usize {
    try testing.expectEqual(off.len, on.len);
    var renamed: usize = 0;
    var kept: usize = 0;
    for (off, on, 0..) |x, y, i| {
        errdefer std.debug.print("call {d}: {s} vs {s}\n", .{ i, x.name, y.name });
        if (eq(x.name, "_scores")) {
            const gather = named(x, "GATHER").?.b;
            try testing.expectEqualStrings(if (gather) "_scores" else "_scores_b", y.name);
            if (gather) kept += 1 else renamed += 1;
        } else try testing.expectEqualStrings(x.name, y.name);
        try testing.expect(x.triton == y.triton and x.glue == y.glue and x.begin == y.begin and std.mem.eql(i64, &x.grid, &y.grid));
        try testing.expectEqual(x.args.len, y.args.len);
        for (x.args, y.args) |p, q| try testing.expectEqualStrings(p.name, q.name);
    }
    return .{ renamed, kept };
}

test "TF_DSV41_INDEX_BOUND reader: unset / 0 off, 1 on, anything else refused" {
    const Fake = struct {
        var v: ?[]const u8 = null;
        fn get(name: []const u8) ?[]const u8 {
            return if (eq(name, "TF_DSV41_INDEX_BOUND")) v else null;
        }
    };
    Fake.v = null;
    try testing.expect(!try pk.indexBound(&Fake.get));
    Fake.v = "0";
    try testing.expect(!try pk.indexBound(&Fake.get));
    Fake.v = " 1 ";
    try testing.expect(try pk.indexBound(&Fake.get));
    for ([_][]const u8{ "2", "on", "" }) |s| {
        Fake.v = s;
        try testing.expectError(error.BadKnob, pk.indexBound(&Fake.get));
    }
}

test "TF_DSV41_INDEX_BOUND: decode windows' dense scores as _scores_b (materialised and row blocks), GATHER kept, row mode alike" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    // short and 131K-class positions; at 900K the 24 / 64-row windows pass the 64 MiB budget: backend._blocked
    for ([_]i64{ 4_000, 150_000, 900_000 }) |start| for ([_]i64{ 1, 5, 16, 24, 64 }) |n| {
        errdefer std.debug.print("start {d}, {d} rows\n", .{ start, n });
        const off = try block.emit(a, &cfg, &w, prodOptions(limit, false), &backbone, n, start, true);
        const on = try block.emit(a, &cfg, &w, prodOptions(limit, true), &backbone, n, start, true);
        const got = try expectRenamed(off, on);
        try testing.expect(got[0] > 0);
        if (n > 1) {
            const ro = try rowmode.transform(a, off, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = prodOptions(limit, false).pool.?.pts });
            const rn = try rowmode.transform(a, on, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = prodOptions(limit, false).pool.?.pts });
            _ = try expectRenamed(ro, rn);
            // row mode turns ROWS on and hands SL the read slots on the twin as on `_scores`
            for (rn) |c| if (eq(c.name, "_scores_b")) try testing.expect(named(c, "ROWS").?.b);
        }
    };
}
