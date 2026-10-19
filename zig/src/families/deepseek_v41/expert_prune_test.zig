//! TF_DSV41_EXPERT_MIN_K on the real decode emitter, host only (test-dsv41-wide): only the prune's min-K moves.

const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const check = @import("m1_check.zig");
const Config = @import("config.zig").Config;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    return w;
}

fn options(limit: i64, r1: bool, min_k: i64) block.Options {
    var o: block.Options = .{ .limit = limit, .r1 = r1, .expert_topp = 0.85, .expert_min_k = min_k, .index_budget = 64 << 20, .rows = true, .rope_rows = limit + 2048 };
    const pages = @divExact(limit, block.Pool.page);
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
    return o;
}

/// The min-K a call carries: the router's prune list (R1, element 3) or `_prune`'s MIN_K; null for any other call.
fn minKOf(c: calls.Call) ?i64 {
    if (std.mem.eql(u8, c.name, "tf_dsv41_router_gemv_v2.route")) {
        const l = switch (c.args[c.args.len - 1].arg) {
            .list => |l| l,
            else => return null,
        };
        return if (l.len == 7) l[3].i else null;
    }
    for (c.args) |x| if (std.mem.eql(u8, x.name, "MIN_K")) return x.arg.i;
    return null;
}

test "TF_DSV41_EXPERT_MIN_K: every backbone prune takes it, R1 and not; nothing else in the window changes" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    for ([_]bool{ true, false }) |r1| for ([_]i64{ 1, 5 }) |n| {
        errdefer std.debug.print("r1 {}, {d} rows\n", .{ r1, n });
        const base = try block.emit(a, &cfg, &w, options(limit, r1, 3), &backbone, n, 300_000, true);
        const two = try block.emit(a, &cfg, &w, options(limit, r1, 2), &backbone, n, 300_000, true);
        try testing.expectEqual(base.len, two.len);
        var pruned: usize = 0;
        for (base, two) |x, y| {
            try testing.expectEqualStrings(x.name, y.name);
            if (minKOf(x)) |k| {
                try testing.expectEqual(@as(i64, 3), k);
                try testing.expectEqual(@as(i64, 2), minKOf(y).?);
                pruned += 1;
            } else try testing.expectEqualDeep(x.args, y.args);
        }
        try testing.expect(pruned > 0 and pruned <= cfg.layers); // a prune per routed backbone layer
    };
}
