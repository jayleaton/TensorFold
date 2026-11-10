//! Model.pooled_budgets: a shared round's drafts as db281878's Rule.budgets allocated them, head steps unpriced.
const std = @import("std");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;
const Rule = depth.Rule;
const State = depth.State;
const Who = depth.Who;

const No = struct {
    fn copy(_: *anyopaque, _: *const Who) bool {
        return false;
    }
};

/// Each of `n` fresh streams' budget (all equal here) under the pooled or the priced rule.
fn budget(n: usize, pooled: bool) !i64 {
    const gpa = std.testing.allocator;
    var shared: Table = .{};
    defer shared.deinit(gpa);
    for (1..257) |w| try shared.put(gpa, @intCast(w), 10.0 + @as(f64, @floatFromInt(w))); // a row costs 1 ms past 10
    const empty: Table = .{};
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.9, 0.8, 0.7, 0.6, 0.5 }, .most_drafts = 15, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 6.0, .plain_guard = false, .node_probabilities = false, .batch_rows = 256, .batched_heads = true, .pooled_budgets = pooled };
    defer rule.deinit();
    var states: [16]?State = @splat(null);
    defer for (&states) |*s| if (s.*) |*x| x.deinit(gpa);
    var whos: [16]Who = undefined;
    for (&whos, &states) |*w, *s| w.* = .{ .state = s, .draft_room = 100, .finished = false, .forced = false };
    var unused: u8 = 0;
    var out: [16]i64 = undefined;
    try rule.budgets(whos[0..n], .{ .ptr = &unused, .copy = No.copy }, false, out[0..n]);
    for (out[1..n]) |d| try std.testing.expectEqual(out[0], d);
    return out[0];
}

test "pooled budgets: N streams draft what db281878's allocation over the measured overhead gives them" {
    // chains 0.9, 0.72, 0.504, 0.3024, 8 ms overhead: three each at (4 + 4 * 2.124) / (10 + 16 + 8) = 0.3675
    try std.testing.expectEqual(@as(i64, 3), try budget(4, true));
    // 16 streams: two each, (16 + 16 * 1.62) / (10 + 48 + 8) = 0.6352, a third draft (0.504) is below that rate
    try std.testing.expectEqual(@as(i64, 2), try budget(16, true));
}

test "pooled budgets: off, the head's 6 ms steps priced by the deepest chain draft each stream less" {
    // 4 streams: two each, 10.48 / (10 + 12 + 8 + 2 * 6) = 0.2495 beats three each at 12.496 / (10 + 16 + 8 + 18)
    try std.testing.expectEqual(@as(i64, 2), try budget(4, false));
    // 16 streams: one each, 30.4 / (10 + 32 + 8 + 6) = 0.5429 beats two each at 41.92 / (10 + 48 + 8 + 12)
    try std.testing.expectEqual(@as(i64, 1), try budget(16, false));
}
