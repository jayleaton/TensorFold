//! Rule depths that a draft head's own stops set: a confidence floor (greedy streams) and opt-in head stops.
const std = @import("std");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;
const Rule = depth.Rule;
const State = depth.State;
const Who = depth.Who;
const Probe = depth.Probe;

test "under a confidence floor a greedy stream drafts the most, and a sampled one keeps its rates' depth" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    const No = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = No.copy };
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.5, 0.5, 0.5 }, .most_drafts = 3, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = true, .confidence_floor = 0.5, .batch_rows = 8 };
    defer rule.deinit();
    var states: [2]?State = .{ null, null };
    defer for (&states) |*s| if (s.*) |*x| x.deinit(gpa);
    const greedy: Who = .{ .state = &states[0], .draft_room = 100, .finished = false, .forced = false };
    const sampled: Who = .{ .state = &states[1], .draft_room = 100, .finished = false, .forced = false, .sampled = true };
    try std.testing.expectEqual(@as(i64, 3), try rule.headDepth(&greedy, null, probe));
    try std.testing.expectEqual(@as(i64, 1), try rule.headDepth(&sampled, null, probe)); // a first rate under 0.8: one draft
}

test "head stops: every ask is the room's most while few streams are live, the rates' depth above that" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    const No = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = No.copy };
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.5, 0.5, 0.5 }, .most_drafts = 3, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = false, .head_stops = true, .head_stops_most = 2, .batch_rows = 8 };
    defer rule.deinit();
    var states: [2]?State = .{ null, null };
    defer for (&states) |*s| if (s.*) |*x| x.deinit(gpa);
    var whos = [_]Who{ .{ .state = &states[0], .draft_room = 100, .finished = false, .forced = false }, .{ .state = &states[1], .draft_room = 3, .finished = false, .forced = false, .sampled = true } };
    var out: [2]i64 = undefined;
    rule.live = 2;
    try std.testing.expect(rule.stopsNow());
    try std.testing.expectEqual(@as(i64, 3), try rule.depth(whos[0]));
    try rule.budgets(&whos, probe, true, &out);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, &out); // the room caps the second: draft_room 3 holds 2 drafts
    rule.live = 3; // past head_stops_most: the rates' rule (a first rate under 0.8 asks one draft)
    try std.testing.expect(!rule.stopsNow());
    try std.testing.expectEqual(@as(i64, 1), try rule.depth(whos[0]));
    rule.head_stops = false;
    rule.live = 1;
    try std.testing.expect(!rule.stopsNow());
}
