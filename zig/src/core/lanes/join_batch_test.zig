//! Config.join_batch: prompts admitted together draft their first drafts in one batch at the next step, same tokens.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const Config = @import("config.zig").Config;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const a = std.testing.allocator;
const prompts = [_][]const u32{ &.{ 3, 1, 4 }, &.{ 1, 5, 9, 2 }, &.{ 6, 5 } };

const Run = struct { tokens: [3][8]u32 = @splat(@splat(0)), at_open: usize = 0, first_batch: usize = 0 };

fn run(join: bool) !Run {
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var cfg = try Config.init(a, .{ .exact_width = 8, .hidden_rows = true, .mtp = true, .speculate = true, .drafts = 7, .draft_streams = true, .join_batch = join }, 8, 7);
    defer cfg.deinit(a);
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.batched(), clock.clock());
    defer e.deinit();
    var ss: [3]sm.Stream = undefined;
    for (&ss, prompts) |*s, p| s.* = try sm.Stream.init(a, .{ .id = "s", .prompt = p, .max_new = 8 });
    defer for (&ss) |*s| s.deinit(a);
    const ptrs = [_]*sm.Stream{ &ss[0], &ss[1], &ss[2] };
    var errs: [3]?anyerror = undefined;
    var done: [3]bool = undefined;
    e.addStreams(&ptrs, &errs, &done);
    for (errs) |x| if (x) |err| return err;
    var r: Run = .{ .at_open = target.drafts_calls };
    for (ss) |s| try std.testing.expectEqual(@as(usize, 1), s.emitted().len); // every first token is out
    try e.step();
    r.first_batch = target.drafts_most;
    var steps: usize = 0;
    while (e.activeCount() > 0 and steps < 200) : (steps += 1) try e.step();
    for (&ss, &r.tokens) |*s, *out| @memcpy(out[0..s.emitted().len], s.emitted());
    return r;
}

test "join batch: no first drafts at open, then one batch for every joiner; the tokens are unchanged" {
    const plain = try run(false);
    const joined = try run(true);
    try std.testing.expectEqual(@as(usize, 3), plain.at_open); // one head call a stream as it opens
    try std.testing.expectEqual(@as(usize, 0), joined.at_open);
    try std.testing.expect(joined.first_batch >= 3); // the three joiners' first drafts in one call
    for (plain.tokens, joined.tokens) |p, j| try std.testing.expectEqualSlices(u32, &p, &j);
}
