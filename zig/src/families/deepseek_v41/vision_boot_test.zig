//! Boot selection uses the actual integer wire frame and a host-only fake of the existing PlanLink interface.
const std = @import("std");
const t = std.testing;
const boot = @import("vision_boot.zig");
const VisionOpen = @import("engine_api").VisionOpen;

const Link = struct {
    rank: u32 = 0,
    world: u32 = 2,
    frame: [boot.frame_words]i64 = @splat(0),
    len: usize = boot.frame_words,
    sends: u32 = 0,
    receives: u32 = 0,
    fail: bool = false,

    pub fn send(l: *Link, words: []const i64) !void {
        l.sends += 1;
        if (l.fail) return error.PeerClosed;
        try t.expectEqual(boot.frame_words, words.len);
        @memcpy(&l.frame, words);
    }
    pub fn recv(l: *Link, out: []i64) ![]i64 {
        l.receives += 1;
        if (l.fail) return error.PeerClosed;
        @memcpy(out[0..l.len], l.frame[0..l.len]);
        return out[0..l.len];
    }
};

test "vision boot root selects limits and follower ignores conflicting local settings" {
    var root: Link = .{};
    const limits: VisionOpen = .{ .image_tokens = 128, .image_tokens_set = true, .max_images = 3 };
    const selected = (try boot.select(&root, limits)).?;
    try t.expectEqual(@as(u32, 1), root.sends);
    try t.expectEqual(@as(u32, 0), root.receives);
    try t.expectEqualSlices(i64, &.{ boot.tag, boot.version, 1, 128, 1, 3 }, &root.frame);
    var follower: Link = .{ .rank = 1, .frame = root.frame };
    const received = (try boot.select(&follower, .{ .image_tokens = 0, .max_images = 0 })).?;
    try t.expectEqualDeep(selected, received);
    try t.expectEqualDeep(limits, received);
    try t.expectEqual(@as(u32, 0), follower.sends);
    try t.expectEqual(@as(u32, 1), follower.receives);
}

test "vision boot disabled root overrides follower enable and implicit token default survives" {
    var root: Link = .{};
    try t.expect(try boot.select(&root, null) == null);
    try t.expectEqualSlices(i64, &.{ boot.tag, boot.version, 0, 0, 0, 0 }, &root.frame);
    var follower: Link = .{ .rank = 1, .frame = root.frame };
    try t.expect(try boot.select(&follower, .{}) == null);
    _ = try boot.select(&root, .{});
    follower.frame = root.frame;
    const limits = (try boot.select(&follower, null)).?;
    try t.expect(!limits.image_tokens_set);
    try t.expectEqual(@as(u32, 4096), limits.image_tokens);
}

test "vision boot single rank performs no transport calls" {
    var solo: Link = .{ .world = 1, .fail = true };
    try t.expect(try boot.select(&solo, null) == null);
    const limits = (try boot.select(&solo, .{ .image_tokens = 512, .image_tokens_set = true })).?;
    try t.expectEqual(@as(u32, 512), limits.image_tokens);
    try t.expectEqual(@as(u32, 0), solo.sends);
    try t.expectEqual(@as(u32, 0), solo.receives);
}

test "vision boot refuses malformed frame tags versions flags lengths and limits" {
    var root: Link = .{};
    _ = try boot.select(&root, .{});
    const good = root.frame;
    const bad_frames = [_][boot.frame_words]i64{
        .{ 0, boot.version, 1, 128, 1, 3 },
        .{ boot.tag, 2, 1, 128, 1, 3 },
        .{ boot.tag, boot.version, 2, 128, 1, 3 },
        .{ boot.tag, boot.version, 1, 128, -1, 3 },
        .{ boot.tag, boot.version, 1, 128, 2, 3 },
        .{ boot.tag, boot.version, 0, 128, 0, 0 },
    };
    for (bad_frames) |frame| {
        var link: Link = .{ .rank = 1, .frame = frame };
        try t.expectError(error.BadVisionBootFrame, boot.select(&link, null));
    }
    const bad_limits = [_][boot.frame_words]i64{
        .{ boot.tag, boot.version, 1, -1, 1, 3 },
        .{ boot.tag, boot.version, 1, 4, 1, 3 },
        .{ boot.tag, boot.version, 1, 65537, 1, 3 },
        .{ boot.tag, boot.version, 1, 128, 1, 0 },
        .{ boot.tag, boot.version, 1, 128, 1, -1 },
        .{ boot.tag, boot.version, 1, 128, 1, @as(i64, std.math.maxInt(u32)) + 1 },
    };
    for (bad_limits) |frame| {
        var link: Link = .{ .rank = 1, .frame = frame };
        try t.expectError(error.BadVisionBootSettings, boot.select(&link, null));
    }
    for ([_]usize{ 0, 1, 5 }) |len| {
        var link: Link = .{ .rank = 1, .frame = good, .len = len };
        try t.expectError(error.BadVisionBootFrame, boot.select(&link, null));
    }
    try t.expectError(error.BadVisionBootSettings, boot.select(&root, .{ .image_tokens = 0 }));
    try t.expectError(error.BadVisionBootSettings, boot.select(&root, .{ .max_images = 0 }));
    var invalid: Link = .{ .world = 0 };
    try t.expectError(error.BadVisionBootRank, boot.select(&invalid, null));
    invalid = .{ .world = 1, .rank = 1 };
    try t.expectError(error.BadVisionBootRank, boot.select(&invalid, null));
}

test "vision boot propagates link failures and legacy native is root-selectable" {
    var failed: Link = .{ .fail = true };
    try t.expectError(error.PeerClosed, boot.select(&failed, null));
    failed.rank = 1;
    try t.expectError(error.PeerClosed, boot.select(&failed, null));
    const native = (try boot.legacy(null, " Native \n")).?;
    try t.expect(!native.image_tokens_set);
    try t.expect(try boot.legacy(null, null) == null);
    try t.expect(try boot.legacy(null, "reject") == null);
    try t.expect(try boot.legacy(null, "placeholder") == null);
    try t.expectError(error.BadImagesMode, boot.legacy(null, "on"));
    const explicit = (try boot.legacy(.{ .image_tokens = 256 }, "bad setting")).?;
    try t.expectEqual(@as(u32, 256), explicit.image_tokens);
}
