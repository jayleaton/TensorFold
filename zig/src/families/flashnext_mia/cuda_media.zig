//! Her vision frontend's prompt media (Python's EncodedVision), kept here: f212's lanes.Stream carries none.
const std = @import("std");
const lanes = @import("lanes");

/// Feature rows (ascending), every row's t, h, w rotary positions, the decode offset and the bf16 features.
pub const Media = struct {
    rows: []const u32,
    positions: []const i32,
    delta: i64,
    features: []const u8,
    width: u32,
    /// a following rank's copy: the feature rows' count only (rows, positions and features arrive by broadcast)
    follower_rows: u32 = 0,

    pub fn rowCount(m: *const Media) usize {
        return if (m.follower_rows > 0) m.follower_rows else m.rows.len;
    }

    pub fn featureBytes(m: *const Media) usize {
        return m.rowCount() * @as(usize, m.width) * 2;
    }

    pub fn check(m: *const Media, prompt_len: usize) error{BadMedia}!void {
        if (m.follower_rows > 0) {
            if (m.rows.len != 0 or m.positions.len != 0 or m.features.len != 0 or m.width == 0 or m.follower_rows > prompt_len) return error.BadMedia;
            return;
        }
        if (m.rows.len == 0 or m.positions.len != 3 * prompt_len or m.width == 0) return error.BadMedia;
        if (m.features.len != 0 and m.features.len != m.featureBytes()) return error.BadMedia;
        var last: ?u32 = null;
        for (m.rows) |r| {
            if (r >= prompt_len or (last != null and r <= last.?)) return error.BadMedia;
            last = r;
        }
        for (m.positions) |p| if (p < 0) return error.BadMedia;
    }
};

/// A stream's media: always null on f212's lanes (the server sends images to in-engine towers, not to this engine).
pub fn of(s: *const lanes.Stream) ?*const Media {
    return if (comptime @hasField(lanes.Stream, "media")) s.media else null;
}

test "media rows, positions and features must cover the prompt" {
    const feats: [16]u8 = @splat(0);
    const pos: [15]i32 = @splat(0);
    var m: Media = .{ .rows = &.{ 1, 2 }, .positions = &pos, .delta = -1, .features = &feats, .width = 4 };
    try m.check(5);
    try std.testing.expectError(error.BadMedia, m.check(4));
    m.rows = &.{ 2, 1 };
    try std.testing.expectError(error.BadMedia, m.check(5));
    m.rows = &.{ 1, 2 };
    m.features = &.{};
    try m.check(5);
    m.features = feats[0..8];
    try std.testing.expectError(error.BadMedia, m.check(5));
}
