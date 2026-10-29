//! Root-selected image boot settings over the existing host plan link, before model planning or admission.
const std = @import("std");
const VisionOpen = @import("engine_api").VisionOpen;

pub const tag: i64 = 0x54465642;
pub const version: i64 = 1;
pub const frame_words = 6;

/// Link has rank/world, send([]const i64) and recv([]i64), as the existing PlanLink does.
pub fn select(link: anytype, requested: ?VisionOpen) !?VisionOpen {
    if (link.world == 0 or link.rank >= link.world) return error.BadVisionBootRank;
    if (link.rank == 0) {
        const frame = try encode(requested);
        if (link.world > 1) try link.send(&frame);
        return decode(&frame);
    }
    var frame: [frame_words]i64 = undefined;
    return decode(try link.recv(&frame));
}

fn encode(requested: ?VisionOpen) ![frame_words]i64 {
    const limits = requested orelse return .{ tag, version, 0, 0, 0, 0 };
    if (limits.image_tokens < 5 or limits.image_tokens > (1 << 16) or limits.max_images == 0)
        return error.BadVisionBootSettings;
    return .{ tag, version, 1, limits.image_tokens, @intFromBool(limits.image_tokens_set), limits.max_images };
}

fn decode(frame: []const i64) !?VisionOpen {
    if (frame.len != frame_words or frame[0] != tag or frame[1] != version) return error.BadVisionBootFrame;
    if (frame[2] == 0) {
        for (frame[3..]) |word| if (word != 0) return error.BadVisionBootFrame;
        return null;
    }
    if (frame[2] != 1 or (frame[4] != 0 and frame[4] != 1)) return error.BadVisionBootFrame;
    if (frame[3] < 5 or frame[3] > (1 << 16) or frame[5] < 1 or frame[5] > std.math.maxInt(u32))
        return error.BadVisionBootSettings;
    return .{ .image_tokens = @intCast(frame[3]), .image_tokens_set = frame[4] == 1, .max_images = @intCast(frame[5]) };
}

/// Only rank zero calls this reader; explicit API options override the legacy environment setting.
pub fn fromEnv(explicit: ?VisionOpen) !?VisionOpen {
    if (explicit != null) return explicit;
    const raw: ?[]const u8 = if (std.c.getenv("TF_DSV41_IMAGES")) |value| std.mem.span(value) else null;
    const max: ?[]const u8 = if (std.c.getenv("TF_DSV41_VISION_MAX_IMAGES")) |value| std.mem.span(value) else null;
    return legacy(null, raw, max);
}

/// Only native enables the tower, with its own images a request (TF_DSV41_VISION_MAX_IMAGES, default 8).
pub fn legacy(explicit: ?VisionOpen, raw: ?[]const u8, max_images: ?[]const u8) !?VisionOpen {
    if (explicit != null) return explicit;
    if (!try @import("image_mode.zig").enabled(raw)) return null;
    var limits: VisionOpen = .{ .max_images = (@import("vision/prep.zig").Settings{}).max_images };
    if (max_images) |text| {
        const trimmed = std.mem.trim(u8, text, " \t\r\n");
        if (trimmed.len > 0) limits.max_images = std.fmt.parseInt(u32, trimmed, 10) catch return error.BadVisionBootSettings;
    }
    if (limits.max_images == 0) return error.BadVisionBootSettings;
    return limits;
}
