//! One legacy image-mode interpretation for CPU asset planning and native model boot.
const std = @import("std");

pub fn enabled(raw: ?[]const u8) !bool {
    const mode = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (mode.len == 0 or std.ascii.eqlIgnoreCase(mode, "placeholder") or std.ascii.eqlIgnoreCase(mode, "reject"))
        return false;
    if (std.ascii.eqlIgnoreCase(mode, "native")) return true;
    return error.BadImagesMode;
}

pub fn fromEnv() !bool {
    const raw = if (std.c.getenv("TF_DSV41_IMAGES")) |value| std.mem.span(value) else null;
    return enabled(raw);
}
