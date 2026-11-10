//! Her settings through platform/environment.zig's flashnext_mia rows; each reader keeps its own reading of the text.
const std = @import("std");
const environment = @import("sys").environment;

/// Refuses the first malformed setting of her rows, before the engine opens anything.
pub fn startup() !void {
    try environment.checkScope(.flashnext_mia, environment.Process{});
}

/// A setting's text as set (null when unset) once its row accepts it; `startup` has refused a malformed one.
pub fn get(comptime name: [:0]const u8) ?[]const u8 {
    comptime environment.registered(name);
    const raw = (environment.Process{}).get(name);
    environment.check(name, raw) catch return null;
    return raw;
}

/// A switch set to 0 or off (unset is not off).
pub fn off(comptime name: [:0]const u8) bool {
    const v = get(name) orelse return false;
    return std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off");
}

/// A switch set to exactly `value`.
pub fn is(comptime name: [:0]const u8, value: []const u8) bool {
    return if (get(name)) |v| std.mem.eql(u8, v, value) else false;
}
