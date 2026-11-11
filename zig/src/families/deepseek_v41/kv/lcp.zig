//! Resume past a rewritten history (TF_DSV41_SESSION_LCP_RESUME): prompt checkpoints, their kind, the image rule.

const std = @import("std");
const sessions = @import("sessions");

/// TF_DSV41_SESSION_LCP_EVERY's default: a checkpoint every 32,768 prompt tokens.
pub const every_default: u64 = 32768;

/// sessions.Entry.kind of a checkpoint (a turn's end is 0, a prompt's replay point 1): the store's anchor kind.
pub const checkpoint_kind: u32 = 2;

/// The checkpoint spacing (TF_DSV41_SESSION_LCP_EVERY, a multiple of the planner's grid); null: off.
pub fn fromEnv() !?u64 {
    const v = std.c.getenv("TF_DSV41_SESSION_LCP_RESUME") orelse return null;
    const on = std.mem.trim(u8, std.mem.span(v), " ");
    if (on.len == 0 or std.mem.eql(u8, on, "0")) return null;
    if (!std.mem.eql(u8, on, "1")) return error.BadSettings;
    const e = std.c.getenv("TF_DSV41_SESSION_LCP_EVERY") orelse return every_default;
    const t = std.mem.trim(u8, std.mem.span(e), " ");
    if (t.len == 0) return every_default;
    const every = std.fmt.parseInt(u64, t, 10) catch return error.BadSettings;
    if (every == 0 or every % @import("sched.zig").grid != 0) return error.BadSettings;
    return every;
}

/// A prompt snapshot's entry kind: a checkpoint at multiples of `every`, else the replay point's.
pub fn kindAt(every: ?u64, pos: u64, prompt_kind: u32) u32 {
    const e = every orelse return prompt_kind;
    return if (pos > 0 and pos % e == 0) checkpoint_kind else prompt_kind;
}

/// The run of image positions (`isVid`) that `pos` falls strictly inside: its first position; null: none.
pub fn imageStart(prompt: []const u32, pos: usize, isVid: *const fn (u32) bool) ?usize {
    if (pos == 0 or pos >= prompt.len or !isVid(prompt[pos - 1]) or !isVid(prompt[pos])) return null;
    var s = pos - 1;
    while (s > 0 and isVid(prompt[s - 1])) s -= 1;
    return s;
}

/// The hit `find` gives, moved back to an entry at or before the image's first position when it lands inside one.
pub fn outsideImages(prompt: []const u32, ctx: anytype, find: anytype, isVid: *const fn (u32) bool) !?sessions.store.Hit {
    var h = (try find(ctx, prompt)) orelse return null;
    while (imageStart(prompt, @intCast(h.pos), isVid)) |s| h = (try find(ctx, prompt[0 .. s + 1])) orelse return null;
    return h;
}
