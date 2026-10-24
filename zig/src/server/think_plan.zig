//! A reply's think plan: budget close, answer reserve, and a think close forced first where Info.first_mask is false.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const grammar = @import("grammar.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;

pub const no_first_mask = "this engine cannot constrain a reply's first token, and the prompt does not end in the think close it could force instead: send the request with thinking on";

/// A grammar request's prompt tail: the think close cut to force first (returned), or an adaptive think block opened.
pub fn cutFirst(srv: *Server, cx: *Cx, f: Value, has_tools: bool, thinking: bool, ids: *[]const u32) errors.Refused!?u32 {
    if (!srv.info.structures) return null;
    const required = has_tools and if (f.get("tool_call_required")) |r| r.truthy() else false;
    if (!required and try grammar.spec(cx, f) == null) return null;
    if (thinking) return try openThink(srv, cx, ids);
    if (srv.info.first_mask) return null;
    return try cut(cx, srv.text.tokenId(srv.markers.close), ids);
}

/// MiniMax's adaptive prompt opened, so the close its grammar starts after is sure to come (or the budget forces it).
fn openThink(srv: *Server, cx: *Cx, ids: *[]const u32) errors.Refused!?u32 {
    const m = srv.mm_think orelse return null;
    const p = ids.*;
    if (p.len == 0 or p[p.len - 1] == m.open or p[p.len - 1] == m.close) return null;
    ids.* = try std.mem.concat(cx.a, u32, &.{ p, &.{m.open} });
    return null;
}

fn cut(cx: *Cx, close: ?u32, ids: *[]const u32) errors.Refused!u32 {
    const p = ids.*;
    const end = close orelse return cx.refuse(no_first_mask);
    if (p.len < 2 or p[p.len - 1] != end) return cx.refuse(no_first_mask);
    ids.* = p[0 .. p.len - 1];
    return end;
}

/// The budget, loop guard and answer reserve of a thinking request, or the forced first token of a cut one.
pub fn apply(srv: *Server, cx: *Cx, f: Value, thinking: bool, forced: ?u32, room: i64, request: *api.Request) errors.Refused!void {
    if (forced) |t| return force(cx, request, t);
    if (!thinking) return;
    const budget_field = f.get("thinking_budget");
    const budget: i64 = if (budget_field != null and budget_field.?.truthy()) budget_field.?.int64() orelse 0 else srv.config.thinking_budget;
    if (budget > 0 and !srv.info.thinking_budget)
        return cx.refuse("thinking_budget is not supported by this native engine");
    if (srv.think_close.len > 0) request.loop_guard = srv.config.loop_guard;
    if ((budget > 0 or request.loop_guard) and srv.think_close.len > 0) {
        if (budget > 0) request.think_budget = @intCast(@min(budget, std.math.maxInt(u32)));
        request.think_close = srv.think_close;
        request.think_end = srv.think_close_end;
    }
    // The answer quota is separate only on engines that enforce it.
    if (srv.info.answer_limit) @import("thinking_reserve.zig").apply(request, srv.config.thinking_reserve, room, srv.think_close_end);
}

/// The grammar starts after `token`, the reply's first token, forced as a one-token budget's close.
fn force(cx: *Cx, request: *api.Request, token: u32) errors.Refused!void {
    if (request.structure) |*s| s.after = token;
    request.think_close = try cx.a.dupe(u32, &.{token});
    request.think_end = token;
    request.think_budget = 1;
}

test "a thinking-off prompt's think close is cut and forced first, so its grammar starts after it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    var ids: []const u32 = &.{ 1, 2, 3, 9 };
    try std.testing.expectEqual(@as(u32, 9), try cut(&cx, 9, &ids));
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, ids);
    var request: api.Request = .{ .prompt = ids, .max_tokens = 8, .structure = .{ .kind = .json, .text = "" } };
    try force(&cx, &request, 9);
    try std.testing.expectEqual(@as(?u32, 9), request.structure.?.after);
    try std.testing.expectEqualSlices(u32, &.{9}, request.think_close);
    try std.testing.expectEqual(@as(u32, 1), request.think_budget);
    try std.testing.expectEqual(@as(?u32, 9), request.think_end);
}

test "a prompt that does not end in the think close is refused, not served with an unmasked first token" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    var ids: []const u32 = &.{ 1, 2, 3 };
    try std.testing.expectError(error.Refused, cut(&cx, 9, &ids));
    try std.testing.expectError(error.Refused, cut(&cx, null, &ids));
    var one: []const u32 = &.{9};
    try std.testing.expectError(error.Refused, cut(&cx, 9, &one));
    try std.testing.expectEqualStrings(no_first_mask, cx.message);
}
