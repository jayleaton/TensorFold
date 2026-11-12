//! tool_choice required, a named function or Anthropic any as an engine call gate (ported from MiaAI-Lab patch 0007).
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const prompt = @import("prompt.zig");
const reply_text = @import("reply_text.zig");
const tool_specs = @import("tool_specs.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

const probe = "tfprobe_fn";

/// The template's call: its opener, the text from it to the name and the mark after the name (null: unread).
pub const Form = struct { opener: []const u8, lead: ?[]const u8, tail: ?[]const u8 };

/// The gate for a reply that must call one of `tools` (the named one alone for a named function).
pub fn gate(srv: *Server, cx: *Cx, tools: []const Value) errors.Refused!api.CallGate {
    const form = try formOf(srv, cx);
    const opener = if (form) |fm| srv.text.tokenId(fm.opener) else null;
    if (opener == null) return cx.fail(.request, "backend '{s}' cannot enforce tool_choice \"required\" or a named function: its chat template marks tool calls with no single opener token; send \"auto\" or \"none\"", .{srv.backendName()});
    var names: std.ArrayList([]const u8) = .empty;
    if (form.?.lead != null) for (tools) |t| try names.append(cx.a, try tool_specs.toolName(cx.a, t));
    const think_open = if (srv.markers.open.len > 0) blk: {
        const ids = srv.text.encode(cx.a, srv.markers.open, false) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Template => &[_]u32{},
        };
        break :blk if (ids.len > 0) ids[0] else null;
    } else srv.text.tokenId("<think>");
    return .{
        .opener = opener.?,
        .lead = form.?.lead orelse "",
        .tail = form.?.tail orelse "",
        .names = names.items,
        .think_open = think_open,
        .think_end = srv.text.tokenId(srv.markers.close),
        .lex = .{ .ctx = srv, .blank = blank, .text = text, .encode = encode },
    };
}

/// The call form of `rendered`, a turn that calls `probe` with no arguments; null when it marks no call.
pub fn formIn(rendered: []const u8, opener_token: *const fn (ctx: *anyopaque, piece: []const u8) bool, ctx: *anyopaque) ?Form {
    if (std.mem.lastIndexOf(u8, rendered, probe)) |at| {
        var best: ?usize = null;
        var opener: []const u8 = "";
        for (reply_text.calls) |c| if (std.mem.lastIndexOf(u8, rendered[0..at], c[0])) |s| {
            if (best == null or s > best.? or (s == best.? and std.mem.order(u8, c[0], opener) == .gt)) {
                best = s;
                opener = c[0];
            }
        };
        if (best) |s| {
            const end = at + probe.len;
            return .{ .opener = opener, .lead = rendered[s + opener.len .. at], .tail = if (end < rendered.len) rendered[end .. end + 1] else "" };
        }
    }
    for (reply_text.calls) |c| if (opener_token(ctx, c[0])) return .{ .opener = c[0], .lead = null, .tail = null };
    return null;
}

fn formOf(srv: *Server, cx: *Cx) errors.Refused!?Form {
    const turn = (try json.parse(cx.a, "[{\"role\": \"user\", \"content\": \"x\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"call_0\", \"type\": \"function\", \"function\": {\"name\": \"" ++ probe ++ "\", \"arguments\": {}}}]}]")).ok;
    var scratch: Cx = .{ .a = cx.a };
    const rendered = if (prompt.renderIds(srv, &scratch, turn, &.{}, false, null, false)) |ids| try srv.text.decode(cx.a, ids) else |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Refused => "",
    };
    return formIn(rendered, isToken, srv);
}

fn isToken(ctx: *anyopaque, piece: []const u8) bool {
    const srv: *Server = @ptrCast(@alignCast(ctx));
    return srv.text.tokenId(piece) != null;
}

fn blank(ctx: *anyopaque, token: u32) bool {
    const srv: *Server = @ptrCast(@alignCast(ctx));
    for (srv.text.eosIds()) |e| if (e == token) return false;
    var buf: [256]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const t = srv.text.decode(fba.allocator(), &.{token}) catch return false;
    return std.mem.trim(u8, t, " \t\r\n").len == 0;
}

/// A token's text; an end token is a byte no lead or name holds, so the gate finishes the call instead of ending it.
fn text(ctx: *anyopaque, a: Allocator, token: u32) anyerror![]const u8 {
    const srv: *Server = @ptrCast(@alignCast(ctx));
    for (srv.text.eosIds()) |e| if (e == token) return "\x00";
    return srv.text.decode(a, &.{token});
}

fn encode(ctx: *anyopaque, a: Allocator, t: []const u8) anyerror![]const u32 {
    const srv: *Server = @ptrCast(@alignCast(ctx));
    return srv.text.encode(a, t, false);
}

test {
    _ = @import("tool_call_gate_test.zig");
}
