//! tool_choice through an engine's call gate: each route's choice reaches the lane core, which forces the call.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const json = @import("json");
const errors = @import("errors.zig");
const mt = @import("model_text.zig");
const Server = @import("server.zig").Server;
const openai = @import("openai.zig");
const responses = @import("responses_translate.zig");
const anthropic = @import("anthropic_translate.zig");
const parse = @import("tool_parse.zig");
const gate = @import("tool_call_gate.zig");
const a = std.testing.allocator;
const t = std.testing;

// one token each, in ids no text below spells: Qwen's call and think markers
const opener: u32 = 94; // "~"
const closer: u32 = 92; // "|"
const think_open: u32 = 64; // "`"
const think_end: u32 = 62; // "^"
const eos: u32 = 96;
const marks = [_]struct { u32, []const u8 }{ .{ opener, "<tool_call>" }, .{ closer, "</tool_call>" }, .{ think_open, "<think>" }, .{ think_end, "</think>" } };

const Text = struct {
    probe: []const u8 = "<tool_call>\n<function=tfprobe_fn>\n</function>\n</tool_call>",
    fn text(x: *Text) mt.Text {
        return .{ .ctx = x, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eosIds, .render = render, .template_source = source } };
    }
    fn encode(_: *anyopaque, g: std.mem.Allocator, s: []const u8, _: bool) mt.Error![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        outer: while (i < s.len) {
            for (marks) |m| if (std.mem.startsWith(u8, s[i..], m[1])) {
                try ids.append(g, m[0]);
                i += m[1].len;
                continue :outer;
            };
            try ids.append(g, if (s[i] == '\n') 95 else @as(u32, s[i]) - 32);
            i += 1;
        }
        return ids.items;
    }
    fn decode(_: *anyopaque, g: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| {
            if (mark(id)) |m| try out.appendSlice(g, m) else if (id < 95) try out.append(g, @intCast(id + 32)) else if (id == 95) try out.append(g, '\n');
        }
        return out.items;
    }
    fn mark(id: u32) ?[]const u8 {
        for (marks) |m| if (m[0] == id) return m[1];
        return null;
    }
    fn tokenId(_: *anyopaque, s: []const u8) ?u32 {
        for (marks) |m| if (std.mem.eql(u8, m[1], s)) return m[0];
        return null;
    }
    fn tokenString(_: *anyopaque, g: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return decode(undefined, g, &.{id});
    }
    fn vocab(_: *anyopaque) u32 {
        return 97;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{eos};
    }
    fn source(_: *anyopaque) []const u8 {
        return "";
    }
    fn render(ptr: *anyopaque, g: std.mem.Allocator, _: json.Value, _: mt.RenderOptions, _: *[]const u8) mt.Error!mt.Rendered {
        const x: *Text = @ptrCast(@alignCast(ptr));
        return .{ .text = try g.dupe(u8, x.probe) };
    }
};

const Stub = struct {
    fn engine(s: *Stub) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .name = "flashnext_mia", .call_gates = true };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        unreachable;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

fn value(g: std.mem.Allocator, s: []const u8) !json.Value {
    return (try json.parse(g, s)).ok;
}

const Route = enum { chat, responses, messages };
const specs = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}},{\"type\":\"function\",\"function\":{\"name\":\"now\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";
const anthropic_tools = "[{\"name\":\"lookup\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}},{\"name\":\"now\",\"input_schema\":{\"type\":\"object\",\"properties\":{}}}]";

/// A body on `route` offering lookup and now with `choice` (that route's spelling), translated for the chat route.
fn body(cx: *errors.Cx, route: Route, choice: []const u8) !json.Value {
    const History = struct {
        pub fn conversation(_: @This(), c: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
            return c.refuse("this fixture has no stored response");
        }
    };
    const b = try json.newObject(cx.a);
    try b.put(cx.a, "model", .{ .string = "m" });
    try b.put(cx.a, "tool_choice", try value(cx.a, choice));
    switch (route) {
        .chat => {
            try b.put(cx.a, "messages", try value(cx.a, "[{\"role\":\"user\",\"content\":\"hi\"}]"));
            try b.put(cx.a, "tools", try value(cx.a, specs));
            return .{ .object = b };
        },
        .responses => {
            try b.put(cx.a, "input", .{ .string = "hi" });
            const offered = (try value(cx.a, specs)).array;
            for (offered) |*tool| {
                const f = tool.get("function").?.object;
                try f.put(cx.a, "type", .{ .string = "function" });
                tool.* = .{ .object = f };
            }
            try b.put(cx.a, "tools", .{ .array = offered });
            return (try responses.translate(cx, .{ .object = b }, History{})).chat;
        },
        .messages => {
            try b.put(cx.a, "max_tokens", try json.intValue(cx.a, 64));
            try b.put(cx.a, "messages", try value(cx.a, "[{\"role\":\"user\",\"content\":\"hi\"}]"));
            try b.put(cx.a, "tools", try value(cx.a, anthropic_tools));
            return anthropic.translate(cx, .{ .object = b }, false);
        },
    }
}

/// The request `choice` on `route` makes on a call-gate engine with no grammar support.
fn gated(srv: *Server, cx: *errors.Cx, route: Route, choice: []const u8) !struct { request: api.Request, tools: []const json.Value } {
    const input = try openai.chatInput(srv, cx, try body(cx, route, choice));
    var request: api.Request = .{ .prompt = &.{1}, .max_tokens = 200 };
    try srv.checkFeatures(cx, input.fields, input.tools.len > 0, false, &.{1}, input.tools, &request);
    try t.expect(request.structure == null);
    return .{ .request = request, .tools = input.tools };
}

const Choice = struct { route: Route, choice: []const u8, names: []const []const u8 };
const required = [_]Choice{
    .{ .route = .chat, .choice = "\"required\"", .names = &.{ "lookup", "now" } },
    .{ .route = .chat, .choice = "{\"type\":\"function\",\"function\":{\"name\":\"now\"}}", .names = &.{"now"} },
    .{ .route = .responses, .choice = "\"required\"", .names = &.{ "lookup", "now" } },
    .{ .route = .responses, .choice = "{\"type\":\"function\",\"name\":\"now\"}", .names = &.{"now"} },
    .{ .route = .messages, .choice = "{\"type\":\"any\"}", .names = &.{ "lookup", "now" } },
    .{ .route = .messages, .choice = "{\"type\":\"tool\",\"name\":\"now\"}", .names = &.{"now"} },
};

test "call gate: required, a named function and Anthropic any reach the engine as a gate over the offered names" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var text: Text = .{};
    var stub: Stub = .{};
    const srv = try Server.init(a, t.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    for (required) |c| {
        const r = try gated(srv, &cx, c.route, c.choice);
        const g = r.request.call orelse return error.TestExpectedGate;
        try t.expectEqual(opener, g.opener);
        try t.expectEqualStrings("\n<function=", g.lead);
        try t.expectEqualStrings(">", g.tail);
        try t.expectEqual(@as(usize, c.names.len), g.names.len);
        for (c.names, g.names) |want, got| try t.expectEqualStrings(want, got);
        try t.expectEqual(@as(?u32, think_open), g.think_open);
        try t.expectEqual(@as(?u32, think_end), g.think_end);
    }
}

test "call gate: auto and none make no gate and no grammar, on every route" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var text: Text = .{};
    var stub: Stub = .{};
    const srv = try Server.init(a, t.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    for ([_]Route{ .chat, .responses, .messages }) |route| for ([_][]const u8{ "auto", "none" }) |kind| {
        const choice = if (route == .messages) try std.fmt.allocPrint(cx.a, "{{\"type\":\"{s}\"}}", .{kind}) else try std.fmt.allocPrint(cx.a, "\"{s}\"", .{kind});
        const r = try gated(srv, &cx, route, choice);
        try t.expect(r.request.call == null);
        try t.expectEqual(std.mem.eql(u8, kind, "none"), r.tools.len == 0);
    };
}

test "call gate: an engine without gates or grammars still refuses required, naming the backend" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var text: Text = .{};
    var stub: Stub = .{};
    const srv = try Server.init(a, t.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    srv.info.call_gates = false;
    var cx: errors.Cx = .{ .a = arena.allocator() };
    try t.expectError(error.Refused, gated(srv, &cx, .chat, "\"required\""));
    try t.expectEqual(@as(u16, 400), cx.status());
    try t.expect(std.mem.indexOf(u8, cx.message, "flashnext_mia") != null);
}

test "call gate: a probe call with no form forces the opener alone, and no opener token means no form" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var text: Text = .{ .probe = "[[call tfprobe_fn]]" };
    var stub: Stub = .{};
    const srv = try Server.init(a, t.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    // no opener in the probe: the first marker that is one token (Qwen's) is forced alone, without a name
    const r = try gated(srv, &cx, .chat, "\"required\"");
    try t.expectEqual(opener, r.request.call.?.opener);
    try t.expectEqual(@as(usize, 0), r.request.call.?.names.len);
    try t.expect(gate.formIn("[[call tfprobe_fn]]", never, undefined) == null);
}

fn never(_: *anyopaque, _: []const u8) bool {
    return false;
}

/// A model that writes `plan`, but once a call header it opened closed, ends the call (as a model would).
const Model = struct {
    prompt_len: usize,
    plan: []const u32,
    close: []const u32, // "\n</function>\n</tool_call>"
    fn next(ctx: *anyopaque, history: []const u32) u32 {
        const m: *Model = @ptrCast(@alignCast(ctx));
        const reply = history[m.prompt_len..];
        if (std.mem.lastIndexOfScalar(u32, reply, opener)) |at| {
            const call = reply[at + 1 ..];
            if (std.mem.indexOfScalar(u32, call, closer) != null) return eos;
            if (std.mem.indexOfScalar(u32, call, '>' - 32)) |header| {
                const rest = call[header + 1 ..];
                return if (rest.len < m.close.len) m.close[rest.len] else eos;
            }
        }
        return if (reply.len < m.plan.len) m.plan[reply.len] else eos;
    }
};

/// The reply the lane engine decodes from `prompt` under `call` (null: ungated), drafted `depth` deep (0: plain).
fn decodeWith(g: std.mem.Allocator, prompt: []const u32, plan: []const u8, call: ?api.CallGate, depth: u32) ![]const u8 {
    return decodeOn(g, prompt, plan, call, depth, false);
}

/// decodeWith on a target whose one-token rounds run a step ahead (`piped`, Config.pipelined) or in windows.
fn decodeOn(g: std.mem.Allocator, prompt: []const u32, plan: []const u8, call: ?api.CallGate, depth: u32, piped: bool) ![]const u8 {
    var model: Model = .{ .prompt_len = prompt.len, .plan = try Text.encode(undefined, g, plan, false), .close = try Text.encode(undefined, g, "\n</function>\n</tool_call>", false) };
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = depth, .hidden_rows = true, .streams_exact = true, .batch_rows = 8, .max_streams = 2, .draft_streams = true, .gpu_tokens = piped }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .script = .{ .ctx = &model, .next = Model.next } };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "gate", .prompt = prompt, .max_new = 200, .drafts = depth > 0, .eos = &.{eos}, .call = call, .think_end = think_end });
    defer s.deinit(a);
    try engine.addStream(&s);
    var steps: usize = 0;
    while (engine.activeCount() > 0 and steps < 1000) : (steps += 1) try engine.step();
    try t.expect(s.finished and s.reason == .stop);
    return Text.decode(undefined, g, s.emitted());
}

test "call gate: the lane engine forces the call a plain answer skipped, drafted equal to plain, and it parses" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var text: Text = .{};
    var stub: Stub = .{};
    const srv = try Server.init(a, t.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = g };
    const prompt = try Text.encode(undefined, g, "user: hi\nassistant:", false);
    for (required) |c| {
        const r = try gated(srv, &cx, c.route, c.choice);
        const plain = try decodeWith(g, prompt, "Sure.", r.request.call, 0);
        errdefer std.debug.print("CALL_GATE_REPLY {s} [{s}]\n", .{ c.choice, plain });
        try t.expectEqualStrings(try std.fmt.allocPrint(g, "<tool_call>\n<function={s}>\n</function>\n</tool_call>", .{c.names[0]}), plain);
        for ([_]u32{ 1, 4 }) |depth| try t.expectEqualStrings(plain, try decodeWith(g, prompt, "Sure.", r.request.call, depth));
        try t.expectEqualStrings(plain, try decodeOn(g, prompt, "Sure.", r.request.call, 0, true)); // a step ahead, gated in windows
        const result = try parse.parse(g, plain, r.tools, null);
        try t.expectEqualStrings("", result.content);
        try t.expectEqual(@as(usize, 1), result.calls.?.len);
        try t.expectEqualStrings(c.names[0], result.calls.?[0].get("function").?.strField("name").?);
        // at the name this plan writes "noo": the offered name it starts is finished (drafted or not)
        for ([_]u32{ 0, 4 }) |depth| try t.expectEqualStrings("<tool_call>\n<function=now>\n</function>\n</tool_call>", try decodeWith(g, prompt, "Sure, it is noon.", r.request.call, depth));
    }
}

test "call gate: a reply that makes the call itself and a reply that thinks first keep every token the model chose" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var text: Text = .{};
    var stub: Stub = .{};
    const srv = try Server.init(a, t.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = g };
    const r = try gated(srv, &cx, .chat, "\"required\"");
    const prompt = try Text.encode(undefined, g, "user: hi\nassistant:", false);
    // a call the model writes itself (with a blank first) is the same reply gated or not
    const own = "\n<tool_call>\n<function=now>\n</function>\n</tool_call>";
    for ([_]u32{ 0, 4 }) |depth| try t.expectEqualStrings(try decodeWith(g, prompt, own, null, depth), try decodeWith(g, prompt, own, r.request.call, depth));
    try t.expectEqualStrings(own, try decodeWith(g, prompt, own, r.request.call, 0));
    // thinking: the open think block is the model's; the gate takes the answer after it
    const thinking = try Text.encode(undefined, g, "user: hi\nassistant:<think>", false);
    const reply = try decodeWith(g, thinking, "noon, so now</think>Sure.", r.request.call, 4);
    try t.expectEqualStrings("noon, so now</think><tool_call>\n<function=lookup>\n</function>\n</tool_call>", reply);
    try t.expectEqualStrings(reply, try decodeWith(g, thinking, "noon, so now</think>Sure.", r.request.call, 0));
}
