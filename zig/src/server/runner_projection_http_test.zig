//! Abstain anchors through the OpenAI HTTP handler: raw runner bytes projected into the final fields.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const model_text = @import("model_text.zig");
const Allocator = std.mem.Allocator;

const close_id: u32 = 300;
const eos_id: u32 = 301;

/// One byte a token below 256; 300 is the model's own `</think>` token and 301 its EOS.
const Bytes = struct {
    fn text(t: *Bytes) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = template } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id == close_id) try out.appendSlice(a, "</think>") else if (id < 256) try out.append(a, @intCast(id));
        return out.items;
    }
    fn tokenId(_: *anyopaque, piece: []const u8) ?u32 {
        return if (std.mem.eql(u8, piece, "</think>")) close_id else null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) ![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocab(_: *anyopaque) u32 {
        return 302;
    }
    fn eos(_: *anyopaque) []const u32 {
        return &.{eos_id};
    }
    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, "prompt") };
    }
    fn template(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Gives its tokens in one round, then ends (stop) with its runner receipt.
const Script = struct {
    tokens: []const u32,
    receipt: []const u8,
    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 8192 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = s.tokens });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop, .runner = s.receipt } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Reply = struct {
    a: Allocator,
    status: ?u16 = null,
    whole: ?json.Value = null,
    events: std.ArrayList(json.Value) = .empty,
    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        if (value) |v| r.events.append(r.a, v) catch return error.Closed;
    }
    fn reply(ctx: *anyopaque, status_code: u16, value: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.status = status_code;
        r.whole = value;
    }
};

/// The client's fields, as a whole reply gives them or its stream's deltas add up to, and the receipt's runner block.
const Got = struct { reasoning: []const u8, content: []const u8, runner: json.Value };

/// A thinking chat reply of `reasoning`, the close token and `content`, through openai.run.
fn post(a: Allocator, reasoning: []const u8, content: []const u8, receipt: []const u8, stream: bool) !Got {
    var ids: std.ArrayList(u32) = .empty;
    for (reasoning) |b| try ids.append(a, b);
    try ids.append(a, close_id);
    for (content) |b| try ids.append(a, b);
    try ids.append(a, eos_id);
    var s: Script = .{ .tokens = ids.items, .receipt = receipt };
    var bytes: Bytes = .{};
    const srv = try Server.init(std.testing.allocator, std.testing.io, s.engine(), bytes.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
    defer srv.deinit();
    const text = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"max_tokens\":500,\"stream\":{}}}", .{stream});
    var conn = Conn{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response = Reply{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, true, (try json.parse(a, text)).ok);
    if (!stream) {
        const m = response.whole.?.get("choices").?.array[0].get("message").?;
        return .{ .reasoning = json.strOr(m.get("reasoning_content")), .content = json.strOr(m.get("content")), .runner = response.whole.?.get("tensorfold").?.get("runner").? };
    }
    var r: std.ArrayList(u8) = .empty;
    var c: std.ArrayList(u8) = .empty;
    var runner: ?json.Value = null;
    for (response.events.items) |v| {
        if (v.get("tensorfold")) |t| if (t.get("runner")) |x| if (x.get("live") == null) {
            runner = x;
        };
        const cs = (v.get("choices") orelse continue).array;
        if (cs.len == 0) continue;
        const d = cs[0].get("delta") orelse continue;
        try r.appendSlice(a, json.strOr(d.get("reasoning_content")));
        try c.appendSlice(a, json.strOr(d.get("content")));
    }
    return .{ .reasoning = r.items, .content = c.items, .runner = runner orelse return error.MissingReceipt };
}

fn num(v: json.Value, key: []const u8) ?i64 {
    return (v.get(key) orelse return null).int64();
}

/// Each abstain's final [source_at, source_end) holds its source in the field the client has; raw_* keeps the engine's.
fn expectAbstain(got: Got, i: usize, source: []const u8, raw: [4]?i64, final: [4]?i64) !void {
    const x = got.runner.get("abstains").?.array[i];
    for ([_][]const u8{ "source_at", "source_end", "source_close_at", "resume_at" }, raw, final) |key, want_raw, want| {
        try std.testing.expectEqual(want, num(x, key));
        var name: [32]u8 = undefined;
        try std.testing.expectEqual(want_raw, num(x, try std.fmt.bufPrint(&name, "raw_{s}", .{key})));
    }
    if (source.len == 0) return;
    const field = if (std.mem.eql(u8, x.strField("source_field").?, "content")) got.content else got.reasoning;
    try std.testing.expectEqualStrings(source, field[@intCast(final[0].?)..@intCast(final[1].?)]);
}

fn abstain(a: Allocator, field: []const u8, source: [3]?usize, resume_field: []const u8, resume_at: usize) ![]const u8 {
    var o: std.ArrayList(u8) = .empty;
    try o.print(a, "{{\"cell\":0,\"form\":\"fence\",\"source_field\":\"{s}\"", .{field});
    for ([_][]const u8{ "source_at", "source_end", "source_close_at" }, source) |key, v|
        if (v) |n| try o.print(a, ",\"{s}\":{d}", .{ key, n }) else try o.print(a, ",\"{s}\":null", .{key});
    try o.print(a, ",\"resume_field\":\"{s}\",\"resume_at\":{d}}}", .{ resume_field, resume_at });
    return o.items;
}

// T5's counterexample: the runner finds `print(undefined_total)\n` at raw content byte 12, the client at byte 10.
const t5_content = "\n\n```python\nprint(undefined_total)\n```\nResult:\n";

test "runner projection: HTTP abstain anchors after the think close's newlines are final content bytes, raw kept" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const receipt = try std.fmt.allocPrint(a, "{{\"cells\":[{{\"cell\":1,\"field\":\"content\",\"at\":12}}],\"abstain_schema\":1,\"abstains\":[{s},{s}]}}", .{
        try abstain(a, "content", .{ 12, 35, 39 }, "content", 47),
        try abstain(a, "content", .{ null, null, null }, "content", 1), // resumed inside the dropped newlines
    });
    for ([_]bool{ false, true }) |stream| {
        const got = try post(a, "plan", t5_content, receipt, stream);
        try std.testing.expectEqualStrings("plan", got.reasoning);
        try std.testing.expectEqualStrings(t5_content[2..], got.content);
        try expectAbstain(got, 0, "print(undefined_total)\n", .{ 12, 35, 39, 47 }, .{ 10, 33, 37, 45 });
        try expectAbstain(got, 1, "", .{ null, null, null, 1 }, .{ null, null, null, null });
        try std.testing.expectEqual(10, num(got.runner.get("cells").?.array[0], "final_at"));
        const content = got.runner.get("field_projection").?.get("content").?;
        try std.testing.expectEqual(47, num(content, "raw_bytes"));
        try std.testing.expectEqual(2, num(content, "dropped_lead"));
        try std.testing.expectEqual(45, num(content, "final_bytes"));
    }
}

test "runner projection: HTTP reasoning anchors follow the whole reply's strip, and the stream's unstripped deltas" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reasoning = " \n x\n```python\nprint(4)\n```\n";
    const receipt = try std.fmt.allocPrint(a, "{{\"cells\":[],\"abstains\":[{s}]}}", .{try abstain(a, "reasoning", .{ 15, 24, 28 }, "reasoning", 28)});
    const whole = try post(a, reasoning, "ok", receipt, false);
    try std.testing.expectEqualStrings(reasoning[3 .. reasoning.len - 1], whole.reasoning);
    // the close and resume bytes sit in the stripped trailing newline: no final byte, only the raw one
    try expectAbstain(whole, 0, "print(4)\n", .{ 15, 24, 28, 28 }, .{ 12, 21, null, null });
    const streamed = try post(a, reasoning, "ok", receipt, true);
    try std.testing.expectEqualStrings(reasoning, streamed.reasoning);
    try expectAbstain(streamed, 0, "print(4)\n", .{ 15, 24, 28, 28 }, .{ 15, 24, 28, 28 });
}

test "runner projection: HTTP a receipt without abstain records keeps its bytes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const receipt = "{\"cells\":[{\"cell\":0,\"field\":\"content\",\"at\":12}]}";
    for ([_]bool{ false, true }) |stream| {
        const got = try post(a, "plan", t5_content, receipt, stream);
        try std.testing.expectEqualStrings(try json.stringify(a, (try json.parse(a, receipt)).ok, .{}), try json.stringify(a, got.runner, .{}));
    }
}
