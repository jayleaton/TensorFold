//! Runner-owned trace lines beside tool calls (AB3 t02): their bytes reach the client whole, lane or not, cut or not.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const HfText = @import("hf_text.zig").HfText;
const tf = @import("control_tokens_fixture.zig");
const lane_test = @import("compute_lane_http_test.zig");
const lane = api.compute_lane;
const t = std.testing;
const Allocator = std.mem.Allocator;
const Note = @typeInfo(@FieldType(api.Event, "cells")).pointer.child;

/// An owned write in a segment: its first token's offset there and its token count.
const Write = struct { at: usize, n: usize };

/// An engine that gives each segment after the last resume, its records first; held at each EOS, else cut (length).
const Script = struct {
    a: Allocator,
    segments: []const []const u32, // each its EOS last
    writes: []const []const Write, // per segment
    sink: ?api.Sink = null,
    id: api.Id = 0,
    sent: usize = 0, // the reply's tokens so far, the inserted results' too: what a record's `token` counts
    next: usize = 0,
    records: usize = 0,

    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .resume_held = resumeHeld } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768 };
    }
    fn segment(s: *@This()) void {
        const sink = s.sink.?;
        var notes: std.ArrayList(Note) = .empty;
        for (s.writes[s.next]) |w| {
            const text = std.fmt.allocPrint(s.a, "{{\"cell\":{d},\"form\":\"trace\",\"field\":\"content\",\"token\":{d},\"ids\":{d},\"landed\":{d}}}", .{ s.records, s.sent + w.at, w.n, w.n }) catch unreachable;
            notes.append(s.a, .{ .index = @intCast(s.records), .content = true, .end = 0, .json = text }) catch unreachable;
            s.records += 1;
        }
        if (notes.items.len > 0) sink.event(sink.ctx, s.id, &.{ .cells = notes.items });
        sink.event(sink.ctx, s.id, &.{ .tokens = s.segments[s.next] });
        s.sent += s.segments[s.next].len;
        s.next += 1;
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        s.sink = sink;
        s.id = id;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        s.segment();
        if (!request.hold_eos) sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .length } });
    }
    fn resumeHeld(ctx: *anyopaque, id: api.Id, r: api.Resume) void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        const sink = s.sink.?;
        if (r.tokens.len == 0 or s.next == s.segments.len) return sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
        sink.event(sink.ctx, id, &.{ .tokens = r.tokens });
        s.sent += r.tokens.len;
        s.segment();
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The reply, whole or as the stream's content deltas added up.
const Reply = struct {
    a: Allocator,
    code: u16 = 0,
    body: json.Value = .null,
    content: std.ArrayList(u8) = .empty,
    calls: bool = false,
    fn out(r: *@This()) @import("openai.zig").Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        const r: *@This() = @ptrCast(@alignCast(ctx));
        const cs = ((value orelse return).get("choices") orelse return).array;
        if (cs.len == 0) return;
        const d = cs[0].get("delta") orelse return;
        r.calls = r.calls or d.get("tool_calls") != null;
        r.content.appendSlice(r.a, json.strOr(d.get("content"))) catch return error.Closed;
    }
    fn reply(ctx: *anyopaque, code: u16, v: json.Value) void {
        const r: *@This() = @ptrCast(@alignCast(ctx));
        r.code = code;
        r.body = v;
    }
};

const trace = "[exact, computed by the engine: 2^66 = 73786976294838206464]\n";
const later = "[exact, computed by the engine: 2^66 + 1 = 73786976294838206465]\n";

/// The template's tokens for an assistant turn making `call` after `history` (its EOS last).
fn callTokens(a: Allocator, text: anytype, history: json.Value, call: []const u8) ![]const u32 {
    var problem: []const u8 = "";
    const turn = try lane_test.obj(a, call);
    const asked = try text.renderIds(a, history, .{ .add_generation_prompt = true }, &problem);
    const called = try text.renderIds(a, try lane_test.appendTo(a, history, &.{turn}), .{ .add_generation_prompt = false }, &problem);
    try t.expect(std.mem.startsWith(u32, called, asked));
    return called[asked.len..];
}

/// The t02 shape twice: an owned trace line, then the lane's python call; a longer one, then a client's call.
fn run(a: Allocator, stream: bool) !Reply {
    const python = "/usr/bin/python3";
    std.Io.Dir.accessAbsolute(t.io, python, .{}) catch return error.SkipZigTest;
    var tmp = try lane_test.fixture(a);
    defer tmp.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, tmp.path, null, a, &problem);
    defer hf.deinit();
    const text = hf.text();
    const history = try tf.messages(a, "What is 2^66?");
    const py = try callTokens(a, text, history,
        \\{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"python","arguments":"{\"code\":\"print(2**66)\"}"}}]}
    );
    const f = try callTokens(a, text, history,
        \\{"role":"assistant","content":"","tool_calls":[{"id":"c2","type":"function","function":{"name":"f","arguments":"{}"}}]}
    );
    const first = try text.encodePlain(a, trace);
    const second = try text.encodePlain(a, later);
    try t.expect(second.len > first.len);
    var script: Script = .{
        .a = a,
        .segments = &.{ try std.mem.concat(a, u32, &.{ first, py }), try std.mem.concat(a, u32, &.{ second, f }) },
        // the first segment's second record lands inside the python call: no byte the prose keeps
        .writes = &.{ &.{ .{ .at = 0, .n = first.len }, .{ .at = first.len + py.len / 2, .n = 1 } }, &.{.{ .at = 0, .n = second.len }} },
    };
    const srv = try @import("server.zig").Server.init(a, t.io, script.engine(), text, .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const cfg: lane.Config = .{ .python = python };
    srv.info.compute = cfg;
    srv.compute = lane.Pool.init(cfg, &.{ "/bin/sh", "-c", "exec \"$0\" -I -B -u -c \"$1\"", python, lane.runner.runner });
    var reply: Reply = .{ .a = a };
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    try body.put(a, "max_tokens", try json.intValue(a, 600));
    try body.put(a, "stream", .{ .bool = stream });
    try body.put(a, "messages", history);
    try body.put(a, "tools", try lane_test.obj(a, "[{\"type\":\"code_interpreter\"},{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]"));
    @import("openai.zig").run(srv, a, reply.out(), .{ .conn = &conn }, true, .{ .object = body });
    try t.expectEqual(@as(usize, 2), script.next); // the python result went in, the client's call ended the reply
    return reply;
}

test "compute lane: owned trace lines before the lane's call and a client's call keep their newlines, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const whole = try run(a, false);
    errdefer std.debug.print("OWNED_REPLY {s}\n", .{json.stringify(a, whole.body, .{}) catch ""});
    try t.expectEqual(@as(u16, 200), whole.code);
    const choice = whole.body.get("choices").?.array[0];
    try t.expectEqualStrings("tool_calls", choice.get("finish_reason").?.string);
    try t.expectEqualStrings(trace ++ later, choice.get("message").?.strField("content").?);
    const streamed = try run(a, true);
    try t.expect(streamed.calls);
    try t.expectEqualStrings(trace ++ later, streamed.content.items);
}

/// The same line after a word (a reply led by `[` streams no call), then a call max_tokens cuts, without the lane.
fn cut(a: Allocator, stream: bool) !Reply {
    var bytes: @import("runner_projection_http_test.zig").Bytes = .{};
    const said = "So:\n" ++ trace ++ "\n<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n<parameter=content>\nThe history of";
    const reply = try a.alloc(u32, said.len);
    for (said, reply) |b, *id| id.* = b;
    var script: Script = .{ .a = a, .segments = &.{reply}, .writes = &.{&.{.{ .at = 4, .n = trace.len }}} };
    const srv = try @import("server.zig").Server.init(a, t.io, script.engine(), bytes.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var out: Reply = .{ .a = a };
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    try body.put(a, "max_tokens", try json.intValue(a, reply.len));
    try body.put(a, "stream", .{ .bool = stream });
    try body.put(a, "messages", try lane_test.obj(a, "[{\"role\":\"user\",\"content\":\"What is 2^66?\"}]"));
    try body.put(a, "tools", try lane_test.obj(a, "[{\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}}}]"));
    @import("openai.zig").run(srv, a, out.out(), .{ .conn = &conn }, true, .{ .object = body });
    return out;
}

test "runner projection: HTTP an owned trace line before a call max_tokens cut keeps its newline, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const whole = try cut(a, false);
    errdefer std.debug.print("OWNED_CUT_REPLY {s}\n", .{json.stringify(a, whole.body, .{}) catch ""});
    try t.expectEqual(@as(u16, 200), whole.code);
    const choice = whole.body.get("choices").?.array[0];
    try t.expectEqualStrings("length", choice.get("finish_reason").?.string);
    const calls = choice.get("message").?.get("tool_calls") orelse .null;
    try t.expect(calls != .null);
    try t.expectEqualStrings("So:\n" ++ trace, choice.get("message").?.strField("content").?);
    const streamed = try cut(a, true);
    try t.expect(streamed.calls);
    try t.expectEqualStrings("So:\n" ++ trace, streamed.content.items);
}
