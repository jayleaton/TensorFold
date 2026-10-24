//! MiniMax-M3.1 replies over HTTP: think split, tool calls, Messages events, effort, grammars with thinking off.
const std = @import("std");
const platform = @import("sys");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;

const ns = "]<]minimax[>[";
const specials = [_][]const u8{ ns, "<mm:think>", "</mm:think>" };
const ns_id = 300;
const open_id = 301;
const close_id = 302;

/// Bytes as tokens, MiniMax's specials as one token each; the prompt ends as the template's generation prompt does.
const Text = struct {
    switched: bool = false, // Mia's template variant: thinking on opens the block, off closes it; else adaptive

    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        outer: while (i < input.len) {
            for (specials, 0..) |s, k| if (std.mem.startsWith(u8, input[i..], s)) {
                try out.append(a, @intCast(ns_id + k));
                i += s.len;
                continue :outer;
            };
            try out.append(a, input[i]);
            i += 1;
        }
        return out.items;
    }
    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id >= ns_id and id < ns_id + specials.len) try out.appendSlice(a, specials[id - ns_id]) else if (id < 256) try out.append(a, @intCast(id));
        return out.items;
    }
    fn tokenId(_: *anyopaque, s: []const u8) ?u32 {
        for (specials, 0..) |x, k| if (std.mem.eql(u8, x, s)) return @intCast(ns_id + k);
        return null;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return decode(undefined, a, &.{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 304;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{303};
    }
    fn render(ptr: *anyopaque, a: std.mem.Allocator, _: json.Value, o: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        const t: *Text = @ptrCast(@alignCast(ptr));
        const tail = if (!o.add_generation_prompt) "" else if (!t.switched) "]~b]ai\n" else if (o.enable_thinking) "]~b]ai\n<mm:think>" else "]~b]ai\n</mm:think>";
        const effort = if (o.reasoning_effort) |e| try std.mem.concat(a, u8, &.{ "<effort>", e, "</effort>" }) else "";
        return .{ .text = try std.mem.concat(a, u8, &.{ effort, "]~b]user\nhi[e~[\n", tail }) };
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Records the submitted request and answers `reply` a token at a time.
const Script = struct {
    reply: []const u32 = &.{},
    first_mask: bool = true,
    prompt: [64]u32 = undefined,
    prompt_len: usize = 0,
    think_close: [8]u32 = undefined,
    think_close_len: usize = 0,
    think_budget: u32 = 0,
    after: ?u32 = null,
    structured: bool = false,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const e: *Script = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 4096, .structures = true, .first_mask = e.first_mask };
    }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Script = @ptrCast(@alignCast(ctx));
        e.prompt_len = @min(r.prompt.len, e.prompt.len);
        @memcpy(e.prompt[0..e.prompt_len], r.prompt[0..e.prompt_len]);
        e.think_close_len = @min(r.think_close.len, e.think_close.len);
        @memcpy(e.think_close[0..e.think_close_len], r.think_close[0..e.think_close_len]);
        e.think_budget = r.think_budget;
        e.structured = r.structure != null;
        e.after = if (r.structure) |s| s.after else null;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (e.reply) |t| sink.event(sink.ctx, id, &.{ .tokens = &.{t} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The grammar compiler's boundary: every grammar compiles; no xgrammar library is loaded.
const Compile = struct {
    fn compiler(c: *Compile) api.grammar.Compiler {
        var out: api.grammar.Compiler = undefined;
        out.h = c;
        out.words = 10;
        out.mutex = .init;
        out.f.compile = compile;
        out.f.free = free;
        out.f.close = free;
        return out;
    }
    fn compile(ptr: *anyopaque, _: c_int, _: [*]const u8, _: usize, _: [*]u8, _: usize) callconv(.c) ?*anyopaque {
        return ptr;
    }
    fn free(_: *anyopaque) callconv(.c) void {}
};

const Exchange = struct { status: u16, body: []const u8 };

fn post(a: std.mem.Allocator, srv: *server.Server, path: []const u8, body: []const u8) !Exchange {
    var pair: [2]platform.posix.socket_t = undefined;
    if (@import("builtin").os.tag == .windows) try platform.posix.socketpair(&pair) else if (platform.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = platform.posix.system.close(pair[0]);
    defer _ = platform.posix.system.close(pair[1]);
    const head = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{ path, body.len });
    try writeAll(pair[0], head);
    try writeAll(pair[0], body);
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    if (try conn.readRequest(a, false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, a);
    var out: std.ArrayList(u8) = .empty;
    var tmp: [4096]u8 = undefined;
    while (true) {
        var fds = [_]platform.posix.pollfd{.{ .fd = pair[0], .events = platform.posix.POLL.IN, .revents = 0 }};
        if (try platform.posix.poll(&fds, 200) == 0) break;
        const n = try platform.posix.read(pair[0], &tmp);
        if (n == 0) break;
        try out.appendSlice(a, tmp[0..n]);
    }
    const split = std.mem.indexOf(u8, out.items, "\r\n\r\n") orelse return error.BadReply;
    return .{ .status = try std.fmt.parseInt(u16, out.items["HTTP/1.1 ".len..][0..3], 10), .body = out.items[split + 4 ..] };
}

fn writeAll(fd: platform.posix.socket_t, data: []const u8) !void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = platform.posix.system.write(fd, data[sent..].ptr, data.len - sent);
        if (n <= 0) return error.Closed;
        sent += @intCast(n);
    }
}

/// What a chat reply carried, whole or rebuilt from its stream's deltas.
const Seen = struct { content: []const u8 = "", reasoning: []const u8 = "", names: []const u8 = "", arguments: []const u8 = "", finish: []const u8 = "" };

fn str(v: ?json.Value) []const u8 {
    const x = v orelse return "";
    return if (x == .string) x.string else "";
}

fn seen(a: std.mem.Allocator, body: []const u8, stream: bool) !Seen {
    var s: Seen = .{};
    if (!stream) {
        const choice = (try json.parse(a, body)).ok.get("choices").?.array[0];
        const m = choice.get("message").?;
        s.content = str(m.get("content"));
        s.reasoning = str(m.get("reasoning_content"));
        s.finish = str(choice.get("finish_reason"));
        if (m.get("tool_calls")) |calls| if (calls == .array) for (calls.array) |c| {
            s.names = try std.mem.concat(a, u8, &.{ s.names, str(c.get("function").?.get("name")), ";" });
            s.arguments = try std.mem.concat(a, u8, &.{ s.arguments, str(c.get("function").?.get("arguments")), ";" });
        };
        return s;
    }
    var calls: std.ArrayList([2][]const u8) = .empty;
    var lines = std.mem.splitScalar(u8, body, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: {")) continue;
        const choices = (try json.parse(a, line["data: ".len..])).ok.get("choices") orelse continue;
        if (choices != .array or choices.array.len == 0) continue;
        const d = choices.array[0].get("delta") orelse continue;
        s.content = try std.mem.concat(a, u8, &.{ s.content, str(d.get("content")) });
        s.reasoning = try std.mem.concat(a, u8, &.{ s.reasoning, str(d.get("reasoning_content")) });
        if (choices.array[0].get("finish_reason")) |f| if (f == .string) {
            s.finish = f.string;
        };
        if (d.get("tool_calls")) |tc| if (tc == .array) for (tc.array) |c| {
            const i: usize = @intCast(c.get("index").?.int64().?);
            while (calls.items.len <= i) try calls.append(a, .{ "", "" });
            const f = c.get("function") orelse continue;
            calls.items[i][0] = try std.mem.concat(a, u8, &.{ calls.items[i][0], str(f.get("name")) });
            calls.items[i][1] = try std.mem.concat(a, u8, &.{ calls.items[i][1], str(f.get("arguments")) });
        };
    }
    for (calls.items) |c| {
        s.names = try std.mem.concat(a, u8, &.{ s.names, c[0], ";" });
        s.arguments = try std.mem.concat(a, u8, &.{ s.arguments, c[1], ";" });
    }
    return s;
}

fn tok(a: std.mem.Allocator, text: []const u8) ![]u32 {
    return Text.encode(undefined, a, text, false);
}

const Setup = struct { text: Text = .{}, script: Script = .{}, compile: Compile = .{}, compiler: api.grammar.Compiler = undefined };

fn start(s: *Setup, thinking: bool) !*server.Server {
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, s.script.engine(), s.text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = thinking, .use_drafts = false }, null);
    s.compiler = s.compile.compiler();
    srv.grammars = &s.compiler;
    return srv;
}

/// The `field` text a Messages stream's `kind` deltas carry, in order.
fn deltas(a: std.mem.Allocator, body: []const u8, kind: []const u8, field: []const u8) ![]const u8 {
    var out: []const u8 = "";
    var lines = std.mem.splitScalar(u8, body, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: {")) continue;
        const d = (try json.parse(a, line["data: ".len..])).ok.get("delta") orelse continue;
        if (std.mem.eql(u8, str(d.get("type")), kind)) out = try std.mem.concat(a, u8, &.{ out, str(d.get(field)) });
    }
    return out;
}

const chat_tools = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}}}}}]";

test "MiniMax replies: reasoning splits by its think markers, whole and streamed, and an adaptive skip leaves no close" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { reply: []const u8, reasoning: []const u8, content: []const u8 }{
        .{ .reply = "<mm:think>\nplan it\n</mm:think>\n\nParis.", .reasoning = "plan it", .content = "Paris." },
        .{ .reply = "</mm:think>\n\nParis.", .reasoning = "", .content = "Paris." },
        .{ .reply = "Paris.", .reasoning = "", .content = "Paris." },
    };
    for (cases) |c| for ([_]bool{ false, true }) |stream| {
        var s: Setup = .{};
        s.script.reply = try tok(a, c.reply);
        const srv = try start(&s, true);
        defer srv.deinit();
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":64,\"stream\":{any},\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}", .{stream});
        const got = try post(a, srv, "/v1/chat/completions", body);
        try std.testing.expectEqual(@as(u16, 200), got.status);
        const r = try seen(a, got.body, stream);
        try std.testing.expectEqualStrings(c.content, r.content);
        try std.testing.expectEqualStrings(c.reasoning, std.mem.trim(u8, r.reasoning, "\n"));
        try std.testing.expect(std.mem.indexOf(u8, got.body, "mm:think") == null);
    };
}

test "MiniMax replies: a thinking reply's parallel calls are tool_calls whole and streamed, never content" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = "<mm:think>two cities</mm:think>" ++ ns ++ "<tool_call>\n" ++ ns ++ "<invoke name=\"get_weather\">" ++ ns ++ "<city>Paris" ++ ns ++ "</city>" ++ ns ++ "<days>3" ++ ns ++ "</days>" ++ ns ++ "</invoke>\n" ++
        ns ++ "<invoke name=\"get_weather\">" ++ ns ++ "<city>Lyon \"centre\"" ++ ns ++ "</city>" ++ ns ++ "</invoke>\n" ++ ns ++ "</tool_call>";
    var want: ?Seen = null;
    for ([_]bool{ false, true }) |stream| {
        var s: Setup = .{};
        s.script.reply = try tok(a, reply);
        const srv = try start(&s, true);
        defer srv.deinit();
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":256,\"stream\":{any},\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],{s}}}", .{ stream, chat_tools });
        const got = try post(a, srv, "/v1/chat/completions", body);
        try std.testing.expectEqual(@as(u16, 200), got.status);
        const r = try seen(a, got.body, stream);
        try std.testing.expectEqualStrings("get_weather;get_weather;", r.names);
        try std.testing.expectEqualStrings("{\"city\":\"Paris\",\"days\":3};{\"city\":\"Lyon \\\"centre\\\"\"};", r.arguments);
        try std.testing.expectEqualStrings("two cities", r.reasoning);
        try std.testing.expectEqualStrings("", r.content);
        try std.testing.expectEqualStrings("tool_calls", r.finish);
        try std.testing.expect(std.mem.indexOf(u8, got.body, "minimax") == null);
        if (want) |w| try std.testing.expectEqualStrings(w.arguments, r.arguments) else want = r;
    }
}

test "MiniMax on Messages: the stream's head counts the prompt, thinking and tool_use blocks carry the reply" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = "<mm:think>look it up</mm:think>" ++ ns ++ "<tool_call>\n" ++ ns ++ "<invoke name=\"get_weather\">" ++ ns ++ "<city>Paris" ++ ns ++ "</city>" ++ ns ++ "</invoke>\n" ++ ns ++ "</tool_call>";
    const tools = "\"tools\":[{\"name\":\"get_weather\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}]";
    for ([_]bool{ false, true }) |stream| {
        var s: Setup = .{};
        s.script.reply = try tok(a, reply);
        const srv = try start(&s, true);
        defer srv.deinit();
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":256,\"stream\":{any},\"thinking\":{{\"type\":\"adaptive\"}},\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],{s}}}", .{ stream, tools });
        const got = try post(a, srv, "/v1/messages", body);
        try std.testing.expectEqual(@as(u16, 200), got.status);
        try std.testing.expectEqualStrings("look it up", if (stream) try deltas(a, got.body, "thinking_delta", "thinking") else "look it up");
        for ([_][]const u8{ "\"thinking\"", if (stream) "thinking_delta" else "look it up", "\"tool_use\"", "\"get_weather\"" }) |w| std.testing.expect(std.mem.indexOf(u8, got.body, w) != null) catch |e| {
            std.debug.print("missing {s} in {s}\n", .{ w, got.body });
            return e;
        };
        try std.testing.expect(std.mem.indexOf(u8, got.body, "minimax") == null and std.mem.indexOf(u8, got.body, "mm:think") == null);
        if (stream) {
            const flat = try std.mem.replaceOwned(u8, a, got.body, " ", "");
            const head = flat[0 .. std.mem.indexOf(u8, flat, "content_block_start") orelse return error.NoBlock];
            const counted = try std.fmt.allocPrint(a, "\"input_tokens\":{d}", .{s.script.prompt_len});
            try std.testing.expect(std.mem.indexOf(u8, head, counted) != null);
            try std.testing.expectEqualStrings("{\"city\":\"Paris\"}", try deltas(a, got.body, "input_json_delta", "partial_json"));
            try std.testing.expect(std.mem.indexOf(u8, flat, "\"stop_reason\":\"tool_use\"") != null);
        } else try std.testing.expect(std.mem.indexOf(u8, try std.mem.replaceOwned(u8, a, got.body, " ", ""), "\"input\":{\"city\":\"Paris\"}") != null);
    }
}

test "MiniMax thinking off: the adaptive prompt gets its close; a grammar on a first-mask-less engine forces it first" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |switched| {
        // plain text: the prompt ends in the close whichever template variant renders it
        var s: Setup = .{ .text = .{ .switched = switched } };
        s.script.reply = try tok(a, "Paris.");
        const srv = try start(&s, false);
        defer srv.deinit();
        const plain = try post(a, srv, "/v1/chat/completions", "{\"model\":\"m\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
        try std.testing.expectEqual(@as(u16, 200), plain.status);
        try std.testing.expectEqual(@as(u32, close_id), s.script.prompt[s.script.prompt_len - 1]);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u32, s.script.prompt[0..s.script.prompt_len], &.{close_id}));
        try std.testing.expect(!s.script.structured and s.script.think_budget == 0);
        // response_format: the close leaves the prompt and is the forced first token; the grammar starts after it
        s.script.first_mask = false;
        srv.info.first_mask = false;
        s.script.reply = try tok(a, "</mm:think>{\"city\": \"Paris\"}");
        for ([_]bool{ false, true }) |stream| {
            const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":32,\"stream\":{any},\"response_format\":{{\"type\":\"json_object\"}},\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}", .{stream});
            const got = try post(a, srv, "/v1/chat/completions", body);
            try std.testing.expectEqual(@as(u16, 200), got.status);
            try std.testing.expect(s.script.structured);
            try std.testing.expect(s.script.prompt[s.script.prompt_len - 1] != close_id);
            try std.testing.expectEqualSlices(u32, &.{close_id}, s.script.think_close[0..s.script.think_close_len]);
            try std.testing.expectEqual(@as(u32, 1), s.script.think_budget);
            try std.testing.expectEqual(@as(?u32, close_id), s.script.after);
            const r = try seen(a, got.body, stream);
            try std.testing.expectEqualStrings("{\"city\": \"Paris\"}", r.content);
            try std.testing.expectEqualStrings("", r.reasoning);
        }
        // the same request with thinking on starts its grammar after the close the model writes
        const on = try post(a, srv, "/v1/chat/completions", "{\"model\":\"m\",\"max_tokens\":32,\"chat_template_kwargs\":{\"enable_thinking\":true},\"response_format\":{\"type\":\"json_object\"},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
        try std.testing.expectEqual(@as(u16, 200), on.status);
        try std.testing.expectEqual(@as(?u32, close_id), s.script.after);
        try std.testing.expectEqual(@as(u32, 0), s.script.think_budget);
        try std.testing.expectEqual(@as(u32, open_id), s.script.prompt[s.script.prompt_len - 1]); // adaptive: opened
        const free = try post(a, srv, "/v1/chat/completions", "{\"model\":\"m\",\"max_tokens\":8,\"chat_template_kwargs\":{\"enable_thinking\":true},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
        try std.testing.expectEqual(@as(u16, 200), free.status);
        try std.testing.expectEqual(@as(u32, if (switched) open_id else '\n'), s.script.prompt[s.script.prompt_len - 1]); // no grammar: as rendered
    }
}

test "MiniMax thinking budget: the forced close is MiniMax's own, and required tools compile its invoke grammar" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var s: Setup = .{};
    s.script.reply = try tok(a, "<mm:think>a</mm:think>ok");
    const srv = try start(&s, true);
    defer srv.deinit();
    const got = try post(a, srv, "/v1/chat/completions", "{\"model\":\"m\",\"max_tokens\":32,\"thinking_budget\":5,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    try std.testing.expectEqual(@as(u16, 200), got.status);
    try std.testing.expectEqual(@as(u32, 5), s.script.think_budget);
    try std.testing.expectEqualSlices(u32, &.{ '\n', close_id, '\n', '\n' }, s.script.think_close[0..s.script.think_close_len]);
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":256,\"tool_choice\":\"required\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],{s}}}", .{chat_tools});
    s.script.reply = try tok(a, "</mm:think>" ++ ns ++ "<tool_call>\n" ++ ns ++ "<invoke name=\"get_weather\">" ++ ns ++ "<city>Paris" ++ ns ++ "</city>" ++ ns ++ "</invoke>\n" ++ ns ++ "</tool_call>");
    const required = try post(a, srv, "/v1/chat/completions", body);
    try std.testing.expectEqual(@as(u16, 200), required.status);
    try std.testing.expect(s.script.structured and s.script.after == close_id);
    const r = try seen(a, required.body, false);
    try std.testing.expectEqualStrings("get_weather;", r.names);
}

test "MiniMax effort: its five levels reach the template as sent, and an unknown one takes the nearest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var s: Setup = .{};
    s.script.reply = try tok(a, "ok");
    const srv = try start(&s, true);
    defer srv.deinit();
    for ([_][2][]const u8{ .{ "low", "low" }, .{ "medium", "medium" }, .{ "high", "high" }, .{ "xhigh", "xhigh" }, .{ "max", "max" }, .{ "minimal", "low" } }) |e| {
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":8,\"reasoning_effort\":\"{s}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}", .{e[0]});
        const got = try post(a, srv, "/v1/chat/completions", body);
        try std.testing.expectEqual(@as(u16, 200), got.status);
        const prompt = try Text.decode(undefined, a, s.script.prompt[0..s.script.prompt_len]);
        try std.testing.expect(std.mem.startsWith(u8, prompt, try std.mem.concat(a, u8, &.{ "<effort>", e[1], "</effort>" })));
    }
}
