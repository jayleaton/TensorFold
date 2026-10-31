//! A completion prompt's reply spans (Request.decode_spans): closed replies always, an open last one on `open_replies`.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

/// Bytes as tokens, '$' the end token; the template is "<u>x" and its generation prompt "<a>".
const ByteText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id < 256) try out.append(a, @intCast(id));
        return out.toOwnedSlice(a);
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{'$'};
    }

    fn render(_: *anyopaque, a: Allocator, _: json.Value, o: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, if (o.add_generation_prompt) "<u>x<a>" else "<u>x") };
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Copies the spans the server submits, then finishes at once.
const Capture = struct {
    open: bool = false,
    spans: [][2]u32 = &.{},

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        return .{ .open_replies = e.open };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        std.testing.allocator.free(e.spans);
        e.spans = std.testing.allocator.dupe([2]u32, request.decode_spans) catch return error.Busy;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'o', 'k' } });
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

const Reply = struct {
    status: ?u16 = null,

    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(_: *anyopaque, _: usize) error{Closed}!void {}

    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}

    fn reply(ctx: *anyopaque, status: u16, _: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.status = status;
    }
};

/// The status of a completion of the text `prompt` (its bytes the ids); `backend.spans` keeps the submitted spans.
fn complete(backend: *Capture, prompt: []const u8) !?u16 {
    var text: ByteText = .{};
    const srv = try Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var ids: std.ArrayList(u8) = .empty;
    for (prompt, 0..) |c, i| try ids.print(a, "{s}{d}", .{ if (i == 0) "" else ",", c });
    const raw = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":2,\"prompt\":[{s}]}}", .{ids.items}))).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var reply: Reply = .{};
    openai.run(srv, a, reply.out(), .{ .conn = &conn }, false, raw);
    return reply.status;
}

test "reply spans: each closed reply through its end token; an open last reply only with open_replies" {
    const prompt = "<u>x<a>hi$<u>y<a>ok";
    for ([_]bool{ false, true }) |open| {
        var backend: Capture = .{ .open = open };
        defer std.testing.allocator.free(backend.spans);
        try std.testing.expectEqual(@as(?u16, 200), try complete(&backend, prompt));
        const closed = [2]u32{ 7, 10 };
        if (open) {
            try std.testing.expectEqualSlices([2]u32, &.{ closed, .{ 17, 19 } }, backend.spans);
        } else try std.testing.expectEqualSlices([2]u32, &.{closed}, backend.spans);
    }
}

test "reply spans: a prompt that ends at a generation prompt has no open reply, open_replies or not" {
    var backend: Capture = .{ .open = true };
    defer std.testing.allocator.free(backend.spans);
    try std.testing.expectEqual(@as(?u16, 200), try complete(&backend, "<u>x<a>hi$<u>y<a>"));
    try std.testing.expectEqualSlices([2]u32, &.{.{ 7, 10 }}, backend.spans);
}
