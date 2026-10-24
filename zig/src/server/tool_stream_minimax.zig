//! MiniMax-M3.1 calls while they are written: a string parameter streams as it comes, any other once it closes.
const std = @import("std");
const json = @import("json");
const ids = @import("ids.zig");
const reply_text = @import("reply_text.zig");
const tool_minimax = @import("tool_minimax.zig");
const tool_params = @import("tool_params.zig");
const wrap = @import("tool_streaming.zig").wrapCalls;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// Offered tools by lowercase name, as the streamer keeps them.
pub const Known = std.StringHashMapUnmanaged([]const u8);

/// What a streamed block reads beside the reply: the tools' schemas, their names, and the last call index sent.
pub const Ctx = struct { a: Allocator, schemas: *const tool_params.Schemas, known: *const Known, index: *i64 };

/// The bytes of `s` before a UTF-8 sequence the end cuts short.
fn whole(s: []const u8) usize {
    var back: usize = 0;
    while (back < @min(s.len, 3) and s[s.len - 1 - back] & 0xc0 == 0x80) back += 1;
    if (back == s.len) return s.len;
    const lead = s[s.len - 1 - back];
    const need = std.unicode.utf8ByteSequenceLength(lead) catch return s.len;
    return if (need > back + 1) s.len - 1 - back else s.len;
}

/// One block from its opener: a header per offered invoke, then each parameter as it closes (`take` reads the same).
pub const Block = struct {
    start: usize, // where the block's inner text starts in the reply
    searched: usize, // the reply holds no closer before this
    reader: tool_minimax.Reader = .{},
    done: usize = 0, // invokes streamed to their end
    seen: bool = false, // the current invoke's name was looked up
    index: ?i64 = null, // the current invoke's call index, null when it is no call
    tool: []const u8 = "",
    params: usize = 0, // its parameters sent whole
    keys: std.StringHashMapUnmanaged(void) = .empty,
    touched: bool = false, // the open parameter was looked at
    skip: bool = false, // it repeats a key: the first value stands
    textual: bool = false,
    quoted: bool = false, // its string is open in the arguments
    sent: usize = 0, // bytes of its text sent
    any: bool = false, // the arguments' `{` went out

    pub fn init(start: usize) Block {
        return .{ .start = start, .searched = start };
    }

    fn args(b: *const Block, c: Ctx, out: *std.ArrayList(Value), fragment: []const u8) Allocator.Error!void {
        const function = try json.newObject(c.a);
        try function.put(c.a, "arguments", .{ .string = fragment });
        const one = try json.newObject(c.a);
        try one.put(c.a, "index", try json.intValue(c.a, b.index.?));
        try one.put(c.a, "function", .{ .object = function });
        try out.append(c.a, try wrap(c.a, .{ .object = one }));
    }

    fn head(b: *const Block, c: Ctx, out: *std.ArrayList(Value)) Allocator.Error!void {
        const function = try json.newObject(c.a);
        try function.put(c.a, "name", .{ .string = b.tool });
        try function.put(c.a, "arguments", .{ .string = "" });
        const one = try json.newObject(c.a);
        try one.put(c.a, "index", try json.intValue(c.a, b.index.?));
        try one.put(c.a, "id", .{ .string = try ids.make(c.a, "call_", 24) });
        try one.put(c.a, "type", .{ .string = "function" });
        try one.put(c.a, "function", .{ .object = function });
        try out.append(c.a, try wrap(c.a, .{ .object = one }));
    }

    fn esc(a: Allocator, text: []const u8) Allocator.Error![]const u8 {
        const q = try json.quote(a, text, .{ .ascii = false });
        return q[1 .. q.len - 1];
    }

    /// `{` or `,`, then the key and its colon.
    fn key(b: *Block, a: Allocator, name: []const u8) Allocator.Error![]const u8 {
        const lead: []const u8 = if (b.any) "," else "{";
        b.any = true;
        return std.mem.concat(a, u8, &.{ lead, try json.quote(a, name, .{ .ascii = false }), ":" });
    }

    /// The deltas the reply's `text` (its whole answer so far) adds; the reply's offset past the block once it closed.
    pub fn feed(b: *Block, c: Ctx, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!?usize {
        const end = reply_text.findTag(text, @max(b.start, b.searched -| (tool_minimax.close.len - 1)), tool_minimax.close);
        if (end == null) b.searched = text.len;
        try b.reader.feed(c.a, text[b.start .. end orelse text.len], end != null);
        while (b.done < b.reader.calls.items.len) {
            const call = b.reader.calls.items[b.done];
            if (!b.seen) {
                b.seen = true;
                if (try tool_minimax.offered(c.a, call.name, c.known)) |tool| {
                    c.index.* += 1;
                    b.index = c.index.*;
                    b.tool = tool;
                    try b.head(c, out);
                }
            }
            if (b.index != null) try b.parameters(c, call, out);
            if (call.end == null) break;
            if (b.index != null) try b.args(c, out, if (b.any) "}" else "{}");
            const kept = b.*;
            b.* = .{ .start = kept.start, .searched = kept.searched, .reader = kept.reader, .done = kept.done + 1 };
        }
        return if (end) |e| e + tool_minimax.close.len else null;
    }

    /// Sends each parameter that closed, and the text the open one gained when it is a string.
    fn parameters(b: *Block, c: Ctx, call: *const tool_minimax.Call, out: *std.ArrayList(Value)) Allocator.Error!void {
        const items = call.args.params.items;
        while (b.params < items.len) {
            const p = items[b.params];
            if (!b.touched) {
                b.touched = true;
                b.skip = b.keys.contains(p.name);
                if (!b.skip) try b.keys.put(c.a, p.name, {});
                b.textual = try tool_minimax.textual(c.a, try tool_params.schemaOf(c.schemas, b.tool, p.name, c.a));
            }
            const stack = call.args.stack.items;
            if (stack.len > 0 and stack[0] == p) {
                // open: a text-only leaf streams once it holds more than whitespace (until then it may hold elements)
                if (b.skip or !b.textual or p.kids.items.len > 0) return;
                const have = p.text.items[0..whole(p.text.items)];
                if (!b.quoted) {
                    if (reply_text.pyStrip(have).len == 0) return;
                    try b.args(c, out, try std.mem.concat(c.a, u8, &.{ try b.key(c.a, p.name), "\"" }));
                    b.quoted = true;
                }
                if (have.len > b.sent) try b.args(c, out, try esc(c.a, have[b.sent..]));
                b.sent = have.len;
                return;
            }
            if (b.quoted) {
                try b.args(c, out, try std.mem.concat(c.a, u8, &.{ try esc(c.a, p.text.items[b.sent..]), "\"" }));
            } else if (!b.skip) {
                const v = try tool_minimax.paramValue(c.a, p, c.schemas, b.tool);
                const k = try b.key(c.a, p.name);
                try b.args(c, out, try std.mem.concat(c.a, u8, &.{ k, try json.stringify(c.a, v, .{ .ascii = false, .compact = true }) }));
            }
            b.params += 1;
            b.touched = false;
            b.quoted = false;
            b.sent = 0;
        }
    }
};
