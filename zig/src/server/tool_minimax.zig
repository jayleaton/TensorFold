//! MiniMax-M3.1's tool calls (`NS<tool_call>` ... `NS</tool_call>`), parameters read back from the template's to_xml.
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const tool_kinds = @import("tool_kinds.zig");
const tool_params = @import("tool_params.zig");
const tool_parse = @import("tool_parse.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const ns = "]<]minimax[>[";
pub const open = ns ++ "<tool_call>";
pub const close = ns ++ "</tool_call>";

pub const Invoke = struct { name: []const u8, arguments: Value };

/// One block found in a reply: where it starts and ends, and the text between its markers.
pub const Block = struct { start: usize, end: usize, payload: []const u8, complete: bool };

/// The first block whose opener starts at or after `from`; an unclosed one runs to the end of the text.
pub fn nextBlock(text: []const u8, from: usize) ?Block {
    const s = reply_text.findTag(text, from, open) orelse return null;
    const e = reply_text.findTag(text, s + open.len, close) orelse return .{ .start = s, .end = text.len, .payload = text[s..], .complete = false };
    return .{ .start = s, .end = e + close.len, .payload = reply_text.pyStrip(text[s + open.len .. e]), .complete = true };
}

/// Every block in `text` appended to `found`, a list of envelopes with `Block`'s four fields.
pub fn appendBlocks(comptime E: type, a: Allocator, found: *std.ArrayList(E), text: []const u8) Allocator.Error!void {
    var pos: usize = 0;
    while (nextBlock(text, pos)) |b| : (pos = b.end) try found.append(a, .{ .start = b.start, .end = b.end, .payload = b.payload, .complete = b.complete });
}

/// Where the block the reply ends inside starts, if it does.
pub fn unclosed(text: []const u8) ?usize {
    var at: usize = 0;
    while (nextBlock(text, at)) |b| : (at = b.end) {
        if (!b.complete) return b.start;
    }
    return null;
}

/// One parameter element: its text, or the elements inside it.
pub const Node = struct {
    name: []const u8,
    text: std.ArrayList(u8) = .empty,
    kids: std.ArrayList(*Node) = .empty,
};

fn allSpace(s: []const u8) bool {
    return reply_text.pyStrip(s).len == 0;
}

/// One invoke's parameter elements as they are read: any run of tags makes a tree, so a known invoke is always a call.
pub const Args = struct {
    params: std.ArrayList(*Node) = .empty,
    stack: std.ArrayList(*Node) = .empty, // the elements open now, outermost first

    fn top(p: *const Args) ?*Node {
        return if (p.stack.items.len > 0) p.stack.items[p.stack.items.len - 1] else null;
    }

    /// An element opens; a leaf already holding text ends here first (the model left out its close).
    fn open(p: *Args, a: Allocator, name: []const u8) Allocator.Error!void {
        if (p.top()) |t| if (t.kids.items.len == 0 and !allSpace(t.text.items)) {
            _ = p.stack.pop();
        };
        const node = try a.create(Node);
        node.* = .{ .name = try a.dupe(u8, name) };
        try (if (p.top()) |t| &t.kids else &p.params).append(a, node);
        try p.stack.append(a, node);
    }

    /// The innermost open element named `name` closes, with any left open inside it; false when none is open.
    fn close(p: *Args, name: []const u8) bool {
        var i = p.stack.items.len;
        while (i > 0) : (i -= 1) if (std.mem.eql(u8, p.stack.items[i - 1].name, name)) {
            p.stack.shrinkRetainingCapacity(i - 1);
            return true;
        };
        return false;
    }

    /// Text is the innermost open element's; between parameters it is dropped.
    fn text(p: *Args, a: Allocator, bytes: []const u8) Allocator.Error!void {
        if (p.top()) |t| try t.text.appendSlice(a, bytes);
    }
};

/// One invoke of a block: where it starts and ends in the block's inner text (end null while it is open), its args.
pub const Call = struct { start: usize, end: ?usize = null, name: []const u8, args: Args = .{} };

const Tag = union(enum) { invoke: []const u8, end_invoke, block, open: []const u8, close: []const u8, literal };

fn nameOk(name: []const u8) bool {
    return name.len > 0 and std.mem.indexOfAny(u8, name, " \t\r\n\"<=/>") == null;
}

/// `invoke name="X"` with any spacing around its parts: X.
fn invokeName(tag: []const u8) ?[]const u8 {
    const blank = " \t\r\n";
    var s = std.mem.trimStart(u8, tag, blank);
    if (!std.mem.startsWith(u8, s, "invoke") or s.len == 6 or std.mem.indexOfScalar(u8, blank, s[6]) == null) return null;
    s = std.mem.trimStart(u8, s[6..], blank);
    if (!std.mem.startsWith(u8, s, "name")) return null;
    s = std.mem.trimStart(u8, s[4..], blank);
    if (!std.mem.startsWith(u8, s, "=")) return null;
    s = std.mem.trimStart(u8, s[1..], blank);
    if (!std.mem.startsWith(u8, s, "\"")) return null;
    const q = std.mem.lastIndexOfScalar(u8, s, '"').?;
    if (q == 0 or std.mem.trim(u8, s[q + 1 ..], blank).len > 0) return null;
    return s[1..q];
}

/// The tag a segment (the bytes after one NS) opens with and its length, or null while more bytes may change it.
fn tagOf(seg: []const u8, complete: bool) ?struct { tag: Tag, len: usize } {
    if (seg.len > 0 and seg[0] != '<') return .{ .tag = .literal, .len = 0 };
    const gt = std.mem.indexOfScalar(u8, seg, '>') orelse return if (complete) .{ .tag = .literal, .len = 0 } else null;
    const inner = seg[1..gt];
    const len = gt + 1;
    if (invokeName(inner)) |n| return .{ .tag = .{ .invoke = n }, .len = len };
    const body = std.mem.trim(u8, inner, " \t\r\n");
    if (std.mem.eql(u8, body, "/invoke")) return .{ .tag = .end_invoke, .len = len };
    if (std.mem.eql(u8, body, "tool_call") or std.mem.eql(u8, body, "/tool_call")) return .{ .tag = .block, .len = len };
    if (body.len > 1 and body[0] == '/' and nameOk(body[1..])) return .{ .tag = .{ .close = body[1..] }, .len = len };
    if (nameOk(body)) return .{ .tag = .{ .open = body }, .len = len };
    return .{ .tag = .literal, .len = 0 };
}

/// A block's inner text read as it grows, whole or streamed alike; an invoke ends at `</invoke>` or the next invoke.
pub const Reader = struct {
    calls: std.ArrayList(*Call) = .empty,
    current: ?*Call = null,
    pos: usize = 0,
    in_text: bool = true, // `pos` is inside a segment's text, else at an NS

    fn end(r: *Reader, at: usize) void {
        const c = r.current orelse return;
        c.end = at;
        c.args.stack.clearRetainingCapacity();
        r.current = null;
    }

    fn put(r: *Reader, a: Allocator, bytes: []const u8) Allocator.Error!void {
        if (r.current) |c| try c.args.text(a, bytes);
    }

    /// Reads `text` (all of the inner text so far) from where the last call stopped; `done`: the block closed there.
    pub fn feed(r: *Reader, a: Allocator, text: []const u8, done: bool) Allocator.Error!void {
        while (true) {
            if (r.in_text) {
                const next = reply_text.findTag(text, r.pos, ns);
                const stop = next orelse if (done) text.len else text.len - reply_text.partialTag(text[r.pos..], ns);
                try r.put(a, text[r.pos..stop]);
                r.pos = stop;
                if (next == null) {
                    if (done) r.end(text.len);
                    return;
                }
                r.in_text = false;
            }
            const seg_start = r.pos + ns.len;
            if (seg_start > text.len) { // the NS waited at was the block's closer, cut off once it was whole
                if (done) r.end(text.len);
                return;
            }
            const next = reply_text.findTag(text, seg_start, ns);
            const t = tagOf(text[seg_start .. next orelse text.len], next != null or done) orelse return;
            const tag_end = seg_start + t.len;
            switch (t.tag) {
                .invoke => |name| {
                    r.end(r.pos);
                    const c = try a.create(Call);
                    c.* = .{ .start = r.pos, .name = try a.dupe(u8, name) };
                    try r.calls.append(a, c);
                    r.current = c;
                },
                .end_invoke => r.end(tag_end),
                .block => r.end(r.pos),
                .open => |name| if (r.current) |c| try c.args.open(a, name),
                .close => |name| if (r.current) |c| {
                    _ = c.args.close(name); // a close nothing open matches is dropped, as a bare NS is
                },
                .literal => {},
            }
            r.pos = tag_end;
            r.in_text = true;
        }
    }
};

/// `$ref` ("#/a/b") followed through the root, a bounded number of times.
fn deref(schema: tool_params.Schema) Value {
    var v = schema.value;
    for (0..32) |_| {
        const ref = (v.get("$ref") orelse return v).str() orelse return v;
        if (!std.mem.startsWith(u8, ref, "#/")) return v;
        var cur = schema.root;
        var parts = std.mem.splitScalar(u8, ref[2..], '/');
        while (parts.next()) |p| {
            if (cur == .array) {
                const i = std.fmt.parseInt(usize, p, 10) catch return v;
                if (i >= cur.array.len) return v;
                cur = cur.array[i];
            } else cur = cur.get(p) orelse return v;
        }
        v = cur;
    }
    return v;
}

/// The schema of member `key` (or of the list's items when `key` is null) under `schema`; empty when it declares none.
fn sub(a: Allocator, schema: tool_params.Schema, key: ?[]const u8) Allocator.Error!tool_params.Schema {
    const here: tool_params.Schema = .{ .value = deref(schema), .root = schema.root };
    if (here.value == .object) {
        const direct: ?Value = if (key) |k| blk: {
            if (here.value.get("properties")) |p| if (p.get(k)) |s| if (s == .object) break :blk s;
            if (here.value.get("additionalProperties")) |s| if (s == .object) break :blk s;
            break :blk null;
        } else if (here.value.get("items")) |s| (if (s == .object) s else null) else null;
        if (direct) |s| return .{ .value = s, .root = schema.root };
        for ([_][]const u8{ "anyOf", "oneOf", "allOf" }) |combiner| {
            const branches = here.value.get(combiner) orelse continue;
            if (branches != .array) continue;
            for (branches.array) |branch| {
                const found = try sub(a, .{ .value = branch, .root = schema.root }, key);
                if (found.value == .object and found.value.object.count() > 0) return found;
            }
        }
    }
    return tool_params.emptySchema(a);
}

/// A leaf under `schema` is its own text whatever it holds: the schema names no type but string.
pub fn textual(a: Allocator, schema: tool_params.Schema) Allocator.Error!bool {
    const here: tool_params.Schema = .{ .value = deref(schema), .root = schema.root };
    const kinds = try tool_kinds.schemaKinds(a, here.value, here.root) orelse return true;
    return !tool_kinds.takesJson(kinds);
}

/// An element's value (the module comment's rules); text beside child elements is dropped.
fn value(a: Allocator, node: *const Node, schema: tool_params.Schema) Allocator.Error!Value {
    const here: tool_params.Schema = .{ .value = deref(schema), .root = schema.root };
    const kinds = try tool_kinds.schemaKinds(a, here.value, here.root);
    if (node.kids.items.len == 0) {
        if (node.text.items.len == 0) if (kinds) |k| {
            if (!k.contains(.string)) {
                if (k.contains(.array) and !k.contains(.object)) return .{ .array = &.{} };
                if (k.contains(.object) and !k.contains(.array)) return .{ .object = try json.newObject(a) };
            }
        };
        return tool_params.decode(a, node.text.items, here, false);
    }
    var items = true;
    for (node.kids.items) |k| items = items and std.mem.eql(u8, k.name, "item");
    var as_list = items;
    if (kinds) |k| {
        if (k.contains(.array) and !k.contains(.object)) as_list = true;
        if (k.contains(.object) and !k.contains(.array)) as_list = false;
    }
    if (as_list) {
        const list = try a.alloc(Value, node.kids.items.len);
        const item_schema = try sub(a, here, null);
        for (node.kids.items, list) |kid, *slot| slot.* = try value(a, kid, item_schema);
        return .{ .array = list };
    }
    const object = try json.newObject(a);
    for (node.kids.items) |kid| try object.put(a, kid.name, try value(a, kid, try sub(a, here, kid.name)));
    return .{ .object = object };
}

/// One top-level parameter of a call to offered tool `tool`, read by its schema.
pub fn paramValue(a: Allocator, node: *const Node, schemas: *const tool_params.Schemas, tool: []const u8) Allocator.Error!Value {
    return value(a, node, try tool_params.schemaOf(schemas, tool, node.name, a));
}

/// A call's arguments: each parameter in order, a repeated one keeping its first value (streamed before the repeat).
pub fn arguments(a: Allocator, args: *const Args, schemas: *const tool_params.Schemas, tool: []const u8) Allocator.Error!Value {
    const object = try json.newObject(a);
    for (args.params.items) |p| if (!object.contains(p.name)) try object.put(a, p.name, try paramValue(a, p, schemas, tool));
    return .{ .object = object };
}

/// The offered spelling of an invoke's name, or null when the request did not offer it.
pub fn offered(a: Allocator, name: []const u8, known: anytype) Allocator.Error!?[]const u8 {
    return known.get(try std.ascii.allocLowerString(a, reply_text.pyStrip(name)));
}

/// A block (markers included) into `calls`; false when it is no MiniMax block. An offered tool's invoke is a call.
pub fn take(a: Allocator, block: []const u8, schemas: *const tool_params.Schemas, known: anytype, max_calls: ?usize, calls: *std.ArrayList(Value), residue: *std.ArrayList(u8)) Allocator.Error!bool {
    if (!std.mem.startsWith(u8, block, open)) return false;
    const inner = block[open.len .. block.len - close.len];
    var r: Reader = .{};
    try r.feed(a, inner, true);
    var found: std.ArrayList(Value) = .empty;
    var kept: std.ArrayList(u8) = .empty;
    var cursor: usize = 0;
    for (r.calls.items) |c| {
        try keep(a, &kept, inner[cursor..c.start]);
        cursor = c.end.?;
        const tool = try offered(a, c.name, known) orelse {
            try keep(a, &kept, inner[c.start..cursor]);
            continue;
        };
        if (max_calls == null or calls.items.len + found.items.len < max_calls.?)
            try found.append(a, try tool_parse.openaiCall(a, tool, try arguments(a, &c.args, schemas, tool)));
    }
    try keep(a, &kept, inner[cursor..]);
    // the rest stays text (newline-joined) when calls are unlimited, the whole block when none is a call
    if (found.items.len == 0) {
        if (max_calls == null) try residue.appendSlice(a, block);
        return true;
    }
    try calls.appendSlice(a, found.items);
    if (max_calls == null) try residue.appendSlice(a, kept.items);
    return true;
}

fn keep(a: Allocator, kept: *std.ArrayList(u8), piece: []const u8) Allocator.Error!void {
    const s = reply_text.pyStrip(piece);
    if (s.len == 0) return;
    if (kept.items.len > 0) try kept.append(a, '\n');
    try kept.appendSlice(a, s);
}

/// What closes the block the reply ended in (`text` from its opener): an open invoke's close, the block's, else "".
pub fn closing(a: Allocator, text: []const u8, tools: []const Value) Allocator.Error![]const u8 {
    const opened = std.mem.lastIndexOf(u8, text, ns ++ "<invoke");
    const shut = std.mem.lastIndexOf(u8, text, ns ++ "</invoke>");
    const tail = if (opened != null and (shut == null or shut.? < opened.?)) ns ++ "</invoke>" ++ close else close;
    const closed = try std.mem.concat(a, u8, &.{ text, tail });
    return if ((try tool_parse.parse(a, closed, tools, 1)).calls != null) tail else "";
}
