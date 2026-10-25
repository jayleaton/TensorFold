//! MiniMax-M3.1's tool calls: `NS<tool_call>` `NS<invoke name="n">` `NS<k>value NS</k>` ... `NS</invoke>` `NS</tool_call>`,
//! NS being the special token `]<]minimax[>[`. Parameters are the chat template's own `to_xml` read back: a mapping is
//! child elements, a list is repeated `<item>` elements, None is omitted, booleans are `true`/`false`, any other scalar is
//! its `str()` text. The template never says how scalars were typed, so reading them back is a decision (UNVERIFIED
//! against MiniMax's own parser): a leaf takes the type its tool's schema names, through the same `tool_params.decode`
//! the Qwen and GLM branches use, and is the text itself (a string) wherever no schema is declared or the text does not
//! spell the declared type. An element with children is a list when its schema says array (not object), an object when
//! it says object (not array), and otherwise a list exactly when every child is `item`. An empty element (an empty list,
//! mapping or string all render as nothing) is "" unless the schema is array-only ([]) or object-only ({}).
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

pub const Error = error{ Invalid, OutOfMemory };
pub const Invoke = struct { name: []const u8, arguments: Value };

/// One block found in a reply: where it starts and ends, and the text between its markers.
pub const Block = struct { start: usize, end: usize, payload: []const u8, complete: bool };

/// The first block whose opener starts at or after `from`; an unclosed one runs to the end of the text.
pub fn nextBlock(text: []const u8, from: usize) ?Block {
    const s = reply_text.findTag(text, from, open) orelse return null;
    const e = reply_text.findTag(text, s + open.len, close) orelse return .{ .start = s, .end = text.len, .payload = text[s..], .complete = false };
    return .{ .start = s, .end = e + close.len, .payload = reply_text.pyStrip(text[s + open.len .. e]), .complete = true };
}

/// Where the block the reply ends inside starts, if it does.
pub fn unclosed(text: []const u8) ?usize {
    var at: usize = 0;
    while (nextBlock(text, at)) |b| : (at = b.end) {
        if (!b.complete) return b.start;
    }
    return null;
}

const Node = struct {
    name: []const u8,
    text: []const u8 = "",
    kids: std.ArrayList(*Node) = .empty,
};

fn allSpace(s: []const u8) bool {
    return reply_text.pyStrip(s).len == 0;
}

/// Every invoke of a block's payload with its arguments typed by `schemas`; Invalid when anything else is in it.
pub fn calls(a: Allocator, block: []const u8, schemas: *const tool_params.Schemas) Error![]Invoke {
    var out: std.ArrayList(Invoke) = .empty;
    var stack: std.ArrayList(*Node) = .empty;
    var parts = std.mem.splitSequence(u8, block, ns);
    if (!allSpace(parts.first())) return error.Invalid;
    while (parts.next()) |seg| {
        if (seg.len == 0 or seg[0] != '<') return error.Invalid;
        const gt = std.mem.indexOfScalar(u8, seg, '>') orelse return error.Invalid;
        const tag = seg[1..gt];
        const rest = seg[gt + 1 ..];
        const invoke_head = "invoke name=\"";
        if (std.mem.startsWith(u8, tag, invoke_head)) {
            if (stack.items.len != 0 or tag.len < invoke_head.len + 1 or tag[tag.len - 1] != '"') return error.Invalid;
            const node = try a.create(Node);
            node.* = .{ .name = tag[invoke_head.len .. tag.len - 1] };
            if (!allSpace(rest)) return error.Invalid;
            try stack.append(a, node);
        } else if (std.mem.eql(u8, tag, "/invoke")) {
            if (stack.items.len != 1 or !allSpace(rest)) return error.Invalid;
            const node = stack.pop().?;
            const arguments = try json.newObject(a);
            for (node.kids.items) |kid| {
                const schema = try tool_params.schemaOf(schemas, node.name, kid.name, a);
                try arguments.put(a, kid.name, try value(a, kid, schema));
            }
            try out.append(a, .{ .name = reply_text.pyStrip(node.name), .arguments = .{ .object = arguments } });
        } else if (tag.len > 1 and tag[0] == '/') {
            if (stack.items.len < 2 or !std.mem.eql(u8, stack.items[stack.items.len - 1].name, tag[1..]) or !allSpace(rest)) return error.Invalid;
            _ = stack.pop();
        } else {
            if (stack.items.len == 0 or tag.len == 0 or std.mem.indexOfAny(u8, tag, " \t\r\n\"<=/") != null) return error.Invalid;
            const node = try a.create(Node);
            node.* = .{ .name = tag, .text = rest };
            try stack.items[stack.items.len - 1].kids.append(a, node);
            try stack.append(a, node);
        }
    }
    if (stack.items.len != 0 or out.items.len == 0) return error.Invalid;
    return out.items;
}

/// `calls`, or null where it is Invalid: the block stays text.
pub fn tryCalls(a: Allocator, block: []const u8, schemas: *const tool_params.Schemas) Allocator.Error!?[]Invoke {
    return calls(a, block, schemas) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Invalid => null,
    };
}

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

fn value(a: Allocator, node: *const Node, schema: tool_params.Schema) Error!Value {
    const here: tool_params.Schema = .{ .value = deref(schema), .root = schema.root };
    const kinds = try tool_kinds.schemaKinds(a, here.value, here.root);
    if (node.kids.items.len == 0) {
        if (node.text.len == 0) if (kinds) |k| {
            if (!k.contains(.string)) {
                if (k.contains(.array) and !k.contains(.object)) return .{ .array = &.{} };
                if (k.contains(.object) and !k.contains(.array)) return .{ .object = try json.newObject(a) };
            }
        };
        return tool_params.decode(a, node.text, here, false);
    }
    if (!allSpace(node.text)) return error.Invalid;
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

/// What closes the block the reply ended inside (`text` from its opener): the shortest closing that makes one whole
/// call to an offered tool, else nothing.
pub fn closing(a: Allocator, text: []const u8, tools: []const Value) Allocator.Error![]const u8 {
    for ([_][]const u8{ close, ns ++ "</invoke>" ++ close }) |tail| {
        const closed = try std.mem.concat(a, u8, &.{ text, tail });
        if ((try tool_parse.parse(a, closed, tools, 1)).calls != null) return tail;
    }
    return "";
}
