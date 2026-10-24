//! MiniMax-M3.1's required calls: one `NS<tool_call>` block of invokes, arguments in xgrammar's `minimax_m3_xml` style.
const std = @import("std");
const json = @import("json");
const tool_grammar = @import("tool_grammar.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const ns = "]<]minimax[>[";
const style = "minimax_m3_xml";

/// The structural tag's JSON text: blanks, the block holding one invoke or (`parallel`) one or more, blanks.
pub fn build(a: Allocator, tools: []const Value, parallel: bool) Allocator.Error![]const u8 {
    const invokes = try a.alloc(Value, tools.len);
    for (tools, invokes) |t, *c| c.* = try invoke(a, try tool_specs.toolName(a, t), try narrow(a, try tool_grammar.parameters(a, t)));
    const one: Value = if (invokes.len == 1) invokes[0] else try tool_grammar.node(a, "or", &.{.{ "elements", .{ .array = invokes } }});
    const ws = try tool_grammar.node(a, "regex", &.{.{ "pattern", .{ .string = tool_grammar.blanks } }});
    var parts: std.ArrayList(Value) = .empty;
    try parts.appendSlice(a, &.{ ws, try tool_grammar.constant(a, ns ++ "<tool_call>\n"), one });
    if (parallel) try parts.append(a, try tool_grammar.node(a, "star", &.{.{ "content", one }}));
    try parts.appendSlice(a, &.{ try tool_grammar.constant(a, ns ++ "</tool_call>"), ws });
    const format = try tool_grammar.node(a, "sequence", &.{.{ "elements", .{ .array = parts.items } }});
    return json.stringify(a, try tool_grammar.node(a, "structural_tag", &.{.{ "format", format }}), .{ .ascii = false, .compact = true });
}

/// `NS<invoke name="NAME">`, the arguments, `NS</invoke>` and the template's newline.
fn invoke(a: Allocator, name: []const u8, schema: Value) Allocator.Error!Value {
    const args = try tool_grammar.node(a, "json_schema", &.{ .{ "json_schema", schema }, .{ "style", .{ .string = style } }, .{ "max_whitespace_cnt", try json.intValue(a, tool_grammar.value_blanks) } });
    const head = try tool_grammar.constant(a, try std.fmt.allocPrint(a, ns ++ "<invoke name=\"{s}\">", .{name}));
    return tool_grammar.node(a, "sequence", &.{.{ "elements", try tool_grammar.list(a, &.{ head, args, try tool_grammar.constant(a, ns ++ "</invoke>\n") }) }});
}

const dropped_everywhere = [_][]const u8{ "patternProperties", "propertyNames", "unevaluatedProperties" };
const dropped_strings = [_][]const u8{ "pattern", "format", "minLength", "maxLength" };
const nested = [_][]const u8{ "anyOf", "oneOf", "allOf", "prefixItems" };

fn text(a: Allocator) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = "string" });
    return .{ .object = o };
}

fn typed(v: Value, name: []const u8) bool {
    const t = v.get("type") orelse return false;
    if (t == .string) return std.mem.eql(u8, t.string, name);
    if (t == .array) for (t.array) |x| if (x == .string and std.mem.eql(u8, x.string, name)) return true;
    return false;
}

/// True when `v` constrains nothing (`{}`, `true`, or only annotations): xgrammar's unconstrained value.
fn unconstrained(v: Value) bool {
    if (v == .bool) return v.bool;
    if (v != .object) return false;
    for (v.object.keys()) |k| {
        for ([_][]const u8{ "description", "title", "default", "examples", "$comment", "deprecated", "readOnly", "writeOnly" }) |note| {
            if (std.mem.eql(u8, k, note)) break;
        } else return false;
    }
    return true;
}

/// `schema` narrowed to fixed shapes: untyped is text, open objects close, string patterns, formats and lengths go.
pub fn narrow(a: Allocator, schema: Value) Allocator.Error!Value {
    if (unconstrained(schema)) return text(a);
    if (schema != .object) return schema;
    const o = try json.newObject(a);
    var it = schema.object.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        const v = e.value_ptr.*;
        if (inList(&dropped_everywhere, k)) continue;
        if (typed(schema, "string") and inList(&dropped_strings, k)) continue;
        if (std.mem.eql(u8, k, "additionalProperties") or std.mem.eql(u8, k, "required")) continue;
        try o.put(a, k, try narrowField(a, k, v));
    }
    const props = schema.get("properties");
    if (props != null or typed(schema, "object")) {
        try o.put(a, "additionalProperties", .{ .bool = false });
        if (schema.get("required")) |req| if (req == .array) {
            var keep: std.ArrayList(Value) = .empty;
            for (req.array) |r| if (r == .string and props != null and props.? == .object and props.?.get(r.string) != null) try keep.append(a, r);
            try o.put(a, "required", .{ .array = keep.items });
        };
    }
    if (typed(schema, "array") and schema.get("items") == null and schema.get("prefixItems") == null) try o.put(a, "items", try text(a));
    return .{ .object = o };
}

fn narrowField(a: Allocator, k: []const u8, v: Value) Allocator.Error!Value {
    if (std.mem.eql(u8, k, "items") or std.mem.eql(u8, k, "not")) return narrow(a, v);
    if (inList(&nested, k) and v == .array) {
        const out = try a.alloc(Value, v.array.len);
        for (v.array, out) |x, *slot| slot.* = try narrow(a, x);
        return .{ .array = out };
    }
    if ((std.mem.eql(u8, k, "properties") or std.mem.eql(u8, k, "$defs") or std.mem.eql(u8, k, "definitions")) and v == .object) {
        const out = try json.newObject(a);
        var it = v.object.iterator();
        while (it.next()) |e| try out.put(a, e.key_ptr.*, try narrow(a, e.value_ptr.*));
        return .{ .object = out };
    }
    return v;
}

fn inList(names: []const []const u8, k: []const u8) bool {
    for (names) |n| if (std.mem.eql(u8, n, k)) return true;
    return false;
}

test "MiniMax's required grammar is one block of invokes in its namespace markup, in the minimax_m3_xml style" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}},{\"type\":\"function\",\"function\":{\"name\":\"now\"}}]")).ok;
    const many = try build(a, tools.array, true);
    for ([_][]const u8{ "\"value\":\"]<]minimax[>[<tool_call>\\n\"", "\"value\":\"]<]minimax[>[<invoke name=\\\"get_weather\\\">\"", "\"value\":\"]<]minimax[>[<invoke name=\\\"now\\\">\"", "\"value\":\"]<]minimax[>[</invoke>\\n\"", "\"value\":\"]<]minimax[>[</tool_call>\"", "\"style\":\"minimax_m3_xml\"", "\"required\":[\"city\"]", "\"type\":\"star\"", "\"type\":\"or\"" }) |want|
        std.testing.expect(std.mem.indexOf(u8, many, want) != null) catch |e| {
            std.debug.print("missing {s} in {s}\n", .{ want, many });
            return e;
        };
    const one = try build(a, tools.array[0..1], false);
    try std.testing.expect(std.mem.indexOf(u8, one, "\"type\":\"star\"") == null and std.mem.indexOf(u8, one, "\"type\":\"or\"") == null);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, one, "<tool_call>"));
}

test "a schema is narrowed to the fixed shapes minimax_m3_xml compiles, at every depth" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const schema = (try json.parse(a,
        \\{"type":"object","properties":{"id":{"type":"string","pattern":"^a","minLength":1,"format":"uuid","description":"d"},
        \\"any":{},"tags":{"type":"array"},"meta":{"type":"object","additionalProperties":true,"patternProperties":{"x":{}}},
        \\"pick":{"anyOf":[{"type":"string","maxLength":3},{"type":"integer","minimum":0}]},"ref":{"$ref":"#/$defs/r"}},
        \\"required":["id","ghost"],"additionalProperties":{"type":"string"},"$defs":{"r":{"type":"array","items":{}}}}
    )).ok;
    const got = try json.stringify(a, try narrow(a, schema), .{ .compact = true });
    const want =
        \\{"type":"object","properties":{"id":{"type":"string","description":"d"},"any":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}},"meta":{"type":"object","additionalProperties":false},"pick":{"anyOf":[{"type":"string"},{"type":"integer","minimum":0}]},"ref":{"$ref":"#/$defs/r"}},"$defs":{"r":{"type":"array","items":{"type":"string"}}},"additionalProperties":false,"required":["id"]}
    ;
    try std.testing.expectEqualStrings(want, got);
    try std.testing.expectEqualStrings("{\"type\":\"string\"}", try json.stringify(a, try narrow(a, .{ .bool = true }), .{ .compact = true }));
}

test "the required grammar for each fixture tool set is the tag tools/zig/minimax_grammar_check.py compiles with xgrammar" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = (try json.parse(a, @embedFile("fixtures/minimax_required_tags.json"))).ok;
    var differ: usize = 0;
    for (cases.array) |c| {
        const got = try build(a, c.get("tools").?.array, c.get("parallel").?.bool);
        if (std.mem.eql(u8, got, c.strField("tag").?)) continue;
        std.debug.print("MM3TAG {s}\t{s}\n", .{ c.strField("name").?, got });
        differ += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), differ);
    try std.testing.expectEqual(@as(usize, 4), cases.array.len);
}
