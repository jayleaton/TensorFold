//! MiniMax-M3.1 tool calls: the chat template's own renderings (fixtures/template_goldens.json, made by Python Jinja2)
//! parsed back, scalar typing against the tool schema, and streaming split at every byte boundary.
const std = @import("std");
const json = @import("json");
const tool_parse = @import("tool_parse.zig");
const tool_stream = @import("tool_stream.zig");
const reply_text = @import("reply_text.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const ns = "]<]minimax[>[";
const nested_call = "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"search\">]<]minimax[>[<q>a<b & \"c\"]<]minimax[>[</q>]<]minimax[>[<filters>]<]minimax[>[<lang>en]<]minimax[>[</lang>]<]minimax[>[<deep>]<]minimax[>[<x>1]<]minimax[>[</x>]<]minimax[>[<y>]<]minimax[>[<item>1]<]minimax[>[</item>]<]minimax[>[<item>2.5]<]minimax[>[</item>]<]minimax[>[<item>z]<]minimax[>[</item>]<]minimax[>[</y>]<]minimax[>[</deep>]<]minimax[>[<flag>true]<]minimax[>[</flag>]<]minimax[>[</filters>]<]minimax[>[<ids>]<]minimax[>[<item>3]<]minimax[>[</item>]<]minimax[>[<item>4]<]minimax[>[</item>]<]minimax[>[<item>5]<]minimax[>[</item>]<]minimax[>[</ids>]<]minimax[>[<off>false]<]minimax[>[</off>]<]minimax[>[<pi>3.14159]<]minimax[>[</pi>]<]minimax[>[<big>12345678901234567890]<]minimax[>[</big>]<]minimax[>[<u>café 中文]<]minimax[>[</u>]<]minimax[>[<empty_list>]<]minimax[>[</empty_list>]<]minimax[>[<empty_map>]<]minimax[>[</empty_map>]<]minimax[>[<list_of_maps>]<]minimax[>[<item>]<]minimax[>[<a>1]<]minimax[>[</a>]<]minimax[>[</item>]<]minimax[>[<item>]<]minimax[>[<b>]<]minimax[>[<item>true]<]minimax[>[</item>]<]minimax[>[<item>false]<]minimax[>[</item>]<]minimax[>[</b>]<]minimax[>[</item>]<]minimax[>[</list_of_maps>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>";
const multi_call = "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">]<]minimax[>[<city>A]<]minimax[>[</city>]<]minimax[>[</invoke>\n]<]minimax[>[<invoke name=\"search\">]<]minimax[>[<q>B]<]minimax[>[</q>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>";
const single_call = "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">]<]minimax[>[<city>Paris]<]minimax[>[</city>]<]minimax[>[<unit>c]<]minimax[>[</unit>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>";
const no_args_call = "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"ping\">]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>";
/// The arguments the nested call was rendered from, Nones dropped as the template drops them.
const nested_args = "{\"q\":\"a<b & \\\"c\\\"\",\"filters\":{\"lang\":\"en\",\"deep\":{\"x\":1,\"y\":[1,2.5,\"z\"]},\"flag\":true},\"ids\":[3,4,5],\"off\":false,\"pi\":3.14159,\"big\":12345678901234567890,\"u\":\"café 中文\",\"empty_list\":[],\"empty_map\":{},\"list_of_maps\":[{\"a\":1},{\"b\":[true,false]}]}";
const nested_tool = "{\"type\": \"function\", \"function\": {\"name\": \"search\", \"parameters\": {\"type\": \"object\", \"properties\": {\"q\": {\"type\": \"string\"}, \"filters\": {\"type\": \"object\", \"properties\": {\"lang\": {\"type\": \"string\"}, \"deep\": {\"type\": \"object\", \"properties\": {\"x\": {\"type\": \"integer\"}, \"y\": {\"type\": \"array\", \"items\": {\"type\": [\"number\", \"string\"]}}}}, \"flag\": {\"type\": \"boolean\"}}}, \"ids\": {\"type\": \"array\", \"items\": {\"type\": \"integer\"}}, \"off\": {\"type\": \"boolean\"}, \"pi\": {\"type\": \"number\"}, \"big\": {\"type\": \"integer\"}, \"u\": {\"type\": \"string\"}, \"empty_list\": {\"type\": \"array\"}, \"empty_map\": {\"type\": \"object\"}, \"list_of_maps\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/$defs/entry\"}}}, \"$defs\": {\"entry\": {\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"integer\"}, \"b\": {\"type\": \"array\", \"items\": {\"type\": \"boolean\"}}}}}}}}";
const plain_tools = "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"}},{\"type\":\"function\",\"function\":{\"name\":\"search\"}},{\"type\":\"function\",\"function\":{\"name\":\"ping\"}}]";
const typed_tools = "[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \"unit\": {\"type\": \"string\"}}}}}, {\"type\": \"function\", \"function\": {\"name\": \"search\", \"parameters\": {\"type\": \"object\", \"properties\": {\"q\": {\"type\": \"string\"}}}}}, {\"type\": \"function\", \"function\": {\"name\": \"ping\"}}]";

fn tools(a: Allocator, text: []const u8) ![]const Value {
    return (try json.parse(a, text)).ok.array;
}

fn canonical(a: Allocator, text: []const u8) ![]const u8 {
    return json.stringify(a, (try json.parse(a, text)).ok, .{ .compact = true });
}

fn argsOf(call: Value) []const u8 {
    return call.get("function").?.get("arguments").?.string;
}

fn nameOf(call: Value) []const u8 {
    return call.get("function").?.get("name").?.string;
}

test "minimax: the nested call the template rendered parses back to the arguments it was rendered from" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const offered = try tools(a, "[" ++ nested_tool ++ "]");
    const r = try tool_parse.parse(a, "Let me search. " ++ nested_call, offered, null);
    try std.testing.expectEqualStrings("Let me search.", r.content);
    try std.testing.expectEqual(@as(usize, 1), r.calls.?.len);
    try std.testing.expectEqualStrings("search", nameOf(r.calls.?[0]));
    try std.testing.expectEqualStrings(try canonical(a, nested_args), try canonical(a, argsOf(r.calls.?[0])));
}

test "minimax: single, multi and argument-free calls" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const offered = try tools(a, typed_tools);
    const one = try tool_parse.parse(a, single_call, offered, null);
    try std.testing.expectEqualStrings("get_weather", nameOf(one.calls.?[0]));
    try std.testing.expectEqualStrings("{\"city\":\"Paris\",\"unit\":\"c\"}", argsOf(one.calls.?[0]));
    const two = try tool_parse.parse(a, multi_call, offered, null);
    try std.testing.expectEqual(@as(usize, 2), two.calls.?.len);
    try std.testing.expectEqualStrings("{\"city\":\"A\"}", argsOf(two.calls.?[0]));
    try std.testing.expectEqualStrings("search", nameOf(two.calls.?[1]));
    try std.testing.expectEqualStrings("{\"q\":\"B\"}", argsOf(two.calls.?[1]));
    const capped = try tool_parse.parse(a, multi_call, offered, 1);
    try std.testing.expectEqual(@as(usize, 1), capped.calls.?.len);
    const none = try tool_parse.parse(a, no_args_call, offered, null);
    try std.testing.expectEqualStrings("{}", argsOf(none.calls.?[0]));
}

test "minimax: scalars are the text unless the schema names another type (UNVERIFIED decision)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const call = ns ++ "<tool_call>\n" ++ ns ++ "<invoke name=\"ping\">" ++ ns ++ "<n>5" ++ ns ++ "</n>" ++ ns ++ "<ok>true" ++ ns ++ "</ok>" ++
        ns ++ "<l>" ++ ns ++ "<item>1" ++ ns ++ "</item>" ++ ns ++ "</l>" ++ ns ++ "<m>" ++ ns ++ "<item>x" ++ ns ++ "</item>" ++ ns ++ "<k>y" ++ ns ++ "</k>" ++ ns ++ "</m>" ++
        ns ++ "<e>" ++ ns ++ "</e>" ++ ns ++ "<s>  pad\n" ++ ns ++ "</s>" ++ ns ++ "</invoke>\n" ++ ns ++ "</tool_call>";
    const bare = try tools(a, "[{\"type\":\"function\",\"function\":{\"name\":\"ping\"}}]");
    const r0 = try tool_parse.parse(a, call, bare, null);
    // no schema: every leaf a string, a list when every child is `item`, "" for an empty element
    try std.testing.expectEqualStrings("{\"n\":\"5\",\"ok\":\"true\",\"l\":[\"1\"],\"m\":{\"item\":\"x\",\"k\":\"y\"},\"e\":\"\",\"s\":\"  pad\\n\"}", argsOf(r0.calls.?[0]));
    const typed = try tools(a, "[{\"type\":\"function\",\"function\":{\"name\":\"ping\",\"parameters\":{\"type\":\"object\",\"properties\":" ++
        "{\"n\":{\"type\":\"integer\"},\"ok\":{\"type\":\"boolean\"},\"l\":{\"type\":\"array\",\"items\":{\"type\":\"integer\"}},\"m\":{\"type\":\"object\"},\"e\":{\"type\":\"array\"},\"s\":{\"type\":\"string\"}}}}}]");
    const r1 = try tool_parse.parse(a, call, typed, null);
    try std.testing.expectEqualStrings("{\"n\":5,\"ok\":true,\"l\":[1],\"m\":{\"item\":\"x\",\"k\":\"y\"},\"e\":[],\"s\":\"  pad\\n\"}", argsOf(r1.calls.?[0]));
    // a number-typed parameter whose text is no number stays the text
    const bad = try tool_parse.parse(a, ns ++ "<tool_call>" ++ ns ++ "<invoke name=\"ping\">" ++ ns ++ "<n>five" ++ ns ++ "</n>" ++ ns ++ "</invoke>" ++ ns ++ "</tool_call>", typed, null);
    try std.testing.expectEqualStrings("{\"n\":\"five\"}", argsOf(bad.calls.?[0]));
}

test "minimax: unoffered, malformed and unclosed calls stay text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const offered = try tools(a, typed_tools);
    const example = ns ++ "<tool_call>\n" ++ ns ++ "<invoke name=\"tool-name-1\">" ++ ns ++ "<param-1>value-1" ++ ns ++ "</param-1>" ++ ns ++ "</invoke>\n" ++ ns ++ "</tool_call>";
    const cases = [_][]const u8{
        example,
        ns ++ "<tool_call>\n" ++ ns ++ "<invoke name=\"search\">" ++ ns ++ "<q>x" ++ ns ++ "</z>" ++ ns ++ "</invoke>\n" ++ ns ++ "</tool_call>",
        ns ++ "<tool_call>\n" ++ ns ++ "<invoke name=\"search\">" ++ ns ++ "<q>x" ++ ns ++ "</q>" ++ ns ++ "</tool_call>",
        ns ++ "<tool_call>\nstray text " ++ ns ++ "<invoke name=\"ping\">" ++ ns ++ "</invoke>\n" ++ ns ++ "</tool_call>",
        ns ++ "<tool_call>\n" ++ ns ++ "<invoke name=\"search\">" ++ ns ++ "<q>x" ++ ns ++ "</q>" ++ ns ++ "</invoke>\n",
    };
    for (cases) |text| {
        const reply = try std.mem.concat(a, u8, &.{ "before ", text });
        const r = try tool_parse.parse(a, reply, offered, null);
        try std.testing.expect(r.calls == null);
        try std.testing.expectEqualStrings(std.mem.trimEnd(u8, reply, "\n"), r.content); // the content is stripped
    }
    // a good call after a bad one is still found, and the bad one stays in the content
    const mixed = try tool_parse.parse(a, example ++ "\n" ++ single_call, offered, null);
    try std.testing.expectEqual(@as(usize, 1), mixed.calls.?.len);
    try std.testing.expectEqualStrings(example, mixed.content);
}

test "minimax: closeCall completes a call the end token left open, and only then" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const offered = try tools(a, typed_tools);
    const head = ns ++ "<tool_call>\n" ++ ns ++ "<invoke name=\"search\">" ++ ns ++ "<q>x" ++ ns ++ "</q>";
    try std.testing.expectEqualStrings(ns ++ "</invoke>" ++ ns ++ "</tool_call>", try tool_parse.closeCall(a, head, offered));
    try std.testing.expectEqualStrings(ns ++ "</tool_call>", try tool_parse.closeCall(a, head ++ ns ++ "</invoke>\n", offered));
    try std.testing.expectEqualStrings("", try tool_parse.closeCall(a, head ++ ns ++ "<r>cut", offered));
    try std.testing.expectEqualStrings("", try tool_parse.closeCall(a, single_call, offered));
    try std.testing.expectEqualStrings("", try tool_parse.closeCall(a, ns ++ "<tool_call>\n" ++ ns ++ "<invoke name=\"nope\">", offered));
}

/// The arguments a client joins from streamed deltas.
fn sent(a: Allocator, deltas: []const Value) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| try out.appendSlice(a, one.get("function").?.get("arguments").?.string);
    return out.items;
}

fn heads(deltas: []const Value) usize {
    var n: usize = 0;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| {
        if (one.get("id") != null) n += 1;
    };
    return n;
}

test "minimax: a streamed call splits at every byte boundary into the same calls" {
    const text = "Sure. " ++ multi_call ++ " done";
    const call_end = "Sure. ".len + multi_call.len;
    for (0..text.len + 1) |boundary| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const offered = try tools(a, typed_tools);
        // one feed of a growing prefix, then the whole text
        var grown = try tool_stream.Streamer.init(a, offered);
        var out: std.ArrayList(Value) = .empty;
        try grown.feed(text[0..boundary], &out);
        if (boundary < call_end) try std.testing.expectEqual(@as(usize, 0), out.items.len);
        try grown.feed(text, &out);
        try grown.feed(text, &out);
        try std.testing.expectEqual(@as(usize, 2), heads(out.items));
        try std.testing.expectEqualStrings("{\"city\":\"A\"}{\"q\":\"B\"}", try sent(a, out.items));
        // a prefix per byte up to the boundary
        var bytes = try tool_stream.Streamer.init(a, offered);
        var out2: std.ArrayList(Value) = .empty;
        for (0..boundary + 1) |n| try bytes.feed(text[0..n], &out2);
        try bytes.feed(text, &out2);
        try std.testing.expectEqual(@as(usize, 2), heads(out2.items));
        try std.testing.expectEqualStrings("{\"city\":\"A\"}{\"q\":\"B\"}", try sent(a, out2.items));
    }
}

test "minimax: a streamed nested call byte by byte yields the nested arguments once" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const offered = try tools(a, "[" ++ nested_tool ++ "]");
    var s = try tool_stream.Streamer.init(a, offered);
    var out: std.ArrayList(Value) = .empty;
    const text = nested_call ++ "\n";
    for (0..text.len + 1) |n| try s.feed(text[0..n], &out);
    try std.testing.expectEqual(@as(usize, 1), heads(out.items));
    try std.testing.expectEqualStrings(try canonical(a, nested_args), try canonical(a, try sent(a, out.items)));
}

test "minimax: a call to an unoffered tool streams nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const offered = try tools(a, "[{\"type\":\"function\",\"function\":{\"name\":\"other\"}}]");
    var s = try tool_stream.Streamer.init(a, offered);
    var out: std.ArrayList(Value) = .empty;
    for (0..multi_call.len + 1) |n| try s.feed(multi_call[0..n], &out);
    try std.testing.expectEqual(@as(usize, 0), out.items.len);
}

test "minimax: the one-call text filter hides the block at every byte boundary and keeps the prose" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = "Sure ]<] ] <b> " ++ multi_call ++ " done" ++ single_call;
    for (0..text.len + 1) |boundary| {
        var policy: tool_stream.Policy = .{ .single = true };
        var shown: std.ArrayList(u8) = .empty;
        try shown.appendSlice(a, try policy.text(a, text[0..boundary]));
        try shown.appendSlice(a, try policy.text(a, text[boundary..]));
        try shown.appendSlice(a, policy.flush());
        try std.testing.expectEqualStrings("Sure ]<] ] <b>  done", shown.items);
    }
    var policy: tool_stream.Policy = .{ .single = true };
    var shown: std.ArrayList(u8) = .empty;
    for (text) |ch| try shown.appendSlice(a, try policy.text(a, &.{ch}));
    try shown.appendSlice(a, policy.flush());
    try std.testing.expectEqualStrings("Sure ]<] ] <b>  done", shown.items);
    // whole envelopes the parser left as text are kept by the policy's parsed content
    const policy2: tool_stream.Policy = .{ .single = true };
    try std.testing.expectEqualStrings("a " ++ multi_call ++ " b", try policy2.parsedContent(a, "a " ++ multi_call ++ " b"));
}

test "minimax: reply_text hides the block and holds a partial opener back" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = "Sure. " ++ multi_call ++ " tail";
    try std.testing.expectEqualStrings("Sure.  tail", try reply_text.hideToolCalls(a, text, true));
    for (0..text.len + 1) |n| {
        const shown = try reply_text.hideToolCalls(a, text[0..n], false);
        try std.testing.expect(shown.len <= "Sure. ".len or std.mem.eql(u8, shown, "Sure.  tail"[0..shown.len]));
        try std.testing.expect(std.mem.indexOf(u8, shown, "]<") == null);
    }
}

test "minimax: <mm:think> splits the reasoning from the answer and from a call, at every byte boundary" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const offered = try tools(a, typed_tools);
    // adaptive or disabled: the reply opens its own block
    const full = "<mm:think>plan A\nplan B</mm:think>\n\n" ++ single_call;
    const done = try reply_text.splitThinking(a, full, true, reply_text.mm_think_markers);
    try std.testing.expectEqualStrings("plan A\nplan B", done.reasoning);
    try std.testing.expectEqualStrings(single_call, done.answer);
    try std.testing.expectEqual(@as(usize, 1), (try tool_parse.parse(a, done.answer, offered, null)).calls.?.len);
    for (0..full.len + 1) |n| {
        const part = try reply_text.splitThinking(a, full[0..n], false, reply_text.mm_think_markers);
        try std.testing.expect(std.mem.startsWith(u8, done.reasoning, part.reasoning));
        try std.testing.expect(std.mem.startsWith(u8, done.answer, part.answer));
    }
    // no think block at all: the whole reply is the answer
    const plain = try reply_text.splitThinking(a, "just an answer", true, reply_text.mm_think_markers);
    try std.testing.expectEqualStrings("just an answer", plain.answer);
    try std.testing.expectEqualStrings("", plain.reasoning);
    // enabled: the prompt ended in <mm:think>, so the reply holds only the close
    const prompted = "plan</mm:think>\nanswer";
    const split = try reply_text.splitThinking(a, prompted, true, reply_text.mm_think_prompt_markers);
    try std.testing.expectEqualStrings("plan", split.reasoning);
    try std.testing.expectEqualStrings("answer", split.answer);
    for (0..prompted.len + 1) |n| {
        const part = try reply_text.splitThinking(a, prompted[0..n], false, reply_text.mm_think_prompt_markers);
        try std.testing.expect(std.mem.startsWith(u8, "plan", part.reasoning));
        try std.testing.expect(std.mem.startsWith(u8, "answer", part.answer));
    }
    // a call written before the block closes is held as reasoning while streaming, the answer once the reply ends
    const early = "plan " ++ single_call;
    const live = try reply_text.splitThinking(a, early, false, reply_text.mm_think_prompt_markers);
    try std.testing.expectEqualStrings("", live.answer);
    const end = try reply_text.splitThinking(a, early, true, reply_text.mm_think_prompt_markers);
    try std.testing.expectEqualStrings(single_call, end.answer);
}
