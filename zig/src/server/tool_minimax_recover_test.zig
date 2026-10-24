//! MiniMax-M3.1 calls the model writes slightly wrong: each invoke read alone, slips recovered, streamed as parsed.
const std = @import("std");
const json = @import("json");
const tool_parse = @import("tool_parse.zig");
const tool_stream = @import("tool_stream.zig");
const check = @import("tool_minimax_check.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const ns = "]<]minimax[>[";
const open = ns ++ "<tool_call>\n";
const close = ns ++ "</tool_call>";
const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":" ++
    "{\"city\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}}}}}," ++
    "{\"type\":\"function\",\"function\":{\"name\":\"Search\",\"parameters\":{\"type\":\"object\",\"properties\":" ++
    "{\"q\":{\"type\":\"string\"},\"filters\":{\"type\":\"object\"},\"ids\":{\"type\":\"array\",\"items\":{\"type\":\"integer\"}}}}}}," ++
    "{\"type\":\"function\",\"function\":{\"name\":\"ping\"}}]";

inline fn invoke(comptime name: []const u8, comptime body: []const u8) []const u8 {
    return ns ++ "<invoke name=\"" ++ name ++ "\">" ++ body ++ ns ++ "</invoke>\n";
}

inline fn el(comptime k: []const u8, comptime v: []const u8) []const u8 {
    return ns ++ "<" ++ k ++ ">" ++ v ++ ns ++ "</" ++ k ++ ">";
}

fn tools(a: Allocator) ![]const Value {
    return (try json.parse(a, tools_json)).ok.array;
}

fn argsOf(call: Value) []const u8 {
    return call.get("function").?.get("arguments").?.string;
}

fn nameOf(call: Value) []const u8 {
    return call.get("function").?.get("name").?.string;
}

/// `text`'s calls (checked streamed at every split) are `want`, name then arguments, in order.
fn expectCalls(a: Allocator, text: []const u8, want: []const [2][]const u8) !tool_parse.Parsed {
    const r = try check.check(a, text, try tools(a));
    const calls = r.calls orelse &.{};
    try std.testing.expectEqual(want.len, calls.len);
    for (want, calls) |w, c| {
        try std.testing.expectEqualStrings(w[0], nameOf(c));
        try std.testing.expectEqualStrings(w[1], argsOf(c));
    }
    return r;
}

test "minimax recovery: a missing </invoke> ends at the next invoke or at the block's close" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const next = open ++ ns ++ "<invoke name=\"get_weather\">" ++ el("city", "Paris") ++ "\n" ++ invoke("search", el("q", "x")) ++ close;
    _ = try expectCalls(a, next, &.{ .{ "get_weather", "{\"city\":\"Paris\"}" }, .{ "Search", "{\"q\":\"x\"}" } });
    const last = open ++ ns ++ "<invoke name=\"get_weather\">" ++ el("city", "Paris") ++ el("days", "3") ++ close;
    _ = try expectCalls(a, last, &.{.{ "get_weather", "{\"city\":\"Paris\",\"days\":3}" }});
}

test "minimax recovery: a parameter left open closes at </invoke>, its parent's close or the next element" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try expectCalls(a, open ++ invoke("get_weather", ns ++ "<city>Paris") ++ close, &.{.{ "get_weather", "{\"city\":\"Paris\"}" }});
    const parent = open ++ invoke("search", ns ++ "<filters>" ++ ns ++ "<lang>en" ++ ns ++ "</filters>" ++ el("q", "x")) ++ close;
    _ = try expectCalls(a, parent, &.{.{ "Search", "{\"filters\":{\"lang\":\"en\"},\"q\":\"x\"}" }});
    const sibling = open ++ invoke("get_weather", ns ++ "<city>Paris" ++ el("unit", "c") ++ ns ++ "<days>2") ++ close;
    _ = try expectCalls(a, sibling, &.{.{ "get_weather", "{\"city\":\"Paris\",\"unit\":\"c\",\"days\":2}" }});
    // a close that matches nothing open is dropped, the text around it kept
    const stray = open ++ invoke("search", ns ++ "<q>x" ++ ns ++ "</z>y" ++ ns ++ "</q>") ++ close;
    _ = try expectCalls(a, stray, &.{.{ "Search", "{\"q\":\"xy\"}" }});
}

test "minimax recovery: whitespace between elements, JSON text for an object or array, repeated keys" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const spaced = ns ++ "<tool_call>\n\n  " ++ ns ++ "<invoke name=\"search\">\n  " ++ ns ++ "<filters>\n    " ++ el("lang", "en") ++ "\n  " ++
        ns ++ "</filters>\n  " ++ ns ++ "<ids>\n    " ++ el("item", "1") ++ "\n    " ++ el("item", " 2\n") ++ "\n  " ++ ns ++ "</ids>\n" ++
        ns ++ "</invoke>\n\n" ++ close;
    _ = try expectCalls(a, spaced, &.{.{ "Search", "{\"filters\":{\"lang\":\"en\"},\"ids\":[1,2]}" }});
    const written = open ++ invoke("search", el("filters", "{\"lang\": \"en\", \"n\": [1, 2]}") ++ el("ids", "\n[3, 4]\n") ++ el("q", "{\"a\": 1}")) ++ close;
    _ = try expectCalls(a, written, &.{.{ "Search", "{\"filters\":{\"lang\":\"en\",\"n\":[1,2]},\"ids\":[3,4],\"q\":\"{\\\"a\\\": 1}\"}" }});
    // the first value of a repeated parameter stands: it may already have streamed
    _ = try expectCalls(a, open ++ invoke("search", el("q", "a") ++ el("q", "b")) ++ close, &.{.{ "Search", "{\"q\":\"a\"}" }});
}

test "minimax salvage: each invoke stands alone; unoffered ones and stray text stay text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bad = ns ++ "<invoke name=\"nope\">" ++ el("x", "1") ++ ns ++ "</invoke>";
    const block = open ++ invoke("get_weather", el("city", "A")) ++ bad ++ "\n" ++ invoke("SEARCH", el("q", "B")) ++ close;
    const r = try expectCalls(a, "Here. " ++ block, &.{ .{ "get_weather", "{\"city\":\"A\"}" }, .{ "Search", "{\"q\":\"B\"}" } });
    try std.testing.expectEqualStrings("Here. " ++ bad, r.content);
    const stray = try expectCalls(a, open ++ "note " ++ invoke("ping", "") ++ close, &.{.{ "ping", "{}" }});
    try std.testing.expectEqualStrings("note", stray.content);
    // no invoke a call: the whole block stays text, as before
    const none = try expectCalls(a, "x " ++ open ++ bad ++ "\n" ++ close, &.{});
    try std.testing.expectEqualStrings("x " ++ open ++ bad ++ "\n" ++ close, none.content);
    // a one-call reply keeps the first call and drops the rest, unoffered ones included
    const one = try tool_parse.parse(a, "Here. " ++ open ++ bad ++ "\n" ++ invoke("search", el("q", "B")) ++ invoke("ping", "") ++ close, try tools(a), 1);
    try std.testing.expectEqual(@as(usize, 1), one.calls.?.len);
    try std.testing.expectEqualStrings("{\"q\":\"B\"}", argsOf(one.calls.?[0]));
    try std.testing.expectEqualStrings("Here.", one.content);
}

test "minimax parallel calls: several invokes in a block and several blocks in a reply" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const first = open ++ invoke("get_weather", el("city", "A")) ++ invoke("get_weather", el("city", "B")) ++ close;
    const second = open ++ invoke("ping", "") ++ close;
    const text = "One. " ++ first ++ "\nTwo. " ++ second ++ " end";
    const r = try expectCalls(a, text, &.{ .{ "get_weather", "{\"city\":\"A\"}" }, .{ "get_weather", "{\"city\":\"B\"}" }, .{ "ping", "{}" } });
    try std.testing.expectEqualStrings("One. \nTwo.  end", r.content);
    const one = try tool_parse.parse(a, text, try tools(a), 1);
    try std.testing.expectEqual(@as(usize, 1), one.calls.?.len);
    try std.testing.expectEqualStrings("{\"city\":\"A\"}", argsOf(one.calls.?[0]));
}

test "minimax streaming: a header once the invoke tag is whole, a string as it comes, partial NS held back" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try tool_stream.Streamer.init(a, try tools(a));
    var out: std.ArrayList(Value) = .empty;
    const head = "ok " ++ open ++ ns ++ "<invoke name=\"get_weather\"";
    try s.feed(head, &out);
    try std.testing.expectEqual(@as(usize, 0), out.items.len);
    try s.feed(head ++ ">" ++ ns ++ "<city>Pa", &out);
    try std.testing.expectEqualStrings("{\"city\":\"Pa", (try check.joined(a, out.items)).args[0]);
    try s.feed(head ++ ">" ++ ns ++ "<city>Par\"is" ++ ns[0..5], &out);
    try std.testing.expectEqualStrings("{\"city\":\"Par\\\"is", (try check.joined(a, out.items)).args[0]);
    // a typed parameter waits for its close
    try s.feed(head ++ ">" ++ ns ++ "<city>Par\"is" ++ ns ++ "</city>" ++ ns ++ "<days>1", &out);
    try std.testing.expectEqualStrings("{\"city\":\"Par\\\"is\"", (try check.joined(a, out.items)).args[0]);
    try s.feed(head ++ ">" ++ ns ++ "<city>Par\"is" ++ ns ++ "</city>" ++ el("days", "12") ++ ns ++ "</invoke>", &out);
    const got = try check.joined(a, out.items);
    try std.testing.expectEqualStrings("get_weather", got.names[0]);
    try std.testing.expectEqualStrings("{\"city\":\"Par\\\"is\",\"days\":12}", got.args[0]);
}

test "minimax streaming: a block the end token left open streams the call closeCall completes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const offered = try tools(a);
    const cut = "ok " ++ open ++ invoke("get_weather", el("city", "A")) ++ ns ++ "<invoke name=\"search\">" ++ ns ++ "<q>caf\xc3\xa9 \xe4\xb8\xad";
    const tail = try tool_parse.closeCall(a, cut, offered);
    try std.testing.expectEqualStrings(ns ++ "</invoke>" ++ close, tail);
    const whole = try std.mem.concat(a, u8, &.{ cut, tail });
    _ = try expectCalls(a, whole, &.{ .{ "get_weather", "{\"city\":\"A\"}" }, .{ "Search", "{\"q\":\"caf\xc3\xa9 \xe4\xb8\xad\"}" } });
}
