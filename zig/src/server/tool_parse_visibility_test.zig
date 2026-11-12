//! Tool visibility: a call to an undeclared function name goes out; a rejected one stays text, outside the call limit.
const std = @import("std");
const json = @import("json");
const tool_parse = @import("tool_parse.zig");
const parse = tool_parse.parse;
const closeCall = tool_parse.closeCall;

const dsml = [_][2][]const u8{
    .{ "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\"><｜DSML｜parameter name=\"q\" string=\"true\">", "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>" },
    .{ "<｜DSML｜ calls><｜DSML｜ invoke name=\"lookup\"><｜DSML｜ parameter name=\"q\" string=\"true\">", "</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>" },
};

test "tool visibility: a first unoffered call remains content in single-call mode" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"execute\"}}]")).ok.array;
    const forms = [_][]const u8{
        "<tool_call><function=tools.session_move><parameter=folder>project</parameter></function></tool_call>",
        "<tool_call>{\"name\":\"tools.session_move\",\"arguments\":{}}</tool_call>",
        "<tool_call>not a tool payload</tool_call>",
    };
    for (forms) |text| for ([_]?usize{ null, 1 }) |limit| {
        const result = try parse(a, text, tools, limit);
        try std.testing.expect(result.calls == null);
        try std.testing.expectEqual(@as(usize, 1), result.rejected_envelopes);
        try std.testing.expectEqualStrings(text, result.content);
    };
}

test "tool visibility: rejected text does not consume a valid call and the second accepted call stays limited" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"execute\"}}]")).ok.array;
    const bad = "<tool_call>{\"name\":\"tools.unknown\",\"arguments\":{}}</tool_call>";
    const good = "<tool_call>{\"name\":\"execute\",\"arguments\":{}}</tool_call>";
    const result = try parse(a, bad ++ good ++ good, tools, 1);
    try std.testing.expectEqualStrings(bad, result.content);
    try std.testing.expectEqual(@as(usize, 1), result.rejected_envelopes);
    try std.testing.expectEqual(@as(usize, 1), result.calls.?.len);
    try std.testing.expectEqualStrings("execute", result.calls.?[0].get("function").?.get("name").?.string);
}

test "a whole call to an undeclared function name goes out under its own name, as vLLM returns it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}},{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]")).ok.array;
    var calls: std.ArrayList([3][]const u8) = .empty; // the reply, the call's name and its arguments
    for (dsml) |sp| {
        const call = try std.mem.concat(a, u8, &.{ sp[0], "ls", sp[1] });
        try calls.append(a, .{ try std.mem.replaceOwned(u8, a, call, "\"lookup\"", "\"bash\""), "bash", "{\"q\":\"ls\"}" });
        try calls.append(a, .{ try std.mem.replaceOwned(u8, a, call, "\"lookup\"", "\"Shell\""), "shell", "{\"q\":\"ls\"}" });
    }
    try calls.append(a, .{ "<tool_call><function=launch><parameter=when>now</parameter></function></tool_call>", "launch", "{\"when\":\"now\"}" });
    try calls.append(a, .{ "<tool_call>{\"name\": \"read-file_2\", \"arguments\": {\"p\": 1}}</tool_call>", "read-file_2", "{\"p\":1}" });
    try calls.append(a, .{ "<tool_call>launch<arg_key>when</arg_key><arg_value>now</arg_value></tool_call>", "launch", "{\"when\":\"now\"}" });
    for (calls.items) |c| for ([_]?usize{ null, 1 }) |max_calls| {
        const r = try parse(a, c[0], tools, max_calls);
        try std.testing.expectEqual(@as(usize, 1), r.calls.?.len);
        try std.testing.expectEqualStrings(c[1], r.calls.?[0].get("function").?.get("name").?.string);
        try std.testing.expectEqualStrings(c[2], r.calls.?[0].get("function").?.get("arguments").?.string);
        try std.testing.expectEqualStrings("", r.content);
        try std.testing.expectEqual(@as(usize, 0), r.rejected_envelopes);
    };
    // with no tools declared the reply makes no call at all
    try std.testing.expect((try parse(a, calls.items[0][0], &.{}, null)).calls == null);
    // an open call the end token cut closes only for a declared tool
    try std.testing.expectEqualStrings("", try closeCall(a, "<tool_call>{\"name\":\"launch\",\"arguments\":{}", tools));
    try std.testing.expectEqualStrings("}</tool_call>", try closeCall(a, "<tool_call>{\"name\":\"lookup\",\"arguments\":{}", tools));
}

test "a call whose undeclared name is no function name, or whose arguments are not finite, stays text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const long = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const whole = [_][]const u8{
        "<tool_call><function=tools.launch><parameter=when>now</parameter></function></tool_call>",
        "<tool_call>{\"name\": \"" ++ long ++ "\", \"arguments\": {}}</tool_call>",
        "<｜DSML｜ calls><｜DSML｜ invoke name=\"run it\"><｜DSML｜ parameter name=\"q\" string=\"true\">ls</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>",
        "<tool_call>{\"name\": \"launch rocket!\", \"arguments\": {\"when\": \"now\"}}</tool_call>",
        "<tool_call>{\"name\": \"launch\", \"arguments\": {\"when\": NaN}}</tool_call>",
    };
    for (whole) |text| {
        const parallel = try parse(a, text, tools, null);
        try std.testing.expect(parallel.calls == null);
        try std.testing.expectEqualStrings(text, parallel.content);
        const single = try parse(a, text, tools, 1);
        try std.testing.expect(single.calls == null);
        try std.testing.expectEqualStrings(text, single.content);
    }
    // an unclosed envelope is not a call in either mode, so the reply keeps it
    const open = "<tool_call>{\"name\":\"launch\",\"arguments\":{\"when\":\"now\"}";
    for ([_]?usize{ null, 1 }) |max_calls| {
        const k = try parse(a, open, tools, max_calls);
        try std.testing.expect(k.calls == null);
        try std.testing.expectEqualStrings(open, k.content);
    }
    const prose = "Launching. <tool_call><function=launch><parameter=when>now</function></tool_call>";
    try std.testing.expectEqualStrings(prose, (try parse(a, prose, tools, 1)).content);
    // an offered tool keeps its offered spelling; bare JSON naming one not offered stays content
    const offered = try parse(a, "<tool_call>{\"name\":\"LOOKUP\"}</tool_call>", tools, null);
    try std.testing.expectEqualStrings("lookup", offered.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expect((try parse(a, "{\"name\":\"launch\",\"arguments\":{}}", tools, null)).calls == null);
}
