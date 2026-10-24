//! Tests' check that a reply's streamed MiniMax calls are its parsed calls, at every way of splitting the reply.
const std = @import("std");
const json = @import("json");
const tool_parse = @import("tool_parse.zig");
const tool_stream = @import("tool_stream.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const ns = "]<]minimax[>[";

/// Calls as a client sees them: names and joined arguments, by index.
pub const Calls = struct { names: [][]const u8, args: [][]const u8 };

/// The calls a client joins from streamed deltas; an index out of order or a fragment before its header fails.
pub fn joined(a: Allocator, deltas: []const Value) !Calls {
    var names: std.ArrayList([]const u8) = .empty;
    var args: std.ArrayList(std.ArrayList(u8)) = .empty;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| {
        const index: usize = @intCast(one.get("index").?.int64().?);
        const function = one.get("function").?;
        if (one.get("id") != null) {
            try std.testing.expectEqual(names.items.len, index);
            try names.append(a, function.get("name").?.string);
            try args.append(a, .empty);
        }
        try std.testing.expect(index < args.items.len);
        try args.items[index].appendSlice(a, function.get("arguments").?.string);
    };
    const out = try a.alloc([]const u8, args.items.len);
    for (args.items, out) |list, *slot| slot.* = list.items;
    return .{ .names = names.items, .args = out };
}

/// The calls streamed when the reply grows to each of `ends` in turn (each a fresh copy, as a decode buffer moves).
pub fn streamed(a: Allocator, text: []const u8, tools: []const Value, ends: []const usize) !Calls {
    var s = try tool_stream.Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    for (ends) |e| try s.feed(try a.dupe(u8, text[0..e]), &out);
    try s.feed(text, &out);
    return joined(a, out.items);
}

fn expectSame(want: Calls, got: Calls) !void {
    try std.testing.expectEqual(want.names.len, got.names.len);
    for (want.names, got.names, want.args, got.args) |wn, gn, wa, ga| {
        try std.testing.expectEqualStrings(wn, gn);
        try std.testing.expectEqualStrings(wa, ga);
    }
}

/// The reply's parse, once its calls streamed per byte, in two at each byte and in token-like pieces equal it exactly.
pub fn check(a: Allocator, text: []const u8, tools: []const Value) !tool_parse.Parsed {
    const final = try tool_parse.parse(a, text, tools, null);
    const list = final.calls orelse &.{};
    const want: Calls = .{ .names = try a.alloc([]const u8, list.len), .args = try a.alloc([]const u8, list.len) };
    for (list, want.names, want.args) |c, *n, *g| {
        n.* = c.get("function").?.get("name").?.string;
        g.* = c.get("function").?.get("arguments").?.string;
    }
    const every = try a.alloc(usize, text.len + 1);
    for (every, 0..) |*e, i| e.* = i;
    try expectSame(want, try streamed(a, text, tools, every));
    for (0..text.len + 1) |k| try expectSame(want, try streamed(a, text, tools, &.{k}));
    var tokens: std.ArrayList(usize) = .empty;
    for (0..text.len) |i| {
        const cut = std.mem.startsWith(u8, text[i..], ns) or (i >= ns.len and std.mem.eql(u8, text[i - ns.len .. i], ns)) or
            text[i] == '<' or (i > 0 and text[i - 1] == '>') or i % 5 == 0;
        if (cut) try tokens.append(a, i);
    }
    try expectSame(want, try streamed(a, text, tools, tokens.items));
    return final;
}
