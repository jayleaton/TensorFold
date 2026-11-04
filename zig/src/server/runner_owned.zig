//! A reply segment's engine-owned content bytes: where its runner cells landed, so no strip cuts into them.
const std = @import("std");
const json = @import("json");
const tool_parse = @import("tool_parse.zig");
const Owned = @import("reply_text.zig").Owned;
const Allocator = std.mem.Allocator;

/// The owned span of `content` (a slice of `raw`, `pieces` its tokens' bytes) before its first tool envelope, or null.
pub fn of(a: Allocator, records: []const []const u8, base: usize, pieces: []const []const u8, raw: []const u8, content: []const u8) Allocator.Error!?Owned {
    if (records.len == 0) return null;
    const from = @intFromPtr(content.ptr) -% @intFromPtr(raw.ptr); // past raw.len: content is not raw's bytes
    // each token's first byte in `raw`, which the token bytes must spell exactly (a lossy decode is no map)
    const at = try a.alloc(usize, pieces.len + 1);
    at[0] = 0;
    for (pieces, 0..) |p, i| {
        at[i + 1] = at[i] + p.len;
        if (at[i + 1] > raw.len or !std.mem.eql(u8, raw[at[i]..at[i + 1]], p)) return null;
    }
    if (at[pieces.len] != raw.len) return null;
    const envs = try tool_parse.envelopes(a, content);
    const limit = if (envs.len > 0) envs[0].start else content.len;
    var span: ?Owned = null;
    for (records) |text| {
        const r = switch (try json.parse(a, text)) {
            .ok => |v| v,
            .err => continue,
        };
        if (!std.mem.eql(u8, r.strField("field") orelse "", "content")) continue;
        const token = std.math.cast(usize, (r.get("token") orelse continue).int64() orelse continue) orelse continue;
        const landed = std.math.cast(usize, (r.get("landed") orelse continue).int64() orelse continue) orelse continue;
        if (token < base or token - base >= pieces.len) continue; // another segment's write
        const lo = at[token - base];
        const hi = at[@min(token - base + landed, pieces.len)];
        if (lo < from or hi - from > limit or hi == lo) continue; // outside the content, in or past a call, or no bytes
        span = Owned.join(span, .{ .lo = lo - from, .hi = hi - from });
    }
    return span;
}

fn record(a: Allocator, field: []const u8, token: usize, landed: usize) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"cell\":0,\"field\":\"{s}\",\"token\":{d},\"landed\":{d}}}", .{ field, token, landed });
}

test "runner projection: owned span is the landed tokens' bytes in the content, by the segment's base" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pieces = [_][]const u8{ "plan", "</think>", "[2^66 = ", "73786976294838206464]", "\n", "<tool_call>{\"name\":\"f\"}</tool_call>" };
    const raw = try std.mem.concat(a, u8, &pieces);
    const content = raw["plan</think>".len..];
    const trace = try record(a, "content", 7, 3);
    try std.testing.expectEqual(Owned{ .lo = 0, .hi = 30 }, (try of(a, &.{trace}, 5, &pieces, raw, content)).?);
    // the same write, read as a later segment's (base past it), or as reasoning: none
    try std.testing.expectEqual(null, try of(a, &.{trace}, 8, &pieces, raw, content));
    try std.testing.expectEqual(null, try of(a, &.{try record(a, "reasoning", 7, 3)}, 5, &pieces, raw, content));
    try std.testing.expectEqual(null, try of(a, &.{try record(a, "content", 7, 0)}, 5, &pieces, raw, content));
    try std.testing.expectEqual(null, try of(a, &.{try record(a, "content", 12, 1)}, 5, &pieces, raw, content));
    // a write that lands in the call block owns nothing the prose keeps
    try std.testing.expectEqual(Owned{ .lo = 0, .hi = 30 }, (try of(a, &.{ trace, try record(a, "content", 10, 1) }, 5, &pieces, raw, content)).?);
    try std.testing.expectEqual(null, try of(a, &.{try record(a, "content", 10, 1)}, 5, &pieces, raw, content));
    // token bytes that do not spell the decoded text, or content that is not its slice: no proof, so none
    const lossy = [_][]const u8{ "plan", "</think>", "[2^66 = ", "73786976294838206464]", "\r", pieces[5] };
    try std.testing.expectEqual(null, try of(a, &.{trace}, 5, &lossy, raw, content));
    try std.testing.expectEqual(null, try of(a, &.{trace}, 5, &pieces, raw, try a.dupe(u8, content)));
}

test "runner projection: owned bytes stay through the finish strip, the cut strip and tool parsing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply_text = @import("reply_text.zig");
    const trace = "[exact, computed by the engine: 2^66 = 73786976294838206464]\n";
    const call = "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>";
    const text = trace ++ "\n" ++ call;
    const own: Owned = .{ .lo = 0, .hi = trace.len };
    try std.testing.expectEqualStrings(trace, try expectValid(try reply_text.finishedProse(a, text, own)));
    try std.testing.expectEqualStrings(trace[0 .. trace.len - 1], try expectValid(try reply_text.finishedProse(a, text, null)));
    try std.testing.expectEqualStrings(trace[0 .. trace.len - 1], try expectValid(try reply_text.finishedProse(a, text, .{ .lo = 0, .hi = text.len })));
    try std.testing.expectEqualStrings(trace, try expectValid(try reply_text.cutProse(a, trace ++ "\n<tool_call>{\"name\":\"f\",\"argu", own)));
    const tools = [_]json.Value{(try json.parse(a, "{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}")).ok};
    const parsed = try tool_parse.parseOwned(a, " \n" ++ text, &tools, null, .{ .lo = 2, .hi = 2 + trace.len });
    try std.testing.expectEqualStrings(trace, try expectValid(parsed.content));
    try std.testing.expectEqual(Owned{ .lo = 0, .hi = trace.len }, parsed.owned.?);
    // an owned lead blank stays too, and a span past the text is no offsets of it
    try std.testing.expectEqualStrings(" 7\n", try expectValid(reply_text.stripOwned("\n 7\n\n", .{ .lo = 1, .hi = 4 })));
    try std.testing.expectEqualStrings("7", try expectValid(reply_text.stripOwned("\n 7\n\n", .{ .lo = 1, .hi = 9 })));
}

/// One byte a token, as T5's counterexample tokenizes it.
fn bytePieces(a: Allocator, raw: []const u8) ![]const []const u8 {
    const pieces = try a.alloc([]const u8, raw.len);
    for (pieces, 0..) |*p, i| p.* = raw[i..][0..1];
    return pieces;
}

fn expectValid(s: []const u8) ![]const u8 {
    try std.testing.expect(std.unicode.utf8ValidateSlice(s));
    return s;
}

test "runner projection: owned trim limits widen to whole code points, so no trim leaves invalid UTF-8" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply_text = @import("reply_text.zig");
    const call = "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>";
    const tools = [_]json.Value{(try json.parse(a, "{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}")).ok};
    // T5-OWNED-WS-REVIEW: raw 78 0a c2 a0, a record owning only the C2 of U+00A0, then a whole call
    const raw = "x\n\u{a0}" ++ call;
    const own = (try of(a, &.{try record(a, "content", 2, 1)}, 0, try bytePieces(a, raw), raw, raw)).?;
    try std.testing.expectEqual(Owned{ .lo = 2, .hi = 3 }, own);
    try std.testing.expectEqualStrings("x\n\u{a0}", try expectValid((try tool_parse.parseOwned(a, raw, &tools, null, own)).content));
    try std.testing.expectEqualStrings("x\n\u{a0}", try expectValid(try reply_text.finishedProse(a, raw, own)));
    // right and left ends inside 2-, 3- and 4-byte code points (U+00A0, U+3000, U+1F600); none takes four as a blank
    const Case = struct { text: []const u8, own: Owned, want: []const u8 };
    const cases = [_]Case{
        .{ .text = "x\n\u{3000}\n", .own = .{ .lo = 2, .hi = 3 }, .want = "x\n\u{3000}" },
        .{ .text = "x\n\u{3000}\n", .own = .{ .lo = 3, .hi = 4 }, .want = "x\n\u{3000}" },
        .{ .text = "x \u{1f600}\u{a0}", .own = .{ .lo = 3, .hi = 4 }, .want = "x \u{1f600}" },
        .{ .text = "\n\u{a0}x", .own = .{ .lo = 2, .hi = 3 }, .want = "\u{a0}x" },
        .{ .text = "\u{3000}x\n", .own = .{ .lo = 1, .hi = 2 }, .want = "\u{3000}x" },
        .{ .text = "\u{3000}x\n", .own = .{ .lo = 2, .hi = 4 }, .want = "\u{3000}x" },
        .{ .text = "\n\u{1f600}x ", .own = .{ .lo = 2, .hi = 3 }, .want = "\u{1f600}x" },
    };
    for (cases) |c| {
        try std.testing.expectEqualStrings(c.want, try expectValid(reply_text.stripOwned(c.text, c.own)));
        const parsed = try tool_parse.parseOwned(a, try std.mem.concat(a, u8, &.{ c.text, call }), &tools, null, c.own);
        try std.testing.expectEqualStrings(c.want, try expectValid(parsed.content));
    }
}
