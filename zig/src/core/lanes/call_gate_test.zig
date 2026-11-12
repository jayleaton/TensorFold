//! The call gate's steps on a toy lexicon (the first two tests are MiaAI-Lab's, from patch 0007).
const std = @import("std");
const call_gate = @import("call_gate.zig");
const Gate = call_gate.Gate;
const Allocator = std.mem.Allocator;
const t = std.testing;

const Lex = struct {
    fn blank(_: *anyopaque, token: u32) bool {
        return token == 1;
    }
    fn text(_: *anyopaque, _: Allocator, token: u32) anyerror![]const u8 {
        return switch (token) {
            2 => "=",
            3 => "ab",
            4 => "ac>",
            5 => "zz",
            6 => "\x00", // an end token, as the server's lexicon spells it
            else => "?",
        };
    }
    fn encode(_: *anyopaque, a: Allocator, s: []const u8) anyerror![]const u32 {
        if (std.mem.eql(u8, s, "=")) return a.dupe(u32, &.{2});
        if (std.mem.eql(u8, s, "ab>")) return a.dupe(u32, &.{ 3, 4 });
        if (std.mem.eql(u8, s, "c>")) return a.dupe(u32, &.{4});
        return a.dupe(u32, &.{});
    }
    fn lex() call_gate.Lex {
        return .{ .ctx = undefined, .blank = blank, .text = text, .encode = encode };
    }
};

fn spec(think_open: ?u32, think_end: ?u32, lead: []const u8, names: []const []const u8, tail: []const u8) call_gate.Spec {
    return .{ .opener = 9, .think_open = think_open, .think_end = think_end, .lead = lead, .names = names, .tail = tail, .lex = Lex.lex() };
}

test "call gate: a required call replaces a plain answer with the opener and finishes the offered name" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var g = Gate.afterPrompt(spec(null, null, "=", &.{ "ab", "ac" }, ">"), &.{});
    defer g.deinit(t.allocator);
    const hit = (try g.cut(arena.allocator(), &.{ 7, 8 })).?;
    try t.expectEqual(@as(usize, 0), hit.at);
    try t.expectEqualSlices(u32, &.{ 9, 2 }, hit.fix);
    try g.observe(t.allocator, 9);
    try g.observe(t.allocator, 2);
    try t.expect(g.phase == .name);
    try g.observe(t.allocator, 3);
    try t.expect(g.phase == .name);
    try g.observe(t.allocator, 4);
    try t.expect(g.done());
}

test "call gate: whitespace before the call is kept and an open think block waits" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var g = Gate.afterPrompt(spec(8, 4, "", &.{}, ""), &.{ 4, 8 });
    defer g.deinit(t.allocator);
    try t.expect(!g.armed);
    try t.expect((try g.cut(a, &.{7})) == null);
    try g.observe(t.allocator, 4);
    try t.expect(g.armed);
    try t.expect((try g.cut(a, &.{1})) == null);
    // armed, a plain token is the opener (no names: the opener alone)
    const hit = (try g.cut(a, &.{ 1, 7 })).?;
    try t.expectEqual(@as(usize, 1), hit.at);
    try t.expectEqualSlices(u32, &.{9}, hit.fix);
}

test "call gate: a prompt that closed its think block is armed, and a think block the reply opens disarms it" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var closed = Gate.afterPrompt(spec(8, 4, "", &.{}, ""), &.{ 8, 7, 4 });
    defer closed.deinit(t.allocator);
    try t.expect(closed.armed);
    try t.expect((try closed.cut(a, &.{ 8, 7, 7 })) == null); // the reply's own think block: no cut inside it
    try t.expectEqual(@as(usize, 3), (try closed.cut(a, &.{ 8, 7, 4, 7 })).?.at); // and the answer after it is cut
}

test "call gate: a name the model starts is finished, an end token in the name is replaced, the cut is found in a window" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var g = Gate.afterPrompt(spec(null, null, "=", &.{ "ab", "ac" }, ">"), &.{});
    defer g.deinit(t.allocator);
    // a drafted window: the opener and lead hold, then a token no offered name starts
    const hit = (try g.cut(a, &.{ 9, 2, 5 })).?;
    try t.expectEqual(@as(usize, 2), hit.at);
    try t.expectEqualSlices(u32, &.{ 3, 4 }, hit.fix); // the first offered name and its tail
    try t.expect(g.phase == .answer); // cut changes nothing: commit's observe moves the gate
    for ([_]u32{ 9, 2 }) |tok| try g.observe(t.allocator, tok);
    const eos = (try g.cut(a, &.{6})).?;
    try t.expectEqualSlices(u32, &.{ 3, 4 }, eos.fix);
}

test "call gate: a committed token that breaks the call ends the gate, and a done gate cuts nothing" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var g = Gate.afterPrompt(spec(null, null, "=", &.{"ab"}, ">"), &.{});
    defer g.deinit(t.allocator);
    try g.observe(t.allocator, 7); // a plain token committed past the gate (nothing corrected it)
    try t.expect(g.done());
    try t.expect((try g.cut(arena.allocator(), &.{ 7, 7 })) == null);
}
