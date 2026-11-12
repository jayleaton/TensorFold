//! The tool_choice call gate (ported from MiaAI-Lab patch 0007): outside a think block the answer opens a call.
const std = @import("std");
const be = @import("backend.zig");
const Stream = @import("stream.zig").Stream;
const Allocator = std.mem.Allocator;

/// Where the answer is: before the call, in the text between the opener and the name, in the name, or past it.
pub const Phase = enum { answer, lead, name, done };

/// The tokenizer the gate asks while a round is decided; `text` and `encode` return slices of the allocator given.
pub const Lex = struct {
    ctx: *anyopaque,
    blank: *const fn (ctx: *anyopaque, token: u32) bool,
    text: *const fn (ctx: *anyopaque, a: Allocator, token: u32) anyerror![]const u8,
    encode: *const fn (ctx: *anyopaque, a: Allocator, text: []const u8) anyerror![]const u32,
};

/// tool_choice "required" or a named function, as the server read the template's call form.
pub const Spec = struct {
    /// The token that opens a call, the template text before the name, and the mark that ends the name.
    opener: u32,
    lead: []const u8 = "",
    tail: []const u8 = "",
    /// Offered names (empty: the opener alone is forced); the gate completes a name its prefix starts.
    names: []const []const u8 = &.{},
    think_open: ?u32 = null,
    think_end: ?u32 = null,
    lex: Lex,
};

fn is(marker: ?u32, token: u32) bool {
    return if (marker) |m| m == token else false;
}

pub const Gate = struct {
    spec: Spec,
    armed: bool = true, // false while a think block is open
    phase: Phase = .answer,
    seen: std.ArrayList(u8) = .empty, // the lead or name text written so far

    /// The gate after `prompt`: it waits while the prompt leaves a think block open.
    pub fn afterPrompt(spec: Spec, prompt: []const u32) Gate {
        var open_at: ?usize = null;
        var end_at: ?usize = null;
        for (prompt, 0..) |t, i| {
            if (is(spec.think_open, t)) open_at = i;
            if (is(spec.think_end, t)) end_at = i;
        }
        const held = spec.think_end != null and open_at != null and (end_at == null or open_at.? > end_at.?);
        return .{ .spec = spec, .armed = !held };
    }

    pub fn deinit(g: *Gate, a: Allocator) void {
        g.seen.deinit(a);
    }

    pub fn done(g: *const Gate) bool {
        return g.phase == .done;
    }

    const Snap = struct { armed: bool, phase: Phase, seen: []const u8 };
    const Step = struct { snap: Snap, fix: ?[]const u32 };

    fn ends(g: *const Gate, char: u8) bool {
        if (std.ascii.isWhitespace(char)) return true;
        if (g.spec.tail.len > 0) return char == g.spec.tail[0];
        return switch (char) {
            '<', '>', '{', '}', '"', '\'', '(', ')', ',' => true,
            else => false,
        };
    }

    fn gaveUp(snap: Snap) Step {
        return .{ .snap = .{ .armed = snap.armed, .phase = .done, .seen = "" }, .fix = null };
    }

    /// The gate after `token`, or the tokens that replace it (the first in its place, the rest forced after it).
    fn step(g: *const Gate, a: Allocator, snap: Snap, token: u32) !Step {
        const sp = &g.spec;
        if (snap.phase == .answer) {
            if (!snap.armed) return .{ .snap = .{ .armed = is(sp.think_end, token), .phase = .answer, .seen = "" }, .fix = null };
            if (is(sp.think_open, token)) return .{ .snap = .{ .armed = false, .phase = .answer, .seen = "" }, .fix = null };
            if (token == sp.opener) return .{ .snap = .{ .armed = true, .phase = .lead, .seen = "" }, .fix = null };
            if (sp.lex.blank(sp.lex.ctx, token)) return .{ .snap = snap, .fix = null };
            var fix: std.ArrayList(u32) = .empty;
            try fix.append(a, sp.opener);
            if (sp.names.len > 0 and sp.lead.len > 0) try fix.appendSlice(a, try sp.lex.encode(sp.lex.ctx, a, sp.lead));
            return .{ .snap = snap, .fix = fix.items };
        }
        if (snap.phase == .done or sp.names.len == 0) return gaveUp(snap);
        const written = try std.mem.concat(a, u8, &.{ snap.seen, try sp.lex.text(sp.lex.ctx, a, token) });
        var owed: []const u8 = "";
        var seen = snap.seen;
        var body = written;
        if (snap.phase == .lead) {
            if (std.mem.startsWith(u8, sp.lead, written)) {
                const full = written.len == sp.lead.len;
                return .{ .snap = .{ .armed = snap.armed, .phase = if (full) .name else .lead, .seen = if (full) "" else written }, .fix = null };
            }
            if (!std.mem.startsWith(u8, written, sp.lead)) {
                const ids = try sp.lex.encode(sp.lex.ctx, a, sp.lead[seen.len..]);
                return if (ids.len == 0) gaveUp(snap) else .{ .snap = snap, .fix = ids };
            }
            owed = sp.lead[seen.len..];
            seen = "";
            body = written[sp.lead.len..];
        }
        var end: usize = body.len;
        for (body, 0..) |c, i| if (g.ends(c)) {
            end = i;
            break;
        };
        if (end == body.len) for (sp.names) |n| if (std.mem.startsWith(u8, n, body)) {
            return .{ .snap = .{ .armed = snap.armed, .phase = .name, .seen = body }, .fix = null };
        };
        if (end < body.len) for (sp.names) |n| if (std.mem.eql(u8, n, body[0..end])) return gaveUp(snap);
        const name = for (sp.names) |n| {
            if (std.mem.startsWith(u8, n, seen)) break n;
        } else return gaveUp(snap);
        const ids = try sp.lex.encode(sp.lex.ctx, a, try std.mem.concat(a, u8, &.{ owed, name[seen.len..], sp.tail }));
        return if (ids.len == 0) gaveUp(snap) else .{ .snap = snap, .fix = ids };
    }

    pub const Hit = struct { at: usize, fix: []const u32 };

    /// The first of `tokens` that breaks the call, with its fix; null when they all keep it (the gate is unchanged).
    pub fn cut(g: *const Gate, a: Allocator, tokens: []const u32) !?Hit {
        var snap: Snap = .{ .armed = g.armed, .phase = g.phase, .seen = g.seen.items };
        for (tokens, 0..) |token, i| {
            if (snap.phase == .done) return null;
            const next = try g.step(a, snap, token);
            if (next.fix) |fix| return .{ .at = i, .fix = fix };
            snap = next.snap;
        }
        return null;
    }

    /// A committed token: the gate moves on (a token that breaks the call ends it, as nothing then corrects it).
    pub fn observe(g: *Gate, a: Allocator, token: u32) !void {
        if (g.done()) return;
        var scratch: std.heap.ArenaAllocator = .init(a);
        defer scratch.deinit();
        const next = try g.step(scratch.allocator(), .{ .armed = g.armed, .phase = g.phase, .seen = g.seen.items }, token);
        g.armed = next.snap.armed;
        g.phase = if (next.fix != null) .done else next.snap.phase;
        const seen = if (next.fix != null) "" else next.snap.seen;
        var kept: std.ArrayList(u8) = .empty;
        try kept.appendSlice(a, seen);
        g.seen.deinit(a);
        g.seen = kept;
    }
};

/// The fix's rest goes out forced after its first token, in place of anything queued.
pub fn queue(gpa: Allocator, s: *Stream, rest: []const u32) !void {
    s.force.clearRetainingCapacity();
    try s.force.appendSlice(gpa, rest);
}

/// A drawn token the gate replaces: the fix's first token (its rest forced), else the token itself.
pub fn correct(e: anytype, s: *Stream, token: u32) !u32 {
    const g = &(s.gate orelse return token);
    const hit = (try g.cut(e.arena.allocator(), &.{token})) orelse return token;
    try queue(e.gpa, s, hit.fix[1..]);
    return hit.fix[0];
}

/// The first token under a gate: the draw is read and corrected before drafts or a queued step use it; the draw read.
pub fn first(e: anytype, s: *Stream, feed: *be.Feed) !?u32 {
    if (s.gate == null or feed.* == .value) return null; // a forced first token (a budget close) goes in as it is
    const drawn = try e.readFeed(feed.*);
    feed.* = .{ .value = try correct(e, s, drawn) };
    return drawn;
}

test {
    _ = @import("call_gate_test.zig");
}
