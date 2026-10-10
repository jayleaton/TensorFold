//! Prompts in pieces on the lane core: a stream begun, its prompt rows run in pieces, then its first token.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Config = @import("config.zig").Config;
const depth = @import("depth.zig");
const accept = @import("accept.zig");
const be = @import("backend.zig");
const sm = @import("stream.zig");
const ev = @import("events.zig");
const win = @import("windows.zig");
const shape = @import("shape.zig");
const plan_lanes = @import("plan_lanes.zig");
const fill = @import("fill.zig");
const trail = @import("trail.zig");
const trim = @import("trim.zig");
const Stream = sm.Stream;
const Feed = be.Feed;
const LogRow = @import("logprob.zig").Row;
const Plan = win.Plan;
const f = ev.f;
const str = trail.str;
const int = trail.int;

const Outcome = struct { got: []const u32, path: []const u32, cut: ?usize, follow: []const u32 };
const Result = struct { rows: u32, keep: u32, got: []const u32 };
const Unread = struct { event: usize, handle: u64 };

const Engine = @import("engine.zig").Engine;

/// A stream whose prompt runs in pieces between rounds: `piece`s, then `finishStream`; a failed begin holds nothing.
pub fn beginStream(e: *Engine, s: *Stream) !be.Backend.Begun {
    try trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) });
    const begin = e.backend.vtable.begin orelse return error.NoPieces;
    const b = try begin(e.backend.ptr, s);
    s.cached = @intCast(b.cached);
    return b;
}

/// Prompt rows [start, end) of a begun stream (`save`: the prompt's snapshot right after them).
pub fn piece(e: *Engine, s: *Stream, start: u64, end: u64, save: bool) !void {
    const run = e.backend.vtable.piece orelse return error.NoPieces;
    return run(e.backend.ptr, s, start, end, save);
}

/// A round's pieces of several streams in one backend call (Config.piece_runs, Backend.pieces); else one at a time.
pub fn pieces(e: *Engine, list: []const be.Backend.Piece) !void {
    if (e.cfg.piece_runs) if (e.backend.vtable.pieces) |run| return run(e.backend.ptr, list);
    for (list) |p| try piece(e, p.stream, p.start, p.end, p.save);
}

/// A round's finished prompts before their `finishStream` calls (Config.replay_runs, Backend.finals).
pub fn prepareFinals(e: *Engine, streams: []const *Stream) !void {
    if (!e.cfg.replay_runs or streams.len < 2) return;
    const run = e.backend.vtable.finals orelse return;
    return run(e.backend.ptr, streams);
}

/// After the prompt: the first token, the first drafts; the stream joins the rounds (or ends at once).
pub fn opened(e: *Engine, s: *Stream) !void {
    s.context.shrinkRetainingCapacity(s.prompt_len);
    s.rows.clearRetainingCapacity();
    s.pending = null;
    s.cache_len = s.prompt_len;
    const position: u64 = s.prompt_len;
    const drawn = try e.backend.first(s, position);
    var feed: Feed = .{ .handle = drawn };
    if (try e.forcedNext(s)) |t| feed = .{ .value = t };
    var asked: ?u32 = null;
    if (e.cfg.family_mtp and s.drafts) {
        // the head reads the last prompt row + first token and drafts on (node probabilities: whole reach)
        var who = win.who(s);
        const d: u32 = @intCast(if (e.cfg.node_probabilities) try e.rule.headDepth(&who, null, win.probe(e)) else try e.rule.depth(who));
        asked = d;
        try e.backend.draft(&.{.{ .stream = s, .follow = &.{}, .first = feed, .rows = null, .start = s.prompt_len, .position = position + 1, .depth = d }});
        s.dropHeld(e.gpa);
        s.next = .{ .count = d };
        if (e.backend.vtable.tree) |tree| if (try tree(e.backend.ptr, s, e.gpa)) |held| {
            s.next = held; // a tree head's first drafts as host tokens, as every later round's
        };
    } else if (e.cfg.pipelined and s.logprobs == null) {
        try e.queueNext(s, feed);
    }
    const value = try e.readFeed(feed);
    if (asked) |d| try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(s.id)), f("depth", int(d)), f("position", int(position + 1)), f("follow", .{ .u32s = &.{value} }), f("rows", .null) });
    if (e.log != null) {
        const first = if (feed == .handle) value else try e.backend.read(drawn);
        try trail.event(e, &.{ f("ev", str("first")), f("stream", str(s.id)), f("position", int(position)), f("drawn", int(first)), f("token", int(value)) });
    }
    var first_row: [1]LogRow = undefined;
    if (s.logprobs != null) first_row[0] = (try e.backend.firstRow(s)).forToken(value);
    _ = try s.commit(e.gpa, &.{value}, if (s.logprobs != null) &first_row else &.{});
    s.pending = value;
    try trail.resolve(e);
    if (s.finished) {
        try trail.finish(e, s);
        e.release(s);
        return;
    }
    try e.live.append(e.gpa, s);
}

/// A begun stream's prompt is in: last row, first token and drafts; a failure leaves it for the host to discard.
pub fn finishStream(e: *Engine, s: *Stream) !void {
    _ = e.arena.reset(.retain_capacity);
    // Config.join_tail: the last token waits for the next round (a forced or budget-cut first token keeps its row)
    if (e.cfg.join_tail and e.cfg.family_mtp and s.drafts and s.prompt_len > 0 and s.force.items.len == 0 and !s.cutsNext()) {
        if (e.backend.vtable.join) |join| {
            if (try join(e.backend.ptr, s)) return joined(e, s);
            return e.opened(s);
        }
    }
    const fin = e.backend.vtable.finish orelse return error.NoPieces;
    try fin(e.backend.ptr, s);
    return e.opened(s);
}

/// A round's finished prompts as `finishStream`, joined ones' first drafts in one call; returns how many joined.
pub fn finishStreams(e: *Engine, streams: []const *Stream, errs: []?anyerror) !usize {
    _ = e.arena.reset(.retain_capacity);
    const a = e.arena.allocator();
    var reqs: std.ArrayList(be.DraftRequest) = .empty;
    var joins: std.ArrayList(usize) = .empty;
    for (streams, errs, 0..) |s, *err, i| {
        err.* = null;
        const can = e.cfg.join_tail and e.cfg.family_mtp and s.drafts and s.prompt_len > 0 and s.force.items.len == 0 and !s.cutsNext();
        const join = if (can) e.backend.vtable.join else null;
        if (join) |j| {
            const ok = j(e.backend.ptr, s) catch |x| {
                err.* = x;
                continue;
            };
            if (ok) {
                try reqs.append(a, joinRequest(e, s) catch |x| {
                    err.* = x;
                    continue;
                });
                try joins.append(a, i);
                continue;
            }
        } else {
            const fin = e.backend.vtable.finish orelse return error.NoPieces;
            fin(e.backend.ptr, s) catch |x| {
                err.* = x;
                continue;
            };
        }
        e.opened(s) catch |x| {
            err.* = x;
        };
    }
    if (reqs.items.len == 0) return 0;
    e.backend.draft(reqs.items) catch |x| {
        for (joins.items) |i| errs[i] = x;
        return 0;
    };
    for (reqs.items) |r| joinedDrafted(e, r) catch |x| {
        for (joins.items) |i| if (streams[i] == r.stream) {
            errs[i] = x;
        };
    };
    return reqs.items.len;
}

/// A joined prompt (Backend.join): its last token pending, drafts asked now, verified in its first shared round.
fn joined(e: *Engine, s: *Stream) !void {
    s.context.shrinkRetainingCapacity(s.prompt_len);
    const last = s.context.items[s.prompt_len - 1];
    s.cache_len = s.prompt_len - 1;
    s.pending = last;
    const position: u64 = s.prompt_len;
    var who = win.who(s);
    const d: u32 = @intCast(if (e.cfg.node_probabilities) try e.rule.headDepth(&who, null, win.probe(e)) else try e.rule.depth(who));
    try e.backend.draft(&.{.{ .stream = s, .follow = &.{}, .first = .{ .value = last }, .rows = null, .start = s.cache_len, .position = position, .depth = d }});
    s.dropHeld(e.gpa);
    s.next = .{ .count = d };
    try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(s.id)), f("depth", int(d)), f("position", int(position)), f("follow", .{ .u32s = &.{last} }), f("rows", .null) });
    try trail.resolve(e);
    try e.live.append(e.gpa, s);
}

/// After its first draft: the joined stream holds it and takes part from the next round.
fn joinedDrafted(e: *Engine, r: be.DraftRequest) !void {
    const s = r.stream;
    s.dropHeld(e.gpa);
    s.next = .{ .count = r.depth };
    try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(s.id)), f("depth", int(r.depth)), f("position", int(r.position)), f("follow", .{ .u32s = &.{s.pending.?} }), f("rows", .null) });
    try trail.resolve(e);
    try e.live.append(e.gpa, s);
}

/// A joined prompt's state and its first draft request (the last token pending at the prompt's last position).
fn joinRequest(e: *Engine, s: *Stream) !be.DraftRequest {
    s.context.shrinkRetainingCapacity(s.prompt_len);
    const last = s.context.items[s.prompt_len - 1];
    s.cache_len = s.prompt_len - 1;
    s.pending = last;
    var who = win.who(s);
    const d: u32 = @intCast(if (e.cfg.node_probabilities) try e.rule.headDepth(&who, null, win.probe(e)) else try e.rule.depth(who));
    return .{ .stream = s, .follow = &.{}, .first = .{ .value = last }, .rows = null, .start = s.cache_len, .position = s.prompt_len, .depth = d };
}
