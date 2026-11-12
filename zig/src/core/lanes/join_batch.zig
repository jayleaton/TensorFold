//! Config.join_batch (opt-in): a joiner's first token goes out at once, its first drafts with the next step's batch.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const be = @import("backend.zig");
const Stream = @import("stream.zig").Stream;
const ev = @import("events.zig");
const win = @import("windows.zig");
const trail = @import("trail.zig");
const LogRow = @import("logprob.zig").Row;
const f = ev.f;
const str = trail.str;
const int = trail.int;

/// Whether `s` opens this way: join_batch on, and a drafting stream with the family's head.
pub fn joins(e: *const Engine, s: *const Stream) bool {
    return e.cfg.join_batch and e.cfg.family_mtp and s.drafts;
}

/// After the prompt pass: the first token drawn and committed, no drafts yet; the stream joins the rounds.
pub fn opened(e: *Engine, s: *Stream, hook: ?be.Committed) !void {
    if (s.isCancelled()) { // cancelled in its last chunk: no first token
        e.backend.release(s);
        return error.Cancelled;
    }
    try e.live.ensureUnusedCapacity(e.gpa, 1);
    s.context.shrinkRetainingCapacity(s.prompt_len);
    s.rows.clearRetainingCapacity();
    s.pending = null;
    s.cache_len = s.prompt_len;
    const position: u64 = s.prompt_len;
    const drawn = try e.drawFirst(s, position);
    var feed: be.Feed = .{ .handle = drawn };
    if (try e.forcedNext(s)) |t| feed = .{ .value = t };
    const drew = try @import("call_gate.zig").first(e, s, &feed); // a call gate reads and fixes the draw first
    const value = try e.readFeed(feed);
    if (e.log != null) {
        const actual = if (feed == .handle) value else drew orelse try e.backend.read(drawn);
        try trail.event(e, &.{ f("ev", str("first")), f("stream", str(s.id)), f("position", int(position)), f("drawn", int(actual)), f("token", int(value)) });
    }
    var row: [1]LogRow = undefined;
    if (s.logprobs != null) row[0] = (try e.backend.firstRow(s)).forToken(value);
    _ = try s.commit(e.gpa, &.{value}, if (s.logprobs != null) &row else &.{});
    s.pending = value;
    s.dropHeld(e.gpa);
    if (s.finished) {
        {
            defer e.release(s); // the host will not discard a finished stream if tracing fails
            try trail.resolve(e);
            try trail.finish(e, s);
        }
        if (hook) |h| h.call(h.ptr);
        return;
    }
    try trail.resolve(e);
    s.join_draft = true; // its first drafts come with the next step's other joiners (`step`)
    e.live.appendAssumeCapacity(s);
    if (hook) |h| h.call(h.ptr);
}

/// A step's start: head stops' live count, then every joiner's first drafts in batches of a head step's streams.
pub fn step(e: *Engine) !void {
    e.rule.live = e.live.items.len;
    if (!e.cfg.join_batch) return;
    var reqs: std.ArrayList(be.DraftRequest) = .empty;
    defer reqs.deinit(e.gpa);
    for (e.live.items) |s| {
        if (!s.join_draft) continue;
        s.join_draft = false;
        if (s.finished or s.paused) continue;
        const d: u32 = @intCast(try e.rule.depth(win.who(s)));
        try reqs.append(e.gpa, .{ .stream = s, .follow = &.{}, .first = .{ .value = s.pending.? }, .rows = null, .start = s.prompt_len, .position = s.prompt_len + 1, .depth = d });
    }
    if (reqs.items.len == 0) return;
    const per: usize = @max(1, e.cfg.batch_streams);
    var at: usize = 0;
    while (at < reqs.items.len) : (at += per) try e.backend.draft(reqs.items[at..@min(reqs.items.len, at + per)]);
    for (reqs.items) |r| {
        const s = r.stream;
        s.dropHeld(e.gpa);
        s.next = .{ .count = r.depth };
        // the drafts the head holds (its stop rule may end the chain short of the ask), as askDrafts reads them
        if (e.backend.vtable.tree) |tree| if (try tree(e.backend.ptr, s, e.gpa)) |held| {
            s.next = held;
        };
        try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(s.id)), f("depth", int(r.depth)), f("position", int(r.position)), f("follow", .{ .u32s = &.{s.pending.?} }), f("rows", .null) });
    }
}
