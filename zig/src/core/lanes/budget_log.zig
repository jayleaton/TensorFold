//! TF_LANES_BUDGET_LOG=1: one stderr line a shared or lone round with its draft budgets, the head's drafts and costs.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const win = @import("windows.zig");

pub const name = "TF_LANES_BUDGET_LOG";
const costs_every = 32; // logged rounds between cost tables; a changed shared-cost table prints at once

/// A round as its line prints it; a `got` below 0 is a stream whose head holds no count.
pub const Line = struct {
    step: u64,
    live: usize,
    streams: usize,
    rows: u64,
    path: []const u8,
    stops: bool,
    ask: []const i64,
    got: []const i64,
    over_ms: f64,
    est_ms: f64,
    head_ms: f64,
};

/// The switch (read once, at engine init) and a shared round's budgets, kept from its budgets to its overhead.
pub const Log = struct {
    on: bool = false,
    logged: u64 = 0,
    costs_hash: ?u64 = null,
    path: []const u8 = "",
    est_ms: f64 = 0,
    sampled: bool = false,
    ask: []const i64 = &.{},
    budgets: []const i64 = &.{}, // the round's own slice, zeroed by root copies as headMs reads it
    heads: []const *Stream = &.{},

    pub fn init() Log {
        comptime @import("sys").environment.registered(name);
        return .{ .on = @import("sys").environment.read(bool, name, false) catch false };
    }
};

/// Which budget rule ran, in Rule.budgets' (and the shared draft allocator's) order.
fn pathName(e: *const Engine) []const u8 {
    const r = &e.rule;
    if (e.cfg.allocate_shared_drafts) return "shared";
    if (r.confidence_floor != null) return "floor";
    if (r.stopsNow()) return "stops";
    if (r.node_probabilities) return "nodes";
    if (r.pooled_budgets) return "pooled";
    return if (r.draft_step_ms != 0) "priced" else "fixed";
}

/// At the round's budgets event: what the rule allocated against and the asks, before root copies zero any.
pub fn budgets(e: *Engine, plans: []const win.Plan, heads: []const usize, depths: []const i64) !void {
    const a = e.arena.allocator();
    const ss = try a.alloc(*Stream, heads.len);
    for (ss, heads) |*s, i| s.* = plans[i].stream;
    var live: usize = 0;
    for (plans) |p| live += @intFromBool(!p.stream.finished and !p.stream.paused);
    const l = &e.blog;
    l.sampled = win.anySampled(plans);
    l.path = pathName(e);
    const n = if (std.mem.eql(u8, l.path, "shared")) live else heads.len;
    l.est_ms = if (std.mem.eql(u8, l.path, "fixed")) e.rule.fixedMs(n, l.sampled) else e.rule.overheadMs(n, l.sampled);
    l.ask = try a.dupe(i64, depths);
    l.budgets = depths;
    l.heads = ss;
}

/// At the round's overhead event (`one`: a lone round's stream): the budget line, and the cost table when due.
pub fn round(e: *Engine, one: ?*Stream, streams: usize, rows: u64, ms: f64) void {
    const l = &e.blog;
    var ask1 = [1]i64{0};
    var got1 = [1]i64{0};
    var got_buf: [256]i64 = undefined;
    var line: Line = .{ .step = e.steps, .live = liveCount(e), .streams = streams, .rows = rows, .path = l.path, .stops = e.rule.stopsNow(), .ask = l.ask, .got = &.{}, .over_ms = ms, .est_ms = l.est_ms, .head_ms = 0 };
    if (one) |s| {
        ask1[0] = s.held_levels;
        got1[0] = held(s);
        line.path = "lone";
        line.ask = &ask1;
        line.got = &got1;
        line.est_ms = e.rule.overheadMs(1, s.sampling != null);
    } else {
        const n = @min(l.heads.len, got_buf.len);
        for (got_buf[0..n], l.heads[0..n]) |*g, s| g.* = held(s);
        line.got = got_buf[0..n];
        line.head_ms = e.rule.headMs(l.budgets);
    }
    const sampled = if (one) |s| s.sampling != null else l.sampled;
    const kept = if (sampled) e.rule.overhead.items else e.rule.greedy_overhead.items;
    const hash = tableHash(e.rule.shared_costs.slots);
    if (l.logged % costs_every == 0 or (l.costs_hash orelse ~hash) != hash) {
        var over: [64][3]f64 = undefined;
        const k = @min(kept.len, over.len);
        for (over[0..k], kept[0..k]) |*o, x| o.* = .{ @floatFromInt(x.streams), x.ms, x.head };
        var table: [16384]u8 = undefined;
        var tw: std.Io.Writer = .fixed(&table);
        writeCosts(&tw, e.steps, e.rule.shared_costs.slots, over[0..k]) catch {};
        std.debug.print("{s}\n", .{table[0..tw.end]});
        l.costs_hash = hash;
    }
    l.logged += 1;
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    writeLine(&w, line) catch {};
    std.debug.print("{s}\n", .{buf[0..w.end]});
    if (one == null) l.* = .{ .on = l.on, .logged = l.logged, .costs_hash = l.costs_hash };
}

fn held(s: *const Stream) i64 {
    return if (s.next) |h| @intCast(h.count) else -1;
}

fn liveCount(e: *const Engine) usize {
    var n: usize = 0;
    for (e.live.items) |s| n += @intFromBool(!s.finished and !s.paused);
    return n;
}

fn tableHash(slots: []const ?f64) u64 {
    var h = std.hash.Wyhash.init(0);
    for (slots, 0..) |slot, k| if (slot) |v| {
        h.update(std.mem.asBytes(&k));
        h.update(std.mem.asBytes(&v));
    };
    return h.final();
}

fn list(w: *std.Io.Writer, xs: []const i64) !void {
    for (xs, 0..) |x, i| {
        if (i > 0) try w.writeAll(",");
        if (x < 0) try w.writeAll("-") else try w.print("{d}", .{x});
    }
}

/// `[lanes-budget] step=.. live=.. streams=.. rows=.. path=.. stops=0|1 ask=a,b got=a,- over_ms= est_ms= head_ms=`.
pub fn writeLine(w: *std.Io.Writer, x: Line) !void {
    try w.print("[lanes-budget] step={d} live={d} streams={d} rows={d} path={s} stops={d} ask=", .{ x.step, x.live, x.streams, x.rows, x.path, @intFromBool(x.stops) });
    try list(w, x.ask);
    try w.writeAll(" got=");
    try list(w, x.got);
    try w.print(" over_ms={d:.3} est_ms={d:.3} head_ms={d:.3}", .{ x.over_ms, x.est_ms, x.head_ms });
}

/// `[lanes-costs] step=.. rows=R:ms,.. over=N:fixed+head,..`: the shared-cost table and the overheads by streams.
pub fn writeCosts(w: *std.Io.Writer, step: u64, slots: []const ?f64, over: []const [3]f64) !void {
    try w.print("[lanes-costs] step={d} rows=", .{step});
    var first = true;
    for (slots, 0..) |slot, k| if (slot) |v| {
        try w.print("{s}{d}:{d:.3}", .{ if (first) "" else ",", k, v });
        first = false;
    };
    try w.writeAll(" over=");
    for (over, 0..) |o, i| try w.print("{s}{d}:{d:.3}+{d:.3}", .{ if (i == 0) "" else ",", @as(u64, @intFromFloat(o[0])), o[1], o[2] });
}

test "budget log: a round's line and the cost table print in the format one parser reads for both engines" {
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeLine(&w, .{ .step = 7, .live = 16, .streams = 3, .rows = 9, .path = "pooled", .stops = false, .ask = &.{ 3, 0, 2 }, .got = &.{ 2, -1, 2 }, .over_ms = 1.25, .est_ms = 1.5, .head_ms = 0 });
    try std.testing.expectEqualStrings("[lanes-budget] step=7 live=16 streams=3 rows=9 path=pooled stops=0 ask=3,0,2 got=2,-,2 over_ms=1.250 est_ms=1.500 head_ms=0.000", buf[0..w.end]);
    w = .fixed(&buf);
    try writeCosts(&w, 7, &.{ null, 5.0, 5.25 }, &.{ .{ 1, 0.5, 0 }, .{ 16, 2, 0.25 } });
    try std.testing.expectEqualStrings("[lanes-costs] step=7 rows=1:5.000,2:5.250 over=1:0.500+0.000,16:2.000+0.250", buf[0..w.end]);
}
