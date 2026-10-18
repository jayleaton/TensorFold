//! What a lane host reports beyond its counts on /health: the rounds' pace against its baseline, the backend's graphs.
const std = @import("std");
const api = @import("engine_api.zig");
const pace_mod = @import("pace.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

/// The pace (core/pace.zig, under the host's mutex) and the backend's graph caches (null: it keeps none).
pub const Watch = struct {
    pace: pace_mod.Pace = .{},
    graphs: ?api.GraphsSource = null,
};

/// The watch's part of Engine.status, read under the host's lock.
pub fn fill(h: *const LaneHost, out: *api.Status, now: i96) void {
    out.oldest_queued_s = if (h.queued.items.len > 0) seconds(now - oldestQueued(h)) else 0;
    out.pace = paceStatus(&h.watch.pace, now);
    if (h.watch.graphs) |g| out.graph_caches = @intCast(g.read(g.ctx, &out.graphs));
}

/// One decode step of the default loop, timed into the pace.
pub fn step(h: *LaneHost) !void {
    const s0 = std.Io.Clock.awake.now(h.io).toNanoseconds();
    try h.core.step();
    const s1 = std.Io.Clock.awake.now(h.io).toNanoseconds();
    round(h, s1, 0, 0, seconds(s1 - s0));
}

/// A round ended at `end`: its prompt rows, the pieces' seconds and the decode step's.
pub fn round(h: *LaneHost, end: i96, rows: u64, piece_s: f64, step_s: f64) void {
    h.mutex.lockUncancelable(h.io);
    defer h.mutex.unlock(h.io);
    h.watch.pace.round(end, rows, piece_s, step_s);
}

/// A request whose client left mid-prompt keeps its finished prefix as a session entry: the same prompt resumes.
pub fn keepLeaving(h: *LaneHost, job: *LaneHost.Job) void {
    const rd = h.rounds orelse return;
    const ask = rd.vtable.leaving orelse return;
    const at = ask(rd.ctx, @intFromPtr(job)) orelse return;
    h.core.piece(&job.stream, at, at, true) catch |e| std.log.warn("lane host: a leaving prompt's {d} rows were not kept ({t})", .{ at, e });
}

fn oldestQueued(h: *const LaneHost) i96 {
    var at = h.queued.items[0].submitted;
    for (h.queued.items) |job| at = @min(at, job.submitted);
    return at;
}

fn paceStatus(p: *const pace_mod.Pace, now: i96) api.PaceStatus {
    return .{
        .last_round_age_s = if (p.last_end) |e| seconds(now - e) else null,
        .last_round_s = @as(f64, @floatFromInt(p.last_ns)) / 1e9,
        .decode_s = if (p.decode.ready()) p.decode.quick else 0,
        .decode_base_s = if (p.decode.ready()) p.decode.base else 0,
        .prefill_rows_s = if (p.prefill.ready()) p.prefill.quick else 0,
        .prefill_base_rows_s = if (p.prefill.ready()) p.prefill.base else 0,
        .slow_s = p.slowFor(now),
    };
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

test {
    _ = @import("pace.zig");
    _ = @import("lane_host_resume_test.zig");
}
