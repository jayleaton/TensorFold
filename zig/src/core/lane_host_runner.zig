//! The in-model runner on the engine thread: each round advances the reply sessions and continues held streams.
const std = @import("std");
const lanes = @import("lanes");
const LaneHost = @import("lane_host.zig").LaneHost;

/// Advances every started job's runner session; a settled result is forced after the trigger it waited at.
pub fn tick(h: *LaneHost) void {
    if (h.runner == null) return;
    for (h.admitted.items) |job| {
        if (!job.started) continue;
        const s = job.runner orelse continue;
        const tokens = s.tick() catch {
            fail(job, "out of memory for a runner result");
            continue;
        } orelse continue;
        defer h.gpa.free(tokens);
        const ok = lanes.hold.resumeAwaiting(&job.stream, h.gpa, tokens) catch {
            fail(job, "out of memory for a runner result");
            continue;
        };
        if (!ok) std.log.warn("lane host: a runner result for request {d}, which is not held, was dropped", .{job.id});
    }
}

/// An idle loop's wait: 1 ms while a stream is held at a runner trigger (its run settles off-thread, waking nothing).
pub fn idleWait(h: *LaneHost, idle_ms: i64) i64 {
    if (h.runner == null) return idle_ms;
    for (h.admitted.items) |job| {
        if (!job.started) continue;
        const s = job.runner orelse continue;
        if (s.awaiting()) return @min(idle_ms, held_poll_ms);
    }
    return idle_ms;
}

const held_poll_ms = 1;

fn fail(job: anytype, words: []const u8) void {
    const st = &job.stream;
    st.held = false;
    st.paused = false;
    st.awaiting = false;
    st.finished = true;
    st.reason = .@"error";
    st.problem = words;
}
