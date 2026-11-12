//! A round's committed tokens cut where the loop guard, the thinking budget or a tool_choice call gate takes over.
const std = @import("std");
const sm = @import("stream.zig");
const call_gate = @import("call_gate.zig");
const Stream = sm.Stream;

/// Cut `committed` and `path` at the earliest taker: its token goes in last and its rest is forced; the cut row.
pub fn apply(e: anytype, s: *Stream, committed: *std.ArrayList(u32), path: *[]u32) !?usize {
    const a = e.arena.allocator();
    const budget_cut = s.thinkCut(committed.items);
    const loop_cut = sm.loopCut(s, committed.items);
    const never = std.math.maxInt(usize);
    // the gate wins ties with the loop guard and yields to the budget (patch 0007's order)
    if (s.gate) |*g| if (try g.cut(a, committed.items)) |hit| if (hit.at <= (loop_cut orelse never) and hit.at < (budget_cut orelse never)) {
        path.* = path.*[0 .. hit.at + 1];
        committed.shrinkRetainingCapacity(hit.at);
        try committed.append(a, hit.fix[0]);
        try call_gate.queue(e.gpa, s, hit.fix[1..]);
        return hit.at;
    };
    if (loop_cut) |c| if (budget_cut == null or c < budget_cut.?) {
        path.* = path.*[0 .. c + 1];
        committed.shrinkRetainingCapacity(c + 1);
        return c;
    };
    if (budget_cut) |c| {
        path.* = path.*[0 .. c + 1];
        committed.shrinkRetainingCapacity(c);
        try committed.append(a, try s.startClose(e.gpa));
        return c;
    }
    return null;
}
