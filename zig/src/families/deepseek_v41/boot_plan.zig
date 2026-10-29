//! Complete prompt buffer planning before checking the enabled request paths.
const Forward = @import("forward.zig").Forward;
const Plan = @import("buffers.zig").Plan;
pub fn prefill(f: *Forward, p: *Plan, enabled: bool) !void {
    if (enabled) try f.planPrefill(p, @intCast(f.opts.prefill_rows));
    try @import("vision_open.zig").checkPlan(f, p);
}
