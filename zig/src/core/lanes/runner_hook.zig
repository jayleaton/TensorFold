//! The runner callbacks used at token commits and terminal handoffs.
pub const RunnerHook = struct {
    ptr: *anyopaque,
    token: *const fn (ptr: *anyopaque, token: u32, think_open: bool, can_hold: bool, think_end: bool) bool,
    eos: ?*const fn (ptr: *anyopaque, think_open: bool, can_hold: bool) bool = null,
    settled: ?*const fn (ptr: *anyopaque) ?[]const u32 = null, // at a hold: a result forced next instead, or null
};
