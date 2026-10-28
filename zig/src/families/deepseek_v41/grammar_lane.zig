//! Shared lane grammar handoff: the matcher and draft cuts stay in core.lanes.
const std = @import("std");
const lanes = @import("lanes");
const Lanes = @import("draft/lanes.zig").Lanes;

pub fn prompt(x: *Lanes, s: *lanes.Stream, slot: u32) !void {
    var arena = std.heap.ArenaAllocator.init(x.gpa);
    defer arena.deinit();
    const mask = if (s.grammar) |g| try g.next(arena.allocator()) else null;
    try x.target.promptMask(slot, mask orelse &.{});
}

pub fn firstMasked(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
    const x: *Lanes = @ptrCast(@alignCast(ptr));
    const st = x.state(s) orelse return error.UnknownStream;
    if (position != st.len) return error.PositionMismatch;
    if (!lanes.grammar.allows(mask, st.first)) return error.GrammarRejected;
    try x.drawn.append(x.gpa, st.first);
    return x.drawn.items.len - 1;
}

pub fn held(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
    const x: *Lanes = @ptrCast(@alignCast(ptr));
    const st = x.state(s) orelse return error.UnknownStream;
    if (out.len > st.nodes()) return error.HeldMismatch;
    const main = @min(out.len, st.n);
    @memcpy(out[0..main], st.held[0..main]);
    var at: usize = main;
    for (st.sib_tokens[0..st.n_sib], st.sib_len[0..st.n_sib]) |tokens, n| {
        const take = @min(n, out.len - at);
        @memcpy(out[at..][0..take], tokens[0..take]);
        at += take;
        if (at == out.len) break;
    }
}
