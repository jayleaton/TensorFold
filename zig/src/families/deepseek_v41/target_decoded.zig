//! TF_DSV41_DECODED_ROWS on the GPU target: a piece's rows in a slot's regions (decoded_rows.zig) as decode windows.

const std = @import("std");
const iface = @import("draft/iface.zig");
const pf = @import("forward_prefill.zig");
const rows = @import("decoded_rows.zig");
const GpuTarget = @import("target.zig").GpuTarget;

pub const Region = rows.Region;
pub const touches = rows.touches;

/// The rows a window takes here: the decode rounds' own row buffers bound them.
const max_window = 64;

pub fn regionsOf(t: *const GpuTarget, slot: u32) []const Region {
    const d = t.decoded orelse return &.{};
    return if (slot < d.len) d[slot] else &.{};
}

pub fn clear(t: *GpuTarget, r: *[]Region) void {
    if (r.len != 0) t.f.gpa.free(r.*);
    r.* = &.{};
}

/// The slots' regions and their table freed (the target's teardown); knob off: nothing was allocated.
pub fn deinit(t: *GpuTarget) void {
    const d = t.decoded orelse return;
    for (d) |*r| clear(t, r);
    t.f.gpa.destroy(d);
    t.decoded = null;
}

/// Iface.Target.spans: the slot's next prompt's regions from its reply spans; regions with an image's rows prefill.
pub fn spans(p: *anyopaque, slot: u32, sp: []const [2]u32, ids: []const u32) anyerror!void {
    const t: *GpuTarget = @ptrCast(@alignCast(p));
    const mode = pf.state(t.f).decoded_rows;
    if (mode == .off) return; // the table stays null, so the knob-off paths never look a region up
    if (slot >= 16) return error.Slots;
    const d = t.decoded orelse made: {
        const fresh = try t.f.gpa.create([16][]Region);
        fresh.* = @splat(&.{});
        t.decoded = fresh;
        break :made fresh;
    };
    clear(t, &d[slot]);
    if (sp.len == 0 or ids.len == 0) return;
    const all = try rows.regions(t.f.gpa, mode, sp, ids.len - 1);
    var n: usize = 0;
    for (all) |r| if (!@import("image_opts.zig").hasImage(ids[@intCast(r[0])..@intCast(r[1])])) {
        all[n] = r;
        n += 1;
    };
    if (n == 0) return t.f.gpa.free(all);
    d[slot] = if (n == all.len) all else blk: {
        defer t.f.gpa.free(all);
        break :blk try t.f.gpa.dupe(Region, all[0..n]);
    };
}

/// Whether a round's pieces include a slot with regions (they then run one at a time).
pub fn anyIn(t: *const GpuTarget, list: []const iface.Target.Piece) bool {
    for (list) |q| if (regionsOf(t, q.slot).len != 0) return true;
    return false;
}

/// Whether the slot's position is a replay point a snapshot can keep: not inside or at the end of a region.
pub fn savable(t: *const GpuTarget, slot: u32) bool {
    const pos = t.f.slot.pos;
    for (regionsOf(t, slot)) |r| if (r[0] < pos and pos <= r[1]) return false;
    return true;
}

/// A piece across regions: prefill sub-ranges as `prefillPiece` runs them, region rows as kept decode windows.
pub fn piece(t: *GpuTarget, slot: u32, ids: []const u32, start: u64, end: u64, save: bool) !void {
    if (t.f.slot.pos != start or start >= end or end + 1 > ids.len) return error.NotAChain;
    var at = start;
    while (at < end) {
        const part = rows.next(regionsOf(t, slot), at, end);
        try t.onSlot(slot);
        // a snapshot only where the piece ends, after prefilled rows
        if (part.decoded) try windows(t, slot, ids, at, part.end) else try t.prefillPiece(slot, ids, at, part.end, save and part.end == end);
        at = part.end;
    }
}

/// Rows [a, b) as decode windows, every row kept: the kernels and row shapes the decode rounds ran them with.
fn windows(t: *GpuTarget, slot: u32, ids: []const u32, a: u64, b: u64) !void {
    // CED: the rows prefilled up to here get their decoder replay first, as at a prompt's end, its taps to the drafter
    if (pf.state(t.f).mode == .replay and t.f.slot.ced.hi == a) {
        pf.state(t.f).taps_rows = 0;
        try t.f.finishPrompt();
        try t.handTaps(slot);
    }
    var path: [max_window]u32 = undefined;
    for (&path, 0..) |*x, i| x.* = @intCast(i);
    var picks: [max_window]u32 = undefined;
    const w: u64 = @max(1, @min(t.max_rows, max_window));
    var at = a;
    while (at < b) {
        const n: usize = @intCast(@min(w, b - at));
        const seg: iface.Segment = .{ .slot = slot, .start = at, .tokens = ids[@intCast(at)..][0..n], .parents = null, .draws = &.{}, .sampling = null };
        var out = [_][]u32{picks[0..n]};
        GpuTarget.window(t, &.{seg}, &out) catch |err| {
            if (err == error.PoolExhausted) t.abandon(slot);
            return err;
        };
        try GpuTarget.keep(t, slot, path[0..n]);
        at += n;
    }
}
