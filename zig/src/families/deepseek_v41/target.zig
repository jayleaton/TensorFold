//! M3's GPU target: draft/iface.zig's `Target` over forward.zig, so the lanes backend (draft/lanes.zig) drives the real
//! model as it drives the CPU twin. This first form is one slot, chain windows (a draft chain under the pending token),
//! greedy choices and the prefill as consecutive decode windows; trees (`parents`) run as chain windows on the slot
//! (draft/branches.zig: the main chain, then a winning sibling's path in place of it); several slots are refused by
//! name until state.zig (per-slot pools) lands. Sampling (`lanes.Sampling`, T > 0, keyed at `draws`) goes through
//! `sampler` (sampling_gpu.zig: Python's candidates, nucleus statistics and keyed choice); without one it is refused.
//! M5 (`sessions`, with the paged pool): a prompt resumes the longest saved entry that strictly prefixes it (RAM or
//! NVMe) and prefills only the rest; a released slot is saved as an entry (its pages shared, its bounded state kept),
//! the residency budget parking or dropping the least recently used chats.

const std = @import("std");
const iface = @import("draft/iface.zig");
const lanes = @import("lanes");
const fwd = @import("forward.zig");
const sg = @import("sessions_gpu.zig");
const batch = @import("batch.zig");
const samp = @import("sampling_gpu.zig");
const branches = @import("draft/branches.zig");

pub const Error = error{ TreeWindow, Sampling, Slots, NotAChain, WindowRows };

pub const GpuTarget = struct {
    f: *fwd.Forward,
    /// rows a prefill window takes (the decode path's buckets; the forward plans buffers for them)
    prefill_rows: u32 = 16,
    /// a slot that already holds the prompt (a gate resuming the reference's state): prefill returns this choice
    /// without running the prompt again
    resume_first: ?u32 = null,
    /// the most rows a window may take: the forward's buffers are planned for its buckets (run.zig has no bounds)
    max_rows: u32 = 16,
    /// the prompt through the forward's own prefill (2,048-row segments, Python's chunks; forward.planPrefill sized
    /// its buffers) instead of `prefill_rows`-row decode windows
    own_prefill: bool = false,
    /// M5: the session store over the paged pool (null: every prompt from an empty slot, nothing kept)
    sessions: ?*sg.Sessions = null,
    /// several live slots (TF_DSV41_SLOTS > 1, batch.zig): windows in row mode over every slot's segment, the prompt on
    /// the activated slot (greedy chains; trees and sampling refused there by name)
    rows: ?*batch.Batch = null,
    /// the keyed sampler (model.zig makes it on every rank): sampled choices; null refuses them (error.Sampling)
    sampler: ?*samp.GpuSampler = null,
    /// draft trees (`parents`): resolved as chain windows on the slot (draft/branches.zig), the last one kept by path
    tree: branches.Resolver = .{},

    pub fn target(t: *GpuTarget) iface.Target {
        return .{ .ptr = t, .vtable = &vtable };
    }

    const vtable: iface.Target.VTable = .{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release };

    fn self_(p: *anyopaque) *GpuTarget {
        return @ptrCast(@alignCast(p));
    }

    fn prefill(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const t = self_(p);
        if (t.rows) |b| try b.beginPrefill(slot) else if (slot != 0) return error.Slots;
        const keyed = try t.keyedOf(sampling);
        t.tree.clear();
        if (t.resume_first) |first| if (t.f.slot.pos == ids.len and t.f.slot.pending == null) return first;
        // a saved entry that prefixes the prompt: the slot continues at its position, the rest is prefilled
        const at: usize = if (t.sessions) |ss| @intCast(try ss.resume_(ids)) else 0;
        const rest = ids[at..];
        if (t.own_prefill) {
            // the keyed choice after the prompt: the last segment's last row
            try t.f.prefill(rest);
            const seg: usize = @intCast(t.f.opts.prefill_rows);
            const last = if (rest.len % seg == 0) seg else rest.len % seg;
            if (keyed) |s| return t.pickOne(@intCast(last - 1), s, draw);
            const picks = try t.f.gpa.alloc(u32, last);
            defer t.f.gpa.free(picks);
            try t.f.greedy(picks);
            return picks[last - 1];
        }
        var at_ = at;
        var choice: [1]u32 = .{0};
        while (at_ < ids.len) {
            const n = @min(@min(t.prefill_rows, t.max_rows), ids.len - at_);
            try t.f.window(ids[at_ .. at_ + n]);
            if (at_ + n == ids.len) {
                // the keyed choice after the prompt's last row: the last row's greedy pick, or its sampled one
                if (keyed) |s| choice[0] = try t.pickOne(@intCast(n - 1), s, draw) else {
                    var rows: [64]u32 = undefined;
                    try t.f.greedy(rows[0..n]);
                    choice[0] = rows[n - 1];
                }
            }
            try t.f.keep(@intCast(n - 1));
            at_ += n;
        }
        return choice[0];
    }

    fn window(p: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        const t = self_(p);
        if (t.rows) |b| return b.targetWindow(segments, choices);
        if (segments.len != 1) return error.Slots;
        const s = segments[0];
        if (s.slot != 0) return error.Slots;
        if (s.parents != null) return t.tree.window(.{ .ptr = t, .vtable = &chains }, s, choices[0]);
        t.tree.clear();
        const keyed = try t.keyedOf(s.sampling);
        // a chain window never kept keeps every row (iface.Target.window): lanes' undrafted step (queue) sends the
        // next row without a keep
        if (t.f.slot.pending) |w| {
            if (s.start != w.start + w.n) return error.NotAChain;
            try t.f.keep(w.n - 1);
        }
        if (s.start != t.f.slot.pos) return error.NotAChain;
        if (s.tokens.len > t.max_rows) return error.WindowRows;
        try t.f.window(s.tokens);
        if (keyed) |smp| return t.sampler.?.choose(0, @intCast(s.tokens.len), smp, s.draws, choices[0][0..s.tokens.len]);
        try t.f.greedy(choices[0][0..s.tokens.len]);
    }

    /// T > 0: the sampling the sampler keys (null: greedy); refused without a sampler.
    fn keyedOf(t: *const GpuTarget, sampling: ?lanes.Sampling) !?lanes.Sampling {
        if (!samp.sampled(sampling)) return null;
        if (t.sampler == null) return error.Sampling;
        return sampling;
    }

    /// The keyed choice of "w.logits" row `row`, drawn at `draw`.
    fn pickOne(t: *GpuTarget, row: u32, s: lanes.Sampling, draw: u64) !u32 {
        var one: [1]u32 = undefined;
        try t.sampler.?.choose(row, 1, s, &.{draw}, &one);
        return one[0];
    }

    fn keep(p: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        const t = self_(p);
        if (t.rows) |b| return b.keepPath(slot, path);
        if (slot != 0) return error.Slots;
        if (t.tree.live) return t.f.keep(try t.tree.keep(path));
        // a chain keeps rows 0..k: the forward commits row 0 + k accepted drafts
        for (path, 0..) |row, i| if (row != i) return error.NotAChain;
        if (path.len == 0) return error.NotAChain;
        try t.f.keep(@intCast(path.len - 1));
    }

    fn taps(p: *anyopaque, slot: u32) iface.Taps {
        const t = self_(p);
        if (t.rows) |b| if (b.tapsOf(slot)) |x| return .{ .address = x.address, .rows = x.rows, .stride = @intCast(t.f.cfg.dspark_targets.items().len * t.f.cfg.hidden) };
        const n: u32 = if (t.f.slot.pending) |w| w.n else @intCast(t.f.ids.len);
        const D: u32 = t.f.cfg.hidden;
        const k: u32 = @intCast(t.f.cfg.dspark_targets.items().len);
        return .{ .address = t.f.runner.addressOf("w.taps") orelse 0, .rows = n, .stride = k * D, .map = t.tree.map() };
    }

    /// The tree resolver's chains: a chain window through `window` (its rules and refusals), dropped on every rank.
    const chains: branches.Chains.VTable = .{ .run = chainRun, .drop = chainDrop };

    fn chainRun(p: *anyopaque, slot: u32, start: u64, tokens: []const u32, draws: []const u64, sampling: ?lanes.Sampling, out: []u32) anyerror!void {
        var one = [_][]u32{out};
        return window(p, &.{.{ .slot = slot, .start = start, .tokens = tokens, .parents = null, .draws = draws, .sampling = sampling }}, &one);
    }

    fn chainDrop(p: *anyopaque, slot: u32) anyerror!void {
        if (slot != 0) return error.Slots;
        return self_(p).f.drop();
    }

    fn release(p: *anyopaque, slot: u32) void {
        const t = self_(p);
        if (t.rows) |b| return b.release(slot) catch |e| std.log.err("dsv41 target: releasing slot {d} failed ({t})", .{ slot, e });
        if (slot != 0) return;
        t.tree.clear();
        if (t.sessions) |ss| {
            // the turn kept for the next one (a pending window is the reply's last token: committed first)
            if (t.f.slot.pending) |w| t.f.keep(w.n - 1) catch |e| std.log.err("sessions: keep before save failed ({t})", .{e});
            _ = ss.save() catch |e| {
                std.log.err("sessions: save failed ({t}); the slot is released", .{e});
                t.f.release() catch |e2| std.log.err("dsv41 target: releasing the slot on the other ranks failed ({t})", .{e2});
            };
            return;
        }
        t.f.release() catch |e| std.log.err("dsv41 target: releasing the slot on the other ranks failed ({t})", .{e});
    }
};
