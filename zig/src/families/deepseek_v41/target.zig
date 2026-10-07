//! M3's GPU target: draft/iface.zig's `Target` over forward.zig, so the lanes backend (draft/lanes.zig) drives the real
//! model as it drives the CPU twin. This first form is one slot, chain windows (a draft chain under the pending token),
//! greedy choices and the prefill as consecutive decode windows; trees (`parents`), sampling (`lanes.Sampling` keyed
//! at `draws`) and several slots are refused by name until state.zig (per-slot pools) and the sampler land.
//! M5 (`sessions`, with the paged pool): a prompt resumes the longest saved entry that strictly prefixes it (RAM or
//! NVMe) and prefills only the rest; a released slot is saved as an entry (its pages shared, its bounded state kept),
//! the residency budget parking or dropping the least recently used chats.

const std = @import("std");
const iface = @import("draft/iface.zig");
const lanes = @import("lanes");
const fwd = @import("forward.zig");
const sg = @import("sessions_gpu.zig");

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

    pub fn target(t: *GpuTarget) iface.Target {
        return .{ .ptr = t, .vtable = &vtable };
    }

    const vtable: iface.Target.VTable = .{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release };

    fn self_(p: *anyopaque) *GpuTarget {
        return @ptrCast(@alignCast(p));
    }

    fn prefill(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32 {
        _ = draw;
        const t = self_(p);
        if (slot != 0) return error.Slots;
        if (sampling != null) return error.Sampling;
        if (t.resume_first) |first| if (t.f.slot.pos == ids.len and t.f.slot.pending == null) return first;
        // a saved entry that prefixes the prompt: the slot continues at its position, the rest is prefilled
        const at: usize = if (t.sessions) |ss| @intCast(try ss.resume_(ids)) else 0;
        const rest = ids[at..];
        if (t.own_prefill) {
            // the keyed choice after the prompt: the last segment's last row
            try t.f.prefill(rest);
            const seg: usize = @intCast(t.f.opts.prefill_rows);
            const last = if (rest.len % seg == 0) seg else rest.len % seg;
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
                // the keyed choice after the prompt's last row: the last row's greedy pick
                var rows: [64]u32 = undefined;
                try t.f.greedy(rows[0..n]);
                choice[0] = rows[n - 1];
            }
            try t.f.keep(@intCast(n - 1));
            at_ += n;
        }
        return choice[0];
    }

    fn window(p: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        const t = self_(p);
        if (segments.len != 1) return error.Slots;
        const s = segments[0];
        if (s.slot != 0) return error.Slots;
        if (s.parents != null) return error.TreeWindow;
        if (s.sampling != null) return error.Sampling;
        // a chain window never kept keeps every row (iface.Target.window): lanes' undrafted step (queue) sends the
        // next row without a keep
        if (t.f.slot.pending) |w| {
            if (s.start != w.start + w.n) return error.NotAChain;
            try t.f.keep(w.n - 1);
        }
        if (s.start != t.f.slot.pos) return error.NotAChain;
        if (s.tokens.len > t.max_rows) return error.WindowRows;
        try t.f.window(s.tokens);
        try t.f.greedy(choices[0][0..s.tokens.len]);
    }

    fn keep(p: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        const t = self_(p);
        if (slot != 0) return error.Slots;
        // a chain keeps rows 0..k: the forward commits row 0 + k accepted drafts
        for (path, 0..) |row, i| if (row != i) return error.NotAChain;
        if (path.len == 0) return error.NotAChain;
        try t.f.keep(@intCast(path.len - 1));
    }

    fn taps(p: *anyopaque, slot: u32) iface.Taps {
        const t = self_(p);
        _ = slot;
        const n: u32 = if (t.f.slot.pending) |w| w.n else @intCast(t.f.ids.len);
        const D: u32 = t.f.cfg.hidden;
        const k: u32 = @intCast(t.f.cfg.dspark_targets.items().len);
        return .{ .address = t.f.runner.addressOf("w.taps") orelse 0, .rows = n, .stride = k * D };
    }

    fn release(p: *anyopaque, slot: u32) void {
        const t = self_(p);
        if (slot != 0) return;
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
