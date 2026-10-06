//! M3's GPU target: draft/iface.zig's `Target` over forward.zig, so the lanes backend (draft/lanes.zig) drives the real
//! model as it drives the CPU twin. This first form is one slot, chain windows (a draft chain under the pending token),
//! greedy choices and the prefill as consecutive decode windows; trees (`parents`), sampling (`lanes.Sampling` keyed
//! at `draws`) and several slots are refused by name until state.zig (per-slot pools) and the sampler land.

const std = @import("std");
const iface = @import("draft/iface.zig");
const lanes = @import("lanes");
const fwd = @import("forward.zig");

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
        var at: usize = 0;
        var choice: [1]u32 = .{0};
        while (at < ids.len) {
            const n = @min(@min(t.prefill_rows, t.max_rows), ids.len - at);
            try t.f.window(ids[at .. at + n]);
            if (at + n == ids.len) {
                // the keyed choice after the prompt's last row: the last row's greedy pick
                var rows: [64]u32 = undefined;
                try t.f.greedy(rows[0..n]);
                choice[0] = rows[n - 1];
            }
            try t.f.keep(@intCast(n - 1));
            at += n;
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
        t.f.slot = .{};
    }
};
