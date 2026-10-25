//! The dspark draft behind core/lanes/drafter.zig: every device step goes through the leader, so rank 1 runs it too.
const std = @import("std");
const lanes = @import("lanes");
const iface = @import("iface.zig");
const ops = @import("engine_ops.zig");
const backend = @import("backend.zig");
const dc = @import("dspark_config.zig");

const dr = lanes.drafter;

/// What the round loop is told of the draft's cost: UNVERIFIED planning figures (docs/dspark-plan.md sections 7 and 8).
pub const Pricing = struct {
    /// One block pass: the five layers' weight reads, the head slice and the exchanges.
    pass_ms: f64 = 6.0,
    /// A one-token target step (the plan's estimate, two ranks).
    step_ms: f64 = 33.0,
    /// Device bytes each stream holds on a rank (its ring and pass scratch): dspark_config.Cost.per_lane.
    stream_bytes: usize = 0,
};

/// Acceptance by draft depth until a stream has its own (UNVERIFIED guesses; the pack's taps are 17-23 % off).
pub const prior = [_]f64{ 0.7, 0.55, 0.45, 0.38, 0.32, 0.27, 0.23 };

/// A stream the draft has opened; its engine slot is known once the target's pass has begun.
const State = struct {
    tokens: [8]i32 = @splat(-1),
    n: u32 = 0,
    conf: [8]f32 = @splat(0),
    fresh: bool = false, // the last hold's results are not read yet
};

pub const Stats = struct { absorbs: u64 = 0, absorbed_rows: u64 = 0, passes: u64 = 0, drafted: u64 = 0, accepted: u64 = 0 };

pub const Draft = struct {
    gpa: std.mem.Allocator,
    cfg: dc.Config,
    lanes: *backend.Lanes,
    price: Pricing,
    costs: []lanes.config.Cost,
    streams: std.AutoHashMapUnmanaged(*lanes.Stream, State) = .empty,
    stats: Stats = .{},

    pub fn init(gpa: std.mem.Allocator, host: *backend.Lanes, cfg: dc.Config, price: Pricing) !*Draft {
        const d = try gpa.create(Draft);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .cfg = cfg, .lanes = host, .price = price, .costs = &.{} };
        d.costs = try windowCosts(gpa, price.step_ms);
        return d;
    }

    pub fn deinit(d: *Draft) void {
        std.log.info("dspark draft: {d} passes, {d} absorbs of {d} rows; the streams drafted {d} tokens and the target kept {d}", .{ d.stats.passes, d.stats.absorbs, d.stats.absorbed_rows, d.stats.drafted, d.stats.accepted });
        d.streams.deinit(d.gpa);
        d.gpa.free(d.costs);
        d.gpa.destroy(d);
    }

    /// The target's forward ms by window rows from the plan's byte model (dspark_config.windowCostRatio), 1 to 16 rows.
    pub fn windowCosts(gpa: std.mem.Allocator, step_ms: f64) ![]lanes.config.Cost {
        const out = try gpa.alloc(lanes.config.Cost, iface.window_rows);
        for (out, 1..) |*c, rows| c.* = .{ .width = @intCast(rows), .ms = step_ms * dc.windowCostRatio(8.8, 5.7, 128, 4, @intCast(rows)) };
        return out;
    }

    pub fn drafter(d: *Draft) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn of(p: *anyopaque) *Draft {
        return @ptrCast(@alignCast(p));
    }

    fn tapsFn(p: *anyopaque) []const u32 {
        const d = of(p);
        return d.cfg.taps[0..d.cfg.n_taps];
    }

    fn factsFn(p: *anyopaque) dr.Facts {
        const d = of(p);
        return .{
            .depth = d.cfg.block,
            .step_ms = d.price.pass_ms / @as(f64, @floatFromInt(d.cfg.block)),
            .prior = &prior,
            .plain_guard = true,
            .batched = false,
            .stream_bytes = d.price.stream_bytes,
            .window_costs = d.costs,
            .lane_costs = d.costs,
        };
    }

    fn openFn(p: *anyopaque, s: *lanes.Stream) anyerror!void {
        const d = of(p);
        if (d.streams.contains(s)) return error.StreamOpen;
        try d.streams.put(d.gpa, s, .{});
    }

    /// The engine slot of an opened stream: the target's pass has begun by the time the draft is asked for anything.
    fn slotOf(d: *Draft, s: *const lanes.Stream) !u32 {
        if (!d.streams.contains(@constCast(s))) return error.UnknownStream;
        const l = d.lanes.lanes.get(s) orelse return error.UnknownStream;
        return l.slot;
    }

    fn absorbFn(p: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = of(p);
        for (items) |it| {
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.rows == 0) return error.UnsupportedFeatures;
            if (it.rows != null) return error.TreesNotBuilt; // verify windows here are chains: a kept prefix
            if (f.row_bytes != d.cfg.fcIn() * 2 or f.offset % f.row_bytes != 0) return error.UnsupportedFeatures;
            const slot = try d.slotOf(it.stream);
            try d.lanes.x.bind(slot);
            try d.lanes.x.absorb(@intCast(it.start), f.rows, @intCast(f.offset / f.row_bytes));
            d.stats.absorbs += 1;
            d.stats.absorbed_rows += f.rows;
        }
    }

    fn holdFn(p: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = of(p);
        for (items) |it| {
            if (it.lanes != null) return error.TreesNotBuilt;
            if (it.depth == 0 or it.depth > d.cfg.block or it.position == 0) return error.BadHold;
            const slot = try d.slotOf(it.stream);
            const st = d.streams.getPtr(it.stream).?;
            try d.lanes.x.bind(slot);
            try d.lanes.x.draftPass(it.pending, @intCast(it.position - 1));
            st.fresh = true;
            st.n = it.depth;
            d.stats.passes += 1;
        }
    }

    fn heldFn(p: *anyopaque, streams: []const *lanes.Stream, out: []const []u32) anyerror!void {
        const d = of(p);
        for (streams, out) |s, o| {
            const st = d.streams.getPtr(s) orelse return error.UnknownStream;
            if (o.len > st.n) return error.TooFewDrafts;
            if (st.fresh) {
                try d.lanes.x.bind(try d.slotOf(s));
                var r: ops.DraftResult = .{};
                try d.lanes.x.e.draftResult(&r);
                st.tokens = r.tokens;
                st.conf = r.conf;
                st.fresh = false;
            }
            for (o, st.tokens[0..o.len]) |*t, v| t.* = if (v >= 0) @intCast(v) else (s.pending orelse 0);
        }
    }

    fn releaseFn(p: *anyopaque, s: *lanes.Stream) void {
        const d = of(p);
        if (d.streams.remove(s)) { // the stream's own count of drafts offered and kept (every kind of draft, copies too)
            d.stats.drafted += s.drafted;
            d.stats.accepted += s.accepted;
        }
    }
};

test "the draft's prior has one entry a drafted level and falls" {
    try std.testing.expectEqual(@as(usize, dc.flash.block), prior.len);
    for (prior[1..], prior[0 .. prior.len - 1]) |a, b| try std.testing.expect(a < b);
}

test "window costs grow with the rows by the plan's byte model and start at one step" {
    const c = try Draft.windowCosts(std.testing.allocator, 33.0);
    defer std.testing.allocator.free(c);
    try std.testing.expectEqual(@as(usize, iface.window_rows), c.len);
    try std.testing.expectEqual(@as(u32, 1), c[0].width);
    try std.testing.expectApproxEqAbs(@as(f64, 33.0), c[0].ms, 1e-9);
    try std.testing.expect(c[7].ms > 3.3 * 33.0 and c[7].ms < 3.5 * 33.0);
}
