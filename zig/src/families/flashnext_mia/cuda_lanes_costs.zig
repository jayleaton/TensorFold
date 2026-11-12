//! Flash Next lanes' cost tables: verify windows, head levels and shared rounds timed on the engine itself.
const std = @import("std");
const cl = @import("cuda_lanes.zig");
const lanes_core = @import("cuda_lanes_core.zig");

const max_rows = cl.max_rows;
const max_drafts = cl.max_drafts;
const max_streams = cl.max_streams;
const shared_points = cl.shared_points;

/// Every cost table width's timing: untimed rounds of the timed shape first, then the median of the timed ones.
pub const shared_warm = 4;
pub const shared_reps = 9;
/// Passes over every width; each width keeps its least median (a pass slowed by boot noise does not price it).
pub const shared_passes = 2;

/// One pass at one width: `shared_warm` untimed rounds, then the median of `shared_reps` timed ones (`round.ms`).
pub fn passMs(round: anytype) !f64 {
    for (0..shared_warm) |_| _ = try round.ms(false);
    var times: [shared_reps]f64 = undefined;
    for (&times) |*x| x.* = try round.ms(true);
    return median(&times);
}

/// Every stream's `w`-row window, then `levels` head drafts (a shared round: `w - 1`), as `passMs` runs it.
fn WindowRound(comptime E: type) type {
    return struct {
        c: *Costing(E),
        w: usize,
        levels: usize,
        text: []const u32,
        io: std.Io,

        fn ms(s: @This(), timed: bool) !f64 {
            var out: f64 = 0;
            try s.c.round(s.w, s.text, if (timed) .{ .io = s.io, .out = &out } else null, s.levels);
            return out;
        }
    };
}

/// One stream's head chaining `levels` drafts, as `passMs` runs it (every chain is timed; warm-ups drop theirs).
fn ChainRound(comptime E: type) type {
    return struct {
        c: *Costing(E),
        levels: usize,
        io: std.Io,

        fn ms(s: @This(), _: bool) !f64 {
            return s.c.chain(s.io, s.levels);
        }
    };
}

pub fn Measure(comptime E: type) type {
    return struct {
        const Self = lanes_core.Lanes(E);

        // -- costs ----------------------------------------------------------------------------------------------

        /// Times verify windows, head levels and shared rounds in ms on `ids` (ranks in step) for the cost tables.
        pub fn measure(self: *Self, io: std.Io, ids: []const u32) !void {
            if (!self.drafting) return;
            const e = self.e;
            if (ids.len < 3 * max_rows) return error.CostTextTooShort;
            try self.settle();
            const top = self.window;
            // 19 prompt rows: a multiple of 16 asks Triton for an M % 16 variant a capture may not hold
            const lead = max_rows + 3;
            const cont = ids[lead..];
            var c = try Costing(E).init(self.gpa, e, @min(self.context, 4096), 1, ids[0..lead]);
            defer c.deinit();
            for (0..32) |_| try c.round(1, cont, null, 0); // the GPU at its working clocks before any window is timed
            // each pass times every width and level; a width keeps its least median, as the shared rounds below
            var verify_all: [max_rows + 1]f64 = @splat(0);
            const verify = verify_all[0 .. top + 1];
            var chain: [2]f64 = undefined;
            const deep = @max(2, @min(8, top - 1));
            for (0..shared_passes) |pass| {
                for (1..top + 1) |w| {
                    const got = try passMs(WindowRound(E){ .c = &c, .w = w, .levels = 0, .text = cont, .io = io });
                    verify[w] = if (pass == 0) got else @min(verify[w], got);
                }
                for ([_]usize{ 1, deep }, &chain) |levels, *out| {
                    const got = try passMs(ChainRound(E){ .c = &c, .levels = levels, .io = io });
                    out.* = if (pass == 0) got else @min(out.*, got);
                }
            }
            // one regime over every width: f212's pooling as hers at db281878 (no regime boundary)
            const k = @import("core").draft_depth.Costs.measured(verify, (chain[1] - chain[0]) / @as(f64, @floatFromInt(deep - 1)), std.math.maxInt(usize));
            self.cost_count = 0;
            for (1..k.rows + 1) |w| {
                self.costs[self.cost_count] = .{ .width = @intCast(w), .ms = k.verify[w] };
                self.cost_count += 1;
            }
            self.mtp_ms = k.level;
            // shared rounds: n streams' full windows in one forward; each pass times every width, the least median kept
            self.shared_count = 0;
            for (0..shared_passes) |pass| {
                var i: usize = 0;
                var n: usize = 2;
                while (n <= self.streams and i < shared_points) : (n = if (n * 2 > self.streams and n < self.streams) self.streams else n * 2) {
                    var many = try Costing(E).init(self.gpa, e, @min(self.context, 2048), n, ids[0..lead]);
                    defer many.deinit();
                    // a whole shared round: forward, draws and the batched head's drafts (the allocator weighs rows)
                    const got = try passMs(WindowRound(E){ .c = &many, .w = top, .levels = top - 1, .text = cont, .io = io });
                    if (pass == 0) self.shared[i] = .{ .width = @intCast(n * top), .ms = got } else self.shared[i].ms = @min(self.shared[i].ms, got);
                    i += 1;
                }
                self.shared_count = i;
            }
        }

        /// The cost tables in one line (rank 0 logs it at boot), `source` measured or kept.
        pub fn logCosts(self: *const Self, source: []const u8) void {
            var buf: [1024]u8 = undefined;
            var n: usize = 0;
            n += (std.fmt.bufPrint(buf[n..], "lanes costs ({s}): verify ", .{source}) catch return).len;
            for (self.costs[0..self.cost_count], 0..) |c, i| {
                n += (std.fmt.bufPrint(buf[n..], "{s}{d}:{d:.1}", .{ if (i == 0) "" else ",", c.width, c.ms }) catch return).len;
            }
            n += (std.fmt.bufPrint(buf[n..], "; level {d:.2} ms; shared ", .{self.mtp_ms}) catch return).len;
            for (self.shared[0..self.shared_count], 0..) |c, i| {
                n += (std.fmt.bufPrint(buf[n..], "{s}{d}:{d:.1}", .{ if (i == 0) "" else ",", c.width, c.ms }) catch return).len;
            }
            std.log.info("{s}", .{buf[0..n]});
        }
    };
}

/// The cost measurement's sequences and rounds: `n` streams each prefilled with `lead` and its head in step.
fn Costing(comptime E: type) type {
    return struct {
        const Self = @This();
        const SeqPtr = @typeInfo(@typeInfo(@TypeOf(E.newSeq)).@"fn".return_type.?).error_union.payload;
        e: *E,
        seqs: [max_streams]SeqPtr = undefined,
        pending: [max_streams]u32 = undefined,
        n: usize = 0,

        fn init(gpa: std.mem.Allocator, e: *E, limit: usize, n: usize, lead: []const u32) !Self {
            _ = gpa;
            var c: Self = .{ .e = e };
            errdefer c.deinit();
            for (0..n) |i| {
                c.seqs[i] = try e.newSeq(limit);
                c.n += 1;
                e.bind(c.seqs[i]);
                if (@hasDecl(E, "setSampling")) e.setSampling(null);
                c.pending[i] = try e.prefill(lead, null);
            }
            var reqs: [max_streams]E.DraftReq = undefined;
            var firsts: [max_streams][1]u32 = undefined;
            var none: [1]u32 = undefined;
            for (0..n) |i| {
                firsts[i] = .{c.pending[i]};
                reqs[i] = .{ .seq = c.seqs[i], .source = .prefill_last, .next = &firsts[i], .count = 0, .out = &none };
            }
            try e.draftMany(reqs[0..n]);
            return c;
        }

        fn deinit(c: *Self) void {
            for (c.seqs[0..c.n]) |q| c.e.freeSeq(q);
            c.n = 0;
        }

        const Timing = struct { io: std.Io, out: *f64 };

        /// One round of every stream's `w`-row window, first row kept and absorbed, then `levels` drafts, timed.
        fn round(c: *Self, w: usize, text: []const u32, timing: ?Timing, levels: usize) !void {
            const e = c.e;
            var ids: [max_streams][max_rows]u32 = undefined;
            var wins: [max_streams]E.Window = undefined;
            for (0..c.n) |i| {
                ids[i][0] = c.pending[i];
                @memcpy(ids[i][1..w], text[0 .. w - 1]);
                wins[i] = .{ .seq = c.seqs[i], .tokens = ids[i][0..w] };
            }
            var sampled: [max_streams][max_rows]u32 = undefined;
            const t0 = if (timing) |t| std.Io.Timestamp.now(t.io, .awake) else undefined;
            try e.verifyMany(wins[0..c.n]);
            for (0..c.n) |i| try e.sampleWindow(i, null, sampled[i][0..w]);
            if (timing != null and levels == 0) timing.?.out.* = msSince(timing.?.io, t0);
            var reqs: [max_streams]E.DraftReq = undefined;
            var drafts: [max_streams][max_drafts]u32 = undefined;
            for (0..c.n) |i| {
                try e.commitWindow(i, 1);
                c.pending[i] = sampled[i][0];
                reqs[i] = .{ .seq = c.seqs[i], .source = .{ .window = i }, .next = sampled[i][0..1], .count = levels, .confidence = 0, .out = &drafts[i] };
            }
            try e.draftMany(reqs[0..c.n]);
            if (timing != null and levels > 0) timing.?.out.* = msSince(timing.?.io, t0);
        }

        /// One stream's kept row, then its head chaining `levels` drafts (greedy, confidence 0), timed.
        fn chain(c: *Self, io: std.Io, levels: usize) !f64 {
            const e = c.e;
            var ids = [1]u32{c.pending[0]};
            var sampled: [1]u32 = undefined;
            try e.verifyMany(&.{.{ .seq = c.seqs[0], .tokens = &ids }});
            try e.sampleWindow(0, null, &sampled);
            try e.commitWindow(0, 1);
            c.pending[0] = sampled[0];
            var drafts: [max_drafts]u32 = undefined;
            var reqs = [1]E.DraftReq{.{ .seq = c.seqs[0], .source = .{ .window = 0 }, .next = &sampled, .count = levels, .confidence = 0, .out = &drafts }};
            const t0 = std.Io.Timestamp.now(io, .awake);
            try e.draftMany(&reqs);
            return msSince(io, t0);
        }
    };
}

fn msSince(io: std.Io, t0: std.Io.Timestamp) f64 {
    const d = t0.durationTo(std.Io.Timestamp.now(io, .awake));
    return @as(f64, @floatFromInt(d.nanoseconds)) / 1e6;
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return if (xs.len % 2 == 1) xs[xs.len / 2] else (xs[xs.len / 2 - 1] + xs[xs.len / 2]) / 2;
}

test "shared round costs: a width's slow first rounds and one outlier rep leave its steady ms" {
    // a fake clock: a width's first 5 rounds run slow (190 ms), its 10th is an outlier (500 ms), the rest take 125 ms
    const Spiky = struct {
        calls: *usize,
        fn ms(f: @This(), _: bool) !f64 {
            defer f.calls.* += 1;
            return if (f.calls.* < 5) 190 else if (f.calls.* == 9) 500 else 125;
        }
    };
    var calls: usize = 0;
    try std.testing.expectEqual(@as(f64, 125), try passMs(Spiky{ .calls = &calls }));
    try std.testing.expectEqual(@as(usize, shared_warm + shared_reps), calls);
}
