//! The keyed sampler on the GPU forward (sampling_pipe.zig's steps over "w.logits"): topk_keys.cu for each rank's
//! candidates, sampling.cu for their packed rows and the nucleus statistics, the TP Collective for the gathers.
//! Rank 0 (the GPU target) sends each call's sampling on the plan link (`Forward.op_sample`) before its gathers;
//! the followers run the same steps from `Forward.follow` (their tokens are dropped).

const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("forward.zig");
const kops = @import("dsv41_kernels").ops;
const pipe = @import("sampling_pipe.zig");
const Gpu = @import("sampling_dev.zig").Gpu;

pub const Sampling = pipe.Sampling;
pub const Seg = pipe.Pipe.Seg;
pub const sampled = pipe.sampled;

/// TF_DSV41_SAMPLING=1 (every rank alike): model.zig makes the sampler; unset, sampled requests are refused as before.
pub fn enabled() bool {
    const v = std.c.getenv("TF_DSV41_SAMPLING") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// One rank's sampler over its forward (model.zig makes one on every rank; it hooks the follower side itself).
pub const GpuSampler = struct {
    gpa: std.mem.Allocator,
    f: *fwd.Forward,
    gpu: Gpu,
    p: pipe.Pipe,
    /// the call in flight (its rows, sampling and first draw position; null pos0: no device choice) for the
    /// download hook; the device choice's pinned copy, its event and the chosen tokens (Forward.after_sample)
    cur: ?Cur = null,
    pinned: ?cuda.HostBuffer = null,
    done: ?cuda.Event = null,
    chosen: ?cuda.DeviceBuffer = null,
    /// TF_DSV41_SAMP_BATCH=1 (rank 0 decides; followers run what op 13 brings): a forward's nucleus segments in one
    /// call (`chooseSegments`), as Python's one candidates gather + one statistics exchange a forward
    batch: bool = false,

    const Cur = struct { row0: u32, s: Sampling, pos0: ?u64 };

    pub fn init(gpa: std.mem.Allocator, f: *fwd.Forward) !*GpuSampler {
        const x = try gpa.create(GpuSampler);
        errdefer gpa.destroy(x);
        const W = f.comm.world();
        x.* = .{ .gpa = gpa, .f = f, .gpu = try Gpu.init(f.runner.d, f.runner.stream, f.runner.kernels), .p = undefined };
        errdefer x.gpu.deinit();
        x.p = try pipe.Pipe.init(gpa, f.comm, f.cfg.vocab / W, try pipe.nucleusEnv());
        f.sampler = .{ .ptr = x, .follow = follow };
        x.p.after_gather = .{ .ctx = x, .run = afterGather };
        x.batch = try batchEnv();
        return x;
    }

    /// TF_DSV41_SAMP_BATCH: 0 (default) | 1.
    pub fn batchEnv() !bool {
        const v = std.mem.span(std.c.getenv("TF_DSV41_SAMP_BATCH") orelse return false);
        if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
        if (std.mem.eql(u8, v, "1")) return true;
        return error.BadSampBatch;
    }

    /// Whether `chooseSegments` takes a segment under `s` (sampling_pipe.Pipe.batchable, with the knob on).
    pub fn batchable(x: *const GpuSampler, s: Sampling) bool {
        return x.batch and x.p.batchable(s);
    }

    /// The most segments one op_sample_many carries.
    pub const max_segments = 16;

    /// Rank 0: several nucleus segments of "w.logits" in one call (op 13 first: [op, n, (row0, rows, T, top_k, top_p,
    /// min_p) a segment]); no device start (nucleus rows have none).
    pub fn chooseSegments(x: *GpuSampler, segs: []const pipe.Pipe.Seg) !void {
        if (segs.len == 0 or segs.len > max_segments) return error.BadSegments;
        var msg: [2 + 6 * max_segments]i64 = undefined;
        msg[0] = fwd.Forward.op_sample_many;
        msg[1] = @intCast(segs.len);
        for (segs, 0..) |g, i| {
            const o = opOf(g.row0, g.n, g.s);
            @memcpy(msg[2 + 6 * i ..][0..6], o[1..7]);
        }
        try x.f.sendSample(msg[0 .. 2 + 6 * segs.len]);
        try x.runSegments(segs);
    }

    fn runSegments(x: *GpuSampler, segs: []const pipe.Pipe.Seg) !void {
        const f = x.f;
        const lg = f.runner.addressOf("w.logits") orelse return error.Unbound;
        if (f.grammar) |g| for (segs) |sg| try g.apply(g.ptr, sg.row0, sg.n);
        x.cur = null; // no device start
        try x.p.chooseSegments(x.gpu.device(), lg, segs);
        if (x.p.whole_rows > 0) std.log.scoped(.dsv41).debug("sampling: {d} rows taken whole ({d} segments)", .{ x.p.whole_rows, segs.len });
    }

    /// A follower: op_sample_many's segments (tokens dropped).
    fn followSegments(x: *GpuSampler, msg: []const i64) !void {
        if (msg.len < 2) return error.BadPlan;
        const ns: usize = @intCast(msg[1]);
        if (ns == 0 or ns > max_segments or msg.len != 2 + 6 * ns) return error.BadPlan;
        var segs: [max_segments]pipe.Pipe.Seg = undefined;
        var outs: [64]u32 = undefined;
        var used: usize = 0;
        for (0..ns) |i| {
            const w = msg[2 + 6 * i ..][0..6];
            const n: u32 = @intCast(w[1]);
            if (n == 0 or used + n > outs.len) return error.BadPlan;
            segs[i] = .{ .row0 = @intCast(w[0]), .n = n, .s = .{ .seed = 0, .temperature = @bitCast(w[2]), .top_k = @intCast(w[3]), .top_p = @bitCast(w[4]), .min_p = @bitCast(w[5]) }, .out = outs[used..][0..n] };
            used += n;
        }
        try x.runSegments(segs[0..ns]);
    }

    pub fn deinit(x: *GpuSampler) void {
        x.f.sampler = null;
        if (x.pinned) |*b| b.free();
        if (x.done) |*e| e.deinit();
        if (x.chosen) |*b| b.free();
        x.gpu.deinit();
        x.p.deinit();
        x.gpa.destroy(x);
    }

    /// Rank 0: the keyed choices of "w.logits" rows [row0, row0 + n) under `s` (T > 0), row i drawn at `draws[i]`.
    pub fn choose(x: *GpuSampler, row0: u32, n: u32, s: Sampling, draws: []const u64, out: []u32) !void {
        // a device start (Forward.after_sample) needs the seed and consecutive draws on every rank: op 12 carries them
        const pos0: ?u64 = if (x.f.after_sample != null and consecutive(draws)) draws[0] else null;
        if (pos0) |p0| try x.f.sendSample(&opDevice(row0, n, s, p0)) else try x.f.sendSample(&opOf(row0, n, s));
        try x.run(row0, n, s, draws, out, pos0);
    }

    fn consecutive(draws: []const u64) bool {
        if (draws.len == 0) return false;
        for (draws, 0..) |d, i| if (d != draws[0] + i) return false;
        return true;
    }

    fn run(x: *GpuSampler, row0: u32, n: u32, s: Sampling, draws: ?[]const u64, out: []u32, pos0: ?u64) !void {
        const f = x.f;
        const V: u64 = f.cfg.vocab / f.comm.world();
        const lg = (f.runner.addressOf("w.logits") orelse return error.Unbound) + 4 * V * row0;
        if (f.grammar) |g| try g.apply(g.ptr, row0, n);
        x.cur = .{ .row0 = row0, .s = s, .pos0 = pos0 };
        defer x.cur = null;
        try x.p.choose(x.gpu.device(), lg, n, s, draws, out);
        if (x.p.whole_rows > 0) std.log.scoped(.dsv41).debug("sampling: {d} of {d} rows taken whole", .{ x.p.whole_rows, n });
    }

    /// The plan-link message: [op, row0, n, T, top_k, top_p, min_p] (floats as their bits; the seed stays on rank 0).
    pub fn opOf(row0: u32, n: u32, s: Sampling) [7]i64 {
        return .{ fwd.Forward.op_sample, row0, n, @bitCast(s.temperature), s.top_k, @bitCast(s.top_p), @bitCast(s.min_p) };
    }

    /// The message with the device start's seed and first draw position: [op 12 ..., seed, pos0] (9 words).
    pub fn opDevice(row0: u32, n: u32, s: Sampling, pos0: u64) [9]i64 {
        const o = opOf(row0, n, s);
        return o ++ [2]i64{ @bitCast(s.seed), @intCast(pos0) };
    }

    /// A follower: the leader's call from its message (tokens dropped; with 9 words its device choice too).
    fn follow(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const x: *GpuSampler = @ptrCast(@alignCast(ptr));
        if (msg[0] == fwd.Forward.op_sample_many) return x.followSegments(msg);
        if (msg.len != 7 and msg.len != 9) return error.BadPlan;
        const n: u32 = @intCast(msg[2]);
        const dev = msg.len == 9;
        const s: Sampling = .{ .seed = if (dev) @bitCast(msg[7]) else 0, .temperature = @bitCast(msg[3]), .top_k = @intCast(msg[4]), .top_p = @bitCast(msg[5]), .min_p = @bitCast(msg[6]) };
        var out: [64]u32 = undefined;
        if (n == 0 or n > out.len) return error.BadPlan;
        try x.run(@intCast(msg[1]), n, s, null, out[0..n], if (dev) @intCast(msg[8]) else null);
    }

    /// sampling_pipe's download hook (every rank): with a device start wanted (Forward.after_sample, a non-nucleus
    /// call with its draw positions), the candidates' copy goes to pinned memory with an event, vsample.choose's
    /// device tokens follow (glue.cu vs_choose) and the pass's start is enqueued behind them; then the host waits for
    /// the copy alone and picks as before (its draws are the authority, unchanged). False: the pipe's own download.
    fn afterGather(ctx: *anyopaque, gathered: u64, n: usize, k: usize, count: usize, host: []u8) anyerror!bool {
        const x: *GpuSampler = @ptrCast(@alignCast(ctx));
        const c = x.cur orelse return false;
        const h = x.f.after_sample orelse return false;
        const p0 = c.pos0 orelse return false;
        const r = x.f.runner;
        const ops = r.kernels.others(r.stream);
        const W: usize = x.f.comm.world();
        if (ops.f.vs_choose == null or pipe.sampled(c.s) == false or isNucleusRow(c.s) or W * k > kops.glue.vs_max) return false;
        if (x.pinned == null or x.pinned.?.bytes.len < host.len) {
            if (x.pinned) |*b| b.free();
            x.pinned = null;
            x.pinned = try cuda.HostBuffer.alloc(r.d, @max(host.len, 1 << 16));
        }
        if (x.chosen == null or x.chosen.?.len < 8 * n) {
            if (x.chosen) |*b| b.free();
            x.chosen = null;
            x.chosen = try cuda.DeviceBuffer.alloc(r.d, @max(8 * n, 1024));
        }
        if (x.done == null) x.done = try cuda.Event.init(r.d, false);
        try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(x.pinned.?.bytes.ptr, gathered, host.len, r.stream.handle), "cuMemcpyDtoHAsync");
        try x.done.?.record(r.stream);
        try ops.vsChoose(gathered, W, n, k, count, c.s.seed, p0, c.s.top_k, c.s.temperature, c.s.top_p, c.s.min_p, x.chosen.?.ptr);
        try h.run(h.ctx, c.row0, @intCast(n), x.chosen.?.ptr);
        try x.done.?.synchronize();
        @memcpy(host, x.pinned.?.bytes[0..host.len]);
        return true;
    }

    fn isNucleusRow(s: Sampling) bool {
        return s.temperature > 0 and s.top_k == 0 and s.top_p > 0 and s.top_p < 1;
    }
};
