//! The keyed sampler on the GPU forward (sampling_pipe.zig's steps over "w.logits"): topk_keys.cu for each rank's
//! candidates, sampling.cu for their packed rows and the nucleus statistics, the TP Collective for the gathers.
//! Rank 0 (the GPU target) sends each call's sampling on the plan link (`Forward.op_sample`) before its gathers;
//! the followers run the same steps from `Forward.follow` (their tokens are dropped).

const std = @import("std");
const fwd = @import("forward.zig");
const pipe = @import("sampling_pipe.zig");
const Gpu = @import("sampling_dev.zig").Gpu;

pub const Sampling = pipe.Sampling;
pub const sampled = pipe.sampled;

/// One rank's sampler over its forward (model.zig makes one on every rank; it hooks the follower side itself).
pub const GpuSampler = struct {
    gpa: std.mem.Allocator,
    f: *fwd.Forward,
    gpu: Gpu,
    p: pipe.Pipe,

    pub fn init(gpa: std.mem.Allocator, f: *fwd.Forward) !*GpuSampler {
        const x = try gpa.create(GpuSampler);
        errdefer gpa.destroy(x);
        const W = f.comm.world();
        x.* = .{ .gpa = gpa, .f = f, .gpu = try Gpu.init(f.runner.d, f.runner.stream, f.runner.kernels), .p = undefined };
        errdefer x.gpu.deinit();
        x.p = try pipe.Pipe.init(gpa, f.comm, f.cfg.vocab / W, try pipe.nucleusEnv());
        f.sampler = .{ .ptr = x, .follow = follow };
        return x;
    }

    pub fn deinit(x: *GpuSampler) void {
        x.f.sampler = null;
        x.gpu.deinit();
        x.p.deinit();
        x.gpa.destroy(x);
    }

    /// Rank 0: the keyed choices of "w.logits" rows [row0, row0 + n) under `s` (T > 0), row i drawn at `draws[i]`.
    pub fn choose(x: *GpuSampler, row0: u32, n: u32, s: Sampling, draws: []const u64, out: []u32) !void {
        try x.f.sendSample(&opOf(row0, n, s));
        try x.run(row0, n, s, draws, out);
    }

    fn run(x: *GpuSampler, row0: u32, n: u32, s: Sampling, draws: ?[]const u64, out: []u32) !void {
        const f = x.f;
        const V: u64 = f.cfg.vocab / f.comm.world();
        const lg = (f.runner.addressOf("w.logits") orelse return error.Unbound) + 4 * V * row0;
        try x.p.choose(x.gpu.device(), lg, n, s, draws, out);
        if (x.p.whole_rows > 0) std.log.scoped(.dsv41).debug("sampling: {d} of {d} rows taken whole", .{ x.p.whole_rows, n });
    }

    /// The plan-link message: [op, row0, n, T, top_k, top_p, min_p] (floats as their bits; the seed stays on rank 0).
    pub fn opOf(row0: u32, n: u32, s: Sampling) [7]i64 {
        return .{ fwd.Forward.op_sample, row0, n, @bitCast(s.temperature), s.top_k, @bitCast(s.top_p), @bitCast(s.min_p) };
    }

    /// A follower: the leader's call from its message (tokens dropped).
    fn follow(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const x: *GpuSampler = @ptrCast(@alignCast(ptr));
        if (msg.len != 7) return error.BadPlan;
        const n: u32 = @intCast(msg[2]);
        const s: Sampling = .{ .seed = 0, .temperature = @bitCast(msg[3]), .top_k = @intCast(msg[4]), .top_p = @bitCast(msg[5]), .min_p = @bitCast(msg[6]) };
        var out: [64]u32 = undefined;
        if (n == 0 or n > out.len) return error.BadPlan;
        try x.run(@intCast(msg[1]), n, s, null, out[0..n]);
    }
};
