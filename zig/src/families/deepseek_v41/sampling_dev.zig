//! The keyed sampler's device steps on one stream (sampling_pipe.Device): topk_keys.cu (`dsv41_kernels`) for each
//! row's candidates, sampling.cu for their packed rows and the nucleus statistics.

const std = @import("std");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const pipe = @import("sampling_pipe.zig");

/// The device steps on one stream: topk_keys.cu (`dsv41_kernels`) and sampling.cu, scratch kept between calls.
pub const Gpu = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    kernels: *const dk.Kernels,
    fns: dk.sampling.Functions,
    bufs: [8]?cuda.DeviceBuffer = @splat(null),

    pub fn init(d: *const cuda.Driver, stream: cuda.Stream, kernels: *const dk.Kernels) !Gpu {
        return .{ .d = d, .stream = stream, .kernels = kernels, .fns = try dk.sampling.Functions.load(d) };
    }

    pub fn deinit(g: *Gpu) void {
        for (&g.bufs) |*b| if (b.*) |*x| x.free();
        g.fns.unload();
    }

    pub fn device(g: *Gpu) pipe.Device {
        return .{ .ptr = g, .vtable = &.{ .top = top, .stats = stats, .scratch = scratch, .download = download, .stream = streamOf } };
    }

    fn self_(p: *anyopaque) *Gpu {
        return @ptrCast(@alignCast(p));
    }

    fn top(p: *anyopaque, lg: u64, n: usize, C: usize, k: usize, id0: u32, out: u64) anyerror!void {
        const g = self_(p);
        const vals = try scratch(p, 5, 4 * n * k);
        const cols = try scratch(p, 6, 8 * n * k);
        try g.kernels.others(g.stream).topKeys(lg, C, n, C, k, vals, cols);
        try g.fns.pack(g.stream, vals, cols, n, k, id0, out);
    }

    fn stats(p: *anyopaque, lg: u64, n: usize, C: usize, t: f64, out: u64) anyerror!void {
        const g = self_(p);
        try g.fns.stats(g.stream, lg, C, n, C, t, out);
    }

    fn scratch(p: *anyopaque, slot: u32, bytes: usize) anyerror!u64 {
        const g = self_(p);
        const b = &g.bufs[slot];
        if (b.* == null or b.*.?.len < bytes) {
            if (b.*) |*x| x.free();
            b.* = null;
            b.* = try cuda.DeviceBuffer.alloc(g.d, bytes);
        }
        return b.*.?.ptr;
    }

    fn download(p: *anyopaque, src: u64, dst: []u8) anyerror!void {
        const g = self_(p);
        try g.stream.synchronize();
        try cuda.DeviceBuffer.download(.{ .d = g.d, .ptr = src, .len = dst.len }, 0, dst);
    }

    fn streamOf(p: *anyopaque) cuda.abi.Stream {
        return self_(p).stream.handle;
    }
};
