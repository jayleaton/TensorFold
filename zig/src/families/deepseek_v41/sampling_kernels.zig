//! sampling.cu's launches (the `dsv41_kernels` module): a window's nucleus statistics and the candidates' packed rows,
//! its own fatbin and module (`dsv41_sampling`, loaded beside `Kernels`).

const std = @import("std");
const cuda = @import("cuda");
const options = @import("dsv41_kernel_options");

const image: []const u8 = if (options.with_kernels) &Blob.bytes else &.{};
const Blob = struct {
    pub const bytes align(16) = @embedFile("dsv41_fatbin_dsv41_sampling").*;
};

pub const Functions = struct {
    module: cuda.Module,
    stats_fn: cuda.Function,
    pack_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Functions {
        if (!options.with_kernels) return error.BuiltWithoutKernels;
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{
            .module = m,
            .stats_fn = try m.function("_ZN14dsv41_sampling12stats_kernelEPKfxidPd"),
            .pack_fn = try m.function("_ZN14dsv41_sampling11pack_kernelEPKfPKxiiPf"),
        };
    }

    pub fn unload(f: *Functions) void {
        f.module.unload();
    }

    /// nucleus.row_stats: float64 [rows, 2] at `out`, each row's (max, sum exp(v / t - max / t)) over its C columns.
    pub fn stats(f: *const Functions, s: cuda.Stream, lg: u64, ld: usize, rows: usize, C: usize, t: f64, out: u64) !void {
        if (rows < 1 or C < 1 or !(t > 0)) return error.Shape;
        var args: cuda.Args = .{};
        args.add(lg);
        args.add(@as(i64, @intCast(ld)));
        args.add(@as(c_int, @intCast(C)));
        args.add(t);
        args.add(out);
        try cuda.launch.launch(f.stats_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &args);
    }

    /// cand_gather's packed rows: fp32 [rows, 2k] at `out` from topk_keys' vals fp32 / cols int64 [rows, k].
    pub fn pack(f: *const Functions, s: cuda.Stream, vals: u64, cols: u64, rows: usize, k: usize, id0: u32, out: u64) !void {
        if (rows < 1 or k < 1) return error.Shape;
        var args: cuda.Args = .{};
        args.add(vals);
        args.add(cols);
        args.add(@as(c_int, @intCast(k)));
        args.add(@as(c_int, @intCast(id0)));
        args.add(out);
        try cuda.launch.launch(f.pack_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &args);
    }
};
