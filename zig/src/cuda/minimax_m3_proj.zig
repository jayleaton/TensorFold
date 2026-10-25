//! MiniMax-M3.1-Flash fp16-weight index projection and per-rank greedy pick (minimax_m3_proj.cu).
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const warps = 4; // outputs a block
pub const row_group = 4; // rows a warp
pub const chunk = 128; // the reduction length is a multiple: 32 lanes of 4 values
pub const threads = warps * 32;
pub const pick_threads = 256;
pub const max_grid_y = 65535;

/// The dtype of the projection's outputs: fp16 (`index_fp16`, the planner's `iq_raw`/`ik_raw`) or bf16.
pub const OutType = enum(i32) { f16 = 0, bf16 = 1 };

pub const Launch = struct { grid: [3]u32, block: u32 };
pub const Error = error{ Empty, TooManyRows, BadStride, Align };

/// One block per 4 outputs (of both matrices), one grid row per 4 rows.
pub fn projLaunch(rows: usize, outputs: usize) Error!Launch {
    if (rows == 0 or outputs == 0) return error.Empty;
    const gy = (rows + row_group - 1) / row_group;
    if (gy > max_grid_y) return error.TooManyRows;
    return .{ .grid = .{ @intCast((outputs + warps - 1) / warps), @intCast(gy), 1 }, .block = threads };
}

pub fn checkProj(k: usize, x_ld: usize) Error!void {
    if (k == 0 or k % chunk != 0 or x_ld < k or x_ld % 4 != 0) return error.BadStride;
}

/// One CTA a row; the padded row (`ld` logits) holds the rank's `n` first.
pub fn pickLaunch(rows: usize, n: usize, ld: usize) Error!Launch {
    if (rows == 0 or n == 0) return error.Empty;
    if (rows > std.math.maxInt(i32)) return error.TooManyRows;
    if (ld < n) return error.BadStride;
    return .{ .grid = .{ @intCast(rows), 1, 1 }, .block = pick_threads };
}

/// One matrix of the projection: fp16 weights [n, k] and the [rows, n] output at `out_ld` elements a row.
pub const Part = struct { w: u64 = 0, n: u32 = 0, out: u64 = 0, out_ld: usize = 0 };

pub const Proj = struct {
    mod: Module,
    bf16_fn: Function,
    f32_fn: Function,
    pick_fn: Function,

    pub fn load(d: *const Driver) !Proj {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var m = try Module.load(d, kernels.minimax_m3_proj);
        errdefer m.unload();
        return .{
            .mod = m,
            .bf16_fn = try m.function("tf_mm3_proj_bf16"),
            .f32_fn = try m.function("tf_mm3_proj_f32"),
            .pick_fn = try m.function("tf_mm3_pick"),
        };
    }

    pub fn unload(g: *Proj) void {
        g.mod.unload();
    }

    fn cfg(l: Launch) launch_.Config {
        return .{ .grid = .{ .x = l.grid[0], .y = l.grid[1], .z = l.grid[2] }, .block = .{ .x = l.block } };
    }

    /// out_i [rows, n_i] = x [rows, k] times fp16 w_i [n_i, k] for the index query `a` and key `b` (b.n may be 0).
    pub fn project(g: *const Proj, st: Stream, x: u64, x_ld: usize, x_f32: bool, k: u32, a: Part, b: Part, rows: usize, ot: OutType) !void {
        const launch = try projLaunch(rows, @as(usize, a.n) + b.n);
        try checkProj(k, x_ld);
        for ([_]Part{ a, b }) |p| {
            if (p.n == 0) continue;
            if (p.w == 0 or p.out == 0 or p.out_ld < p.n or p.w % 8 != 0 or p.out % 2 != 0) return error.BadStride;
        }
        if (x % (if (x_f32) @as(u64, 16) else 8) != 0) return error.Align;
        var args: launch_.Args = .{};
        args.add(x);
        args.add(@as(i64, @intCast(x_ld)));
        for ([_]Part{ a, b }) |p| {
            args.add(p.w);
            args.add(@as(i32, @intCast(p.n)));
            args.add(p.out);
            args.add(@as(i64, @intCast(p.out_ld)));
        }
        args.add(@as(i32, @intCast(rows)));
        args.add(@as(i32, @intCast(k)));
        args.add(@backingInt(ot));
        try launch_.launch(if (x_f32) g.f32_fn else g.bf16_fn, cfg(launch), st, &args);
    }

    /// pairs [rows] = (logit bits, id0 + index) of each padded row's best of `n`; ties to the lower id, NaN last.
    pub fn pick(g: *const Proj, st: Stream, logits: u64, ld: usize, n: u32, id0: u32, rows: usize, pairs: u64) !void {
        const launch = try pickLaunch(rows, n, ld);
        if (logits == 0 or pairs == 0 or logits % 4 != 0 or pairs % 4 != 0 or @as(u64, id0) + n > std.math.maxInt(i32)) return error.BadStride;
        var args: launch_.Args = .{};
        args.add(logits);
        args.add(@as(i64, @intCast(ld)));
        args.add(@as(i32, @intCast(n)));
        args.add(@as(i32, @intCast(id0)));
        args.add(pairs);
        try launch_.launch(g.pick_fn, cfg(launch), st, &args);
    }
};

test "launch geometry: 4 outputs a block, 4 rows a grid row, one CTA a pick row" {
    const l = try projLaunch(5, 384);
    try std.testing.expectEqual([3]u32{ 96, 2, 1 }, l.grid);
    try std.testing.expectEqual(@as(u32, 128), l.block);
    try std.testing.expectError(error.Empty, projLaunch(0, 384));
    try std.testing.expectError(error.Empty, projLaunch(1, 0));
    try std.testing.expectError(error.TooManyRows, projLaunch(4 * max_grid_y + 1, 128));
    try checkProj(6144, 6144);
    try std.testing.expectError(error.BadStride, checkProj(6000, 6000));
    try std.testing.expectError(error.BadStride, checkProj(6144, 6142));
    const p = try pickLaunch(7, 100_000, 100_096);
    try std.testing.expectEqual([3]u32{ 7, 1, 1 }, p.grid);
    try std.testing.expectError(error.BadStride, pickLaunch(1, 10, 9));
    try std.testing.expectError(error.Empty, pickLaunch(1, 0, 9));
}
