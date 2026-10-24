//! Prompt-sized dense EXL3 rows on the tensor units (exl3_pfd.cu): pfdense over the trellis as stored, mul1.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const exl3 = @import("exl3.zig");
const block = @import("exl3_block_exl3.zig");
const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// The widths built (2 * bits) and pfdense_args.h's configurations (BM, WM, WN, KS, NST) of the image.
pub const k2s = [_]u32{ 8, 10, 12, 16 };
pub const cfg_ids = [_]usize{ 0, 3, 6 };
pub const cfgs = [_][5]u32{ .{ 128, 2, 4, 2, 3 }, .{ 64, 2, 2, 2, 4 }, .{ 32, 1, 4, 2, 4 } };
/// Row tiles a raster group sweeps before the next column block (Triton _gemm's GROUP).
const group = 8;

/// pfdense_args.h Args, passed by value.
pub const Args = extern struct {
    xh: u64 = 0,
    T: u64 = 0,
    stride_k: i64 = 0,
    stride_nb: i64 = 0,
    svh: u64 = 0,
    bias: u64 = 0,
    out: u64 = 0,
    o_stride: i64 = 0,
    M: c_int = 0,
    K: c_int = 0,
    N: c_int = 0,
    group: c_int = 0,
    out_type: c_int = 0,
};

comptime {
    std.debug.assert(@offsetOf(Args, "M") == 64 and @sizeOf(Args) == 88);
}

pub fn threads(c: usize) u32 {
    return cfgs[c][1] * cfgs[c][2] * 32;
}

/// Lay<K2, false, Cfg>::SMEM.
pub fn smem(c: usize, k2: u32) u32 {
    const bm, const wm, _, const ks, const nst = cfgs[c];
    const ring = nst * (bm * (16 * ks) * 2 + ks * 8 * (4 * k2) * 4);
    return @max(ring + 2 * (ks * 8 * 32 * 16), (bm / wm) * 128 * 2 * 2);
}

fn symbol(comptime k2: u32, comptime c: usize) [:0]const u8 {
    const t = cfgs[c];
    return std.fmt.comptimePrint("_ZN9dsv41_pfd10pfd_kernelILi{d}ELb0ENS_3CfgILi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEEEvNS_4ArgsE", .{ k2, t[0], t[1], t[2], t[3], t[4] });
}

/// Whether pfdense keeps a dense weight's formula: mul1 whole widths it builds, stored tiles, whole 128 blocks.
pub fn eligible(w: exl3.Weight) bool {
    if (w.codebook != .mul1 or std.mem.indexOfScalar(u32, &k2s, w.halves) == null) return false;
    if (w.k < 128 or w.n < 128 or w.k % 128 != 0 or w.n % 128 != 0) return false;
    const st = exl3.strides(w.k, w.n, w.halves, false);
    return w.stride_k == st.k and w.stride_nb == st.nb;
}

/// pfdense.heuristic: the largest row tile whose last wave is not mostly empty (configurations 0, 3, 6).
pub fn heuristic(n: usize, k2: u32, m: usize, sms: usize) usize {
    var best: usize = 0;
    var best_eff: f64 = -1.0;
    for (0..cfgs.len) |c| {
        const blocks: usize = if (2 * (@as(usize, smem(c, k2)) + 1024) <= 102400) 2 else 1;
        const ctas = ((m + cfgs[c][0] - 1) / cfgs[c][0]) * (n / 128);
        const slots = sms * blocks;
        const eff = @as(f64, @floatFromInt(ctas)) / @as(f64, @floatFromInt(((ctas + slots - 1) / slots) * slots));
        if (eff > best_eff + 0.10) {
            best = c;
            best_eff = eff;
        }
    }
    return best;
}

fn deviceAttribute(d: *const Driver, a: anytype) !u32 {
    var dev: @import("abi.zig").Device = undefined;
    try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
    var v: c_int = 0;
    try d.check(d.api.cuDeviceGetAttribute(&v, a, dev), "cuDeviceGetAttribute");
    return @intCast(v);
}

pub const Dense = struct {
    mod: Module,
    fns: [k2s.len][cfgs.len]?Function,
    sms: usize,

    pub fn load(d: *const Driver) !Dense {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var p: Dense = .{ .mod = try Module.load(d, kernels.exl3_pfd), .fns = undefined, .sms = 0 };
        errdefer p.mod.unload();
        const optin = try deviceAttribute(d, .max_shared_memory_per_block_optin);
        p.sms = try deviceAttribute(d, .multiprocessor_count);
        inline for (k2s, 0..) |k2, ki| inline for (0..cfgs.len) |c| {
            const f = try p.mod.function(symbol(k2, c));
            p.fns[ki][c] = if (try block.fitShared(f, symbol(k2, c), smem(c, k2), optin)) f else null;
        };
        return p;
    }

    pub fn unload(p: *Dense) void {
        p.mod.unload();
    }

    /// y [m, n] = xh [m, k] (fp16, rotated as tf_exl3_rot_in rotates) through the weight, svh and bias in the epilogue.
    pub fn gemm(p: *const Dense, s: Stream, w: exl3.Weight, xh: u64, m: usize, y: u64, y_dtype: exl3.DType) !void {
        if (!eligible(w) or m < 1 or m > std.math.maxInt(c_int)) return error.Exl3Shape;
        const ki = std.mem.indexOfScalar(u32, &k2s, w.halves).?;
        var c = heuristic(w.n, w.halves, m, p.sms);
        if (p.fns[ki][c] == null) c = 0;
        const f = p.fns[ki][c] orelse return error.Unsupported;
        const a: Args = .{ .xh = xh, .T = w.trellis, .stride_k = @intCast(w.stride_k), .stride_nb = @intCast(w.stride_nb), .svh = w.svh, .bias = w.bias, .out = y, .o_stride = @intCast(w.n), .M = @intCast(m), .K = @intCast(w.k), .N = @intCast(w.n), .group = group, .out_type = @backingInt(y_dtype) };
        var args: launch_.Args = .{};
        args.add(a);
        const ctas = ((m + cfgs[c][0] - 1) / cfgs[c][0]) * (w.n / 128);
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(ctas) }, .block = .{ .x = threads(c) }, .shared = smem(c, w.halves) }, s, &args);
    }
};

test "the configurations' shared bytes and threads as pfdense_args.h" {
    try std.testing.expectEqual(@as(u32, 256), threads(0));
    try std.testing.expectEqual(@as(u32, 128), threads(1));
    try std.testing.expectEqual(@as(u32, 3 * (128 * 32 * 2 + 2 * 8 * 48 * 4) + 2 * (2 * 8 * 32 * 16)), smem(0, 12));
    try std.testing.expectEqualStrings("_ZN9dsv41_pfd10pfd_kernelILi12ELb0ENS_3CfgILi128ELi2ELi4ELi2ELi3EEEEEvNS_4ArgsE", symbol(12, 0));
}

test "pfdense takes stored mul1 widths it builds" {
    const st = exl3.strides(6144, 4096, 12, false);
    var w: exl3.Weight = .{ .trellis = 1, .suh = 2, .svh = 3, .k = 6144, .n = 4096, .halves = 12, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    try std.testing.expect(eligible(w));
    w.halves = 7; // a mul1 half step: not built
    try std.testing.expect(!eligible(w));
    w.halves = 12;
    w.codebook = .mcg;
    try std.testing.expect(!eligible(w));
    w.codebook = .mul1;
    w.stride_nb += 4; // strips or repacked: not the stored layout
    try std.testing.expect(!eligible(w));
}

test "the row tile keeps the last wave full" {
    try std.testing.expectEqual(@as(usize, 0), heuristic(4096, 12, 2048, 48)); // 16 x 32 = 512 CTAs: the big tile
    try std.testing.expectEqual(@as(usize, 1), heuristic(4096, 12, 100, 48)); // 32 CTAs of 128 rows fill a third
}
