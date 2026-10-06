//! DeepSeek-V4.1's CUDA kernels: the Python engine's .cu device code (zig/kernels/cuda/deepseek_v41, kept in sync by
//! its sync.py) as fatbins built with each extension's nvcc flags and embedded here, and their launches.
//!
//! The module is `dsv41_kernels` (zig/build/cuda.zig): it imports `cuda` and the fatbins; the family imports it.

const std = @import("std");
const cuda = @import("cuda");
const options = @import("dsv41_kernel_options");

pub const exl3 = @import("kernels_exl3.zig");
pub const dense = @import("kernels_dense.zig");
pub const replay = @import("kernels_replay.zig");
pub const ops = @import("kernels_ops.zig");

fn Blob(comptime import_name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(import_name).*;
    };
}

/// False in host-only builds (no nvcc, no prebuilt fatbins); every image is then empty.
pub const available = options.with_kernels;

/// The fatbins, by the names zig/build/cuda.zig gives them (dsv41_<file>).
pub const images = struct {
    pub const exl3_experts: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_exl3_experts").bytes else &.{};
    pub const x3ld: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_x3ld").bytes else &.{};
    pub const x3pf: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_x3pf").bytes else &.{};
    pub const x3gm: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_x3gm").bytes else &.{};
    pub const linear: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_linear").bytes else &.{};
    pub const x3seg: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_x3seg").bytes else &.{};
    pub const dense3: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_dense3").bytes else &.{};
    pub const attn: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_attn").bytes else &.{};
    pub const topk: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_topk").bytes else &.{};
    pub const mhc: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_mhc").bytes else &.{};
    pub const mhc_pf: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_mhc_pf").bytes else &.{};
    pub const router_gemv: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_router_gemv").bytes else &.{};
    pub const pfdense_k8: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_pfdense_k8").bytes else &.{};
    pub const pfdense_k10: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_pfdense_k10").bytes else &.{};
    pub const pfdense_k12: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_pfdense_k12").bytes else &.{};
    pub const pfdense_k16: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_pfdense_k16").bytes else &.{};
    pub const engram_gate: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_engram_gate").bytes else &.{};
    pub const l2pace: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_l2pace").bytes else &.{};
    pub const l2pf: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_l2pf").bytes else &.{};
};

pub const Kernels = struct {
    mods: [20]cuda.Module,
    exl3: exl3.Functions,
    dense: dense.Functions,
    ops: ops.Functions,
    sms: usize,

    /// Loads every module and resolves every instance the Python wrappers can launch.
    pub fn load(ctx: *const cuda.Context) !Kernels {
        if (!available) return error.BuiltWithoutKernels;
        const d = ctx.d;
        var k: Kernels = undefined;
        const list = [_][]const u8{ images.exl3_experts, images.x3ld, images.x3pf, images.x3gm, images.linear, images.x3seg, images.dense3, images.attn, images.topk, images.mhc, images.mhc_pf, images.router_gemv, images.pfdense_k8, images.pfdense_k10, images.pfdense_k12, images.pfdense_k16, images.engram_gate, images.l2pace, images.l2pf };
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        for (list, 0..) |img, i| {
            k.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        k.exl3 = try exl3.Functions.resolve(k.mods[0], k.mods[1], k.mods[2], k.mods[3]);
        k.dense = try dense.Functions.resolve(k.mods[4], k.mods[5], k.mods[6]);
        const m = k.mods;
        k.ops = try ops.Functions.resolve(m[7], m[8], m[9], m[10], m[11], .{ m[12], m[13], m[14], m[15] }, m[16], m[17], m[18]);
        k.sms = @intCast(try ctx.attribute(.multiprocessor_count));
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for (&k.mods) |*m| m.unload();
    }

    /// The routed-expert launches on one stream.
    pub fn experts(k: *const Kernels, s: cuda.Stream) exl3.Ops {
        return .{ .f = &k.exl3, .s = s, .sms = k.sms };
    }

    /// Attention, top-k, mHC, router, pfdense, Engram gate and L2 launches on one stream.
    pub fn others(k: *const Kernels, s: cuda.Stream) ops.Ops {
        return .{ .f = &k.ops, .s = s };
    }

    /// The dense-linear launches on one stream.
    pub fn linears(k: *const Kernels, s: cuda.Stream) dense.Ops {
        return .{ .f = &k.dense, .s = s };
    }
};

test {
    std.testing.refAllDecls(exl3);
    std.testing.refAllDecls(exl3.Ops);
    std.testing.refAllDecls(exl3.Functions);
    std.testing.refAllDecls(dense);
    std.testing.refAllDecls(dense.Ops);
    std.testing.refAllDecls(dense.Functions);
    std.testing.refAllDecls(Kernels);
    std.testing.refAllDecls(replay);
    std.testing.refAllDecls(ops);
    std.testing.refAllDecls(ops.Ops);
    std.testing.refAllDecls(ops.Functions);
}
