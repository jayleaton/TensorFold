//! TF_DSV41_PF_ROT2 / TF_DSV41_PF_WOROT (ours, pfrot.cu): rot_in for two projections at once, a GEMM's next rotation.
const std = @import("std");
const cuda = @import("cuda");
const pfd = @import("kernels_ops.zig").pfd;
const fitShared = @import("kernels_exl3.zig").fitShared;

/// pfdr_kernel's widths (the lanes form only, wo_a's)
pub const k2s = [_]u32{ 8, 10, 12 };

pub const rot2_symbol: [:0]const u8 = "_ZN11dsv41_pfrot11rot2_kernelENS_8Rot2ArgsE";

/// pfdr_kernel<K2, true, Cfg<BM, WM, WN, KS, NST>>(dsv41_pfd::Args, RotOut) at pfdense_args.h's configuration `cfg`
pub fn symbol(comptime k2: u32, comptime cfg: usize) [:0]const u8 {
    const c = pfd.cfgs[cfg];
    return std.fmt.comptimePrint("_ZN11dsv41_pfrot11pfdr_kernelILi{d}ELb1EN9dsv41_pfd3CfgILi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEEEvNS1_4ArgsENS_6RotOutE", .{ k2, c[0], c[1], c[2], c[3], c[4] });
}

/// dsv41_pfrot::Rot2Args: x [M, K] bf16 (x_bf16) or fp16, contiguous; xh0 / xh1 [M, K] fp16, the two rotations.
pub const Rot2Args = extern struct {
    x: u64 = 0,
    suh0: u64 = 0,
    suh1: u64 = 0,
    xh0: u64 = 0,
    xh1: u64 = 0,
    x_bf16: c_int = 0,
    M: c_int = 0,
    K: c_int = 0,
};

/// dsv41_pfrot::RotOut: the next projection's suh at this GEMM's output column 0, its fp16 input (row stride x_stride).
pub const RotOut = extern struct {
    suh: u64 = 0,
    xh: u64 = 0,
    x_stride: i64 = 0,
};

comptime {
    std.debug.assert(@offsetOf(Rot2Args, "x_bf16") == 40 and @offsetOf(Rot2Args, "K") == 48 and @sizeOf(Rot2Args) == 56);
    std.debug.assert(@offsetOf(RotOut, "x_stride") == 16 and @sizeOf(RotOut) == 24);
}

pub const Error = error{ BadKnob, Shape, Unsupported };

/// TF_DSV41_PF_ROT2 / TF_DSV41_PF_WOROT (`name`): unset, empty or 0 off, 1 on; any other value refused.
pub fn enabled(comptime name: []const u8, v: ?[]const u8) Error!bool {
    const s = std.mem.trim(u8, v orelse return false, " ");
    if (s.len == 0 or std.mem.eql(u8, s, "0")) return false;
    if (std.mem.eql(u8, s, "1")) return true;
    std.log.warn(name ++ "={s}: 0 or 1", .{s});
    return error.BadKnob;
}

/// rot_in_kernel's grid: a warp a (row, 128-column block), 4 warps a CTA.
pub fn rot2Grid(M: usize, K: usize) [3]usize {
    return .{ (K / 128 + 3) / 4, M, 1 };
}

/// rot2's checks: whole 128-column blocks, 8-byte aligned rows (its uint2 loads and stores).
pub fn rot2Check(a: Rot2Args) Error!void {
    if (a.M < 1 or a.K < 128 or @rem(a.K, 128) != 0) return error.Shape;
    for ([_]u64{ a.x, a.suh0, a.suh1, a.xh0, a.xh1 }) |p| if (p == 0 or p % 8 != 0) return error.Shape;
}

/// gemm_rot's own checks on top of pfdense's: lanes, a bf16 output, a built width, the rotation's aligned views.
pub fn gemmRotCheck(a: pfd.Args, r: RotOut, k2: u32, lanes: bool) Error!void {
    if (!lanes or a.out_type != @backingInt(pfd.OutType.bf16) or std.mem.indexOfScalar(u32, &k2s, k2) == null) return error.Unsupported;
    if (@rem(a.o_stride, 4) != 0 or r.x_stride < a.N or @rem(r.x_stride, 4) != 0) return error.Shape;
    for ([_]u64{ a.out, r.suh, r.xh }) |p| if (p == 0 or p % 8 != 0) return error.Shape;
}

pub const PfRot = struct {
    module: cuda.Module,
    rot2: cuda.Function,
    /// [k2][cfg]: null where Lay<K2, true, C>::SMEM does not fit this GPU (as pfd's)
    pfdr: [k2s.len][pfd.cfgs.len]?cuda.Function,

    pub fn resolve(module: cuda.Module, optin: u32) !PfRot {
        var r: PfRot = .{ .module = module, .rot2 = try module.function(rot2_symbol), .pfdr = undefined };
        inline for (k2s, 0..) |k2, ki| inline for (0..pfd.cfgs.len) |c| {
            const fun = try module.function(symbol(k2, c));
            r.pfdr[ki][c] = if (try fitShared(fun, symbol(k2, c), pfd.smem(c, k2), optin)) fun else null;
        };
        return r;
    }
};

test "TF_DSV41_PF_ROT2 / _WOROT: off unless 1, junk refused" {
    inline for (.{ "TF_DSV41_PF_ROT2", "TF_DSV41_PF_WOROT" }) |n| {
        try std.testing.expect(!try enabled(n, null));
        try std.testing.expect(!try enabled(n, ""));
        try std.testing.expect(!try enabled(n, "0"));
        try std.testing.expect(!try enabled(n, " 0 "));
        try std.testing.expect(try enabled(n, "1"));
        try std.testing.expect(try enabled(n, " 1"));
        for ([_][]const u8{ "2", "on", "true", "yes", "1,0", "-1" }) |v| try std.testing.expectError(error.BadKnob, enabled(n, v));
    }
}

test "PF_ROT Args: Rot2Args' and RotOut's layouts (56 B, 24 B; dsv41_pfd::Args 88 B before RotOut)" {
    const at = [_]struct { []const u8, usize }{
        .{ "x", 0 },       .{ "suh0", 8 }, .{ "suh1", 16 }, .{ "xh0", 24 }, .{ "xh1", 32 },
        .{ "x_bf16", 40 }, .{ "M", 44 },   .{ "K", 48 },
    };
    inline for (at) |p| try std.testing.expectEqual(p[1], @offsetOf(Rot2Args, p[0]));
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(Rot2Args));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(Rot2Args));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(RotOut, "suh"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(RotOut, "xh"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(RotOut, "x_stride"));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(RotOut));
    try std.testing.expectEqual(@as(usize, 88), @sizeOf(pfd.Args));
}

test "PF_ROT symbols: rot2_kernel, pfdr_kernel at K2 8 / 10 / 12 and every pfdense configuration, pfd's shared memory" {
    try std.testing.expectEqualStrings("_ZN11dsv41_pfrot11pfdr_kernelILi8ELb1EN9dsv41_pfd3CfgILi128ELi2ELi4ELi2ELi3EEEEEvNS1_4ArgsENS_6RotOutE", symbol(8, 0));
    try std.testing.expectEqualStrings("_ZN11dsv41_pfrot11pfdr_kernelILi10ELb1EN9dsv41_pfd3CfgILi32ELi1ELi4ELi2ELi4EEEEEvNS1_4ArgsENS_6RotOutE", symbol(10, 6));
    try std.testing.expectEqualStrings("_ZN11dsv41_pfrot11pfdr_kernelILi12ELb1EN9dsv41_pfd3CfgILi256ELi4ELi2ELi2ELi3EEEEEvNS1_4ArgsENS_6RotOutE", symbol(12, 7));
    inline for (k2s) |k2| inline for (0..pfd.cfgs.len) |c| {
        const s = symbol(k2, c);
        try std.testing.expect(std.mem.indexOf(u8, s, std.fmt.comptimePrint("pfdr_kernelILi{d}ELb1E", .{k2})) != null);
        try std.testing.expect(std.mem.endsWith(u8, s, "EEEEEvNS1_4ArgsENS_6RotOutE"));
    };
    try std.testing.expectEqual(@as(u32, 3 * 10752 + 16384), pfd.smem(0, 10));
}

test "PF_ROT grid and checks: rot_in's grid, aligned 128-column rows, lanes bf16 widths 8-12 only" {
    try std.testing.expectEqual([3]usize{ 10, 4096, 1 }, rot2Grid(4096, 5120));
    try std.testing.expectEqual([3]usize{ 1, 7, 1 }, rot2Grid(7, 128));
    var r2: Rot2Args = .{ .x = 64, .suh0 = 64, .suh1 = 64, .xh0 = 64, .xh1 = 64, .x_bf16 = 1, .M = 16, .K = 5120 };
    try rot2Check(r2);
    r2.K = 5000;
    try std.testing.expectError(error.Shape, rot2Check(r2));
    r2.K = 5120;
    r2.xh1 = 0;
    try std.testing.expectError(error.Shape, rot2Check(r2));
    const a: pfd.Args = .{ .out = 64, .o_stride = 4096, .M = 16, .K = 1024, .N = 1024, .group = 8, .out_type = 1 };
    const r: RotOut = .{ .suh = 64, .xh = 128, .x_stride = 4096 };
    try gemmRotCheck(a, r, 10, true);
    try std.testing.expectError(error.Unsupported, gemmRotCheck(a, r, 10, false));
    try std.testing.expectError(error.Unsupported, gemmRotCheck(a, r, 16, true));
    var a16 = a;
    a16.out_type = 0;
    try std.testing.expectError(error.Unsupported, gemmRotCheck(a16, r, 10, true));
    var narrow = r;
    narrow.x_stride = 512;
    try std.testing.expectError(error.Shape, gemmRotCheck(a, narrow, 10, true));
}
