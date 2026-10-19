//! TF_DSV41_MHC_PF_FOLD (ours, mhc_pf_fold.cu): a prefill site's mhc_pf.run and its `_finish_k` in one launch.
const std = @import("std");
const cuda = @import("cuda");
const mhc_pf = @import("kernels_ops.zig").mhc_pf;
const fitShared = @import("kernels_exl3.zig").fitShared;

/// fold_kernel<POST, COLL, false, false> by mhc_pf's mode: 0 boundary, 1 site from stream 0, 2 with a carried pre-mix
pub const symbols = [3][:0]const u8{
    "_ZN14dsv41_mhc_fold11fold_kernelILb1ELi2ELb0ELb0EEEvNS_4ArgsE",
    "_ZN14dsv41_mhc_fold11fold_kernelILb0ELi1ELb0ELb0EEEvNS_4ArgsE",
    "_ZN14dsv41_mhc_fold11fold_kernelILb0ELi2ELb0ELb0EEEvNS_4ArgsE",
};
pub const NT = mhc_pf.NT;
pub const BM = mhc_pf.BM;
/// mhc_pf's Lay<false>::BYTES: the site's staging, which the finish reuses (its static 1,040 B come on top)
pub const smem: u32 = mhc_pf.smem(false);

/// dsv41_mhc_fold::Args: mhc_pf's Args, then _finish_k's (base / scale, the next coefficients, the norm, cnt).
pub const Args = extern struct {
    s: mhc_pf.Args = .{},
    base: u64 = 0,
    scale: u64 = 0,
    opre: u64 = 0,
    opost: u64 = 0,
    ocomb: u64 = 0,
    nw: u64 = 0,
    nwbf16: c_int = 0,
    _pad: c_int = 0,
    out: u64 = 0,
    /// int32 [ceil(R / 16)], zero before the first launch; the last CTA of a row tile resets its entry
    cnt: u64 = 0,
    eps: f32 = 0,
    hc_eps: f32 = 0,
    post_alpha: f32 = 0,
    iters: c_int = 0,
};

comptime {
    std.debug.assert(@offsetOf(Args, "base") == 120 and @offsetOf(Args, "nw") == 160 and @offsetOf(Args, "nwbf16") == 168);
    std.debug.assert(@offsetOf(Args, "out") == 176 and @offsetOf(Args, "cnt") == 184 and @offsetOf(Args, "eps") == 192);
    std.debug.assert(@offsetOf(Args, "iters") == 204 and @sizeOf(Args) == 208);
}

pub const Error = error{ BadKnob, Shape, Unsupported };

/// TF_DSV41_MHC_PF_FOLD: unset, empty or 0 off, 1 on; any other value refused.
pub fn enabled(v: ?[]const u8) Error!bool {
    const s = std.mem.trim(u8, v orelse return false, " ");
    if (s.len == 0 or std.mem.eql(u8, s, "0")) return false;
    if (std.mem.eql(u8, s, "1")) return true;
    std.log.warn("TF_DSV41_MHC_PF_FOLD={s}: 0 or 1", .{s});
    return error.BadKnob;
}

/// The counters a launch of R rows uses: one a row tile.
pub fn cntLen(R: usize) usize {
    return (R + BM - 1) / BM;
}

/// dispatch's grid: mhc_pf's (40 column blocks, row tiles of 16).
pub fn grid(R: usize) [3]usize {
    return .{ mhc_pf.NB, cntLen(R), 1 };
}

/// dispatch's checks, and the instances it has: bf16 fn, no taps, a counter buffer.
pub fn check(a: Args, mode: u32) Error!void {
    if (a.s.R < 1 or a.s.world < 1 or a.s.world > mhc_pf.MAXW or a.s.tap != 0 or a.cnt == 0) return error.Shape;
    if (mode > 2) return error.Unsupported;
}

pub const Fold = struct {
    module: cuda.Module,
    f: [3]cuda.Function,

    pub fn resolve(module: cuda.Module, optin: u32) !Fold {
        var r: Fold = .{ .module = module, .f = undefined };
        inline for (symbols, 0..) |name, i| {
            r.f[i] = try module.function(name);
            if (!try fitShared(r.f[i], name, smem, optin)) return error.SharedMemoryTooSmall;
        }
        return r;
    }
};

test "TF_DSV41_MHC_PF_FOLD: off unless 1, junk refused" {
    try std.testing.expect(!try enabled(null));
    try std.testing.expect(!try enabled(""));
    try std.testing.expect(!try enabled("0"));
    try std.testing.expect(!try enabled(" 0 "));
    try std.testing.expect(try enabled("1"));
    try std.testing.expect(try enabled(" 1"));
    for ([_][]const u8{ "2", "on", "true", "yes", "1,0", "-1" }) |v| try std.testing.expectError(error.BadKnob, enabled(v));
}

test "MHC_PF_FOLD Args: dsv41_mhc_fold::Args' layout (mhc_pf's 120 B, then the finish's, 208 B)" {
    try std.testing.expectEqual(@as(usize, 120), @sizeOf(mhc_pf.Args));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Args, "s"));
    const at = [_]struct { []const u8, usize }{
        .{ "base", 120 },   .{ "scale", 128 },      .{ "opre", 136 },  .{ "opost", 144 }, .{ "ocomb", 152 },
        .{ "nw", 160 },     .{ "nwbf16", 168 },     .{ "out", 176 },   .{ "cnt", 184 },   .{ "eps", 192 },
        .{ "hc_eps", 196 }, .{ "post_alpha", 200 }, .{ "iters", 204 },
    };
    inline for (at) |p| try std.testing.expectEqual(p[1], @offsetOf(Args, p[0]));
    try std.testing.expectEqual(@as(usize, 208), @sizeOf(Args));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(Args));
}

test "MHC_PF_FOLD symbols: fold_kernel's mangled instances in dispatch's mode order, mhc_pf's site_kernel pattern" {
    const want = [3]struct { post: u1, coll: u32 }{ .{ .post = 1, .coll = 2 }, .{ .post = 0, .coll = 1 }, .{ .post = 0, .coll = 2 } };
    inline for (want, 0..) |w, i| {
        const s = std.fmt.comptimePrint("_ZN14dsv41_mhc_fold11fold_kernelILb{d}ELi{d}ELb0ELb0EEEvNS_4ArgsE", .{ w.post, w.coll });
        try std.testing.expectEqualStrings(s, symbols[i]);
    }
    try std.testing.expectEqual(@as(u32, 47872), smem);
}

test "MHC_PF_FOLD grid and checks: mhc_pf's tiles, a counter a row tile, bf16 fn without taps only" {
    try std.testing.expectEqual([3]usize{ 40, 128, 1 }, grid(2048));
    try std.testing.expectEqual([3]usize{ 40, 17, 1 }, grid(257));
    try std.testing.expectEqual(@as(usize, 16), cntLen(256));
    var a: Args = .{ .s = .{ .R = 256 }, .cnt = 64 };
    for ([_]u32{ 0, 1, 2 }) |m| try check(a, m);
    try std.testing.expectError(error.Unsupported, check(a, 3));
    a.s.tap = 64;
    try std.testing.expectError(error.Shape, check(a, 0));
    a.s.tap = 0;
    a.cnt = 0;
    try std.testing.expectError(error.Shape, check(a, 0));
    a.cnt = 64;
    a.s.world = 9;
    try std.testing.expectError(error.Shape, check(a, 0));
}
