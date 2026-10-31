//! TF_DSV41_ROT_FOLD bit 1 (attn_rot.cuh): R1's split attention whose merge also writes wo_a's rot_in rows.

const std = @import("std");
const cuda = @import("cuda");
const ops = @import("kernels_ops.zig");
const fitShared = @import("cuda").qlinear.block.experts.fitShared;

const attn = ops.attn;

/// The attention kernels' head width (attn_cuda.cu D): a merge warp holds 128 of one head's columns.
pub const head_dim = 512;
pub const max_groups = 8;

/// attn_rot.cuh RotIn: wo_a group s's fp16 rows (row stride gk) and suh, s reading o's columns [s gk, (s + 1) gk).
pub const RotIn = extern struct { xh: [max_groups]u64 = @splat(0), suh: [max_groups]u64 = @splat(0), gk: c_int = 0 };

comptime {
    std.debug.assert(@sizeOf(RotIn) == 136 and @offsetOf(RotIn, "gk") == 128);
}

/// split_rot_kernel<4, ROWS, false, P>: the decode window's KW and no window ends; P 1 and 4 as attn.splitFor.
pub fn symbol(comptime rows: bool, comptime p: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN10dsv41_attn16split_rot_kernelILi4ELb{d}ELb0ELi{d}EEEvNS_4ArgsENS_5RotInE", .{ @intFromBool(rows), p });
}

/// The fold's instances in the attention image (resolved only with the knob: an older image still loads).
pub const Rot = struct {
    split: [2][2]cuda.Function, // [P 1, P 4][rows]

    pub fn resolve(m_attn: cuda.Module, optin: u32) !Rot {
        var r: Rot = undefined;
        inline for (.{ 1, 4 }, 0..) |p, pi| inline for (.{ false, true }, 0..) |rows, ri| {
            const name = symbol(rows, p);
            r.split[pi][ri] = try m_attn.function(name);
            if (!try fitShared(r.split[pi][ri], name, attn.splitSmem(p), optin)) return error.SharedMemoryTooSmall;
        };
        return r;
    }
};

/// One launch's RotIn for `heads` heads: group s's suh `suh[s]`, its rows at xh + 2 offs[s] (fp16 elements).
pub fn rotIn(heads: usize, suh: []const u64, gk: usize, xh: u64, offs: []const usize) !RotIn {
    // whole groups of whole Hadamard blocks: a merge warp's 128 columns are then one rot_in block of one group
    if (suh.len < 1 or suh.len > max_groups or offs.len != suh.len) return error.Unsupported;
    if (gk == 0 or gk % 128 != 0 or gk > std.math.maxInt(c_int) or suh.len * gk != heads * head_dim) return error.Unsupported;
    var r: RotIn = .{ .gk = @intCast(gk) };
    for (suh, offs, 0..) |s, off, i| {
        if (s == 0 or xh == 0) return error.Unsupported;
        r.xh[i] = xh + 2 * off;
        r.suh[i] = s;
    }
    return r;
}

/// A launch the fold admits: `heads` heads' columns covered by the set groups, a built (KW, window ends, P).
pub fn admits(ro: RotIn, heads: c_int, kw: u32, has_hi: bool, split: u32) bool {
    if (kw != 4 or has_hi or (split != 1 and split != 4) or ro.gk <= 0 or @mod(ro.gk, 128) != 0) return false;
    var groups: i64 = 0;
    for (ro.xh, ro.suh) |x, s| {
        if (x == 0 or s == 0) break;
        groups += 1;
    }
    return groups >= 1 and groups * ro.gk == @as(i64, heads) * head_dim;
}

/// ops.attention's split launch of split_rot_kernel: o as before, plus wo_a's rot_in rows `ro`.
pub fn launch(o: ops.Ops, a: attn.Args, ro: RotIn, kw: u32, rows: bool, has_hi: bool, split: u32) !void {
    const f = o.f.attn_rot orelse return error.RotFoldNotLoaded;
    _ = try attn.instance(a, kw, rows, has_hi);
    if (!admits(ro, a.H, kw, has_hi, split)) return error.Unsupported;
    var args: cuda.Args = .{};
    args.add(a);
    args.add(ro);
    const hb: usize = @intCast(@divTrunc(a.H, attn.BMQ));
    const R: usize = @intCast(a.R);
    const cluster: ?cuda.Dim3 = if (split == 4) .{ .x = 4 } else null;
    try o.go(f.split[@intFromBool(split == 4)][@intFromBool(rows)], .{ attn.NCH * split, hb, R }, attn.NT, attn.splitSmem(split), cluster, &args);
}

test "attention rot fold: symbols, RotIn's layout and the groups it admits" {
    const t = std.testing;
    try t.expectEqualStrings("_ZN10dsv41_attn16split_rot_kernelILi4ELb0ELb0ELi1EEEvNS_4ArgsENS_5RotInE", symbol(false, 1));
    try t.expectEqualStrings("_ZN10dsv41_attn16split_rot_kernelILi4ELb1ELb0ELi4EEEvNS_4ArgsENS_5RotInE", symbol(true, 4));
    // TP2's rank: 32 heads, 4 groups of 4,096 columns, rows of 3 at L.o.xh's fusedRot offsets
    const suh = [_]u64{ 0x1000, 0x3000, 0x5000, 0x7000 };
    const offs = [_]usize{ 0, 3 * 4096, 6 * 4096, 9 * 4096 };
    const ro = try rotIn(32, &suh, 4096, 0x100000, &offs);
    try t.expectEqual(@as(u64, 0x100000 + 2 * 6 * 4096), ro.xh[2]);
    try t.expectEqual(@as(u64, 0x5000), ro.suh[2]);
    try t.expectEqual(@as(u64, 0), ro.xh[4]);
    try t.expect(admits(ro, 32, 4, false, 1));
    try t.expect(admits(ro, 32, 4, false, 4));
    try t.expect(!admits(ro, 64, 4, false, 4)); // the groups cover only half of 64 heads' columns
    try t.expect(!admits(ro, 32, 2, false, 4)); // no KW 2 instance
    try t.expect(!admits(ro, 32, 4, true, 4)); // nor with window ends
    try t.expect(!admits(ro, 32, 4, false, 0)); // attn_kernel (no split) has no fold
    // groups that do not tile the heads' columns, a group off the 128-column grid, too many groups: refused
    try t.expectError(error.Unsupported, rotIn(32, suh[0..3], 4096, 0x100000, offs[0..3]));
    try t.expectError(error.Unsupported, rotIn(1, suh[0..1], 512 + 64, 0x100000, offs[0..1]));
    const nine: [9]u64 = @splat(0x1000);
    const at: [9]usize = @splat(0);
    try t.expectError(error.Unsupported, rotIn(72, &nine, 4096, 0x100000, &at));
    try t.expectError(error.Unsupported, rotIn(32, &suh, 4096, 0, &offs));
}
