//! TF_DSV41_ROT_FOLD (dense3_rot.cuh): dense3 linears whose epilogue also writes the next projection's rot_in rows.

const std = @import("std");
const cuda = @import("root.zig");
const dense = @import("exl3_block_dense.zig");

const dense3_k2s = dense.dense3_k2s;
const max_seg = dense.max_seg;

/// dense3_rot.cuh RotOut: each segment's rotated rows (fp16, row stride `ld`) and suh, at its first output column.
pub const RotOut = extern struct { xh: [max_seg]u64 = @splat(0), suh: [max_seg]u64 = @splat(0), ld: c_int = 0 };

comptime {
    std.debug.assert(@sizeOf(RotOut) == 136 and @offsetOf(RotOut, "ld") == 128);
}

/// lanes_rot_kernel's (WK, MINB) instances: the two block.fusedNorm launches (WK 4 at MINB 3, WK 8 at MINB 1).
pub const rot_cfgs = [_][2]u32{ .{ 4, 3 }, .{ 8, 1 } };

fn rotSymbol(comptime k2: u32, comptime wk: u32, comptime mb: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN12dsv41_dense316lanes_rot_kernelILi{d}ELi2ELi{d}ELi{d}EEEvN11dsv41_x3seg8LinTableEiiNS_6RotOutE", .{ k2, wk, mb });
}

/// The fold's instances in the dense3 image (resolved only with the knob: an older dense3 fatbin still loads).
pub const Rot = struct {
    lanes: [dense3_k2s.len][rot_cfgs.len]cuda.Function,

    pub fn resolve(dense3: cuda.Module) !Rot {
        var r: Rot = undefined;
        inline for (dense3_k2s, 0..) |k2, i| inline for (rot_cfgs, 0..) |c, j| {
            r.lanes[i][j] = try dense3.function(rotSymbol(k2, c[0], c[1]));
        };
        return r;
    }
};

/// TF_DSV41_ROT_FOLD's mask: unset / empty / 0 none, 1-7 (bit 0 wo_a -> wo_b, bits 1-2 later stages); else refused
pub fn rotFoldMask(v: ?[]const u8) !u3 {
    const t = std.mem.trim(u8, v orelse "", " \t\r\n");
    if (t.len == 0) return 0;
    if (t.len != 1 or t[0] < '0' or t[0] > '7') return error.BadRotFold;
    return @intCast(t[0] - '0');
}

/// One launch's RotOut: segment j's rows at column `cols[j]` of `xh` (fp16, row stride `ld`) and of `suh`.
pub fn rotOut(cols: []const usize, xh: u64, suh: u64, ld: usize) !RotOut {
    // whole Hadamard blocks: rot_in transforms 128-column blocks of the row, a linear warp 128 output columns
    if (cols.len < 1 or cols.len > max_seg or ld == 0 or ld % 128 != 0 or ld > std.math.maxInt(c_int)) return error.Unsupported;
    var r: RotOut = .{ .ld = @intCast(ld) };
    for (cols, 0..) |c, j| {
        if (c % 128 != 0 or c >= ld) return error.Unsupported;
        r.xh[j] = xh + 2 * c;
        r.suh[j] = suh + 2 * c;
    }
    return r;
}

/// The output columns of a layout launch's segments, in its table's order.
pub fn launchCols(l: dense.Launch, segs: []const dense.Seg) [max_seg]usize {
    var out: [max_seg]usize = @splat(0);
    for (l.segs[0..l.count], 0..) |i, j| out[j] = segs[i].col;
    return out;
}

/// A launch the fold admits: whole 128-column blocks, every segment's rows set, a built width and (WK, MINB).
pub fn admits(tab: dense.LinTable, ro: RotOut, k2: u32, wk: usize, minb: u32) bool {
    if (std.mem.indexOfScalar(u32, &dense3_k2s, k2) == null or ro.ld <= 0 or @mod(ro.ld, 128) != 0) return false;
    const cfg_ok = for (rot_cfgs) |c| {
        if (c[0] == wk and c[1] == minb) break true;
    } else false;
    if (!cfg_ok or tab.n < 1 or tab.n > max_seg) return false;
    for (tab.s[0..@intCast(tab.n)], 0..) |s, j| {
        if (ro.xh[j] == 0 or ro.suh[j] == 0 or @mod(s.N, 128) != 0) return false;
    }
    return true;
}

/// dense3's lanes program plus the rotated rows `ro` (never lin2's variants: the fold has only this program).
pub fn lanesRotLinear(o: dense.Ops, tab: dense.LinTable, ro: RotOut, y_dtype: dense.DType, M: usize, programs: usize, k2: u32, wk: usize, minb: u32, pdl: bool) !void {
    const f = o.f.rot orelse return error.RotFoldNotLoaded;
    if (!admits(tab, ro, k2, wk, minb)) return error.Unsupported;
    if (M < 1 or M > dense.rows_max) return error.Shape;
    for (tab.s[0..@intCast(tab.n)]) |s| {
        const per_warp: usize = @intCast(@divTrunc(@divTrunc(@divTrunc(s.K, 16), s.SK), s.wk));
        if (per_warp % dense.groupSteps(k2) != 0) return error.Shape;
    }
    const ki = std.mem.indexOfScalar(u32, &dense3_k2s, k2).?;
    const ci: usize = if (wk == 8) 1 else 0;
    var a: cuda.Args = .{};
    a.add(tab);
    a.add(@backingInt(y_dtype));
    a.add(@as(c_int, @intCast(M)));
    a.add(ro);
    const shared = wk * @min(M, 8) * 128 * 4;
    try cuda.launch.launch(f.lanes[ki][ci], .{ .grid = .{ .x = @intCast(programs), .y = 1, .z = 1 }, .block = .{ .x = @intCast(wk * 32) }, .shared = @intCast(shared), .pdl = pdl }, o.s, &a);
}

test "TF_DSV41_ROT_FOLD mask: unset / empty / 0 none, 1-7 a mask, anything else refused" {
    const t = std.testing;
    for ([_]?[]const u8{ null, "", " 0 " }) |v| try t.expectEqual(@as(u3, 0), try rotFoldMask(v));
    try t.expectEqual(@as(u3, 1), try rotFoldMask(" 1\n"));
    try t.expectEqual(@as(u3, 7), try rotFoldMask("7"));
    for ([_][]const u8{ "8", "-1", "01", "on", "0x1", "1,2" }) |v| try t.expectError(error.BadRotFold, rotFoldMask(v));
}

test "rot tables: each segment's rows and suh at its output column; misaligned segments refused" {
    const t = std.testing;
    const segs = [_]dense.Seg{
        .{ .k = 4096, .n = 1024, .k2 = 10, .sk = 8, .wk = 4, .col = 0 },
        .{ .k = 4096, .n = 1024, .k2 = 12, .sk = 8, .wk = 4, .col = 1024 },
        .{ .k = 4096, .n = 1024, .k2 = 10, .sk = 8, .wk = 4, .col = 2048 },
    };
    const lay = try dense.layout(&segs, 3);
    const l0 = lay.launches[0];
    const cols = launchCols(l0, &segs);
    try t.expectEqualSlices(usize, &.{ 0, 2048 }, cols[0..l0.count]);
    const ro = try rotOut(cols[0..l0.count], 0x10000, 0x20000, 3072);
    try t.expectEqual(@as(u64, 0x10000 + 2 * 2048), ro.xh[1]);
    try t.expectEqual(@as(u64, 0x20000 + 2 * 2048), ro.suh[1]);
    try t.expectEqual(@as(u64, 0), ro.xh[2]);
    const bufs = [_]dense.SegBufs{ .{ .T = 1, .svh = 2, .y = 0x30000, .ld = 3072 }, .{ .T = 3, .svh = 4, .y = 0x30000, .ld = 3072 }, .{ .T = 5, .svh = 6, .y = 0x30000, .ld = 3072 } };
    const tab = dense.linTable(lay, l0, &segs, &bufs, .bf16, 0x40000, 0x50000, 0x60000);
    try t.expect(admits(tab, ro, 10, 4, 3));
    try t.expect(!admits(tab, ro, 16, 4, 3)); // no dense3 width 16
    try t.expect(!admits(tab, ro, 10, 4, 4)); // a (WK, MINB) the emitter never asks for
    // a segment starting mid-block, rows not a whole number of blocks, a column past the row: refused
    try t.expectError(error.Unsupported, rotOut(&.{ 0, 1088 }, 0x10000, 0x20000, 3072));
    try t.expectError(error.Unsupported, rotOut(&.{0}, 0x10000, 0x20000, 3008));
    try t.expectError(error.Unsupported, rotOut(&.{ 0, 3072 }, 0x10000, 0x20000, 3072));
    var unset = ro;
    unset.suh[1] = 0;
    try t.expect(!admits(tab, unset, 10, 4, 3));
}
