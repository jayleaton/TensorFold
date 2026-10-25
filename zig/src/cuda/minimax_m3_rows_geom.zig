//! MiniMax-M3.1-Flash row kernels: launch geometry, flag bits and block-table arithmetic (std only, host-testable).
const std = @import("std");

pub const d = 128; // head_dim and index_head_dim
pub const block = 128; // tokens a page
pub const rows_threads = 256; // 8 warps, a warp a head
pub const rows_warps = 8;
pub const router_warps = 4; // experts a block
pub const router_rows = 4; // rows a warp
pub const router_chunk = 128; // reduction length is a multiple: 32 lanes of 4 values
pub const router_threads = router_warps * 32;
pub const max_rot = 128;
pub const max_rot_pairs = 64;
pub const max_grid_y = 65535;

/// Bits of the kernel's `flags` argument; the defaults are core/minimax_ref.zig's Switches defaults.
pub const Flags = struct {
    norm_qk: bool = true,
    norm_index: bool = true,
    index_rope: bool = true,
    index_fp16: bool = true,
    round_norm: bool = false,
    gptj: bool = false,
    /// The pool is the fp8 layout: k and v are written as e4m3 rows with an fp32 scale each (docs/fp8-kv.md).
    kv8: bool = false,

    pub fn bits(f: Flags) i32 {
        var b: i32 = 0;
        if (f.norm_qk) b |= 1;
        if (f.norm_index) b |= 2;
        if (f.index_rope) b |= 4;
        if (f.index_fp16) b |= 8;
        if (f.round_norm) b |= 16;
        if (f.gptj) b |= 32;
        if (f.kv8) b |= 64;
        return b;
    }
};

/// Bits the kernel ORs into its status word.
pub const status_no_page: i32 = 1;
pub const status_past_table: i32 = 2;

pub const Launch = struct { grid: [3]u32, block: u32 };

pub const RowsError = error{ Empty, TooManyRows, BadRot, BadStride, Heads };

/// Head slots a row works through: q, index q, k, v and the index key.
pub fn slots(nq: usize, niq: usize, kvh: usize, has_ik: bool) usize {
    return nq + niq + 2 * kvh + @intFromBool(has_ik);
}

/// One CTA a row; its warps take the slots in order.
pub fn rowsLaunch(rows: usize) RowsError!Launch {
    if (rows == 0) return error.Empty;
    if (rows > std.math.maxInt(i32)) return error.TooManyRows;
    return .{ .grid = .{ @intCast(rows), 1, 1 }, .block = rows_threads };
}

/// Rotation width must be even and fit the head; strides keep 8-byte vector loads and stores aligned.
pub fn checkRows(rot: usize, width: usize, ld: usize) RowsError!void {
    if (rot == 0 or rot % 2 != 0 or rot > max_rot) return error.BadRot;
    if (width == 0 or width % d != 0 or ld < width or ld % 4 != 0) return error.BadStride;
}

pub fn routerLaunch(rows: usize, experts: usize) RowsError!Launch {
    if (rows == 0 or experts == 0) return error.Empty;
    const gy = (rows + router_rows - 1) / router_rows;
    if (gy > max_grid_y) return error.TooManyRows;
    return .{ .grid = .{ @intCast((experts + router_warps - 1) / router_warps), @intCast(gy), 1 }, .block = router_threads };
}

pub fn checkRouter(k: usize, x_ld: usize) RowsError!void {
    if (k == 0 or k % router_chunk != 0 or x_ld < k or x_ld % 4 != 0) return error.BadStride;
}

pub const TableError = error{ BadSlot, TableFull, Overflow, NoPage, PastTable, Gap };

/// Entry offset of a slot's table row inside the device copy: slot * max_pages, which must fit an int32.
pub fn entryOffset(slot: usize, max_pages: usize) TableError!i32 {
    const v = std.math.mul(usize, slot, max_pages) catch return error.Overflow;
    if (v > std.math.maxInt(i32)) return error.Overflow;
    return @intCast(v);
}

/// Byte range of entries [from, from + n) of a slot's row in the device copy.
pub fn entryBytes(slots_n: usize, max_pages: usize, slot: usize, from: usize, n: usize) TableError!struct { off: usize, len: usize } {
    if (slot >= slots_n) return error.BadSlot;
    if (from > max_pages or n > max_pages - from) return error.TableFull;
    _ = try entryOffset(slots_n, max_pages);
    return .{ .off = (slot * max_pages + from) * @sizeOf(i32), .len = n * @sizeOf(i32) };
}

/// A stream's rows in a call: rows [row0, row0 + rows) belong to slot `slot`.
pub const Seg = struct { slot: u32, rows: usize };

/// row_tab for consecutive segments covering every row: each row's entry is its slot's offset.
pub fn fillRowTab(out: []i32, segs: []const Seg, slots_n: usize, max_pages: usize) TableError!void {
    var at: usize = 0;
    for (segs) |s| {
        if (s.slot >= slots_n) return error.BadSlot;
        if (s.rows > out.len - at) return error.Gap;
        const off = try entryOffset(s.slot, max_pages);
        @memset(out[at..][0..s.rows], off);
        at += s.rows;
    }
    if (at != out.len) return error.Gap;
}

/// The page the kernels read for position `pos` of the row whose table starts at entry `off` (the kernel's own lookup).
pub fn pageOf(tables: []const i32, max_pages: usize, off: usize, pos: usize) TableError!u32 {
    const b = pos / block;
    if (b >= max_pages) return error.PastTable;
    if (off + max_pages > tables.len) return error.TableFull;
    const id = tables[off + b];
    return if (id < 0) error.NoPage else @intCast(id);
}

/// Checks every row of a call before it launches: a page for the row's position. `row_tab` null: one table at entry 0.
pub fn checkCall(tables: []const i32, max_pages: usize, row_tab: ?[]const i32, pos: ?[]const i32, pos0: i32, rows: usize) TableError!void {
    for (0..rows) |r| {
        const p: i32 = if (pos) |a| a[r] else pos0 + @as(i32, @intCast(r));
        if (p < 0) continue;
        const off: usize = if (row_tab) |t| @intCast(t[r]) else 0;
        _ = try pageOf(tables, max_pages, off, @intCast(p));
    }
}

/// Blocks (= pages) a stream with positions up to `pos` needs.
pub fn pagesNeeded(pos: usize) usize {
    return pos / block + 1;
}

test "launch geometry and flag bits" {
    const l = try rowsLaunch(300);
    try std.testing.expectEqual([3]u32{ 300, 1, 1 }, l.grid);
    try std.testing.expectEqual(@as(u32, 256), l.block);
    try std.testing.expectError(error.Empty, rowsLaunch(0));
    const r = try routerLaunch(5, 128);
    try std.testing.expectEqual([3]u32{ 32, 2, 1 }, r.grid);
    try std.testing.expectEqual(@as(u32, 128), r.block);
    try std.testing.expectEqual(@as(usize, 32 + 2 + 4 + 1), slots(32, 2, 2, true));
    try std.testing.expectEqual(@as(i32, 1 | 2 | 4 | 8), (Flags{}).bits());
    try std.testing.expectEqual(@as(i32, 16 | 32), (Flags{ .norm_qk = false, .norm_index = false, .index_rope = false, .index_fp16 = false, .round_norm = true, .gptj = true }).bits());
}
