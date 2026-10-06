//! The run-time weight forms the Python engine's Block builds at load (blocks.materialize, exl3 experts.prepare), from our named host tensors: linear trellises as strips, and the routed + shared experts' tables and scale stacks.

const std = @import("std");

/// Trellis int16 [kt, nt, words] -> int32 [nt/8, kt, 8, words/2] (exl3/linear.py strips): each 128-column block's
/// tiles in K order, one copy of 8 tiles at a time.
pub fn strips(gpa: std.mem.Allocator, src: []const u8, kt: usize, nt: usize, words: usize) ![]align(16) u8 {
    if (nt % 8 != 0 or src.len != kt * nt * words * 2) return error.BadTrellis;
    const run = 8 * words * 2; // 8 tiles of one K row inside one 128-column block
    const out = try gpa.alignedAlloc(u8, .@"16", src.len);
    const blocks = nt / 8;
    for (0..blocks * kt) |i| {
        const nb = i / kt;
        const k = i % kt;
        @memcpy(out[i * run ..][0..run], src[(k * nt + nb * 8) * words * 2 ..][0..run]);
    }
    return out;
}

/// One expert of a projection: its trellis's device address and K2, its scales as stored (fp16).
pub const Expert = struct { trellis: u64, k2: i32, suh: []const u8, svh: []const u8 };

/// A projection's tables as exl3 experts.prepare builds them: int64 addresses, int32 K2s, suh [E, K] and svh [E, N] fp16.
pub const Tables = struct {
    ptrs: []i64,
    k2s: []i32,
    suh: []align(16) u8,
    svh: []align(16) u8,
    k2_lo: i32,
    k2_hi: i32,

    pub fn deinit(t: *Tables, gpa: std.mem.Allocator) void {
        gpa.free(t.ptrs);
        gpa.free(t.k2s);
        gpa.free(t.suh);
        gpa.free(t.svh);
        t.* = undefined;
    }
};

/// The tables of `experts` in order (the routed ones, then the shared expert as the last entry, as the Python MoE
/// appends it); every expert's suh and svh must have the projection's K and N.
pub fn tables(gpa: std.mem.Allocator, experts: []const Expert) !Tables {
    if (experts.len == 0) return error.NoExperts;
    const kb = experts[0].suh.len;
    const nb = experts[0].svh.len;
    var t: Tables = .{
        .ptrs = try gpa.alloc(i64, experts.len),
        .k2s = try gpa.alloc(i32, experts.len),
        .suh = try gpa.alignedAlloc(u8, .@"16", experts.len * kb),
        .svh = try gpa.alignedAlloc(u8, .@"16", experts.len * nb),
        .k2_lo = std.math.maxInt(i32),
        .k2_hi = 0,
    };
    errdefer t.deinit(gpa);
    for (experts, 0..) |e, i| {
        if (e.suh.len != kb or e.svh.len != nb) return error.ExpertShapesDiffer;
        t.ptrs[i] = @bitCast(e.trellis);
        t.k2s[i] = e.k2;
        @memcpy(t.suh[i * kb ..][0..kb], e.suh);
        @memcpy(t.svh[i * nb ..][0..nb], e.svh);
        t.k2_lo = @min(t.k2_lo, e.k2);
        t.k2_hi = @max(t.k2_hi, e.k2);
    }
    return t;
}

const testing = std.testing;

test "strips: each 128-column block's tiles in K order" {
    const a = testing.allocator;
    const kt = 3;
    const nt = 16;
    const words = 2; // one int32 a tile, to keep the check readable
    var src: [kt * nt * words * 2]u8 = undefined;
    // tile (k, n) holds the byte 16 k + n
    for (0..kt) |k| for (0..nt) |n| @memset(src[(k * nt + n) * words * 2 ..][0 .. words * 2], @intCast(16 * k + n));
    const out = try strips(a, &src, kt, nt, words);
    defer a.free(out);
    // out[nb][k][j] = tile (k, 8 nb + j)
    for (0..2) |nb| for (0..kt) |k| for (0..8) |j| {
        try testing.expectEqual(@as(u8, @intCast(16 * k + 8 * nb + j)), out[((nb * kt + k) * 8 + j) * words * 2]);
    };
    try testing.expectError(error.BadTrellis, strips(a, src[0..8], kt, nt, words));
}

test "expert tables: addresses, K2s and stacked scales in order, the width range" {
    const a = testing.allocator;
    const s1 = [_]u8{ 1, 2, 3, 4 };
    const s2 = [_]u8{ 5, 6, 7, 8 };
    const v1 = [_]u8{ 9, 9 };
    const v2 = [_]u8{ 7, 7 };
    var t = try tables(a, &.{ .{ .trellis = 0x7f00_0000_1000, .k2 = 4, .suh = &s1, .svh = &v1 }, .{ .trellis = 0x7f00_0002_0000, .k2 = 10, .suh = &s2, .svh = &v2 } });
    defer t.deinit(a);
    try testing.expectEqualSlices(i64, &.{ 0x7f00_0000_1000, 0x7f00_0002_0000 }, t.ptrs);
    try testing.expectEqualSlices(i32, &.{ 4, 10 }, t.k2s);
    try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, t.suh);
    try testing.expectEqualSlices(u8, &.{ 9, 9, 7, 7 }, t.svh);
    try testing.expect(t.k2_lo == 4 and t.k2_hi == 10);
    try testing.expectError(error.ExpertShapesDiffer, tables(a, &.{ .{ .trellis = 1, .k2 = 4, .suh = &s1, .svh = &v1 }, .{ .trellis = 2, .k2 = 4, .suh = s2[0..2], .svh = &v2 } }));
}
