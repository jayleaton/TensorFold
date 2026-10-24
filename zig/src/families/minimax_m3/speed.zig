//! The speed settings both ranks fingerprint, and the graph keys' pure parts: context buckets, layouts, room.
const std = @import("std");

/// TF_MM3_SCORE_MIB default: the block-score scratch's bound on one rank.
pub const score_mib_default: u32 = 64;

/// Streams one keyed layout names (iface.max_shared; graphs.zig checks the two agree).
pub const max_streams = 16;

/// Context buckets 0-15 are `bucket` tokens each; past them each ends 1.25x further, rounded up to `bucket`.
pub const linear = 16;
pub const grow = 1.25;

pub const Graphs = struct {
    on: bool = true, // TF_MM3_GRAPHS=0: every window eager
    tp: bool = false, // TF_MM3_GRAPHS_TP=1: also capture the exchanges of two or more ranks
    max: u32 = 48, // TF_MM3_GRAPHS_MAX: graphs held at most
    rows_max: u32 = 64, // windows of more rows run eagerly (iface.shared_rows)
    bucket: u32 = 2048, // tokens of one linear context bucket
    hold: bool = true, // under the memory floor a miss runs eagerly and every held graph stays
    cost_mib: u32 = 64, // a capture's price against the floor
    floor_mib: u32 = 4096, // MemAvailable a capture must leave

    /// Off under TF_GRAPH_CACHE=0 or off, the operator's switch for every family's graph cache.
    pub fn operator(s: Graphs, value: ?[]const u8) Graphs {
        var out = s;
        const v = value orelse return out;
        if (std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off")) out.on = false;
        return out;
    }

    pub fn fromEnv(s: Graphs) Graphs {
        return s.operator(if (std.c.getenv("TF_GRAPH_CACHE")) |v| std.mem.span(v) else null);
    }

    /// Whether a forward over `world` ranks graphs at all.
    pub fn active(s: Graphs, world: u32) bool {
        return s.on and s.max > 0 and (world == 1 or s.tp);
    }

    /// The effective settings as one word for the ranks' fingerprint.
    pub fn digest(s: Graphs) u32 {
        const e = s.fromEnv();
        const words = [_]u64{ @intFromBool(e.on), @intFromBool(e.tp), e.max, e.rows_max, e.bucket, @intFromBool(e.hold), e.cost_mib, e.floor_mib };
        const h = fnv(&words);
        return @truncate(h ^ (h >> 32));
    }
};

fn fnv(words: []const u64) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (words) |w| {
        var x = w;
        for (0..8) |_| {
            h = (h ^ (x & 0xff)) *% 0x100000001b3;
            x >>= 8;
        }
    }
    return h;
}

/// The end (exclusive) of bucket `b`.
pub fn endOf(b: u32, size: u64) u64 {
    var e: u64 = size * @min(b + 1, linear);
    var i: u32 = linear;
    while (i <= b) : (i += 1) {
        const prod: u64 = @intFromFloat(@trunc(@as(f64, @floatFromInt(e)) * grow));
        e = @max(((prod + size - 1) / size) * size, e + size);
    }
    return e;
}

/// The bucket whose end lies past `last`.
pub fn bucketOf(last: u64, size: u64) u32 {
    var b: u32 = 0;
    while (endOf(b, size) <= last) b += 1;
    return b;
}

/// The last position bucket `b`'s graphs score keys for, inside the block table's `positions`.
pub fn bucketEnd(b: u32, size: u64, positions: u64) u64 {
    return @min(endOf(b, size) - 1, positions - 1);
}

/// Each stream's window in a keyed layout as slot << 8 | rows; zeros without taps.
pub const Layout = [max_streams]u32;

pub const Key = struct {
    rows: u32,
    ctx: u32,
    head: u8,
    mode: u8,
    layout: Layout = @splat(0),
};

/// The layout of `segs` (anything with `slot` and `rows`, in row order).
pub fn layoutOf(segs: anytype) error{BadCall}!Layout {
    if (segs.len > max_streams) return error.BadCall;
    var out: Layout = @splat(0);
    for (segs, out[0..segs.len]) |s, *o| {
        if (s.rows == 0 or s.rows > 255 or s.slot > 0xffffff) return error.BadCall;
        o.* = @as(u32, s.slot) << 8 | s.rows;
    }
    return out;
}

/// One more graph fits when its price still leaves the floor (unknown memory never blocks).
pub fn fits(s: Graphs, avail_mib: ?u64) bool {
    const a = avail_mib orelse return true;
    return a >= @as(u64, s.cost_mib) + s.floor_mib;
}

pub fn memAvailableMiB() ?u64 {
    var buf: [4096]u8 = undefined;
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return null;
    return memAvailableOf(buf[0..@intCast(n)]);
}

fn memAvailableOf(text: []const u8) ?u64 {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, "MemAvailable:")) continue;
        var f = std.mem.tokenizeScalar(u8, line["MemAvailable:".len..], ' ');
        const kb = std.fmt.parseInt(u64, f.next() orelse return null, 10) catch return null;
        return kb / 1024;
    }
    return null;
}

const testing = std.testing;

test "context buckets: 16 linear, then 1.25x rounded up; a window lands in the first bucket ending past it" {
    try testing.expectEqual(@as(u64, 2048), endOf(0, 2048));
    try testing.expectEqual(@as(u64, 32768), endOf(15, 2048));
    try testing.expectEqual(@as(u64, 40960), endOf(16, 2048));
    try testing.expectEqual(@as(u32, 0), bucketOf(2047, 2048));
    try testing.expectEqual(@as(u32, 1), bucketOf(2048, 2048));
    try testing.expectEqual(@as(u32, 16), bucketOf(32768, 2048));
    var last: u64 = 0;
    while (last < 1 << 20) : (last += 997) try testing.expect(bucketEnd(bucketOf(last, 2048), 2048, 1 << 21) >= last);
    try testing.expectEqual(@as(u64, 4095), bucketEnd(40, 2048, 4096));
    try testing.expect(bucketOf(1 << 20, 2048) < 40);
}

test "a layout names each stream's slot and rows; zero rows or too many streams refuse" {
    const Seg = struct { slot: u32, rows: u32 };
    const l = try layoutOf(@as([]const Seg, &.{ .{ .slot = 3, .rows = 2 }, .{ .slot = 0, .rows = 1 } }));
    try testing.expectEqual(@as(u32, 3 << 8 | 2), l[0]);
    try testing.expectEqual(@as(u32, 1), l[1]);
    try testing.expectEqual(@as(u32, 0), l[2]);
    try testing.expectError(error.BadCall, layoutOf(@as([]const Seg, &.{.{ .slot = 1, .rows = 0 }})));
    const many: [max_streams + 1]Seg = @splat(.{ .slot = 0, .rows = 1 });
    try testing.expectError(error.BadCall, layoutOf(@as([]const Seg, &many)));
}

test "settings: one rank graphs by default, ranks only with the TP opt-in; the digest follows every field" {
    const s: Graphs = .{};
    try testing.expect(s.active(1));
    try testing.expect(!s.active(2));
    try testing.expect((Graphs{ .tp = true }).active(2));
    try testing.expect(!(Graphs{ .on = false }).active(1));
    try testing.expect(!s.operator("off").on);
    try testing.expect(s.operator("1").on);
    try testing.expect(s.digest() != (Graphs{ .max = 11 }).digest());
    try testing.expect(s.digest() != (Graphs{ .tp = true }).digest());
}

test "room: a capture's price must leave the floor; unknown memory never blocks" {
    try testing.expect(fits(.{}, 8192));
    try testing.expect(!fits(.{}, 4100));
    try testing.expect(fits(.{}, null));
    try testing.expectEqual(@as(?u64, 2048), memAvailableOf("MemTotal: 9 kB\nMemAvailable:    2097152 kB\n"));
}
