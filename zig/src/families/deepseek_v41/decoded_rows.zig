//! TF_DSV41_DECODED_ROWS: the prompt rows that run as decode windows, so a re-sent reply keeps decoding's bits.

const std = @import("std");
const DecodedRows = @import("serve_knobs.zig").DecodedRows;

/// Rows [lo, hi) of the prompt that run as decode windows.
pub const Region = [2]u64;

/// The regions of a prompt with `spans` (Request.decode_spans: sorted, disjoint), its last row (`finish`'s) at `last`.
pub fn regions(a: std.mem.Allocator, mode: DecodedRows, spans: []const [2]u32, last: u64) ![]Region {
    var out: std.ArrayList(Region) = .empty;
    errdefer out.deinit(a);
    if (mode == .off or spans.len == 0) return out.toOwnedSlice(a);
    for (spans) |sp| {
        const lo: u64 = @as(u64, sp[0]) -| 1; // the reply's request ran row s - 1 as its first verify row
        // `.after`: to the end, so rows inserted after a reply (the compute lane's results) are windows too
        const hi: u64 = if (mode == .after) last else @min(sp[1], last);
        if (lo >= hi) continue;
        // a reply right after another one (no rows between them) joins it
        if (out.items.len > 0 and out.items[out.items.len - 1][1] >= lo) {
            out.items[out.items.len - 1][1] = @max(out.items[out.items.len - 1][1], hi);
        } else try out.append(a, .{ lo, hi });
        if (mode == .after) break;
    }
    return out.toOwnedSlice(a);
}

/// The sub-range of [at, end) that starts at `at`: its end, and whether its rows are decode windows.
pub fn next(regs: []const Region, at: u64, end: u64) struct { end: u64, decoded: bool } {
    for (regs) |r| {
        if (at >= r[1]) continue;
        if (at >= r[0]) return .{ .end = @min(end, r[1]), .decoded = true };
        return .{ .end = @min(end, r[0]), .decoded = false };
    }
    return .{ .end = end, .decoded = false };
}

/// Whether any row of [start, end) is in a region.
pub fn touches(regs: []const Region, start: u64, end: u64) bool {
    for (regs) |r| if (r[0] < end and start < r[1]) return true;
    return false;
}

/// Where the decode windows before `at` end: the last region's end at or before `at` (0: none).
pub fn decodedTo(regs: []const Region, at: u64) u64 {
    var to: u64 = 0;
    for (regs) |r| if (r[1] <= at) {
        to = r[1];
    };
    return to;
}

const testing = std.testing;

test "decoded rows: a reply span runs from its request's last prompt row through its EOS row" {
    const a = testing.allocator;
    const spans = [_][2]u32{ .{ 10, 20 }, .{ 30, 40 } };
    const r = try regions(a, .replies, &spans, 50);
    defer a.free(r);
    try testing.expectEqualSlices(Region, &.{ .{ 9, 20 }, .{ 29, 40 } }, r);
    // the prompt's last row stays finish's verify row
    const cut = try regions(a, .replies, &spans, 35);
    defer a.free(cut);
    try testing.expectEqualSlices(Region, &.{ .{ 9, 20 }, .{ 29, 35 } }, cut);
    const after = try regions(a, .after, &spans, 50);
    defer a.free(after);
    try testing.expectEqualSlices(Region, &.{.{ 9, 50 }}, after);
}

test "decoded rows: off, no spans, adjacent replies and a span past the last row" {
    const a = testing.allocator;
    const spans = [_][2]u32{ .{ 10, 20 }, .{ 21, 30 }, .{ 60, 70 } };
    const off = try regions(a, .off, &spans, 50);
    defer a.free(off);
    try testing.expectEqual(0, off.len);
    const none = try regions(a, .replies, &.{}, 50);
    defer a.free(none);
    try testing.expectEqual(0, none.len);
    const r = try regions(a, .replies, &spans, 50);
    defer a.free(r);
    try testing.expectEqualSlices(Region, &.{.{ 9, 30 }}, r);
    const first = [_][2]u32{.{ 0, 4 }};
    const z = try regions(a, .replies, &first, 50);
    defer a.free(z);
    try testing.expectEqualSlices(Region, &.{.{ 0, 4 }}, z);
}

test "decoded rows: a piece splits at region edges, and windows before a position" {
    const regs = [_]Region{ .{ 9, 20 }, .{ 29, 40 } };
    try testing.expectEqual(@as(u64, 9), next(&regs, 0, 50).end);
    try testing.expect(!next(&regs, 0, 50).decoded);
    try testing.expectEqual(@as(u64, 20), next(&regs, 9, 50).end);
    try testing.expect(next(&regs, 9, 50).decoded);
    try testing.expectEqual(@as(u64, 15), next(&regs, 12, 15).end);
    try testing.expectEqual(@as(u64, 29), next(&regs, 20, 50).end);
    try testing.expect(!next(&regs, 40, 50).decoded);
    try testing.expect(touches(&regs, 0, 10));
    try testing.expect(!touches(&regs, 20, 29));
    try testing.expectEqual(@as(u64, 0), decodedTo(&regs, 19));
    try testing.expectEqual(@as(u64, 20), decodedTo(&regs, 29));
    try testing.expectEqual(@as(u64, 40), decodedTo(&regs, 49));
}
