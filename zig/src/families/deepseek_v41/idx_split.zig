//! TF_DSV41_IDX_SPLIT: a TP2 prefill segment's index selection split by rows between the ranks, int32 rows exchanged.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel

/// off: every rank selects every row (the replicated path); auto: past `auto_work`; all: every eligible segment.
pub const Mode = enum { off, auto, all };

/// Segments under this many rows stay replicated (a rank's part is at least 64 rows; CED's decoder pass never splits).
pub const min_rows: i64 = 256;

/// auto: a layer splits when rows x visible keys reach this (4,096 rows at 65,536 compressed keys).
pub const auto_work: i64 = 4096 * 65536;

/// The cut's grid when the layer's own loop is one block: rows a multiple of this.
pub const grid: i64 = 64;

/// TF_DSV41_IDX_SPLIT: unset / 0 / off = off, auto = the work threshold, 1 / on = every eligible segment.
pub fn mode(get: *const fn ([]const u8) ?[]const u8) !Mode {
    const raw = std.mem.trim(u8, get("TF_DSV41_IDX_SPLIT") orelse return .off, " \t");
    if (raw.len == 0 or std.mem.eql(u8, raw, "0") or std.ascii.eqlIgnoreCase(raw, "off")) return .off;
    if (std.ascii.eqlIgnoreCase(raw, "auto")) return .auto;
    if (std.mem.eql(u8, raw, "1") or std.ascii.eqlIgnoreCase(raw, "on")) return .all;
    if (!quiet) std.log.warn("TF_DSV41_IDX_SPLIT={s}: expected 0 / off, auto or 1", .{raw});
    return error.BadKnob;
}

/// Whether a layer over `keys` visible keys splits a segment of `n` rows; from values every rank holds alike.
pub fn engages(m: Mode, world: u32, rows_mode: bool, n: i64, keys: i64) bool {
    if (m == .off or world != 2 or rows_mode or n < min_rows) return false;
    return m == .all or n * keys >= auto_work;
}

/// Rank 0 selects rows [0, h0), rank 1 rows [h0, n); each sends `big` rows, the other's copied out of the gather.
pub const Cut = struct {
    n: i64,
    h0: i64,

    /// The cut of `n` rows whose loop steps `st` rows: the loop's own boundary when about even, else the 64-row grid's.
    pub fn of(n: i64, st: i64) Cut {
        const even: Cut = .{ .n = n, .h0 = @divFloor(n, 2 * grid) * grid };
        if (st <= 0 or st >= n) return even;
        var h0: i64 = st;
        var s: i64 = 1;
        while (s * st < n) : (s += 1) {
            if (@max(s * st, n - s * st) < @max(h0, n - h0)) h0 = s * st;
        }
        const at: Cut = .{ .n = n, .h0 = h0 };
        return if (at.big() <= even.big() + grid) at else even;
    }

    /// This rank's rows [lo, hi).
    pub fn mine(c: Cut, rank: u32) [2]i64 {
        return if (rank == 0) .{ 0, c.h0 } else .{ c.h0, c.n };
    }

    /// Rows each rank sends: the larger part's (the gather's equal counts).
    pub fn big(c: Cut) i64 {
        return @max(c.h0, c.n - c.h0);
    }

    /// The first row a rank sends: rank 0 rows [0, big) (its own first), rank 1 rows [n - big, n) (its own last).
    pub fn sendRow(c: Cut, rank: u32) i64 {
        return if (rank == 0) 0 else c.n - c.big();
    }

    /// The other rank's rows: from row `src` of the gathered [2 x big] rows into row `dst`, `rows` of them.
    pub fn take(c: Cut, rank: u32) struct { src: i64, dst: i64, rows: i64 } {
        const b = c.big();
        if (rank == 0) return .{ .src = b + c.h0 - (c.n - b), .dst = c.h0, .rows = c.n - c.h0 };
        return .{ .src = 0, .dst = 0, .rows = c.h0 };
    }

    pub fn valid(c: Cut) bool {
        return c.h0 > 0 and c.h0 < c.n;
    }
};
