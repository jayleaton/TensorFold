//! TF_DSV41_IDX_SPLIT: a TP2 prefill segment's index selection split by rows between the ranks, int32 rows exchanged.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel

/// off: every rank selects every row (the replicated path); auto: past `auto_work`; all: every eligible segment.
pub const Mode = enum { off, auto, all };

/// Segments under this many rows stay replicated (a rank's part is at least 64 rows; CED's decoder pass never splits).
pub const min_rows: i64 = 256;

/// auto: a layer splits when rows x visible keys reach this (4,096 rows at 65,536 compressed keys).
pub const auto_work: i64 = 4096 * 65536;

/// The cut's grid: rank 0's rows a multiple of this.
pub const grid: i64 = 64;

const name = "TF_DSV41_IDX_SPLIT";

/// The setting as read: a mode, or a value no rank accepts (kept apart so the ranks can agree on refusing it).
const Read = union(enum) { mode: Mode, bad: []const u8 };

fn read(get: *const fn ([]const u8) ?[]const u8) Read {
    const raw = std.mem.trim(u8, get(name) orelse return .{ .mode = .off }, " \t");
    if (raw.len == 0 or std.mem.eql(u8, raw, "0") or std.ascii.eqlIgnoreCase(raw, "off")) return .{ .mode = .off };
    if (std.ascii.eqlIgnoreCase(raw, "auto")) return .{ .mode = .auto };
    if (std.mem.eql(u8, raw, "1") or std.ascii.eqlIgnoreCase(raw, "on")) return .{ .mode = .all };
    return .{ .bad = raw };
}

/// TF_DSV41_IDX_SPLIT: unset / 0 / off = off, auto = the work threshold, 1 / on = every eligible segment.
pub fn mode(get: *const fn ([]const u8) ?[]const u8) !Mode {
    return switch (read(get)) {
        .mode => |m| m,
        .bad => |raw| {
            if (!quiet) std.log.warn("{s}={s}: expected 0 / off, auto or 1", .{ name, raw });
            return error.BadKnob;
        },
    };
}

/// The bytes the TP bootstrap's settings agreement carries for this setting (tp.Config.family); null when off.
pub fn agreement(get: *const fn ([]const u8) ?[]const u8) ?[32]u8 {
    // off adds nothing: an off / off boot agrees on exactly the base digest; any other pair differs and both refuse
    const tag: []const u8 = switch (read(get)) {
        .mode => |m| switch (m) {
            .off => return null,
            .auto => "auto",
            .all => "all",
        },
        .bad => "invalid",
    };
    var out: [32]u8 = undefined;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(name ++ "=");
    h.update(tag);
    h.final(&out);
    return out;
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

    /// The cut of `n` rows: the 64-row grid's even one, from the rows alone (no rank-local budget or step).
    pub fn of(n: i64) Cut {
        return .{ .n = n, .h0 = @divFloor(n, 2 * grid) * grid };
    }

    /// This rank's rows [lo, hi).
    pub fn mine(c: Cut, rank: u32) [2]i64 {
        return if (rank == 0) .{ 0, c.h0 } else .{ c.h0, c.n };
    }

    /// Rows each rank sends: the larger part's (the gather's equal counts).
    pub fn big(c: Cut) i64 {
        return @max(c.h0, c.n - c.h0);
    }

    pub fn valid(c: Cut) bool {
        return c.h0 > 0 and c.h0 < c.n;
    }

    /// The glue's transfers for `rank`: the gather and the copy of the other rank's rows (`Plan`).
    pub fn plan(c: Cut, rank: u32) Plan {
        const b = c.big();
        // rank 0 sends rows [0, big) (its own first), rank 1 rows [n - big, n) (its own last); the rest is dropped
        if (rank == 0) return .{ .send = 0, .count = b, .src = b + c.h0 - (c.n - b), .dst = c.h0, .rows = c.n - c.h0 };
        return .{ .send = c.n - b, .count = b, .src = 0, .dst = 0, .rows = c.h0 };
    }
};

/// In rows: send rows [send, send + count) into the [2 x count] gather, then its rows [src, src + rows) to row `dst`.
pub const Plan = struct { send: i64, count: i64, src: i64, dst: i64, rows: i64 };

/// The glue's checks of `ix_gather(sel [n, cols], tmp, h0)`: two ranks, a valid cut, contiguous rows, room in `tmp`.
pub fn glueCut(world: usize, n: i64, h0: i64, cols: i64, stride: i64, tmp_elems: usize) error{BadGlue}!Cut {
    const c: Cut = .{ .n = n, .h0 = h0 };
    if (world != 2 or !c.valid() or cols < 1 or stride != cols) return error.BadGlue;
    if (tmp_elems < @as(usize, @intCast(2 * c.big() * cols))) return error.BadGlue;
    return c;
}
