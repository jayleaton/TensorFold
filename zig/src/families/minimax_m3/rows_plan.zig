//! Prompt chunk rows under a bounded block-score scratch: a pure function of position, the same on every rank.
const std = @import("std");
const iface = @import("iface.zig");
const pool = @import("pool.zig");

/// Block maxima are fp32 per (row, kv head, 128-key block); a call's stride is the blocks its last row reaches.
pub const block = pool.page_tokens;

pub fn blocks(last_pos: usize) usize {
    return last_pos / block + 1;
}

/// Scratch bytes `rows` rows need when the last of them sits at `last_pos`.
pub fn need(kvh: u32, rows: usize, last_pos: usize) u64 {
    return @as(u64, rows) * kvh * blocks(last_pos) * 4;
}

pub const Plan = struct {
    kvh: u32,
    max_rows: u32 = iface.prefill_rows, // a chunk's rows at most (the forward's buffers hold this many)
    positions: usize, // the largest position plus one a sequence reaches (max_pages whole pages)
    scratch: u64, // bytes of the block-score scratch both ranks allocate

    /// The plan of a rank with `kvh` kv heads and `max_pages` table entries under a `mib` bound (0: never bounded).
    pub fn of(kvh: u32, max_pages: usize, mib: u32) Plan {
        const positions = max_pages * block;
        const full = need(kvh, iface.prefill_rows, positions - 1);
        // a window of shared rows and a one-page chunk always fit, so decode never depends on the bound
        const floor = @max(need(kvh, iface.shared_rows, positions - 1), need(kvh, pool.page_tokens, positions - 1));
        const bound: u64 = if (mib == 0) full else @as(u64, mib) << 20;
        return .{ .kvh = kvh, .positions = positions, .scratch = @max(floor, @min(full, bound)) };
    }

    /// Fixed rows with scratch for all of them at every position (the unbounded form; host fakes use it).
    pub fn fixed(kvh: u32, max_pages: usize) Plan {
        return of(kvh, max_pages, 0);
    }

    pub fn fits(p: Plan, rows: usize, last_pos: usize) bool {
        return need(p.kvh, rows, last_pos) <= p.scratch;
    }

    /// Rows of the chunk at `start` of a `len`-id prompt: the most whole pages that fit, or the prompt's tail.
    pub fn rowsAt(p: Plan, start: usize, len: usize) usize {
        const left = len - start;
        if (left <= p.max_rows and p.fits(left, len - 1)) return left;
        var r: usize = @min(p.max_rows, left) / block * block;
        while (r > block and !p.fits(r, start + r - 1)) r -= block;
        return @max(r, block);
    }

    /// The end of the chunk that starts at `start`: its rows, or the first mark inside them.
    pub fn chunkEnd(p: Plan, start: usize, len: usize, marks: ?iface.Marks) usize {
        var end = start + p.rowsAt(start, len);
        if (marks) |m| for (m.at) |at| {
            if (at > start and at < end) end = at;
        };
        return end;
    }
};

const testing = std.testing;

test "unbounded plans keep the fixed 2048-row chunks and scratch for them at the last position" {
    const p = Plan.fixed(2, iface.maxPages(65536 + iface.window_rows));
    try testing.expectEqual(@as(usize, 2048), p.rowsAt(0, 5000));
    try testing.expectEqual(@as(usize, 904), p.rowsAt(4096, 5000));
    try testing.expectEqual(need(2, 2048, p.positions - 1), p.scratch);
}

test "a bound shrinks chunks in whole pages as positions grow, never below a page, never back up" {
    const p = Plan.of(2, iface.maxPages((1 << 20) + iface.window_rows), 16);
    var start: usize = 0;
    var last: usize = p.max_rows;
    const len: usize = 1 << 20;
    while (start < len) {
        const r = p.rowsAt(start, len);
        try testing.expect(r >= block or start + r == len);
        try testing.expect(start + r == len or r % block == 0);
        try testing.expect(p.fits(r, start + r - 1));
        try testing.expect(r <= last or start + r == len);
        last = r;
        start += r;
    }
    try testing.expect(last < p.max_rows);
    try testing.expectEqual(@as(usize, 2048), p.rowsAt(0, len));
}

test "the floor keeps decode windows and one-page chunks inside the scratch at the last position" {
    const p = Plan.of(4, iface.maxPages((1 << 20) + iface.window_rows), 1);
    try testing.expect(p.fits(iface.shared_rows, p.positions - 1));
    try testing.expectEqual(@as(usize, block), p.rowsAt(p.positions - 4 * block, p.positions - 1));
}

test "marks still cut a chunk and every boundary stays on a page" {
    var at = [_]u32{ 512, 3072 };
    var ctx: u8 = 0;
    const m: iface.Marks = .{ .at = &at, .ctx = &ctx, .call = struct {
        fn f(_: *anyopaque, _: u32) void {}
    }.f };
    const p = Plan.fixed(2, iface.maxPages(65536 + iface.window_rows));
    try testing.expectEqual(@as(usize, 512), p.chunkEnd(0, 5000, m));
    try testing.expectEqual(@as(usize, 2560), p.chunkEnd(512, 5000, m));
    try testing.expectEqual(@as(usize, 3072), p.chunkEnd(2560, 5000, m));
}
