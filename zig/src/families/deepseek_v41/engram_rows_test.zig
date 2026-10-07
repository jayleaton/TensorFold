//! engram_rows.zig against the Python engine's own Engram host code (fixtures/engram-rows-ref.txt, written by
//! tools/zig/dsv41_perf/engram_fixture.py with torch on the CPU): the bf16 table, then a packed shard read back
//! through the synchronous path and through io_uring, prefetched, cached and evicted.

const std = @import("std");
const testing = std.testing;
const er = @import("engram_rows.zig");
const eh = @import("engram_host.zig");

const ref = @embedFile("fixtures/engram-rows-ref.txt");

/// engram_fixture.record: byte j of global row r.
fn recordByte(r: u64, j: u64) u8 {
    const h = (r * 2654435761 + j * 40503) % (1 << 32);
    return if (j < 256) @truncate(h >> 7) else @intCast(100 + (h >> 9) % 40);
}

const Fx = struct { lo: u64, rows: u64, total: u64, sets: [3]struct { idx: []i64, sha: [32]u8 } };

fn parseFx(a: std.mem.Allocator, lut_sha: *[32]u8) !Fx {
    var fx: Fx = undefined;
    var set: usize = 0;
    var it = std.mem.tokenizeScalar(u8, ref, '\n');
    while (it.next()) |line| {
        var w = std.mem.tokenizeScalar(u8, line, ' ');
        const kind = w.next().?;
        if (std.mem.eql(u8, kind, "lut")) {
            _ = try std.fmt.hexToBytes(lut_sha, w.next().?);
        } else if (std.mem.eql(u8, kind, "shard")) {
            fx.lo = try std.fmt.parseInt(u64, w.next().?, 10);
            fx.rows = try std.fmt.parseInt(u64, w.next().?, 10);
            fx.total = try std.fmt.parseInt(u64, w.next().?, 10);
        } else {
            var ids: std.ArrayList(i64) = .empty;
            var c = std.mem.tokenizeScalar(u8, w.next().?, ',');
            while (c.next()) |x| try ids.append(a, try std.fmt.parseInt(i64, x, 10));
            fx.sets[set].idx = try ids.toOwnedSlice(a);
            _ = try std.fmt.hexToBytes(&fx.sets[set].sha, w.next().?);
            set += 1;
        }
    }
    return fx;
}

/// Writes the fixture's shard (Python write_shard's layout) as engram-l1-r0of2.bin under `dir`.
fn writeShard(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, fx: Fx) !void {
    const n: usize = @intCast(er.header_bytes + fx.rows * er.row_bytes);
    const b = try a.alloc(u8, n);
    defer a.free(b);
    @memset(b, 0);
    for ([_]u64{ er.magic, 1, fx.lo, fx.lo + fx.rows, fx.total, er.row_bytes }, 0..) |v, i| std.mem.writeInt(u64, b[8 * i ..][0..8], v, .little);
    for (0..fx.rows) |k| for (0..er.row_bytes) |j| {
        b[er.header_bytes + k * er.row_bytes + j] = recordByte(fx.lo + k, j);
    };
    try dir.writeFile(io, .{ .sub_path = "engram-l1-r0of2.bin", .data = b });
}

fn sha(out: []const u16) [32]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(out), &d, .{});
    return d;
}

test "engram rows: the bf16 table equals torch's dequant(...).to(bfloat16)" {
    var want: [32]u8 = undefined;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    _ = try parseFx(arena.allocator(), &want);
    const t = try testing.allocator.create([65536]u16);
    defer testing.allocator.destroy(t);
    er.fillTable(t);
    try testing.expectEqual(want, sha(t));
    // spot checks: 1.0 x 2^0, -0.0, the smallest subnormal x 2^-6, scale byte 0 (x 0)
    try testing.expectEqual(@as(u16, 0x3F80), t[127 << 8 | 0x38]);
    try testing.expectEqual(@as(u16, 0x8000), t[127 << 8 | 0x80]);
    try testing.expectEqual(@as(u16, 0x3800), t[121 << 8 | 0x01]); // 2^-9 x 2^-6
    try testing.expectEqual(@as(u16, 0x0000), t[0 << 8 | 0x38]);
}

fn openRows(a: std.mem.Allocator, io: std.Io, tmp: *testing.TmpDir, uring: bool) !*er.Rows {
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const r = try er.Rows.open(a, root, 0, 2, &.{1}, 4096);
    if (!uring) if (r.ring) |*g| {
        g.deinit();
        r.ring = null;
    };
    return r;
}

test "engram rows: a packed shard read as Python's ShardRows.rows(...).to(bfloat16), synchronously and by io_uring" {
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var lut_sha: [32]u8 = undefined;
    const fx = try parseFx(arena.allocator(), &lut_sha);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, io, tmp.dir, fx);
    for ([_]bool{ false, true }) |uring| {
        const r = try openRows(a, io, &tmp, uring);
        defer r.close();
        try testing.expectEqual(fx.lo, r.tables[0].h.lo);
        for (fx.sets) |s| {
            const out = try a.alloc(u16, s.idx.len * 256);
            defer a.free(out);
            try r.rows(0, s.idx, 256, out);
            try testing.expectEqual(s.sha, sha(out));
            // again: every record from the cache now, the same bits
            const hits = r.stats.hits;
            try r.rows(0, s.idx, 256, out);
            try testing.expectEqual(s.sha, sha(out));
            try testing.expect(r.stats.hits > hits);
        }
        try testing.expectError(error.EngramRowOutsideShard, r.rows(0, &.{@intCast(fx.lo - 1)}, 256, try arena.allocator().alloc(u16, 256)));
    }
}

test "engram rows: issued ahead, then waited for; bulk reads past the cache; eviction keeps the bits" {
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var lut_sha: [32]u8 = undefined;
    const fx = try parseFx(arena.allocator(), &lut_sha);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, io, tmp.dir, fx);
    for ([_]bool{ false, true }) |uring| {
        const r = try openRows(a, io, &tmp, uring);
        defer r.close();
        const s = fx.sets[1];
        try r.issue(0, s.idx);
        try testing.expectEqual(@as(u64, s.idx.len), r.stats.issued);
        try r.issue(0, s.idx); // nothing new: every row is in flight or cached
        try testing.expectEqual(@as(u64, s.idx.len), r.stats.issued);
        const out = try a.alloc(u16, s.idx.len * 256);
        defer a.free(out);
        try r.rows(0, s.idx, 256, out);
        try testing.expectEqual(s.sha, sha(out));
        try testing.expectEqual(@as(u64, 0), r.stats.misses);
        // every row of the shard once (a prompt's bulk read: past cap / 4, read through), then the sets again
        const all = try a.alloc(i64, fx.rows);
        defer a.free(all);
        for (all, 0..) |*x, k| x.* = @intCast(fx.lo + fx.rows - 1 - k);
        const big = try a.alloc(u16, all.len * 256);
        defer a.free(big);
        try r.rows(0, all, 256, big);
        for (all, 0..) |x, k| for (0..256) |j| {
            const row: u64 = @intCast(x);
            const v = recordByte(row, j);
            const sc = recordByte(row, 256 + j / 32);
            try testing.expectEqual(r.lut[@as(usize, sc) << 8 | v], big[k * 256 + j]);
        };
        for (fx.sets) |q| {
            const o = try a.alloc(u16, q.idx.len * 256);
            defer a.free(o);
            try r.rows(0, q.idx, 256, o);
            try testing.expectEqual(q.sha, sha(o));
        }
        try testing.expect(r.tables[0].used <= 4096);
    }
}

test "engram rows: shard headers" {
    var b: [48]u8 = undefined;
    for ([_]u64{ er.magic, 14, 10, 20, 100, er.row_bytes }, 0..) |v, i| std.mem.writeInt(u64, b[8 * i ..][0..8], v, .little);
    const h = try er.Header.parse(&b, er.header_bytes + 10 * er.row_bytes);
    try testing.expectEqual(@as(u64, 14), h.layer);
    try testing.expectError(error.BadEngramShard, er.Header.parse(&b, er.header_bytes + 9 * er.row_bytes));
    b[0] ^= 1;
    try testing.expectError(error.NotEngramShard, er.Header.parse(&b, er.header_bytes + 10 * er.row_bytes));
}

test "engram lookback: the tail, then the kept rows of the window in flight" {
    var buf: [16]u32 = undefined;
    // nothing in flight: the next window starts at the slot's position
    try testing.expectEqualSlices(u32, &.{ 7, 8, 9 }, eh.lookbackAt(&.{ 7, 8, 9 }, 50, null, 50, 3, &buf).?);
    try testing.expect(eh.lookbackAt(&.{ 7, 8, 9 }, 50, null, 51, 3, &buf) == null);
    // a window [a, d1, d2] at 50 not kept yet; the next starts at 52: a and d1 were kept
    try testing.expectEqualSlices(u32, &.{ 9, 100, 101 }, eh.lookbackAt(&.{ 7, 8, 9 }, 50, &.{ 100, 101, 102 }, 52, 3, &buf).?);
    try testing.expectEqualSlices(u32, &.{ 100, 101, 102 }, eh.lookbackAt(&.{ 7, 8, 9 }, 50, &.{ 100, 101, 102 }, 53, 3, &buf).?);
    try testing.expect(eh.lookbackAt(&.{ 7, 8, 9 }, 50, &.{ 100, 101, 102 }, 54, 3, &buf) == null);
    // the sequence start: fewer than keep_n
    try testing.expectEqualSlices(u32, &.{5}, eh.lookbackAt(&.{}, 0, &.{ 5, 6 }, 1, 3, &buf).?);
}
