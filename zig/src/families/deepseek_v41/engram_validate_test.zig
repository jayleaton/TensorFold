//! Packed files with internally valid headers can still omit rows the rank's hasher will request.
const std = @import("std");
const t = std.testing;
const guard = @import("engram_validate.zig");
const format = @import("engram_rows.zig");
const hash = @import("engram_hash.zig");
const portable = @import("engram_rows_portable.zig");

pub fn metadata(id: u32) hash.Layer {
    var meta: hash.Layer = .{ .layer_id = id, .rows = 48, .multipliers = @splat(1), .primes = @splat(2), .offsets = undefined };
    for (&meta.offsets, 0..) |*x, i| x.* = @intCast(2 * i);
    return meta;
}

pub fn shard(dir: std.Io.Dir, meta: hash.Layer, rank: u32, lo: u64, hi: u64, total: u64, id: u32) !void {
    var bytes: [4096 + 48 * 264]u8 = @splat(0);
    for ([_]u64{ format.magic, id, lo, hi, total, format.row_bytes }, 0..) |x, i|
        std.mem.writeInt(u64, bytes[i * 8 ..][0..8], x, .little);
    for (0..@as(usize, @intCast(hi - lo))) |row| for (0..264) |col| {
        bytes[4096 + row * 264 + col] = if (col < 256) @intCast((row * 7 + col * 3 + id) % 120) else 127;
    };
    var name: [64]u8 = undefined;
    try dir.writeFile(t.io, .{
        .sub_path = try std.fmt.bufPrint(&name, "engram-l{d}-r{d}of2.bin", .{ meta.layer_id, rank }),
        .data = bytes[0..@intCast(format.header_bytes + (hi - lo) * format.row_bytes)],
    });
}

fn decoded(root: []const u8, rank: u32, out: *[1536]u16) !void {
    const rows = try portable.Rows.openWithIo(t.allocator, t.io, root, rank, 2, &.{ 1, 14 }, 4);
    defer rows.close();
    const base: i64 = @as(i64, rank) * 24;
    const ids = [_]i64{ base, base + 12, base + 23 };
    try rows.rows(0, &ids, 256, out[0..768]);
    try rows.rows(1, &ids, 256, out[768..]);
}

test "Engram startup validation is opt in and off never opens a shard" {
    for ([_]?[]const u8{ null, "", "0", " 0 " }) |raw|
        try guard.apply(t.io, raw, "no-such-engram-directory", 0, 2, &.{metadata(1)});
    try guard.apply(t.io, null, null, 0, 0, &.{});
    try t.expectError(error.BadKnob, guard.apply(t.io, "on", null, 0, 2, &.{}));
    try t.expectError(error.BadKnob, guard.apply(t.io, "2", null, 0, 2, &.{}));
    try t.expectError(error.MissingEngramDirectory, guard.apply(t.io, "1", null, 0, 2, &.{}));
    try t.expectError(error.BadEngramShard, guard.apply(t.io, "1", ".", 2, 2, &.{}));
    try t.expectError(error.BadEngramShard, guard.apply(t.io, "1", ".", 0, 0, &.{}));
}

test "Engram startup validation accepts both ranks and leaves packed bytes unchanged" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    const metas = [_]hash.Layer{ metadata(1), metadata(14) };
    for (metas) |meta| {
        try shard(tmp.dir, meta, 0, 0, 24, 48, meta.layer_id);
        try shard(tmp.dir, meta, 1, 24, 48, 48, meta.layer_id);
    }
    const before = try tmp.dir.readFileAlloc(t.io, "engram-l1-r0of2.bin", t.allocator, .limited(1 << 20));
    defer t.allocator.free(before);
    for (0..2) |rank| {
        var plain: [1536]u16 = undefined;
        var checked: [1536]u16 = undefined;
        try guard.apply(t.io, "0", root, @intCast(rank), 2, &metas);
        try decoded(root, @intCast(rank), &plain);
        try guard.apply(t.io, "1", root, @intCast(rank), 2, &metas);
        try decoded(root, @intCast(rank), &checked);
        try t.expectEqualSlices(u16, &plain, &checked);
    }
    const after = try tmp.dir.readFileAlloc(t.io, "engram-l1-r0of2.bin", t.allocator, .limited(1 << 20));
    defer t.allocator.free(after);
    try t.expectEqualSlices(u8, before, after);
    try guard.apply(t.io, "0", root, 0, 2, &metas);
    try guard.apply(t.io, "1", root, 0, 2, &metas);
}

test "internally valid Engram headers missing global or rank-owned rows fail before serving" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    const meta = metadata(1);
    const cases = [_]struct { rank: u32, lo: u64, hi: u64, total: u64, id: u32 }{
        .{ .rank = 0, .lo = 0, .hi = 24, .total = 47, .id = 1 },
        .{ .rank = 0, .lo = 1, .hi = 24, .total = 48, .id = 1 },
        .{ .rank = 0, .lo = 0, .hi = 23, .total = 48, .id = 1 },
        .{ .rank = 1, .lo = 25, .hi = 48, .total = 48, .id = 1 },
        .{ .rank = 1, .lo = 24, .hi = 47, .total = 48, .id = 1 },
        .{ .rank = 0, .lo = 0, .hi = 24, .total = 48, .id = 14 },
    };
    for (cases) |c| {
        try shard(tmp.dir, meta, c.rank, c.lo, c.hi, c.total, c.id);
        var name: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(&name, "engram-l1-r{d}of2.bin", .{c.rank});
        const bytes = try tmp.dir.readFileAlloc(t.io, path, t.allocator, .limited(1 << 20));
        defer t.allocator.free(bytes);
        _ = try format.Header.parse(bytes[0..48], bytes.len);
        try guard.apply(t.io, "0", root, c.rank, 2, &.{meta});
        try t.expectError(error.BadEngramShard, guard.apply(t.io, "1", root, c.rank, 2, &.{meta}));
    }
}

test "Engram startup validation checks every layer and refuses truncated files" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    const metas = [_]hash.Layer{ metadata(1), metadata(14) };
    try shard(tmp.dir, metas[0], 0, 0, 24, 48, 1);
    try shard(tmp.dir, metas[1], 0, 0, 24, 47, 14);
    try t.expectError(error.BadEngramShard, guard.apply(t.io, "1", root, 0, 2, &metas));
    try tmp.dir.writeFile(t.io, .{ .sub_path = "engram-l1-r0of2.bin", .data = "short" });
    try t.expectError(error.BadEngramShard, guard.apply(t.io, "1", root, 0, 2, &metas));
}

test "Engram refusal diagnostics retain file and expected versus actual geometry" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    var diagnostic: guard.Diagnostic = .{};
    const meta = metadata(1);
    try shard(tmp.dir, meta, 0, 0, 24, 47, 1);
    try t.expectError(error.BadEngramShard, guard.applyDiagnostic(t.io, "1", root, 0, 2, &.{meta}, &diagnostic));
    for ([_][]const u8{ root, "engram-l1-r0of2.bin", "rank 0 layer 1", "expected total >= 48, rows [0, 24)", "actual layer 1, total 47, rows [0, 24)" }) |part|
        try t.expect(std.mem.indexOf(u8, diagnostic.text(), part) != null);
    try shard(tmp.dir, meta, 1, 25, 48, 48, 1);
    try t.expectError(error.BadEngramShard, guard.applyDiagnostic(t.io, "1", root, 1, 2, &.{meta}, &diagnostic));
    for ([_][]const u8{ "engram-l1-r1of2.bin", "expected total >= 48, rows [24, 48)", "actual layer 1, total 48, rows [25, 48)" }) |part|
        try t.expect(std.mem.indexOf(u8, diagnostic.text(), part) != null);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "engram-l1-r0of2.bin", .data = "short" });
    try t.expectError(error.BadEngramShard, guard.applyDiagnostic(t.io, "1", root, 0, 2, &.{meta}, &diagnostic));
    try t.expect(std.mem.indexOf(u8, diagnostic.text(), "engram-l1-r0of2.bin") != null);
    try t.expect(std.mem.indexOf(u8, diagnostic.text(), "actual unavailable") != null);
    try t.expectError(error.FileNotFound, guard.applyDiagnostic(t.io, "1", root, 0, 2, &.{metadata(14)}, &diagnostic));
    try t.expect(std.mem.indexOf(u8, diagnostic.text(), "engram-l14-r0of2.bin") != null);
    try shard(tmp.dir, meta, 0, 0, 24, 48, 1);
    const file = try tmp.dir.openFile(t.io, "engram-l1-r0of2.bin", .{ .mode = .read_write });
    defer file.close(t.io);
    try file.setLength(t.io, 48);
    try t.expectError(error.BadEngramShard, guard.applyDiagnostic(t.io, "1", root, 0, 2, &.{meta}, &diagnostic));
    try t.expect(std.mem.indexOf(u8, diagnostic.text(), "actual layer 1, total 48, rows [0, 24)") != null);
}

test "canonical Engram unequal buckets accept padding and reject inward edges" {
    for (hash.layers, 0..) |meta, layer_index| {
        const tp2_boundary: u64 = if (layer_index == 0) 192001740 else 192007016;
        try t.expectEqual(tp2_boundary, guard.required(meta, 0, 2).hi);
        try t.expectEqual(tp2_boundary, guard.required(meta, 1, 2).lo);
        for ([_]u32{ 2, 5 }) |world| for (0..world) |r| {
            const rank: u32 = @intCast(r);
            const first = @min(rank * ((24 + world - 1) / world), 24);
            const end = @min(first + ((24 + world - 1) / world), 24);
            var lo: u64 = 0;
            var hi: u64 = 0;
            for (meta.primes[0..end], 0..) |prime, col| {
                hi += prime;
                if (col < first) lo += prime;
            }
            var header: format.Header = .{ .layer = meta.layer_id, .total = meta.rows, .lo = lo, .hi = hi, .row_bytes = format.row_bytes };
            try guard.coverage(header, meta, rank, world);
            header.lo = lo -| 1;
            header.hi = hi + 1;
            header.total += 1;
            try guard.coverage(header, meta, rank, world);
            header.lo = lo + 1;
            try t.expectError(error.BadEngramShard, guard.coverage(header, meta, rank, world));
            header.lo = lo;
            header.hi = hi - 1;
            try t.expectError(error.BadEngramShard, guard.coverage(header, meta, rank, world));
        };
    }
}
