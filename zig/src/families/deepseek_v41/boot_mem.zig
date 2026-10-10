//! A rank's boot memory: the rope tables uploaded from a scratch read freed before serving.

const std = @import("std");

const log = std.log.scoped(.dsv41);

/// rope.bin's header bytes, then the main and the compressed table (f32 [rows, 64] each)
pub const rope_header = 16;

/// Reads `path` into scratch memory, gives each table's `want` bytes to `sink.table(i, bytes)`, then `sink.done()`.
pub fn uploadRope(gpa: std.mem.Allocator, io: std.Io, path: []const u8, want: usize, sink: anytype) !void {
    var scratch = std.heap.ArenaAllocator.init(gpa);
    defer scratch.deinit();
    const rope = try std.Io.Dir.cwd().readFileAlloc(io, path, scratch.allocator(), .unlimited);
    const half = (rope.len -| rope_header) / 2;
    if (half < want) {
        log.err("rope.bin holds {d} positions, the limit needs {d} (dsv41_rope_tables.py --rows)", .{ half / 256, want / 256 });
        return error.RopeTooShort;
    }
    for (0..2) |i| try sink.table(i, rope[rope_header + i * half ..][0..want]);
    try sink.done();
}

test "the rope tables reach the sink byte for byte and the read is freed" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const rows = 5;
    const half = rows * 64 * 4;
    const data = try gpa.alloc(u8, rope_header + 2 * half);
    defer gpa.free(data);
    for (data, 0..) |*b, i| b.* = @truncate(i *% 131 +% 7);
    try tmp.dir.writeFile(io, .{ .sub_path = "rope.bin", .data = data });
    const path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/rope.bin", .{&tmp.sub_path});
    defer gpa.free(path);
    const Sink = struct {
        got: [2][]const u8 = undefined,
        copies: [2][4 * 64 * 4]u8 = undefined,
        fn table(s: *@This(), i: usize, bytes: []const u8) !void {
            @memcpy(s.copies[i][0..bytes.len], bytes);
            s.got[i] = s.copies[i][0..bytes.len];
        }
        fn done(_: *@This()) !void {}
    };
    var sink: Sink = .{};
    const want = 4 * 64 * 4;
    try uploadRope(gpa, io, path, want, &sink);
    try std.testing.expectEqualSlices(u8, data[rope_header..][0..want], sink.got[0]);
    try std.testing.expectEqualSlices(u8, data[rope_header + half ..][0..want], sink.got[1]);
}
