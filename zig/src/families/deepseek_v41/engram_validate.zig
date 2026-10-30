//! Opt-in startup checks that each packed Engram shard covers the hasher's rank-owned rows.
const std = @import("std");
const format = @import("engram_rows.zig");
const hash = @import("engram_hash.zig");

/// TF_DSV41_ENGRAM_VALIDATE: off by default; no file is opened before CUDA boot unless explicitly enabled.
pub fn fromEnv(io: std.Io, rank: u32, world: u32) !void {
    const raw = if (std.c.getenv("TF_DSV41_ENGRAM_VALIDATE")) |v| std.mem.span(v) else null;
    const dir = if (std.c.getenv("TF_DSV41_ENGRAM_DIR")) |v| std.mem.span(v) else null;
    try apply(io, raw, dir, rank, world, &hash.layers);
}

/// Checks only metadata; table bytes and the served read/arithmetic paths are unchanged.
pub fn apply(io: std.Io, raw: ?[]const u8, dir: ?[]const u8, rank: u32, world: u32, layers: []const hash.Layer) !void {
    const mode = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (mode.len == 0 or std.mem.eql(u8, mode, "0")) return;
    if (!std.mem.eql(u8, mode, "1")) return error.BadKnob;
    if (world == 0 or rank >= world) return error.BadEngramShard;
    const root = dir orelse return error.MissingEngramDirectory;
    for (layers) |meta| {
        var buf: [4096]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/engram-l{d}-r{d}of{d}.bin", .{ root, meta.layer_id, rank, world });
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        var bytes: [48]u8 = undefined;
        if (try file.readPositionalAll(io, &bytes, 0) != bytes.len) return error.BadEngramShard;
        const header = try format.Header.parse(&bytes, try file.length(io));
        try coverage(header, meta, rank, world);
    }
}

fn coverage(header: format.Header, meta: hash.Layer, rank: u32, world: u32) !void {
    if (header.layer != meta.layer_id or header.total < meta.rows) return error.BadEngramShard;
    const part = (hash.columns + @as(u64, world) - 1) / world;
    const first: usize = @intCast(@min(rank * part, hash.columns));
    const end: usize = @intCast(@min(first + part, hash.columns));
    if (first == end) return;
    const lo = meta.offsets[first];
    const last = end - 1;
    const hi = @as(u64, meta.offsets[last]) + meta.primes[last];
    if (header.lo > lo or header.hi < hi) return error.BadEngramShard;
}
