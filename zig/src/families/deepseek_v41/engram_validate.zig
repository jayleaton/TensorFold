//! Opt-in startup checks that each packed Engram shard covers the hasher's rank-owned rows.
const std = @import("std");
const format = @import("engram_rows.zig");
const hash = @import("engram_hash.zig");

pub const Diagnostic = struct {
    bytes: [8192]u8 = undefined,
    len: usize = 0,

    pub fn text(d: *const Diagnostic) []const u8 {
        return d.bytes[0..d.len];
    }

    fn set(d: *Diagnostic, path: []const u8, rank: u32, meta: hash.Layer, bounds: Bounds, h: ?format.Header, err: anyerror) void {
        const actual = h orelse format.Header{ .layer = 0, .total = 0, .lo = 0, .hi = 0, .row_bytes = 0 };
        const message = std.fmt.bufPrint(
            &d.bytes,
            "{s}: rank {d} layer {d}: {s}; expected total >= {d}, rows [{d}, {d}); actual {s}layer {d}, total {d}, rows [{d}, {d})",
            .{ path, rank, meta.layer_id, @errorName(err), meta.rows, bounds.lo, bounds.hi, if (h == null) "unavailable: " else "", actual.layer, actual.total, actual.lo, actual.hi },
        ) catch unreachable;
        d.len = message.len;
    }
};

pub fn enabled(raw: ?[]const u8) !bool {
    const mode = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (mode.len == 0 or std.mem.eql(u8, mode, "0")) return false;
    if (!std.mem.eql(u8, mode, "1")) return error.BadKnob;
    return true;
}

/// Checks only metadata; table bytes and the served read/arithmetic paths are unchanged.
pub fn apply(io: std.Io, raw: ?[]const u8, dir: ?[]const u8, rank: u32, world: u32, layers: []const hash.Layer) !void {
    var diagnostic: Diagnostic = .{};
    try applyDiagnostic(io, raw, dir, rank, world, layers, &diagnostic);
}

pub fn applyDiagnostic(io: std.Io, raw: ?[]const u8, dir: ?[]const u8, rank: u32, world: u32, layers: []const hash.Layer, diagnostic: *Diagnostic) !void {
    diagnostic.len = 0;
    if (!try enabled(raw)) return;
    if (world == 0 or rank >= world) return error.BadEngramShard;
    const root = dir orelse return error.MissingEngramDirectory;
    for (layers) |meta| {
        var buf: [4096]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/engram-l{d}-r{d}of{d}.bin", .{ root, meta.layer_id, rank, world });
        try checkFile(io, path, meta, rank, world, diagnostic);
    }
}

fn checkFile(io: std.Io, path: []const u8, meta: hash.Layer, rank: u32, world: u32, diagnostic: *Diagnostic) !void {
    var actual: ?format.Header = null;
    checkShard(io, path, meta, rank, world, &actual) catch |err| {
        diagnostic.set(path, rank, meta, required(meta, rank, world), actual, err);
        return err;
    };
}

fn checkShard(io: std.Io, path: []const u8, meta: hash.Layer, rank: u32, world: u32, actual: *?format.Header) !void {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var bytes: [48]u8 = undefined;
    if (try file.readPositionalAll(io, &bytes, 0) != bytes.len) return error.BadEngramShard;
    actual.* = .{
        .layer = std.mem.readInt(u64, bytes[8..16], .little),
        .lo = std.mem.readInt(u64, bytes[16..24], .little),
        .hi = std.mem.readInt(u64, bytes[24..32], .little),
        .total = std.mem.readInt(u64, bytes[32..40], .little),
        .row_bytes = std.mem.readInt(u64, bytes[40..48], .little),
    };
    const header = try format.Header.parse(&bytes, try file.length(io));
    try coverage(header, meta, rank, world);
}

pub const Bounds = struct { lo: u64, hi: u64 };

pub fn required(meta: hash.Layer, rank: u32, world: u32) Bounds {
    const part = (hash.columns + @as(u64, world) - 1) / world;
    const first: usize = @intCast(@min(rank * part, hash.columns));
    const end: usize = @intCast(@min(first + part, hash.columns));
    if (first == end) return .{ .lo = meta.rows, .hi = meta.rows };
    return .{ .lo = meta.offsets[first], .hi = @as(u64, meta.offsets[end - 1]) + meta.primes[end - 1] };
}

pub fn coverage(header: format.Header, meta: hash.Layer, rank: u32, world: u32) !void {
    if (header.layer != meta.layer_id or header.total < meta.rows) return error.BadEngramShard;
    const bounds = required(meta, rank, world);
    if (bounds.lo == bounds.hi) return;
    if (header.lo > bounds.lo or header.hi < bounds.hi) return error.BadEngramShard;
}
