//! CPU-only rank agreement before Engram validation can refuse CUDA startup.
const std = @import("std");
const tp = @import("tp");
const guard = @import("engram_validate.zig");
const hash = @import("engram_hash.zig");

pub fn fromEnv(io: std.Io, cfg: tp.Config) !?tp.Bootstrap {
    const raw = if (std.c.getenv("TF_DSV41_ENGRAM_VALIDATE")) |v| std.mem.span(v) else null;
    const dir = if (std.c.getenv("TF_DSV41_ENGRAM_DIR")) |v| std.mem.span(v) else null;
    var diagnostic: guard.Diagnostic = .{};
    return start(io, raw, dir, cfg, &hash.layers, &diagnostic) catch |err| {
        if (diagnostic.len != 0) std.log.err("Engram preflight: {s}", .{diagnostic.text()});
        return err;
    };
}

/// The returned sockets continue into Session.openBootstrapped; off does not open files or sockets.
pub fn start(io: std.Io, raw: ?[]const u8, dir: ?[]const u8, cfg: tp.Config, layers: []const hash.Layer, diagnostic: *guard.Diagnostic) !?tp.Bootstrap {
    if (!(guard.enabled(raw) catch true)) return null;
    const local: ?anyerror = blk: {
        guard.applyDiagnostic(io, raw, dir, cfg.rank, cfg.world, layers, diagnostic) catch |err| {
            if (diagnostic.len == 0) {
                const message = try std.fmt.bufPrint(&diagnostic.bytes, "rank {d}: {s}", .{ cfg.rank, @errorName(err) });
                diagnostic.len = message.len;
            }
            break :blk err;
        };
        break :blk null;
    };
    var b = try tp.Bootstrap.open(cfg);
    errdefer b.close();
    try agree(&b, local, diagnostic);
    return b;
}

pub fn agree(b: *tp.Bootstrap, local: ?anyerror, diagnostic: *guard.Diagnostic) !void {
    var bytes: [tp.bootstrap.max_ranks * 8192]u8 = undefined;
    var blobs: [tp.bootstrap.max_ranks][]const u8 = undefined;
    const all = try b.allGather(if (local != null) diagnostic.text() else "", &bytes, &blobs);
    if (local) |err| return err;
    for (all) |message| if (message.len != 0) {
        if (message.len > diagnostic.bytes.len) return error.TooLarge;
        @memcpy(diagnostic.bytes[0..message.len], message);
        diagnostic.len = message.len;
        return error.PeerEngramValidationFailed;
    };
}
