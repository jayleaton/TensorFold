//! Rank-local malformed shards must stop both ranks before the GPU startup seam.
const std = @import("std");
const t = std.testing;
const tp = @import("tp");
const guard = @import("engram_validate.zig");
const preflight = @import("engram_preflight.zig");
const fixtures = @import("engram_validate_test.zig");

const Result = struct {
    err: ?anyerror = null,
    diagnostic: guard.Diagnostic = .{},
    startup_reached: bool = false,
};

fn rankStart(cfg: tp.Config, root: []const u8, result: *Result) void {
    const metas = .{ fixtures.metadata(1), fixtures.metadata(14) };
    var boot = preflight.start(t.io, "1", root, cfg, &metas, &result.diagnostic) catch |err| {
        result.err = err;
        return;
    };
    defer if (boot) |*b| b.close();
    result.startup_reached = true;
    // Session startup consumes these same sockets; a host-only continuation verifies their ownership.
    if (boot) |*b| b.agree("session continuation", "test") catch |err| {
        result.err = err;
    };
}

test "Engram preflight refuses both ranks for either rank-local failure before CUDA startup" {
    for ([_]?u32{ 0, 1, 2, null }) |bad_rank| {
        var dirs = [_]t.TmpDir{ t.tmpDir(.{}), t.tmpDir(.{}) };
        defer for (&dirs) |*dir| dir.cleanup();
        var roots: [2][:0]const u8 = undefined;
        for (&dirs, 0..) |*dir, rank| {
            roots[rank] = try dir.dir.realPathFileAlloc(t.io, ".", t.allocator);
            for ([_]u32{ 1, 14 }) |layer| {
                if (bad_rank == 2 and rank == 1 and layer == 14) continue;
                const invalid = bad_rank == @as(u32, @intCast(rank)) or (bad_rank == 2 and rank == 0);
                const hi: u64 = (rank + 1) * 24 - @intFromBool(invalid and layer == 14);
                try fixtures.shard(dir.dir, fixtures.metadata(layer), @intCast(rank), rank * 24, hi, 48, layer);
            }
        }
        defer for (roots) |root| t.allocator.free(root);
        const listener = try tp.bootstrap.testListener();
        tp.sock.close(listener.fd);
        const cfg: tp.Config = .{ .world = 2, .rank = 0, .link_port = listener.port, .connect_timeout_ns = 2 * std.time.ns_per_s };
        var follower = cfg;
        follower.rank = 1;
        var results: [2]Result = .{ .{}, .{} };
        const before = tp.sock.nowNs();
        const thread = try std.Thread.spawn(.{}, rankStart, .{ follower, roots[1], &results[1] });
        rankStart(cfg, roots[0], &results[0]);
        thread.join();
        try t.expect(tp.sock.nowNs() - before < 2 * std.time.ns_per_s);
        if (bad_rank == 2) {
            try t.expectEqual(error.BadEngramShard, results[0].err.?);
            try t.expectEqual(error.FileNotFound, results[1].err.?);
            for (&results, 0..) |*result, rank| {
                try t.expect(!result.startup_reached);
                try t.expect(std.mem.indexOf(u8, result.diagnostic.text(), roots[rank]) != null);
                try t.expect(std.mem.indexOf(u8, result.diagnostic.text(), @errorName(result.err.?)) != null);
            }
        } else if (bad_rank) |bad| {
            for (&results) |*result| {
                try t.expect(result.err != null);
                try t.expect(!result.startup_reached);
                try t.expect(std.mem.indexOf(u8, result.diagnostic.text(), roots[bad]) != null);
                try t.expect(std.mem.indexOf(u8, result.diagnostic.text(), "expected total >= 48") != null);
                try t.expect(std.mem.indexOf(u8, result.diagnostic.text(), "actual layer 14") != null);
            }
            try t.expectEqual(error.BadEngramShard, results[bad].err.?);
            try t.expectEqual(error.PeerEngramValidationFailed, results[1 - bad].err.?);
            try t.expectEqualStrings(results[0].diagnostic.text(), results[1].diagnostic.text());
        } else {
            for (results) |result| {
                try t.expectEqual(@as(?anyerror, null), result.err);
                try t.expect(result.startup_reached);
            }
        }
    }
}

test "Engram preflight off avoids rendezvous even with an absent peer and invalid world" {
    var diagnostic: guard.Diagnostic = .{};
    for ([_]?[]const u8{ null, "", "0", " 0 " }) |raw| {
        try t.expect(try preflight.start(t.io, raw, "missing", .{ .world = 0 }, &.{}, &diagnostic) == null);
    }
}
