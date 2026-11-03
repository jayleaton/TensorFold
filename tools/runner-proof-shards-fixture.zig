//! Disposable real test functions for the parent to exercise through the native server runner.
const std = @import("std");

fn mode(wanted: []const u8) bool {
    const value = std.c.getenv("PROOF_SHARD_FIXTURE_MODE") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), wanted);
}

test {
    try std.testing.expect(true);
}

test "runner proof: fixture pass" {
    try std.testing.expectEqual(@as(u32, 42), @as(u32, 6 * 7));
}

test "compute runner proof: fixture substring population" {
    try std.testing.expect(true);
}

test "runner proof: fixture skip" {
    if (mode("skip")) return error.SkipZigTest;
}

test "runner proof: fixture failure" {
    if (mode("fail")) return error.PlantedProofFailure;
}

test "runner proof: fixture logged error" {
    if (mode("log")) std.log.err("planted proof error", .{});
}

test "runner proof: fixture allocation leak" {
    if (mode("leak")) {
        const bytes = try std.testing.allocator.alloc(u8, 1);
        bytes[0] = 1;
        std.debug.print("planted leaked byte {d}\n", .{bytes[0]});
    }
}

test "runner proof: fixture panic" {
    if (mode("panic")) @panic("planted proof panic");
}

test "runner proof: fixture JSON \"quoted\" \\ slash\t tab\n newline" {
    try std.testing.expect(true);
}
