//! The fate channel between two ranks over real TCP: a failure or a vanished rank ends both, a normal stop ends neither.
const std = @import("std");
const sock = @import("sock.zig");
const fate = @import("fate.zig");
const bootstrap = @import("bootstrap.zig");
const Config = @import("config.zig").Config;
const Fate = fate.Fate;

var exits: std.atomic.Value(u32) = .init(0);
var hooks: std.atomic.Value(u32) = .init(0);

fn recordExit(code: u8) void {
    std.debug.assert(code == fate.exit_code);
    _ = exits.fetchAdd(1, .acq_rel);
}

fn hook(_: *anyopaque, reason: []const u8) void {
    std.debug.assert(reason.len > 0);
    _ = hooks.fetchAdd(1, .acq_rel);
}

const grace = 50 * std.time.ns_per_ms;

/// Two connected ranks' fate channels (rank 0 first), started, with recorded exits and hooks.
fn pair(fates: *[2]Fate) !void {
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
    const a = try sock.connect("127.0.0.1", l.port, deadline);
    const b = try sock.accept(l.fd, deadline);
    var fds0: [fate.max_ranks]sock.Fd = @splat(-1);
    var fds1: [fate.max_ranks]sock.Fd = @splat(-1);
    fds0[1] = b;
    fds1[0] = a;
    for (fates, [2][fate.max_ranks]sock.Fd{ fds0, fds1 }, 0..) |*f, fds, r| {
        f.* = Fate.init(.{ .world = 2, .rank = @intCast(r), .grace_ns = grace, .report_ns = 20 * std.time.ns_per_ms }, fds);
        f.exit_fn = recordExit;
        f.usable = fakeUsable;
        f.onFail(.{ .ctx = f, .call = hook });
    }
    for (fates) |*f| try f.start();
}

fn fakeUsable() u64 {
    return 123 << 20;
}

fn waitExits(want: u32) !u64 {
    const t0 = sock.nowNs();
    while (exits.load(.acquire) < want) {
        if (sock.nowNs() - t0 > 5 * std.time.ns_per_s) return error.Timeout;
        sock.sleepNs(std.time.ns_per_ms);
    }
    return sock.nowNs() - t0;
}

fn reset() void {
    exits.store(0, .release);
    hooks.store(0, .release);
}

test "rank 1's failure ends both ranks with its reason" {
    reset();
    var fates: [2]Fate = undefined;
    try pair(&fates);
    defer for (&fates) |*f| f.close();
    sock.sleepNs(60 * std.time.ns_per_ms);
    const mem = fates[0].peerMem(1) orelse return error.NoReport;
    try std.testing.expectEqual(@as(u64, 123 << 20), mem.bytes);
    fates[1].fatal("prefill", error.OutOfMemory);
    const took = try waitExits(2);
    try std.testing.expect(took < grace + 500 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(u32, 2), hooks.load(.acquire));
    try std.testing.expect(std.mem.indexOf(u8, fates[0].reason(), "rank 1 failed in prefill: OutOfMemory") != null);
}

test "rank 0's failure ends rank 1" {
    reset();
    var fates: [2]Fate = undefined;
    try pair(&fates);
    defer for (&fates) |*f| f.close();
    fates[0].fatal("window", error.Unexpected);
    _ = try waitExits(2);
    try std.testing.expect(std.mem.indexOf(u8, fates[1].reason(), "rank 0 failed in window") != null);
}

test "a rank that vanishes without a word ends the other" {
    reset();
    var fates: [2]Fate = undefined;
    try pair(&fates);
    defer fates[0].close();
    // rank 1 killed: its socket closes with no FAIL frame (its own fate thread ends quietly first)
    fates[1].expectStop();
    fates[1].close();
    _ = try waitExits(1);
    try std.testing.expect(std.mem.indexOf(u8, fates[0].reason(), "rank 1 closed the fate channel") != null);
}

test "a normal stop ends nobody" {
    reset();
    var fates: [2]Fate = undefined;
    try pair(&fates);
    fates[0].expectStop();
    fates[1].expectStop();
    fates[1].close();
    fates[0].close();
    sock.sleepNs(2 * grace);
    try std.testing.expectEqual(@as(u32, 0), exits.load(.acquire));
}
