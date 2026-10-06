//! The round plan from rank 0 to the other ranks over its own TCP connection (Python's planlink.TcpLink): one frame a plan, a 4-byte length then little-endian int64s.
const std = @import("std");
const sock = @import("sock.zig");
const builtin = @import("builtin");

pub const max_ranks = 64;
const more: u32 = 0x8000; // MSG_MORE: the length and the ints leave as one segment

pub const Error = error{ PeerClosed, TooLarge, NotRankZero, NotFollower };

comptime {
    std.debug.assert(builtin.cpu.arch.endian() == .little); // the ints go out as they are in memory
}

pub const PlanLink = struct {
    rank: u32,
    world: u32,
    /// Rank 0: rank i's socket at index i; the others: rank 0's at index 0 (the bootstrap's `releasePlan`).
    fds: [max_ranks]sock.Fd,
    sent: u64 = 0,
    received: u64 = 0,

    pub fn init(rank: u32, world: u32, fds: [max_ranks]sock.Fd) PlanLink {
        return .{ .rank = rank, .world = world, .fds = fds };
    }

    /// Rank 0: one plan to every other rank. Host only: nothing waits on a GPU.
    pub fn send(p: *PlanLink, ints: []const i64) Error!void {
        if (p.rank != 0) return error.NotRankZero;
        const bytes = std.mem.sliceAsBytes(ints);
        if (bytes.len > std.math.maxInt(u32)) return error.TooLarge;
        const len: u32 = @intCast(bytes.len);
        for (p.fds[1..p.world]) |fd| {
            if (std.c.send(fd, &len, 4, std.posix.MSG.NOSIGNAL | more) != 4) return error.PeerClosed;
            sock.sendAll(fd, bytes) catch return error.PeerClosed;
        }
        p.sent += 1;
    }

    /// The other ranks: the next plan into `out` (returned: the ints it holds). Blocks as long as rank 0 is idle; a
    /// rank 0 that is gone ends it (`error.PeerClosed`; the fate channel ends the process right after).
    pub fn recv(p: *PlanLink, out: []i64) Error![]i64 {
        if (p.rank == 0) return error.NotFollower;
        const fd = p.fds[0];
        var len: u32 = undefined;
        sock.recvExact(fd, std.mem.asBytes(&len), null) catch return error.PeerClosed;
        if (len % 8 != 0 or len / 8 > out.len) {
            // the frame is read and dropped, so the next plan still starts on a frame boundary
            var left: usize = len;
            var scratch: [4096]u8 = undefined;
            while (left > 0) : (left -= @min(left, scratch.len)) sock.recvExact(fd, scratch[0..@min(left, scratch.len)], null) catch return error.PeerClosed;
            return error.TooLarge;
        }
        const n = len / 8;
        sock.recvExact(fd, std.mem.sliceAsBytes(out[0..n]), null) catch return error.PeerClosed;
        p.received += 1;
        return out[0..n];
    }

    pub fn close(p: *PlanLink) void {
        for (&p.fds) |*fd| if (fd.* >= 0) {
            sock.close(fd.*);
            fd.* = -1;
        };
    }
};

test "plans arrive whole and in order; a closed rank 0 ends the follower" {
    const bootstrap = @import("bootstrap.zig");
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
    const a = try sock.connect("127.0.0.1", l.port, deadline);
    const b = try sock.accept(l.fd, deadline);
    var f0: [max_ranks]sock.Fd = @splat(-1);
    var f1: [max_ranks]sock.Fd = @splat(-1);
    f0[1] = b;
    f1[0] = a;
    var lead = PlanLink.init(0, 2, f0);
    var follow = PlanLink.init(1, 2, f1);
    defer follow.close();
    var big: [5000]i64 = undefined;
    for (&big, 0..) |*v, i| v.* = @as(i64, @intCast(i)) * -3 + 0x7FFF_0000_0000;
    try lead.send(&.{ 0x504C414E, 2, 1 });
    try lead.send(&big);
    try lead.send(&.{});
    var buf: [8192]i64 = undefined;
    try std.testing.expectEqualSlices(i64, &.{ 0x504C414E, 2, 1 }, try follow.recv(&buf));
    try std.testing.expectEqualSlices(i64, &big, try follow.recv(&buf));
    try std.testing.expectEqual(@as(usize, 0), (try follow.recv(&buf)).len);
    try std.testing.expectError(error.NotRankZero, follow.send(&.{1}));
    var small: [2]i64 = undefined;
    try lead.send(&.{ 1, 2, 3 });
    try std.testing.expectError(error.TooLarge, follow.recv(&small));
    lead.close();
    try std.testing.expectError(error.PeerClosed, follow.recv(&buf));
}
