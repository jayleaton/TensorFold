//! Peer loss at two ranks: each rank's watchdog on the control link and the gate's one-int agreement.
const std = @import("std");
const envs = @import("cuda_env.zig");
const cuda = @import("cuda");
const Comm = @import("cuda_comm.zig").Comm;

// -- peer loss ----------------------------------------------------------------------------------------------------

/// Each rank's watchdog: a closed link or a missed call deadline means the peer is lost; the process exits (code 3).
pub const Watch = struct {
    io: std.Io,
    fd: std.posix.socket_t,
    rank: u32,
    /// a call's deadline at least, in seconds (TF_TP_TIMEOUT_S, 300 by default: a long prompt's chunks)
    base_s: u64 = 300,
    deadline_ms: std.atomic.Value(i64) = .init(0),
    done: std.atomic.Value(bool) = .init(false),
    /// rank 1 waits on the link with no stream open: the link closing then is rank 0's stop, not a loss
    idle: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    /// what a lost peer does: abort `comm` and exit; the tests record the reason instead
    lost: *const fn (w: *Watch, why: []const u8) void = abortAndExit,
    comm: ?*const Comm = null,

    pub fn start(w: *Watch) !void {
        keepalive(w.fd);
        if (envs.get("TF_TP_TIMEOUT_S")) |v| w.base_s = std.fmt.parseInt(u64, v, 10) catch w.base_s;
        w.thread = try std.Thread.spawn(.{}, run, .{w});
    }

    /// A clean stop is under way: the other rank may close the link now.
    pub fn quiet(w: *Watch) void {
        w.done.store(true, .release);
    }

    pub fn deinit(w: *Watch) void {
        w.done.store(true, .release);
        if (w.thread) |t| t.join();
        w.thread = null;
    }

    fn now(w: *const Watch) i64 {
        return @intCast(@divTrunc(std.Io.Timestamp.now(w.io, .awake).toNanoseconds(), std.time.ns_per_ms));
    }

    /// A call starts that must end within `seconds`.
    pub fn arm(w: *Watch, seconds: u64) void {
        w.deadline_ms.store(w.now() + @as(i64, @intCast(seconds * 1000)), .release);
    }

    pub fn disarm(w: *Watch) void {
        w.deadline_ms.store(0, .release);
    }

    fn run(w: *Watch) void {
        const rdhup: i16 = 0x2000; // POLLRDHUP: the peer closed its end
        while (!w.done.load(.acquire)) {
            var fds = [_]std.posix.pollfd{.{ .fd = w.fd, .events = rdhup | std.posix.POLL.IN, .revents = 0 }};
            const n = std.posix.poll(&fds, 200) catch 0;
            // readable data alone is a frame for the loop: wait for it to be read, not for the peer
            if (n > 0 and fds[0].revents & (rdhup | std.posix.POLL.HUP | std.posix.POLL.ERR | std.posix.POLL.NVAL) == 0) {
                std.Io.sleep(w.io, .fromMilliseconds(50), .awake) catch {};
            }
            if (n > 0 and fds[0].revents & (rdhup | std.posix.POLL.HUP | std.posix.POLL.ERR | std.posix.POLL.NVAL) != 0) {
                // with bytes still unread (rank 0's stop) the loop gets 2 s to read them and quiet the watch
                var grace: u32 = 0;
                const unread = fds[0].revents & std.posix.POLL.IN != 0;
                while (unread and grace < 20 and !w.done.load(.acquire)) : (grace += 1) std.Io.sleep(w.io, .fromMilliseconds(100), .awake) catch {};
                if (w.done.load(.acquire)) return;
                if (w.idle.load(.acquire)) {
                    // the follow loop reads the close and returns; give it a moment, then leave cleanly anyway
                    var settle: u32 = 0;
                    while (settle < 20 and !w.done.load(.acquire)) : (settle += 1) std.Io.sleep(w.io, .fromMilliseconds(100), .awake) catch {};
                    if (w.done.load(.acquire)) return;
                    std.log.info("rank {d}: the other rank closed the link while no stream was open: stopping", .{w.rank});
                    std.c._exit(0);
                }
                return w.lost(w, "the other rank closed the control link (it exited, or the link is down)");
            }
            const d = w.deadline_ms.load(.acquire);
            if (d != 0 and w.now() > d) return w.lost(w, "a call passed its deadline (TF_TP_TIMEOUT_S): the other rank stalled");
        }
    }

    /// Log and _exit at once; no ncclCommAbort, which can free the communicator under another thread's collective.
    fn abortAndExit(w: *Watch, why: []const u8) void {
        std.log.err("rank {d}: {s}; exiting", .{ w.rank, why });
        std.c._exit(3);
    }
};

/// The lane backend's gate at two ranks: prepare outcomes in one 1-int all-gather before the call's first collective.
pub const Agreement = struct {
    comm: *const Comm,
    stream: cuda.abi.Stream,
    device: cuda.DeviceBuffer,
    host: cuda.HostBuffer,
    watch: *Watch,
    count: u64 = 0,

    pub fn init(d: *const cuda.Driver, comm: *const Comm, stream: cuda.abi.Stream, watch: *Watch) !Agreement {
        var device = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer device.free();
        return .{ .comm = comm, .stream = stream, .device = device, .host = try cuda.HostBuffer.alloc(d, 16), .watch = watch };
    }

    pub fn deinit(a: *Agreement) void {
        a.host.free();
        a.device.free();
    }

    pub fn gate(a: *Agreement) @import("cuda_lanes.zig").Gate {
        return .{ .ptr = a, .agree = agreeFn, .fatal = fatalFn };
    }

    fn of(p: *anyopaque) *Agreement {
        return @ptrCast(@alignCast(p));
    }

    fn agreeFn(p: *anyopaque, ok: bool) anyerror!bool {
        const a = of(p);
        a.count += 1;
        const h = a.host.slice(i32);
        h[0] = @intFromBool(ok);
        try a.device.uploadAsync(0, a.host.bytes[0..4], a.stream);
        try a.comm.allGather(a.device.ptr, try a.device.at(8), 1, .i32, a.stream);
        try a.device.downloadAsync(8, a.host.bytes[8..16], a.stream);
        try cuda.Stream.synchronize(.{ .d = a.device.d, .handle = a.stream });
        return h[2] == 1 and h[3] == 1;
    }

    fn fatalFn(p: *anyopaque, what: []const u8, err: anyerror) void {
        const a = of(p);
        var buf: [160]u8 = undefined;
        const why = std.fmt.bufPrint(&buf, "{s} failed after the ranks agreed ({s})", .{ what, @errorName(err) }) catch "an engine call failed after the ranks agreed";
        a.watch.lost(a.watch, why);
    }
};

/// TCP keepalive on the control link: a link that goes down errs within ~11 s (5 s idle, 3 probes 2 s apart).
fn keepalive(fd: std.posix.socket_t) void {
    const posix = std.posix;
    const on: c_int = 1;
    posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.KEEPALIVE, std.mem.asBytes(&on)) catch {};
    for ([_]struct { u32, c_int }{ .{ 4, 5 }, .{ 5, 2 }, .{ 6, 3 } }) |o| { // TCP_KEEPIDLE, KEEPINTVL, KEEPCNT
        posix.setsockopt(fd, posix.IPPROTO.TCP, o[0], std.mem.asBytes(&o[1])) catch {};
    }
}
