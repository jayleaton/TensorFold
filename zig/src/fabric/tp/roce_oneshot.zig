//! TF_TP_ROCE_ONESHOT=1: all-gathers up to a cap by upstream's one-shot RoCE gather (cuda/roce.zig), the rest inner.
const std = @import("std");
const cuda = @import("cuda");
const collective = @import("collective.zig");
const Bootstrap = @import("bootstrap.zig").Bootstrap;
const Collective = collective.Collective;
const DevicePtr = collective.DevicePtr;
const DType = collective.DType;
const Op = collective.Op;
const Error = collective.Error;
const Gather = cuda.roce.Gather;

/// Upstream's group seam over our bootstrap: `gather` of one value a rank, rank order (the stream is unused).
const Link = struct {
    b: *Bootstrap,
    rank: usize,
    world: usize,

    pub fn gather(l: *const Link, comptime T: type, _: *const cuda.Driver, _: cuda.Stream, mine: T, out: []T) !void {
        if (out.len != l.world) return error.BadWorld;
        var blobs: [2][]const u8 = undefined;
        var buf: [2 * @sizeOf(T)]u8 = undefined;
        const all = try l.b.allGather(std.mem.asBytes(&mine), &buf, &blobs);
        for (all, out) |blob, *o| {
            if (blob.len != @sizeOf(T)) return error.BadRecord;
            @memcpy(std.mem.asBytes(o), blob);
        }
    }
};

/// Whether a gather of `bytes` a rank goes one-shot: within both caps, whole 16-byte packs (size alone: ranks agree).
pub fn routed(cap: usize, gather_fits: bool, bytes: usize) bool {
    return bytes > 0 and bytes <= cap and bytes % 16 == 0 and gather_fits;
}

pub const OneShot = struct {
    gather: *Gather,
    inner: Collective,
    cap: usize,
    aborted: std.atomic.Value(bool) = .init(false),
    /// One-shot gathers this rank launched or captured (logs, the bench).
    calls: u64 = 0,

    /// Both ranks or neither: an empty `image` or a rank without verbs / an active HCA refuses on every rank.
    pub fn open(d: *const cuda.Driver, b: *Bootstrap, image: []const u8, cap: usize, inner: Collective) !OneShot {
        if (b.world != 2) return error.Unsupported;
        if (image.len == 0) std.log.warn("[tensorfold] tp: TF_TP_ROCE_ONESHOT=1 but this build has no roce_gather kernel image", .{});
        var stream = try cuda.Stream.init(d, true);
        defer stream.deinit();
        const link: Link = .{ .b = b, .rank = b.rank, .world = b.world };
        const io = std.Io.Threaded.global_single_threaded.io();
        const g = try Gather.init(std.heap.page_allocator, io, d, &link, stream, image.len > 0, cap, image) orelse {
            std.log.warn("[tensorfold] tp: TF_TP_ROCE_ONESHOT=1: the one-shot gather is not up on both ranks (kernel image, libibverbs, TF_ROCE_HCA / NCCL_IB_HCA, an active port); refusing to start", .{});
            return error.OneshotUnavailable;
        };
        return .{ .gather = g, .inner = inner, .cap = @min(cap, g.max_bytes) };
    }

    pub fn deinit(o: *OneShot) void {
        o.gather.deinit();
    }

    /// The GPU or the proxy recorded a failure (a timed-out wait poisons the exchange).
    pub fn failed(o: *OneShot) bool {
        if (o.aborted.load(.acquire)) return true;
        o.gather.check() catch return true;
        return false;
    }

    pub fn iface(o: *OneShot) Collective {
        return .{ .ptr = o, .vtable = &vtable };
    }

    const vtable: Collective.VTable = .{
        .kind = .roce,
        .rank = rankOf,
        .world = worldOf,
        .all_reduce = allReduce,
        .all_gather = allGather,
        .exchange = exchange,
        .send = send,
        .recv = recv,
        .barrier = barrier,
        .check = check,
        .abort = abort,
        .all_gather_v = allGatherV,
        .var_fits = varFits,
        .var_agreed = varAgreed,
    };

    fn self_(ptr: *anyopaque) *OneShot {
        return @ptrCast(@alignCast(ptr));
    }

    fn allGather(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, stream: collective.Stream) Error!void {
        const o = self_(ptr);
        const bytes = count * t.size();
        if (!routed(o.cap, o.gather.fits(bytes), bytes)) return o.inner.allGather(s, r, count, t, stream);
        if (o.aborted.load(.acquire)) return error.Aborted;
        // a misaligned buffer is an error, never this rank alone on the inner path (the peer would wait one-shot)
        o.gather.gather(stream, s, r, bytes) catch |e| return switch (e) {
            error.Misaligned => error.Invalid,
            else => error.BackendFailed,
        };
        o.calls += 1;
    }

    fn check(ptr: *anyopaque) Error!void {
        const o = self_(ptr);
        o.gather.check() catch return error.PeerFailed;
        if (o.aborted.load(.acquire)) return error.Aborted;
        return o.inner.check();
    }

    /// New one-shot launches refuse; a kernel already waiting ends at its spin limit or with the process (fail-fast).
    fn abort(ptr: *anyopaque) void {
        const o = self_(ptr);
        o.aborted.store(true, .release);
        o.inner.abort();
    }

    fn rankOf(ptr: *anyopaque) u32 {
        return self_(ptr).inner.rank();
    }

    fn worldOf(ptr: *anyopaque) u32 {
        return self_(ptr).inner.world();
    }

    fn allReduce(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, op: Op, stream: collective.Stream) Error!void {
        return self_(ptr).inner.allReduce(s, r, count, t, op, stream);
    }

    fn allGatherV(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, stride: usize, lens: DevicePtr, t: DType, stream: collective.Stream) Error!void {
        return self_(ptr).inner.allGatherV(s, r, stride, lens, t, stream);
    }

    fn varFits(ptr: *anyopaque, bytes: usize) bool {
        return self_(ptr).inner.varFits(bytes);
    }

    fn varAgreed(ptr: *anyopaque) Error!bool {
        return self_(ptr).inner.varAgreed();
    }

    fn exchange(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, peer: u32, stream: collective.Stream) Error!void {
        return self_(ptr).inner.exchange(s, r, count, t, peer, stream);
    }

    fn send(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: collective.Stream) Error!void {
        return self_(ptr).inner.send(buf, count, t, peer, stream);
    }

    fn recv(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: collective.Stream) Error!void {
        return self_(ptr).inner.recv(buf, count, t, peer, stream);
    }

    fn barrier(ptr: *anyopaque) Error!void {
        return self_(ptr).inner.barrier();
    }
};

test "one-shot route: the size alone, whole packs, both caps" {
    try std.testing.expect(routed(64 << 10, true, 8192));
    try std.testing.expect(routed(64 << 10, true, 64 << 10));
    try std.testing.expect(!routed(64 << 10, true, (64 << 10) + 16));
    try std.testing.expect(!routed(64 << 10, true, 8));
    try std.testing.expect(!routed(64 << 10, true, 0));
    try std.testing.expect(!routed(64 << 10, false, 8192));
}
