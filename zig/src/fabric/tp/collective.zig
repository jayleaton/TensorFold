//! The tensor-parallel communicator every backend implements: collectives enqueued on the caller's stream, graph-capturable, rank-order sums.
const std = @import("std");
const cuda = @import("cuda");

pub const DevicePtr = cuda.abi.DevicePtr;
pub const Stream = cuda.abi.Stream;

pub const Error = error{ Aborted, Unsupported, Invalid, PeerFailed, Timeout, BackendFailed };

/// Element types the collectives move (`reduce` takes the float ones and the 32/64-bit integers).
pub const DType = enum(u8) {
    u8,
    i32,
    i64,
    f16,
    bf16,
    f32,
    f64,

    pub fn size(t: DType) usize {
        return switch (t) {
            .u8 => 1,
            .f16, .bf16 => 2,
            .i32, .f32 => 4,
            .i64, .f64 => 8,
        };
    }
};

pub const Op = enum(u8) { sum, max, min };

/// Which implementation is behind a `Collective` (logs, bench labels).
pub const Kind = enum { nccl, mailbox, hybrid, roce, host };

pub const Collective = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        kind: Kind,
        rank: *const fn (ptr: *anyopaque) u32,
        world: *const fn (ptr: *anyopaque) u32,
        /// recv[count] <- the element-wise reduction of every rank's send[count], added in rank order.
        all_reduce: *const fn (ptr: *anyopaque, send: DevicePtr, recv: DevicePtr, count: usize, t: DType, op: Op, stream: Stream) Error!void,
        /// recv[world * count] <- every rank's send[count], rank 0's first (NCCL's layout).
        all_gather: *const fn (ptr: *anyopaque, send: DevicePtr, recv: DevicePtr, count: usize, t: DType, stream: Stream) Error!void,
        /// Two ranks trade: this rank's send[count] lands in `peer`'s recv[count] and the peer's in ours (one group).
        exchange: *const fn (ptr: *anyopaque, send: DevicePtr, recv: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void,
        send: *const fn (ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void,
        recv: *const fn (ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void,
        /// Host: returns once every rank has called it and this rank's earlier collectives have finished.
        barrier: *const fn (ptr: *anyopaque) Error!void,
        /// Host: a failure the transport recorded since the last check (a peer that timed out, an NCCL async error).
        check: *const fn (ptr: *anyopaque) Error!void,
        /// Any thread: end pending and later work at once; safe while another thread is inside a call.
        abort: *const fn (ptr: *anyopaque) void,
        /// recv [world][stride]: rank r's first lens[r] bytes land at recv + r x stride (lens: device int32 [world] of bytes, the
        /// same on every rank, each <= the stride's bytes); bytes past a length are unspecified. Null: `all_gather` of whole strides.
        all_gather_v: ?*const fn (ptr: *anyopaque, send: DevicePtr, recv: DevicePtr, stride: usize, lens: DevicePtr, t: DType, stream: Stream) Error!void = null,
        /// Whether `all_gather_v` moves only the lengths for shards of `bytes` on this rank's transport (rank-local). Null: never.
        var_fits: ?*const fn (ptr: *anyopaque, bytes: usize) bool = null,
        /// Host: whether every rank's transport takes device lengths (`var_fits(16)` on all), agreed once per communicator by an
        /// int all-gather and kept on the implementation. Null: false.
        var_agreed: ?*const fn (ptr: *anyopaque) Error!bool = null,
        /// send's rows of each peer's `rowShare` to it; recv [world - 1][ceil(rows / world)][width]: theirs of ours
        scatter_rows: ?*const fn (ptr: *anyopaque, send: DevicePtr, recv: DevicePtr, rows: usize, width: usize, t: DType, stream: Stream) Error!void = null,
    };

    pub fn kind(c: Collective) Kind {
        return c.vtable.kind;
    }

    pub fn rank(c: Collective) u32 {
        return c.vtable.rank(c.ptr);
    }

    pub fn world(c: Collective) u32 {
        return c.vtable.world(c.ptr);
    }

    pub fn allReduce(c: Collective, src: DevicePtr, dst: DevicePtr, count: usize, t: DType, op: Op, stream: Stream) Error!void {
        return c.vtable.all_reduce(c.ptr, src, dst, count, t, op, stream);
    }

    pub fn allGather(c: Collective, src: DevicePtr, dst: DevicePtr, count: usize, t: DType, stream: Stream) Error!void {
        return c.vtable.all_gather(c.ptr, src, dst, count, t, stream);
    }

    /// `allGather` of `stride` elements a rank where only rank r's first lens[r] bytes matter (see `VTable.all_gather_v`):
    /// a transport that takes device lengths moves just those, any other the whole strides (the same bytes where lens says).
    pub fn allGatherV(c: Collective, src: DevicePtr, dst: DevicePtr, stride: usize, lens: DevicePtr, t: DType, stream: Stream) Error!void {
        if (c.vtable.all_gather_v) |f| return f(c.ptr, src, dst, stride, lens, t, stream);
        return c.vtable.all_gather(c.ptr, src, dst, stride, t, stream);
    }

    /// Whether `allGatherV` of `bytes`-byte strides moves only the lengths here (callers agree first: `varAgreed`).
    pub fn varFits(c: Collective, bytes: usize) bool {
        const f = c.vtable.var_fits orelse return false;
        return f(c.ptr, bytes);
    }

    /// Every rank's transport takes device lengths: one agreement a communicator (every rank calls it alike, outside any graph).
    pub fn varAgreed(c: Collective) Error!bool {
        const f = c.vtable.var_agreed orelse return false;
        return f(c.ptr);
    }

    pub fn exchange(c: Collective, src: DevicePtr, dst: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void {
        return c.vtable.exchange(c.ptr, src, dst, count, t, peer, stream);
    }

    /// A rank-order reduce-scatter's transfer (`VTable.scatter_rows`); else two equal shares trade by `exchange`.
    pub fn scatterRows(c: Collective, src: DevicePtr, dst: DevicePtr, rows: usize, width: usize, t: DType, stream: Stream) Error!void {
        if (c.vtable.scatter_rows) |f| return f(c.ptr, src, dst, rows, width, t, stream);
        const w = c.world();
        if (w != 2 or rows % 2 != 0) return error.Unsupported;
        const peer = 1 - c.rank();
        return c.exchange(src + rowShare(rows, w, peer)[0] * width * t.size(), dst, rows / 2 * width, t, peer, stream);
    }

    pub fn send(c: Collective, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void {
        return c.vtable.send(c.ptr, buf, count, t, peer, stream);
    }

    pub fn recv(c: Collective, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void {
        return c.vtable.recv(c.ptr, buf, count, t, peer, stream);
    }

    pub fn barrier(c: Collective) Error!void {
        return c.vtable.barrier(c.ptr);
    }

    pub fn check(c: Collective) Error!void {
        return c.vtable.check(c.ptr);
    }

    pub fn abort(c: Collective) void {
        c.vtable.abort(c.ptr);
    }
};

/// Refuses a peer that is this rank or out of range.
pub fn checkPeer(rank: u32, world: u32, peer: u32) Error!void {
    if (peer == rank or peer >= world) return error.Invalid;
}

/// Rank `rank`'s rows [lo, hi) of a reduce-scatter: ceil(rows / world) a rank from row 0, cut at `rows`.
pub fn rowShare(rows: usize, world: usize, rank: usize) [2]usize {
    const c = (rows + world - 1) / world;
    return .{ @min(rows, rank * c), @min(rows, (rank + 1) * c) };
}

/// recv's slot (of world - 1) for `peer`'s rows of this rank's share (`VTable.scatter_rows`).
pub fn scatterSlot(rank: usize, peer: usize) usize {
    return if (peer < rank) peer else peer - 1;
}

test "row shares tile the rows in rank order, ceil(rows / world) a rank; two ranks match upstream's equal cut" {
    for ([_]usize{ 0, 1, 2, 3, 255, 256, 257, 4095, 4096 }) |rows| for (1..6) |world| {
        var at: usize = 0;
        for (0..world) |r| {
            const s = rowShare(rows, world, r);
            try std.testing.expectEqual(at, s[0]);
            try std.testing.expect(s[1] - s[0] <= (rows + world - 1) / world);
            at = s[1];
        }
        try std.testing.expectEqual(rows, at);
    };
    try std.testing.expectEqual([2]usize{ 0, 129 }, rowShare(257, 2, 0));
    try std.testing.expectEqual([2]usize{ 129, 257 }, rowShare(257, 2, 1));
    for ([_][3]usize{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 1, 3, 2 }, .{ 2, 1, 1 } }) |x| try std.testing.expectEqual(x[2], scatterSlot(x[0], x[1]));
}

test "dtype sizes" {
    try std.testing.expectEqual(@as(usize, 2), DType.bf16.size());
    try std.testing.expectEqual(@as(usize, 8), DType.i64.size());
}
