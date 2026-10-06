//! Small exchanges through the mailbox, the rest through NCCL; the route depends on the call, type and size only, so ranks always agree.
const std = @import("std");
const collective = @import("collective.zig");
const Mailbox = @import("mailbox.zig").Mailbox;
const Nccl = @import("nccl.zig").Nccl;
const DevicePtr = collective.DevicePtr;
const DType = collective.DType;
const Op = collective.Op;
const Error = collective.Error;

pub const Hybrid = struct {
    mail: *Mailbox,
    nccl: *Nccl,
    /// Largest shard a rank sends through the mailbox.
    max_bytes: usize,

    pub fn iface(h: *Hybrid) collective.Collective {
        return .{ .ptr = h, .vtable = &vtable };
    }

    const vtable: collective.Collective.VTable = .{
        .kind = .hybrid,
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
    };

    fn self_(ptr: *anyopaque) *Hybrid {
        return @ptrCast(@alignCast(ptr));
    }

    fn small(h: *const Hybrid, n: usize) bool {
        return n <= h.max_bytes and h.mail.fits(n);
    }

    fn rankOf(ptr: *anyopaque) u32 {
        return self_(ptr).nccl.me;
    }

    fn worldOf(ptr: *anyopaque) u32 {
        return self_(ptr).nccl.size;
    }

    fn allReduce(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, op: Op, stream: collective.Stream) Error!void {
        const h = self_(ptr);
        if (op == .sum and (t == .f32 or t == .bf16) and h.small(count * t.size())) return h.mail.sum(s, r, count, t, stream);
        return Nccl.allReduce(h.nccl, s, r, count, t, op, stream);
    }

    fn allGather(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, stream: collective.Stream) Error!void {
        const h = self_(ptr);
        if (h.small(count * t.size())) return h.mail.allGatherBytes(s, r, count * t.size(), stream);
        return Nccl.allGather(h.nccl, s, r, count, t, stream);
    }

    fn exchange(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, peer: u32, stream: collective.Stream) Error!void {
        const h = self_(ptr);
        try collective.checkPeer(h.nccl.me, h.nccl.size, peer);
        if (h.small(count * t.size())) return h.mail.exchangeBytes(s, r, count * t.size(), stream);
        return Nccl.exchange(h.nccl, s, r, count, t, peer, stream);
    }

    fn send(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: collective.Stream) Error!void {
        return Nccl.iface(self_(ptr).nccl).send(buf, count, t, peer, stream);
    }

    fn recv(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: collective.Stream) Error!void {
        return Nccl.iface(self_(ptr).nccl).recv(buf, count, t, peer, stream);
    }

    fn barrier(ptr: *anyopaque) Error!void {
        return Nccl.iface(self_(ptr).nccl).barrier();
    }

    fn check(ptr: *anyopaque) Error!void {
        const h = self_(ptr);
        try h.mail.check();
        try Nccl.iface(h.nccl).check();
    }

    fn abort(ptr: *anyopaque) void {
        const h = self_(ptr);
        h.mail.abort();
        Nccl.iface(h.nccl).abort();
    }
};
