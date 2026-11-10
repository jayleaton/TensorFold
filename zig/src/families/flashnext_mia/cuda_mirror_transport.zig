//! The mirror's transports: link frames (host tests) and fixed-size records on an NCCL broadcast or a link.
const std = @import("std");
const cuda = @import("cuda");
const Link = @import("cuda_link.zig").Link;
const Comm = @import("cuda_comm.zig").Comm;
const Allocator = std.mem.Allocator;
const codec = @import("cuda_mirror.zig");
const Channel = codec.Channel;
const Watch = @import("cuda_mirror_watch.zig").Watch;

/// Every batch a `round` frame on the control link, `stop` a stop frame (the host tests' transport).
pub const Frames = struct {
    gpa: Allocator,
    link: Link,
    last: ?[]u8 = null,

    pub fn deinit(f: *Frames) void {
        if (f.last) |b| f.gpa.free(b);
        f.last = null;
    }

    pub fn channel(f: *Frames) Channel {
        return .{ .ptr = f, .vtable = &.{ .send = sendFn, .recv = recvFn, .idle = idleFn, .stop = stopFn } };
    }

    fn of(p: *anyopaque) *Frames {
        return @ptrCast(@alignCast(p));
    }
    fn sendFn(p: *anyopaque, bytes: []const u8) anyerror!void {
        try of(p).link.send(.round, bytes);
    }
    fn idleFn(_: *anyopaque) anyerror!void {}
    fn stopFn(p: *anyopaque) anyerror!void {
        try of(p).link.send(.stop, &.{});
    }
    fn recvFn(p: *anyopaque) anyerror!?[]const u8 {
        const f = of(p);
        f.deinit();
        const fr = try f.link.recv(f.gpa);
        switch (fr.tag) {
            .round => {
                f.last = fr.bytes;
                return fr.bytes;
            },
            .stop => {
                fr.deinit(f.gpa);
                return null;
            },
            else => {
                fr.deinit(f.gpa);
                return error.UnexpectedFrame;
            },
        }
    }
};

/// A record's bytes (the NCCL broadcast's size: 16 KiB holds a 64-stream round's windows; small ones cost alike).
pub const record_bytes = 16384;
const record_head = 8;

/// inline_call: batch follows; on_link: batch is the next link `round` frame; stop: rank 1 leaves; idle: wait request.
pub const Kind = enum(u8) { inline_call = 0, on_link = 1, stop = 2, idle = 3, _ };

/// `bytes` as a record into `out` (record_bytes); a batch too long for it goes on the link (`on_link`).
pub fn packRecord(out: []u8, kind: Kind, bytes: []const u8) Kind {
    @memset(out[0..record_head], 0);
    const k: Kind = if (kind == .inline_call and bytes.len > record_bytes - record_head) .on_link else kind;
    std.mem.writeInt(u32, out[0..4], @intCast(bytes.len), .little);
    out[4] = @backingInt(k);
    if (k == .inline_call) @memcpy(out[record_head..][0..bytes.len], bytes);
    return k;
}

pub const Record = struct { kind: Kind, len: u32, bytes: []const u8 };

pub fn unpackRecord(rec: []const u8) !Record {
    if (rec.len < record_bytes) return error.ShortRecord;
    const len = std.mem.readInt(u32, rec[0..4], .little);
    const kind: Kind = @fromBackingInt(rec[4]);
    switch (kind) {
        .inline_call => {
            if (len > record_bytes - record_head) return error.BadRecord;
            return .{ .kind = kind, .len = len, .bytes = rec[record_head..][0..len] };
        },
        .on_link, .stop, .idle => return .{ .kind = kind, .len = len, .bytes = &.{} },
        _ => return error.BadRecord,
    }
}

/// How a record goes from rank 0 to rank 1: an NCCL broadcast (`Broadcast`) or, in the host tests, a link.
pub const Wire = struct {
    ptr: *anyopaque,
    put: *const fn (ptr: *anyopaque, record: []const u8) anyerror!void,
    get: *const fn (ptr: *anyopaque, record: []u8) anyerror!void,
};

/// One record broadcast from rank 0 on the engine's stream, so it lands on rank 1 in the order the collectives run.
pub const Broadcast = struct {
    comm: *const Comm,
    stream: cuda.abi.Stream,
    device: cuda.DeviceBuffer,
    host: cuda.HostBuffer,
    copied: cuda.Event, // rank 0: the record left the pinned buffer (it may be written again)
    pending: bool = false,

    pub fn init(d: *const cuda.Driver, comm: *const Comm, stream: cuda.abi.Stream) !Broadcast {
        var device = try cuda.DeviceBuffer.alloc(d, record_bytes);
        errdefer device.free();
        var host = try cuda.HostBuffer.alloc(d, record_bytes);
        errdefer host.free();
        return .{ .comm = comm, .stream = stream, .device = device, .host = host, .copied = try cuda.Event.init(d, false) };
    }

    pub fn deinit(b: *Broadcast) void {
        if (b.pending) b.copied.synchronize() catch {};
        b.copied.deinit();
        b.host.free();
        b.device.free();
    }

    pub fn wire(b: *Broadcast) Wire {
        return .{ .ptr = b, .put = putFn, .get = getFn };
    }

    fn of(p: *anyopaque) *Broadcast {
        return @ptrCast(@alignCast(p));
    }

    fn putFn(p: *anyopaque, record: []const u8) anyerror!void {
        const b = of(p);
        if (b.pending) try b.copied.synchronize();
        b.pending = false;
        const used = record_head + (if (record[4] == @backingInt(Kind.inline_call)) std.mem.readInt(u32, record[0..4], .little) else 0);
        @memcpy(b.host.bytes[0..used], record[0..used]);
        try b.device.uploadAsync(0, b.host.bytes[0..used], b.stream);
        try b.copied.record(.{ .d = b.device.d, .handle = b.stream });
        b.pending = true;
        try b.comm.broadcast(b.device.ptr, record_bytes, 0, b.stream);
    }

    fn getFn(p: *anyopaque, record: []u8) anyerror!void {
        const b = of(p);
        try b.comm.broadcast(b.device.ptr, record_bytes, 0, b.stream);
        try b.device.downloadAsync(0, b.host.bytes[0..record_bytes], b.stream);
        try cuda.Stream.synchronize(.{ .d = b.device.d, .handle = b.stream });
        @memcpy(record[0..record_bytes], b.host.bytes[0..record_bytes]);
    }
};

/// Batches as records on a wire, long ones on the control link; while idle rank 1 waits on the link, not a collective.
pub const Records = struct {
    gpa: Allocator,
    wire: Wire,
    link: Link,
    watch: ?*Watch = null,
    idle: bool = true,
    rec: [record_bytes]u8 = undefined,
    last: ?[]u8 = null,

    pub fn deinit(r: *Records) void {
        if (r.last) |b| r.gpa.free(b);
        r.last = null;
    }

    pub fn channel(r: *Records) Channel {
        return .{ .ptr = r, .vtable = &.{ .send = sendFn, .recv = recvFn, .idle = idleFn, .stop = stopFn } };
    }

    fn of(p: *anyopaque) *Records {
        return @ptrCast(@alignCast(p));
    }

    fn put(r: *Records, kind: Kind, bytes: []const u8) !void {
        if (r.idle) try r.link.send(.request, &.{}); // rank 1 back on the wire
        r.idle = false;
        const k = packRecord(&r.rec, kind, bytes);
        // record first, then the long batch: rank 1 must be draining the link while a large write is still going
        try r.wire.put(r.wire.ptr, &r.rec);
        if (k == .on_link) try r.link.send(.round, bytes);
    }

    fn sendFn(p: *anyopaque, bytes: []const u8) anyerror!void {
        try of(p).put(.inline_call, bytes);
    }
    fn idleFn(p: *anyopaque) anyerror!void {
        const r = of(p);
        if (r.idle) return;
        try r.put(.idle, &.{});
        r.idle = true;
    }
    fn stopFn(p: *anyopaque) anyerror!void {
        const r = of(p);
        if (r.idle) return r.link.send(.stop, &.{});
        try r.put(.stop, &.{});
        r.idle = true;
    }
    fn recvFn(p: *anyopaque) anyerror!?[]const u8 {
        const r = of(p);
        if (r.last) |b| r.gpa.free(b);
        r.last = null;
        while (true) {
            if (r.idle) {
                // idle, rank 0 closing the link is its stop (a SIGTERM'd server sends no stop frame): exit 0
                if (r.watch) |w| w.idle.store(true, .release);
                const f = r.link.recv(r.gpa) catch |e| switch (e) {
                    error.PeerClosed => {
                        if (r.watch) |w| w.quiet();
                        std.log.info("rank 1: rank 0 closed the link while no stream was open: stopping", .{});
                        return null;
                    },
                    else => return e,
                };
                if (r.watch) |w| w.idle.store(false, .release);
                defer f.deinit(r.gpa);
                switch (f.tag) {
                    .request => r.idle = false,
                    .stop => {
                        if (r.watch) |w| w.quiet();
                        return null;
                    },
                    else => return error.UnexpectedFrame,
                }
            }
            // on the wire rank 0 is busy: its next record is due (a stall past the deadline is a lost rank)
            if (r.watch) |w| w.arm(w.base_s);
            defer if (r.watch) |w| w.disarm();
            try r.wire.get(r.wire.ptr, &r.rec);
            const rec = try unpackRecord(&r.rec);
            switch (rec.kind) {
                .inline_call => return rec.bytes,
                .stop => {
                    if (r.watch) |w| w.quiet();
                    return null;
                },
                .idle => r.idle = true,
                .on_link => {
                    const b = try r.link.expect(r.gpa, .round);
                    if (b.len != rec.len) {
                        r.gpa.free(b);
                        return error.RecordMismatch;
                    }
                    r.last = b;
                    return b;
                },
                _ => return error.BadRecord,
            }
        }
    }
};

test "a record holds a short call inline and sends a long one to the link" {
    var rec: [record_bytes]u8 = undefined;
    try std.testing.expectEqual(Kind.inline_call, packRecord(&rec, .inline_call, "abc"));
    const r = try unpackRecord(&rec);
    try std.testing.expectEqual(Kind.inline_call, r.kind);
    try std.testing.expectEqualStrings("abc", r.bytes);
    try std.testing.expectEqualSlices(u8, &.{ 3, 0, 0, 0, 0, 0, 0, 0, 'a', 'b', 'c' }, rec[0..11]);
    const long: [record_bytes - record_head + 1]u8 = @splat(7);
    try std.testing.expectEqual(Kind.on_link, packRecord(&rec, .inline_call, &long));
    const l = try unpackRecord(&rec);
    try std.testing.expectEqual(Kind.on_link, l.kind);
    try std.testing.expectEqual(@as(u32, long.len), l.len);
    const fits: [record_bytes - record_head]u8 = @splat(7);
    try std.testing.expectEqual(Kind.inline_call, packRecord(&rec, .inline_call, &fits));
    try std.testing.expectEqual(Kind.stop, packRecord(&rec, .stop, &.{}));
    try std.testing.expectEqual(Kind.stop, (try unpackRecord(&rec)).kind);
    rec[4] = 9;
    try std.testing.expectError(error.BadRecord, unpackRecord(&rec));
    try std.testing.expectError(error.ShortRecord, unpackRecord(rec[0..100]));
}
