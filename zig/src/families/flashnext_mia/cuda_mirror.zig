//! Two ranks behind one lane backend: rank 0's Leader records each engine call, rank 1's Follower replays it.
const std = @import("std");
const lanes = @import("lanes");
const be = lanes.backend;
const Stream = lanes.Stream;
const Sampling = lanes.Sampling;
const Allocator = std.mem.Allocator;
const mirror_transport = @import("cuda_mirror_transport.zig");
const mirror_watch = @import("cuda_mirror_watch.zig");
const mirror_ranks = @import("cuda_mirror_ranks.zig");

pub const Op = enum(u8) { prefill = 1, first = 2, queue = 3, verify = 4, keep = 5, draft = 6, unspeculate = 7, release = 8, note = 9, prefill_begin = 10, prefill_step = 11, prefill_many = 12, _ };

pub const StreamSpec = struct {
    key: u64,
    max_new: u32,
    drafts: bool,
    sampling: ?Sampling,
    prompt: []const u32,
    eos: []const u32,
    chunks: []const u32,
    /// the kept prompt state the pass resumes (rank 0's key of it; 0: none), its position, and the pass's marks
    resume_key: u64 = 0,
    resume_at: u32 = 0,
    marks: []const u32 = &.{},
    /// the prompt's images and video frames: rows, rotary positions and offset (features follow by NCCL broadcast)
    media: ?MediaSpec = null,
};

/// The media's shape for rank 1: feature row count, decode offset and width (the data follows by NCCL broadcast).
pub const MediaSpec = struct { rows: u32, delta: i64, width: u32 };

pub const WindowCall = struct {
    key: u64,
    pending: u32,
    held: u32,
    tokens: []const u32,
    parents: ?[]const i32,
    positions: []const u64,
    early: bool,

    pub fn rows(w: WindowCall) usize {
        return 1 + @as(usize, w.held) + w.tokens.len;
    }
};

pub const DraftCall = struct {
    key: u64,
    follow: []const u32,
    first: ?be.Feed,
    rows: ?[]const u32,
    start: u64,
    position: u64,
    depth: u32,
    early: bool,
    ranks: bool,
};

pub const Call = union(enum) {
    prefill: StreamSpec,
    first: struct { key: u64, position: u64 },
    queue: struct { key: u64, feed: be.Feed, position: u64 },
    verify: []const WindowCall,
    keep: struct { windows: []const WindowCall, paths: []const []const u32 },
    draft: []const DraftCall,
    unspeculate: u64,
    release: u64,
    /// the lane backend's own message to rank 1 (rank 0's prompt-cache decisions), opaque here
    note: []const u8,
    /// a prompt that fills between rounds: its start (prefill's fields), then slices of every filling stream
    prefill_begin: StreamSpec,
    prefill_step: []const u64,
    /// a burst of prompts prefilled together (each stream's prefill fields)
    prefill_many: []const StreamSpec,

    fn op(c: Call) Op {
        return switch (c) {
            .prefill => .prefill,
            .first => .first,
            .queue => .queue,
            .verify => .verify,
            .keep => .keep,
            .draft => .draft,
            .unspeculate => .unspeculate,
            .release => .release,
            .note => .note,
            .prefill_begin => .prefill_begin,
            .prefill_step => .prefill_step,
            .prefill_many => .prefill_many,
        };
    }
};

pub fn keyOf(s: *const Stream) u64 {
    return @intFromPtr(s);
}

// -- encoding -----------------------------------------------------------------------------------------------------

const Out = struct {
    list: *std.ArrayList(u8),
    gpa: Allocator,

    fn spec(o: Out, p: StreamSpec) !void {
        try o.int(u64, p.key);
        try o.int(u32, p.max_new);
        try o.flag(p.drafts);
        try o.flag(p.sampling != null);
        if (p.sampling) |s| {
            try o.int(u64, s.seed);
            try o.float(s.temperature);
            try o.int(u32, s.top_k);
            try o.float(s.top_p);
            try o.float(s.min_p);
        }
        try o.many(u32, p.prompt);
        try o.many(u32, p.eos);
        try o.many(u32, p.chunks);
        try o.int(u64, p.resume_key);
        try o.int(u32, p.resume_at);
        try o.many(u32, p.marks);
        try o.flag(p.media != null);
        if (p.media) |m| {
            try o.int(u32, m.rows);
            try o.int(i64, m.delta);
            try o.int(u32, m.width);
        }
    }
    fn int(o: Out, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try o.list.appendSlice(o.gpa, &b);
    }
    fn flag(o: Out, v: bool) !void {
        try o.int(u8, @intFromBool(v));
    }
    fn float(o: Out, v: f64) !void {
        try o.int(u64, @bitCast(v));
    }
    fn many(o: Out, comptime T: type, xs: []const T) !void {
        try o.int(u32, @intCast(xs.len));
        for (xs) |x| try o.int(T, x);
    }
    fn feed(o: Out, f: be.Feed) !void {
        switch (f) {
            .handle => |h| {
                try o.int(u8, 0);
                try o.int(u64, h);
            },
            .value => |v| {
                try o.int(u8, 1);
                try o.int(u64, v);
            },
        }
    }
    fn windows(o: Out, ws: []const WindowCall) !void {
        try o.int(u32, @intCast(ws.len));
        for (ws) |w| {
            try o.int(u64, w.key);
            try o.int(u32, w.pending);
            try o.int(u32, w.held);
            try o.many(u32, w.tokens);
            try o.flag(w.parents != null);
            if (w.parents) |p| try o.many(i32, p);
            try o.many(u64, w.positions);
            try o.flag(w.early);
        }
    }
};

/// `call` as a `round` frame's bytes, appended to `list`.
pub fn encode(list: *std.ArrayList(u8), gpa: Allocator, call: Call) !void {
    const o: Out = .{ .list = list, .gpa = gpa };
    try o.int(u8, @backingInt(call.op()));
    switch (call) {
        .prefill, .prefill_begin => |p| try o.spec(p),
        .prefill_many => |ps| {
            try o.int(u32, @intCast(ps.len));
            for (ps) |p| try o.spec(p);
        },
        .first => |f| {
            try o.int(u64, f.key);
            try o.int(u64, f.position);
        },
        .queue => |q| {
            try o.int(u64, q.key);
            try o.feed(q.feed);
            try o.int(u64, q.position);
        },
        .verify => |ws| try o.windows(ws),
        .keep => |k| {
            if (k.paths.len != k.windows.len) return error.PathsWindowsMismatch;
            try o.windows(k.windows);
            for (k.paths) |p| try o.many(u32, p);
        },
        .draft => |rs| {
            try o.int(u32, @intCast(rs.len));
            for (rs) |r| {
                try o.int(u64, r.key);
                try o.many(u32, r.follow);
                try o.flag(r.first != null);
                if (r.first) |f| try o.feed(f);
                try o.flag(r.rows != null);
                if (r.rows) |rows| try o.many(u32, rows);
                try o.int(u64, r.start);
                try o.int(u64, r.position);
                try o.int(u32, r.depth);
                try o.flag(r.early);
                try o.flag(r.ranks);
            }
        },
        .unspeculate, .release => |key| try o.int(u64, key),
        .note => |bytes| try o.many(u8, bytes),
        .prefill_step => |keys| try o.many(u64, keys),
    }
}

// -- decoding -----------------------------------------------------------------------------------------------------

const In = struct {
    bytes: []const u8,
    at: usize = 0,
    arena: Allocator,

    fn int(i: *In, comptime T: type) !T {
        const n = @sizeOf(T);
        if (i.bytes.len - i.at < n) return error.ShortFrame;
        const v = std.mem.readInt(T, i.bytes[i.at..][0..n], .little);
        i.at += n;
        return v;
    }
    fn flag(i: *In) !bool {
        return switch (try i.int(u8)) {
            0 => false,
            1 => true,
            else => error.BadFrame,
        };
    }
    fn float(i: *In) !f64 {
        return @bitCast(try i.int(u64));
    }
    fn many(i: *In, comptime T: type) ![]T {
        const n = try i.int(u32);
        if ((i.bytes.len - i.at) / @sizeOf(T) < n) return error.ShortFrame;
        const out = try i.arena.alloc(T, n);
        for (out) |*x| x.* = try i.int(T);
        return out;
    }
    fn feed(i: *In) !be.Feed {
        const kind = try i.int(u8);
        const v = try i.int(u64);
        return switch (kind) {
            0 => .{ .handle = v },
            1 => .{ .value = std.math.cast(u32, v) orelse return error.BadFrame },
            else => error.BadFrame,
        };
    }
    fn count(i: *In) !usize {
        const n = try i.int(u32);
        // every element takes at least a byte: a corrupt count cannot ask for more than the frame holds
        if (n > i.bytes.len - i.at) return error.ShortFrame;
        return n;
    }
    fn spec(i: *In) !StreamSpec {
        return .{
            .key = try i.int(u64),
            .max_new = try i.int(u32),
            .drafts = try i.flag(),
            .sampling = if (try i.flag()) .{
                .seed = try i.int(u64),
                .temperature = try i.float(),
                .top_k = try i.int(u32),
                .top_p = try i.float(),
                .min_p = try i.float(),
            } else null,
            .prompt = try i.many(u32),
            .eos = try i.many(u32),
            .chunks = try i.many(u32),
            .resume_key = try i.int(u64),
            .resume_at = try i.int(u32),
            .marks = try i.many(u32),
            .media = if (try i.flag()) .{
                .rows = try i.int(u32),
                .delta = try i.int(i64),
                .width = try i.int(u32),
            } else null,
        };
    }

    fn windows(i: *In) ![]WindowCall {
        const ws = try i.arena.alloc(WindowCall, try i.count());
        for (ws) |*w| w.* = .{
            .key = try i.int(u64),
            .pending = try i.int(u32),
            .held = try i.int(u32),
            .tokens = try i.many(u32),
            .parents = if (try i.flag()) try i.many(i32) else null,
            .positions = try i.many(u64),
            .early = try i.flag(),
        };
        return ws;
    }
};

/// A `round` frame's call; its slices live in `arena`.
pub fn decode(arena: Allocator, bytes: []const u8) !Call {
    var i: In = .{ .bytes = bytes, .arena = arena };
    const op: Op = @fromBackingInt(try i.int(u8));
    const call: Call = switch (op) {
        .prefill => .{ .prefill = try i.spec() },
        .prefill_begin => .{ .prefill_begin = try i.spec() },
        .prefill_step => .{ .prefill_step = try i.many(u64) },
        .prefill_many => blk: {
            const ps = try arena.alloc(StreamSpec, try i.count());
            for (ps) |*p| p.* = try i.spec();
            break :blk .{ .prefill_many = ps };
        },
        .first => .{ .first = .{ .key = try i.int(u64), .position = try i.int(u64) } },
        .queue => .{ .queue = .{ .key = try i.int(u64), .feed = try i.feed(), .position = try i.int(u64) } },
        .verify => .{ .verify = try i.windows() },
        .keep => blk: {
            const ws = try i.windows();
            const paths = try arena.alloc([]const u32, ws.len);
            for (paths) |*p| p.* = try i.many(u32);
            break :blk .{ .keep = .{ .windows = ws, .paths = paths } };
        },
        .draft => blk: {
            const rs = try arena.alloc(DraftCall, try i.count());
            for (rs) |*r| r.* = .{
                .key = try i.int(u64),
                .follow = try i.many(u32),
                .first = if (try i.flag()) try i.feed() else null,
                .rows = if (try i.flag()) try i.many(u32) else null,
                .start = try i.int(u64),
                .position = try i.int(u64),
                .depth = try i.int(u32),
                .early = try i.flag(),
                .ranks = try i.flag(),
            };
            break :blk .{ .draft = rs };
        },
        .unspeculate => .{ .unspeculate = try i.int(u64) },
        .release => .{ .release = try i.int(u64) },
        .note => .{ .note = try i.many(u8) },
        _ => return error.UnknownOp,
    };
    if (i.at != bytes.len) return error.TrailingBytes;
    return call;
}

// -- batches ------------------------------------------------------------------------------------------------------

/// Calls in one send: u16 count, then each call as a u32 length and its bytes.
pub const Batch = struct {
    buf: std.ArrayList(u8) = .empty,
    count: u16 = 0,
    scratch: std.ArrayList(u8) = .empty,

    pub fn deinit(b: *Batch, gpa: Allocator) void {
        b.buf.deinit(gpa);
        b.scratch.deinit(gpa);
    }

    pub fn add(b: *Batch, gpa: Allocator, call: Call) !void {
        if (b.count == 0) {
            b.buf.clearRetainingCapacity();
            try b.buf.appendSlice(gpa, &.{ 0, 0 });
        }
        b.scratch.clearRetainingCapacity();
        try encode(&b.scratch, gpa, call);
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(b.scratch.items.len), .little);
        try b.buf.appendSlice(gpa, &len);
        try b.buf.appendSlice(gpa, b.scratch.items);
        b.count += 1;
        std.mem.writeInt(u16, b.buf.items[0..2], b.count, .little);
    }

    /// The batch's bytes; empty when it holds no call.
    pub fn bytes(b: *const Batch) []const u8 {
        return if (b.count == 0) &.{} else b.buf.items;
    }

    pub fn clear(b: *Batch) void {
        b.count = 0;
    }
};

/// The calls of a batch's bytes, in order.
pub const Calls = struct {
    bytes: []const u8,
    left: u16,
    at: usize = 2,

    pub fn of(bytes: []const u8) !Calls {
        if (bytes.len < 2) return error.ShortBatch;
        return .{ .bytes = bytes, .left = std.mem.readInt(u16, bytes[0..2], .little) };
    }

    pub fn next(c: *Calls) !?[]const u8 {
        if (c.left == 0) return if (c.at == c.bytes.len) null else error.TrailingBytes;
        if (c.bytes.len - c.at < 4) return error.ShortBatch;
        const n = std.mem.readInt(u32, c.bytes[c.at..][0..4], .little);
        c.at += 4;
        if (c.bytes.len - c.at < n) return error.ShortBatch;
        defer c.at += n;
        c.left -= 1;
        return c.bytes[c.at..][0..n];
    }
};

// -- transport ----------------------------------------------------------------------------------------------------

/// Where rank 0's batches go and rank 1 reads them, in order.
pub const Channel = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// rank 0: one batch
        send: *const fn (ptr: *anyopaque, bytes: []const u8) anyerror!void,
        /// rank 1: the next batch (valid until the next recv); null once rank 0 stopped
        recv: *const fn (ptr: *anyopaque) anyerror!?[]const u8,
        /// rank 0: no stream is open; rank 1 may wait for the next one where a lost peer shows (the link)
        idle: *const fn (ptr: *anyopaque) anyerror!void,
        /// rank 0: rank 1 leaves its loop
        stop: *const fn (ptr: *anyopaque) anyerror!void,
    };

    pub fn send(c: Channel, bytes: []const u8) !void {
        return c.vtable.send(c.ptr, bytes);
    }
    pub fn recv(c: Channel) !?[]const u8 {
        return c.vtable.recv(c.ptr);
    }
    pub fn idle(c: Channel) !void {
        return c.vtable.idle(c.ptr);
    }
    pub fn stop(c: Channel) !void {
        return c.vtable.stop(c.ptr);
    }
};

pub const Frames = mirror_transport.Frames;
pub const record_bytes = mirror_transport.record_bytes;
pub const Kind = mirror_transport.Kind;
pub const packRecord = mirror_transport.packRecord;
pub const Record = mirror_transport.Record;
pub const unpackRecord = mirror_transport.unpackRecord;
pub const Wire = mirror_transport.Wire;
pub const Broadcast = mirror_transport.Broadcast;
pub const Records = mirror_transport.Records;
pub const Watch = mirror_watch.Watch;
pub const Agreement = mirror_watch.Agreement;
pub const Leader = mirror_ranks.Leader;
pub const Follower = mirror_ranks.Follower;

// -- tests --------------------------------------------------------------------------------------------------------

test "a prefill record carries the prompt's media shape: rows' count, offset and width" {
    var ar: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer ar.deinit();
    const c = try roundTrip(&ar, .{ .prefill_begin = .{ .key = 7, .max_new = 9, .drafts = true, .sampling = null, .prompt = &.{ 1, 2, 3, 4, 5 }, .eos = &.{}, .chunks = &.{}, .media = .{ .rows = 2, .delta = -1, .width = 2560 } } });
    const m = c.prefill_begin.media.?;
    try std.testing.expectEqual(@as(u32, 2), m.rows);
    try std.testing.expectEqual(@as(i64, -1), m.delta);
    try std.testing.expectEqual(@as(u32, 2560), m.width);
    const t = try roundTrip(&ar, .{ .prefill = .{ .key = 8, .max_new = 1, .drafts = false, .sampling = null, .prompt = &.{1}, .eos = &.{}, .chunks = &.{} } });
    try std.testing.expect(t.prefill.media == null);
}

const roundTrip = @import("cuda_mirror_codec_test.zig").roundTrip;
const net = std.Io.net;
const link_mod = @import("cuda_link.zig");
const Rank1 = @import("cuda_mirror_test.zig").Rank1;

test "rank 1 ends with an error when rank 0 goes away without a stop" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    var rank1: Rank1 = .{ .target = .{ .gpa = gpa } };
    defer rank1.target.deinit();
    const t = try std.Thread.spawn(.{}, Rank1.run, .{ &rank1, io, server.port() });
    const l = try server.accept(5000);
    l.close();
    t.join();
    try std.testing.expectEqual(@as(?anyerror, error.PeerClosed), rank1.err);
}
