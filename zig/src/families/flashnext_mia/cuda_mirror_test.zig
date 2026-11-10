//! The mirror in tests over local links: rank 1 replays rank 0's calls, records, idle waits, stops and losses.
const std = @import("std");
const lanes = @import("lanes");
const Link = @import("cuda_link.zig").Link;
const be = lanes.backend;
const Stream = lanes.Stream;
const Sampling = lanes.Sampling;
const Allocator = std.mem.Allocator;
const mirror = @import("cuda_mirror.zig");
const Frames = mirror.Frames;
const record_bytes = mirror.record_bytes;
const Wire = mirror.Wire;
const Records = mirror.Records;
const Watch = mirror.Watch;
const Leader = mirror.Leader;
const Follower = mirror.Follower;

const fake = lanes.fake;
const net = std.Io.net;
const link_mod = @import("cuda_link.zig");

fn testConfig(gpa: Allocator) !lanes.Config {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .max_streams = 1 }, 16, 15);
}

const Case = struct { prompt: []const u32, max_new: u32, sampling: ?Sampling = null, drafts: bool = true };

/// The cases one after another on `backend` through the round loop; each stream's emitted tokens.
fn drive(gpa: Allocator, backend: be.Backend, cases: []const Case) ![][]u32 {
    var cfg = try testConfig(gpa);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, backend, clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(Stream, cases.len);
    defer gpa.free(streams);
    for (cases, streams) |c, *s| s.* = try Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts });
    defer for (streams) |*s| s.deinit(gpa);
    // two streams at once (rounds alternate between them), then the rest
    for (streams[0..@min(2, streams.len)]) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    for (streams[@min(2, streams.len)..]) |*s| {
        try engine.addStream(s);
        while (engine.activeCount() > 0) try engine.step();
    }
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

fn freeRuns(gpa: Allocator, runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

pub const Rank1 = struct {
    target: fake.Fake,
    err: ?anyerror = null,
    calls: u64 = 0,
    failed: u64 = 0,
    sliced: bool = false,

    pub fn run(r: *Rank1, io: std.Io, port: u16) void {
        const gpa = std.testing.allocator;
        const l = Link.join(io, net.IpAddress.parse("127.0.0.1", port) catch unreachable, 5000) catch |e| {
            r.err = e;
            return;
        };
        defer l.close();
        var frames: Frames = .{ .gpa = gpa, .link = l };
        defer frames.deinit();
        var f = Follower.init(gpa, if (r.sliced) r.target.stepped() else r.target.backend(), frames.channel());
        defer f.deinit();
        f.run() catch |e| {
            r.err = e;
        };
        r.calls = f.calls;
        r.failed = f.failed;
    }
};

test "rank 1 makes rank 0's calls over a local link: the same draws, rounds and caches" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const cases = [_]Case{
        .{ .prompt = &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, .max_new = 60 },
        .{ .prompt = &.{ 2, 7, 1, 8 }, .max_new = 45, .sampling = .{ .seed = 11, .temperature = 0.8 } },
        .{ .prompt = &.{ 1, 1, 2, 3, 5, 8 }, .max_new = 30, .drafts = false },
        .{ .prompt = &.{ 9, 9 }, .max_new = 25, .sampling = .{ .seed = 5 } },
    };
    // the same cases on one rank, no link
    var alone: fake.Fake = .{ .gpa = gpa };
    defer alone.deinit();
    const want = try drive(gpa, alone.backend(), &cases);
    defer freeRuns(gpa, want);

    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    var rank1: Rank1 = .{ .target = .{ .gpa = gpa } };
    defer rank1.target.deinit();
    const t = try std.Thread.spawn(.{}, Rank1.run, .{ &rank1, io, server.port() });
    var rank0: fake.Fake = .{ .gpa = gpa };
    defer rank0.deinit();
    const l = try server.accept(5000);
    defer l.close();
    var frames: Frames = .{ .gpa = gpa, .link = l };
    var leader = Leader.init(gpa, rank0.backend(), frames.channel());
    const got = drive(gpa, leader.backend(), &cases) catch |e| {
        leader.deinit();
        t.join();
        return e;
    };
    defer freeRuns(gpa, got);
    leader.deinit(); // sends stop
    t.join();
    if (rank1.err) |e| return e;
    for (want, got) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    // rank 1 drew every token rank 0 drew, in order, and ran as many forwards
    try std.testing.expectEqualSlices(u32, rank0.drawn.items, rank1.target.drawn.items);
    try std.testing.expectEqual(rank0.rounds, rank1.target.rounds);
    try std.testing.expectEqual(rank0.prefill_count, rank1.target.prefill_count);
    try std.testing.expect(rank1.calls > cases.len * 3);
    try std.testing.expectEqual(@as(u64, 0), rank1.failed);
    // every stream was released on both ranks
    try std.testing.expectEqual(@as(u32, 0), rank0.lanes.count());
    try std.testing.expectEqual(@as(u32, 0), rank1.target.lanes.count());
}

/// The records' wire in the host tests: a second local link carrying each record as a frame.
const LinkWire = struct {
    link: Link,

    fn wire(w: *LinkWire) Wire {
        return .{ .ptr = w, .put = put, .get = get };
    }
    fn put(p: *anyopaque, record: []const u8) anyerror!void {
        const w: *LinkWire = @ptrCast(@alignCast(p));
        try w.link.send(.ack, record);
    }
    fn get(p: *anyopaque, record: []u8) anyerror!void {
        const w: *LinkWire = @ptrCast(@alignCast(p));
        const b = try w.link.expect(std.testing.allocator, .ack);
        defer std.testing.allocator.free(b);
        if (b.len != record_bytes) return error.TestBadRecord;
        @memcpy(record[0..record_bytes], b);
    }
};

const Rank1Records = struct {
    target: fake.Fake,
    err: ?anyerror = null,
    calls: u64 = 0,
    failed: u64 = 0,

    fn run(r: *Rank1Records, io: std.Io, port: u16, wire_port: u16) void {
        const gpa = std.testing.allocator;
        const l = Link.join(io, net.IpAddress.parse("127.0.0.1", port) catch unreachable, 5000) catch |e| {
            r.err = e;
            return;
        };
        defer l.close();
        const wl = Link.join(io, net.IpAddress.parse("127.0.0.1", wire_port) catch unreachable, 5000) catch |e| {
            r.err = e;
            return;
        };
        defer wl.close();
        var lw: LinkWire = .{ .link = wl };
        var records: Records = .{ .gpa = gpa, .wire = lw.wire(), .link = l };
        defer records.deinit();
        var f = Follower.init(gpa, r.target.backend(), records.channel());
        defer f.deinit();
        f.run() catch |e| {
            r.err = e;
        };
        r.calls = f.calls;
        r.failed = f.failed;
    }
};

test "the mirror over records: batched rounds, idle waits on the link, long prompts on the link, stop when idle" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var long: [1500]u32 = undefined;
    for (&long, 0..) |*x, i| x.* = @intCast((i * 7) % 90);
    const cases = [_]Case{
        .{ .prompt = &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, .max_new = 60 },
        .{ .prompt = &long, .max_new = 30, .sampling = .{ .seed = 3, .temperature = 0.9 } },
        .{ .prompt = &.{ 1, 1, 2, 3, 5, 8 }, .max_new = 30, .drafts = false },
        .{ .prompt = &.{ 9, 9 }, .max_new = 25, .sampling = .{ .seed = 5 } },
    };
    var alone: fake.Fake = .{ .gpa = gpa };
    defer alone.deinit();
    const want = try drive(gpa, alone.backend(), &cases);
    defer freeRuns(gpa, want);

    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const wire_server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer wire_server.close();
    var rank1: Rank1Records = .{ .target = .{ .gpa = gpa } };
    defer rank1.target.deinit();
    const t = try std.Thread.spawn(.{}, Rank1Records.run, .{ &rank1, io, server.port(), wire_server.port() });
    var rank0: fake.Fake = .{ .gpa = gpa };
    defer rank0.deinit();
    const l = try server.accept(5000);
    defer l.close();
    const wl = try wire_server.accept(5000);
    defer wl.close();
    var lw: LinkWire = .{ .link = wl };
    var records: Records = .{ .gpa = gpa, .wire = lw.wire(), .link = l };
    defer records.deinit();
    var leader = Leader.init(gpa, rank0.backend(), records.channel());
    // drive() runs two streams together, then the rest one by one: rank 0 goes idle between them
    const got = drive(gpa, leader.backend(), &cases) catch |e| {
        leader.deinit();
        t.join();
        return e;
    };
    defer freeRuns(gpa, got);
    try std.testing.expect(records.idle); // every stream released
    try std.testing.expect(leader.sends < leader.calls); // keep, first and release rode with the next call
    leader.deinit(); // stop while idle: a stop frame on the link
    t.join();
    if (rank1.err) |e| return e;
    for (want, got) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    try std.testing.expectEqualSlices(u32, rank0.drawn.items, rank1.target.drawn.items);
    try std.testing.expectEqual(rank0.rounds, rank1.target.rounds);
    try std.testing.expectEqual(leader.calls, rank1.calls);
    try std.testing.expectEqual(@as(u64, 0), rank1.failed);
    try std.testing.expectEqual(@as(u32, 0), rank1.target.lanes.count());
}

/// Both ends of a small socket pair with a deadline: an undrained large write blocks, then errors instead of hanging.
fn smallBuffers(links: []const Link) !void {
    const room: c_int = 4096;
    const deadline: std.posix.timeval = .{ .sec = 3, .usec = 0 };
    for (links) |l| {
        try std.posix.setsockopt(l.fd, std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, std.mem.asBytes(&room));
        try std.posix.setsockopt(l.fd, std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, std.mem.asBytes(&room));
        try std.posix.setsockopt(l.fd, std.posix.SOL.SOCKET, std.posix.SO.SNDTIMEO, std.mem.asBytes(&deadline));
        try std.posix.setsockopt(l.fd, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&deadline));
    }
}

const BatchReader = struct {
    got: ?[]u8 = null,
    err: ?anyerror = null,

    fn run(r: *BatchReader, io: std.Io, port: u16, wire_port: u16) void {
        const gpa = std.testing.allocator;
        const l = Link.join(io, net.IpAddress.parse("127.0.0.1", port) catch unreachable, 5000) catch |e| {
            r.err = e;
            return;
        };
        defer l.close();
        const wl = Link.join(io, net.IpAddress.parse("127.0.0.1", wire_port) catch unreachable, 5000) catch |e| {
            r.err = e;
            return;
        };
        defer wl.close();
        smallBuffers(&.{ l, wl }) catch |e| {
            r.err = e;
            return;
        };
        var lw: LinkWire = .{ .link = wl };
        var records: Records = .{ .gpa = gpa, .wire = lw.wire(), .link = l, .idle = false };
        defer records.deinit();
        const batch = records.channel().recv() catch |e| {
            r.err = e;
            return;
        };
        r.got = gpa.dupe(u8, batch orelse &.{}) catch |e| {
            r.err = e;
            return;
        };
    }
};

test "a batch larger than the link's buffers reaches rank 1: its record goes out before the batch is written" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const wire_server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer wire_server.close();
    var reader: BatchReader = .{};
    defer if (reader.got) |b| gpa.free(b);
    const t = try std.Thread.spawn(.{}, BatchReader.run, .{ &reader, io, server.port(), wire_server.port() });
    const l = try server.accept(5000);
    defer l.close();
    const wl = try wire_server.accept(5000);
    defer wl.close();
    try smallBuffers(&.{ l, wl });
    var lw: LinkWire = .{ .link = wl };
    var records: Records = .{ .gpa = gpa, .wire = lw.wire(), .link = l, .idle = false };
    defer records.deinit();
    const batch = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(batch);
    for (batch, 0..) |*b, i| b.* = @intCast((i * 31 + i / 251) % 256);
    const sent = records.channel().send(batch);
    t.join();
    try sent;
    if (reader.err) |e| return e;
    try std.testing.expectEqualSlices(u8, batch, reader.got.?);
}

test "rank 0 closing the link while no stream is open stops rank 1 cleanly" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const wire_server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer wire_server.close();
    var rank1: Rank1Records = .{ .target = .{ .gpa = gpa } };
    defer rank1.target.deinit();
    const t = try std.Thread.spawn(.{}, Rank1Records.run, .{ &rank1, io, server.port(), wire_server.port() });
    const l = try server.accept(5000);
    const wl = try wire_server.accept(5000);
    defer wl.close();
    l.close(); // no stop frame: a server stopped by SIGTERM
    t.join();
    try std.testing.expectEqual(@as(?anyerror, null), rank1.err);
}

const Lost = struct {
    var why: ?[]const u8 = null;
    fn record(_: *Watch, reason: []const u8) void {
        why = reason;
    }
};

test "the watchdog sees the other rank close the link, and a call past its deadline" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const peer = try Link.join(io, try net.IpAddress.parse("127.0.0.1", server.port()), 5000);
    const l = try server.accept(5000);
    defer l.close();
    Lost.why = null;
    var w: Watch = .{ .io = io, .fd = l.fd, .rank = 0, .lost = Lost.record };
    try w.start();
    std.Io.sleep(io, .fromMilliseconds(300), .awake) catch {};
    try std.testing.expect(Lost.why == null); // an open, quiet link is fine
    peer.close();
    var waited: u32 = 0;
    while (Lost.why == null and waited < 50) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
    w.deinit();
    try std.testing.expect(std.mem.indexOf(u8, Lost.why orelse "", "closed") != null);

    const peer2 = try Link.join(io, try net.IpAddress.parse("127.0.0.1", server.port()), 5000);
    defer peer2.close();
    const l2 = try server.accept(5000);
    defer l2.close();
    Lost.why = null;
    var w2: Watch = .{ .io = io, .fd = l2.fd, .rank = 1, .lost = Lost.record, .base_s = 0 };
    try w2.start();
    w2.base_s = 0; // start() may read TF_TP_TIMEOUT_S
    w2.arm(0);
    waited = 0;
    while (Lost.why == null and waited < 50) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
    w2.deinit();
    try std.testing.expect(std.mem.indexOf(u8, Lost.why orelse "", "deadline") != null);
}

test "prompts filling between rounds are mirrored: rank 1 fills the same slices and draws the same tokens" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const prompts = [_][]const u32{ &.{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 }, &.{ 2, 7, 1, 8 }, &.{ 1, 1, 2, 3, 5, 8, 13, 21 } };
    const Drive = struct {
        fn run(backend: be.Backend) ![3][]u32 {
            var cfg = try testConfig(gpa);
            defer cfg.deinit(gpa);
            var clock: fake.FixedClock = .{};
            var engine = lanes.Engine.init(gpa, &cfg, backend, clock.clock());
            defer engine.deinit();
            var streams: [3]Stream = undefined;
            for (&streams, prompts, 0..) |*s, p, i| s.* = try Stream.init(gpa, .{ .id = "s", .prompt = p, .max_new = 20 + 5 * @as(u32, @intCast(i)), .eos = &.{96} });
            defer for (&streams) |*s| s.deinit(gpa);
            // one fills alone, then the others arrive two steps apart while it decodes (f212's host steps fills)
            var filling: [3]bool = .{ true, false, false };
            var fresh: [3]bool = @splat(true);
            var next: usize = 1;
            var steps: usize = 0;
            while (steps < 10_000) : (steps += 1) {
                if (next < 3 and steps % 2 == 1) {
                    filling[next] = true;
                    next += 1;
                }
                var any = false;
                for (&streams, &filling, &fresh) |*s, *fl, *fr| if (fl.*) {
                    any = true;
                    if (try engine.fillStream(s, fr.*)) fl.* = false;
                    fr.* = false;
                };
                if (engine.activeCount() > 0) try engine.step() else if (!any and next >= 3) break;
            }
            var out: [3][]u32 = undefined;
            for (&out, &streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
            return out;
        }
    };
    var alone: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer alone.deinit();
    const want = try Drive.run(alone.stepped());
    defer for (want) |w| gpa.free(w);

    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    var rank1: Rank1 = .{ .target = .{ .gpa = gpa, .prefill_chunks = 3 }, .sliced = true };
    defer rank1.target.deinit();
    const t = try std.Thread.spawn(.{}, Rank1.run, .{ &rank1, io, server.port() });
    var rank0: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer rank0.deinit();
    const l = try server.accept(5000);
    defer l.close();
    var frames: Frames = .{ .gpa = gpa, .link = l };
    var leader = Leader.init(gpa, rank0.stepped(), frames.channel());
    const got = Drive.run(leader.backend()) catch |e| {
        leader.deinit();
        t.join();
        return e;
    };
    defer for (got) |g| gpa.free(g);
    leader.deinit();
    t.join();
    if (rank1.err) |e| return e;
    for (want, got) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    try std.testing.expect(rank0.prefill_count > 3); // every prompt in 3 steps
    try std.testing.expectEqual(rank0.prefill_count, rank1.target.prefill_count);
    try std.testing.expectEqualSlices(u32, rank0.drawn.items, rank1.target.drawn.items);
    try std.testing.expectEqual(@as(u64, 0), rank1.failed);
}
