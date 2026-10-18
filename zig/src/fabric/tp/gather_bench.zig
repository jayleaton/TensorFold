//! `tf-tp-test gather-bench [N]`: our RoCE all-gather vs upstream's one-shot vs NCCL at decode sizes, bytes checked.
const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const Collective = tp.Collective;

/// Bytes a rank: 4 KiB, then 1-6 decode rows of D = 4096 bf16 (8 KiB each), and 64 KiB.
const sizes = [_]usize{ 4 << 10, 8 << 10, 16 << 10, 24 << 10, 32 << 10, 40 << 10, 48 << 10, 64 << 10 };
const max_bytes = 64 << 10;
/// A decode window's exchanges in one graph (the window mode's per-call time is a replay's over this many).
const window = 83;

const Backend = struct { comm: Collective, label: []const u8 };
const Mode = enum { eager, graph, window };

const Bench = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    rank: u32,
    stream: cuda.Stream,
    in: cuda.DeviceBuffer,
    outs: cuda.DeviceBuffer,
    meet: Collective,
    events: []cuda.Event,
    us: []f64,

    fn issue(b: *Bench, comm: Collective, bytes: usize, slot: usize) !void {
        try comm.allGather(b.in.ptr, b.outs.ptr + slot * 2 * max_bytes, bytes / 2, .bf16, b.stream.handle);
    }

    /// GPU microseconds between consecutive events (`n` of them after the first) into `us[0..n]`, scaled by 1 / `per`.
    fn deltas(b: *Bench, n: usize, per: usize) ![]f64 {
        try b.events[n].synchronize();
        for (0..n) |i| b.us[i] = @as(f64, try cuda.Event.elapsedMs(b.events[i], b.events[i + 1])) * 1000.0 / @as(f64, @floatFromInt(per));
        return b.us[0..n];
    }

    fn eager(b: *Bench, comm: Collective, bytes: usize, n: usize) ![]f64 {
        for (0..20) |_| try b.issue(comm, bytes, 0);
        try b.stream.synchronize();
        try b.meet.barrier();
        try b.events[0].record(b.stream);
        for (0..n) |i| {
            try b.issue(comm, bytes, 0);
            try b.events[i + 1].record(b.stream);
        }
        return b.deltas(n, 1);
    }

    /// `replays` launches of one graph of `per` calls, an event between each.
    fn graphed(b: *Bench, comm: Collective, bytes: usize, per: usize, replays: usize) ![]f64 {
        try cuda.graph.beginCapture(b.stream, .thread_local);
        const err: ?tp.collective.Error = blk: {
            for (0..per) |_| b.issue(comm, bytes, 0) catch |e| break :blk e;
            break :blk null;
        };
        var g = try cuda.graph.endCapture(b.stream);
        defer g.deinit();
        if (err) |e| return e;
        var exec = try g.instantiate();
        defer exec.deinit();
        try exec.upload(b.stream);
        for (0..3) |_| try exec.launchOn(b.stream);
        try b.stream.synchronize();
        try b.meet.barrier();
        try b.events[0].record(b.stream);
        for (0..replays) |i| {
            try exec.launchOn(b.stream);
            try b.events[i + 1].record(b.stream);
        }
        return b.deltas(replays, per);
    }

    /// Each backend's gather of this size once into its own output: equal to NCCL's and to the host reference.
    fn same(b: *Bench, backends: []const Backend, bytes: usize, seed: u64) !bool {
        const mine = try b.gpa.alloc(u8, bytes);
        defer b.gpa.free(mine);
        const want = try b.gpa.alloc(u8, 2 * bytes);
        defer b.gpa.free(want);
        const got = try b.gpa.alloc(u8, 2 * bytes);
        defer b.gpa.free(got);
        for (0..2) |r| tp.reference.fill(want[r * bytes ..][0..bytes], .bf16, @intCast(r), seed);
        @memcpy(mine, want[b.rank * bytes ..][0..bytes]);
        try b.in.uploadAsyncDirect(0, mine, b.stream.handle);
        try b.outs.fill8(0xa5, b.stream.handle);
        for (backends, 0..) |x, i| try b.issue(x.comm, bytes, i);
        try b.stream.synchronize();
        var ok = true;
        for (backends, 0..) |x, i| {
            try x.comm.check();
            try b.outs.download(i * 2 * max_bytes, got);
            ok = ok and std.mem.eql(u8, got, want);
        }
        return ok;
    }
};

fn stats(v: []f64) struct { median: f64, p99: f64, mean: f64 } {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    var sum: f64 = 0;
    for (v) |x| sum += x;
    return .{ .median = v[v.len / 2], .p99 = v[@min(v.len - 1, v.len * 99 / 100)], .mean = sum / @as(f64, @floatFromInt(v.len)) };
}

/// Needs TF_COMM_BACKEND=roce and TF_TP_ROCE_ONESHOT=1 (ours is skipped when RoCE fell back to NCCL).
pub fn run(s: *tp.Session, d: *const cuda.Driver, gpa: std.mem.Allocator, cfg: tp.Config, iters: usize) !u8 {
    if (s.oneshot == null) {
        std.debug.print("REFUSED gather-bench needs TF_COMM_BACKEND=roce TF_TP_ROCE_ONESHOT=1\n", .{});
        return 2;
    }
    var list: [3]Backend = undefined;
    var n: usize = 0;
    list[n] = .{ .comm = s.nccl.iface(), .label = "nccl" };
    n += 1;
    if (s.roce != null) {
        list[n] = .{ .comm = s.route.iface(), .label = "roce" };
        n += 1;
    } else std.debug.print("INFO rank {d}: our RoCE route is not up (NCCL fallback): nccl vs oneshot only\n", .{cfg.rank});
    list[n] = .{ .comm = s.comm(), .label = "oneshot" };
    n += 1;
    const replays = @max(iters, 20);
    var b: Bench = .{ .d = d, .gpa = gpa, .rank = cfg.rank, .stream = try .init(d, true), .in = try .alloc(d, max_bytes), .outs = try .alloc(d, 3 * 2 * max_bytes), .meet = s.nccl.iface(), .events = try gpa.alloc(cuda.Event, replays + 1), .us = try gpa.alloc(f64, replays) };
    defer {
        b.stream.deinit();
        b.in.free();
        b.outs.free();
        gpa.free(b.events);
        gpa.free(b.us);
    }
    var made: usize = 0;
    defer for (b.events[0..made]) |*e| e.deinit();
    for (b.events) |*e| {
        e.* = try cuda.Event.init(d, true);
        made += 1;
    }
    var failures: u32 = 0;
    for (sizes, 0..) |bytes, k| {
        const ok = try b.same(list[0..n], bytes, 0x6761_7468 + k);
        if (!ok) failures += 1;
        std.debug.print("{s} gather-bench rank {d} bytes {d}: gathered bytes equal across {d} backends and the reference\n", .{ if (ok) "PASS" else "FAIL", cfg.rank, bytes, n });
        const rows: usize = if (bytes % (8 << 10) == 0 and bytes <= 48 << 10) bytes / (8 << 10) else 0;
        for (list[0..n]) |x| for ([_]Mode{ .eager, .graph, .window }) |mode| {
            const v = switch (mode) {
                .eager => try b.eager(x.comm, bytes, iters),
                .graph => try b.graphed(x.comm, bytes, 1, iters),
                .window => try b.graphed(x.comm, bytes, window, @max(iters / window, 20)),
            };
            try x.comm.check();
            const st = stats(v);
            if (cfg.rank == 0) std.debug.print("RESULT {{\"op\":\"gather_bench\",\"backend\":\"{s}\",\"mode\":\"{t}\",\"bytes\":{d},\"rows\":{d},\"n\":{d},\"median_us\":{d:.2},\"p99_us\":{d:.2},\"mean_us\":{d:.2}}}\n", .{ x.label, mode, bytes, rows, v.len, st.median, st.p99, st.mean });
        };
    }
    if (s.fate) |*f| f.expectStop();
    try s.nccl.iface().barrier();
    std.debug.print("SUMMARY gather-bench rank {d}: {d} sizes, {d} byte mismatches, {d} one-shot calls\n", .{ cfg.rank, sizes.len, failures, s.oneshot.?.calls });
    s.close();
    return if (failures == 0) 0 else 1;
}
