//! The checkpoint streamer: runs coalesced into large aligned windows, read by several threads, copied in file order.
const std = @import("std");
const builtin = @import("builtin");
const core = @import("core");
const dio = core.direct_io;
const load = @import("load.zig");
const Catalog = load.Catalog;
const Plan = load.Plan;
const Sink = load.Sink;

/// Told when the last run of a shard has been read: its pages can leave the page cache.
pub const ShardEnd = struct { ctx: *anyopaque, call: *const fn (ctx: *anyopaque, shard: u16) void };

pub const Options = struct {
    /// Bytes of each aligned buffer: the most one read covers (less 8 KiB of alignment slack).
    chunk: usize = Streamer.default_chunk,
    /// Reader threads, so reads in flight (0: every read runs on the calling thread, as `runSerial` does).
    threads: usize = Streamer.default_threads,
    /// Buffers beyond one per reader: windows read ahead of the copy that waits for them.
    ahead: usize = 2,
    /// Runs this close (bytes between them) share a read; the bytes between are read and dropped.
    gap: usize = Streamer.default_gap,
    /// Most bytes one sink put carries: a device copy stages its source, so a put is kept to a size that stage reuses.
    put_max: usize = Streamer.default_put,

    /// The defaults with TF_MM3_LOAD_THREADS, _CHUNK_MIB, _GAP_KIB and _PUT_MIB applied, to tune a slow mount.
    pub fn fromEnv() Options {
        var o: Options = .{};
        o.threads = envInt("TF_MM3_LOAD_THREADS", o.threads, 1);
        o.chunk = envInt("TF_MM3_LOAD_CHUNK_MIB", o.chunk >> 20, 1) << 20;
        o.gap = envInt("TF_MM3_LOAD_GAP_KIB", o.gap >> 10, 1) << 10;
        o.put_max = envInt("TF_MM3_LOAD_PUT_MIB", o.put_max >> 20, 1) << 20;
        return o;
    }
};

/// The integer in environment variable `name` when it holds one of at least `min` and under 4096, else `fallback`.
fn envInt(name: [:0]const u8, fallback: usize, min: usize) usize {
    const v = std.c.getenv(name) orelse return fallback;
    const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return fallback;
    return if (n >= min and n < 4096) n else fallback;
}

/// What the last `run` read, for the benchmark and the tests.
pub const Stats = struct {
    windows: u64 = 0, // reads planned
    read_bytes: u64 = 0, // bytes those reads cover, gaps included (before 4 KiB widening)
    wanted_bytes: u64 = 0, // bytes the plan asked for
    markers: u64 = 0, // codebook markers checked
    marker_windows: u64 = 0, // reads they took
};

/// Reads a plan's byte runs as coalesced windows with several reads in flight.
pub const Streamer = struct {
    gpa: std.mem.Allocator,
    cat: *const Catalog,
    bufs: []dio.Buffer,
    folded: []f32,
    ended: ?ShardEnd = null,
    opts: Options,
    stats: Stats = .{},

    pub const default_chunk = 16 << 20;
    pub const default_threads = 6;
    pub const default_gap = 1 << 20;
    pub const default_put = 4 << 20;

    pub fn init(gpa: std.mem.Allocator, cat: *const Catalog, chunk: usize) !Streamer {
        var o = Options.fromEnv();
        if (chunk != default_chunk) o.chunk = chunk;
        return initWith(gpa, cat, o);
    }

    pub fn initWith(gpa: std.mem.Allocator, cat: *const Catalog, opts: Options) !Streamer {
        if (opts.chunk < 4 * dio.alignment or opts.chunk % dio.alignment != 0 or opts.put_max == 0) return error.BadChunk;
        const n = opts.threads + opts.ahead + @intFromBool(opts.threads == 0);
        const bufs = try gpa.alloc(dio.Buffer, n);
        errdefer gpa.free(bufs);
        var made: usize = 0;
        errdefer for (bufs[0..made]) |b| gpa.free(b);
        while (made < n) : (made += 1) bufs[made] = try gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), opts.chunk);
        const folded = try gpa.alloc(f32, dio.fits(opts.chunk) / 2);
        return .{ .gpa = gpa, .cat = cat, .bufs = bufs, .folded = folded, .opts = opts };
    }

    pub fn deinit(s: *Streamer) void {
        for (s.bufs) |b| s.gpa.free(b);
        s.gpa.free(s.bufs);
        s.gpa.free(s.folded);
        s.* = undefined;
    }

    /// Bytes of buffers held: the loader's whole read-side memory.
    pub fn bufferBytes(s: *const Streamer) usize {
        return s.bufs.len * s.opts.chunk;
    }

    /// Every EXL3 marker holds mul1, else the stream is refused before any copy; each is read once, in file order.
    pub fn verifyMarkers(s: *Streamer, plan: *const Plan) !void {
        const marks = try sortedMarks(s.gpa, plan);
        defer s.gpa.free(marks);
        var ctx: MarkCtx = .{ .plan = plan };
        const wins = try schedule(Marks, s.gpa, .{ .marks = marks }, dio.fits(s.opts.chunk), s.opts.gap);
        defer s.gpa.free(wins);
        s.stats.markers = marks.len;
        s.stats.marker_windows = wins.len;
        try s.pipeline(Marks, wins, &ctx, MarkCtx.consume);
    }

    /// The reads `run` would make for `plan`, counted without reading anything.
    pub fn measure(s: *const Streamer, plan: *const Plan) !Stats {
        const room = dio.fits(s.opts.chunk);
        var out: Stats = .{ .markers = plan.matrices.len };
        const marks = try sortedMarks(s.gpa, plan);
        defer s.gpa.free(marks);
        const mw = try schedule(Marks, s.gpa, .{ .marks = marks }, room, s.opts.gap);
        defer s.gpa.free(mw);
        out.marker_windows = mw.len;
        const wins = try schedule(Runs, s.gpa, .{ .plan = plan }, room, s.opts.gap);
        defer s.gpa.free(wins);
        out.windows = wins.len;
        for (wins) |w| out.read_bytes += w.end - w.lo;
        for (plan.items) |it| out.wanted_bytes += it.span.bytes();
        return out;
    }

    pub fn run(s: *Streamer, plan: *const Plan, sink: Sink) !void {
        try s.verifyMarkers(plan);
        const room = dio.fits(s.opts.chunk);
        const wins = try schedule(Runs, s.gpa, .{ .plan = plan }, room, s.opts.gap);
        defer s.gpa.free(wins);
        s.stats.windows = wins.len;
        s.stats.read_bytes = 0;
        s.stats.wanted_bytes = 0;
        for (wins) |w| s.stats.read_bytes += w.end - w.lo;
        var ctx: PlanCtx = .{ .s = s, .sink = sink };
        try s.pipeline(Runs, wins, &ctx, PlanCtx.consume);
        if (ctx.shard_open) |o| s.shardEnded(o);
        s.stats.wanted_bytes = ctx.wanted;
    }

    fn shardEnded(s: *Streamer, shard: u16) void {
        if (s.ended) |e| e.call(e.ctx, shard);
    }

    // ---- the pipeline: windows read by threads, consumed in order by the caller ----

    fn Consume(comptime I: type) type {
        return *const fn (ctx: *anyopaque, index: usize, w: Window(I), data: []u8) anyerror!void;
    }

    fn pipeline(s: *Streamer, comptime I: type, wins: []const Window(I), ctx: *anyopaque, consume: Consume(I)) !void {
        if (wins.len == 0) return;
        const want = @min(s.opts.threads, wins.len);
        if (want == 0) return s.inline_(I, wins, ctx, consume);
        const state = try s.gpa.alloc(std.atomic.Value(u32), wins.len);
        defer s.gpa.free(state);
        for (state) |*x| x.* = .init(0);
        const fail = try s.gpa.alloc(u16, wins.len);
        defer s.gpa.free(fail);
        const got = try s.gpa.alloc([]u8, wins.len);
        defer s.gpa.free(got);
        const reads = try s.gpa.alloc(Read, wins.len);
        defer s.gpa.free(reads);
        for (reads, wins) |*r, w| r.* = .{ .shard = w.shard, .lo = w.lo, .len = @intCast(w.end - w.lo) };
        var sh: Shared = .{ .s = s, .reads = reads, .state = state, .fail = fail, .got = got };
        const threads = try s.gpa.alloc(std.Thread, want);
        defer s.gpa.free(threads);
        var spawned: usize = 0;
        while (spawned < want) : (spawned += 1) {
            threads[spawned] = std.Thread.spawn(.{}, Shared.worker, .{&sh}) catch break;
        }
        defer {
            sh.abort.store(true, .release);
            sh.bell.ring();
            for (threads[0..spawned]) |t| t.join();
        }
        // no thread could start: the calling thread reads every window itself
        if (spawned == 0) return s.inline_(I, wins, ctx, consume);
        for (wins, 0..) |w, i| {
            while (true) {
                const seen = sh.bell.now();
                if (state[i].load(.acquire) != 0) break;
                sh.bell.wait(seen);
            }
            if (state[i].load(.acquire) == 2) return @errorFromInt(fail[i]);
            try consume(ctx, i, w, got[i]);
            sh.released.store(i + 1, .release);
            sh.bell.ring();
        }
    }

    fn inline_(s: *Streamer, comptime I: type, wins: []const Window(I), ctx: *anyopaque, consume: Consume(I)) !void {
        for (wins, 0..) |w, i| {
            const data = try s.cat.files.items[w.shard].read(s.bufs[0], w.lo, @intCast(w.end - w.lo));
            try consume(ctx, i, w, data);
        }
    }

    /// The old strategy, kept as the test reference and the benchmark baseline: one read per run, one at a time.
    pub fn verifyMarkersSerial(s: *Streamer, plan: *const Plan) !void {
        for (plan.matrices) |m| {
            const got = try s.cat.files.items[m.marker_shard].read(s.bufs[0], m.marker_at, 4);
            if (std.mem.readInt(u32, got[0..4], .little) != load.mul1_multiplier) {
                std.log.warn("layer {d} {t} expert {d}: codebook marker is not mul1", .{ m.layer, m.proj, m.expert });
                return error.BadCodebookMarker;
            }
        }
    }

    pub fn runSerial(s: *Streamer, plan: *const Plan, sink: Sink) !void {
        try s.verifyMarkersSerial(plan);
        const buf = s.bufs[0];
        const room = dio.fits(buf.len);
        var shard_open: ?u16 = null;
        for (plan.items) |it| {
            if (shard_open) |o| if (o != it.shard) s.shardEnded(o);
            shard_open = it.shard;
            const file = s.cat.files.items[it.shard];
            const limit = plan.slots[it.slot].bytes;
            if (it.fold) {
                // a norm weight: one read, bf16 to fp32 1 + w
                if (it.span.len > room or it.span.count != 1 or it.dst != 0) return error.BadPlan;
                const got = try file.read(buf, it.first(), @intCast(it.span.len));
                const out = load.foldGemma(s.folded, got);
                try sink.put(sink.ctx, it.slot, 0, std.mem.sliceAsBytes(out));
                continue;
            }
            var run_i: u64 = 0;
            while (run_i < it.span.count) : (run_i += 1) {
                var at = it.base + it.span.at(run_i);
                var dst = it.dst + run_i * it.span.len;
                var left: u64 = it.span.len;
                while (left > 0) {
                    const n: usize = @intCast(@min(left, room));
                    if (dst + n > limit) return error.BadPlan;
                    try sink.put(sink.ctx, it.slot, dst, try file.read(buf, at, n));
                    at += n;
                    dst += n;
                    left -= n;
                }
            }
        }
        if (shard_open) |o| s.shardEnded(o);
    }
};

/// A thread's wake-up: any change rings it, a waiter re-checks its condition (a futex where the OS has one).
const Bell = struct {
    seq: std.atomic.Value(u32) = .init(0),

    fn now(b: *Bell) u32 {
        return b.seq.load(.acquire);
    }

    fn ring(b: *Bell) void {
        _ = b.seq.fetchAdd(1, .release);
        if (builtin.os.tag == .linux) _ = std.os.linux.futex_3arg(&b.seq.raw, .{ .cmd = .WAKE, .private = true }, std.math.maxInt(i32));
    }

    /// Returns once the bell has rung since `seen` (or at once, spuriously).
    fn wait(b: *Bell, seen: u32) void {
        if (builtin.os.tag == .linux) {
            _ = std.os.linux.futex_4arg(&b.seq.raw, .{ .cmd = .WAIT, .private = true }, seen, null);
        } else if (b.seq.load(.acquire) == seen) std.Thread.yield() catch {};
    }
};

/// One read: `len` bytes at absolute offset `lo` of a shard.
const Read = struct { shard: u16, lo: u64, len: usize };

const Shared = struct {
    s: *Streamer,
    reads: []const Read,
    state: []std.atomic.Value(u32), // per window: 0 pending, 1 read, 2 failed
    fail: []u16,
    got: [][]u8,
    next: std.atomic.Value(usize) = .init(0),
    released: std.atomic.Value(usize) = .init(0), // windows copied: their buffers are free
    abort: std.atomic.Value(bool) = .init(false),
    bell: Bell = .{},

    /// Claims windows in order; window i reads into buffer i % n once the copy of window i - n is done with it.
    fn worker(sh: *Shared) void {
        const nbuf = sh.s.bufs.len;
        while (true) {
            const i = sh.next.fetchAdd(1, .monotonic);
            if (i >= sh.reads.len) return;
            while (true) {
                const seen = sh.bell.now();
                if (sh.abort.load(.acquire)) return;
                if (i < sh.released.load(.acquire) + nbuf) break;
                sh.bell.wait(seen);
            }
            const r = sh.reads[i];
            if (sh.s.cat.files.items[r.shard].read(sh.s.bufs[i % nbuf], r.lo, r.len)) |data| {
                sh.got[i] = data;
                sh.state[i].store(1, .release);
            } else |e| {
                sh.fail[i] = @intFromError(e);
                sh.state[i].store(2, .release);
            }
            sh.bell.ring();
        }
    }
};

// ---- the schedule: runs in file order folded into windows ----

/// A piece of one run still to read: `len` bytes at absolute offset `at`, bound for `dst` of `slot`.
const Piece = struct { shard: u16, at: u64, len: u64, slot: u32, dst: u64, fold: bool };

/// One read: the runs from `start` up to `stop` fall in [lo, end) of a shard, and none crosses `hard`.
fn Window(comptime I: type) type {
    return struct { shard: u16, lo: u64, end: u64, hard: u64, start: I, stop: I };
}

/// A plan's runs in the plan's (shard, offset) order, a run being cut where a window ends.
const Runs = struct {
    plan: *const Plan,
    item: usize = 0,
    run: u64 = 0,
    used: u64 = 0, // bytes of the current run already taken

    fn head(r: *Runs) ?Piece {
        while (r.item < r.plan.items.len) {
            const it = r.plan.items[r.item];
            if (r.run < it.span.count and it.span.len > 0) return .{
                .shard = it.shard,
                .at = it.base + it.span.at(r.run) + r.used,
                .len = it.span.len - r.used,
                .slot = it.slot,
                .dst = it.dst + r.run * it.span.len + r.used,
                .fold = it.fold,
            };
            r.item += 1;
            r.run = 0;
            r.used = 0;
        }
        return null;
    }

    fn skip(r: *Runs, n: u64) void {
        r.used += n;
        if (r.used >= r.plan.items[r.item].span.len) {
            r.run += 1;
            r.used = 0;
        }
    }

    /// Whether `a` and `b` are at the same place; each is first moved past runs with nothing in them.
    fn same(a: *Runs, b: Runs) bool {
        var other = b;
        _ = a.head();
        _ = other.head();
        return a.item == other.item and a.run == other.run and a.used == other.used;
    }

    /// The part of `p` a window takes must stay in its slot; a norm weight is one whole run (folded as a unit).
    fn admit(r: *const Runs, p: Piece, n: u64, room: u64) !void {
        if (p.dst + n > r.plan.slots[p.slot].bytes) return error.BadPlan;
        if (p.fold) {
            const it = r.plan.items[r.item];
            if (it.span.len > room or it.span.count != 1 or it.dst != 0) return error.BadPlan;
        }
    }
};

const Mark = struct { shard: u16, at: u64, matrix: u32 };

/// Every matrix' marker by (shard, offset).
fn sortedMarks(gpa: std.mem.Allocator, plan: *const Plan) ![]Mark {
    const marks = try gpa.alloc(Mark, plan.matrices.len);
    errdefer gpa.free(marks);
    for (marks, plan.matrices, 0..) |*mk, m, i| mk.* = .{ .shard = m.marker_shard, .at = m.marker_at, .matrix = @intCast(i) };
    std.mem.sort(Mark, marks, {}, struct {
        fn less(_: void, a: Mark, b: Mark) bool {
            return if (a.shard != b.shard) a.shard < b.shard else a.at < b.at;
        }
    }.less);
    return marks;
}

/// The codebook markers, sorted by (shard, offset), four bytes each.
const Marks = struct {
    marks: []const Mark,
    i: usize = 0,

    fn head(r: *Marks) ?Piece {
        if (r.i >= r.marks.len) return null;
        const m = r.marks[r.i];
        return .{ .shard = m.shard, .at = m.at, .len = 4, .slot = 0, .dst = 0, .fold = false };
    }

    fn skip(r: *Marks, _: u64) void {
        r.i += 1;
    }

    fn same(a: *Marks, b: Marks) bool {
        return a.i == b.i;
    }

    fn admit(_: *const Marks, _: Piece, _: u64, _: u64) !void {}
};

/// Folds the runs from `start` into windows of at most `room` bytes, one read each, joining runs under `gap` apart.
fn schedule(comptime I: type, gpa: std.mem.Allocator, start: I, room: u64, gap: u64) ![]Window(I) {
    const W = Window(I);
    var list: std.ArrayList(W) = .empty;
    errdefer list.deinit(gpa);
    var it = start;
    var open = false;
    var w: W = undefined;
    while (it.head()) |h| {
        if (open and (h.shard != w.shard or h.at < w.end or h.at >= w.hard or h.at - w.end > gap)) {
            w.stop = it;
            try list.append(gpa, w);
            open = false;
        }
        if (!open) {
            w = .{ .shard = h.shard, .lo = h.at, .end = h.at, .hard = h.at + room, .start = it, .stop = undefined };
            open = true;
        }
        const n = @min(h.len, w.hard - h.at);
        if (h.fold and n < h.len) {
            // a folded run is never split: it opens a window of its own
            if (w.end == w.lo) return error.BadPlan;
            w.stop = it;
            try list.append(gpa, w);
            open = false;
            continue;
        }
        try it.admit(h, n, room);
        w.end = h.at + n;
        it.skip(n);
    }
    if (open) {
        w.stop = it;
        try list.append(gpa, w);
    }
    return list.toOwnedSlice(gpa);
}

/// The plan's side of the pipeline: copies a window's runs to the sink, packed per destination.
const PlanCtx = struct {
    s: *Streamer,
    sink: Sink,
    shard_open: ?u16 = null,
    wanted: u64 = 0,
    since_drop: u64 = 0,

    /// Bytes of a file read between page-cache drops, for a file system that refused the direct read.
    const drop_every = 512 << 20;

    fn consume(ctx: *anyopaque, _: usize, w: Window(Runs), data: []u8) anyerror!void {
        const c: *PlanCtx = @ptrCast(@alignCast(ctx));
        if (c.shard_open) |o| if (o != w.shard) {
            c.s.shardEnded(o);
            c.since_drop = 0;
        };
        c.shard_open = w.shard;
        const file = c.s.cat.files.items[w.shard];
        var it = w.start;
        var packed_to: usize = 0; // wanted bytes are packed to the front of `data` as they are met
        var group: ?Group = null;
        while (!it.same(w.stop)) {
            const h = it.head().?;
            const n: usize = @intCast(@min(h.len, w.hard - h.at));
            const src: usize = @intCast(h.at - w.lo);
            if (h.fold) {
                try c.flush(&group, data, packed_to);
                const out = load.foldGemma(c.s.folded, data[src..][0..n]);
                try c.sink.put(c.sink.ctx, h.slot, h.dst, std.mem.sliceAsBytes(out));
            } else {
                if (group) |g| if (g.slot != h.slot or g.dst + (packed_to - g.from) != h.dst) try c.flush(&group, data, packed_to);
                if (group == null) group = .{ .slot = h.slot, .dst = h.dst, .from = packed_to };
                if (packed_to != src) @memmove(data[packed_to..][0..n], data[src..][0..n]);
                packed_to += n;
            }
            c.wanted += n;
            it.skip(n);
        }
        try c.flush(&group, data, packed_to);
        c.since_drop += data.len;
        if (c.since_drop >= drop_every) {
            file.dropCache();
            c.since_drop = 0;
        }
    }

    const Group = struct { slot: u32, dst: u64, from: usize };

    fn flush(c: *PlanCtx, group: *?Group, data: []u8, packed_to: usize) !void {
        const g = group.* orelse return;
        group.* = null;
        var at: usize = 0;
        const all = data[g.from..packed_to];
        while (at < all.len) {
            const n = @min(all.len - at, c.s.opts.put_max);
            try c.sink.put(c.sink.ctx, g.slot, g.dst + at, all[at..][0..n]);
            at += n;
        }
    }
};

/// The marker side: each marker of a window is the mul1 multiplier or the stream stops.
const MarkCtx = struct {
    plan: *const Plan,

    fn consume(ctx: *anyopaque, _: usize, w: Window(Marks), data: []u8) anyerror!void {
        const c: *MarkCtx = @ptrCast(@alignCast(ctx));
        var it = w.start;
        while (!it.same(w.stop)) : (it.skip(4)) {
            const m = it.marks[it.i];
            const at: usize = @intCast(m.at - w.lo);
            if (std.mem.readInt(u32, data[at..][0..4], .little) != load.mul1_multiplier) {
                const x = c.plan.matrices[m.matrix];
                std.log.warn("layer {d} {t} expert {d}: codebook marker is not mul1", .{ x.layer, x.proj, x.expert });
                return error.BadCodebookMarker;
            }
        }
    }
};
