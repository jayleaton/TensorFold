//! mm3-loadbench: the MiniMax-M3.1-Flash checkpoint reader's diagnosis and micro-benchmark (host only, no GPU).
const std = @import("std");
const core = @import("core");
const dio = core.direct_io;
const load = @import("load.zig");
const config = @import("config.zig");
const shard = @import("shard.zig");
const names = @import("names.zig");

const usage =
    \\
    \\  mm3-loadbench stats   <model dir> <rank> <world>              run lengths of a rank's plan, reads old and new
    \\  mm3-loadbench model   <model dir> <rank> <world> [options]    stream a slice of the real plan, old and new strategy
    \\  mm3-loadbench pattern <file> <run> <stride> <wanted MiB> [options]   strided runs of one file, old and new strategy
    \\  mm3-loadbench raw     <file> <block KiB> <threads> <MiB>      sequential direct reads: the device's ceiling
    \\
    \\options: --layers N --first L (model: the plan is cut to N layers from layer L, no embedding or head), --only old|new,
    \\         --threads N --chunk MiB --gap KiB (the new strategy), --limit-gib G (default 8: the most one test may read).
    \\Every read is O_DIRECT, so the page cache does not grow; all buffers together stay within 512 MiB; the tool stops when
    \\MemAvailable is under 9 GiB, before and while it reads.
;

const gpa = std.heap.smp_allocator;
const min_avail_kib: u64 = 9 << 20;
const max_buffer_bytes: u64 = 512 << 20;
const mib = 1024.0 * 1024.0;

var io_: std.Io = undefined;

fn out(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(fmt, args);
}

fn availKiB() !u64 {
    // /proc files report size 0, so read until the end by hand
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.BadMeminfo;
    defer _ = std.c.close(fd);
    var text: [8192]u8 = undefined;
    const n = std.c.read(fd, &text, text.len);
    if (n <= 0) return error.BadMeminfo;
    var lines = std.mem.tokenizeScalar(u8, text[0..@intCast(n)], '\n');
    while (lines.next()) |l| {
        if (!std.mem.startsWith(u8, l, "MemAvailable:")) continue;
        var f = std.mem.tokenizeAny(u8, l["MemAvailable:".len..], " \tkB");
        return try std.fmt.parseInt(u64, f.next() orelse return error.BadMeminfo, 10);
    }
    return error.BadMeminfo;
}

fn guard() !void {
    const kib = try availKiB();
    if (kib < min_avail_kib) {
        out("MemAvailable {d} MiB is under 9 GiB: stopping\n", .{kib >> 10});
        return error.LowMemory;
    }
}

fn now() i96 {
    return std.Io.Clock.awake.now(io_).toNanoseconds();
}

/// Counts what it is given, touches a few bytes of it, and watches MemAvailable.
const Null = struct {
    bytes: u64 = 0,
    puts: u64 = 0,
    sum: u64 = 0,
    since: u64 = 0,
    min_avail: u64 = std.math.maxInt(u64),
    fault: ?anyerror = null,

    fn put(ctx: *anyopaque, _: u32, _: u64, bytes: []const u8) anyerror!void {
        const n: *Null = @ptrCast(@alignCast(ctx));
        n.bytes += bytes.len;
        n.puts += 1;
        if (bytes.len >= 8) n.sum +%= std.mem.readInt(u64, bytes[0..8], .little) +% std.mem.readInt(u64, bytes[bytes.len - 8 ..][0..8], .little);
        n.since += bytes.len;
        if (n.since >= 256 << 20) {
            n.since = 0;
            const kib = try availKiB();
            n.min_avail = @min(n.min_avail, kib);
            if (kib < min_avail_kib) return error.LowMemory;
        }
    }

    fn sink(n: *Null) load.Sink {
        return .{ .ctx = n, .put = put };
    }
};

const Settings = struct {
    layers: usize = 1,
    first: u16 = 10,
    only: ?enum { old, new } = null,
    opts: load.StreamOptions = .{},
    limit_gib: f64 = 8,
};

fn parseOptions(args: []const [:0]const u8, s: *Settings) !void {
    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return error.OptionValueRequired;
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--only")) {
            s.only = std.meta.stringToEnum(@TypeOf(s.only.?), v) orelse return error.BadOption;
        } else if (std.mem.eql(u8, k, "--limit-gib")) {
            s.limit_gib = try std.fmt.parseFloat(f64, v);
        } else {
            const n = try std.fmt.parseInt(usize, v, 10);
            if (std.mem.eql(u8, k, "--layers")) s.layers = n else if (std.mem.eql(u8, k, "--first")) s.first = @intCast(n) else if (std.mem.eql(u8, k, "--threads")) s.opts.threads = n else if (std.mem.eql(u8, k, "--chunk")) s.opts.chunk = n << 20 else if (std.mem.eql(u8, k, "--gap")) s.opts.gap = n << 10 else if (std.mem.eql(u8, k, "--ahead")) s.opts.ahead = n else return error.BadOption;
        }
    }
}

// ---- stats ----

const Bucket = struct { name: []const u8, below: u64 };
const buckets = [_]Bucket{
    .{ .name = "<4K", .below = 4 << 10 },
    .{ .name = "4-16K", .below = 16 << 10 },
    .{ .name = "16-64K", .below = 64 << 10 },
    .{ .name = "64-256K", .below = 256 << 10 },
    .{ .name = "256K-1M", .below = 1 << 20 },
    .{ .name = "1-8M", .below = 8 << 20 },
    .{ .name = ">=8M", .below = std.math.maxInt(u64) },
};

fn bucketOf(len: u64) usize {
    for (buckets, 0..) |b, i| if (len < b.below) return i;
    unreachable;
}

const Agg = struct {
    tensors: u64 = 0,
    runs: u64 = 0,
    bytes: u64 = 0,
    hist_runs: [buckets.len]u64 = @splat(0),
    hist_bytes: [buckets.len]u64 = @splat(0),
    min_len: u64 = std.math.maxInt(u64),
    max_len: u64 = 0,
};

fn kindNames() []const []const u8 {
    comptime {
        var all: []const []const u8 = &.{};
        for (std.meta.fieldNames(names.Proj)) |p| for ([_][]const u8{ "trellis", "suh", "svh" }) |part| {
            all = all ++ &[_][]const u8{p ++ "." ++ part};
        };
        for (std.meta.fieldNames(names.Plain)) |p| all = all ++ &[_][]const u8{"plain." ++ p};
        return all;
    }
}

fn stats(dir: []const u8, rank: usize, world: usize, s: Settings) !void {
    var cat = try load.Catalog.open(gpa, io_, dir);
    defer cat.deinit();
    const c = try loadConfig(dir);
    var plan = try load.Plan.build(gpa, &cat, &c, try shard.Share.of(&c, rank, world), .{});
    defer plan.deinit();
    const kinds = comptime kindNames();
    const n_proj = std.meta.fieldNames(names.Proj).len;
    const kind_of = try gpa.alloc(u16, plan.slots.len);
    defer gpa.free(kind_of);
    @memset(kind_of, std.math.maxInt(u16));
    for (plan.matrices) |m| {
        const base: u16 = @as(u16, @backingInt(m.proj)) * 3;
        kind_of[m.trellis] = base;
        kind_of[m.suh] = base + 1;
        kind_of[m.svh] = base + 2;
    }
    for (plan.plain) |p| kind_of[p.slot] = @intCast(n_proj * 3 + @as(usize, @backingInt(p.role)));
    const aggs = try gpa.alloc(Agg, kinds.len);
    defer gpa.free(aggs);
    @memset(aggs, .{});
    var all: Agg = .{};
    var old_reads: u64 = 0;
    const room = dio.fits(8 << 20); // the old streamer's buffer
    for (plan.items) |it| {
        const a = &aggs[kind_of[it.slot]];
        a.tensors += 1;
        a.runs += it.span.count;
        a.bytes += it.span.bytes();
        if (it.span.len > 0 and it.span.count > 0) {
            const b = bucketOf(it.span.len);
            a.hist_runs[b] += it.span.count;
            a.hist_bytes[b] += it.span.bytes();
            a.min_len = @min(a.min_len, it.span.len);
            a.max_len = @max(a.max_len, it.span.len);
            old_reads += it.span.count * ((it.span.len + room - 1) / room);
        }
    }
    for (aggs) |a| {
        all.tensors += a.tensors;
        all.runs += a.runs;
        all.bytes += a.bytes;
        for (0..buckets.len) |i| {
            all.hist_runs[i] += a.hist_runs[i];
            all.hist_bytes[i] += a.hist_bytes[i];
        }
        all.min_len = @min(all.min_len, a.min_len);
        all.max_len = @max(all.max_len, a.max_len);
    }
    out("\n## rank {d} of {d}: {d} shards, {d} tensors read, {d:.2} GiB wanted\n\n", .{ rank, world, cat.files.items.len, all.tensors, @as(f64, @floatFromInt(all.bytes)) / 1024 / 1024 / 1024 });
    out("| kind | tensors | runs | bytes MiB | min run | max run |", .{});
    for (buckets) |b| out(" runs {s} |", .{b.name});
    out("\n|---|---:|---:|---:|---:|---:|", .{});
    for (buckets) |_| out("---:|", .{});
    out("\n", .{});
    for (kinds, aggs) |k, a| {
        if (a.runs == 0) continue;
        out("| {s} | {d} | {d} | {d:.1} | {d} | {d} |", .{ k, a.tensors, a.runs, @as(f64, @floatFromInt(a.bytes)) / mib, a.min_len, a.max_len });
        for (a.hist_runs) |h| out(" {d} |", .{h});
        out("\n", .{});
    }
    out("| all | {d} | {d} | {d:.1} | {d} | {d} |", .{ all.tensors, all.runs, @as(f64, @floatFromInt(all.bytes)) / mib, all.min_len, all.max_len });
    for (all.hist_runs) |h| out(" {d} |", .{h});
    out("\n\nbytes by run length bucket (MiB):", .{});
    for (buckets, all.hist_bytes) |b, h| out(" {s}={d:.1}", .{ b.name, @as(f64, @floatFromInt(h)) / mib });
    var st = try load.Streamer.initWith(gpa, &cat, s.opts);
    defer st.deinit();
    const m = try st.measure(&plan);
    out("\n\nold strategy: {d} payload reads + {d} marker reads = {d} reads, {d:.1} KiB per payload read\n", .{
        old_reads, plan.matrices.len, old_reads + plan.matrices.len, @as(f64, @floatFromInt(all.bytes)) / 1024 / @as(f64, @floatFromInt(old_reads)),
    });
    out("new strategy (chunk {d} MiB, gap {d} KiB): {d} payload reads of {d:.2} MiB mean + {d} marker reads; reads cover {d:.2} GiB for {d:.2} GiB wanted ({d:.2}x)\n", .{
        s.opts.chunk >> 20,                                           s.opts.gap >> 10,
        m.windows,                                                    @as(f64, @floatFromInt(m.read_bytes)) / mib / @as(f64, @floatFromInt(m.windows)),
        m.marker_windows,                                             @as(f64, @floatFromInt(m.read_bytes)) / 1024 / 1024 / 1024,
        @as(f64, @floatFromInt(m.wanted_bytes)) / 1024 / 1024 / 1024, @as(f64, @floatFromInt(m.read_bytes)) / @as(f64, @floatFromInt(m.wanted_bytes)),
    });
}

fn loadConfig(dir: []const u8) !config.Config {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io_, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    return config.parse(gpa, text);
}

// ---- running a plan both ways ----

const Result = struct { seconds: f64, wanted: u64, read: u64, reads: u64, markers: f64, puts: u64, min_avail: u64, sum: u64 };

fn report(name: []const u8, r: Result) void {
    out("{s:<4} {d:8.1} MiB/s wanted  {d:8.1} MiB/s read  {d:7.2} s total ({d:.2} s markers)  {d} reads  {d:.1} KiB/read  {d} puts  min MemAvailable {d} MiB  chk {x}\n", .{
        name,
        @as(f64, @floatFromInt(r.wanted)) / mib / r.seconds,
        @as(f64, @floatFromInt(r.read)) / mib / r.seconds,
        r.seconds,
        r.markers,
        r.reads,
        @as(f64, @floatFromInt(r.read)) / 1024 / @as(f64, @floatFromInt(r.reads)),
        r.puts,
        r.min_avail >> 10,
        r.sum,
    });
}

/// Streams `plan` with the strategy `new` selects; the reads are counted from the plan, not measured.
fn stream(cat: *const load.Catalog, plan: *const load.Plan, s: Settings, new: bool) !Result {
    var st = try load.Streamer.initWith(gpa, cat, if (new) s.opts else .{ .chunk = 8 << 20, .threads = 0, .ahead = 0 });
    defer st.deinit();
    if (st.bufferBytes() > max_buffer_bytes) return error.TooMuchBuffer;
    var sink: Null = .{};
    try guard();
    // the marker pass alone, then the whole stream (which checks the markers again): the second is the load's time
    const t0 = now();
    if (new) try st.verifyMarkers(plan) else try st.verifyMarkersSerial(plan);
    const markers = @as(f64, @floatFromInt(now() - t0)) / 1e9;
    const t1 = now();
    if (new) try st.run(plan, sink.sink()) else try st.runSerial(plan, sink.sink());
    const seconds = @as(f64, @floatFromInt(now() - t1)) / 1e9;
    var reads: u64 = 0;
    var read: u64 = 0;
    var wanted: u64 = 0;
    for (plan.items) |it| wanted += it.span.bytes();
    if (new) {
        const m = try st.measure(plan);
        reads = m.windows + m.marker_windows;
        read = m.read_bytes + 4096 * m.marker_windows;
    } else {
        const room = dio.fits(8 << 20);
        for (plan.items) |it| {
            if (it.span.len == 0) continue;
            reads += it.span.count * ((it.span.len + room - 1) / room);
        }
        read = wanted;
        reads += plan.matrices.len;
        read += 4096 * plan.matrices.len;
    }
    return .{ .seconds = seconds, .wanted = wanted, .read = read, .reads = reads, .markers = markers, .puts = sink.puts, .min_avail = sink.min_avail, .sum = sink.sum };
}

fn bothWays(cat: *const load.Catalog, plan: *const load.Plan, s: Settings) !void {
    var wanted: u64 = 0;
    for (plan.items) |it| wanted += it.span.bytes();
    var probe = try load.Streamer.initWith(gpa, cat, s.opts);
    const m = try probe.measure(plan);
    probe.deinit();
    const worst: f64 = @floatFromInt(@max(m.read_bytes, wanted));
    out("plan: {d} items, {d} matrices, {d:.2} GiB wanted, new strategy reads {d:.2} GiB in {d} reads (+{d} marker reads)\n", .{ plan.items.len, plan.matrices.len, @as(f64, @floatFromInt(wanted)) / 1024 / mib, @as(f64, @floatFromInt(m.read_bytes)) / 1024 / mib, m.windows, m.marker_windows });
    if (worst / 1024 / mib > s.limit_gib) {
        out("this test would read {d:.2} GiB, over the limit of {d:.2} GiB: use fewer layers\n", .{ worst / 1024 / mib, s.limit_gib });
        return error.OverReadLimit;
    }
    // old, new, old, new: a second pair shows the run to run spread
    for ([_]bool{ false, true, false, true }) |new| {
        if (s.only) |o| if ((o == .new) != new) continue;
        report(if (new) "new" else "old", try stream(cat, plan, s, new));
    }
}

/// The plan cut to `layers` layers from `first`, without the embedding and the head (one in-memory slice).
fn slice(base: *const load.Plan, keep_first: u16, keep_n: usize) !load.Plan {
    var p = base.*;
    var keep: std.ArrayList(bool) = .empty;
    defer keep.deinit(gpa);
    try keep.appendNTimes(gpa, true, base.slots.len);
    for (base.plain) |t| if (t.role == .embed or (t.layer < keep_first or t.layer >= keep_first + keep_n)) {
        keep.items[t.slot] = false;
    };
    for (base.matrices) |m| if (m.proj == .head or m.layer < keep_first or m.layer >= keep_first + keep_n) {
        keep.items[m.trellis] = false;
        keep.items[m.suh] = false;
        keep.items[m.svh] = false;
    };
    var items: std.ArrayList(load.Item) = .empty;
    errdefer items.deinit(gpa);
    for (base.items) |it| if (keep.items[it.slot]) try items.append(gpa, it);
    var ms: std.ArrayList(load.Matrix) = .empty;
    errdefer ms.deinit(gpa);
    for (base.matrices) |m| if (keep.items[m.trellis]) try ms.append(gpa, m);
    p.items = try items.toOwnedSlice(gpa);
    p.matrices = try ms.toOwnedSlice(gpa);
    return p;
}

fn model(dir: []const u8, rank: usize, world: usize, s: Settings) !void {
    var cat = try load.Catalog.open(gpa, io_, dir);
    defer cat.deinit();
    const c = try loadConfig(dir);
    var layers: [64]u16 = undefined;
    for (layers[0..s.layers], 0..) |*l, i| l.* = @intCast(s.first + i);
    var full = try load.Plan.build(gpa, &cat, &c, try shard.Share.of(&c, rank, world), .{ .layers = layers[0..s.layers] });
    defer full.deinit();
    const p = try slice(&full, s.first, s.layers);
    defer gpa.free(p.items);
    defer gpa.free(p.matrices);
    out("model {s} rank {d}/{d}: layers {d}..{d}\n", .{ dir, rank, world, s.first, s.first + s.layers });
    try bothWays(&cat, &p, s);
}

// ---- strided runs of one file ----

fn pattern(path: [:0]const u8, run: u64, stride: u64, wanted_mib: u64, s: Settings) !void {
    var cat = load.Catalog.init(gpa);
    defer cat.deinit();
    try cat.files.append(gpa, try dio.File.open(path));
    const count = wanted_mib * (1 << 20) / run;
    var slots = [_]load.Slot{.{ .bytes = count * run }};
    var items = [_]load.Item{.{ .shard = 0, .base = 0, .span = .{ .offset = 0, .len = run, .stride = stride, .count = count }, .slot = 0, .dst = 0 }};
    const c = config.flash;
    const plan: load.Plan = .{
        .arena = .init(gpa),
        .share = try shard.Share.of(&c, 0, 2),
        .slots = &slots,
        .items = &items,
        .matrices = &.{},
        .plain = &.{},
        .moe = &.{},
        .ignored = 0,
    };
    out("pattern {s} (direct {}): {d} runs of {d} bytes every {d} bytes ({d} MiB wanted, spans {d:.2} GiB of the file)\n", .{ path, cat.files.items[0].direct, count, run, stride, wanted_mib, @as(f64, @floatFromInt(count * stride)) / 1024 / mib });
    try bothWays(&cat, &plan, s);
}

// ---- the device's ceiling ----

const Raw = struct {
    file: dio.File,
    block: usize,
    blocks: u64,
    next: std.atomic.Value(u64) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    fn worker(r: *Raw) void {
        const buf = gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), r.block) catch {
            r.failed.store(true, .release);
            return;
        };
        defer gpa.free(buf);
        while (true) {
            const i = r.next.fetchAdd(1, .monotonic);
            if (i >= r.blocks or r.failed.load(.acquire)) return;
            _ = r.file.read(buf, i * r.block, r.block) catch {
                r.failed.store(true, .release);
                return;
            };
        }
    }
};

fn raw(path: [:0]const u8, block_kib: usize, threads: usize, total_mib: usize) !void {
    if (@as(u64, block_kib) * 1024 * threads > max_buffer_bytes) return error.TooMuchBuffer;
    try guard();
    var r: Raw = .{ .file = try dio.File.open(path), .block = block_kib * 1024, .blocks = total_mib * (1 << 20) / (block_kib * 1024) };
    defer r.file.close();
    const hs = try gpa.alloc(std.Thread, threads);
    defer gpa.free(hs);
    const t0 = now();
    for (hs) |*h| h.* = try std.Thread.spawn(.{}, Raw.worker, .{&r});
    for (hs) |h| h.join();
    const secs = @as(f64, @floatFromInt(now() - t0)) / 1e9;
    if (r.failed.load(.acquire)) return error.ReadFailed;
    out("raw {s} (direct {}): {d} KiB blocks x {d} threads: {d:.1} MiB/s ({d} MiB in {d:.2} s)\n", .{ path, r.file.direct, block_kib, threads, @as(f64, @floatFromInt(total_mib)) / secs, total_mib, secs });
}

pub fn main(init: std.process.Init) !void {
    io_ = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        out("{s}\n", .{usage});
        return error.Usage;
    }
    try guard();
    var s: Settings = .{};
    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "stats") and args.len >= 5) {
        try parseOptions(args[5..], &s);
        return stats(args[2], try std.fmt.parseInt(usize, args[3], 10), try std.fmt.parseInt(usize, args[4], 10), s);
    }
    if (std.mem.eql(u8, cmd, "model") and args.len >= 5) {
        try parseOptions(args[5..], &s);
        return model(args[2], try std.fmt.parseInt(usize, args[3], 10), try std.fmt.parseInt(usize, args[4], 10), s);
    }
    if (std.mem.eql(u8, cmd, "pattern") and args.len >= 6) {
        try parseOptions(args[6..], &s);
        return pattern(args[2], try std.fmt.parseInt(u64, args[3], 10), try std.fmt.parseInt(u64, args[4], 10), try std.fmt.parseInt(u64, args[5], 10), s);
    }
    if (std.mem.eql(u8, cmd, "raw") and args.len >= 6) {
        return raw(args[2], try std.fmt.parseInt(usize, args[3], 10), try std.fmt.parseInt(usize, args[4], 10), try std.fmt.parseInt(usize, args[5], 10));
    }
    return error.Usage;
}
