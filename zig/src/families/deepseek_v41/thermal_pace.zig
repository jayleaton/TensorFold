//! TF_DSV41_THERMAL_PACE (ours): idle gaps between prefill segments and decode windows while hot; the same bits.
const std = @import("std");

pub const Error = error{BadKnob};

/// Reads one setting; null when unset or empty (the environment, a fixture in tests).
pub const Get = *const fn ([]const u8) ?[]const u8;

/// The largest gap / busy ratio: rounds at most 3x slower stay under /health's slow limits (core/pace.zig: 4x, 8x).
pub const max_ratio: f64 = 2;
/// A decode window more than this after the previous one starts a new busy span (the gap was idle or a prefill's).
pub const idle_ns: u64 = 500 * std.time.ns_per_ms;
/// The zone files read at most (the default scan stops here too).
pub const max_zones = 16;
/// An unreadable source: no gap.
const unknown: i32 = std.math.minInt(i32);

/// The pacing curve in millidegrees C, and the longest single gap.
pub const Curve = struct {
    soft_mc: i32 = 86_000,
    full_mc: i32 = 90_000,
    ratio: f64 = 1,
    max_ns: u64 = 4 * std.time.ns_per_s,

    /// The gap ratio at `temp_mc` (null: unknown).
    pub fn ratioAt(c: Curve, temp_mc: ?i32) f64 {
        const t = temp_mc orelse return 0;
        if (t <= c.soft_mc) return 0;
        const over: f64 = @floatFromInt(@as(i64, t) - c.soft_mc);
        const span: f64 = @floatFromInt(@as(i64, c.full_mc) - c.soft_mc);
        return @min(max_ratio, c.ratio * over / span);
    }

    /// The idle gap after `busy_ns` of work at `temp_mc`, at most `max_ns`.
    pub fn gapNs(c: Curve, busy_ns: u64, temp_mc: ?i32) u64 {
        const ns = c.ratioAt(temp_mc) * @as(f64, @floatFromInt(busy_ns));
        if (!(ns > 0)) return 0;
        return @min(c.max_ns, @as(u64, @intFromFloat(@min(ns, 1e18))));
    }
};

/// The knobs: null when TF_DSV41_THERMAL_PACE is off.
pub const Knobs = struct {
    curve: Curve = .{},
    read_ns: u64 = 250 * std.time.ns_per_ms,
    /// TF_DSV41_THERMAL_PACE_ZONES (comma separated), empty: every /sys/class/thermal/thermal_zone*/temp
    zones: []const u8 = "",
};

/// TF_DSV41_THERMAL_PACE (0 / unset off, 1 on) and its _SOFT_C, _FULL_C, _RATIO, _MAX_MS, _READ_MS, _ZONES.
pub fn knobs(get: Get) Error!?Knobs {
    const on = std.mem.trim(u8, get("TF_DSV41_THERMAL_PACE") orelse "0", " ");
    if (std.mem.eql(u8, on, "0")) return null;
    if (!std.mem.eql(u8, on, "1")) return bad("TF_DSV41_THERMAL_PACE", on, "0 or 1");
    var k: Knobs = .{};
    const soft = try num(get, "TF_DSV41_THERMAL_PACE_SOFT_C", 86, 30, 110);
    const full = try num(get, "TF_DSV41_THERMAL_PACE_FULL_C", 90, 30, 120);
    if (full < soft + 0.5) return bad("TF_DSV41_THERMAL_PACE_FULL_C", get("TF_DSV41_THERMAL_PACE_FULL_C") orelse "90", "at least SOFT_C + 0.5");
    k.curve.soft_mc = @intFromFloat(@round(soft * 1000));
    k.curve.full_mc = @intFromFloat(@round(full * 1000));
    k.curve.ratio = try num(get, "TF_DSV41_THERMAL_PACE_RATIO", 1, 0.01, max_ratio);
    k.curve.max_ns = @intFromFloat(try num(get, "TF_DSV41_THERMAL_PACE_MAX_MS", 4000, 1, 60_000) * std.time.ns_per_ms);
    k.read_ns = @intFromFloat(try num(get, "TF_DSV41_THERMAL_PACE_READ_MS", 250, 10, 10_000) * std.time.ns_per_ms);
    k.zones = std.mem.trim(u8, get("TF_DSV41_THERMAL_PACE_ZONES") orelse "", " ");
    return k;
}

fn num(get: Get, name: []const u8, default: f64, lo: f64, hi: f64) Error!f64 {
    const raw = get(name) orelse return default;
    const v = std.fmt.parseFloat(f64, std.mem.trim(u8, raw, " ")) catch return bad(name, raw, "a number");
    if (!(v >= lo and v <= hi)) return bad(name, raw, "in range");
    return v;
}

fn bad(name: []const u8, raw: []const u8, want: []const u8) Error {
    std.log.scoped(.dsv41).warn("{s}={s}: expected {s} (thermal_pace.zig)", .{ name, raw, want });
    return error.BadKnob;
}

/// A sysfs temperature's text (millidegrees C, an integer and a newline); null when it is not one.
pub fn parseMilli(text: []const u8) ?i32 {
    const v = std.fmt.parseInt(i32, std.mem.trim(u8, text, " \t\r\n"), 10) catch return null;
    // below -50 C or above 200 C: an unpopulated or broken sensor, not a reading
    return if (v < -50_000 or v > 200_000) null else v;
}

/// One zone file's reading (null: unreadable).
pub fn readMilli(path: [:0]const u8) ?i32 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var buf: [32]u8 = undefined;
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return null;
    return parseMilli(buf[0..@intCast(n)]);
}

/// The hottest readable zone (null: none).
pub fn hottest(paths: []const [:0]const u8) ?i32 {
    var best: ?i32 = null;
    for (paths) |p| if (readMilli(p)) |v| {
        best = if (best) |b| @max(b, v) else v;
    };
    return best;
}

/// The zone files: TF_DSV41_THERMAL_PACE_ZONES's list, else every readable thermal_zone<i>/temp, i < 64.
pub fn zonePaths(a: std.mem.Allocator, list: []const u8) ![]const [:0]const u8 {
    var out: std.ArrayList([:0]const u8) = .empty;
    if (list.len > 0) {
        var it = std.mem.tokenizeScalar(u8, list, ',');
        while (it.next()) |raw| {
            if (out.items.len == max_zones) return error.BadKnob;
            try out.append(a, try a.dupeSentinel(u8, std.mem.trim(u8, raw, " "), 0));
        }
        return out.items;
    }
    for (0..64) |i| {
        if (out.items.len == max_zones) break;
        const p = try std.fmt.allocPrintSentinel(a, "/sys/class/thermal/thermal_zone{d}/temp", .{i}, 0);
        if (readMilli(p) != null) try out.append(a, p) else a.free(p);
    }
    return out.items;
}

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub fn sleepNs(ns: u64) void {
    var ts: std.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    while (std.c.nanosleep(&ts, &ts) != 0) {}
}

/// A decode busy span: rank 0's windows since the last gap, reset after an idle stretch.
pub const Span = struct {
    mark: u64 = 0,

    /// The busy time a window at `now` closes (0 after an idle stretch or before the first window).
    pub fn busy(s: *const Span, now: u64) u64 {
        if (s.mark == 0 or now < s.mark or now - s.mark > idle_ns) return 0;
        return now - s.mark;
    }
};

/// What a rank paced, for its log line.
pub const Totals = struct {
    gaps: u64 = 0,
    idle_ns: u64 = 0,
    hottest_mc: i32 = unknown,

    pub fn add(t: *Totals, ns: u64, temp_mc: ?i32) bool {
        if (ns == 0) return false;
        t.gaps += 1;
        t.idle_ns += ns;
        if (temp_mc) |v| t.hottest_mc = @max(t.hottest_mc, v);
        return t.gaps == 1 or t.gaps % 256 == 0;
    }
};

/// The pacer: its knobs, zone files, reader thread and the latest reading.
pub const Pace = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    k: Knobs,
    paths: []const [:0]const u8 = &.{},
    temp: std.atomic.Value(i32) = .init(unknown),
    stop: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    span: Span = .{},
    totals: Totals = .{},
    rank: u32,

    /// The pacer when TF_DSV41_THERMAL_PACE=1 (null when off); its reader thread started.
    pub fn init(gpa: std.mem.Allocator, get: Get, rank: u32) !?*Pace {
        const k = try knobs(get) orelse return null;
        const p = try gpa.create(Pace);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .arena = .init(gpa), .k = k, .rank = rank };
        errdefer p.arena.deinit();
        p.paths = try zonePaths(p.arena.allocator(), k.zones);
        const log = std.log.scoped(.dsv41);
        if (p.paths.len == 0) log.warn("thermal pace: no readable zone file on rank {d}: this rank proposes no gap", .{rank});
        p.temp.store(hottest(p.paths) orelse unknown, .release);
        p.thread = try std.Thread.spawn(.{ .stack_size = 64 << 10 }, loop, .{p});
        const c = k.curve;
        log.info("thermal pace on rank {d}: {d} zones (hottest {d:.1} C), gaps from {d:.1} C, ratio {d:.2} at {d:.1} C, at most {d} ms", .{ rank, p.paths.len, celsius(p.reading()), celsius(c.soft_mc), c.ratio, celsius(c.full_mc), c.max_ns / std.time.ns_per_ms });
        return p;
    }

    pub fn deinit(p: *Pace) void {
        p.stop.store(true, .release);
        if (p.thread) |t| t.join();
        p.arena.deinit();
        p.gpa.destroy(p);
    }

    fn loop(p: *Pace) void {
        while (!p.stop.load(.acquire)) {
            p.temp.store(hottest(p.paths) orelse unknown, .release);
            sleepNs(p.k.read_ns);
        }
    }

    /// The latest reading in millidegrees C (null: no zone readable).
    pub fn reading(p: *const Pace) ?i32 {
        const v = p.temp.load(.acquire);
        return if (v == unknown) null else v;
    }

    /// This rank's gap after a prefill segment of `busy_ns` (the ranks then agree on the largest).
    pub fn segmentGap(p: *const Pace, busy_ns: u64) u64 {
        return p.k.curve.gapNs(busy_ns, p.reading());
    }

    /// Rank 0 before a decode window: the gap for the busy span since the last one, slept now.
    pub fn decode(p: *Pace) void {
        const now = nowNs();
        const t = p.reading();
        p.rest(p.k.curve.gapNs(p.span.busy(now), t), t, "decode");
        p.span.mark = nowNs();
    }

    /// Idles `ns` and counts it (`what` names the path in the log line).
    pub fn rest(p: *Pace, ns: u64, temp_mc: ?i32, what: []const u8) void {
        if (ns == 0) return;
        sleepNs(ns);
        if (p.totals.add(ns, temp_mc))
            std.log.scoped(.dsv41).info("thermal pace rank {d}: {s} gap {d} ms at {d:.1} C; {d} gaps, {d:.1} s idle, hottest {d:.1} C", .{ p.rank, what, ns / std.time.ns_per_ms, celsius(temp_mc), p.totals.gaps, @as(f64, @floatFromInt(p.totals.idle_ns)) / 1e9, celsius(p.totals.hottest_mc) });
    }
};

fn celsius(mc: ?i32) f64 {
    const v = mc orelse return -1;
    return if (v == unknown) -1 else @as(f64, @floatFromInt(v)) / 1000;
}

const testing = std.testing;

fn fixture(name: []const u8) ?[]const u8 {
    const table = [_][2][]const u8{
        .{ "TF_DSV41_THERMAL_PACE", "1" },
        .{ "TF_DSV41_THERMAL_PACE_SOFT_C", "85.5" },
        .{ "TF_DSV41_THERMAL_PACE_FULL_C", "89" },
        .{ "TF_DSV41_THERMAL_PACE_RATIO", "0.5" },
        .{ "TF_DSV41_THERMAL_PACE_MAX_MS", "2500" },
        .{ "TF_DSV41_THERMAL_PACE_ZONES", " /tmp/a , /tmp/b" },
    };
    for (table) |kv| if (std.mem.eql(u8, kv[0], name)) return kv[1];
    return null;
}

fn unset(_: []const u8) ?[]const u8 {
    return null;
}

test "thermal pace knobs: off by default, the fixture's values, junk refused" {
    try testing.expect(try knobs(&unset) == null);
    const k = (try knobs(&fixture)).?;
    try testing.expectEqual(@as(i32, 85_500), k.curve.soft_mc);
    try testing.expectEqual(@as(i32, 89_000), k.curve.full_mc);
    try testing.expectEqual(@as(f64, 0.5), k.curve.ratio);
    try testing.expectEqual(2500 * std.time.ns_per_ms, k.curve.max_ns);
    try testing.expectEqualStrings("/tmp/a , /tmp/b", k.zones);
    const Junk = struct {
        var name: []const u8 = "";
        var value: []const u8 = "";
        fn get(n: []const u8) ?[]const u8 {
            if (std.mem.eql(u8, n, name)) return value;
            return if (std.mem.eql(u8, n, "TF_DSV41_THERMAL_PACE")) "1" else null;
        }
    };
    const cases = [_][2][]const u8{
        .{ "TF_DSV41_THERMAL_PACE", "on" },         .{ "TF_DSV41_THERMAL_PACE_SOFT_C", "hot" },
        .{ "TF_DSV41_THERMAL_PACE_SOFT_C", "200" }, .{ "TF_DSV41_THERMAL_PACE_FULL_C", "86.2" },
        .{ "TF_DSV41_THERMAL_PACE_RATIO", "2.5" },  .{ "TF_DSV41_THERMAL_PACE_RATIO", "0" },
        .{ "TF_DSV41_THERMAL_PACE_MAX_MS", "0" },   .{ "TF_DSV41_THERMAL_PACE_READ_MS", "nan" },
    };
    for (cases) |c| {
        Junk.name = c[0];
        Junk.value = c[1];
        try testing.expectEqualStrings(c[1], Junk.get(c[0]).?);
        try testing.expectError(error.BadKnob, knobs(&Junk.get));
    }
}

test "thermal pace curve: no gap at or below soft, the ratio at full, capped past it and by max_ms" {
    const c: Curve = .{};
    try testing.expectEqual(@as(u64, 0), c.gapNs(std.time.ns_per_s, null));
    try testing.expectEqual(@as(u64, 0), c.gapNs(std.time.ns_per_s, 70_000));
    try testing.expectEqual(@as(u64, 0), c.gapNs(std.time.ns_per_s, 86_000));
    try testing.expectEqual(@as(u64, 250 * std.time.ns_per_ms), c.gapNs(std.time.ns_per_s, 87_000));
    try testing.expectEqual(@as(u64, std.time.ns_per_s), c.gapNs(std.time.ns_per_s, 90_000));
    try testing.expectEqual(@as(u64, 1_500 * std.time.ns_per_ms), c.gapNs(std.time.ns_per_s, 92_000));
    // the ratio stops at max_ratio (94 C would be 2.0 on the line, 99 C 3.25)
    try testing.expectEqual(@as(u64, 2 * std.time.ns_per_s), c.gapNs(std.time.ns_per_s, 99_000));
    // one gap is at most max_ns however long the segment was
    try testing.expectEqual(c.max_ns, c.gapNs(10 * std.time.ns_per_s, 91_000));
    try testing.expectEqual(@as(u64, 0), c.gapNs(0, 95_000));
    // monotone in temperature
    var last: u64 = 0;
    var t: i32 = 80_000;
    while (t < 100_000) : (t += 100) {
        const g = c.gapNs(std.time.ns_per_s, t);
        try testing.expect(g >= last);
        last = g;
    }
}

test "thermal pace readings: sysfs millidegrees, junk and absurd values refused, the hottest of the files" {
    try testing.expectEqual(@as(?i32, 90_100), parseMilli("90100\n"));
    try testing.expectEqual(@as(?i32, -5_000), parseMilli("-5000"));
    try testing.expectEqual(@as(?i32, null), parseMilli("90.1\n"));
    try testing.expectEqual(@as(?i32, null), parseMilli(""));
    try testing.expectEqual(@as(?i32, null), parseMilli("-274000"));
    try testing.expectEqual(@as(?i32, null), parseMilli("250000"));
    var dir = testing.tmpDir(.{});
    defer dir.cleanup();
    const io = testing.io;
    try dir.dir.writeFile(io, .{ .sub_path = "a", .data = "71000\n" });
    try dir.dir.writeFile(io, .{ .sub_path = "b", .data = "88250\n" });
    try dir.dir.writeFile(io, .{ .sub_path = "c", .data = "broken\n" });
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const base = try dir.dir.realPathFileAlloc(io, ".", a);
    const list = try std.fmt.allocPrint(a, "{s}/a, {s}/b,{s}/c ,{s}/missing", .{ base, base, base, base });
    const paths = try zonePaths(a, list);
    try testing.expectEqual(@as(usize, 4), paths.len);
    try testing.expectEqual(@as(?i32, 88_250), hottest(paths));
    try testing.expectEqual(@as(?i32, null), hottest(paths[2..]));
}

test "thermal pace decode span: busy since the last gap, reset after an idle stretch" {
    var s: Span = .{};
    try testing.expectEqual(@as(u64, 0), s.busy(1_000));
    s.mark = std.time.ns_per_s;
    try testing.expectEqual(@as(u64, 20 * std.time.ns_per_ms), s.busy(std.time.ns_per_s + 20 * std.time.ns_per_ms));
    try testing.expectEqual(@as(u64, 0), s.busy(std.time.ns_per_s + idle_ns + 1));
    try testing.expectEqual(@as(u64, 0), s.busy(std.time.ns_per_s - 1));
}

test "thermal pace totals: counted gaps only; a log line at the first and every 256th" {
    var t: Totals = .{};
    try testing.expect(!t.add(0, 90_000));
    try testing.expect(t.add(5, 88_000));
    try testing.expect(!t.add(5, 91_000));
    try testing.expectEqual(@as(u64, 2), t.gaps);
    try testing.expectEqual(@as(u64, 10), t.idle_ns);
    try testing.expectEqual(@as(i32, 91_000), t.hottest_mc);
}
