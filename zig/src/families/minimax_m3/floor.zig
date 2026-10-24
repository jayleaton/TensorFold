//! The ranks' memory floor: chunk gates gather every rank's MemAvailable and stop together; new prompts wait under it.
const std = @import("std");

pub const GiB: u64 = 1 << 30;
pub const hard_var = "TF_MM3_FLOOR_HARD_GIB";
pub const target_var = "TF_MM3_FLOOR_GIB";
pub const transient_var = "TF_MM3_FLOOR_TRANSIENT_GIB";

pub const Get = *const fn ([:0]const u8) ?[]const u8;

/// MemAvailable thresholds in bytes, the same on every rank (the ranks' fingerprint carries `gateMib`).
pub const Floor = struct {
    hard: u64 = 4 * GiB, // TF_MM3_FLOOR_HARD_GIB: no chunk starts below it, on any rank
    target: u64 = 5 * GiB, // TF_MM3_FLOOR_GIB: a new prompt waits below it while other streams run
    transient: u64 = GiB / 2, // TF_MM3_FLOOR_TRANSIENT_GIB: what one chunk may take beyond the preallocated scratch

    /// The three variables (GiB, 0 < hard <= target, transient >= 0); unset keeps 4, 5 and 0.5.
    pub fn read(get: Get, problem: *[]const u8) error{BadSetting}!Floor {
        var f: Floor = .{};
        f.hard = try gib(get, hard_var, f.hard, problem);
        f.target = try gib(get, target_var, @max(f.target, f.hard), problem);
        f.transient = try gib(get, transient_var, f.transient, problem);
        if (f.hard == 0 or f.hard > f.target) {
            problem.* = "TF_MM3_FLOOR_HARD_GIB / TF_MM3_FLOOR_GIB: expected 0 < hard <= target (4 and 5 by default)";
            return error.BadSetting;
        }
        return f;
    }

    /// The gate's one threshold in MiB: hard floor plus a chunk's transient; ranks that differ refuse at boot.
    pub fn gateMib(f: Floor) u32 {
        return @intCast(@min((f.hard + f.transient) >> 20, std.math.maxInt(u32)));
    }

    /// A chunk stops when the lowest rank's MemAvailable less a chunk's transient is under the hard floor.
    pub fn stops(f: Floor, low: u64) bool {
        return low -| f.transient < f.hard;
    }

    /// A new prompt starts beside other streams only when the lowest known MemAvailable keeps the target.
    pub fn admits(f: Floor, low: u64) bool {
        return low -| f.transient >= f.target;
    }

    /// After the pool is made: under the hard floor refuses the start, under the target warns.
    pub fn boot(f: Floor, low: u64) enum { ok, under_target, under_hard } {
        if (f.stops(low)) return .under_hard;
        return if (low < f.target) .under_target else .ok;
    }
};

fn gib(get: Get, name: [:0]const u8, default: u64, problem: *[]const u8) error{BadSetting}!u64 {
    const v = std.mem.trim(u8, get(name) orelse return default, " ");
    if (v.len == 0) return default;
    const g = std.fmt.parseFloat(f64, v) catch return bad(name, problem);
    if (!(g >= 0 and g < 1 << 20)) return bad(name, problem);
    return @intFromFloat(@round(g * @as(f64, @floatFromInt(GiB))));
}

fn bad(name: [:0]const u8, problem: *[]const u8) error{BadSetting} {
    problem.* = if (std.mem.eql(u8, name, transient_var)) "TF_MM3_FLOOR_TRANSIENT_GIB: a number of GiB, 0 or more" else "TF_MM3_FLOOR_HARD_GIB / TF_MM3_FLOOR_GIB: a positive number of GiB";
    return error.BadSetting;
}

/// MemAvailable from /proc/meminfo's text, in bytes; null when the line is missing.
pub fn available(text: []const u8) ?u64 {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, "MemAvailable:")) continue;
        var f = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \tkB");
        const kib = std.fmt.parseInt(u64, f.next() orelse return null, 10) catch return null;
        return kib * 1024;
    }
    return null;
}

/// Where a rank reads its MemAvailable: /proc/meminfo in production (`proc`), a value in tests.
pub const Mem = struct {
    ctx: ?*anyopaque = null,
    read_fn: *const fn (ctx: ?*anyopaque, io: std.Io) ?u64,

    pub fn read(m: Mem, io: std.Io) ?u64 {
        return m.read_fn(m.ctx, io);
    }
};

pub const proc: Mem = .{ .read_fn = procRead };

fn procRead(_: ?*anyopaque, io: std.Io) ?u64 {
    var buf: [8192]u8 = undefined;
    var file = std.Io.Dir.cwd().openFile(io, "/proc/meminfo", .{}) catch return null;
    defer file.close(io);
    var r = file.readerStreaming(io, &buf); // procfs reports size 0: read it as a stream
    var text: [16384]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&text);
    return available(r.interface.allocRemaining(fba.allocator(), .limited(text.len)) catch return null);
}

/// What a rank brings to a gate: its MemAvailable (unreadable: the most, no say). 16 bytes: a RoCE-sized gather.
pub const Sample = extern struct { avail: u64, pad: u64 = 0 };

pub const unknown: u64 = std.math.maxInt(u64);

/// The gathered samples' lowest MemAvailable and its rank, and the lowest of the ranks other than `me`.
pub const Low = struct { low: u64, rank: usize, others: u64 };

pub fn lowest(all: []const Sample, me: usize) Low {
    var out: Low = .{ .low = unknown, .rank = 0, .others = unknown };
    for (all, 0..) |s, r| {
        if (s.avail < out.low) out = .{ .low = s.avail, .rank = r, .others = out.others };
        if (r != me) out.others = @min(out.others, s.avail);
    }
    return out;
}

/// Short GiB for log lines.
pub fn gibs(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(GiB));
}

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: [:0]const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "the floor's variables: 4 / 5 / 0.5 GiB by default, hard above target or a bad number refused" {
    var problem: []const u8 = "";
    Fake.pairs = &.{};
    const d = try Floor.read(&Fake.get, &problem);
    try std.testing.expectEqual(Floor{}, d);
    try std.testing.expectEqual(@as(u32, 4608), d.gateMib());
    Fake.pairs = &.{ .{ hard_var, "6" }, .{ transient_var, "0" } };
    const six = try Floor.read(&Fake.get, &problem);
    try std.testing.expectEqual(6 * GiB, six.target); // the target follows a higher hard floor
    try std.testing.expectEqual(@as(u64, 0), six.transient);
    Fake.pairs = &.{ .{ hard_var, "6" }, .{ target_var, "5" } };
    try std.testing.expectError(error.BadSetting, Floor.read(&Fake.get, &problem));
    Fake.pairs = &.{.{ hard_var, "0" }};
    try std.testing.expectError(error.BadSetting, Floor.read(&Fake.get, &problem));
    Fake.pairs = &.{.{ transient_var, "-1" }};
    try std.testing.expectError(error.BadSetting, Floor.read(&Fake.get, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "TRANSIENT") != null);
}

test "the gate stops under hard + transient, admission keeps target + transient, boot refuses under the gate" {
    const f: Floor = .{};
    try std.testing.expect(!f.stops(4 * GiB + GiB / 2));
    try std.testing.expect(f.stops(4 * GiB + GiB / 2 - 1));
    try std.testing.expect(f.stops(0));
    try std.testing.expect(f.admits(5 * GiB + GiB / 2));
    try std.testing.expect(!f.admits(5 * GiB + GiB / 2 - 1));
    try std.testing.expect(!f.stops(unknown) and f.admits(unknown)); // no reading: no say
    try std.testing.expectEqual(.under_hard, f.boot(4 * GiB));
    try std.testing.expectEqual(.under_target, f.boot(4 * GiB + GiB / 2));
    try std.testing.expectEqual(.ok, f.boot(8 * GiB));
}

test "the lowest of the ranks names its rank; `others` leaves this rank out" {
    const all = [_]Sample{ .{ .avail = 9 * GiB }, .{ .avail = 3 * GiB } };
    const a = lowest(&all, 0);
    try std.testing.expectEqual(Low{ .low = 3 * GiB, .rank = 1, .others = 3 * GiB }, a);
    const b = lowest(&all, 1);
    try std.testing.expectEqual(@as(u64, 9 * GiB), b.others);
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Sample));
}

test "MemAvailable from meminfo text, and none without its line" {
    try std.testing.expectEqual(@as(?u64, 4893208 * 1024), available("MemTotal:       125481616 kB\nMemAvailable:    4893208 kB\n"));
    try std.testing.expectEqual(@as(?u64, null), available("MemTotal: 1 kB\n"));
    const live = proc.read(std.testing.io); // this host's, where it has one
    if (@import("builtin").os.tag == .linux) try std.testing.expect(live != null and live.? > 0);
}
