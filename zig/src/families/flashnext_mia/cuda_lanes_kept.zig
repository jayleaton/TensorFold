//! Lane cost tables kept between loads (lanes.cost_cache): timed once per build, GPU, settings and environment.
const std = @import("std");
const lanes = @import("lanes");
const cl = @import("cuda_lanes.zig");

const max_rows = cl.max_rows;
const shared_points = cl.shared_points;

/// The tables `measure` fills, in fixed-width fields (no padding to read back as a value).
pub const Kept = extern struct {
    cost_count: u32,
    shared_count: u32,
    mtp_ms: f64,
    widths: [max_rows]u32,
    ms: [max_rows]f64,
    shared_widths: [shared_points]u32,
    shared_ms: [shared_points]f64,
};

/// The lane set's measured tables as kept bytes.
pub fn of(l: anytype) Kept {
    var k = std.mem.zeroes(Kept);
    k.cost_count = @intCast(l.cost_count);
    k.shared_count = @intCast(l.shared_count);
    k.mtp_ms = l.mtp_ms;
    for (l.costs[0..l.cost_count], 0..) |c, i| {
        k.widths[i] = c.width;
        k.ms[i] = c.ms;
    }
    for (l.shared[0..l.shared_count], 0..) |c, i| {
        k.shared_widths[i] = c.width;
        k.shared_ms[i] = c.ms;
    }
    return k;
}

/// True when `measure` could have written `k` for a `window`-row set: widths 1.. in order, all ms finite and > 0.
pub fn fits(k: Kept, window: usize) bool {
    if (k.cost_count == 0 or k.cost_count > @min(window, max_rows) or k.shared_count > shared_points) return false;
    if (!std.math.isFinite(k.mtp_ms) or k.mtp_ms < 0) return false;
    for (0..k.cost_count) |i| if (k.widths[i] != i + 1 or !positive(k.ms[i])) return false;
    for (0..k.shared_count) |i| {
        if (!positive(k.shared_ms[i]) or k.shared_widths[i] == 0) return false;
        if (i > 0 and k.shared_widths[i] <= k.shared_widths[i - 1]) return false;
    }
    return true;
}

fn positive(x: f64) bool {
    return std.math.isFinite(x) and x > 0;
}

/// The lane set takes a table `fits` accepted.
pub fn apply(k: Kept, l: anytype) void {
    l.cost_count = k.cost_count;
    for (0..k.cost_count) |i| l.costs[i] = .{ .width = k.widths[i], .ms = k.ms[i] };
    l.shared_count = k.shared_count;
    for (0..k.shared_count) |i| l.shared[i] = .{ .width = k.shared_widths[i], .ms = k.shared_ms[i] };
    l.mtp_ms = k.mtp_ms;
}

/// The environ's speed entries (TF_FLASHNEXT_, TF_TP_, TF_CUDA_, TF_ROCE_, NCCL_, CUDA_), sorted, one a line.
pub fn envLines(a: std.mem.Allocator, environ: []const u8) ![]u8 {
    const prefixes = [_][]const u8{ "TF_FLASHNEXT_", "TF_TP_", "TF_CUDA_", "TF_ROCE_", "NCCL_", "CUDA_" };
    var kept: std.ArrayList([]const u8) = .empty;
    defer kept.deinit(a);
    var it = std.mem.splitScalar(u8, environ, 0);
    while (it.next()) |entry| {
        for (prefixes) |p| if (std.mem.startsWith(u8, entry, p) and std.mem.indexOfScalar(u8, entry, '=') != null) {
            try kept.append(a, entry);
            break;
        };
    }
    std.mem.sort([]const u8, kept.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.less);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    for (kept.items) |entry| {
        try out.appendSlice(a, entry);
        try out.append(a, '\n');
    }
    return out.toOwnedSlice(a);
}

/// The key: build, GPU, driver, rank, agreed settings and speed env; null (timed, not kept) when unreadable.
pub fn key(a: std.mem.Allocator, io: std.Io, ctx: anytype, settings: []const u8, rank: u32, world: u32) ?[32]u8 {
    var name: [256]u8 = undefined;
    const gpu = ctx.name(&name) catch return null;
    var dev: [96]u8 = undefined;
    const device = std.fmt.bufPrint(&dev, "sm{d} sms{d} driver{d} rank{d}/{d}", .{
        ctx.capability() catch return null, ctx.attribute(.multiprocessor_count) catch return null, ctx.d.version() catch return null, rank, world,
    }) catch return null;
    const environ = processEnviron(a, io) orelse return null;
    defer a.free(environ);
    const env = envLines(a, environ) catch return null;
    defer a.free(env);
    const parts = [_][]const u8{ "flashnext-mia-cuda/lane-costs-v1", gpu, device, settings, env };
    return lanes.cost_cache.key(a, io, &parts) catch null;
}

/// /proc/self/environ, read as a stream (/proc reports size 0); empty where absent, null when out of memory.
fn processEnviron(a: std.mem.Allocator, io: std.Io) ?[]u8 {
    const empty = a.alloc(u8, 0) catch return null;
    const f = std.Io.Dir.cwd().openFile(io, "/proc/self/environ", .{}) catch return empty;
    defer f.close(io);
    var buf: [4096]u8 = undefined;
    var r = f.readerStreaming(io, &buf);
    const got = r.interface.allocRemaining(a, .limited(1 << 20)) catch return empty;
    a.free(empty);
    return got;
}

fn table(cost_count: usize, shared_count: usize) Kept {
    var k = std.mem.zeroes(Kept);
    k.cost_count = @intCast(cost_count);
    k.shared_count = @intCast(shared_count);
    k.mtp_ms = 1.19;
    for (0..cost_count) |i| {
        k.widths[i] = @intCast(i + 1);
        k.ms[i] = 33.0 + @as(f64, @floatFromInt(i));
    }
    for (0..shared_count) |i| {
        k.shared_widths[i] = @as(u32, 32) << @intCast(i);
        k.shared_ms[i] = 75.0 + 20.0 * @as(f64, @floatFromInt(i));
    }
    return k;
}

test "kept lane costs: a table round-trips through a lane set and the cache envelope" {
    const L = struct {
        costs: [max_rows]lanes.config.Cost = undefined,
        cost_count: usize = 0,
        shared: [shared_points]lanes.config.Cost = undefined,
        shared_count: usize = 0,
        mtp_ms: f64 = 0,
    };
    const k = table(16, 4);
    try std.testing.expect(fits(k, 16));
    var l: L = .{};
    apply(k, &l);
    try std.testing.expectEqual(@as(usize, 16), l.cost_count);
    try std.testing.expectEqual(@as(u32, 128), l.shared[2].width);
    try std.testing.expectEqualDeep(k, of(&l));

    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const at: [32]u8 = @splat(7);
    try std.testing.expectEqual(@as(?Kept, null), lanes.cost_cache.loadAt(Kept, a, io, at, root));
    lanes.cost_cache.saveAt(Kept, a, io, at, k, root);
    try std.testing.expectEqualDeep(k, lanes.cost_cache.loadAt(Kept, a, io, at, root).?);
}

test "kept lane costs: a table no measurement could have written is refused" {
    try std.testing.expect(!fits(table(16, 4), 8)); // more verify widths than the window
    try std.testing.expect(!fits(table(0, 0), 16));
    var gap = table(16, 4);
    gap.widths[3] = 5;
    try std.testing.expect(!fits(gap, 16));
    var nan = table(16, 4);
    nan.ms[0] = std.math.nan(f64);
    try std.testing.expect(!fits(nan, 16));
    var flat = table(16, 4);
    flat.shared_widths[2] = flat.shared_widths[1];
    try std.testing.expect(!fits(flat, 16));
    var slow = table(16, 4);
    slow.shared_ms[3] = 0;
    try std.testing.expect(!fits(slow, 16));
    try std.testing.expect(fits(table(7, 0), 16)); // one stream: no shared rounds
}

test "kept lane costs: the key's environment keeps speed settings only, sorted" {
    const a = std.testing.allocator;
    const raw = "PATH=/bin\x00TF_FLASHNEXT_FILL=1\x00NCCL_IB_HCA=rocep1s0f1\x00TF_RUNNER=1\x00HOME=/root\x00CUDA_CACHE_PATH=/cache/nv\x00NCCL_NOVALUE";
    const got = try envLines(a, raw);
    defer a.free(got);
    try std.testing.expectEqualStrings("CUDA_CACHE_PATH=/cache/nv\nNCCL_IB_HCA=rocep1s0f1\nTF_FLASHNEXT_FILL=1\n", got);
    const none = try envLines(a, "");
    defer a.free(none);
    try std.testing.expectEqualStrings("", none);
}

test "kept lane costs: /proc's environment reads whole (its size reads as 0)" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const environ = processEnviron(a, std.testing.io).?;
    defer a.free(environ);
    try std.testing.expect(std.mem.indexOf(u8, environ, "PATH=") != null);
}
