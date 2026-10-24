//! CUDA graphs of windows in rows form over cuda/graph_cache.zig: one graph serves any slots; replay bits equal eager.
const std = @import("std");
const cuda = @import("cuda");
const iface = @import("iface.zig");
const rows_plan = @import("rows_plan.zig");
const speed = @import("speed.zig");

pub const gc = cuda.graph_cache;
pub const Settings = speed.Graphs;
pub const Key = speed.Key;
pub const Cache = gc.Cache(Key);

comptime {
    if (speed.max_streams != iface.max_shared) @compileError("a keyed layout names every stream of a shared window");
    if ((Settings{}).rows_max != iface.shared_rows) @compileError("graphs cover the shared window's rows");
}

var rule: Settings = .{};

fn room() bool {
    return speed.fits(rule, speed.memAvailableMiB());
}

fn meter() ?f64 {
    const m = speed.memAvailableMiB() orelse return null;
    return @as(f64, @floatFromInt(m)) / 1024;
}

/// Every rank's verdict on a miss: one byte a rank all-gathered over the group, outside any capture.
pub const Agreement = struct {
    group: *const cuda.tp_group.Group,
    stream: cuda.Stream,
    dev: cuda.DeviceBuffer,
    world: u32,

    pub const max_world = 8;

    pub fn init(d: *const cuda.Driver, group: *const cuda.tp_group.Group, stream: cuda.Stream, world: u32) !Agreement {
        if (world > max_world) return error.TooManyRanks;
        return .{ .group = group, .stream = stream, .dev = try cuda.DeviceBuffer.alloc(d, 16 * (world + 1)), .world = world };
    }

    pub fn deinit(a: *Agreement) void {
        a.dev.free();
    }

    pub fn agree(a: *Agreement) gc.Agree {
        return .{ .ctx = a, .all = all };
    }

    fn all(ctx: *anyopaque, ok: bool) anyerror!bool {
        const a: *Agreement = @ptrCast(@alignCast(ctx));
        var mine: [16]u8 = @splat(0);
        mine[0] = @intFromBool(ok);
        try a.dev.uploadAsync(0, &mine, a.stream.handle);
        try a.group.allGather(a.stream, a.dev.ptr, a.dev.ptr + 16, 16);
        var every: [16 * max_world]u8 = undefined;
        const got = every[0 .. 16 * a.world];
        try a.dev.downloadAsync(16, got, a.stream.handle);
        try a.stream.synchronize();
        var r: usize = 0;
        while (r < a.world) : (r += 1) if (got[r * 16] == 0) return false;
        return true;
    }
};

/// Keys remembered as seen once at most.
pub const seen_max = 4096;

/// A forward's graphs: the engine, the cache, the keys seen once (a key is captured the second time it comes).
pub const Graphs = struct {
    gpa: std.mem.Allocator,
    settings: Settings,
    engine: gc.CudaEngine,
    cache: Cache,
    agreement: ?Agreement = null,
    seen: std.AutoHashMapUnmanaged(Key, void) = .empty,

    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, s: Settings) !*Graphs {
        const g = try gpa.create(Graphs);
        g.* = .{ .gpa = gpa, .settings = s, .engine = .{ .d = d }, .cache = undefined };
        g.cache = Cache.init(gpa, g.engine.engine(), .{ .on = s.on, .max = s.max, .hold = s.hold });
        g.cache.name = "mm3 row graphs";
        g.cache.room = room;
        g.cache.meter = meter;
        rule = s;
        return g;
    }

    /// Ranks of a group vote on every miss (a lone rank needs no vote).
    pub fn join(g: *Graphs, d: *const cuda.Driver, group: *const cuda.tp_group.Group, stream: cuda.Stream, world: u32) !void {
        if (world < 2) return;
        g.agreement = try Agreement.init(d, group, stream, world);
        g.cache.agree = g.agreement.?.agree();
    }

    pub fn destroy(g: *Graphs) void {
        g.cache.deinit();
        if (g.agreement) |*a| a.deinit();
        g.seen.deinit(g.gpa);
        g.gpa.destroy(g);
    }

    /// True the first time `key` comes: that step runs eagerly, so lazily grown scratch settles before a capture.
    pub fn first(g: *Graphs, key: Key) !bool {
        if (g.cache.has(key)) return false;
        if (g.seen.count() >= seen_max) g.seen.clearRetainingCapacity(); // tap layouts are many: forget, never grow
        const r = try g.seen.getOrPut(g.gpa, key);
        return !r.found_existing;
    }

    /// The key of a window: its rows, context bucket and head, plus its streams' layout when the forward taps.
    pub fn keyOf(g: *const Graphs, rows: u32, max_pos: u32, head: u8, mode: u8, segs: []const iface.Seg, taps: bool) !Key {
        return .{
            .rows = rows,
            .ctx = speed.bucketOf(max_pos, g.settings.bucket),
            .head = head,
            .mode = mode,
            .layout = if (taps) try speed.layoutOf(segs) else @splat(0),
        };
    }

    /// The largest position a key's graph scores: its bucket's end inside the table's positions.
    pub fn scorePos(g: *const Graphs, k: Key, positions: usize) u32 {
        return @intCast(speed.bucketEnd(k.ctx, g.settings.bucket, positions));
    }
};

test "a decode window always fits a bounded plan's scratch at the bucket's end" {
    const p = rows_plan.Plan.of(4, iface.maxPages((1 << 20) + iface.window_rows), 1);
    const end = speed.bucketEnd(speed.bucketOf(p.positions - 2, 2048), 2048, p.positions);
    try std.testing.expect(p.fits(iface.shared_rows, end));
}

test {
    _ = speed;
}
