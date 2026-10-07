//! The sampling pipeline (sampling_pipe.zig) on in-process ranks (``tp.host``) against the Python engine's choices:
//! the serving goldens (zig/tests/server/dsv41/gen_sampling.py: rows of logits, their sampling, position and token as
//! ``batch._choose`` picks them from the ranks' candidates), each row split over its world's ranks and picked by the
//! pipeline's own steps (each rank's top k, the gathers, the nucleus statistics, the whole-row fallback).

const std = @import("std");
const testing = std.testing;
const tp = @import("tp");
const json = @import("json");
const serve = @import("dsv41_serve");
const pipe = @import("sampling_pipe.zig");

const Case = struct {
    logits: []const f32,
    world: u32,
    rows: usize,
    sampling: pipe.Sampling,
    positions: []const u64,
    nucleus: u32 = serve.sampling.nucleus_count,
    got: []u32,
    whole: []usize,
};

/// Rank `me` of a window: its vocabulary slice of each row (contiguous [rows, V]), the pipeline's tokens on rank 0.
fn rankBody(comm: tp.collective.Collective, me: u32, c: *Case) anyerror!void {
    const a = testing.allocator;
    const vocab = c.logits.len / c.rows;
    const V = vocab / c.world;
    const mine = try a.alloc(f32, c.rows * V);
    defer a.free(mine);
    for (0..c.rows) |r| @memcpy(mine[r * V ..][0..V], c.logits[r * vocab + me * V ..][0..V]);
    var dev: pipe.HostDevice = .{ .gpa = a };
    defer dev.deinit();
    var p = try pipe.Pipe.init(a, comm, @intCast(V), c.nucleus);
    defer p.deinit();
    const out = try a.alloc(u32, c.rows);
    defer a.free(out);
    try p.choose(dev.device(), @intFromPtr(mine.ptr), c.rows, c.sampling, if (me == 0) c.positions else null, out);
    if (me == 0) {
        @memcpy(c.got, out);
        c.whole[0] = p.whole_rows;
    }
}

fn run(c: *Case) !void {
    try tp.host.run(c.world, c, rankBody);
}

fn parse(x: std.mem.Allocator, line: []const u8) !struct { logits: []f32, world: u32, s: ?pipe.Sampling, position: u64, full: bool, token: u32 } {
    const c = (try json.parseText(x, line)).ok;
    const b64 = c.get("logits").?.string;
    const raw = try x.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(b64));
    try std.base64.standard.Decoder.decode(raw, b64);
    const logits = try x.alloc(f32, raw.len / 4);
    for (logits, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));
    const sv = c.get("sampling").?;
    const s: ?pipe.Sampling = if (sv == .array) .{
        .seed = @truncate(@as(u128, @bitCast(@as(i128, try std.fmt.parseInt(i128, sv.array[0].int, 10))))),
        .temperature = sv.array[1].float64().?,
        .top_k = @intCast(sv.array[2].int64().?),
        .top_p = sv.array[3].float64().?,
        .min_p = sv.array[4].float64().?,
    } else null;
    return .{ .logits = logits, .world = @intCast(c.get("world").?.int64().?), .s = s, .position = @intCast(c.get("position").?.int64().?), .full = c.get("full").?.bool, .token = @intCast(c.get("token").?.int64().?) };
}

test "sampled rows on two ranks pick the Python engine's tokens (the committed serving goldens)" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var it = std.mem.splitScalar(u8, serve.fixtures.sampling, '\n');
    var rows: usize = 0;
    var bad: usize = 0;
    var whole: usize = 0;
    var fallbacks: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        _ = arena.reset(.retain_capacity);
        const g = try parse(arena.allocator(), line);
        const s = g.s orelse continue; // greedy rows: forward.greedy's path
        if (!pipe.sampled(s) or g.logits.len % g.world != 0) continue;
        var got: [1]u32 = .{0};
        var w: [1]usize = .{0};
        var c: Case = .{ .logits = g.logits, .world = g.world, .rows = 1, .sampling = s, .positions = &.{g.position}, .got = &got, .whole = &w };
        try run(&c);
        rows += 1;
        whole += w[0];
        fallbacks += @intFromBool(g.full);
        if (got[0] != g.token) {
            bad += 1;
            if (bad <= 5) std.debug.print("row {d}: want {d} got {d} (vocab {d}, world {d}, sampling {any})\n", .{ rows, g.token, got[0], g.logits.len, g.world, s });
        }
    }
    std.debug.print("sampling pipeline: {d}/{d} sampled rows as Python picks them ({d} taken whole, {d} nucleus fallbacks in the goldens)\n", .{ rows - bad, rows, whole, fallbacks });
    try testing.expect(rows > 100 and fallbacks > 0);
    try testing.expectEqual(@as(usize, 0), bad);
}

test "a window's rows pick as each row does alone (one call, several positions, every path)" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(41);
    const rnd = prng.random();
    const vocab = 4096;
    const n = 6;
    const logits = try a.alloc(f32, n * vocab);
    defer a.free(logits);
    for (logits, 0..) |*v, i| {
        const x: f32 = @floatCast(rnd.floatNorm(f64) * 3.0 + @as(f64, if (i % vocab == 77) 6 else 0));
        v.* = @bitCast(@as(u32, @bitCast(x)) & 0xFFFF_0000); // bf16-valued, as kit_logits leaves them
    }
    const positions = [_]u64{ 100, 101, 102, 103, 104, 105 };
    const cases = [_]pipe.Sampling{
        .{ .seed = 7, .temperature = 0.7, .top_k = 0, .top_p = 0.9 }, // nucleus
        .{ .seed = 9, .temperature = 1.0, .top_k = 40, .top_p = 0.95, .min_p = 0.05 }, // top-k
        .{ .seed = 3, .temperature = 1.3, .top_k = 0, .top_p = 1.0 }, // whole rows (k > max_k)
        .{ .seed = 1 << 63 | 5, .temperature = 0.6, .top_k = 0, .top_p = 0.8 },
    };
    for (cases) |s| for ([_]u32{ 1, 2 }) |world| {
        var all: [n]u32 = undefined;
        var w: [1]usize = .{0};
        var c: Case = .{ .logits = logits, .world = world, .rows = n, .sampling = s, .positions = &positions, .got = &all, .whole = &w };
        try run(&c);
        for (0..n) |r| {
            var one: [1]u32 = undefined;
            var w1: [1]usize = .{0};
            var c1: Case = .{ .logits = logits[r * vocab ..][0..vocab], .world = world, .rows = 1, .sampling = s, .positions = positions[r..][0..1], .got = &one, .whole = &w1 };
            try run(&c1);
            try testing.expectEqual(one[0], all[r]);
        }
        // the keyed choice is the host rule's over the whole row (lanes.sampling.choose = exact_sampling.choose)
        for (0..n) |r| {
            const row = logits[r * vocab ..][0..vocab];
            const vals = try a.alloc(f64, vocab);
            defer a.free(vals);
            const ids = try a.alloc(u64, vocab);
            defer a.free(ids);
            for (vals, ids, row, 0..) |*v, *id, x, i| {
                v.* = x;
                id.* = i;
            }
            const want = try @import("lanes").sampling.choose(a, vals, ids, positions[r], s);
            try testing.expectEqual(@as(u32, @intCast(want)), all[r]);
        }
    };
}
