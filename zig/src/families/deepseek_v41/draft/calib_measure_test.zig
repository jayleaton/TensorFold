//! calib_measure.zig against fixtures/calib-measure-golden.json: same runs, same gathered ints, same Costs bits.
const std = @import("std");
const Value = std.json.Value;
const cm = @import("calib_measure.zig");
const costs_mod = @import("costs.zig");

const gpa = std.testing.allocator;
const golden = @embedFile("fixtures/calib-measure-golden.json");

fn num(v: Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

fn floats(a: std.mem.Allocator, v: Value) ![]f64 {
    const out = try a.alloc(f64, v.array.items.len);
    for (out, v.array.items) |*o, x| o.* = num(x);
    return out;
}

const Env = struct {
    var obj: ?std.json.ObjectMap = null;
    fn get(name: [*:0]const u8) ?[]const u8 {
        const o = obj orelse return null;
        return if (o.get(std.mem.span(name))) |v| v.string else null;
    }
};

fn methodOf(env: Value) !cm.Method {
    Env.obj = env.object;
    defer Env.obj = null;
    return cm.method(&Env.get);
}

test "the measurement equals Python's calib.measure: runs, gathered ints and the table" {
    var parsed = try std.json.parseFromSlice(Value, gpa, golden, .{});
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = parsed.value.object.get("cases").?.array.items;
    try std.testing.expect(cases.len >= 5);
    for (cases) |cv| {
        const c = cv.object;
        const m = try methodOf(c.get("env").?);
        const slots: u32 = @intCast(c.get("slots").?.integer);
        var p = try cm.Plan.init(gpa, slots, @intCast(c.get("cap").?.integer), cm.max_rows, m, c.get("draft").?.bool);
        defer p.deinit(gpa);
        try std.testing.expectEqual(@as(u32, @intCast(c.get("n").?.integer)), p.n);
        const wide = c.get("wide").?.array.items;
        try std.testing.expectEqual(wide.len, p.wideBucketsOf().len);
        for (wide, p.wideBucketsOf()) |w, b| try std.testing.expectEqual(@as(u32, @intCast(w.integer)), b);
        // the same runs in the same order: each window's (slot, rows) segments, or the DSpark pass
        const calls = c.get("calls").?.array.items;
        try std.testing.expectEqual(calls.len, p.runs.items.len);
        for (calls, p.runs.items) |call, run| {
            if (call == .string) {
                try std.testing.expectEqualStrings("draft", call.string);
                try std.testing.expect(run.what == .draft and run.nsegs == 0);
                continue;
            }
            const segs = call.array.items;
            try std.testing.expectEqual(segs.len, run.nsegs);
            for (segs, run.segments()) |s, r| {
                try std.testing.expectEqual(@as(u32, @intCast(s.array.items[0].integer)), r.slot);
                try std.testing.expectEqual(@as(u32, @intCast(s.array.items[1].integer)), r.rows);
            }
        }
        // each rank's kept values as the golden's gathered ints, then the table from both
        var ranks: [2][]f64 = undefined;
        for (0..2) |rk| {
            const samples = try floats(a, c.get("samples").?.array.items[rk]);
            ranks[rk] = try a.alloc(f64, p.entries());
            try p.values(samples, ranks[rk]);
            const ints = c.get("ints").?.array.items[rk].array.items;
            try std.testing.expectEqual(ints.len, ranks[rk].len);
            for (ints, ranks[rk]) |want, got| try std.testing.expectEqual(want.integer, costs_mod.roundEven(got * 1000.0));
        }
        var got = try p.build(gpa, &.{ ranks[0], ranks[1] });
        defer got.deinit(gpa);
        const want = c.get("costs").?.object;
        const verify = want.get("verify").?.array.items;
        try std.testing.expectEqual(verify.len, got.verify.len);
        for (verify, got.verify) |x, y| try std.testing.expectEqual(num(x), y);
        try std.testing.expectEqual(num(want.get("draft").?), got.draft);
        try std.testing.expectEqual(num(want.get("slot").?), got.slot);
        // and as the ranks share a cached table (Costs.encode / decode): microseconds, half to even
        const ints = try got.encode(gpa);
        defer gpa.free(ints);
        var back = try costs_mod.Costs.decode(gpa, ints);
        defer back.deinit(gpa);
        for (back.verify, got.verify) |x, y| try std.testing.expectEqual(@as(f64, @floatFromInt(@max(1, costs_mod.roundEven(y * 1000.0)))) / 1000.0, x);
    }
}

test "method, statistic, table_of, extend, split, wide_buckets and fallback_ids equal Python's" {
    var parsed = try std.json.parseFromSlice(Value, gpa, golden, .{});
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const pure = parsed.value.object.get("pure").?.object;
    for (pure.get("statistic").?.array.items) |x| {
        const xs = try floats(a, x.object.get("xs").?);
        try std.testing.expectEqual(num(x.object.get("median").?), try cm.statistic(xs, .median));
        try std.testing.expectEqual(num(x.object.get("min").?), try cm.statistic(xs, .min));
    }
    for (pure.get("table").?.array.items) |x| {
        const times = try floats(a, x.object.get("times").?);
        for ([_]cm.Shape{ .raw, .fit }) |sh| {
            const want = try floats(a, x.object.get(@tagName(sh)).?);
            const got = try cm.tableOf(gpa, times, sh);
            defer gpa.free(got);
            try std.testing.expectEqualSlices(f64, want, got);
        }
    }
    for (pure.get("extend").?.array.items) |x| {
        const table = try floats(a, x.object.get("table").?);
        const pv = x.object.get("points").?.array.items;
        const pts = try a.alloc(costs_mod.Point, pv.len);
        for (pts, pv) |*pt, v| pt.* = .{ .rows = @intCast(v.array.items[0].integer), .ms = num(v.array.items[1]) };
        const got = try costs_mod.extend(gpa, table, pts);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(f64, try floats(a, x.object.get("out").?), got);
    }
    for (pure.get("split").?.array.items) |x| {
        var out: [8]u32 = undefined;
        const n: usize = @intCast(x.object.get("n").?.integer);
        cm.split(@intCast(x.object.get("total").?.integer), out[0..n]);
        for (x.object.get("out").?.array.items, out[0..n]) |w, g| try std.testing.expectEqual(@as(u32, @intCast(w.integer)), g);
    }
    for (pure.get("wide").?.array.items) |x| {
        var buf: [4]u32 = undefined;
        const got = cm.wideBuckets(@intCast(x.object.get("slots").?.integer), @intCast(x.object.get("cap").?.integer), cm.max_rows, &buf);
        const want = x.object.get("out").?.array.items;
        try std.testing.expectEqual(want.len, got.len);
        for (want, got) |w, g| try std.testing.expectEqual(@as(u32, @intCast(w.integer)), g);
    }
    try std.testing.expectEqualStrings(pure.get("text").?.string, cm.text); // the calibration text
    var methods: usize = 0;
    for (pure.get("method").?.array.items) |x| {
        const got = methodOf(x.object.get("env").?);
        const want = x.object.get("out").?;
        if (want == .null) {
            if (got) |_| return error.TestExpectedError else |_| {}
            continue;
        }
        const m = try got;
        const o = want.object;
        try std.testing.expectEqual(@as(u32, @intCast(o.get("reps").?.integer)), m.reps);
        try std.testing.expectEqual(@as(u32, @intCast(o.get("warm").?.integer)), m.warm);
        try std.testing.expectEqual(@as(u32, @intCast(o.get("recheck").?.integer)), m.recheck);
        try std.testing.expectEqual(@as(u32, @intCast(o.get("cycle").?.integer)), m.cycle);
        try std.testing.expectEqual(@as(u32, @intCast(o.get("deep").?.integer)), m.deep);
        try std.testing.expectEqualStrings(o.get("stat").?.string, @tagName(m.stat));
        try std.testing.expectEqualStrings(o.get("shape").?.string, @tagName(m.shape));
        methods += 1;
    }
    try std.testing.expect(methods >= 4);
    for (pure.get("fallback").?.array.items) |x| {
        var buf: [64]u32 = undefined;
        const got = cm.fallbackIds(@intCast(x.object.get("vocab").?.integer), &buf);
        for (x.object.get("out").?.array.items, got) |w, g| try std.testing.expectEqual(@as(u32, @intCast(w.integer)), g);
    }
}

test "the calibration prompt: ids past the vocabulary dropped, at most 256, fallback under 16; slot cuts" {
    var ids: [300]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(if (i % 10 == 3) 200_000 else i);
    var fb: [64]u32 = undefined;
    const got = cm.promptIds(&ids, 129280, &fb);
    try std.testing.expectEqual(@as(usize, 256), got.len);
    try std.testing.expectEqual(@as(u32, 4), got[3]);
    var few = [_]u32{ 1, 2, 3 };
    try std.testing.expectEqual(@as(usize, 64), cm.promptIds(&few, 129280, &fb).len);
    try std.testing.expectEqual(@as(usize, 120), cm.cutOf(120, 0));
    try std.testing.expectEqual(@as(usize, 113), cm.cutOf(120, 1));
    try std.testing.expectEqual(@as(usize, 20), cm.cutOf(20, 2)); // 20 > 14 + 8 fails: the whole prompt
    try std.testing.expect(std.mem.endsWith(u8, cm.text, "and the") and std.mem.startsWith(u8, cm.text, "The committee"));
}

// Rank 0's kept rows 1..16 (ms) from Spark TP2 boots: bursts (S10 row 6; all-on 5-6 + deep; E06 5,10,15), 3 clean.
const spiked_s10 = [_]f64{ 22.772, 27.019, 30.387, 34.693, 39.26, 63.096, 46.47, 51.122, 67.116, 61.769, 71.768, 69.244, 73.629, 85.339, 77.017, 79.448 };
const spiked_all_on = [_]f64{ 22.823, 27.038, 30.416, 35.529, 60.167, 67.206, 48.494, 49.727, 59.171, 96.94, 109.488, 115.328, 129.475, 76.827, 144.058, 144.776 };
const spiked_e06 = [_]f64{ 22.874, 26.99, 30.296, 34.766, 47.874, 44.323, 46.106, 51.241, 57.974, 81.059, 66.091, 68.658, 74.165, 75.87, 124.086, 80.051 };
const clean = [_][16]f64{
    .{ 22.71, 26.866, 30.071, 34.575, 39.486, 44.699, 47.567, 50.591, 57.581, 65.043, 66.205, 69.061, 73.217, 76.339, 78.011, 80.667 },
    .{ 23.289, 26.772, 30.405, 35.209, 38.856, 43.218, 46.93, 49.595, 59.344, 62.237, 65.004, 68.633, 73.667, 77.915, 78.217, 80.34 },
    .{ 22.846, 26.953, 30.288, 34.791, 39.194, 45.385, 47.106, 49.819, 58.188, 63.24, 65.447, 68.609, 73.129, 75.523, 76.988, 78.614 },
};

fn flaggedRows(times: []const f64, buf: *[cm.max_rows]u32) ![]const u32 {
    var flag: [cm.max_rows]bool = undefined;
    const n = try cm.suspects(gpa, times, 0.15, flag[0..times.len]);
    var k: usize = 0;
    for (flag[0..times.len], 0..) |f, i| if (f) {
        buf[k] = @intCast(i + 1);
        k += 1;
    };
    try std.testing.expectEqual(n, k);
    return buf[0..k];
}

test "suspect rows: a burst over one row's runs is found, clean boots are left alone" {
    var buf: [cm.max_rows]u32 = undefined;
    try std.testing.expectEqualSlices(u32, &.{6}, try flaggedRows(&spiked_s10, &buf));
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 10, 11, 12, 13, 15, 16 }, try flaggedRows(&spiked_all_on, &buf));
    try std.testing.expectEqualSlices(u32, &.{ 5, 10, 15 }, try flaggedRows(&spiked_e06, &buf));
    for (&clean) |c| try std.testing.expectEqual(@as(usize, 0), (try flaggedRows(&c, &buf)).len);
    // too few rows for a line: nothing to judge
    try std.testing.expectEqual(@as(usize, 0), (try flaggedRows(&.{ 22.0, 60.0, 30.0 }, &buf)).len);
}

test "retime: suspect rows cycled after one warm-up round, each kept the lower; the table loses the burst" {
    const m: cm.Method = .{};
    var p = try cm.Plan.init(gpa, 4, 64, cm.max_rows, m, false);
    defer p.deinit(gpa);
    // one rank's values: S10's rows, the shared windows 1 ms over row 16
    const v = try gpa.alloc(f64, p.entries());
    defer gpa.free(v);
    @memset(v, spiked_s10[15] + 1.0);
    v[0] = 0.0;
    @memcpy(v[1..17], &spiked_s10);
    var before = try p.build(gpa, &.{v});
    defer before.deinit(gpa);
    try std.testing.expectEqual(spiked_s10[5], before.verify[7]); // the burst carried to row 8 by the envelope
    var both: [cm.max_rows]f64 = undefined;
    try p.combined(&.{v}, &both);
    var flag: [cm.max_rows]bool = undefined;
    try std.testing.expectEqual(@as(usize, 1), try cm.suspects(gpa, &both, m.spike, &flag));
    var runs: std.ArrayList(cm.Run) = .empty;
    defer runs.deinit(gpa);
    try p.retimeRuns(gpa, &flag, &runs);
    try std.testing.expectEqual(@as(usize, 1 + m.reps), runs.items.len);
    try std.testing.expect(runs.items[0].warm and !runs.items[1].warm);
    try std.testing.expectEqual(@as(u32, 6), runs.items[1].segments()[0].rows);
    // the retime sees row 6 at 43 ms (the warm-up at 90 is dropped); a slower retime would not raise a row
    var ms: [6]f64 = .{ 90.0, 43.2, 42.9, 43.1, 43.0, 44.0 };
    try p.mergeRetime(runs.items, &ms, v);
    try std.testing.expectEqual(@as(f64, 43.1), v[6]);
    @memset(&ms, 99.0);
    try p.mergeRetime(runs.items, &ms, v);
    try std.testing.expectEqual(@as(f64, 43.1), v[6]);
    var after = try p.build(gpa, &.{v});
    defer after.deinit(gpa);
    try std.testing.expect(after.verify[5] < 44.0 and after.verify[7] < 52.0);
    try p.combined(&.{v}, &both);
    try std.testing.expectEqual(@as(usize, 0), try cm.suspects(gpa, &both, m.spike, &flag));
}

test "retime knobs: rounds 0..4, spike fraction 0.02..2" {
    Env.obj = null;
    const d = try cm.method(&Env.get);
    try std.testing.expectEqual(@as(u32, 2), d.retime);
    try std.testing.expectEqual(@as(f64, 0.15), d.spike);
    var parsed = try std.json.parseFromSlice(Value, gpa, "{\"TF_DSV41_CALIB_RETIME\": \"0\", \"TF_DSV41_CALIB_SPIKE\": \"0.3\"}", .{});
    defer parsed.deinit();
    const m = try methodOf(parsed.value);
    try std.testing.expectEqual(@as(u32, 0), m.retime);
    try std.testing.expectEqual(@as(f64, 0.3), m.spike);
    for ([_][]const u8{ "{\"TF_DSV41_CALIB_RETIME\": \"5\"}", "{\"TF_DSV41_CALIB_SPIKE\": \"0.01\"}", "{\"TF_DSV41_CALIB_SPIKE\": \"nan\"}" }) |bad| {
        var x = try std.json.parseFromSlice(Value, gpa, bad, .{});
        defer x.deinit();
        try std.testing.expectError(error.CalibKnob, methodOf(x.value));
    }
}
