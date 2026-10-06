//! The lane engine on the CPU twin with DeepSeek's backend and depth: drafted == serial (greedy and keyed sampling),
//! shared rounds == alone, trees, zero depth and joint depth, every round checked against the twin's serial decode.
const std = @import("std");
const lanes = @import("lanes");
const twin = @import("twin.zig");
const ln = @import("lanes.zig");
const costs_mod = @import("costs.zig");
const depth = @import("depth.zig");

const gpa = std.testing.allocator;

const Case = struct { prompt: []const u32, max_new: u32 = 60, sampling: ?lanes.Sampling = null, drafts: bool = true, copies: bool = false };

const Setup = struct { siblings: u32 = 0, joint: u2 = 1, skip: bool = true, costs: Prices = .prod, slots: u32 = 4, dims: twin.Dims = .{}, depth: depth.Settings = .{} };

const Prices = enum { prod, flat, cheap };

/// The G15 boot's table (GB10, TP=2): rows 1-16 measured, the wide buckets 24/32/48/64, the pass and slot ms.
pub fn prodCosts(alloc: std.mem.Allocator) !costs_mod.Costs {
    const rows = [_]f64{ 21.954, 26.573, 31.977, 36.158, 40.852, 44.776, 47.237, 51.089, 57.951, 59.663, 60.793, 62.458, 69.962, 70.85, 70.971, 72.362 };
    const slot: f64 = 0.83; // the log's bucket values had the slot overhead taken out: put it back as calib.py timed them
    const wide = [_]costs_mod.Point{ .{ .rows = 24, .ms = @as(f64, 100.2) + slot }, .{ .rows = 32, .ms = @as(f64, 115.2) + slot }, .{ .rows = 48, .ms = @as(f64, 147.1) + 2 * slot }, .{ .rows = 64, .ms = @as(f64, 196.3) + 3 * slot } };
    return costs_mod.measured(alloc, &rows, &wide, 3.61, 0.83);
}

const Run = struct {
    out: [][]u32,
    drafted: u64,
    accepted: u64,
    zero: u64,
    trees: u64,
    sib_wins: u64,
    joint: u64,
    kept: [16]u64, // drafts kept by position (depth.Stats)
    widest: usize, // the most rows one forward held

    fn deinit(r: Run) void {
        for (r.out) |o| gpa.free(o);
        gpa.free(r.out);
    }
};

fn run(cases: []const Case, setup: Setup) !Run {
    const t = try twin.Twin.init(gpa, setup.dims, setup.slots + 1);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, setup.slots + 1);
    defer d.deinit();
    var c = switch (setup.costs) {
        .prod => try prodCosts(gpa),
        .flat => try costs_mod.defaults(gpa, 64),
        .cheap => cheap: { // rows nearly free: every draft pays, so windows fill
            const v = try gpa.alloc(f64, 64);
            for (v, 0..) |*x, r| x.* = 20.0 + 0.01 * @as(f64, @floatFromInt(r));
            break :cheap costs_mod.Costs{ .verify = v, .draft = 1.0 };
        },
    };
    defer c.deinit(gpa);
    const x = try ln.Lanes.init(gpa, t.target(), d.pass(), c, .{ .shape = d.shape, .siblings = setup.siblings, .slots = setup.slots, .depth = d: {
        var set = setup.depth;
        set.joint = setup.joint;
        set.skip = setup.skip;
        break :d set;
    } });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cfg = try lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    const props = try gpa.alloc(lanes.SuffixLookup, cases.len);
    defer gpa.free(props);
    for (cases, streams, props) |cs, *s, *p| {
        p.* = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = cs.prompt, .max_new = cs.max_new, .sampling = cs.sampling, .drafts = cs.drafts, .proposer = if (cs.copies) p.proposer() else null });
    }
    defer for (streams, props) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try e.addStream(s);
    while (e.activeCount() > 0) try e.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return .{ .out = out, .drafted = e.drafted, .accepted = e.accepted, .zero = x.policy.depth.stats.zero, .trees = x.policy.trees, .sib_wins = x.policy.sib_wins, .joint = x.policy.depth.stats.joint[0], .kept = x.policy.depth.stats.kept, .widest = t.widest };
}

/// The twin's own serial decode of a case (no lanes at all).
fn serial(cs: Case, dims: twin.Dims) ![]u32 {
    const t = try twin.Twin.init(gpa, dims, 1);
    defer t.deinit();
    return t.serial(0, cs.prompt, cs.max_new, cs.sampling, null);
}

fn expectSerial(cases: []const Case, r: Run, dims: twin.Dims) !void {
    for (cases, r.out) |cs, got| {
        const want = try serial(cs, dims);
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u32, want, got);
    }
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5 };
const p3 = [_]u32{ 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30 };
const hot: lanes.Sampling = .{ .seed = 11, .temperature = 0.8, .top_k = 0, .top_p = 0.9 };
const warm: lanes.Sampling = .{ .seed = 5, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.02 };

test "drafted rounds commit the serial decode, greedy and keyed sampling" {
    for ([_]?lanes.Sampling{ null, hot, warm }) |s| {
        const cases = [_]Case{.{ .prompt = &p1, .sampling = s, .max_new = 80 }};
        const r = try run(&cases, .{});
        defer r.deinit();
        try expectSerial(&cases, r, .{});
        try std.testing.expect(r.accepted > 0 and r.kept[1] > 0); // windows kept two drafts and more
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .max_new = 80, .drafts = false }}, .{});
        defer plain.deinit();
        try std.testing.expectEqualSlices(u32, plain.out[0], r.out[0]);
        try std.testing.expectEqual(@as(u64, 0), plain.drafted);
    }
}

test "a confident drafter keeps whole blocks and stays serial" {
    const dims: twin.Dims = .{ .bigram = 8.0, .head = 0.08 };
    const cases = [_]Case{ .{ .prompt = &p3, .max_new = 120 }, .{ .prompt = &p1, .sampling = warm, .max_new = 120 } };
    const r = try run(&cases, .{ .dims = dims });
    defer r.deinit();
    try expectSerial(&cases, r, dims);
    try std.testing.expect(r.kept[3] > 0 and r.accepted * 10 >= r.drafted * 8);
}

test "shared rounds commit what each stream commits alone, every joint mode" {
    const cases = [_]Case{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = hot, .max_new = 45 }, .{ .prompt = &p3, .copies = true }, .{ .prompt = &p2, .drafts = false, .max_new = 30 } };
    for ([_]u2{ 0, 1, 2 }) |mode| {
        const r = try run(&cases, .{ .joint = mode });
        defer r.deinit();
        try expectSerial(&cases, r, .{});
        if (mode > 0) try std.testing.expect(r.joint > 0);
    }
}

test "first-position trees keep the serial decode" {
    var trees: u64 = 0;
    var wins: u64 = 0;
    for ([_]?lanes.Sampling{ null, hot }) |s| {
        const cases = [_]Case{.{ .prompt = &p2, .sampling = s, .max_new = 120 }};
        const r = try run(&cases, .{ .siblings = 2 });
        defer r.deinit();
        try expectSerial(&cases, r, .{});
        trees += r.trees;
        wins += r.sib_wins;
    }
    try std.testing.expect(trees > 0 and wins > 0);
}

test "zero depth under an overconfident head keeps the serial decode" {
    const dims: twin.Dims = .{ .conf = 0.3 };
    for ([_]u2{ 1, 2 }) |mode| {
        const cases = [_]Case{ .{ .prompt = &p2, .max_new = 100 }, .{ .prompt = &p1, .sampling = hot, .max_new = 60 } };
        const r = try run(cases[0..1], .{ .dims = dims, .joint = mode });
        defer r.deinit();
        try expectSerial(cases[0..1], r, dims);
        try std.testing.expect(r.zero > 0);
        const shared = try run(&cases, .{ .dims = dims, .joint = mode });
        defer shared.deinit();
        try expectSerial(&cases, shared, dims);
    }
}

test "shared forwards past 32 rows (Python's 64) keep every stream serial" {
    const dims: twin.Dims = .{ .bigram = 8.0, .head = 0.08, .block = 10 }; // a wider block than V4.1's 5 fills the rows
    const flat = [_]u32{ 7, 3, 9, 1 };
    const cases = [_]Case{ .{ .prompt = &p1, .max_new = 90 }, .{ .prompt = &p2, .max_new = 90 }, .{ .prompt = &p3, .max_new = 90 }, .{ .prompt = &flat, .max_new = 90 }, .{ .prompt = &p1, .sampling = warm, .max_new = 90 }, .{ .prompt = &p2, .sampling = hot, .max_new = 90 } };
    const r = try run(&cases, .{ .dims = dims, .slots = 6, .costs = .cheap, .depth = .{ .cap = 10 } });
    defer r.deinit();
    try expectSerial(&cases, r, dims);
    try std.testing.expect(r.widest > 32 and r.widest <= 64);
}
