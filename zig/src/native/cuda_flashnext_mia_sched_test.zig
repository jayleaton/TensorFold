//! Her lane scheduling under TF_FLASHNEXT_ENGINE=mia: bursts past a small cache hit, and her shared rounds' budgets.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const mia = @import("flashnext_mia");
const family = @import("cuda_flashnext_mia.zig").family;
const lane_test = @import("cuda_flashnext_mia_lane_test.zig");
const L = mia.lane_backend;
const Toy = L.Toy;
const Box = lane_test.Box;

const t = std.testing;
const gpa = t.allocator;

/// Her lanes over the toy as cuda_native.openOn serves them: drafting, her stop rule, `streams` a forward.
fn served(toy: *Toy, streams: usize) L.Lanes(Toy) {
    var b = L.Lanes(Toy).init(gpa, toy, true, 1 << 16, L.max_rows);
    b.streams = streams;
    b.confidence = mia.engine.confidenceSetting();
    b.product_streams = mia.engine.productStreams();
    for (b.costs[0..L.max_rows], 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 10.0 + @as(f64, @floatFromInt(w)) };
    b.cost_count = L.max_rows;
    // shared rounds: a row costs 1 ms past 10 up to 64 rows
    for (b.shared[0..2], [_]u32{ 32, 64 }) |*c, w| c.* = .{ .width = w, .ms = 10.0 + @as(f64, @floatFromInt(w)) };
    b.shared_count = 2;
    b.mtp_ms = 6.0;
    return b;
}

test "flashnext_mia family: sixteen queued prompts with a 3-token cache hit prefill in one burst" {
    var toy: Toy = .{ .gpa = gpa };
    var b = served(&toy, 16);
    defer b.deinit();
    var cfg = try lanes.Config.init(gpa, b.facts(), L.max_rows, L.max_rows - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer core.deinit();
    // the server's store over her lanes (native/cuda.zig: her snap_* calls, f212's pass rules)
    const pc = api.prompt_cache;
    const vt: pc.Snapshots.VTable = .{ .bytes = L.Lanes(Toy).bytesFn, .save = L.Lanes(Toy).saveFn, .restore = L.Lanes(Toy).restoreFn, .drop = L.Lanes(Toy).dropFn };
    var store = pc.Store.init(gpa, .{ .ptr = &b, .vtable = &vt }, @import("cuda_snaps.zig").passRules(family, family.prompt_rows), 1 << 30);
    defer store.deinit();

    // one request keeps the state at its 3-token history (the chat template's opening, as in the c16 wave)
    var first = api.LaneHost.init(gpa, t.io, &core, .{ .lanes = 16 });
    first.cache = &store;
    const seed = [_]u32{ 1, 2, 3, 9, 8, 7, 6, 5 };
    var seed_req: api.Request = .{ .prompt = &seed, .max_tokens = 4, .history_len = 3 };
    var seed_box: Box = .{};
    defer seed_box.tokens.deinit(gpa);
    try first.engine().submit(1, &seed_req, .{ .ctx = &seed_box, .event = Box.event });
    try first.start();
    _ = try seed_box.wait();
    first.stop();
    try t.expect(store.entries.items.len > 0 and store.entries.items[0].at == 3);

    // sixteen queued before the host starts, each resuming at 3: one pass for all, each reply the target's own
    var host = api.LaneHost.init(gpa, t.io, &core, .{ .lanes = 16 });
    host.cache = &store;
    var prompts: [16][8]u32 = undefined;
    var requests: [16]api.Request = undefined;
    var boxes: [16]Box = @splat(.{});
    defer for (&boxes) |*x| x.tokens.deinit(gpa);
    const e = host.engine();
    for (&prompts, &requests, &boxes, 0..) |*p, *r, *x, i| {
        p.* = .{ 1, 2, 3, @intCast(10 + i), 4, 5, 6, @intCast(30 + i) };
        r.* = .{ .prompt = p, .max_tokens = 12, .history_len = 3 };
        try e.submit(@intCast(10 + i), r, .{ .ctx = x, .event = Box.event });
    }
    const passes = toy.passes;
    try host.start();
    defer host.stop();
    for (&prompts, &boxes) |*p, *x| {
        try t.expectEqual(api.Reason.length, try x.wait());
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, p);
        for (x.tokens.items) |token| {
            try t.expectEqual(lane_test.greedy(history.items), token);
            try history.append(gpa, token);
        }
    }
    try t.expectEqual(@as(usize, 1), toy.many); // f212's host refused the whole burst: sixteen passes one by one
    try t.expectEqual(@as(usize, 16), toy.many_prompts);
    try t.expectEqual(passes + 16, toy.passes); // no pass of its own beside the burst's
}

test "flashnext_mia family: four live streams' budgets are db281878's allocation over her shared costs" {
    var toy: Toy = .{ .gpa = gpa };
    var b = served(&toy, 4);
    defer b.deinit();
    var cfg = try lanes.Config.init(gpa, b.facts(), L.max_rows, L.max_rows - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer core.deinit();
    core.rule.live = 4; // past her product rule's two streams: the round loop's allocation sizes each chain
    var states: [4]?lanes.depth.State = @splat(null);
    defer for (&states) |*s| if (s.*) |*x| x.deinit(gpa);
    var whos: [4]lanes.depth.Who = undefined;
    for (&whos, &states) |*w, *s| w.* = .{ .state = s, .draft_room = 400, .finished = false, .forced = false };
    const No = struct {
        fn copy(_: *anyopaque, _: *const lanes.depth.Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    var out: [4]i64 = undefined;
    try core.rule.budgets(&whos, .{ .ptr = &unused, .copy = No.copy }, false, &out);
    // her prior's chains 0.82, 0.6232, 0.4487, 0.3051 at 8 ms overhead: (4 + 4 * 1.8919) / 34 = 0.3402 is the best
    try t.expectEqualSlices(i64, &.{ 3, 3, 3, 3 }, &out); // f212 priced: two each (9.7728 / 42 > 11.5676 / 52)
}
