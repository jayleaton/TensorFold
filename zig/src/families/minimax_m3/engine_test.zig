//! Host tests of the engine and its follower: chunking, resume, windows, cancellation, rank failure, fingerprints.
const std = @import("std");
const lanes = @import("lanes");
const iface = @import("iface.zig");
const engine = @import("engine.zig");
const follow = @import("follow.zig");
const fakes = @import("engine_fake.zig");
const dev = @import("engine_dev.zig");
const ranks = @import("ranks.zig");
const pool = @import("pool.zig");

const gpa = std.testing.allocator;
const World = fakes.World;

fn prompt(n: usize, salt: u32) ![]u32 {
    const p = try gpa.alloc(u32, n);
    for (p, 0..) |*t, i| t.* = @intCast((i * 7 + salt) % 90);
    return p;
}

/// A lease for `slot` as the backend takes one: pages promised, covered and written to both tables by the leader.
fn admit(w: *World, slot: u32, n: usize) !pool_state {
    var st: pool_state = .{ .st = .{ .slot = slot } };
    const ids = try prompt(n, 1);
    defer gpa.free(ids);
    var sink = w.pg.tables.sink(slot);
    st.marks = try w.pg.cache.admit(gpa, &st.st, ids, 0, 16, w.pg.window, sink.table(), w.pg.max_pages);
    try w.lanes.x.bind(slot);
    try w.lanes.x.begin(null, st.st.lease.ids.items);
    return st;
}
const pool_state = struct {
    st: @import("state.zig").Stream,
    marks: []const u32 = &.{},
    fn done(s: *pool_state, w: *World) void {
        gpa.free(s.marks);
        w.pg.cache.end(&s.st, &w.pg.tables);
    }
};

test "a prompt crosses in chunks of at most 2048 rows and rank 1 runs the same forwards on the same pages" {
    const w = try World.create(gpa, 6000, 2, 120);
    defer w.destroy();
    const ids = try prompt(5000, 3);
    defer gpa.free(ids);
    var s = try admit(w, 0, ids.len);
    defer s.done(w);
    const first = try w.lanes.x.prefill(ids, null, null, null);
    try std.testing.expectEqual(@as(usize, 3), w.r0.runs.items.len); // 2048 + 2048 + 904
    var at: usize = 0;
    for (w.r0.runs.items) |r| {
        try std.testing.expectEqualSlices(u32, ids[at..][0..r.ids.len], r.ids);
        try std.testing.expectEqual(@as(u32, @intCast(at)), r.starts[0]);
        try std.testing.expect(r.ids.len <= iface.prefill_rows);
        at += r.ids.len;
    }
    try std.testing.expectEqual(ids.len, at);
    try std.testing.expectEqual(ids.len, w.e0.pos());
    try std.testing.expectEqual(lanes.fake.next(ids, null, ids.len), first); // the token the target draws after the prompt
    try w.replay();
    try std.testing.expectEqual(w.r0.runs.items.len, w.r1.runs.items.len);
    for (w.r0.runs.items, w.r1.runs.items) |a, b| {
        try std.testing.expectEqualSlices(u32, a.ids, b.ids);
        try std.testing.expectEqualSlices(u32, a.starts, b.starts);
        try std.testing.expectEqualSlices(iface.Seg, a.segs, b.segs);
    }
    try std.testing.expectEqualSlices(u32, w.r0.kv, w.r1.kv);
    try std.testing.expectEqualSlices(i32, w.r0.table, w.r1.table);
    try std.testing.expectEqual(w.r0.draws, w.r1.draws); // the follower sent its span for every draw
}

test "chunk and resume equivalence: any cut of the prompt, and a pass resumed from adopted pages, give the same rows and token" {
    const w = try World.create(gpa, 6000, 2, 120);
    defer w.destroy();
    const ids = try prompt(4096 + 300, 5);
    defer gpa.free(ids);
    var whole = try admit(w, 0, ids.len);
    defer whole.done(w);
    const want = try w.lanes.x.prefill(ids, null, null, null);
    // a second slot whose first 2048 rows are the first slot's pages (adopted), then the rest in cuts at 512-row marks
    var other = try admit(w, 1, ids.len);
    defer other.done(w);
    const cut = [_]u32{ 2560, 3072, 4096 };
    var calls: usize = 0;
    const Hook = struct {
        fn call(ctx: *anyopaque, _: u32) void {
            @as(*usize, @ptrCast(@alignCast(ctx))).* += 1;
        }
    };
    // share the first 16 pages (2048 rows) the way a resume adopts them
    for (0..16) |i| w.pg.pages.share(&other.st.lease, i, whole.st.lease.ids.items[i]);
    try w.pg.tables.write(1, other.st.lease.ids.items, 0);
    try w.lanes.x.bind(1);
    try w.lanes.x.begin(null, other.st.lease.ids.items);
    const got = try w.lanes.x.prefill(ids, .{ .at = 2048 }, null, .{ .at = &cut, .ctx = &calls, .call = Hook.call });
    try std.testing.expectEqual(want, got);
    try std.testing.expectEqual(@as(usize, 3), calls); // every mark in the pass was called, once, at a chunk start
    var rows: usize = 2048;
    var chunks: usize = 0;
    for (w.r0.runs.items[3..]) |r| { // the second pass starts at the resume point and covers the rest, cut at the marks
        try std.testing.expectEqual(@as(u32, @intCast(rows)), r.starts[0]);
        rows += r.ids.len;
        chunks += 1;
    }
    try std.testing.expectEqual(ids.len, rows);
    try std.testing.expectEqual(@as(usize, 4), chunks); // 2048..2560..3072..4096..4396: a chunk ends at each mark
    // the pages the resumed pass read are the first slot's, so its rows are bitwise the first slot's
    try w.replay();
    try std.testing.expectEqualSlices(u32, w.r0.kv, w.r1.kv);
}

test "a window verifies at the committed position, keep commits the accepted rows, and shared windows keep their own pages" {
    const w = try World.create(gpa, 2000, 3, 60);
    defer w.destroy();
    const a_ids = try prompt(300, 2);
    defer gpa.free(a_ids);
    const b_ids = try prompt(700, 9);
    defer gpa.free(b_ids);
    var a = try admit(w, 0, a_ids.len);
    defer a.done(w);
    var b = try admit(w, 1, b_ids.len);
    defer b.done(w);
    try w.lanes.x.bind(0);
    const fa = try w.lanes.x.prefill(a_ids, null, null, null);
    try w.lanes.x.bind(1);
    const fb = try w.lanes.x.prefill(b_ids, null, null, null);
    var pa = [_]u32{ fa, 4, 5 };
    var pb = [_]u32{ fb, 6 };
    var oa: [3]u32 = undefined;
    var ob: [2]u32 = undefined;
    try w.lanes.x.verifyShared(&.{ .{ .slot = 0, .ids = &pa, .out = &oa }, .{ .slot = 1, .ids = &pb, .out = &ob } });
    const shared = w.r0.runs.items[w.r0.runs.items.len - 1];
    try std.testing.expectEqual(@import("engine_ops.zig").Rows.shared, shared.kind);
    try std.testing.expectEqualSlices(u32, &.{ 300, 700 }, shared.starts);
    try std.testing.expectEqual(@as(u32, 3), shared.segs[1].row0);
    // row 0 of window a is drawn after prompt + fa: the target's own function of those tokens
    var hist: [301]u32 = undefined;
    @memcpy(hist[0..300], a_ids);
    hist[300] = fa;
    try std.testing.expectEqual(lanes.fake.next(&hist, null, 301), oa[0]);
    try w.lanes.x.bind(0);
    try w.lanes.x.keep(2); // two of three rows accepted
    try std.testing.expectEqual(@as(usize, 302), w.e0.pos());
    try w.lanes.x.bind(1);
    try std.testing.expectEqual(@as(usize, 700), w.e0.pos()); // slot 1 was not committed
    try std.testing.expectError(error.BadKeep, w.e0.keep(5));
    var long: [iface.window_rows + 1]u32 = @splat(1);
    var out: [iface.window_rows + 1]u32 = undefined;
    try std.testing.expectError(error.WindowTooWide, w.e0.verify(&long, &out, 0));
    try w.replay();
    try std.testing.expectEqualSlices(u32, w.r0.kv, w.r1.kv);
}

test "cancellation at a chunk gate stops every rank before the same forward" {
    const w = try World.create(gpa, 6000, 2, 120);
    defer w.destroy();
    const ids = try prompt(5000, 4);
    defer gpa.free(ids);
    var s = try admit(w, 0, ids.len);
    defer s.done(w);
    const Flag = struct {
        calls: usize = 0,
        fn check(p: ?*anyopaque) bool {
            const f: *@This() = @ptrCast(@alignCast(p.?));
            f.calls += 1;
            return f.calls > 1; // the second chunk's gate sees it
        }
    };
    var flag: Flag = .{};
    try std.testing.expectError(error.Cancelled, w.lanes.x.prefill(ids, null, .{ .ptr = &flag, .check = Flag.check }, null));
    try std.testing.expectEqual(@as(usize, 1), w.r0.runs.items.len);
    try std.testing.expectEqualSlices(f64, &.{ 0, 1 }, w.r0.results.items);
    try w.lanes.x.release(0);
    try w.replay(); // the follower hears the same answers, skips the same forward, and goes on to the release
    try std.testing.expectEqual(@as(usize, 1), w.r1.runs.items.len);
    try std.testing.expectEqual(@as(usize, 1), w.tape.count(.release));
    try std.testing.expectEqual(@as(usize, 0), w.hooks1.failures);
}

test "another rank's vote cancels this rank's chunk too" {
    const w = try World.create(gpa, 3000, 2, 60);
    defer w.destroy();
    const ids = try prompt(2500, 6);
    defer gpa.free(ids);
    var s = try admit(w, 0, ids.len);
    defer s.done(w);
    try w.r0.inject.append(gpa, 1); // rank 1's gate says stop
    try std.testing.expectError(error.Cancelled, w.lanes.x.prefill(ids, null, null, null));
    try std.testing.expectEqual(@as(usize, 0), w.r0.runs.items.len);
}

test "a follower that fails tells the others and stops; rank 0 ends cleanly when the wire breaks" {
    {
        const w = try World.create(gpa, 3000, 2, 60);
        defer w.destroy();
        const ids = try prompt(500, 1);
        defer gpa.free(ids);
        var s = try admit(w, 0, ids.len);
        defer s.done(w);
        _ = try w.lanes.x.prefill(ids, null, null, null);
        w.r1.fail_run_at = 0; // rank 1's kernel fails on its first forward
        try std.testing.expectError(error.FakeKernel, w.replay());
        try std.testing.expectEqual(@as(usize, 1), w.hooks1.failures);
        try std.testing.expectEqualStrings("FakeKernel", w.hooks1.failed[0]);
    }
    {
        const w = try World.create(gpa, 3000, 2, 60);
        defer w.destroy();
        const ids = try prompt(500, 1);
        defer gpa.free(ids);
        var s = try admit(w, 0, ids.len);
        defer s.done(w);
        w.tape.fail_at = w.tape.steps.items.len + 1; // the chunk goes out; the next send finds rank 1 gone
        _ = try w.lanes.x.prefill(ids, null, null, null);
        var window = [_]u32{ 1, 2 };
        var out: [2]u32 = undefined;
        try std.testing.expectError(error.RankLost, w.lanes.x.verify(&window, &out, 0));
        try std.testing.expectEqual(@as(usize, 1), w.hooks0.failures);
        try std.testing.expectEqualStrings("WireBroken", w.hooks0.failed[0]);
        const sent = w.tape.steps.items.len;
        try std.testing.expectError(error.RankLost, w.lanes.x.keep(1)); // refused without touching the wire
        try std.testing.expectError(error.RankLost, w.lanes.x.release(0));
        try w.lanes.x.end(); // nothing to send, nothing to wait for
        try std.testing.expectEqual(sent, w.tape.steps.items.len);
        try std.testing.expectEqual(@as(usize, 1), w.hooks0.failures);
    }
}

test "rank 0 wakes the sleeping ranks once, and idles them only when no stream is left" {
    const w = try World.create(gpa, 2000, 2, 40);
    defer w.destroy();
    try std.testing.expectEqual(@as(usize, 1), w.hooks0.wakes); // the attach step
    try w.lanes.x.idle();
    try w.lanes.x.idle(); // already asleep: no second step
    try std.testing.expectEqual(@as(usize, 1), w.tape.count(.idle) - 1);
    const ids = try prompt(200, 1);
    defer gpa.free(ids);
    var s = try admit(w, 0, ids.len);
    defer s.done(w);
    try std.testing.expectEqual(@as(usize, 2), w.hooks0.wakes);
    try w.replay();
    try std.testing.expectEqual(@as(usize, 2), w.hooks1.sleeps); // at its start and at the idle step
}

test "the ranks' fingerprints are gathered as bytes and a difference refuses on every rank" {
    var r: fakes.Rank = .{ .gpa = gpa, .rank = 0, .world = 2 };
    const set: ranks.Settings = .{};
    const mine = ranks.Fingerprint.of(.{ .rank = 0, .world = 2 }, set, 4096, @splat(1), @splat(2), @splat(3), @splat(4));
    var same = mine;
    same.rank = 1;
    r.peers = &.{ std.mem.asBytes(&mine), std.mem.asBytes(&same) };
    const c = r.parts(4096).comm;
    try dev.agreeFingerprints(gpa, c, mine);
    var other = same;
    other.weights = @splat(9);
    r.peers = &.{ std.mem.asBytes(&mine), std.mem.asBytes(&other) };
    try std.testing.expectError(error.RankMismatch, dev.agreeFingerprints(gpa, c, mine));
}

test "a prompt past the capacity or a chunk off the page grid is refused by the engine" {
    const w = try World.create(gpa, 2000, 2, 40);
    defer w.destroy();
    var long: [2100]u32 = @splat(1);
    try std.testing.expectError(error.PromptTooLong, w.e0.prefill(&long, null, null, null));
    try std.testing.expectError(error.BadResume, w.e0.prefill(long[0..300], .{ .at = 100 }, null, null));
    try std.testing.expectError(error.BadChunk, w.e0.prefillChunk(long[0..100], 0, 300, null, null, null)); // a middle chunk off a page
    try std.testing.expectError(error.BadChunk, w.e0.prefillChunk(long[0..100], 128, 228, null, null, null)); // not at the position
}

test "the table row bytes and chunk grid are the contract's" {
    const w = try World.create(gpa, 2000, 2, 40);
    defer w.destroy();
    try std.testing.expectEqual(@as(u64, 4 * iface.maxPages(w.e0.capacity)), w.e0.seqBytes());
    try std.testing.expect(iface.prefill_rows % pool.page_tokens == 0);
}
