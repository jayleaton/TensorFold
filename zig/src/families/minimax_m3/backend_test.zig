//! Host tests of the lane backend against core/lanes/fake.zig: same tokens, shared rounds, the radix cache, admission.
const std = @import("std");
const lanes = @import("lanes");
const iface = @import("iface.zig");
const fakes = @import("engine_fake.zig");
const backend = @import("backend.zig");
const pool = @import("pool.zig");

const gpa = std.testing.allocator;
const World = fakes.World;
const be = lanes.backend;

const Case = struct { prompt: []const u32, max_new: u32 = 40, sampling: ?lanes.Sampling = null, history_len: u32 = 0 };

fn tokens(n: usize, salt: u32) ![]u32 {
    const p = try gpa.alloc(u32, n);
    for (p, 0..) |*t, i| t.* = @intCast((i * 7 + salt) % 90);
    return p;
}

/// Every case's emitted tokens when the cases run together through the lane core on `backend`.
fn drive(b: be.Backend, facts: lanes.Model, cases: []const Case) ![][]u32 {
    var cfg = try lanes.Config.init(gpa, facts, iface.window_rows, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, b, clock.clock());
    defer core.deinit();
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(lanes.SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |c, *s, *p| {
        p.* = try lanes.SuffixLookup.init(gpa, .{ .min_match = 5 });
        s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .proposer = p.proposer(), .history_len = c.history_len });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try core.addStream(s);
    while (core.activeCount() > 0) try core.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

fn free(runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

/// The fake target on the same facts: what the reference backend emits for the cases.
fn reference(facts: lanes.Model, cases: []const Case) ![][]u32 {
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    return drive(target.backend(), facts, cases);
}

fn expectSame(want: [][]u32, got: [][]u32) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |a, b| try std.testing.expectEqualSlices(u32, a, b);
}

test "streams through the lane core emit what the fake backend emits, greedy and sampled, rank 1 mirroring" {
    const w = try World.create(gpa, 4000, 4, 160);
    defer w.destroy();
    const p1 = try tokens(300, 1);
    defer gpa.free(p1);
    const p2 = try tokens(2600, 2); // two chunks
    defer gpa.free(p2);
    const cases = [_]Case{
        .{ .prompt = p1, .max_new = 60 },
        .{ .prompt = p2, .max_new = 30, .sampling = .{ .seed = 7, .temperature = 0.8, .top_k = 20, .top_p = 0.9 } },
    };
    for (cases) |c| {
        const want = try reference(w.lanes.facts(), &.{c});
        defer free(want);
        const got = try drive(w.lanes.backend(), w.lanes.facts(), &.{c});
        defer free(got);
        try expectSame(want, got);
        try std.testing.expect(got[0].len > 0);
    }
    try std.testing.expectEqual(@as(usize, 0), w.lanes.lanes.count()); // released: slots free, pages back
    try std.testing.expect(w.pg.pages.idle());
    try w.replay();
    try std.testing.expectEqualSlices(u32, w.r0.kv, w.r1.kv);
    try std.testing.expectEqualSlices(i32, w.r0.table, w.r1.table);
    try std.testing.expectEqual(w.r0.runs.items.len, w.r1.runs.items.len);
}

test "several streams in one round share a forward and still emit what each emits alone" {
    const w = try World.create(gpa, 3000, 4, 200);
    defer w.destroy();
    const p1 = try tokens(200, 3);
    defer gpa.free(p1);
    const p2 = try tokens(450, 4);
    defer gpa.free(p2);
    const p3 = try tokens(130, 5);
    defer gpa.free(p3);
    const cases = [_]Case{ .{ .prompt = p1, .max_new = 50 }, .{ .prompt = p2, .max_new = 40, .sampling = .{ .seed = 3, .temperature = 1.0 } }, .{ .prompt = p3, .max_new = 30 } };
    const want = try reference(w.lanes.facts(), &cases);
    defer free(want);
    const got = try drive(w.lanes.backend(), w.lanes.facts(), &cases);
    defer free(got);
    try expectSame(want, got);
    try std.testing.expect(w.lanes.shared > 0);
    var shared_runs: usize = 0;
    for (w.r0.runs.items) |r| shared_runs += @intFromBool(r.kind == .shared);
    try std.testing.expect(shared_runs > 0);
    try w.replay();
    try std.testing.expectEqualSlices(u32, w.r0.kv, w.r1.kv);
}

test "the radix cache: a later turn resumes from the kept pages, emits what a fresh pass emits, rank 1 swaps the same pages" {
    const w = try World.create(gpa, 4000, 3, 120);
    defer w.destroy();
    w.pg.limit(60, 8);
    const base = try tokens(2700, 8);
    defer gpa.free(base);
    const turn2 = try gpa.alloc(u32, 3000);
    defer gpa.free(turn2);
    @memcpy(turn2[0..2700], base);
    for (turn2[2700..], 0..) |*t, i| t.* = @intCast((i * 11 + 5) % 90);
    const first = [_]Case{.{ .prompt = base, .max_new = 20, .history_len = 2700 }};
    const second = [_]Case{.{ .prompt = turn2, .max_new = 20, .history_len = 2700 }};
    const got1 = try drive(w.lanes.backend(), w.lanes.facts(), &first);
    defer free(got1);
    const runs_before = w.r0.runs.items.len;
    try std.testing.expect(w.pg.cache.live >= 1); // the pass kept a snapshot at its mark
    var swapped: usize = 0;
    for (w.tape.steps.items) |s| swapped += @intFromBool(s.h.kind == @backingInt(iface.Kind.keep) and s.h.flags & 1 != 0);
    try std.testing.expectEqual(@as(usize, 1), swapped); // rank 1 heard the page swap
    const got2 = try drive(w.lanes.backend(), w.lanes.facts(), &second);
    defer free(got2);
    try std.testing.expectEqual(@as(u64, 1), w.lanes.resumed);
    try std.testing.expectEqual(@as(u32, 1), w.pg.cache.store.tree.counts.hits);
    try std.testing.expectEqual(@as(u32, 2688), w.r0.runs.items[runs_before].starts[0]); // resumed at the kept mark's rows
    const fresh = try reference(w.lanes.facts(), &second);
    defer free(fresh);
    try expectSame(fresh, got2);
    try std.testing.expect(w.pg.pages.idle());
    try w.replay();
    try std.testing.expectEqualSlices(u32, w.r0.kv, w.r1.kv); // the follower's pages hold the same rows under the same ids
    try std.testing.expectEqualSlices(i32, w.r0.table, w.r1.table);
}

test "admission: vision, a context past the budget, a pool too small and a table without a free row are refused untouched" {
    const w = try World.create(gpa, 6000, 1, 20);
    defer w.destroy();
    const b = w.lanes.backend();
    const sent = w.tape.steps.items.len;
    const long = try tokens(5900, 1);
    defer gpa.free(long);
    const short = try tokens(500, 2);
    defer gpa.free(short);
    var vision = try lanes.Stream.init(gpa, .{ .id = "v", .prompt = short, .max_new = 10, .images = &.{.{ .pixels = &.{}, .gh = 1, .gw = 1, .at = 0, .tokens = 1 }} });
    defer vision.deinit(gpa);
    try std.testing.expectError(error.VisionUnsupported, b.prefill(&vision));
    var big = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = long, .max_new = 200 });
    defer big.deinit(gpa);
    try std.testing.expectError(error.PromptTooLong, b.prefill(&big));
    const mid = try tokens(3000, 3); // 24 pages against a pool of 20
    defer gpa.free(mid);
    var pool_big = try lanes.Stream.init(gpa, .{ .id = "p", .prompt = mid, .max_new = 100 });
    defer pool_big.deinit(gpa);
    try std.testing.expectError(error.ContextTooLong, b.prefill(&pool_big));
    try std.testing.expectEqual(sent, w.tape.steps.items.len); // no rank heard of any of them
    try std.testing.expect(w.pg.pages.idle());
    var a = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = short, .max_new = 10 });
    defer a.deinit(gpa);
    try b.prefill(&a);
    var again = try lanes.Stream.init(gpa, .{ .id = "c", .prompt = short, .max_new = 10 });
    defer again.deinit(gpa);
    try std.testing.expectError(error.LaneMemory, b.prefill(&again)); // one table row, taken
    try std.testing.expectEqual(@as(usize, 1), w.lanes.lanes.count());
    b.release(&a);
    try std.testing.expect(w.pg.pages.idle());
}

test "a cancelled prompt pass frees its pages and its slot on both ranks" {
    const w = try World.create(gpa, 6000, 2, 120);
    defer w.destroy();
    const ids = try tokens(5000, 7);
    defer gpa.free(ids);
    const Flag = struct {
        n: usize = 0,
        fn check(p: *anyopaque) bool {
            const f: *@This() = @ptrCast(@alignCast(p));
            f.n += 1;
            return f.n > 3; // true from the third chunk's gate on
        }
    };
    var flag: Flag = .{};
    var s = try lanes.Stream.init(gpa, .{ .id = "x", .prompt = ids, .max_new = 10, .cancel_check = .{ .ptr = &flag, .check = Flag.check } });
    defer s.deinit(gpa);
    const b = w.lanes.backend();
    try std.testing.expectError(error.Cancelled, b.prefill(&s));
    b.release(&s);
    try std.testing.expectEqual(@as(usize, 0), w.lanes.lanes.count());
    try std.testing.expect(w.pg.pages.idle());
    try w.replay();
    try std.testing.expectEqual(@as(usize, 0), w.hooks1.failures);
    for (w.r1.table) |e| try std.testing.expectEqual(@as(i32, -1), e); // the follower's row went back to -1
}

test "lane facts: exact windows over several streams, copy drafts only, no head" {
    const w = try World.create(gpa, 1000, 1, 10);
    defer w.destroy();
    const f = w.lanes.facts();
    try std.testing.expectEqual(iface.window_rows, f.exact_width);
    try std.testing.expectEqual(iface.shared_rows, f.batch_rows);
    try std.testing.expectEqual(iface.max_shared, f.max_streams);
    try std.testing.expect(f.hidden_rows and f.streams_exact and f.context_copies and !f.mtp and !f.speculate);
    var cfg = try lanes.Config.init(gpa, f, iface.window_rows, 15);
    defer cfg.deinit(gpa);
    try std.testing.expect(cfg.family_streams and !cfg.family_mtp);
    try std.testing.expectEqual(iface.window_rows - 1, cfg.max_copy);
}

test "the rest of the contract: no pipelining, no head, no trees, handles expire" {
    const w = try World.create(gpa, 1000, 1, 10);
    defer w.destroy();
    const b = w.lanes.backend();
    var s: lanes.Stream = undefined;
    try std.testing.expectError(error.NotPipelined, b.queue(&s, .{ .value = 1 }, 5));
    try std.testing.expectError(error.NoDraftHead, b.draft(&.{}));
    try std.testing.expectError(error.NoSuchToken, b.read(3));
    try std.testing.expectEqual(@as(u32, 128), pool.page_tokens);
}
