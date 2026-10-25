//! Host tests of the dspark draft in the engine: drafted replies equal plain decode, both ranks absorb and draft alike.
const std = @import("std");
const lanes = @import("lanes");
const iface = @import("iface.zig");
const fakes = @import("engine_fake.zig");
const dd = @import("dspark_drafter.zig");
const dc = @import("dspark_config.zig");
const engine = @import("engine.zig");

const gpa = std.testing.allocator;
const World = fakes.World;
const be = lanes.backend;

const Case = struct { prompt: []const u32, max_new: u32 = 60, sampling: ?lanes.Sampling = null, history_len: u32 = 0 };

fn tokens(n: usize, salt: u32) ![]u32 {
    const p = try gpa.alloc(u32, n);
    for (p, 0..) |*t, i| t.* = @intCast((i * 7 + salt) % 90);
    return p;
}

/// The cases through the lane core on `b`; every case's emitted tokens.
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

fn plain(facts: lanes.Model, cases: []const Case) ![][]u32 {
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    return drive(target.backend(), facts, cases);
}

fn expectSame(want: [][]u32, got: [][]u32) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |a, b| try std.testing.expectEqualSlices(u32, a, b);
}

/// The world's lanes backend behind the draft adapter and the core's Drafted wrapper.
const Setup = struct {
    draft: *dd.Draft,
    wrapped: lanes.drafted.Drafted,

    fn init(w: *World) !Setup {
        const d = try dd.Draft.init(gpa, &w.lanes, dc.flash, .{});
        errdefer d.deinit();
        return .{ .draft = d, .wrapped = try lanes.drafted.Drafted.init(gpa, w.lanes.backend(), d.drafter()) };
    }

    fn deinit(s: *Setup) void {
        s.wrapped.deinit();
        s.draft.deinit();
    }
};

fn runDrafted(w: *World, cases: []const Case) ![][]u32 {
    var s = try Setup.init(w);
    defer s.deinit();
    const facts = s.wrapped.facts(w.lanes.facts());
    return drive(s.wrapped.backend(), facts, cases);
}

test "drafted decode equals plain decode, greedy and sampled, and the draft is right often enough to save forwards" {
    const w = try World.createWith(gpa, 4000, 2, 160, true);
    defer w.destroy();
    const p1 = try tokens(300, 1);
    defer gpa.free(p1);
    const p2 = try tokens(2600, 2); // two chunks
    defer gpa.free(p2);
    const cases = [_]Case{
        .{ .prompt = p1, .max_new = 80 },
        .{ .prompt = p2, .max_new = 40, .sampling = .{ .seed = 7, .temperature = 0.8, .top_k = 20, .top_p = 0.9 } },
    };
    for (cases) |c| {
        const want = try plain(w.lanes.facts(), &.{c});
        defer free(want);
        const runs_before = w.r0.runs.items.len;
        const got = try runDrafted(w, &.{c});
        defer free(got);
        try expectSame(want, got);
        try std.testing.expect(got[0].len > 0);
        _ = runs_before;
    }
    try std.testing.expect(w.r0.fd.passes > 0 and w.r0.fd.absorbs > 0);
    try std.testing.expectEqual(@as(usize, 0), w.r0.fd.gaps); // every absorb continued its ring
    // the greedy case: the fake draft predicts the fake target exactly, so most forwards carry several accepted tokens
    var wide: usize = 0;
    for (w.r0.runs.items) |r| wide += @intFromBool(r.ids.len > 2);
    try std.testing.expect(wide > 0);
    try std.testing.expectEqual(@as(usize, 0), w.lanes.lanes.count());
    try std.testing.expect(w.pg.pages.idle());
}

test "a draft that is often wrong costs acceptance and nothing else" {
    const w = try World.createWith(gpa, 3000, 2, 120, true);
    defer w.destroy();
    w.r0.fd.wrong_every = 3;
    w.r1.fd.wrong_every = 3;
    const p = try tokens(500, 4);
    defer gpa.free(p);
    const cases = [_]Case{.{ .prompt = p, .max_new = 90 }};
    const want = try plain(w.lanes.facts(), &cases);
    defer free(want);
    const got = try runDrafted(w, &cases);
    defer free(got);
    try expectSame(want, got);
    try std.testing.expect(w.r0.fd.made > 0);
}

test "several streams in shared rounds draft and verify as each does alone" {
    const w = try World.createWith(gpa, 3000, 4, 200, true);
    defer w.destroy();
    const p1 = try tokens(200, 3);
    defer gpa.free(p1);
    const p2 = try tokens(450, 4);
    defer gpa.free(p2);
    const p3 = try tokens(130, 5);
    defer gpa.free(p3);
    const cases = [_]Case{ .{ .prompt = p1, .max_new = 50 }, .{ .prompt = p2, .max_new = 40, .sampling = .{ .seed = 3, .temperature = 1.0 } }, .{ .prompt = p3, .max_new = 30 } };
    const want = try plain(w.lanes.facts(), &cases);
    defer free(want);
    const got = try runDrafted(w, &cases);
    defer free(got);
    try expectSame(want, got);
    try std.testing.expectEqual(@as(usize, 0), w.r0.fd.gaps);
}

test "rank 1 absorbs and drafts what rank 0 does, step for step" {
    const w = try World.createWith(gpa, 3000, 3, 150, true);
    defer w.destroy();
    const p1 = try tokens(700, 6);
    defer gpa.free(p1);
    const p2 = try tokens(260, 9);
    defer gpa.free(p2);
    const cases = [_]Case{ .{ .prompt = p1, .max_new = 40 }, .{ .prompt = p2, .max_new = 40 } };
    const got = try runDrafted(w, &cases);
    defer free(got);
    try std.testing.expect(w.tape.count(.absorb) > 0 and w.tape.count(.draft) > 0);
    try w.replay();
    try std.testing.expectEqual(w.r0.fd.absorbs, w.r1.fd.absorbs);
    try std.testing.expectEqual(w.r0.fd.passes, w.r1.fd.passes);
    try std.testing.expectEqualSlices(u32, w.r0.kv, w.r1.kv);
    for (w.r0.fd.rings.items, w.r1.fd.rings.items) |a, b| {
        try std.testing.expectEqual(a.lo, b.lo);
        try std.testing.expectEqualSlices(u32, a.toks.items, b.toks.items);
    }
}

test "the draft's taps must be the engine's: any other layers are refused before a forward" {
    const w = try World.createWith(gpa, 1000, 1, 20, true);
    defer w.destroy();
    const b = w.lanes.backend();
    try b.vtable.prepare_features.?(b.ptr, dc.flash.tapIds());
    try std.testing.expectError(error.TapsMismatch, b.vtable.prepare_features.?(b.ptr, &.{ 1, 2, 3 }));
    const w0 = try World.create(gpa, 1000, 1, 20);
    defer w0.destroy();
    const b0 = w0.lanes.backend();
    try std.testing.expectError(error.TapsMismatch, b0.vtable.prepare_features.?(b0.ptr, dc.flash.tapIds())); // an engine that does not tap
}

test "a prompt's last row survives another stream's forward until its first draft" {
    const w = try World.createWith(gpa, 3000, 2, 120, true);
    defer w.destroy();
    const a_ids = try tokens(300, 2);
    defer gpa.free(a_ids);
    const b_ids = try tokens(520, 8);
    defer gpa.free(b_ids);
    var a = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = a_ids, .max_new = 20 });
    defer a.deinit(gpa);
    var b = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = b_ids, .max_new = 20 });
    defer b.deinit(gpa);
    const t = w.lanes.backend();
    try t.prefill(&a);
    try t.prefill(&b); // another stream's whole prompt runs before a's first draft
    const f = try t.features(&a, dc.flash.tapIds(), a_ids.len - 1, 1);
    try std.testing.expectEqual(@as(u32, 1), f.rows);
    const row = f.offset / f.row_bytes;
    try std.testing.expectEqual(@as(u64, iface.tapWindowRow(0)), row);
    const tr = w.r0.taps[row];
    try std.testing.expectEqual(@as(u32, @intCast(a_ids.len - 1)), tr.pos);
    try std.testing.expectEqual(a_ids[a_ids.len - 1], tr.tok);
    t.release(&a);
    t.release(&b);
}

test "the radix cache keeps the draft's ring with a prompt mark: a resumed turn drafts from the whole context" {
    const w = try World.createWith(gpa, 4000, 3, 120, true);
    defer w.destroy();
    w.pg.limit(60, 8);
    const base = try tokens(2700, 8);
    defer gpa.free(base);
    const turn2 = try gpa.alloc(u32, 3000);
    defer gpa.free(turn2);
    @memcpy(turn2[0..2700], base);
    for (turn2[2700..], 0..) |*t, i| t.* = @intCast((i * 11 + 5) % 90);
    const first = [_]Case{.{ .prompt = base, .max_new = 20, .history_len = 2700 }};
    const second = [_]Case{.{ .prompt = turn2, .max_new = 40, .history_len = 2700 }};
    const got1 = try runDrafted(w, &first);
    defer free(got1);
    try std.testing.expect(w.pg.cache.live >= 1);
    try std.testing.expectEqual(@as(usize, 1), w.r0.fd.snaps.count()); // the mark's ring was saved on both ranks
    const got2 = try runDrafted(w, &second);
    defer free(got2);
    try std.testing.expectEqual(@as(u64, 1), w.lanes.resumed);
    const fresh = try plain(w.lanes.facts(), &second);
    defer free(fresh);
    try expectSame(fresh, got2);
    try std.testing.expectEqual(@as(usize, 0), w.r0.fd.partial); // every pass saw the whole context, the restored prefix included
    try std.testing.expect(w.pg.pages.idle());
    try w.replay();
    try std.testing.expectEqual(w.r0.fd.snaps.count(), w.r1.fd.snaps.count());
    for (w.r0.fd.rings.items, w.r1.fd.rings.items) |a, b| try std.testing.expectEqualSlices(u32, a.toks.items, b.toks.items);
}

test "without a ring snapshot a resumed turn drafts from its resume point and still replies the same" {
    const w = try World.createWith(gpa, 4000, 3, 120, true);
    defer w.destroy();
    w.pg.limit(60, 8);
    w.pg.cache.ring = null; // the cache keeps no rings
    const base = try tokens(2700, 8);
    defer gpa.free(base);
    const turn2 = try gpa.alloc(u32, 3000);
    defer gpa.free(turn2);
    @memcpy(turn2[0..2700], base);
    for (turn2[2700..], 0..) |*t, i| t.* = @intCast((i * 11 + 5) % 90);
    const first = [_]Case{.{ .prompt = base, .max_new = 20, .history_len = 2700 }};
    const second = [_]Case{.{ .prompt = turn2, .max_new = 40, .history_len = 2700 }};
    const got1 = try runDrafted(w, &first);
    defer free(got1);
    const got2 = try runDrafted(w, &second);
    defer free(got2);
    const fresh = try plain(w.lanes.facts(), &second);
    defer free(fresh);
    try expectSame(fresh, got2);
    try std.testing.expect(w.r0.fd.partial > 0); // the passes had only the rows since the resume
}
