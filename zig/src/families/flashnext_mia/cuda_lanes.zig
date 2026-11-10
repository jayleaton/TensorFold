//! Flash Next CUDA lane backend: stream windows share one forward; tokens equal serial decoding's; generic over `E`.
const std = @import("std");
const envs = @import("cuda_env.zig");
const lanes = @import("lanes");
const media_mod = @import("cuda_media.zig");
const decode = @import("cuda_decode.zig");
const lanes_core = @import("cuda_lanes_core.zig");
const toy_part = @import("cuda_lanes_toy.zig");

const be = lanes.backend;

/// A verify window's rows at most (the pending token and 15 drafts; cuda_state.max_rows).
pub const max_rows = 16;
pub const max_drafts = max_rows - 1;
/// The MTP head's chain stops before a later draft under this temperature-1 probability (Python CONFIDENCE).
pub const default_confidence = 0.70;

/// Head acceptance prior at depth 1, 2, ... given the ones before, until a stream has its own rounds.
pub const draft_prior = [_]f64{ 0.82, 0.76, 0.72, 0.68, 0.65, 0.62, 0.6, 0.58, 0.56, 0.54, 0.52, 0.5, 0.5, 0.5, 0.5 };

/// Her store kept and resumed nothing for shorter prompts (db281878 Rules.min_prompt), so her host burst them all.
pub const burst_short = 4096;

/// First tokens a handle names (the prompt's draw is on the host once prefill returns).
pub const ring = 1024;
/// a pass keeps its state late only this close to the prompt's end (the indexer key ring holds 4096+ rows)
pub const late_keep_rows = 1024;

/// Two ranks' prepare / agree / commit: local fallible steps, one gather, then collectives; `fatal` ends a split rank.
pub const Gate = struct {
    ptr: *anyopaque,
    agree: *const fn (ptr: *anyopaque, ok: bool) anyerror!bool,
    fatal: *const fn (ptr: *anyopaque, what: []const u8, err: anyerror) void,
};

/// Where rank 0's prompt-cache decisions go (cuda_mirror's Leader: a note call to rank 1).
pub const Notes = struct { ptr: *anyopaque, send: *const fn (ptr: *anyopaque, bytes: []const u8) void };

pub const Note = enum(u8) { pass = 1, drop = 2, _ };

/// A filling stream's state after a step of her layer-sliced fill (her lanes core's FillState; f212 has none).
pub const FillState = enum { filling, done, cancelled };

/// The prompt cache's family calls (core/prompt_cache.zig Snapshots.VTable) and budget for native/cuda.zig.
pub const CacheHooks = struct {
    ptr: *anyopaque,
    bytes: *const fn (ptr: *anyopaque, at: u32) u64,
    save: *const fn (ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque,
    restore: *const fn (ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void,
    drop: *const fn (ptr: *anyopaque, saved: *anyopaque) void,
    budget: u64,
};

pub const Lanes = lanes_core.Lanes;

/// Shared rounds' cost points at most (2, 4, ... 64 streams).
pub const shared_points = 8;
/// Streams one shared forward holds at most (the engine's round segments).
pub const max_streams = 64;

// -- tests: a toy engine with several sequences -------------------------------------------------------------------

pub const Toy = toy_part.Toy;

const Case = struct { prompt: []const u32, max_new: u32, sampling: ?lanes.Sampling = null, drafts: bool = true };

/// Serial decoding on the toy alone: the tokens every lane run must reproduce.
fn serialTokens(gpa: std.mem.Allocator, c: Case) ![]u32 {
    var t: Toy = .{ .gpa = gpa };
    const q = try t.newSeq(1 << 20);
    defer t.freeSeq(q);
    t.bind(q);
    const first = try t.prefill(c.prompt, c.sampling);
    var r = try decode.serial(Toy, &t, gpa, first, .{ .count = c.max_new });
    defer r.deinit(gpa);
    return gpa.dupe(u32, r.tokens.items);
}

fn laneTokens(gpa: std.mem.Allocator, cases: []const Case, together: bool, n_streams: usize, widest: *usize) ![][]u32 {
    var t: Toy = .{ .gpa = gpa };
    var b = Lanes(Toy).init(gpa, &t, true, 1 << 20, max_rows);
    defer b.deinit();
    b.streams = n_streams;
    for (b.shared[0..2], [_]usize{ 2, 4 }) |*c, n| c.* = .{ .width = @intCast(n * max_rows), .ms = 9.0 + 0.2 * @as(f64, @floatFromInt(n * max_rows)) };
    b.shared_count = if (n_streams > 1) 2 else 0;
    var costs: [max_rows]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 8.0 + 0.3 * @as(f64, @floatFromInt(w)) };
    b.costs = costs;
    b.cost_count = max_rows;
    b.mtp_ms = 0.4;
    var cfg = try lanes.Config.init(gpa, b.facts(), max_rows, max_drafts);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    for (cases, streams) |c, *s| s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .sampling = c.sampling, .drafts = c.drafts });
    defer for (streams) |*s| s.deinit(gpa);
    if (together) {
        for (streams) |*s| try engine.addStream(s);
        while (engine.activeCount() > 0) try engine.step();
    } else for (streams) |*s| {
        try engine.addStream(s);
        while (engine.activeCount() > 0) try engine.step();
    }
    try std.testing.expectEqual(@as(usize, 0), t.seqs); // every sequence freed at release
    widest.* = t.widest;
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

test "lane rounds over the toy engine equal serial decoding: one stream at a time, interleaved, and in shared rounds" {
    const gpa = std.testing.allocator;
    const cases = [_]Case{
        .{ .prompt = &.{ 3, 1, 4, 1, 5 }, .max_new = 80 },
        .{ .prompt = &.{ 2, 7, 1, 8, 2, 8 }, .max_new = 64, .sampling = .{ .seed = 9, .temperature = 0.7 } },
        .{ .prompt = &.{ 6, 6 }, .max_new = 40, .drafts = false },
        .{ .prompt = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, .max_new = 120, .sampling = .{ .seed = 3 } },
    };
    for ([_]struct { bool, usize }{ .{ false, 1 }, .{ true, 1 }, .{ true, 4 }, .{ true, 2 } }) |run| {
        const together, const streams = run;
        var widest: usize = 0;
        const got = try laneTokens(gpa, &cases, together, streams, &widest);
        // several streams' windows shared a forward when the backend holds several
        try std.testing.expect(widest <= streams and (streams == 1 or widest > 1));
        defer {
            for (got) |g| gpa.free(g);
            gpa.free(got);
        }
        for (cases, got) |c, g| {
            const want = try serialTokens(gpa, c);
            defer gpa.free(want);
            try std.testing.expectEqualSlices(u32, want, g);
        }
    }
}

test "the confidence rule's short chains reach the round loop as the held count" {
    const gpa = std.testing.allocator;
    var t: Toy = .{ .gpa = gpa };
    var b = Lanes(Toy).init(gpa, &t, true, 4096, max_rows);
    defer b.deinit();
    const be_ = b.backend();
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 1, 2, 3 }, .max_new = 50 });
    defer s.deinit(gpa);
    try be_.prefill(&s);
    const h = try be_.first(&s, 3);
    const first = try be_.read(h);
    try std.testing.expectEqual(Toy.next(&.{ 1, 2, 3 }, &.{}, null, 3), first);
    try be_.draft(&.{.{ .stream = &s, .follow = &.{}, .first = .{ .handle = h }, .rows = null, .start = 3, .position = 4, .depth = 15 }});
    const held = (try be_.vtable.tree.?(be_.ptr, &s, gpa)).?;
    try std.testing.expect(held.count >= 1 and held.count <= 15);
    // a window asking for more held rows than the chain made still verifies (the extra rows repeat its last draft)
    var positions: [16]u64 = undefined;
    for (&positions, 0..) |*p, r| p.* = 4 + r;
    var sampled: [16]u32 = undefined;
    var drafts: [15]u32 = undefined;
    var out = [_]be.Verified{.{ .sampled = &sampled, .drafts = &drafts }};
    try be_.verify(&.{.{ .stream = &s, .pending = first, .held = 15, .tokens = &.{}, .parents = null, .positions = &positions }}, &out);
    try std.testing.expectEqual(Toy.next(&.{ 1, 2, 3 }, &.{first}, null, 4), sampled[0]);
    try be_.keep(&.{.{ .stream = &s, .pending = first, .held = 15, .tokens = &.{}, .parents = null, .positions = &positions }}, &.{&.{0}});
    try std.testing.expectEqual(@as(u64, 4), t.pos());
    be_.release(&s);
    try std.testing.expectEqual(@as(usize, 0), t.seqs);
}

test "a prompt past the window is refused before a sequence is made" {
    const gpa = std.testing.allocator;
    var t: Toy = .{ .gpa = gpa };
    var b = Lanes(Toy).init(gpa, &t, false, 32, max_rows);
    defer b.deinit();
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 1, 2, 3 }, .max_new = 20 });
    defer s.deinit(gpa);
    try std.testing.expectError(error.PromptTooLong, b.backend().prefill(&s));
    try std.testing.expectEqual(@as(usize, 0), t.seqs);
    try std.testing.expectEqual(@as(u32, 1), b.facts().exact_width);
}

test "costs are measured by the same calls a decode makes" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    var t: Toy = .{ .gpa = gpa };
    var b = Lanes(Toy).init(gpa, &t, true, 1 << 16, 7);
    defer b.deinit();
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i % 50);
    b.streams = 4;
    try b.measure(threaded.io(), &ids);
    try std.testing.expectEqual(@as(usize, 7), b.cost_count);
    try std.testing.expectEqual(@as(usize, 2), b.shared_count); // 2 and 4 streams' full windows
    try std.testing.expectEqual(@as(u32, 28), b.shared[1].width);
    try std.testing.expectEqual(@as(u32, 6), b.facts().drafts);
    try std.testing.expectEqual(@as(usize, 0), t.seqs);
    for (b.costs[0..b.cost_count], 1..) |c, w| try std.testing.expectEqual(@as(u32, @intCast(w)), c.width);
}

test "every verify width, head level and shared width is warmed and timed in each pass" {
    const gpa = std.testing.allocator;
    const costs = @import("cuda_lanes_costs.zig");
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    var t: Toy = .{ .gpa = gpa };
    var b = Lanes(Toy).init(gpa, &t, true, 1 << 16, 7);
    defer b.deinit();
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i % 50);
    b.streams = 4;
    try b.measure(threaded.io(), &ids);
    // 32 clock-warming rounds, then per pass: 7 verify widths, 2 head levels and 2 shared widths, each a pass
    const pass = costs.shared_warm + costs.shared_reps;
    try std.testing.expectEqual(@as(usize, 32 + costs.shared_passes * (7 + 2 + 2) * pass), t.rounds);
}

test "a rank's failed prepare stops both ranks before a collective; an engine error after the agreement is fatal" {
    const gpa = std.testing.allocator;
    const G = struct {
        refuse: bool = false,
        agreements: usize = 0,
        fatal_what: ?[]const u8 = null,
        fn agree(p: *anyopaque, ok: bool) anyerror!bool {
            const g: *@This() = @ptrCast(@alignCast(p));
            g.agreements += 1;
            return ok and !g.refuse;
        }
        fn fatal(p: *anyopaque, what: []const u8, _: anyerror) void {
            const g: *@This() = @ptrCast(@alignCast(p));
            g.fatal_what = what;
        }
    };
    var g: G = .{ .refuse = true };
    var t: Toy = .{ .gpa = gpa };
    var b = Lanes(Toy).init(gpa, &t, true, 4096, max_rows);
    defer b.deinit();
    b.gate = .{ .ptr = &g, .agree = G.agree, .fatal = G.fatal };
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 1, 2, 3 }, .max_new = 20 });
    defer s.deinit(gpa);
    // the other rank failed its prepare: no sequence is left, the engine never ran
    try std.testing.expectError(error.OtherRankFailed, b.backend().prefill(&s));
    try std.testing.expectEqual(@as(usize, 0), t.seqs);
    try std.testing.expectEqual(@as(usize, 1), g.agreements);
    // this rank's own failure still reaches the agreement, so the other rank hears of it
    var long = try lanes.Stream.init(gpa, .{ .id = "l", .prompt = &.{ 1, 2, 3 }, .max_new = 5000 });
    defer long.deinit(gpa);
    g.refuse = false;
    try std.testing.expectError(error.PromptTooLong, b.backend().prefill(&long));
    try std.testing.expectEqual(@as(usize, 2), g.agreements);
    // agreed, then the engine fails (the toy's sequence is too short for the window): fatal
    try b.backend().prefill(&s);
    try std.testing.expectEqual(@as(usize, 3), g.agreements);
    b.lanes.get(&s).?.seq.limit = 4;
    var positions = [_]u64{ 4, 5 };
    var sampled: [2]u32 = undefined;
    var drafts: [1]u32 = undefined;
    var out = [_]be.Verified{.{ .sampled = &sampled, .drafts = &drafts }};
    try std.testing.expectError(error.SeqFull, b.backend().verify(&.{.{ .stream = &s, .pending = 1, .held = 0, .tokens = &.{2}, .parents = null, .positions = &positions }}, &out));
    try std.testing.expectEqualStrings("a round's forward", g.fatal_what.?);
    b.backend().release(&s);
    try std.testing.expectEqual(@as(usize, 0), t.seqs);
}

test "the ranks agree on admission before a sequence is made: its growth runs a collective the other rank must join" {
    const gpa = std.testing.allocator;
    const G = struct {
        toy: *Toy,
        refuse: bool = false,
        seqs_at: [4]usize = undefined,
        agreements: usize = 0,
        fn agree(p: *anyopaque, ok: bool) anyerror!bool {
            const g: *@This() = @ptrCast(@alignCast(p));
            g.seqs_at[g.agreements] = g.toy.seqs;
            g.agreements += 1;
            return ok and !g.refuse;
        }
        fn fatal(_: *anyopaque, _: []const u8, _: anyerror) void {}
    };
    var t: Toy = .{ .gpa = gpa };
    var g: G = .{ .toy = &t };
    var b = Lanes(Toy).init(gpa, &t, true, 4096, max_rows);
    defer b.deinit();
    b.gate = .{ .ptr = &g, .agree = G.agree, .fatal = G.fatal };
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 1, 2, 3 }, .max_new = 20 });
    defer s.deinit(gpa);
    // admitted on both ranks: the vote comes first, the sequence after it
    try b.backend().prefill(&s);
    try std.testing.expectEqual(@as(usize, 1), g.agreements);
    try std.testing.expectEqual(@as(usize, 0), g.seqs_at[0]);
    try std.testing.expectEqual(@as(usize, 1), t.seqs);
    b.backend().release(&s);
    // the other rank refused (its NoRoom): this rank made no sequence, nothing reserved
    g.refuse = true;
    try std.testing.expectError(error.OtherRankFailed, b.backend().prefill(&s));
    try std.testing.expectEqual(@as(usize, 0), g.seqs_at[1]);
    try std.testing.expectEqual(@as(usize, 0), t.seqs);
    try std.testing.expectEqual(@as(usize, 0), b.reserved);
}

test "admission reserves each stream's whole window: a prompt that would not fit beside the others waits for room" {
    const gpa = std.testing.allocator;
    var t: Toy = .{ .gpa = gpa };
    // room for two streams of 3 + 81 + 16 = 100 rows (10,000 bytes each)
    t.budget.limit = 20_000;
    var b = Lanes(Toy).init(gpa, &t, true, 4096, max_rows);
    defer b.deinit();
    var ss: [3]lanes.Stream = undefined;
    for (&ss) |*s| s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 1, 2, 3 }, .max_new = 81 });
    defer for (&ss) |*s| s.deinit(gpa);
    try b.backend().prefill(&ss[0]);
    try b.backend().prefill(&ss[1]);
    try std.testing.expectError(error.NoRoom, b.backend().prefill(&ss[2]));
    try std.testing.expectEqual(@as(usize, 2), t.seqs);
    // a stream prefilled again keeps its own reservation
    try b.backend().prefill(&ss[1]);
    b.backend().release(&ss[0]);
    try b.backend().prefill(&ss[2]);
    try std.testing.expectEqual(@as(usize, 20_000), b.reserved);
    // overcommit admits past the reservations
    b.overcommit = true;
    try b.backend().prefill(&ss[0]);
    for (&ss) |*s| b.backend().release(s);
    try std.testing.expectEqual(@as(usize, 0), b.reserved);
}

/// The prompt cache part a test plays: keeps every mark the pass offers (LaneHost hook -> Store.keep -> family save).
const KeepAll = struct {
    hooks: CacheHooks,
    saved: std.ArrayList(*anyopaque) = .empty,

    fn hook(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
        const k: *KeepAll = @ptrCast(@alignCast(ptr));
        const got = k.hooks.save(k.hooks.ptr, s, at) catch return;
        k.saved.append(std.testing.allocator, got) catch {};
    }
};

test "a resumed prompt equals a fresh one: kept at the pass's marks, any later stream resumes from them" {
    const gpa = std.testing.allocator;
    var t: Toy = .{ .gpa = gpa };
    var b = Lanes(Toy).init(gpa, &t, true, 1 << 16, max_rows);
    defer b.deinit();
    var store: KeepAll = .{ .hooks = b.cacheHooks(1 << 30) };
    defer store.saved.deinit(gpa);
    var long: [600]u32 = undefined;
    for (&long, 0..) |*x, i| x.* = @intCast((i * 13 + 5) % 47);
    // turn 1: marks at 200 (a shared block) and 500 (its history); two states kept, the pass's first draw as cold
    var s1 = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = long[0..520], .max_new = 8 });
    defer s1.deinit(gpa);
    s1.reuse = .{ .marks = &.{ 200, 500 }, .hook = .{ .ptr = &store, .at = KeepAll.hook } };
    try b.backend().prefill(&s1);
    const first1 = try b.backend().read(try b.backend().first(&s1, 520));
    try std.testing.expectEqual(Toy.next(long[0..520], &.{}, null, 520), first1);
    try std.testing.expectEqual(@as(usize, 2), store.saved.items.len);
    try std.testing.expectEqual(@as(u32, 0), s1.cached);
    b.backend().release(&s1); // the kept states outlive the stream (their keepers hold the rows)
    // turn 2 (a new stream) resumes at 500: its first draw equals a fresh pass's
    var s2 = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &long, .max_new = 8 });
    defer s2.deinit(gpa);
    s2.reuse = .{ .saved = store.saved.items[1], .at = 500 };
    const resumed0 = t.resumed;
    try b.backend().prefill(&s2);
    try std.testing.expectEqual(@as(u32, 500), s2.cached);
    try std.testing.expect(t.resumed > resumed0);
    try std.testing.expectEqual(Toy.next(&long, &.{}, null, 600), try b.backend().read(try b.backend().first(&s2, 600)));
    b.backend().release(&s2);
    // a state the store drops goes with its keeper and snapshot
    for (store.saved.items) |sv| b.cacheHooks(0).drop(&b, sv);
    try std.testing.expectEqual(@as(usize, 0), t.snaps);
    try std.testing.expectEqual(@as(usize, 0), t.seqs);
    try std.testing.expectEqual(@as(u64, 0), b.kept_bytes);
}

test "rank 1 keeps what rank 0's store kept, through the notes, and resumes the same state" {
    const gpa = std.testing.allocator;
    const Sink = struct {
        notes: std.ArrayList(u8) = .empty,
        lens: std.ArrayList(usize) = .empty,
        fn send(ptr: *anyopaque, bytes: []const u8) void {
            const k: *@This() = @ptrCast(@alignCast(ptr));
            k.lens.append(std.testing.allocator, bytes.len) catch {};
            k.notes.appendSlice(std.testing.allocator, bytes) catch {};
        }
    };
    var sink: Sink = .{};
    defer sink.notes.deinit(gpa);
    defer sink.lens.deinit(gpa);
    var t0: Toy = .{ .gpa = gpa };
    var r0 = Lanes(Toy).init(gpa, &t0, true, 1 << 16, max_rows);
    defer r0.deinit();
    r0.notes = .{ .ptr = &sink, .send = Sink.send };
    var t1: Toy = .{ .gpa = gpa };
    var r1 = Lanes(Toy).init(gpa, &t1, true, 1 << 16, max_rows);
    defer r1.deinit();
    r1.follower = true;
    var store: KeepAll = .{ .hooks = r0.cacheHooks(1 << 30) };
    defer store.saved.deinit(gpa);
    var long: [300]u32 = undefined;
    for (&long, 0..) |*x, i| x.* = @intCast((i * 7 + 3) % 41);
    // the same pass on both ranks; rank 0's store keeps only the mark at 250 (rank 1 made both)
    var a0 = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = long[0..280], .max_new = 4 });
    defer a0.deinit(gpa);
    const Only250 = struct {
        fn hook(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            if (at == 250) KeepAll.hook(ptr, s, at);
        }
    };
    a0.reuse = .{ .marks = &.{ 100, 250 }, .hook = .{ .ptr = &store, .at = Only250.hook } };
    var a1 = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = long[0..280], .max_new = 4 });
    defer a1.deinit(gpa);
    a1.reuse = .{ .marks = &.{ 100, 250 } };
    try r0.backend().prefill(&a0);
    try r1.backend().prefill(&a1);
    try std.testing.expectEqual(@as(usize, 1), r0.kept.items.len);
    try std.testing.expectEqual(@as(usize, 2), r1.tentative.items.len);
    try r1.applyNote(sink.notes.items[0..sink.lens.items[0]]);
    try std.testing.expectEqual(@as(usize, 1), r1.kept.items.len);
    const key = r0.kept.items[0].key;
    try std.testing.expect(r1.savedFor(key) != null);
    r0.backend().release(&a0);
    r1.backend().release(&a1);
    // the next turn resumes on both ranks
    var b0 = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &long, .max_new = 4 });
    defer b0.deinit(gpa);
    b0.reuse = .{ .saved = store.saved.items[0], .at = 250 };
    var b1 = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &long, .max_new = 4 });
    defer b1.deinit(gpa);
    b1.reuse = .{ .saved = r1.savedFor(key), .at = 250 };
    try r0.backend().prefill(&b0);
    try r1.backend().prefill(&b1);
    try std.testing.expectEqual(@as(u32, 250), b0.cached);
    try std.testing.expectEqual(@as(u32, 250), b1.cached);
    r0.backend().release(&b0);
    r1.backend().release(&b1);
    // rank 0's store drops it: the note frees rank 1's
    r0.cacheHooks(0).drop(&r0, store.saved.items[0]);
    try std.testing.expectEqual(@as(usize, 2), sink.lens.items.len); // the pass note, then the drop
    try r1.applyNote(sink.notes.items[sink.lens.items[0]..][0..sink.lens.items[1]]);
    try std.testing.expectEqual(@as(usize, 0), r1.kept.items.len);
    try std.testing.expectEqual(@as(usize, 0), t1.snaps);
}

test "prompts filling in slices between rounds equal serial decoding, the prompt cache's marks kept on the way" {
    const gpa = std.testing.allocator;
    var t: Toy = .{ .gpa = gpa };
    var b = Lanes(Toy).init(gpa, &t, true, 1 << 20, max_rows);
    defer b.deinit();
    b.streams = 4;
    b.sliced = true;
    b.fill_layers = 2;
    var costs: [max_rows]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 8.0 + 0.3 * @as(f64, @floatFromInt(w)) };
    b.costs = costs;
    b.cost_count = max_rows;
    b.mtp_ms = 0.4;
    for (b.shared[0..2], [_]usize{ 2, 4 }) |*c, n| c.* = .{ .width = @intCast(n * max_rows), .ms = 9.0 + 0.2 * @as(f64, @floatFromInt(n * max_rows)) };
    b.shared_count = 2;
    var cfg = try lanes.Config.init(gpa, b.facts(), max_rows, max_drafts);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer engine.deinit();
    try std.testing.expect(engine.fills());
    var store: KeepAll = .{ .hooks = b.cacheHooks(1 << 30) };
    defer store.saved.deinit(gpa);
    var long: [300]u32 = undefined;
    for (&long, 0..) |*x, i| x.* = @intCast((i * 11 + 2) % 43);
    const cases = [_]Case{
        .{ .prompt = &long, .max_new = 40 },
        .{ .prompt = &.{ 2, 7, 1, 8, 2, 8 }, .max_new = 30, .sampling = .{ .seed = 9, .temperature = 0.7 } },
        .{ .prompt = &.{ 6, 6, 1 }, .max_new = 25, .drafts = false },
    };
    var streams: [cases.len]lanes.Stream = undefined;
    for (cases, &streams) |c, *s| s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .sampling = c.sampling, .drafts = c.drafts });
    defer for (&streams) |*s| s.deinit(gpa);
    streams[0].reuse = .{ .marks = &.{ 100, 250 }, .hook = .{ .ptr = &store, .at = KeepAll.hook } };
    // the first stream fills alone; the others arrive while it decodes; f212's host steps each fill between rounds
    var filling: [cases.len]bool = .{ true, false, false };
    var fresh: [cases.len]bool = @splat(true);
    var next: usize = 1;
    var steps: usize = 0;
    while (steps < 10_000) : (steps += 1) {
        if (next < streams.len and steps % 3 == 2) {
            filling[next] = true;
            next += 1;
        }
        var any = false;
        for (&streams, &filling, &fresh) |*s, *fl, *fr| if (fl.*) {
            any = true;
            if (try engine.fillStream(s, fr.*)) fl.* = false;
            fr.* = false;
        };
        if (engine.activeCount() > 0) try engine.step() else if (!any and next >= streams.len) break;
    }
    for (cases, &streams) |c, *s| {
        const want = try serialTokens(gpa, c);
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u32, want, s.emitted());
    }
    try std.testing.expectEqual(@as(usize, 2), store.saved.items.len); // both marks kept
    try std.testing.expectEqual(@as(usize, 0), t.fills);
    try std.testing.expectEqual(@as(usize, 0), t.seqs - 2); // the two keepers stay with their states
}

test "a state the store evicts during the pass that kept it is left out of rank 1's note and freed on both ranks" {
    const gpa = std.testing.allocator;
    const Sink = struct {
        notes: std.ArrayList(u8) = .empty,
        lens: std.ArrayList(usize) = .empty,
        fn send(ptr: *anyopaque, bytes: []const u8) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.lens.append(std.testing.allocator, bytes.len) catch unreachable;
            self.notes.appendSlice(std.testing.allocator, bytes) catch unreachable;
        }
    };
    // Store.keep with room for one state: evicts the pass's own earlier mark before saving the next one
    const KeepOne = struct {
        hooks: CacheHooks,
        saved: std.ArrayList(*anyopaque) = .empty,
        fn hook(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const k: *@This() = @ptrCast(@alignCast(ptr));
            if (k.saved.items.len > 0) k.hooks.drop(k.hooks.ptr, k.saved.orderedRemove(0));
            const got = k.hooks.save(k.hooks.ptr, s, at) catch return;
            k.saved.append(std.testing.allocator, got) catch {};
        }
    };
    var sink: Sink = .{};
    defer sink.notes.deinit(gpa);
    defer sink.lens.deinit(gpa);
    var t0: Toy = .{ .gpa = gpa };
    var t1: Toy = .{ .gpa = gpa };
    var r0 = Lanes(Toy).init(gpa, &t0, true, 4096, max_rows);
    defer r0.deinit();
    var r1 = Lanes(Toy).init(gpa, &t1, true, 4096, max_rows);
    defer r1.deinit();
    r0.notes = .{ .ptr = &sink, .send = Sink.send };
    r1.follower = true;
    var store: KeepOne = .{ .hooks = r0.cacheHooks(1 << 20) };
    defer store.saved.deinit(gpa);
    var long: [300]u32 = undefined;
    for (&long, 0..) |*x, i| x.* = @intCast((i * 5 + 1) % 37);
    var a0 = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &long, .max_new = 4 });
    defer a0.deinit(gpa);
    var a1 = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &long, .max_new = 4 });
    defer a1.deinit(gpa);
    // 100 gets a keeper in the pass; 250 is the late one, kept at the next call
    a0.reuse = .{ .marks = &.{ 100, 250 }, .hook = .{ .ptr = &store, .at = KeepOne.hook } };
    a1.reuse = .{ .marks = &.{ 100, 250 } };
    try r0.backend().prefill(&a0);
    try r1.backend().prefill(&a1);
    // one note, the pass's, naming only the surviving state: no drop note for a state rank 1 never adopted
    try std.testing.expectEqual(@as(usize, 1), sink.lens.items.len);
    try std.testing.expectEqual(@as(usize, 5 + 12), sink.lens.items[0]);
    try std.testing.expectEqual(@as(usize, 1), r0.kept.items.len);
    try std.testing.expectEqual(@as(u32, 250), r0.kept.items[0].at);
    try std.testing.expectEqual(@as(usize, 0), r0.tentative.items.len);
    try r1.applyNote(sink.notes.items[0..sink.lens.items[0]]);
    // both ranks hold the same state, the same bytes and the same engine objects
    try std.testing.expectEqual(@as(usize, 1), r1.kept.items.len);
    try std.testing.expect(r1.savedFor(r0.kept.items[0].key) != null);
    try std.testing.expectEqual(r0.kept_bytes, r1.kept_bytes);
    try std.testing.expectEqual(t0.snaps, t1.snaps);
    try std.testing.expectEqual(t0.seqs, t1.seqs);
    r0.backend().release(&a0);
    r1.backend().release(&a1);
    // the store drops the survivor later: the drop note frees rank 1's copy and nothing is left on either rank
    r0.cacheHooks(0).drop(&r0, store.saved.items[0]);
    try std.testing.expectEqual(@as(usize, 2), sink.lens.items.len);
    try r1.applyNote(sink.notes.items[sink.lens.items[0]..][0..sink.lens.items[1]]);
    for ([_]*Lanes(Toy){ &r0, &r1 }) |r| {
        try std.testing.expectEqual(@as(usize, 0), r.kept.items.len);
        try std.testing.expectEqual(@as(u64, 0), r.kept_bytes);
    }
    for ([_]*Toy{ &t0, &t1 }) |t| {
        try std.testing.expectEqual(@as(usize, 0), t.snaps);
        try std.testing.expectEqual(@as(usize, 0), t.seqs);
    }
}
