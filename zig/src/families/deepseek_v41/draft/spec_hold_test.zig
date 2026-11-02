//! TF_DSV41_RUNNER_SPEC_HOLD on the CPU twin: runner fires keep every token, and lose fewer speculative passes.
const std = @import("std");
const lanes = @import("lanes");
const twin = @import("twin.zig");
const ln = @import("lanes.zig");
const spec = @import("spec.zig");

const gpa = std.testing.allocator;
const fires = [_]usize{ 20, 45, 70 };
const result = [_]u32{ 7, 8, 9 };
const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5 };

/// A runner hook holding its stream at the given emitted indices.
const Hook = struct {
    seen: usize = 0,
    fn token(ptr: *anyopaque, _: u32, _: bool, can_hold: bool) bool {
        const h: *Hook = @ptrCast(@alignCast(ptr));
        defer h.seen += 1;
        return can_hold and std.mem.indexOfScalar(usize, &fires, h.seen) != null;
    }
};

const Mode = enum { off, sync, device };

/// Pass counts: the drafter's passes, and the speculation's launches, uses, drops, held slots and fed launches.
const Counts = struct {
    passes: u64 = 0,
    launched: u64 = 0,
    hit: u64 = 0,
    dropped: u64 = 0,
    held: u64 = 0,
    fed: u64 = 0,

    fn of(d: *const twin.Drafter, x: *const ln.Lanes) Counts {
        const st = if (x.spec) |sp| sp.stats else spec.Stats{};
        return .{ .passes = d.passes, .launched = st.launched, .hit = st.hit, .dropped = st.missed(), .held = st.held, .fed = st.fed };
    }

    fn since(a: Counts, b: Counts) Counts {
        return .{ .passes = a.passes - b.passes, .launched = a.launched - b.launched, .hit = a.hit - b.hit, .dropped = a.dropped - b.dropped, .held = a.held - b.held, .fed = a.fed - b.fed };
    }

    fn add(a: *Counts, b: Counts) void {
        a.passes += b.passes;
        a.launched += b.launched;
        a.hit += b.hit;
        a.dropped += b.dropped;
        a.held += b.held;
        a.fed += b.fed;
    }
};

/// The firing reply's tokens (and the peer's), the whole run's counts and the counts over each fire's three rounds.
const Run = struct {
    out: []u32,
    peer: []u32,
    all: Counts,
    near: Counts,

    fn deinit(r: Run) void {
        gpa.free(r.out);
        gpa.free(r.peer);
    }
};

fn run(mode: Mode, hold: bool, peer_on: bool) !Run {
    const slots: u32 = 4;
    const t = try twin.Twin.init(gpa, .{}, slots + 1);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, slots + 1);
    defer d.deinit();
    var c = try @import("lanes_test.zig").servedCosts(gpa);
    defer c.deinit(gpa);
    const pass = if (mode == .device) d.passAsync() else d.pass();
    const x = try ln.Lanes.init(gpa, t.target(), pass, c, .{ .shape = d.shape, .slots = slots, .spec = .{ .on = mode != .off, .hold = hold } });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cfg = try lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    var hook: Hook = .{};
    var s = try lanes.Stream.init(gpa, .{ .id = "fires", .prompt = &p1, .max_new = 100, .drafts = true, .runner = .{ .ptr = &hook, .token = Hook.token } });
    defer s.deinit(gpa);
    var peer = try lanes.Stream.init(gpa, .{ .id = "peer", .prompt = &p2, .max_new = 100, .drafts = true });
    defer peer.deinit(gpa);
    try e.addStream(&s);
    if (peer_on) try e.addStream(&peer);
    var near: Counts = .{};
    var before: Counts = .{};
    var left: u32 = 0; // rounds of the current fire still counted: its trigger, the feed and the round after
    while (e.activeCount() > 0) {
        if (s.awaiting) try std.testing.expect(try lanes.hold.resumeAwaiting(&s, gpa, &result));
        const pre = Counts.of(d, x);
        try e.step();
        if (s.awaiting) {
            before = pre;
            left = 3;
        }
        if (left > 0) {
            left -= 1;
            if (left == 0) near.add(Counts.of(d, x).since(before));
        }
    }
    return .{ .out = try gpa.dupe(u32, s.emitted()), .peer = try gpa.dupe(u32, peer.emitted()), .all = Counts.of(d, x), .near = near };
}

test "spec hold: fires keep every token; off is the old count, on releases held slots and speculates after the feed" {
    // the counts without the knob, as the code before it gives them: a lone firing reply, then beside a peer
    const old = [2][2]Counts{
        .{ .{ .passes = 52, .launched = 46, .hit = 43, .dropped = 3 }, .{ .passes = 11, .launched = 6, .hit = 4, .dropped = 2 } },
        .{ .{ .passes = 53, .launched = 46, .hit = 40, .dropped = 6 }, .{ .passes = 13, .launched = 9, .hit = 5, .dropped = 4 } },
    };
    for ([_]bool{ false, true }, old) |peer_on, want| {
        const ref = try run(.off, false, peer_on);
        defer ref.deinit();
        for ([_]Mode{ .sync, .device }) |m| {
            const off = try run(m, false, peer_on);
            defer off.deinit();
            const on = try run(m, true, peer_on);
            defer on.deinit();
            for ([_]Run{ off, on }) |r| {
                try std.testing.expectEqualSlices(u32, ref.out, r.out);
                try std.testing.expectEqualSlices(u32, ref.peer, r.peer);
            }
            // off (the negative control): the old counts, nothing held or fed
            try std.testing.expectEqual(want[0], off.all);
            try std.testing.expectEqual(want[1], off.near);
            // on: held slots leave unscored, feeds launch, nothing drops; a fire's one serial pass is the held stream's
            try std.testing.expectEqual(@as(u64, fires.len), on.near.held);
            try std.testing.expectEqual(@as(u64, fires.len), on.near.fed);
            try std.testing.expectEqual(@as(u64, 0), on.near.dropped);
            try std.testing.expectEqual(@as(u64, fires.len), on.near.passes - on.near.launched);
            try std.testing.expect(on.all.dropped < off.all.dropped and on.all.hit > off.all.hit);
        }
    }
}
