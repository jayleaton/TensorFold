//! Issue #80 on the host: a history rewritten mid-way (a dropped image) through the planner, pool, pages and store.

const std = @import("std");
const sessions = @import("sessions");
const sched = @import("sched.zig");
const sess = @import("sess.zig");
const t = std.testing;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

const fams = [_]sessions.Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = 4 },
    .{ .name = "comp.1", .ratio = 1, .row_bytes = 4 },
};
const page: u32 = 32;
const tag: u32 = 41;
const vbase: i32 = 1 << 24;

/// A fake model's bounded state: a hash of every token so far, as the SWA rings and carries depend on the history.
const Bound = struct {
    state: u64,
    fn nbytes(_: *anyopaque) u64 {
        return 8;
    }
    fn blobs(_: *anyopaque, _: std.mem.Allocator) ![]sessions.disk.Blob {
        return error.Unused;
    }
    fn release(ptr: *anyopaque) void {
        t.allocator.destroy(@as(*Bound, @ptrCast(@alignCast(ptr))));
    }
    fn make(state: u64) !sessions.store.Bounded {
        const b = try t.allocator.create(Bound);
        b.* = .{ .state = state };
        return .{ .ptr = b, .vtable = &.{ .nbytes = nbytes, .blobs = blobs, .release = release } };
    }
};

fn mix(s: u64, id: i32) u64 {
    var h = std.hash.Wyhash.init(s);
    h.update(std.mem.asBytes(&id));
    return h.final();
}

const World = struct {
    pool: sessions.Pool,
    host: sessions.HostPool,
    store: sessions.Store,
    slot: *sessions.Slot,
    /// the prompt snapshots of the last turn
    saves: std.ArrayList(u64) = .empty,
    /// every piece, snapshot and row of the turns: the knob-off trace
    trace: std.hash.Wyhash = .init(0),

    fn init(w: *World) !void {
        w.pool = try sessions.Pool.init(t.allocator, .{ .families = &fams, .page = page }, 64 * 1024, 1, 0);
        w.host = try sessions.HostPool.init(t.allocator, &w.pool);
        w.store = try sessions.Store.init(t.allocator, t.io, &w.pool, w.host.store(), null, .{ .skip_covered = false, .delta = false });
        w.slot = try w.pool.newSlot(4096);
        w.saves = .empty;
        w.trace = .init(0);
    }

    fn deinit(w: *World) void {
        w.saves.deinit(t.allocator);
        w.store.deinit();
        w.host.deinit();
        w.pool.deinit();
    }

    fn row(w: *World, slot: *sessions.Slot, fi: usize, pos: usize) []u8 {
        const f = fams[fi];
        const off = (pos % page) / f.ratio * f.row_bytes;
        return w.host.pageOf(@intCast(fi), slot.tableAt(@intCast(pos / page)))[off..][0..f.row_bytes];
    }

    /// The fake forward over ids[start, end) from `state`: each position's rows, then the state past it.
    fn prefill(w: *World, slot: *sessions.Slot, state: *u64, ids: []const i32, start: usize, end: usize) !void {
        try slot.ensure(end);
        for (start..end) |pos| {
            state.* = mix(state.*, ids[pos]);
            for (fams, 0..) |f, fi| if (pos % f.ratio == 0) {
                const v: u32 = @truncate(state.* +% fi);
                @memcpy(w.row(slot, fi, pos), std.mem.asBytes(&v));
            };
        }
    }

    /// A fresh run of ids[0, n) in its own slot has the same rows and state as `slot` resumed to `state`.
    fn sameAsFresh(w: *World, slot: *sessions.Slot, state: u64, ids: []const i32, n: usize) !bool {
        const fresh = try w.pool.newSlot(4096);
        defer _ = fresh.releaseAll() catch unreachable;
        var s: u64 = 0;
        try w.prefill(fresh, &s, ids, 0, n);
        if (s != state) return false;
        for (fams, 0..) |f, fi| {
            var pos: usize = 0;
            while (pos < n) : (pos += f.ratio) if (!std.mem.eql(u8, w.row(fresh, fi, pos), w.row(slot, fi, pos))) return false;
        }
        return true;
    }

    const Ctx = struct { w: *World, ids: []const i32 };

    fn findFn(ctx: *anyopaque, _: usize) anyerror!?sched.Store.Hit {
        const c: *Ctx = @ptrCast(@alignCast(ctx));
        const h = (try c.w.store.find(tag, c.ids)) orelse return null;
        return .{ .id = h.id, .pos = h.pos, .ram = h.ram };
    }

    fn poolFn(_: *anyopaque, _: usize, _: ?sched.Store.Hit, _: []const u32, _: u32, _: *std.ArrayList(u32)) anyerror!sess.Plan {
        return .{ .need = 0, .want = 0 };
    }

    const Turn = struct { cached: u64, state: u64 };

    /// One turn's prompt ids[0, n) through the planner: admitted (a hit restored), pieces prefilled, snapshots saved.
    fn turn(w: *World, p: *sched.Planner, ids: []const i32, n: usize) !Turn {
        var ctx: Ctx = .{ .w = w, .ids = ids[0..n] };
        const st: sched.Store = .{ .ctx = &ctx, .vtable = &.{ .find = findFn, .pool = poolFn } };
        var round = p.beginAdmit();
        defer round.deinit(t.allocator);
        const adm = (try p.admit(&round, .{ .key = 1, .n = n, .max_new = ids.len - n + 1, .submitted = 0 }, st, 0, null)).admit;
        var state: u64 = 0;
        if (adm.hit) |h| state = @as(*Bound, @ptrCast(@alignCast((try w.store.restoreRam(h.id, w.slot)).ptr))).state;
        w.saves.clearRetainingCapacity();
        var pieces: std.ArrayList(sched.Piece) = .empty;
        defer pieces.deinit(t.allocator);
        var finals: std.ArrayList(usize) = .empty;
        defer finals.deinit(t.allocator);
        while (finals.items.len == 0) {
            try p.plan(null, &pieces, &finals);
            for (pieces.items) |pc| {
                try w.prefill(w.slot, &state, ids, @intCast(pc.start), @intCast(pc.end));
                w.trace.update(std.mem.asBytes(&[_]u64{ pc.start, pc.end, @intFromBool(pc.save) }));
                if (!pc.save) continue;
                _ = try w.store.save(w.slot, try Bound.make(state), ids[0..@intCast(pc.end)], tag, 1);
                try w.saves.append(t.allocator, pc.end);
            }
        }
        p.ended(1);
        const cached = if (adm.hit) |h| h.pos else 0;
        w.trace.update(std.mem.asBytes(&[_]u64{ cached, state }));
        return .{ .cached = cached, .state = state };
    }

    /// The turn's last prompt token and its reply (ids[n-1, len)), its end saved, the slot emptied.
    fn reply(w: *World, state: u64, ids: []const i32, n: usize) !void {
        var s = state;
        try w.prefill(w.slot, &s, ids, n - 1, ids.len);
        _ = try w.store.save(w.slot, try Bound.make(s), ids, tag, 0);
        _ = try w.slot.releaseAll();
    }
};

/// A long agent turn holding a screenshot at [300, 364), then the next turn with that image replaced by a short note.
const Chat = struct {
    a: [1050]i32 = undefined,
    b: [1041]i32 = undefined,
    const a_prompt = 1000;
    const b_prompt = 1021;
    const image = .{ 300, 364 };

    fn init() Chat {
        var c: Chat = .{};
        for (&c.a, 0..) |*x, i| x.* = if (i >= image[0] and i < image[1]) vbase + @as(i32, @intCast(i)) * 7 else @intCast(i % 997 + 1);
        @memcpy(c.b[0..image[0]], c.a[0..image[0]]);
        for (c.b[image[0]..][0..5], 0..) |*x, i| x.* = @intCast(9001 + i);
        const kept = c.a[image[1]..];
        @memcpy(c.b[image[0] + 5 ..][0..kept.len], kept);
        for (c.b[image[0] + 5 + kept.len ..], 0..) |*x, i| x.* = @intCast(5000 + i);
        return c;
    }
};

fn planner(lcp: bool) !sched.Planner {
    if (lcp) {
        try t.expectEqual(@as(c_int, 0), setenv("TF_DSV41_SESSION_LCP_RESUME", "1", 1));
        try t.expectEqual(@as(c_int, 0), setenv("TF_DSV41_SESSION_LCP_EVERY", "272", 1));
    } else {
        _ = unsetenv("TF_DSV41_SESSION_LCP_RESUME");
        _ = unsetenv("TF_DSV41_SESSION_LCP_EVERY");
    }
    defer _ = unsetenv("TF_DSV41_SESSION_LCP_RESUME");
    defer _ = unsetenv("TF_DSV41_SESSION_LCP_EVERY");
    var s = try sched.Settings.fromEnv();
    s.rows = 128;
    s.sessions = true;
    return sched.Planner.init(t.allocator, s);
}

test "issue #80: a turn whose image was dropped resumes from the last checkpoint before it, equal to a fresh prefill" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    var p = try planner(true);
    defer p.deinit();
    const c = Chat.init();
    const first = try w.turn(&p, &c.a, Chat.a_prompt);
    const saved = try t.allocator.dupe(u64, w.saves.items);
    defer t.allocator.free(saved);
    try t.expectEqual(@as(u64, 0), first.cached);
    try w.reply(first.state, &c.a, Chat.a_prompt);
    // base: no entry prefixes the rewritten history, so all 1,021 tokens prefill again
    const second = try w.turn(&p, &c.b, Chat.b_prompt);
    try t.expectEqual(@as(u64, 272), second.cached);
    try t.expectEqualSlices(u64, &.{ 272, 544, 816, 992 }, saved);
    try t.expect(try w.sameAsFresh(w.slot, second.state, &c.b, Chat.b_prompt - 1));
    _ = try w.slot.releaseAll();
}

test "issue #80 negctl: the longer entry cut at the divergence keeps its own bounded state and the rows differ" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    var p = try planner(false);
    defer p.deinit();
    const c = Chat.init();
    const first = try w.turn(&p, &c.a, Chat.a_prompt);
    try w.reply(first.state, &c.a, Chat.a_prompt);
    // the turn's end shares tokens [0, 300) with the rewritten prompt: adopt it, cut its pages there, prefill on
    var longer: [1051]i32 = undefined;
    @memcpy(longer[0..1050], &c.a);
    longer[1050] = 1;
    const h = (try w.store.find(tag, &longer)).?;
    try t.expectEqual(@as(u64, 1050), h.pos);
    var state = @as(*Bound, @ptrCast(@alignCast((try w.store.restoreRam(h.id, w.slot)).ptr))).state;
    _ = try w.slot.truncate(Chat.image[0]);
    try w.prefill(w.slot, &state, &c.b, Chat.image[0], Chat.b_prompt - 1);
    try t.expect(!try w.sameAsFresh(w.slot, state, &c.b, Chat.b_prompt - 1));
    _ = try w.slot.releaseAll();
}

test "issue #80, knob off: no checkpoints, the rewritten turn prefills from 0, and the trace is base's" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    var p = try planner(false);
    defer p.deinit();
    const c = Chat.init();
    const first = try w.turn(&p, &c.a, Chat.a_prompt);
    try t.expectEqualSlices(u64, &.{992}, w.saves.items);
    try w.reply(first.state, &c.a, Chat.a_prompt);
    const second = try w.turn(&p, &c.b, Chat.b_prompt);
    try t.expectEqual(@as(u64, 0), second.cached);
    try t.expect(try w.sameAsFresh(w.slot, second.state, &c.b, Chat.b_prompt - 1));
    for (fams, 0..) |f, fi| {
        var pos: usize = 0;
        while (pos < Chat.b_prompt - 1) : (pos += f.ratio) w.trace.update(w.row(w.slot, fi, pos));
    }
    _ = try w.slot.releaseAll();
    // 7b79ef92d's value: the planner's pieces and snapshots, the resumes and every row of both turns
    try t.expectEqual(@as(u64, 5385444483688252178), w.trace.final());
}
