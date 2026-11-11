//! TF_DSV41_SESSION_LCP_RESUME's parts: the knob, checkpoint kinds, the image rule, the planner and store retention.

const std = @import("std");
const sessions = @import("sessions");
const lcp = @import("lcp.zig");
const sched = @import("sched.zig");
const t = std.testing;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

const fams = [_]sessions.Family{.{ .name = "comp.1", .ratio = 1, .row_bytes = 4 }};

const Bound = struct {
    fn nbytes(_: *anyopaque) u64 {
        return 8;
    }
    fn blobs(_: *anyopaque, gpa: std.mem.Allocator) ![]sessions.disk.Blob {
        const b = try gpa.alloc(sessions.disk.Blob, 1);
        b[0] = .{ .name = try gpa.dupe(u8, "s.ring"), .bytes = try gpa.dupe(u8, "ringdata") };
        return b;
    }
    fn release(_: *anyopaque) void {}
    var one: u8 = 0;
    fn make() sessions.store.Bounded {
        return .{ .ptr = &one, .vtable = &.{ .nbytes = nbytes, .blobs = blobs, .release = release }, .logical_bytes = 8 };
    }
};

fn vid(x: u32) bool {
    return x >= 1 << 24;
}

fn env(name: [*:0]const u8, value: ?[*:0]const u8) !void {
    if (value) |v| try t.expectEqual(@as(c_int, 0), setenv(name, v, 1)) else _ = unsetenv(name);
}

test "LCP resume: the knob is off unless 1, and its spacing sits on the planner's grid" {
    defer env("TF_DSV41_SESSION_LCP_RESUME", null) catch {};
    defer env("TF_DSV41_SESSION_LCP_EVERY", null) catch {};
    try env("TF_DSV41_SESSION_LCP_RESUME", null);
    try env("TF_DSV41_SESSION_LCP_EVERY", null);
    try t.expectEqual(@as(?u64, null), try lcp.fromEnv());
    try env("TF_DSV41_SESSION_LCP_RESUME", "0");
    try t.expectEqual(@as(?u64, null), try lcp.fromEnv());
    try t.expectEqual(@as(u64, 0), (try sched.Settings.fromEnv()).lcp_every);
    try env("TF_DSV41_SESSION_LCP_RESUME", "1");
    try t.expectEqual(@as(?u64, lcp.every_default), try lcp.fromEnv());
    try env("TF_DSV41_SESSION_LCP_EVERY", "272");
    try t.expectEqual(@as(?u64, 272), try lcp.fromEnv());
    try t.expectEqual(@as(u64, 272), (try sched.Settings.fromEnv()).lcp_every);
    for ([_][*:0]const u8{ "100", "0", "x" }) |bad| {
        try env("TF_DSV41_SESSION_LCP_EVERY", bad);
        try t.expectError(error.BadSettings, lcp.fromEnv());
    }
    try env("TF_DSV41_SESSION_LCP_RESUME", "yes");
    try t.expectError(error.BadSettings, lcp.fromEnv());
}

test "LCP resume: a snapshot at a multiple of the spacing is a checkpoint; off, every prompt snapshot keeps its kind" {
    try t.expectEqual(lcp.checkpoint_kind, lcp.kindAt(272, 544, 1));
    try t.expectEqual(@as(u32, 1), lcp.kindAt(272, 992, 1));
    try t.expectEqual(@as(u32, 1), lcp.kindAt(null, 544, 1));
    try t.expectEqual(@as(u32, 1), lcp.kindAt(272, 0, 1));
}

test "LCP resume: the planner snapshots at each checkpoint, then at the replay point; damaged, it starts over" {
    for ([_]u64{ 0, 272 }) |every| {
        var p = sched.Planner.init(t.allocator, .{ .rows = 128, .sessions = true, .lcp_every = every });
        defer p.deinit();
        var round = p.beginAdmit();
        defer round.deinit(t.allocator);
        _ = try p.admit(&round, .{ .key = 1, .n = 1000, .max_new = 8, .submitted = 0 }, null, 0, null);
        var pieces: std.ArrayList(sched.Piece) = .empty;
        defer pieces.deinit(t.allocator);
        var finals: std.ArrayList(usize) = .empty;
        defer finals.deinit(t.allocator);
        var saves: std.ArrayList(u64) = .empty;
        defer saves.deinit(t.allocator);
        var rounds: u32 = 0;
        while (finals.items.len == 0) : (rounds += 1) {
            try p.plan(null, &pieces, &finals);
            for (pieces.items) |pc| if (pc.save) try saves.append(t.allocator, pc.end);
            if (rounds == 1 and every > 0) {
                p.damaged(1);
                p.after(0, 0);
                try t.expectEqual(@as(?u64, 272), p.seq(1).?.save_at);
            }
        }
        const want: []const u64 = if (every == 0) &.{992} else &.{ 272, 544, 816, 992 };
        try t.expectEqualSlices(u64, want, saves.items);
        try t.expectEqual(@as(?u64, null), p.seq(1).?.save_at);
    }
}

test "LCP resume: spacings below the snapshot minimum start at their first multiple at or past it" {
    for ([_]u64{ 16, 32, 48 }) |every| {
        var p = sched.Planner.init(t.allocator, .{ .rows = 128, .sessions = true, .lcp_every = every });
        defer p.deinit();
        var round = p.beginAdmit();
        defer round.deinit(t.allocator);
        _ = try p.admit(&round, .{ .key = 1, .n = 1000, .max_new = 8, .submitted = 0 }, null, 0, null);
        var pieces: std.ArrayList(sched.Piece) = .empty;
        defer pieces.deinit(t.allocator);
        var finals: std.ArrayList(usize) = .empty;
        defer finals.deinit(t.allocator);
        var saves: std.ArrayList(u64) = .empty;
        defer saves.deinit(t.allocator);
        while (finals.items.len == 0) {
            try p.plan(null, &pieces, &finals);
            for (pieces.items) |pc| if (pc.save) try saves.append(t.allocator, pc.end);
        }
        var want: std.ArrayList(u64) = .empty;
        defer want.deinit(t.allocator);
        var c = (p.s.session_min + every - 1) / every * every;
        while (c < 992) : (c += every) try want.append(t.allocator, c);
        try want.append(t.allocator, 992);
        try t.expectEqualSlices(u64, want.items, saves.items);
    }
}

fn findStore(st: *sessions.Store, prompt: []const u32) !?sessions.store.Hit {
    var ids: [1024]i32 = undefined;
    for (ids[0..prompt.len], prompt) |*d, x| d.* = @intCast(x);
    const h = (try st.find(7, ids[0..prompt.len])) orelse return null;
    return if (h.pos >= prompt.len) null else h;
}

const World = struct {
    pool: sessions.Pool,
    host: sessions.HostPool,
    store: sessions.Store,
    slot: *sessions.Slot,
    ids: [512]i32,

    fn init(w: *World, policy: sessions.store.Policy, disk: ?*sessions.Disk) !void {
        w.pool = try sessions.Pool.init(t.allocator, .{ .families = &fams, .page = 16 }, 64 * 1024, 1, 0);
        w.host = try sessions.HostPool.init(t.allocator, &w.pool);
        w.store = try sessions.Store.init(t.allocator, t.io, &w.pool, w.host.store(), disk, policy);
        w.slot = try w.pool.newSlot(2048);
        for (&w.ids, 0..) |*x, i| x.* = @intCast(i + 1);
    }

    fn deinit(w: *World) void {
        w.store.deinit();
        w.host.deinit();
        w.pool.deinit();
    }

    fn save(w: *World, pos: usize, kind: u32) !u32 {
        try w.slot.ensure(pos);
        return (try w.store.save(w.slot, Bound.make(), w.ids[0..pos], 7, kind)).?;
    }

    /// The slot emptied, then entry id back in it (a turn resumes its chat).
    fn resume_(w: *World, id: u32) !void {
        _ = try w.slot.releaseAll();
        _ = try w.store.restoreRam(id, w.slot);
    }
};

test "LCP resume: a hit inside an image's positions moves back to an entry at or before the image" {
    var w: World = undefined;
    try w.init(.{}, null);
    defer w.deinit();
    for (w.ids[300..340], 0..) |*x, i| x.* = @intCast((1 << 24) + i);
    _ = try w.save(256, 0);
    _ = try w.save(320, 0); // a checkpoint inside the image [300, 340)
    var prompt: [400]u32 = undefined;
    for (&prompt, w.ids[0..400]) |*d, x| d.* = @intCast(x);
    prompt[330] = (1 << 24) + 999; // the image changed past the checkpoint
    try t.expectEqual(@as(u64, 320), (try findStore(&w.store, &prompt)).?.pos);
    try t.expectEqual(@as(u64, 256), (try lcp.outsideImages(&prompt, &w.store, findStore, &vid)).?.pos);
    try t.expectEqual(@as(?usize, 300), lcp.imageStart(&prompt, 320, &vid));
    try t.expectEqual(@as(?usize, null), lcp.imageStart(&prompt, 300, &vid)); // at its first position: outside
    try t.expectEqual(@as(?usize, null), lcp.imageStart(&prompt, 256, &vid));
    prompt[0] = 0; // nothing outside the image prefixes it: a miss
    try t.expectEqual(@as(?sessions.store.Hit, null), try lcp.outsideImages(&prompt, &w.store, findStore, &vid));
}

test "LCP resume: checkpoints keep their chat's recency under the RAM budget; without anchors the oldest goes" {
    for ([_]bool{ true, false }) |anchors| {
        var w: World = undefined;
        try w.init(.{ .ram_bytes = 3 * 8, .skip_covered = false, .delta = false, .anchor_kind = if (anchors) lcp.checkpoint_kind else null }, null);
        defer w.deinit();
        const ck = try w.save(64, lcp.checkpoint_kind);
        var id = try w.save(100, 0);
        for ([_]usize{ 150, 200, 250 }) |pos| {
            try w.resume_(id);
            id = try w.save(pos, 0);
        }
        _ = try w.slot.releaseAll();
        const e = w.store.entry(ck);
        try t.expectEqual(anchors, e.live and e.ram and e.pos == 64);
        var prompt: [300]u32 = undefined;
        for (&prompt, w.ids[0..300]) |*d, x| d.* = @intCast(x);
        prompt[70] = 0; // the history rewritten past the checkpoint
        const h = try findStore(&w.store, &prompt);
        if (anchors) try t.expectEqual(@as(u64, 64), h.?.pos) else try t.expectEqual(@as(?sessions.store.Hit, null), h);
    }
}

test "LCP resume: an evicted checkpoint is dropped, never parked; a turn's end still parks" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(t.io, "tier");
    const root = try tmp.dir.realPathFileAlloc(t.io, "tier", t.allocator);
    defer t.allocator.free(root);
    const disk = try sessions.Disk.open(t.allocator, t.io, .{ .root = root, .compat = @splat(7), .min_tokens = 16, .chunk = 1 << 20, .staging = 4, .lanes = 2, .direct = false });
    defer disk.close();
    var w: World = undefined;
    try w.init(.{ .skip_covered = false, .delta = false, .anchor_kind = lcp.checkpoint_kind }, disk);
    defer w.deinit();
    const ck = try w.save(64, lcp.checkpoint_kind);
    const end = try w.save(100, 0);
    _ = try w.slot.releaseAll();
    _ = try w.store.evict(ck);
    try t.expectEqual(@as(u64, 0), w.store.stats.parks);
    try t.expect(!w.store.entry(ck).live);
    _ = try w.store.evict(end);
    try t.expectEqual(@as(u64, 1), w.store.stats.parks);
}
