//! Split KV's exchange on two in-process ranks (tp's host Collective): each rank stores only the comp rows of pages it owns, and every
//! row the attention would read after the exchange equals the replicated pool's, dense (row mode, 3 slots) and union (cap splits).

const std = @import("std");
const testing = std.testing;
const sessions = @import("sessions");
const tp = @import("tp");
const split = @import("split.zig");

const page = 64; // tokens; ratio 2: 32 rows a page
const rb = 40; // a short row keeps the test fast; the code takes any width
const fams = [_]sessions.Family{.{ .name = "comp.2", .ratio = 2, .row_bytes = rb, .split = true }};
const psh = 5;

/// The replicated pool's row t of slot s: what every exchanged row must equal.
fn value(s: usize, t: usize, i: usize) u8 {
    return @truncate(s *% 61 +% t *% 7 +% i *% 3 +% 1);
}

const Ctx = struct { seed: u64 };

fn rank(comm: tp.Collective, me: u32, ctx: *const Ctx) anyerror!void {
    const gpa = testing.allocator;
    var pool = try sessions.Pool.init(gpa, .{ .families = &fams, .page = page }, 64 * page, 2, me);
    defer pool.deinit();
    var host = try sessions.HostPool.init(gpa, &pool);
    defer host.deinit();
    const lens = [_]u64{ 11 * page + 5, 3 * page, 17 * page + 40 };
    var slots: [3]*sessions.Slot = undefined;
    for (&slots, lens) |*s, n| {
        s.* = try pool.newSlot(24 * page);
        try s.*.ensure(n);
    }
    // each rank writes the rows it owns (the other rank's go to the discard page, as kv_store does)
    for (slots, 0..) |s, si| for (s.mapped(), 0..) |pg, k| {
        const v = host.pageOf(0, pool.localPage(pg));
        if (!pool.owns(pg)) continue;
        for (0..page / 2) |r| for (0..rb) |i| {
            v[r * rb + i] = value(si, k * (page / 2) + r, i);
        };
    };
    const pts = 24;
    var tables: [3 * pts]i32 = undefined;
    for (slots, 0..) |s, si| for (0..pts) |k| {
        tables[si * pts + k] = @intCast(s.localTableAt(@intCast(k)));
    };
    var prng = std.Random.DefaultPrng.init(ctx.seed); // the same selections on both ranks
    const rnd = prng.random();
    var hk: split.HostKernels = .{};
    const x: split.Exchange = .{ .comm = comm, .kernels = hk.kernels() };
    const base = @intFromPtr(host.tensors[0].ptr);

    // dense, row mode: 6 rows over 3 slots, 16 entries each, some -1
    const R = 6;
    const K = 16;
    var sel: [R * K]i32 = undefined;
    var rslot: [R]i32 = undefined;
    for (&rslot, 0..) |*s, r| s.* = @intCast(r % 3);
    for (&sel, 0..) |*t, i| {
        const rows: u32 = @intCast(lens[@intCast(rslot[i / K])] / 2);
        t.* = if (rnd.uintLessThan(u32, 8) == 0) -1 else @intCast(rnd.uintLessThan(u32, rows));
    }
    var send: [R * K * rb]u8 = undefined;
    var recv: [2 * R * K * rb]u8 = undefined;
    var tok: [R * K]i32 = undefined;
    try x.dense(.{ .sel = @intFromPtr(&sel), .rows = R, .k = K, .table = @intFromPtr(&tables), .pts = pts, .rslot = @intFromPtr(&rslot), .psh = psh, .base = base, .row_bytes = rb, .world = 2, .send = @intFromPtr(&send), .tok = @intFromPtr(&tok) }, @intFromPtr(&recv), null);
    for (sel, tok, 0..) |t, o, i| {
        if (t < 0) {
            try testing.expectEqual(@as(i32, -1), o);
            continue;
        }
        const row = recv[@as(usize, @intCast(o)) * rb ..][0..rb];
        for (row, 0..) |b, j| try testing.expectEqual(value(@intCast(rslot[i / K]), @intCast(t), j), b);
    }

    // union over a 40-row prefill segment of slot 2, with a cap that splits it into blocks
    const S = 40;
    var useg: [S * K]i32 = undefined;
    for (&useg) |*t| t.* = if (rnd.uintLessThan(u32, 10) == 0) -1 else @intCast(rnd.uintLessThan(u32, @intCast(lens[2] / 2)));
    var table2: [pts]u32 = undefined;
    for (&table2, 0..) |*t, k| t.* = slots[2].localTableAt(@intCast(k));
    var blocks: std.ArrayList(split.Block) = .empty;
    defer {
        for (blocks.items) |*b| b.deinit(gpa);
        blocks.deinit(gpa);
    }
    try split.planUnion(gpa, .{ .sel = &useg, .k = K, .table = &table2, .psh = psh, .world = 2, .rank = me, .row_bytes = rb, .cap = 3 * 120 * rb }, &blocks);
    try testing.expect(blocks.items.len > 1);
    for (blocks.items) |blk| {
        const s2 = try gpa.alloc(u8, blk.m * rb);
        defer gpa.free(s2);
        const r2 = try gpa.alloc(u8, 2 * blk.m * rb);
        defer gpa.free(r2);
        try x.unionBlock(base, rb, @intFromPtr(blk.phys.ptr), blk.m, @intFromPtr(s2.ptr), @intFromPtr(r2.ptr), null);
        for (useg[blk.a * K .. blk.b * K], blk.tokens) |t, o| {
            if (t < 0) continue;
            const row = r2[@as(usize, @intCast(o)) * rb ..][0..rb];
            for (row, 0..) |b, j| try testing.expectEqual(value(2, @intCast(t), j), b);
        }
    }
}

test "split == replicated: dense row-mode and capped union exchanges on two ranks" {
    for ([_]u64{ 1, 2, 3 }) |seed| try tp.host.run(2, &Ctx{ .seed = seed }, rank);
}
