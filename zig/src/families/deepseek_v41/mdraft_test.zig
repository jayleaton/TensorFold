//! Host tests of DSpark over several slots (`zig build test-dsv41-mdraft`): the real emitter's passes at 1-3 slots
//! against the one-slot pass (the same launches, every Triton launch in the one-slot pass's specialization, the rings
//! stacked), the default emission unchanged, and dspark_rows.zig's statics, grouping, messages and taps placement.

const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const buffers = @import("buffers.zig");
const check = @import("m1_check.zig");
const emit = @import("dspark_emit.zig");
const rows = @import("dspark_rows.zig");
const Config = @import("config.zig").Config;

test {
    _ = rows;
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    // the ingest's main_proj (not in the backbone fixture): a width of the DSpark blocks' own
    try w.dense.put(a, "dspark.main_proj", try w.k2("L40.attn.wkv"));
    return w;
}

const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config = .{},
    w: block.Widths = undefined,
    o: block.Options = .{ .taps = true },

    fn init(f: *Fixture) !void {
        f.arena = std.heap.ArenaAllocator.init(testing.allocator);
        f.w = try widths(f.arena.allocator());
    }
};

fn roleOf(t: calls.Tensor) []const u8 {
    return switch (t.role) {
        .buf => |b| b,
        .weight => |w| w,
        .empty => "",
    };
}

/// An argument's Triton specialization class: a tensor's role and address residue mod 16 (roles are 256-aligned),
/// an int's (== 1, multiple of 16); anything else its tag.
fn class(x: calls.Arg) [3]i64 {
    return switch (x) {
        .t => |t| .{ 1, @mod(t.offset, 16), @intFromEnum(t.dt) },
        .i => |v| .{ 2, @intFromBool(v == 1), @intFromBool(@mod(v, 16) == 0) },
        .f => .{ 3, 0, 0 },
        .b => |v| .{ 4, @intFromBool(v), 0 },
        .none => .{ 5, 0, 0 },
        else => .{ 6, 0, 0 },
    };
}

fn sameCall(a: calls.Call, b: calls.Call) !void {
    try testing.expectEqualStrings(a.name, b.name);
    try testing.expectEqual(a.args.len, b.args.len);
    for (a.args, b.args) |x, y| try testing.expectEqual(std.meta.activeTag(x.arg), std.meta.activeTag(y.arg));
}

fn equalArgs(x: calls.Arg, y: calls.Arg) bool {
    return switch (x) {
        .t, .opaque_table => |t| blk: {
            const u = switch (y) {
                .t, .opaque_table => |u| u,
                else => break :blk false,
            };
            break :blk std.mem.eql(u8, roleOf(t), roleOf(u)) and std.meta.activeTag(t.role) == std.meta.activeTag(u.role) and t.dt == u.dt and t.offset == u.offset and std.mem.eql(i64, t.shape, u.shape) and std.mem.eql(i64, t.stride, u.stride);
        },
        .list => |l| y == .list and l.len == y.list.len and for (l, y.list) |p, q| {
            if (!equalArgs(p, q)) break false;
        } else true,
        else => std.meta.eql(x, y),
    };
}

fn identical(a: []const calls.Call, b: []const calls.Call) !void {
    try testing.expectEqual(a.len, b.len);
    for (a, b) |x, y| {
        try sameCall(x, y);
        try testing.expectEqual(x.triton, y.triton);
        try testing.expectEqual(x.glue, y.glue);
        try testing.expectEqual(x.begin, y.begin);
        try testing.expectEqual(x.grid, y.grid);
        for (x.args, y.args) |p, q| {
            try testing.expectEqualStrings(p.name, q.name);
            try testing.expect(equalArgs(p.arg, q.arg));
        }
    }
}

test "one slot: Layout.of(1) is the one-slot statics, and the default emission is the one-slot pass / ingest" {
    var f: Fixture = .{ .arena = undefined };
    try f.init();
    defer f.arena.deinit();
    const a = f.arena.allocator();
    const n: i64 = f.cfg.dspark_block;
    const one: emit.Statics = .{ .n = n, .window = f.cfg.window };
    const l = emit.Layout.of(1, n, f.cfg.window);
    const lay: emit.Statics = .{ .n = n, .window = f.cfg.window, .lay = l };
    inline for (.{ "ids", "positions", "readRows", "writeRows", "len64", "tokens", "counts", "lo", "hi", "len32" }) |m|
        try testing.expectEqual(@field(emit.Statics, m)(one), @field(emit.Statics, m)(lay));
    try identical(try emit.emitPass(a, &f.cfg, &f.w, f.o, n), try emit.emitPassSlots(a, &f.cfg, &f.w, f.o, n, 1, 1));
    try identical(try emit.emitIngest(a, &f.cfg, &f.w, f.o, 7, 3), try emit.emitIngestAt(a, &f.cfg, &f.w, f.o, 7, 3, .{}));
}

test "several slots: the same launches in the one-slot pass's Triton specializations, rows x slots, rings stacked" {
    var f: Fixture = .{ .arena = undefined };
    try f.init();
    defer f.arena.deinit();
    const a = f.arena.allocator();
    const n: i64 = f.cfg.dspark_block;
    const S: i64 = 4;
    const ring = emit.ringRows(&f.cfg);
    const base = try emit.emitPass(a, &f.cfg, &f.w, f.o, n);
    for (1..rows.maxGroup(f.cfg.dspark_block) + 1) |k| {
        const cs = try emit.emitPassSlots(a, &f.cfg, &f.w, f.o, n, @intCast(k), S);
        try testing.expect(@as(i64, @intCast(k)) * n < 16);
        try testing.expectEqual(base.len, cs.len);
        var rings: usize = 0;
        for (base, cs) |x, y| {
            try sameCall(x, y);
            if (x.triton) for (x.args, y.args) |p, q| {
                // a Triton launch's ints that are rows here are never 1 or a multiple of 16 at k x n rows
                try testing.expectEqual(class(p.arg), class(q.arg));
            };
            for (y.args) |q| if (q.arg == .t) {
                const r = roleOf(q.arg.t);
                if (std.mem.startsWith(u8, r, "s.L") and std.mem.indexOf(u8, r, ".ring.") != null) {
                    rings += 1;
                    try testing.expectEqual(S * ring, q.arg.t.shape[0]);
                }
            };
        }
        try testing.expect(rings > 0);
    }
    // the plan sizes the rings for every slot and the statics for the largest pass
    var p: buffers.Plan = .{ .a = a };
    const g = rows.maxGroup(f.cfg.dspark_block);
    try p.add(try emit.emitPassSlots(a, &f.cfg, &f.w, f.o, n, g, S));
    try testing.expectEqual(@as(u64, @intCast(S * ring * 576)), p.sizes.get("s.L40.ring.v").?);
    const l = emit.Layout.of(g, n, f.cfg.window);
    try testing.expect(p.sizes.get("w.ds.i64").? >= 8 * @as(u64, @intCast(l.len64)));
    try testing.expect(p.sizes.get("w.ds.i32").? >= 4 * @as(u64, @intCast(l.end32)));
}

test "an ingest writes its slot's ring (a view at slot x ring rows) from the chosen taps role" {
    var f: Fixture = .{ .arena = undefined };
    try f.init();
    defer f.arena.deinit();
    const a = f.arena.allocator();
    const ring = emit.ringRows(&f.cfg);
    const base = try emit.emitIngest(a, &f.cfg, &f.w, f.o, 6, 2);
    const cs = try emit.emitIngestAt(a, &f.cfg, &f.w, f.o, 6, 2, .{ .ring_slots = 4, .slot = 3, .taps_role = "s.rows.taps" });
    try testing.expectEqual(base.len, cs.len);
    var stores: usize = 0;
    var taps: usize = 0;
    for (base, cs) |x, y| {
        try sameCall(x, y);
        if (x.triton) for (x.args, y.args) |p, q| try testing.expectEqual(class(p.arg), class(q.arg));
        for (x.args, y.args) |p, q| if (p.arg == .t) {
            const r = roleOf(p.arg.t);
            if (std.mem.indexOf(u8, r, ".ring.v") != null) {
                stores += 1;
                try testing.expectEqual(3 * ring * 576, q.arg.t.offset);
                try testing.expectEqual(ring, q.arg.t.shape[0]);
            } else if (std.mem.indexOf(u8, r, ".ring.s") != null) {
                try testing.expectEqual(3 * ring * 8, q.arg.t.offset);
            } else if (std.mem.eql(u8, r, "w.taps")) {
                taps += 1;
                try testing.expectEqualStrings("s.rows.taps", roleOf(q.arg.t));
                try testing.expectEqual(p.arg.t.offset, q.arg.t.offset);
            }
        };
    }
    try testing.expectEqual(@as(usize, f.cfg.mtp_layers), stores);
    try testing.expect(taps > 0);
}

test "stage: Python's row-mode DraftInputs (ids, positions, ring slots, the context in the stacked ring)" {
    const n: u32 = 5;
    const W: u32 = 128;
    const ring: u32 = 256;
    const lay = emit.Layout.of(2, n, W);
    var h64: [256]i64 = undefined;
    var h32: [4096]i32 = undefined;
    const ms = [_]rows.Member{ .{ .slot = 3, .anchor = 77, .start = 300, .valid = 0 }, .{ .slot = 1, .anchor = 9, .start = 40, .valid = 35 } };
    rows.stage(lay, n, W, ring, 2, &ms, &h64, &h32);
    const at = struct {
        fn i64s(h: []const i64, o: i64, i: usize) i64 {
            return h[@as(usize, @intCast(o)) + i];
        }
        fn i32s(h: []const i32, o: i64, i: usize) i32 {
            return h[@as(usize, @intCast(o)) + i];
        }
    };
    try testing.expectEqual(77, at.i64s(&h64, lay.ids, 0));
    try testing.expectEqual(2, at.i64s(&h64, lay.ids, 1));
    try testing.expectEqual(9, at.i64s(&h64, lay.ids, 5));
    try testing.expectEqual(304, at.i64s(&h64, lay.positions, 4));
    try testing.expectEqual(40, at.i64s(&h64, lay.positions, 5));
    try testing.expectEqual(3, at.i64s(&h64, lay.read, 4));
    try testing.expectEqual(1, at.i64s(&h64, lay.write, 9));
    // slot 3 at 300: positions 172 .. 299 (the window), rows 3 x 256 + p % 256, every row of the slot alike
    try testing.expectEqual(W, @as(u32, @intCast(at.i32s(&h32, lay.counts, 0))));
    try testing.expectEqual(3 * 256 + 172, at.i32s(&h32, lay.tokens, 0));
    try testing.expectEqual(3 * 256 + 299 % 256, at.i32s(&h32, lay.tokens, 4 * W + W - 1));
    // slot 1 at 40 with valid 35: positions 35 .. 39 only
    try testing.expectEqual(5, at.i32s(&h32, lay.counts, 5));
    try testing.expectEqual(256 + 35, at.i32s(&h32, lay.tokens, 5 * W));
    try testing.expectEqual(-1, at.i32s(&h32, lay.tokens, 5 * W + 5));
    try testing.expectEqual(40, at.i32s(&h32, lay.lo, 9));
    try testing.expectEqual(44, at.i32s(&h32, lay.hi, 9));
    // no array overlaps the next one
    try testing.expect(lay.positions >= lay.ids + 10 and lay.read >= lay.positions + 10 and lay.write >= lay.read + 10);
    try testing.expect(lay.counts >= lay.tokens + 10 * W and lay.lo >= lay.counts + 10 and lay.hi >= lay.lo + 10);
}

test "groups, messages, taps placement, the gather's order" {
    var gs: [16][2]usize = undefined;
    try testing.expectEqual(@as(usize, 2), rows.groups(4, 3, &gs));
    try testing.expectEqualSlices(usize, &.{ 0, 2 }, &gs[0]);
    try testing.expectEqualSlices(usize, &.{ 2, 4 }, &gs[1]);
    try testing.expectEqual(@as(usize, 1), rows.groups(3, 3, &gs));
    try testing.expectEqual(@as(usize, 2), rows.groups(5, 3, &gs));
    try testing.expectEqualSlices(usize, &.{ 3, 5 }, &gs[1]);
    try testing.expectEqual(@as(usize, 4), rows.groups(4, 1, &gs));
    try testing.expectEqual(@as(u32, 3), rows.maxGroup(5));
    try testing.expectEqual(@as(u32, 1), rows.maxGroup(16));

    const ms = [_]rows.Member{ .{ .slot = 0, .anchor = 5, .start = 10, .valid = 0 }, .{ .slot = 2, .anchor = 6, .start = 70, .valid = 0 } };
    var buf: [50]i64 = undefined;
    const msg = try rows.proposeMsg(&ms, &buf);
    const valid = [_]u64{ 4, 0, 61, 0 };
    var out: [3]rows.Member = undefined;
    const got = try rows.proposeOf(msg, &valid, &out);
    try testing.expectEqual(@as(usize, 2), got.len);
    try testing.expectEqual(@as(u64, 4), got[0].valid);
    try testing.expectEqual(@as(u64, 61), got[1].valid);
    try testing.expectEqual(@as(u32, 6), got[1].anchor);
    try testing.expectError(error.BadPlan, rows.proposeOf(msg, &valid, out[0..1]));

    const taps: u64 = 0x10000;
    const stride: u64 = 2 * 3 * 4096;
    const stash = [2]u64{ 0x900000, 16 * 4 * stride };
    // in "w.taps": the slot's rows from row 4 of the window, rows 1 .. 3 of them
    try testing.expectEqual(rows.TapsAt{ .src = 0, .skip = 5 }, try rows.locate(taps + 4 * stride, 6, stride, 1, 3, 6, taps, stash));
    // stashed: slot 2's block of the stash
    try testing.expectEqual(rows.TapsAt{ .src = 1, .skip = 32 }, try rows.locate(stash[0] + 32 * stride, 4, stride, 0, 4, 6, taps, stash));
    // past the taps (a resumed prompt) or past the planned rows: not on the device
    try testing.expectEqual(@as(i64, -1), (try rows.locate(taps, 6, stride, 0, 7, 16, taps, stash)).src);
    try testing.expectEqual(@as(i64, -1), (try rows.locate(taps, 128, stride, 0, 100, 16, taps, stash)).src);
    try testing.expectError(error.Rows, rows.locate(taps + 3, 6, stride, 0, 1, 6, taps, stash));

    // [W][R][2k] -> rows of [W k]
    const all = [_]f32{ 1, 2, @bitCast(@as(i32, 10)), @bitCast(@as(i32, 11)), 3, 4, @bitCast(@as(i32, 12)), @bitCast(@as(i32, 13)), 5, 6, @bitCast(@as(i32, 20)), @bitCast(@as(i32, 21)), 7, 8, @bitCast(@as(i32, 22)), @bitCast(@as(i32, 23)) };
    var cand: [8]i32 = undefined;
    var cval: [8]f32 = undefined;
    rows.ungather(&all, 2, 2, 2, &cand, &cval);
    try testing.expectEqualSlices(i32, &.{ 10, 11, 20, 21, 12, 13, 22, 23 }, &cand);
    try testing.expectEqualSlices(f32, &.{ 1, 2, 5, 6, 3, 4, 7, 8 }, &cval);
}
