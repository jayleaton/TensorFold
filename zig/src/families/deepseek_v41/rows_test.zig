//! Host tests of row mode on the real emitter (`zig build test-dsv41-rows`): block.emit's whole backbone window on the
//! release config and the q28-v2 widths, in pool mode (paged and split), rewritten by rowmode.transform; every CSA2 step
//! must read its rows' positions and slots from the row table (compress.py / index.py / attn_cuda.py's row-mode
//! arguments), the stacked roles must size S slots, and nothing else may change.

const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const buffers = @import("buffers.zig");
const check = @import("m1_check.zig");
const graphs = @import("graphs.zig");
const rowtab = @import("rowtab.zig");
const rowmode = @import("rowmode.zig");
const Config = @import("config.zig").Config;

test {
    _ = rowtab;
    _ = rowmode;
}

const S: u32 = 4;
const rmax: u32 = 16;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    return w;
}

/// One slot's pool at `limit` positions (kv_state.zig's view: every page + null; split: half + null + discard).
fn poolOf(limit: i64, split: bool) block.Pool {
    const pages = @divExact(limit, block.Pool.page);
    return .{ .comp_pages = if (split) @divExact(pages, 2) + 2 else pages + 1, .ik_pages = pages + 1, .pts = pages, .split = split };
}

fn named(c: calls.Call, name: []const u8) ?calls.Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    return null;
}

/// The row table's column `c` as the rewritten calls must pass it.
fn isCol(x: calls.Arg, c: rowtab.Col, R: i64) bool {
    const t = switch (x) {
        .t => |t| t,
        else => return false,
    };
    const role = switch (t.role) {
        .buf => |b| b,
        else => return false,
    };
    return std.mem.eql(u8, role, rowmode.table_role) and t.dt == .i64 and t.shape.len == 1 and t.shape[0] == R and t.offset == rowtab.colOffset(c, rmax);
}

fn roleOf(x: calls.Arg) ?[]const u8 {
    return switch (x) {
        .t => |t| switch (t.role) {
            .buf => |b| b,
            else => null,
        },
        else => null,
    };
}

const Counts = struct { kv_store: usize = 0, pool_norm: usize = 0, index_k: usize = 0, scores: usize = 0, rope: usize = 0, attn: usize = 0, topk: usize = 0, dense: usize = 0 };

fn checkProgram(one: []const calls.Call, rows: []const calls.Call, R: i64, pts: i64) !Counts {
    var n: Counts = .{};
    try testing.expectEqual(one.len, rows.len);
    for (one, rows) |o, c| {
        // the same launch: name, grid, argument count
        try testing.expectEqualStrings(o.name, c.name);
        try testing.expectEqual(o.grid, c.grid);
        try testing.expectEqual(o.args.len, c.args.len);
        const nm = c.name;
        if (c.triton) {
            const writes = std.mem.eql(u8, nm, "_kv_store") or std.mem.eql(u8, nm, "_pool_norm") or std.mem.eql(u8, nm, "_index_k");
            const reads = std.mem.eql(u8, nm, "_scores");
            if (std.mem.eql(u8, nm, "_rope")) {
                try testing.expect(isCol(named(c, "POS").?, .pos, R));
                n.rope += 1;
            }
            if (writes or reads) {
                try testing.expect(isCol(named(c, "POS").?, .pos, R));
                try testing.expect(isCol(named(c, "SL").?, if (writes) .wslot else .rslot, R));
                try testing.expect(named(c, "ROWS").?.b);
                // PTS where the kernel pages (compress._row_paging): the stacked tables' row stride with a table
                if (named(c, "PTS")) |pv| {
                    const pt = named(c, "PT").?;
                    if (roleOf(pt)) |role| {
                        try testing.expect(rowmode.isTable(role));
                        try testing.expectEqual(pts, pv.i);
                        try testing.expectEqualSlices(i64, &.{ S, pts }, pt.t.shape);
                    } else try testing.expectEqual(@as(i64, 0), pv.i);
                }
            }
            if (std.mem.eql(u8, nm, "_kv_store")) {
                n.kv_store += 1;
                if (named(c, "RATIO").?.i == 0) {
                    // the SWA ring: S rings stacked
                    const v = named(c, "V").?.t;
                    try testing.expect(rowmode.isRing(roleOf(named(c, "V").?).?));
                    try testing.expectEqual(@as(i64, S) * named(o, "V").?.t.shape[0], v.shape[0]);
                }
            }
            if (std.mem.eql(u8, nm, "_pool_norm")) {
                n.pool_norm += 1;
                try testing.expect(isCol(named(c, "PREV").?, .prev, R));
                const split = named(c, "SPLIT").?.b;
                try testing.expectEqual(@as(i64, if (split) S else 0), named(c, "OFF").?.i);
                if (split) try testing.expectEqualSlices(i64, &.{ S, named(o, "CARRY").?.t.shape[1] }, named(c, "CARRY").?.t.shape);
            }
            if (std.mem.eql(u8, nm, "_index_k")) n.index_k += 1;
            if (std.mem.eql(u8, nm, "_scores")) n.scores += 1;
        } else if (std.mem.eql(u8, nm, "tf_dsv41_attn_cuda_v1.attn")) {
            n.attn += 1;
            try testing.expect(isCol(c.args[9].arg, .pos, R));
            try testing.expect(isCol(c.args[10].arg, .rslot, R));
            try testing.expect(rowmode.isRing(roleOf(c.args[5].arg).?));
            try testing.expectEqual(@as(i64, S) * o.args[5].arg.t.shape[0], c.args[5].arg.t.shape[0]);
            const pt_on = roleOf(c.args[11].arg) != null and c.args[11].arg.t.role != .empty;
            try testing.expectEqual(if (pt_on) pts else 0, c.args[21].arg.i);
        } else if (std.mem.eql(u8, nm, "tf_dsv41_attn_cuda_v1.topk")) {
            n.topk += 1;
            try testing.expect(isCol(c.args[17].arg, .pos, R));
        } else if (std.mem.eql(u8, nm, "glue.kx_dense")) {
            n.dense += 1;
            try testing.expectEqualSlices(i64, &.{ S, pts }, c.args[2].arg.t.shape);
        }
        // the one-slot position tensor is read by nobody but the positions glue
        if (!c.glue) for (c.args) |x| if (roleOf(x.arg)) |role| try testing.expect(!std.mem.eql(u8, role, "w.pos"));
    }
    return n;
}

test "row mode: the whole backbone window, paged pool, every CSA2 step on the row table" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    for ([_]bool{ false, true }) |r1| {
        var o: block.Options = .{ .limit = 8192, .r1 = r1, .expert_topp = 0.85 };
        o.pool = poolOf(o.limit, false);
        const pts = o.pool.?.pts;
        for ([_]u32{ 1, 4, 12, 16 }) |R| {
            const s: graphs.Settings = .{};
            const ctx = graphs.bucketOf(5000, s.bucket, s.grow);
            const start = graphs.emitStart(R, ctx, s, @intCast(o.limit));
            const one = try block.emit(a, &cfg, &w, o, &backbone, R, start, true);
            const rows = try rowmode.transform(a, one, R, .{ .slots = S, .rmax = rmax, .pts = pts });
            const n = try checkProgram(one, rows, R, pts);
            // every backbone layer stores its SWA row; every full layer pools, keys and scores; every layer attends
            try testing.expect(n.kv_store >= 40 and n.attn == 40 and n.pool_norm > 0 and n.pool_norm == n.index_k and n.scores > 0 and n.topk == n.scores);
            try testing.expect(r1 or n.rope >= 40);
            // the indexer scores the context bucket's keys: NK = (bucket end + 1) / ratio
            for (rows) |c| if (std.mem.eql(u8, c.name, "_scores") and !named(c, "GATHER").?.b) {
                const ratio = named(c, "RATIO").?.i;
                const end: i64 = @intCast(graphs.bucketEnd(ctx, s.bucket, @intCast(o.limit), s.grow) + 1);
                try testing.expectEqual(@divFloor(end, ratio), named(c, "NK").?.i);
            };
            // the plan: S rings, S carries, S page tables; every other role as one slot's
            var p1: buffers.Plan = .{ .a = a };
            try p1.add(one);
            var pr: buffers.Plan = .{ .a = a };
            try pr.add(rows);
            var stacked: usize = 0;
            for (p1.sizes.keys(), p1.sizes.values()) |role, v| {
                const got = pr.sizes.get(role).?;
                if (rowmode.isRing(role) or rowmode.isCarry(role) or rowmode.isTable(role)) {
                    try testing.expectEqual(@as(u64, S) * v, got);
                    stacked += 1;
                } else try testing.expectEqual(v, got);
            }
            try testing.expect(stacked > 40);
            try testing.expect(pr.sizes.get(rowmode.table_role).? <= rowtab.tableBytes(rmax));
        }
    }
}

test "row mode: split KV's dense exchange reads the stacked split table; a union is refused" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    var o: block.Options = .{ .limit = 8192 };
    o.pool = poolOf(o.limit, true);
    const pts = o.pool.?.pts;
    const one = try block.emit(a, &cfg, &w, o, &backbone, 8, 2040, true);
    const rows = try rowmode.transform(a, one, 8, .{ .slots = S, .rmax = rmax, .pts = pts });
    const n = try checkProgram(one, rows, 8, pts);
    try testing.expect(n.dense > 0);
    const big = try block.emit(a, &cfg, &w, o, &backbone, 24, 2040, true);
    try testing.expectError(error.UnionInRows, rowmode.transform(a, big, 24, .{ .slots = S, .rmax = 64, .pts = pts }));
    // the contiguous slot has no row mode
    try testing.expectError(error.NoPool, rowmode.transform(a, one, 8, .{ .slots = S, .rmax = rmax, .pts = 0 }));
}
