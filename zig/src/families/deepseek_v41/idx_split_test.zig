//! Host tests of TF_DSV41_IDX_SPLIT: the reader, the cut and its exchange, the split programs on both ranks, the plan.
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const check = @import("calls_check.zig");
const fp = @import("forward_prefill.zig");
const xs = @import("idx_split.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "TF_DSV41_IDX_SPLIT: unset / empty / 0 / off off, auto the threshold, 1 / on every segment, anything else refused" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expectEqual(xs.Mode.off, try xs.mode(&Fake.get));
    for ([_][2][]const u8{ .{ "", "off" }, .{ "0", "off" }, .{ "off", "off" }, .{ " OFF ", "off" }, .{ "auto", "auto" }, .{ "1", "all" }, .{ "on", "all" } }) |c| {
        Fake.pairs = &.{.{ "TF_DSV41_IDX_SPLIT", c[0] }};
        try testing.expectEqual(std.meta.stringToEnum(xs.Mode, c[1]).?, try xs.mode(&Fake.get));
    }
    for ([_][]const u8{ "2", "yes", "-1", "auto1" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_IDX_SPLIT", v }};
        try testing.expectError(error.BadKnob, xs.mode(&Fake.get));
    }
}

test "TF_DSV41_IDX_SPLIT agreement: off adds nothing; auto, 1 and any refused value each carry their own bytes" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expect(xs.agreement(&Fake.get) == null);
    for ([_][]const u8{ "", "0", "off", " OFF " }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_IDX_SPLIT", v }};
        try testing.expect(xs.agreement(&Fake.get) == null);
    }
    var got: [6][32]u8 = undefined;
    for ([_][]const u8{ "auto", "1", "on", "yes", "2", "auto1" }, &got) |v, *g| {
        Fake.pairs = &.{.{ "TF_DSV41_IDX_SPLIT", v }};
        g.* = xs.agreement(&Fake.get).?;
    }
    // 1 and on agree; every refused value agrees with every other (both ranks then refuse at apply)
    try testing.expectEqualSlices(u8, &got[1], &got[2]);
    try testing.expectEqualSlices(u8, &got[3], &got[4]);
    try testing.expectEqualSlices(u8, &got[3], &got[5]);
    // auto, all and refused differ pairwise: a mixed pair fails the bootstrap's agreement on both ranks
    for ([_][2]usize{ .{ 0, 1 }, .{ 0, 3 }, .{ 1, 3 } }) |p| try testing.expect(!std.mem.eql(u8, &got[p[0]], &got[p[1]]));
}

test "idx split: every cut gives each rank whole rows, equal gather counts, and both ranks every row's bytes back" {
    const k = 3;
    var ref: [3 * 4200]i32 = undefined;
    var local: [2][3 * 4200]i32 = undefined;
    var tmp: [2 * 3 * 4200]i32 = undefined;
    for (&ref, 0..) |*v, i| v.* = @intCast(i * 7 + 1);
    var n: i64 = xs.min_rows;
    while (n <= 4200) : (n += 1) {
        const c = xs.Cut.of(n);
        try testing.expect(c.valid() and @mod(c.h0, xs.grid) == 0);
        try testing.expect(2 * c.big() - n <= 2 * xs.grid);
        const un: usize = @intCast(n);
        const big: usize = @intCast(c.big());
        // the glue's checks: the cut, two ranks, contiguous rows, a gather buffer of two `big` parts
        try testing.expectEqual(c, try xs.glueCut(2, n, c.h0, k, k, 2 * big * k));
        try testing.expectError(error.BadGlue, xs.glueCut(2, n, c.h0, k, k, 2 * big * k - 1));
        try testing.expectError(error.BadGlue, xs.glueCut(1, n, c.h0, k, k, 2 * big * k));
        try testing.expectError(error.BadGlue, xs.glueCut(2, n, c.h0, k, k + 1, 2 * big * k));
        try testing.expectError(error.BadGlue, xs.glueCut(2, n, 0, k, k, 2 * big * k));
        // a rank holds its own rows' selection; the rest of its buffer is another layer's (stale) rows
        for (0..2) |r| {
            const mi = c.mine(@intCast(r));
            const m: [2]usize = .{ @intCast(mi[0]), @intCast(mi[1]) };
            for (0..un) |row| for (0..k) |j| {
                const own = row >= m[0] and row < m[1];
                local[r][row * k + j] = if (own) ref[row * k + j] else -7;
            };
            const p = c.plan(@intCast(r));
            const s0: usize = @intCast(p.send);
            try testing.expectEqual(c.big(), p.count);
            try testing.expect(s0 + big <= un and s0 <= m[0] and m[1] <= s0 + big);
            @memcpy(tmp[r * big * k ..][0 .. big * k], local[r][s0 * k ..][0 .. big * k]);
        }
        for (0..2) |r| {
            const p = c.plan(@intCast(r));
            const src: usize, const dst: usize, const rows: usize = .{ @intCast(p.src), @intCast(p.dst), @intCast(p.rows) };
            try testing.expect(src + rows <= 2 * big);
            @memcpy(local[r][dst * k ..][0 .. rows * k], tmp[src * k ..][0 .. rows * k]);
            try testing.expectEqualSlices(i32, ref[0 .. un * k], local[r][0 .. un * k]);
        }
    }
}

fn roleEq(x: calls.Role, y: calls.Role) bool {
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
    return switch (x) {
        .buf => |b| std.mem.eql(u8, b, y.buf),
        .weight => |w| std.mem.eql(u8, w, y.weight),
        .empty => true,
    };
}

fn argEq(x: calls.Arg, y: calls.Arg) bool {
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
    return switch (x) {
        .t, .opaque_table => |t| blk: {
            const u = if (y == .t) y.t else y.opaque_table;
            break :blk roleEq(t.role, u.role) and t.dt == u.dt and t.offset == u.offset and std.mem.eql(i64, t.shape, u.shape) and std.mem.eql(i64, t.stride, u.stride);
        },
        .i => |v| v == y.i,
        .f => |v| @as(u64, @bitCast(v)) == @as(u64, @bitCast(y.f)),
        .b => |v| v == y.b,
        .none => true,
        .list => |l| l.len == y.list.len and for (l, y.list) |p, q| {
            if (!argEq(p, q)) break false;
        } else true,
    };
}

fn same(x: calls.Call, y: calls.Call) bool {
    if (!std.mem.eql(u8, x.name, y.name) or x.triton != y.triton or x.glue != y.glue or x.begin != y.begin) return false;
    if (!std.mem.eql(i64, &x.grid, &y.grid) or x.side != y.side or x.fork != y.fork or x.join != y.join) return false;
    if (x.mark != y.mark or x.wait != y.wait or x.defer_side != y.defer_side or x.defer_join != y.defer_join) return false;
    if (x.args.len != y.args.len) return false;
    for (x.args, y.args) |p, q| if (!std.mem.eql(u8, p.name, q.name) or !argEq(p.arg, q.arg)) return false;
    return true;
}

fn expectSame(x: []const calls.Call, y: []const calls.Call) !void {
    try testing.expectEqual(x.len, y.len);
    for (x, y) |p, q| try testing.expect(same(p, q));
}

/// A call of the selection's row loop (its blocks differ between the ranks); every other call is the replicated one.
fn rowLoop(c: calls.Call) bool {
    const names = [_][]const u8{ "_stream", "_scores", "_keys", "_block_keys", "glue.stream_merge", "glue.block_pos", "glue.topk", "glue.cand_keys", "glue.ix_gather" };
    for (names) |nm| if (std.mem.eql(u8, c.name, nm)) return true;
    return false;
}

fn argOf(c: calls.Call, name: []const u8) ?calls.Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    return null;
}

fn roleName(x: calls.Arg) []const u8 {
    return if (x == .t and x.t.role == .buf) x.t.role.buf else "";
}

/// Rows [first, first + count) of a row-major view of `cols` int32 columns.
fn rowsIn(x: calls.Arg, cols: i64) [2]i64 {
    const a = @divExact(x.t.offset, cols * 4);
    return .{ a, a + x.t.shape[0] };
}

const Fix = struct {
    a: std.mem.Allocator,
    cfg: Config = .{},
    w: block.Widths = .{},

    fn init(a: std.mem.Allocator) !Fix {
        var f: Fix = .{ .a = a };
        try check.parseDense(a, &f.w, @embedFile("fixtures/dense-k2.txt"));
        try check.parseExperts(a, &f.w, @embedFile("fixtures/expert-k2.txt"));
        return f;
    }
};

fn longOpts(n: i64) block.Options {
    var o: block.Options = .{ .gm_v2 = .one, .limit = 1 << 20, .rope_rows = (1 << 20) + 2048, .index_budget = 64 << 20 };
    if (n > 2048) {
        o.pf4k = true;
        o.prefill_rows = 4096;
    }
    return o;
}

/// The rows each rank's layer `L` selects ("w.sel<L>" outputs), marked in `seen` (a row twice: a failure).
fn markRows(cs: []const calls.Call, L: u32, k: i64, seen: []u8, mine: ?[2]i64) !void {
    var nb: [32]u8 = undefined;
    const role = try std.fmt.bufPrint(&nb, "w.sel{d}", .{L});
    for (cs) |c| {
        const is_out = std.mem.eql(u8, c.name, "glue.stream_merge") or std.mem.eql(u8, c.name, "glue.topk");
        if (!is_out or !std.mem.eql(u8, roleName(c.args[1].arg), role)) continue;
        const r = rowsIn(c.args[1].arg, k);
        if (mine) |m| try testing.expect(r[0] >= m[0] and r[1] <= m[1]);
        for (@intCast(r[0])..@intCast(r[1])) |row| {
            try testing.expectEqual(@as(u8, 0), seen[row]);
            seen[row] = 1;
        }
    }
}

/// Each collective of a program in order: its name and the elements a rank sends.
fn collectives(a: std.mem.Allocator, cs: []const calls.Call) ![][2]i64 {
    var out: std.ArrayList([2]i64) = .empty;
    for (cs) |c| {
        if (std.mem.eql(u8, c.name, "glue.ix_gather")) {
            const g = c.args[1].arg.t;
            try out.append(a, .{ 1, g.shape[1] * g.shape[2] });
        } else if (calls.isExchange(c)) {
            var e: i64 = 1;
            for (c.args[0].arg.t.shape) |d| e *= d;
            try out.append(a, .{ 0, e });
        }
    }
    return out.items;
}

test "idx split: off and auto under its threshold emit the replicated program; split, each rank selects its own rows" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const fx = try Fix.init(aa);
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    const k: i64 = fx.cfg.index_topk;
    for ([_]i64{ 4096, 4095, 2048, 2047, 300, 256, 255 }) |n| for ([_]i64{ 0, 8192, 262144 }) |start| {
        const base = longOpts(n);
        const off = try bp.emitReplay(aa, &fx.cfg, &fx.w, base, &enc, n, start, .encoder);
        for ([_]xs.Mode{ .auto, .all }) |md| {
            var o = base;
            o.idx_split = md;
            var r: [2][]const calls.Call = undefined;
            for (0..2) |rank| {
                o.rank = @intCast(rank);
                r[rank] = try bp.emitReplay(aa, &fx.cfg, &fx.w, o, &enc, n, start, .encoder);
            }
            const nvis = @divFloor(start + n, 2); // layers 2 / 8 / 14 (ratio 2) select here
            if (!xs.engages(md, 2, false, n, nvis)) {
                // the replicated program, call for call, on both ranks
                for (r) |cs| try expectSame(off, cs);
                continue;
            }
            for ([_]u32{ 2, 8, 14 }) |L| {
                const seen = try aa.alloc(u8, @intCast(n));
                @memset(seen, 0);
                const role = try std.fmt.allocPrint(aa, "w.sel{d}", .{L});
                var cut: ?xs.Cut = null;
                for (r, 0..) |cs, rank| {
                    for (cs, 0..) |c, i| if (std.mem.eql(u8, c.name, "glue.ix_gather") and std.mem.eql(u8, roleName(c.args[0].arg), role)) {
                        const got: xs.Cut = .{ .n = n, .h0 = c.args[2].arg.i };
                        if (cut) |x| try testing.expectEqual(x.h0, got.h0) else cut = got;
                        // the gather lands before the counts read the whole selection
                        try testing.expectEqualStrings("glue.counts", cs[i + 1].name);
                    };
                    try markRows(cs, L, k, seen, cut.?.mine(@intCast(rank)));
                }
                for (seen) |v| try testing.expectEqual(@as(u8, 1), v);
            }
            // both ranks issue the same collectives, in order, of the same sizes
            const c0 = try collectives(aa, r[0]);
            const c1 = try collectives(aa, r[1]);
            try testing.expectEqual(c0.len, c1.len);
            for (c0, c1) |x, y| try testing.expectEqual(x, y);
            var gathers: usize = 0;
            for (c0) |x| gathers += @intCast(x[0]);
            try testing.expectEqual(@as(usize, 3), gathers);
            // outside the selection's row loop, every call is the replicated program's
            for (r) |cs| {
                var i: usize = 0;
                var j: usize = 0;
                while (true) {
                    while (i < off.len and rowLoop(off[i])) i += 1;
                    while (j < cs.len and rowLoop(cs[j])) j += 1;
                    if (i == off.len or j == cs.len) break;
                    try testing.expect(same(off[i], cs[j]));
                    i += 1;
                    j += 1;
                }
                try testing.expect(i == off.len and j == cs.len);
            }
            // a split block's launch: the replicated launch's constants, its own rows' keys and first position
            for (r) |cs| for (cs, 0..) |c, i| if (std.mem.eql(u8, c.name, "_stream")) {
                const q_row: i64 = @as(i64, fx.cfg.index_heads) * fx.cfg.index_dim * 2;
                const a = @divExact(argOf(c, "QI").?.t.offset, q_row);
                try testing.expectEqual(@max(@divFloor(start + a + c.grid[0], 2), 1), argOf(c, "NK").?.i);
                try testing.expectEqualStrings("glue.block_pos", cs[i - 1].name);
                try testing.expectEqual(start + a, cs[i - 1].args[1].arg.i);
                const ref = for (off) |x| {
                    if (std.mem.eql(u8, x.name, "_stream")) break x;
                } else unreachable;
                for ([_][]const u8{ "H", "D", "BP", "SPLIT", "K", "CAP", "BS", "MODE", "RATIO", "w_stride", "k_stride" }) |nm| try testing.expectEqual(argOf(ref, nm).?.i, argOf(c, nm).?.i);
                for ([_][]const u8{ "WS", "SCALE" }) |nm| try testing.expectEqual(argOf(ref, nm).?.f, argOf(c, nm).?.f);
            };
        }
    };
}

test "idx split: the candidate source and its reindex layers take one cut, so candidates never cross the ranks" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const fx = try Fix.init(aa);
    var layers: [40]u32 = undefined;
    for (&layers, 0..) |*l, i| l.* = @intCast(i);
    const k: i64 = fx.cfg.index_topk;
    const cb: i64 = fx.cfg.candidate_blocks;
    for ([_]i64{ 2048, 1500 }) |n| for ([_]i64{ 4096, 131072 }) |start| {
        var o = longOpts(n);
        o.idx_split = .all;
        var lo: [2][2]i64 = undefined;
        var gathers: [2]usize = .{ 0, 0 };
        for (0..2) |rank| {
            o.rank = @intCast(rank);
            const cs = try bp.emitPrefill(aa, &fx.cfg, &fx.w, o, &layers, n, start, false);
            var h0: ?i64 = null;
            for (cs) |c| if (std.mem.eql(u8, c.name, "glue.ix_gather")) {
                gathers[rank] += 1;
                if (std.mem.eql(u8, roleName(c.args[0].arg), "w.sel20")) h0 = c.args[2].arg.i;
            };
            const m = (xs.Cut{ .n = n, .h0 = h0.? }).mine(@intCast(rank));
            lo[rank] = m;
            // the candidates this rank writes (layer 20) and reads (layers 24-36): its own rows only
            for (cs) |c| {
                const writes = std.mem.eql(u8, c.name, "glue.topk") and std.mem.eql(u8, roleName(c.args[1].arg), "w.cand");
                const reads = std.mem.eql(u8, c.name, "glue.cand_keys");
                if (!writes and !reads) continue;
                const r = rowsIn(if (writes) c.args[1].arg else c.args[0].arg, cb);
                try testing.expect(r[0] >= m[0] and r[1] <= m[1]);
            }
            for ([_]u32{ 20, 24, 28, 32, 36 }) |L| {
                const seen = try aa.alloc(u8, @intCast(n));
                @memset(seen, 0);
                try markRows(cs, L, k, seen, m);
                for (seen[@intCast(m[0])..@intCast(m[1])]) |v| try testing.expectEqual(@as(u8, 1), v);
            }
        }
        try testing.expectEqual(lo[0][1], lo[1][0]);
        // layers 2 / 8 / 14, the candidate source and its four reindex layers
        try testing.expectEqual(@as(usize, 8), gathers[0]);
        try testing.expectEqual(gathers[0], gathers[1]);
    };
}

test "idx split: ranks with unequal index budgets still cut alike, issue the same collectives and own each row once" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const fx = try Fix.init(aa);
    var layers: [40]u32 = undefined;
    for (&layers, 0..) |*l, i| l.* = @intCast(i);
    const k: i64 = fx.cfg.index_topk;
    const n: i64 = 2048;
    for ([_][2]i64{ .{ 32 << 20, 64 << 20 }, .{ 64 << 20, 1 << 20 } }) |budget| for ([_]xs.Mode{ .auto, .all }) |md| {
        var cs: [2][]const calls.Call = undefined;
        for (0..2) |rank| {
            var o = longOpts(n);
            o.idx_split = md;
            o.rank = @intCast(rank);
            o.index_budget = budget[rank];
            cs[rank] = try bp.emitPrefill(aa, &fx.cfg, &fx.w, o, &layers, n, 131072, false);
        }
        const c0 = try collectives(aa, cs[0]);
        const c1 = try collectives(aa, cs[1]);
        try testing.expectEqual(c0.len, c1.len);
        for (c0, c1) |x, y| try testing.expectEqual(x, y);
        // each split layer: one cut (the rows' canonical one) on both ranks, every row selected on exactly one
        var gathers: usize = 0;
        for (cs[0]) |c| if (std.mem.eql(u8, c.name, "glue.ix_gather")) {
            gathers += 1;
            const role = roleName(c.args[0].arg);
            const L = try std.fmt.parseInt(u32, role["w.sel".len..], 10);
            const cut = xs.Cut.of(n);
            try testing.expectEqual(cut.h0, c.args[2].arg.i);
            const seen = try aa.alloc(u8, @intCast(n));
            @memset(seen, 0);
            for (cs, 0..) |r, rank| try markRows(r, L, k, seen, cut.mine(@intCast(rank)));
            for (seen) |v| try testing.expectEqual(@as(u8, 1), v);
        };
        // 1: every eligible layer (2 / 8 / 14, the source, its 4 reindex layers); auto: the equal-budget pair's layers
        var o = longOpts(n);
        o.idx_split = md;
        var want: usize = 0;
        for (try bp.emitPrefill(aa, &fx.cfg, &fx.w, o, &layers, n, 131072, false)) |c| want += @intFromBool(std.mem.eql(u8, c.name, "glue.ix_gather"));
        if (md == .all) try testing.expectEqual(@as(usize, 8), want);
        try testing.expect(want > 0);
        try testing.expectEqual(want, gathers);
    };
}

test "idx split: never off, on one rank or four, in row mode, under 256 rows; auto from 4,096 x 65,536" {
    for ([_]xs.Mode{ .auto, .all }) |m| {
        try testing.expect(xs.engages(m, 2, false, 4096, 1 << 20));
        for ([_]u32{ 1, 4 }) |w| try testing.expect(!xs.engages(m, w, false, 4096, 1 << 20));
        try testing.expect(!xs.engages(m, 2, true, 4096, 1 << 20));
        try testing.expect(!xs.engages(m, 2, false, xs.min_rows - 1, 1 << 20));
    }
    try testing.expect(!xs.engages(.off, 2, false, 4096, 1 << 20));
    // a 4K piece's ratio-2 layers engage from token 131,072 under auto, a 2K piece's from 262,144
    try testing.expect(xs.engages(.auto, 2, false, 4096, 131072 / 2) and !xs.engages(.auto, 2, false, 4096, 131072 / 2 - 1));
    try testing.expect(xs.engages(.auto, 2, false, 2048, 262144 / 2) and !xs.engages(.auto, 2, false, 2048, 262144 / 2 - 1));
    try testing.expect(xs.engages(.all, 2, false, xs.min_rows, 1));
}

/// Every role a program names, planned at least its bytes (the workspace's under the K4 tag).
fn covered(a: std.mem.Allocator, cs: []const calls.Call, p: *const buffers.Plan, wp: *const buffers.Plan) !void {
    var need: buffers.Plan = .{ .a = a };
    try need.add(cs);
    for (need.sizes.keys(), need.sizes.values()) |role, bytes| {
        const have = if (std.mem.indexOf(u8, role, fp.K4.tag) != null) wp.sizes.get(role) orelse 0 else p.sizes.get(role) orelse 0;
        if (have < bytes) std.debug.print("idx split: {s}: need {d}, planned {d}\n", .{ role, bytes, have });
        try testing.expect(have >= bytes);
    }
}

test "idx split: the buffer plan holds every ragged split segment's gather, on both ranks, 2K and 4K" {
    const ws = @import("forward_prefill_k4.zig");
    for ([_]bool{ false, true }) |k4| for ([_]xs.Mode{ .auto, .all }) |md| for (0..2) |rank| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const fx = try Fix.init(a);
        var layers: [25]u32 = undefined;
        for (&layers, 0..) |*l, i| l.* = @intCast(i);
        const o: block.Options = .{ .gm_v2 = .one, .limit = 299008, .rope_rows = 301056, .index_budget = 64 << 20, .taps = true, .idx_split = md, .rank = @intCast(rank) };
        var f = @import("forward.zig").Forward.init(a, &fx.cfg, &fx.w, o, undefined, undefined);
        f.layers = &layers;
        var state: fp.State = .{ .mode = .replay, .k4 = k4 };
        f.prefill_state = &state;
        var p: buffers.Plan = .{ .a = a };
        var wp: buffers.Plan = .{ .a = a };
        try fp.planPrograms(&f, &p, 2048, &layers);
        if (k4) try ws.planWorkspace(&f, &wp, 2048, &layers);
        try testing.expect(p.sizes.contains("L.ix.gather") or wp.sizes.contains("L.k4~ix.gather"));
        const rows: [4]i64 = if (k4) .{ 2049, 3071, 4095, 4096 } else .{ 256, 1025, 2047, 2048 };
        for (rows) |n| for ([_]i64{ 0, 131072, 290816 }) |start| {
            var tmp = std.heap.ArenaAllocator.init(testing.allocator);
            defer tmp.deinit();
            const ta = tmp.allocator();
            const op = if (k4) fp.k4Options(o) else o;
            const plain = try bp.emitReplay(ta, &fx.cfg, &fx.w, op, layers[0..21], n, start, .encoder);
            try covered(ta, if (k4) try bp.ownCalls(ta, plain, fp.K4.tag) else plain, &p, &wp);
        };
    };
}

test "idx split: aot-needs lists the other rank's split passes and the replicated ones beside the served passes" {
    const an = @import("aot_needs.zig");
    var base: an.Served = .{ .o = .{ .prefill_rows = 4096, .pf4k = true }, .slots = 4, .cap = 64, .drafts = false, .slot_drafts = false, .group = 1, .graphs = .{}, .replay = true, .replay_runs = true };
    for (an.promptPasses(base)[2..]) |p| try testing.expect(p == null);
    base.o.idx_split = .auto;
    const ps = an.promptPasses(base);
    for (ps) |p| try testing.expect(p != null);
    for (ps[0..2]) |p| try testing.expect(p.?.replay and p.?.o.rank == 0 and p.?.o.idx_split == .auto);
    for (ps[2..4]) |p| try testing.expect(!p.?.replay and !p.?.replay_runs and p.?.o.rank == 1 and p.?.o.idx_split == .auto);
    for (ps[4..6]) |p| try testing.expect(!p.?.replay and p.?.o.rank == 0 and p.?.o.idx_split == .off);
    for ([_]usize{ 0, 2, 4 }) |i| try testing.expect(ps[i].?.o.pf4k and !ps[i + 1].?.o.pf4k);
    base.o.world = 4;
    for (an.promptPasses(base)[2..]) |p| try testing.expect(p == null);
}
