//! TF_DSV41_IDX_IOTA: the knob, the emitted windows on and off, and the top-k's score independence on a host twin.

const std = @import("std");
const testing = std.testing;
const block = @import("block.zig");
const calls = @import("block_calls.zig");
const check = @import("calls_check.zig");
const pk = @import("serve_knobs.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "TF_DSV41_IDX_IOTA: unset and 0 off, 1 on, anything else refused" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expect(!try pk.idxIota(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_IDX_IOTA", "0" }};
    try testing.expect(!try pk.idxIota(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_IDX_IOTA", " 1 " }};
    try testing.expect(try pk.idxIota(&Fake.get));
    for ([_][]const u8{ "2", "on", "true", "yes", "-1" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_IDX_IOTA", v }};
        try testing.expectError(error.BadKnob, pk.idxIota(&Fake.get));
    }
    try testing.expect(!(block.Options{}).idx_iota);
}

// -- the emitted windows ---------------------------------------------------------------------------------------

fn argKey(a: std.mem.Allocator, out: *std.ArrayList(u8), x: calls.Arg) !void {
    switch (x) {
        .t, .opaque_table => |t| {
            try out.print(a, " {s}:{t}@{d}[", .{ switch (t.role) {
                .weight, .buf => |r| r,
                .empty => "-",
            }, t.dt, t.offset });
            for (t.shape, t.stride) |n, s| try out.print(a, "{d}/{d},", .{ n, s });
            try out.append(a, ']');
        },
        .list => |l| {
            try out.appendSlice(a, " (");
            for (l) |y| try argKey(a, out, y);
            try out.append(a, ')');
        },
        .i => |v| try out.print(a, " {d}", .{v}),
        .f => |v| try out.print(a, " {e}", .{v}),
        .b => |v| try out.print(a, " {any}", .{v}),
        .none => try out.appendSlice(a, " none"),
    }
}

/// A call as compared here: name, grid, begin, stream, every argument (roles, views, scalars); not the fork mark.
fn key(a: std.mem.Allocator, c: calls.Call) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.print(a, "{s} {any} {t} side={any} join={any} dside={any} djoin={any}", .{ c.name, c.grid, c.begin, c.side, c.join, c.defer_side, c.defer_join });
    for (c.args) |x| {
        try out.print(a, " {s}=", .{x.name});
        try argKey(a, &out, x.arg);
    }
    return out.items;
}

fn readsQi(x: calls.Arg) bool {
    return switch (x) {
        .t => |t| t.role == .buf and std.mem.startsWith(u8, t.role.buf, "L.qi"),
        .list => |l| for (l) |y| {
            if (readsQi(y)) break true;
        } else false,
        else => false,
    };
}

/// The calls the knob drops on an eligible layer: the index query (rot_in, linear), its rope and the scores.
fn dropped(c: calls.Call) bool {
    if (scores(c)) return true;
    for (c.args) |x| if (readsQi(x.arg)) return true;
    return false;
}

fn isTopk(c: calls.Call) bool {
    return std.mem.endsWith(u8, c.name, ".topk") and !c.glue;
}

/// The window's calls a layer each (a layer starts at its `begin`; the head stays with the last).
fn layersOf(a: std.mem.Allocator, cs: []const calls.Call) ![]const []const calls.Call {
    var out: std.ArrayList([]const calls.Call) = .empty;
    var from: usize = 0;
    for (cs, 0..) |c, i| if (i > 0 and c.begin == .layer) {
        try out.append(a, cs[from..i]);
        from = i;
    };
    try out.append(a, cs[from..]);
    return out.items;
}

fn count(cs: []const calls.Call, comptime f: fn (calls.Call) bool) usize {
    var k: usize = 0;
    for (cs) |c| k += @intFromBool(f(c));
    return k;
}

fn scores(c: calls.Call) bool {
    return std.mem.startsWith(u8, c.name, "_scores");
}

fn forks(c: calls.Call) bool {
    return c.fork;
}

/// Index layers whose window ending at `end` qualifies (flash1017: ratio 2 at 2 / 8 / 14, ratio 1 from 20 on).
fn expected(end: i64, L: u32) bool {
    const ratio2 = L == 2 or L == 8 or L == 14;
    const ratio1 = L == 20 or L == 24 or L == 28 or L == 32 or L == 36;
    return (ratio2 and @divFloor(end, 2) <= 512) or (ratio1 and end <= 512);
}

test "TF_DSV41_IDX_IOTA: eligible index layers drop query and scores, keep one top-k; the rest is unchanged" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 8192;
    const pages = @divExact(limit, block.Pool.page);
    var served: block.Options = .{ .limit = limit, .r1 = true, .index_bound = true, .expert_topp = 0.85, .branches = true, .mhc_defer = true, .kv_norm_store = true };
    served.pool = .{ .comp_pages = @divExact(pages, 2) + 2, .ik_pages = pages + 1, .pts = pages, .split = true };
    var rows = served;
    rows.rows = true;
    const sets = [_]block.Options{ .{ .limit = limit }, served, rows };
    // window ends: inside both thresholds, at 512 / 513 (ratio 1), 1025 / 1026 (ratio 2), the 2048 bucket's end
    const ends = [_]i64{ 4, 300, 512, 513, 1000, 1025, 1026, 2048, 4096 };
    var hits: usize = 0;
    for (sets) |o| for (ends) |end| for ([_]i64{ 1, 5, 16 }) |n| {
        if (end < n) continue;
        var on = o;
        on.idx_iota = true;
        const off_cs = try block.emit(a, &cfg, &w, o, &backbone, n, end - n, true);
        const on_cs = try block.emit(a, &cfg, &w, on, &backbone, n, end - n, true);
        const off_l = try layersOf(a, off_cs);
        const on_l = try layersOf(a, on_cs);
        try testing.expectEqual(backbone.len, off_l.len);
        try testing.expectEqual(off_l.len, on_l.len);
        // stream marks: a dropped first side call hands its fork to the next side call (the top-k)
        try testing.expectEqual(count(off_cs, forks), count(on_cs, forks));
        for (backbone, off_l, on_l) |L, x, y| {
            const index_layer = cfg.mode(L) == .full or cfg.mode(L) == .reindex;
            // flag off: every index layer still scores (the H chain)
            if (index_layer) try testing.expectEqual(@as(usize, 1), count(x, scores));
            const want = index_layer and expected(end, L);
            var kept: std.ArrayList([]const u8) = .empty;
            for (x) |c| if (!(want and dropped(c))) try kept.append(a, try key(a, c));
            try testing.expectEqual(kept.items.len, y.len);
            for (kept.items, y) |k, c| try testing.expectEqualStrings(k, try key(a, c));
            if (!want) continue;
            hits += 1;
            // the query projection (rot_in + linear), its rope and the scores: gone; the one top-k launch stays
            try testing.expect(x.len - y.len >= 4);
            try testing.expectEqual(@as(usize, 0), count(y, dropped));
            try testing.expectEqual(@as(usize, 1), count(y, isTopk));
        }
    };
    try testing.expect(hits > 0);
}

test "TF_DSV41_IDX_IOTA: a window past the index budget keeps its row blocks (no layer qualifies)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    const cfg: Config = .{};
    var backbone: [20]u32 = undefined; // up to the candidate source: reindex layers would refuse this budget
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const o: block.Options = .{ .index_budget = 1 << 10 };
    var on = o;
    on.idx_iota = true;
    const off_cs = try block.emit(a, &cfg, &w, o, &backbone, 16, 300, true);
    const on_cs = try block.emit(a, &cfg, &w, on, &backbone, 16, 300, true);
    try testing.expectEqual(off_cs.len, on_cs.len);
    for (off_cs, on_cs) |x, y| try testing.expectEqualStrings(try key(a, x), try key(a, y));
}

// -- the top-k's selection on the host (topk_cuda.cu topk_kernel / topk_b.cu topk_b_kernel, one row) ----------

/// topk_cuda.cu image(): 0 marks a missing entry, so a read entry's image is never 0.
fn image(s: f32) u32 {
    var b: i32 = @bitCast(s);
    if (s == 0) b = 0;
    if (std.math.isNan(s)) b = 0x7FC00000;
    if (b < 0) b ^= 0x7FFFFFFF;
    return @as(u32, @bitCast(b)) ^ 0x80000000;
}

fn fmaxf(x: f32, y: f32) f32 {
    if (std.math.isNan(x)) return y;
    if (std.math.isNan(y)) return x;
    return if (x > y) x else y;
}

const Job = struct { s: []const f32, cand: []const i32 = &.{}, nk: usize, k: usize, mode: u2, bs: i64, nvis: i64, bound: bool };

/// One row's selection into `out` [k]: every image above the K-th, ties in scan order, -1 after; the visible count.
fn topkRow(a: std.mem.Allocator, j: Job, out: []i32) !i64 {
    const nk: i64 = @intCast(j.nk);
    const k: i64 = @intCast(j.k);
    const B: i64 = if (!j.bound) nk else switch (j.mode) {
        0 => @min(nk, @max(j.nvis, k)),
        2 => @min(nk, @max(0, @divTrunc(j.nvis + j.bs - 1, j.bs))),
        else => nk,
    };
    const img = try a.alloc(u32, j.nk);
    defer a.free(img);
    for (img, 0..) |*u, ei| {
        const e: i64 = @intCast(ei);
        u.* = 0;
        if (e >= B) continue;
        switch (j.mode) {
            2 => {
                var m: f32 = 0;
                var q: i64 = 0;
                while (q < j.bs) : (q += 1) {
                    const p = e * j.bs + q;
                    const v = if (p < j.nvis) j.s[@intCast(p)] else -std.math.inf(f32);
                    m = if (q > 0) fmaxf(m, v) else v;
                }
                if (e == @divTrunc(j.nvis - 1, j.bs)) m = std.math.inf(f32);
                if (e * j.bs < j.nvis) u.* = image(m);
            },
            3 => if (j.cand[@intCast(@divTrunc(e, j.bs))] >= 0) {
                u.* = image(j.s[ei]);
            },
            else => u.* = image(j.s[ei]),
        }
    }
    var prefix: u32 = 0;
    var pmask: u32 = 0;
    var above: i64 = 0;
    var quota = false;
    for (0..4) |p| {
        const shift: u5 = @intCast(24 - 8 * p);
        var hist: [256]i64 = @splat(0);
        for (img) |v| if (v != 0 and (v & pmask) == prefix) {
            hist[(v >> shift) & 255] += 1;
        };
        var all: i64 = 0;
        for (hist) |h| all += h;
        if (p == 0 and all <= k) break;
        // the highest digit whose suffix reaches K (the kernel's lane scan)
        var dg: usize = 256;
        var ge: i64 = 0;
        var suf: i64 = 0;
        var d: usize = 256;
        while (d > 0) {
            d -= 1;
            suf += hist[d];
            if (above + suf >= k) {
                dg = d;
                ge = suf;
                break;
            }
        }
        try testing.expect(dg < 256);
        const hd = hist[dg];
        above += ge - hd;
        prefix |= @as(u32, @intCast(dg)) << shift;
        pmask |= @as(u32, 255) << shift;
        if (above + hd == k) break;
        if (p == 3) quota = true;
    }
    const need = k - above;
    var eq: i64 = 0;
    var o: usize = 0;
    var vis: i64 = 0;
    for (img, 0..) |v, ei| {
        if (v == 0) continue;
        const mv = v & pmask;
        var s = mv > prefix;
        if (!s and mv == prefix) {
            s = !quota or eq < need;
            eq += 1;
        }
        if (!s) continue;
        const e: i64 = @intCast(ei);
        const pp: i64 = if (j.mode == 3) j.cand[@intCast(@divTrunc(e, j.bs))] * j.bs + @rem(e, j.bs) else e;
        out[o] = @intCast(pp);
        o += 1;
        vis += @intFromBool(pp < j.nvis);
    }
    for (out[o..]) |*x| x.* = -1;
    return vis;
}

/// Ascending 0 .. m - 1, then -1.
fn isIota(out: []const i32, m: i64) bool {
    for (out, 0..) |x, i| if (x != (if (@as(i64, @intCast(i)) < m) @as(i32, @intCast(i)) else -1)) return false;
    return true;
}

fn garbage(r: std.Random, s: []f32) void {
    for (s) |*x| x.* = @bitCast(r.int(u32));
}

test "top-k twin: up to K entries the selection is every entry ascending, whatever the scores; K + 1 is not" {
    const a = testing.allocator;
    const K = 512;
    var prng = std.Random.DefaultPrng.init(0x1d07a);
    const r = prng.random();
    const s = try a.alloc(f32, K + 1);
    defer a.free(s);
    const out = try a.alloc(i32, K);
    defer a.free(out);
    for ([_]bool{ false, true }) |bound| {
        // every program NK up to K at a few visible counts, then every visible count at NK = K (a bucket's rows)
        for (0..K + 1) |nk| for ([_]usize{ 0, 1, nk / 2, nk }) |nv| for (0..2) |_| {
            garbage(r, s);
            const c = try topkRow(a, .{ .s = s, .nk = nk, .k = K, .mode = 0, .bs = 1, .nvis = @intCast(nv), .bound = bound }, out);
            try testing.expect(isIota(out, @intCast(nk)));
            try testing.expectEqual(@as(i64, @intCast(@min(nk, nv))), c);
        };
        for (1..K + 1) |nv| {
            garbage(r, s);
            const c = try topkRow(a, .{ .s = s, .nk = K, .k = K, .mode = 0, .bs = 1, .nvis = @intCast(nv), .bound = bound }, out);
            try testing.expect(isIota(out, K));
            try testing.expectEqual(@as(i64, @intCast(nv)), c);
        }
        // NK = K + 1 with every row visible: the lowest score is left out, so the scores matter
        for (s, 0..) |*x, i| x.* = @floatFromInt(i);
        _ = try topkRow(a, .{ .s = s, .nk = K + 1, .k = K, .mode = 0, .bs = 1, .nvis = K + 1, .bound = bound }, out);
        try testing.expect(!isIota(out, K));
        try testing.expectEqual(@as(i32, 1), out[0]);
    }
}

test "top-k twin: candidate blocks, then reindex over them, are score independent up to K gathered entries" {
    const a = testing.allocator;
    const K = 512;
    const cfg: Config = .{};
    const cb: usize = cfg.candidate_blocks;
    const bs: i64 = cfg.candidate_block_size;
    var prng = std.Random.DefaultPrng.init(0xb10c5);
    const r = prng.random();
    const s = try a.alloc(f32, cb * @as(usize, @intCast(bs)));
    defer a.free(s);
    const cand = try a.alloc(i32, cb);
    defer a.free(cand);
    const out = try a.alloc(i32, K);
    defer a.free(out);
    for ([_]bool{ false, true }) |bound| {
        var nv: i64 = 0;
        while (nv <= K + 1) : (nv += if (nv < 24 or nv > K - 24) 1 else 7) {
            const nkc: usize = @max(K, @as(usize, @intCast(nv))); // the candidate source's NK (ratio 1)
            const nb = std.math.divCeil(usize, nkc, @intCast(bs)) catch unreachable;
            garbage(r, s);
            _ = try topkRow(a, .{ .s = s, .nk = nb, .k = cb, .mode = 2, .bs = bs, .nvis = nv, .bound = bound }, cand);
            const blocks = @divTrunc(nv + bs - 1, bs);
            try testing.expect(isIota(cand, blocks));
            garbage(r, s);
            const c = try topkRow(a, .{ .s = s, .cand = cand, .nk = s.len, .k = K, .mode = 3, .bs = bs, .nvis = nv, .bound = false }, out);
            if (nv <= K) {
                try testing.expect(isIota(out, blocks * bs));
                try testing.expectEqual(nv, c);
            } else {
                // 520 gathered entries past K: increasing scores leave the first ones out
                for (s, 0..) |*x, i| x.* = @floatFromInt(i);
                _ = try topkRow(a, .{ .s = s, .cand = cand, .nk = s.len, .k = K, .mode = 3, .bs = bs, .nvis = nv, .bound = false }, out);
                try testing.expect(!isIota(out, K));
            }
        }
    }
}
