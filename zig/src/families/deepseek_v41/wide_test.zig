//! Decode windows of more than 16 rows (block_wide.zig, TF_DSV41_WIDE_ROWS) on the real emitter, host only: every
//! mHC launch and the router's configuration against a table derived from Python's rules at 8474f31, independently of
//! block_wide.zig:
//!
//! - forward._layers' order: the first block's site (`mhc.site`, collapse 1), an Engram block's `post_only` then site
//!   (collapse 2 with the previous pre), every other attention site and every FFN site a `boundary`, then `final`;
//! - mhc._launch: mhc_cuda / mhc_dec take <= 16 rows, so every site is Triton `_site`; SPLIT = a mixing launch of
//!   <= TF_DSV41_MHC_SPLIT_ROWS (32) rows that does not post in place; the forward's second streams buffer exists
//!   for <= 32 rows (mhc.out_of_place), so a boundary is out of place there and in place above; `final` and
//!   `post_only` never mix; a mixing site's `_finish_k` has COEF, the final's not; DSpark's tap at the boundary
//!   entering each target layer;
//! - router_gemv.config / row_tile: (8 warps, 2 experts a warp) past 16 rows, row tile 16.
//! The window also has to pass rowmode.transform (row mode, 4 slots), as batch.zig serves it.

const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const check = @import("m1_check.zig");
const graphs = @import("graphs.zig");
const rowmode = @import("rowmode.zig");
const dspark_emit = @import("dspark_emit.zig");
const Config = @import("config.zig").Config;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    return w;
}

fn prodOptions(limit: i64) block.Options {
    var o: block.Options = .{ .limit = limit, .r1 = true, .expert_topp = 0.85, .index_budget = 64 << 20, .rows = true, .rope_rows = limit + 2048, .taps = true };
    const pages = @divExact(limit, block.Pool.page);
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
    return o;
}

fn arg(c: calls.Call, name: []const u8) calls.Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    std.debug.print("{s}: no argument {s}\n", .{ c.name, name });
    @panic("argument");
}

fn roleOf(x: calls.Arg) []const u8 {
    return switch (x) {
        .t => |t| t.role.buf,
        else => "",
    };
}

/// One expected mHC site (Python's call and its launch parameters).
const Site = struct { post: bool, collapse: i64, mix: bool, split: bool, tap: bool, in_place: bool };

fn expected(a: std.mem.Allocator, cfg: *const Config, n: i64) ![]Site {
    var out: std.ArrayList(Site) = .empty;
    const alt = n <= 32; // mhc.out_of_place
    const split_ok = n <= 32; // mhc.split_rows
    for (0..cfg.layers) |i| {
        const L: u32 = @intCast(i);
        const engram = std.mem.indexOfScalar(u32, cfg.engram_layers.items(), L) != null;
        const aux = std.mem.indexOfScalar(u32, cfg.dspark_targets.items(), L) != null;
        if (i == 0) {
            try out.append(a, .{ .post = false, .collapse = 1, .mix = true, .split = split_ok, .tap = false, .in_place = false });
        } else if (engram) {
            try out.append(a, .{ .post = true, .collapse = 0, .mix = false, .split = false, .tap = false, .in_place = true });
            try out.append(a, .{ .post = false, .collapse = 2, .mix = true, .split = split_ok, .tap = false, .in_place = false });
        } else {
            try out.append(a, .{ .post = true, .collapse = 2, .mix = true, .split = split_ok and alt, .tap = aux, .in_place = !alt });
        }
        try out.append(a, .{ .post = true, .collapse = 2, .mix = true, .split = split_ok and alt, .tap = false, .in_place = !alt });
    }
    try out.append(a, .{ .post = true, .collapse = 2, .mix = false, .split = false, .tap = false, .in_place = true });
    return out.items;
}

fn expectSites(a: std.mem.Allocator, cfg: *const Config, cs: []const calls.Call, n: i64) !void {
    const want = try expected(a, cfg, n);
    var k: usize = 0;
    var finishes: usize = 0;
    var mixes: usize = 0;
    for (cs, 0..) |c, ci| {
        if (std.mem.startsWith(u8, c.name, "tf_dsv41_mhc_cuda")) {
            std.debug.print("n {d}: mhc_cuda call at {d}\n", .{ n, ci });
            return error.TestUnexpectedResult;
        }
        if (std.mem.eql(u8, c.name, "_finish_k")) {
            finishes += 1;
            // after a mixing site: COEF; after the final: not
            try testing.expect(k > 0);
            try testing.expectEqual(want[k - 1].mix, arg(c, "COEF").b);
            try testing.expectEqual(@as(i64, n), c.grid[0]);
            continue;
        }
        if (!std.mem.eql(u8, c.name, "_site")) continue;
        if (k >= want.len) return error.TestUnexpectedResult;
        const s = want[k];
        k += 1;
        if (s.mix) mixes += 1;
        errdefer std.debug.print("n {d}: site {d} (call {d}) want {any}\n", .{ n, k - 1, ci, s });
        try testing.expectEqual(s.post, arg(c, "POST_ON").b);
        try testing.expectEqual(s.collapse, arg(c, "COLLAPSE").i);
        try testing.expectEqual(s.mix, arg(c, "MIX").b);
        try testing.expectEqual(s.split, arg(c, "SPLIT").b);
        try testing.expectEqual(s.tap, arg(c, "TAP_ON").b);
        try testing.expectEqual(@as(i64, 16), arg(c, "BM").i);
        try testing.expectEqual(n, arg(c, "R").i);
        try testing.expectEqual([3]i64{ std.math.divCeil(i64, n, 16) catch unreachable, 40, if (s.split) 4 else 1 }, c.grid);
        if (s.post) try testing.expectEqual(s.in_place, std.mem.eql(u8, roleOf(arg(c, "X")), roleOf(arg(c, "XOUT"))));
    }
    try testing.expectEqual(want.len, k);
    try testing.expectEqual(mixes + 1, finishes);
}

fn expectRouters(cs: []const calls.Call) !usize {
    var m: usize = 0;
    for (cs) |c| if (std.mem.eql(u8, c.name, "tf_dsv41_router_gemv_v2.route")) {
        // (row tile, warps, experts a warp) after (x, gate, bias, pick, wts, logits, cnt, topk, slots, scale)
        try testing.expectEqual(@as(i64, 16), c.args[10].arg.i);
        try testing.expectEqual(@as(i64, 8), c.args[11].arg.i);
        try testing.expectEqual(@as(i64, 2), c.args[12].arg.i);
        m += 1;
    };
    return m;
}

test "wide rows: decode windows of 17-64 rows take Python's > 16-row mHC and router path, and row mode takes them" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    const o = prodOptions(limit);
    const s: graphs.Settings = .{};
    for ([_]i64{ 17, 20, 24, 32, 33, 48, 64 }) |n| {
        for ([_]u64{ 5000, 300_000, @intCast(limit - 1) }) |last| {
            const ctx = graphs.bucketOf(last, s.bucket, s.grow);
            const start = graphs.emitStart(@intCast(n), ctx, s, @intCast(limit));
            const one = try block.emit(a, &cfg, &w, o, &backbone, n, start, true);
            try expectSites(a, &cfg, one, n);
            try testing.expectEqual(@as(usize, cfg.layers), try expectRouters(one));
            const rows = try rowmode.transform(a, one, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = o.pool.?.pts });
            try testing.expect(rows.len > 0);
        }
    }
}

test "wide rows: 16-row windows keep mhc_cuda and the narrow router" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const o = prodOptions(1 << 20);
    const one = try block.emit(a, &cfg, &w, o, &backbone, 16, 4080, true);
    var cuda: usize = 0;
    for (one) |c| {
        if (std.mem.eql(u8, c.name, "tf_dsv41_mhc_cuda_v1.run")) cuda += 1;
        try testing.expect(!std.mem.eql(u8, c.name, "_finish_k"));
        if (std.mem.eql(u8, c.name, "tf_dsv41_router_gemv_v2.route")) {
            try testing.expectEqual(@as(i64, 16), c.args[10].arg.i);
            try testing.expectEqual(@as(i64, 2), c.args[11].arg.i);
            try testing.expectEqual(@as(i64, 0), c.args[12].arg.i);
        }
    }
    try testing.expect(cuda > 0);
}

test "wide rows: a 4-slot DSpark pass (20 rows) as Python's drafter at pad_slots(4)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    const o = prodOptions(1 << 20);
    const cs = try dspark_emit.emitPassSlots(a, &cfg, &w, o, cfg.dspark_block, 4, 4);
    // Drafter._device_pass: block 0's site (collapse 1, split: <= 32 rows, no post), each block's two boundaries in
    // place over x (boundary(x, x, ...): never split), the final (no mix)
    var sites: std.ArrayList(Site) = .empty;
    for (0..cfg.mtp_layers) |b| {
        if (b == 0) try sites.append(a, .{ .post = false, .collapse = 1, .mix = true, .split = true, .tap = false, .in_place = false }) else try sites.append(a, .{ .post = true, .collapse = 2, .mix = true, .split = false, .tap = false, .in_place = true });
        try sites.append(a, .{ .post = true, .collapse = 2, .mix = true, .split = false, .tap = false, .in_place = true });
    }
    try sites.append(a, .{ .post = true, .collapse = 2, .mix = false, .split = false, .tap = false, .in_place = true });
    var k: usize = 0;
    for (cs) |c| {
        try testing.expect(!std.mem.startsWith(u8, c.name, "tf_dsv41_mhc_cuda"));
        if (!std.mem.eql(u8, c.name, "_site")) continue;
        const s = sites.items[k];
        k += 1;
        try testing.expectEqual(s.post, arg(c, "POST_ON").b);
        try testing.expectEqual(s.collapse, arg(c, "COLLAPSE").i);
        try testing.expectEqual(s.mix, arg(c, "MIX").b);
        try testing.expectEqual(s.split, arg(c, "SPLIT").b);
        try testing.expectEqual(@as(i64, 20), arg(c, "R").i);
    }
    try testing.expectEqual(sites.items.len, k);
    try testing.expectEqual(cfg.mtp_layers, try expectRouters(cs));
}
