//! Host tests of TF_DSV41_PF_SHX (test-dsv41-knobs): its readers; the prefill shared expert on the side stream.
const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const branches = @import("branches.zig");
const check = @import("m1_check.zig");
const pk = @import("prod_knobs.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "TF_DSV41_PF_SHX: unset / 0 off, 1 on; its x3gm cap 0 or 8-48; its company; anything else refused" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expect(!try pk.pfShx(&Fake.get));
    try testing.expectEqual(@as(u32, 0), try pk.pfShxGmSms(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_PF_SHX", "1" }};
    try testing.expect(try pk.pfShx(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_PF_SHX", "0" }};
    try testing.expect(!try pk.pfShx(&Fake.get));
    for ([_][]const u8{ "2", "on", "-1" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_PF_SHX", v }};
        try testing.expectError(error.BadKnob, pk.pfShx(&Fake.get));
    }
    for ([_][2][]const u8{ .{ "0", "0" }, .{ "8", "8" }, .{ "32", "32" }, .{ "48", "48" } }) |c| {
        Fake.pairs = &.{.{ "TF_DSV41_PF_SHX_GM_SMS", c[0] }};
        try testing.expectEqual(try std.fmt.parseInt(u32, c[1], 10), try pk.pfShxGmSms(&Fake.get));
    }
    for ([_][]const u8{ "7", "49", "all" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_PF_SHX_GM_SMS", v }};
        try testing.expectError(error.BadKnob, pk.pfShxGmSms(&Fake.get));
    }
    try testing.expect(pk.pfShxFits(true, 0, false));
    try testing.expect(!pk.pfShxFits(false, 0, false));
    try testing.expect(!pk.pfShxFits(true, 2, false));
    try testing.expect(!pk.pfShxFits(true, 0, true));
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    return w;
}

fn eqArg(x: calls.Arg, y: calls.Arg) bool {
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
    return switch (x) {
        .t, .opaque_table => |t| blk: {
            const u = if (y == .t) y.t else y.opaque_table;
            if (std.meta.activeTag(t.role) != std.meta.activeTag(u.role) or t.dt != u.dt or t.offset != u.offset) break :blk false;
            const same = switch (t.role) {
                .buf => |b| std.mem.eql(u8, b, u.role.buf),
                .weight => |w| std.mem.eql(u8, w, u.role.weight),
                .empty => true,
            };
            break :blk same and std.mem.eql(i64, t.shape, u.shape) and std.mem.eql(i64, t.stride, u.stride);
        },
        .i => |v| v == y.i,
        .f => |v| v == y.f,
        .b => |v| v == y.b,
        .none => true,
        .list => |l| blk: {
            if (l.len != y.list.len) break :blk false;
            for (l, y.list) |p, q| if (!eqArg(p, q)) break :blk false;
            break :blk true;
        },
    };
}

/// The same launch: name, grid, scope and every argument (the stream marks aside).
fn eqCall(x: calls.Call, y: calls.Call) bool {
    if (!std.mem.eql(u8, x.name, y.name) or x.triton != y.triton or x.glue != y.glue or x.begin != y.begin) return false;
    if (!std.mem.eql(i64, &x.grid, &y.grid) or x.args.len != y.args.len) return false;
    for (x.args, y.args) |p, q| if (!std.mem.eql(u8, p.name, q.name) or !eqArg(p.arg, q.arg)) return false;
    return true;
}

fn names(c: calls.Call, comptime what: []const u8) bool {
    for (c.args) |x| if (x.arg == .t) switch (x.arg.t.role) {
        .buf => |b| if (std.mem.indexOf(u8, b, what) != null) return true,
        .weight => |w| if (std.mem.indexOf(u8, w, what) != null) return true,
        .empty => {},
    };
    return false;
}

/// A shared-expert launch: its projections (rot_in, unpack / GEMM on ".shared.0." words) and the SiLU product.
fn isShared(c: calls.Call) bool {
    return !std.mem.eql(u8, c.name, "glue.moe_sum") and (names(c, ".shared.0.") or names(c, "L.sh."));
}

fn count(cs: []const calls.Call, comptime f: []const u8) usize {
    var n: usize = 0;
    for (cs) |c| n += @intFromBool(@field(c, f));
    return n;
}

/// `on`: `off` with the shared expert on the side (each stream in order), forked before the router, joined at moe_sum
fn sameSplit(a: std.mem.Allocator, off: []const calls.Call, on: []const calls.Call) !void {
    var layers: usize = 0; // MoE calls
    for (off) |c| layers += @intFromBool(std.mem.eql(u8, c.name, "glue.moe_sum"));
    try testing.expect(layers > 0);
    try testing.expectEqual(off.len, on.len);
    try testing.expectEqual(@as(usize, 0), count(off, "side"));
    var main: std.ArrayList(calls.Call) = .empty;
    var side: std.ArrayList(calls.Call) = .empty;
    for (on) |c| try (if (c.side) &side else &main).append(a, c);
    var i: usize = 0;
    var j: usize = 0;
    for (off) |c| if (isShared(c)) {
        try testing.expect(j < side.items.len and eqCall(c, side.items[j]));
        j += 1;
    } else {
        try testing.expect(i < main.items.len and eqCall(c, main.items[i]));
        i += 1;
    };
    try testing.expectEqual(main.items.len, i);
    try testing.expectEqual(side.items.len, j);
    try testing.expect(j >= 4 * layers);
    try testing.expectEqual(count(off, "fork") + layers, count(on, "fork"));
    try testing.expectEqual(count(off, "join") + layers, count(on, "join"));
    var open = false;
    for (on, 0..) |c, k| {
        if (c.fork) {
            try testing.expect(c.side and !open);
            open = true;
        }
        if (c.side) try testing.expect(open);
        if (c.side and !on[k + 1].side) try testing.expectEqualStrings("tf_dsv41_router_gemv_v2.route", on[k + 1].name);
        if (c.join) {
            try testing.expect(open and !c.side);
            try testing.expectEqualStrings("glue.moe_sum", c.name);
            open = false;
        }
        // the main stream inside the fork only reads the MoE input (the router and the experts' rotation)
        if (open and !c.side and names(c, "w.out")) try testing.expect(std.mem.eql(u8, c.name, "tf_dsv41_router_gemv_v2.route") or std.mem.eql(u8, c.name, "tf_dsv41_x3gm_v1.rot"));
    }
    try testing.expect(!open);
    const sh = try branches.shared(a, on);
    try testing.expectEqual(@as(usize, 1), sh.len);
    try testing.expectEqualStrings("w.out", sh[0]);
}

const Mode = struct { n: i64, o: block.Options };

test "TF_DSV41_PF_SHX on the prefill emitter: the shared expert forked onto the side stream before the router, joined at moe_sum; every launch unchanged" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    const fused: pk.PfDense = .{};
    const base: block.Options = .{ .gm_v2 = .one, .branches = true, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048 };
    var big = base;
    big.pf4k = true;
    big.prefill_rows = 4096;
    var big_fused = big;
    big_fused.pfd = &fused;
    for ([_]Mode{ .{ .n = 2048, .o = base }, .{ .n = 512, .o = base }, .{ .n = 4096, .o = big }, .{ .n = 4096, .o = big_fused } }) |md| {
        var on_o = md.o;
        on_o.pf_shx = true;
        const off = try bp.emitPrefill(aa, &cfg, &w, md.o, &layers, md.n, 0, false);
        try sameSplit(aa, off, try bp.emitPrefill(aa, &cfg, &w, on_o, &layers, md.n, 0, false));
        if (md.n != 4096) continue;
        // CED's encoder segment (prod's 4K path): layers 0 .. 20, 20 the decoder's first (its site and compressor)
        var enc: [21]u32 = undefined;
        for (&enc, 0..) |*l, i| l.* = @intCast(i);
        const e_off = try bp.emitReplay(aa, &cfg, &w, md.o, &enc, md.n, 0, .encoder);
        try sameSplit(aa, e_off, try bp.emitReplay(aa, &cfg, &w, on_o, &enc, md.n, 0, .encoder));
    }
    // under 512 rows, no side stream, exchange pieces or TBO: the knob-off program, launch and stream alike
    for ([_]Mode{ .{ .n = 496, .o = base }, .{ .n = 2048, .o = blk: {
        var o = base;
        o.branches = false;
        break :blk o;
    } }, .{ .n = 2048, .o = blk: {
        var o = base;
        o.pf_overlap = 2;
        break :blk o;
    } }, .{ .n = 2048, .o = blk: {
        var o = base;
        o.pf_tbo = true;
        break :blk o;
    } } }) |md| {
        var on_o = md.o;
        on_o.pf_shx = true;
        const off = try bp.emitPrefill(aa, &cfg, &w, md.o, &layers, md.n, 0, false);
        const same = try bp.emitPrefill(aa, &cfg, &w, on_o, &layers, md.n, 0, false);
        try testing.expectEqual(off.len, same.len);
        for (off, same) |x, y| {
            try testing.expect(eqCall(x, y));
            try testing.expect(x.side == y.side and x.fork == y.fork and x.join == y.join and x.mark == y.mark and x.wait == y.wait);
        }
    }
}
