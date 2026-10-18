//! TF_DSV41_PF_RS host tests (`zig build test-dsv41-knobs`): the reader, the CPU twin on in-process ranks, the emitter.
const std = @import("std");
const testing = std.testing;
const tp = @import("tp");
const calls = @import("calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const check = @import("m1_check.zig");
const pk = @import("prod_knobs.zig");
const rs = @import("pf_rs.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "TF_DSV41_PF_RS: unset / 0 off, 1 on, anything else refused; only TP=2 without exchange pieces or TBO" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expect(!try pk.pfRs(&Fake.get));
    for ([_][2][]const u8{ .{ "0", "0" }, .{ "1", "1" }, .{ " 1 ", "1" } }) |c| {
        Fake.pairs = &.{.{ "TF_DSV41_PF_RS", c[0] }};
        try testing.expectEqual(c[1][0] == '1', try pk.pfRs(&Fake.get));
    }
    for ([_][]const u8{ "2", "on", "-1", "" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_PF_RS", v }};
        try testing.expectError(error.BadKnob, pk.pfRs(&Fake.get));
    }
    try testing.expect(pk.pfRsFits(2, 0, false));
    try testing.expect(!pk.pfRsFits(1, 0, false) and !pk.pfRsFits(4, 0, false));
    try testing.expect(!pk.pfRsFits(2, 4, false) and !pk.pfRsFits(2, 0, true));
}

test "-0.0: one rank's sum read alone keeps its sign; a +0 second term would not (hence world 1, not a zero slot)" {
    try testing.expectEqual(@as(u16, 0x8000), rs.post(&.{0x8000}));
    try testing.expectEqual(@as(u16, 0x0000), rs.post(&.{ 0x8000, 0x0000 }));
    try testing.expectEqual(@as(u16, 0x8000), rs.post(&.{ 0x8000, 0x8000 }));
    try testing.expectEqual(@as(u16, 0x7FFF), rs.post(&.{ 0x7F80, 0xFF80 }));
    try testing.expectEqual(@as(u16, 0x7F80), rs.post(&.{ 0x7F7F, 0x7F7F }));
}

/// zeros, subnormals, the smallest normals, the largest finite, infinities, NaNs (quiet, signed, payload), ones.
const special = [_]u16{ 0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x3F80, 0xBF81 };

/// Rank r's term of element i: the first 16^W elements every combination of the specials, then splitmix bits.
fn term(r: usize, i: usize, W: usize) u16 {
    const all = std.math.pow(usize, special.len, W);
    if (i < all) return special[(i / std.math.pow(usize, special.len, r)) % special.len];
    var z: u64 = (@as(u64, i) << 8 | r) +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return @truncate(z ^ (z >> 31));
}

const Case = struct { n: usize, D: usize };

const Ranks = struct {
    fn go(c: tp.Collective, me: u32, k: Case) !void {
        const W: usize = c.world();
        const a = std.heap.page_allocator;
        const send = try a.alloc(u16, k.n * k.D);
        defer a.free(send);
        const into = try a.alloc(u16, W * k.n * k.D);
        defer a.free(into);
        for (send, 0..) |*x, i| x.* = term(me, i, W);
        @memset(into, 0xEEEE);
        const mine = tp.collective.rowShare(k.n, W, me);
        try rs.exchange(c, rs.Twin{}, @intFromPtr(send.ptr), @intFromPtr(into.ptr), k.n, k.D, W, me, mine, null);
        var terms: [8]u16 = undefined;
        for (0..k.n * k.D) |i| {
            for (0..W) |r| terms[r] = term(r, i, W);
            const want = rs.post(terms[0..W]);
            // the boundary over the sums (world 1) uses what it uses over every rank's partials (world W)
            try testing.expectEqual(want, rs.post(&.{into[i]}));
            try testing.expectEqual(want, into[i]);
        }
        try c.barrier();
    }
};

test "the reduce-scatter + gather on in-process ranks: every element's bits are mhc_pf's post of the all-gather" {
    for ([_]u32{ 2, 3 }) |W| for ([_]Case{ .{ .n = 256, .D = 8 }, .{ .n = 257, .D = 8 }, .{ .n = 255, .D = 16 }, .{ .n = 3, .D = 8 }, .{ .n = 1001, .D = 8 } }) |k| {
        try testing.expect(rs.fits(k.n, W));
        try tp.host.run(W, k, Ranks.go);
    };
    try testing.expect(!rs.fits(2, 2) and !rs.fits(1, 2) and !rs.fits(256, 1));
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    return w;
}

fn roleOf(x: calls.Arg) []const u8 {
    return if (x == .t and x.t.role == .buf) x.t.role.buf else "";
}

fn nameOf(r: calls.Role) []const u8 {
    return switch (r) {
        .buf, .weight => |b| b,
        .empty => "",
    };
}

fn sameArg(x: calls.Arg, y: calls.Arg) bool {
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
    return switch (x) {
        .t => |t| std.meta.activeTag(t.role) == std.meta.activeTag(y.t.role) and std.mem.eql(u8, nameOf(t.role), nameOf(y.t.role)) and t.dt == y.t.dt and t.offset == y.t.offset and std.mem.eql(i64, t.shape, y.t.shape) and std.mem.eql(i64, t.stride, y.t.stride),
        .i => |v| v == y.i,
        .f => |v| v == y.f,
        .b => |v| v == y.b,
        else => true,
    };
}

const site = "tf_dsv41_mhc_pf_v1.run";

test "TF_DSV41_PF_RS on the prefill emitter: a boundary's exchanges become exchange_rs, its mhc_pf reads [1, rows, D]" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 12, 13, 14, 15 }; // 14 an Engram block: the MoE exchange before it feeds a post alone
    const D: i64 = cfg.hidden;
    const base: block.Options = .{ .gm_v2 = .one, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048 };
    var big = base;
    big.pf4k = true;
    big.prefill_rows = 4096;
    var pieces = base;
    pieces.mhc_site_rows = 512;
    for ([_]struct { i64, block.Options, bool }{ .{ 2048, base, false }, .{ 1027, base, true }, .{ 4096, big, true }, .{ 2048, pieces, false }, .{ 100, base, false } }) |md| {
        const n = md[0];
        var on_o = md[1];
        on_o.pf_rs = true;
        const off = try bp.emitPrefill(aa, &cfg, &w, md[1], &layers, n, 0, md[2]);
        const on = try bp.emitPrefill(aa, &cfg, &w, on_o, &layers, n, 0, md[2]);
        try testing.expectEqual(off.len, on.len);
        var into: []const u8 = ""; // the role the last exchange_rs filled, until the boundary that reads it
        var swaps: usize = 0;
        var reads: usize = 0;
        for (off, on) |x, y| {
            try testing.expectEqual(x.args.len, y.args.len);
            var done = false;
            if (std.mem.eql(u8, y.name, "glue.exchange_rs")) {
                try testing.expectEqualStrings("glue.exchange", x.name);
                try testing.expect(into.len == 0);
                into = roleOf(y.args[1].arg);
                swaps += 1;
            } else try testing.expectEqualStrings(x.name, y.name);
            for (x.args, y.args, 0..) |p, q, j| {
                if (into.len > 0 and std.mem.eql(u8, y.name, site) and j == 2 and std.mem.eql(u8, roleOf(q.arg), into)) {
                    // the gathered [2, rows, D] (rank stride n D) becomes the sums [1, rows, D] at the same rows
                    try testing.expectEqual(@as(i64, 2), p.arg.t.shape[0]);
                    try testing.expectEqualSlices(i64, p.arg.t.shape[1..], q.arg.t.shape[1..]);
                    try testing.expectEqual(@as(i64, 1), q.arg.t.shape[0]);
                    try testing.expectEqual(p.arg.t.offset, q.arg.t.offset);
                    reads += 1;
                    done = @divExact(q.arg.t.offset, 2 * D) + q.arg.t.shape[1] == n; // the whole site's or its last piece
                    continue;
                }
                // nothing else reads the role between the exchange and its boundary
                if (into.len > 0 and !std.mem.eql(u8, y.name, "glue.exchange_rs")) try testing.expect(!std.mem.eql(u8, roleOf(q.arg), into));
                try testing.expect(sameArg(p.arg, q.arg));
            }
            if (done) into = "";
        }
        try testing.expect(into.len == 0);
        // 4 attention exchanges and the MoE ones into 13 and 15; the MoE ones into Engram 14 and after 15 stay
        try testing.expectEqual(@as(usize, if (n < 256) 0 else 6), swaps);
        try testing.expect(reads >= swaps);
    }
}
