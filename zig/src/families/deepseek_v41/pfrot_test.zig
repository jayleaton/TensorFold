//! Host tests of TF_DSV41_PF_ROT2 / _WOROT (`zig build test-dsv41-knobs`): the readers and the prefill emitter's calls.
const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const pk = @import("prod_knobs.zig");
const pkf = @import("prod_knobs_pf.zig");
const t4 = @import("pf4k_test.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "TF_DSV41_PF_ROT2 / _WOROT: unset / 0 off, 1 on, anything else refused; apply sets both" {
    defer Fake.pairs = &.{};
    inline for (.{ .{ "TF_DSV41_PF_ROT2", pkf.pfRot2 }, .{ "TF_DSV41_PF_WOROT", pkf.pfWorot } }) |r| {
        Fake.pairs = &.{};
        try testing.expect(!try r[1](&Fake.get));
        Fake.pairs = &.{.{ r[0], "0" }};
        try testing.expect(!try r[1](&Fake.get));
        Fake.pairs = &.{.{ r[0], " 1 " }};
        try testing.expect(try r[1](&Fake.get));
        for ([_][]const u8{ "2", "on", "true", "x", "-1" }) |v| {
            Fake.pairs = &.{.{ r[0], v }};
            try testing.expectError(error.BadKnob, r[1](&Fake.get));
        }
    }
    var o: block.Options = .{};
    Fake.pairs = &.{ .{ "TF_DSV41_PF_ROT2", "1" }, .{ "TF_DSV41_PF_WOROT", "1" } };
    try pkf.apply(&o, &Fake.get);
    try testing.expect(o.pf_rot2 and o.pf_worot and !o.mhc_pf_fold);
    Fake.pairs = &.{.{ "TF_DSV41_MHC_PF_FOLD", "1" }};
    try pkf.apply(&o, &Fake.get);
    try testing.expect(!o.pf_rot2 and !o.pf_worot and o.mhc_pf_fold);
}

const rot_in = "tensorfold_exl3_linear_v4.rot_in";
const gemm = "tf_dsv41_pfdense_v1.gemm";
const rot2 = "tf_dsv41_pfrot_v1.rot2";
const gemm_rot = "tf_dsv41_pfrot_v1.gemm_rot";
/// The pairs TF_DSV41_PF_ROT2 covers, by their weights' suffixes
const covered = [_][2][]const u8{ .{ ".attn.wq_a", ".attn.wkv" }, .{ ".attn.comp_wkv", ".attn.comp_wgate" }, .{ ".shared.0.w1", ".shared.0.w3" } };

fn is(c: calls.Call, name: []const u8) bool {
    return std.mem.eql(u8, c.name, name);
}

/// A fused GEMM's weight prefix (its svh's role without ".svh").
fn weightOf(c: calls.Call) []const u8 {
    const r = t4.roleOf(c.args[4].arg);
    return r[0 .. r.len - ".svh".len];
}

/// The listed pair `w0`, `w1` is (its index), or null.
fn pairOf(w0: []const u8, w1: []const u8) ?usize {
    for (covered, 0..) |p, i| if (std.mem.endsWith(u8, w0, p[0]) and std.mem.endsWith(u8, w1, p[1]) and w0.len - p[0].len == w1.len - p[1].len) return i;
    return null;
}

/// The knob-off pattern ROT2 replaces at `off[i]`: rot_in, gemm, rot_in of the same bf16 / fp16 x, gemm, a listed pair.
fn pairAt(off: []const calls.Call, i: usize) ?usize {
    if (i + 3 >= off.len or !is(off[i], rot_in) or !is(off[i + 1], gemm) or !is(off[i + 2], rot_in) or !is(off[i + 3], gemm)) return null;
    const x = off[i].args[0].arg;
    if (!t4.argEql(x, off[i + 2].args[0].arg) or (x.t.dt != .bf16 and x.t.dt != .f16)) return null;
    return pairOf(weightOf(off[i + 1]), weightOf(off[i + 3]));
}

fn expectSame(p: calls.Call, q: calls.Call) !void {
    try testing.expectEqualStrings(p.name, q.name);
    try testing.expectEqual(p.grid, q.grid);
    try testing.expectEqual(p.args.len, q.args.len);
    for (p.args, q.args) |x, y| {
        try testing.expectEqualStrings(x.name, y.name);
        try testing.expect(t4.argEql(x.arg, y.arg));
    }
}

/// `q` is `p` but for its first argument's role (`role`): a GEMM reading another workspace of the same shape.
fn expectSameBut0(p: calls.Call, q: calls.Call, role: []const u8) !void {
    try testing.expectEqual(p.args.len, q.args.len);
    for (p.args[1..], q.args[1..]) |x, y| try testing.expect(t4.argEql(x.arg, y.arg));
    try testing.expectEqualStrings(role, t4.roleOf(q.args[0].arg));
    try testing.expectEqualSlices(i64, p.args[0].arg.t.shape, q.args[0].arg.t.shape);
    try testing.expectEqual(p.args[0].arg.t.dt, q.args[0].arg.t.dt);
}

/// ROT2's program against the knob-off one: each listed pair's four calls as rot2 + its two GEMMs; per pair, the rot2s.
fn checkRot2(off: []const calls.Call, on: []const calls.Call) ![covered.len]usize {
    var seen: [covered.len]usize = @splat(0);
    var i: usize = 0;
    var j: usize = 0;
    while (i < off.len) {
        try testing.expect(j < on.len);
        if (pairAt(off, i)) |p| {
            const r = on[j];
            try testing.expectEqualStrings(rot2, r.name);
            try testing.expectEqual(@as(usize, 5), r.args.len);
            try testing.expect(t4.argEql(off[i].args[0].arg, r.args[0].arg));
            try testing.expect(t4.argEql(off[i].args[1].arg, r.args[1].arg));
            try testing.expect(t4.argEql(off[i + 2].args[1].arg, r.args[2].arg));
            try testing.expect(t4.argEql(off[i].args[2].arg, r.args[3].arg));
            try testing.expectEqualStrings("s.pf.xh2", t4.roleOf(r.args[4].arg));
            try testing.expectEqualSlices(i64, off[i + 2].args[2].arg.t.shape, r.args[4].arg.t.shape);
            try expectSame(off[i + 1], on[j + 1]);
            try expectSameBut0(off[i + 3], on[j + 2], "s.pf.xh2");
            try testing.expect(t4.argEql(r.args[4].arg, on[j + 2].args[0].arg));
            seen[p] += 1;
            i += 4;
            j += 3;
            continue;
        }
        try expectSame(off[i], on[j]);
        i += 1;
        j += 1;
    }
    try testing.expectEqual(on.len, j);
    var n: usize = 0;
    for (seen) |v| n += v;
    try testing.expectEqual(n, t4.count(on, rot2));
    try testing.expectEqual(t4.count(off, rot_in) - 2 * n, t4.count(on, rot_in));
    return seen;
}

/// WOROT's program against the knob-off one: wo_a's GEMMs as gemm_rot (+ suh_b, xhb), wo_b's GEMM on xhb, no rot_in.
fn checkWorot(cfg: *const Config, o: block.Options, off: []const calls.Call, on: []const calls.Call) !usize {
    const groups: i64 = cfg.o_groups / o.world;
    const ol: i64 = cfg.o_lora;
    var rots: usize = 0;
    var i: usize = 0;
    var j: usize = 0;
    while (i < off.len) {
        try testing.expect(j < on.len);
        if (is(on[j], gemm_rot)) {
            try testing.expectEqualStrings(gemm, off[i].name);
            const w = weightOf(off[i]);
            const at = std.mem.lastIndexOf(u8, w, ".wo_a.") orelse return error.TestUnexpectedResult;
            const g = try std.fmt.parseInt(i64, w[at + ".wo_a.".len ..], 10);
            try testing.expectEqual(@as(usize, 13), on[j].args.len);
            for (off[i].args, on[j].args[0..11]) |x, y| try testing.expect(t4.argEql(x.arg, y.arg));
            const suh = on[j].args[11].arg.t;
            const sr = t4.roleOf(on[j].args[11].arg);
            try testing.expect(std.mem.startsWith(u8, sr, w[0 .. at + 1]) and std.mem.eql(u8, sr[at + 1 ..], "wo_b.suh"));
            try testing.expectEqualSlices(i64, &.{ol}, suh.shape);
            try testing.expectEqual(g * ol * 2, suh.offset);
            const xhb = on[j].args[12].arg.t;
            try testing.expectEqualStrings("s.pf.xhb", t4.roleOf(on[j].args[12].arg));
            try testing.expectEqualSlices(i64, &.{ off[i].args[0].arg.t.shape[0], ol }, xhb.shape);
            try testing.expectEqualSlices(i64, &.{ groups * ol, 1 }, xhb.stride);
            try testing.expectEqual(g * ol * 2, xhb.offset);
            rots += 1;
            i += 1;
            j += 1;
            continue;
        }
        if (is(off[i], rot_in) and std.mem.endsWith(u8, t4.roleOf(off[i].args[1].arg), ".attn.wo_b.suh") and !is(on[j], rot_in)) {
            try testing.expectEqualStrings(gemm, off[i + 1].name);
            try expectSameBut0(off[i + 1], on[j], "s.pf.xhb");
            try testing.expectEqual(groups * ol, on[j].args[0].arg.t.shape[1]);
            i += 2;
            j += 1;
            continue;
        }
        try expectSame(off[i], on[j]);
        i += 1;
        j += 1;
    }
    try testing.expectEqual(on.len, j);
    return rots;
}

const Mode = struct { n: i64, o: block.Options, pieces: bool = false };

test "TF_DSV41_PF_ROT2 / _WOROT on the prefill emitter: off the same program, ROT2 one rot2 a pair, WOROT gemm_rot into wo_b" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try t4.widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    const fused: pk.PfDense = .{};
    const base: block.Options = .{ .gm_v2 = .one, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048, .pfd = &fused };
    var ovl = base;
    ovl.branches = true;
    ovl.pf_overlap = 2;
    for ([_]Mode{ .{ .n = 2048, .o = base }, .{ .n = 300, .o = base }, .{ .n = 2048, .o = ovl, .pieces = true } }) |md| {
        const off = try bp.emitPrefill(aa, &cfg, &w, md.o, &layers, md.n, 0, false);
        try testing.expectEqual(@as(usize, 0), t4.count(off, rot2) + t4.count(off, gemm_rot));
        // the knobs without the fused dense path: today's program, call for call
        var nofuse = md.o;
        nofuse.pfd = null;
        var nofuse_on = nofuse;
        nofuse_on.pf_rot2 = true;
        nofuse_on.pf_worot = true;
        const plain = try bp.emitPrefill(aa, &cfg, &w, nofuse, &layers, md.n, 0, false);
        const plain_on = try bp.emitPrefill(aa, &cfg, &w, nofuse_on, &layers, md.n, 0, false);
        try testing.expectEqual(plain.len, plain_on.len);
        for (plain, plain_on) |p, q| try expectSame(p, q);
        // ROT2: every listed pair once, the row-piece shared expert too
        var r_o = md.o;
        r_o.pf_rot2 = true;
        const seen = try checkRot2(off, try bp.emitPrefill(aa, &cfg, &w, r_o, &layers, md.n, 0, false));
        try testing.expectEqual(@as(usize, layers.len), seen[0]);
        try testing.expect(seen[2] > 0);
        var gates: usize = 0;
        for (off) |c| gates += @intFromBool(is(c, gemm) and std.mem.endsWith(u8, weightOf(c), ".attn.comp_wgate"));
        try testing.expectEqual(gates, seen[1]);
        // WOROT: wo_a's groups of every layer, none under the exchange pieces
        var w_o = md.o;
        w_o.pf_worot = true;
        const won = try bp.emitPrefill(aa, &cfg, &w, w_o, &layers, md.n, 0, false);
        const rots = try checkWorot(&cfg, md.o, off, won);
        const groups: usize = cfg.o_groups / md.o.world;
        try testing.expectEqual(if (md.pieces) 0 else layers.len * groups, rots);
        try testing.expectEqual(rots, t4.count(won, gemm_rot));
        // both: the two rewrites side by side
        var b_o = r_o;
        b_o.pf_worot = true;
        const both = try bp.emitPrefill(aa, &cfg, &w, b_o, &layers, md.n, 0, false);
        try testing.expectEqual(t4.count(won, gemm_rot), t4.count(both, gemm_rot));
        var nseen: usize = 0;
        for (seen) |v| nseen += v;
        try testing.expectEqual(nseen, t4.count(both, rot2));
        // a pair's four calls are three; WOROT drops each layer's wo_b rot_in
        try testing.expectEqual(off.len - nseen - (if (md.pieces) 0 else layers.len), both.len);
    }
}
