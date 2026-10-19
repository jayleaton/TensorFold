//! Host tests of TF_DSV41_MHC_PF_FOLD (`zig build test-dsv41-knobs`): the reader and the prefill emitter's fold runs.
const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const pk = @import("prod_knobs.zig");
const t4 = @import("pf4k_test.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "TF_DSV41_MHC_PF_FOLD: unset / 0 off, 1 on, anything else refused" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expect(!try pk.mhcPfFold(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_MHC_PF_FOLD", "0" }};
    try testing.expect(!try pk.mhcPfFold(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_MHC_PF_FOLD", "1" }};
    try testing.expect(try pk.mhcPfFold(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_MHC_PF_FOLD", " 1 " }};
    try testing.expect(try pk.mhcPfFold(&Fake.get));
    for ([_][]const u8{ "2", "on", "true", "x", "-1" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_MHC_PF_FOLD", v }};
        try testing.expectError(error.BadKnob, pk.mhcPfFold(&Fake.get));
    }
}

fn count(cs: []const calls.Call, name: []const u8) usize {
    var n: usize = 0;
    for (cs) |c| n += @intFromBool(std.mem.eql(u8, c.name, name));
    return n;
}

/// The `_finish_k` argument named `name`.
fn named(c: calls.Call, name: []const u8) !calls.Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    return error.TestUnexpectedResult;
}

const pf_run = "tf_dsv41_mhc_pf_v1.run";
const fold_run = "tf_dsv41_mhc_pf_fold_v1.run";
/// `_finish_k`'s arguments in the fold's order after mhc_pf.run's 12 (index 19, the counter, is the fold's own)
const finish_at = [_]struct { []const u8, usize }{
    .{ "BASE", 12 }, .{ "SCALE", 13 }, .{ "PRE", 14 },    .{ "POST", 15 },       .{ "COMB", 16 },  .{ "NW", 17 },
    .{ "OUT", 18 },  .{ "eps", 20 },   .{ "hc_eps", 21 }, .{ "post_alpha", 22 }, .{ "ITERS", 23 },
};

const Mode = struct { n: i64, layers: []const u32, taps: bool };

test "TF_DSV41_MHC_PF_FOLD on the prefill emitter: a whole tapless site one fold run, no `_finish_k`; every other call unchanged" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try t4.widths(aa);
    const modes = [_]Mode{
        .{ .n = 2048, .layers = &.{ 1, 2, 3 }, .taps = false },
        .{ .n = 300, .layers = &.{ 1, 2 }, .taps = false },
        .{ .n = 2048, .layers = &.{ 36, 37, 38 }, .taps = true },
    };
    for (modes) |md| {
        const off_o: block.Options = .{ .gm_v2 = .one, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048, .taps = md.taps };
        var on_o = off_o;
        on_o.mhc_pf_fold = true;
        const off = try bp.emitPrefill(aa, &cfg, &w, off_o, md.layers, md.n, 0, false);
        const on = try bp.emitPrefill(aa, &cfg, &w, on_o, md.layers, md.n, 0, false);
        try testing.expectEqual(@as(usize, 0), count(off, fold_run));
        var i: usize = 0;
        var j: usize = 0;
        var folds: usize = 0;
        var kept: usize = 0;
        while (i < off.len) : ({
            i += 1;
            j += 1;
        }) {
            try testing.expect(j < on.len);
            const site = std.mem.eql(u8, off[i].name, pf_run);
            if (site and off[i].args[9].arg.t.role != .empty) kept += 1;
            if (site and off[i].args[9].arg.t.role == .empty) {
                // the knob-off pair: mhc_pf.run, then its `_finish_k` with the next coefficients
                const fin = off[i + 1];
                try testing.expectEqualStrings("_finish_k", fin.name);
                try testing.expect((try named(fin, "COEF")).b);
                const f = on[j];
                try testing.expectEqualStrings(fold_run, f.name);
                try testing.expectEqual(@as(usize, 24), f.args.len);
                for (0..12) |k| try testing.expect(t4.argEql(off[i].args[k].arg, f.args[k].arg));
                for (finish_at) |p| try testing.expect(t4.argEql(try named(fin, p[0]), f.args[p[1]].arg));
                const cnt = f.args[19].arg.t;
                try testing.expectEqualStrings("w.mhc.fcnt", cnt.role.buf);
                try testing.expectEqual(calls.Dt.i32, cnt.dt);
                try testing.expectEqualSlices(i64, &.{@divFloor(md.n + 15, 16)}, cnt.shape);
                try testing.expectEqual(@as(i64, 0), cnt.offset);
                folds += 1;
                i += 1; // the `_finish_k` the fold run replaces
                continue;
            }
            try testing.expectEqualStrings(off[i].name, on[j].name);
            try testing.expectEqual(off[i].grid, on[j].grid);
            try testing.expectEqual(off[i].args.len, on[j].args.len);
            for (off[i].args, on[j].args) |p, q| {
                try testing.expectEqualStrings(p.name, q.name);
                try testing.expect(t4.argEql(p.arg, q.arg));
            }
        }
        try testing.expectEqual(on.len, j);
        try testing.expect(folds > 0);
        try testing.expectEqual(folds, count(on, fold_run));
        try testing.expectEqual(kept, count(on, pf_run));
        try testing.expectEqual(count(off, "_finish_k") - folds, count(on, "_finish_k"));
        try testing.expectEqual(count(off, pf_run), folds + kept);
        if (md.taps) try testing.expect(kept > 0);
    }
}
