//! Host tests of 8,192-row prefill segments (TF_DSV41_PF_8K, test-dsv41-knobs): the knob and the emitter under it.
const std = @import("std");
const testing = std.testing;
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const pk = @import("prod_knobs.zig");
const fp = @import("forward_prefill.zig");
const t4 = @import("pf4k_test.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

const ok8 = [2][]const u8{ "TF_DSV41_PF_8K_MEMORY_OK", "1" };
const on8 = [2][]const u8{ "TF_DSV41_PF_8K", "1" };
const ok4 = [2][]const u8{ "TF_DSV41_PF_4K_MEMORY_OK", "1" };
const on4 = [2][]const u8{ "TF_DSV41_PF_4K", "1" };

test "TF_DSV41_PF_8K: 0 / 1 only, refused without TF_DSV41_PF_8K_MEMORY_OK=1, 8,192-row segments over PF_4K's" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expectEqual(@as(u32, 0), try pk.bigRows(&Fake.get));
    try testing.expectEqual(@as(?u32, null), try pk.prefillRows(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_PF_8K", "0" }};
    try testing.expect(!try pk.pf8k(&Fake.get));
    try testing.expectEqual(@as(u32, 0), try pk.bigRows(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_PF_8K", "2" }};
    try testing.expectError(error.BadKnob, pk.pf8k(&Fake.get));
    Fake.pairs = &.{on8};
    try testing.expectError(error.BadKnob, pk.bigRows(&Fake.get));
    try testing.expectError(error.BadKnob, pk.prefillRows(&Fake.get));
    Fake.pairs = &.{ on8, .{ "TF_DSV41_PF_8K_MEMORY_OK", "0" } };
    try testing.expectError(error.BadKnob, pk.pf8k(&Fake.get));
    Fake.pairs = &.{ on8, ok8 };
    try testing.expectEqual(@as(u32, 8192), try pk.bigRows(&Fake.get));
    try testing.expectEqual(@as(?u32, 8192), try pk.prefillRows(&Fake.get));
    Fake.pairs = &.{ on8, ok8, .{ "TF_DSV41_PREFILL_CHUNK", "8192" }, .{ "TF_DSV41_PREFILL_ROWS", "8192" } };
    try testing.expectEqual(@as(?u32, 8192), try pk.prefillRows(&Fake.get));
    // prod's 4K words with 8K on: refused, not 4,096-row segments under an 8K label
    Fake.pairs = &.{ on8, ok8, .{ "TF_DSV41_PREFILL_CHUNK", "4096" } };
    try testing.expectError(error.BadKnob, pk.prefillRows(&Fake.get));
    Fake.pairs = &.{ on8, ok8, .{ "TF_DSV41_PREFILL_ROWS", "2048" } };
    try testing.expectError(error.BadKnob, pk.prefillRows(&Fake.get));
    // over prod's PF_4K (its own acknowledgement still read): 8,192
    Fake.pairs = &.{ on4, ok4, on8, ok8, .{ "TF_DSV41_PREFILL_CHUNK", "8192" }, .{ "TF_DSV41_PREFILL_ROWS", "8192" } };
    try testing.expectEqual(@as(u32, 8192), try pk.bigRows(&Fake.get));
    try testing.expectEqual(@as(?u32, 8192), try pk.prefillRows(&Fake.get));
    Fake.pairs = &.{ on4, on8, ok8 };
    try testing.expectError(error.BadKnob, pk.bigRows(&Fake.get));
    // PF_4K alone as it was; without either the 2,048 limit stands
    Fake.pairs = &.{ on4, ok4 };
    try testing.expectEqual(@as(u32, 4096), try pk.bigRows(&Fake.get));
    try testing.expectEqual(@as(?u32, 4096), try pk.prefillRows(&Fake.get));
    Fake.pairs = &.{ on4, ok4, .{ "TF_DSV41_PREFILL_ROWS", "8192" } };
    try testing.expectError(error.BadKnob, pk.prefillRows(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_PREFILL_CHUNK", "8192" }};
    try testing.expectError(error.BadKnob, pk.prefillRows(&Fake.get));
}

const opts: block.Options = .{ .gm_v2 = .one, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048 };

fn big(rows: i64) block.Options {
    var o = opts;
    o.pf4k = true;
    o.prefill_rows = rows;
    return o;
}

test "an 8,192-row segment under TF_DSV41_PF_8K: the 2K / 4K program's calls, the router in four CHUNK launches, x3gm one block" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try t4.widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    const D: i64 = cfg.hidden;
    const four = big(4096);
    const eight = big(8192);
    try testing.expectError(error.Unsupported, bp.emitPrefill(aa, &cfg, &w, four, &layers, 8192, 8192, false));
    try testing.expectError(error.Unsupported, bp.emitPrefill(aa, &cfg, &w, eight, &layers, 8208, 8192, false));
    // starts past the stream threshold for every size (a ratio-2 layer's visible keys >= 4,096 at the 2K segment's end)
    for ([_]i64{ 8192, 40960 }) |start| {
        const s2 = try bp.emitPrefill(aa, &cfg, &w, opts, &layers, 2048, start, false);
        const s4 = try bp.emitPrefill(aa, &cfg, &w, four, &layers, 4096, start, false);
        const b = try bp.emitPrefill(aa, &cfg, &w, eight, &layers, 8192, start, false);
        try t4.sameCalls(s2, b, 4);
        try t4.sameCalls(s4, b, 2);
        const moes = t4.count(b, "tf_dsv41_x3gm_v1.rot");
        try testing.expectEqual(@as(usize, layers.len), moes);
        try testing.expectEqual(moes, t4.count(b, "tf_dsv41_x3gm_v1.gateup2"));
        try testing.expectEqual(moes, t4.count(b, "tf_dsv41_x3gm_v1.down2"));
        try testing.expectEqual(moes, t4.count(b, "glue.gm_plan"));
        var routes: usize = 0;
        for (b) |c| {
            if (std.mem.eql(u8, c.name, "tf_dsv41_x3gm_v1.rot")) {
                try testing.expectEqual(@as(i64, 8192 * 6), c.args[9].arg.i); // P: 49,152 pairs, one block
                try testing.expectEqual(@as(i64, 8192 * 6), c.args[5].arg.t.shape[0]);
            }
            if (!t4.isRoute(c)) continue;
            const q: i64 = @intCast(routes % 4);
            const x = c.args[0].arg.t;
            try testing.expectEqual(@as(i64, 2048), x.shape[0]);
            try testing.expectEqual(q * 2048 * D * 2, x.offset);
            for ([_]usize{ 3, 4, 5 }) |k| {
                const t = c.args[k].arg.t;
                try testing.expectEqual(@as(i64, 2048), t.shape[0]);
                try testing.expectEqual(q * 2048 * t.shape[1] * 4, t.offset);
            }
            try testing.expectEqual(@as(i64, 16), c.args[10].arg.i);
            try testing.expectEqual(@as(i64, 8), c.args[11].arg.i);
            try testing.expectEqual(@as(i64, 2), c.args[12].arg.i);
            routes += 1;
        }
        try testing.expectEqual(4 * moes, routes);
    }
    // ragged: 8,200 rows are 8,192 + an 8-row segment; a 6,152-row segment's last launch is 8 rows at (8, 1)
    _ = try bp.emitPrefill(aa, &cfg, &w, eight, &layers, 8, 8192 + 8192, false);
    const r = try bp.emitPrefill(aa, &cfg, &w, eight, &layers, 6152, 8192, false);
    var k: usize = 0;
    for (r) |c| if (t4.isRoute(c)) {
        const last = k % 4 == 3;
        try testing.expectEqual(@as(i64, if (last) 8 else 2048), c.args[0].arg.t.shape[0]);
        try testing.expectEqual(@as(i64, if (last) 8 else 16), c.args[10].arg.i);
        try testing.expectEqual(@as(i64, if (last) 1 else 2), c.args[12].arg.i);
        k += 1;
    };
    try testing.expectEqual(4 * layers.len, k);
    // TF_DSV41_GM_GU2 at 8K: gate / up's own 128-member plan, one a MoE call
    var gu = eight;
    gu.gm_gu2 = true;
    var plans: usize = 0;
    for (try bp.emitPrefill(aa, &cfg, &w, gu, &layers, 8192, 8192, false)) |c| {
        if (std.mem.eql(u8, c.name, "glue.gm_plan") and c.args[8].arg.i == 128) plans += 1;
    }
    try testing.expectEqual(layers.len, plans);
    // CED's encoder pass and a two-segment piece at 8,192 rows
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    _ = try bp.emitReplay(aa, &cfg, &w, eight, &enc, 8192, 0, .encoder);
    const sp = [_]bp.Span{ .{ .slot = 0, .start = 0, .n = 8192 - 17 }, .{ .slot = 1, .start = 0, .n = 17 } };
    _ = try bp.emitMulti(aa, &cfg, &w, eight, &enc, &sp, .encoder);
}

test "TF_DSV41_PF_8K leaves a segment of 4,096 rows or fewer as the 4K program, launch for launch, but for the staging ring" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try t4.widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    for ([_]i64{ 4096, 2048, 1024, 300, 17, 12 }) |n| {
        const a = try bp.emitPrefill(aa, &cfg, &w, big(4096), &layers, n, 4000, n <= 2048);
        const b = try bp.emitPrefill(aa, &cfg, &w, big(8192), &layers, n, 4000, n <= 2048);
        try testing.expectEqual(a.len, b.len);
        for (a, b) |x, y| {
            try testing.expectEqualStrings(x.name, y.name);
            try testing.expectEqual(x.grid, y.grid);
            try testing.expectEqual(x.args.len, y.args.len);
            for (x.args, y.args) |p, q| {
                // the staging ring's size: ceilPow2(prefill_rows + 127), 8,192 -> 16,384, its roles and its row count
                if (p.arg == .t and std.mem.startsWith(u8, t4.roleOf(p.arg), "s.stage.")) continue;
                if (p.arg == .i and p.arg.i == 8192 and q.arg == .i and q.arg.i == 16384) continue;
                if (!(std.meta.eql(p.arg, q.arg) or t4.argEql(p.arg, q.arg))) {
                    std.debug.print("{s} arg {s}: {any} vs {any}\n", .{ x.name, p.name, p.arg, q.arg });
                    return error.TestUnexpectedResult;
                }
            }
        }
    }
}

test "TF_DSV41_PF_8K adds no byte to the forward's buffer plan: the 8K program's own roles are its workspace's (with PF_OVERLAP too)" {
    const base: block.Options = .{ .gm_v2 = .one, .limit = 1 << 20, .rope_rows = (1 << 20) + 2048, .index_budget = 64 << 20, .taps = true };
    try t4.planWithin(base, 8192);
    var ovl = base;
    ovl.branches = true;
    ovl.pf_overlap = 4;
    try t4.planWithin(ovl, 8192);
    try testing.expectEqual(@as(i64, 8192), fp.k4Options(base, 8192).prefill_rows);
}
