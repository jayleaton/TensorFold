//! test-minimax-dspark: the pack's header and config (fixtures) against the name map, the rank plan and the budget.
const std = @import("std");
// The float64 forward runs on a tiny random draft (dspark_model_test.zig): no test reads the 2.3 GB of weights.
const core = @import("core");
// A passing test shows the code agrees with ITSELF, never that the semantics match the upstream model.
const mr = core.minimax_ref;
const target = @import("config.zig");
const shard = @import("shard.zig");
const load = @import("load.zig");
const dc = @import("dspark_config.zig");
const dn = @import("dspark_names.zig");
const dl = @import("dspark_load.zig");
const ref = @import("dspark_ref.zig");
const twin = @import("dspark_twin.zig");
const budget = @import("budget.zig");
const pool = @import("pool.zig");

test {
    _ = @import("dspark_model_test.zig");
    _ = dc;
    _ = dn;
    _ = ref;
    _ = budget;
}

const gpa = std.testing.allocator;

// ---- the pack's real header: names, shapes, cuts -----------------------------------------------------------------

const header_json = @embedFile("fixtures/dspark_header.json");
const header_data_len: usize = 2_301_852_674; // model.safetensors.index.json total_size

fn catalog() !load.Catalog {
    var cat = load.Catalog.init(gpa);
    errdefer cat.deinit();
    try dl.addHeader(&cat, 0, header_json, header_data_len);
    return cat;
}

fn catalogWith(from: []const u8, to: []const u8) !load.Catalog {
    const json = try std.mem.replaceOwned(u8, gpa, header_json, from, to);
    defer gpa.free(json);
    try std.testing.expect(!std.mem.eql(u8, json, header_json));
    var cat = load.Catalog.init(gpa);
    errdefer cat.deinit();
    try dl.addHeader(&cat, 0, json, header_data_len);
    return cat;
}

test "the header has the 97 tensors the name map says, every one a role, nothing else" {
    var cat = try catalog();
    defer cat.deinit();
    try std.testing.expectEqual(@as(usize, 97), cat.map.count());
    var plain: usize = 0;
    var lin: usize = 0;
    var it = cat.map.iterator();
    while (it.next()) |kv| {
        const n = dn.parse(kv.key_ptr.*) orelse return error.UnknownTensor;
        switch (n.what) {
            .plain => |p| {
                plain += 1;
                try std.testing.expectEqual(st_bf16, kv.value_ptr.dtype);
                _ = p;
            },
            .lin => |l| {
                lin += 1;
                try std.testing.expectEqual(if (l.part == .weight) core.safetensors.DType.f8_e4m3 else core.safetensors.DType.u8, kv.value_ptr.dtype);
            },
        }
    }
    try std.testing.expectEqual(@as(usize, 27), plain); // 7 global + 5 x 4 per layer
    try std.testing.expectEqual(@as(usize, 70), lin); // 5 layers x 7 linears x (codes, scales)
}
const st_bf16 = core.safetensors.DType.bf16;

fn share(rank: usize, world: usize) !shard.Share {
    return shard.Share.of(&target.flash, rank, world);
}

test "the plan for one rank and for two: bytes equal the arithmetic of dspark_config" {
    const c = dc.flash;
    for ([_]usize{ 1, 2 }) |world| for (0..world) |rank| {
        var cat = try catalog();
        defer cat.deinit();
        var plan = try dl.Plan.build(gpa, &cat, &c, try share(rank, world), .{});
        defer plan.deinit();
        const want = dc.rankWeights(&c, try share(rank, world), false);
        try std.testing.expectEqual(want.total(), plan.slotBytes());
        try std.testing.expectEqual(@as(usize, 35), plan.mats.len);
        try std.testing.expectEqual(@as(usize, 27), plan.plain.len);
        try std.testing.expectEqual(@as(usize, 0), plan.ignored);
        // the item runs read exactly the slot bytes, none twice
        var read: u64 = 0;
        for (plan.items) |i| read += i.span.bytes();
        try std.testing.expectEqual(plan.slotBytes(), read);
    };
}

test "two ranks' runs tile every tensor exactly once; replicated tensors are read whole by both" {
    const c = dc.flash;
    var cat = try catalog();
    defer cat.deinit();
    var plans: [2]dl.Plan = undefined;
    for (&plans, 0..) |*p, r| p.* = try dl.Plan.build(gpa, &cat, &c, try share(r, 2), .{});
    defer for (&plans) |*p| p.deinit();
    // each tensor's runs of both ranks sorted: cut tensors abut exactly, replicated ones are two whole copies
    var by_base: std.AutoHashMapUnmanaged(u64, std.ArrayList([2]u64)) = .empty;
    defer {
        var vit = by_base.valueIterator();
        while (vit.next()) |v| v.deinit(gpa);
        by_base.deinit(gpa);
    }
    for (plans) |p| for (p.items) |i| {
        const e = try by_base.getOrPut(gpa, i.base);
        if (!e.found_existing) e.value_ptr.* = .empty;
        for (0..i.span.count) |n| try e.value_ptr.append(gpa, .{ i.span.at(n), i.span.at(n) + i.span.len });
    };
    var it = cat.map.iterator();
    while (it.next()) |kv| {
        const runs = by_base.getPtr(kv.value_ptr.begin) orelse return error.TensorNotRead;
        std.mem.sort([2]u64, runs.items, {}, struct {
            fn less(_: void, a: [2]u64, b: [2]u64) bool {
                return a[0] < b[0];
            }
        }.less);
        const n = dn.parse(kv.key_ptr.*).?;
        const whole = n.what == .plain and (n.what.plain != .fc or true); // plain tensors are replicated
        if (whole) {
            // two identical copies of [0, len)
            try std.testing.expectEqual(@as(usize, 2), runs.items.len);
            for (runs.items) |r| try std.testing.expectEqual([2]u64{ 0, kv.value_ptr.len }, r);
        } else {
            var at: u64 = 0;
            for (runs.items) |r| {
                try std.testing.expectEqual(at, r[0]);
                at = r[1];
            }
            try std.testing.expectEqual(kv.value_ptr.len, at);
        }
    }
}

test "the cuts: q 4096 rows, kv 256, o and down input halves, gate/up 6144 rows; scales follow in 32s" {
    const c = dc.flash;
    var cat = try catalog();
    defer cat.deinit();
    var plan = try dl.Plan.build(gpa, &cat, &c, try share(1, 2), .{});
    defer plan.deinit();
    const want = [_]struct { p: dn.Lin, n: u32, k: u32 }{
        .{ .p = .q, .n = 4096, .k = 6144 },
        .{ .p = .k, .n = 256, .k = 6144 },
        .{ .p = .v, .n = 256, .k = 6144 },
        .{ .p = .o, .n = 6144, .k = 4096 },
        .{ .p = .gate, .n = 6144, .k = 6144 },
        .{ .p = .up, .n = 6144, .k = 6144 },
        .{ .p = .down, .n = 6144, .k = 6144 },
    };
    for (want) |w| {
        const m = plan.mat(w.p, 4).?;
        try std.testing.expectEqual(w.n, m.n);
        try std.testing.expectEqual(w.k, m.k);
        try std.testing.expectEqual(@as(u64, w.n) * w.k, plan.slots[m.codes].bytes);
        try std.testing.expectEqual(@as(u64, w.n) * w.k / 32, plan.slots[m.scales].bytes);
    }
    // rank 1's o_proj: columns [4096, 8192) of every row, 6144 runs a row apart (scale runs: 128 bytes)
    var runs: usize = 0;
    for (plan.items) |i| if (i.slot == plan.mat(.o, 0).?.codes) {
        try std.testing.expectEqual(@as(u64, 4096), i.span.offset);
        try std.testing.expectEqual(@as(u64, 4096), i.span.len);
        try std.testing.expectEqual(@as(u64, 8192), i.span.stride);
        try std.testing.expectEqual(@as(u64, 6144), i.span.count);
        runs += 1;
    };
    try std.testing.expectEqual(@as(usize, 1), runs);
    // the Markov tables whole by default, by the head's vocabulary units when asked
    try std.testing.expectEqual(@as(u32, 200064), plan.tensor(.markov_w1, 0).?.rows);
    var cut = try dl.Plan.build(gpa, &cat, &c, try share(1, 2), .{ .markov_cut = true });
    defer cut.deinit();
    try std.testing.expectEqual(@as(u32, 781 * 128), cut.tensor(.markov_w2, 0).?.rows);
    // nothing is folded: the norms are stored as they are (the Gemma offset is an unverified switch)
    for (plan.items) |i| try std.testing.expect(!i.fold);
}

test "a missing, misshaped, retyped or extra tensor refuses the plan" {
    const c = dc.flash;
    const s = try share(0, 2);
    {
        var cat = try catalogWith("language_model.model.dspark.fc.weight", "language_model.model.dspark.fx.weight");
        defer cat.deinit();
        try std.testing.expectError(error.MissingTensor, dl.Plan.build(gpa, &cat, &c, s, .{}));
    }
    {
        // codes retyped to e5m2 (same size)
        var cat = try catalogWith("layers.2.decoder_layer.mlp.up_proj.weight\":{\"dtype\":\"F8_E4M3\"", "layers.2.decoder_layer.mlp.up_proj.weight\":{\"dtype\":\"F8_E5M2\"");
        defer cat.deinit();
        try std.testing.expectError(error.MisshapedTensor, dl.Plan.build(gpa, &cat, &c, s, .{}));
    }
    {
        // scales retyped to i8
        var cat = try catalogWith("layers.4.decoder_layer.mlp.up_proj.weight_scale_inv\":{\"dtype\":\"U8\"", "layers.4.decoder_layer.mlp.up_proj.weight_scale_inv\":{\"dtype\":\"I8\"");
        defer cat.deinit();
        try std.testing.expectError(error.MisshapedTensor, dl.Plan.build(gpa, &cat, &c, s, .{}));
    }
    {
        // a tensor of the draft's namespace no role names (the draft has no head of its own)
        var cat = try catalog();
        defer cat.deinit();
        try cat.add("language_model.model.dspark.lm_head.weight", .{ .shard = 0, .dtype = .bf16, .rank = 1, .shape = .{ 1, 1, 1, 1, 1 }, .begin = 0, .len = 2 });
        try std.testing.expectError(error.UnexpectedTensor, dl.Plan.build(gpa, &cat, &c, s, .{}));
    }
    {
        // a tensor of another namespace is ignored, not refused
        var cat = try catalog();
        defer cat.deinit();
        try cat.add("language_model.model.layers.0.input_layernorm.weight", .{ .shard = 0, .dtype = .bf16, .rank = 1, .shape = .{ 6144, 1, 1, 1, 1 }, .begin = 0, .len = 12288 });
        var plan = try dl.Plan.build(gpa, &cat, &c, s, .{});
        defer plan.deinit();
        try std.testing.expectEqual(@as(usize, 1), plan.ignored);
    }
}

// ---- the float8 decode against TensorFold's own codec ----------------------------------------------------------

test "e4m3 and ue8m0 equal TensorFold's codec for every byte" {
    const codec = core.exlr8.codec;
    for (0..256) |i| {
        const code: u8 = @intCast(i);
        const mine = ref.e4m3(code);
        const theirs: f64 = codec.e4m3Value(code);
        if (std.math.isNan(theirs)) try std.testing.expect(std.math.isNan(mine)) else try std.testing.expectEqual(theirs, mine);
        // exlr8_dense.zig dequantizes an mxfp8 scale as ldexp(1, byte - 127); the NaN byte is qmmf.cu's
        if (i < 255) try std.testing.expectEqual(std.math.ldexp(@as(f64, 1), @as(i32, @intCast(i)) - 127), ref.ue8m0(code, 127));
    }
    try std.testing.expect(std.math.isNan(ref.ue8m0(255, 127)));
}

// ---- memory: what the draft adds to a rank's budget -----------------------------------------------------------------

test "budget: the draft takes its weights and runtime before any KV and a ring a stream; the refusal names it" {
    const c = dc.flash;
    const s = try share(0, 2);
    const dims = try pool.Dims.of(&target.flash, s);
    const total: u64 = 119 * budget.GiB; // assumed GB10 memory; the real number is measured at start-up
    const plain = try budget.tensorfold(total, null);
    const cost = dc.cost(&c, s, 16, 2048, false);
    const drafted = budget.withDraft(plain, cost);
    var in: budget.Input = .{ .available = total, .weights = 91 * budget.GiB, .terms = plain };
    var in_d = in;
    in_d.terms = drafted;
    const without = budget.check(dims, in, null).ok;
    const with = budget.check(dims, in_d, null).ok;
    // the draft costs weights + runtime + one stream's ring and scratch, which the pool gives up in whole pages
    const spent = cost.weights + cost.runtime + cost.per_lane;
    const lost = (without.pool_tokens - with.pool_tokens) * dims.tokenBytes();
    // (whole pages, and the score chunk shrinks with the shorter context: a few pages of slack either way)
    try std.testing.expect(@abs(@as(i64, @intCast(lost)) - @as(i64, @intCast(spent))) <= @as(i64, @intCast(3 * dims.pageBytes())));
    try std.testing.expect(with.context < without.context);
    try std.testing.expectEqual(spent, budget.draftBytes(drafted, 1));
    try std.testing.expectEqual(@as(u64, 0), budget.draftBytes(plain, 4));
    // a context that fit without the draft but not with it is refused with the draft named
    in_d.weights = 106 * budget.GiB;
    in.weights = 106 * budget.GiB;
    const lone = budget.check(dims, in, null);
    switch (lone) {
        .ok => |cap| {
            const asked = cap.context;
            const verdict = budget.check(dims, in_d, asked);
            if (verdict == .refused) {
                const text = try budget.refusalText(gpa, verdict.refused);
                defer gpa.free(text);
                try std.testing.expect(std.mem.indexOf(u8, text, "the dspark draft takes") != null);
                try std.testing.expect(std.mem.indexOf(u8, text, "serve without --drafter") != null);
            } else return error.DraftCostNothing;
        },
        .refused => {},
    }
    // lanes: each extra stream adds its ring
    var four = in_d;
    four.lanes = 4;
    try std.testing.expectEqual(cost.weights + cost.runtime + 4 * cost.per_lane, budget.draftBytes(four.terms, 4));
}
