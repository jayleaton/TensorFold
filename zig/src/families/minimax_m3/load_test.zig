//! Loader tests: the shipped config and trimmed index against the plan, and a synthetic checkpoint written by the test.
const std = @import("std");
const core = @import("core");
const st = core.safetensors;
const load = @import("load.zig");
const config = @import("config.zig");
const names = @import("names.zig");
const shard = @import("shard.zig");

const gpa = std.testing.allocator;
const Config = config.Config;

// ---- the shipped files: no shard is needed, the catalog comes from the shape table ----

fn dtypeOf(text: []const u8) !st.DType {
    const table = [_][2][]const u8{ .{ "torch.bfloat16", "BF16" }, .{ "torch.float16", "F16" }, .{ "torch.float32", "F32" }, .{ "torch.int16", "I16" }, .{ "torch.int32", "I32" } };
    for (table) |t| if (std.mem.eql(u8, text, t[0])) return st.DType.parse(t[1]).?;
    return error.UnknownDType;
}

fn ref(shard_no: usize, dtype: st.DType, shape: []const usize, at: *u64) load.Ref {
    var r: load.Ref = .{ .shard = @intCast(shard_no), .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .begin = at.*, .len = dtype.size() };
    for (shape, 0..) |d, i| {
        r.shape[i] = d;
        r.len *= d;
    }
    at.* += r.len;
    return r;
}

fn shardNumber(file: []const u8) !usize {
    return (try std.fmt.parseInt(usize, file["model-".len..][0..5], 10)) - 1;
}

fn num(v: std.json.Value) usize {
    return @intCast(v.integer);
}

/// The trimmed index's tensors, shaped from fixtures/tensor_shapes.json (EXL3 parts derive from [K, N, words]).
fn fixtureCatalog() !load.Catalog {
    const shapes = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("fixtures/tensor_shapes.json"), .{});
    defer shapes.deinit();
    const index = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("fixtures/model.safetensors.index.json"), .{});
    defer index.deinit();
    const groups = shapes.value.object.get("groups").?.object;
    const plain = shapes.value.object.get("plain").?.object;
    var cat = load.Catalog.init(gpa);
    errdefer cat.deinit();
    var at: u64 = 0;
    var it = index.value.object.get("weight_map").?.object.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        const n = try shardNumber(kv.value_ptr.string);
        if (plain.get(name)) |p| {
            var dims: [4]usize = undefined;
            for (p.array.items[1..], 0..) |d, i| dims[i] = num(d);
            try cat.add(name, ref(n, st.DType.parse(p.array.items[0].string).?, dims[0 .. p.array.items.len - 1], &at));
        } else if (std.mem.endsWith(u8, name, ".e_score_correction_bias")) {
            // absent from quantization_config.json; the index total_size implies 512 bytes: fp32 [128]
            try cat.add(name, ref(n, .f32, &.{128}, &at));
        } else {
            const dot = std.mem.lastIndexOfScalar(u8, name, '.').?;
            const g = (groups.get(name[0..dot]) orelse return error.UnknownFixtureTensor).array.items;
            const k, const m, const words = .{ num(g[0]), num(g[1]), num(g[2]) };
            const part = names.parse(name).?.what.part.part;
            try cat.add(name, switch (part) {
                .trellis => ref(n, .i16, &.{ k / 16, m / 16, words }, &at),
                .suh => ref(n, .f16, &.{k}, &at),
                .svh => ref(n, .f16, &.{m}, &at),
                .mul1 => ref(n, .i32, &.{}, &at),
            });
        }
    }
    return cat;
}

const layers6 = [_]u16{ 0, 3, 17, 18, 46, 59 };

fn flashConfig() !Config {
    return config.parse(gpa, config.fixture);
}

fn halvesOf(plan: *const load.Plan, proj: names.Proj, layer: u32, expert: u32) u8 {
    return plan.matrix(proj, layer, expert).?.halves;
}

/// What every plan owes: the item bytes fill the slots exactly (a folded tensor is held as fp32).
fn expectFilled(plan: *const load.Plan) !void {
    var want: u64 = 0;
    for (plan.items) |i| want += i.span.bytes() * @as(u64, if (i.fold != .none) 2 else 1);
    try std.testing.expectEqual(want, plan.slotBytes());
    for (plan.items[1..], plan.items[0 .. plan.items.len - 1]) |b, a| try std.testing.expect(a.shard < b.shard or (a.shard == b.shard and a.base + a.span.offset <= b.base + b.span.offset));
}

test "the trimmed index has the section 1 counts and every name is a known role" {
    var cat = try fixtureCatalog();
    defer cat.deinit();
    // embedding, final norm and head (6), a dense layer (36), five MoE layers of 1,574 (24 + 1,550)
    try std.testing.expectEqual(@as(usize, 6 + 36 + 5 * 1574), cat.map.count());
    var it = cat.map.keyIterator();
    while (it.next()) |k| try std.testing.expect(names.parse(k.*) != null);
}

test "layers 0, 3, 17, 18, 46 and 59 plan with the section 1 bit table" {
    const c = try flashConfig();
    var cat = try fixtureCatalog();
    defer cat.deinit();
    var plan = try load.Plan.build(gpa, &cat, &c, try shard.Share.of(&c, 0, 1), .{ .layers = &layers6 });
    defer plan.deinit();
    try expectFilled(&plan);
    // q, k, v, o in six layers, three dense matrices, 3 shared + 384 routed in five MoE layers, the head
    try std.testing.expectEqual(@as(usize, 24 + 3 + 5 * (3 + 384) + 1), plan.matrices.len);
    try std.testing.expectEqual(@as(usize, 5), plan.moe.len);
    const experts = [_]struct { layer: u32, w1: u8, w3: u8, w2: u8 }{
        .{ .layer = 3, .w1 = 8, .w3 = 8, .w2 = 8 },
        .{ .layer = 17, .w1 = 8, .w3 = 8, .w2 = 6 }, // the one layer mixed within itself
        .{ .layer = 18, .w1 = 6, .w3 = 6, .w2 = 6 },
        .{ .layer = 46, .w1 = 8, .w3 = 8, .w2 = 8 },
        .{ .layer = 59, .w1 = 8, .w3 = 8, .w2 = 8 },
    };
    for (experts) |x| {
        const m = plan.moeLayer(x.layer).?;
        for ([_]struct { load.MoeProj, u8 }{ .{ m.w1, x.w1 }, .{ m.w3, x.w3 }, .{ m.w2, x.w2 } }) |pair| {
            try std.testing.expectEqual(@as(u32, 128), pair[0].histogram[pair[1]]); // all 128 experts at that K2
            for (pair[0].halves) |h| try std.testing.expectEqual(@as(i32, pair[1]), h);
        }
        try std.testing.expectEqual(@as(u8, x.w1), halvesOf(&plan, .w1, x.layer, 127));
    }
    for (layers6) |l| for ([_]names.Proj{ .q, .k, .v, .o }) |p| try std.testing.expectEqual(@as(u8, 12), halvesOf(&plan, p, l, 0));
    for ([_]names.Proj{ .dense_gate, .dense_up, .dense_down }) |p| try std.testing.expectEqual(@as(u8, 10), halvesOf(&plan, p, 0, 0));
    for ([_]names.Proj{ .shared_gate, .shared_up, .shared_down }) |p| try std.testing.expectEqual(@as(u8, 10), halvesOf(&plan, p, 17, 0));
    try std.testing.expectEqual(@as(u8, 12), halvesOf(&plan, .head, 0, 0));
    // plain tensors: fp16 index projections and the fp32 router stay as they are, norms are folded to fp32
    var folded: usize = 0;
    for (plan.plain) |p| {
        switch (p.role) {
            .index_q_proj, .index_k_proj => try std.testing.expect(p.dtype == .f16 and !p.folded),
            .router => try std.testing.expect(p.dtype == .f32 and !p.folded and p.rows == 128 and p.cols == 6144),
            .embed => try std.testing.expect(p.dtype == .bf16 and !p.folded),
            else => {},
        }
        if (p.role.isNorm()) {
            try std.testing.expect(p.folded and p.dtype == .f32);
            folded += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 6 * 6 + 1), folded); // five norms a layer plus the final norm
}

test "TP2 cuts of the real shapes: the head 782/781 units, experts halved, replicated parts whole" {
    const c = try flashConfig();
    var cat = try fixtureCatalog();
    defer cat.deinit();
    var a = try load.Plan.build(gpa, &cat, &c, try shard.Share.of(&c, 0, 2), .{ .layers = &layers6 });
    defer a.deinit();
    var b = try load.Plan.build(gpa, &cat, &c, try shard.Share.of(&c, 1, 2), .{ .layers = &layers6 });
    defer b.deinit();
    try expectFilled(&a);
    try expectFilled(&b);
    const ha = a.matrix(.head, 0, 0).?;
    const hb = b.matrix(.head, 0, 0).?;
    try std.testing.expectEqual(@as(u32, 782 * 128), ha.n);
    try std.testing.expectEqual(@as(u32, 781 * 128), hb.n);
    try std.testing.expectEqual(@as(u64, 384 * 12504 * 192), a.slots[ha.trellis].bytes + b.slots[hb.trellis].bytes);
    const m17 = b.moeLayer(17).?;
    try std.testing.expectEqual(@as(u32, 1536), m17.w1.n);
    try std.testing.expectEqual(@as(u32, 1536), m17.w2.k);
    try std.testing.expectEqual(@as(u64, 128 * 384 * 96 * 128), b.slots[m17.w1.trellis_slot].bytes); // 4-bit: 128 B a tile
    try std.testing.expectEqual(@as(u64, 128 * 96 * 384 * 96), b.slots[m17.w2.trellis_slot].bytes); // 3-bit down: 96 B a tile, K halved
    try std.testing.expectEqual(@as(u64, 128 * 1536 * 2), b.slots[m17.w1.svh_slot].bytes);
    try std.testing.expectEqual(@as(u64, 128 * 6144 * 2), b.slots[m17.w1.suh_slot].bytes);
    for (a.plain, b.plain) |pa, pb| switch (pa.role) {
        .index_q_proj => try std.testing.expect(pa.rows == 256 and pb.rows == 256), // two of four index heads each
        .index_k_proj, .router, .embed => try std.testing.expectEqual(pa.rows, pb.rows),
        else => {},
    };
}

test "the loader names no mapping of the shards" {
    const src = @embedFile("load.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "MemoryMap") == null);
    try std.testing.expect(std.mem.indexOf(u8, src, "openModel") == null);
    try std.testing.expect(std.mem.indexOf(u8, src, "mmap") == null);
}

test "Gemma 1 + w is applied once, in fp32" {
    const bf = [_]u8{ 0x80, 0x3F, 0x80, 0xBF, 0x00, 0x3F, 0x00, 0x00 }; // 1.0, -1.0, 0.5, 0.0
    var out: [4]f32 = undefined;
    try std.testing.expectEqualSlices(f32, &.{ 2.0, 0.0, 1.5, 1.0 }, load.foldGemma(&out, &bf));
}

// ---- the whole 89,832-tensor pack: only when its metadata is on disk (MM3_META_DIR) ----

fn metaDir() ?[]const u8 {
    const p = std.c.getenv("MM3_META_DIR") orelse return null;
    return std.mem.span(p);
}

fn readMeta(io: std.Io, dir: []const u8, file: []const u8) ![]u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, file });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
}

test "the full pack: 89,832 tensors, 22,309 groups, the per-layer bit table, TP2 totals" {
    const dir = metaDir() orelse return error.SkipZigTest;
    const io = std.testing.io;
    const index_text = try readMeta(io, dir, "model.safetensors.index.json");
    defer gpa.free(index_text);
    const quant_text = try readMeta(io, dir, "quantization_config.json");
    defer gpa.free(quant_text);
    const index = try std.json.parseFromSlice(std.json.Value, gpa, index_text, .{});
    defer index.deinit();
    const quant = try std.json.parseFromSlice(std.json.Value, gpa, quant_text, .{});
    defer quant.deinit();
    const wm = index.value.object.get("weight_map").?.object;
    var cat = load.Catalog.init(gpa);
    defer cat.deinit();
    var at: u64 = 0;
    var from_quant: usize = 0;
    var bytes: u64 = 0;
    for (quant.value.object.get("tensor_storage").?.object.values()) |entry| {
        var it = entry.object.get("stored_tensors").?.object.iterator();
        while (it.next()) |kv| {
            var dims: [4]usize = undefined;
            const shape = kv.value_ptr.object.get("shape").?.array.items;
            for (shape, 0..) |d, i| dims[i] = num(d);
            const n = try shardNumber(wm.get(kv.key_ptr.*).?.string);
            const dt = try dtypeOf(kv.value_ptr.object.get("dtype").?.string);
            try cat.add(kv.key_ptr.*, ref(n, dt, dims[0..shape.len], &at));
            try std.testing.expectEqual(num(kv.value_ptr.object.get("n_bytes").?), cat.get(kv.key_ptr.*).?.len);
            bytes += cat.get(kv.key_ptr.*).?.len;
            from_quant += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 89775), from_quant);
    // the 57 router biases are in the index only; the index' total_size is consistent with fp32 [128] each
    var it = wm.iterator();
    while (it.next()) |kv| if (cat.get(kv.key_ptr.*) == null) {
        try std.testing.expect(std.mem.endsWith(u8, kv.key_ptr.*, ".e_score_correction_bias"));
        try cat.add(kv.key_ptr.*, ref(try shardNumber(kv.value_ptr.string), .f32, &.{128}, &at));
        bytes += 512;
    };
    try std.testing.expectEqual(@as(usize, 89832), cat.map.count());
    try std.testing.expectEqual(@as(u64, @intCast(index.value.object.get("metadata").?.object.get("total_size").?.integer)), bytes);

    const c = try flashConfig();
    var whole = try load.Plan.build(gpa, &cat, &c, try shard.Share.of(&c, 0, 1), .{});
    defer whole.deinit();
    try std.testing.expectEqual(@as(usize, 22309), whole.matrices.len);
    try std.testing.expectEqual(@as(usize, 0), whole.ignored);
    try std.testing.expectEqual(@as(usize, 57), whole.moe.len);
    for (whole.moe) |m| {
        const l = m.layer;
        const w1: u32 = if (l <= 16 or l >= 46) 8 else if (l == 17) 8 else 6;
        const w2: u32 = if (l <= 16 or l >= 46) 8 else 6;
        try std.testing.expectEqual(@as(u32, 128), m.w1.histogram[w1]);
        try std.testing.expectEqual(@as(u32, 128), m.w3.histogram[w1]);
        try std.testing.expectEqual(@as(u32, 128), m.w2.histogram[w2]);
    }
    var r0 = try load.Plan.build(gpa, &cat, &c, try shard.Share.of(&c, 0, 2), .{});
    defer r0.deinit();
    var r1 = try load.Plan.build(gpa, &cat, &c, try shard.Share.of(&c, 1, 2), .{});
    defer r1.deinit();
    try expectFilled(&r0);
    try expectFilled(&r1);
    // plan section 5: about 91 GiB a rank (the folded norms are fp32, so a hair more than the stored bytes)
    const gib = 1024.0 * 1024.0 * 1024.0;
    for ([_]*load.Plan{ &r0, &r1 }) |p| {
        const g = @as(f64, @floatFromInt(p.slotBytes())) / gib;
        try std.testing.expect(g > 89.0 and g < 93.0);
    }
}
