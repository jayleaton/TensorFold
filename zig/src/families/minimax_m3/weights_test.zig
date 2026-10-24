//! weights.zig and weights_device.zig on the synthetic mini checkpoint, with fake slot bases and sinks; no GPU.
const std = @import("std");
const load = @import("load.zig");
const weights = @import("weights.zig");
const device = @import("weights_device.zig");
const shard = @import("shard.zig");
const ffn = @import("ffn.zig");
const synth = @import("load_synth_test.zig");
const native = @import("native.zig");
const budget = @import("budget.zig");

const gpa = std.testing.allocator;
const base0: u64 = 0x7000_0000_0000;

fn fakeBases(plan: *const load.Plan, out: []u64) void {
    for (plan.slots, out, 0..) |s, *b, i| b.* = base0 + i * 0x1_0000_0000 + (s.bytes & 0);
}

const Fixture = struct {
    dir: synth.Dir,
    cat: load.Catalog,
    pack: synth.Pack,

    fn init(io: std.Io, c: *const @import("config.zig").Config, mutate: bool) !Fixture {
        var pack = synth.Pack.init();
        errdefer pack.deinit();
        try pack.all(c);
        if (mutate) pack.ts.items[0].data[0] ^= 0xff;
        var dir = try synth.Dir.init(io);
        errdefer dir.deinit();
        try synth.write(io, dir.tmp.dir, &pack, .{});
        return .{ .dir = dir, .cat = try load.Catalog.open(gpa, io, dir.path), .pack = pack };
    }

    fn deinit(f: *Fixture) void {
        f.cat.deinit();
        f.dir.deinit();
        f.pack.deinit();
    }
};

test "describe: every record points at its slot base plus its offset, and the experts' widths decide uniform_k2" {
    const io = std.testing.io;
    const c = synth.mini();
    var f = try Fixture.init(io, &c, false);
    defer f.deinit();
    var plan = try load.Plan.build(gpa, &f.cat, &c, try shard.Share.of(&c, 1, 2), .{});
    defer plan.deinit();
    var bases: [4096]u64 = undefined;
    fakeBases(&plan, bases[0..plan.slots.len]);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const d = try weights.describe(arena.allocator(), &plan, &c, bases[0..plan.slots.len], c.layers);
    const w = d.weights;
    try std.testing.expectEqual(@as(usize, 4), w.layers.len);
    const q = plan.matrix(.q, 2, 0).?;
    try std.testing.expectEqual(bases[q.trellis] + q.trellis_off, w.layers[2].q.trellis);
    try std.testing.expectEqual(@as(u32, 12), w.layers[2].q.halves);
    try std.testing.expectEqual(q.n, w.layers[2].q.n);
    try std.testing.expect(w.layers[0].ffn == .dense and w.layers[1].ffn == .moe);
    try std.testing.expect(w.embed != 0 and w.final_norm != 0 and w.head.n == plan.matrix(.head, 0, 0).?.n);
    try std.testing.expect(w.layers[0].input_ln != 0 and w.layers[0].index_q_proj != 0 and w.layers[0].index_k_norm != 0);
    try std.testing.expect(w.layers[1].ffn.moe.router != 0 and w.layers[1].ffn.moe.bias != 0);
    // layer 1: w1, w3 and w2 are all 4-bit; layer 2 mixes 4/4/3; layer 3 has one 4-bit expert among 3-bit
    try std.testing.expectEqual(@as(u32, 8), w.layers[1].ffn.moe.experts.uniform_k2);
    try std.testing.expectEqual(@as(u32, 0), w.layers[2].ffn.moe.experts.uniform_k2);
    try std.testing.expectEqual(@as(u32, 0), w.layers[3].ffn.moe.experts.uniform_k2);
    try std.testing.expect(!w.layers[1].ffn.moe.experts.shared_input);
    try std.testing.expectEqual(@as(usize, 3), d.tables.len);
    const m1 = plan.moeLayer(1).?;
    try std.testing.expectEqual(@as(usize, c.experts), d.tables[0].gate.len);
    for (d.tables[0].gate, m1.w1.trellis_off) |p, off| try std.testing.expectEqual(bases[m1.w1.trellis_slot] + off, p);
    try std.testing.expectEqual(bases[m1.w2.svh_slot], w.layers[1].ffn.moe.experts.svh_d);
    try std.testing.expectEqual(@as(usize, m1.w1.k), w.layers[1].ffn.moe.experts.dims);
}

test "a thin model describes its first layers only, and a wrong plain dtype or missing tensor is refused" {
    const io = std.testing.io;
    const c = synth.mini();
    var f = try Fixture.init(io, &c, false);
    defer f.deinit();
    var plan = try load.Plan.build(gpa, &f.cat, &c, try shard.Share.of(&c, 0, 2), .{ .layers = &.{ 0, 1 } });
    defer plan.deinit();
    var bases: [4096]u64 = undefined;
    fakeBases(&plan, bases[0..plan.slots.len]);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const d = try weights.describe(arena.allocator(), &plan, &c, bases[0..plan.slots.len], 2);
    try std.testing.expectEqual(@as(usize, 2), d.weights.layers.len);
    try std.testing.expectEqual(@as(usize, 1), d.tables.len);
    try std.testing.expectError(error.MissingTensor, weights.describe(arena.allocator(), &plan, &c, bases[0..plan.slots.len], 3));
    var odd = plan.plain[0];
    odd.role = .index_q_proj;
    odd.dtype = .bf16;
    try std.testing.expectError(error.BadDtype, weights.checkPlain(odd));
    odd.role = .router;
    odd.dtype = .f32;
    try weights.checkPlain(odd);
}

test "a plan cut to layer 2 alone describes that layer at its own index; unordered or foreign ids are refused" {
    const io = std.testing.io;
    const c = synth.mini();
    var f = try Fixture.init(io, &c, false);
    defer f.deinit();
    var plan = try load.Plan.build(gpa, &f.cat, &c, try shard.Share.of(&c, 0, 1), .{ .layers = &.{2} });
    defer plan.deinit();
    var bases: [4096]u64 = undefined;
    fakeBases(&plan, bases[0..plan.slots.len]);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const d = try weights.describeLayers(arena.allocator(), &plan, &c, bases[0..plan.slots.len], &.{2});
    try std.testing.expectEqual(@as(usize, 3), d.weights.layers.len);
    const q = plan.matrix(.q, 2, 0).?;
    try std.testing.expectEqual(bases[q.trellis] + q.trellis_off, d.weights.layers[2].q.trellis);
    try std.testing.expect(d.weights.layers[2].ffn.moe.router != 0 and d.weights.layers[2].input_ln != 0);
    try std.testing.expectEqual(@as(usize, 1), d.tables.len);
    try std.testing.expectEqual(@as(u16, 2), d.tables[0].layer);
    try std.testing.expectError(error.MissingTensor, weights.describeLayers(arena.allocator(), &plan, &c, bases[0..plan.slots.len], &.{ 1, 2 }));
    try std.testing.expectError(error.BadLayer, weights.describeLayers(arena.allocator(), &plan, &c, bases[0..plan.slots.len], &.{ 2, 2 }));
    var full = try load.Plan.build(gpa, &f.cat, &c, try shard.Share.of(&c, 0, 1), .{});
    defer full.deinit();
    fakeBases(&full, bases[0..full.slots.len]);
    const sparse = try weights.describeLayers(arena.allocator(), &full, &c, bases[0..full.slots.len], &.{ 0, 3 });
    try std.testing.expect(sparse.weights.layers[0].ffn == .dense and sparse.weights.layers[3].ffn.moe.bias != 0);
}

test "the tables pack to the layout the pointers say, and a gate/up swap exchanges both pairs" {
    const io = std.testing.io;
    const c = synth.mini();
    var f = try Fixture.init(io, &c, false);
    defer f.deinit();
    var plan = try load.Plan.build(gpa, &f.cat, &c, try shard.Share.of(&c, 0, 1), .{});
    defer plan.deinit();
    var bases: [4096]u64 = undefined;
    fakeBases(&plan, bases[0..plan.slots.len]);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var d = try weights.describe(arena.allocator(), &plan, &c, bases[0..plan.slots.len], c.layers);
    const t = d.tables[1]; // layer 2
    const bytes = try weights.packTables(gpa, t);
    defer gpa.free(bytes);
    const n = t.gate.len;
    try std.testing.expectEqual(n * 36, bytes.len);
    const at = weights.tablePtrs(1000, n);
    try std.testing.expectEqual(@as(u64, 1000 + 24 * n), at.gate_k2);
    try std.testing.expectEqual(t.up[2], std.mem.readInt(u64, bytes[at.up - 1000 ..][16..24], .little));
    try std.testing.expectEqual(t.down_k2[3], std.mem.readInt(i32, bytes[at.down_k2 - 1000 ..][12..16], .little));
    weights.attachTables(&d.weights, t, at, .{ .w1_is_gate = false });
    const e = d.weights.layers[2].ffn.moe.experts;
    try std.testing.expectEqual(at.up, e.gate);
    try std.testing.expectEqual(at.gate, e.up);
    try std.testing.expectEqual(at.up_k2, e.gate_k2);
    try std.testing.expectEqual(at.down, e.down);
    try std.testing.expect(e.act_mode >= 2);
}

test "the digest reads headers and sizes: data bytes do not move it, a changed header does" {
    const io = std.testing.io;
    const c = synth.mini();
    var a = try Fixture.init(io, &c, false);
    defer a.deinit();
    var b = try Fixture.init(io, &c, true);
    defer b.deinit();
    const da = try device.digest(gpa, &a.cat);
    try std.testing.expectEqualSlices(u8, &da, &(try device.digest(gpa, &b.cat)));
    var other = try Fixture.init(io, &c, false);
    defer other.deinit();
    try other.pack.reshape("language_model.model.layers.0.input_layernorm.weight", "BF16", &.{c.hidden * 2});
    var dir2 = try synth.Dir.init(io);
    defer dir2.deinit();
    try synth.write(io, dir2.tmp.dir, &other.pack, .{});
    var cat2 = try load.Catalog.open(gpa, io, dir2.path);
    defer cat2.deinit();
    try std.testing.expect(!std.mem.eql(u8, &da, &(try device.digest(gpa, &cat2))));
}

const Count = struct {
    bytes: u64 = 0,
    ended: std.ArrayList(u16) = .empty,
    fn put(ctx: *anyopaque, _: u32, _: u64, bytes: []const u8) anyerror!void {
        const s: *Count = @ptrCast(@alignCast(ctx));
        s.bytes += bytes.len;
    }
    fn done(ctx: *anyopaque, shard_no: u16) void {
        const s: *Count = @ptrCast(@alignCast(ctx));
        s.ended.append(gpa, shard_no) catch unreachable;
    }
};

test "feed tells the hook once per shard, in shard order, after that shard's last run" {
    const io = std.testing.io;
    const c = synth.mini();
    var f = try Fixture.init(io, &c, false);
    defer f.deinit();
    var plan = try load.Plan.build(gpa, &f.cat, &c, try shard.Share.of(&c, 0, 1), .{});
    defer plan.deinit();
    var count: Count = .{};
    defer count.ended.deinit(gpa);
    try device.feed(gpa, &f.cat, &plan, .{ .ctx = &count, .put = Count.put }, .{ .ctx = &count, .call = Count.done });
    try std.testing.expectEqualSlices(u16, &.{ 0, 1 }, count.ended.items);
    try std.testing.expect(count.bytes > 0);
    var again: Count = .{};
    defer again.ended.deinit(gpa);
    try device.feed(gpa, &f.cat, &plan, .{ .ctx = &again, .put = Count.put }, null);
    try std.testing.expectEqual(count.bytes, again.bytes);
    try std.testing.expectEqual(@as(usize, 0), again.ended.items.len);
    var drop: device.CacheDrop = .{ .cat = &f.cat };
    drop.hook().call(drop.hook().ctx, 0); // advice on an open file: must not fail or change what is read
    try device.feed(gpa, &f.cat, &plan, .{ .ctx = &again, .put = Count.put }, drop.hook());
    try std.testing.expectEqual(2 * count.bytes, again.bytes);
}

test "the device loader compiles: its signature is the one native.zig calls" {
    _ = &device.loadWeights;
    _ = ffn.Switches;
}

test "admission: the plan's exact bytes and the digest come from the headers, and the host's room decides the verdict" {
    const io = std.testing.io;
    const c = synth.mini();
    var f = try Fixture.init(io, &c, false);
    defer f.deinit();
    const place: @import("ranks.zig").Place = .{ .rank = 1, .world = 2 };
    const o: native.Options = .{ .context = 65536, .drafts = false, .grant = 20 << 30 };
    const a = try native.admission(gpa, io, f.dir.path, &c, place, o);
    var plan = try load.Plan.build(gpa, &f.cat, &c, try shard.Share.of(&c, 1, 2), .{});
    defer plan.deinit();
    try std.testing.expectEqual(plan.slotBytes(), a.weights);
    try std.testing.expectEqualSlices(u8, &(try device.digest(gpa, &f.cat)), &a.fingerprint_weights);
    try std.testing.expect(a.verdict.? == .ok);
    try std.testing.expect(a.verdict.?.ok.context == 65536);
    const tight = try native.admission(gpa, io, f.dir.path, &c, place, .{ .context = 65536, .drafts = false, .grant = 5 << 30 });
    try std.testing.expect(tight.verdict.? == .refused);
    const words = try budget.refusalText(gpa, tight.verdict.?.refused);
    defer gpa.free(words);
    try std.testing.expect(std.mem.indexOf(u8, words, "CONTEXT=65536") != null or std.mem.indexOf(u8, words, "no KV room") != null);
    const unknown = try native.admission(gpa, io, f.dir.path, &c, place, .{ .context = 65536, .drafts = false });
    try std.testing.expect(unknown.verdict == null); // no host room named: nothing to check
    try std.testing.expectEqual(@as(u64, 0), budget.hosted().reserve);
}
