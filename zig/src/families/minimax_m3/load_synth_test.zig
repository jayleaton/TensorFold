//! Loader tests on a synthetic mini-checkpoint whose safetensors headers the test writes, good and broken.
const std = @import("std");
const load = @import("load.zig");
const config = @import("config.zig");
const names = @import("names.zig");
const shard = @import("shard.zig");

const gpa = std.testing.allocator;
const Config = config.Config;

/// A two-expert-pair model the kernels' widths allow: 4 layers (one dense), 4 experts, every width a multiple of 256.
pub fn mini() Config {
    var c = config.flash;
    c.hidden = 256;
    c.layers = 4;
    c.heads = 4;
    c.kv_heads = 2;
    c.vocab = 256;
    c.dense_inter = 256;
    c.moe_inter = 256;
    c.shared_inter = 256;
    c.experts = 4;
    c.topk = 2;
    c.index.heads = 2;
    c.moe = std.StaticBitSet(config.max_layers).empty;
    c.moe.setRangeValue(.{ .start = 1, .end = 4 }, true);
    return c;
}

/// Half-bit widths of the mini pack: layer 2 mixes 4/4/3 bits like the real 17; layer 3 is 3-bit but expert 2's w1.
fn halvesFor(proj: names.Proj, layer: u32, expert: u32) u32 {
    return switch (proj) {
        .q, .k, .v, .o, .head => 12,
        .dense_gate, .dense_up, .dense_down, .shared_gate, .shared_up, .shared_down => 10,
        .w1 => if (layer == 3 and expert != 2) 6 else 8,
        .w3 => if (layer == 3) 6 else 8,
        .w2 => if (layer == 1) 8 else 6,
    };
}

const T = struct { name: []const u8, dtype: []const u8, shape: []const usize, data: []u8 };

fn dtypeSize(dtype: []const u8) usize {
    if (std.mem.eql(u8, dtype, "F32") or std.mem.eql(u8, dtype, "I32")) return 4;
    return if (std.mem.eql(u8, dtype, "I8") or std.mem.eql(u8, dtype, "U8")) 1 else 2;
}

pub const Pack = struct {
    arena: std.heap.ArenaAllocator,
    ts: std.ArrayList(T) = .empty,

    pub fn init() Pack {
        return .{ .arena = .init(gpa) };
    }

    pub fn deinit(p: *Pack) void {
        p.arena.deinit();
    }

    /// A tensor of `dtype` and `shape` whose bytes are a function of its name (bf16 holds small exact values).
    pub fn add(p: *Pack, name: []const u8, dtype: []const u8, shape: []const usize) !void {
        const a = p.arena.allocator();
        var n: usize = dtypeSize(dtype);
        for (shape) |d| n *= d;
        const data = try a.alloc(u8, n);
        var prng = std.Random.DefaultPrng.init(std.hash.Wyhash.hash(0, name));
        prng.random().bytes(data);
        if (std.mem.eql(u8, dtype, "BF16")) {
            for (0..n / 2) |i| {
                const w: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.125;
                std.mem.writeInt(u16, data[2 * i ..][0..2], @intCast(@as(u32, @bitCast(w)) >> 16), .little);
            }
        }
        if (std.mem.endsWith(u8, name, ".mul1") and data.len >= 4) std.mem.writeInt(u32, data[0..4], load.mul1_multiplier, .little);
        try p.ts.append(a, .{ .name = try a.dupe(u8, name), .dtype = dtype, .shape = try a.dupe(usize, shape), .data = data });
    }

    pub fn find(p: *Pack, name: []const u8) *T {
        for (p.ts.items) |*t| if (std.mem.eql(u8, t.name, name)) return t;
        unreachable;
    }

    fn drop(p: *Pack, name: []const u8) void {
        for (p.ts.items, 0..) |t, i| if (std.mem.eql(u8, t.name, name)) {
            _ = p.ts.swapRemove(i);
            return;
        };
        unreachable;
    }

    pub fn reshape(p: *Pack, name: []const u8, dtype: []const u8, shape: []const usize) !void {
        p.drop(name);
        try p.add(name, dtype, shape);
    }

    fn group(p: *Pack, proj: names.Proj, layer: u32, expert: u32, k: usize, n: usize) !void {
        var b: names.Buf = undefined;
        var b2: names.Buf = undefined;
        const prefix = names.group(&b, proj, layer, expert);
        const words = halvesFor(proj, layer, expert) * 8;
        try p.add(names.part(&b2, prefix, .trellis), "I16", &.{ k / 16, n / 16, words });
        try p.add(names.part(&b2, prefix, .suh), "F16", &.{k});
        try p.add(names.part(&b2, prefix, .svh), "F16", &.{n});
        try p.add(names.part(&b2, prefix, .mul1), "I32", &.{});
    }

    fn plain(p: *Pack, role: names.Plain, layer: u32, dtype: []const u8, shape: []const usize) !void {
        var b: names.Buf = undefined;
        try p.add(names.plain(&b, role, layer), dtype, shape);
    }

    pub fn all(p: *Pack, c: *const Config) !void {
        const q = c.heads * c.head_dim;
        const kv = c.kv_heads * c.head_dim;
        try p.plain(.embed, 0, "BF16", &.{ c.vocab, c.hidden });
        try p.plain(.final_norm, 0, "BF16", &.{c.hidden});
        try p.group(.head, 0, 0, c.hidden, c.vocab);
        for (0..c.layers) |l| {
            try p.plain(.input_ln, @intCast(l), "BF16", &.{c.hidden});
            try p.plain(.post_ln, @intCast(l), "BF16", &.{c.hidden});
            try p.plain(.q_norm, @intCast(l), "BF16", &.{c.head_dim});
            try p.plain(.k_norm, @intCast(l), "BF16", &.{c.head_dim});
            try p.plain(.index_q_proj, @intCast(l), "F16", &.{ c.index.heads * c.index.dim, c.hidden });
            try p.plain(.index_k_proj, @intCast(l), "F16", &.{ c.index.dim, c.hidden });
            try p.plain(.index_q_norm, @intCast(l), "BF16", &.{c.index.dim});
            try p.plain(.index_k_norm, @intCast(l), "BF16", &.{c.index.dim});
            const i: u32 = @intCast(l);
            try p.group(.q, i, 0, c.hidden, q);
            try p.group(.k, i, 0, c.hidden, kv);
            try p.group(.v, i, 0, c.hidden, kv);
            try p.group(.o, i, 0, q, c.hidden);
            if (!c.isMoe(i)) {
                try p.group(.dense_gate, i, 0, c.hidden, c.dense_inter);
                try p.group(.dense_up, i, 0, c.hidden, c.dense_inter);
                try p.group(.dense_down, i, 0, c.dense_inter, c.hidden);
                continue;
            }
            try p.plain(.router, i, "F32", &.{ c.experts, c.hidden });
            try p.plain(.router_bias, i, "F32", &.{c.experts});
            try p.group(.shared_gate, i, 0, c.hidden, c.shared_inter);
            try p.group(.shared_up, i, 0, c.hidden, c.shared_inter);
            try p.group(.shared_down, i, 0, c.shared_inter, c.hidden);
            for (0..c.experts) |e| {
                try p.group(.w1, i, @intCast(e), c.hidden, c.moe_inter);
                try p.group(.w3, i, @intCast(e), c.hidden, c.moe_inter);
                try p.group(.w2, i, @intCast(e), c.moe_inter, c.hidden);
            }
        }
    }
};

pub const Write = struct { shards: usize = 2, index: bool = true, cut_last: usize = 0 };

pub fn shardFile(buf: *[48]u8, i: usize, of: usize) []const u8 {
    return std.fmt.bufPrint(buf, "model-{d:0>5}-of-{d:0>5}.safetensors", .{ i + 1, of }) catch unreachable;
}

/// Tensors alternate over the shards (a group's parts are in both) with their headers and an index.json.
pub fn write(io: std.Io, dir: std.Io.Dir, p: *const Pack, w: Write) !void {
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(gpa);
    try index.appendSlice(gpa, "{\"metadata\":{},\"weight_map\":{");
    var first = true;
    for (0..w.shards) |s| {
        var header: std.ArrayList(u8) = .empty;
        defer header.deinit(gpa);
        var data: std.ArrayList(u8) = .empty;
        defer data.deinit(gpa);
        var fb: [48]u8 = undefined;
        const file = shardFile(&fb, s, w.shards);
        try header.append(gpa, '{');
        for (p.ts.items, 0..) |t, i| {
            if (i % w.shards != s) continue;
            if (data.items.len != 0) try header.append(gpa, ',');
            try header.print(gpa, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.name, t.dtype });
            for (t.shape, 0..) |d, j| try header.print(gpa, "{s}{d}", .{ if (j > 0) "," else "", d });
            try header.print(gpa, "],\"data_offsets\":[{d},{d}]}}", .{ data.items.len, data.items.len + t.data.len });
            try data.appendSlice(gpa, t.data);
            try index.print(gpa, "{s}\"{s}\":\"{s}\"", .{ if (first) "" else ",", t.name, file });
            first = false;
        }
        try header.append(gpa, '}');
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(gpa);
        try bytes.appendNTimes(gpa, 0, 8);
        std.mem.writeInt(u64, bytes.items[0..8], header.items.len, .little);
        try bytes.appendSlice(gpa, header.items);
        try bytes.appendSlice(gpa, data.items);
        const cut = if (s + 1 == w.shards) w.cut_last else 0;
        try dir.writeFile(io, .{ .sub_path = file, .data = bytes.items[0 .. bytes.items.len - cut] });
    }
    try index.appendSlice(gpa, "}}");
    if (w.index) try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}

pub const Dir = struct {
    tmp: std.testing.TmpDir,
    path: [:0]u8,

    pub fn init(io: std.Io) !Dir {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        return .{ .tmp = tmp, .path = try tmp.dir.realPathFileAlloc(io, ".", gpa) };
    }

    pub fn deinit(d: *Dir) void {
        gpa.free(d.path);
        d.tmp.cleanup();
    }
};

// ---- what a good checkpoint streams ----

/// Records every put and fills a host buffer per slot, so the test can look at what a device would hold.
pub const Host = struct {
    bufs: [][]u8,
    written: []u64,

    pub fn init(plan: *const load.Plan) !Host {
        const bufs = try gpa.alloc([]u8, plan.slots.len);
        const written = try gpa.alloc(u64, plan.slots.len);
        @memset(written, 0);
        for (bufs, plan.slots) |*b, s| b.* = try gpa.alloc(u8, s.bytes);
        return .{ .bufs = bufs, .written = written };
    }

    pub fn deinit(h: *Host) void {
        for (h.bufs) |b| gpa.free(b);
        gpa.free(h.bufs);
        gpa.free(h.written);
    }

    fn put(ctx: *anyopaque, slot: u32, offset: u64, bytes: []const u8) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        if (offset + bytes.len > h.bufs[slot].len) return error.OutOfSlot;
        @memcpy(h.bufs[slot][offset..][0..bytes.len], bytes);
        h.written[slot] += bytes.len;
    }

    pub fn sink(h: *Host) load.Sink {
        return .{ .ctx = h, .put = put };
    }
};

pub fn rankPlan(cat: *load.Catalog, c: *const Config, rank: usize, world: usize) !load.Plan {
    return load.Plan.build(gpa, cat, c, try shard.Share.of(c, rank, world), .{});
}

/// Trellis tiles [lo, hi) along one axis, taken tile by tile from the full tensor (not through Span arithmetic).
fn expectedTrellis(full: []const u8, shape: []const usize, axis: names.Cut, lo: usize, hi: usize) ![]u8 {
    const tile = shape[2] * 2;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    switch (axis) {
        .col => for (0..shape[0]) |kt| for (lo..hi) |nt| try out.appendSlice(gpa, full[(kt * shape[1] + nt) * tile ..][0..tile]),
        .row => for (lo..hi) |kt| for (0..shape[1]) |nt| try out.appendSlice(gpa, full[(kt * shape[1] + nt) * tile ..][0..tile]),
    }
    return out.toOwnedSlice(gpa);
}

fn slice(h: *const Host, slot: u32, off: u64, len: usize) []const u8 {
    return h.bufs[slot][off..][0..len];
}

pub fn expectStreamed(c: *const Config, pack: *Pack, plan: *const load.Plan, host: *const Host) !void {
    for (host.written, plan.slots) |w, s| try std.testing.expectEqual(s.bytes, w);
    var b: names.Buf = undefined;
    var b2: names.Buf = undefined;
    for (plan.matrices) |m| {
        const prefix = names.group(&b, m.proj, m.layer, m.expert);
        const tr = pack.find(names.part(&b2, prefix, .trellis));
        const su = pack.find(names.part(&b2, prefix, .suh)).data;
        const sv = pack.find(names.part(&b2, prefix, .svh)).data;
        const cut = plan.share.cut(c, m.proj);
        const axis = m.proj.cut();
        const want = try expectedTrellis(tr.data, tr.shape, axis, cut.lo / 16, cut.hi / 16);
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u8, want, slice(host, m.trellis, m.trellis_off, want.len));
        const su_want = if (axis == .row) su[cut.lo * 2 .. cut.hi * 2] else su;
        const sv_want = if (axis == .col) sv[cut.lo * 2 .. cut.hi * 2] else sv;
        try std.testing.expectEqualSlices(u8, su_want, slice(host, m.suh, m.suh_off, su_want.len));
        try std.testing.expectEqualSlices(u8, sv_want, slice(host, m.svh, m.svh_off, sv_want.len));
        try std.testing.expectEqual(@as(u32, @intCast(if (axis == .row) cut.len() else su.len / 2)), m.k);
        try std.testing.expectEqual(@as(u32, @intCast(if (axis == .col) cut.len() else sv.len / 2)), m.n);
    }
    for (plan.plain) |pt| {
        const full = pack.find(names.plain(&b, pt.role, pt.layer)).data;
        const got = host.bufs[pt.slot];
        if (pt.folded) {
            try std.testing.expectEqual(full.len * 2, got.len);
            for (0..full.len / 2) |i| {
                const w: f32 = @bitCast(@as(u32, std.mem.readInt(u16, full[2 * i ..][0..2], .little)) << 16);
                const g: f32 = @bitCast(std.mem.readInt(u32, got[4 * i ..][0..4], .little));
                try std.testing.expectEqual(1.0 + w, g);
            }
        } else if (pt.role == .index_q_proj) {
            const rows = plan.share.indexRows(c);
            const row = full.len / (c.index.heads * c.index.dim);
            try std.testing.expectEqualSlices(u8, full[rows.lo * row .. rows.hi * row], got);
        } else try std.testing.expectEqualSlices(u8, full, got);
    }
}

test "a synthetic checkpoint streams, for one rank and for two, exactly the cut tiles and scales" {
    const io = std.testing.io;
    const c = mini();
    var pack = Pack.init();
    defer pack.deinit();
    try pack.all(&c);
    var dir = try Dir.init(io);
    defer dir.deinit();
    try write(io, dir.tmp.dir, &pack, .{});
    var cat = try load.Catalog.open(gpa, io, dir.path);
    defer cat.deinit();
    try std.testing.expectEqual(pack.ts.items.len, cat.map.count());
    for ([_]usize{ 1, 2 }) |world| for (0..world) |rank| for ([_]usize{ 32 * 1024, load.Streamer.default_chunk }) |chunk| {
        var plan = try rankPlan(&cat, &c, rank, world);
        defer plan.deinit();
        var host = try Host.init(&plan);
        defer host.deinit();
        var streamer = try load.Streamer.init(gpa, &cat, chunk);
        defer streamer.deinit();
        try streamer.run(&plan, host.sink());
        try expectStreamed(&c, &pack, &plan, &host);
    };
}

test "per-expert widths and pointer offsets: layer 2 mixes 4/4/3 bits, layer 3 has one 4-bit expert among 3-bit" {
    const io = std.testing.io;
    const c = mini();
    var pack = Pack.init();
    defer pack.deinit();
    try pack.all(&c);
    var dir = try Dir.init(io);
    defer dir.deinit();
    try write(io, dir.tmp.dir, &pack, .{});
    var cat = try load.Catalog.open(gpa, io, dir.path);
    defer cat.deinit();
    for ([_]usize{ 1, 2 }) |world| {
        var plan = try rankPlan(&cat, &c, 0, world);
        defer plan.deinit();
        const l2 = plan.moeLayer(2).?;
        try std.testing.expectEqualSlices(i32, &.{ 8, 8, 8, 8 }, l2.w1.halves);
        try std.testing.expectEqualSlices(i32, &.{ 8, 8, 8, 8 }, l2.w3.halves);
        try std.testing.expectEqualSlices(i32, &.{ 6, 6, 6, 6 }, l2.w2.halves);
        const l3 = plan.moeLayer(3).?;
        try std.testing.expectEqualSlices(i32, &.{ 6, 6, 8, 6 }, l3.w1.halves);
        try std.testing.expectEqual(@as(u32, 3), l3.w1.histogram[6]);
        try std.testing.expectEqual(@as(u32, 1), l3.w1.histogram[8]);
        // a tile is 16 * bits int16 words, i.e. 16 * halves bytes; K x N is 256 x 256 whole, 256 x 128 on a TP2 rank
        const tiles: u64 = if (world == 1) 16 * 16 else 16 * 8;
        var at: u64 = 0;
        for (l3.w1.halves, l3.w1.trellis_off) |h, off| {
            try std.testing.expectEqual(at, off);
            at += tiles * 16 * @as(u64, @intCast(h));
        }
        try std.testing.expectEqual(at, plan.slots[l3.w1.trellis_slot].bytes);
        try std.testing.expectEqual(@as(u64, 4 * 256 * 2), plan.slots[l3.w1.suh_slot].bytes);
        // every expert's pointer lands on its own trellis: the matrix table agrees with the stacked table
        for (0..4) |e| try std.testing.expectEqual(l3.w1.trellis_off[e], plan.matrix(.w1, 3, @intCast(e)).?.trellis_off);
    }
}

test "three shards plan the same: layer 1's down projections are one width on a TP2 rank's half rows" {
    const io = std.testing.io;
    const c = mini();
    var pack = Pack.init();
    defer pack.deinit();
    try pack.all(&c);
    var dir = try Dir.init(io);
    defer dir.deinit();
    try write(io, dir.tmp.dir, &pack, .{ .shards = 3 });
    var cat = try load.Catalog.open(gpa, io, dir.path);
    defer cat.deinit();
    var plan = try rankPlan(&cat, &c, 1, 2);
    defer plan.deinit();
    try std.testing.expectEqual(@as(u32, 4), plan.moeLayer(1).?.w2.histogram[8]);
    try std.testing.expectEqual(@as(u32, 128), plan.moeLayer(1).?.w2.k);
}

// ---- what a broken checkpoint must not do ----

fn openFails(want: anyerror, pack: *Pack, w: Write) !void {
    const io = std.testing.io;
    var dir = try Dir.init(io);
    defer dir.deinit();
    try write(io, dir.tmp.dir, pack, w);
    try std.testing.expectError(want, load.Catalog.open(gpa, io, dir.path));
}

fn planFails(want: anyerror, pack: *Pack) !void {
    const io = std.testing.io;
    const c = mini();
    var dir = try Dir.init(io);
    defer dir.deinit();
    try write(io, dir.tmp.dir, pack, .{});
    var cat = try load.Catalog.open(gpa, io, dir.path);
    defer cat.deinit();
    try std.testing.expectError(want, rankPlan(&cat, &c, 0, 2));
}

const q0 = "language_model.model.layers.0.self_attn.q_proj";
const e1 = "language_model.model.layers.1.block_sparse_moe.experts.1.w1";

test "truncated files fail closed: a short data region, a header longer than the file, a file shorter than 8 bytes" {
    const io = std.testing.io;
    var pack = Pack.init();
    defer pack.deinit();
    try pack.all(&mini());
    try openFails(error.BadSafetensors, &pack, .{ .cut_last = 100 });
    try openFails(error.BadSafetensors, &pack, .{ .cut_last = 1 });
    var dir = try Dir.init(io);
    defer dir.deinit();
    var lie: [20]u8 = @splat(0);
    std.mem.writeInt(u64, lie[0..8], 1000, .little);
    try dir.tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &lie });
    try std.testing.expectError(error.BadSafetensors, load.Catalog.open(gpa, io, dir.path));
    try dir.tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = "abc" });
    try std.testing.expectError(error.BadSafetensors, load.Catalog.open(gpa, io, dir.path));
}

test "duplicate and aliased tensors fail closed" {
    const io = std.testing.io;
    // the same name in two shards (no index could name both)
    var dup = Pack.init();
    defer dup.deinit();
    try dup.add("a.weight", "U8", &.{4});
    try dup.add("x.weight", "U8", &.{4});
    try dup.add("y.weight", "U8", &.{4});
    try dup.add("a.weight", "U8", &.{4});
    try openFails(error.DuplicateTensor, &dup, .{ .index = false });
    // the same key twice inside one header
    var dir = try Dir.init(io);
    defer dir.deinit();
    const twice = "{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[4,8]}}";
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(gpa);
    try file.appendNTimes(gpa, 0, 8);
    std.mem.writeInt(u64, file.items[0..8], twice.len, .little);
    try file.appendSlice(gpa, twice);
    try file.appendNTimes(gpa, 0, 8);
    try dir.tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file.items });
    try std.testing.expectError(error.DuplicateTensor, load.Catalog.open(gpa, io, dir.path));
    // two names over the same bytes
    const alias = "{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[2,6]}}";
    file.clearRetainingCapacity();
    try file.appendNTimes(gpa, 0, 8);
    std.mem.writeInt(u64, file.items[0..8], alias.len, .little);
    try file.appendSlice(gpa, alias);
    try file.appendNTimes(gpa, 0, 6);
    try dir.tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file.items });
    try std.testing.expectError(error.OverlappingTensors, load.Catalog.open(gpa, io, dir.path));
}

test "an index that disagrees with the headers is refused" {
    const io = std.testing.io;
    var pack = Pack.init();
    defer pack.deinit();
    try pack.all(&mini());
    var dir = try Dir.init(io);
    defer dir.deinit();
    try write(io, dir.tmp.dir, &pack, .{});
    // an index naming one tensor no header holds, then one whose names are not plain shard files
    try dir.tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"ghost\":\"model-00001-of-00002.safetensors\"}}" });
    try std.testing.expectError(error.IndexMismatch, load.Catalog.open(gpa, io, dir.path));
    try dir.tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"../x\":\"../escape.safetensors\"}}" });
    try std.testing.expectError(error.BadIndex, load.Catalog.open(gpa, io, dir.path));
}

test "misshaped tensors fail closed with the reason that names them" {
    const c = mini();
    const cases = [_]struct { name: []const u8, dtype: []const u8, shape: []const usize, want: anyerror }{
        .{ .name = q0 ++ ".trellis", .dtype = "I16", .shape = &.{ 16, 32, 100 }, .want = error.BadTrellis }, // not 16 * bits
        .{ .name = q0 ++ ".trellis", .dtype = "I32", .shape = &.{ 16, 32, 96 }, .want = error.BadTrellis },
        .{ .name = q0 ++ ".trellis", .dtype = "I16", .shape = &.{ 16, 32 }, .want = error.BadTrellis },
        .{ .name = q0 ++ ".trellis", .dtype = "I16", .shape = &.{ 16, 32, 32 }, .want = error.UnsupportedWidth }, // 2 bits
        .{ .name = q0 ++ ".trellis", .dtype = "I16", .shape = &.{ 16, 32, 88 }, .want = error.UnsupportedWidth }, // 5.5 bits
        .{ .name = e1 ++ ".trellis", .dtype = "I16", .shape = &.{ 16, 16, 128 }, .want = error.UnsupportedWidth }, // expert, 8 bits
        .{ .name = q0 ++ ".trellis", .dtype = "I16", .shape = &.{ 17, 32, 96 }, .want = error.NotAligned }, // K 272
        .{ .name = q0 ++ ".trellis", .dtype = "I16", .shape = &.{ 8, 32, 96 }, .want = error.MisshapedTensor }, // K 128, not 256
        .{ .name = q0 ++ ".suh", .dtype = "F16", .shape = &.{255}, .want = error.MisshapedTensor },
        .{ .name = q0 ++ ".svh", .dtype = "BF16", .shape = &.{512}, .want = error.MisshapedTensor },
        .{ .name = q0 ++ ".mul1", .dtype = "I32", .shape = &.{2}, .want = error.MisshapedTensor },
        .{ .name = q0 ++ ".mul1", .dtype = "I16", .shape = &.{}, .want = error.MisshapedTensor },
        .{ .name = "language_model.lm_head.trellis", .dtype = "I16", .shape = &.{ 16, 16, 80 }, .want = error.UnsupportedWidth }, // head_bits is 6
        .{ .name = "language_model.model.layers.1.block_sparse_moe.gate.weight", .dtype = "F16", .shape = &.{ 4, 256 }, .want = error.MisshapedTensor },
        .{ .name = "language_model.model.layers.1.block_sparse_moe.gate.weight", .dtype = "F32", .shape = &.{ 3, 256 }, .want = error.MisshapedTensor },
        .{ .name = "language_model.model.layers.1.self_attn.index_q_proj.weight", .dtype = "F32", .shape = &.{ 256, 256 }, .want = error.MisshapedTensor },
        .{ .name = "language_model.model.layers.1.self_attn.index_k_proj.weight", .dtype = "F16", .shape = &.{ 64, 256 }, .want = error.MisshapedTensor },
        .{ .name = "language_model.model.embed_tokens.weight", .dtype = "BF16", .shape = &.{ 256, 255 }, .want = error.MisshapedTensor },
        .{ .name = "language_model.model.layers.2.input_layernorm.weight", .dtype = "F32", .shape = &.{256}, .want = error.MisshapedTensor },
    };
    for (cases) |k| {
        var pack = Pack.init();
        defer pack.deinit();
        try pack.all(&c);
        try pack.reshape(k.name, k.dtype, k.shape);
        try planFails(k.want, &pack);
    }
}

test "missing, extra and foreign tensors" {
    const c = mini();
    var pack = Pack.init();
    defer pack.deinit();
    try pack.all(&c);
    pack.drop(e1 ++ ".svh");
    try planFails(error.MissingTensor, &pack);

    var extra = Pack.init();
    defer extra.deinit();
    try extra.all(&c);
    try extra.add("language_model.model.layers.1.self_attn.q_proj.bias", "F16", &.{512});
    try planFails(error.UnexpectedTensor, &extra);

    var moe0 = Pack.init(); // MoE tensors in a layer the config calls dense
    defer moe0.deinit();
    try moe0.all(&c);
    try moe0.plain(.router, 0, "F32", &.{ 4, 256 });
    try planFails(error.UnexpectedTensor, &moe0);

    // a vision tower beside the text model is left unread and counted
    const io = std.testing.io;
    var tower = Pack.init();
    defer tower.deinit();
    try tower.all(&c);
    try tower.add("vision_tower.blocks.0.attn.qkv.weight", "BF16", &.{ 3, 3 });
    var dir = try Dir.init(io);
    defer dir.deinit();
    try write(io, dir.tmp.dir, &tower, .{});
    var cat = try load.Catalog.open(gpa, io, dir.path);
    defer cat.deinit();
    var plan = try rankPlan(&cat, &c, 0, 1);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 1), plan.ignored);
}

test "a codebook marker that is not the mul1 multiplier stops the stream before any byte is copied" {
    const io = std.testing.io;
    const c = mini();
    var pack = Pack.init();
    defer pack.deinit();
    try pack.all(&c);
    std.mem.writeInt(u32, pack.find(e1 ++ ".mul1").data[0..4], 0xCBAC1FED, .little); // the mcg marker
    var dir = try Dir.init(io);
    defer dir.deinit();
    try write(io, dir.tmp.dir, &pack, .{});
    var cat = try load.Catalog.open(gpa, io, dir.path);
    defer cat.deinit();
    var plan = try rankPlan(&cat, &c, 0, 2);
    defer plan.deinit();
    var host = try Host.init(&plan);
    defer host.deinit();
    var streamer = try load.Streamer.init(gpa, &cat, 64 * 1024);
    defer streamer.deinit();
    try std.testing.expectError(error.BadCodebookMarker, streamer.run(&plan, host.sink()));
    for (host.written) |w| try std.testing.expectEqual(@as(u64, 0), w);
}

test "a plan cut for more ranks than a width has units is refused before any file is read" {
    const c = mini();
    try std.testing.expectError(error.EmptyShare, shard.Share.of(&c, 2, 3)); // 2 kv heads
    try std.testing.expectError(error.EmptyShare, shard.Share.of(&c, 2, 4));
}
