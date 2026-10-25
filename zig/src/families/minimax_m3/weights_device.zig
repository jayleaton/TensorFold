//! A rank's weights on the device: a buffer per slot, direct I/O, page cache dropped per shard, a header digest.
const std = @import("std");
const cuda = @import("cuda");
const load = @import("load.zig");
const shard_ = @import("shard.zig");
const weights = @import("weights.zig");
const ffn = @import("ffn.zig");
const Config = @import("config.zig").Config;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// Bytes put between stream synchronizations: bounds the pinned staging the async copies hold.
pub const sync_every: u64 = 256 << 20;

/// sha256 over each tensor's name, shard, dtype, shape and byte range in name order; no data byte is read (NFS-safe).
pub fn digest(gpa: std.mem.Allocator, cat: *const load.Catalog) ![32]u8 {
    const keys = try gpa.alloc([]const u8, cat.map.count());
    defer gpa.free(keys);
    var it = cat.map.keyIterator();
    for (keys) |*k| k.* = it.next().?.*;
    std.mem.sort([]const u8, keys, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.less);
    var h = Sha256.init(.{});
    for (keys) |k| {
        const r = cat.map.get(k).?;
        h.update(k);
        h.update(&.{0});
        const fields = [_]u64{ r.shard, @intFromEnum(r.dtype), r.rank, r.begin, r.len };
        for (fields) |f| h.update(std.mem.asBytes(&std.mem.nativeToLittle(u64, f)));
        for (r.shape[0..r.rank]) |d| h.update(std.mem.asBytes(&std.mem.nativeToLittle(u64, d)));
    }
    return h.finalResult();
}

/// Streams `plan` through `sink`, calling `ended` after each shard's last run.
pub fn feed(gpa: std.mem.Allocator, cat: *const load.Catalog, plan: *const load.Plan, sink: load.Sink, ended: ?load.ShardEnd) !void {
    var s = try load.Streamer.init(gpa, cat, load.Streamer.default_chunk);
    defer s.deinit();
    s.ended = ended;
    try s.run(plan, sink);
}

/// The shard-end hook that empties the page cache of the shard's file.
pub const CacheDrop = struct {
    cat: *const load.Catalog,

    pub fn hook(d: *CacheDrop) load.ShardEnd {
        return .{ .ctx = d, .call = call };
    }

    fn call(ctx: *anyopaque, shard: u16) void {
        const d: *CacheDrop = @ptrCast(@alignCast(ctx));
        d.cat.files.items[shard].dropCache();
    }
};

/// The sink onto per-slot device buffers: async copies on `stream`, synchronized every `sync_every` bytes.
const Upload = struct {
    bufs: []const cuda.DeviceBuffer,
    stream: cuda.Stream,
    since: u64 = 0,

    fn put(ctx: *anyopaque, slot: u32, offset: u64, bytes: []const u8) anyerror!void {
        const u: *Upload = @ptrCast(@alignCast(ctx));
        try u.bufs[slot].uploadAsync(@intCast(offset), bytes, u.stream.handle);
        u.since += bytes.len;
        if (u.since >= sync_every) {
            try u.stream.synchronize();
            u.since = 0;
        }
    }
};

pub const Loaded = struct {
    arena: std.heap.ArenaAllocator,
    bufs: []cuda.DeviceBuffer,
    tables: []cuda.DeviceBuffer,
    weights: weights.Weights,
    digest: [32]u8,
    bytes: u64, // device bytes of the slots

    pub fn deinit(l: *Loaded) void {
        for (l.bufs) |*b| b.free();
        for (l.tables) |*b| b.free();
        l.arena.deinit();
        l.* = undefined;
    }
};

/// Plans rank `share` (the first `layers` layers; 0: all), fills the slots, uploads the expert tables. UNEXERCISED.
pub fn loadWeights(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, dir: []const u8, c: *const Config, share: shard_.Share, layers: usize) !Loaded {
    var cat = try load.Catalog.open(gpa, io, dir);
    defer cat.deinit();
    const sum = try digest(gpa, &cat);
    var first: [64]u16 = undefined;
    const subset: []const u16 = if (layers == 0 or layers >= c.layers) &.{} else blk: {
        for (first[0..layers], 0..) |*f, i| f.* = @intCast(i);
        break :blk first[0..layers];
    };
    var plan = try load.Plan.build(gpa, &cat, c, share, .{ .layers = subset });
    defer plan.deinit();
    var out: Loaded = .{ .arena = .init(gpa), .bufs = &.{}, .tables = &.{}, .weights = undefined, .digest = sum, .bytes = plan.slotBytes() };
    errdefer out.arena.deinit();
    const a = out.arena.allocator();
    var made: std.ArrayList(cuda.DeviceBuffer) = .empty;
    errdefer made.deinit(gpa);
    errdefer for (made.items) |*b| b.free();
    const bases = try a.alloc(u64, plan.slots.len);
    for (plan.slots, bases) |s, *base| {
        const b = try cuda.DeviceBuffer.alloc(d, @intCast(s.bytes));
        try made.append(gpa, b);
        base.* = b.ptr;
    }
    var up: Upload = .{ .bufs = made.items, .stream = stream };
    var drop: CacheDrop = .{ .cat = &cat };
    try feed(gpa, &cat, &plan, .{ .ctx = &up, .put = Upload.put }, drop.hook());
    try stream.synchronize();
    var described = try weights.describe(a, &plan, c, bases, if (layers == 0) c.layers else layers);
    var tab: std.ArrayList(cuda.DeviceBuffer) = .empty;
    errdefer tab.deinit(gpa);
    errdefer for (tab.items) |*b| b.free();
    for (described.tables) |t| {
        const packed_ = try weights.packTables(gpa, t);
        defer gpa.free(packed_);
        const b = try cuda.DeviceBuffer.fromHost(d, packed_);
        try tab.append(gpa, b);
        weights.attachTables(&described.weights, t, weights.tablePtrs(b.ptr, t.gate.len), ffn.Switches{});
    }
    out.bufs = try a.dupe(cuda.DeviceBuffer, made.items);
    out.tables = try a.dupe(cuda.DeviceBuffer, tab.items);
    made.deinit(gpa);
    tab.deinit(gpa);
    out.weights = described.weights;
    return out;
}
