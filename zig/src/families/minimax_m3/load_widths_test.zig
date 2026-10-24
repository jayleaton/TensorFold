//! Width admission: every dense matrix (k and v included) may be 3 to 8 bits, a routed expert 3 to 6.
const std = @import("std");
const load = @import("load.zig");
const names = @import("names.zig");
const synth = @import("load_synth_test.zig");

const gpa = std.testing.allocator;

/// Sets one group's trellis to `halves` (2 * bits) over its unchanged K x N tiles.
fn widen(pack: *synth.Pack, proj: names.Proj, layer: u32, expert: u32, halves: usize) !void {
    var b: names.Buf = undefined;
    var b2: names.Buf = undefined;
    const name = names.part(&b2, names.group(&b, proj, layer, expert), .trellis);
    const t = pack.find(name);
    try pack.reshape(name, "I16", &.{ t.shape[0], t.shape[1], 8 * halves });
}

fn plan(cat: *load.Catalog, rank: usize, world: usize) !load.Plan {
    const c = synth.mini();
    return synth.rankPlan(cat, &c, rank, world);
}

fn open(io: std.Io, dir: *synth.Dir, pack: *const synth.Pack) !load.Catalog {
    try synth.write(io, dir.tmp.dir, pack, .{});
    return load.Catalog.open(gpa, io, dir.path);
}

test "8-bit k and v (and 7-bit q) plan, cut for TP2 and stream the same tiles as any other width" {
    const io = std.testing.io;
    const c = synth.mini();
    var pack = synth.Pack.init();
    defer pack.deinit();
    try pack.all(&c);
    for (0..c.layers) |l| {
        try widen(&pack, .k, @intCast(l), 0, 16);
        try widen(&pack, .v, @intCast(l), 0, 16);
        try widen(&pack, .q, @intCast(l), 0, 14);
    }
    var dir = try synth.Dir.init(io);
    defer dir.deinit();
    var cat = try open(io, &dir, &pack);
    defer cat.deinit();
    for ([_]usize{ 1, 2 }) |world| for (0..world) |rank| {
        var p = try plan(&cat, rank, world);
        defer p.deinit();
        for (0..c.layers) |l| {
            for ([_]names.Proj{ .k, .v }) |proj| {
                const m = p.matrix(proj, @intCast(l), 0).?;
                try std.testing.expectEqual(@as(u8, 16), m.halves);
                try std.testing.expectEqual(@as(u32, @intCast(c.kv_heads * c.head_dim / world)), m.n); // whole kv heads a rank
                try std.testing.expectEqual(@as(u64, (c.hidden / 16) * (m.n / 16) * 16 * 16), p.slots[m.trellis].bytes);
            }
            try std.testing.expectEqual(@as(u8, 14), p.matrix(.q, @intCast(l), 0).?.halves);
        }
        var host = try synth.Host.init(&p);
        defer host.deinit();
        var streamer = try load.Streamer.init(gpa, &cat, 32 * 1024);
        defer streamer.deinit();
        try streamer.run(&p, host.sink());
        try synth.expectStreamed(&c, &pack, &p, &host);
    };
}

test "a routed expert past 6 bits, a dense matrix past 8 or under 3 bits are refused" {
    const io = std.testing.io;
    const c = synth.mini();
    const cases = [_]struct { proj: names.Proj, layer: u32, expert: u32, halves: usize, want: anyerror }{
        .{ .proj = .w1, .layer = 1, .expert = 0, .halves = 16, .want = error.UnsupportedWidth },
        .{ .proj = .w2, .layer = 2, .expert = 3, .halves = 14, .want = error.UnsupportedWidth },
        .{ .proj = .k, .layer = 0, .expert = 0, .halves = 4, .want = error.UnsupportedWidth },
        .{ .proj = .v, .layer = 1, .expert = 0, .halves = 18, .want = error.BadTrellis }, // 144 words: past 16 * 8
    };
    for (cases) |k| {
        var pack = synth.Pack.init();
        defer pack.deinit();
        try pack.all(&c);
        try widen(&pack, k.proj, k.layer, k.expert, k.halves);
        var dir = try synth.Dir.init(io);
        defer dir.deinit();
        var cat = try open(io, &dir, &pack);
        defer cat.deinit();
        try std.testing.expectError(k.want, plan(&cat, 0, 2));
    }
}
