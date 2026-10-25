//! The coalescing streamer against the one-run-at-a-time streamer on the synthetic mini checkpoint, no GPU.
const std = @import("std");
const load = @import("load.zig");
const synth = @import("load_synth_test.zig");

const gpa = std.testing.allocator;

test "the coalescing streamer puts the bytes of the one-run-at-a-time streamer, whatever the readers, buffers and gap" {
    const io = std.testing.io;
    const c = synth.mini();
    var pack = synth.Pack.init();
    defer pack.deinit();
    try pack.all(&c);
    const Case = struct { shards: usize, world: usize, o: load.StreamOptions };
    const cases = [_]Case{
        .{ .shards = 2, .world = 1, .o = .{} },
        .{ .shards = 2, .world = 2, .o = .{} },
        .{ .shards = 3, .world = 2, .o = .{ .chunk = 16 * 1024, .threads = 3, .gap = 0 } },
        .{ .shards = 3, .world = 2, .o = .{ .chunk = 16 * 1024, .threads = 1, .ahead = 0, .gap = 4096 } },
        .{ .shards = 5, .world = 2, .o = .{ .chunk = 32 * 1024, .threads = 0, .gap = 1 << 20 } },
        .{ .shards = 1, .world = 2, .o = .{ .chunk = 20 * 1024, .threads = 8, .ahead = 1, .gap = 100, .put_max = 5000 } },
    };
    for (cases) |case| {
        var dir = try synth.Dir.init(io);
        defer dir.deinit();
        try synth.write(io, dir.tmp.dir, &pack, .{ .shards = case.shards });
        var cat = try load.Catalog.open(gpa, io, dir.path);
        defer cat.deinit();
        for (0..case.world) |rank| {
            var plan = try synth.rankPlan(&cat, &c, rank, case.world);
            defer plan.deinit();
            var old = try synth.Host.init(&plan);
            defer old.deinit();
            var reference = try load.Streamer.init(gpa, &cat, 64 * 1024);
            defer reference.deinit();
            try reference.runSerial(&plan, old.sink());
            var new = try synth.Host.init(&plan);
            defer new.deinit();
            var streamer = try load.Streamer.initWith(gpa, &cat, case.o);
            defer streamer.deinit();
            try streamer.run(&plan, new.sink());
            for (old.bufs, new.bufs, old.written, new.written) |a, b, wa, wb| {
                try std.testing.expectEqualSlices(u8, a, b);
                try std.testing.expectEqual(wa, wb);
            }
            try synth.expectStreamed(&c, &pack, &plan, &new);
            var runs: u64 = 0;
            var wanted: u64 = 0;
            for (plan.items) |it| {
                runs += it.span.count;
                wanted += it.span.bytes();
            }
            try std.testing.expectEqual(wanted, streamer.stats.wanted_bytes);
            try std.testing.expect(streamer.stats.read_bytes >= wanted);
            try std.testing.expectEqual(plan.matrices.len, streamer.stats.markers);
            // coalescing is the point: with a gap the many short runs of a cut share reads
            if (case.o.gap >= 4096 and case.world == 2) try std.testing.expect(streamer.stats.windows < runs);
        }
    }
}

test "a read that fails stops the coalescing streamer with that error and no put follows it" {
    const io = std.testing.io;
    const c = synth.mini();
    var pack = synth.Pack.init();
    defer pack.deinit();
    try pack.all(&c);
    var dir = try synth.Dir.init(io);
    defer dir.deinit();
    try synth.write(io, dir.tmp.dir, &pack, .{});
    var cat = try load.Catalog.open(gpa, io, dir.path);
    defer cat.deinit();
    var plan = try synth.rankPlan(&cat, &c, 1, 2);
    defer plan.deinit();
    var host = try synth.Host.init(&plan);
    defer host.deinit();
    // the shards are emptied under the open descriptors after their headers were read: every read hits the end of file
    for (0..2) |i| {
        var fb: [48]u8 = undefined;
        try dir.tmp.dir.writeFile(io, .{ .sub_path = synth.shardFile(&fb, i, 2), .data = "" });
    }
    var streamer = try load.Streamer.initWith(gpa, &cat, .{ .chunk = 16 * 1024, .threads = 3 });
    defer streamer.deinit();
    if (streamer.run(&plan, host.sink())) |_| return error.TestUnexpectedResult else |_| {}
}
