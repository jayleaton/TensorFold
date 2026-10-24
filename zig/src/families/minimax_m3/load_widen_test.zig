//! An f16 or bf16 router (and bias) is widened to the fp32 the router kernels read, losslessly, as it streams.
const std = @import("std");
const load = @import("load.zig");
const names = @import("names.zig");
const synth = @import("load_synth_test.zig");

const gpa = std.testing.allocator;

fn bits(x: f32) u32 {
    return @bitCast(x);
}

test "the widening keeps every value: zeros, subnormals, the extremes, infinities and NaN" {
    var out: [8]f32 = undefined;
    const h = [_]u16{ 0x0000, 0x8000, 0x0001, 0x3c00, 0x7bff, 0xfc00, 0x7e00, 0xb555 };
    const want_h = [_]u32{ 0x00000000, 0x80000000, 0x33800000, 0x3f800000, 0x477fe000, 0xff800000, 0x7fc00000, 0xbeaaa000 };
    const got_h = load.applyFold(&out, .f16, std.mem.sliceAsBytes(&h));
    for (got_h, want_h) |g, w| try std.testing.expectEqual(w, bits(g));
    const bf = [_]u16{ 0x0000, 0x8000, 0x0001, 0x3f80, 0x7f7f, 0xff80, 0x7fc0, 0xc0a0 };
    const got_b = load.applyFold(&out, .bf16, std.mem.sliceAsBytes(&bf));
    for (got_b, bf) |g, w| try std.testing.expectEqual(@as(u32, w) << 16, bits(g));
    try std.testing.expectEqual(@as(usize, 8), got_b.len);
}

test "only the router and its bias are widened; index projections stay fp16 and norms take the Gemma fold" {
    try std.testing.expectEqual(load.Fold.f16, load.foldOf(true, .router, .f16));
    try std.testing.expectEqual(load.Fold.bf16, load.foldOf(true, .router, .bf16));
    try std.testing.expectEqual(load.Fold.none, load.foldOf(true, .router, .f32));
    try std.testing.expectEqual(load.Fold.f16, load.foldOf(false, .router_bias, .f16));
    try std.testing.expectEqual(load.Fold.bf16, load.foldOf(true, .router_bias, .bf16));
    try std.testing.expectEqual(load.Fold.none, load.foldOf(true, .index_q_proj, .f16));
    try std.testing.expectEqual(load.Fold.none, load.foldOf(true, .embed, .bf16));
    try std.testing.expectEqual(load.Fold.gemma, load.foldOf(true, .input_ln, .bf16));
    try std.testing.expectEqual(load.Fold.none, load.foldOf(false, .input_ln, .bf16));
}

test "f16 and bf16 routers plan as fp32 slots and stream their values widened, for one rank and two" {
    const io = std.testing.io;
    const c = synth.mini();
    var pack = synth.Pack.init();
    defer pack.deinit();
    try pack.all(&c);
    var b: names.Buf = undefined;
    // layer 1 stores fp16 (the MM3.1 pack's form), layer 2 bf16 with an fp16 bias, layer 3 keeps fp32
    try pack.reshape(names.plain(&b, .router, 1), "F16", &.{ c.experts, c.hidden });
    try pack.reshape(names.plain(&b, .router, 2), "BF16", &.{ c.experts, c.hidden });
    try pack.reshape(names.plain(&b, .router_bias, 2), "F16", &.{c.experts});
    var dir = try synth.Dir.init(io);
    defer dir.deinit();
    try synth.write(io, dir.tmp.dir, &pack, .{});
    var cat = try load.Catalog.open(gpa, io, dir.path);
    defer cat.deinit();
    for ([_]usize{ 1, 2 }) |world| for (0..world) |rank| for ([_]usize{ 32 * 1024, load.Streamer.default_chunk }) |chunk| {
        var plan = try synth.rankPlan(&cat, &c, rank, world);
        defer plan.deinit();
        var routers: usize = 0;
        for (plan.plain) |p| if (p.role == .router or p.role == .router_bias) {
            try std.testing.expect(p.dtype == .f32 and !p.folded);
            const n: u64 = if (p.role == .router) c.experts * c.hidden else c.experts;
            try std.testing.expectEqual(4 * n, plan.slots[p.slot].bytes);
            routers += 1;
        };
        try std.testing.expectEqual(@as(usize, 6), routers);
        for (0..2) |serial| {
            var host = try synth.Host.init(&plan);
            defer host.deinit();
            var streamer = try load.Streamer.init(gpa, &cat, chunk);
            defer streamer.deinit();
            if (serial == 1) try streamer.runSerial(&plan, host.sink()) else try streamer.run(&plan, host.sink());
            try synth.expectStreamed(&c, &pack, &plan, &host);
        }
    };
}
