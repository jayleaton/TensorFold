//! Host registration for the JIT mask operator; device execution and compiled resource receipts remain unrun.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const masks = @import("grammar_rows.zig");

pub const launch = contract.registry.dsv41_grammar.launch;

pub const spec: c.Spec = .{
    .name = launch.name,
    .accum = .none,
    .depth = depth,
    .out = .f32,
    .in_place = true,
    .nan = .{ .stated = "allowed logits retain their bits including NaN; disallowed logits become negative infinity" },
    .tiles = .{ .{ .size = 16 }, .{ .size = 32, .step = 32, .min = 32 }, .{} },
    .served = &.{
        .{ .rows = 1, .n = 129280, .k = 0 },
        .{ .rows = 16, .n = 64640, .k = 0 },
        .{ .rows = 64, .n = 32384, .k = 0 },
    },
    .sums = .{ .f16 = 0, .bf16 = 0, .why = "bitmask stores only; no accumulation" },
};

pub const DeviceGate = enum { unrun };
pub const device_gate: DeviceGate = .unrun;
pub const resource_receipt: ?contract.limits.Usage = null;

fn depth(_: c.Shape) usize {
    return 0;
}

fn reference(input: []const f32, output: []f32, idx: []const u32, bits: []const u32, width: usize, words: usize, lo: usize) void {
    @memcpy(output, input);
    for (idx, 0..) |row, r| for (0..width) |col| {
        const token = lo + col;
        if ((bits[r * words + token / 32] >> @intCast(token % 32)) & 1 == 0)
            output[row * width + col] = -std.math.inf(f32);
    };
}

test "grammar contract: registered launch and arithmetic contract name the production kernel" {
    try std.testing.expectEqualStrings(masks.symbol, launch.symbol);
    try std.testing.expectEqual(masks.threads, launch.threads);
    var found = false;
    for (contract.registry.all_cuda) |entry| {
        if (std.mem.eql(u8, entry.launch.symbol, launch.symbol)) found = true;
    }
    try std.testing.expect(found);
    try std.testing.expectEqual(@as(u32, 128), launch.threads);
    try std.testing.expect(spec.in_place);
    try std.testing.expectEqual(c.Accum.none, spec.accum);
    try std.testing.expectEqual(@as(?contract.limits.Usage, null), resource_receipt);
    try std.testing.expectEqual(.unrun, device_gate);
}

test "grammar contract: nonfinite, repeat, shape edges, guards and corruption control on host" {
    const a = std.testing.allocator;
    const shapes = try c.shapes(a, spec);
    defer a.free(shapes);
    for (shapes) |shape| {
        const width: u32 = @intCast(shape.n);
        const lo: u32 = 128;
        const words = (lo + width + 31) / 32;
        const count: usize = @min(shape.rows, 3);
        const len = (count + 2) * width;
        const input = try a.alloc(f32, len);
        defer a.free(input);
        const got = try a.alloc(f32, len + 2);
        defer a.free(got);
        const want = try a.alloc(f32, len);
        defer a.free(want);
        const bits = try a.alloc(u32, count * words);
        defer a.free(bits);
        const saved = try a.alloc(u32, bits.len);
        defer a.free(saved);
        for (input, 0..) |*v, i| v.* = switch (i % 5) {
            0 => @bitCast(@as(u32, 0x7fc01234)),
            1 => std.math.inf(f32),
            2 => -std.math.inf(f32),
            3 => -0.0,
            else => @floatFromInt(i % 97),
        };
        for (bits, 0..) |*v, i| v.* = switch (i % 4) {
            0 => 0,
            1 => std.math.maxInt(u32),
            2 => 0x55555555,
            else => 0xaaaaaaaa,
        };
        @memcpy(saved, bits);
        const indices = [_]u32{ 1, 2, 3 };
        reference(input, want, indices[0..count], bits, width, words, lo);
        for (0..5) |_| {
            got[0] = 12345;
            got[len + 1] = 54321;
            @memcpy(got[1 .. len + 1], input);
            masks.maskRows(got[1 .. len + 1], width, indices[0..count], bits, words, lo);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got[1 .. len + 1]));
            try std.testing.expectEqualSlices(u32, saved, bits);
            try std.testing.expectEqual(@as(f32, 12345), got[0]);
            try std.testing.expectEqual(@as(f32, 54321), got[len + 1]);
        }
        const damaged = 1 + @as(usize, indices[0]) * width;
        got[damaged] = 9;
        try std.testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got[1 .. len + 1])));
    }
}
