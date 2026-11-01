//! Proof 2: the multiply and loop replies on the real frontend and the real CPU circuit (TF_RUNNER_CIRCUIT).
const std = @import("std");
const rt = @import("runner_test.zig");
const circuit = @import("runner_circuit.zig");
const backend = @import("runner_circuit_backend.zig");
const frontend = @import("runner_frontend.zig");

const gpa = std.testing.allocator;

/// One reply per lead on the circuit at TF_RUNNER_CIRCUIT; each must equal its lead, `want`, then the model's tail.
fn prove(leads: []const []const u8, wants: []const []const u8) !void {
    const path = std.c.getenv("TF_RUNNER_CIRCUIT") orelse {
        std.debug.print("skip runner proof: TF_RUNNER_CIRCUIT unset\n", .{});
        return error.SkipZigTest;
    };
    var c = try circuit.Circuit.load(gpa, std.mem.span(path));
    defer c.deinit(gpa);
    const b = try backend.backend(gpa, &c, 2);
    defer backend.destroy(b);
    var fake: rt.Back = .{};
    defer fake.deinit();
    const models = try gpa.alloc(rt.Model, leads.len);
    defer gpa.free(models);
    for (models, leads) |*m, l| m.* = .{ .lead = l };
    // TF_RUNNER_LLD: an absolute ld.lld for hosts without lld beside clang (the image installs lld-18 there)
    var ctx: frontend.Context = .{ .io = std.testing.io };
    if (std.c.getenv("TF_RUNNER_LLD")) |l| ctx.compiler.linker = std.mem.span(l);
    const texts = try rt.replies(models, .{ .frontend = ctx.frontend(), .backend = b, .wait_s = 600 }, &fake);
    defer rt.freeAll(texts);
    try std.testing.expectEqual(@as(u32, 0), fake.started);
    for (texts, leads, wants) |got, l, w| {
        const want = try std.mem.concat(gpa, u8, &.{ l, w, "\nDone." });
        defer gpa.free(want);
        try std.testing.expectEqualStrings(want, got);
    }
}

test "runner proof: a 30-digit multiply cell's Result: is the exact product, computed by the circuit" {
    try prove(&.{rt.mul_cell}, &.{rt.product});
}

test "runner proof: loop cells (a generator sum and an accumulator) print 285 through the circuit" {
    const sum = "```python\nprint(sum(i*i for i in range(10)))\n```\nOutput:\n";
    const acc = "```python\ntotal = 0\nfor i in range(10):\n    total += i * i\nprint(total)\n```\nResult:\n";
    try prove(&.{ sum, acc }, &.{ "285", "285" });
}
