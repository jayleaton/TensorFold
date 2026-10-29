const std = @import("std");
const t = std.testing;
const block = @import("block.zig");
const listing = @import("widths_listing.zig");
const pf = @import("forward_prefill.zig");
const hooks = @import("forward_hooks.zig");
test "rank boot completes prefill roles before admitting image segments" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var widths: block.Widths = .{};
    try listing.parseDense(a, &widths, @embedFile("fixtures/dense-k2.txt"));
    try listing.parseExperts(a, &widths, @embedFile("fixtures/expert-k2.txt"));
    for (0..40) |L| try widths.gm.put(a, @intCast(L), .{ (1 << 3) | (1 << 4), (1 << 3) | (1 << 4) });
    var cfg: @import("config.zig").Config = .{};
    const saved_hooks = hooks.set;
    hooks.install(.{ .prefill = &pf.hooks_table });
    defer hooks.install(saved_hooks);
    for ([_]bool{ false, true }) |replay| for (0..2) |rank| {
        var f = @import("forward.zig").Forward.init(a, &cfg, &widths, .{ .world = 2, .rank = @intCast(rank), .prefill_rows = 232, .limit = 8192, .rope_rows = 10240, .r1 = replay, .gm_v2 = .one }, undefined, undefined);
        defer f.deinitAfterDrain();
        try pf.attach(&f);
        if (replay) {
            pf.state(&f).mode = .replay;
            pf.state(&f).k4 = true;
        }
        var plan = try f.plan(&.{1});
        try @import("boot_plan.zig").prefill(&f, &plan, false);
        f.images = true;
        f.image_bias = true;
        try t.expectError(error.ImagesUnplanned, @import("vision_open.zig").checkPlan(&f, &plan));
        try t.expectError(error.ImagesUnplanned, @import("boot_plan.zig").prefill(&f, &plan, false));
        try @import("boot_plan.zig").prefill(&f, &plan, true);
        try @import("vision_open.zig").checkPlan(&f, &plan);
    };
}
