//! MiniMax-M3's host checks: config, token ids and the unported-operator refusal, no GPU.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/minimax_m3/minimax_m3.zig"), .target = target, .optimize = optimize });
}

pub fn tests(b: *std.Build, test_step: *std.Build.Step) void {
    const filter = b.option([]const u8, "minimax-m3-filter", "Run one MiniMax-M3 host test");
    const cases = b.addTest(.{ .name = "minimax-m3-host", .root_module = module(b, b.graph.host, .Debug), .filters = if (filter) |f| &.{f} else &.{} });
    const run = b.addRunArtifact(cases);
    b.step("test-minimax-m3", "MiniMax-M3 config, token and refusal host tests").dependOn(&run.step);
    test_step.dependOn(&run.step);
    const install = b.addInstallArtifact(cases, .{ .dest_dir = .{ .override = .{ .custom = "host-checks" } } });
    b.step("compile-minimax-m3-host", "Build the MiniMax-M3 host test binary without running it").dependOn(&install.step);
}
