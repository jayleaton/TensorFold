//! test-minimax-dspark: the dspark draft's config, names, rank plan, float64 reference, twins and budget; no GPU.
const std = @import("std");

pub fn step(b: *std.Build, host: std.Build.ResolvedTarget, core: *std.Build.Module, umbrella: *std.Build.Step) *std.Build.Step {
    const canon = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon.zig"), .target = host, .optimize = .debug });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/minimax_m3/dspark_test.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "canon", .module = canon }, .{ .name = "engine_api", .module = api } },
    });
    const s = b.step("test-minimax-dspark", "MiniMax-M3.1-Flash dspark draft: config, names, plan, float64 reference, twins with mutants, budget; no GPU");
    const run = b.addRunArtifact(b.addTest(.{ .root_module = family }));
    s.dependOn(&run.step);
    umbrella.dependOn(&run.step);
    return s;
}
