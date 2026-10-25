//! test-minimax-pool: MiniMax-M3.1-Flash's KV pages, block tables, snapshots and memory budget on the host, no GPU.
const std = @import("std");

pub fn step(b: *std.Build, host: std.Build.ResolvedTarget, core: *std.Build.Module) *std.Build.Step {
    const canon = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon.zig"), .target = host, .optimize = .debug });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/minimax_m3/pool.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "canon", .module = canon }, .{ .name = "engine_api", .module = api } },
    });
    const s = b.step("test-minimax-pool", "MiniMax-M3.1-Flash KV pages, block tables, prompt snapshots and memory budget without a GPU");
    s.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family })).step);
    return s;
}
