//! test-minimax-load: MiniMax-M3.1-Flash's config, names, shard cuts and checkpoint loader on the host, no GPU.
const std = @import("std");

pub fn step(b: *std.Build, host: std.Build.ResolvedTarget, core: *std.Build.Module) *std.Build.Step {
    const canon = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon.zig"), .target = host, .optimize = .debug });
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/minimax_m3/load.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "canon", .module = canon } },
    });
    const s = b.step("test-minimax-load", "MiniMax-M3.1-Flash config, tensor names, TP cuts and EXL3 loader without a GPU");
    s.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family })).step);
    const bench = b.addExecutable(.{ .name = "mm3-loadbench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/minimax_m3/loadbench.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "canon", .module = canon } },
    }) });
    b.step("mm3-loadbench", "MiniMax-M3.1-Flash checkpoint reader diagnosis and micro-benchmark (host only; see loadbench.zig)")
        .dependOn(&b.addInstallArtifact(bench, .{}).step);
    return s;
}
