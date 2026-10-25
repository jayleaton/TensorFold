//! test-minimax-dspark-kernels: the dspark draft kernels' host twins, mutants, launch geometry and records, no GPU.
const std = @import("std");

pub fn step(b: *std.Build, host: std.Build.ResolvedTarget, core: *std.Build.Module, cuda: *std.Build.Module, umbrella: *std.Build.Step) void {
    const m = b.createModule(.{
        .root_source_file = b.path("zig/tests/minimax_dspark_tests.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cuda } },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-minimax-dspark-kernels", "MiniMax-M3.1-Flash dspark draft kernels: host twins, mutants, launch geometry and records without a GPU").dependOn(&run.step);
    umbrella.dependOn(&run.step);
}
