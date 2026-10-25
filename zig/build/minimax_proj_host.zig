//! test-minimax-proj: MiniMax-M3.1-Flash fp16 index projection and per-rank pick on the host, no GPU.
const std = @import("std");

pub fn step(b: *std.Build, host: std.Build.ResolvedTarget, core: *std.Build.Module, cuda: *std.Build.Module, umbrella: *std.Build.Step) void {
    const canon = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon.zig"), .target = host, .optimize = .debug });
    const tp = b.createModule(.{
        .root_source_file = b.path("zig/src/families/minimax_m3/tp.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "canon", .module = canon } },
    });
    const m = b.createModule(.{
        .root_source_file = b.path("zig/tests/minimax_proj_tests.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cuda }, .{ .name = "mm3_tp", .module = tp } },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-minimax-proj", "MiniMax-M3.1-Flash fp16 index projection and per-rank pick: host twins, mutants and launch records without a GPU").dependOn(&run.step);
    umbrella.dependOn(&run.step);
}
