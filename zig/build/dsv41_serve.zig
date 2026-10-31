//! The installable DeepSeek-V4.1 CUDA pair on real images: tensorfold-dsv41 (rank 0) and tf-dsv41 (the other ranks).
const std = @import("std");
const family = @import("dsv41_cuda.zig");
const images = @import("dsv41_images.zig");

/// `dsv41-serve` installs both over the CUDA runtime `cuda`; family images come from -Ddsv41-nvcc or -Ddsv41-fatbins.
pub fn install(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, runtime_images: bool) void {
    const step = b.step("dsv41-serve", "Install tensorfold-dsv41 and tf-dsv41 over the CUDA runtime and family images");
    const fixtures = b.step("dsv41-test-runner", "Install the device fixture runner tf-dsv41-test over the real images");
    if (!runtime_images) {
        const fail = &b.addFail("dsv41-serve and dsv41-test-runner need the CUDA runtime's images: -Dnvcc=PATH or -Dfatbins=DIR").step;
        step.dependOn(fail);
        return fixtures.dependOn(fail);
    }
    const im = images.built orelse {
        const fail = &b.addFail("dsv41-serve and dsv41-test-runner need -Ddsv41-nvcc=PATH or -Ddsv41-fatbins=DIR").step;
        step.dependOn(fail);
        return fixtures.dependOn(fail);
    };
    const dk = images.module(b, target, optimize, cuda, im);
    const tool = b.addExecutable(.{ .name = "tf-dsv41", .root_module = family.assemble(b, target, optimize, "tool_main.zig", cuda, dk) });
    step.dependOn(&b.addInstallArtifact(tool, .{}).step);
    const engine = family.assemble(b, target, optimize, "serve_engine.zig", cuda, dk);
    step.dependOn(&b.addInstallArtifact(@import("dsv41_draft.zig").serverOn(b, target, optimize, engine), .{}).step);
    const runner = b.addExecutable(.{ .name = "tf-dsv41-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/dsv41_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "dsv41_kernels", .module = dk } },
    }) });
    fixtures.dependOn(&b.addInstallArtifact(runner, .{}).step);
}
