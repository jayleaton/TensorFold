//! MiaAI-Lab's Flash Next engine on CUDA (zig/src/families/flashnext_mia): its module and its own check steps.
const std = @import("std");

/// The family over the runtime, the core and the lane core.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext_mia/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    return m;
}

/// Her two-rank transport test and weights check as their own steps, not in `install`.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, mia: *std.Build.Module) void {
    const imports: []const std.Build.Module.Import = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "flashnext_mia", .module = mia } };
    const comm = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/flashnext_mia/comm_test.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = imports });
    const comm_exe = b.addExecutable(.{ .name = "tf-flashnext-mia-comm", .root_module = comm });
    b.step("flashnext-mia-comm", "tf-flashnext-mia-comm RANK MASTER PORT: two-rank all-gathers, exact and timed").dependOn(&b.addInstallArtifact(comm_exe, .{}).step);
    const weights = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/flashnext_mia/weights_check.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = imports });
    const weights_exe = b.addExecutable(.{ .name = "tf-flashnext-mia-weights", .root_module = weights });
    b.step("flashnext-mia-weights", "tf-flashnext-mia-weights SNAPSHOT [WEIGHTS.json]: loaded weights' digests against the oracle's").dependOn(&b.addInstallArtifact(weights_exe, .{}).step);
}
