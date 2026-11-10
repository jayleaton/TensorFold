//! MiaAI-Lab's Flash Next engine on CUDA (zig/src/families/flashnext_mia): its module and its own check steps.
const std = @import("std");

/// The family over the runtime, the core, the lane core and upstream flashnext (its compute lane and runner bindings).
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, flashnext: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext_mia/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    m.addImport("flashnext", flashnext); // the same qwen4_exp markup: its compute and runner bindings are hers
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

/// Her module's host tests, roce_mia's, the lane opt-ins her family takes and its native contract; in test-cuda-host.
pub fn hostTests(b: *std.Build, cuda: *std.Build.Module, mia: *std.Build.Module, lanes: *std.Build.Module, native: *std.Build.Module, all: *std.Build.Step) void {
    const step = b.step("test-flashnext-mia-host", "MiaAI-Lab's Flash Next engine: its host tests and the lane opt-ins it takes, no GPU");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mia })).step);
    const roce = &.{ "the timeout reader compiles", "prefetch blocks:", "NCCL_IB_HCA lists", "the region layout:" };
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = roce })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{ "join batch:", "head stops:", "pooled budgets:" } })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"flashnext_mia family:"} })).step);
    all.dependOn(step);
}

/// check-flashnext-mia-cli: the tensorfold CLI with her commands compiled (no kernels); in test-flashnext-mia-host.
pub fn cliCheck(b: *std.Build, cli: *std.Build.Module, dory: *std.Build.Module, cluster: *std.Build.Module) void {
    cli.addImport("dory", dory); // what `main` reaches past the host tests' imports, as the installed CLI has them
    cli.addImport("cluster", cluster);
    const obj = b.addExecutable(.{ .name = "tensorfold-cli-check", .root_module = cli }); // an executable: `main` is analyzed
    const step = b.step("check-flashnext-mia-cli", "Compile the tensorfold CUDA CLI with MiaAI-Lab's commands, no kernels");
    step.dependOn(&obj.step);
    b.top_level_steps.get("test-flashnext-mia-host").?.step.dependOn(step);
}
