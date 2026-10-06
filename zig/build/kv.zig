//! KV memory's build: the family-neutral `sessions` module (zig/src/sessions) and DeepSeek-V4.1's `kv` module over it, with host tests on any OS.

const std = @import("std");

/// The CUDA runtime without the engine's kernels (kv needs the driver, memory and streams; tp's collectives need the same).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    return cuda;
}

pub fn sessions(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/sessions/sessions.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

/// The `kv` module: imports sessions, cuda and tp (an engine passes its own runtime and tp modules).
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, tp: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kv/kv.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("sessions", sessions(b, target, optimize));
    m.addImport("cuda", cuda);
    m.addImport("tp", tp);
    return m;
}

pub fn add(b: *std.Build, test_step: *std.Build.Step, tp_module: fn (*std.Build, std.Build.ResolvedTarget, std.builtin.OptimizeMode, *std.Build.Module) *std.Build.Module) void {
    const host = b.graph.host;
    const cuda = runtime(b, host, .debug);
    const s = b.addRunArtifact(b.addTest(.{ .root_module = sessions(b, host, .debug) }));
    const k = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, tp_module(b, host, .debug, cuda)) }));
    const step = b.step("test-kv", "KV memory's host tests: pool, sessions, NVMe tier on real files, split KV on in-process ranks (no GPU)");
    step.dependOn(&s.step);
    step.dependOn(&k.step);
    test_step.dependOn(&s.step);
    test_step.dependOn(&k.step);
}
