//! The MiniMax-M3.1-Flash float64 CPU reference's host-only test step (no GPU, no nvcc).
const std = @import("std");

pub fn steps(b: *std.Build, host: std.Build.ResolvedTarget, core: *std.Build.Module, umbrella: *std.Build.Step) void {
    const ref = b.createModule(.{
        .root_source_file = b.path("zig/src/families/minimax_m3/ref.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .imports = &.{.{ .name = "core", .module = core }},
    });
    const tests = b.createModule(.{
        .root_source_file = b.path("zig/tests/minimax_ref_tests.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "minimax_ref", .module = ref } },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = tests }));
    b.step("test-minimax-ref", "MiniMax-M3.1-Flash float64 CPU reference tests without a GPU").dependOn(&run.step);
    umbrella.dependOn(&run.step);
    const geom = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/minimax_m3_geom.zig"), .target = host, .optimize = .ReleaseSafe });
    const sparse = b.createModule(.{
        .root_source_file = b.path("zig/tests/minimax_sparse_tests.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "mm3_geom", .module = geom } },
    });
    const sparse_run = b.addRunArtifact(b.addTest(.{ .root_module = sparse }));
    b.step("test-minimax-sparse", "MiniMax-M3.1-Flash sparse attention geometry, host twins and launch records without a GPU").dependOn(&sparse_run.step);
    umbrella.dependOn(&sparse_run.step);
}

/// The FFN wiring's host tests (T4): switches to kernel arguments, layer descriptors, the router plan.
pub fn ffn(b: *std.Build, host: std.Build.ResolvedTarget, core: *std.Build.Module, cuda: *std.Build.Module, umbrella: *std.Build.Step) void {
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/families/minimax_m3/ffn.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cuda } },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-minimax-ffn", "MiniMax-M3.1-Flash FFN wiring (swigluoai switches, router plan) without a GPU").dependOn(&run.step);
    umbrella.dependOn(&run.step);
}

/// The row kernels' host tests: norm and RoPE writer, fp32 router, block-table mirror arithmetic, launch records.
pub fn rows(b: *std.Build, host: std.Build.ResolvedTarget, core: *std.Build.Module, cuda: *std.Build.Module, umbrella: *std.Build.Step) void {
    const pool = b.createModule(.{
        .root_source_file = b.path("zig/src/families/minimax_m3/pool.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .imports = &.{.{ .name = "core", .module = core }},
    });
    const plan = b.createModule(.{
        .root_source_file = b.path("zig/src/families/minimax_m3/rows.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cuda } },
    });
    const m = b.createModule(.{
        .root_source_file = b.path("zig/tests/minimax_rows_tests.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{
            .{ .name = "core", .module = core },
            .{ .name = "cuda", .module = cuda },
            .{ .name = "mm3_pool", .module = pool },
            .{ .name = "mm3_plan", .module = plan },
        },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-minimax-rows", "MiniMax-M3.1-Flash qk-norm/RoPE writer, fp32 router and block-table mirror without a GPU").dependOn(&run.step);
    umbrella.dependOn(&run.step);
}
