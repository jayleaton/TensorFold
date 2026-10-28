//! Native image host checks join the full test suite; no device access or compiler images are needed.
const std = @import("std");
pub fn steps(b: *std.Build, test_step: *std.Build.Step) void {
    const m = @import("dsv41_cuda.zig").module(b, b.graph.host, "vision_host_test.zig", false);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    run.setCwd(b.path("."));
    b.step("test-dsv41-vision", "Native image decoders, geometry, ownership, request lifecycle and planning").dependOn(&run.step);
    test_step.dependOn(&run.step);
    const target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu });
    const cross = b.addTest(.{ .root_module = @import("dsv41_cuda.zig").module(b, target, "vision_host_test.zig", true) });
    _ = cross.getEmittedBin();
    b.step("compile-dsv41-vision", "Emit native image host checks for Linux without execution").dependOn(&cross.step);
}
