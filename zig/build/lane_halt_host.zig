//! CPU checks for halt ordering, with the real lane host and fake target.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const interrupted = b.createModule(.{ .root_source_file = b.path("zig/src/core/lane_host_interrupted_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    b.step("test-lane-interrupted-host", "Interrupted prompt retention and clean retries without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = interrupted, .filters = &.{"interrupted prefill:"} })).step);
    const halt = b.addTest(.{ .root_module = api, .filters = &.{ "halt (", "halt fails" } });
    b.step("test-lane-halt-host", "Halt ordering with a fake backend and no GPU dependencies").dependOn(&b.addRunArtifact(halt).step);
    b.step("test-lane-lone-host", "Existing lone-driver handoff and reclaim with a fake backend").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"lone driver"} })).step);
    b.step("test-lane-api-host", "Shared lane host and engine API CPU regressions").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api })).step);
    b.step("test-snapshot-index-host", "Prompt-cache lookup index against the whole-store scan, without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"snapshot index:"} })).step);
    b.step("test-rows-prefetch-host", "Disk states read ahead for queued requests: equality, cancellation, fallbacks").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"prefetch:"} })).step);
    b.step("test-embedding-lane-host", "Prompt-only embedding jobs on the actual lane worker without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"embedding lane:"} })).step);
    b.step("test-prefill-service-host", "The opt-in prompt-row budget beside decoding streams on the lane host with a fake backend").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"prefill service:"} })).step);
    b.step("test-runner-host", "The in-model runner: scanner, held results and delivery with a fake circuit backend").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"runner"} })).step);
    // Proof 2 on the real circuit (TF_RUNNER_CIRCUIT): optimized, since a debug evaluator takes minutes a run
    const lanes_fast = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .ReleaseSafe, .link_libc = true });
    const api_fast = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes_fast }} });
    b.step("test-runner-proof-host", "Multiply and loop replies on the real frontend and CPU circuit (TF_RUNNER_CIRCUIT)").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api_fast, .filters = &.{"runner proof:"} })).step);
    b.step("test-compute-runner-host", "S3: lane calls on the circuit, their fallbacks and knobs, with a scripted backend").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"compute runner:"} })).step);
    b.step("test-compute-runner-proof-host", "S3 on recorded arm-d calls: the real frontend and CPU circuit (TF_RUNNER_CIRCUIT)").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api_fast, .filters = &.{"compute runner proof:"} })).step);
    const copies = b.step("test-root-copy-host", "Root copy precedence, drafter absorption and real worker checks without GPU");
    copies.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"root copy:"} })).step);
    copies.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"root copy:"} })).step);
}
