//! The family mask kernel's PTX; the driver compiles it for the actual GPU.
const std = @import("std");

pub const image_name = "dsv41_grammar_mask.ptx";

pub fn image(b: *std.Build) std.Build.LazyPath {
    const target = b.resolveTargetQuery(.{ .cpu_arch = .nvptx64, .os_tag = .cuda, .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_80 } });
    const obj = b.addObject(.{ .name = "dsv41_grammar_mask", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/grammar_mask_kernel.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    }) });
    return obj.getEmittedAsm();
}

pub fn addTo(b: *std.Build, m: *std.Build.Module) void {
    m.addAnonymousImport("grammar_mask_ptx", .{ .root_source_file = image(b) });
}

pub fn hostTests(b: *std.Build, all: *std.Build.Step) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/grammar_contract.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .imports = &.{.{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .Debug) }},
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-dsv41-grammar-contract-host", "Shared grammar mask contract and planted corruption on host; device unrun").dependOn(&run.step);
    all.dependOn(&run.step);
}
