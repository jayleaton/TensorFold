//! DeepSeek-V4.1's host-side steps under `test` (no GPU): host, drafting, serving, sampling, grammar, image, TP, KV.
const std = @import("std");

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    hostTests(b, test_step);
    draftTests(b, test_step);
    @import("dsv41_server.zig").tests(b, test_step);
    const tp = @import("tp.zig");
    tp.add(b, target, optimize, test_step);
    @import("kv.zig").add(b, target, optimize, test_step, tp.module);
    @import("dsv41_sampling.zig").add(b, test_step);
    @import("dsv41_grammar.zig").add(b, test_step);
    @import("dsv41_vision.zig").add(b, test_step);
}

/// The family's host side (config, pack headers, the rank's weight plan) on any host; TF_DSV41_HEADERS adds real dumps.
fn hostTests(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = host, .optimize = .debug, .link_libc = true });
    core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = host, .optimize = .debug, .link_libc = true }));
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/deepseek_v41.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = core }},
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = family }));
    b.step("test-dsv41", "DeepSeek-V4.1's host-side unit tests (no GPU work)").dependOn(&run.step);
    test_step.dependOn(&run.step);
}

/// The family's drafting (DSpark depth, trees, the lanes backend) on the CPU twin and the Python golden traces.
fn draftTests(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const draft = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/draft/draft.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "lanes", .module = lanes }},
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = draft }));
    b.step("test-dsv41-draft", "DeepSeek-V4.1's drafting and lanes backend on the CPU twin (no GPU work)").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
