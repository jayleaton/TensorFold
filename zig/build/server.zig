//! The HTTP server's and DeepSeek-V4.1 serving side's host builds: unit tests on any host, the tokenizer benchmark.
//! Goldens beyond the committed ones: TF_DSV41_MODEL (the release tokenizer.json's directory) and TF_DSV41_GOLDEN
//! (zig/tests/server/dsv41/gen_*.py's output directory).
const std = @import("std");

pub const Modules = struct { json: *std.Build.Module, serve: *std.Build.Module };

/// DeepSeek-V4.1's serving module (tokenizer, encoding, DSML tools, sampling) and the server's JSON values it reads.
pub fn modules(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Modules {
    const json = b.createModule(.{ .root_source_file = b.path("zig/src/server/json.zig"), .target = target, .optimize = optimize });
    const unicode = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/unicode.zig"), .target = target, .optimize = optimize });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const fixtures = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/fixtures/serve/fixtures.zig"), .target = target, .optimize = optimize });
    const serve = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/serve/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "json", .module = json },
            .{ .name = "tok_unicode", .module = unicode },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "dsv41_serve_fixtures", .module = fixtures },
        },
    });
    return .{ .json = json, .serve = serve };
}

/// `test-dsv41-serve`: the serving module's tests and goldens; also part of `test`. `tf-dsv41-tokbench`: throughput.
pub fn tests(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const m = modules(b, host, .Debug);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m.serve }));
    const json_run = b.addRunArtifact(b.addTest(.{ .root_module = m.json }));
    const step = b.step("test-dsv41-serve", "DeepSeek-V4.1's serving side: tokenizer, encoding, DSML tools, sampling (goldens: TF_DSV41_MODEL, TF_DSV41_GOLDEN)");
    step.dependOn(&run.step);
    step.dependOn(&json_run.step);
    test_step.dependOn(step);
    const fast = modules(b, host, .ReleaseFast);
    const bench = b.addExecutable(.{ .name = "tf-dsv41-tokbench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/dsv41/tokbench.zig"),
        .target = host,
        .optimize = .ReleaseFast,
        .link_libc = true,
        .imports = &.{.{ .name = "dsv41_serve", .module = fast.serve }},
    }) });
    b.step("tf-dsv41-tokbench", "The DeepSeek-V4.1 tokenizer's throughput over gen_tokenizer.py's corpus").dependOn(&b.addInstallArtifact(bench, .{}).step);
}
