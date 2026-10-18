//! `dsv41_serve` for every server module, its goldens (TF_DSV41_MODEL, TF_DSV41_GOLDEN) and the tokenizer bench.
const std = @import("std");
const vision_build = @import("dsv41_vision.zig");

/// DeepSeek-V4.1's serving module (tokenizer, encoding, DSML tools, sampling) over the modules the server shares.
pub fn serveModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, json: *std.Build.Module, tokenizer: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const fixtures = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/fixtures/serve/fixtures.zig"), .target = target, .optimize = optimize });
    return b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/serve/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "json", .module = json },
            .{ .name = "tokenizer", .module = tokenizer },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "dsv41_serve_fixtures", .module = fixtures },
            .{ .name = "dsv41_vision", .module = vision_build.module(b, target, optimize, json) },
        },
    });
}

/// tensorfold-native's root module with `natives` as its built-in engines (zig/build/dsv41_cuda.zig: tensorfold-dsv41).
pub fn withEngine(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, natives: *std.Build.Module, core: *std.Build.Module, build_options: *std.Build.Step.Options) *std.Build.Module {
    const tokenizer = core.import_table.get("tokenizer").?; // the checkpoint commands' core reads with the same one
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "native_engines", .module = natives }, @import("exlr8.zig").cliImport(b, target, optimize, core), .{ .name = "core", .module = core } } });
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "engine_api", .module = api },
            .{ .name = "tokenizer", .module = tokenizer },
            .{ .name = "template", .module = template },
            .{ .name = "native_engines", .module = natives },
            .{ .name = "checkpoint_cli", .module = cli },
        },
    });
    m.addOptions("build_options", build_options);
    _ = @import("server.zig").addTo(b, m, target, optimize);
    natives.addImport("engine_api", api);
    natives.addImport("lanes", lanes); // an engine on the lane core shares the server's (engine_api's) lanes types
    natives.addImport("dsv41_serve", m.import_table.get("dsv41_serve").?); // and its serving module (the exact sampler)
    natives.addImport("tokenizer", tokenizer); // TF_DSV41_CALIB=measure's calibration text (calib_gpu.zig)
    return m;
}

/// `test-dsv41-serve`: the serving module's goldens, part of `test`; `tf-dsv41-tokbench`: the tokenizer's throughput.
pub fn tests(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const serve_step = b.step("test-dsv41-serve", "DeepSeek-V4.1's serving side: tokenizer, encoding, DSML tools, sampling (goldens: TF_DSV41_MODEL, TF_DSV41_GOLDEN)");
    serve_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = standalone(b, host, .Debug) })).step);
    test_step.dependOn(serve_step);
    const bench = b.addExecutable(.{ .name = "tf-dsv41-tokbench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/dsv41/tokbench.zig"),
        .target = host,
        .optimize = .ReleaseFast,
        .link_libc = true,
        .imports = &.{.{ .name = "dsv41_serve", .module = standalone(b, host, .ReleaseFast) }},
    }) });
    b.step("tf-dsv41-tokbench", "The DeepSeek-V4.1 tokenizer's throughput over gen_tokenizer.py's corpus").dependOn(&b.addInstallArtifact(bench, .{}).step);
}

/// The serving module with its own json, tokenizer and lanes modules, for `target`.
pub fn standalone(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const json = @import("server.zig").json(b, target, optimize);
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    return serveModule(b, target, optimize, json, tokenizer, lanes);
}
