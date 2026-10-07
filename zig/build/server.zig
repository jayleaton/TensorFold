//! The HTTP server's and DeepSeek-V4.1 serving side's host builds: unit tests on any host, the tokenizer benchmark,
//! and the engine-less server binary. Goldens beyond the committed ones: TF_DSV41_MODEL (the release tokenizer.json's
//! directory) and TF_DSV41_GOLDEN (zig/tests/server/dsv41/gen_*.py's output directory).
const std = @import("std");

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
        },
    });
}

const Server = struct { json: *std.Build.Module, serve: *std.Build.Module, server: *std.Build.Module, api: *std.Build.Module };

/// The server module (zig/src/server/root.zig) and everything it imports, for ``target``.
fn server(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, root: []const u8, natives: ?*std.Build.Module) Server {
    const json = b.createModule(.{ .root_source_file = b.path("zig/src/server/json.zig"), .target = target, .optimize = optimize });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const serve = serveModule(b, target, optimize, json, tokenizer, lanes);
    const m = b.createModule(.{
        .root_source_file = b.path(root),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "engine_api", .module = api },
            .{ .name = "tokenizer", .module = tokenizer },
            .{ .name = "template", .module = template },
            .{ .name = "json", .module = json },
            .{ .name = "dsv41_serve", .module = serve },
        },
    });
    if (natives) |n| {
        n.addImport("engine_api", api);
        n.addImport("lanes", lanes); // an engine on the lane core shares the server's (engine_api's) lanes types
        n.addImport("dsv41_serve", serve); // and its serving module (DeepSeek-V4.1's exact sampler)
        m.addImport("native_engines", n);
    }
    return .{ .json = json, .serve = serve, .server = m, .api = api };
}

/// tensorfold-native's root module with `natives` as its built-in engines (zig/build/cuda.zig: tensorfold-dsv41).
pub fn withEngine(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, natives: *std.Build.Module) *std.Build.Module {
    return server(b, target, optimize, "zig/src/server/main.zig", natives).server;
}

/// `test-dsv41-serve`: the serving module's goldens; `test-server`: the HTTP server's tests (DeepSeek over HTTP
/// included); both part of `test`. `tf-dsv41-tokbench`: throughput. `server-none`: tensorfold-native without an engine.
pub fn tests(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const s = server(b, host, .Debug, "zig/src/server/root.zig", null);
    const serve_step = b.step("test-dsv41-serve", "DeepSeek-V4.1's serving side: tokenizer, encoding, DSML tools, sampling (goldens: TF_DSV41_MODEL, TF_DSV41_GOLDEN)");
    serve_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = s.serve })).step);
    test_step.dependOn(serve_step);
    if (host.result.os.tag != .macos) { // macOS runs the server's tests in nativeServer, with its Metal engines
        const server_step = b.step("test-server", "The native HTTP server's tests, DeepSeek-V4.1 over HTTP included (TF_DSV41_MODEL: the release tokenizer too)");
        server_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = s.server })).step);
        server_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = s.json })).step);
        server_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = s.api })).step);
        test_step.dependOn(server_step);
        const none = b.createModule(.{ .root_source_file = b.path("zig/src/native/none.zig"), .target = host, .optimize = .ReleaseSafe, .link_libc = true });
        const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = server(b, host, .ReleaseSafe, "zig/src/server/main.zig", none).server });
        b.step("server-none", "tensorfold-native without an engine (the server's own checks) into zig-out/server").dependOn(&b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "server" } } }).step);
    }
    const fast = server(b, host, .ReleaseFast, "zig/src/server/root.zig", null);
    const bench = b.addExecutable(.{ .name = "tf-dsv41-tokbench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/dsv41/tokbench.zig"),
        .target = host,
        .optimize = .ReleaseFast,
        .link_libc = true,
        .imports = &.{.{ .name = "dsv41_serve", .module = fast.serve }},
    }) });
    b.step("tf-dsv41-tokbench", "The DeepSeek-V4.1 tokenizer's throughput over gen_tokenizer.py's corpus").dependOn(&b.addInstallArtifact(bench, .{}).step);
}
