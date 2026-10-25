//! MiniMax-M3.1-Flash's CUDA family module and every one of its host-only test steps, wired from one call.
const std = @import("std");

const Mod = *std.Build.Module;

/// The shares module (cluster/canon.zig); one per compilation, so a build that also holds GLM's passes GLM's here.
pub fn canonModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Mod {
    return b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon.zig"), .target = target, .optimize = optimize });
}

/// The family over the runtime, the core, the lane core and the engine API, as native/cuda.zig registers it.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: Mod, core: Mod, lanes: Mod, api: Mod, canon: Mod) Mod {
    return moduleAt(b, target, optimize, cuda, core, lanes, api, canon, "zig/src/families/minimax_m3/cuda.zig");
}

/// The same module rooted at another file of the family (a test step's root).
fn moduleAt(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: Mod, core: Mod, lanes: Mod, api: Mod, canon: Mod, root: []const u8) Mod {
    return b.createModule(.{
        .root_source_file = b.path(root),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "core", .module = core },
            .{ .name = "canon", .module = canon },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "engine_api", .module = api },
        },
    });
}

/// Every `test-minimax-*` step; `umbrella` (test-cuda-host) depends on each.
pub fn steps(b: *std.Build, host: std.Build.ResolvedTarget, cuda: Mod, core: Mod, lanes: Mod, api: Mod, tokenizer: Mod, template: Mod, server_add: *const fn (*std.Build, Mod, std.Build.ResolvedTarget, std.builtin.OptimizeMode) Mod, umbrella: *std.Build.Step) void {
    @import("minimax_ref_host.zig").steps(b, host, core, umbrella);
    _ = @import("minimax_load.zig").step(b, host, core);
    _ = @import("minimax_dspark.zig").step(b, host, core, umbrella);
    _ = @import("minimax_pool.zig").step(b, host, core);
    _ = @import("minimax_tp.zig").step(b, host, core);
    @import("minimax_ref_host.zig").ffn(b, host, core, cuda, umbrella);
    @import("minimax_ref_host.zig").rows(b, host, core, cuda, umbrella);
    @import("minimax_host.zig").steps(b, host, api, tokenizer, template, server_add, umbrella);
    forward(b, host, cuda, core, lanes, api, umbrella);
    @import("minimax_proj_host.zig").step(b, host, core, cuda, umbrella);
    kv8(b, host, core, lanes, api, umbrella);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, core, lanes, api, canonModule(b, host, .debug)) }));
    b.step("test-minimax-engine-host", "MiniMax-M3.1-Flash interface, weights descriptors, page state and native host half without a GPU").dependOn(&run.step);
    umbrella.dependOn(&run.step);
    const engine = b.addRunArtifact(b.addTest(.{ .root_module = moduleAt(b, host, .debug, cuda, core, lanes, api, canonModule(b, host, .debug), "zig/src/families/minimax_m3/engine_tests.zig") }));
    b.step("test-minimax-engine", "MiniMax-M3.1-Flash engine loop, follower, lanes backend and radix cache flow on recording fakes, no GPU").dependOn(&engine.step);
    umbrella.dependOn(&engine.step);
    // the dspark draft in the engine: drafted decode equals plain decode on fakes, both ranks
    const dev = b.addRunArtifact(b.addTest(.{ .root_module = moduleAt(b, host, .debug, cuda, core, lanes, api, canonModule(b, host, .debug), "zig/src/families/minimax_m3/dspark_dev_tests.zig") }));
    b.step("test-minimax-dspark-dev", "MiniMax-M3.1-Flash dspark draft in the engine: drafted decode equals plain decode, taps, K/V rings, snapshots, both ranks, on fakes, no GPU").dependOn(&dev.step);
    umbrella.dependOn(&dev.step);
}

/// The forward planner, the device executor's compile check and the prompt driver (forward.zig), host only.
fn forward(b: *std.Build, host: std.Build.ResolvedTarget, cuda: Mod, core: Mod, lanes: Mod, api: Mod, umbrella: *std.Build.Step) void {
    const m = module(b, host, .debug, cuda, core, lanes, api, canonModule(b, host, .debug));
    m.root_source_file = b.path("zig/src/families/minimax_m3/forward.zig");
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-minimax-forward", "MiniMax-M3.1-Flash forward launch plan, op order, widths and prompt chunks without a GPU").dependOn(&run.step);
    umbrella.dependOn(&run.step);
}

/// The fp8 KV cache's host tests (docs/fp8-kv.md): codes, layout, quantizer and attention twins, mutants, family side.
fn kv8(b: *std.Build, host: std.Build.ResolvedTarget, core: Mod, lanes: Mod, api: Mod, umbrella: *std.Build.Step) void {
    const geom = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/minimax_m3_geom.zig"), .target = host, .optimize = .ReleaseSafe });
    const rows_geom = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/minimax_m3_rows_geom.zig"), .target = host, .optimize = .ReleaseSafe });
    const ref = b.createModule(.{
        .root_source_file = b.path("zig/tests/minimax_kv8_tests.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "mm3_geom", .module = geom }, .{ .name = "mm3_rows_geom", .module = rows_geom } },
    });
    const canon = canonModule(b, host, .debug);
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/minimax_m3/kv8_test.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "canon", .module = canon }, .{ .name = "lanes", .module = lanes }, .{ .name = "engine_api", .module = api } },
    });
    const s = b.step("test-minimax-kv8", "MiniMax-M3.1-Flash fp8 KV cache (e4m3 rows, page layout, budget, quantizer and attention twins) without a GPU");
    s.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ref })).step);
    s.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family })).step);
    umbrella.dependOn(s);
}
