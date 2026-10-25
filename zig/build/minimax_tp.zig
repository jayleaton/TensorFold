//! test-minimax-tp: MiniMax-M3.1-Flash's rank contract, collective plan, rank-order reduction and fingerprint on the host, no GPU.
const std = @import("std");

pub fn step(b: *std.Build, host: std.Build.ResolvedTarget, core: *std.Build.Module) *std.Build.Step {
    const canon = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon.zig"), .target = host, .optimize = .debug });
    const imports: []const std.Build.Module.Import = &.{ .{ .name = "core", .module = core }, .{ .name = "canon", .module = canon } };
    const s = b.step("test-minimax-tp", "MiniMax-M3.1-Flash rank contract, TP collectives, rank-order sums and rank fingerprint without a GPU");
    for ([_][]const u8{ "tp.zig", "ranks.zig" }) |file| {
        const m = b.createModule(.{
            .root_source_file = b.path(b.fmt("zig/src/families/minimax_m3/{s}", .{file})),
            .target = host,
            .optimize = .debug,
            .link_libc = true,
            .imports = imports,
        });
        s.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    }
    return s;
}
