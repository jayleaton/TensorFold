//! `test-minimax-template`: MiniMax-M3.1-Flash tokenizer, chat template and tool-call parser tests, host only, no GPU.
const std = @import("std");

pub fn steps(b: *std.Build, host: std.Build.ResolvedTarget, api: *std.Build.Module, tokenizer: *std.Build.Module, template: *std.Build.Module, server_add: *const fn (*std.Build, *std.Build.Module, std.Build.ResolvedTarget, std.builtin.OptimizeMode) *std.Build.Module, all: *std.Build.Step) void {
    const step = b.step("test-minimax-template", "MiniMax-M3.1-Flash tokenizer, chat template (vs Python Jinja2) and tool-call parser, host only");
    const opts = b.addOptions();
    opts.addOption([]const u8, "tokenizer_path", b.option([]const u8, "mm3-tokenizer", "Path to MiniMax-M3.1-Flash tokenizer.json (the tokenizer test skips without it)") orelse "zig/src/families/minimax_m3/fixtures/tokenizer.json");
    const text = b.createModule(.{ .root_source_file = b.path("zig/src/families/minimax_m3/text_test.zig"), .target = host, .optimize = .Debug, .link_libc = true, .imports = &.{
        .{ .name = "tokenizer", .module = tokenizer },
        .{ .name = "template", .module = template },
        .{ .name = "mm3_options", .module = opts.createModule() },
    } });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = text })).step);
    const parse = b.createModule(.{ .root_source_file = b.path("zig/src/server/tool_parse_test.zig"), .target = host, .optimize = .Debug, .link_libc = true, .imports = &.{
        .{ .name = "engine_api", .module = api },
        .{ .name = "tokenizer", .module = tokenizer },
        .{ .name = "template", .module = template },
    } });
    _ = server_add(b, parse, host, .Debug);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = parse })).step);
    all.dependOn(step);
}
