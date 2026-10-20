//! MiniMax-M3 (MiniMaxAI/MiniMax-M3) and its M3.1 successors: config reading and a clean refusal until ported.
const std = @import("std");
pub const config = @import("config.zig");
pub const tokens = @import("tokens.zig");
pub const ops = @import("ops.zig");

pub const model_types = [_][]const u8{ "minimax_m3_vl", "minimax_m3" };

pub fn claims(model_type: []const u8) bool {
    for (model_types) |t| if (std.mem.eql(u8, t, model_type)) return true;
    return false;
}

/// Why the engine cannot serve the checkpoint in `dir`: its config's broken key, or the operators not yet ported.
pub fn refusal(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![]const u8 {
    var p: config.Problem = .{};
    const c = config.Config.read(a, io, dir, &p) catch |e| return switch (e) {
        error.UnsupportedMiniMaxM3 => std.fmt.allocPrint(a, "MiniMax-M3: config.json's {s} names an operator this family does not implement", .{p.key}),
        error.BadMiniMaxM3Config => std.fmt.allocPrint(a, "MiniMax-M3: config.json is malformed at {s}", .{p.key}),
        error.OutOfMemory => error.OutOfMemory,
        else => std.fmt.allocPrint(a, "MiniMax-M3: cannot read config.json ({s})", .{@errorName(e)}),
    };
    return describe(a, &c);
}

/// A server's open for these model types: no engine, with the refusal as its problem.
pub fn refuse(comptime Opened: type, a: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) !?Opened {
    problem.* = try refusal(a, io, dir);
    return null;
}

/// The refusal for a config that reads: its shape, then each operator still to port.
pub fn describe(a: std.mem.Allocator, c: *const config.Config) ![]const u8 {
    var buf: [ops.count]ops.Op = undefined;
    const left = ops.missing(c, &buf);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.print(a, "MiniMax-M3 ({d} layers, {d} sparse, {d} experts top-{d}): not implemented in this build:", .{ c.layers, c.sparseLayers(), c.experts, c.topk });
    for (left, 0..) |op, i| try out.print(a, "{s} {s}", .{ if (i == 0) "" else ";", op.text() });
    if (left.len == 0) try out.appendSlice(a, " no operator is missing, but this family has no engine open yet");
    return out.toOwnedSlice(a);
}

test {
    _ = config;
    _ = tokens;
    _ = ops;
}

test "minimax_m3 refusal: a published config reads, then names what is not implemented" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    try std.testing.expectStringStartsWith(try refusal(a, io, dir), "MiniMax-M3: cannot read config.json (FileNotFound)");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = @embedFile("config_fixture.json") });
    const why = try refusal(a, io, dir);
    try std.testing.expectStringStartsWith(why, "MiniMax-M3 (60 layers, 57 sparse, 128 experts top-4): not implemented in this build: checkpoint");
    try std.testing.expect(std.mem.indexOf(u8, why, "MSA index branch") != null);
    try std.testing.expect(std.mem.endsWith(u8, why, "; vision tower and patch-merge projector"));
    const gated = try std.mem.replaceOwned(u8, a, @embedFile("config_fixture.json"), "\"attention_output_gate\": false", "\"attention_output_gate\": true");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = gated });
    try std.testing.expectEqualStrings("MiniMax-M3: config.json's attention_output_gate names an operator this family does not implement", try refusal(a, io, dir));
}

test "minimax_m3 claims its model types only" {
    try std.testing.expect(claims("minimax_m3_vl") and claims("minimax_m3"));
    try std.testing.expect(!claims("minimax_m2") and !claims("minimax"));
}
