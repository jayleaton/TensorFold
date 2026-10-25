//! MiniMax-M3.1-Flash takes --kv-cache fp8: the refusal passes it, the host forwards it, bf16 families still refuse.
const std = @import("std");
const minimax = @import("minimax_cuda");
const refusals = @import("cuda_refusal.zig");
const options = @import("cuda_options.zig");

test "MiniMax-M3.1-Flash takes --kv-cache fp8 and forwards it; a family with bf16 caches still refuses it" {
    try std.testing.expect(@hasField(minimax.native.Options, "kv8"));
    const open: @import("engine_api").Open = .{ .dir = "d", .model_type = minimax.native.model_type, .kv8 = true };
    try std.testing.expect(refusals.family(minimax.native, open) == null);
    var opts: minimax.native.Options = .{ .context = 4096, .drafts = false };
    var why: []const u8 = "";
    try std.testing.expect(options.set(minimax.native, open, &opts, &why));
    try std.testing.expect(opts.kv8);
    const Bf16Only = struct {
        pub const Options = struct { context: usize };
    };
    try std.testing.expectStringStartsWith(refusals.family(Bf16Only, .{ .dir = "d", .model_type = "m", .kv8 = true }) orelse "", "--kv-cache fp8");
}
