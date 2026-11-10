//! MiaAI-Lab's Flash Next engine (flashnext_mia) for qwen4_exp checkpoints, opened only with TF_FLASHNEXT_ENGINE=mia.
const std = @import("std");
const api = @import("engine_api");
const mia = @import("flashnext_mia");
const refusals = @import("cuda_refusal.zig");

/// Her family's namespace, which native/cuda.zig opens with openWith as every registry family (f212's contract).
pub const family = mia.native;

/// Whether `o` asks for this engine: a qwen4_exp checkpoint with TF_FLASHNEXT_ENGINE=mia (unset: upstream's flashnext).
pub fn wanted(o: api.Open) bool {
    return std.mem.eql(u8, o.model_type, family.model_type) and mia.selected();
}

test "flashnext_mia family: two ranks and fp8 caches through the options, no --vision or --slide" {
    const F = family;
    try std.testing.expect(refusals.family(F, .{ .dir = "d", .model_type = F.model_type, .tp = 2, .rank = 1, .master = "192.0.2.1", .kv8 = true }) == null);
    try std.testing.expect(!@hasField(F.Options, "vision")); // refusals.family refuses --vision for it
    try std.testing.expect(!@hasDecl(F, "slide") and F.max_segments == 1);
    try std.testing.expect(F.explain(null, error.CapturedKernelsNeeded) != null);
    var opts: F.Options = .{ .context = 1 << 20, .drafts = true };
    var why: []const u8 = "";
    const o: api.Open = .{ .dir = "d", .model_type = F.model_type, .lanes = 16, .tp = 2, .rank = 1, .master = "192.0.2.1", .master_port = 29551, .kv8 = true, .prompt_cache_gib = 4 };
    try std.testing.expect(@import("cuda_options.zig").set(F, o, &opts, &why));
    try std.testing.expectEqual(@as(u32, 16), opts.lanes);
    try std.testing.expect(opts.kv8 and opts.tp == 2 and opts.rank == 1 and opts.master_port == 29551);
    try std.testing.expectEqualStrings("192.0.2.1", opts.master.?);
    try std.testing.expectEqual(@as(?f64, 4), opts.prompt_cache_gib);
}
