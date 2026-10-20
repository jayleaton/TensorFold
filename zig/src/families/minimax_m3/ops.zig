//! The operators a MiniMax-M3 checkpoint needs and which of them this engine runs; the open refuses until all do.
const std = @import("std");
const config = @import("config.zig");

pub const Op = enum {
    weights,
    attention,
    sparse_index,
    sparse_attention,
    router,
    experts,
    cache,
    tensor_parallel,
    mtp,
    vision,

    pub fn text(op: Op) []const u8 {
        return switch (op) {
            .weights => "checkpoint tensor map and EXL3 packs",
            .attention => "GQA attention with partial rotary and per-head QK norm",
            .sparse_index => "MSA index branch and block top-k",
            .sparse_attention => "block-selected sparse attention",
            .router => "sigmoid router with routing bias",
            .experts => "routed and shared swigluoai experts",
            .cache => "KV and index-key cache pages",
            .tensor_parallel => "two-rank head and expert split",
            .mtp => "MTP draft modules",
            .vision => "vision tower and patch-merge projector",
        };
    }
};

pub const count = std.enums.values(Op).len;

/// Flip an entry when its operator lands with its gate; the open admits a checkpoint only when every needed one has.
pub const ported = std.EnumArray(Op, bool).initFill(false);

/// Whether `c` uses `op`: drafts and the tower only when the checkpoint has them.
pub fn needed(c: *const config.Config, op: Op) bool {
    return switch (op) {
        .sparse_index, .sparse_attention => c.sparseLayers() > 0,
        .experts, .router => c.denseLayers() < c.layers,
        .mtp => c.mtp_modules > 0 and c.nextn_layers > 0,
        .vision => c.vision != null,
        else => true,
    };
}

/// The operators `c` needs that this engine does not run yet, in pipeline order.
pub fn missing(c: *const config.Config, out: *[count]Op) []const Op {
    var n: usize = 0;
    for (std.enums.values(Op)) |op| if (needed(c, op) and !ported.get(op)) {
        out[n] = op;
        n += 1;
    };
    return out[0..n];
}

test "minimax_m3 ops: the published M3 needs every operator, drafts and tower included" {
    var p: config.Problem = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("config_fixture.json"), &p);
    var buf: [count]Op = undefined;
    const left = missing(&c, &buf);
    try std.testing.expectEqual(count, left.len);
    try std.testing.expectEqual(Op.weights, left[0]);
    try std.testing.expectEqual(Op.vision, left[left.len - 1]);
}

test "minimax_m3 ops: a checkpoint without drafts or a tower does not wait on them" {
    var c = config.Config{ .layers = 2, .mtp_modules = 0 };
    c.layer[0] = .{ .moe = false, .sparse = false, .index_value_off = false };
    c.layer[1] = .{ .moe = true, .sparse = true, .index_value_off = false };
    var buf: [count]Op = undefined;
    const left = missing(&c, &buf);
    for (left) |op| try std.testing.expect(op != .mtp and op != .vision);
    try std.testing.expectEqual(count - 2, left.len);
}
