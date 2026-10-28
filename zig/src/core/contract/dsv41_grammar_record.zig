//! The shared-lane grammar mask launch; compiled resource and device qualification remain pending.
const limits = @import("limits.zig");
const registry = @import("registry.zig");

pub const launch: limits.CudaLaunch = .{
    .name = "DeepSeek shared-lane grammar mask",
    .symbol = "tf_grammar_mask",
    .image = "dsv41_grammar_mask.ptx",
    .threads = 128,
    .min_sm = 80,
};

pub const records = [_]registry.CudaRecord{.{ .launch = launch, .usage = registry.none }};
