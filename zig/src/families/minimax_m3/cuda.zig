//! MiniMax-M3.1-Flash on CUDA: the root of the `minimax_cuda` module the native server registers (native/cuda.zig).
pub const native = @import("native.zig");
pub const iface = @import("iface.zig");
pub const weights = @import("weights.zig");
pub const weights_device = @import("weights_device.zig");
pub const pages = @import("pages.zig");
pub const engine = @import("engine.zig");
pub const engine_ops = @import("engine_ops.zig");
pub const engine_dev = @import("engine_dev.zig");
pub const follow = @import("follow.zig");
pub const backend = @import("backend.zig");
pub const draw = @import("draw.zig");
pub const forward = @import("forward.zig");
pub const forward_ops = @import("forward_ops.zig");
pub const forward_plan = @import("forward_plan.zig");
pub const pool = @import("pool.zig");
pub const prefill = @import("prefill.zig");
pub const config = @import("config.zig");
pub const shard = @import("shard.zig");
pub const dspark_config = @import("dspark_config.zig");
pub const dspark_names = @import("dspark_names.zig");
pub const dspark_ref = @import("dspark_ref.zig");
pub const dspark_device = @import("dspark_device.zig");
pub const dspark_drafter = @import("dspark_drafter.zig");

test {
    _ = native;
    _ = iface;
    _ = weights;
    _ = weights_device;
    _ = pages;
    _ = engine;
    _ = engine_ops;
    _ = engine_dev;
    _ = follow;
    _ = backend;
    _ = draw;
    _ = forward;
    _ = forward_ops;
    _ = prefill;
    _ = @import("load.zig");
    _ = @import("ffn.zig");
    _ = @import("tp.zig");
    _ = @import("pool.zig");
    _ = @import("ranks.zig");
}
