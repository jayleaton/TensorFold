//! The root of the `test-minimax-dspark-dev` step: the dspark draft in the engine on recording fakes, no GPU.
test {
    _ = @import("dspark_drafter.zig");
    _ = @import("dspark_device.zig");
    _ = @import("dspark_weights.zig");
    _ = @import("dspark_folder.zig");
    _ = @import("dspark_engine_test.zig");
    _ = @import("forward_test.zig"); // the tap ops of the planner
}
