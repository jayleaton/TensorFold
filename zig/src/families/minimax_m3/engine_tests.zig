//! The root of the `test-minimax-engine` step: the engine loop's files and their host tests.
test {
    _ = @import("engine.zig");
    _ = @import("engine_ops.zig");
    _ = @import("engine_dev.zig");
    _ = @import("follow.zig");
    _ = @import("backend.zig");
    _ = @import("draw.zig");
    _ = @import("engine_fake.zig");
    _ = @import("engine_test.zig");
    _ = @import("backend_test.zig");
    _ = @import("floor.zig");
    _ = @import("floor_test.zig");
}
