//! The real processor, decoder corpus, native API, ownership lifecycle and image buffer planning on the host.
test {
    _ = @import("vision_boot_test.zig");
    _ = @import("vision_native_test.zig");
    _ = @import("vision_engine_test.zig");
    _ = @import("vision_map_churn_test.zig");
    _ = @import("vision_plan_test.zig");
    _ = @import("vision_open_test.zig");
    _ = @import("vision/golden_test.zig");
    _ = @import("vision/webp_golden_test.zig");
    _ = @import("vision/webp_vp8l_budget_test.zig");
    @import("std").testing.refAllDecls(@import("vision/decode.zig"));
    @import("std").testing.refAllDecls(@import("vision/prep.zig"));
}
