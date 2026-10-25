//! MiniMax-M3.1-Flash's CUDA launch records (minimax_m3_act.cu): swigluoai and the sigmoid-plus-bias router.
pub fn records(comptime Record: type) [3]Record {
    return .{
        .{ .launch = .{ .name = "minimax_m3_act tf_mm3_swigluoai_bf16", .symbol = "tf_mm3_swigluoai_bf16", .image = "minimax_m3_act", .threads = 256 }, .usage = .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 } } },
        .{ .launch = .{ .name = "minimax_m3_act tf_mm3_swigluoai_f32", .symbol = "tf_mm3_swigluoai_f32", .image = "minimax_m3_act", .threads = 256 }, .usage = .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 } } },
        .{ .launch = .{ .name = "minimax_m3_act tf_mm3_route_warp", .symbol = "tf_mm3_route_warp", .image = "minimax_m3_act", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 33, .static = 0, .bound = 256 }, .{ .regs = 33, .static = 0, .bound = 256 } } },
    };
}
