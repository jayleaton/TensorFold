//! MiniMax-M3.1-Flash's CUDA launch records (minimax_m3_act.cu): swigluoai for the dense layers and the shared expert, and
//! the sigmoid-plus-bias router. Registers per nvcc 13.0 compile for sm_80, 86, 89, 120, 121 (cuobjdump --dump-resource-usage).
pub fn records(comptime Record: type) [3]Record {
    return .{
        .{ .launch = .{ .name = "minimax_m3_act tf_mm3_swigluoai_bf16", .symbol = "tf_mm3_swigluoai_bf16", .image = "minimax_m3_act", .threads = 256 }, .usage = .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 } } },
        .{ .launch = .{ .name = "minimax_m3_act tf_mm3_swigluoai_f32", .symbol = "tf_mm3_swigluoai_f32", .image = "minimax_m3_act", .threads = 256 }, .usage = .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 } } },
        .{ .launch = .{ .name = "minimax_m3_act tf_mm3_route_warp", .symbol = "tf_mm3_route_warp", .image = "minimax_m3_act", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 33, .static = 0, .bound = 256 }, .{ .regs = 33, .static = 0, .bound = 256 } } },
    };
}
