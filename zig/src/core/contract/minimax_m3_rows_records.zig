//! MiniMax-M3.1-Flash row kernels' launch records: nvcc 13.0.88 cuobjdump, sm_80, 86, 89, 120, 121.
const L = @import("limits.zig");

fn u(regs: u32, static: u32, bound: u32) L.Usage {
    return .{ .regs = regs, .static = static, .bound = bound };
}

pub fn records(comptime Record: type) [3]Record {
    return .{
        .{ .launch = .{ .name = "minimax_m3_rows tf_mm3_rows", .symbol = "tf_mm3_rows", .image = "minimax_m3_rows", .threads = 256 }, .usage = .{ u(49, 4608, 256), u(52, 4608, 256), u(48, 4608, 256), u(48, 4608, 256), u(48, 4608, 256) } },
        .{ .launch = .{ .name = "minimax_m3_rows tf_mm3_router_bf16", .symbol = "tf_mm3_router_bf16", .image = "minimax_m3_rows", .threads = 128 }, .usage = .{ u(48, 0, 128), u(48, 0, 128), u(48, 0, 128), u(48, 0, 128), u(48, 0, 128) } },
        .{ .launch = .{ .name = "minimax_m3_rows tf_mm3_router_f32", .symbol = "tf_mm3_router_f32", .image = "minimax_m3_rows", .threads = 128 }, .usage = .{ u(56, 0, 128), u(56, 0, 128), u(56, 0, 128), u(51, 0, 128), u(51, 0, 128) } },
    };
}
