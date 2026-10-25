//! MiniMax-M3.1-Flash sparse attention CUDA launch records: nvcc 13.0.88 resource usage for sm_80 to sm_121.
const L = @import("limits.zig");

fn u(regs: u32, static: u32, bound: u32) L.Usage {
    return .{ .regs = regs, .static = static, .bound = bound };
}

pub fn records(comptime Record: type) [3]Record {
    return .{
        .{ .launch = .{ .name = "minimax_m3 tf_mm3_index", .symbol = "tf_mm3_index", .image = "minimax_m3_index", .threads = 128 }, .usage = .{ u(32, 16, 128), u(38, 16, 128), u(38, 16, 128), u(40, 16, 128), u(40, 16, 128) } },
        .{ .launch = .{ .name = "minimax_m3 tf_mm3_select", .symbol = "tf_mm3_select", .image = "minimax_m3_select", .threads = 256 }, .usage = .{ u(30, 136, 256), u(26, 136, 256), u(26, 136, 256), u(32, 136, 256), u(33, 136, 256) } },
        .{ .launch = .{ .name = "minimax_m3 tf_mm3_attn", .symbol = "tf_mm3_attn", .image = "minimax_m3_attn", .threads = 256 }, .usage = .{ u(58, 27136, 256), u(77, 27136, 256), u(77, 27136, 256), u(76, 27136, 256), u(76, 27136, 256) } },
    };
}
