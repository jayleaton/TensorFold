//! MiniMax-M3.1-Flash index projection and pick launch records: nvcc 13.0.88 cuobjdump, sm_80, 86, 89, 120, 121.
const L = @import("limits.zig");

fn u(regs: u32, static: u32, bound: u32) L.Usage {
    return .{ .regs = regs, .static = static, .bound = bound };
}

pub fn records(comptime Record: type) [3]Record {
    return .{
        .{ .launch = .{ .name = "minimax_m3_proj tf_mm3_proj_bf16", .symbol = "tf_mm3_proj_bf16", .image = "minimax_m3_proj", .threads = 128 }, .usage = .{ u(50, 0, 128), u(48, 0, 128), u(48, 0, 128), u(56, 0, 128), u(56, 0, 128) } },
        .{ .launch = .{ .name = "minimax_m3_proj tf_mm3_proj_f32", .symbol = "tf_mm3_proj_f32", .image = "minimax_m3_proj", .threads = 128 }, .usage = .{ u(56, 0, 128), u(56, 0, 128), u(56, 0, 128), u(58, 0, 128), u(58, 0, 128) } },
        .{ .launch = .{ .name = "minimax_m3_proj tf_mm3_pick", .symbol = "tf_mm3_pick", .image = "minimax_m3_proj", .threads = 256 }, .usage = .{ u(22, 64, 256), u(24, 64, 256), u(24, 64, 256), u(22, 64, 256), u(22, 64, 256) } },
    };
}
