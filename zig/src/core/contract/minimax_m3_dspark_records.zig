//! MiniMax-M3.1-Flash dspark draft launch records: nvcc 13.0.88 cuobjdump, sm_80, 86, 89, 120, 121.
const L = @import("limits.zig");

fn u(regs: u32, static: u32, bound: u32) L.Usage {
    return .{ .regs = regs, .static = static, .bound = bound };
}

pub fn records(comptime Record: type) [11]Record {
    return .{
        .{ .launch = .{ .name = "minimax_m3_dspark tf_mm3_ds_tap", .symbol = "tf_mm3_ds_tap", .image = "minimax_m3_dspark", .threads = 256 }, .usage = .{ u(22, 0, 256), u(24, 0, 256), u(24, 0, 256), u(20, 0, 256), u(20, 0, 256) } },
        .{ .launch = .{ .name = "minimax_m3_dspark tf_mm3_ds_add_bf16", .symbol = "tf_mm3_ds_add_bf16", .image = "minimax_m3_dspark", .threads = 256 }, .usage = .{ u(18, 0, 256), u(18, 0, 256), u(18, 0, 256), u(28, 0, 256), u(28, 0, 256) } },
        .{ .launch = .{ .name = "minimax_m3_dspark tf_mm3_ds_add_f32", .symbol = "tf_mm3_ds_add_f32", .image = "minimax_m3_dspark", .threads = 256 }, .usage = .{ u(14, 0, 256), u(14, 0, 256), u(14, 0, 256), u(26, 0, 256), u(26, 0, 256) } },
        .{ .launch = .{ .name = "minimax_m3_dspark tf_mm3_ds_norm_bf16", .symbol = "tf_mm3_ds_norm_bf16", .image = "minimax_m3_dspark", .threads = 256 }, .usage = .{ u(32, 32, 256), u(34, 32, 256), u(34, 32, 256), u(32, 32, 256), u(32, 32, 256) } },
        .{ .launch = .{ .name = "minimax_m3_dspark tf_mm3_ds_norm_f32", .symbol = "tf_mm3_ds_norm_f32", .image = "minimax_m3_dspark", .threads = 256 }, .usage = .{ u(32, 32, 256), u(34, 32, 256), u(34, 32, 256), u(31, 32, 256), u(31, 32, 256) } },
        .{ .launch = .{ .name = "minimax_m3_dspark tf_mm3_ds_block_heads", .symbol = "tf_mm3_ds_block_heads", .image = "minimax_m3_dspark", .threads = 256 }, .usage = .{ u(45, 4608, 256), u(44, 4608, 256), u(44, 4608, 256), u(35, 4608, 256), u(35, 4608, 256) } },
        .{ .launch = .{ .name = "minimax_m3_dspark tf_mm3_ds_ring_put", .symbol = "tf_mm3_ds_ring_put", .image = "minimax_m3_dspark", .threads = 256 }, .usage = .{ u(46, 4608, 256), u(45, 4608, 256), u(45, 4608, 256), u(38, 4608, 256), u(38, 4608, 256) } },
        .{ .launch = .{ .name = "minimax_m3_dspark tf_mm3_ds_attn", .symbol = "tf_mm3_ds_attn", .image = "minimax_m3_dspark", .threads = 256, .dynamic = 74240, .opt_in = true }, .usage = .{ u(39, 0, 256), u(39, 0, 256), u(39, 0, 256), u(40, 0, 256), u(40, 0, 256) } },
        .{ .launch = .{ .name = "minimax_m3_dspark tf_mm3_ds_attn_merge", .symbol = "tf_mm3_ds_attn_merge", .image = "minimax_m3_dspark", .threads = 128 }, .usage = .{ u(40, 0, 128), u(46, 0, 128), u(46, 0, 128), u(40, 0, 128), u(40, 0, 128) } },
        .{ .launch = .{ .name = "minimax_m3_dspark tf_mm3_ds_topc", .symbol = "tf_mm3_ds_topc", .image = "minimax_m3_dspark", .threads = 256 }, .usage = .{ u(32, 4128, 256), u(40, 4128, 256), u(40, 4128, 256), u(30, 4128, 256), u(30, 4128, 256) } },
        .{ .launch = .{ .name = "minimax_m3_dspark tf_mm3_ds_chain", .symbol = "tf_mm3_ds_chain", .image = "minimax_m3_dspark", .threads = 256 }, .usage = .{ u(32, 6288, 256), u(40, 6288, 256), u(40, 6288, 256), u(39, 6288, 256), u(39, 6288, 256) } },
    };
}
