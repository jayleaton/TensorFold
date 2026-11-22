//! Flash Next's checkpoint-math SwiGLU experts (ck_experts.cu): launch geometry; usage from kernel-contract-limits.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "ck tf_ck4_swiglu_nvfp4", .symbol = "tf_ck4_swiglu_nvfp4", .image = "ck_experts", .threads = 128, .min_sm = 120 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "ck tf_ck4_swiglu_nvfp4_few", .symbol = "tf_ck4_swiglu_nvfp4_few", .image = "ck_experts", .threads = 128, .min_sm = 120 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 118, .static = 0, .bound = 128 }, .{ .regs = 118, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "ck tf_ck4_swiglu_nvfp4_prompt", .symbol = "tf_ck4_swiglu_nvfp4_prompt", .image = "ck_experts", .threads = 256, .min_sm = 120 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 116, .static = 46080, .bound = 256 }, .{ .regs = 116, .static = 46080, .bound = 256 } } },
    .{ .launch = .{ .name = "ck tf_ck4_down_f32_skip", .symbol = "tf_ck4_down_f32_skip", .image = "ck_experts", .threads = 128, .min_sm = 120 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "ck tf_ck4_down_f32_skip_few", .symbol = "tf_ck4_down_f32_skip_few", .image = "ck_experts", .threads = 128, .min_sm = 120 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "ck tf_ck4_down_f32_skip_prompt", .symbol = "tf_ck4_down_f32_skip_prompt", .image = "ck_experts", .threads = 256, .min_sm = 120 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 115, .static = 36864, .bound = 256 }, .{ .regs = 115, .static = 36864, .bound = 256 } } },
};

test "each launch refuses every served SM kernel-contract-limits has not measured, and fits every one it has" {
    for (records) |r| {
        var missing: usize = 0;
        for (limits.sms, r.usage) |sm, u| missing += @intFromBool(registry.targetDecision(r, sm, u) == .applicable and u.regs == 0);
        try std.testing.expectEqual(missing, try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
