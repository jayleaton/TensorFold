//! MiniMax-M3.1-Flash row kernels' wiring: the launch parameters the unverified T0 switches become. Host-testable.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const mr = core.minimax_ref;
const rows = cuda.minimax_m3_rows;

pub const Switches = mr.Switches;

/// What the loader left in a norm tensor: raw Gemma `w` or the folded fp32 `1 + w` (Streamer.fold does the latter).
pub const Stored = enum { raw, folded };

pub const Rope = struct { rot: u32 = 64, theta: f64 = 1e7, eps: f32 = 1e-6 };

pub const Error = error{FoldedPlain};

fn add(gemma: bool, stored: Stored) Error!f32 {
    if (gemma) return if (stored == .raw) 1 else 0;
    return if (stored == .raw) 0 else error.FoldedPlain;
}

/// The kernel's arguments for a set of switches. Folded weights cannot give a plain (non-Gemma) norm back.
pub fn opts(sw: Switches, r: Rope, qk: Stored, index: Stored, index_fp16: bool) Error!rows.Opts {
    return .{
        .rot = r.rot,
        .theta = r.theta,
        .eps = r.eps,
        .w_add = try add(sw.gemma_qk_norms, qk),
        .w_add_index = try add(sw.gemma_index_norms, index),
        .flags = .{
            .norm_qk = true,
            .norm_index = true,
            .index_rope = sw.index_rope,
            .index_fp16 = index_fp16,
            .round_norm = sw.bf16_norm_out,
            .gptj = sw.rope_layout == .gptj_interleaved,
        },
    };
}

/// The router: 128 experts over the 6144 hidden values, fp32 weights, x bf16 (the fp32 form takes the cast hidden).
pub const router: rows.Router = .{ .experts = 128, .k = 6144 };

test "switches become flags and weight offsets" {
    const o = try opts(.{}, .{}, .folded, .folded, true);
    try std.testing.expectEqual(@as(f32, 0), o.w_add);
    try std.testing.expectEqual(@as(i32, 1 | 2 | 4 | 8), o.flags.bits());
    const raw = try opts(.{ .rope_layout = .gptj_interleaved, .index_rope = false, .bf16_norm_out = true }, .{}, .raw, .raw, false);
    try std.testing.expectEqual(@as(f32, 1), raw.w_add_index);
    try std.testing.expectEqual(@as(i32, 1 | 2 | 16 | 32), raw.flags.bits());
    try std.testing.expectError(error.FoldedPlain, opts(.{ .gemma_qk_norms = false }, .{}, .folded, .folded, true));
    const plain = try opts(.{ .gemma_index_norms = false }, .{}, .folded, .raw, true);
    try std.testing.expectEqual(@as(f32, 0), plain.w_add_index);
}
