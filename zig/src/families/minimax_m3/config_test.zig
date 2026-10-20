//! MiniMax-M3 config reader host checks against the published config.json (MiniMaxAI/MiniMax-M3 at f0e1c1e0).
const std = @import("std");
const config = @import("config.zig");

const fixture = @embedFile("config_fixture.json");

fn edited(from: []const u8, to: []const u8, p: *config.Problem) !config.Config {
    try std.testing.expect(std.mem.indexOf(u8, fixture, from) != null);
    const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, fixture, from, to);
    defer std.testing.allocator.free(bytes);
    return config.parse(std.testing.allocator, bytes, p);
}

test "minimax_m3 config: the published M3 shape" {
    var p: config.Problem = .{};
    const c = try config.parse(std.testing.allocator, fixture, &p);
    try std.testing.expectEqual(@as(u32, 60), c.layers);
    try std.testing.expectEqual(@as(u32, 6144), c.hidden);
    try std.testing.expectEqual(@as(u32, 16), c.group());
    try std.testing.expectEqual(@as(u32, 64), c.rotary_dim);
    try std.testing.expectEqual(@as(f32, 5e6), c.rope_theta);
    try std.testing.expect(c.gemma_norm and !c.tie_embeddings);
    try std.testing.expectEqual(@as(u32, 3), c.denseLayers());
    try std.testing.expectEqual(@as(u32, 57), c.sparseLayers());
    try std.testing.expect(!c.layer[2].moe and c.layer[3].moe and !c.layer[2].sparse and c.layer[3].sparse);
    try std.testing.expect(c.layer[59].index_value_off and !c.layer[0].index_value_off);
    try std.testing.expectEqual(@as(u32, 128), c.experts);
    try std.testing.expectEqual(@as(u32, 4), c.topk);
    try std.testing.expectEqual(@as(u32, 1), c.shared_experts);
    try std.testing.expectEqual(@as(u32, 12288), c.dense_intermediate);
    try std.testing.expectEqual(@as(f32, 2.0), c.routed_scale);
    try std.testing.expectEqual(@as(f32, 7.0), c.swiglu_limit);
    try std.testing.expectEqual(@as(u32, 7), c.mtp_modules);
    try std.testing.expectEqual(@as(u32, 1), c.nextn_layers);
    try std.testing.expectEqual(@as(u32, 1048576), c.context);
    try std.testing.expectEqual(@as(u32, 200064), c.vocab);
    const v = c.vision.?;
    try std.testing.expectEqual(@as(u32, 32), v.layers);
    try std.testing.expectEqual(@as(u32, 1280), v.hidden);
    try std.testing.expectEqual(@as(u32, 2), v.spatial_merge);
    try std.testing.expectEqual(@as(u32, 6144), v.projection);
    try std.testing.expectEqual(@as(?u32, 200025), c.image_token);
    try std.testing.expectEqual(@as(?u32, 200026), c.video_token);
}

test "minimax_m3 config: cache bytes a token adds, whole and per tensor rank" {
    var p: config.Problem = .{};
    const c = try config.parse(std.testing.allocator, fixture, &p);
    // bf16: 2 x 60 layers x 4 heads x 128 for K and V, and one 128-wide index key in each of the 57 sparse layers
    const one = c.cacheBytes(2, 1);
    try std.testing.expectEqual(@as(u64, 122880), one.kv);
    try std.testing.expectEqual(@as(u64, 14592), one.index);
    const two = c.cacheBytes(2, 2);
    try std.testing.expectEqual(@as(u64, 61440), two.kv);
    try std.testing.expectEqual(@as(u64, 14592), two.index);
    const fp8 = c.cacheBytes(1, 2);
    try std.testing.expectEqual(@as(u64, 30720 + 7296), fp8.kv + fp8.index);
    try std.testing.expectEqual(@as(u64, 17 * 128), c.sparse.attended());
}

test "minimax_m3 config: operators outside the plan are refused, not guessed" {
    const Case = struct { from: []const u8, to: []const u8, key: []const u8 };
    const cases = [_]Case{
        .{ .from = "\"attention_output_gate\": false", .to = "\"attention_output_gate\": true", .key = "attention_output_gate" },
        .{ .from = "\"hidden_act\": \"swigluoai\"", .to = "\"hidden_act\": \"silu\"", .key = "hidden_act" },
        .{ .from = "\"scoring_func\": \"sigmoid\"", .to = "\"scoring_func\": \"softmax\"", .key = "scoring_func" },
        .{ .from = "\"sparse_score_type\": \"max\"", .to = "\"sparse_score_type\": \"mean\"", .key = "sparse_score_type" },
        .{ .from = "\"qk_norm_type\": \"per_head\"", .to = "\"qk_norm_type\": \"per_layer\"", .key = "qk_norm_type" },
        .{ .from = "\"use_sparse_attention\": true", .to = "\"use_sparse_attention\": false", .key = "use_sparse_attention" },
        .{ .from = "\"MiniMaxM3SparseForConditionalGeneration\"", .to = "\"MiniMaxM2ForCausalLM\"", .key = "architectures" },
        .{ .from = "\"num_experts_per_tok\": 4", .to = "\"num_experts_per_tok\": 17", .key = "num_local_experts" },
    };
    for (cases) |k| {
        var p: config.Problem = .{};
        try std.testing.expectError(error.UnsupportedMiniMaxM3, edited(k.from, k.to, &p));
        try std.testing.expectEqualStrings(k.key, p.key);
    }
}

test "minimax_m3 config: malformed shapes name the key that broke" {
    const Case = struct { from: []const u8, to: []const u8, key: []const u8 };
    const cases = [_]Case{
        .{ .from = "\"sparse_topk_blocks\"", .to = "\"sparse_topk_blockz\"", .key = "sparse_topk_blocks" },
        .{ .from = "\"num_hidden_layers\": 60", .to = "\"num_hidden_layers\": 59", .key = "moe_layer_freq" },
        .{ .from = "\"rotary_dim\": 64", .to = "\"rotary_dim\": 63", .key = "rotary_dim" },
        .{ .from = "\"partial_rotary_factor\": 0.5", .to = "\"partial_rotary_factor\": 0.25", .key = "partial_rotary_factor" },
        .{ .from = "\"num_key_value_heads\": 4", .to = "\"num_key_value_heads\": 5", .key = "num_key_value_heads" },
        .{ .from = "\"image_token_index\": 200025", .to = "\"image_token_index\": 300000", .key = "image_token_index" },
    };
    for (cases) |k| {
        var p: config.Problem = .{};
        try std.testing.expectError(error.BadMiniMaxM3Config, edited(k.from, k.to, &p));
        try std.testing.expectEqualStrings(k.key, p.key);
    }
    var p: config.Problem = .{};
    try std.testing.expectError(error.BadMiniMaxM3Config, config.parse(std.testing.allocator, "[1, 2]", &p));
    try std.testing.expectError(error.BadMiniMaxM3Config, config.parse(std.testing.allocator, "{\"architectures\":", &p));
}

test "minimax_m3 config: a text-only checkpoint has no tower and no image tokens" {
    var p: config.Problem = .{};
    const c = try edited("\"vision_config\"", "\"vision_config_absent\"", &p);
    try std.testing.expect(c.vision == null and c.image_token == null);
    try std.testing.expectEqual(@as(u32, 57), c.sparseLayers());
    try std.testing.expectError(error.BadMiniMaxM3Config, edited("\"projection_dim\": 6144", "\"projection_dim\": 4096", &p));
    try std.testing.expectEqualStrings("projection_dim", p.key);
}
