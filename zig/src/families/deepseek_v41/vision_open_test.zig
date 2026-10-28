//! Checkpoint geometry and shared native --vision limits; missing metadata fails before GPU initialization.
const std = @import("std");
const t = std.testing;
const readConfig = @import("vision_open.zig").readConfig;
const config =
    \\{"hidden_size":5120,"image_token_id":129264,"vision_config":{
    \\ "num_hidden_layers":32,"hidden_size":1024,"num_attention_heads":16,"intermediate_size":2816,
    \\ "patch_size":14,"downsample_ratio":3,"rope_theta":5000,"max_image_tokens":768,"min_pixels":295936}}
;

test "vision boot reads checkpoint processor and tower geometry, with shared Open.vision budgets" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try readConfig(a, config, .{});
    try t.expectEqual(@as(f32, 5000), c.shape.theta);
    try t.expectEqual(@as(u32, 32), c.shape.layers);
    try t.expectEqual(@as(u32, 5120), c.shape.out);
    try t.expectEqual(@as(u32, 768), c.limits.image_tokens);
    try t.expect(!c.settings.fetch);
    try t.expect(c.settings.bias_vl);
    const smaller = try readConfig(a, config, .{ .image_tokens = 128, .image_tokens_set = true, .max_images = 2 });
    try t.expectEqual(@as(u32, 128), smaller.settings.budget());
    try t.expectEqual(@as(u32, 2), smaller.limits.max_images);
    try t.expectError(error.NoVisionTower, readConfig(a, "{}", .{}));
    try t.expectError(error.BadVisionConfig, readConfig(a, "{\"vision_config\":{}}", .{}));
    try t.expectError(error.BadVisionLimits, readConfig(a, config, .{ .image_tokens = 0, .image_tokens_set = true }));
}

test "vision boot accepts actual public1017 nested checkpoint metadata" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const c = try readConfig(arena.allocator(), @embedFile("fixtures/flash1017-config.json"), .{});
    try t.expectEqual(@as(u32, 5120), c.shape.out);
    try t.expectEqual(@as(u32, 1024), c.shape.dim);
    try t.expectEqual(@as(u32, 32), c.shape.layers);
    try t.expectEqual(@as(u32, 16), c.shape.heads);
    try t.expectEqual(@as(u32, 2816), c.shape.inter);
    try t.expectEqual(@as(u32, 14), c.shape.patch);
    try t.expectEqual(@as(u32, 3), c.shape.ratio);
    try t.expectEqual(@as(f32, 10000), c.shape.theta);
    try t.expectEqual(@as(u32, 129264), c.settings.image_token);
    try t.expectEqual(@as(u32, 1024), c.limits.image_tokens);
}

test "vision output width follows text_config, llm_config, root precedence" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const vision =
        \\{"num_hidden_layers":32,"hidden_size":1024,"num_attention_heads":16,"intermediate_size":2816,
        \\ "patch_size":14,"downsample_ratio":3}
    ;
    const cases = .{
        .{ "\"text_config\":{\"hidden_size\":5120},\"llm_config\":{\"hidden_size\":4096},\"hidden_size\":2048", 5120 },
        .{ "\"llm_config\":{\"hidden_size\":4096},\"hidden_size\":2048", 4096 },
        .{ "\"hidden_size\":2048", 2048 },
        .{ "\"text_config\":null,\"llm_config\":{\"hidden_size\":4096}", 4096 },
        .{ "\"text_config\":{\"hidden_size\":0},\"llm_config\":{\"hidden_size\":4096}", 4096 },
        .{ "\"text_config\":{\"hidden_size\":-1},\"hidden_size\":2048", 2048 },
    };
    inline for (cases) |case| {
        const text = try std.fmt.allocPrint(a, "{{{s},\"vision_config\":{s}}}", .{ case[0], vision });
        const c = try readConfig(a, text, .{});
        try t.expectEqual(@as(u32, case[1]), c.shape.out);
    }
    const missing = try std.fmt.allocPrint(a, "{{\"text_config\":{{}},\"vision_config\":{s}}}", .{vision});
    try t.expectError(error.BadVisionConfig, readConfig(a, missing, .{}));
}
