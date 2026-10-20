//! MiniMax-M3's special token ids and sampling defaults from generation_config.json and tokenizer_config.json.
const std = @import("std");

pub const max_eos = 4;

pub const Error = error{ BadMiniMaxM3Tokens, OutOfMemory };

pub const Generation = struct {
    bos: u32 = 0,
    eos: [max_eos]u32 = undefined,
    eos_count: u32 = 0,
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,

    pub fn stops(g: *const Generation) []const u32 {
        return g.eos[0..g.eos_count];
    }
};

fn id(v: std.json.Value) Error!u32 {
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.BadMiniMaxM3Tokens;
    return @intCast(v.integer);
}

fn number(v: std.json.Value) Error!f32 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => error.BadMiniMaxM3Tokens,
    };
}

/// generation_config.json: eos_token_id is one id or a list, and the sampling keys are optional.
pub fn generation(gpa: std.mem.Allocator, bytes: []const u8) Error!Generation {
    var doc = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch |e|
        return if (e == error.OutOfMemory) error.OutOfMemory else error.BadMiniMaxM3Tokens;
    defer doc.deinit();
    if (doc.value != .object) return error.BadMiniMaxM3Tokens;
    const o = doc.value.object;
    var g = Generation{};
    g.bos = try id(o.get("bos_token_id") orelse return error.BadMiniMaxM3Tokens);
    const eos = o.get("eos_token_id") orelse return error.BadMiniMaxM3Tokens;
    if (eos == .array) {
        if (eos.array.items.len == 0 or eos.array.items.len > max_eos) return error.BadMiniMaxM3Tokens;
        for (eos.array.items, 0..) |v, i| g.eos[i] = try id(v);
        g.eos_count = @intCast(eos.array.items.len);
    } else {
        g.eos[0] = try id(eos);
        g.eos_count = 1;
    }
    if (o.get("temperature")) |v| g.temperature = try number(v);
    if (o.get("top_p")) |v| g.top_p = try number(v);
    if (!(g.temperature >= 0) or !(g.top_p > 0 and g.top_p <= 1)) return error.BadMiniMaxM3Tokens;
    return g;
}

/// The id tokenizer_config.json's added_tokens_decoder gives a special token's text, such as its eos_token.
pub fn special(gpa: std.mem.Allocator, bytes: []const u8, key: []const u8) Error!u32 {
    var doc = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch |e|
        return if (e == error.OutOfMemory) error.OutOfMemory else error.BadMiniMaxM3Tokens;
    defer doc.deinit();
    if (doc.value != .object) return error.BadMiniMaxM3Tokens;
    const text = doc.value.object.get(key) orelse return error.BadMiniMaxM3Tokens;
    if (text != .string) return error.BadMiniMaxM3Tokens;
    const added = doc.value.object.get("added_tokens_decoder") orelse return error.BadMiniMaxM3Tokens;
    if (added != .object) return error.BadMiniMaxM3Tokens;
    var it = added.object.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* != .object) continue;
        const content = e.value_ptr.object.get("content") orelse continue;
        if (content != .string or !std.mem.eql(u8, content.string, text.string)) continue;
        return std.fmt.parseInt(u32, e.key_ptr.*, 10) catch error.BadMiniMaxM3Tokens;
    }
    return error.BadMiniMaxM3Tokens;
}

test "minimax_m3 tokens: the published generation and tokenizer configs agree on bos and eos" {
    const a = std.testing.allocator;
    const g = try generation(a, @embedFile("generation_fixture.json"));
    try std.testing.expectEqual(@as(u32, 200019), g.bos);
    try std.testing.expectEqualSlices(u32, &.{200020}, g.stops());
    try std.testing.expectEqual(@as(f32, 0.95), g.top_p);
    const tok = @embedFile("tokenizer_fixture.json");
    try std.testing.expectEqual(g.bos, try special(a, tok, "bos_token"));
    try std.testing.expectEqual(g.stops()[0], try special(a, tok, "eos_token"));
    try std.testing.expectEqual(@as(u32, 200000), try special(a, tok, "pad_token"));
}

test "minimax_m3 tokens: lists, missing ids and bad sampling values" {
    const a = std.testing.allocator;
    const two = try generation(a, "{\"bos_token_id\": 1, \"eos_token_id\": [2, 3]}");
    try std.testing.expectEqualSlices(u32, &.{ 2, 3 }, two.stops());
    try std.testing.expectError(error.BadMiniMaxM3Tokens, generation(a, "{\"eos_token_id\": 2}"));
    try std.testing.expectError(error.BadMiniMaxM3Tokens, generation(a, "{\"bos_token_id\": 1, \"eos_token_id\": []}"));
    try std.testing.expectError(error.BadMiniMaxM3Tokens, generation(a, "{\"bos_token_id\": 1, \"eos_token_id\": 2, \"top_p\": 0}"));
    try std.testing.expectError(error.BadMiniMaxM3Tokens, special(a, "{\"eos_token\": \"x\", \"added_tokens_decoder\": {}}", "eos_token"));
}
