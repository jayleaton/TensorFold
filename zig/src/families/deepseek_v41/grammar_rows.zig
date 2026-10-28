//! Per-row vocabulary masks supplied by shared lanes, before either greedy or keyed selection.
const std = @import("std");

pub const threads: u32 = 128;
pub const symbol = "tf_grammar_mask";

pub inline fn maskWord(row: [*]f32, word: u32, col0: u32, width: u32) void {
    if (word == std.math.maxInt(u32)) return;
    for (0..32) |b| {
        const col = col0 + @as(u32, @intCast(b));
        if (col < width and (word >> @intCast(b)) & 1 == 0) row[col] = -std.math.inf(f32);
    }
}

pub fn maskRows(logits: []f32, width: u32, indices: []const u32, bits: []const u32, words: u32, lo: u32) void {
    for (indices, 0..) |row, r| {
        for (0..width / 32) |k| maskWord(logits[@as(usize, row) * width ..].ptr, bits[r * words + lo / 32 + k], @intCast(k * 32), width);
    }
}

pub const Row = struct { index: ?u32, bits: []const u32 };

pub fn validate(rows: []const Row, words: u32, max_rows: u32) !void {
    if (words == 0 or rows.len > max_rows) return error.GrammarRows;
    for (rows) |r| {
        if (r.bits.len != words) return error.GrammarWords;
        if (r.index) |i| if (i >= max_rows) return error.GrammarRows;
        if (std.mem.allEqual(u32, r.bits, 0)) return error.GrammarDeadEnd;
    }
}
