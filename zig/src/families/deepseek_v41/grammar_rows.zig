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

/// TF_DSV41_GRAMMAR: unset, empty or 1 (the default) serves structured output; 0 refuses it. Every rank must agree.
pub fn enabled() !bool {
    const v = std.c.getenv("TF_DSV41_GRAMMAR") orelse return true;
    const s = std.mem.span(v);
    if (std.mem.eql(u8, s, "0")) return false;
    if (s.len == 0 or std.mem.eql(u8, s, "1")) return true;
    return error.BadGrammarKnob;
}

/// Whether `bits` allows every token below `vocab`: masking a row with it changes no logit on any rank.
pub fn free(bits: []const u32, vocab: u32) bool {
    const full = vocab / 32;
    if (!std.mem.allEqual(u32, bits[0..full], std.math.maxInt(u32))) return false;
    const tail: u5 = @intCast(vocab % 32);
    return tail == 0 or (~bits[full] & ((@as(u32, 1) << tail) - 1)) == 0;
}

/// The rows of `list` that disallow a token, into `out`; null when none do and none are staged (nothing to change).
pub fn constraining(list: []const Row, vocab: u32, staged: u32, out: []Row) ?[]Row {
    var n: usize = 0;
    for (list) |r| if (!free(r.bits, vocab)) {
        out[n] = r;
        n += 1;
    };
    return if (n == 0 and staged == 0) null else out[0..n];
}

pub fn validate(rows: []const Row, words: u32, max_rows: u32) !void {
    if (words == 0 or rows.len > max_rows) return error.GrammarRows;
    for (rows) |r| {
        if (r.bits.len != words) return error.GrammarWords;
        if (r.index) |i| if (i >= max_rows) return error.GrammarRows;
        if (std.mem.allEqual(u32, r.bits, 0)) return error.GrammarDeadEnd;
    }
}
