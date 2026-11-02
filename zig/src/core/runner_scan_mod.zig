//! The inline modular step `A mod M =` / `A % M =` (Scanner, Config.inline_mod): its shape and literal modulus.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// A modular step closing a line: `A` as the model wrote it, its literal modulus, and the space flag as inlineExpr.
pub const Mod = struct {
    left: []const u8,
    modulus: []const u8,
    space: bool,

    /// `A mod M` in plain digits for the cell (the graph binder's math syntax; `%` there would read as a percent).
    pub fn code(m: Mod, a: Allocator, out: *std.ArrayList(u8)) Allocator.Error!void {
        var i: usize = 0;
        while (i < m.left.len) : (i += 1) {
            if (std.mem.startsWith(u8, m.left[i..], "×")) {
                try out.append(a, '*');
                i += "×".len - 1;
            } else if (std.mem.startsWith(u8, m.left[i..], "−")) {
                try out.append(a, '-');
                i += "−".len - 1;
            } else if (m.left[i] != ',') try out.append(a, m.left[i]);
        }
        try out.appendSlice(a, " mod ");
        if (std.mem.indexOfScalar(u8, m.modulus, 'e')) |e| {
            try out.print(a, "({s}*10^{s})", .{ m.modulus[0..e], m.modulus[e + 1 ..] });
        } else for (m.modulus) |c| if (c != ',') try out.append(a, c);
    }
};

/// `A mod M =` or `A % M =` ending a line, after any earlier `=`: A over `* × ^ **`, M a nonzero literal; no names.
pub fn inlineMod(line: []const u8) ?Mod {
    var t = line;
    var space = true;
    if (std.mem.endsWith(u8, t, " ")) {
        t = t[0 .. t.len - 1];
        space = false;
    }
    if (t.len < 6 or t[t.len - 1] != '=') return null;
    const before = t[0 .. t.len - 1];
    if (std.mem.indexOfScalar(u8, "=<>!", before[before.len - 1]) != null) return null;
    var seg = before;
    if (std.mem.lastIndexOfScalar(u8, before, '=')) |k| {
        if (k > 0 and std.mem.indexOfScalar(u8, "=<>!", before[k - 1]) != null) return null;
        seg = before[k + 1 ..];
    }
    const s = std.mem.trimEnd(u8, seg, " \t");
    var m_start = s.len;
    while (m_start > 0 and (std.ascii.isDigit(s[m_start - 1]) or std.mem.indexOfScalar(u8, ",^*e", s[m_start - 1]) != null))
        m_start -= 1;
    const modulus = s[m_start..];
    if (!literalModulus(modulus)) return null;
    var rest = std.mem.trimEnd(u8, s[0..m_start], " \t");
    if (std.mem.endsWith(u8, rest, "%")) {
        rest = rest[0 .. rest.len - 1];
    } else if (std.mem.endsWith(u8, rest, "mod") and rest.len > 3 and (rest[rest.len - 4] == ' ' or rest[rest.len - 4] == '\t')) {
        rest = rest[0 .. rest.len - 3];
    } else return null;
    const left_text = std.mem.trimEnd(u8, rest, " \t");
    var start = left_text.len;
    while (start > 0) {
        const b = left_text[start - 1];
        if (std.ascii.isDigit(b) or std.mem.indexOfScalar(u8, " \t*^()+-,", b) != null) {
            start -= 1;
        } else if (start >= 2 and left_text[start - 2] == 0xC3 and b == 0x97) {
            start -= 2; // ×
        } else if (start >= 3 and left_text[start - 3] == 0xE2 and left_text[start - 2] == 0x88 and b == 0x92) {
            start -= 3; // −
        } else break;
    }
    const lead = left_text[0..start];
    const left = std.mem.trim(u8, left_text[start..], " \t");
    if (left.len == 0 or left.len > 120 or !grouped(left)) return null;
    if (!(std.ascii.isDigit(left[0]) or left[0] == '(')) return null;
    if (!(std.ascii.isDigit(left[left.len - 1]) or left[left.len - 1] == ')')) return null;
    if (lead.len > 0) {
        const stop = lead[lead.len - 1];
        // glued to a word or a decimal (x3, 3.5), or after an operator, comparison or symbol: not a whole operand
        if ((std.ascii.isAlphanumeric(stop) or stop == '_' or stop == '$' or stop == '.') and
            left_text[start] != ' ' and left_text[start] != '\t') return null;
        const word = std.mem.trimEnd(u8, lead, " \t");
        if (word.len > 0 and (word[word.len - 1] >= 0x80 or std.mem.indexOfScalar(u8, ",/%<>!", word[word.len - 1]) != null))
            return null;
    }
    var depth: u32 = 0;
    var prev: enum { none, num, op, open, close } = .none;
    var gap = false;
    var i: usize = 0;
    while (i < left.len) : (i += 1) {
        const c = left[i];
        if (c == ' ' or c == '\t') {
            gap = true;
            continue;
        }
        defer gap = false;
        if (std.ascii.isDigit(c) or c == ',') {
            if ((prev == .num and gap) or prev == .close) return null;
            prev = .num;
        } else if (c == '(') {
            if (prev == .num or prev == .close) return null;
            depth += 1;
            prev = .open;
        } else if (c == ')') {
            if (depth == 0 or prev == .op or prev == .open) return null;
            depth -= 1;
            prev = .close;
        } else {
            const n: usize = if (c == 0xC3) 2 else if (c == 0xE2) 3 else 1;
            // which side `mod` takes in `2 + 3 mod 7` is a convention; inside parentheses a sum is one operand
            if ((c == '+' or c == '-' or c == 0xE2) and depth == 0) return null;
            if (prev != .num and prev != .close) return null;
            if (c == '*' and i + 1 < left.len and left[i + 1] == '*') i += 1;
            i += n - 1;
            prev = .op;
        }
    }
    if (depth != 0 or prev == .op) return null;
    return .{ .left = left, .modulus = modulus, .space = space };
}

/// Thousands groups only: each `,` has a digit before it and exactly three digits after it.
fn grouped(text: []const u8) bool {
    for (text, 0..) |c, i| if (c == ',') {
        if (i == 0 or !std.ascii.isDigit(text[i - 1]) or i + 4 > text.len) return false;
        for (text[i + 1 .. i + 4]) |d| if (!std.ascii.isDigit(d)) return false;
        if (i + 4 < text.len and std.ascii.isDigit(text[i + 4])) return false;
    };
    return true;
}

/// A nonzero literal modulus: digits, `b^k`, `b**k`, `dek` (1e9), or thousands groups.
fn literalModulus(m: []const u8) bool {
    // a zero or a leading zero (0, 007, 0^3, 0e5, 0,000): Python raises or reads it otherwise
    if (m.len == 0 or !std.ascii.isDigit(m[0]) or m[0] == '0') return false;
    if (std.mem.indexOfScalar(u8, m, ',') != null) {
        if (std.mem.indexOfAny(u8, m, "^*e") != null) return false;
        const first = std.mem.indexOfScalar(u8, m, ',').?;
        return first <= 3 and grouped(m);
    }
    const sep = std.mem.indexOfAny(u8, m, "^*e") orelse return true;
    const width: usize = if (std.mem.startsWith(u8, m[sep..], "**")) 2 else if (m[sep] == '*') return false else 1;
    const exponent = m[sep + width ..];
    if (exponent.len == 0 or exponent.len > 3) return false;
    for (m[0..sep]) |c| if (!std.ascii.isDigit(c)) return false;
    for (exponent) |c| if (!std.ascii.isDigit(c)) return false;
    return !(exponent[0] == '0' and exponent.len > 1);
}
