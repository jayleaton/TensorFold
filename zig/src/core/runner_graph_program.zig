//! TF_DSV41_RUNNER_BACKEND=graph: a straight-line integer cell (assignments, divmod, prints) as graph packets.
const std = @import("std");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const Allocator = std.mem.Allocator;

/// A straight-line cell's prints run as packets in turn; each piece is `<I|B><N|S> <packet>` on its own line.
pub const seq_magic = "TFGS1\n";

/// Straight-line integer Python (assignments, divmod, `;`, prints) as graph bytes over literals; null: other shapes.
pub fn program(a: Allocator, bank: *const graph.Bank, code: []const u8) Allocator.Error!?[]u8 {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    if (code.len > 4096 or std.mem.indexOfAny(u8, code, "'\"\\[]{}.:") != null) return null;
    var defs: std.StringHashMapUnmanaged([]const u8) = .empty;
    var pieces: std.ArrayList(u8) = .empty;
    var n_pieces: usize = 0;
    var lone: ?[]const u8 = null;
    var lines = std.mem.splitScalar(u8, code, '\n');
    while (lines.next()) |raw| {
        const line = raw[0 .. std.mem.indexOfScalar(u8, raw, '#') orelse raw.len];
        if (line.len > 0 and (line[0] == ' ' or line[0] == '\t') and std.mem.trim(u8, line, " \t\r").len > 0) return null;
        var stmts = std.mem.splitScalar(u8, line, ';');
        while (stmts.next()) |st| {
            const t = std.mem.trim(u8, st, " \t\r");
            if (t.len == 0) continue;
            if (std.mem.startsWith(u8, t, "print(") and t[t.len - 1] == ')') {
                const args = try splitArgs(arena, t[6 .. t.len - 1]) orelse return null;
                if (args.len == 0) return null;
                for (args, 0..) |arg, i| {
                    const last = i + 1 == args.len;
                    const one = try piece(arena, bank, &defs, arg) orelse return null;
                    if (n_pieces >= 16) return null;
                    try pieces.print(arena, "{c}{c} {s}\n", .{ one.kind, @as(u8, if (last) 'N' else 'S'), one.packet });
                    lone = if (n_pieces == 0 and one.kind == 'I' and last) one.packet else null;
                    n_pieces += 1;
                }
                continue;
            }
            const eq = std.mem.indexOfScalar(u8, t, '=') orelse return null;
            if (eq + 1 < t.len and t[eq + 1] == '=') return null;
            const lhs = std.mem.trim(u8, t[0..eq], " \t()");
            const rhs = std.mem.trim(u8, t[eq + 1 ..], " \t");
            if (std.mem.indexOfScalar(u8, lhs, ',')) |comma| {
                // q, r = divmod(x, y): Python's floor quotient and remainder
                const q = std.mem.trim(u8, lhs[0..comma], " \t");
                const rem = std.mem.trim(u8, lhs[comma + 1 ..], " \t");
                if (!name(q) or !name(rem) or !std.mem.startsWith(u8, rhs, "divmod(") or rhs[rhs.len - 1] != ')') return null;
                const args = try splitArgs(arena, rhs[7 .. rhs.len - 1]) orelse return null;
                if (args.len != 2) return null;
                const x = try expand(arena, &defs, args[0]) orelse return null;
                const y = try expand(arena, &defs, args[1]) orelse return null;
                try defs.put(arena, q, try std.mem.concat(arena, u8, &.{ "(", x, "//", y, ")" }));
                try defs.put(arena, rem, try std.mem.concat(arena, u8, &.{ "(", x, "%", y, ")" }));
                continue;
            }
            if (!name(lhs)) return null;
            const v = try expand(arena, &defs, rhs) orelse return null;
            // a plain literal stays bare, so it can still be an exponent
            const bare = v.len > 0 and for (v) |c| {
                if (!std.ascii.isDigit(c)) break false;
            } else true;
            try defs.put(arena, lhs, if (bare) v else try std.mem.concat(arena, u8, &.{ "(", v, ")" }));
        }
    }
    if (n_pieces == 0) return null;
    if (n_pieces == 1) if (lone) |p| return try a.dupe(u8, p);
    return try std.mem.concat(a, u8, &.{ seq_magic, pieces.items });
}

const Piece = struct { kind: u8, packet: []const u8 };

/// One print argument: an integer expression ('I'), or one `==`/`!=` printed as True/False ('B').
fn piece(arena: Allocator, bank: *const graph.Bank, defs: anytype, arg: []const u8) Allocator.Error!?Piece {
    const cmp = std.mem.indexOf(u8, arg, "==") orelse std.mem.indexOf(u8, arg, "!=");
    if (cmp == null) {
        const e = try expand(arena, defs, arg) orelse return null;
        return .{ .kind = 'I', .packet = try route.lower(arena, bank, e) orelse return null };
    }
    const at = cmp.?;
    const l = try expand(arena, defs, arg[0..at]) orelse return null;
    const r = try expand(arena, defs, arg[at + 2 ..]) orelse return null;
    if (std.mem.indexOfAny(u8, l, "=!<>") != null or std.mem.indexOfAny(u8, r, "=!<>") != null) return null;
    // d = l - r; with M = 10^D above |d|: (d+M-1)//M is 1 for d > 0, (-d+M-1)//M is 1 for d < 0, both 0 at d = 0
    const d = try std.mem.concat(arena, u8, &.{ "(", l, ")-(", r, ")" });
    const nodes = try route.shape(arena, d) orelse return null;
    const digits = nodes[nodes.len - 1].digits;
    if (digits > 57) return null;
    const m = try arena.alloc(u8, digits + 1);
    m[0] = '1';
    @memset(m[1..], '0');
    const nz = try std.mem.concat(arena, u8, &.{ "(", d, "+", m, "-1)//", m, "+((", r, ")-(", l, ")+", m, "-1)//", m });
    const e = if (arg[at] == '=') try std.mem.concat(arena, u8, &.{ "1-(", nz, ")" }) else nz;
    return .{ .kind = 'B', .packet = try route.lower(arena, bank, e) orelse return null };
}

/// `src` with every name replaced by its definition and blanks dropped (calc input); null for an undefined name.
fn expand(arena: Allocator, defs: anytype, src: []const u8) Allocator.Error!?[]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < src.len) {
        const c = src[i];
        if (std.ascii.isAlphabetic(c) or c == '_') {
            const start = i;
            while (i < src.len and (std.ascii.isAlphanumeric(src[i]) or src[i] == '_')) i += 1;
            try out.appendSlice(arena, defs.get(src[start..i]) orelse return null);
            continue;
        }
        if (c != ' ' and c != '\t') try out.append(arena, c);
        i += 1;
    }
    return if (out.items.len == 0) null else out.items;
}

/// Top-level comma-separated arguments; null for unbalanced parentheses.
fn splitArgs(arena: Allocator, src: []const u8) Allocator.Error!?[]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var depth: usize = 0;
    var start: usize = 0;
    for (src, 0..) |c, i| switch (c) {
        '(' => depth += 1,
        ')' => depth = std.math.sub(usize, depth, 1) catch return null,
        ',' => if (depth == 0) {
            try out.append(arena, std.mem.trim(u8, src[start..i], " \t"));
            start = i + 1;
        },
        else => {},
    };
    if (depth != 0) return null;
    const last = std.mem.trim(u8, src[start..], " \t");
    if (last.len > 0) try out.append(arena, last) else if (out.items.len > 0) return null;
    return out.items;
}

fn name(s: []const u8) bool {
    if (s.len == 0 or !(std.ascii.isAlphabetic(s[0]) or s[0] == '_')) return false;
    for (s) |c| if (!(std.ascii.isAlphanumeric(c) or c == '_')) return false;
    for ([_][]const u8{ "print", "divmod", "True", "False", "None" }) |k| if (std.mem.eql(u8, s, k)) return false;
    return true;
}

/// Whether earlier cells rebind `print` or `divmod`, which this cell's meaning depends on.
pub fn shadowed(prior: []const []const u8) bool {
    for (prior) |code| for ([_][]const u8{ "print", "divmod" }) |k| {
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, code, at, k)) |i| : (at = i + 1) {
            const rest = std.mem.trimStart(u8, code[i + k.len ..], " \t");
            if (rest.len > 0 and (rest[0] == '=' and (rest.len == 1 or rest[1] != '=')) or std.mem.startsWith(u8, rest, ",")) return true;
            if (i >= 4 and std.mem.eql(u8, code[i - 4 .. i], "def ")) return true;
        }
    };
    return false;
}
