//! The runner's `trace` form: whole integer computations the reasoning states, and the lines where hand work starts.
const std = @import("std");
const text = @import("runner_trace_text.zig");
const expr = @import("runner_trace_expr.zig");
const Allocator = std.mem.Allocator;
const Expr = expr.Expr;

/// The integer frontend's width: a statement whose syntax bound is wider is never a candidate.
pub const max_digits = 62;

/// A stated computation: Python source for the frontend, the expression shown to the model, divmod's re-shaping.
pub const Found = struct { source: []const u8, display: []const u8, divmod: bool = false, same_line: bool = false };

/// Per reply: the names the reasoning defined and the sources already owned (`keep`); one line's work (`scratch`).
pub const Tracer = struct {
    keep: std.heap.ArenaAllocator,
    scratch: std.heap.ArenaAllocator,
    names: std.ArrayList(expr.Name) = .empty,
    fired: std.ArrayList([]const u8) = .empty,

    pub fn init(gpa: Allocator) Tracer {
        return .{ .keep = .init(gpa), .scratch = .init(gpa) };
    }

    pub fn deinit(t: *Tracer) void {
        t.keep.deinit();
        t.scratch.deinit();
    }

    /// The source owned once: the same statement never fires twice in a reply.
    pub fn fire(t: *Tracer, source: []const u8) Allocator.Error!void {
        const a = t.keep.allocator();
        try t.fired.append(a, try a.dupe(u8, source));
    }

    fn firedHas(t: *const Tracer, source: []const u8) bool {
        for (t.fired.items) |f| if (std.mem.eql(u8, f, source)) return true;
        return false;
    }

    fn poison(t: *Tracer, n: []const u8) void {
        for (t.names.items) |*m| if (std.mem.eql(u8, m.name, n)) {
            m.expr = null;
        };
    }

    fn lookup(t: *const Tracer, n: []const u8) ?*const expr.Name {
        for (t.names.items) |*e| if (std.mem.eql(u8, e.name, n)) return e;
        return null;
    }

    /// One completed reasoning line: its definitions, then its first viable statement; text valid until the next call.
    pub fn line(t: *Tracer, raw: []const u8) Allocator.Error!?Found {
        _ = t.scratch.reset(.retain_capacity);
        const a = t.scratch.allocator();
        const clean = try text.stripNotes(a, raw);
        var define: ?Found = null;
        try t.definitions(clean, &define);
        var heads: std.ArrayList(text.Head) = .empty;
        try text.goalHeads(a, clean, &heads);
        try text.bareHeads(a, clean, &heads);
        std.mem.sort(text.Head, heads.items, {}, text.Head.less);
        var first: ?Found = null;
        var last_head: usize = 0;
        for (heads.items) |h| {
            last_head = @max(last_head, h.end);
            if (first != null) continue;
            if (try t.statement(clean, h)) |f| first = f;
        }
        if (first) |*f| {
            f.same_line = text.cueIn(clean[last_head..]);
            return f.*;
        }
        return define;
    }

    fn definitions(t: *Tracer, clean: []const u8, define: *?Found) Allocator.Error!void {
        const a = t.keep.allocator();
        var i: usize = 0;
        while (i < clean.len) {
            const d = text.defAt(clean, i) orelse {
                i += 1;
                continue;
            };
            i = d.rhs;
            if (!text.defLeftOk(clean[0..d.start])) continue;
            const txt_end = text.defEnd(clean, d.rhs) orelse continue;
            const txt = clean[d.rhs..txt_end];
            if (text.approx(clean[d.rhs..charsOn(clean, txt_end, 3)])) continue;
            if (std.mem.indexOfAny(u8, txt, "[]") != null) continue;
            if (text.hypothetical(text.sentenceAt(clean, d.start))) continue;
            const e = try t.parse(txt) orelse continue;
            const n = clean[d.start..d.name_end];
            if (t.lookup(n)) |old| {
                // a second definition by other text poisons the name
                if (old.expr != null and !std.mem.eql(u8, old.expr.?.source, e.source)) t.poison(n);
                continue;
            }
            const kept: Expr = .{ .source = try a.dupe(u8, e.source), .display = try a.dupe(u8, e.display), .bound = e.bound, .ops = e.ops, .constant = e.constant };
            try t.names.append(a, .{ .name = try a.dupe(u8, n), .expr = kept });
            // a first wide definition with nothing after it states that computation ("N = A*2^64 + B.")
            if (hasArith(e.source) and e.bound >= 12 and e.bound <= max_digits and text.sentenceEndsAt(clean, txt_end) and
                !t.firedHas(e.source)) define.* = .{ .source = e.source, .display = e.display };
        }
    }

    fn statement(t: *Tracer, clean: []const u8, h: text.Head) Allocator.Error!?Found {
        const a = t.scratch.allocator();
        const rest = clean[h.end..];
        const ph = text.phrase(rest) orelse return null;
        if (text.approx(rest[0..ph.end])) return null;
        if (text.listed(rest, ph)) return null;
        if (std.mem.indexOfScalar(u8, text.sentenceAt(clean, h.end), '?') != null) return null;
        var parts: [2]Expr = undefined;
        for (ph.parts[0..ph.n], 0..) |p, k| parts[k] = try t.parse(rest[p[0]..p[1]]) orelse return null;
        var f: Found = undefined;
        var bound: u32 = 0;
        switch (ph.kind) {
            .expr => {
                if (parts[0].ops == 0) return null;
                f = .{ .source = parts[0].source, .display = parts[0].display };
                bound = parts[0].bound;
            },
            .divmod => {
                f = .{ .source = try std.fmt.allocPrint(a, "divmod({s}, {s})", .{ parts[0].source, parts[1].source }), .display = try std.fmt.allocPrint(a, "{s} ÷ {s}", .{ try wrapped(a, parts[0]), try wrapped(a, parts[1]) }), .divmod = true };
                bound = parts[0].bound;
            },
            .mul => {
                f = .{ .source = try std.fmt.allocPrint(a, "({s}) * ({s})", .{ parts[0].source, parts[1].source }), .display = try std.fmt.allocPrint(a, "{s} * {s}", .{ try wrapped(a, parts[0]), try wrapped(a, parts[1]) }) };
                bound = expr.sat(parts[0].bound + parts[1].bound);
            },
            .subfrom => {
                f = .{ .source = try std.fmt.allocPrint(a, "({s}) - ({s})", .{ parts[1].source, parts[0].source }), .display = try std.fmt.allocPrint(a, "{s} - {s}", .{ try wrapped(a, parts[1]), try wrapped(a, parts[0]) }) };
                bound = expr.sat(@max(parts[0].bound, parts[1].bound) + 1);
            },
        }
        if (t.firedHas(f.source) or bound > max_digits) return null;
        // a bare sentence counts only when it is plainly a computation: a power or a six-digit literal
        if (h.bare and std.mem.indexOf(u8, f.source, "**") == null and !hasDigitRun(f.source, 6)) return null;
        return f;
    }

    fn parse(t: *Tracer, s: []const u8) Allocator.Error!?Expr {
        return expr.parse(t.scratch.allocator(), t.names.items, s);
    }
};

/// "[exact: <display> = <value>]\n" from a circuit's printed value, or null (abstain: not one integer, or not (q, r)).
pub fn owned(a: Allocator, display: []const u8, divmod: bool, out: []const u8) Allocator.Error!?[]u8 {
    const v = if (std.mem.endsWith(u8, out, "\n")) out[0 .. out.len - 1] else out;
    if (v.len == 0 or std.mem.indexOfScalar(u8, v, '\n') != null) return null;
    if (!divmod) {
        if (!isInt(v)) return null;
        return try std.fmt.allocPrint(a, "[exact: {s} = {s}]\n", .{ display, v });
    }
    if (v.len < 6 or v[0] != '(' or v[v.len - 1] != ')') return null;
    const inner = v[1 .. v.len - 1];
    const comma = std.mem.indexOf(u8, inner, ", ") orelse return null;
    const q = inner[0..comma];
    const r = inner[comma + 2 ..];
    if (!isInt(q) or !isInt(r)) return null;
    return try std.fmt.allocPrint(a, "[exact: {s} = {s} remainder {s}]\n", .{ display, q, r });
}

fn isInt(s: []const u8) bool {
    const d = if (s.len > 0 and s[0] == '-') s[1..] else s;
    if (d.len == 0) return false;
    for (d) |c| if (!std.ascii.isDigit(c)) return false;
    return true;
}

fn wrapped(a: Allocator, e: Expr) Allocator.Error![]const u8 {
    return if (e.ops > 0) std.fmt.allocPrint(a, "({s})", .{e.display}) else e.display;
}

fn hasArith(source: []const u8) bool {
    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        if (source[i] == '*' and i + 1 < source.len and source[i + 1] == '*') {
            i += 1;
            continue;
        }
        if (std.mem.indexOfScalar(u8, "-+*%", source[i]) != null) return true;
    }
    return false;
}

/// The byte index `n` UTF-8 characters past `i` (the study counts characters: "≈" is one).
fn charsOn(s: []const u8, i: usize, n: usize) usize {
    var j = i;
    var left = n;
    while (j < s.len and left > 0) {
        j += 1;
        while (j < s.len and s[j] & 0xC0 == 0x80) j += 1;
        left -= 1;
    }
    return j;
}

fn hasDigitRun(s: []const u8, n: usize) bool {
    var run: usize = 0;
    for (s) |c| {
        run = if (std.ascii.isDigit(c)) run + 1 else 0;
        if (run >= n) return true;
    }
    return false;
}

/// A hand trace starts in this text (a step, table or digit line, long division, carry, borrow, by hand, ...).
pub const cue = text.cueIn;
