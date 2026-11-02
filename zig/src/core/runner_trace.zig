//! The runner's `trace` form: whole integer computations the reasoning states, and the lines where hand work starts.
const std = @import("std");
const text = @import("runner_trace_text.zig");
const expr = @import("runner_trace_expr.zig");
const Allocator = std.mem.Allocator;
const Expr = expr.Expr;

/// The integer frontend's width: a statement whose syntax bound is wider is never a candidate.
pub const max_digits = 62;

/// A stated computation: Python source for the frontend, the expression shown to the model, divmod's re-shaping.
pub const Found = struct { source: []const u8, display: []const u8, divmod: bool = false, same_line: bool = false, run: []const u8 };

/// Per reply: the names the reasoning defined and the sources already owned (`keep`); one line's work (`scratch`).
pub const Tracer = struct {
    keep: std.heap.ArenaAllocator,
    scratch: std.heap.ArenaAllocator,
    names: std.ArrayList(expr.Name) = .empty,
    fired: std.ArrayList([]const u8) = .empty,
    repeat: u8 = 0, // times a displayed line may be owned again when the reasoning restates it (TraceAsk.repeat)
    whole: bool = false, // TF_DSV41_RUNNER_STATED: also comma sentences, gcd(x, y), `L = R.` restated and a hex value

    pub fn init(gpa: Allocator) Tracer {
        return .{ .keep = .init(gpa), .scratch = .init(gpa) };
    }

    pub fn deinit(t: *Tracer) void {
        t.keep.deinit();
        t.scratch.deinit();
    }

    /// A displayed line owned: a restatement (also through names) fires again only `repeat` more times.
    pub fn fire(t: *Tracer, display: []const u8) Allocator.Error!void {
        const a = t.keep.allocator();
        try t.fired.append(a, try a.dupe(u8, display));
    }

    fn firedHas(t: *const Tracer, display: []const u8) bool {
        var n: usize = 0;
        for (t.fired.items) |f| n += @intFromBool(std.mem.eql(u8, f, display));
        return n > t.repeat;
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
        try text.bareHeads(a, clean, t.whole, &heads);
        std.mem.sort(text.Head, heads.items, {}, text.Head.less);
        var first: ?Found = null;
        var last_head: usize = 0;
        for (heads.items) |h| {
            last_head = @max(last_head, h.end);
            if (first != null) continue;
            if (try t.statement(clean, h)) |f| first = f;
        }
        var tail = last_head;
        if (first == null and t.whole) if (try t.wholeShape(clean)) |w| {
            first = w.found;
            tail = w.end;
        };
        if (first) |*f| {
            f.same_line = text.cueIn(clean[tail..]);
            f.run = try printed(a, f.run);
            return f.*;
        }
        if (define) |*d| d.run = try printed(a, d.run);
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
            const kept: Expr = .{ .source = try a.dupe(u8, e.source), .display = try a.dupe(u8, e.display), .bound = e.bound, .ops = e.ops, .constant = e.constant, .run = try a.dupe(u8, e.run) };
            try t.names.append(a, .{ .name = try a.dupe(u8, n), .expr = kept });
            // a first wide definition with nothing after it states that computation ("N = A*2^64 + B.")
            if (hasArith(e.source) and e.bound >= 12 and e.bound <= max_digits and text.sentenceEndsAt(clean, txt_end) and
                !t.firedHas(e.display)) define.* = .{ .source = e.source, .display = e.display, .run = e.run };
        }
    }

    fn statement(t: *Tracer, clean: []const u8, h: text.Head) Allocator.Error!?Found {
        const a = t.scratch.allocator();
        const rest = clean[h.end..];
        const ph = text.phrase(rest, t.whole) orelse return null;
        if (text.approx(rest[0..ph.end])) return null;
        if (text.listed(rest, ph)) return null;
        if (std.mem.indexOfScalar(u8, text.sentenceAt(clean, h.end), '?') != null) return null;
        if (ph.kind == .isqrt or ph.kind == .edit) return t.named(rest, ph);
        var parts: [2]Expr = undefined;
        for (ph.parts[0..ph.n], 0..) |p, k| parts[k] = try t.parse(rest[p[0]..p[1]]) orelse return null;
        var f: Found = undefined;
        var bound: u32 = 0;
        switch (ph.kind) {
            .expr => {
                if (parts[0].ops == 0) return null;
                f = .{ .source = parts[0].source, .display = parts[0].display, .run = parts[0].run };
                bound = parts[0].bound;
            },
            .divmod => {
                f = .{ .source = try std.fmt.allocPrint(a, "divmod({s}, {s})", .{ parts[0].source, parts[1].source }), .display = try std.fmt.allocPrint(a, "{s} ÷ {s}", .{ try wrapped(a, parts[0]), try wrapped(a, parts[1]) }), .divmod = true, .run = try std.fmt.allocPrint(a, "divmod({s}, {s})", .{ parts[0].run, parts[1].run }) };
                bound = parts[0].bound;
            },
            .mul => {
                f = .{ .source = try std.fmt.allocPrint(a, "({s}) * ({s})", .{ parts[0].source, parts[1].source }), .display = try std.fmt.allocPrint(a, "{s} * {s}", .{ try wrapped(a, parts[0]), try wrapped(a, parts[1]) }), .run = try std.fmt.allocPrint(a, "({s}) * ({s})", .{ parts[0].run, parts[1].run }) };
                bound = expr.sat(parts[0].bound + parts[1].bound);
            },
            .gcd => {
                f = .{ .source = try std.fmt.allocPrint(a, "gcd({s}, {s})", .{ parts[0].source, parts[1].source }), .display = try std.fmt.allocPrint(a, "gcd({s}, {s})", .{ parts[0].display, parts[1].display }), .run = try std.fmt.allocPrint(a, "from math import gcd\nprint(gcd({s}, {s}))", .{ parts[0].run, parts[1].run }) };
                bound = @max(parts[0].bound, parts[1].bound);
            },
            .subfrom => {
                f = .{ .source = try std.fmt.allocPrint(a, "({s}) - ({s})", .{ parts[1].source, parts[0].source }), .display = try std.fmt.allocPrint(a, "{s} - {s}", .{ try wrapped(a, parts[1]), try wrapped(a, parts[0]) }), .run = try std.fmt.allocPrint(a, "({s}) - ({s})", .{ parts[1].run, parts[0].run }) };
                bound = expr.sat(@max(parts[0].bound, parts[1].bound) + 1);
            },
            .isqrt, .edit => unreachable,
        }
        if (t.firedHas(f.display) or bound > max_digits) return null;
        // a bare sentence counts only when it is plainly a computation: a power or a six-digit literal
        if (h.bare and std.mem.indexOf(u8, f.source, "**") == null and !hasDigitRun(f.source, 6)) return null;
        return f;
    }

    /// A named problem over literals (STATED): isqrt on the bank's plan, edit distance as one RV32 program.
    fn named(t: *Tracer, rest: []const u8, ph: text.Phrase) Allocator.Error!?Found {
        const a = t.scratch.allocator();
        const x = rest[ph.parts[0][0]..ph.parts[0][1]];
        const f: Found = if (ph.kind == .isqrt) blk: {
            const n = try std.mem.replaceOwned(u8, a, x, ",", "");
            // a zero-led literal is no Python integer, and the plan is as wide as the integer frontend
            if ((n.len > 1 and n[0] == '0') or n.len > max_digits) return null;
            const call = try std.fmt.allocPrint(a, "isqrt({s})", .{n});
            break :blk .{ .source = call, .display = call, .run = try std.fmt.allocPrint(a, "from math import isqrt\nprint({s})", .{call}) };
        } else blk: {
            const y = rest[ph.parts[1][0]..ph.parts[1][1]];
            const call = try std.fmt.allocPrint(a, "edit_distance(\"{s}\", \"{s}\")", .{ x, y });
            break :blk .{ .source = call, .display = call, .run = try std.fmt.allocPrint(a, edit_program, .{ x, y }) };
        };
        return if (t.firedHas(f.display)) null else f;
    }

    /// TF_DSV41_RUNNER_STATED's line shapes: `L = R.` with L over names and literally R, or a hex literal's value.
    fn wholeShape(t: *Tracer, clean: []const u8) Allocator.Error!?struct { found: Found, end: usize } {
        const a = t.scratch.allocator();
        for (clean, 0..) |c, k| {
            if (c != '=' or (k > 0 and std.mem.indexOfScalar(u8, "=<>!", clean[k - 1]) != null)) continue;
            if (k + 1 < clean.len and clean[k + 1] == '=') continue;
            const sentence = text.sentenceAt(clean, k);
            if (text.hypothetical(sentence)) continue;
            const from = @intFromPtr(sentence.ptr) - @intFromPtr(clean.ptr);
            // the left side: names and operators only (a definition or a stated literal is not a restatement)
            const left = std.mem.trim(u8, clean[from..k], " \t");
            if (left.len == 0 or std.mem.indexOfAny(u8, left, "0123456789") != null) continue;
            const l = try t.parse(left) orelse continue;
            if (l.ops == 0) continue;
            var r0 = k + 1;
            while (r0 < clean.len and text.isSpace(clean[r0])) r0 += 1;
            const rest = clean[r0..];
            const ph = text.phrase(rest, false) orelse continue;
            if (ph.kind != .expr or text.approx(rest[0..ph.end])) continue;
            const r = try t.parse(rest[ph.parts[0][0]..ph.parts[0][1]]) orelse continue;
            if (r.ops == 0 or r.bound > max_digits or t.firedHas(r.display)) continue;
            if (!std.mem.eql(u8, try literalParens(a, l.source), r.source)) continue;
            return .{ .found = .{ .source = r.source, .display = r.display, .run = r.run }, .end = r0 + ph.end };
        }
        // `Hex = 0x….` / `This is 0x….`: ten or more hex digits ending a sentence; the radix runs on the circuit
        var i: usize = 0;
        while (std.mem.indexOfPos(u8, clean, i, "0x")) |h| : (i = h + 2) {
            const lead = std.mem.trimEnd(u8, clean[0..h], " \t");
            if (h > 0 and !text.isSpace(clean[h - 1])) continue;
            if (!std.mem.endsWith(u8, lead, "=") and !(lead.len >= 2 and text.wordAt(lead, lead.len - 2, "is", true))) continue;
            var j = h + 2;
            while (j < clean.len and std.ascii.isHex(clean[j])) j += 1;
            if (j - h - 2 < 10 or j - h - 2 > 100 or (j < clean.len and text.isWord(clean[j]))) continue;
            if (!text.sentenceEndsAt(clean, j)) continue;
            const lit = clean[h..j];
            const e = try t.parse(lit) orelse continue;
            if (e.bound > max_digits or t.firedHas(lit)) continue;
            return .{ .found = .{ .source = try a.dupe(u8, lit), .display = try a.dupe(u8, lit), .run = try a.dupe(u8, lit) }, .end = j };
        }
        return null;
    }

    fn parse(t: *Tracer, s: []const u8) Allocator.Error!?Expr {
        return expr.parse(t.scratch.allocator(), t.names.items, s);
    }
};

/// Levenshtein distance (T5's census program): the graph route runs it as a DP packet (runner_graph_edit.zig).
pub const edit_program = @import("runner_graph_edit.zig").program;

/// The owned line's default wording (`{}` is `<display> = <value>`); seeded S2 round 3: 42% fewer tokens than off.
pub const default_format = "[exact, computed by the engine: {}]";

/// The most a request may ask a restated line to be owned again (runner_trace_repeat).
pub const max_repeat = 3;

/// The longest wording a request may give (runner_trace_format).
pub const max_format = 128;

/// A request's wording: one line, at most `max_format` bytes, `{}` exactly once.
pub fn validFormat(f: []const u8) bool {
    return f.len <= max_format and std.mem.count(u8, f, "{}") == 1 and std.mem.indexOfAny(u8, f, "\r\n") == null;
}

/// `format` around "<display> = <value>\n" from a circuit's printed value, or null (not one integer, or not (q, r)).
pub fn owned(a: Allocator, format: []const u8, display: []const u8, divmod: bool, out: []const u8) Allocator.Error!?[]u8 {
    const f = if (format.len > 0) format else default_format;
    const at = std.mem.indexOf(u8, f, "{}").?;
    const v = if (std.mem.endsWith(u8, out, "\n")) out[0 .. out.len - 1] else out;
    if (v.len == 0 or std.mem.indexOfScalar(u8, v, '\n') != null) return null;
    if (!divmod) {
        if (!isInt(v)) return null;
        return try std.fmt.allocPrint(a, "{s}{s} = {s}{s}\n", .{ f[0..at], display, v, f[at + 2 ..] });
    }
    if (v.len < 6 or v[0] != '(' or v[v.len - 1] != ')') return null;
    const inner = v[1 .. v.len - 1];
    const comma = std.mem.indexOf(u8, inner, ", ") orelse return null;
    const q = inner[0..comma];
    const r = inner[comma + 2 ..];
    if (!isInt(q) or !isInt(r)) return null;
    return try std.fmt.allocPrint(a, "{s}{s} = {s} remainder {s}{s}\n", .{ f[0..at], display, q, r, f[at + 2 ..] });
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

/// A whole-statement pow(a, b, m) as `print(...)`: the form the graph bank's modular-power plan takes; else `run`.
fn printed(a: Allocator, run: []const u8) Allocator.Error![]const u8 {
    if (!std.mem.startsWith(u8, run, "pow(") or run[run.len - 1] != ')') return run;
    var depth: usize = 0;
    for (run[3 .. run.len - 1]) |c| {
        if (c == '(') depth += 1;
        if (c == ')') depth -= 1;
        if (depth == 0) return run; // pow(...) is only part of the statement
    }
    return std.mem.concat(a, u8, &.{ "print(", run, ")" });
}

/// A name's source with the brackets around each plain literal dropped: `(98765432109)*(12345678901)` reads as written.
fn literalParens(a: Allocator, source: []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        if (source[i] == '(') {
            var j = i + 1;
            while (j < source.len and std.ascii.isDigit(source[j])) j += 1;
            if (j > i + 1 and j < source.len and source[j] == ')') {
                try out.appendSlice(a, source[i + 1 .. j]);
                i = j;
                continue;
            }
        }
        try out.append(a, source[i]);
    }
    return out.items;
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
