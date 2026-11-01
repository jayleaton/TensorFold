//! The in-model runner's scanner: closed code cells and the model's own result headers, from committed reply bytes.
const std = @import("std");
const api = @import("runner_api.zig");
const Allocator = std.mem.Allocator;

pub const Config = struct {
    fence: bool = true, // ```lang … ``` blocks
    repl: bool = true, // `>>> ` lines
    lookahead: u32 = 64, // tokens after a closed fence in which a result header counts
    reasoning_only: bool = false, // triggers only while the think block is open
};

/// A closed cell; `lang` and `code` live in the scanner until the next `feed`.
pub const Closed = struct { index: u32, form: api.Form, lang: []const u8, code: []const u8 };

/// The model began cell `index`'s result: the runner owns what follows; `fenced`: it opened a fence to close.
pub const Trigger = struct { index: u32, fenced: bool };

pub const Events = struct { closed: ?Closed = null, trigger: ?Trigger = null };

const State = enum { prose, fence, after };

pub const Scanner = struct {
    cfg: Config,
    state: State = .prose,
    line: std.ArrayList(u8) = .empty,
    lang: std.ArrayList(u8) = .empty,
    code: std.ArrayList(u8) = .empty,
    cells: u32 = 0, // cells closed so far; the next one's index
    since: u32 = 0, // tokens since the pending cell closed (state .after)
    armed: ?Trigger = null, // a trigger this token set; later non-blank bytes in the same token spoil it

    pub fn deinit(s: *Scanner, a: Allocator) void {
        s.line.deinit(a);
        s.lang.deinit(a);
        s.code.deinit(a);
    }

    /// One committed token's bytes; `think_open`: the think block is open at this token.
    pub fn feed(s: *Scanner, a: Allocator, bytes: []const u8, think_open: bool) Allocator.Error!Events {
        var ev: Events = .{};
        s.armed = null;
        if (s.state == .after) {
            s.since += 1;
            if (s.since > s.cfg.lookahead) s.state = .prose;
        }
        for (bytes) |b| {
            if (s.armed != null and b != '\n' and b != ' ' and b != '\t' and b != '\r') s.armed = null;
            if (b != '\n') {
                try s.line.append(a, b);
                continue;
            }
            try s.endLine(a, &ev, think_open);
            s.line.clearRetainingCapacity();
        }
        ev.trigger = s.armed;
        return ev;
    }

    fn endLine(s: *Scanner, a: Allocator, ev: *Events, think_open: bool) Allocator.Error!void {
        const raw = std.mem.trimEnd(u8, s.line.items, "\r");
        const allowed = think_open or !s.cfg.reasoning_only;
        switch (s.state) {
            .fence => {
                if (fenceClose(raw)) {
                    ev.closed = s.close(.fence);
                    s.state = .after;
                    s.since = 0;
                } else {
                    try s.code.appendSlice(a, raw);
                    try s.code.append(a, '\n');
                }
            },
            .prose, .after => {
                if (s.state == .after and allowed) {
                    if (header(raw)) return s.fire(false);
                    if (resultFence(raw)) return s.fire(true);
                }
                if (s.cfg.fence) if (fenceOpen(raw)) |lang| {
                    s.lang.clearRetainingCapacity();
                    for (lang) |c| try s.lang.append(a, std.ascii.toLower(c));
                    s.code.clearRetainingCapacity();
                    s.state = .fence;
                    return;
                };
                if (s.cfg.repl) if (replLine(raw)) |src| {
                    s.lang.clearRetainingCapacity();
                    try s.lang.appendSlice(a, "python");
                    s.code.clearRetainingCapacity();
                    try s.code.appendSlice(a, src);
                    ev.closed = s.close(.repl);
                    s.state = .after;
                    if (allowed) s.fire(false);
                    return;
                };
            },
        }
    }

    fn close(s: *Scanner, form: api.Form) Closed {
        const c: Closed = .{ .index = s.cells, .form = form, .lang = s.lang.items, .code = s.code.items };
        s.cells += 1;
        return c;
    }

    fn fire(s: *Scanner, fenced: bool) void {
        s.armed = .{ .index = s.cells - 1, .fenced = fenced };
        s.state = .prose;
    }
};

fn indentOk(line: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < line.len and line[i] == ' ') i += 1;
    return if (i <= 3) line[i..] else null;
}

/// "```lang" (lang may be empty; result-like tags excluded): the lowercase-able tag, else null.
fn fenceOpen(line: []const u8) ?[]const u8 {
    const t = indentOk(line) orelse return null;
    if (!std.mem.startsWith(u8, t, "```")) return null;
    const tag = std.mem.trim(u8, t[3..], " \t");
    for (tag) |c| if (!(std.ascii.isAlphanumeric(c) or c == '+' or c == '#' or c == '-' or c == '_' or c == '.')) return null;
    if (isResultTag(tag)) return null;
    return tag;
}

fn fenceClose(line: []const u8) bool {
    const t = indentOk(line) orelse return false;
    return std.mem.eql(u8, std.mem.trimEnd(u8, t, " \t"), "```");
}

const result_tags = [_][]const u8{ "output", "text", "console", "stdout", "plaintext", "txt", "result" };

fn isResultTag(tag: []const u8) bool {
    for (result_tags) |r| if (std.ascii.eqlIgnoreCase(tag, r)) return true;
    return false;
}

/// "```output" / "```text" / …: the model opening a block for the output.
fn resultFence(line: []const u8) bool {
    const t = indentOk(line) orelse return false;
    if (!std.mem.startsWith(u8, t, "```")) return false;
    return isResultTag(std.mem.trim(u8, t[3..], " \t"));
}

const header_words = [_][]const u8{
    "result",      "results",      "output",           "outputs",            "it prints",
    "this prints", "prints",       "running it gives", "running this gives", "expected output",
    "the output",  "this outputs", "it outputs",       "the result",         "output is",
    "running it",  "running this", "executing it",     "executing this",
};

/// A short line naming the result and ending in ':' ("Result:", "**Output:**", "Running this gives:").
fn header(line: []const u8) bool {
    var t = std.mem.trim(u8, line, " \t*#->_");
    if (t.len < 2 or t.len > 60 or t[t.len - 1] != ':') return false;
    t = std.mem.trim(u8, t[0 .. t.len - 1], " \t*_");
    var buf: [64]u8 = undefined;
    const low = std.ascii.lowerString(&buf, t);
    for (header_words) |w| if (std.mem.eql(u8, low, w)) return true;
    return false;
}

/// ">>> code" (one line, not opening a block): the code, else null.
fn replLine(line: []const u8) ?[]const u8 {
    const t = indentOk(line) orelse return null;
    if (!std.mem.startsWith(u8, t, ">>> ")) return null;
    const src = std.mem.trim(u8, t[4..], " \t");
    if (src.len == 0 or src[src.len - 1] == ':') return null;
    return src;
}

fn run(a: Allocator, s: *Scanner, tokens: []const []const u8, out: *std.ArrayList(Events)) !void {
    for (tokens) |t| try out.append(a, try s.feed(a, t, true));
}

test "runner scan: a fence closes a cell; the header after it triggers that cell, plain" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{} };
    defer s.deinit(a);
    var ev: std.ArrayList(Events) = .empty;
    defer ev.deinit(a);
    try run(a, &s, &.{ "let me check.\n```", "Python\n", "print(2*3)\n", "```\n", "Result", ":\n", "6" }, &ev);
    try std.testing.expectEqual(@as(?Closed, null), ev.items[2].closed);
    const c = ev.items[3].closed.?;
    try std.testing.expectEqual(@as(u32, 0), c.index);
    try std.testing.expectEqual(api.Form.fence, c.form);
    try std.testing.expectEqual(Trigger{ .index = 0, .fenced = false }, ev.items[5].trigger.?);
    for (ev.items[0..5]) |e| try std.testing.expectEqual(@as(?Trigger, null), e.trigger);
}

test "runner scan: cell code and lang are kept exactly" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{} };
    defer s.deinit(a);
    _ = try s.feed(a, "```py\nx = 3\nprint(x*x)\n", true);
    const e = try s.feed(a, "```\n", true);
    try std.testing.expectEqualStrings("py", e.closed.?.lang);
    try std.testing.expectEqualStrings("x = 3\nprint(x*x)\n", e.closed.?.code);
}

test "runner scan: a result fence after a cell triggers fenced; a header inside prose without a cell does not" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{} };
    defer s.deinit(a);
    try std.testing.expectEqual(@as(?Trigger, null), (try s.feed(a, "Result:\n", true)).trigger);
    _ = try s.feed(a, "```python\nprint(1)\n```\n", true);
    try std.testing.expectEqual(Trigger{ .index = 0, .fenced = true }, (try s.feed(a, "```text\n", true)).trigger.?);
}

test "runner scan: a REPL line is a cell and its own trigger; a block opener is not" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{} };
    defer s.deinit(a);
    const e = try s.feed(a, ">>> 2**100\n", true);
    try std.testing.expectEqualStrings("2**100", e.closed.?.code);
    try std.testing.expectEqual(api.Form.repl, e.closed.?.form);
    try std.testing.expectEqual(Trigger{ .index = 0, .fenced = false }, e.trigger.?);
    const f = try s.feed(a, ">>> for i in range(3):\n", true);
    try std.testing.expect(f.closed == null and f.trigger == null);
}

test "runner scan: no trigger past the lookahead, when the model already writes in the trigger's token, or a claim line" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{ .lookahead = 2 } };
    defer s.deinit(a);
    _ = try s.feed(a, "```python\nprint(1)\n```\n", true);
    _ = try s.feed(a, "So", true);
    _ = try s.feed(a, " then", true);
    try std.testing.expectEqual(@as(?Trigger, null), (try s.feed(a, "\nOutput:\n", true)).trigger);
    var t: Scanner = .{ .cfg = .{} };
    defer t.deinit(a);
    _ = try t.feed(a, "```python\nprint(1)\n```\n", true);
    try std.testing.expectEqual(@as(?Trigger, null), (try t.feed(a, "Output:\n1", true)).trigger);
    try std.testing.expectEqual(@as(?Trigger, null), (try t.feed(a, "Result: we need O(n)\n", true)).trigger);
}

test "runner scan: reasoning_only ignores headers outside the think block" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{ .reasoning_only = true } };
    defer s.deinit(a);
    _ = try s.feed(a, "```python\nprint(1)\n```\n", false);
    try std.testing.expectEqual(@as(?Trigger, null), (try s.feed(a, "Output:\n", false)).trigger);
}
