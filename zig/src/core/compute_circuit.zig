//! S3: a compute lane call whose code the frontend accepts runs on the circuit (in-model runner), else in the sandbox.
const std = @import("std");
const api = @import("runner_api.zig");
const runner = @import("runner.zig");
const lane = @import("compute_lane.zig");
const Allocator = std.mem.Allocator;

/// The executor names a report gives; a number never adds the two.
pub const in_model = "in-model runner";
pub const code_tool = "code tool";

/// S3's knobs beside the runner's under the same prefix (PREFIX_WALL_MS, PREFIX_BUSY_MS, PREFIX_VERIFY).
pub const Config = struct {
    wall_ms: u32 = 2000, // the safety net: a run still going is cancelled and the call goes to the sandbox
    busy_ms: u32 = 2000, // how long a call waits for a free backend slot before it goes to the sandbox
    verify: bool = false, // shadow: the sandbox runs too, its result is inserted and the bodies compared

    pub fn parse(get: *const fn ([]const u8) ?[]const u8, comptime prefix: []const u8) error{BadKnob}!Config {
        var c: Config = .{};
        c.wall_ms = try knob(u32, get, prefix ++ "_WALL_MS", c.wall_ms, 1, 3_600_000);
        c.busy_ms = try knob(u32, get, prefix ++ "_BUSY_MS", c.busy_ms, 0, 3_600_000);
        c.verify = try knob(u1, get, prefix ++ "_VERIFY", 0, 0, 1) == 1;
        return c;
    }

    fn knob(comptime T: type, get: *const fn ([]const u8) ?[]const u8, comptime name: []const u8, default: T, min: T, max: T) error{BadKnob}!T {
        const text = get(name) orelse return default;
        if (text.len == 0) return default;
        const v = std.fmt.parseInt(T, text, 10) catch return bad(name, text);
        return if (v < min or v > max) bad(name, text) else v;
    }

    fn bad(name: []const u8, value: []const u8) error{BadKnob} {
        std.log.warn("{s}={s} is not a valid value", .{ name, value });
        return error.BadKnob;
    }
};

/// What the engine hands the server: the runner S3 opened for itself (its own backend) and S3's knobs.
pub const Setup = struct { runner: *runner.Runner, cfg: Config };

/// Why a call went to the sandbox: the frontend's or backend's abstain, a non-ok run, no slot, the wall net, or gone.
pub const Decline = enum { language, unsupported, syntax, name, too_long, too_big, context, abstain, budget, fault, busy, wall, cancelled };
pub const Kind = @FieldType(api.Program, "kind");
/// A call the circuit ran to its halt: the lane's result body made of its stdout, and what ran it.
pub const Owned = struct { body: []const u8, kind: Kind, steps: u64, exec_ns: u64 };
pub const Outcome = union(enum) { owned: Owned, declined: Decline };

/// One server's S3 executor; request threads share it, and every backend call happens under its lock.
pub const Exec = struct {
    frontend: api.Frontend,
    backend: api.Backend,
    max_steps: u64,
    cfg: Config,
    lock: std.atomic.Value(bool) = .init(false),
    next_stream: std.atomic.Value(u64) = .init(1),

    pub fn init(s: Setup) Exec {
        return .{ .frontend = s.runner.frontend, .backend = s.runner.backend, .max_steps = s.runner.cfg.max_steps, .cfg = s.cfg };
    }

    /// The call's code as a fresh fence cell (a lane call sees no earlier call's names), run to its halt or declined.
    pub fn run(e: *Exec, a: Allocator, code: []const u8, hook: ?lane.Cancel) Allocator.Error!Outcome {
        const cell = try fold(a, code) orelse code;
        const t = try e.frontend.translate(e.frontend.ctx, a, .{ .form = .fence, .lang = "python", .code = cell });
        const program = switch (t) {
            .program => |p| p,
            .abstain => |why| return .{ .declined = from(why) },
        };
        const stream = e.next_stream.fetchAdd(1, .monotonic);
        defer e.release(stream);
        const t0 = nowNs();
        const job = while (true) {
            if (e.start(stream, program)) |j| break j else |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Busy => if (nowNs() - t0 >= @as(u64, e.cfg.busy_ms) * std.time.ns_per_ms) return .{ .declined = .busy },
            }
            if (gone(hook)) return .{ .declined = .cancelled };
            sleepMs(1);
        };
        const t1 = nowNs();
        while (true) {
            switch (e.poll(job)) {
                .running => {},
                .done => |r| return switch (r.status) {
                    // `out` is the backend's until its next call for this stream: the body is copied first
                    .ok => .{ .owned = .{ .body = try body(a, r.out), .kind = program.kind, .steps = r.steps, .exec_ns = r.exec_ns } },
                    .abstain => .{ .declined = if (r.abstain) |why| from(why) else .abstain },
                    .budget => .{ .declined = .budget },
                    .fault => .{ .declined = .fault },
                },
            }
            const late = nowNs() - t1 >= @as(u64, e.cfg.wall_ms) * std.time.ns_per_ms;
            if (late or gone(hook)) {
                e.cancel(job);
                return .{ .declined = if (late) .wall else .cancelled };
            }
            sleepMs(1);
        }
    }

    fn start(e: *Exec, stream: u64, program: api.Program) error{ OutOfMemory, Busy }!api.JobId {
        e.acquire();
        defer e.lock.store(false, .release);
        return e.backend.start(stream, program, e.max_steps);
    }

    fn poll(e: *Exec, job: api.JobId) api.Poll {
        e.acquire();
        defer e.lock.store(false, .release);
        return e.backend.poll(job);
    }

    fn cancel(e: *Exec, job: api.JobId) void {
        e.acquire();
        defer e.lock.store(false, .release);
        e.backend.vtable.cancel(e.backend.ctx, job);
    }

    fn release(e: *Exec, stream: u64) void {
        e.acquire();
        defer e.lock.store(false, .release);
        e.backend.vtable.release(e.backend.ctx, stream);
    }

    fn acquire(e: *Exec) void {
        while (e.lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }
};

/// The lane's result body for a run that printed `out` and ended ok: the sandbox's rule, so the bytes match.
pub fn body(a: Allocator, out: []const u8) Allocator.Error![]u8 {
    return lane.runner.body(a, .{ .status = .ok, .stdout = out }, 0, 0);
}

/// `a = ...` / `a op= ...` lines, then print(expr) or an expr: that last line with names bound in parentheses, or null.
pub fn fold(a: Allocator, code: []const u8) Allocator.Error!?[]u8 {
    for (code) |c| switch (c) {
        '0'...'9', 'a'...'z', 'A'...'Z', '_', ' ', '\t', '\r', '\n', '+', '-', '*', '/', '%', '(', ')', ',', '=', '#' => {},
        else => return null,
    };
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const s = arena.allocator();
    var names: [32]Binding = undefined;
    var n: usize = 0;
    var last: ?[]const u8 = null;
    var lines = std.mem.splitScalar(u8, code, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw[0 .. std.mem.indexOfScalar(u8, raw, '#') orelse raw.len], " \t\r");
        if (line.len == 0) continue;
        if (last != null or line[0] == ' ' or line[0] == '\t') return null;
        const eq = std.mem.indexOfScalar(u8, line, '=') orelse {
            last = line;
            continue;
        };
        if (std.mem.indexOfScalarPos(u8, line, eq + 1, '=') != null) return null;
        var lhs = std.mem.trimEnd(u8, line[0..eq], " \t");
        const op = augmented(lhs);
        lhs = std.mem.trimEnd(u8, lhs[0 .. lhs.len - op.len], " \t");
        if (!identifier(lhs) or reserved(lhs)) return null;
        var value = try substitute(s, names[0..n], line[eq + 1 ..]) orelse return null;
        const slot = for (names[0..n], 0..) |x, i| {
            if (std.mem.eql(u8, x.name, lhs)) break i;
        } else null;
        if (op.len > 0) {
            const old = &names[slot orelse return null]; // NameError in Python
            old.used = true;
            value = try std.fmt.allocPrint(s, "{s} {s} ({s})", .{ old.value, op, value });
        }
        if (value.len > max_cell) return null;
        const b: Binding = .{ .name = lhs, .value = try std.fmt.allocPrint(s, "({s})", .{value}) };
        if (slot) |i| {
            if (!names[i].used) return null; // an unread `x = 1//0` raises in Python: a fold would drop it
            names[i] = b;
        } else {
            if (n == names.len) return null;
            names[n] = b;
            n += 1;
        }
    }
    const tail = last orelse return null;
    if (n == 0) return null;
    const call = std.mem.startsWith(u8, tail, "print(") and (closes(tail, "print(".len - 1) orelse 0) == tail.len - 1;
    const expr = if (call) tail["print(".len .. tail.len - 1] else tail;
    if (topComma(expr)) return null;
    const folded = try substitute(s, names[0..n], expr) orelse return null;
    for (names[0..n]) |b| if (!b.used) return null; // the same: every binding is read
    if (folded.len > max_cell) return null;
    return if (call) try std.fmt.allocPrint(a, "print({s})", .{folded}) else try a.dupe(u8, folded);
}

const Binding = struct { name: []const u8, value: []const u8, used: bool = false };
const max_cell = 4096; // the frontend's own length limit

/// `expr` with bound names replaced by their parenthesised values; null on other names (pow and abs pass through).
fn substitute(s: Allocator, names: []Binding, expr: []const u8) Allocator.Error!?[]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < expr.len) {
        const c = expr[i];
        const start = i;
        if (std.ascii.isAlphanumeric(c) or c == '_') {
            while (i < expr.len and (std.ascii.isAlphanumeric(expr[i]) or expr[i] == '_')) i += 1;
            const word = expr[start..i];
            if (std.ascii.isDigit(c)) {
                try out.appendSlice(s, word); // a literal, kept as written
                continue;
            }
            const b = for (names) |*x| {
                if (std.mem.eql(u8, x.name, word)) break x;
            } else {
                if (!std.mem.eql(u8, word, "pow") and !std.mem.eql(u8, word, "abs")) return null;
                try out.appendSlice(s, word);
                continue;
            };
            b.used = true;
            try out.appendSlice(s, b.value);
            if (out.items.len > max_cell) return null;
            continue;
        }
        try out.append(s, c);
        i += 1;
    }
    const text = std.mem.trim(u8, out.items, " \t");
    return if (text.len == 0) null else text;
}

/// The operator of an augmented assignment's left side (`a +=` gives "+"), "" for plain `=`.
fn augmented(lhs: []const u8) []const u8 {
    for ([_][]const u8{ "**", "//", "+", "-", "*", "/", "%" }) |op| if (std.mem.endsWith(u8, lhs, op)) return op;
    return "";
}

fn identifier(word: []const u8) bool {
    if (word.len == 0 or std.ascii.isDigit(word[0])) return false;
    for (word) |c| if (!std.ascii.isAlphanumeric(c) and c != '_') return false;
    return true;
}

/// Names a binding may not take: keywords and the calls the fold treats as builtins.
fn reserved(word: []const u8) bool {
    const words = [_][]const u8{ "print", "pow", "abs", "and", "or", "not", "in", "is", "if", "else", "for", "while", "def", "del", "lambda", "return", "import", "from", "as", "pass", "class", "global", "nonlocal", "assert", "with", "yield", "try", "except", "finally", "raise", "break", "continue", "elif", "await", "async", "True", "False", "None" };
    for (words) |w| if (std.mem.eql(u8, w, word)) return true;
    return false;
}

/// The index of the parenthesis closing the one at `open`, or null.
fn closes(text: []const u8, open: usize) ?usize {
    var depth: usize = 0;
    for (text[open..], open..) |c, i| switch (c) {
        '(' => depth += 1,
        ')' => {
            depth -= 1;
            if (depth == 0) return i;
        },
        else => {},
    };
    return null;
}

/// A comma outside parentheses: print would join several values, a bare expression would be a tuple.
fn topComma(expr: []const u8) bool {
    var depth: usize = 0;
    for (expr) |c| switch (c) {
        '(' => depth += 1,
        ')' => depth -|= 1,
        ',' => if (depth == 0) return true,
        else => {},
    };
    return false;
}

fn from(why: api.Abstain) Decline {
    return switch (why) {
        inline else => |w| @field(Decline, @tagName(w)),
    };
}

fn gone(hook: ?lane.Cancel) bool {
    const c = hook orelse return false;
    return c.gone(c.ctx);
}

fn sleepMs(ms: u64) void {
    const ts: std.c.timespec = .{ .sec = 0, .nsec = @intCast(ms * std.time.ns_per_ms) };
    _ = std.c.nanosleep(&ts, null);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
