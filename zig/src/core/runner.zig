//! The in-model runner: per-reply sessions run closed cells on the circuit and own the model's result positions.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("runner_api.zig");
const scan = @import("runner_scan.zig");
pub const place = @import("runner_place.zig");
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.runner);

pub const Config = struct {
    scan: scan.Config = .{},
    max_steps: u64 = 4_000_000, // per run (a step count: wall time never changes a result)
    max_tokens: u32 = 512, // owned result tokens before the cut note
    workers: u8 = 2,
    device: enum { cpu, cuda } = .cpu,
    circuit: []const u8 = "",
    exact: bool = false, // `calc` / `exact` cells (firmware exact rationals, not any language's print) run
    place: place.Config = .{}, // PREFIX_CPUS / PREFIX_NICE: where the runner's threads run (core/runner_place.zig)

    /// PREFIX "1" enables (unset, "" or "0": null); PREFIX_* knobs override defaults; a bad value is error.BadKnob.
    pub fn parse(get: *const fn ([]const u8) ?[]const u8, comptime prefix: []const u8) error{BadKnob}!?Config {
        const on = get(prefix) orelse return null;
        if (on.len == 0 or std.mem.eql(u8, on, "0")) return null;
        if (!std.mem.eql(u8, on, "1")) return bad(prefix, on);
        var c: Config = .{};
        if (get(prefix ++ "_BACKEND")) |b| if (b.len > 0 and !std.mem.eql(u8, b, "circuit")) return bad(prefix ++ "_BACKEND", b);
        c.circuit = get(prefix ++ "_CIRCUIT") orelse "";
        if (c.circuit.len == 0) return bad(prefix ++ "_CIRCUIT", "(unset)");
        if (get(prefix ++ "_DEVICE")) |d| if (d.len > 0) {
            c.device = std.meta.stringToEnum(@TypeOf(c.device), d) orelse return bad(prefix ++ "_DEVICE", d);
        };
        c.workers = try knob(u8, get, prefix ++ "_WORKERS", c.workers, 1, 64);
        c.max_steps = try knob(u64, get, prefix ++ "_MAX_STEPS", c.max_steps, 1, 1 << 40);
        c.max_tokens = try knob(u32, get, prefix ++ "_MAX_TOKENS", c.max_tokens, 1, 1 << 20);
        c.exact = try knob(u1, get, prefix ++ "_EXACT", 0, 0, 1) == 1;
        c.place = try place.Config.parse(get, prefix);
        c.scan.lookahead = try knob(u32, get, prefix ++ "_LOOKAHEAD", c.scan.lookahead, 1, 4096);
        c.scan.float_window = try knob(u32, get, prefix ++ "_FLOAT_WINDOW", c.scan.float_window, 0, 65536);
        if (get(prefix ++ "_FORMS")) |f| if (f.len > 0) {
            c.scan.fence = false;
            c.scan.repl = false;
            c.scan.inline_math = false;
            var it = std.mem.splitScalar(u8, f, ',');
            while (it.next()) |form| {
                if (std.mem.eql(u8, form, "fence")) c.scan.fence = true else if (std.mem.eql(u8, form, "repl")) c.scan.repl = true else if (std.mem.eql(u8, form, "inline")) c.scan.inline_math = true else return bad(prefix ++ "_FORMS", f);
            }
        };
        if (get(prefix ++ "_SPANS")) |sp| if (sp.len > 0) {
            if (std.mem.eql(u8, sp, "reasoning")) c.scan.reasoning_only = true else if (!std.mem.eql(u8, sp, "all")) return bad(prefix ++ "_SPANS", sp);
        };
        return c;
    }

    fn knob(comptime T: type, get: *const fn ([]const u8) ?[]const u8, comptime name: []const u8, default: T, min: T, max: T) error{BadKnob}!T {
        const text = get(name) orelse return default;
        if (text.len == 0) return default;
        const v = std.fmt.parseInt(T, text, 10) catch return bad(name, text);
        if (v < min or v > max) return bad(name, text);
        return v;
    }

    fn bad(name: []const u8, value: []const u8) error{BadKnob} {
        log.warn("{s}={s} is not a valid value", .{ name, value });
        return error.BadKnob;
    }
};

/// The engine's tokenizer as the runner needs it: one token's bytes, and text to token ids.
pub const Text = struct {
    ctx: ?*anyopaque = null,
    bytes: *const fn (ctx: ?*anyopaque, token: u32) []const u8,
    encode: *const fn (ctx: ?*anyopaque, a: Allocator, text: []const u8) Allocator.Error![]u32,
};

/// The server-wide runner: a frontend, a backend and the server's tokenizer (set at its start); sessions are per reply.
pub const Runner = struct {
    gpa: Allocator,
    cfg: Config,
    frontend: api.Frontend,
    backend: api.Backend,
    text: ?Text = null,
    next_id: u64 = 1,
    placer: ?*place.Placer = null, // the backend's job threads and the translations run placed

    /// A reply's session; null until the server has given the runner its tokenizer.
    pub fn session(r: *Runner) Allocator.Error!?*Session {
        if (r.text == null) return null;
        const s = try r.gpa.create(Session);
        s.* = .{ .r = r, .id = r.next_id, .scan = .{ .cfg = r.cfg.scan } };
        r.next_id += 1;
        return s;
    }
};

const Cell = struct {
    lang: []u8,
    code: []u8,
    form: api.Form,
    arena: std.heap.ArenaAllocator,
    thread: ?std.Thread = null,
    translated: std.atomic.Value(bool) = .init(false),
    translation: api.Translation = .{ .abstain = .unsupported },
    job: ?api.JobId = null,
    result: ?api.Result = null, // out copied into the arena
    abstained: bool = false,
};

/// One reply's cells, runs and the trigger it waits on; every method runs on the engine thread.
pub const Session = struct {
    r: *Runner,
    id: u64,
    scan: scan.Scanner,
    cells: std.ArrayList(*Cell) = .empty,
    waiting: ?scan.Trigger = null,
    held_at: u64 = 0, // monotonic ns when the stream began to wait (the serving log's latency, never a result)
    owned_n: u32 = 0,
    abstained_n: u32 = 0,
    broken: bool = false, // out of memory once: no more cells, no more holds

    pub fn hook(s: *Session) lanes.stream.RunnerHook {
        return .{ .ptr = s, .token = onToken };
    }

    fn onToken(ptr: *anyopaque, token: u32, think_open: bool, can_hold: bool) bool {
        const s: *Session = @ptrCast(@alignCast(ptr));
        if (s.broken) return false;
        return s.feed(token, think_open, can_hold) catch {
            s.broken = true;
            return false;
        };
    }

    fn feed(s: *Session, token: u32, think_open: bool, can_hold: bool) Allocator.Error!bool {
        const tx = s.r.text.?;
        const ev = try s.scan.feed(s.r.gpa, tx.bytes(tx.ctx, token), think_open);
        if (ev.closed) |c| try s.open(c);
        const t = ev.trigger orelse return false;
        if (!can_hold or t.index >= s.cells.items.len) return false;
        s.waiting = t;
        s.held_at = nowNs();
        return true;
    }

    fn open(s: *Session, c: scan.Closed) Allocator.Error!void {
        const gpa = s.r.gpa;
        const cell = try gpa.create(Cell);
        errdefer gpa.destroy(cell);
        cell.* = .{ .lang = try gpa.dupe(u8, c.lang), .code = undefined, .form = c.form, .arena = .init(gpa) };
        errdefer gpa.free(cell.lang);
        cell.code = try gpa.dupe(u8, c.code);
        errdefer gpa.free(cell.code);
        // earlier code cells are the notebook; inline values define nothing
        var n: usize = 0;
        for (s.cells.items) |p| n += @intFromBool(p.form != .inline_expr);
        const prior = try cell.arena.allocator().alloc([]const u8, n);
        n = 0;
        for (s.cells.items) |p| if (p.form != .inline_expr) {
            prior[n] = p.code;
            n += 1;
        };
        try s.cells.append(gpa, cell);
        // exact inline math would contradict a reply reasoning in Python floats
        if (c.float_context) {
            cell.translation = .{ .abstain = .context };
            cell.translated.store(true, .release);
            return;
        }
        if (!s.r.cfg.exact and (std.mem.eql(u8, cell.lang, "calc") or std.mem.eql(u8, cell.lang, "exact"))) {
            cell.translation = .{ .abstain = .language };
            cell.translated.store(true, .release);
            return;
        }
        const task: Translate = .{ .cell = cell, .fe = s.r.frontend, .prior = prior, .placer = s.r.placer };
        cell.thread = std.Thread.spawn(.{}, Translate.spawned, .{task}) catch blk: {
            task.run();
            break :blk null;
        };
    }

    /// Advances translations and runs; the tokens to force when the awaited cell is settled (caller frees them).
    pub fn tick(s: *Session) Allocator.Error!?[]u32 {
        for (s.cells.items) |c| s.advance(c);
        const w = s.waiting orelse return null;
        const c = s.cells.items[w.index];
        if (!c.abstained and c.result == null) return null;
        s.waiting = null;
        const held_ms = (nowNs() -| s.held_at) / std.time.ns_per_ms;
        // an inline value is owned only when the run halted with a one-line value
        if (!c.abstained and w.inline_space != null and (c.result.?.status != .ok or std.mem.indexOfScalar(u8, std.mem.trimEnd(u8, c.result.?.out, "\n"), '\n') != null)) c.abstained = true;
        if (c.abstained) {
            s.abstained_n += 1;
            log.info("reply {d} cell {d} ({s} {t}): abstain ({s}), held {d} ms", .{ s.id, w.index, c.lang, c.form, why(c), held_ms });
            return try s.r.gpa.alloc(u32, 0);
        }
        const res = c.result.?;
        const ids = try s.owned(res, w);
        s.owned_n += 1;
        log.info("reply {d} cell {d} ({s} {t}): owned {d} tokens, {t} after {d} steps, run {d} ms, held {d} ms", .{ s.id, w.index, c.lang, c.form, ids.len, res.status, res.steps, res.exec_ns / std.time.ns_per_ms, held_ms });
        return ids;
    }

    fn why(c: *const Cell) []const u8 {
        if (c.translation == .abstain) return @tagName(c.translation.abstain);
        const res = c.result orelse return "backend start";
        if (res.status == .abstain) return if (res.abstain) |a| @tagName(a) else "backend";
        if (res.status != .ok) return @tagName(res.status); // inline: budget or fault
        return if (res.out.len == 0) "empty output" else "multi-line inline value";
    }

    fn advance(s: *Session, c: *Cell) void {
        if (c.thread) |th| {
            if (!c.translated.load(.acquire)) return;
            th.join();
            c.thread = null;
        } else if (!c.translated.load(.acquire)) return;
        if (c.abstained or c.result != null) return;
        const prog = switch (c.translation) {
            .abstain => {
                c.abstained = true;
                return;
            },
            .program => |p| p,
        };
        const job = c.job orelse {
            c.job = s.r.backend.start(s.id, prog, s.r.cfg.max_steps) catch |e| switch (e) {
                error.Busy => return, // tried again next round: tokens never depend on it
                error.OutOfMemory => {
                    c.abstained = true;
                    return;
                },
            };
            return;
        };
        switch (s.r.backend.poll(job)) {
            .running => {},
            .done => |res| {
                c.job = null;
                var kept = res;
                kept.out = c.arena.allocator().dupe(u8, res.out) catch {
                    c.abstained = true;
                    return;
                };
                c.result = kept;
                if (res.status == .abstain or (res.status == .ok and res.out.len == 0)) c.abstained = true;
            },
        }
    }

    /// The owned text for a settled run (INTERFACES.md §5), as token ids within the token cap.
    fn owned(s: *Session, res: api.Result, w: scan.Trigger) Allocator.Error![]u32 {
        const gpa = s.r.gpa;
        const tx = s.r.text.?;
        var body: std.ArrayList(u8) = .empty;
        defer body.deinit(gpa);
        const out = if (std.mem.endsWith(u8, res.out, "\n")) res.out[0 .. res.out.len - 1] else res.out;
        if (w.inline_space) |space| {
            // only the value after `=`; past the cap it is the model's own text again
            try body.appendSlice(gpa, if (space) " " else "");
            try body.appendSlice(gpa, out);
            const ids = try tx.encode(tx.ctx, gpa, body.items);
            if (ids.len <= s.r.cfg.max_tokens) return ids;
            gpa.free(ids);
            return try gpa.alloc(u32, 0);
        }
        try body.appendSlice(gpa, out);
        switch (res.status) {
            .ok, .abstain => {},
            .budget => try body.print(gpa, "\n[stopped after {d} steps]", .{res.steps}),
            .fault => try body.appendSlice(gpa, "\n[machine fault]"),
        }
        const ids = try tx.encode(tx.ctx, gpa, body.items);
        defer gpa.free(ids);
        const tail = try tx.encode(tx.ctx, gpa, if (w.fenced) "\n```\n" else "\n");
        defer gpa.free(tail);
        const note = if (ids.len > s.r.cfg.max_tokens) try tx.encode(tx.ctx, gpa, "\n[output cut]") else try gpa.alloc(u32, 0);
        defer gpa.free(note);
        const keep = ids[0..@min(ids.len, s.r.cfg.max_tokens)];
        return std.mem.concat(gpa, u32, &.{ keep, note, tail });
    }

    /// Whether the session waits on a run (its stream is held at a trigger).
    pub fn awaiting(s: *const Session) bool {
        return s.waiting != null;
    }

    /// Cancels runs, joins translations and frees the session; the backend drops the stream's machines.
    pub fn destroy(s: *Session) void {
        const gpa = s.r.gpa;
        if (s.cells.items.len > 0) log.info("reply {d}: {d} cells closed, {d} owned, {d} abstained at a header, {d} never claimed", .{ s.id, s.cells.items.len, s.owned_n, s.abstained_n, s.cells.items.len - s.owned_n - s.abstained_n });
        for (s.cells.items) |c| {
            if (c.thread) |th| th.join();
            if (c.job) |j| s.r.backend.cancel(j);
            gpa.free(c.lang);
            gpa.free(c.code);
            c.arena.deinit();
            gpa.destroy(c);
        }
        s.r.backend.release(s.id);
        s.cells.deinit(gpa);
        s.scan.deinit(gpa);
        gpa.destroy(s);
    }
};

/// Monotonic nanoseconds for the serving log only.
fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const Translate = struct {
    cell: *Cell,
    fe: api.Frontend,
    prior: []const []const u8,
    placer: ?*place.Placer = null,

    /// On its own thread: placed first. The inline fallback (no thread) stays on the engine thread's CPU and niceness.
    fn spawned(t: Translate) void {
        if (t.placer) |p| p.enter();
        t.run();
    }

    fn run(t: Translate) void {
        const c = t.cell;
        const cell: api.Cell = .{ .form = c.form, .lang = c.lang, .code = c.code, .prior = t.prior };
        c.translation = t.fe.translate(t.fe.ctx, c.arena.allocator(), cell) catch .{ .abstain = .too_big };
        c.translated.store(true, .release);
    }
};

test {
    _ = @import("runner_scan.zig");
    _ = @import("runner_api.zig");
}
