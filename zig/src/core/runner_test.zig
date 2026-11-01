//! The in-model runner end to end on a lane host: a scripted model writes a cell and its result header; runs own it.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const rapi = @import("runner_api.zig");
const runner = @import("runner.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;

const gpa = std.testing.allocator;
const EOS: u32 = 96;
const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
const product = "121932631137021795226185032733622923332237463801111263526900";
const mul_cell = "Let me check.\n```python\nprint(123456789012345678901234567890 * 987654321098765432109876543210)\n```\nResult:\n";

/// Character tokens: printable ASCII 32..126 are 0..94, newline 95, EOS 96 (the fake's vocab is 97).
fn id(c: u8) u32 {
    return if (c == '\n') 95 else c - 32;
}

fn ids(a: std.mem.Allocator, text: []const u8) ![]u32 {
    const out = try a.alloc(u32, text.len);
    for (text, out) |c, *t| t.* = id(c);
    return out;
}

const table = blk: {
    var t: [97][]const u8 = undefined;
    for (0..95) |i| t[i] = &[_]u8{@intCast(i + 32)};
    t[95] = "\n";
    t[96] = "";
    break :blk t;
};

fn tokenBytes(_: ?*anyopaque, token: u32) []const u8 {
    return if (token < table.len) table[token] else "";
}

fn encode(_: ?*anyopaque, a: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error![]u32 {
    return ids(a, text);
}

fn decode(a: std.mem.Allocator, tokens: []const u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    for (tokens) |t| try out.appendSlice(a, tokenBytes(null, t));
    return out.toOwnedSlice(a);
}

/// A model that writes `lead`, then its own guess on one line, then `tail` and EOS, whatever the guess line became.
const Model = struct {
    lead: []const u8,
    guess: []const u8 = "42",
    tail: []const u8 = "Done.",

    fn next(ctx: *anyopaque, history: []const u32) u32 {
        const m: *Model = @ptrCast(@alignCast(ctx));
        var buf: [1024]u8 = undefined;
        const reply = history[prompt.len..];
        var n: usize = 0;
        for (reply) |t| for (tokenBytes(null, t)) |c| {
            buf[n] = c;
            n += 1;
        };
        const r = buf[0..n];
        if (r.len < m.lead.len) return id(m.lead[r.len]);
        var rest = r[m.lead.len..];
        const nl = std.mem.indexOfScalar(u8, rest, '\n') orelse return if (rest.len < m.guess.len) id(m.guess[rest.len]) else id('\n');
        rest = rest[nl + 1 ..];
        if (std.mem.startsWith(u8, rest, "```\n")) rest = rest[4..];
        return if (rest.len < m.tail.len) id(m.tail[rest.len]) else EOS;
    }
};

/// `print(X)` and REPL `X` become calc X; code naming `undefined_` abstains.
fn translate(_: ?*anyopaque, a: std.mem.Allocator, cell: rapi.Cell) std.mem.Allocator.Error!rapi.Translation {
    if (std.mem.indexOf(u8, cell.code, "undefined_") != null) return .{ .abstain = .name };
    var code = std.mem.trim(u8, cell.code, " \n");
    if (std.mem.startsWith(u8, code, "print(") and std.mem.endsWith(u8, code, ")")) code = code[6 .. code.len - 1];
    return .{ .program = .{ .kind = .calc, .bytes = try a.dupe(u8, code) } };
}

/// Answers from a table after `delay` polls, or only once `gate` opens.
const Back = struct {
    delay: u32 = 0,
    gate: std.atomic.Value(bool) = .init(true),
    jobs: std.ArrayList(struct { prog: []u8, polls: u32 }) = .empty,
    started: u32 = 0,

    const answers = [_][2][]const u8{
        .{ "123456789012345678901234567890 * 987654321098765432109876543210", product ++ "\n" },
        .{ "sum(i*i for i in range(10))", "285\n" },
        .{ "2**100", "1267650600228229401496703205376\n" },
    };

    fn backend(b: *Back) rapi.Backend {
        return .{ .ctx = b, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
    }
    fn start(ctx: *anyopaque, _: u64, p: rapi.Program, _: u64) error{ OutOfMemory, Busy }!rapi.JobId {
        const b: *Back = @ptrCast(@alignCast(ctx));
        try b.jobs.append(gpa, .{ .prog = try gpa.dupe(u8, p.bytes), .polls = 0 });
        b.started += 1;
        return b.jobs.items.len - 1;
    }
    fn poll(ctx: *anyopaque, job: rapi.JobId) rapi.Poll {
        const b: *Back = @ptrCast(@alignCast(ctx));
        const j = &b.jobs.items[job];
        j.polls += 1;
        if (j.polls <= b.delay or !b.gate.load(.acquire)) return .running;
        for (answers) |x| if (std.mem.eql(u8, x[0], j.prog)) return .{ .done = .{ .status = .ok, .out = x[1], .steps = 100 } };
        return .{ .done = .{ .status = .fault, .out = "" } };
    }
    fn cancel(_: *anyopaque, _: rapi.JobId) void {}
    fn release(_: *anyopaque, _: u64) void {}
    fn deinit(b: *Back) void {
        for (b.jobs.items) |j| gpa.free(j.prog);
        b.jobs.deinit(gpa);
    }
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?Reason = null,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }
    fn finished(b: *@This()) bool {
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        return b.done != null;
    }
    fn wait(b: *@This()) !Reason {
        for (0..20_000) |_| {
            if (b.finished()) return b.done.?;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

const Opts = struct { runner: bool = true, run_cells: bool = true, delay: u32 = 0, planner: bool = false, streams: usize = 1 };

/// Runs `models` (one request each, concurrently) on one host; the reply texts.
fn replies(models: []Model, o: Opts, back: *Back) ![][]u8 {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var router: Router = .{ .models = models };
    target.script = .{ .ctx = &router, .next = Router.next };
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4 });
    back.delay = o.delay;
    var run: runner.Runner = .{ .gpa = gpa, .cfg = .{ .circuit = "test" }, .frontend = .{ .translate = translate }, .backend = back.backend(), .text = .{ .bytes = tokenBytes, .encode = encode } };
    if (o.runner) host.runner = &run;
    const boxes = try gpa.alloc(Box, models.len);
    defer gpa.free(boxes);
    const prompts = try gpa.alloc([prompt.len + 1]u32, models.len);
    defer gpa.free(prompts);
    const requests = try gpa.alloc(api.Request, models.len);
    defer gpa.free(requests);
    const e = host.engine();
    try host.start();
    defer host.stop();
    for (models, 0..) |_, i| {
        boxes[i] = .{};
        prompts[i] = prompt ++ [_]u32{@intCast(i)}; // the router tells the streams apart by their last prompt token
        requests[i] = .{ .prompt = &prompts[i], .max_tokens = 400, .eos = &.{EOS}, .run_cells = o.run_cells };
        try e.submit(@intCast(i + 1), &requests[i], .{ .ctx = &boxes[i], .event = Box.event });
    }
    const out = try gpa.alloc([]u8, models.len);
    for (boxes, out) |*b, *text| {
        try std.testing.expectEqual(Reason.stop, try b.wait());
        text.* = try decode(gpa, b.tokens.items);
        b.tokens.deinit(gpa);
    }
    return out;
}

/// Each stream's script by the extra prompt token.
const Router = struct {
    models: []Model,
    fn next(ctx: *anyopaque, history: []const u32) u32 {
        const r: *Router = @ptrCast(@alignCast(ctx));
        return Model.next(&r.models[history[prompt.len]], history[1..]);
    }
};

fn freeAll(texts: [][]u8) void {
    for (texts) |t| gpa.free(t);
    gpa.free(texts);
}

fn one(lead: []const u8, o: Opts) ![]u8 {
    var back: Back = .{};
    defer back.deinit();
    var m = [_]Model{.{ .lead = lead }};
    const texts = try replies(&m, o, &back);
    defer gpa.free(texts);
    return texts[0];
}

test "runner: the model's own Result: after a multiply cell is the exact product, in one reply" {
    const got = try one(mul_cell, .{});
    defer gpa.free(got);
    try std.testing.expectEqualStrings(mul_cell ++ product ++ "\nDone.", got);
}

test "runner: a loop cell's result is 285; a REPL line's next line is its value" {
    const lead = "```python\nprint(sum(i*i for i in range(10)))\n```\nOutput:\n";
    const got = try one(lead, .{});
    defer gpa.free(got);
    try std.testing.expectEqualStrings(lead ++ "285\nDone.", got);
    const repl = "Check:\n>>> 2**100\n";
    const r = try one(repl, .{});
    defer gpa.free(r);
    try std.testing.expectEqualStrings(repl ++ "1267650600228229401496703205376\nDone.", r);
}

test "runner: a fenced result block is filled and closed" {
    const lead = "```python\nprint(2**100)\n```\n```text\n";
    const got = try one(lead, .{});
    defer gpa.free(got);
    try std.testing.expectEqualStrings(lead ++ "1267650600228229401496703205376\n```\nDone.", got);
}

test "runner: a run that ends late gives the same reply as one that ends early" {
    const early = try one(mul_cell, .{ .delay = 0 });
    defer gpa.free(early);
    const late = try one(mul_cell, .{ .delay = 8 });
    defer gpa.free(late);
    try std.testing.expectEqualStrings(early, late);
}

test "runner: a held stream waits while another stream decodes to its end" {
    var back: Back = .{};
    defer back.deinit();
    back.gate.store(false, .release);
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var models = [_]Model{ .{ .lead = mul_cell }, .{ .lead = "No code here.\n", .guess = "", .tail = "" } };
    var router: Router = .{ .models = &models };
    target.script = .{ .ctx = &router, .next = Router.next };
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4 });
    var run: runner.Runner = .{ .gpa = gpa, .cfg = .{ .circuit = "test" }, .frontend = .{ .translate = translate }, .backend = back.backend(), .text = .{ .bytes = tokenBytes, .encode = encode } };
    host.runner = &run;
    var boxes = [_]Box{ .{}, .{} };
    const p0 = prompt ++ [_]u32{0};
    const p1 = prompt ++ [_]u32{1};
    const r0: api.Request = .{ .prompt = &p0, .max_tokens = 400, .eos = &.{EOS}, .run_cells = true };
    const r1: api.Request = .{ .prompt = &p1, .max_tokens = 400, .eos = &.{EOS}, .run_cells = true };
    const e = host.engine();
    try host.start();
    defer host.stop();
    try e.submit(1, &r0, .{ .ctx = &boxes[0], .event = Box.event });
    try e.submit(2, &r1, .{ .ctx = &boxes[1], .event = Box.event });
    try std.testing.expectEqual(Reason.stop, try boxes[1].wait());
    std.Io.sleep(std.testing.io, .fromMilliseconds(50), .awake) catch {};
    try std.testing.expect(!boxes[0].finished());
    const held = try decode(gpa, boxes[0].tokens.items);
    defer gpa.free(held);
    try std.testing.expectEqualStrings(mul_cell, held); // it stopped exactly at its trigger
    back.gate.store(true, .release);
    try std.testing.expectEqual(Reason.stop, try boxes[0].wait());
    const got = try decode(gpa, boxes[0].tokens.items);
    defer gpa.free(got);
    try std.testing.expectEqualStrings(mul_cell ++ product ++ "\nDone.", got);
    for (&boxes) |*b| b.tokens.deinit(gpa);
}

test "runner: four concurrent replies equal their solo runs" {
    const leads = [_][]const u8{ mul_cell, "```python\nprint(sum(i*i for i in range(10)))\n```\nResult:\n", "Check:\n>>> 2**100\n", "Plain prose.\n" };
    var models: [4]Model = undefined;
    for (&models, leads) |*m, l| m.* = .{ .lead = l };
    var back: Back = .{ .delay = 3 };
    defer back.deinit();
    const together = try replies(&models, .{ .delay = 3 }, &back);
    defer freeAll(together);
    for (leads, together) |l, t| {
        const solo = try one(l, .{ .delay = 0 });
        defer gpa.free(solo);
        try std.testing.expectEqualStrings(solo, t);
    }
}

test "runner: an abstained cell, a cell with no header, and the runner off all keep the model's own text" {
    const pseudo = "```python\nprint(undefined_total)\n```\nResult:\n";
    const a = try one(pseudo, .{});
    defer gpa.free(a);
    try std.testing.expectEqualStrings(pseudo ++ "42\nDone.", a);
    const quiet = "```python\nprint(2**100)\n```\nSo that works.\n";
    const b = try one(quiet, .{});
    defer gpa.free(b);
    try std.testing.expectEqualStrings(quiet ++ "42\nDone.", b);
    inline for (.{ Opts{ .runner = false }, Opts{ .run_cells = false } }) |o| {
        const c = try one(mul_cell, o);
        defer gpa.free(c);
        try std.testing.expectEqualStrings(mul_cell ++ "42\nDone.", c);
    }
}

test "runner: knobs parse; off unless the flag is 1 with a circuit" {
    const K = struct {
        var set: []const [2][]const u8 = &.{};
        fn get(name: []const u8) ?[]const u8 {
            for (set) |k| if (std.mem.eql(u8, k[0], name)) return k[1];
            return null;
        }
    };
    K.set = &.{};
    try std.testing.expectEqual(null, try runner.Config.parse(&K.get, "TF_R"));
    K.set = &.{.{ "TF_R", "1" }};
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_FORMS", "repl" }, .{ "TF_R_SPANS", "reasoning" }, .{ "TF_R_MAX_STEPS", "99" } };
    const c = (try runner.Config.parse(&K.get, "TF_R")).?;
    try std.testing.expect(!c.scan.fence and c.scan.repl and c.scan.reasoning_only and c.max_steps == 99);
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_BACKEND", "sandbox" } };
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
}

test {
    _ = @import("runner.zig");
}
