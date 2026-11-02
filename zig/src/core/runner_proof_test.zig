//! Proof 2: the multiply and loop replies on the real frontend and the real CPU circuit (TF_RUNNER_CIRCUIT).
const std = @import("std");
const rt = @import("runner_test.zig");
const circuit = @import("runner_circuit.zig");
const backend = @import("runner_circuit_backend.zig");
const frontend = @import("runner_frontend.zig");

const gpa = std.testing.allocator;

/// One reply per lead on the circuit at TF_RUNNER_CIRCUIT; each must equal its lead, `want`, then the model's tail.
fn prove(leads: []const []const u8, wants: []const []const u8) !void {
    const path = std.c.getenv("TF_RUNNER_CIRCUIT") orelse {
        std.debug.print("skip runner proof: TF_RUNNER_CIRCUIT unset\n", .{});
        return error.SkipZigTest;
    };
    var c = try circuit.Circuit.load(gpa, std.mem.span(path));
    defer c.deinit(gpa);
    const b = try backend.backend(gpa, &c, 2);
    defer backend.destroy(b);
    var fake: rt.Back = .{};
    defer fake.deinit();
    const models = try gpa.alloc(rt.Model, leads.len);
    defer gpa.free(models);
    for (models, leads) |*m, l| m.* = .{ .lead = l };
    // TF_RUNNER_LLD: an absolute ld.lld for hosts without lld beside clang (the image installs lld-18 there)
    var ctx: frontend.Context = .{ .io = std.testing.io };
    if (std.c.getenv("TF_RUNNER_CLANG")) |clang| ctx.compiler.clang = std.mem.span(clang);
    if (std.c.getenv("TF_RUNNER_LLD")) |l| ctx.compiler.linker = std.mem.span(l);
    const texts = try rt.replies(models, .{ .frontend = ctx.frontend(), .backend = b, .wait_s = 600 }, &fake);
    defer rt.freeAll(texts);
    try std.testing.expectEqual(@as(u32, 0), fake.started);
    for (texts, leads, wants) |got, l, w| {
        const want = try std.mem.concat(gpa, u8, &.{ l, w, "\nDone." });
        defer gpa.free(want);
        try std.testing.expectEqualStrings(want, got);
    }
}

test "runner proof: a 30-digit multiply cell's Result: is the exact product, computed by the circuit" {
    try prove(&.{rt.mul_cell}, &.{rt.product});
}

test "runner proof: loop cells (a generator sum and an accumulator) print 285 through the circuit" {
    const sum = "```python\nprint(sum(i*i for i in range(10)))\n```\nOutput:\n";
    const acc = "```python\ntotal = 0\nfor i in range(10):\n    total += i * i\nprint(total)\n```\nResult:\n";
    try prove(&.{ sum, acc }, &.{ "285", "285" });
}

const rapi = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");

/// A backend that counts the jobs it starts.
const Count = struct {
    inner: rapi.Backend,
    started: u32 = 0,

    fn backend(c: *Count) rapi.Backend {
        return .{ .ctx = c, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
    }
    fn start(ctx: *anyopaque, stream: u64, p: rapi.Program, budget: u64) error{ OutOfMemory, Busy }!rapi.JobId {
        const c: *Count = @ptrCast(@alignCast(ctx));
        c.started += 1;
        return c.inner.start(stream, p, budget);
    }
    fn poll(ctx: *anyopaque, job: rapi.JobId) rapi.Poll {
        const c: *Count = @ptrCast(@alignCast(ctx));
        return c.inner.poll(job);
    }
    fn cancel(ctx: *anyopaque, job: rapi.JobId) void {
        const c: *Count = @ptrCast(@alignCast(ctx));
        c.inner.cancel(job);
    }
    fn release(ctx: *anyopaque, stream: u64) void {
        const c: *Count = @ptrCast(@alignCast(ctx));
        c.inner.release(stream);
    }
};

/// The real circuit (TF_RUNNER_CIRCUIT) and graph bank (TF_RUNNER_GRAPH_BANK), or a skip.
const Both = struct {
    circuit: circuit.Circuit,
    bank: graph.Bank,
    calc: rapi.Backend,
    graph: rapi.Backend,

    fn open(b: *Both) !void {
        const c = std.c.getenv("TF_RUNNER_CIRCUIT") orelse return error.SkipZigTest;
        const g = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest;
        b.circuit = try circuit.Circuit.load(gpa, std.mem.span(c));
        errdefer b.circuit.deinit(gpa);
        b.bank = try graph.Bank.load(gpa, std.mem.span(g));
        errdefer b.bank.deinit();
        b.calc = try backend.backend(gpa, &b.circuit, 2);
        errdefer backend.destroy(b.calc);
        b.graph = try graph.backend(gpa, &b.bank, 2);
    }
    fn close(b: *Both) void {
        graph.destroy(b.graph);
        backend.destroy(b.calc);
        b.bank.deinit();
        b.circuit.deinit(gpa);
    }
};

fn finish(b: rapi.Backend, job: rapi.JobId) rapi.Result {
    while (true) switch (b.poll(job)) {
        .running => std.Thread.yield() catch {},
        .done => |r| return r,
    };
}

test "runner proof: a REPL bignum cell takes the graph route and gives the calc route's tokens" {
    var both: Both = undefined;
    try both.open();
    defer both.close();
    const leads = [_][]const u8{ "Check:\n>>> 2**100 - 3**50\n", "```python\nprint(2**100 - 3**50)\n```\nOutput:\n" };
    var ctx: frontend.Context = .{ .io = std.testing.io };
    if (std.c.getenv("TF_RUNNER_LLD")) |l| ctx.compiler.linker = std.mem.span(l);
    var fake: rt.Back = .{};
    defer fake.deinit();
    var models = [_]rt.Model{ .{ .lead = leads[0] }, .{ .lead = leads[1] } };
    var calc_only: Count = .{ .inner = both.calc };
    const calc_texts = try rt.replies(&models, .{ .frontend = ctx.frontend(), .backend = calc_only.backend(), .wait_s = 600 }, &fake);
    defer rt.freeAll(calc_texts);
    var calc_fallback: Count = .{ .inner = both.calc };
    var on_graph: Count = .{ .inner = both.graph };
    var routed: route.Routed = .{ .inner = ctx.frontend(), .bank = &both.bank };
    var dispatch: route.Dispatch = .{ .gpa = gpa, .circuit = calc_fallback.backend(), .graph = on_graph.backend() };
    defer dispatch.deinit();
    const graph_texts = try rt.replies(&models, .{ .frontend = routed.frontend(), .backend = dispatch.backend(), .wait_s = 600 }, &fake);
    defer rt.freeAll(graph_texts);
    // one character per token, so equal texts are equal token sequences
    for (calc_texts, graph_texts, leads) |c, g, l| {
        const want = try std.mem.concat(gpa, u8, &.{ l, "1267649882330241709644114435127\nDone." });
        defer gpa.free(want);
        try std.testing.expectEqualStrings(want, c);
        try std.testing.expectEqualStrings(c, g);
    }
    try std.testing.expectEqual(@as(u32, 2), calc_only.started);
    try std.testing.expectEqual(@as(u32, 2), on_graph.started);
    try std.testing.expectEqual(@as(u32, 0), calc_fallback.started);
    try std.testing.expectEqual(@as(u32, 0), fake.started);
}

test "runner proof: pure-integer calc programs print Python's value on the graph bank, and the circuit's where it has one" {
    var both: Both = undefined;
    try both.open();
    defer both.close();
    for ([_][2][]const u8{
        .{ "2**100-3**50", "1267649882330241709644114435127" },
        .{ "47*83", "3901" },
        .{ "-7//2", "-4" },
        .{ "-7%3", "2" },
        .{ "1071%462", "147" },
        .{ "(-3)**5", "-243" },
        .{ "-(2+3)*7", "-35" },
        .{ "314159*271828", "85397212652" },
        .{ "10**61", "10000000000000000000000000000000000000000000000000000000000000" },
        .{ "3-10", "-7" },
        .{ "2**37", "137438953472" },
        .{ "7//-2", "-4" },
        .{ "12345678901234567890*98765432109876543210", "1219326311370217952237463801111263526900" },
        .{ "-(2**40)//3", "-366503875926" },
        .{ "10**62", "100000000000000000000000000000000000000000000000000000000000000" },
        .{ "(10**80+123456789)**2-10**160", "24691357800000000000000000000000000000000000000000000000000000000000000015241578750190521" },
    }) |c| {
        const bytes = (try route.lower(gpa, &both.bank, c[0])) orelse {
            std.debug.print("not routed: {s}\n", .{c[0]});
            return error.TestExpectedRoute;
        };
        defer gpa.free(bytes);
        const g = finish(both.graph, try both.graph.start(1, .{ .kind = .graph, .bytes = bytes }, 1_000_000));
        const got = try gpa.dupe(u8, g.out);
        defer gpa.free(got);
        // TF_DSV41_RUNNER_SETTLE_US: a run here within its steps is the job's result; one step short, a job's again
        const prog: rapi.Program = .{ .kind = .graph, .bytes = bytes };
        const here = both.graph.now(prog, 1_000_000, g.steps).?;
        try std.testing.expect(here.status == g.status and here.steps == g.steps and std.mem.eql(u8, here.out, got));
        try std.testing.expectEqual(null, both.graph.now(prog, 1_000_000, g.steps - 1));
        const want = try std.mem.concat(gpa, u8, &.{ c[1], "\n" });
        defer gpa.free(want);
        // calc firmware prints `error` past its own width (10**61); there the graph's value is checked alone
        const r = finish(both.calc, try both.calc.start(2, .{ .kind = .calc, .bytes = c[0] }, 50_000_000));
        const calc_value = r.status == .ok and !std.mem.eql(u8, r.out, "error\n");
        if (g.status != .ok or !std.mem.eql(u8, want, got) or (calc_value and !std.mem.eql(u8, r.out, got))) {
            std.debug.print("{s}: circuit {t} '{s}', graph {t} '{s}' in {d} steps\n", .{ c[0], r.status, r.out, g.status, got, g.steps });
            return error.TestExpectedEqual;
        }
    }
}

/// S1's d_mirror reply to big-number-019 (s1-guide-20261101T194134Z): the whole problem as one cell, in the content.
const divmod_cell = "```python\na = 314159265358979323846264338327950288419716939937510\nb = 271828182845904523536\nq, r = divmod(a, b)\nprint(q)\nprint(r)\nprint(q*b + r == a)\n```\n\nResult:\n";

test "runner proof: a whole-problem divmod cell with a bool print is owned on the graph bank (S1 d_mirror)" {
    var both: Both = undefined;
    try both.open();
    defer both.close();
    var ctx: frontend.Context = .{ .io = std.testing.io };
    if (std.c.getenv("TF_RUNNER_LLD")) |l| ctx.compiler.linker = std.mem.span(l);
    var fake: rt.Back = .{};
    defer fake.deinit();
    var models = [_]rt.Model{.{ .lead = divmod_cell }};
    var calc_fallback: Count = .{ .inner = both.calc };
    var on_graph: Count = .{ .inner = both.graph };
    var routed: route.Routed = .{ .inner = ctx.frontend(), .bank = &both.bank };
    var dispatch: route.Dispatch = .{ .gpa = gpa, .circuit = calc_fallback.backend(), .graph = on_graph.backend() };
    defer dispatch.deinit();
    const texts = try rt.replies(&models, .{ .frontend = routed.frontend(), .backend = dispatch.backend(), .wait_s = 600 }, &fake);
    defer rt.freeAll(texts);
    try std.testing.expectStringStartsWith(texts[0], divmod_cell ++ "1155727349790921717910215407028\n23663814570294126502\nTrue\n");
    try std.testing.expectEqual(@as(u32, 3), on_graph.started);
    try std.testing.expectEqual(@as(u32, 0), calc_fallback.started);
}

test "runner proof: straight-line integer cells print Python's lines on the graph bank; other shapes refuse" {
    var both: Both = undefined;
    try both.open();
    defer both.close();
    var dispatch: route.Dispatch = .{ .gpa = gpa, .circuit = both.calc, .graph = both.graph };
    defer dispatch.deinit();
    const b = dispatch.backend();
    for ([_][2][]const u8{
        .{ divmod_cell[10 .. divmod_cell.len - 14], "1155727349790921717910215407028\n23663814570294126502\nTrue\n" },
        .{ "a = 17\nb = 5\nq, r = divmod(a, b); print(q, r)\nprint(q*b + r != a)", "3 2\nFalse\n" },
        .{ "x = -7\nprint(x // 2); print(x % 3)  # floor", "-4\n2\n" },
        .{ "print(2**64 == 18446744073709551616)", "True\n" },
        .{ "a=10\nprint(a == 11)", "False\n" },
        .{ "n = 37\nprint(2**n - 1, n)", "137438953471 37\n" },
        .{ "a = 2**10\nprint(a)", "1024\n" },
        .{ "s = 2 + 3\nprint(s * 4)", "20\n" },
        .{ "print(pow(3, 200, 10000000000000000000000000000000000000007))", "6037814813667445162521849728381872075460\n" },
        .{ "import math\nprint(math.gcd(1267650600228229401496703205375, 42391158275216203514294433200))", "8525\n" },
        .{ "from math import isqrt, factorial\nprint(isqrt(100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000))\nprint(factorial(50))", "316227766016837933199889354443271853371955513932521\n30414093201713378043612608166064768844377641568960512000000000000\n" },
    }) |c| {
        const bytes = (try route.program(gpa, &both.bank, c[0])) orelse {
            std.debug.print("not lowered: {s}\n", .{c[0]});
            return error.TestExpectedRoute;
        };
        defer gpa.free(bytes);
        const r = finish(b, try b.start(1, .{ .kind = .graph, .bytes = bytes }, 4_000_000));
        if (r.status != .ok or !std.mem.eql(u8, c[1], r.out)) {
            std.debug.print("{s}: {t} '{s}'\n", .{ c[0], r.status, r.out });
            return error.TestExpectedEqual;
        }
    }
    b.release(1);
    for ([_][]const u8{
        "print(a)",                         "a = 1.5\nprint(a)",             "print(divmod(7, 2))",
        "for i in range(3):\n    print(i)", "a = 7\nprint(a / 2)",           "print('x')",
        "a = 1\nprint(a < 2)",              "q, r = divmod(7, 0)\nprint(q)", "a = 1\nb = a == 1\nprint(b)",
        "print(1 == 1 == 1)",               "x = 5",                         "print = 3\nprint(print)",
        "print(math.gcd(4, 6))",            "print(gcd(4, 6))",              "import math\nprint(math.lcm(4, 6))",
        "print(pow(3, 2, 0) + 1)",          "from math import sqrt",         "import math\nprint(math.pi)",
    }) |src| if (try route.program(gpa, &both.bank, src)) |got| {
        gpa.free(got);
        std.debug.print("lowered: {s}\n", .{src});
        return error.TestUnexpectedRoute;
    };
}

test "runner proof: big-number-020 past the frontend's 62 digits, inline and REPL, runs on the 128-digit bank" {
    var both: Both = undefined;
    try both.open();
    defer both.close();
    var ctx: frontend.Context = .{ .io = std.testing.io };
    if (std.c.getenv("TF_RUNNER_LLD")) |l| ctx.compiler.linker = std.mem.span(l);
    var routed: route.Routed = .{ .inner = ctx.frontend(), .bank = &both.bank };
    const front = routed.frontend();
    const Case = struct { form: rapi.Form, code: []const u8, graph: bool };
    for ([_]Case{
        .{ .form = .inline_expr, .code = "(10^80 + 123456789)^2 − 10^160", .graph = true },
        .{ .form = .inline_expr, .code = "(10^80+123456789)^2-10^160", .graph = true },
        .{ .form = .repl, .code = "(10**80 + 123456789)**2 - 10**160", .graph = true },
        .{ .form = .repl, .code = "(10^80+123456789)^2-10^160", .graph = false }, // XOR in Python
        .{ .form = .inline_expr, .code = "(10^80+123456789)^2 % 7", .graph = false },
        .{ .form = .inline_expr, .code = "10^200 + 1", .graph = false },
    }) |c| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const t = try front.translate(front.ctx, arena.allocator(), .{ .form = c.form, .lang = "python", .code = c.code });
        const on_graph = t == .program and t.program.kind == .graph;
        if (on_graph != c.graph) {
            std.debug.print("{s}: {any}\n", .{ c.code, t });
            return error.TestExpectedEqual;
        }
        if (!on_graph) continue;
        const r = finish(both.graph, try both.graph.start(1, t.program, 4_000_000));
        try std.testing.expectEqualStrings("24691357800000000000000000000000000000000000000000000000000000000000000015241578750190521\n", r.out);
    }
    both.graph.release(1);
}

comptime {
    _ = @import("runner_frontend_inline_regression_test.zig");
    _ = @import("runner_inline_mod_test.zig");
    _ = @import("runner_inline_arith_test.zig");
    _ = @import("runner_inline_chain_test.zig");
}
