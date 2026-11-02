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
const program = @import("runner_graph_program.zig");

/// A backend that counts the jobs it starts.
const Count = struct {
    inner: rapi.Backend,
    started: u32 = 0,

    fn backend(c: *Count) rapi.Backend {
        return .{ .ctx = c, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
    }
    fn start(ctx: *anyopaque, stream: u64, p: rapi.Program, budget: u64) error{ OutOfMemory, Busy }!rapi.JobId {
        const c: *Count = @ptrCast(@alignCast(ctx));
        // jobs the inner backend took (a Busy refusal starts nothing)
        const id = try c.inner.start(stream, p, budget);
        c.started += 1;
        return id;
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
    var dispatch: route.Dispatch = .{ .gpa = gpa, .circuit = calc_fallback.backend(), .graph = on_graph.backend(), .bank = &both.bank };
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
    var dispatch: route.Dispatch = .{ .gpa = gpa, .circuit = calc_fallback.backend(), .graph = on_graph.backend(), .bank = &both.bank };
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
    var dispatch: route.Dispatch = .{ .gpa = gpa, .circuit = both.calc, .graph = both.graph, .bank = &both.bank };
    defer dispatch.deinit();
    const b = dispatch.backend();
    for ([_][2][]const u8{
        .{ divmod_cell[10 .. divmod_cell.len - 14], "1155727349790921717910215407028\n23663814570294126502\nTrue\n" },
        .{ "a = 17\nb = 5\nq, r = divmod(a, b); print(q, r)\nprint(q*b + r != a)", "3 2\nFalse\n" },
        .{ "x = -7\nprint(x // 2); print(x % 3)  # floor", "-4\n2\n" },
        .{ "print(2**64 == 18446744073709551616)", "True\n" },
        .{ "a=10\nprint(a == 11)", "False\n" },
        .{ divmod_cell[10..112] ++ "print(q*b + r == a + 1, q*b + r != a - 1, q*b + r - a == 0, a != a)", "False True True False\n" },
        .{ "a = -10**40\nb = 10**40\nprint(a == b, a != b, a == -b, b - a == 2*b)", "False True True True\n" },
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

test "runner proof: bn-011's inline cells (smoke r0-583d0cd875f6): powers to 3^256 own CPython's value, 3^400 abstains" {
    var both: Both = undefined;
    try both.open();
    defer both.close();
    var ctx: frontend.Context = .{ .io = std.testing.io };
    if (std.c.getenv("TF_RUNNER_LLD")) |l| ctx.compiler.linker = std.mem.span(l);
    var routed: route.Routed = .{ .inner = ctx.frontend(), .bank = &both.bank };
    const front = routed.frontend();
    // the reply's own cells verbatim (S1-UNOWNED-CELLS.md "T3 smoke r0-583d0cd875f6"); null: abstains today
    for ([_]struct { []const u8, ?[]const u8 }{
        .{ "3^64", "3433683820292512484657849089281" },
        .{ "3^128", "11790184577738583171520872861412518665678211592275841109096961" },
        .{ "3^256", "139008452377144732764939786789661303114218850808529137991604824430036072629766435941001769154109609521811665540548899435521" },
        .{ "1000^6", "1000000000000000000" },
        .{ "262144*707", "185335808" },
        // 191 digits: past the bank's 128-digit width and the frontend's 62, so `unsupported` (all six abstains)
        .{ "3^400", null },
    }) |c| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const t = try front.translate(front.ctx, arena.allocator(), .{ .form = .inline_expr, .lang = "python", .code = c[0] });
        const want = c[1] orelse {
            if (t != .abstain or t.abstain != .unsupported) {
                std.debug.print("{s}: {any}\n", .{ c[0], t });
                return error.TestExpectedEqual;
            }
            continue;
        };
        if (t != .program or t.program.kind != .graph) {
            std.debug.print("not on the graph route: {s}: {any}\n", .{ c[0], t });
            return error.TestExpectedRoute;
        }
        const r = finish(both.graph, try both.graph.start(1, t.program, 4_000_000));
        try std.testing.expectEqualStrings(want, std.mem.trimEnd(u8, r.out, "\n"));
    }
    both.graph.release(1);
}

test "runner proof: a print reads an earlier piece's value (TFGS2), so d_mirror's check reuses q and r; cheap names do not" {
    var both: Both = undefined;
    try both.open();
    defer both.close();
    const cell = divmod_cell[10 .. divmod_cell.len - 14];
    const bytes = (try route.program(gpa, &both.bank, cell)).?;
    defer gpa.free(bytes);
    try std.testing.expectStringStartsWith(bytes, "TFGS2\n");
    const want = "BN @=$0*271828182845904523536+$1;314159265358979323846264338327950288419716939937510\n";
    try std.testing.expect(std.mem.endsWith(u8, bytes, want));
    // a value with no division or power is recomputed beside the rest, as before
    const cheap = (try route.program(gpa, &both.bank, "a = 7 + 3\nprint(a)\nprint(a * 2)")).?;
    defer gpa.free(cheap);
    try std.testing.expectStringStartsWith(cheap, "TFGS1\n");
    // a template bound to the bank's values: the comparison reads them, so a wrong value moves the answer
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var dispatch: route.Dispatch = .{ .gpa = gpa, .circuit = both.calc, .graph = both.graph, .bank = &both.bank };
    defer dispatch.deinit();
    const b = dispatch.backend();
    for ([_]struct { []const u8, [2]?[]const u8, []const u8 }{
        .{ "@=$0*3+$1;14", .{ "4", "2" }, "1" },
        .{ "@=$0*3+$1;14", .{ "4", "3" }, "0" },
        .{ "@!$0;-5", .{ "-5", null }, "0" },
        .{ "@I$0*$1-1", .{ "-12", "12" }, "-145" },
    }) |c| {
        const p = (try program.bind(arena, &both.bank, c[0], &c[1])).?;
        const r = finish(b, try b.start(1, .{ .kind = .graph, .bytes = p }, 4_000_000));
        try std.testing.expectEqualStrings(c[2], std.mem.trimEnd(u8, r.out, "\n"));
    }
    b.release(1);
    // a reference past the values or to a missing one, and an unknown template op, bind nothing
    try std.testing.expectEqual(null, try program.bind(arena, &both.bank, "@I$2+1", &.{ "1", "2" }));
    try std.testing.expectEqual(null, try program.bind(arena, &both.bank, "@I$1+1", &.{ "1", null }));
    try std.testing.expectEqual(null, try program.bind(arena, &both.bank, "@<$0;1", &.{"1"}));
}

test "runner proof: S1's and S2's six saved cells take the graph route and print CPython's stdout (T1 direct binding)" {
    var both: Both = undefined;
    try both.open();
    defer both.close();
    var ctx: frontend.Context = .{ .io = std.testing.io };
    if (std.c.getenv("TF_RUNNER_LLD")) |l| ctx.compiler.linker = std.mem.span(l);
    var routed: route.Routed = .{ .inner = ctx.frontend(), .bank = &both.bank };
    const front = routed.frontend();
    var dispatch: route.Dispatch = .{ .gpa = gpa, .circuit = both.calc, .graph = both.graph, .bank = &both.bank };
    defer dispatch.deinit();
    const b = dispatch.backend();
    // S1 d_mirror-c1-g2 cells (S1-UNOWNED-CELLS.md) verbatim; S2's saved 008/011/015 as printed cells
    for ([_][2][]const u8{
        .{ "import math\na=12345678901234567890\nb=98765432109876543210\nprint(math.gcd(a,b))", "900000000090\n" },
        .{ "import math\nn = 123456789012345678901234\nr = math.isqrt(n)\nprint(r)\nprint(r*r)\nprint((r+1)*(r+1))\nprint(n - r*r)", "351364182882\n123456789012335541825924\n123456789013038270191689\n10137075310\n" },
        .{ "print(pow(7, 83, 1000000007))", "269823221\n" },
        .{ "print(7**123%1000000007)", "937329259\n" },
        .{ "print(3**400%10**18)", "192443283334088001\n" },
        .{ "print(0xFEDCBA9876543210*2**64+0x0123456789ABCDEF)", "338770000845734292516042252062085074415\n" },
        .{ "a=3\nprint(100//(a+1))\n", "25\n" }, // a computed divisor (T2's table refused it before graph-v4)
    }) |c| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const t = try front.translate(front.ctx, arena.allocator(), .{ .form = .fence, .lang = "python", .code = c[0] });
        if (t != .program or t.program.kind != .graph) {
            std.debug.print("not on the graph route: {s}: {any}\n", .{ c[0], t });
            return error.TestExpectedRoute;
        }
        const r = finish(b, try b.start(1, t.program, 4_000_000));
        if (r.status != .ok or !std.mem.eql(u8, c[1], r.out)) {
            std.debug.print("{s}: {t} '{s}'\n", .{ c[0], r.status, r.out });
            return error.TestExpectedEqual;
        }
    }
    // a zero divisor is Python's ZeroDivisionError: no printed value is owned
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const z = try front.translate(front.ctx, arena.allocator(), .{ .form = .fence, .lang = "python", .code = "a=-1\nprint(100//(a+1))\n" });
    if (z == .program) {
        const r = finish(b, try b.start(1, z.program, 4_000_000));
        std.debug.print("zero divisor: {t} '{s}'\n", .{ r.status, r.out });
        try std.testing.expect(r.status != .ok);
    }
    b.release(1);
}

comptime {
    _ = @import("runner_frontend_inline_regression_test.zig");
    _ = @import("runner_inline_mod_test.zig");
    _ = @import("runner_inline_arith_test.zig");
    _ = @import("runner_inline_chain_test.zig");
    _ = @import("runner_inline_context_test.zig");
    _ = @import("runner_inline_hex_test.zig");
}
