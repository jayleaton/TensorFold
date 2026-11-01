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
    var dispatch: route.Dispatch = .{ .circuit = calc_fallback.backend(), .graph = on_graph.backend() };
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
    }) |c| {
        const bytes = (try route.lower(gpa, &both.bank, c[0])) orelse {
            std.debug.print("not routed: {s}\n", .{c[0]});
            return error.TestExpectedRoute;
        };
        defer gpa.free(bytes);
        const g = finish(both.graph, try both.graph.start(1, .{ .kind = .graph, .bytes = bytes }, 1_000_000));
        const got = try gpa.dupe(u8, g.out);
        defer gpa.free(got);
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
