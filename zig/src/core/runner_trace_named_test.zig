//! TF_DSV41_RUNNER_STATED named problems (S1 R4 d_mirror rank 1b): isqrt and edit distance over literals.
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const ropen = @import("runner_open.zig");
const trace = @import("runner_trace.zig");
const gpa = std.testing.allocator;

/// The last line's stated computation as display and run, after every line went through one tracer (null: none).
fn expect(whole: bool, lines: []const []const u8, want: ?[2][]const u8) !void {
    var t = trace.Tracer.init(gpa);
    defer t.deinit();
    t.whole = whole;
    var got: ?trace.Found = null;
    for (lines) |l| got = try t.line(l);
    if (want) |w| {
        const f = got orelse {
            std.debug.print("nothing stated (whole {}): {s}\n", .{ whole, lines[lines.len - 1] });
            return error.TestExpectedStated;
        };
        try std.testing.expectEqualStrings(w[0], f.display);
        try std.testing.expectEqualStrings(w[1], f.run);
    } else if (got) |f| {
        std.debug.print("stated (whole {}): {s} from {s}\n", .{ whole, f.display, lines[lines.len - 1] });
        return error.TestUnexpectedStated;
    }
}

fn edit(comptime x: []const u8, comptime y: []const u8) [2][]const u8 {
    return .{ "edit_distance(\"" ++ x ++ "\", \"" ++ y ++ "\")", std.fmt.comptimePrint(trace.edit_program, .{ x, y }) };
}

/// The live rank-1b lines (sqrt-a, sqrt-b, edit, edit-b) and the other spellings of the same statements.
const live = [_]struct { line: []const u8, want: [2][]const u8 }{
    .{ .line = "We need answer. Need compute integer sqrt of 98765432109876543210. Need use Python.", .want = .{ "isqrt(98765432109876543210)", "from math import isqrt\nprint(isqrt(98765432109876543210))" } },
    .{ .line = "We need answer integer sqrt of N=123456789876543210123456. Need compute exactly.", .want = .{ "isqrt(123456789876543210123456)", "from math import isqrt\nprint(isqrt(123456789876543210123456))" } },
    .{ .line = "We need answer compute Levenshtein distance between \"intention\" and \"execution\". Need be careful.", .want = edit("intention", "execution") },
    .{ .line = "We need answer compute Levenshtein distance between \"algorithm\" and \"logarithm\". Need return distance.", .want = edit("algorithm", "logarithm") },
    .{ .line = "Compute the edit distance between 'kitten' and 'sitting'.", .want = edit("kitten", "sitting") },
    .{ .line = "We need the Levenshtein distance from “flaw” to “lawn” exactly.", .want = edit("flaw", "lawn") },
    .{ .line = "Need isqrt(1,000,000,007).", .want = .{ "isqrt(1000000007)", "from math import isqrt\nprint(isqrt(1000000007))" } },
    .{ .line = "We need the floor square root of 10,000,000,000,000,000.", .want = .{ "isqrt(10000000000000000)", "from math import isqrt\nprint(isqrt(10000000000000000))" } },
};

test "runner trace stated: R4 rank-1b named problems are stated with TF_DSV41_RUNNER_STATED and not without it" {
    for (live) |c| {
        try expect(true, &.{c.line}, c.want);
        try expect(false, &.{c.line}, null);
    }
}

test "runner trace stated: near misses of a named problem refuse" {
    for ([_][]const u8{
        "The edit distance is roughly 5.",
        "Need edit distance between 'kitten' and 'sitting' is about 3.",
        "Need edit distance between 'kitten' and 'sitting' maybe.",
        "Need edit distance between 'kitten' and 'sitting'?",
        "We need compute Levenshtein kitten to sitting.", // bare words: g2-edit, not literals
        "Need Levenshtein algorithm to compute.",
        "Need edit distance between s and t.",
        "Need edit distance between 'kitten' and t.",
        "Need edit distance between 'kit\\ten' and 'sitting'.",
        "Need edit distance between 'abcdefghijklmnopqrstuvwxyzabcdefg' and 'b'.", // 33 bytes
        "Need sqrt of 98765432109876543210.", // the real root, maybe: no "integer" or "floor"
        "Need integer sqrt of N.",
        "Need integer sqrt of 0123.",
        "Need integer sqrt of 2^64.",
        "Need integer sqrt of 98765432109876543210.5.",
        "Need integer sqrt of 98765432109876543210 roughly.",
        "Need isqrt(98765432109876543210 + 1).",
        "Need integer sqrt of 1234567890123456789012345678901234567890123456789012345678901234.", // 64 digits
    }) |line| try expect(true, &.{line}, null);
}

fn python(code: []const u8) ![]u8 {
    const r = try std.process.run(gpa, std.testing.io, .{ .argv = &.{ "/usr/bin/python3", "-I", "-S", "-c", code }, .stdout_limit = .limited(4096), .stderr_limit = .limited(4096) });
    defer gpa.free(r.stderr);
    errdefer gpa.free(r.stdout);
    try std.testing.expect(r.term.success());
    return r.stdout;
}

/// The engine's runner (runner_open: Dispatch over the bank and circuit) and a routed frontend beside it.
const Engine = struct {
    runner: *ropen.Runner,
    bank: graph.Bank,
    context: @import("runner_frontend.zig").Context,
    job: u64 = 0,

    var one: ?*Engine = null;

    /// One per test binary; null without the circuit, bank and clang (TF_RUNNER_CLANG) this needs.
    fn open() !?*Engine {
        if (one) |e| return e;
        const circuit = std.c.getenv("TF_RUNNER_CIRCUIT") orelse return null;
        const bank = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return null;
        const clang = std.c.getenv("TF_RUNNER_CLANG") orelse return null;
        // a runner lives as long as the engine: never freed, as in compute_circuit_proof_test
        const p = std.heap.page_allocator;
        const e = try p.create(Engine);
        e.* = .{
            .runner = try ropen.open(p, .{ .circuit = std.mem.span(circuit), .backend = .graph, .graph_bank = std.mem.span(bank), .workers = 1 }),
            .bank = try graph.Bank.load(p, std.mem.span(bank)),
            .context = .{ .io = std.testing.io, .compiler = .{ .clang = std.mem.span(clang), .linker = if (std.c.getenv("TF_RUNNER_LLD")) |l| std.mem.span(l) else null } },
        };
        one = e;
        return e;
    }

    /// The trace cell for `run` (a repl cell, no prior) at frontend step `budget`; null: abstained.
    fn translate(e: *Engine, run: []const u8, budget: usize) !?api.Program {
        e.context.compiler.step_budget = budget;
        var routed: route.Routed = .{ .inner = e.context.frontend(), .bank = &e.bank };
        const f = routed.frontend();
        const tr = try f.translate(f.ctx, gpa, .{ .form = .repl, .lang = "python", .code = run });
        return if (tr == .program) tr.program else null;
    }

    fn exec(e: *Engine, p: api.Program) ![]u8 {
        const b = e.runner.backend;
        e.job += 1;
        const job = try b.start(e.job, p, e.runner.cfg.max_steps);
        const res = while (true) switch (b.poll(job)) {
            .running => std.Thread.yield() catch {},
            .done => |r| break r,
        };
        defer b.release(e.job);
        try std.testing.expectEqual(api.Status.ok, res.status);
        return gpa.dupe(u8, res.out);
    }
};

test "runner proof: trace stated: isqrt runs on the bank's plan and prints CPython's root; a mutated input follows (TF_RUNNER_GRAPH_BANK)" {
    const e = try Engine.open() orelse return error.SkipZigTest;
    for ([_][2][]const u8{ .{ live[0].want[1], "98765432109876543211" }, .{ live[1].want[1], "123456789876543210123457" }, .{ live[6].want[1], "" } }) |c| {
        const p = (try e.translate(c[0], 10522)) orelse return error.TestExpectedGraph;
        defer gpa.free(p.bytes);
        try std.testing.expect(p.kind == .graph);
        const want = try python(c[0]);
        defer gpa.free(want);
        const out = try e.exec(p);
        defer gpa.free(out);
        try std.testing.expectEqualStrings(want, out);
        if (c[1].len == 0) continue;
        // the circuit owns the root: the input changed in the packet prints CPython's root of the changed input
        const at = std.mem.indexOf(u8, c[0], "isqrt(").? + 6;
        const n = c[0][at .. c[0].len - 2];
        const from = try std.fmt.allocPrint(gpa, "\"{s}\"", .{n});
        defer gpa.free(from);
        const to = try std.fmt.allocPrint(gpa, "\"{s}\"", .{c[1]});
        defer gpa.free(to);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, p.bytes, from));
        const mutated = try std.mem.replaceOwned(u8, gpa, p.bytes, from, to);
        defer gpa.free(mutated);
        const src = try std.fmt.allocPrint(gpa, "from math import isqrt\nprint(isqrt({s}))", .{c[1]});
        defer gpa.free(src);
        const want2 = try python(src);
        defer gpa.free(want2);
        const out2 = try e.exec(.{ .kind = .graph, .bytes = mutated });
        defer gpa.free(out2);
        try std.testing.expectEqualStrings(want2, out2);
    }
}

test "runner proof: trace stated: edit distance runs as RV32, past the default budget only with the opt-in, and a mutated string follows (TF_RUNNER_GRAPH_BANK)" {
    const e = try Engine.open() orelse return error.SkipZigTest;
    // kitten/sitting costs 28,360 estimated steps: over the default 10,522, so only the opt-in admits it
    if (try e.translate(live[4].want[1], 10522)) |p| {
        gpa.free(p.bytes);
        return error.TestDefaultBudget;
    }
    for ([_]struct { want: [2][]const u8 }{ .{ .want = live[4].want }, .{ .want = live[5].want }, .{ .want = edit("abc", "yabd") } }) |c| {
        const p = (try e.translate(c.want[1], 32768)) orelse {
            std.debug.print("opt-in budget refused {s}\n", .{c.want[0]});
            return error.TestExpectedProgram;
        };
        defer gpa.free(p.bytes);
        try std.testing.expect(p.kind == .elf);
        const want = try python(c.want[1]);
        defer gpa.free(want);
        const out = try e.exec(p);
        defer gpa.free(out);
        try std.testing.expectEqualStrings(want, out);
        const line = (try trace.owned(gpa, "[exact: {}]", c.want[0], false, out)) orelse return error.TestExpectedOwned;
        defer gpa.free(line);
        const golden = try std.fmt.allocPrint(gpa, "[exact: {s} = {s}]\n", .{ c.want[0], std.mem.trimEnd(u8, want, "\n") });
        defer gpa.free(golden);
        try std.testing.expectEqualStrings(golden, line);
    }
    // the circuit computes the distance: "kitten" read as "sitten" in the ELF prints CPython's 2, not the source's 3
    const p = (try e.translate(live[4].want[1], 32768)) orelse return error.TestExpectedProgram;
    defer gpa.free(p.bytes);
    if (std.mem.count(u8, p.bytes, "kitten") != 1) {
        std.debug.print("\"kitten\" {d} times in the ELF\n", .{std.mem.count(u8, p.bytes, "kitten")});
        return error.TestExpectedInput;
    }
    const mutated = try std.mem.replaceOwned(u8, gpa, p.bytes, "kitten", "sitten");
    defer gpa.free(mutated);
    const out = try e.exec(.{ .kind = .elf, .bytes = mutated });
    defer gpa.free(out);
    try std.testing.expectEqualStrings("2\n", out);
}
