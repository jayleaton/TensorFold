//! S3 on recorded arm-d calls (imr-measure-20261101T155322Z): real frontend, CPU circuit, graph first given a bank.
const std = @import("std");
const s3 = @import("compute_circuit.zig");
const circuit = @import("runner_circuit.zig");
const backend = @import("runner_circuit_backend.zig");
const frontend = @import("runner_frontend.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const ropen = @import("runner_open.zig");

const gpa = std.testing.allocator;
const Recorded = struct { task: []const u8, call: u32, code: []const u8, status: []const u8, output: []const u8 };
const File = struct { source: []const u8, calls: []const Recorded };

/// Calls each route must own, so the proof is never empty: calc lowers these on any backend.
const calc_owned = [_][]const u8{ "add30-06", "add30-11", "add30-22", "add30-23", "add30-25", "add30-30", "add30-33", "add30-40", "mul30-14", "mul30-16", "mul_big-05", "mul_big-30", "mul_big-40", "bignum-18" };
/// Two prints: one graph sequence on the bank, or an ELF when compiling (the build has the whole translate deadline).
const bank_owned = [_][]const u8{"bignum-10"};
/// Floor division with divmod and Fraction, Decimal: the graph bank's, or an ELF's when compiling.
const graph_owned = [_][]const u8{ "div_big-04", "div_big-08", "div_big-20", "div_big-23", "div_big-32", "div_big-36", "mul_big-08", "mul_big-17", "mul_big-18" };
/// One-expression integer defs with asserts: the bank's since the def inlining, an ELF's otherwise.
const def_owned = [_][]const u8{ "mbpp-80", "mbpp-86" };
/// MBPP functions only an ELF runs.
const elf_owned = [_][]const u8{ "mbpp-56", "mbpp-66" };

test "compute runner proof: served S3 (no ELF): owned calls print the sandbox's bytes, never as an ELF" {
    try proof(false);
}

test "compute runner proof: S3 with ELF programs: owned calls print the sandbox's bytes" {
    try proof(true);
}

fn listed(list: []const []const u8, task: []const u8) bool {
    for (list) |t| if (std.mem.eql(u8, t, task)) return true;
    return false;
}

fn proof(elf: bool) !void {
    const path = std.c.getenv("TF_RUNNER_CIRCUIT") orelse {
        std.debug.print("skip compute runner proof: TF_RUNNER_CIRCUIT unset\n", .{});
        return error.SkipZigTest;
    };
    var c = try circuit.Circuit.load(gpa, std.mem.span(path));
    defer c.deinit(gpa);
    const b = try backend.backend(gpa, &c, 2);
    defer backend.destroy(b);
    var ctx: frontend.Context = .{ .io = std.testing.io };
    if (std.c.getenv("TF_RUNNER_CLANG")) |cxx| ctx.compiler.clang = std.mem.span(cxx);
    if (std.c.getenv("TF_RUNNER_LLD")) |l| ctx.compiler.linker = std.mem.span(l);
    if (!elf) ctx.compiler.scratch_root = ""; // runner_open's no-compile frontend (TF_DSV41_COMPUTE_LANE_RUNNER_ELF=0)
    var e: s3.Exec = .{ .frontend = ctx.frontend(), .backend = b, .max_steps = 4_000_000, .cfg = .{ .wall_ms = 600_000, .elf = elf } };
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // TF_RUNNER_GRAPH_BANK: the served graph backend's route (runner_open), T1's bank first and the circuit behind it
    var bank: graph.Bank = undefined;
    var routed: route.Routed = undefined;
    var dispatch: route.Dispatch = undefined;
    const with_bank = std.c.getenv("TF_RUNNER_GRAPH_BANK");
    if (with_bank) |g| {
        bank = try graph.Bank.load(gpa, std.mem.span(g));
        routed = .{ .inner = e.frontend, .bank = &bank };
        dispatch = .{ .gpa = gpa, .circuit = b, .graph = try graph.backend(gpa, &bank, 2), .bank = &bank };
        e.frontend = routed.frontend();
        e.backend = dispatch.backend();
    }
    defer if (with_bank != null) {
        const on_graph = dispatch.graph;
        dispatch.deinit();
        graph.destroy(on_graph);
        bank.deinit();
    };
    std.debug.print("compute runner proof: backend {s}, ELF {s}\n", .{ if (with_bank != null) "graph, circuit behind it" else "circuit", if (elf) "on" else "off" });
    const file = try std.json.parseFromSliceLeaky(File, a, @embedFile("compute_circuit_armd.json"), .{});
    var owned: u32 = 0;
    var wrong: u32 = 0;
    for (file.calls) |r| {
        const got = try e.run(a, r.code, null);
        switch (got) {
            .owned => |w| {
                owned += 1;
                std.debug.print("{s}#{d}: in-model runner ({t}, {d} steps, {d} us)\n", .{ r.task, r.call, w.kind, w.steps, w.exec_ns / 1000 });
                // only a call the sandbox ran ok can match: its recorded body is then the same bytes
                if (!std.mem.eql(u8, r.status, "ok") or !std.mem.eql(u8, r.output, w.body)) {
                    std.debug.print("  DIFFERS: sandbox {s} [{s}], circuit [{s}]\n", .{ r.status, r.output, w.body });
                    wrong += 1;
                }
                if (!elf and w.kind == .elf) {
                    std.debug.print("  ELF: built with the compile off\n", .{});
                    wrong += 1;
                }
            },
            .declined => |why| std.debug.print("{s}#{d}: code tool ({t})\n", .{ r.task, r.call, why }),
        }
        const must = listed(&calc_owned, r.task) or (listed(&bank_owned, r.task) and (with_bank != null or elf)) or
            (listed(&graph_owned, r.task) and (with_bank != null or elf)) or (listed(&elf_owned, r.task) and elf) or
            (listed(&def_owned, r.task) and (with_bank != null or elf));
        if (must and got != .owned) {
            std.debug.print("  MUST OWN: {s} went to the code tool\n", .{r.task});
            wrong += 1;
        }
    }
    std.debug.print("compute runner proof: {d} of {d} recorded calls ran on the circuit\n", .{ owned, file.calls.len });
    try std.testing.expectEqual(@as(u32, 0), wrong);
}

test "compute runner proof: S3 beside the cell runner loads the circuit (and graph bank) once and runs on it" {
    const path = std.c.getenv("TF_RUNNER_CIRCUIT") orelse return error.SkipZigTest;
    // a runner lives as long as the engine (no deinit, workers idle on its memory): this test never frees them
    const p = std.heap.page_allocator;
    var cfg: ropen.Config = .{ .circuit = std.mem.span(path), .workers = 1 };
    if (std.c.getenv("TF_RUNNER_GRAPH_BANK")) |g| {
        cfg.backend = .graph;
        cfg.graph_bank = std.mem.span(g);
    }
    const files: u32 = if (cfg.backend == .graph) 2 else 1;
    const before = ropen.loaded();
    const cells = try ropen.open(p, cfg);
    try std.testing.expectEqual(before + files, ropen.loaded());
    var lane_cfg = cfg;
    lane_cfg.compile = false; // as served: S3's runner never compiles
    const lane = try ropen.openSharing(p, lane_cfg, cells);
    try std.testing.expectEqual(before + files, ropen.loaded());
    try std.testing.expect(ropen.compiles(cells) and !ropen.compiles(lane));
    var e = s3.Exec.init(.{ .runner = lane, .cfg = .{ .wall_ms = 600_000 } });
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const got = try e.run(arena.allocator(), "a = 6\nb = 7\nprint(a * b)", null);
    try std.testing.expect(got == .owned);
    try std.testing.expectEqualStrings("42", got.owned.body);
    // another path to the same file is another file to this rule: it loads (strings compare, never inodes)
    var other = cfg;
    other.circuit = try std.fmt.allocPrint(p, "/{s}", .{cfg.circuit});
    _ = try ropen.openSharing(p, other, cells);
    try std.testing.expectEqual(before + files + 1, ropen.loaded());
}
