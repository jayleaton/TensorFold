//! The served in-model runner: knobs to a circuit file, its CPU backend and the frontend, for a family's engine.
const std = @import("std");
const runner = @import("runner.zig");
const circuit = @import("runner_circuit.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.runner);

pub const Config = runner.Config;
pub const Runner = runner.Runner;
pub const Text = runner.Text;
pub const TraceAsk = runner.TraceAsk;
pub const validTraceFormat = @import("runner_trace.zig").validFormat;
pub const maxTraceRepeat = @import("runner_trace.zig").max_repeat;
pub const place = runner.place;

const Opened = struct {
    runner: Runner,
    circuit: circuit.Circuit,
    placer: runner.place.Placer,
    bank: graph.Bank = undefined,
    routed: route.Routed = undefined,
    dispatch: route.Dispatch = undefined,
};

/// The runner for `cfg` (from Config.parse); lives as long as the engine. CUDA waits for its backend (DESIGN T4).
pub fn open(gpa: Allocator, cfg: Config) !*Runner {
    if (cfg.device != .cpu) {
        log.err("TF_DSV41_RUNNER_DEVICE=cuda: the CUDA circuit backend is not in this build", .{});
        return error.BadKnob;
    }
    const o = try gpa.create(Opened);
    errdefer gpa.destroy(o);
    o.* = .{ .runner = undefined, .circuit = undefined, .placer = .{ .cfg = cfg.place } };
    o.circuit = circuit.Circuit.load(gpa, cfg.circuit) catch |e| {
        log.err("TF_DSV41_RUNNER_CIRCUIT={s}: {t}", .{ cfg.circuit, e });
        return e;
    };
    errdefer o.circuit.deinit(gpa);
    try probe(gpa, &o.circuit);
    const back = try @import("runner_circuit_backend.zig").placed(gpa, &o.circuit, cfg.workers, &o.placer);
    var frontend = @import("runner_frontend.zig").frontend();
    var backend = back;
    // backend graph: T1's bank loads once here; pure-integer calc programs run on it, the rest on the circuit
    if (cfg.backend == .graph) {
        const t0 = nowNs();
        o.bank = graph.Bank.load(gpa, cfg.graph_bank) catch |e| {
            log.err("TF_DSV41_RUNNER_GRAPH_BANK={s}: {t}", .{ cfg.graph_bank, e });
            return e;
        };
        o.routed = .{ .inner = frontend, .bank = &o.bank };
        // settling holds: graph jobs leave the engine thread's CPU as the circuit's do (unplaced, they share its pin)
        const placer: ?*runner.place.Placer = if (cfg.settle_us > 0) &o.placer else null;
        o.dispatch = .{ .gpa = gpa, .circuit = back, .graph = try graph.placed(gpa, &o.bank, cfg.workers, placer) };
        frontend = o.routed.frontend();
        backend = o.dispatch.backend();
        log.info("in-model runner: graph bank {s} loaded in {d} ms", .{ cfg.graph_bank, (nowNs() - t0) / std.time.ns_per_ms });
    }
    o.runner = .{ .gpa = gpa, .cfg = cfg, .frontend = frontend, .backend = backend, .placer = &o.placer };
    log.info("in-model runner: circuit {s}, cpu, {d} workers, {d} steps a run, backend {t}, settle {d} us", .{ cfg.circuit, cfg.workers, cfg.max_steps, cfg.backend, cfg.settle_us });
    return &o.runner;
}

/// The circuit route at boot (served compiler, this circuit, unplaced): a wrong loop value refuses the runner.
fn probe(gpa: Allocator, c: *const circuit.Circuit) !void {
    const t0 = nowNs();
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const compiler = try @import("runner_frontend_boot.zig").selfTest(gpa, threaded.io(), .{});
    const back = try @import("runner_circuit_backend.zig").backend(gpa, c, 1);
    defer @import("runner_circuit_backend.zig").destroy(back);
    var why: [256]u8 = undefined;
    const got = try @import("runner_probe.zig").run(gpa, @import("runner_frontend.zig").frontend(), back, compiler, &why);
    if (got.wrong) |w| {
        log.err("in-model runner probe: {s}; the runner refuses this circuit and compiler", .{w});
        return error.RunnerProbe;
    }
    if (!compiler) return log.info("in-model runner probe: no RV32 compiler, the circuit route abstains", .{});
    log.info("in-model runner probe: {d} probe cells print the right output on the circuit in {d} steps ({d} ms)", .{ got.ran, got.steps, (nowNs() - t0) / std.time.ns_per_ms });
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
