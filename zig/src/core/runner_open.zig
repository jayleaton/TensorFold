//! The served in-model runner: knobs to a circuit file, its CPU backend and the frontend, for a family's engine.
const std = @import("std");
const runner = @import("runner.zig");
const circuit = @import("runner_circuit.zig");
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.runner);

pub const Config = runner.Config;
pub const Runner = runner.Runner;
pub const Text = runner.Text;

const Opened = struct { runner: Runner, circuit: circuit.Circuit };

/// The runner for `cfg` (from Config.parse); lives as long as the engine. CUDA waits for its backend (DESIGN T4).
pub fn open(gpa: Allocator, cfg: Config) !*Runner {
    if (cfg.device != .cpu) {
        log.err("TF_DSV41_RUNNER_DEVICE=cuda: the CUDA circuit backend is not in this build", .{});
        return error.BadKnob;
    }
    const o = try gpa.create(Opened);
    errdefer gpa.destroy(o);
    o.circuit = circuit.Circuit.load(gpa, cfg.circuit) catch |e| {
        log.err("TF_DSV41_RUNNER_CIRCUIT={s}: {t}", .{ cfg.circuit, e });
        return e;
    };
    errdefer o.circuit.deinit(gpa);
    const back = try @import("runner_circuit_backend.zig").backend(gpa, &o.circuit, cfg.workers);
    o.runner = .{ .gpa = gpa, .cfg = cfg, .frontend = @import("runner_frontend.zig").frontend(), .backend = back };
    log.info("in-model runner: circuit {s}, cpu, {d} workers, {d} steps a run", .{ cfg.circuit, cfg.workers, cfg.max_steps });
    return &o.runner;
}
