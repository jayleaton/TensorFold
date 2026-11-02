//! S3 on the recorded arm-d calls (G + lane, imr-measure-20261101T155322Z): the real frontend and CPU circuit backend.
const std = @import("std");
const s3 = @import("compute_circuit.zig");
const circuit = @import("runner_circuit.zig");
const backend = @import("runner_circuit_backend.zig");
const frontend = @import("runner_frontend.zig");

const gpa = std.testing.allocator;
const Recorded = struct { task: []const u8, call: u32, code: []const u8, status: []const u8, output: []const u8 };
const File = struct { source: []const u8, calls: []const Recorded };

/// Calls whose code is integer Python the frontend admits: each must run on the circuit, so the proof is never empty.
const must_own = [_][]const u8{ "add30-06", "add30-11", "add30-22", "add30-23", "add30-25", "add30-30", "add30-33", "add30-40", "mul30-14", "mul30-16", "mul_big-05", "mul_big-30", "mul_big-40", "bignum-18" };

test "compute runner proof: every recorded call the circuit runs prints the sandbox's bytes; integer calls run there" {
    const path = std.c.getenv("TF_RUNNER_CIRCUIT") orelse {
        std.debug.print("skip compute runner proof: TF_RUNNER_CIRCUIT unset\n", .{});
        return error.SkipZigTest;
    };
    var c = try circuit.Circuit.load(gpa, std.mem.span(path));
    defer c.deinit(gpa);
    const b = try backend.backend(gpa, &c, 2);
    defer backend.destroy(b);
    var ctx: frontend.Context = .{ .io = std.testing.io };
    if (std.c.getenv("TF_RUNNER_LLD")) |l| ctx.compiler.linker = std.mem.span(l);
    var e: s3.Exec = .{ .frontend = ctx.frontend(), .backend = b, .max_steps = 4_000_000, .cfg = .{ .wall_ms = 600_000 } };
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const file = try std.json.parseFromSliceLeaky(File, a, @embedFile("compute_circuit_armd.json"), .{});
    var owned: u32 = 0;
    var wrong: u32 = 0;
    for (file.calls) |r| {
        const got = try e.run(a, r.code, null);
        switch (got) {
            .owned => |w| {
                owned += 1;
                std.debug.print("{s}#{d}: in-model runner ({t}, {d} steps)\n", .{ r.task, r.call, w.kind, w.steps });
                // only a call the sandbox ran ok can match: its recorded body is then the same bytes
                if (!std.mem.eql(u8, r.status, "ok") or !std.mem.eql(u8, r.output, w.body)) {
                    std.debug.print("  DIFFERS: sandbox {s} [{s}], circuit [{s}]\n", .{ r.status, r.output, w.body });
                    wrong += 1;
                }
            },
            .declined => |why| std.debug.print("{s}#{d}: code tool ({t})\n", .{ r.task, r.call, why }),
        }
        const must = for (must_own) |t| {
            if (std.mem.eql(u8, t, r.task)) break true;
        } else false;
        if (must and got != .owned) {
            std.debug.print("  MUST OWN: {s} went to the code tool\n", .{r.task});
            wrong += 1;
        }
    }
    std.debug.print("compute runner proof: {d} of {d} recorded calls ran on the circuit\n", .{ owned, file.calls.len });
    try std.testing.expectEqual(@as(u32, 0), wrong);
}
