const std = @import("std");
const model = @import("runner_circuit_model.zig");
const State = @import("runner_circuit_state.zig").State;
const eval = @import("runner_circuit_eval.zig");
const rc = @import("runner_circuit.zig");
fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
fn cpuNow() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.PROCESS_CPUTIME_ID, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const load_start = now();
    const load_cpu_start = cpuNow();
    var c = try model.Circuit.load(a, argv[1]);
    const load_ns = now() - load_start;
    const load_cpu_ns = cpuNow() - load_cpu_start;
    defer c.deinit(a);
    const want = try std.Io.Dir.cwd().readFileAlloc(init.io, argv[2], a, .limited(1000));
    defer a.free(want);
    const repeats = if (argv.len > 4) try std.fmt.parseInt(u32, argv[4], 10) else 5;
    for (0..repeats) |rep| {
        var machine = if (argv.len > 3 and !std.mem.eql(u8, argv[3], "-"))
            try rc.Machine.boot(a, &c, .{ .kind = .calc, .bytes = argv[3] })
        else
            rc.Machine{ .state = try State.init(a, &c), .next = c.go };
        defer machine.deinit(a);
        const s = &machine.state;
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(a);
        var token = c.go;
        var digest = std.crypto.hash.sha2.Sha256.init(.{});
        var encoded: [4]u8 = undefined;
        std.mem.writeInt(u32, &encoded, token, .little);
        digest.update(&encoded);
        var steps: u64 = 0;
        const start = now();
        const cpu_start = cpuNow();
        while (token != c.halt and steps < 100000) {
            token = try eval.step(s, token);
            std.mem.writeInt(u32, &encoded, token, .little);
            digest.update(&encoded);
            steps += 1;
            const fam, const pay = c.split(token).?;
            if (fam == c.byte_family and pay < 256) try out.append(a, @intCast(pay));
        }
        const elapsed = now() - start;
        const cpu_elapsed = cpuNow() - cpu_start;
        if (token != c.halt or !std.mem.eql(u8, out.items, want)) return error.NativeOutputMismatch;
        const hash = std.fmt.bytesToHex(digest.finalResult(), .lower);
        std.debug.print("{{\"repeat\":{d},\"steps\":{d},\"run_ns\":{d},\"cpu_ns\":{d},\"load_ns\":{d},\"load_cpu_ns\":{d},\"batches\":{d},\"nonzeros\":{d},\"trace_sha256\":\"{s}\",\"pass\":true}}\n", .{ rep, steps, elapsed, cpu_elapsed, load_ns, load_cpu_ns, c.batches.len, c.nonzeros(), hash });
    }
    return 0;
}
