//! S3 lane calls (whole tool scripts) through the served runner, run as S3 runs them: route, steps, ms; no GPU.
const std = @import("std");
const api = @import("runner_api.zig");
const runner = @import("runner.zig");
const ropen = @import("runner_open.zig");
const prog = @import("runner_graph_program.zig");
const lane = @import("compute_lane.zig");
const rc = @import("runner_circuit.zig");
const elf_cache = @import("runner_frontend_elf_cache.zig");
const linux = std.os.linux;

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn env(name: []const u8) ?[]const u8 {
    var buf: [96]u8 = undefined;
    if (name.len >= buf.len) return null;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const v = std.mem.span(std.c.getenv(buf[0..name.len :0]) orelse return null);
    return if (v.len == 0) null else v;
}

fn pinTo(cpu: u32) void {
    var set: [16]u64 = @splat(0);
    set[cpu / 64] = @as(u64, 1) << @intCast(cpu % 64);
    _ = linux.syscall3(.sched_setaffinity, 0, @sizeOf(@TypeOf(set)), @intFromPtr(&set));
}

const Cell = struct { id: []const u8, code: []const u8, want: []const u8 };

/// The route a program takes on the served backend: one bank packet, a sequence of them, calc or ELF on the circuit.
fn route(p: api.Program) []const u8 {
    return switch (p.kind) {
        .graph => if (std.mem.startsWith(u8, p.bytes, prog.seq_magic)) "graph-seq" else "graph",
        .calc => "calc",
        .elf => "elf",
    };
}

/// Pieces in a graph sequence (1 for anything else).
fn pieces(p: api.Program) usize {
    if (p.kind != .graph or !std.mem.startsWith(u8, p.bytes, prog.seq_magic)) return 1;
    return std.mem.count(u8, p.bytes[prog.seq_magic.len..], "\n");
}

/// One cell's staged run: translate, then start and spin-poll to the end (the backend's own time, no S3 naps).
const Staged = struct { route: []const u8 = "abstain", pieces: usize = 0, bytes: usize = 0, translate: u64 = 0, run: u64 = 0, steps: u64 = 0, exec: u64 = 0, boot: u64 = 0, eval: u64 = 0, ok: bool = false };

/// A circuit program's boot (the snapshot and the program loaded) and its steps, timed apart on this thread.
fn split(c: *const rc.Circuit, p: api.Program, budget: u64, s: *Staged) !void {
    const gpa = std.heap.c_allocator;
    const t0 = now();
    var m = try rc.Machine.boot(gpa, c, p);
    defer m.deinit(gpa);
    const t1 = now();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    const never: std.atomic.Value(bool) = .init(false);
    _ = try m.run(budget, &out, null, &never);
    s.boot = t1 - t0;
    s.eval = now() - t1;
}

fn staged(a: std.mem.Allocator, r: *runner.Runner, c: Cell, stream: u64, circuit: ?*const rc.Circuit) !Staged {
    var s: Staged = .{};
    const t0 = now();
    const t = try r.frontend.translate(r.frontend.ctx, a, .{ .form = .fence, .lang = "python", .code = c.code });
    const t1 = now();
    s.translate = t1 - t0;
    const p = switch (t) {
        .program => |p| p,
        .abstain => return s,
    };
    s.route = route(p);
    s.pieces = pieces(p);
    s.bytes = p.bytes.len;
    defer r.backend.release(stream);
    const job = try r.backend.start(stream, p, r.cfg.max_steps);
    const res = while (true) switch (r.backend.poll(job)) {
        .running => std.atomic.spinLoopHint(),
        .done => |x| break x,
    };
    s.run = now() - t1;
    s.steps = res.steps;
    s.exec = res.exec_ns;
    s.ok = res.status == .ok and std.mem.eql(u8, res.out, c.want);
    if (circuit) |k| if (p.kind != .graph) try split(k, p, r.cfg.max_steps, &s);
    return s;
}

/// One S3 call as compute_circuit.Exec.run makes it (Doug-lane S3): translate, start, poll with 1 ms naps; its body.
fn call(a: std.mem.Allocator, r: *runner.Runner, code: []const u8, stream: u64) !?[]u8 {
    const t = try r.frontend.translate(r.frontend.ctx, a, .{ .form = .fence, .lang = "python", .code = code });
    const p = switch (t) {
        .program => |p| p,
        .abstain => return null,
    };
    defer r.backend.release(stream);
    const job = while (true) {
        if (r.backend.start(stream, p, r.cfg.max_steps)) |j| break j else |e| switch (e) {
            error.OutOfMemory => return e,
            error.Busy => nap(),
        }
    };
    while (true) {
        switch (r.backend.poll(job)) {
            .running => {},
            .done => |x| return if (x.status == .ok) try lane.runner.body(a, .{ .status = .ok, .stdout = x.out }, 0, 0) else null,
        }
        nap();
    }
}

fn nap() void {
    const ts: std.c.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
    _ = std.c.nanosleep(&ts, null);
}

const Row = struct { st: Staged, e2e: [2]u64, served_ok: bool };

fn median(xs: []u64) u64 {
    std.mem.sort(u64, xs, {}, std.sort.asc(u64));
    return xs[xs.len / 2];
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// runner-s3-bench <cells.jsonl> [reps] [engine-cpu]; a line is {"id", "code", "want"} (want: CPython's stdout).
pub fn main(init: std.process.Init) !u8 {
    const gpa = std.heap.c_allocator;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 2) {
        std.debug.print("usage: runner-s3-bench <cells.jsonl> [reps] [engine-cpu] (TF_DSV41_COMPUTE_LANE_RUNNER* knobs)\n", .{});
        return 2;
    }
    const reps = if (argv.len > 2) try std.fmt.parseInt(u32, argv[2], 10) else 5;
    if (argv.len > 3) pinTo(try std.fmt.parseInt(u32, argv[3], 10));
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, argv[1], gpa, .limited(1 << 26));
    var cells: std.ArrayList(Cell) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| if (line.len > 0) {
        const p = try std.json.parseFromSlice(Cell, gpa, line, .{ .ignore_unknown_fields = true });
        try cells.append(gpa, p.value);
    };
    const cfg = (try runner.Config.parse(&env, "TF_DSV41_COMPUTE_LANE_RUNNER")) orelse {
        std.debug.print("TF_DSV41_COMPUTE_LANE_RUNNER is off\n", .{});
        return 2;
    };
    const r = try ropen.open(gpa, cfg);
    // a second copy of the circuit: ELF and calc programs' boot and steps timed apart from the backend
    var circuit = try rc.Circuit.load(gpa, cfg.circuit);
    defer circuit.deinit(gpa);
    const rows = try gpa.alloc(Row, cells.items.len);
    const e2es = try gpa.alloc(u64, reps);
    const tr = try gpa.alloc(u64, reps);
    const run = try gpa.alloc(u64, reps);
    var stream: u64 = 1 << 40;
    // TF_S3_BENCH_COLD=1 empties the compiled-image cache before every call: each ELF cell pays its compile
    const cold = env("TF_S3_BENCH_COLD") != null;
    std.debug.print("| cell | route | pieces | bytes | steps | translate ms | backend ms | exec ms | boot ms | steps ms | S3 first ms | S3 p50 ms | body |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n", .{});
    for (cells.items, rows) |c, *row| {
        row.* = .{ .st = .{}, .e2e = .{ 0, 0 }, .served_ok = true };
        for (0..reps) |i| {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            // the served call first (cold on rep 0: no translation or run of this cell before it)
            stream += 1;
            if (cold) elf_cache.reset(init.io);
            const t0 = now();
            const got = try call(arena.allocator(), r, c.code, stream);
            e2es[i] = now() - t0;
            const want = try lane.runner.body(arena.allocator(), .{ .status = .ok, .stdout = c.want }, 0, 0);
            if (got == null or !std.mem.eql(u8, got.?, want)) row.served_ok = false;
            stream += 1;
            if (cold) elf_cache.reset(init.io);
            const st = try staged(arena.allocator(), r, c, stream, &circuit);
            row.st = st;
            row.st.route = st.route;
            tr[i] = st.translate;
            run[i] = st.run;
            if (!st.ok) row.served_ok = false;
        }
        row.e2e = .{ e2es[0], median(e2es) };
        row.st.translate = median(tr);
        row.st.run = median(run);
        std.debug.print("| {s} | {s} | {d} | {d} | {d} | {d:.2} | {d:.2} | {d:.2} | {d:.2} | {d:.2} | {d:.2} | {d:.2} | {s} |\n", .{ c.id, row.st.route, row.st.pieces, row.st.bytes, row.st.steps, ms(row.st.translate), ms(row.st.run), ms(row.st.exec), ms(row.st.boot), ms(row.st.eval), ms(row.e2e[0]), ms(row.e2e[1]), if (row.served_ok) "= CPython" else "DIFFERS" });
    }
    var all: std.ArrayList(u64) = .empty;
    var first: std.ArrayList(u64) = .empty;
    var under: u32 = 0;
    var same: u32 = 0;
    for (rows) |row| {
        try all.append(gpa, row.e2e[1]);
        try first.append(gpa, row.e2e[0]);
        under += @intFromBool(row.e2e[1] < 40 * std.time.ns_per_ms);
        same += @intFromBool(row.served_ok);
    }
    const v = all.items;
    std.mem.sort(u64, v, {}, std.sort.asc(u64));
    std.debug.print("\nS3 call p50 over cells: median {d:.2} ms, p90 {d:.2} ms, max {d:.2} ms; {d}/{d} under 40 ms; {d}/{d} bodies = CPython\n", .{ ms(v[v.len / 2]), ms(v[(v.len * 9) / 10]), ms(v[v.len - 1]), under, v.len, same, v.len });
    const w = first.items;
    std.mem.sort(u64, w, {}, std.sort.asc(u64));
    std.debug.print("S3 first call over cells: median {d:.2} ms, max {d:.2} ms; image cache {s}, {d} hits\n", .{ ms(w[w.len / 2]), ms(w[w.len - 1]), if (cold) "emptied per call" else "kept", elf_cache.hitCount(init.io) });
    return if (same == v.len) 0 else 1;
}
