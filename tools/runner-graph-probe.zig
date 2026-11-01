//! Native graph evidence: evaluator tokens, exact stdout, prepared-bank boot and backend lifecycle.
const std = @import("std");
const graph = @import("runner_graph.zig");
const api = @import("runner_api.zig");
const Allocator = std.mem.Allocator;
const Hash = std.crypto.hash.sha2.Sha256;
const byte_base: u32 = 3 * 32768 + 4 + 2 * 65536;
const halt: u32 = byte_base + 512 + 8 + 16 + 16 + 128 + 256 + 5;

const Case = struct {
    id: []const u8,
    packet: std.json.Value,
    expected_output: []const u8,
    expected_status: []const u8 = "halted",
    expect_mismatch: bool = false,
};

const Row = struct {
    id: []const u8,
    pass: bool = false,
    status: []const u8 = "not_started",
    reason: []const u8 = "",
    steps: u64 = 0,
    raw_tokens: usize = 0,
    alu_halts: usize = 0,
    trace_valid: bool = false,
    output_equal: bool = false,
    detected_mismatch: bool = false,
    machine_boot_ns: u64 = 0,
    run_ns: u64 = 0,
    cpu_ns: u64 = 0,
    actual_output: []const u8 = "",
    actual_canonical_sha256: []const u8 = "",
    expected_canonical_sha256: []const u8 = "",
    raw_trace_sha256: []const u8 = "",
};

const Seed = struct { packet: []const u8, output: []const u8, op: []const u8 };
const Lifecycle = struct { pass: bool = false, checks: usize = 0, error_name: []const u8 = "not_requested" };
const Options = struct {
    manifest: []const u8,
    cases: []const u8,
    start: usize = 0,
    count: usize = std.math.maxInt(usize),
    budget: u64 = 100000,
    lifecycle: bool = false,
};
const Report = struct {
    pass: bool,
    boot_wall_ns: u64,
    boot_process_cpu_ns: u64,
    run_wall_ns: u64,
    run_process_cpu_ns: u64,
    case_count: usize,
    failed_cases: usize,
    named_requests: usize,
    generic_requests: usize,
    steps: u64,
    request_compiled_graphs: u64 = 0,
    actual_canonical_sha256: []const u8,
    raw_audit_trace_sha256: []const u8,
    trace_encoding: []const u8 = "LEu32; corpus order; generated tokens only; GO excluded",
    projection: []const u8 = "actual BYTE tokens and actual final HALT; internal ALU HALTs excluded",
    compilation_evidence: []const u8 = "Bank.load prepared packs once; no Circuit.load or compiler calls",
    lifecycle: Lifecycle,
    cases: []const Row,
};

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

fn tokenHash(h: *Hash, token: u32) void {
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, token, .little);
    h.update(&bytes);
}

fn hex(a: Allocator, h: *Hash) ![]const u8 {
    const bytes = std.fmt.bytesToHex(h.finalResult(), .lower);
    return a.dupe(u8, &bytes);
}

fn expectedHash(a: Allocator, bytes: []const u8, halted: bool) ![]const u8 {
    var h = Hash.init(.{});
    for (bytes) |b| tokenHash(&h, byte_base + b);
    if (halted) tokenHash(&h, halt);
    return hex(a, &h);
}

fn options(args: []const []const u8) !Options {
    if (args.len < 3) return error.ExpectedManifestAndCases;
    var o: Options = .{ .manifest = args[1], .cases = args[2] };
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--lifecycle")) {
            o.lifecycle = true;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingOptionValue;
        const key = args[i];
        i += 1;
        if (std.mem.eql(u8, key, "--start")) {
            o.start = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, key, "--count") or std.mem.eql(u8, key, "--maxcases")) {
            o.count = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, key, "--budget")) {
            o.budget = try std.fmt.parseInt(u64, args[i], 10);
        } else return error.UnknownOption;
    }
    return o;
}

fn canonicalBank(bank: *const graph.Bank) !void {
    for (bank.templates) |t| {
        if (t.circuit.halt != halt or t.circuit.byte_family != 6) return error.NoncanonicalBank;
        const family, const payload = t.circuit.split(byte_base + 255) orelse return error.NoncanonicalBank;
        if (family != 6 or payload != 255) return error.NoncanonicalBank;
    }
}

fn audit(a: Allocator, row: *Row, trace: []const u32, out: []const u8, halted: bool, all_raw: *Hash, all_visible: *Hash) !void {
    var raw = Hash.init(.{});
    var visible = Hash.init(.{});
    var valid = trace.len == row.steps;
    var cursor: usize = 0;
    for (trace, 0..) |token, i| {
        tokenHash(&raw, token);
        tokenHash(all_raw, token);
        const is_byte = token >= byte_base and token < byte_base + 256;
        if (is_byte) {
            if (i < row.alu_halts or cursor >= out.len) {
                valid = false;
            } else if (token != byte_base + out[cursor]) valid = false;
            cursor += 1;
            tokenHash(&visible, token);
            tokenHash(all_visible, token);
        } else {
            if (token != halt) valid = false;
            if (halted and i + 1 == trace.len) {
                tokenHash(&visible, token);
                tokenHash(all_visible, token);
            } else if (i >= row.alu_halts) valid = false;
        }
    }
    if (cursor != out.len) valid = false;
    if (halted) {
        if (trace.len != row.alu_halts + out.len + 1) valid = false;
        if (trace.len == 0 or trace[trace.len - 1] != halt) valid = false;
    }
    row.trace_valid = valid;
    row.raw_tokens = trace.len;
    row.actual_canonical_sha256 = try hex(a, &visible);
    row.raw_trace_sha256 = try hex(a, &raw);
}

fn runCase(a: Allocator, keep: Allocator, bank: *const graph.Bank, c: Case, budget: u64, all_raw: *Hash, all_visible: *Hash, seeds: *std.ArrayList(Seed)) !Row {
    var row: Row = .{ .id = try keep.dupe(u8, c.id) };
    row.expected_canonical_sha256 = try expectedHash(keep, c.expected_output, std.mem.eql(u8, c.expected_status, "halted"));
    const bytes = try std.json.Stringify.valueAlloc(a, c.packet, .{});
    defer a.free(bytes);
    const boot_start = now();
    var m = graph.Machine.boot(a, bank, bytes) catch |err| {
        row.machine_boot_ns = now() - boot_start;
        row.status = try std.fmt.allocPrint(keep, "error:{s}", .{@errorName(err)});
        row.pass = !c.expect_mismatch and std.mem.eql(u8, row.status, c.expected_status) and
            c.expected_output.len == 0;
        if (!row.pass) row.reason = "machine boot status differs";
        return row;
    };
    defer m.deinit();
    row.machine_boot_ns = now() - boot_start;
    if (m.ops.len > 4096) {
        row.reason = "case exceeds 4096 operation limit";
        return row;
    }
    for (m.ops) |op| if (!std.mem.eql(u8, op.op, "bind")) {
        row.alu_halts += 1;
    };
    var trace: std.ArrayList(u32) = .empty;
    defer trace.deinit(a);
    m.trace = &trace;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    const start = now();
    const cpu_start = cpuNow();
    const end: ?api.RunEnd = m.run(budget, &out, null) catch |err| blk: {
        row.status = try std.fmt.allocPrint(keep, "error:{s}", .{@errorName(err)});
        break :blk null;
    };
    row.cpu_ns = cpuNow() - cpu_start;
    row.run_ns = now() - start;
    row.steps = m.steps;
    if (end) |e| row.status = @tagName(e.end);
    const halted = std.mem.eql(u8, row.status, "halted");
    try audit(keep, &row, trace.items, out.items, halted, all_raw, all_visible);
    row.actual_output = try keep.dupe(u8, out.items);
    row.output_equal = std.mem.eql(u8, out.items, c.expected_output);
    row.detected_mismatch = halted and row.trace_valid and !row.output_equal;
    row.pass = if (c.expect_mismatch) row.detected_mismatch else row.trace_valid and row.output_equal and std.mem.eql(u8, row.status, c.expected_status);
    if (!row.pass) row.reason = "status, output or generated token trace differs";
    if (row.pass and halted and !c.expect_mismatch and seeds.items.len < 2) {
        const op = if (m.ops.len > 0) m.ops[0].op else "identity";
        if (seeds.items.len == 0 or !std.mem.eql(u8, op, seeds.items[0].op)) {
            try seeds.append(keep, .{ .packet = try keep.dupe(u8, bytes), .output = row.actual_output, .op = try keep.dupe(u8, op) });
        }
    }
    return row;
}

fn wait(b: api.Backend, id: api.JobId) !api.Result {
    const until = now() + 300 * std.time.ns_per_s;
    while (true) {
        switch (b.poll(id)) {
            .done => |r| return r,
            .running => {},
        }
        if (now() >= until) return error.BackendTimeout;
        const delay: std.c.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
        _ = std.c.nanosleep(&delay, null);
    }
}

fn require(ok: bool) !void {
    if (!ok) return error.BackendContract;
}

fn result(b: api.Backend, job: api.JobId, want: []const u8) !void {
    const r = try wait(b, job);
    try require(r.status == .ok and std.mem.eql(u8, r.out, want));
    try require(r.steps > 0);
}

fn retryStart(b: api.Backend, stream: u64, program: api.Program, budget: u64) !api.JobId {
    const until = now() + 300 * std.time.ns_per_s;
    while (true) {
        return b.start(stream, program, budget) catch |err| {
            if (err != error.Busy) return err;
            if (now() >= until) return error.BackendTimeout;
            const delay: std.c.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
            _ = std.c.nanosleep(&delay, null);
            continue;
        };
    }
}

fn lifecycle(a: Allocator, bank: *const graph.Bank, seeds: []const Seed, budget: u64, checks: *usize) !void {
    const Kind = @FieldType(api.Program, "kind");
    if (comptime !@hasField(Kind, "graph")) return error.GraphKindUnavailable;
    const kind = std.meta.stringToEnum(Kind, "graph") orelse return error.GraphKindUnavailable;
    if (seeds.len != 2) return error.NeedTwoDistinctOperations;
    const p: api.Program = .{ .kind = kind, .bytes = seeds[0].packet };
    const q: api.Program = .{ .kind = kind, .bytes = seeds[1].packet };
    const b = try graph.backend(a, bank, 2);
    defer graph.destroy(b);
    const one = try b.start(1, p, budget);
    const two = try b.start(2, q, budget);
    if (b.start(3, p, budget)) |_| return error.MissingBusy else |err| try require(err == error.Busy);
    checks.* += 1;
    try result(b, two, seeds[1].output);
    try result(b, one, seeds[0].output);
    checks.* += 1;
    switch (b.poll(one)) {
        .running => return error.DeliveredTwice,
        .done => |r| try require(r.status == .abstain and r.out.len == 0),
    }
    checks.* += 1;
    const zero = try b.start(1, p, 0);
    const limited = try wait(b, zero);
    try require(limited.status == .budget and limited.steps == 0 and limited.out.len == 0);
    checks.* += 1;
    try result(b, try b.start(1, p, budget), seeds[0].output);
    checks.* += 1;
    const cancelled = try b.start(4, p, budget);
    b.cancel(cancelled);
    try result(b, try retryStart(b, 4, q, budget), seeds[1].output);
    checks.* += 1;
    _ = try b.start(5, q, budget);
    b.release(5);
    try result(b, try retryStart(b, 5, p, budget), seeds[0].output);
    checks.* += 1;
    b.release(1);
    b.release(2);
    b.release(4);
    b.release(5);
    try result(b, try b.start(1, q, budget), seeds[1].output);
    checks.* += 1;
    var m = try graph.Machine.boot(a, bank, seeds[0].packet);
    defer m.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    const flag = std.atomic.Value(bool).init(true);
    const end = try m.run(budget, &out, &flag);
    try require(end.end == .cancelled and m.steps == 0 and out.items.len == 0);
    checks.* += 1;
}

fn run(init: std.process.Init) !u8 {
    const a = init.gpa;
    const keep = init.arena.allocator();
    const args = try init.minimal.args.toSlice(keep);
    const o = try options(args);
    const boot_start = now();
    const boot_cpu_start = cpuNow();
    var bank = try graph.Bank.load(a, o.manifest);
    const boot_cpu_ns = cpuNow() - boot_cpu_start;
    const boot_ns = now() - boot_start;
    defer bank.deinit();
    try canonicalBank(&bank);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, o.cases, a, .limited(128 << 20));
    defer a.free(bytes);
    var lines = std.mem.splitScalar(u8, bytes, '\n');
    var rows: std.ArrayList(Row) = .empty;
    var seeds: std.ArrayList(Seed) = .empty;
    var raw = Hash.init(.{});
    var visible = Hash.init(.{});
    var index: usize = 0;
    var failed: usize = 0;
    var named: usize = 0;
    var generic: usize = 0;
    var steps: u64 = 0;
    const start = now();
    const cpu_start = cpuNow();
    while (lines.next()) |line| {
        if (std.mem.trim(u8, line, "\r\t ").len == 0) continue;
        const current = index;
        index += 1;
        if (current < o.start) continue;
        if (rows.items.len >= o.count) break;
        const parsed = std.json.parseFromSlice(Case, a, line, .{ .ignore_unknown_fields = true }) catch |err| {
            const row: Row = .{ .id = try std.fmt.allocPrint(keep, "line-{d}", .{current + 1}), .reason = try keep.dupe(u8, @errorName(err)) };
            try rows.append(keep, row);
            failed += 1;
            continue;
        };
        defer parsed.deinit();
        const c = parsed.value;
        if (c.packet == .object and c.packet.object.get("plan") != null) named += 1 else generic += 1;
        const row = try runCase(a, keep, &bank, c, o.budget, &raw, &visible, &seeds);
        try rows.append(keep, row);
        if (!row.pass) failed += 1;
        steps += row.steps;
        if (rows.items.len % 100 == 0) std.debug.print("graph probe: {d} cases, {d} failures\n", .{
            rows.items.len, failed,
        });
    }
    const run_ns = now() - start;
    const cpu_ns = cpuNow() - cpu_start;
    var life: Lifecycle = .{};
    if (o.lifecycle) {
        life.error_name = "";
        lifecycle(a, &bank, seeds.items, o.budget, &life.checks) catch |err| {
            life.error_name = @errorName(err);
        };
        life.pass = life.error_name.len == 0 and life.checks == 9;
    }
    const report: Report = .{
        .pass = rows.items.len > 0 and failed == 0 and (!o.lifecycle or life.pass),
        .boot_wall_ns = boot_ns,
        .boot_process_cpu_ns = boot_cpu_ns,
        .run_wall_ns = run_ns,
        .run_process_cpu_ns = cpu_ns,
        .case_count = rows.items.len,
        .failed_cases = failed,
        .named_requests = named,
        .generic_requests = generic,
        .steps = steps,
        .actual_canonical_sha256 = try hex(keep, &visible),
        .raw_audit_trace_sha256 = try hex(keep, &raw),
        .lifecycle = life,
        .cases = rows.items,
    };
    const json = try std.json.Stringify.valueAlloc(keep, report, .{});
    try std.Io.File.stdout().writeStreamingAll(init.io, json);
    try std.Io.File.stdout().writeStreamingAll(init.io, "\n");
    return if (report.pass) 0 else 1;
}

pub fn main(init: std.process.Init) !u8 {
    return run(init) catch |err| {
        const json = try std.json.Stringify.valueAlloc(init.arena.allocator(), .{
            .pass = false,
            .fatal_error = @errorName(err),
        }, .{});
        try std.Io.File.stdout().writeStreamingAll(init.io, json);
        try std.Io.File.stdout().writeStreamingAll(init.io, "\n");
        return 1;
    };
}
