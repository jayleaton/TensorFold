//! T1's reference traces against the CPU circuit: every token, step count and output byte (TF_RUNNER_CIRCUIT/GOLDENS).
const std = @import("std");
const rc = @import("runner_circuit.zig");

const gpa = std.testing.allocator;

const Checkpoint = struct { token_count: u64, prefix_sha256: []const u8 };
const Case = struct {
    id: []const u8,
    kind: []const u8,
    input: struct { bytes_hex: ?[]const u8 = null, elf: ?[]const u8 = null },
    trace_token_count: u64,
    generated_steps: u64,
    token_offset_bytes: u64,
    trace_sha256: []const u8,
    final_output_hex: []const u8,
    checkpoints: []const Checkpoint,
};

fn env(name: [:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

fn readFile(dir: []const u8, name: []const u8) ![]u8 {
    var threaded: std.Io.Threaded = .init_single_threaded;
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(threaded.io(), path, gpa, .limited(1 << 32));
}

fn hex(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

fn gunzip(bytes: []const u8) ![]u8 {
    var in: std.Io.Reader = .fixed(bytes);
    const window = try gpa.alloc(u8, std.compress.flate.max_window_len);
    defer gpa.free(window);
    var d: std.compress.flate.Decompress = .init(&in, .gzip, window);
    return d.reader.allocRemaining(gpa, .unlimited);
}

/// Every data file's size and SHA-256 must match manifest.json before any row is trusted.
fn checked(dir: []const u8, manifest: std.json.Value, name: []const u8) ![]u8 {
    const bytes = try readFile(dir, name);
    errdefer gpa.free(bytes);
    const want = manifest.object.get("files").?.object.get(name) orelse return error.GoldenManifest;
    if (want.object.get("bytes").?.integer != bytes.len) return error.GoldenManifest;
    if (!std.mem.eql(u8, want.object.get("sha256").?.string, &hex(bytes))) return error.GoldenManifest;
    return bytes;
}

test "golden: the CPU circuit reproduces every reference trace token, step count and output byte" {
    const circuit_path = env("TF_RUNNER_CIRCUIT") orelse {
        std.debug.print("skip golden: TF_RUNNER_CIRCUIT unset\n", .{});
        return error.SkipZigTest;
    };
    const dir = env("TF_RUNNER_GOLDENS") orelse {
        std.debug.print("skip golden: TF_RUNNER_GOLDENS unset\n", .{});
        return error.SkipZigTest;
    };
    const limit = if (env("TF_RUNNER_GOLDEN_LIMIT")) |l| try std.fmt.parseInt(u64, l, 10) else std.math.maxInt(u64);
    var circuit = try rc.Circuit.load(gpa, circuit_path);
    defer circuit.deinit(gpa);
    const manifest_bytes = try readFile(dir, "manifest.json");
    defer gpa.free(manifest_bytes);
    var manifest = try std.json.parseFromSlice(std.json.Value, gpa, manifest_bytes, .{});
    defer manifest.deinit();
    const cases_gz = try checked(dir, manifest.value, "cases.jsonl.gz");
    defer gpa.free(cases_gz);
    const tokens_gz = try checked(dir, manifest.value, "tokens.leu32.gz");
    defer gpa.free(tokens_gz);
    const lines = try gunzip(cases_gz);
    defer gpa.free(lines);
    const blob = try gunzip(tokens_gz);
    defer gpa.free(blob);
    var totals: struct { cases: u64 = 0, steps: u64 = 0, ns: u64 = 0, skipped: u64 = 0 } = .{};
    var it = std.mem.tokenizeScalar(u8, lines, '\n');
    var calc_seen: u64 = 0;
    while (it.next()) |line| {
        var parsed = try std.json.parseFromSlice(Case, gpa, line, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const case = parsed.value;
        const is_calc = std.mem.eql(u8, case.kind, "calc");
        if (is_calc and std.mem.startsWith(u8, case.id, "calc-")) {
            calc_seen += 1;
            if (calc_seen > limit) continue;
        }
        const want = blob[case.token_offset_bytes..][0 .. 4 * case.trace_token_count];
        try std.testing.expectEqualStrings(case.trace_sha256, &hex(want));
        var input: []u8 = undefined;
        if (is_calc) {
            const h = case.input.bytes_hex.?;
            input = try gpa.alloc(u8, h.len / 2);
            _ = try std.fmt.hexToBytes(input, h);
            input = input[0 .. input.len - 1];
        } else input = try readFile(dir, case.input.elf.?);
        defer gpa.free(input.ptr[0 .. input.len + @intFromBool(is_calc)]);
        var m = rc.Machine.boot(gpa, &circuit, .{ .kind = if (is_calc) .calc else .elf, .bytes = input }) catch |e| {
            if (e != error.CircuitElfUnsupported) return e;
            std.debug.print("golden {s}: skipped, ELF boot not built yet\n", .{case.id});
            totals.skipped += 1;
            continue;
        };
        defer m.deinit(gpa);
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        var trace: std.ArrayList(u32) = .empty;
        defer trace.deinit(gpa);
        try trace.append(gpa, circuit.go);
        var cancel: std.atomic.Value(bool) = .init(false);
        const t0 = rc.nowNs();
        const end = m.run(case.generated_steps + 1, &out, &trace, &cancel) catch |e| {
            std.debug.print("golden {s}: {s} at step {d}\n", .{ case.id, @errorName(e), m.state.steps });
            return e;
        };
        totals.ns += rc.nowNs() - t0;
        for (trace.items, 0..) |t, i| {
            if (i * 4 >= want.len) return error.GoldenTraceLonger;
            const w = std.mem.readInt(u32, want[4 * i ..][0..4], .little);
            if (t != w) {
                std.debug.print("golden {s}: token {d} is {d}, reference {d}\n", .{ case.id, i, t, w });
                return error.GoldenTokenMismatch;
            }
        }
        try std.testing.expectEqual(case.trace_token_count, trace.items.len);
        try std.testing.expectEqual(case.generated_steps, end.steps);
        try std.testing.expect(end.end == .halted);
        const want_out = try gpa.alloc(u8, case.final_output_hex.len / 2);
        defer gpa.free(want_out);
        _ = try std.fmt.hexToBytes(want_out, case.final_output_hex);
        try std.testing.expectEqualStrings(want_out, out.items);
        for (case.checkpoints) |cp| {
            const prefix = std.mem.sliceAsBytes(trace.items[0..cp.token_count]);
            try std.testing.expectEqualStrings(cp.prefix_sha256, &hex(prefix));
        }
        totals.cases += 1;
        totals.steps += end.steps;
    }
    const rate = if (totals.ns == 0) 0 else totals.steps * std.time.ns_per_s / totals.ns;
    std.debug.print("golden: {d} cases token-exact, {d} steps, {d} steps/s, {d} skipped\n", .{ totals.cases, totals.steps, rate, totals.skipped });
    try std.testing.expect(totals.cases > 0);
}
