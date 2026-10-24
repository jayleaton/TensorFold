//! tf-cuda-test mm3-pack: 8-bit k/v (6144 x 512, TP2 rank cuts of 256) through exl3.Dense, then layer 18's experts.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3fmt = @import("core").exl3_format;
const dense_t = @import("exl3_dense_tests.zig");
const exl3 = cuda.exl3;
const Run = @import("exl3_tests.zig").Run;
const Rig = quant_ref_tests.RigT;
const bf = quant_ref_tests.helpers.bfOf;

const K = 6144; // hidden
const N = 512; // 4 kv heads x 128
const RANK_N = 256; // 2 kv heads a rank at TP2
const HALVES = 16; // 8 bits
const PROMPT = 300; // a prompt call: 128-row blocks and a 44-row tail
const decode_rows = [_]usize{ 1, 2, 16, 128 };

/// bf16 outputs within exl3_tests' tolerance (|want| / 128 + 2^-11 of |t| sums, above gamma(6144) = 3.7e-4 of them).
fn compare(r: *Rig, id: []const u8, ptr: u64, m: usize, n: usize, want: []const f64, tol: []const f64, ld: usize, col0: usize) !void {
    const raw = try r.back(ptr, m * n * 2);
    var most: f64 = 0;
    var at_r: usize = 0;
    var at_c: usize = 0;
    var max_abs: f64 = 0;
    var max_rel: f64 = 0;
    var top: f64 = 0;
    for (0..m) |row| for (0..n) |col| {
        top = @max(top, @abs(want[row * ld + col0 + col]));
    };
    for (0..m) |row| for (0..n) |col| {
        const w = want[row * ld + col0 + col];
        const got = bf(std.mem.bytesToValue(u16, raw[2 * (row * n + col) ..][0..2]));
        const e = @abs(got - w);
        const q = e / tol[row * ld + col0 + col];
        if (!(q <= most)) {
            most = q;
            at_r = row;
            at_c = col;
        }
        max_abs = @max(max_abs, e);
        if (@abs(w) > 0.01 * top) max_rel = @max(max_rel, e / @abs(w));
    };
    std.debug.print("RESULT {s} rows={d}: worst {d:.4} of the tolerance (row {d} col {d}), max abs err {e:.3} (max |want| {e:.3}), max rel err {e:.3} where |want| > 1% of max\n", .{ id, m, most, at_r, at_c, max_abs, top, max_rel });
    if (!(most <= 1)) {
        std.debug.print("FAIL {s} rows={d}: largest difference {d:.3} of its tolerance\n", .{ id, m, most });
        return error.TestFailed;
    }
}

/// A rank's columns [rank * RANK_N, +RANK_N) of the stored [K/16][N/16][W] trellis, repacked contiguous, with its svh.
fn cut(r: *Rig, trellis: []const i16, svh: []const u16, rank: usize) !struct { []i16, []u16 } {
    const words = (exl3fmt.Width{ .halves = HALVES }).tileWords();
    const nt = N / 16;
    const rt = RANK_N / 16;
    const out = try r.a.alloc(i16, (K / 16) * rt * words);
    for (0..K / 16) |kt| @memcpy(out[kt * rt * words ..][0 .. rt * words], trellis[(kt * nt + rank * rt) * words ..][0 .. rt * words]);
    return .{ out, try r.a.dupe(u16, svh[rank * RANK_N ..][0..RANK_N]) };
}

fn kv(r: *Rig, de: *exl3.Dense, which: []const u8) !void {
    const width: exl3fmt.Width = .{ .halves = HALVES };
    const trellis, const suh, const svh = try dense_t.tensors(r, K, N, width);
    const x = try r.bfs(PROMPT * K, 1.0);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const ref = try dense_t.promptReference(r, width, .mul1, trellis, K, N, x, suh, svh, PROMPT);
    const w = try dense_t.weight(r, trellis, suh, svh, K, N, HALVES, .mul1);
    const full_id = try std.fmt.allocPrint(r.a, "mm3.{s}_proj.exl3-8bit.k6144.n512", .{which});
    for (decode_rows) |m| {
        const y = try r.zeros(m * N * 2);
        try de.decode(r.s, w, dx, .bf16, m, y, .bf16);
        try compare(r, full_id, y, m, N, ref.want, ref.tol, N, 0);
    }
    const yfull = try r.zeros(PROMPT * N * 2);
    try de.prompt(r.s, w, dx, .bf16, PROMPT, yfull, .bf16);
    try compare(r, full_id, yfull, PROMPT, N, ref.want, ref.tol, N, 0);
    const full = try r.back(yfull, PROMPT * N * 2);
    for (0..2) |rank| {
        const tr, const sv = try cut(r, trellis, svh, rank);
        const wr = try dense_t.weight(r, tr, suh, sv, K, RANK_N, HALVES, .mul1);
        const id = try std.fmt.allocPrint(r.a, "mm3.{s}_proj.exl3-8bit.k6144.n256.rank{d}", .{ which, rank });
        for ([_]usize{ 1, 16 }) |m| {
            const y = try r.zeros(m * RANK_N * 2);
            try de.decode(r.s, wr, dx, .bf16, m, y, .bf16);
            try compare(r, id, y, m, RANK_N, ref.want, ref.tol, N, rank * RANK_N);
        }
        const y = try r.zeros(PROMPT * RANK_N * 2);
        try de.prompt(r.s, wr, dx, .bf16, PROMPT, y, .bf16);
        try compare(r, id, y, PROMPT, RANK_N, ref.want, ref.tol, N, rank * RANK_N);
        // informational: the rank cut against the full matrix' columns (the same plan, so the same sums are expected)
        const got = try r.back(y, PROMPT * RANK_N * 2);
        var differ: usize = 0;
        for (0..PROMPT) |row| {
            if (!std.mem.eql(u8, got[row * RANK_N * 2 ..][0 .. RANK_N * 2], full[(row * N + rank * RANK_N) * 2 ..][0 .. RANK_N * 2])) differ += 1;
        }
        std.debug.print("INFO {s}: {d} of {d} prompt rows differ in bytes from the full matrix' columns (plan sk {d} vs {d})\n", .{ id, differ, PROMPT, exl3.plan(K, RANK_N, 192, 8).sk, exl3.plan(K, N, 192, 8).sk });
    }
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var state = Run{};
    {
        var arena = std.heap.ArenaAllocator.init(gpu.gpa);
        defer arena.deinit();
        var prng = std.Random.DefaultPrng.init(0x6b76);
        var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
        defer {
            for (r.bufs.items) |*b| b.free();
            r.bufs.deinit(gpu.gpa);
        }
        var de = try exl3.Dense.load(gpu.d);
        defer de.unload();
        for ([_][]const u8{ "k", "v" }) |which| {
            const id = try std.fmt.allocPrint(r.a, "mm3.{s}_proj.exl3-8bit", .{which});
            if (kv(&r, &de, which)) check.pass("{s}", .{id}) else |e| state.fail(e, id);
        }
    }
    try @import("mm3_pack_moe.zig").run(gpu, s, &state);
    if (state.failed) return error.TestFailed;
}
