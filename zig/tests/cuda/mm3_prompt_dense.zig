//! MiniMax-M3.1-Flash's dense prompt rows (pfdense) by class: FP64 bounds beside the decode blocks', rows bit-stable.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const cand = @import("exl3_candidate_tests.zig");
const Gpu = check.Gpu;
const Rig = cand.Rig;

const CAP = 2048;
const row_counts = [_]usize{ 1, 17, 128, 300, 2048 };

/// One projection class at a TP2 rank's shape, its bits and the output type forward_exec gives it.
const Class = struct { name: []const u8, k: usize, n: usize, bits: u32, f32_out: bool };
const classes = [_]Class{
    .{ .name = "q", .k = 6144, .n = 4096, .bits = 6, .f32_out = false },
    .{ .name = "k/v", .k = 6144, .n = 256, .bits = 8, .f32_out = false },
    .{ .name = "o", .k = 4096, .n = 6144, .bits = 6, .f32_out = true },
    .{ .name = "dense gate/up", .k = 6144, .n = 6144, .bits = 6, .f32_out = false },
    .{ .name = "dense down", .k = 6144, .n = 6144, .bits = 6, .f32_out = true },
    .{ .name = "shared gate/up", .k = 6144, .n = 1536, .bits = 6, .f32_out = false },
    .{ .name = "shared down", .k = 1536, .n = 6144, .bits = 6, .f32_out = true },
    .{ .name = "4-bit", .k = 6144, .n = 1536, .bits = 4, .f32_out = false },
    .{ .name = "5-bit", .k = 1536, .n = 6144, .bits = 5, .f32_out = true },
};

/// The rows the reference checks: the first 16 and 17 spread up to the chunk's last.
fn sampleRows() [33]usize {
    var s: [33]usize = undefined;
    for (0..16) |i| s[i] = i;
    for (0..16) |i| s[16 + i] = 16 + i * 127;
    s[32] = CAP - 1;
    return s;
}

fn ofBf16(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

const Score = struct { frac: f64 = 0, err: f64 = 0, col: f64 = 0, rms: f64 = 0 };

fn score(got: []const f32, n_out: usize, rows: []const usize, n: usize, want: []const f64, bound: []const f64) Score {
    var s: Score = .{};
    var n_vals: f64 = 0;
    var scale: f64 = 0;
    for (rows, 0..) |row, ri| if (row < n) {
        for (want[ri * n_out ..][0..n_out]) |w| scale = @max(scale, @abs(w));
    };
    for (0..n_out) |j| {
        var col: f64 = 0;
        for (rows, 0..) |row, ri| if (row < n) {
            const e = @abs(@as(f64, got[row * n_out + j]) - want[ri * n_out + j]);
            s.frac = @max(s.frac, e / bound[ri * n_out + j]);
            s.err = @max(s.err, e);
            s.rms += e * e;
            n_vals += 1;
            col = @max(col, e);
        };
        if (scale > 0) s.col = @max(s.col, col / scale);
    }
    s.rms = @sqrt(s.rms / @max(n_vals, 1));
    return s;
}

/// The first `n` rows of `y` as f32, after a sync.
fn fetch(gpu: Gpu, s: cuda.Stream, y: u64, n: usize, n_out: usize, f32_out: bool, out: []f32) ![]f32 {
    try s.synchronize();
    const vals = out[0 .. n * n_out];
    if (f32_out) {
        const dev: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = y, .len = vals.len * 4 };
        try dev.download(0, std.mem.sliceAsBytes(vals));
        return vals;
    }
    const raw = try gpu.gpa.alloc(u16, vals.len);
    defer gpu.gpa.free(raw);
    const dev: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = y, .len = vals.len * 2 };
    try dev.download(0, std.mem.sliceAsBytes(raw));
    for (vals, raw) |*v, b| v.* = ofBf16(b);
    return vals;
}

/// `tf-cuda-test mm3-prompt-dense`: every class's FP64 bounds, old and new, and the new rows' bytes at every count.
pub fn run(gpu: Gpu) !void {
    const a = gpu.gpa;
    var old = try cuda.exl3.Dense.load(gpu.d);
    defer old.unload();
    var new = try cuda.exl3.Dense.load(gpu.d);
    defer new.unload();
    try new.usePromptTiles();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0xD3A5);
    const rows = sampleRows();
    var failed = false;
    for (classes) |c| {
        const r = prng.random();
        var rig: Rig = .{ .gpu = gpu };
        defer rig.deinit();
        const h = try cand.randomMatrix(a, r, .{ .k = c.k, .n = c.n, .bits = c.bits, .book = .mul1 });
        defer cand.freeHost(a, h);
        const x = try cand.randomRows(a, r, CAP, c.k);
        defer a.free(x);
        const xs = try a.alloc(u16, rows.len * c.k);
        defer a.free(xs);
        for (&rows, 0..) |row, i| @memcpy(xs[i * c.k ..][0..c.k], x[row * c.k ..][0..c.k]);
        const want = try a.alloc(f64, rows.len * c.n);
        defer a.free(want);
        const bound = try a.alloc(f64, rows.len * c.n);
        defer a.free(bound);
        try cand.reference(a, h, xs, rows.len, want, bound, true);
        const st = cuda.exl3.strides(c.k, c.n, 2 * c.bits, false);
        const w: cuda.exl3.Weight = .{ .trellis = try rig.put(h.trellis), .suh = try rig.put(h.suh), .svh = try rig.put(h.svh), .k = @intCast(c.k), .n = @intCast(c.n), .halves = 2 * c.bits, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
        const dx = try rig.put(x);
        const ydt: cuda.exl3.DType = if (c.f32_out) .f32 else .bf16;
        const y = try rig.zeros(CAP * c.n * 4);
        const got = try a.alloc(f32, CAP * c.n);
        defer a.free(got);
        const keep = try a.alloc(f32, CAP * c.n);
        defer a.free(keep);
        var stable = true;
        for (row_counts) |n| {
            try new.prompt(s, w, dx, .bf16, n, y, ydt);
            const sn = score(try fetch(gpu, s, y, n, c.n, c.f32_out, got), c.n, &rows, n, want, bound);
            if (n == 1) @memcpy(keep[0..c.n], got[0..c.n]);
            if (n == CAP) {
                // every shorter call's rows are a prefix of this one's, byte for byte
                stable = std.mem.eql(f32, keep[0..c.n], got[0..c.n]);
                @memcpy(keep, got);
            }
            try old.prompt(s, w, dx, .bf16, n, y, ydt);
            const so = score(try fetch(gpu, s, y, n, c.n, c.f32_out, got), c.n, &rows, n, want, bound);
            std.debug.print("TABLE mm3.dense {s} k={d} n={d} bits={d} out={s} rows={d} new_frac={e:.3} old_frac={e:.3} new_err={e:.3} old_err={e:.3} new_col={e:.3} old_col={e:.3} new_rms={e:.3} old_rms={e:.3}\n", .{ c.name, c.k, c.n, c.bits, if (c.f32_out) "f32" else "bf16", n, sn.frac, so.frac, sn.err, so.err, sn.col, so.col, sn.rms, so.rms });
            if (!(sn.frac <= 1)) {
                std.debug.print("FAIL mm3.dense {s} rows={d}: pfdense at {d:.4} of its FP64 bound\n", .{ c.name, n, sn.frac });
                failed = true;
            }
            if (!(so.frac <= 1)) std.debug.print("FINDING mm3.dense {s} rows={d}: the decode blocks at {d:.4} of the bound\n", .{ c.name, n, so.frac });
        }
        for ([_]usize{ 17, 300 }) |n| {
            try new.prompt(s, w, dx, .bf16, n, y, ydt);
            stable = stable and std.mem.eql(f32, try fetch(gpu, s, y, n, c.n, c.f32_out, got), keep[0 .. n * c.n]);
        }
        if (!stable) {
            std.debug.print("FAIL mm3.dense {s}: a row's bytes depend on the call's row count\n", .{c.name});
            failed = true;
        }
    }
    if (failed) return error.TestFailed;
    check.pass("mm3.dense: pfdense within its FP64 bounds for every class and row count, each row's bytes independent of the call", .{});
}
