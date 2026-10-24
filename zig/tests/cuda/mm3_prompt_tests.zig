//! MiniMax-M3.1-Flash's prompt experts (x3gm) by width: FP64 bounds beside the decode kernel's, bit-stable tiles, rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const cand = @import("exl3_candidate_tests.zig");
const fmt = core.exl3_format;
const ex = cuda.exl3_experts;
const gm = cuda.exl3_gm;
const Gpu = check.Gpu;
const Rig = cand.Rig;

const E = 128; // one rank's TP2 share: every expert at half the 3,072-row width
const D = 6144;
const I = 1536;
const SLOTS = 4;
const CAP = 2048; // rows built: one prompt chunk
const ALPHA: f32 = 1.702;
const LIMIT: f32 = 7.0;
const UP_BIAS: f32 = 1.0;
const u: f64 = 1.0 / 16777216.0;
const row_counts = [_]usize{ 1, 17, 300, 2048 };

/// Expert e's gate (= up) and down bits: gate[e % 2], down[e % 2]; one value each is a uniform layer.
const Combo = struct { name: []const u8, gate: [2]u32, down: [2]u32 };
const combos = [_]Combo{
    .{ .name = "3/3/3", .gate = .{ 3, 3 }, .down = .{ 3, 3 } },
    .{ .name = "4/4/4", .gate = .{ 4, 4 }, .down = .{ 4, 4 } },
    .{ .name = "3/3/4", .gate = .{ 3, 3 }, .down = .{ 4, 4 } },
    .{ .name = "ragged", .gate = .{ 3, 4 }, .down = .{ 4, 3 } },
};

fn uniformOf(c: Combo) gm.Widths {
    if (c.gate[0] != c.gate[1] or c.down[0] != c.down[1]) return .{};
    return .{ .gu = 2 * c.gate[0], .down = 2 * c.down[0] };
}

/// One synthetic layer: device tensors, the host trellises and scales the reference reads, the seeded routing.
const Built = struct {
    layer: ex.Layer,
    gbits: [E]u32,
    dbits: [E]u32,
    gate: [E][]i16,
    up: [E][]i16,
    down: [E][]i16,
    suh_g: []f16,
    suh_u: []f16,
    svh_g: []f16,
    svh_u: []f16,
    suh_d: []f16,
    svh_d: []f16,
    x: []u16, // bf16 [CAP, D]
    pick: []i32, // [CAP, SLOTS]
    wts: []f32,
    dx: u64,
    dpick: u64,
    dwts: u64,
};

fn randI16(a: std.mem.Allocator, r: std.Random, n: usize) ![]i16 {
    const out = try a.alloc(i16, n);
    for (out) |*v| v.* = r.int(i16);
    return out;
}

/// Moderate scales, one in 16 large: the large gate and up columns reach swigluoai's clamp.
fn scales(a: std.mem.Allocator, r: std.Random, n: usize, large: bool) ![]f16 {
    const out = try a.alloc(f16, n);
    for (out, 0..) |*v, i| {
        const m: f32 = if (large and i % 16 == 5) 4 + 4 * r.float(f32) else 0.125 * r.float(f32);
        v.* = @floatCast(if (r.boolean()) m else -m);
    }
    return out;
}

fn build(gpu: Gpu, rig: *Rig, r: std.Random, c: Combo) !Built {
    const a = gpu.gpa;
    var b: Built = undefined;
    var gp: [E]u64 = undefined;
    var up_p: [E]u64 = undefined;
    var dp: [E]u64 = undefined;
    var gk2: [E]i32 = undefined;
    var dk2: [E]i32 = undefined;
    for (0..E) |e| {
        b.gbits[e] = c.gate[e % 2];
        b.dbits[e] = c.down[e % 2];
        b.gate[e] = try randI16(a, r, (D / 16) * (I / 16) * 16 * b.gbits[e]);
        b.up[e] = try randI16(a, r, (D / 16) * (I / 16) * 16 * b.gbits[e]);
        b.down[e] = try randI16(a, r, (I / 16) * (D / 16) * 16 * b.dbits[e]);
        gp[e] = try rig.put(b.gate[e]);
        up_p[e] = try rig.put(b.up[e]);
        dp[e] = try rig.put(b.down[e]);
        gk2[e] = @intCast(2 * b.gbits[e]);
        dk2[e] = @intCast(2 * b.dbits[e]);
    }
    b.suh_g = try scales(a, r, E * D, false);
    b.suh_u = try scales(a, r, E * D, false);
    b.svh_g = try scales(a, r, E * I, true);
    b.svh_u = try scales(a, r, E * I, true);
    b.suh_d = try scales(a, r, E * I, false);
    b.svh_d = try scales(a, r, E * D, false);
    b.layer = .{
        .gate = try rig.put(&gp),
        .up = try rig.put(&up_p),
        .down = try rig.put(&dp),
        .gate_k2 = try rig.put(&gk2),
        .up_k2 = try rig.put(&gk2),
        .down_k2 = try rig.put(&dk2),
        .suh_g = try rig.put(b.suh_g),
        .suh_u = try rig.put(b.suh_u),
        .svh_g = try rig.put(b.svh_g),
        .svh_u = try rig.put(b.svh_u),
        .suh_d = try rig.put(b.suh_d),
        .svh_d = try rig.put(b.svh_d),
        .experts = E,
        .dims = D,
        .width = I,
        .codebook = .mul1,
        .limit = LIMIT,
        .act_mode = 2,
        .alpha = ALPHA,
        .up_bias = UP_BIAS,
    };
    b.x = try a.alloc(u16, CAP * D);
    for (b.x) |*v| v.* = cand.bf16(r.float(f32) - 0.5);
    b.pick = try a.alloc(i32, CAP * SLOTS);
    for (0..CAP) |row| {
        var idx: [E]i32 = undefined;
        for (&idx, 0..) |*v, i| v.* = @intCast(i);
        var n: usize = E;
        for (0..SLOTS) |k| {
            // a hot expert: a quarter of the rows send slot 0 to expert 0, so its passes split
            var j = r.uintLessThan(usize, n);
            if (k == 0 and r.uintLessThan(u32, 4) == 0) j = std.mem.indexOfScalar(i32, idx[0..n], 0) orelse j;
            b.pick[row * SLOTS + k] = idx[j];
            idx[j] = idx[n - 1];
            n -= 1;
        }
    }
    b.wts = try a.alloc(f32, CAP * SLOTS);
    for (b.wts) |*v| v.* = 0.1 + 0.9 * r.float(f32);
    b.dx = try rig.put(b.x);
    b.dpick = try rig.put(b.pick);
    b.dwts = try rig.put(b.wts);
    return b;
}

fn freeBuilt(a: std.mem.Allocator, b: *Built) void {
    for (0..E) |e| {
        a.free(b.gate[e]);
        a.free(b.up[e]);
        a.free(b.down[e]);
    }
    for ([_][]f16{ b.suh_g, b.suh_u, b.svh_g, b.svh_u, b.suh_d, b.svh_d }) |s| a.free(s);
    a.free(b.x);
    a.free(b.pick);
    a.free(b.wts);
}

/// The rows the reference checks: the first 32 and 33 spread up to the chunk's last.
fn sampleRows() [65]usize {
    var s: [65]usize = undefined;
    for (0..32) |i| s[i] = i;
    for (0..32) |i| s[32 + i] = 32 + i * 63;
    s[64] = CAP - 1;
    return s;
}

fn f16v(v: f16) f64 {
    return @floatCast(v);
}

fn bfv(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

/// Each 128-block along the row axis replaced by the sum of its |entries| over sqrt(128) (the worst case).
fn absRotate(x: []f64, nrows: usize, cols: usize) void {
    const inv = 1.0 / @sqrt(@as(f64, 128.0));
    var r: usize = 0;
    while (r < nrows) : (r += 128) {
        for (0..cols) |c| {
            var s: f64 = 0;
            for (0..128) |n| s += @abs(x[(r + n) * cols + c]);
            for (0..128) |n| x[(r + n) * cols + c] = s * inv;
        }
    }
}

/// exl3_experts_f64_tests' f64 mirror and worst-case bound on the sampled rows, swigluoai's fp32 steps for the SiLU.
fn reference(a: std.mem.Allocator, b: *const Built, rows: []const usize, want: []f64, tol: []f64) !void {
    const p = rows.len * SLOTS;
    const pair = struct {
        fn row(rs: []const usize, s0: usize) usize {
            return rs[s0 / SLOTS] * SLOTS + s0 % SLOTS;
        }
    }.row;
    // xg / xu: fp16((x * suh) H) per slot, their error the fp16 rounding plus the fp32 rotation's, of the block's sum
    const xg = try a.alloc(f16, p * D);
    defer a.free(xg);
    const xu = try a.alloc(f16, p * D);
    defer a.free(xu);
    const xgerr = try a.alloc(f64, p * D);
    defer a.free(xgerr);
    const xuerr = try a.alloc(f64, p * D);
    defer a.free(xuerr);
    const xs = try a.alloc(f64, p * D);
    defer a.free(xs);
    for ([_]struct { []f16, []f64, []const f16 }{ .{ xg, xgerr, b.suh_g }, .{ xu, xuerr, b.suh_u } }) |m| {
        for (0..p) |s0| {
            const e: usize = @intCast(b.pick[pair(rows, s0)]);
            for (0..D) |k| xs[s0 * D + k] = bfv(b.x[rows[s0 / SLOTS] * D + k]) * f16v(m[2][e * D + k]);
        }
        fmt.rotate(xs, p, D, 1);
        for (m[0], xs) |*o, v| o.* = @floatCast(v);
        for (m[1], xs) |*o, v| o.* = @abs(v);
        for (0..p) |s0| {
            var c: usize = 0;
            while (c < D) : (c += 128) {
                var s: f64 = 0;
                for (0..128) |n| s += @abs(m[1][s0 * D + c + n]);
                for (0..128) |n| m[1][s0 * D + c + n] = s / @sqrt(128.0) * (1.0 / 2048.0 + 8 * u);
            }
        }
    }
    const zg = try a.alloc(f64, I * p);
    defer a.free(zg);
    const zu = try a.alloc(f64, I * p);
    defer a.free(zu);
    const eg = try a.alloc(f64, I * p);
    defer a.free(eg);
    const eu = try a.alloc(f64, I * p);
    defer a.free(eu);
    const s1g = try a.alloc(f64, I * p);
    defer a.free(s1g);
    const s1u = try a.alloc(f64, I * p);
    defer a.free(s1u);
    for ([_][]f64{ zg, zu, eg, eu, s1g, s1u }) |z| @memset(z, 0);
    const wq = try a.alloc(f16, D * I);
    defer a.free(wq);
    for (0..E) |e| {
        var used = false;
        for (0..p) |s0| used = used or b.pick[pair(rows, s0)] == @as(i32, @intCast(e));
        if (!used) continue;
        const w: fmt.Width = .{ .halves = @intCast(2 * b.gbits[e]) };
        for ([_]struct { []const i16, []const f16, []const f64, []f64, []f64, []f64 }{ .{ b.gate[e], xg, xgerr, zg, s1g, eg }, .{ b.up[e], xu, xuerr, zu, s1u, eu } }) |m| {
            fmt.unpack(w, .mul1, m[0], D / 16, I / 16, wq);
            for (0..p) |s0| {
                if (b.pick[pair(rows, s0)] != @as(i32, @intCast(e))) continue;
                for (0..D) |k| {
                    const t: f64 = f16v(m[1][s0 * D + k]);
                    const xe = m[2][s0 * D + k];
                    for (0..I) |n| {
                        const wv = f16v(wq[k * I + n]);
                        m[3][n * p + s0] += t * wv;
                        m[4][n * p + s0] += @abs(t) * @abs(wv);
                        m[5][n * p + s0] += @abs(wv) * xe;
                    }
                }
            }
        }
    }
    // the fp32 accumulation (2^-17 of 128 roundings) on the |t||w| sums, then the output Hadamard and its rotation
    for ([_]struct { []f64, []f64, []f64 }{ .{ zg, eg, s1g }, .{ zu, eu, s1u } }) |m| {
        for (m[1], m[2]) |*e, s| e.* += s * (1.0 / 131072.0);
        fmt.rotate(m[0], I, p, 0);
        absRotate(m[1], I, p);
        for (m[2], m[0]) |*o, v| o.* = @abs(v);
        absRotate(m[2], I, p);
        for (m[1], m[2]) |*e, v| e.* += v * 8 * u;
    }
    // swigluoai in fp32: the clamps are 1-Lipschitz, |d(g sigmoid(a g))/dg| <= 1.1, sigmoid's steps (|a g| + 4) u
    const v = try a.alloc(f64, I * p);
    defer a.free(v);
    const verr = try a.alloc(f64, I * p);
    defer a.free(verr);
    for (0..I) |n| for (0..p) |s0| {
        const e: usize = @intCast(b.pick[pair(rows, s0)]);
        const at = n * p + s0;
        const vg = zg[at] * f16v(b.svh_g[e * I + n]);
        const vu = zu[at] * f16v(b.svh_u[e * I + n]);
        const g = @min(vg, @as(f64, LIMIT));
        const uu = @min(@max(vu, -@as(f64, LIMIT)), @as(f64, LIMIT));
        const sg = g / (1.0 + @exp(-@as(f64, ALPHA) * g));
        const act = sg * (uu + UP_BIAS);
        v[at] = act * f16v(b.suh_d[e * I + n]);
        const egk = eg[at] * @abs(f16v(b.svh_g[e * I + n])) + @abs(vg) * 4 * u;
        const euk = eu[at] * @abs(f16v(b.svh_u[e * I + n])) + @abs(vu) * 4 * u;
        const esg = 1.1 * egk + @abs(sg) * (@abs(ALPHA * g) + 4) * u;
        const eact = @abs(uu + UP_BIAS) * esg + @abs(sg) * (euk + @abs(uu + UP_BIAS) * u) + @abs(act) * 2 * u;
        verr[at] = eact * @abs(f16v(b.suh_d[e * I + n])) + @abs(v[at]) * u;
    };
    fmt.rotate(v, I, p, 0);
    absRotate(verr, I, p);
    const vrot = try a.alloc(f64, I * p);
    defer a.free(vrot);
    for (vrot, v) |*o, val| o.* = @abs(val);
    absRotate(vrot, I, p);
    const xd = try a.alloc(f16, p * I);
    defer a.free(xd);
    const xderr = try a.alloc(f64, p * I);
    defer a.free(xderr);
    for (0..p) |s0| for (0..I) |k| {
        xd[s0 * I + k] = @floatCast(v[k * p + s0]);
        xderr[s0 * I + k] = verr[k * p + s0] + vrot[k * p + s0] * 8 * u + @abs(f16v(xd[s0 * I + k])) / 2048;
    };
    const zd = try a.alloc(f64, D * p);
    defer a.free(zd);
    const s1d = try a.alloc(f64, D * p);
    defer a.free(s1d);
    const de = try a.alloc(f64, D * p);
    defer a.free(de);
    for ([_][]f64{ zd, s1d, de }) |z| @memset(z, 0);
    const wd = try a.alloc(f16, I * D);
    defer a.free(wd);
    for (0..E) |e| {
        var used = false;
        for (0..p) |s0| used = used or b.pick[pair(rows, s0)] == @as(i32, @intCast(e));
        if (!used) continue;
        fmt.unpack(.{ .halves = @intCast(2 * b.dbits[e]) }, .mul1, b.down[e], I / 16, D / 16, wd);
        for (0..p) |s0| {
            if (b.pick[pair(rows, s0)] != @as(i32, @intCast(e))) continue;
            for (0..I) |k| {
                const t: f64 = f16v(xd[s0 * I + k]);
                const xe = xderr[s0 * I + k];
                for (0..D) |d| {
                    const wv = f16v(wd[k * D + d]);
                    zd[d * p + s0] += t * wv;
                    s1d[d * p + s0] += @abs(t) * @abs(wv);
                    de[d * p + s0] += @abs(wv) * xe;
                }
            }
        }
    }
    for (de, s1d) |*e, s| e.* += s * (1.0 / 2048.0 + 1.0 / 131072.0);
    fmt.rotate(zd, D, p, 0);
    absRotate(de, D, p);
    const zderr = try a.alloc(f64, D * p);
    defer a.free(zderr);
    for (zderr, zd) |*o, val| o.* = @abs(val);
    absRotate(zderr, D, p);
    for (de, zderr) |*e, val| e.* += val * 8 * u;
    // the per-slot outputs and the weighted fp32 fma chain over the slots
    for (rows, 0..) |row, ri| for (0..D) |d| {
        var acc: f32 = 0;
        var err: f64 = 0;
        var swy: f64 = 0;
        for (0..SLOTS) |k| {
            const s0 = ri * SLOTS + k;
            const e: usize = @intCast(b.pick[row * SLOTS + k]);
            const y: f32 = @floatCast(zd[d * p + s0] * f16v(b.svh_d[e * D + d]));
            const w = b.wts[row * SLOTS + k];
            acc = @mulAdd(f32, w, y, acc);
            err += @abs(w) * (de[d * p + s0] * @abs(f16v(b.svh_d[e * D + d])) + @abs(y) * u);
            swy += @abs(w * y);
        }
        want[ri * D + d] = acc;
        tol[ri * D + d] = err + swy / 2097152 + @abs(@as(f64, acc)) * u + 1e-30;
    };
}

/// Sampled rows below `n`: worst fraction of the bound, largest error, worst column's error over the largest output.
const Score = struct { frac: f64 = 0, err: f64 = 0, col: f64 = 0, rms: f64 = 0 };

fn score(got: []const f32, rows: []const usize, n: usize, want: []const f64, tol: []const f64) Score {
    var s: Score = .{};
    var n_vals: f64 = 0;
    var scale: f64 = 0;
    for (rows, 0..) |row, ri| if (row < n) {
        for (want[ri * D ..][0..D]) |w| scale = @max(scale, @abs(w));
    };
    for (0..D) |d| {
        var col: f64 = 0;
        for (rows, 0..) |row, ri| if (row < n) {
            const e = @abs(@as(f64, got[row * D + d]) - want[ri * D + d]);
            s.frac = @max(s.frac, e / tol[ri * D + d]);
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

const Runner = struct {
    gpu: Gpu,
    s: cuda.Stream,
    old: *const ex.Experts,
    new: *gm.Experts,
    old_sc: u64,
    new_sc: u64,
    out: u64,
    host: []f32,

    fn newRun(r: *Runner, b: *const Built, w: gm.Widths, row0: usize, n: usize) !void {
        const sc = gm.scratch(r.new_sc, b.layer, CAP, SLOTS);
        try r.new.run(r.gpu.d, r.s, b.layer, w, b.dx + row0 * D * 2, D, b.dpick + row0 * SLOTS * 4, b.dwts + row0 * SLOTS * 4, n, r.out + row0 * D * 4, sc);
    }

    fn oldRun(r: *Runner, b: *const Built, n: usize) !void {
        const sc = try ex.scratch(r.old_sc, b.layer, 256, SLOTS);
        var at: usize = 0;
        while (at < n) : (at += 256) {
            const m = @min(256, n - at);
            try r.old.run(r.s, b.layer, b.dx + at * D * 2, D, .bf16, b.dpick + at * SLOTS * 4, b.dwts + at * SLOTS * 4, m, r.out + at * D * 4, sc);
        }
    }

    /// The fp32 outputs of the first `n` rows, after a sync.
    fn fetch(r: *Runner, n: usize) ![]f32 {
        try r.s.synchronize();
        const dev: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = r.out, .len = n * D * 4 };
        try dev.download(0, std.mem.sliceAsBytes(r.host[0 .. n * D]));
        return r.host[0 .. n * D];
    }

    fn clear(r: *Runner) !void {
        const dev: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = r.out, .len = CAP * D * 4 };
        try dev.fill8(0xff, r.s.handle); // NaN canaries: a row left unwritten fails every comparison
    }
};

/// `tf-cuda-test mm3-prompt-contract`: every width combination's FP64 bounds, tiles, v2 and chunk splits.
pub fn run(gpu: Gpu) !void {
    const a = gpu.gpa;
    var old = try ex.Experts.load(gpu.d);
    defer old.unload();
    var new = try gm.Experts.load(gpu.d);
    defer new.unload();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x3A3A);
    const rows = sampleRows();
    const want = try a.alloc(f64, rows.len * D);
    defer a.free(want);
    const tol = try a.alloc(f64, rows.len * D);
    defer a.free(tol);
    const keep = try a.alloc(f32, CAP * D);
    defer a.free(keep);
    var failed = false;
    for (combos) |c| {
        var rig: Rig = .{ .gpu = gpu };
        defer rig.deinit();
        var b = try build(gpu, &rig, prng.random(), c);
        defer freeBuilt(a, &b);
        var r: Runner = .{ .gpu = gpu, .s = s, .old = &old, .new = &new, .old_sc = try rig.zeros(try ex.scratchBytes(b.layer, 256, SLOTS)), .new_sc = try rig.zeros(gm.scratchBytes(b.layer, CAP, SLOTS)), .out = try rig.zeros(CAP * D * 4), .host = try a.alloc(f32, CAP * D) };
        defer a.free(r.host);
        const w = uniformOf(c);
        try reference(a, &b, &rows, want, tol);
        for (row_counts) |n| {
            new.force = null;
            try r.clear();
            try r.newRun(&b, w, 0, n);
            const sn = score(try r.fetch(n), &rows, n, want, tol);
            if (n == CAP) @memcpy(keep, r.host);
            try r.clear();
            try r.oldRun(&b, n);
            const so = score(try r.fetch(n), &rows, n, want, tol);
            var diff: f64 = 0;
            for (&rows, 0..) |row, ri| if (row < n) for (0..D) |d| {
                diff = @max(diff, @abs(@as(f64, keep[row * D + d]) - r.host[row * D + d]) / tol[ri * D + d]);
            };
            if (n != CAP) diff = std.math.nan(f64); // the kept rows are the 2,048-row call's
            std.debug.print("TABLE mm3.experts {s} rows={d} new_frac={e:.3} old_frac={e:.3} new_err={e:.3} old_err={e:.3} new_col={e:.3} old_col={e:.3} new_rms={e:.3} old_rms={e:.3} new_vs_old={e:.3}\n", .{ c.name, n, sn.frac, so.frac, sn.err, so.err, sn.col, so.col, sn.rms, so.rms, diff });
            if (!(sn.frac <= 1)) {
                std.debug.print("FAIL mm3.experts {s} rows={d}: x3gm at {d:.4} of its FP64 bound\n", .{ c.name, n, sn.frac });
                failed = true;
            }
            if (!(so.frac <= 1)) std.debug.print("FINDING mm3.experts {s} rows={d}: the decode kernel at {d:.4} of the bound\n", .{ c.name, n, so.frac });
        }
        bitChecks(&r, &b, c, w, keep) catch |e| {
            std.debug.print("FAIL mm3.experts {s}: {t}\n", .{ c.name, e });
            failed = true;
        };
    }
    if (failed) return error.TestFailed;
    check.pass("mm3.experts: x3gm within its FP64 bounds at every width combination and row count; tiles, v2 and chunk splits bit-equal", .{});
}

/// The 2,048-row call's bytes (`keep`) again from every tile it builds, v2's kernel, and two other chunk splits.
fn bitChecks(r: *Runner, b: *const Built, c: Combo, w: gm.Widths, keep: []const f32) !void {
    var tiles: usize = 0;
    const ngu: usize = if (w.gu == 0) 2 else cuda.exl3.block.x3gm.gu_tiles.len;
    const ndn: usize = if (w.gu == 0) 1 else cuda.exl3.block.x3gm.dn_tiles.len;
    for (0..ngu) |gu| for (0..ndn) |dn| {
        r.new.force = .{ .gu = gu, .dn = dn };
        try r.clear();
        r.newRun(b, w, 0, CAP) catch |e| if (e == error.Unsupported) continue else return e;
        try check.sameBytes("mm3.experts tile", std.mem.sliceAsBytes(try r.fetch(CAP)), std.mem.sliceAsBytes(keep));
        tiles += 1;
    };
    r.new.force = null;
    if (w.gu != 0) {
        try r.clear();
        try r.newRun(b, .{}, 0, CAP);
        try check.sameBytes("mm3.experts v2", std.mem.sliceAsBytes(try r.fetch(CAP)), std.mem.sliceAsBytes(keep));
    }
    try r.clear();
    try r.newRun(b, w, 0, 1000);
    try r.newRun(b, w, 1000, CAP - 1000);
    try check.sameBytes("mm3.experts split 1000", std.mem.sliceAsBytes(try r.fetch(CAP)), std.mem.sliceAsBytes(keep));
    try r.clear();
    for (0..CAP / 128) |k| try r.newRun(b, w, k * 128, 128);
    try check.sameBytes("mm3.experts 128-row calls", std.mem.sliceAsBytes(try r.fetch(CAP)), std.mem.sliceAsBytes(keep));
    check.pass("mm3.experts {s}: {d} tile pairs{s}, a 1000 + 1048 split and 128-row calls give the 2,048-row call's bytes", .{ c.name, tiles, if (w.gu != 0) ", v2" else "" });
}
