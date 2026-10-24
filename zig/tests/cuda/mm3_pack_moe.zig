//! MiniMax-M3.1-Flash layer 18's routed experts (w1/w3 3 bits, w2 4, 128 experts, top-4) through the family's tables.
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("minimax_cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3fmt = @import("core").exl3_format;
const exl3e = cuda.exl3_experts;
const Run = @import("exl3_tests.zig").Run;
const Rig = quant_ref_tests.RigT;
const bf = quant_ref_tests.helpers.bfOf;

const E = 128;
const D = 6144;
const SLOTS = 4;
const halves = [3]u32{ 6, 6, 8 }; // gate (w1), up (w3), down (w2): 3, 3 and 4 bits for every expert
const cap_rows = 256; // forward_plan.zig expert_rows: the scratch capacity the engine sizes
const eps24 = 0x1p-24;
const eps23 = 0x1p-23; // tensor-core sums may truncate: one unit of fp32 per add, not half

pub const Case = struct { id: []const u8, width: usize, rows: []const usize };

/// A TP2 rank's slice of the 3072-wide experts (the shape the engine launches), then the whole width.
pub const cases = [_]Case{
    .{ .id = "mm3.moe.layer18.w3-3-4.rank-width-1536", .width = 1536, .rows = &.{ 1, 2, 16, 32 } },
    .{ .id = "mm3.moe.layer18.w3-3-4.full-width-3072", .width = 3072, .rows = &.{ 1, 8 } },
};

const Built = struct {
    layer: exl3e.Layer,
    w: [3][E][]i16,
    sh: [6][]u16, // suh_g, suh_u [E, D], svh_g, svh_u, suh_d [E, I], svh_d [E, D]
    x: []u16,
    pick: []i32,
    wts: []f32,
    dx: u64,
    dpick: u64,
    dwts: u64,
};

fn f16v(v: u16) f64 {
    return @floatCast(@as(f16, @bitCast(v)));
}

fn randScale(r: *Rig, out: []u16, half_range: f32) void {
    for (out) |*v| v.* = @bitCast(@as(f16, @floatCast((r.rng.float(f32) * 2 - 1) * half_range)));
}

/// Random trellises for 128 experts, scales that cross the clamp of 7; row 0 routes to 127, 0, 64, 31, others random.
fn build(r: *Rig, width: usize, rows: usize) !Built {
    var b: Built = undefined;
    var ptrs: [3][]u64 = undefined;
    var k2: [3][E]i32 = undefined;
    for (0..3) |j| {
        ptrs[j] = try r.a.alloc(u64, E);
        const kd: usize = if (j == 2) width else D;
        const nd: usize = if (j == 2) D else width;
        const words = (exl3fmt.Width{ .halves = @intCast(halves[j]) }).tileWords();
        for (0..E) |e| {
            b.w[j][e] = try r.a.alloc(i16, (kd / 16) * (nd / 16) * words);
            r.rng.bytes(std.mem.sliceAsBytes(b.w[j][e]));
            ptrs[j][e] = try r.dev(std.mem.sliceAsBytes(b.w[j][e]));
            k2[j][e] = @intCast(halves[j]);
        }
    }
    const lens = [6]usize{ E * D, E * D, E * width, E * width, E * width, E * D };
    const ranges = [6]f32{ 0.25, 0.25, 1.0, 1.0, 0.25, 0.25 };
    for (&b.sh, lens, ranges) |*s, n, h| {
        s.* = try r.a.alloc(u16, n);
        randScale(r, s.*, h);
    }
    var dsh: [6]u64 = undefined;
    for (&dsh, b.sh) |*d, s| d.* = try r.dev(std.mem.sliceAsBytes(s));
    const t: mm.weights.Tables = .{ .layer = 0, .gate = ptrs[0], .up = ptrs[1], .down = ptrs[2], .gate_k2 = &k2[0], .up_k2 = &k2[1], .down_k2 = &k2[2] };
    const base = try r.dev(try mm.weights.packTables(r.a, t));
    const zw: cuda.exl3.Weight = .{ .trellis = 0, .suh = 0, .svh = 0, .k = 0, .n = 0, .halves = 0, .codebook = .mul1, .stride_k = 0, .stride_nb = 0 };
    // as weights.moeLayer describes a routed layer; uniformK2 is 0 here because w2's width differs from w1's and w3's
    const l0: exl3e.Layer = .{ .gate = 0, .up = 0, .down = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = dsh[0], .suh_u = dsh[1], .svh_g = dsh[2], .svh_u = dsh[3], .suh_d = dsh[4], .svh_d = dsh[5], .experts = E, .dims = D, .width = width, .codebook = .mul1, .uniform_k2 = 0, .shared_input = false };
    var lays = [1]mm.weights.Layer{.{ .q = zw, .k = zw, .v = zw, .o = zw, .ffn = .{ .moe = .{ .router = 0, .bias = 0, .experts = l0, .shared = .{ .gate = zw, .up = zw, .down = zw } } } }};
    var wt: mm.weights.Weights = .{ .head = zw, .layers = &lays };
    mm.weights.attachTables(&wt, t, mm.weights.tablePtrs(base, E), .{});
    b.layer = wt.layers[0].ffn.moe.experts;
    b.x = try r.bfs(rows * D, 1.0);
    b.pick = try r.a.alloc(i32, rows * SLOTS);
    for (0..rows) |row| {
        var idx: [E]i32 = undefined;
        for (&idx, 0..) |*v, i| v.* = @intCast(i);
        var n: usize = E;
        for (0..SLOTS) |k| {
            const j = r.rng.uintLessThan(usize, n);
            b.pick[row * SLOTS + k] = idx[j];
            idx[j] = idx[n - 1];
            n -= 1;
        }
    }
    @memcpy(b.pick[0..SLOTS], &[_]i32{ 127, 0, 64, 31 });
    b.wts = try r.a.alloc(f32, rows * SLOTS);
    for (b.wts) |*v| v.* = 0.1 + 0.9 * r.rng.float(f32);
    b.dx = try r.dev(std.mem.sliceAsBytes(b.x));
    b.dpick = try r.dev(std.mem.sliceAsBytes(b.pick));
    b.dwts = try r.dev(std.mem.sliceAsBytes(b.wts));
    return b;
}

/// Each 128-run along a row replaced by the sum of its |entries| over sqrt(128): a Hadamard output's worst case.
fn absRotateRows(x: []f64) void {
    const inv = 1.0 / @sqrt(@as(f64, 128.0));
    var c: usize = 0;
    while (c + 128 <= x.len) : (c += 128) {
        var s: f64 = 0;
        for (x[c..][0..128]) |v| s += @abs(v);
        for (x[c..][0..128]) |*v| v.* = s * inv;
    }
}

fn used(b: *const Built, ex: usize, p: usize) bool {
    for (b.pick[0..p]) |v| if (v == @as(i32, @intCast(ex))) return true;
    return false;
}

/// Two fp32 roundings a 16-wide tensor-core step (adds may truncate: a whole unit), the split sums, and the margin.
fn depth(k: usize) f64 {
    return @floatFromInt(k / 8 + 8);
}

/// Rows of a slot's GEMM over k: z [p][n] and the sums of |x w|, from inputs x [p][k], expert by expert.
fn gemmRows(r: *Rig, b: *const Built, j: usize, x: []const f64, k: usize, n: usize, p: usize, wh: u32, z: []f64, s1: []f64) !void {
    const wq = try r.a.alloc(f16, k * n);
    @memset(z, 0);
    @memset(s1, 0);
    for (0..E) |ex| {
        if (!used(b, ex, p)) continue;
        exl3fmt.unpack(.{ .halves = @intCast(wh) }, .mul1, b.w[j][ex], k / 16, n / 16, wq);
        for (0..p) |s0| {
            if (b.pick[s0] != @as(i32, @intCast(ex))) continue;
            const zr = z[s0 * n ..][0..n];
            const sr = s1[s0 * n ..][0..n];
            for (0..k) |ki| {
                const t = x[s0 * k + ki];
                const ta = @abs(t);
                const row = wq[ki * n ..][0..n];
                for (zr, sr, row) |*zv, *sv, wv| {
                    const w: f64 = @floatCast(wv);
                    zv.* += t * w;
                    sv.* += ta * @abs(w);
                }
            }
        }
    }
}

/// z rotated along each slot's row in place; `err` gets its bound from the sums' depth and the Hadamard.
fn rotRows(r: *Rig, z: []f64, s1: []const f64, n: usize, p: usize, k: usize, err: []f64) !void {
    const za = try r.a.alloc(f64, z.len);
    for (err, s1, za, z) |*e, s, *a, v| {
        e.* = depth(k) * eps23 * s;
        a.* = @abs(v);
    }
    exl3fmt.rotate(z, p, n, 1);
    absRotateRows(err);
    absRotateRows(za);
    for (err, za) |*e, a| e.* += 9 * eps24 * a;
}

/// rot_in: x * suh rotated to fp16 (exact values and each one's bound: the fp32 product and Hadamard, the fp16 store).
fn inStage(b: *const Built, j: usize, p: usize, out: []f64, tol: []f64) void {
    for (0..p) |s0| {
        const ex: usize = @intCast(b.pick[s0]);
        const o = out[s0 * D ..][0..D];
        const t = tol[s0 * D ..][0..D];
        for (o, t, 0..) |*v, *e, i| {
            v.* = bf(b.x[(s0 / SLOTS) * D + i]) * f16v(b.sh[j][ex * D + i]);
            e.* = @abs(v.*);
        }
        exl3fmt.rotate(o, 1, D, 1);
        absRotateRows(t);
        for (t, o) |*e, v| e.* = (9 * eps24 * e.* + @abs(v) * 0x1p-11) * (1 + 0x1p-10) + 0x1p-25;
    }
}

/// Gate/up GEMMs from fp16 inputs, Hadamard, svh, fp32 swigluoai (up bias `ub`), suh_d, Hadamard: xd [p][I], bounds.
fn actStage(r: *Rig, b: *const Built, I: usize, p: usize, xg: []const f64, xu: []const f64, ub: f64, out: []f64, tol: []f64) !void {
    const l = b.layer;
    var z: [2][]f64 = undefined;
    var e: [2][]f64 = undefined;
    const s1 = try r.a.alloc(f64, p * I);
    for (0..2) |j| {
        z[j] = try r.a.alloc(f64, p * I);
        e[j] = try r.a.alloc(f64, p * I);
        try gemmRows(r, b, j, if (j == 0) xg else xu, D, I, p, halves[j], z[j], s1);
        try rotRows(r, z[j], s1, I, p, D, e[j]);
    }
    const lim: f64 = l.limit;
    const alpha: f64 = l.alpha;
    for (0..p) |s0| for (0..I) |n| {
        const ex: usize = @intCast(b.pick[s0]);
        const i = s0 * I + n;
        const hg = f16v(b.sh[2][ex * I + n]);
        const hu = f16v(b.sh[3][ex * I + n]);
        const vg = z[0][i] * hg;
        const vu = z[1][i] * hu;
        const egk = e[0][i] * @abs(hg) + 3 * eps24 * @abs(vg);
        const euk = e[1][i] * @abs(hu) + 3 * eps24 * @abs(vu);
        const gg = if (l.act_mode == 3) std.math.clamp(vg, -lim, lim) else @min(vg, lim);
        const uu = std.math.clamp(vu, -lim, lim);
        const x = alpha * gg;
        const sg = gg / (1 + @exp(-x));
        const act = sg * (uu + ub);
        const sd = f16v(b.sh[4][ex * I + n]);
        out[i] = act * sd;
        // d(g sigmoid(a g))/dg <= 1.1, d(act)/du = sg, clamps 1-Lipschitz; exp 2 ulps, add, div, add, mul
        const eact = 1.1 * egk * (@abs(uu + ub) + euk) + @abs(sg) * euk + @abs(act) * (8 + @abs(x)) * eps24;
        tol[i] = eact * @abs(sd) + @abs(out[i]) * eps24;
    };
    const va = try r.a.alloc(f64, p * I);
    for (va, out) |*a, v| a.* = @abs(v);
    exl3fmt.rotate(out, p, I, 1);
    absRotateRows(tol);
    absRotateRows(va);
    for (tol, va, out) |*t, a, v| t.* = (t.* + 9 * eps24 * a + @abs(v) * 0x1p-11) * (1 + 0x1p-10) + 0x1p-25;
}

/// Down from fp16 inputs [p][I] at `wh` half-bits, Hadamard and svh, the fma chain over 4 slots: [m][D] and bounds.
fn downStage(r: *Rig, b: *const Built, I: usize, m: usize, xd: []const f64, wh: u32, want: []f64, tol: []f64) !void {
    const p = m * SLOTS;
    const z = try r.a.alloc(f64, p * D);
    const s1 = try r.a.alloc(f64, p * D);
    const e = try r.a.alloc(f64, p * D);
    try gemmRows(r, b, 2, xd, I, D, p, wh, z, s1);
    try rotRows(r, z, s1, D, p, I, e);
    for (0..m) |row| for (0..D) |d| {
        var acc: f64 = 0;
        var err: f64 = 0;
        var swy: f64 = 0;
        for (0..SLOTS) |k| {
            const s0 = row * SLOTS + k;
            const ex: usize = @intCast(b.pick[s0]);
            const sv = f16v(b.sh[5][ex * D + d]);
            const y = z[s0 * D + d] * sv;
            const w: f64 = b.wts[s0];
            acc += w * y;
            err += @abs(w) * (e[s0 * D + d] * @abs(sv) + 3 * eps24 * @abs(y));
            swy += @abs(w * y);
        }
        want[row * D + d] = acc;
        tol[row * D + d] = err + (SLOTS + 1) * eps24 * (swy + err) + 1e-30;
    };
}

/// The worst |got - want| / tol, where it is, and the max abs and relative errors (relative where |want| > 1% of max).
const Miss = struct { q: f64 = 0, at: usize = 0, abs: f64 = 0, rel: f64 = 0, top: f64 = 0 };

fn miss(got: []const f64, want: []const f64, tol: ?[]const f64) Miss {
    var o: Miss = .{};
    for (want) |w| o.top = @max(o.top, @abs(w));
    for (got, want, 0..) |g, w, i| {
        const e = @abs(g - w);
        if (tol) |t| {
            const q = e / t[i];
            if (!(q <= o.q)) {
                o.q = q;
                o.at = i;
            }
        }
        o.abs = @max(o.abs, e);
        if (@abs(w) > 0.01 * o.top) o.rel = @max(o.rel, e / @abs(w));
    }
    return o;
}

fn halfs(r: *Rig, ptr: u64, n: usize) ![]f64 {
    const raw = try r.back(ptr, n * 2);
    const out = try r.a.alloc(f64, n);
    for (out, 0..) |*o, i| o.* = f16v(std.mem.bytesToValue(u16, raw[2 * i ..][0..2]));
    return out;
}

fn floats(r: *Rig, ptr: u64, n: usize) ![]f64 {
    const raw = try r.back(ptr, n * 4);
    const out = try r.a.alloc(f64, n);
    for (out, 0..) |*o, i| o.* = std.mem.bytesToValue(f32, raw[4 * i ..][0..4]);
    return out;
}

fn say(id: []const u8, what: []const u8, m: usize, cols: usize, x: Miss) void {
    std.debug.print("RESULT {s} {s} rows={d}: worst {d:.4} of the bound (slot or row {d}, col {d}), max abs err {e:.3} (max |want| {e:.3}), max rel err {e:.3}\n", .{ id, what, m, x.q, x.at / cols, x.at % cols, x.abs, x.top, x.rel });
}

/// Each stage from the kernel's previous one (scratch xg, xu, xd) within its bound; two planted controls must miss.
fn stages(r: *Rig, b: *const Built, I: usize, m: usize, sc: exl3e.Scratch, out: u64, id: []const u8) !void {
    const p = m * SLOTS;
    const xg = try halfs(r, sc.xg, p * D);
    const xu = try halfs(r, sc.xu, p * D);
    const xd = try halfs(r, sc.xd, p * I);
    const got = try floats(r, out, m * D);
    const v_in = try r.a.alloc(f64, p * D);
    const t_in = try r.a.alloc(f64, p * D);
    const v_in2 = try r.a.alloc(f64, p * D);
    var fail = false;
    for ([_][]const f64{ xg, xu }, 0..) |k_in, j| {
        inStage(b, j, p, if (j == 0) v_in2 else v_in, t_in);
        const x = miss(k_in, if (j == 0) v_in2 else v_in, t_in);
        say(id, if (j == 0) "stage rot_in gate" else "stage rot_in up", m, D, x);
        fail = fail or !(x.q <= 1);
    }
    const v_xd = try r.a.alloc(f64, p * I);
    const t_xd = try r.a.alloc(f64, p * I);
    try actStage(r, b, I, p, xg, xu, b.layer.up_bias, v_xd, t_xd);
    const xa = miss(xd, v_xd, t_xd);
    say(id, "stage gate/up + swigluoai -> fp16 down input", m, I, xa);
    fail = fail or !(xa.q <= 1);
    const want = try r.a.alloc(f64, m * D);
    const tol = try r.a.alloc(f64, m * D);
    try downStage(r, b, I, m, xd, halves[2], want, tol);
    const xo = miss(got, want, tol);
    say(id, "stage down + combine", m, D, xo);
    fail = fail or !(xo.q <= 1);
    // planted controls: each must miss its stage's bound
    try actStage(r, b, I, p, xg, xu, 0, v_xd, t_xd);
    const c1 = miss(xd, v_xd, t_xd);
    std.debug.print("CONTROL {s} up bias 0: {d:.1} of the bound ({s})\n", .{ id, c1.q, if (c1.q > 1) "red, as planted" else "MISSED" });
    try downStage(r, b, I, m, xd, 6, want, tol);
    const c2 = miss(got, want, tol);
    std.debug.print("CONTROL {s} down read at 3 bits: {d:.1} of the bound ({s})\n", .{ id, c2.q, if (c2.q > 1) "red, as planted" else "MISSED" });
    fail = fail or !(c1.q > 1) or !(c2.q > 1);
    // the whole chain in float64 from x: informational error figures
    inStage(b, 0, p, v_in2, t_in);
    inStage(b, 1, p, v_in, t_in);
    try actStage(r, b, I, p, v_in2, v_in, b.layer.up_bias, v_xd, t_xd);
    try downStage(r, b, I, m, v_xd, halves[2], want, tol);
    say(id, "end to end vs float64 (not gated)", m, D, miss(got, want, null));
    if (fail) return error.TestFailed;
}

fn one(gpu: check.Gpu, s: cuda.Stream, exp: *const exl3e.Experts, c: Case, seed: u64) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*x| x.free();
        r.bufs.deinit(gpu.gpa);
    }
    var most: usize = 0;
    for (c.rows) |m| most = @max(most, m);
    const b = try build(&r, c.width, most);
    if (b.layer.act_mode != 2 or b.layer.limit != 7 or b.layer.alpha != 1.702 or b.layer.up_bias != 1) return error.NotSwigluoai;
    const sb = try r.zeros(try exl3e.scratchBytes(b.layer, cap_rows, SLOTS));
    const sc = try exl3e.scratch(sb, b.layer, cap_rows, SLOTS);
    const big = try r.zeros(most * D * 4);
    try exp.run(r.s, b.layer, b.dx, D, .bf16, b.dpick, b.dwts, most, big, sc);
    try stages(&r, &b, c.width, most, sc, big, c.id);
    // every smaller call's rows are the first rows of the largest call, byte for byte
    const ref = try r.back(big, most * D * 4);
    for (c.rows) |m| {
        if (m == most) continue;
        const out = try r.zeros(m * D * 4);
        try exp.run(r.s, b.layer, b.dx, D, .bf16, b.dpick, b.dwts, m, out, sc);
        const id = try std.fmt.allocPrint(r.a, "{s} rows={d} against the first rows of rows={d}", .{ c.id, m, most });
        try check.sameBytes(id, try r.back(out, m * D * 4), ref[0 .. m * D * 4]);
    }
}

pub fn run(gpu: check.Gpu, s: cuda.Stream, state: *Run) !void {
    var exp = try exl3e.Experts.load(gpu.d);
    defer exp.unload();
    for (cases, 0..) |c, i| {
        one(gpu, s, &exp, c, 0x6d6f65 + i) catch |e| {
            state.fail(e, c.id);
            continue;
        };
        check.pass("{s}", .{c.id});
    }
}
