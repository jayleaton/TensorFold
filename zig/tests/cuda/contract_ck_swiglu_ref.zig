//! The decode forms' own gates 1 and 3: float64 SwiGLU and down from the NVFP4 values, every NVFP4 block decoded.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const sw = @import("contract_ck_swiglu.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const uf32 = 0x1p-24; // fp32's unit roundoff
const ubf = 0x1p-8; // bf16's

/// silu in float64 with fp32's exp overflow: the kernel's expf(-x) is infinite past 88.72, which makes silu -0.
pub fn silu(x: f64) f64 {
    const e = if (-x > 88.72283905206835) std.math.inf(f64) else @exp(-x);
    return x / (1 + e);
}

/// tf_ck4_down_f32_skip against float64: every class, finite values within the spec's bound, skipped rows 0xff.
pub fn down(x: *sw.Experts, got: []const u8, at: g.Gate) !?[]const u8 {
    const a = x.rig.gpu.gpa;
    const n = x.shape.n;
    const w = try a.alloc(f64, x.shape.k);
    defer a.free(w);
    const s = sw.spec(x.kind, x.fault);
    var odd: usize = 0;
    var plain: usize = 0;
    for (0..sw.experts) |e| for (0..n) |j| {
        x.weights(e, j, w);
        for (x.of(e)) |p| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (x.row(p), w) |v, y| {
                sum += v * y;
                abs += @abs(v * y);
            }
            const want = sum * x.alpha[e];
            const have = c.read(.f32, got, p * n + j);
            if (c.classOf(have) != c.classOf(want)) return try x.say("pair {d} column {d} is {t} where float64 gives {t}", .{ p, j, c.classOf(have), c.classOf(want) });
            if (c.classOf(want) != .finite) {
                odd += 1;
                continue;
            }
            plain += 1;
            const allow = c.bound(s, x.shape, want, @abs(x.alpha[e]) * abs);
            if (!(@abs(have - want) <= allow)) return try x.say("pair {d} column {d} is {e:.6} where float64 gives {e:.6}: {d:.2} of its bound", .{ p, j, have, want, @abs(have - want) / allow });
        }
    };
    if (at == .nonfinite and (odd == 0 or plain == 0)) return try x.say("{d} non-finite and {d} finite routed outputs: the planted values prove nothing", .{ odd, plain });
    return x.untouched(got);
}

/// tf_ck4_swiglu_nvfp4 against float64 SwiGLU: each 16-output block decoded and bounded, skipped rows 0xff.
pub fn up(x: *sw.Experts, got: []const u8, at: g.Gate) !?[]const u8 {
    const a = x.rig.gpu.gpa;
    const no = x.shape.n / 2;
    const k = x.shape.k;
    const m = x.pairs() * no;
    const buf = try a.alloc(f64, 2 * k + 2 * m);
    defer a.free(buf);
    const wg = buf[0..k];
    const wu = buf[k .. 2 * k];
    const v = buf[2 * k ..][0..m];
    const err = buf[2 * k + m ..];
    const gam = c.gamma(@floatFromInt(sw.stepDepth(x.shape)), uf32);
    for (0..sw.experts) |e| for (0..no) |o| {
        x.weights(e, sw.gateCol(o), wg);
        x.weights(e, sw.gateCol(o) + 8, wu);
        const ag: f64 = x.alpha[2 * e];
        const au: f64 = x.alpha[2 * e + 1];
        for (x.of(e)) |p| {
            var s = [4]f64{ 0, 0, 0, 0 }; // gate's sum and term magnitude, then up's
            for (x.row(p), wg, wu) |r, q, t| {
                s[0] += r * q;
                s[1] += @abs(r * q);
                s[2] += r * t;
                s[3] += @abs(r * t);
            }
            v[p * no + o] = silu(s[0] * ag) * (s[2] * au);
            err[p * no + o] = slack(s[0] * ag, s[2] * au, @abs(ag) * gam * s[1], @abs(au) * gam * s[3]);
        }
    };
    var seen = [3]usize{ 0, 0, 0 }; // NaN, infinite and finite blocks
    for (0..sw.experts) |e| for (x.of(e)) |p| for (0..no / 16) |b| {
        const at_ = p * no + b * 16;
        if (try block(x, got, v[at_..][0..16], err[at_..][0..16], p, b, e, &seen)) |why| return why;
    };
    if (at == .nonfinite and (seen[0] + seen[1] == 0 or seen[2] == 0)) return try x.say("{d} NaN, {d} infinite and {d} finite routed blocks: the planted values prove nothing", .{ seen[0], seen[1], seen[2] });
    return x.untouched(got);
}

/// |kernel - float64| for one output before its NVFP4 rounding: the sums' fp32 error, then each bf16 and fp32 rounding.
fn slack(gate: f64, upv: f64, dsum_g: f64, dsum_u: f64) f64 {
    const r = ubf + 2 * uf32; // an fp32 product rounded to bf16
    const dg = dsum_g * (1 + 2 * uf32) + r * (@abs(gate) + dsum_g);
    const sv = @abs(silu(gate));
    const ds = 1.1 * dg + (ubf + 8 * uf32) * (sv + 1.1 * dg); // silu's slope is within 1.1; then fp32 exp and bf16
    const du = dsum_u * (1 + 2 * uf32) + r * (@abs(upv) + dsum_u);
    const au = @abs(upv);
    return 1.01 * (ds * au + sv * du + ds * du + r * (sv + ds) * (au + du));
}

/// One block: a NaN in float64 needs a NaN scale; infinities 448's scale and +-6 there; else codes within a step.
fn block(x: *sw.Experts, got: []const u8, v: []const f64, err: []const f64, p: usize, b: usize, e: usize, seen: *[3]usize) !?[]const u8 {
    const no = x.shape.n / 2;
    const sb = got[x.pairs() * no / 2 + p * no / 16 + b];
    const codes = got[p * no / 2 + b * 8 ..][0..8];
    var nan = false;
    var inf = false;
    var amax: f64 = 0;
    var emax: f64 = 0;
    for (v, err) |y, d| {
        if (std.math.isNan(y)) {
            nan = true;
        } else if (std.math.isInf(y)) {
            inf = true;
        } else {
            amax = @max(amax, @abs(y));
            emax = @max(emax, d);
        }
    }
    if (nan) {
        seen[0] += 1;
        if ((sb & 0x7f) == 0x7f) return null;
        return try x.say("pair {d} block {d} holds a NaN in float64 but its scale byte is 0x{x}", .{ p, b, sb });
    }
    const sv = core.quant.fp8.e4m3(sb);
    if (std.math.isNan(sv)) return try x.say("pair {d} block {d} is not NaN in float64 but its scale is", .{ p, b });
    if (inf) {
        seen[1] += 1;
        if (sb != 0x7e) return try x.say("pair {d} block {d} holds an infinity but its scale byte is 0x{x}, not 448's", .{ p, b, sb });
        for (v, 0..) |y, i| {
            if (std.math.isInf(y) and nibble(codes, i) != @as(u8, if (y < 0) 15 else 7)) return try x.say("pair {d} output {d} is {d} in float64 but codes {d}", .{ p, b * 16 + i, y, nibble(codes, i) });
        }
        return null;
    }
    seen[2] += 1;
    const ge: f64 = cuda.ck.inverse(x.act[e]);
    if (sv == 0) {
        for (0..16) |i| if ((nibble(codes, i) & 7) != 0) return try x.say("pair {d} output {d} codes {d} under a zero scale", .{ p, b * 16 + i, nibble(codes, i) });
        if (ge * (amax - emax) / 6 > 0x1p-10 * (1 + 0x1p-20)) return try x.say("pair {d} block {d} has a zero scale where float64's largest is {e:.4}", .{ p, b, amax });
        return null;
    }
    var most: f64 = 0;
    var room: f64 = 0;
    for (v, err, 0..) |y, d, i| {
        const q = y * ge / sv; // float64's value in units of the block's scale
        const dq = d * ge / sv + 0x1p-18;
        const m = core.quant.nvfp4.e2m1(nibble(codes, i));
        most = @max(most, @abs(q));
        room = @max(room, dq);
        const ok = if (@abs(m) == 6) q * std.math.sign(m) >= 5 - dq else @abs(q - m) <= step(@abs(m)) + dq;
        if (!ok) return try x.say("pair {d} output {d} codes {d} where float64 gives {d:.4} scales (within {d:.4})", .{ p, b * 16 + i, m, q, dq });
    }
    // inside e4m3's normal range the scale rounds within 1/16, so the block's largest lands in [5.625, 6.375]
    const level = ge * amax / 6;
    if (level >= 0x1p-6 and level <= 400 and !(most >= 5.62 - room and most <= 6.38 + room)) return try x.say("pair {d} block {d}'s largest is {d:.4} scales, not about 6", .{ p, b, most });
    return null;
}

/// The farthest an e2m1 code's inputs lie from it: half the larger gap beside it.
fn step(m: f64) f64 {
    return if (m < 2) 0.25 else if (m < 4) 0.5 else 1;
}

fn nibble(codes: []const u8, i: usize) u8 {
    const byte = codes[i / 2];
    return if (i % 2 == 0) byte & 0xf else byte >> 4;
}
