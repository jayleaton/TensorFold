//! MiniMax-M3.1-Flash float64 reference helpers (engine-port-plan.md section 2); every UNVERIFIED semantic is a Switch.
const std = @import("std");
const bf = @import("bf16_round.zig");
const attention = @import("attention_ref.zig");
const route = @import("route_ref.zig");

/// UNVERIFIED: RoPE pairing. neox_half rotates (i, i + rot/2), the plan's reading; gptj_interleaved (2i, 2i + 1).
pub const RopeLayout = enum { neox_half, gptj_interleaved };
/// UNVERIFIED: sparse_init_block n. count forces blocks [0, n) (0: none); inclusive forces [0, n] (0: block 0).
pub const InitBlocks = enum { count, inclusive };
/// UNVERIFIED: sparse_local_block n. own_block_if_positive forces the own block when n > 0; trailing_window the last n.
pub const LocalBlocks = enum { own_block_if_positive, trailing_window };
/// Which block wins among equal block scores when the top-k cuts through them; the plan defines lower_block.
pub const Tie = enum { lower_block, higher_block };
/// UNVERIFIED: router weights: divide by the picks' sum, then times routed_scaling_factor (or either alone).
pub const RouterNorm = enum { normalize_then_scale, scale_only, normalize_only };
/// UNVERIFIED: whether routed_scaling_factor multiplies the routed sum only or routed plus the shared expert.
pub const ScaleOn = enum { routed, routed_plus_shared };
/// UNVERIFIED: swigluoai's gate clamp; upper_only is min(g, limit), symmetric is clamp(g, -limit, limit).
pub const GateClamp = enum { upper_only, symmetric };

/// Every semantic not yet checked against upstream MiniMax code; defaults are the plan's unread REFERENCE reading.
pub const Switches = struct {
    rope_layout: RopeLayout = .neox_half,
    /// UNVERIFIED: the index query and key get the main rotary (vLLM shares it); false leaves them unrotated.
    index_rope: bool = true,
    sparse_init_block: usize = 0,
    init_mode: InitBlocks = .count,
    sparse_local_block: usize = 1,
    local_mode: LocalBlocks = .own_block_if_positive,
    tie: Tie = .lower_block,
    /// UNVERIFIED: w1 is the swish (gate) side of a routed expert and w3 the linear (up) side; false swaps them.
    w1_is_gate: bool = true,
    gate_clamp: GateClamp = .upper_only,
    /// UNVERIFIED: swigluoai multiplies (u + 1); false multiplies u.
    up_plus_one: bool = true,
    router_norm: RouterNorm = .normalize_then_scale,
    /// UNVERIFIED: the picked weights are sigmoid(logit) without the bias; true takes sigmoid(logit) + bias.
    router_weights_use_bias: bool = false,
    scale_on: ScaleOn = .routed,
    /// UNVERIFIED: the (1 + w) Gemma form for the layer norms, the q/k norms and the index norms (false: plain w).
    gemma_layer_norms: bool = true,
    gemma_qk_norms: bool = true,
    gemma_index_norms: bool = true,
    /// UNVERIFIED rounding points: each true rounds that value to bf16 (RNE); all false is pure float64.
    bf16_norm_out: bool = false,
    bf16_proj_out: bool = false,
    bf16_rope_out: bool = false,
    bf16_attn_out: bool = false,
    bf16_act_out: bool = false,
    bf16_residual: bool = false,
    /// UNVERIFIED: the bf16 activation is cast to fp16 before the fp16 index projections.
    fp16_index_input: bool = false,
};

pub fn round16(on: bool, v: f64) f64 {
    return if (on) bf.bf16(v) else v;
}

pub fn roundAll(on: bool, x: []f64) void {
    if (on) for (x) |*v| {
        v.* = bf.bf16(v.*);
    };
}

pub fn roundFp16(on: bool, x: []f64) void {
    if (on) for (x) |*v| {
        const h: f16 = @floatCast(v.*);
        v.* = h;
    };
}

pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// out = x / sqrt(mean(x^2) + eps) * (w + 1 if `offset` else w): Gemma RMSNorm, one row.
pub fn gemmaNorm(x: []const f64, w: []const f64, eps: f64, offset: bool, out: []f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
    const one: f64 = if (offset) 1 else 0;
    for (out, x, w) |*o, v, ww| o.* = v * inv * (one + ww);
}

/// RoPE on the first `rot` entries of x at `pos`, inv_freq[i] = theta^(-2i/rot); the rest of x is untouched.
pub fn rope(x: []f64, rot: usize, pos: usize, theta: f64, layout: RopeLayout) void {
    const half = rot / 2;
    for (0..half) |i| {
        const ang = @as(f64, @floatFromInt(pos)) * std.math.pow(f64, theta, -2 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(rot)));
        const c = @cos(ang);
        const s = @sin(ang);
        const ia: usize = if (layout == .neox_half) i else 2 * i;
        const ib: usize = if (layout == .neox_half) i + half else 2 * i + 1;
        const a = x[ia];
        const b = x[ib];
        x[ia] = a * c - b * s;
        x[ib] = b * c + a * s;
    }
}

/// swigluoai(g, u): gate clamped above (or both sides) at `limit`, u clamped to +-limit, g sigmoid(alpha g) (u + 1).
pub fn swigluoai(g: f64, u: f64, alpha: f64, limit: f64, sw: Switches) f64 {
    if (std.math.isNan(g) or std.math.isNan(u)) return std.math.nan(f64);
    const gc = if (sw.gate_clamp == .symmetric) std.math.clamp(g, -limit, limit) else @min(g, limit);
    const uc = std.math.clamp(u, -limit, limit);
    return gc * sigmoid(alpha * gc) * (uc + if (sw.up_plus_one) @as(f64, 1) else 0);
}

/// The `ids.len` largest scores, largest first; equal scores go to the lower index (the higher with Tie.higher_block).
pub fn topk(scores: []const f64, ids: []usize, tie: Tie) void {
    for (ids, 0..) |*id, k| {
        var best: ?usize = null;
        var bv: f64 = 0;
        for (scores, 0..) |s, i| {
            if (std.mem.indexOfScalar(usize, ids[0..k], i) != null) continue;
            const v = if (std.math.isNan(s)) -std.math.inf(f64) else s;
            if (best == null or v > bv or (v == bv and tie == .higher_block)) {
                best = i;
                bv = v;
            }
        }
        id.* = best.?;
    }
}

/// Block scores of one index head: max over the causal keys j <= pos of each block of (iq . ik_j) * scale; returns nb.
pub fn blockScores(iq: []const f64, ik: []const f64, pos: usize, d: usize, scale: f64, block: usize, out: []f64) usize {
    const nb = pos / block + 1;
    for (out[0..nb], 0..) |*o, b| {
        o.* = -std.math.inf(f64);
        for (b * block..@min((b + 1) * block, pos + 1)) |j| {
            var s: f64 = 0;
            for (0..d) |e| s += iq[e] * ik[j * d + e];
            o.* = @max(o.*, s * scale);
        }
    }
    return nb;
}

/// Whether block b is forced for a query in block `own`, from sparse_init_block and sparse_local_block.
pub fn forced(b: usize, own: usize, sw: Switches) bool {
    const init = switch (sw.init_mode) {
        .count => b < sw.sparse_init_block,
        .inclusive => b <= sw.sparse_init_block,
    };
    const local = switch (sw.local_mode) {
        .own_block_if_positive => sw.sparse_local_block > 0 and b == own,
        .trailing_window => b <= own and own - b < sw.sparse_local_block,
    };
    return init or local;
}

/// The blocks a query attends, ascending, into ids, and how many: all when nb <= k, else the top k with forced at +inf.
pub fn selectBlocks(bs: []const f64, own: usize, k: usize, sw: Switches, scratch: []f64, ids: []usize) usize {
    const nb = bs.len;
    if (nb <= k) {
        for (ids[0..nb], 0..) |*id, b| id.* = b;
        return nb;
    }
    for (scratch[0..nb], bs, 0..) |*s, v, b| s.* = if (forced(b, own, sw)) std.math.inf(f64) else v;
    topk(scratch[0..nb], ids[0..k], sw.tie);
    std.mem.sort(usize, ids[0..k], {}, std.sort.asc(usize));
    return k;
}

/// Attention of `rep` query heads (rep x d) over the keys of blocks `ids` up to `pos` in one kv head's k and v (n x d).
pub fn attendBlocks(q: []const f64, k: []const f64, v: []const f64, ids: []const usize, pos: usize, block: usize, d: usize, rep: usize, keys: []usize, out: []f64) void {
    var n: usize = 0;
    for (ids) |b| for (b * block..@min((b + 1) * block, pos + 1)) |j| {
        keys[n] = j;
        n += 1;
    };
    var tol: [512]f64 = undefined;
    const scale = std.math.pow(f64, @floatFromInt(d), -0.5);
    for (0..rep) |h| attention.head(q[h * d ..][0..d], k, v, keys[0..n], d, scale, out[h * d ..][0..d], tol[0..d]);
}

/// A router's picks by sigmoid(logit) + bias (lower id on ties) and their weights per Switches, `scale` applied here.
pub fn routerPick(logits: []const f64, bias: []const f64, scale: f64, sw: Switches, scratch: []f64, ids: []usize, weights: []f64) void {
    for (scratch[0..logits.len], logits, bias) |*s, l, b| s.* = sigmoid(l) + b;
    var unused: [64]f64 = undefined;
    route.topk(scratch[0..logits.len], ids, unused[0..ids.len]);
    var sum: f64 = 0;
    for (ids, weights) |id, *w| {
        w.* = if (sw.router_weights_use_bias) scratch[id] else sigmoid(logits[id]);
        sum += w.*;
    }
    const norm = sw.router_norm != .scale_only;
    const factor = if (sw.router_norm == .normalize_only or sw.scale_on == .routed_plus_shared) 1 else scale;
    for (weights) |*w| w.* = (if (norm) w.* / sum else w.*) * factor;
}

/// What the combined routed + shared output is multiplied by after the add (1 unless scale_on is routed_plus_shared).
pub fn totalScale(sw: Switches, scale: f64) f64 {
    return if (sw.scale_on == .routed_plus_shared and sw.router_norm != .normalize_only) scale else 1;
}

// fp8 KV cache (docs/fp8-kv.md): a 128-value row of K or V is e4m3 codes and one fp32 scale amax / 448 (floor 2^-126).

// A NaN or infinity in a row gives scale NaN and zero codes; the fp32 form multiplies by the fp32 inverse, as the GPU.

pub const kv8_max: f64 = 448;
pub const kv8_min_scale: f64 = 0x1p-126;

/// The value of e4m3 code `b` (0x7f and 0xff are NaN).
pub fn e4m3Value(b: u8) f64 {
    if (b & 0x7f == 0x7f) return std.math.nan(f64);
    const e: i32 = (b >> 3) & 0xf;
    const m: f64 = @floatFromInt(b & 7);
    const mag = if (e == 0) m / 8.0 * 0x1p-6 else (1 + m / 8.0) * std.math.pow(f64, 2, @floatFromInt(e - 7));
    return if (b & 0x80 != 0) -mag else mag;
}

/// The nearest e4m3 code of x, ties to the even code, saturating at +-448 (cvt.rn.satfinite); NaN is 0x7f.
pub fn e4m3Code(x: f64) u8 {
    if (std.math.isNan(x)) return 0x7f;
    const sign: u8 = if (std.math.signbit(x)) 0x80 else 0;
    const a = @abs(x);
    if (a >= 464) return sign | 0x7e;
    var i: u8 = 0;
    while (i < 0x7e and e4m3Value(i + 1) <= a) i += 1;
    if (i == 0x7e) return sign | i;
    const lo = e4m3Value(i);
    const hi = e4m3Value(i + 1);
    const mid = (lo + hi) / 2;
    const up = a > mid or (a == mid and i % 2 == 1);
    return sign | (if (up) i + 1 else i);
}

/// Float64 reference quantization of a row: codes and the scale (a non-finite value gives scale NaN, zero codes).
pub fn quantRow(x: []const f64, codes: []u8) f64 {
    var amax: f64 = 0;
    var bad = false;
    for (x) |v| {
        if (!std.math.isFinite(v)) bad = true else amax = @max(amax, @abs(v));
    }
    if (bad) {
        @memset(codes, 0);
        return std.math.nan(f64);
    }
    const scale = if (amax > 0) @max(amax / kv8_max, kv8_min_scale) else 1;
    for (codes, x) |*c, v| c.* = e4m3Code(v / scale);
    return scale;
}

/// code * scale for each code; a NaN scale gives NaN throughout.
pub fn dequantRow(codes: []const u8, scale: f64, out: []f64) void {
    for (out, codes) |*o, c| o.* = e4m3Value(c) * scale;
}

/// The kernel's fp32 form of `quantRow`: inv = 448 / amax and y = x * inv in fp32, then exact e4m3 rounding.
pub fn quantRowF32(x: []const f32, codes: []u8) f32 {
    var amax: f32 = 0;
    var bad = false;
    for (x) |v| {
        if (!std.math.isFinite(v)) bad = true else amax = @max(amax, @abs(v));
    }
    if (bad) {
        @memset(codes, 0);
        return std.math.nan(f32);
    }
    var inv: f32 = 1;
    var scale: f32 = 1;
    if (amax > 0) {
        inv = 448.0 / amax;
        scale = amax / 448.0;
        if (scale < 0x1p-126) {
            scale = 0x1p-126;
            inv = 1.0 / scale;
        }
    }
    for (codes, x) |*c, v| c.* = e4m3Code(@as(f64, v * inv));
    return scale;
}

/// attendBlocks over an fp8 cache: each key and value row is quantized, dequantized in float64 first (kq, vq scratch).
pub fn attendBlocksKv8(q: []const f64, k: []const f64, v: []const f64, ids: []const usize, pos: usize, block: usize, d: usize, rep: usize, keys: []usize, kq: []f64, vq: []f64, out: []f64) void {
    var codes: [512]u8 = undefined;
    const rows = @min(k.len / d, pos + 1);
    for (0..rows) |j| {
        dequantRow(codes[0..d], quantRow(k[j * d ..][0..d], codes[0..d]), kq[j * d ..][0..d]);
        dequantRow(codes[0..d], quantRow(v[j * d ..][0..d], codes[0..d]), vq[j * d ..][0..d]);
    }
    attendBlocks(q, kq, vq, ids, pos, block, d, rep, keys, out);
}
