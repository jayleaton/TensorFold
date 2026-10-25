//! The dspark draft on the CPU in float64: mxfp8 decode, taps, context state, a draft block, Markov chain, confidence.
const std = @import("std");
// The upstream modeling code was unavailable: every semantic not forced by a shape is a `Switches` field, UNVERIFIED.
const core = @import("core");
// Defaults follow what TensorFold ships (deepseek_v41 DSpark chain, qwen3_5 DFlash2 taps, fc and K/V ring).
const mr = core.minimax_ref;

// ---- mxfp8 ------------------------------------------------------------------------------------------------------

/// float8 e4m3 (OCP "FN" form, bias 7, no infinities): 0x7F and 0xFF are NaN, exponent 0 is subnormal (m / 8 * 2^-6).
pub fn e4m3(code: u8) f64 {
    const sign: f64 = if (code & 0x80 != 0) -1 else 1;
    const e: i32 = (code >> 3) & 0xF;
    const m: f64 = @floatFromInt(code & 7);
    if (e == 0xF and (code & 7) == 7) return std.math.nan(f64);
    if (e == 0) return sign * m * 0x1p-9;
    return sign * (1 + m / 8) * std.math.ldexp(@as(f64, 1), e - 7);
}

/// A ue8m0 scale byte: 2^(byte - bias), 255 is NaN. UNVERIFIED bias 127 (k_proj bytes sit at 114-118).
pub fn ue8m0(b: u8, bias: i32) f64 {
    if (b == 255) return std.math.nan(f64);
    return std.math.ldexp(@as(f64, 1), @as(i32, b) - bias);
}

/// w[j][i] = e4m3(codes[j][i]) * ue8m0(scales[j][i / block]) for a [n, k] matrix.
pub fn dequant(codes: []const u8, scales: []const u8, n: usize, k: usize, block: usize, bias: i32, out: []f64) void {
    for (0..n) |j| for (0..k) |i| {
        out[j * k + i] = e4m3(codes[j * k + i]) * ue8m0(scales[j * (k / block) + i / block], bias);
    };
}

// ---- switches ---------------------------------------------------------------------------------------------------

/// UNVERIFIED: which residual a tap of layer `id` reads: its output (HF hidden_states[id + 1]) or its input.
pub const TapSite = enum { layer_output, layer_input };
/// UNVERIFIED: fc's input is the taps side by side in config order; reversed is the planted alternative.
pub const ConcatOrder = enum { listed, reversed };
/// UNVERIFIED: sliding_window: per_row (seen iff rp - kp < window) or context_tail (the last `window` context rows).
pub const WindowRule = enum { per_row, context_tail };
/// UNVERIFIED: rows of the draft block see each other (DFlash/DSpark blocks are parallel) or only earlier rows.
pub const BlockAttention = enum { bidirectional, causal };
/// UNVERIFIED: block row i sits at position P + i (P: the pending token's), or every row at P.
pub const BlockPositions = enum { consecutive, all_pending };
/// UNVERIFIED: next_token_all_rows (row i gives the token at P + 1 + i, `block` drafts) or rows 1.. (DFlash).
pub const DraftRows = enum { next_token_all_rows, rows_after_anchor };
/// UNVERIFIED: the Markov bias is added over the whole vocabulary or only to the top `candidates` of the base logits.
pub const MarkovDomain = enum { full_vocab, candidates };
/// UNVERIFIED: the first draft's previous token is the anchor (pending token) or none (no bias, zero embedding).
pub const FirstPrev = enum { anchor, none };
/// UNVERIFIED: the confidence head reads the block's hidden row after final_norm or before it.
pub const ConfInput = enum { post_final_norm, pre_final_norm };
/// UNVERIFIED: how per-draft confidences cut a block: the running product, or each one alone.
pub const ConfGate = enum { product, each };

pub const Switches = struct {
    /// Target-side switches the draft shares (rope layout, swigluoai, Gemma offsets, bf16 points): mr.Switches.
    base: mr.Switches = .{},
    tap_site: TapSite = .layer_output,
    /// UNVERIFIED: the tap of the last layer is read after the model's final norm (HF's last hidden_states entry is).
    last_tap_post_norm: bool = false,
    /// UNVERIFIED: the taps reach fc as bf16 (the engine would store them so).
    tap_bf16: bool = true,
    concat_order: ConcatOrder = .listed,
    /// UNVERIFIED: hidden_norm and final_norm use 1 + w (hidden_norm's mean is 0.05, final_norm's 0.41).
    gemma_hidden_norm: bool = true,
    gemma_final_norm: bool = true,
    /// UNVERIFIED: the context state passes each layer's input_layernorm before that layer's k_proj and v_proj.
    ctx_input_norm: bool = false,
    window_rule: WindowRule = .per_row,
    block_attention: BlockAttention = .bidirectional,
    block_positions: BlockPositions = .consecutive,
    draft_rows: DraftRows = .next_token_all_rows,
    markov_domain: MarkovDomain = .full_vocab,
    /// UNVERIFIED: bias(prev, id) = markov_w2[id] . markov_w1[prev]; true swaps the two tables.
    markov_swap: bool = false,
    first_prev: FirstPrev = .anchor,
    conf_input: ConfInput = .post_final_norm,
    /// UNVERIFIED: the confidence is sigmoid(w . [h ; markov_w1[prev]] + b); false returns the logit.
    conf_sigmoid: bool = true,
    conf_gate: ConfGate = .product,
    /// UNVERIFIED: the auxiliary GEMMs (head logits, Markov, confidence) take bf16 inputs (dspark_aux_gemm_precision).
    aux_bf16: bool = true,
    /// UNVERIFIED: ue8m0 exponent bias.
    scale_bias: i32 = 127,
};

/// The draft's draft count and the row giving draft i under the switch.
pub fn draftCount(sw: Switches, block: usize) usize {
    return switch (sw.draft_rows) {
        .next_token_all_rows => block,
        .rows_after_anchor => block - 1,
    };
}

pub fn rowFor(sw: Switches, i: usize) usize {
    return switch (sw.draft_rows) {
        .next_token_all_rows => i,
        .rows_after_anchor => i + 1,
    };
}

pub const Config = struct {
    hidden: usize = 6144,
    n_q: usize = 64,
    n_kv: usize = 4,
    head_dim: usize = 128,
    rot_dim: usize = 64,
    rope_theta: f64 = 1e7,
    eps: f64 = 1e-6,
    ffn: usize = 12288,
    layers: usize = 5,
    n_taps: usize = 5,
    window: usize = 4096,
    block: usize = 7,
    mask_token: usize = 200058,
    rank: usize = 256,
    vocab: usize = 200064,
    candidates: usize = 64,
    alpha: f64 = 1.702,
    limit: f64 = 7.0,
    sw: Switches = .{},

    pub fn rep(c: Config) usize {
        return c.n_q / c.n_kv;
    }
    pub fn kvw(c: Config) usize {
        return c.n_kv * c.head_dim;
    }
};

/// y = W x for W stored [n][k] row-major, the checkpoint's layout (the target's ref.Linear is [k][n]).
pub const Dense = struct {
    n: usize,
    k: usize,
    w: []const f64,

    pub fn apply(d: Dense, x: []const f64, out: []f64) void {
        d.applyRows(0, d.n, x, out);
    }

    /// Output rows [lo, hi) only.
    pub fn applyRows(d: Dense, lo: usize, hi: usize, x: []const f64, out: []f64) void {
        for (lo..hi) |j| {
            var s: f64 = 0;
            for (x[0..d.k], d.w[j * d.k ..][0..d.k]) |a, b| s += a * b;
            out[j - lo] = s;
        }
    }

    /// Inputs [lo, lo + x.len) only, all output rows: the partial sum of a row-cut layer.
    pub fn applyCols(d: Dense, lo: usize, x: []const f64, out: []f64) void {
        for (0..d.n) |j| {
            var s: f64 = 0;
            for (x, d.w[j * d.k + lo ..][0..x.len]) |a, b| s += a * b;
            out[j] = s;
        }
    }
};

pub const Layer = struct {
    input_norm: []const f64,
    post_norm: []const f64,
    q_norm: []const f64,
    k_norm: []const f64,
    q: Dense,
    k: Dense,
    v: Dense,
    o: Dense,
    gate: Dense,
    up: Dense,
    down: Dense,
};

pub const Draft = struct {
    cfg: Config,
    fc: Dense, // [hidden, n_taps x hidden]
    hidden_norm: []const f64,
    final_norm: []const f64,
    layers: []const Layer,
    markov_w1: []const f64, // [vocab][rank]
    markov_w2: []const f64,
    conf_w: []const f64, // [hidden + rank]
    conf_b: f64,
    /// The TARGET's embedding [vocab][hidden] and lm_head [vocab, hidden]: the draft has neither (UNVERIFIED reuse).
    embed: []const f64,
    head: Dense,
};

// ---- small pieces -----------------------------------------------------------------------------------------------

pub fn norm(c: Config, x: []const f64, w: []const f64, gemma: bool, out: []f64) void {
    mr.gemmaNorm(x, w, c.eps, gemma, out);
    mr.roundAll(c.sw.base.bf16_norm_out, out);
}

/// Per-head norm then RoPE at `pos` (null: none) over `n` heads of x.
pub fn heads(c: Config, x: []f64, n: usize, w: []const f64, pos: ?usize) void {
    const d = c.head_dim;
    var t: [256]f64 = undefined;
    for (0..n) |h| {
        const s = x[h * d ..][0..d];
        norm(c, s, w, c.sw.base.gemma_qk_norms, t[0..d]);
        @memcpy(s, t[0..d]);
        if (pos) |p| mr.rope(s, c.rot_dim, p, c.rope_theta, c.sw.base.rope_layout);
        mr.roundAll(c.sw.base.bf16_rope_out, s);
    }
}

/// A target residual a tap reads: index r (0 = embeddings, r = output of layer r - 1), and whether final norm applies.
pub const TapSource = struct { r: usize, normed: bool };

pub fn tapSource(sw: Switches, id: usize, target_layers: usize) TapSource {
    const r = switch (sw.tap_site) {
        .layer_output => id + 1,
        .layer_input => id,
    };
    return .{ .r = r, .normed = sw.last_tap_post_norm and r == target_layers };
}

/// One token's tapped row (n_taps x hidden, config order) from the target's residuals; bf16 rounding: contextState.
pub fn tapRow(c: Config, ids: []const usize, target_layers: usize, resid: []const []const f64, target_final_norm: []const f64, target_gemma: bool, out: []f64) void {
    for (ids, 0..) |id, t| {
        const src = tapSource(c.sw, id, target_layers);
        const dst = out[t * c.hidden ..][0..c.hidden];
        if (src.normed) mr.gemmaNorm(resid[src.r], target_final_norm, c.eps, target_gemma, dst) else @memcpy(dst, resid[src.r][0..c.hidden]);
    }
}

// ---- the context --------------------------------------------------------------------------------------------------

/// The context state of one token: hidden_norm(fc(taps)). `taps` is n_taps x hidden in config order.
pub fn contextState(d: *const Draft, a: std.mem.Allocator, taps: []const f64, out: []f64) !void {
    const c = d.cfg;
    const in = try a.alloc(f64, c.n_taps * c.hidden);
    defer a.free(in);
    for (0..c.n_taps) |t| {
        const src = if (c.sw.concat_order == .listed) t else c.n_taps - 1 - t;
        @memcpy(in[t * c.hidden ..][0..c.hidden], taps[src * c.hidden ..][0..c.hidden]);
    }
    mr.roundAll(c.sw.tap_bf16, in);
    const y = try a.alloc(f64, c.hidden);
    defer a.free(y);
    d.fc.apply(in, y);
    mr.roundAll(c.sw.base.bf16_proj_out, y);
    norm(c, y, d.hidden_norm, c.sw.gemma_hidden_norm, out);
}

/// Layer `l`'s K (normed, rotated at `pos`) and V of one context token, all kv heads: [n_kv x head_dim] each.
pub fn contextKv(d: *const Draft, a: std.mem.Allocator, l: usize, ctx_state: []const f64, pos: usize, k: []f64, v: []f64) !void {
    const c = d.cfg;
    const ly = d.layers[l];
    var x = ctx_state;
    var tmp: ?[]f64 = null;
    defer if (tmp) |t| a.free(t);
    if (c.sw.ctx_input_norm) {
        tmp = try a.alloc(f64, c.hidden);
        norm(c, ctx_state, ly.input_norm, c.sw.base.gemma_layer_norms, tmp.?);
        x = tmp.?;
    }
    ly.k.apply(x, k);
    ly.v.apply(x, v);
    mr.roundAll(c.sw.base.bf16_proj_out, k);
    mr.roundAll(c.sw.base.bf16_proj_out, v);
    heads(c, k, c.n_kv, ly.k_norm, pos);
}

/// The context K/V of positions first .. first + n - 1, every layer: `k[l]` and `v[l]` are [n][kvw].
pub const Ctx = struct {
    first: usize,
    n: usize,
    k: [][]f64,
    v: [][]f64,

    pub fn alloc(a: std.mem.Allocator, c: Config, first: usize, n: usize) !Ctx {
        const k = try a.alloc([]f64, c.layers);
        const v = try a.alloc([]f64, c.layers);
        for (k, v) |*kk, *vv| {
            kk.* = try a.alloc(f64, n * c.kvw());
            vv.* = try a.alloc(f64, n * c.kvw());
        }
        return .{ .first = first, .n = n, .k = k, .v = v };
    }
};

/// The reference context: every row recomputed from its tapped features (`feats`: [n][n_taps x hidden]).
pub fn buildCtx(d: *const Draft, a: std.mem.Allocator, feats: []const f64, first: usize, n: usize) !Ctx {
    const c = d.cfg;
    const cx = try Ctx.alloc(a, c, first, n);
    const st = try a.alloc(f64, c.hidden);
    defer a.free(st);
    for (0..n) |j| {
        try contextState(d, a, feats[j * c.n_taps * c.hidden ..][0 .. c.n_taps * c.hidden], st);
        for (0..c.layers) |l| try contextKv(d, a, l, st, first + j, cx.k[l][j * c.kvw() ..][0..c.kvw()], cx.v[l][j * c.kvw() ..][0..c.kvw()]);
    }
    return cx;
}

// ---- one draft block ----------------------------------------------------------------------------------------------

/// The block's hidden rows [block][hidden]: `pre` before final_norm, `post` after it.
pub const Block = struct { pre: []f64, post: []f64 };

/// Whether a key at position `kp` is seen by the row at `rp` (the pending token sits at `p`).
pub fn sees(sw: Switches, window: usize, kp: usize, rp: usize, p: usize) bool {
    return switch (sw.window_rule) {
        .per_row => kp <= rp and rp - kp < window,
        .context_tail => kp + window >= p,
    };
}

/// The block for the pending token `anchor` at position `p`, over the context `cx` (positions first .. p - 1 at most).
pub fn blockForward(d: *const Draft, a: std.mem.Allocator, cx: *const Ctx, anchor: usize, p: usize) !Block {
    const c = d.cfg;
    const sw = c.sw;
    const hd = c.head_dim;
    const kvw = c.kvw();
    const nb = c.block;
    const rep = c.rep();
    if (cx.first + cx.n > p) return error.ContextPastPending;
    const h = try a.alloc(f64, nb * c.hidden);
    for (0..nb) |i| {
        const tok = if (i == 0) anchor else c.mask_token;
        @memcpy(h[i * c.hidden ..][0..c.hidden], d.embed[tok * c.hidden ..][0..c.hidden]);
    }
    const x = try a.alloc(f64, c.hidden);
    defer a.free(x);
    const scale = std.math.pow(f64, @floatFromInt(hd), -0.5);
    for (d.layers, 0..) |ly, l| {
        const q = try a.alloc(f64, nb * c.n_q * hd);
        defer a.free(q);
        const kb = try a.alloc(f64, nb * kvw);
        defer a.free(kb);
        const vb = try a.alloc(f64, nb * kvw);
        defer a.free(vb);
        for (0..nb) |i| {
            norm(c, h[i * c.hidden ..][0..c.hidden], ly.input_norm, sw.base.gemma_layer_norms, x);
            const qi = q[i * c.n_q * hd ..][0 .. c.n_q * hd];
            const ki = kb[i * kvw ..][0..kvw];
            const vi = vb[i * kvw ..][0..kvw];
            ly.q.apply(x, qi);
            ly.k.apply(x, ki);
            ly.v.apply(x, vi);
            for ([_][]f64{ qi, ki, vi }) |t| mr.roundAll(sw.base.bf16_proj_out, t);
            const pos = switch (sw.block_positions) {
                .consecutive => p + i,
                .all_pending => p,
            };
            heads(c, qi, c.n_q, ly.q_norm, pos);
            heads(c, ki, c.n_kv, ly.k_norm, pos);
        }
        const attn = try a.alloc(f64, nb * c.n_q * hd);
        defer a.free(attn);
        for (0..nb) |i| {
            const rp = switch (sw.block_positions) {
                .consecutive => p + i,
                .all_pending => p,
            };
            for (0..c.n_q) |hh| {
                const g = hh / rep;
                const qh = q[(i * c.n_q + hh) * hd ..][0..hd];
                const total = cx.n + nb;
                const s = try a.alloc(f64, total);
                defer a.free(s);
                var best: f64 = -std.math.inf(f64);
                for (0..total) |j| {
                    s[j] = -std.math.inf(f64);
                    var key: []const f64 = undefined;
                    if (j < cx.n) {
                        if (!sees(sw, c.window, cx.first + j, rp, p)) continue;
                        key = cx.k[l][j * kvw + g * hd ..][0..hd];
                    } else {
                        const m = j - cx.n;
                        if (sw.block_attention == .causal and m > i) continue;
                        key = kb[m * kvw + g * hd ..][0..hd];
                    }
                    var dot: f64 = 0;
                    for (qh, key) |u, w| dot += u * w;
                    s[j] = dot * scale;
                    best = @max(best, s[j]);
                }
                var z: f64 = 0;
                const out = attn[(i * c.n_q + hh) * hd ..][0..hd];
                @memset(out, 0);
                for (0..total) |j| {
                    if (s[j] == -std.math.inf(f64)) continue;
                    const w = @exp(s[j] - best);
                    z += w;
                    const val = if (j < cx.n) cx.v[l][j * kvw + g * hd ..][0..hd] else vb[(j - cx.n) * kvw + g * hd ..][0..hd];
                    for (out, val) |*o, vv| o.* += w * vv;
                }
                for (out) |*o| o.* /= z;
            }
        }
        mr.roundAll(sw.base.bf16_attn_out, attn);
        const o = try a.alloc(f64, c.hidden);
        defer a.free(o);
        const g = try a.alloc(f64, c.ffn);
        defer a.free(g);
        const u = try a.alloc(f64, c.ffn);
        defer a.free(u);
        const f = try a.alloc(f64, c.hidden);
        defer a.free(f);
        for (0..nb) |i| {
            const hi = h[i * c.hidden ..][0..c.hidden];
            ly.o.apply(attn[i * c.n_q * hd ..][0 .. c.n_q * hd], o);
            for (hi, o) |*hv, ov| hv.* = mr.round16(sw.base.bf16_residual, hv.* + ov);
            norm(c, hi, ly.post_norm, sw.base.gemma_layer_norms, x);
            ly.gate.apply(x, g);
            ly.up.apply(x, u);
            for (g, u) |*gv, uv| gv.* = mr.round16(sw.base.bf16_act_out, mr.swigluoai(gv.*, uv, c.alpha, c.limit, sw.base));
            ly.down.apply(g, f);
            for (hi, f) |*hv, fv| hv.* = mr.round16(sw.base.bf16_residual, hv.* + fv);
        }
    }
    const post = try a.alloc(f64, nb * c.hidden);
    for (0..nb) |i| norm(c, h[i * c.hidden ..][0..c.hidden], d.final_norm, sw.gemma_final_norm, post[i * c.hidden ..][0..c.hidden]);
    return .{ .pre = h, .post = post };
}

// ---- heads ------------------------------------------------------------------------------------------------------

fn auxRound(c: Config, x: []f64) void {
    mr.roundAll(c.sw.aux_bf16, x);
}

/// The target's head applied to a (final-normed) draft row: logits over the vocabulary.
pub fn baseLogits(d: *const Draft, a: std.mem.Allocator, row: []const f64, out: []f64) !void {
    const x = try a.dupe(f64, row);
    defer a.free(x);
    auxRound(d.cfg, x);
    d.head.apply(x, out);
}

/// Markov bias of candidate `id` after `prev`: the two tables' rows dotted (vanilla head, no nonlinearity).
pub fn markovBias(d: *const Draft, prev: usize, id: usize) f64 {
    const r = d.cfg.rank;
    const a1 = if (d.cfg.sw.markov_swap) d.markov_w2 else d.markov_w1;
    const a2 = if (d.cfg.sw.markov_swap) d.markov_w1 else d.markov_w2;
    var s: f64 = 0;
    for (a1[prev * r ..][0..r], a2[id * r ..][0..r]) |x, y| s += x * y;
    return s;
}

/// The confidence of the draft that follows `prev` (null: the first draft with first_prev = none) from the block's row.
pub fn confidence(d: *const Draft, a: std.mem.Allocator, pre: []const f64, post: []const f64, prev: ?usize) !f64 {
    const c = d.cfg;
    const row = try a.dupe(f64, if (c.sw.conf_input == .post_final_norm) post else pre);
    defer a.free(row);
    auxRound(c, row);
    var s: f64 = d.conf_b;
    for (row, d.conf_w[0..c.hidden]) |x, w| s += x * w;
    if (prev) |pv| for (d.markov_w1[pv * c.rank ..][0..c.rank], d.conf_w[c.hidden..][0..c.rank]) |x, w| {
        s += x * w;
    };
    return if (c.sw.conf_sigmoid) mr.sigmoid(s) else s;
}

/// One pass's chain: draft i = argmax of base_i[id] + bias(prev_i, id), ties to the lower id; prev_0 is the anchor.
pub fn chain(d: *const Draft, a: std.mem.Allocator, b: Block, anchor: usize, drafts: []usize, conf: []f64) !void {
    const c = d.cfg;
    std.debug.assert(drafts.len == draftCount(c.sw, c.block) and conf.len == drafts.len);
    const base = try a.alloc(f64, c.vocab);
    defer a.free(base);
    const allowed = try a.alloc(bool, c.vocab);
    defer a.free(allowed);
    var prev: ?usize = if (c.sw.first_prev == .anchor) anchor else null;
    for (drafts, conf, 0..) |*t, *q, i| {
        const row = rowFor(c.sw, i);
        try baseLogits(d, a, b.post[row * c.hidden ..][0..c.hidden], base);
        @memset(allowed, c.sw.markov_domain == .full_vocab);
        if (c.sw.markov_domain == .candidates) {
            for (0..c.candidates) |_| {
                var top: ?usize = null;
                for (0..c.vocab) |id| {
                    if (allowed[id]) continue;
                    if (top == null or base[id] > base[top.?]) top = id;
                }
                allowed[top.?] = true;
            }
        }
        var best: ?usize = null;
        var bz: f64 = 0;
        for (0..c.vocab) |id| {
            if (!allowed[id]) continue;
            const z = base[id] + if (prev) |pv| markovBias(d, pv, id) else 0;
            if (best == null or z > bz) {
                best = id;
                bz = z;
            }
        }
        q.* = try confidence(d, a, b.pre[row * c.hidden ..][0..c.hidden], b.post[row * c.hidden ..][0..c.hidden], prev);
        t.* = best.?;
        prev = t.*;
    }
}

/// Drafts kept before the first whose confidence (running product, or alone) falls under `threshold`; at least `least`.
pub fn gateLen(sw: Switches, conf: []const f64, threshold: f64, least: usize) usize {
    var running: f64 = 1;
    var n: usize = 0;
    for (conf) |q| {
        running *= q;
        const v = if (sw.conf_gate == .product) running else q;
        if (v < threshold and n >= least) break;
        n += 1;
    }
    return n;
}

test "e4m3 and ue8m0 against hand values" {
    try std.testing.expectEqual(@as(f64, 1), e4m3(0x38));
    try std.testing.expectEqual(@as(f64, -1), e4m3(0xB8));
    try std.testing.expectEqual(@as(f64, 448), e4m3(0x7E));
    try std.testing.expectEqual(@as(f64, 0x1p-9), e4m3(0x01)); // smallest subnormal
    try std.testing.expectEqual(@as(f64, 0x1p-6 * 7 / 8.0), e4m3(0x07));
    try std.testing.expectEqual(@as(f64, 0x1p-6), e4m3(0x08)); // smallest normal
    try std.testing.expectEqual(@as(f64, 1.5), e4m3(0x3C));
    try std.testing.expect(std.math.isNan(e4m3(0x7F)) and std.math.isNan(e4m3(0xFF)));
    try std.testing.expectEqual(@as(f64, 0), e4m3(0x00));
    try std.testing.expectEqual(@as(f64, 1), ue8m0(127, 127));
    try std.testing.expectEqual(@as(f64, 0x1p-12), ue8m0(115, 127));
    try std.testing.expectEqual(@as(f64, 0x1p-13), ue8m0(115, 128)); // the bias is a switch
    try std.testing.expect(std.math.isNan(ue8m0(255, 127)));
}

test "dequant: one scale per 32 inputs of a row" {
    var codes: [2 * 64]u8 = @splat(0x38); // 1.0
    codes[63] = 0xB8; // -1.0 in the last block of row 0
    const scales = [_]u8{ 127, 128, 126, 127 }; // row 0: 1, 2; row 1: 0.5, 1
    var w: [2 * 64]f64 = undefined;
    dequant(&codes, &scales, 2, 64, 32, 127, &w);
    try std.testing.expectEqual(@as(f64, 1), w[0]);
    try std.testing.expectEqual(@as(f64, 1), w[31]);
    try std.testing.expectEqual(@as(f64, 2), w[32]);
    try std.testing.expectEqual(@as(f64, -2), w[63]);
    try std.testing.expectEqual(@as(f64, 0.5), w[64]);
    try std.testing.expectEqual(@as(f64, 1), w[127]);
}
