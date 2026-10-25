//! MiniMax-M3.1-Flash on the CPU in float64: a full layer, a small model and the head (engine-port-plan.md T0).
const std = @import("std");
const core = @import("core");
const mr = core.minimax_ref;

pub const Switches = mr.Switches;

/// The text model's shape; the defaults are the real checkpoint's, tests shrink hidden, experts and the block count.
pub const Config = struct {
    hidden: usize = 6144,
    n_q: usize = 64,
    n_kv: usize = 4,
    head_dim: usize = 128,
    rot_dim: usize = 64,
    rope_theta: f64 = 1e7,
    eps: f64 = 1e-6,
    /// Index heads: one per kv head, all scoring the one shared index key (index head g selects for kv head g).
    n_index: usize = 4,
    block: usize = 128,
    topk_blocks: usize = 16,
    n_experts: usize = 128,
    experts_topk: usize = 4,
    routed_scale: f64 = 2.0,
    alpha: f64 = 1.702,
    limit: f64 = 7.0,
    sw: Switches = .{},

    pub fn rep(c: Config) usize {
        return c.n_q / c.n_kv;
    }
};

/// y = x W for W stored [k][n] row-major, the layout exl3_format.dequantize writes.
pub const Linear = struct {
    k: usize,
    n: usize,
    w: []const f64,

    pub fn apply(l: Linear, x: []const f64, out: []f64) void {
        @memset(out[0..l.n], 0);
        for (x[0..l.k], 0..) |xv, i| {
            const row = l.w[i * l.n ..][0..l.n];
            for (out[0..l.n], row) |*o, wv| o.* += xv * wv;
        }
    }
};

/// A gated MLP: w1, w3 are [hidden][inter], w2 [inter][hidden]; dense and shared MLPs: gate_proj is w1, up_proj w3.
pub const Mlp = struct {
    w1: Linear,
    w3: Linear,
    w2: Linear,

    /// out = w2 swigluoai(gate, up); `routed` applies Switches.w1_is_gate, dense and shared MLPs never swap.
    pub fn apply(m: Mlp, c: Config, x: []const f64, routed: bool, a: std.mem.Allocator, out: []f64) !void {
        const n = m.w1.n;
        const p = try a.alloc(f64, 2 * n);
        m.w1.apply(x, p[0..n]);
        m.w3.apply(x, p[n..]);
        const swap = routed and !c.sw.w1_is_gate;
        for (0..n) |i| {
            const g, const u = if (swap) .{ p[n + i], p[i] } else .{ p[i], p[n + i] };
            p[i] = mr.round16(c.sw.bf16_act_out, mr.swigluoai(g, u, c.alpha, c.limit, c.sw));
        }
        m.w2.apply(p[0..n], out);
    }
};

pub const Moe = struct {
    gate: Linear,
    bias: []const f64,
    experts: []const Mlp,
    shared: Mlp,
};

pub const Ffn = union(enum) { dense: Mlp, moe: Moe };

pub const Layer = struct {
    input_norm: []const f64,
    post_norm: []const f64,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: []const f64,
    k_norm: []const f64,
    index_q: Linear,
    index_k: Linear,
    index_q_norm: []const f64,
    index_k_norm: []const f64,
    ffn: Ffn,
};

pub const Model = struct {
    cfg: Config,
    embed: []const f64,
    layers: []const Layer,
    final_norm: []const f64,
    head: Linear,
};

/// One layer's K, V ([kv head][pos][d]) and index-key (IK [pos][d]) history, `cap` positions.
pub const LayerCache = struct {
    k: []f64,
    v: []f64,
    ik: []f64,
    cap: usize,
};

pub const Cache = struct {
    layers: []LayerCache,

    pub fn init(a: std.mem.Allocator, c: Config, n_layers: usize, cap: usize) !Cache {
        const ls = try a.alloc(LayerCache, n_layers);
        for (ls) |*l| l.* = .{
            .k = try a.alloc(f64, c.n_kv * cap * c.head_dim),
            .v = try a.alloc(f64, c.n_kv * cap * c.head_dim),
            .ik = try a.alloc(f64, cap * c.head_dim),
            .cap = cap,
        };
        return .{ .layers = ls };
    }
};

fn norm(c: Config, x: []const f64, w: []const f64, gemma: bool, out: []f64) void {
    mr.gemmaNorm(x, w, c.eps, gemma, out);
    mr.roundAll(c.sw.bf16_norm_out, out);
}

fn heads(c: Config, x: []f64, n: usize, w: []const f64, gemma: bool, pos: ?usize) void {
    const d = c.head_dim;
    var t: [512]f64 = undefined;
    for (0..n) |h| {
        const s = x[h * d ..][0..d];
        norm(c, s, w, gemma, t[0..d]);
        @memcpy(s, t[0..d]);
        if (pos) |p| mr.rope(s, c.rot_dim, p, c.rope_theta, c.sw.rope_layout);
        mr.roundAll(c.sw.bf16_rope_out, s);
    }
}

/// The attention half of a layer for the token at `pos`: h += o_proj(sparse attention(norm(h))), cache appended.
pub fn attention(c: Config, ly: Layer, lc: *LayerCache, h: []f64, pos: usize, a: std.mem.Allocator) !void {
    const d = c.head_dim;
    const x = try a.alloc(f64, c.hidden);
    norm(c, h, ly.input_norm, c.sw.gemma_layer_norms, x);
    const q = try a.alloc(f64, c.n_q * d);
    const k = try a.alloc(f64, c.n_kv * d);
    const v = try a.alloc(f64, c.n_kv * d);
    ly.q.apply(x, q);
    ly.k.apply(x, k);
    ly.v.apply(x, v);
    for ([_][]f64{ q, k, v }) |t| mr.roundAll(c.sw.bf16_proj_out, t);
    heads(c, q, c.n_q, ly.q_norm, c.sw.gemma_qk_norms, pos);
    heads(c, k, c.n_kv, ly.k_norm, c.sw.gemma_qk_norms, pos);
    const iq = try a.alloc(f64, c.n_index * d);
    const ik = try a.alloc(f64, d);
    mr.roundFp16(c.sw.fp16_index_input, x);
    ly.index_q.apply(x, iq);
    ly.index_k.apply(x, ik);
    mr.roundAll(c.sw.bf16_proj_out, iq);
    mr.roundAll(c.sw.bf16_proj_out, ik);
    heads(c, iq, c.n_index, ly.index_q_norm, c.sw.gemma_index_norms, if (c.sw.index_rope) pos else null);
    heads(c, ik, 1, ly.index_k_norm, c.sw.gemma_index_norms, if (c.sw.index_rope) pos else null);
    for (0..c.n_kv) |g| {
        @memcpy(lc.k[(g * lc.cap + pos) * d ..][0..d], k[g * d ..][0..d]);
        @memcpy(lc.v[(g * lc.cap + pos) * d ..][0..d], v[g * d ..][0..d]);
    }
    @memcpy(lc.ik[pos * d ..][0..d], ik);
    const bs = try a.alloc(f64, pos / c.block + 1);
    const ids = try a.alloc(usize, c.topk_blocks);
    const scratch = try a.alloc(f64, bs.len);
    const keys = try a.alloc(usize, c.topk_blocks * c.block);
    const ao = try a.alloc(f64, c.n_q * d);
    const rep = c.rep();
    for (0..c.n_kv) |g| {
        _ = mr.blockScores(iq[g * d ..][0..d], lc.ik, pos, d, std.math.pow(f64, @floatFromInt(d), -0.5), c.block, bs);
        const n = mr.selectBlocks(bs, pos / c.block, c.topk_blocks, c.sw, scratch, ids);
        const base = g * lc.cap * d;
        mr.attendBlocks(q[g * rep * d ..][0 .. rep * d], lc.k[base..][0 .. lc.cap * d], lc.v[base..][0 .. lc.cap * d], ids[0..n], pos, c.block, d, rep, keys, ao[g * rep * d ..][0 .. rep * d]);
    }
    mr.roundAll(c.sw.bf16_attn_out, ao);
    const o = try a.alloc(f64, c.hidden);
    ly.o.apply(ao, o);
    for (h, o) |*hv, ov| hv.* = mr.round16(c.sw.bf16_residual, hv.* + ov);
}

/// The feed-forward half: h += ffn(norm(h)); dense MLP, or routed experts plus the shared expert.
pub fn feedForward(c: Config, ly: Layer, h: []f64, a: std.mem.Allocator) !void {
    const m = try a.alloc(f64, c.hidden);
    norm(c, h, ly.post_norm, c.sw.gemma_layer_norms, m);
    const f = try a.alloc(f64, c.hidden);
    switch (ly.ffn) {
        .dense => |mlp| try mlp.apply(c, m, false, a, f),
        .moe => |moe| {
            const logits = try a.alloc(f64, c.n_experts);
            moe.gate.apply(m, logits);
            const ids = try a.alloc(usize, c.experts_topk);
            const w = try a.alloc(f64, c.experts_topk);
            mr.routerPick(logits, moe.bias, c.routed_scale, c.sw, try a.alloc(f64, c.n_experts), ids, w);
            @memset(f, 0);
            const y = try a.alloc(f64, c.hidden);
            for (ids, w) |e, we| {
                try moe.experts[e].apply(c, m, true, a, y);
                for (f, y) |*o, yv| o.* += we * yv;
            }
            try moe.shared.apply(c, m, false, a, y);
            const t = mr.totalScale(c.sw, c.routed_scale);
            for (f, y) |*o, yv| o.* = (o.* + yv) * t;
        },
    }
    for (h, f) |*hv, fv| hv.* = mr.round16(c.sw.bf16_residual, hv.* + fv);
}

/// One whole layer for the token at `pos` (h in and out); `a` is scratch the caller may reset afterwards.
pub fn layerStep(c: Config, ly: Layer, lc: *LayerCache, h: []f64, pos: usize, a: std.mem.Allocator) !void {
    try attention(c, ly, lc, h, pos, a);
    try feedForward(c, ly, h, a);
}

/// The head: logits = (gemma_norm(h, model.norm)) lm_head, as the engine's final fp32 logits.
pub fn head(m: Model, h: []const f64, a: std.mem.Allocator, logits: []f64) !void {
    const n = try a.alloc(f64, m.cfg.hidden);
    norm(m.cfg, h, m.final_norm, m.cfg.sw.gemma_layer_norms, n);
    m.head.apply(n, logits);
}

/// The model on one token at `pos` against its cache: embedding, every layer, the head into `logits`.
pub fn step(m: Model, cache: Cache, token: usize, pos: usize, logits: []f64, gpa: std.mem.Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const h = try a.dupe(f64, m.embed[token * m.cfg.hidden ..][0..m.cfg.hidden]);
    for (m.layers, cache.layers) |ly, *lc| try layerStep(m.cfg, ly, lc, h, pos, a);
    try head(m, h, a, logits);
}

/// Every token in order; the last token's logits are left in `logits`.
pub fn run(m: Model, cache: Cache, tokens: []const usize, logits: []f64, gpa: std.mem.Allocator) !void {
    for (tokens, 0..) |t, pos| try step(m, cache, t, pos, logits, gpa);
}
