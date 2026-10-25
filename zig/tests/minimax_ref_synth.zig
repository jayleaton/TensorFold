//! Synthetic MiniMax weights for the CPU reference tests: random EXL3 mul1 trellises at 6 to 12 halves, plain tensors.
const std = @import("std");
const core = @import("core");
const ref = @import("minimax_ref");
const exl3 = core.exl3_format;

pub const Options = struct {
    seed: u64 = 1,
    n_layers: usize = 2,
    vocab: usize = 128,
    /// Multiplies every MLP matrix (expert, shared, dense) so activations reach swigluoai's limit.
    gain: f64 = 1,
    /// Zero index_q_proj: every block score is exactly 0, so only the tie rule picks blocks.
    zero_index_q: bool = false,
};

pub fn tiny(sw: ref.Switches) ref.Config {
    return .{ .hidden = 128, .n_q = 4, .n_kv = 2, .n_index = 2, .topk_blocks = 2, .n_experts = 8, .experts_topk = 2, .sw = sw };
}

pub const Gen = struct {
    a: std.mem.Allocator,
    rng: std.Random.DefaultPrng,

    fn uni(g: *Gen) f64 {
        return g.rng.random().float(f64) * 2 - 1;
    }

    fn vec(g: *Gen, n: usize, scale: f64, bf16: bool) ![]f64 {
        const v = try g.a.alloc(f64, n);
        for (v) |*x| x.* = if (bf16) core.bf16_round.bf16(g.uni() * scale) else g.uni() * scale;
        return v;
    }

    /// A plain matrix as the checkpoint stores fp16 (index) or fp32 (router) weights: fp16 rounded when `half`.
    fn plain(g: *Gen, k: usize, n: usize, gain: f64, half: bool) !ref.Linear {
        const w = try g.vec(k * n, gain * 1.7 / @sqrt(@as(f64, @floatFromInt(k))), false);
        if (half) for (w) |*x| {
            const h: f16 = @floatCast(x.*);
            x.* = h;
        };
        return .{ .k = k, .n = n, .w = w };
    }

    /// An EXL3 mul1 matrix of `halves` (6, 8, 10, 12 = 3..6 bits) from random trellis words, suh and svh.
    pub fn exl3Linear(g: *Gen, halves: u16, k: usize, n: usize, gain: f64) !ref.Linear {
        const tiles_k = k / 16;
        const tiles_n = n / 16;
        const trellis = try g.a.alloc(i16, tiles_k * tiles_n * @as(usize, halves) * 8);
        for (trellis) |*t| t.* = @bitCast(g.rng.random().int(u16));
        const suh = try g.a.alloc(f16, k);
        const svh = try g.a.alloc(f16, n);
        for (suh) |*s| s.* = @floatCast(if (g.rng.random().boolean()) @as(f64, 1) else -1);
        const amp = gain / @sqrt(@as(f64, @floatFromInt(k)));
        for (svh) |*s| s.* = @floatCast((if (g.rng.random().boolean()) @as(f64, 1) else -1) * amp * (0.5 + g.rng.random().float(f64)));
        const w = try g.a.alloc(f64, k * n);
        const scratch = try g.a.alloc(f64, 2 * k * n);
        exl3.dequantize(.{ .halves = halves }, .mul1, trellis, tiles_k, tiles_n, suh, svh, w, scratch);
        return .{ .k = k, .n = n, .w = w };
    }

    fn mlp(g: *Gen, h: usize, inter: usize, halves: u16, gain: f64) !ref.Mlp {
        return .{ .w1 = try g.exl3Linear(halves, h, inter, gain), .w3 = try g.exl3Linear(halves, h, inter, gain), .w2 = try g.exl3Linear(halves, inter, h, gain) };
    }
};

pub fn gen(a: std.mem.Allocator, seed: u64) Gen {
    return .{ .a = a, .rng = std.Random.DefaultPrng.init(seed) };
}

/// A model of `o.n_layers` layers: layer 0 dense (5-bit), the rest MoE (experts 3/4-bit alternating, shared 5-bit).
pub fn model(a: std.mem.Allocator, c: ref.Config, o: Options) !ref.Model {
    var g = Gen{ .a = a, .rng = std.Random.DefaultPrng.init(o.seed) };
    const h = c.hidden;
    const d = c.head_dim;
    const layers = try a.alloc(ref.Layer, o.n_layers);
    for (layers, 0..) |*ly, li| {
        const experts = try a.alloc(ref.Mlp, if (li == 0) 0 else c.n_experts);
        for (experts) |*e| e.* = try g.mlp(h, 128, if (li % 2 == 1) 8 else 6, o.gain);
        ly.* = .{
            .input_norm = try g.vec(h, 0.3, true),
            .post_norm = try g.vec(h, 0.3, true),
            .q = try g.exl3Linear(12, h, c.n_q * d, 1),
            .k = try g.exl3Linear(12, h, c.n_kv * d, 1),
            .v = try g.exl3Linear(12, h, c.n_kv * d, 1),
            .o = try g.exl3Linear(12, c.n_q * d, h, 1),
            .q_norm = try g.vec(d, 0.3, true),
            .k_norm = try g.vec(d, 0.3, true),
            .index_q = try g.plain(h, c.n_index * d, if (o.zero_index_q) 0 else 1, true),
            .index_k = try g.plain(h, d, 1, true),
            .index_q_norm = try g.vec(d, 0.3, true),
            .index_k_norm = try g.vec(d, 0.3, true),
            .ffn = if (li == 0) .{ .dense = try g.mlp(h, 256, 10, o.gain) } else .{ .moe = .{
                .gate = try g.plain(h, c.n_experts, 1, false),
                .bias = try g.vec(c.n_experts, 0.1, false),
                .experts = experts,
                .shared = try g.mlp(h, 128, 10, o.gain),
            } },
        };
    }
    return .{
        .cfg = c,
        .embed = try g.vec(o.vocab * h, 1, true),
        .layers = layers,
        .final_norm = try g.vec(h, 0.3, true),
        .head = try g.exl3Linear(12, h, o.vocab, 1),
    };
}

/// Logits after every position of `tokens`, [tokens.len][vocab].
pub fn trace(a: std.mem.Allocator, m: ref.Model, toks: []const usize, vocab: usize) ![]f64 {
    const cache = try ref.Cache.init(a, m.cfg, m.layers.len, toks.len);
    const out = try a.alloc(f64, toks.len * vocab);
    for (toks, 0..) |t, pos| try ref.step(m, cache, t, pos, out[pos * vocab ..][0..vocab], a);
    return out;
}

pub fn maxDiff(x: []const f64, y: []const f64) f64 {
    var m: f64 = 0;
    for (x, y) |p, q| m = @max(m, @abs(p - q));
    return m;
}

pub fn tokens(a: std.mem.Allocator, n: usize, vocab: usize, seed: u64) ![]usize {
    var rng = std.Random.DefaultPrng.init(seed);
    const t = try a.alloc(usize, n);
    for (t) |*x| x.* = rng.random().uintLessThan(usize, vocab);
    return t;
}
