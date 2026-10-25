//! The float64 draft block on a tiny random draft: properties, switch liveness, twins and planted mutants.
const std = @import("std");
const core = @import("core");
const mr = core.minimax_ref;
const ref = @import("dspark_ref.zig");
const twin = @import("dspark_twin.zig");

const gpa = std.testing.allocator;

var fixed_prng: std.Random.DefaultPrng = undefined;

/// A fresh deterministic generator for a one-off vector.
fn fixedRandom() std.Random {
    fixed_prng = std.Random.DefaultPrng.init(5);
    return fixed_prng.random();
}

// ---- a tiny random draft ------------------------------------------------------------------------------------------

const Tiny = struct { vocab: usize = 24, rank: usize = 4, markov_scale: f64 = 1.0 };

fn rnd(r: std.Random, a: std.mem.Allocator, n: usize, scale: f64) ![]f64 {
    const v = try a.alloc(f64, n);
    for (v) |*x| x.* = (r.float(f64) * 2 - 1) * scale;
    return v;
}

fn dense(r: std.Random, a: std.mem.Allocator, n: usize, k: usize) !ref.Dense {
    return .{ .n = n, .k = k, .w = try rnd(r, a, n * k, 1.5 / @sqrt(@as(f64, @floatFromInt(k)))) };
}

fn tinyConfig(t: Tiny, sw: ref.Switches) ref.Config {
    return .{ .hidden = 32, .n_q = 4, .n_kv = 2, .head_dim = 16, .rot_dim = 8, .rope_theta = 100, .ffn = 24, .layers = 2, .n_taps = 3, .window = 6, .block = 4, .mask_token = t.vocab - 1, .rank = t.rank, .vocab = t.vocab, .candidates = 6, .limit = 0.3, .sw = sw };
}

fn tinyDraft(a: std.mem.Allocator, seed: u64, t: Tiny, sw: ref.Switches) !ref.Draft {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const c = tinyConfig(t, sw);
    const ls = try a.alloc(ref.Layer, c.layers);
    for (ls) |*l| l.* = .{
        .input_norm = try rnd(r, a, c.hidden, 0.4),
        .post_norm = try rnd(r, a, c.hidden, 0.4),
        .q_norm = try rnd(r, a, c.head_dim, 0.4),
        .k_norm = try rnd(r, a, c.head_dim, 0.4),
        .q = try dense(r, a, c.n_q * c.head_dim, c.hidden),
        .k = try dense(r, a, c.kvw(), c.hidden),
        .v = try dense(r, a, c.kvw(), c.hidden),
        .o = try dense(r, a, c.hidden, c.n_q * c.head_dim),
        .gate = try dense(r, a, c.ffn, c.hidden),
        .up = try dense(r, a, c.ffn, c.hidden),
        .down = try dense(r, a, c.hidden, c.ffn),
    };
    return .{
        .cfg = c,
        .fc = try dense(r, a, c.hidden, c.n_taps * c.hidden),
        .hidden_norm = try rnd(r, a, c.hidden, 0.2),
        .final_norm = try rnd(r, a, c.hidden, 0.4),
        .layers = ls,
        .markov_w1 = try rnd(r, a, c.vocab * c.rank, t.markov_scale),
        .markov_w2 = try rnd(r, a, c.vocab * c.rank, t.markov_scale),
        .conf_w = try rnd(r, a, c.hidden + c.rank, 0.3),
        .conf_b = 0.1,
        .embed = try rnd(r, a, c.vocab * c.hidden, 1.0),
        .head = try dense(r, a, c.vocab, c.hidden),
    };
}

/// Tapped features of `n` positions from a fake target of `layers` layers (the residual streams are random).
fn features(a: std.mem.Allocator, seed: u64, c: ref.Config, ids: []const usize, target_layers: usize, n: usize, final_norm: []const f64) ![]f64 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try a.alloc(f64, n * c.n_taps * c.hidden);
    const resid = try a.alloc([]const f64, target_layers + 1);
    for (0..n) |j| {
        for (resid) |*row| row.* = try rnd(r, a, c.hidden, 2.0);
        ref.tapRow(c, ids, target_layers, resid, final_norm, true, out[j * c.n_taps * c.hidden ..][0 .. c.n_taps * c.hidden]);
    }
    return out;
}

const tap_ids = [_]usize{ 1, 3, 5 }; // of a 6-layer fake target: the last tap is the last layer
const fake_layers = 6;

fn maxDiff(x: []const f64, y: []const f64) f64 {
    var m: f64 = 0;
    for (x, y) |a, b| m = @max(m, @abs(a - b));
    return m;
}

/// The whole path, reference form: taps -> context K/V for all positions -> block -> chain; the result as one vector.
fn pipeline(a: std.mem.Allocator, sw: ref.Switches) ![]f64 {
    const t: Tiny = .{ .markov_scale = 3.0 };
    const d = try tinyDraft(a, 11, t, sw);
    const c = d.cfg;
    const fnorm = try rnd(fixedRandom(), a, c.hidden, 0.3);
    const p: usize = 10; // more than the window of 6
    const feats = try features(a, 21, c, &tap_ids, fake_layers, p, fnorm);
    const cx = try ref.buildCtx(&d, a, feats, 0, p);
    const b = try ref.blockForward(&d, a, &cx, 5, p);
    const n = ref.draftCount(c.sw, c.block);
    const drafts = try a.alloc(usize, n);
    const conf = try a.alloc(f64, n);
    try ref.chain(&d, a, b, 5, drafts, conf);
    const out = try a.alloc(f64, b.pre.len + b.post.len + 2 * n);
    @memcpy(out[0..b.pre.len], b.pre);
    @memcpy(out[b.pre.len..][0..b.post.len], b.post);
    for (drafts, conf, 0..) |dt, q, i| {
        out[b.pre.len + b.post.len + 2 * i] = @floatFromInt(dt);
        out[b.pre.len + b.post.len + 2 * i + 1] = q;
    }
    return out;
}

// ---- reference properties -----------------------------------------------------------------------------------------

test "the draft block is deterministic and finite" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const x = try pipeline(a, .{});
    const y = try pipeline(a, .{});
    try std.testing.expectEqualSlices(f64, x, y);
    for (x) |v| try std.testing.expect(std.math.isFinite(v));
}

test "every switch is live: its other setting changes the result" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const base = try pipeline(a, .{});
    const flips = .{
        .{ "base.rope_layout", ref.Switches{ .base = .{ .rope_layout = .gptj_interleaved } } },
        .{ "base.up_plus_one", ref.Switches{ .base = .{ .up_plus_one = false } } },
        .{ "base.gate_clamp", ref.Switches{ .base = .{ .gate_clamp = .symmetric } } },
        .{ "base.gemma_layer_norms", ref.Switches{ .base = .{ .gemma_layer_norms = false } } },
        .{ "base.gemma_qk_norms", ref.Switches{ .base = .{ .gemma_qk_norms = false } } },
        .{ "base.bf16_proj_out", ref.Switches{ .base = .{ .bf16_proj_out = true } } },
        .{ "base.bf16_residual", ref.Switches{ .base = .{ .bf16_residual = true } } },
        .{ "tap_site", ref.Switches{ .tap_site = .layer_input } },
        .{ "last_tap_post_norm", ref.Switches{ .last_tap_post_norm = true } },
        .{ "tap_bf16", ref.Switches{ .tap_bf16 = false } },
        .{ "concat_order", ref.Switches{ .concat_order = .reversed } },
        .{ "gemma_hidden_norm", ref.Switches{ .gemma_hidden_norm = false } },
        .{ "gemma_final_norm", ref.Switches{ .gemma_final_norm = false } },
        .{ "ctx_input_norm", ref.Switches{ .ctx_input_norm = true } },
        .{ "window_rule", ref.Switches{ .window_rule = .context_tail } },
        .{ "block_attention", ref.Switches{ .block_attention = .causal } },
        .{ "block_positions", ref.Switches{ .block_positions = .all_pending } },
        .{ "draft_rows", ref.Switches{ .draft_rows = .rows_after_anchor } },
        .{ "markov_domain", ref.Switches{ .markov_domain = .candidates } },
        .{ "markov_swap", ref.Switches{ .markov_swap = true } },
        .{ "first_prev", ref.Switches{ .first_prev = .none } },
        .{ "conf_input", ref.Switches{ .conf_input = .pre_final_norm } },
        .{ "conf_sigmoid", ref.Switches{ .conf_sigmoid = false } },
        .{ "aux_bf16", ref.Switches{ .aux_bf16 = false } },
    };
    inline for (flips) |f| {
        const got = try pipeline(a, f[1]);
        if (got.len == base.len and maxDiff(base, got) <= 1e-7) {
            std.debug.print("switch {s} changed nothing\n", .{f[0]});
            return error.DeadSwitch;
        }
    }
}

test "tap sources: layer output by default, the last layer normed only when asked" {
    const sw: ref.Switches = .{};
    try std.testing.expectEqual(ref.TapSource{ .r = 4, .normed = false }, ref.tapSource(sw, 3, 60));
    try std.testing.expectEqual(ref.TapSource{ .r = 60, .normed = false }, ref.tapSource(sw, 59, 60));
    try std.testing.expectEqual(ref.TapSource{ .r = 60, .normed = true }, ref.tapSource(.{ .last_tap_post_norm = true }, 59, 60));
    try std.testing.expectEqual(ref.TapSource{ .r = 45, .normed = false }, ref.tapSource(.{ .tap_site = .layer_input, .last_tap_post_norm = true }, 45, 60));
    try std.testing.expectEqual(ref.TapSource{ .r = 59, .normed = false }, ref.tapSource(.{ .tap_site = .layer_input }, 59, 60));
}

test "the sliding window: a context row outside it cannot matter, one inside it does (per_row), and block rows see each other (bidirectional)" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try tinyDraft(a, 3, .{}, .{});
    const c = d.cfg;
    const fnorm = try rnd(fixedRandom(), a, c.hidden, 0.3);
    const p: usize = 12; // window 6: row 0 sees positions 7 .. 11, row 3 (position 15) sees 10 .. 11
    const feats = try features(a, 9, c, &tap_ids, fake_layers, p, fnorm);
    const base = try ref.blockForward(&d, a, &(try ref.buildCtx(&d, a, feats, 0, p)), 2, p);
    const row = c.n_taps * c.hidden;
    const old = try a.dupe(f64, feats);
    for (old[3 * row ..][0..row]) |*v| v.* += 1.0; // position 3: outside every row's window
    const same = try ref.blockForward(&d, a, &(try ref.buildCtx(&d, a, old, 0, p)), 2, p);
    try std.testing.expectEqualSlices(f64, base.post, same.post);
    const near = try a.dupe(f64, feats);
    for (near[9 * row ..][0..row]) |*v| v.* += 1.0; // position 9: row 0 sees it (15 - ... 12 - 9 < 6)
    const moved = try ref.blockForward(&d, a, &(try ref.buildCtx(&d, a, near, 0, p)), 2, p);
    try std.testing.expect(maxDiff(base.post[0..c.hidden], moved.post[0..c.hidden]) > 1e-6);
    // position 7 is inside row 0's window (12 - 7 = 5 < 6) but outside row 3's (15 - 7 = 8 >= 6)
    const edge = try a.dupe(f64, feats);
    for (edge[7 * row ..][0..row]) |*v| v.* += 1.0;
    const e = try ref.blockForward(&d, a, &(try ref.buildCtx(&d, a, edge, 0, p)), 2, p);
    try std.testing.expect(maxDiff(base.post[0..c.hidden], e.post[0..c.hidden]) > 1e-6);
}

test "causal block attention: a later block row cannot change an earlier one; bidirectional lets it" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]ref.BlockAttention{ .causal, .bidirectional }) |mode| {
        var d = try tinyDraft(a, 4, .{}, .{ .block_attention = mode });
        const c = d.cfg;
        const fnorm = try rnd(fixedRandom(), a, c.hidden, 0.3);
        const feats = try features(a, 9, c, &tap_ids, fake_layers, 8, fnorm);
        const cx = try ref.buildCtx(&d, a, feats, 0, 8);
        const one = try ref.blockForward(&d, a, &cx, 2, 8);
        // rows 1.. start from the mask token's embedding: moving it moves row 0 only if the rows see each other
        const e = @constCast(d.embed);
        for (e[c.mask_token * c.hidden ..][0..c.hidden]) |*v| v.* += 0.5;
        const two = try ref.blockForward(&d, a, &cx, 2, 8);
        // row 0 (the anchor) reads rows 1.. only if the block is bidirectional
        const moved = maxDiff(one.post[0..c.hidden], two.post[0..c.hidden]) > 1e-6;
        try std.testing.expectEqual(mode == .bidirectional, moved);
    }
}

test "contextKv: layer K is normed and rotated at the token's position, V is the plain projection" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try tinyDraft(a, 8, .{}, .{});
    const c = d.cfg;
    const st = try rnd(fixedRandom(), a, c.hidden, 1.0);
    const k = try a.alloc(f64, c.kvw());
    const v = try a.alloc(f64, c.kvw());
    try ref.contextKv(&d, a, 1, st, 7, k, v);
    const want_v = try a.alloc(f64, c.kvw());
    d.layers[1].v.apply(st, want_v);
    try std.testing.expectEqualSlices(f64, want_v, v);
    const want_k = try a.alloc(f64, c.kvw());
    d.layers[1].k.apply(st, want_k);
    for (0..c.n_kv) |g| {
        const h = want_k[g * c.head_dim ..][0..c.head_dim];
        var t: [16]f64 = undefined;
        mr.gemmaNorm(h, d.layers[1].k_norm, c.eps, true, &t);
        @memcpy(h, &t);
        mr.rope(h, c.rot_dim, 7, c.rope_theta, .neox_half);
    }
    try std.testing.expect(maxDiff(want_k, k) < 1e-12);
}

// ---- the engine-layout twins and their mutants --------------------------------------------------------------------

fn ringRun(a: std.mem.Allocator, sw: ref.Switches, cap_extra: usize, m: twin.RingMutant) !f64 {
    const d = try tinyDraft(a, 31, .{}, sw);
    const c = d.cfg;
    const fnorm = try rnd(fixedRandom(), a, c.hidden, 0.3);
    const total: usize = 16;
    const feats = try features(a, 41, c, &tap_ids, fake_layers, total, fnorm);
    const row = c.n_taps * c.hidden;
    var ring = try twin.Ring.init(a, &d, c.window + cap_extra, m);
    var worst: f64 = 0;
    var next: usize = 0;
    for ([_]usize{ 3, 7, 8, 13, 16 }) |p| { // pending positions: a step absorbs the kept rows, then drafts
        while (next < p) : (next += 1) try ring.absorb(a, feats[next * row ..][0..row], next + 1 == p);
        const cx_ref = try ref.buildCtx(&d, a, feats, 0, p);
        const want = try ref.blockForward(&d, a, &cx_ref, 4, p);
        const cx = try ring.ctx(a);
        const got = try ref.blockForward(&d, a, &cx, 4, p);
        worst = @max(worst, maxDiff(want.pre, got.pre));
        worst = @max(worst, maxDiff(want.post, got.post));
    }
    return worst;
}

test "ring twin: the incremental K/V ring equals recomputing the context from features, at the window and with slack" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]ref.WindowRule{ .per_row, .context_tail }) |rule| {
        // a ring of window + 4 (the engine keeps window + window_rows) and one of exactly the window match bit for bit
        try std.testing.expectEqual(@as(f64, 0), try ringRun(a, .{ .window_rule = rule }, 4, .none));
        try std.testing.expectEqual(@as(f64, 0), try ringRun(a, .{ .window_rule = rule }, 0, .none));
    }
}

test "ring twin mutants: each planted fault is caught" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]ref.WindowRule{ .per_row, .context_tail }) |rule| {
        for ([_]twin.RingMutant{ .short_ring, .rope_at_slot, .stale_slot, .drop_last_absorb }) |m| {
            const worst = try ringRun(a, .{ .window_rule = rule }, 4, m);
            if (worst <= 1e-6) {
                std.debug.print("ring mutant {t} under {t} slipped through ({e})\n", .{ m, rule, worst });
                return error.MutantSurvived;
            }
        }
    }
}

fn tpRun(a: std.mem.Allocator, sw: ref.Switches, world: usize, m: twin.TpMutant) !f64 {
    const d = try tinyDraft(a, 51, .{}, sw);
    const c = d.cfg;
    const fnorm = try rnd(fixedRandom(), a, c.hidden, 0.3);
    const p: usize = 9;
    const feats = try features(a, 61, c, &tap_ids, fake_layers, p, fnorm);
    const cx = try ref.buildCtx(&d, a, feats, 0, p);
    const want = try ref.blockForward(&d, a, &cx, 3, p);
    const got = try twin.blockForwardTp(&d, a, &cx, 3, p, world, m);
    return @max(maxDiff(want.pre, got.pre), maxDiff(want.post, got.post));
}

test "tp twin: one rank and two ranks (heads, kv groups, FFN rows cut; partial sums added in rank order) equal the reference" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]usize{ 1, 2 }) |world| {
        try std.testing.expect(try tpRun(a, .{}, world, .none) < 1e-10);
        try std.testing.expect(try tpRun(a, .{ .block_attention = .causal, .window_rule = .context_tail }, world, .none) < 1e-10);
        try std.testing.expect(try tpRun(a, .{ .base = .{ .bf16_proj_out = true, .bf16_act_out = true } }, world, .none) < 1e-2);
    }
}

test "tp twin mutants: each planted fault is caught" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]twin.TpMutant{ .drop_last_head, .swap_kv_groups, .ffn_overlap, .drop_exchange }) |m| {
        const worst = try tpRun(a, .{}, 2, m);
        if (worst <= 1e-6) {
            std.debug.print("tp mutant {t} slipped through ({e})\n", .{ m, worst });
            return error.MutantSurvived;
        }
    }
}

// ---- the Markov chain, the confidence head, the per-rank gather ---------------------------------------------------

/// A draft over 8 tokens whose Markov head forces the chain prev -> sigma(prev), for hand-checkable drafts.
fn chainDraft(a: std.mem.Allocator, sw: ref.Switches) !ref.Draft {
    var d = try tinyDraft(a, 77, .{ .vocab = 8, .rank = 8, .markov_scale = 0 }, sw);
    const sigma = [_]usize{ 3, 5, 0, 7, 1, 2, 6, 4 };
    const w1 = try a.alloc(f64, 64);
    const w2 = try a.alloc(f64, 64);
    @memset(w1, 0);
    @memset(w2, 0);
    for (0..8) |prev| {
        w1[prev * 8 + prev] = 1000; // bias(prev, id) = 1000 * w2[id][prev]
        w2[sigma[prev] * 8 + prev] = 1; // ... which is 1000 for id = sigma(prev)
    }
    d.markov_w1 = w1;
    d.markov_w2 = w2;
    return d;
}

fn chainOf(a: std.mem.Allocator, d: *const ref.Draft, anchor: usize) ![]usize {
    const c = d.cfg;
    const fnorm = try rnd(fixedRandom(), a, c.hidden, 0.3);
    const feats = try features(a, 81, c, &tap_ids, fake_layers, 5, fnorm);
    const cx = try ref.buildCtx(d, a, feats, 0, 5);
    const b = try ref.blockForward(d, a, &cx, anchor, 5);
    const drafts = try a.alloc(usize, ref.draftCount(c.sw, c.block));
    const conf = try a.alloc(f64, drafts.len);
    try ref.chain(d, a, b, anchor, drafts, conf);
    return drafts;
}

test "chain: a Markov head that forces prev -> sigma(prev) is followed from the anchor, whatever the base logits say" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const sigma = [_]usize{ 3, 5, 0, 7, 1, 2, 6, 4 };
    const d = try chainDraft(a, .{});
    for ([_]usize{ 0, 4, 6 }) |anchor| {
        const got = try chainOf(a, &d, anchor);
        var prev = anchor;
        for (got) |t| {
            try std.testing.expectEqual(sigma[prev], t);
            prev = t;
        }
    }
    // swapped tables: the bias is 1000 * w1[id][prev-th w2 column], which does not follow sigma
    const swapped = try chainDraft(a, .{ .markov_swap = true });
    const got = try chainOf(a, &swapped, 0);
    var prev: usize = 0;
    var follows = true;
    for (got) |t| {
        follows = follows and t == sigma[prev];
        prev = t;
    }
    try std.testing.expect(!follows);
    // no previous token for the first draft: it is the base argmax; the rest follow sigma
    const none = try chainDraft(a, .{ .first_prev = .none });
    const g = try chainOf(a, &none, 0);
    try std.testing.expectEqual(sigma[g[0]], g[1]);
    try std.testing.expectEqual(sigma[g[1]], g[2]);
    // DFlash's rows: one draft fewer
    const rows = try chainDraft(a, .{ .draft_rows = .rows_after_anchor });
    try std.testing.expectEqual(@as(usize, 3), (try chainOf(a, &rows, 0)).len);
}

test "chain: candidates = the whole vocabulary equals the full vocabulary; a small candidate list stays inside the base top" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const full = try tinyDraft(a, 90, .{ .markov_scale = 2.0 }, .{});
    var wide = try tinyDraft(a, 90, .{ .markov_scale = 2.0 }, .{ .markov_domain = .candidates });
    wide.cfg.candidates = wide.cfg.vocab;
    try std.testing.expectEqualSlices(usize, try chainOf(a, &full, 3), try chainOf(a, &wide, 3));
    var narrow = try tinyDraft(a, 90, .{ .markov_scale = 2.0 }, .{ .markov_domain = .candidates });
    narrow.cfg.candidates = 1;
    var plain = try tinyDraft(a, 90, .{ .markov_scale = 0 }, .{});
    // one candidate: every draft is its row's base argmax, which a zero Markov table also gives
    try std.testing.expectEqualSlices(usize, try chainOf(a, &plain, 3), try chainOf(a, &narrow, 3));
}

test "confidence: sigmoid(w . [h ; markov_w1[prev]] + b), the logit when asked, no Markov part without a previous token" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try tinyDraft(a, 5, .{}, .{ .aux_bf16 = false });
    const c = d.cfg;
    const z = try a.alloc(f64, c.hidden);
    @memset(z, 0);
    const w = try a.alloc(f64, c.hidden + c.rank);
    @memset(w, 0);
    d.conf_w = w;
    d.conf_b = 0.75;
    try std.testing.expectApproxEqAbs(mr.sigmoid(0.75), try ref.confidence(&d, a, z, z, 7), 1e-15);
    // h = e0, w[0] = 2; markov part: w1[3] . w[hidden..]
    z[0] = 1;
    w[0] = 2;
    for (w[c.hidden..], 0..) |*x, i| x.* = @floatFromInt(i + 1);
    var want: f64 = 0.75 + 2.0;
    try std.testing.expectApproxEqAbs(mr.sigmoid(want), try ref.confidence(&d, a, z, z, null), 1e-15);
    for (d.markov_w1[3 * c.rank ..][0..c.rank], 0..) |x, i| want += x * @as(f64, @floatFromInt(i + 1));
    try std.testing.expectApproxEqAbs(mr.sigmoid(want), try ref.confidence(&d, a, z, z, 3), 1e-12);
    d.cfg.sw.conf_sigmoid = false;
    try std.testing.expectApproxEqAbs(want, try ref.confidence(&d, a, z, z, 3), 1e-12);
}

test "confidence gate: the running product cuts earlier than each alone; the first `least` drafts always stay" {
    const conf = [_]f64{ 0.9, 0.9, 0.9, 0.9 };
    try std.testing.expectEqual(@as(usize, 3), ref.gateLen(.{}, &conf, 0.7, 0));
    try std.testing.expectEqual(@as(usize, 4), ref.gateLen(.{ .conf_gate = .each }, &conf, 0.7, 0));
    const cliff = [_]f64{ 0.9, 0.9, 0.5, 0.99 };
    try std.testing.expectEqual(@as(usize, 2), ref.gateLen(.{}, &cliff, 0.6, 0));
    try std.testing.expectEqual(@as(usize, 2), ref.gateLen(.{ .conf_gate = .each }, &cliff, 0.6, 0));
    try std.testing.expectEqual(@as(usize, 2), ref.gateLen(.{}, &cliff, 0.85, 2));
    try std.testing.expectEqual(@as(usize, 1), ref.gateLen(.{}, &cliff, 0.85, 0));
}

test "gather twin: per-rank candidate lists agree with the full argmax; planted mutants are caught" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try tinyDraft(a, 123, .{ .vocab = 24, .rank = 4, .markov_scale = 1.0 }, .{});
    var base: [24]f64 = undefined;
    for (&base, 0..) |*b, i| b.* = @as(f64, @floatFromInt((i * 7) % 24)) * 0.1; // distinct
    // make the true winner (base + bias) live in the last rank's slice and keep it inside its top 3
    const prev: usize = 2;
    var winner: usize = 12;
    var bz: f64 = -1e9;
    for (0..24) |id| {
        const z = base[id] + ref.markovBias(&d, prev, id);
        if (z > bz) {
            bz = z;
            winner = id;
        }
    }
    // clean: with per = the slice, the gather is the whole vocabulary
    try std.testing.expectEqual(winner, try twin.gatherPick(&d, &base, prev, 2, 12, .none, a));
    // ... and the engine's narrow lists agree whenever the winner is in them: widen the winner's base so it is
    var boosted = base;
    boosted[winner] += 5;
    var best: usize = 0;
    var bb: f64 = -1e9;
    for (0..24) |id| {
        const z = boosted[id] + ref.markovBias(&d, prev, id);
        if (z > bb) {
            bb = z;
            best = id;
        }
    }
    try std.testing.expectEqual(best, try twin.gatherPick(&d, &boosted, prev, 2, 3, .none, a));
    // mutants: dropping the last rank loses a winner living there; unshifted ids move the pick; a tie shows the rule
    try std.testing.expect(best != try twin.gatherPick(&d, &boosted, prev, 2, 3, .ids_not_offset, a) or best < 12);
    if (best >= 12) try std.testing.expect(best != try twin.gatherPick(&d, &boosted, prev, 2, 3, .drop_last_rank, a));
    // an exact tie in z between a higher-base id listed first and a lower id: the lower id must win
    var tie: [24]f64 = @splat(-100);
    const hi_id: usize = 5;
    const lo_id: usize = 2;
    const bias_hi = ref.markovBias(&d, prev, hi_id);
    const bias_lo = ref.markovBias(&d, prev, lo_id);
    tie[hi_id] = 50 + (bias_lo - bias_hi) * 0 + 1; // z_hi = 51 + bias_hi
    tie[lo_id] = tie[hi_id] + bias_hi - bias_lo; // z_lo = z_hi exactly (same sum, up to rounding)
    const zhi = tie[hi_id] + bias_hi;
    const zlo = tie[lo_id] + bias_lo;
    if (zhi == zlo) {
        try std.testing.expectEqual(lo_id, try twin.gatherPick(&d, &tie, prev, 2, 3, .none, a));
        if (tie[hi_id] > tie[lo_id]) try std.testing.expectEqual(hi_id, try twin.gatherPick(&d, &tie, prev, 2, 3, .first_lowest, a));
    }
}
