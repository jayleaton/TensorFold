//! MiniMax-M3.1-Flash float64 reference: layers and a 2-layer model on synthetic EXL3 weights, boundaries, switches.
const std = @import("std");
const core = @import("core");
const ref = @import("minimax_ref");
const synth = @import("minimax_ref_synth.zig");
const mr = core.minimax_ref;

const expect = std.testing.expect;

fn arena() std.heap.ArenaAllocator {
    return std.heap.ArenaAllocator.init(std.heap.page_allocator);
}

fn small(sw: ref.Switches) ref.Config {
    var c = synth.tiny(sw);
    c.block = 16;
    c.topk_blocks = 3;
    return c;
}

fn run(a: std.mem.Allocator, c: ref.Config, o: synth.Options, toks: []const usize) ![]f64 {
    return synth.trace(a, try synth.model(a, c, o), toks, o.vocab);
}

test "EXL3 mul1 linears at 3, 4, 5 and 6 bits dequantize to finite, distinct, nonzero matrices" {
    var ar = arena();
    defer ar.deinit();
    var g = synth.gen(ar.allocator(), 4);
    var prev: ?ref.Linear = null;
    for ([_]u16{ 6, 8, 10, 12 }) |halves| {
        const l = try g.exl3Linear(halves, 128, 128, 1);
        var sum: f64 = 0;
        for (l.w) |v| {
            try expect(std.math.isFinite(v));
            sum += v * v;
        }
        try expect(sum > 1e-3);
        if (prev) |p| try expect(synth.maxDiff(p.w, l.w) > 0);
        prev = l;
        var y: [128]f64 = undefined;
        var x: [128]f64 = @splat(0);
        x[5] = 2;
        l.apply(&x, &y);
        for (y, 0..) |v, j| try std.testing.expectApproxEqAbs(2 * l.w[5 * 128 + j], v, 1e-15);
    }
}

test "model: deterministic, finite, and causal (a later token never changes an earlier position's logits)" {
    var ar = arena();
    defer ar.deinit();
    const a = ar.allocator();
    const c = small(.{});
    const toks = try synth.tokens(a, 40, 128, 2);
    const x = try run(a, c, .{}, toks);
    const y = try run(a, c, .{}, toks);
    try std.testing.expectEqualSlices(f64, x, y);
    for (x) |v| try expect(std.math.isFinite(v));
    var other = try a.dupe(usize, toks);
    other[39] = (other[39] + 1) % 128;
    const z = try run(a, c, .{}, other);
    try std.testing.expectEqualSlices(f64, x[0 .. 39 * 128], z[0 .. 39 * 128]);
    try expect(synth.maxDiff(x[39 * 128 ..], z[39 * 128 ..]) > 1e-6);
}

test "block boundary: pos 127 is dense with one block, pos 128 and 129 leave the older block out when topk = 1" {
    var ar = arena();
    defer ar.deinit();
    const a = ar.allocator();
    const toks = try synth.tokens(a, 131, 128, 6);
    var one = synth.tiny(.{});
    one.topk_blocks = 1;
    var two = one;
    two.topk_blocks = 2;
    var wide = one;
    wide.topk_blocks = 16;
    const x1 = try run(a, one, .{}, toks);
    const x2 = try run(a, two, .{}, toks);
    const x16 = try run(a, wide, .{}, toks);
    try std.testing.expectEqualSlices(f64, x2, x16);
    try std.testing.expectEqualSlices(f64, x1[0 .. 128 * 128], x2[0 .. 128 * 128]);
    for ([_]usize{ 128, 129, 130 }) |p| try expect(synth.maxDiff(x1[p * 128 ..][0..128], x2[p * 128 ..][0..128]) > 1e-6);
}

test "partial last block and more blocks than topk: the model runs on 70 tokens of 16-key blocks (5 blocks, topk 3)" {
    var ar = arena();
    defer ar.deinit();
    const a = ar.allocator();
    const toks = try synth.tokens(a, 70, 128, 8);
    const sparse = try run(a, small(.{}), .{}, toks);
    var dense = small(.{});
    dense.topk_blocks = 16;
    const full = try run(a, dense, .{}, toks);
    try std.testing.expectEqualSlices(f64, sparse[0 .. 48 * 128], full[0 .. 48 * 128]);
    try expect(synth.maxDiff(sparse[69 * 128 ..], full[69 * 128 ..]) > 1e-9);
}

fn proj(l: ref.Linear, x: []const f64, out: []f64) void {
    for (0..l.n) |j| {
        var s: f64 = 0;
        for (0..l.k) |i| s += x[i] * l.w[i * l.n + j];
        out[j] = s;
    }
}

fn headNorm(w: []const f64, v: []f64, pos: ?usize, c: ref.Config) void {
    var t: [128]f64 = undefined;
    for (0..v.len / 128) |h| {
        mr.gemmaNorm(v[h * 128 ..][0..128], w, c.eps, true, &t);
        if (pos) |p| mr.rope(&t, c.rot_dim, p, c.rope_theta, c.sw.rope_layout);
        @memcpy(v[h * 128 ..][0..128], &t);
    }
}

const Scores = struct {
    s: []const f64,
    fn lt(self: @This(), x: usize, y: usize) bool {
        return if (self.s[x] != self.s[y]) self.s[x] > self.s[y] else x < y;
    }
};

/// Layer 0's attention at the last position, re-derived from the embeddings alone with sorted brute-force selection.
fn oracle(a: std.mem.Allocator, m: ref.Model, toks: []const usize) ![]f64 {
    const c = m.cfg;
    const ly = m.layers[0];
    const n = toks.len;
    const p = n - 1;
    const ks = try a.alloc(f64, n * 256);
    const vs = try a.alloc(f64, n * 256);
    const iks = try a.alloc(f64, n * 128);
    var q: [512]f64 = undefined;
    var iq: [256]f64 = undefined;
    for (toks, 0..) |t, j| {
        var x: [128]f64 = undefined;
        mr.gemmaNorm(m.embed[t * 128 ..][0..128], ly.input_norm, c.eps, true, &x);
        proj(ly.k, &x, ks[j * 256 ..][0..256]);
        proj(ly.v, &x, vs[j * 256 ..][0..256]);
        proj(ly.index_k, &x, iks[j * 128 ..][0..128]);
        headNorm(ly.k_norm, ks[j * 256 ..][0..256], j, c);
        headNorm(ly.index_k_norm, iks[j * 128 ..][0..128], j, c);
        if (j == p) {
            proj(ly.q, &x, &q);
            proj(ly.index_q, &x, &iq);
            headNorm(ly.q_norm, &q, p, c);
            headNorm(ly.index_q_norm, &iq, p, c);
        }
    }
    const ao = try a.alloc(f64, 512);
    for (0..c.n_kv) |g| {
        const nb = p / c.block + 1;
        const sc = try a.alloc(f64, nb);
        for (sc, 0..) |*b, bi| {
            b.* = -std.math.inf(f64);
            for (bi * c.block..@min(bi * c.block + c.block, p + 1)) |j| {
                var s: f64 = 0;
                for (0..128) |e| s += iq[g * 128 + e] * iks[j * 128 + e];
                b.* = @max(b.*, s / @sqrt(128.0));
            }
        }
        sc[nb - 1] = std.math.inf(f64);
        const order = try a.alloc(usize, nb);
        for (order, 0..) |*o, i| o.* = i;
        std.mem.sort(usize, order, Scores{ .s = sc }, Scores.lt);
        const take = @min(nb, c.topk_blocks);
        std.mem.sort(usize, order[0..take], {}, std.sort.asc(usize));
        for (g * 2..g * 2 + 2) |h| {
            var sm = try a.alloc(f64, n);
            var cnt: usize = 0;
            var mx: f64 = -std.math.inf(f64);
            for (order[0..take]) |b| for (b * c.block..@min(b * c.block + c.block, p + 1)) |j| {
                var s: f64 = 0;
                for (0..128) |e| s += q[h * 128 + e] * ks[j * 256 + g * 128 + e];
                sm[j] = s / @sqrt(128.0);
                mx = @max(mx, sm[j]);
                cnt += 1;
            };
            var z: f64 = 0;
            @memset(ao[h * 128 ..][0..128], 0);
            for (order[0..take]) |b| for (b * c.block..@min(b * c.block + c.block, p + 1)) |j| {
                const w = @exp(sm[j] - mx);
                z += w;
                for (0..128) |e| ao[h * 128 + e] += w * vs[j * 256 + g * 128 + e];
            };
            for (ao[h * 128 ..][0..128]) |*v| v.* /= z;
        }
    }
    const h = try a.dupe(f64, m.embed[toks[p] * 128 ..][0..128]);
    var o: [128]f64 = undefined;
    proj(ly.o, ao, &o);
    for (h, o) |*x, y| x.* += y;
    return h;
}

fn checkOracle(c: ref.Config, n: usize) !void {
    var ar = arena();
    defer ar.deinit();
    const a = ar.allocator();
    const m = try synth.model(a, c, .{});
    const toks = try synth.tokens(a, n, 128, 12);
    const lc = (try ref.Cache.init(a, c, 1, n)).layers;
    var h: [128]f64 = undefined;
    for (toks, 0..) |t, pos| {
        @memcpy(&h, m.embed[t * 128 ..][0..128]);
        try ref.attention(c, m.layers[0], &lc[0], &h, pos, a);
    }
    const want = try oracle(a, m, toks);
    for (want, h) |p, q| try std.testing.expectApproxEqAbs(p, q, 1e-12);
}

test "layer attention equals a from-scratch brute-force derivation (70 tokens x 16-key blocks; 130 tokens x 128-key blocks)" {
    try checkOracle(small(.{}), 70);
    var c = synth.tiny(.{});
    c.topk_blocks = 1;
    try checkOracle(c, 130);
    try checkOracle(c, 129);
    try checkOracle(c, 128);
}

const Case = struct { name: []const u8, sw: ref.Switches };

test "every UNVERIFIED switch changes the model's output (default vs the other branch)" {
    var ar = arena();
    defer ar.deinit();
    const a = ar.allocator();
    const toks = try synth.tokens(a, 70, 128, 21);
    const base = try run(a, small(.{}), .{}, toks);
    const cases = [_]Case{
        .{ .name = "rope gptj", .sw = .{ .rope_layout = .gptj_interleaved } },
        .{ .name = "index_rope off", .sw = .{ .index_rope = false } },
        .{ .name = "init count 1", .sw = .{ .sparse_init_block = 1 } },
        .{ .name = "init inclusive", .sw = .{ .init_mode = .inclusive } },
        .{ .name = "local 0", .sw = .{ .sparse_local_block = 0 } },
        .{ .name = "local window 2", .sw = .{ .local_mode = .trailing_window, .sparse_local_block = 2 } },
        .{ .name = "w1 is up", .sw = .{ .w1_is_gate = false } },
        .{ .name = "up_plus_one off", .sw = .{ .up_plus_one = false } },
        .{ .name = "router scale only", .sw = .{ .router_norm = .scale_only } },
        .{ .name = "router normalize only", .sw = .{ .router_norm = .normalize_only } },
        .{ .name = "router bias in weights", .sw = .{ .router_weights_use_bias = true } },
        .{ .name = "scale routed+shared", .sw = .{ .scale_on = .routed_plus_shared } },
        .{ .name = "plain layer norms", .sw = .{ .gemma_layer_norms = false } },
        .{ .name = "plain qk norms", .sw = .{ .gemma_qk_norms = false } },
        .{ .name = "plain index norms", .sw = .{ .gemma_index_norms = false } },
        .{ .name = "bf16 norm", .sw = .{ .bf16_norm_out = true } },
        .{ .name = "bf16 proj", .sw = .{ .bf16_proj_out = true } },
        .{ .name = "bf16 rope", .sw = .{ .bf16_rope_out = true } },
        .{ .name = "bf16 attn", .sw = .{ .bf16_attn_out = true } },
        .{ .name = "bf16 act", .sw = .{ .bf16_act_out = true } },
        .{ .name = "bf16 residual", .sw = .{ .bf16_residual = true } },
    };
    for (cases) |cs| {
        const x = try run(a, small(cs.sw), .{}, toks);
        const d = synth.maxDiff(base, x);
        std.testing.expect(d > 1e-9 and std.math.isFinite(d)) catch |e| {
            std.debug.print("switch {s} did not change the output (diff {e})\n", .{ cs.name, d });
            return e;
        };
    }
}

test "switches that need a special scenario: gate clamp (large activations) and the tie rule (all block scores equal)" {
    var ar = arena();
    defer ar.deinit();
    const a = ar.allocator();
    const toks = try synth.tokens(a, 20, 128, 5);
    const big: synth.Options = .{ .gain = 60 };
    const upper = try run(a, small(.{}), big, toks);
    const sym = try run(a, small(.{ .gate_clamp = .symmetric }), big, toks);
    try expect(synth.maxDiff(upper, sym) > 1e-9);
    const flat: synth.Options = .{ .zero_index_q = true };
    const long = try synth.tokens(a, 70, 128, 5);
    const lo = try run(a, small(.{}), flat, long);
    const hi = try run(a, small(.{ .tie = .higher_block }), flat, long);
    try expect(synth.maxDiff(lo, hi) > 1e-9);
    const again = try run(a, small(.{}), flat, long);
    try std.testing.expectEqualSlices(f64, lo, again);
}

test "bf16 residual switch leaves the hidden state on the bf16 grid after attention and after the MoE FFN" {
    var ar = arena();
    defer ar.deinit();
    const a = ar.allocator();
    const m = try synth.model(a, small(.{ .bf16_residual = true }), .{});
    const cache = try ref.Cache.init(a, m.cfg, 2, 4);
    const h = try a.dupe(f64, m.embed[3 * 128 ..][0..128]);
    try ref.attention(m.cfg, m.layers[1], &cache.layers[1], h, 0, a);
    for (h) |v| try std.testing.expectEqual(core.bf16_round.bf16(v), v);
    @memset(h, 0.3);
    try ref.feedForward(m.cfg, m.layers[1], h, a);
    for (h) |v| try std.testing.expectEqual(core.bf16_round.bf16(v), v);
    var any = false;
    for (h) |v| any = any or v != 0.3;
    try expect(any);
}

test "fp16 index input changes the cached index keys (selection may not flip, the keys do)" {
    var ar = arena();
    defer ar.deinit();
    const a = ar.allocator();
    const m = try synth.model(a, small(.{}), .{});
    var m16 = m;
    m16.cfg.sw.fp16_index_input = true;
    const toks = try synth.tokens(a, 5, 128, 3);
    const c0 = try ref.Cache.init(a, m.cfg, 1, 5);
    const c1 = try ref.Cache.init(a, m.cfg, 1, 5);
    for (toks, 0..) |t, pos| {
        const h0 = try a.dupe(f64, m.embed[t * 128 ..][0..128]);
        const h1 = try a.dupe(f64, h0);
        try ref.attention(m.cfg, m.layers[0], &c0.layers[0], h0[0..], pos, a);
        try ref.attention(m16.cfg, m16.layers[0], &c1.layers[0], h1[0..], pos, a);
    }
    try expect(synth.maxDiff(c0.layers[0].ik, c1.layers[0].ik) > 1e-9);
    try std.testing.expectEqualSlices(f64, c0.layers[0].v, c1.layers[0].v);
}
