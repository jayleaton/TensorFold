//! Host twins of the draft's planned engine layout, each with planted mutants (like tests/minimax_*_twin.zig).
const std = @import("std");
// The K/V ring the engine would keep, the two-rank tensor-parallel block, and the per-rank candidate gather.
const core = @import("core");
// dspark_model_test.zig runs each clean twin against dspark_ref.zig and requires every mutant to be caught.
const mr = core.minimax_ref;
const ref = @import("dspark_ref.zig");

// ---- the context ring ---------------------------------------------------------------------------------------------

pub const RingMutant = enum {
    none,
    short_ring, // six rows too few: the oldest rows the window still sees are gone
    rope_at_slot, // K rotated by the ring slot, not the token position
    stale_slot, // a slot already written is never overwritten after the ring wraps
    drop_last_absorb, // the newest absorbed row is not stored
};

/// The engine's per-stream context: the draft layers' K/V of the last `cap` positions at slot (position mod cap).
pub const Ring = struct {
    d: *const ref.Draft,
    cap: usize,
    k: [][]f64,
    v: [][]f64,
    end: usize = 0, // next position to absorb
    m: RingMutant,

    pub fn init(a: std.mem.Allocator, d: *const ref.Draft, cap: usize, m: RingMutant) !Ring {
        const c = d.cfg;
        const eff = if (m == .short_ring) cap - 6 else cap;
        const k = try a.alloc([]f64, c.layers);
        const v = try a.alloc([]f64, c.layers);
        for (k, v) |*kk, *vv| {
            kk.* = try a.alloc(f64, eff * c.kvw());
            vv.* = try a.alloc(f64, eff * c.kvw());
            @memset(kk.*, 0);
            @memset(vv.*, 0);
        }
        return .{ .d = d, .cap = eff, .k = k, .v = v, .m = m };
    }

    /// Takes the tapped row of position `end` (positions are absorbed in order, as the verify keeps them).
    pub fn absorb(r: *Ring, a: std.mem.Allocator, feats: []const f64, last: bool) !void {
        const c = r.d.cfg;
        const pos = r.end;
        r.end += 1;
        if (last and r.m == .drop_last_absorb) return;
        const slot = pos % r.cap;
        if (r.m == .stale_slot and pos >= r.cap) return;
        const st = try a.alloc(f64, c.hidden);
        defer a.free(st);
        try ref.contextState(r.d, a, feats, st);
        for (0..c.layers) |l| {
            const rope_pos = if (r.m == .rope_at_slot) slot else pos;
            try ref.contextKv(r.d, a, l, st, rope_pos, r.k[l][slot * c.kvw() ..][0..c.kvw()], r.v[l][slot * c.kvw() ..][0..c.kvw()]);
        }
    }

    /// The valid rows as a Ctx in position order (what the attention kernel walks through the ring's slot list).
    pub fn ctx(r: *const Ring, a: std.mem.Allocator) !ref.Ctx {
        const c = r.d.cfg;
        const n = @min(r.end, r.cap);
        const first = r.end - n;
        const cx = try ref.Ctx.alloc(a, c, first, n);
        for (0..n) |j| {
            const slot = (first + j) % r.cap;
            for (0..c.layers) |l| {
                @memcpy(cx.k[l][j * c.kvw() ..][0..c.kvw()], r.k[l][slot * c.kvw() ..][0..c.kvw()]);
                @memcpy(cx.v[l][j * c.kvw() ..][0..c.kvw()], r.v[l][slot * c.kvw() ..][0..c.kvw()]);
            }
        }
        return cx;
    }
};

// ---- two-rank tensor parallel block -----------------------------------------------------------------------------

pub const TpMutant = enum {
    none,
    drop_last_head, // the last rank's last query head is not computed
    swap_kv_groups, // the last rank reads the first rank's kv group
    ffn_overlap, // the last rank's FFN cut starts one row early
    drop_exchange, // only rank 0's attention partial is added
};

/// blockForward as `world` ranks run it: kv groups, their query heads and FFN rows cut; partials added in rank order.
pub fn blockForwardTp(d: *const ref.Draft, a: std.mem.Allocator, cx: *const ref.Ctx, anchor: usize, p: usize, world: usize, m: TpMutant) !ref.Block {
    const c = d.cfg;
    const sw = c.sw;
    const hd = c.head_dim;
    const kvw = c.kvw();
    const nb = c.block;
    const rep = c.rep();
    const groups = c.n_kv / world;
    const ffn_cut = c.ffn / world;
    const h = try a.alloc(f64, nb * c.hidden);
    for (0..nb) |i| {
        const tok = if (i == 0) anchor else c.mask_token;
        @memcpy(h[i * c.hidden ..][0..c.hidden], d.embed[tok * c.hidden ..][0..c.hidden]);
    }
    const x = try a.alloc(f64, c.hidden);
    const scale = std.math.pow(f64, @floatFromInt(hd), -0.5);
    for (d.layers, 0..) |ly, l| {
        // attention: every rank computes its own heads for every row, then the partial o_proj sums are exchanged
        const part = try a.alloc(f64, world * nb * c.hidden);
        @memset(part, 0);
        for (0..world) |r| {
            var g0 = r * groups;
            if (m == .swap_kv_groups and r == world - 1) g0 = 0;
            const own_q0 = r * groups * rep; // the q heads this rank multiplies o_proj columns for
            var nq = groups * rep;
            if (m == .drop_last_head and r == world - 1) nq -= 1;
            const q = try a.alloc(f64, nb * nq * hd);
            const kb = try a.alloc(f64, nb * groups * hd);
            const vb = try a.alloc(f64, nb * groups * hd);
            for (0..nb) |i| {
                ref.norm(c, h[i * c.hidden ..][0..c.hidden], ly.input_norm, sw.base.gemma_layer_norms, x);
                const pos = if (sw.block_positions == .consecutive) p + i else p;
                const qi = q[i * nq * hd ..][0 .. nq * hd];
                ly.q.applyRows(own_q0 * hd, (own_q0 + nq) * hd, x, qi);
                ly.k.applyRows(g0 * hd, (g0 + groups) * hd, x, kb[i * groups * hd ..][0 .. groups * hd]);
                ly.v.applyRows(g0 * hd, (g0 + groups) * hd, x, vb[i * groups * hd ..][0 .. groups * hd]);
                for ([_][]f64{ qi, kb[i * groups * hd ..][0 .. groups * hd], vb[i * groups * hd ..][0 .. groups * hd] }) |t| mr.roundAll(sw.base.bf16_proj_out, t);
                ref.heads(c, qi, nq, ly.q_norm, pos);
                ref.heads(c, kb[i * groups * hd ..][0 .. groups * hd], groups, ly.k_norm, pos);
            }
            const attn = try a.alloc(f64, nb * nq * hd);
            for (0..nb) |i| {
                const rp = if (sw.block_positions == .consecutive) p + i else p;
                for (0..nq) |hh| {
                    const g = hh / rep; // the rank-local kv group
                    const qh = q[(i * nq + hh) * hd ..][0..hd];
                    const total = cx.n + nb;
                    const s = try a.alloc(f64, total);
                    var best: f64 = -std.math.inf(f64);
                    for (0..total) |j| {
                        s[j] = -std.math.inf(f64);
                        var key: []const f64 = undefined;
                        if (j < cx.n) {
                            if (!ref.sees(sw, c.window, cx.first + j, rp, p)) continue;
                            key = cx.k[l][j * kvw + (g0 + g) * hd ..][0..hd];
                        } else {
                            const mm = j - cx.n;
                            if (sw.block_attention == .causal and mm > i) continue;
                            key = kb[mm * groups * hd + g * hd ..][0..hd];
                        }
                        var dot: f64 = 0;
                        for (qh, key) |u, w| dot += u * w;
                        s[j] = dot * scale;
                        best = @max(best, s[j]);
                    }
                    var z: f64 = 0;
                    const out = attn[(i * nq + hh) * hd ..][0..hd];
                    @memset(out, 0);
                    for (0..total) |j| {
                        if (s[j] == -std.math.inf(f64)) continue;
                        const w = @exp(s[j] - best);
                        z += w;
                        const val = if (j < cx.n) cx.v[l][j * kvw + (g0 + g) * hd ..][0..hd] else vb[(j - cx.n) * groups * hd + g * hd ..][0..hd];
                        for (out, val) |*o, vv| o.* += w * vv;
                    }
                    for (out) |*o| o.* /= z;
                }
            }
            mr.roundAll(sw.base.bf16_attn_out, attn);
            for (0..nb) |i| ly.o.applyCols(own_q0 * hd, attn[i * nq * hd ..][0 .. nq * hd], part[(r * nb + i) * c.hidden ..][0..c.hidden]);
        }
        // exchange: partials added in rank order 0 then 1, every rank holds the same sum
        for (0..nb) |i| {
            const hi = h[i * c.hidden ..][0..c.hidden];
            for (0..c.hidden) |e| {
                var s: f64 = 0;
                for (0..world) |r| {
                    if (m == .drop_exchange and r != 0) continue;
                    s += part[(r * nb + i) * c.hidden + e];
                }
                hi[e] = mr.round16(sw.base.bf16_residual, hi[e] + s);
            }
        }
        // FFN
        @memset(part, 0);
        for (0..world) |r| {
            var lo = r * ffn_cut;
            if (m == .ffn_overlap and r == world - 1) lo -= 1;
            const hi_row = (r + 1) * ffn_cut;
            const g = try a.alloc(f64, hi_row - lo);
            const u = try a.alloc(f64, hi_row - lo);
            for (0..nb) |i| {
                ref.norm(c, h[i * c.hidden ..][0..c.hidden], ly.post_norm, sw.base.gemma_layer_norms, x);
                ly.gate.applyRows(lo, hi_row, x, g);
                ly.up.applyRows(lo, hi_row, x, u);
                for (g, u) |*gv, uv| gv.* = mr.round16(sw.base.bf16_act_out, mr.swigluoai(gv.*, uv, c.alpha, c.limit, sw.base));
                ly.down.applyCols(lo, g, part[(r * nb + i) * c.hidden ..][0..c.hidden]);
            }
        }
        for (0..nb) |i| {
            const hi = h[i * c.hidden ..][0..c.hidden];
            for (0..c.hidden) |e| {
                var s: f64 = 0;
                for (0..world) |r| s += part[(r * nb + i) * c.hidden + e];
                hi[e] = mr.round16(sw.base.bf16_residual, hi[e] + s);
            }
        }
    }
    const post = try a.alloc(f64, nb * c.hidden);
    for (0..nb) |i| ref.norm(c, h[i * c.hidden ..][0..c.hidden], d.final_norm, sw.gemma_final_norm, post[i * c.hidden ..][0..c.hidden]);
    return .{ .pre = h, .post = post };
}

// ---- the head's per-rank candidates -----------------------------------------------------------------------------

pub const GatherMutant = enum {
    none,
    drop_last_rank, // the last rank's candidates never arrive
    ids_not_offset, // a rank's local candidate ids are not moved to global ids
    first_lowest, // the merge keeps the first of equal values instead of the lowest id
};

/// One pick as the engine makes it: per-rank top `per` base logits gathered, Markov bias on the gathered ids only.
pub fn gatherPick(d: *const ref.Draft, base: []const f64, prev: ?usize, world: usize, per: usize, m: GatherMutant, a: std.mem.Allocator) !usize {
    const c = d.cfg;
    const slice = c.vocab / world;
    var ids: std.ArrayList(usize) = .empty;
    defer ids.deinit(a);
    for (0..world) |r| {
        if (m == .drop_last_rank and r == world - 1) continue;
        const lo = r * slice;
        const taken = try a.alloc(bool, slice);
        defer a.free(taken);
        @memset(taken, false);
        for (0..per) |_| {
            var top: ?usize = null;
            for (0..slice) |i| {
                if (taken[i]) continue;
                if (top == null or base[lo + i] > base[lo + top.?]) top = i;
            }
            taken[top.?] = true;
            try ids.append(a, if (m == .ids_not_offset) top.? else lo + top.?);
        }
    }
    var best: ?usize = null;
    var bz: f64 = 0;
    for (ids.items) |id| {
        const z = base[id] + if (prev) |pv| ref.markovBias(d, pv, id) else 0;
        const better = best == null or z > bz or (z == bz and m != .first_lowest and id < best.?);
        if (better) {
            best = id;
            bz = z;
        }
    }
    return best.?;
}
