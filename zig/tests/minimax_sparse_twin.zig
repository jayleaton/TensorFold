//! Host twins of minimax_m3_{index,select,attn}.cu (same fp32 order, no fma) and planted mutants each breaking a rule.
const std = @import("std");
const core = @import("core");
const geom = @import("mm3_geom");

const bf = core.bf16_round;
const nan32 = std.math.nan(f32);
const inf32 = std.math.inf(f32);

pub const Mutant = enum {
    none,
    /// The block maximum skips the last valid key (a partial tail dropped).
    drop_tail,
    /// A NaN key score poisons the block maximum instead of being dropped by fmax.
    nan_poison,
    /// The own block is not forced.
    no_forcing,
    /// The tie rule is flipped.
    tie_flipped,
    /// The attention's key tile depends on how many keys the row sees (a window-dependent split).
    window_split,
};

/// One score as tf_mm3_index computes it: 32 lanes of 4 dims in order, then the xor tree 16, 8, 4, 2, 1; times scale.
pub fn score(iq: []const f32, ik: []const f32, scale: f32) f32 {
    var lane: [32]f32 = undefined;
    for (&lane, 0..) |*s, l| {
        var a = iq[4 * l] * ik[4 * l];
        a = a + iq[4 * l + 1] * ik[4 * l + 1];
        a = a + iq[4 * l + 2] * ik[4 * l + 2];
        a = a + iq[4 * l + 3] * ik[4 * l + 3];
        s.* = a;
    }
    var o: usize = 16;
    while (o > 0) : (o >>= 1) {
        var next: [32]f32 = undefined;
        for (&next, 0..) |*n, l| n.* = lane[l] + lane[l ^ o];
        lane = next;
    }
    return lane[0] * scale;
}

/// Block maxima of one index head over ik [pos + 1, 128]; returns the block count.
pub fn blockMax(iq: []const f32, ik: []const f32, pos: usize, scale: f32, m: Mutant, out: []f32) usize {
    const nb = geom.blocks(pos);
    for (out[0..nb], 0..) |*o, b| {
        var valid = @min(geom.block, pos + 1 - b * geom.block);
        if (m == .drop_tail and valid > 0) valid -= 1;
        var best: f32 = -inf32;
        for (0..valid) |j| {
            const s = score(iq, ik[(b * geom.block + j) * 128 ..][0..128], scale);
            if (m == .nan_poison) {
                if (std.math.isNan(s) or std.math.isNan(best)) best = nan32 else best = @max(best, s);
            } else best = @max(best, s); // fmaxf: a NaN operand yields the other
        }
        o.* = best;
    }
    return nb;
}

fn forced(b: i32, own: i32, r: geom.Rules) bool {
    const init = if (r.init_inclusive) b <= r.init_n else b < r.init_n;
    const local = if (r.local_trailing) (b <= own and own - b < r.local_n) else (r.local_n > 0 and b == own);
    return init or local;
}

fn before(sa: f32, ia: i32, sb: f32, ib: i32, higher: bool) bool {
    return sa > sb or (sa == sb and (if (higher) ia > ib else ia < ib));
}

/// tf_mm3_select for one (row, head): ascending ids, -1 padded.
pub fn select(bmax: []const f32, pos: usize, rules: geom.Rules, m: Mutant, out: *[geom.topk]i32) void {
    const nb: i32 = @intCast(geom.blocks(pos));
    const own = nb - 1;
    var r = rules;
    if (m == .no_forcing) {
        r.local_n = 0;
        r.init_n = 0;
        r.init_inclusive = false;
    }
    if (m == .tie_flipped) r.higher_tie = !r.higher_tie;
    if (nb <= geom.topk) {
        for (out, 0..) |*o, t| o.* = if (t < nb) @intCast(t) else -1;
        return;
    }
    var chosen: [geom.topk]i32 = undefined;
    var prev_s: f32 = 0;
    var prev_i: i32 = 0;
    for (0..geom.topk) |k| {
        var bs: f32 = -inf32;
        var bi: i32 = -1;
        var b: i32 = 0;
        while (b < nb) : (b += 1) {
            var s: f32 = if (forced(b, own, r)) inf32 else bmax[@intCast(b)];
            if (std.math.isNan(s)) s = -inf32;
            if (k > 0 and !before(prev_s, prev_i, s, b, r.higher_tie)) continue;
            if (bi < 0 or before(s, b, bs, bi, r.higher_tie)) {
                bs = s;
                bi = b;
            }
        }
        chosen[k] = bi;
        prev_s = bs;
        prev_i = bi;
    }
    for (&chosen, 0..) |c, t| {
        var rank: usize = 0;
        for (chosen) |u| rank += @intFromBool(u < chosen[t]);
        out[rank] = c;
    }
}

/// tf_mm3_attn for one (row, kv head) into bf16-rounded out [16][128]; `.window_split`: 16-key tiles under 1024 keys.
pub fn attend(q: []const f32, k: []const f32, v: []const f32, ids: []const i32, pos: usize, scale: f32, m: Mutant, out: []f32) void {
    attendImpl(q, k, v, ids, pos, scale, m, true, out);
}

/// The same with the fp32 result before the bf16 store, where a tile change shows.
pub fn attendRaw(q: []const f32, k: []const f32, v: []const f32, ids: []const i32, pos: usize, scale: f32, m: Mutant, out: []f32) void {
    attendImpl(q, k, v, ids, pos, scale, m, false, out);
}

fn attendImpl(q: []const f32, k: []const f32, v: []const f32, ids: []const i32, pos: usize, scale: f32, m: Mutant, round: bool, out: []f32) void {
    for (0..geom.group) |h| {
        var o: [128]f32 = @splat(0);
        var mx: f32 = -inf32;
        var l: f32 = 0;
        var seen: usize = 0;
        for (ids) |b| if (b >= 0) {
            seen += 128;
        };
        const tile: usize = if (m == .window_split and seen < 1024) 16 else 32;
        for (ids) |b| {
            if (b < 0) break;
            var k0: usize = 0;
            while (k0 < geom.block) : (k0 += tile) {
                const first = @as(usize, @intCast(b)) * geom.block + k0;
                if (first > pos) break;
                const nvalid = @min(tile, pos + 1 - first);
                var sc: [32]f32 = undefined;
                for (0..nvalid) |j| {
                    var a: f32 = 0;
                    for (0..128) |e| a = a + q[h * 128 + e] * k[(first + j) * 128 + e];
                    sc[j] = a * scale;
                }
                var mt = sc[0];
                for (sc[1..nvalid]) |s| mt = if (std.math.isNan(mt)) mt else if (std.math.isNan(s)) s else if (mt > s) mt else s;
                const mn = if (std.math.isNan(mx)) mx else if (std.math.isNan(mt)) mt else if (mx > mt) mx else mt;
                const alpha: f32 = if (mx == -inf32) 0 else @exp(mx - mn);
                l = l * alpha;
                for (&o) |*x| x.* = x.* * alpha;
                for (0..nvalid) |j| {
                    const p = @exp(sc[j] - mn);
                    l = l + p;
                    for (&o, 0..) |*x, e| x.* = x.* + p * v[(first + j) * 128 + e];
                }
                mx = mn;
            }
        }
        for (0..128) |e| out[h * 128 + e] = if (round) bf.round32(o[e] / l) else o[e] / l;
    }
}
