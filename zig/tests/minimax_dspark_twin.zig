//! Host twins of minimax_m3_dspark.cu: the kernels' fp32 operation orders (norm, head norm + RoPE, chunked attention
const std = @import("std");
const core = @import("core");

const c = core.contract.spec;

pub const Mutant = enum {
    none,
    /// The sum of squares runs in one sequential chain instead of the 256-thread, xor-tree order.
    sequential_sum,
    /// The Gemma offset is dropped (w instead of 1 + w).
    no_offset,
    /// The rotation angle is computed in fp32 (off by 1e-2 at position 1e6).
    angle_f32,
    /// Rotates (2i, 2i + 1) pairs instead of (i, i + rot/2).
    interleaved,
    /// The ring slot is the position, not the position modulo the ring.
    slot_unwrapped,
    /// The merge weights every chunk alike.
    merge_unweighted,
    /// The merge forgets the block's own chunk.
    merge_drops_block,
    /// Each context chunk drops its last key.
    drops_last_key,
    /// The window starts one key early.
    window_early,
    /// The window start ignores the ring's `lo`.
    ignores_lo,
    /// The block rows see only earlier block rows.
    causal_block,
    /// The scores are not scaled.
    unscaled,
    /// Equal keys are collected from the highest index.
    ties_high,
    /// A NaN ranks above every number.
    nan_high,
    /// -0 and +0 are different keys.
    signed_zero,
    /// The padding after the valid logits is scanned.
    reads_padding,
    /// An equal score goes to the higher id.
    chain_ties_high,
    /// The previous token stays the anchor.
    chain_prev_anchor,
    /// Only the first rank's candidates are scored.
    chain_first_rank,
    /// A NaN score beats every number.
    chain_nan_wins,
};

pub fn tree32(v: *[32]f32) f32 {
    var o: usize = 16;
    while (o > 0) : (o >>= 1) {
        var next: [32]f32 = undefined;
        for (&next, 0..) |*n, l| n.* = v[l] + v[l ^ o];
        v.* = next;
    }
    return v[0];
}

// ---- norm

/// tf_mm3_ds_norm on one row: x exact bf16 or fp32 values, w exact bf16 values; returns the bf16 bits.
pub fn norm(x: []const f32, w: []const f32, eps: f32, gemma: bool, out: []u16, m: Mutant) void {
    const k = x.len;
    var tot: f32 = 0;
    if (m == .sequential_sum) {
        for (x) |v| tot += v * v;
    } else {
        var thread: [256]f32 = undefined;
        for (&thread, 0..) |*s, t| {
            s.* = 0;
            var i = t;
            while (i < k) : (i += 256) s.* += x[i] * x[i];
        }
        var warp: [8]f32 = undefined;
        for (&warp, 0..) |*s, wi| {
            var lanes: [32]f32 = thread[wi * 32 ..][0..32].*;
            s.* = tree32(&lanes);
        }
        tot = warp[0];
        for (warp[1..]) |s| tot += s;
    }
    const inv = 1 / @sqrt(tot / @as(f32, @floatFromInt(k)) + eps);
    const add: f32 = if (gemma and m != .no_offset) 1 else 0;
    for (out, x, w) |*o, v, wv| o.* = c.toBf16((v * inv) * (wv + add));
}

// ---- per-head norm + RoPE

/// One 128-wide head: norm (4 values a lane, xor tree), then RoPE on the first `rot` dims at `pos`; bf16 bits.
pub fn head(x: *const [128]f32, w: *const [128]f32, eps: f32, gemma: bool, rot: usize, theta: f64, pos: usize, out: *[128]u16, m: Mutant) void {
    var lanes: [32]f32 = undefined;
    for (&lanes, 0..) |*s, l| {
        var ss = x[4 * l] * x[4 * l];
        for (1..4) |i| ss += x[4 * l + i] * x[4 * l + i];
        s.* = ss;
    }
    const inv = 1 / @sqrt(tree32(&lanes) / 128.0 + eps);
    var st: [128]f32 = undefined;
    const add: f32 = if (gemma) 1 else 0;
    for (&st, x, w) |*o, v, wv| o.* = (v * inv) * (wv + add);
    const half = rot / 2;
    var cs: [2][64]f32 = undefined;
    for (0..half) |i| {
        const freq = std.math.pow(f64, theta, -2.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(rot)));
        if (m == .angle_f32) {
            const ang: f32 = @as(f32, @floatFromInt(pos)) * @as(f32, @floatCast(freq));
            cs[0][i] = @cos(ang);
            cs[1][i] = @sin(ang);
        } else {
            const ang = @as(f64, @floatFromInt(pos)) * freq;
            cs[0][i] = @floatCast(@cos(ang));
            cs[1][i] = @floatCast(@sin(ang));
        }
    }
    for (0..128) |d| {
        var v = st[d];
        if (d < rot) {
            if (m == .interleaved) {
                const i = d / 2;
                const a = st[2 * i];
                const b = st[2 * i + 1];
                v = if (d % 2 == 0) a * cs[0][i] - b * cs[1][i] else b * cs[0][i] + a * cs[1][i];
            } else {
                const i = if (d < half) d else d - half;
                const a = st[i];
                const b = st[i + half];
                v = if (d < half) a * cs[0][i] - b * cs[1][i] else b * cs[0][i] + a * cs[1][i];
            }
        }
        out[d] = c.toBf16(v);
    }
}

/// The ring slot of a position.
pub fn slot(pos: usize, ring: usize, m: Mutant) usize {
    return if (m == .slot_unwrapped) pos else pos % ring;
}

// ---- attention

/// The first context position row `row` (position p + row) sees.
pub fn windowStart(p: usize, row: usize, window: usize, lo: usize, m: Mutant) usize {
    var w0 = if (p + row + 1 > window) p + row + 1 - window else 0;
    if (m == .window_early and w0 > 0) w0 -= 1;
    return if (m == .ignores_lo) w0 else @max(lo, w0);
}

const Partial = struct { m: f32, l: f32, acc: [128]f32 };

fn partial(q: *const [128]f32, keys: []const [128]f32, vals: []const [128]f32, ok: []const bool, scale: f32, mut: Mutant) Partial {
    var s: [128]f32 = undefined;
    var mx: f32 = -std.math.inf(f32);
    for (keys, 0..) |key, j| {
        s[j] = -std.math.inf(f32);
        if (!ok[j]) continue;
        var acc: f32 = 0;
        for (0..128) |d| acc = @mulAdd(f32, q[d], key[d], acc);
        s[j] = if (mut == .unscaled) acc else acc * scale;
        mx = @max(mx, s[j]);
    }
    var lane: [32]f32 = @splat(0);
    var p: [128]f32 = undefined;
    for (keys, 0..) |_, j| {
        p[j] = if (s[j] == -std.math.inf(f32)) 0 else @floatCast(@exp(@as(f64, s[j] - mx)));
        lane[j % 32] += p[j];
    }
    var out: Partial = .{ .m = mx, .l = tree32(&lane), .acc = @splat(0) };
    for (keys, vals, 0..) |_, v, j| {
        for (0..128) |d| out.acc[d] = @mulAdd(f32, p[j], v[d], out.acc[d]);
    }
    return out;
}

/// One query head over `ctx` context keys (already the row's visible ones, in position order) and `blk` block keys.
pub fn attend(q: *const [128]f32, ck: []const [128]f32, cv: []const [128]f32, bk: []const [128]f32, bv: []const [128]f32, row: usize, scale: f32, out: *[128]f32, mut: Mutant) void {
    var parts: [64]Partial = undefined;
    var n: usize = 0;
    var at: usize = 0;
    var ok: [128]bool = undefined;
    while (at < ck.len) : (at += 128) {
        const len = @min(128, ck.len - at);
        const used = if (mut == .drops_last_key and len > 1) len - 1 else len;
        @memset(&ok, true);
        parts[n] = partial(q, ck[at..][0..used], cv[at..][0..used], ok[0..used], scale, mut);
        n += 1;
    }
    for (0..bk.len) |j| ok[j] = mut != .causal_block or j <= row;
    parts[n] = partial(q, bk, bv, ok[0..bk.len], scale, mut);
    n += 1;
    var mx = parts[0].m;
    for (parts[1..n]) |p| mx = @max(mx, p.m);
    if (mut == .merge_drops_block) n -= 1;
    var den: f32 = 0;
    var num: [128]f32 = @splat(0);
    for (parts[0..n]) |p| {
        const w: f32 = if (mut == .merge_unweighted) 1 else if (p.m == -std.math.inf(f32)) 0 else @floatCast(@exp(@as(f64, p.m - mx)));
        den += w * p.l;
        for (&num, p.acc) |*a, v| a.* += w * v;
    }
    for (out, num) |*o, v| o.* = v / den;
}

// ---- top-C

pub fn orderKey(v: f32, m: Mutant) u32 {
    if (std.math.isNan(v)) return if (m == .nan_high) 0xffff_ffff else 0;
    var u: u32 = @bitCast(v);
    if (u == 0x8000_0000 and m != .signed_zero) u = 0;
    return if (u & 0x8000_0000 != 0) ~u else u | 0x8000_0000;
}

pub const Pair = struct { logit: f32, id: i32 };

/// tf_mm3_ds_topc on one row: radix-select the c-th key, collect the keys above it and the first equal ones in index
pub fn topc(row: []const f32, n_valid: usize, c_want: usize, id0: u32, out: []Pair, m: Mutant) void {
    const n = if (m == .reads_padding) row.len else n_valid;
    const want = @min(c_want, n);
    var prefix: u32 = 0;
    var rem: u32 = @intCast(want);
    if (want > 0) {
        var pass: u5 = 0;
        while (pass < 4) : (pass += 1) {
            const shift: u5 = @intCast(24 - 8 * @as(u32, pass));
            const mask: u32 = if (pass == 0) 0 else ~@as(u32, 0) << @intCast(32 - 8 * @as(u32, pass));
            var hist: [256]u32 = @splat(0);
            for (row[0..n]) |v| {
                const k = orderKey(v, m);
                if (k & mask == prefix) hist[(k >> shift) & 255] += 1;
            }
            var acc: u32 = 0;
            var b: usize = 255;
            while (b > 0) : (b -= 1) {
                if (acc + hist[b] >= rem) break;
                acc += hist[b];
            }
            prefix |= @as(u32, @intCast(b)) << shift;
            rem -= acc;
        }
    }
    var picked: [128]u64 = undefined;
    var np: usize = 0;
    var gt: usize = 0;
    for (row[0..n]) |v| gt += @intFromBool(orderKey(v, m) > prefix);
    const need_eq = want - gt;
    var eq_seen: usize = 0;
    var eq_total: usize = 0;
    for (row[0..n]) |v| eq_total += @intFromBool(orderKey(v, m) == prefix);
    for (row[0..n], 0..) |v, i| {
        const k = orderKey(v, m);
        if (want == 0) break;
        if (k > prefix) {
            picked[np] = (@as(u64, k) << 32) | (0xffff_ffff - @as(u32, @intCast(i)));
            np += 1;
        } else if (k == prefix) {
            const rank = if (m == .ties_high) eq_total - 1 - eq_seen else eq_seen;
            if (rank < need_eq) {
                picked[np] = (@as(u64, k) << 32) | (0xffff_ffff - @as(u32, @intCast(i)));
                np += 1;
            }
            eq_seen += 1;
        }
    }
    std.mem.sort(u64, picked[0..np], {}, std.sort.desc(u64));
    for (out, 0..) |*o, j| {
        if (j < np) {
            const i = 0xffff_ffff - @as(u32, @truncate(picked[j]));
            o.* = .{ .logit = row[i], .id = @intCast(id0 + i) };
        } else o.* = .{ .logit = -std.math.inf(f32), .id = -1 };
    }
}

// ---- the Markov chain

fn beats(bv: f32, bi: i32, av: f32, ai: i32, m: Mutant) bool {
    const an = std.math.isNan(av);
    const bn = std.math.isNan(bv);
    if (m == .chain_nan_wins) {
        if (bn) return !an or bi < ai;
        if (an) return false;
    } else {
        if (an and bn) return bi < ai;
        if (bn) return false;
        if (an) return true;
    }
    if (m == .chain_ties_high) return bv > av or (bv == av and bi > ai);
    return bv > av or (bv == av and bi < ai);
}

/// dot over `rank` dims, 32 lanes each owning rank / 32 consecutive dims, fma chain, xor tree.
pub fn dotRank(a: []const f32, b: []const f32) f32 {
    const per = a.len / 32;
    var lane: [32]f32 = undefined;
    for (&lane, 0..) |*s, l| {
        var acc: f32 = 0;
        for (0..per) |d| acc = @mulAdd(f32, a[l * per + d], b[l * per + d], acc);
        s.* = acc;
    }
    return tree32(&lane);
}

/// The token of one step: the best of `ranks` x `cand` candidates by z = base + dot(w2[id], w1[prev]), -1 if none
pub fn step(base: []const f32, ids: []const i32, w1prev: []const f32, w2: []const []const f32, cand: usize, m: Mutant) i32 {
    var bv = std.math.nan(f32);
    var bi: i32 = std.math.maxInt(i32);
    for (ids, base, 0..) |id, b, i| {
        if (id < 0) continue;
        if (m == .chain_first_rank and i >= cand) continue;
        const z = b + dotRank(w2[i], w1prev);
        if (beats(z, id, bv, bi, m)) {
            bv = z;
            bi = id;
        }
    }
    return if (bi == std.math.maxInt(i32)) -1 else bi;
}

/// The chain's previous-token rule: the anchor first, then the last token (the anchor again after -1).
pub fn prevAfter(token: i32, anchor: u32, m: Mutant) u32 {
    if (m == .chain_prev_anchor or token < 0) return anchor;
    return @intCast(token);
}
