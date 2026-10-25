//! Host twins of minimax_m3_rows.cu: the same fp32 operation order (fma only in the router) and planted mutants.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");

const geom = cuda.minimax_m3_rows.geom;
const c = core.contract.spec;

pub const Mutant = enum {
    none,
    /// The weight offset is ignored: raw weights are used as if they were the folded 1 + w.
    no_plus_one,
    /// RoPE pairs (2i, 2i + 1) where the switch says (i, i + rot/2).
    wrong_layout,
    /// RoPE covers all 128 dims instead of the first 64.
    rope_all_dims,
    /// The rotation runs before the norm.
    rope_first,
    /// The angle is computed in fp32.
    f32_angle,
    /// K goes to the next token slot of its page.
    next_slot,
    /// The router's chunk count follows the row count (a split-K whose depth depends on the call).
    split_by_rows,
    /// The router's four lane chains are summed in a different order.
    reorder_chains,
};

pub const Opts = struct {
    rot: usize = 64,
    theta: f64 = 1e7,
    eps: f32 = 1e-6,
    w_add: f32 = 0,
    w_add_index: f32 = 0,
    flags: geom.Flags = .{},
};

fn tree(v: *[32]f32) f32 {
    var o: usize = 16;
    while (o > 0) : (o >>= 1) {
        var next: [32]f32 = undefined;
        for (&next, 0..) |*n, l| n.* = v[l] + v[l ^ o];
        v.* = next;
    }
    return v[0];
}

/// The kernel's norm of one head: lane l squares its 4 values in order, then the xor tree; y = (x * inv) * (w + add).
pub fn norm(x: *[128]f32, w: *const [128]f32, o: Opts, add: f32) void {
    var lane: [32]f32 = undefined;
    for (&lane, 0..) |*s, l| {
        var a = x[4 * l] * x[4 * l];
        a = a + x[4 * l + 1] * x[4 * l + 1];
        a = a + x[4 * l + 2] * x[4 * l + 2];
        a = a + x[4 * l + 3] * x[4 * l + 3];
        s.* = a;
    }
    const ss = tree(&lane);
    const inv: f32 = 1.0 / @sqrt(ss / 128.0 + o.eps);
    for (x, w) |*v, wv| v.* = v.* * inv * (wv + add);
    if (o.flags.round_norm) for (x) |*v| {
        v.* = c.fromBf16(c.toBf16(v.*));
    };
}

/// The cos and sin of pair i at `pos`: float64 angle and sincos rounded once to fp32 (the mutant rounds the angle).
pub fn angle(pos: i32, i: usize, rot: usize, theta: f64, m: Mutant) [2]f32 {
    const inv = std.math.pow(f64, theta, -2 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(rot)));
    var ang: f64 = @as(f64, @floatFromInt(pos)) * inv;
    if (m == .f32_angle) ang = @as(f32, @floatCast(ang));
    return .{ @floatCast(@cos(ang)), @floatCast(@sin(ang)) };
}

/// The kernel's `rotated` for every dim of a staged head.
pub fn rope(x: *[128]f32, pos: i32, rot: usize, theta: f64, gptj: bool, m: Mutant) void {
    var out = x.*;
    const r = if (m == .rope_all_dims) 128 else rot;
    const g = if (m == .wrong_layout) !gptj else gptj;
    const half = r / 2;
    for (0..r) |d| {
        const i = if (g) d >> 1 else if (d < half) d else d - half;
        const ia = if (g) 2 * i else i;
        const ib = if (g) ia + 1 else i + half;
        const cs = angle(pos, i, r, theta, m);
        const a = x[ia];
        const b = x[ib];
        const first = if (g) d & 1 == 0 else d < half;
        out[d] = if (first) a * cs[0] - b * cs[1] else b * cs[0] + a * cs[1];
    }
    x.* = out;
}

pub const Heads = struct { nq: usize = 32, niq: usize = 2, kvh: usize = 2, ik: bool = true };

/// Per-head weights, fp32 [128] each.
pub const Norms = struct { q: [128]f32, k: [128]f32, iq: [128]f32, ik: [128]f32 };

/// The projections of the rows of a call as exact fp32 values (bf16, or fp16 for the index heads), [rows][width].
pub const Inputs = struct { q: []const f32, k: []const f32, v: []const f32, iq: []const f32, ik: []const f32 };

/// Where a call writes: bf16 bit scratch for q and iq, the pool as bf16 bits, and the tables.
pub const Target = struct {
    q: []u16,
    iq: []u16,
    pool: []u16,
    tables: []const i32,
    max_pages: usize,
    row_tab: ?[]const i32,
    pos: ?[]const i32,
    pos0: i32,
    page_bytes: usize,
    layer_off: usize,
};

/// One head: norm (if on), rotation (if on), bf16 bits out.
fn head(src: []const f32, w: *const [128]f32, on_norm: bool, add: f32, on_rope: bool, pos: i32, o: Opts, m: Mutant, out: *[128]u16) void {
    var x: [128]f32 = undefined;
    @memcpy(&x, src[0..128]);
    if (m == .rope_first and on_rope) rope(&x, pos, o.rot, o.theta, o.flags.gptj, .none);
    if (on_norm) norm(&x, w, o, if (m == .no_plus_one) 0 else add);
    if (m != .rope_first and on_rope) rope(&x, pos, o.rot, o.theta, o.flags.gptj, m);
    for (out, x) |*b, v| b.* = c.toBf16(v);
}

/// tf_mm3_rows over `rows` rows; returns the status word.
pub fn call(rows: usize, h: Heads, o: Opts, in: Inputs, nm: *const Norms, t: Target, m: Mutant) i32 {
    var status: i32 = 0;
    const f = o.flags;
    for (0..rows) |r| {
        const p: i32 = if (t.pos) |a| a[r] else t.pos0 + @as(i32, @intCast(r));
        if (p < 0) continue;
        const blk: usize = @intCast(@divTrunc(p, 128));
        const tok: usize = @intCast(@mod(p, 128));
        const off: usize = if (t.row_tab) |a| @intCast(a[r]) else 0;
        var page: i64 = -1;
        if (blk < t.max_pages) page = t.tables[off + blk] else status |= geom.status_past_table;
        if (blk < t.max_pages and page < 0) status |= geom.status_no_page;
        const base: usize = if (page >= 0) (@as(usize, @intCast(page)) * t.page_bytes + t.layer_off) / 2 else 0;
        var buf: [128]u16 = undefined;
        for (0..h.nq) |i| {
            head(in.q[(r * h.nq + i) * 128 ..], &nm.q, f.norm_qk, o.w_add, true, p, o, m, &buf);
            @memcpy(t.q[(r * h.nq + i) * 128 ..][0..128], &buf);
        }
        for (0..h.niq) |i| {
            head(in.iq[(r * h.niq + i) * 128 ..], &nm.iq, f.norm_index, o.w_add_index, f.index_rope, p, o, m, &buf);
            @memcpy(t.iq[(r * h.niq + i) * 128 ..][0..128], &buf);
        }
        if (page < 0) continue;
        const kvh = h.kvh;
        for (0..kvh) |i| {
            head(in.k[(r * kvh + i) * 128 ..], &nm.k, f.norm_qk, o.w_add, true, p, o, m, &buf);
            const tk = if (m == .next_slot) (tok + 1) % 128 else tok;
            @memcpy(t.pool[base + elem(kvh, 0, i, tk) ..][0..128], &buf);
            for (0..128) |e| buf[e] = c.toBf16(in.v[(r * kvh + i) * 128 + e]);
            @memcpy(t.pool[base + elem(kvh, 1, i, tok) ..][0..128], &buf);
        }
        if (h.ik) {
            head(in.ik[r * 128 ..], &nm.ik, f.norm_index, o.w_add_index, f.index_rope, p, o, m, &buf);
            @memcpy(t.pool[base + ikElem(kvh, tok) ..][0..128], &buf);
        }
    }
    return status;
}

/// minimax_m3_sparse.cuh kv_elem and ik_elem, in bf16 elements.
pub fn elem(kvh: usize, which: usize, h: usize, t: usize) usize {
    return (which * kvh + h) * 128 * 128 + t * 128;
}

pub fn ikElem(kvh: usize, t: usize) usize {
    return 2 * kvh * 128 * 128 + t * 128;
}

/// One router output as the kernel sums it: lane fma chains over chunks l + 32 j, then the 4 chains, then the tree.
pub fn router(x: []const f32, w: []const f32, k: usize, rows_in_call: usize, m: Mutant) f32 {
    var lane: [32]f32 = undefined;
    const slices: usize = if (m == .split_by_rows) @max(1, @min(rows_in_call, 4)) else 1;
    for (&lane, 0..) |*s, l| {
        var acc: [4]f32 = @splat(0);
        var j: usize = 0;
        while (j < k / 128) : (j += 1) {
            const at = 4 * (l + 32 * j);
            for (0..4) |i| {
                const slot = if (slices > 1) (j % slices) % 4 else i;
                acc[slot] = @mulAdd(f32, x[at + i], w[at + i], acc[slot]);
            }
        }
        s.* = if (m == .reorder_chains) (acc[0] + acc[2]) + (acc[1] + acc[3]) else (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }
    return tree(&lane);
}
