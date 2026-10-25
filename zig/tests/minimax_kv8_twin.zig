//! Host twins of quant_row8 and tf_mm3_attn's kv8 mode (docs/fp8-kv.md), in the kernels' fp32 order, with mutants.
const std = @import("std");
const core = @import("core");

const mr = core.minimax_ref;
const bf = core.bf16_round;
const inf32 = std.math.inf(f32);
const nan32 = std.math.nan(f32);

pub const Mutant = enum {
    none,
    /// Codes round toward zero instead of to nearest even.
    truncate,
    /// A tie goes up (away from zero) instead of to the even code.
    ties_up,
    /// The scale divides by 240, not 448 (the e4m3fnuz maximum).
    wrong_max,
    /// A NaN or infinity does not poison its row: the row is scaled by its finite maximum.
    nan_ignored,
    /// The scale's 2^-126 floor is missing: a tiny row's inverse overflows.
    no_floor,
    /// The scale is taken over the whole window of rows, not the row (a row's bits would depend on its neighbours).
    batch_amax,
    /// The key is quantized from its fp32 value, not from the bf16 the bf16 cache would hold.
    k_unrounded,
    /// The key's scale is not applied to its score.
    k_scale_missing,
    /// The value's scale is not applied to its weight.
    v_scale_missing,
    /// The value's scale multiplies the softmax sum l as well (it must not).
    v_scale_on_sum,
    /// The attention's key tile depends on how many keys the row sees.
    window_split,
};

/// Code of |a| by the chosen rounding; mutants change the rule, `none` is mr.e4m3Code's.
fn code(x: f32, m: Mutant) u8 {
    if (m != .truncate and m != .ties_up) return mr.e4m3Code(@as(f64, x));
    if (std.math.isNan(x)) return 0x7f;
    const sign: u8 = if (std.math.signbit(x)) 0x80 else 0;
    const a: f64 = @abs(x);
    if (a >= 448) return sign | 0x7e;
    var i: u8 = 0;
    while (i < 0x7e and mr.e4m3Value(i + 1) <= a) i += 1;
    if (m == .truncate or i == 0x7e) return sign | i;
    const mid = (mr.e4m3Value(i) + mr.e4m3Value(i + 1)) / 2;
    return sign | (if (a >= mid) i + 1 else i);
}

/// quant_row8 for a row of 128 fp32 values: writes the codes, returns the scale (`window_amax` is for `.batch_amax`).
pub fn quantRow(x: []const f32, m: Mutant, window_amax: f32, codes: []u8) f32 {
    var amax: f32 = 0;
    var bad = false;
    for (x) |v| {
        if (!std.math.isFinite(v)) bad = true else amax = @max(amax, @abs(v));
    }
    if (m == .batch_amax) amax = @max(amax, window_amax);
    if (bad and m != .nan_ignored) {
        @memset(codes, 0);
        return nan32;
    }
    const top: f32 = if (m == .wrong_max) 240.0 else 448.0;
    var inv: f32 = 1;
    var scale: f32 = 1;
    if (amax > 0) {
        inv = top / amax;
        scale = amax / top;
        if (scale < 0x1p-126 and m != .no_floor) {
            scale = 0x1p-126;
            inv = 1.0 / scale;
        }
    }
    for (codes, x) |*c, v| c.* = code(if (std.math.isFinite(v)) v * inv else v, m);
    return scale;
}

/// A key as the writer stores it: the normed and rotated fp32 value rounded to bf16, then quantized.
pub fn quantKey(o: []const f32, m: Mutant, window_amax: f32, codes: []u8) f32 {
    var r: [128]f32 = undefined;
    for (&r, o) |*d, v| d.* = if (m == .k_unrounded) v else bf.round32(v);
    return quantRow(&r, m, window_amax, codes);
}

/// The rows a launch writes: each quantized alone (the kernel's rule) or, as a mutant, by the window's amax.
pub fn quantWindow(rows: []const [128]f32, m: Mutant, codes: [][128]u8, scales: []f32) void {
    var wmax: f32 = 0;
    for (rows) |r| for (r) |v| {
        if (std.math.isFinite(v)) wmax = @max(wmax, @abs(v));
    };
    for (rows, codes, scales) |r, *c, *s| s.* = quantRow(&r, m, wmax, c);
}

fn widen(c: u8) f32 {
    return @floatCast(mr.e4m3Value(c));
}

fn nmax(a: f32, b: f32) f32 {
    return if (std.math.isNan(a)) a else if (std.math.isNan(b)) b else if (a > b) a else b;
}

/// tf_mm3_attn in kv8 mode for one (row, kv head): q [16][128], codes kc, vc [pos + 1][128], scales ks, vs, ids.
pub fn attend(q: []const f32, kc: []const u8, ks: []const f32, vc: []const u8, vs: []const f32, ids: []const i32, pos: usize, scale: f32, m: Mutant, round: bool, out: []f32) void {
    for (0..16) |h| {
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
            while (k0 < 128) : (k0 += tile) {
                const first = @as(usize, @intCast(b)) * 128 + k0;
                if (first > pos) break;
                const nvalid = @min(tile, pos + 1 - first);
                var sc: [32]f32 = undefined;
                for (0..nvalid) |j| {
                    var a: f32 = 0;
                    for (0..128) |e| a = a + q[h * 128 + e] * widen(kc[(first + j) * 128 + e]);
                    if (m != .k_scale_missing) a = a * ks[first + j];
                    sc[j] = a * scale;
                }
                var mt = sc[0];
                for (sc[1..nvalid]) |s| mt = nmax(mt, s);
                const mn = nmax(mx, mt);
                const alpha: f32 = if (mx == -inf32) 0 else @exp(mx - mn);
                l = l * alpha;
                for (&o) |*x| x.* = x.* * alpha;
                for (0..nvalid) |j| {
                    const p = @exp(sc[j] - mn);
                    l = l + (if (m == .v_scale_on_sum) p * vs[first + j] else p);
                    const ps = if (m == .v_scale_missing) p else p * vs[first + j];
                    for (&o, 0..) |*x, e| x.* = x.* + ps * widen(vc[(first + j) * 128 + e]);
                }
                mx = mn;
            }
        }
        for (0..128) |e| out[h * 128 + e] = if (round) bf.round32(o[e] / l) else o[e] / l;
    }
}
