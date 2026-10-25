//! Host twins of minimax_m3_proj.cu: the same fp32 operation order of the projection and the pick, with mutants.
const std = @import("std");
const core = @import("core");

const c = core.contract.spec;

pub const Mutant = enum {
    none,
    /// The projection's chain assignment follows the row count (a split-K whose depth depends on the call).
    split_by_rows,
    /// The four lane chains are summed in a different order.
    reorder_chains,
    /// The result is rounded to bf16 although fp16 was asked.
    wrong_out,
    /// The last chunk of the reduction is not read.
    drops_tail,
    /// The chains accumulate in fp16.
    half_accum,
    /// The pick: equal logits go to the higher id.
    ties_to_last,
    /// The pick: a NaN beats every number.
    nan_wins,
    /// The pick: the id is the index in the slice, not the vocabulary id.
    forgets_id0,
    /// The pick: the padding after the slice is scanned too.
    scans_padding,
    /// The pick: the first NaN a lane sees stays (the comparison tp.zig had before it ranked NaN last).
    nan_first_stays,
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

pub const OutType = enum { f16, bf16 };

/// One output's fp32 sum as the kernel forms it: x exact bf16 or fp32 values, w exact fp16 values.
pub fn projSum(x: []const f32, w: []const f32, k: usize, rows_in_call: usize, m: Mutant) f32 {
    var lane: [32]f32 = undefined;
    const slices: usize = if (m == .split_by_rows) @max(1, @min(rows_in_call, 4)) else 1;
    const chunks = if (m == .drops_tail) k / 128 - 1 else k / 128;
    for (&lane, 0..) |*s, l| {
        var acc: [4]f32 = @splat(0);
        var j: usize = 0;
        while (j < chunks) : (j += 1) {
            const at = 4 * (l + 32 * j);
            for (0..4) |i| {
                const slot = if (slices > 1) (j % slices) % 4 else i;
                if (m == .half_accum) {
                    const h: f16 = @floatCast(@mulAdd(f32, x[at + i], w[at + i], acc[slot]));
                    acc[slot] = h;
                } else acc[slot] = @mulAdd(f32, x[at + i], w[at + i], acc[slot]);
            }
        }
        s.* = if (m == .reorder_chains) (acc[0] + acc[2]) + (acc[1] + acc[3]) else (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }
    return tree(&lane);
}

/// The bits of the sum's single rounding to fp16 or bf16 (the planted `wrong_out` rounds to the other).
pub fn proj(x: []const f32, w: []const f32, k: usize, rows_in_call: usize, ot: OutType, m: Mutant) u16 {
    const sum = projSum(x, w, k, rows_in_call, m);
    const to_bf = (ot == .bf16) != (m == .wrong_out);
    if (to_bf) return c.toBf16(sum);
    const h: f16 = @floatCast(sum);
    return @bitCast(h);
}

pub const Pick = struct { logit: f32, id: u32 };

/// The kernel's order: true when (bv, bi) beats (av, ai).
fn beats(bv: f32, bi: u32, av: f32, ai: u32, m: Mutant) bool {
    const an = std.math.isNan(av);
    const bn = std.math.isNan(bv);
    if (m == .nan_wins) {
        if (bn) return !an or bi < ai;
        if (an) return false;
    } else if (m == .nan_first_stays) {
        if (an or bn) return false;
    } else {
        if (an and bn) return bi < ai;
        if (bn) return false;
        if (an) return true;
    }
    if (m == .ties_to_last) return bv > av or (bv == av and bi > ai);
    return bv > av or (bv == av and bi < ai);
}

/// tf_mm3_pick on one padded row: 256 threads stride the slice, a warp's lanes meet in an xor tree, then the warps.
pub fn pick(row: []const f32, n: usize, id0: u32, m: Mutant) Pick {
    const nan = std.math.nan(f32);
    const scan = if (m == .scans_padding) row.len else n;
    var t: [256]Pick = undefined;
    for (&t, 0..) |*p, th| {
        p.* = .{ .logit = nan, .id = std.math.maxInt(i32) };
        var i = th;
        while (i < scan) : (i += 256) {
            if (beats(row[i], @intCast(i), p.logit, p.id, m)) p.* = .{ .logit = row[i], .id = @intCast(i) };
        }
    }
    var warp: [8]Pick = undefined;
    for (&warp, 0..) |*w, wi| {
        var lanes: [32]Pick = t[wi * 32 ..][0..32].*;
        var o: usize = 16;
        while (o > 0) : (o >>= 1) {
            var next = lanes;
            for (&next, 0..) |*p, l| {
                const other = lanes[l ^ o];
                if (beats(other.logit, other.id, p.logit, p.id, m)) p.* = other;
            }
            lanes = next;
        }
        w.* = lanes[0];
    }
    var best = warp[0];
    for (warp[1..]) |p| if (beats(p.logit, p.id, best.logit, best.id, m)) {
        best = p;
    };
    return .{ .logit = best.logit, .id = if (m == .forgets_id0) best.id else id0 + best.id };
}
