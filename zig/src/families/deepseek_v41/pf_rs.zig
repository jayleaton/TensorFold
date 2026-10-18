//! TF_DSV41_PF_RS: a prefill exchange as a rank-order reduce-scatter of bf16 partials, then a gather of the sums.
const std = @import("std");

/// The rows each rank gathers (its share's slot, the last share's tail past n unread).
pub fn slotRows(n: usize, W: usize) usize {
    return (n + W - 1) / W;
}

/// Whether the gathered sums and the peers' terms both fit in `into`'s W n rows (two or more ranks).
pub fn fits(n: usize, W: usize) bool {
    return W >= 2 and n > 0 and (2 * W - 1) * slotRows(n, W) <= W * n;
}

/// glue exchange_rs: peers' terms at rows [W c, (2W - 1) c) of `into`, summed in place, gathered to rows [0, W c).
pub fn exchange(comm: anytype, sum: anytype, send: u64, into: u64, n: usize, D: usize, W: usize, me: usize, mine: [2]usize, s: anytype) !void {
    if (!fits(n, W)) return error.BadGlue;
    const c = slotRows(n, W);
    const terms = into + W * c * D * 2;
    try comm.scatterRows(send, terms, n, D, .bf16, s);
    try sum.rankSum(send + mine[0] * D * 2, terms, W, me, (mine[1] - mine[0]) * D, c * D, terms);
    try comm.allGather(terms, into, c * D, .bf16, s);
}

/// cvt.rn.bf16.f32 (glue.cu rank_sum, mhc_pf's bfbits): round to nearest even, NaN the canonical 0x7FFF.
pub fn cvt(f: f32) u16 {
    if (std.math.isNan(f)) return 0x7FFF;
    const u: u32 = @bitCast(f);
    return @intCast((u +% (0x7FFF + ((u >> 16) & 1))) >> 16);
}

pub fn widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// mhc_pf's post for one element: rank 0's term, + each later rank's in fp32, then bfr (the bits of the bf16 it uses).
pub fn post(terms: []const u16) u16 {
    var acc = widen(terms[0]);
    for (terms[1..]) |t| acc += widen(t);
    return cvt(acc);
}

/// glue.cu rank_sum_kernel on the host (the CPU twin): `own` this rank's terms, peer q's at slot q (scatterSlot).
pub const Twin = struct {
    pub fn rankSum(_: Twin, own: u64, peers: u64, W: usize, me: usize, n: usize, stride: usize, out: u64) !void {
        const o: [*]const u16 = @ptrFromInt(own);
        const p: [*]const u16 = @ptrFromInt(peers);
        const y: [*]u16 = @ptrFromInt(out);
        for (0..n) |i| {
            var acc = widen(if (me == 0) o[i] else p[i]);
            for (1..W) |r| acc += widen(if (r == me) o[i] else p[(if (r < me) r else r - 1) * stride + i]);
            y[i] = cvt(acc);
        }
    }
};
