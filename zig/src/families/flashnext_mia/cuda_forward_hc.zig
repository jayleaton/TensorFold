//! Flash Next hyper-connection glue: write-backs, read-outs, out projections and the final head.
const std = @import("std");
const tri = @import("cuda_triton.zig");
const state = @import("cuda_state.zig");
const comms = @import("cuda_comm.zig");
const prompt = @import("cuda_prompt.zig");
const fp8 = @import("cuda_fp8.zig");
const W = @import("cuda_weights.zig");
const fwd = @import("cuda_forward.zig");
const Forward = fwd.Forward;
const Pending = fwd.Pending;
const Bufs = fwd.Bufs;
const Seg = fwd.Seg;
const Split = fwd.Split;

/// _writeback: a pending branch into the streams h (in place), no read-out.
pub fn writeback(f: *Forward, h: u64, x: *const Bufs, R: usize, p: Pending) !void {
    try f.t.hcWriteback(h, h, x.b.pss, p.inject, p.branch, R, f.g.hidden, f.g.streams);
}

/// `writeback`, split: this rank's rows only (their streams, squared sums and inject gates at their offset).
pub fn writebackRows(f: *Forward, h: u64, x: *const Bufs, R: usize, p: Pending, sp: ?Split) !void {
    const q = sp orelse return f.writeback(h, x, R, p);
    const g = f.g;
    try f.t.hcWriteback(h + q.o * g.wide() * 2, h + q.o * g.wide() * 2, x.b.pss + q.o * pssRow(g), p.inject + q.o * g.streams * 2, p.branch, q.rh, g.hidden, g.streams);
}

pub fn pssRow(g: state.Geometry) usize {
    return (g.hidden / 256) * g.streams * 4;
}

/// hc_block: pending branch written back into h, then the read-out into b.mixed (split: both ranks' rows gathered).
pub fn hcBlock(f: *Forward, hc: *const W.Hc, x: *const Bufs, R: usize, h: u64, pending: ?Pending, inject_out: u64, sp: ?Split) !void {
    const g = f.g;
    const b = &x.b;
    if (sp) |q| {
        const o = q.o;
        const hh = h + o * g.wide() * 2;
        const ps = b.pss + o * pssRow(g);
        const fuse = f.fusesNorm(hc, b, pending);
        const nm: ?tri.Tri.NormOut = if (fuse) .{ .scale = hc.scale, .normed = b.normed, .eps = f.eps } else null;
        if (pending) |p| {
            try f.t.hcWritebackNorm(hh, hh, ps, p.inject + o * g.streams * 2, p.branch, q.rh, g.hidden, g.streams, nm);
        } else try f.t.hcWritebackNorm(hh, hh, ps, null, .none, q.rh, g.hidden, g.streams, nm);
        try f.mark(.hc_writeback);
        try f.readoutAt(hc, x, hh, ps, q.rh, if (hc.inject) inject_out + o * g.streams * 2 else null, b.mixed + o * g.hidden * 2, b.xs_mixed + o * (g.hidden / 32) * 4, fuse);
        try f.gatherRows(b.mixed, q, g.hidden * 2, .bf16, 2);
        try f.mark(.gather);
        return;
    }
    const fuse = f.fusesNorm(hc, b, pending);
    const nm: ?tri.Tri.NormOut = if (fuse) .{ .scale = hc.scale, .normed = b.normed, .eps = f.eps } else null;
    if (pending) |p| {
        try f.t.hcWritebackNorm(h, h, x.b.pss, p.inject, p.branch, R, g.hidden, g.streams, nm);
    } else try f.t.hcWritebackNorm(h, h, x.b.pss, null, .none, R, g.hidden, g.streams, nm);
    try f.mark(.hc_writeback);
    try f.readoutAt(hc, x, h, x.b.pss, R, if (hc.inject) inject_out else null, x.b.mixed, x.b.xs_mixed, fuse);
}

/// Whether this write-back also makes the read-out's normed rows (`_hc_wb_norm`): prompt chunks, no 4-bit MTP.
pub fn fusesNorm(f: *const Forward, hc: *const W.Hc, b: *const state.Buffers, pending: ?Pending) bool {
    _ = hc;
    if (!f.wb_norm or !b.prefill or f.mtp_q4 != null) return false;
    if (pending) |p| switch (p.branch) {
        .slices => return false,
        else => {},
    };
    return true;
}

/// _readout_b16: norm, the down projection in fp32, SiLU and the inject gates, the up projection, the mix.
pub fn readout(f: *Forward, hc: *const W.Hc, x: *const Bufs, h: u64, R: usize, inject: ?u64) !void {
    try f.readoutAt(hc, x, h, x.b.pss, R, inject, x.b.mixed, x.b.xs_mixed, false);
}

/// `readout` of rows whose squared sums are at `pss`, the mix into `mixed` / `xs` (the scratch from row 0).
pub fn readoutAt(f: *Forward, hc: *const W.Hc, x: *const Bufs, h: u64, pss: u64, R: usize, inject: ?u64, mixed: u64, xs: u64, normed_done: bool) !void {
    const b = &x.b;
    const g = f.g;
    const Wd = g.wide();
    if (!normed_done) {
        try f.t.hcNormed(h, pss, hc.scale, b.normed, b.xs_normed, R, g.hidden, g.streams, f.eps);
        try f.mark(.hc_normed);
    }
    const dn_n: usize = hc.downN();
    if (hc.down8) |d8| {
        // fast-fp8: the low-rank rows on block FP8 and the inject rows bf16, each into its columns of dn (fp32)
        try fp8.matmul(f.k8 orelse return error.NoFp8Kernels, f.s, b.normed, Wd, d8, f.sc.p8, true, R);
        try f.th.slotCopy(f.sc.p8, d8.n * 4, f.sc.dn, dn_n * 4, d8.n * 4, R);
        if (hc.down.n > 0) {
            try f.mmAt(null, b.normed, Wd, hc.down, f.sc.pb, true, R);
            try f.th.slotCopy(f.sc.pb, hc.down.n * 4, f.sc.dn + d8.n * 4, dn_n * 4, hc.down.n * 4, R);
        }
        try f.mark(.hc_down);
    } else try f.mmxAt(.hc_down, b.normed, Wd, b.xs_normed, hc.down, f.sc.dn, true, R);
    try f.t.hcAct(f.sc.dn, b.act, b.xs_act, inject, R, dn_n, g.streams, g.low);
    try f.mark(.hc_act);
    // prompt chunks: up projection and mix in one pass (cuda_prompt.zig upMix, same bytes; no 32-group sums)
    if (f.up_mix and b.prefill and R >= 16 and hc.up8 == null and f.mtp_q4 == null and hc.up.n == g.wide() and hc.up.k == g.low and g.low % 64 == 0) {
        try prompt.upMix(f.t, b.act, hc.up.weight, b.normed, mixed, R, g.hidden, g.streams, g.low, 64);
        try f.mark(.hc_mix);
        return;
    }
    try f.denseAt(.hc_up, b.act, g.low, b.xs_act, hc.up, hc.up8, b.up, false, R);
    try f.t.hcMix(b.up, b.normed, mixed, xs, R, g.hidden, g.streams);
    try f.mark(.hc_mix);
}

/// _out_proj: one GPU, the bf16 branch (mode 1); two ranks, the fp32 partials gathered in rank order (mode 3).
pub fn outProj(f: *Forward, x: *const Bufs, in: u64, r: W.Rows, l8: ?fp8.Linear, R: usize) !tri.Branch {
    return f.outProjX(x, in, 0, r, l8, R);
}

/// outProj with the input's group sums (the MTP head's o_proj in 4-bit, D5).
pub fn outProjX(f: *Forward, x: *const Bufs, in: u64, xs: u64, r: W.Rows, l8: ?fp8.Linear, R: usize) !tri.Branch {
    const b = &x.b;
    if (f.splitOf(x, R)) |q| {
        // this rank's rows into its slot, the peer's rows sent while the peer's partial of ours arrives
        const D = f.g.hidden;
        const cm = f.comm.?;
        _ = cm;
        try f.denseAt(.out_proj, in + q.po * r.k * 2, r.k, 0, r, l8, b.part_branch, true, q.rh);
        try f.exchangeAsync(b.part_branch, b.g_branch + (1 - q.rank) * q.rh * D * 4, q.rh * D, .f32);
        try f.denseAt(.out_proj, in + q.o * r.k * 2, r.k, 0, r, l8, b.g_branch + q.rank * q.rh * D * 4, true, q.rh);
        try f.exchangeWait();
        try f.mark(.gather);
        return .{ .ranks = .{ .part = b.g_branch, .world = 2 } };
    }
    if (f.comm) |cm| {
        try f.denseAt(.out_proj, in, r.k, xs, r, l8, b.part_branch, true, R);
        try cm.gatherPartialsThen(b.part_branch, b.g_branch, R, f.g.hidden, f.s.handle, f.pf_mixer);
        try f.mark(.gather);
        return .{ .ranks = .{ .part = b.g_branch, .world = cm.world } };
    }
    try f.denseAt(.out_proj, in, r.k, xs, r, l8, b.branch, false, R);
    return .{ .bf16 = b.branch };
}

/// finish: last write-back into b.streams, the mixer's read-out, the head (`logits`), candidates gathered.
pub fn finish(f: *Forward, mixer: *const W.Hc, x: *const Bufs, R: usize, pending: Pending, logits: bool) !?u64 {
    return f.finishSegs(mixer, x, R, pending, logits, &.{});
}

/// `finish` of a multi-prompt chunk: read-out and head on each prompt's last row (`segs.len` rows).
pub fn finishSegs(f: *Forward, mixer: *const W.Hc, x: *const Bufs, R: usize, pending: Pending, logits: bool, segs: []const Seg) !?u64 {
    const b = &x.b;
    const g = f.g;
    const Wd = g.wide();
    try f.mark(.other);
    if (f.splitOf(x, R)) |q| {
        // this rank's rows written back, then both halves of the streams and their squared sums gathered
        try f.th.copy(b.streams + q.o * Wd * 2, b.h + q.o * Wd * 2, q.rh * Wd * 2);
        try f.writebackRows(b.streams, x, R, pending, q);
        try f.gatherRows(b.streams, q, Wd * 2, .bf16, 2);
        try f.gatherRows(b.pss, q, pssRow(g), .f32, 4);
    } else {
        try f.th.copy(b.streams, b.h, R * Wd * 2);
        try f.writeback(b.streams, x, R, pending);
    }
    try f.mark(.finish);
    var rows = b.streams;
    var n = R;
    if (b.prefill) {
        const prow = (g.hidden / 256) * g.streams * 4;
        if (segs.len > 1) {
            if (segs.len > state.max_rows) return error.TooManyPrompts;
            // last rows gathered into rows 0.. of h and sums (a source never precedes its slot, so in order)
            for (segs, 0..) |sg, i| {
                try f.th.copy(b.h + i * Wd * 2, b.streams + (sg.a1 - 1) * Wd * 2, Wd * 2);
                if (sg.a1 - 1 != i) try f.th.copy(b.pss + i * prow, b.pss + (sg.a1 - 1) * prow, prow);
            }
            rows = b.h;
            n = segs.len;
        } else {
            try f.th.copy(b.pss, b.pss + (R - 1) * prow, prow);
            rows = b.streams + (R - 1) * Wd * 2;
            n = 1;
        }
    }
    try f.readout(mixer, x, rows, n, null);
    try f.put("final_streams", b.streams, R * Wd * 2);
    try f.put("final_mixed", b.mixed, n * g.hidden * 2);
    if (!logits) return null;
    const head = f.w.head;
    try f.headMm(b.mixed, g.hidden, b.logits, n);
    try f.mark(.head);
    if (f.comm) |cm| {
        try f.candidates(x, b.logits, n, head.n, null, f.vocab_offset, cm);
        try f.mark(.cands);
    }
    try f.put("logits", b.logits, n * head.n * 2);
    return b.logits;
}

/// forward.candidates: each rank's top CAND values, ids and log-sum-exp gathered into b.cand_all.
pub fn candidates(f: *Forward, x: *const Bufs, logits: u64, rows: usize, columns: usize, id_map: ?u64, offset: i64, cm: *const comms.Comm) !void {
    const b = &x.b;
    const s = f.sample orelse return error.NoSampleScratch;
    try f.th.rankCandidates(logits, rows, columns, state.cand, id_map, offset, s, b.cand);
    try cm.allGather(b.cand, b.cand_all, rows * (2 * state.cand + 1), .f32, f.s.handle);
}
