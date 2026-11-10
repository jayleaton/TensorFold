//! Flash Next attention block: attn_prep, the sparse indexer, chunked and multi-stream attention.
const std = @import("std");
const tri = @import("cuda_triton.zig");
const state = @import("cuda_state.zig");
const prompt = @import("cuda_prompt.zig");
const W = @import("cuda_weights.zig");
const fwd = @import("cuda_forward.zig");
const Forward = fwd.Forward;
const Bufs = fwd.Bufs;
const Seg = fwd.Seg;
const multi_ptrs = fwd.multi_ptrs;

/// attn_block: projection, attn_prep, indexer pool/select past budget, chunked attention and merge, the gate.
pub fn attnBlock(f: *Forward, layer: *const W.Layer, segs: []const Seg, x: *const Bufs, R: usize, mtp: bool) !tri.Branch {
    const b = &x.b;
    const g = f.g;
    const a = &layer.attn.?;
    if (a.proj8 != null) try f.projFace(.attn_proj, b.mixed, g.hidden, a.proj8, a.proj, b.pa, g.attnWidth(), R) else try f.mmxAt(.attn_proj, b.mixed, g.hidden, b.xs_mixed, a.proj, b.pa, false, R);
    const ci: usize = if (mtp) g.attention_layers else f.sub[@intCast(layer.index)];
    const d = f.dims;
    const ag = x.attn;
    const sc: tri.AttnScratch = .{ .po = b.att_po, .pm = b.att_pm, .pl = b.att_pl, .ids = b.att_ids, .nk = b.att_nk, .sparse = b.att_sparse, .scores = b.att_scores };
    const qrow = g.heads * g.head_dim * 2;
    const irow = g.index_heads * g.index_dim * 2;
    const prow = g.attnWidth() * 2;
    var o = b.att_out;
    // eager rounds only: a graph's launches are fixed at capture, the sparse selects and chunks are not
    if (!b.prefill and !mtp and f.round_multi and f.contexts == null and f.context == null) {
        try f.attnMulti(layer, segs, x, R);
        o = b.attn_o;
        try f.t.attnGate(o, b.pa, b.gated, b.xs_gated, R, d);
        try f.mark(.attn_gate);
        return f.outProjX(x, b.gated, b.xs_gated, a.o, a.o8, R);
    }
    for (segs, 0..) |sg, si| {
        const st = &sg.seq.st;
        const n = sg.a1 - sg.a0;
        const pos = if (mtp) st.mtp_pos else st.pos_dev;
        const host_pos = if (mtp) st.mtp_len else st.pos;
        const cache: tri.Cache = .{ .k = st.kc[ci], .v = st.vc[ci], .ks = f.sc.kv_scale, .vs = f.sc.kv_scale, .fp8 = g.kv == .fp8 };
        const ikc = st.ikc[ci];
        const pooled = st.pooled[ci];
        const q = b.q + sg.a0 * qrow;
        const iq = b.iq + sg.a0 * irow;
        const rope = sg.seq.rope();
        try f.t.attnPrepRope(.{ .p = b.pa + sg.a0 * prow, .pos0 = pos, .qw = a.q_scale, .kw = a.k_scale, .iw = a.iq_scale, .inv = f.w.inv_freq, .q = q, .cache = cache, .iq = iq, .ikc = ikc }, n, d, rope);
        try f.mark(.attn_prep);
        if (b.prefill) {
            if (ag.qsa) try f.t.qsaPoolRope(ikc, pooled, pos, a.ik_scale, f.w.inv_freq, n, ag, d, rope);
            try f.mark(.qsa_pool);
            if (f.splitOf(x, R)) |sp| if (segs.len == 1 and f.cs != null and f.overlap & 1 != 0) {
                // the peer's rows first (their 256-row blocks), their gate and out_proj sent while ours run
                const D = g.hidden;
                const pa = sp.po;
                const pb = @min(sp.po + sp.rh, n);
                var phase: usize = 0;
                while (phase < 2) : (phase += 1) {
                    var r1: usize = 0;
                    while (r1 < n) : (r1 += state.att_rows) {
                        const m = @min(state.att_rows, n - r1);
                        const peer = r1 < pb and r1 + m > pa;
                        if (peer != (phase == 0)) continue;
                        try f.th.fill32(b.pos_blk, @intCast(host_pos + r1), 1);
                        const ends = host_pos + r1 + m;
                        // blocks in a rank's own order: indexer rows are not split (exchanges pair blocks)
                        if (ag.qsa) try f.qsaBlockOpt(iq + r1 * irow, pooled, b.pos_blk, host_pos + r1, sc, m, ag, ends, false);
                        try f.t.attention(q + r1 * qrow, cache, b.pos_blk, sc, b.attn_o + r1 * qrow, m, ag, ends, d);
                        try f.mark(.attention);
                    }
                    const rows0 = if (phase == 0) sp.po else sp.o;
                    try f.t.attnGate(b.attn_o + rows0 * qrow, b.pa + rows0 * prow, b.gated + rows0 * qrow, b.xs_gated + rows0 * (qrow / 64) * 4, sp.rh, d);
                    try f.mark(.attn_gate);
                    if (phase == 0) {
                        try f.denseAt(.out_proj, b.gated + sp.po * qrow, qrow / 2, 0, a.o, a.o8, b.part_branch, true, sp.rh);
                        try f.exchangeAsync(b.part_branch, b.g_branch + (1 - sp.rank) * sp.rh * D * 4, sp.rh * D, .f32);
                    } else {
                        try f.denseAt(.out_proj, b.gated + sp.o * qrow, qrow / 2, 0, a.o, a.o8, b.g_branch + sp.rank * sp.rh * D * 4, true, sp.rh);
                        try f.exchangeWait();
                        try f.mark(.gather);
                    }
                }
                return .{ .ranks = .{ .part = b.g_branch, .world = 2 } };
            };
            var r0: usize = 0;
            while (r0 < n) : (r0 += state.att_rows) {
                const m = @min(state.att_rows, n - r0);
                try f.th.fill32(b.pos_blk, @intCast(host_pos + r0), 1);
                const ends = host_pos + r0 + m;
                if (ag.qsa) try f.qsaBlock(iq + r0 * irow, pooled, b.pos_blk, host_pos + r0, sc, m, ag, ends);
                try f.t.attention(q + r0 * qrow, cache, b.pos_blk, sc, b.attn_o + (sg.a0 + r0) * qrow, m, ag, ends, d);
                try f.mark(.attention);
            }
            continue;
        }
        const keys = if (f.contexts) |cs| cs[si] else f.context orelse host_pos + n;
        if (ag.qsa) try f.t.qsaSelectRope(iq, ikc, pooled, pos, a.ik_scale, f.w.inv_freq, sc, n, ag, keys, d, rope);
        try f.t.attention(q, cache, pos, sc, b.att_out, n, ag, keys, d);
        // the scratch output is the next stream's too
        if (segs.len > 1) try f.th.copy(b.attn_o + sg.a0 * qrow, b.att_out, n * qrow);
        try f.mark(.attention);
    }
    if (b.prefill or segs.len > 1) o = b.attn_o;
    try f.t.attnGate(o, b.pa, b.gated, b.xs_gated, R, d);
    try f.mark(.attn_gate);
    return f.outProjX(x, b.gated, b.xs_gated, a.o, a.o8, R);
}

/// The indexer's scores and key lists of a prompt block's `m` rows; two ranks split rows and exchange the lists.
pub fn qsaBlock(f: *Forward, iq: u64, pooled: u64, pos0: u64, pos: usize, sc: tri.AttnScratch, m: usize, ag: tri.AttnGeometry, ends: usize) !void {
    return f.qsaBlockOpt(iq, pooled, pos0, pos, sc, m, ag, ends, true);
}

pub fn qsaBlockOpt(f: *Forward, iq: u64, pooled: u64, pos0: u64, pos: usize, sc: tri.AttnScratch, m: usize, ag: tri.AttnGeometry, ends: usize, may_split: bool) !void {
    const d = f.dims;
    const irow = d.index_heads * d.index_dim * 2;
    const split = may_split and f.comm != null and f.qsa_split_keys > 0 and ends >= f.qsa_split_keys and m >= 32;
    if (!split) {
        if (f.prof == null and f.qsa_rt == 0 and f.qsa_fast == null and ag.blocks(ends) < f.qsa_tiles_from) return f.t.qsaRows(iq, pooled, pos0, sc, m, ag, ends, d);
        try f.qsaScoresSelect(iq, pooled, pos0, sc, m, ag, ends);
        return;
    }
    const cm = f.comm.?;
    const h = std.mem.alignForward(usize, (m + 1) / 2, 16);
    const own0: usize = if (cm.rank == 0) 0 else h;
    const own1: usize = if (cm.rank == 0) h else m;
    const peer0: usize = if (cm.rank == 0) h else 0;
    const peer1: usize = if (cm.rank == 0) m else h;
    try f.th.fill32(f.sc.qsa_pos, @intCast(pos + own0), 1);
    const so = prompt.scratchAt(sc, ag, own0);
    try f.qsaScoresSelect(iq + own0 * irow, pooled, f.sc.qsa_pos, so, own1 - own0, ag, ends);
    const sp = prompt.scratchAt(sc, ag, peer0);
    const no = own1 - own0;
    const np = peer1 - peer0;
    try cm.exchangeBytes(&.{ .{ so.ids, no * ag.idw * 4 }, .{ so.nk, no * 4 }, .{ so.sparse, no * 4 } }, &.{ .{ sp.ids, np * ag.idw * 4 }, .{ sp.nk, np * 4 }, .{ sp.sparse, np * 4 } }, f.s.handle);
    try f.mark(.qsa_exchange);
}

pub fn qsaScoresSelect(f: *Forward, iq: u64, pooled: u64, pos0: u64, sc: tri.AttnScratch, m: usize, ag: tri.AttnGeometry, ends: usize) !void {
    if (f.qsa_fast) |q| {
        try q.run(f.s, iq, pooled, pos0, sc, m, ag, ends);
    } else if (f.qsa_rt > 0 and m >= 16) {
        try prompt.qsaScoresRows(f.t, iq, pooled, pos0, sc, m, ag, ends, f.dims, f.qsa_rt);
    } else try prompt.qsaScores(f.t, iq, pooled, pos0, sc, m, ag, ends, f.dims);
    try f.mark(.qsa_scores);
    if (ag.blocks(ends) >= f.qsa_tiles_from) {
        try prompt.qsaSelectTiles(f.t, pos0, sc, m, ag);
    } else try prompt.qsaSelect(f.t, pos0, sc, m, ag, ends);
    try f.mark(.qsa_select);
}

/// attn_multi.layer: every stream's decode-round rows in one launch a kernel into b.attn_o (multiTables).
pub fn attnMulti(f: *Forward, layer: *const W.Layer, segs: []const Seg, x: *const Bufs, R: usize) !void {
    const b = &x.b;
    const g = f.g;
    const a = &layer.attn.?;
    const d = f.dims;
    const ag = x.attn;
    const r = &(f.round orelse return error.NoRoundScratch);
    const n = segs.len;
    const ai: usize = f.sub[@intCast(layer.index)];
    const cp = r.aptr + ai * multi_ptrs * n * 8;
    const m: tri.Tri.Multi = .{ .posr = r.posr, .sid = r.sid, .first = r.first, .counts = r.acounts, .n = n };
    try f.t.prepMulti(m, cp, b.pa, a.q_scale, a.k_scale, a.iq_scale, f.w.inv_freq, b.q, b.iq, R, d);
    try f.mark(.attn_prep);
    const top = ag.budget / ag.ratio;
    const irow = g.index_heads * g.index_dim * 2;
    var keys: usize = 0;
    var most: usize = 0;
    for (segs) |sg| {
        keys = @max(keys, sg.seq.st.pos + sg.a1 - sg.a0);
        most = @max(most, sg.a1 - sg.a0);
    }
    if (ag.qsa) {
        try f.t.poolMulti(m, cp, a.ik_scale, f.w.inv_freq, most, ag, d);
        try f.mark(.qsa_pool);
        // a stream whose rows pass TOP blocks selects its own keys (the one-stream kernels on its rows)
        for (segs) |sg| {
            const st = &sg.seq.st;
            const end = st.pos + sg.a1 - sg.a0;
            if (end / ag.ratio <= top) continue;
            const a0 = sg.a0;
            const sc: tri.AttnScratch = .{ .po = b.att_po, .pm = b.att_pm, .pl = b.att_pl, .ids = b.att_ids + a0 * ag.idw * 4, .nk = b.att_nk + a0 * 4, .sparse = b.att_sparse + a0 * 4, .scores = b.att_scores + a0 * ag.nb * 4 };
            try f.t.qsaRows(b.iq + a0 * irow, st.pooled[ai], st.pos_dev, sc, sg.a1 - sg.a0, ag, end, d);
        }
        try f.mark(.qsa_rows);
    }
    const sc: tri.AttnScratch = .{ .po = b.att_po, .pm = b.att_pm, .pl = b.att_pl, .ids = b.att_ids, .nk = b.att_nk, .sparse = b.att_sparse, .scores = b.att_scores };
    try f.t.attentionMulti(m, cp, b.q, sc, b.attn_o, R, ag, keys, d);
    try f.mark(.attention);
}
