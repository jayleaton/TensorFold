//! Flash Next forward pass: embedding, the decoder layers, and their n-gram and DeltaNet blocks.
const std = @import("std");
const envs = @import("cuda_env.zig");
const tri = @import("cuda_triton.zig");
const state = @import("cuda_state.zig");
const comms = @import("cuda_comm.zig");
const W = @import("cuda_weights.zig");
const fwd = @import("cuda_forward.zig");
const Forward = fwd.Forward;
const Pending = fwd.Pending;
const Bufs = fwd.Bufs;
const Seq = fwd.Seq;
const Seg = fwd.Seg;
const Round = fwd.Round;

// -- the forward ------------------------------------------------------------------------------------------

/// forward.compute on staged rows: embedding, every layer, then `finish`; returns the head's logits or null.
pub fn computeSegs(f: *Forward, segs: []const Seg, x: *const Bufs, logits: bool) !?u64 {
    const b = &x.b;
    const R = segs[segs.len - 1].a1;
    try f.mark(.stage);
    try f.t.embed(b.ids, f.w.embed, b.h, R, f.g.hidden, f.g.streams);
    if (b.prefill) try f.injectMedia(segs[0].seq, x, R);
    try f.mark(.embed);
    var pending: ?Pending = null;
    for (f.w.layers, 0..) |*layer, i| {
        f.pf_moe = prefetchHc(if (i + 1 < f.w.layers.len) &f.w.layers[i + 1].attn_hc else &f.w.mixer);
        pending = try f.layerForward(layer, segs, x, R, pending, false);
    }
    return f.finishSegs(&f.w.mixer, x, R, pending.?, logits, segs);
}

/// A weight's first bytes to prefetch: all of it up to TF_FLASHNEXT_L2PF_BYTES (default 8 MiB).
pub fn prefetchOf(r: W.Rows) comms.Prefetch {
    const cap: usize = if (envs.get("TF_FLASHNEXT_L2PF_BYTES")) |v| std.fmt.parseInt(usize, v, 10) catch (8 << 20) else (8 << 20);
    return .{ .ptr = r.weight, .bytes = @min(@as(usize, r.n) * r.k * 2, cap) };
}

/// A read-out's down projection to prefetch: its block-FP8 bytes (fast-fp8), else its bf16 rows.
pub fn prefetchHc(h: *const W.Hc) comms.Prefetch {
    const d8 = h.down8 orelse return prefetchOf(h.down);
    const cap: usize = if (envs.get("TF_FLASHNEXT_L2PF_BYTES")) |v| std.fmt.parseInt(usize, v, 10) catch (8 << 20) else (8 << 20);
    return .{ .ptr = d8.w8, .bytes = @min(@as(usize, d8.npad) * d8.k, cap) };
}

/// One stream's staged rows [0, R).
pub fn compute(f: *Forward, seq: *Seq, x: *const Bufs, R: usize, logits: bool) !?u64 {
    return f.computeSegs(&.{.{ .seq = seq, .a0 = 0, .a1 = R }}, x, logits);
}

/// The window `tokens` at st.pos ..: staged, then computed (Python forward; the state commits later).
pub fn forward(f: *Forward, seq: *Seq, x: *const Bufs, tokens: []const u32, logits: bool) !?u64 {
    try f.stage(seq, x, tokens);
    return f.compute(seq, x, tokens.len, logits);
}

/// layer_forward: one decoder layer on b.h[:R] (every segment's rows); returns the new pending write-back.
pub fn layerForward(f: *Forward, layer: *const W.Layer, segs: []const Seg, x: *const Bufs, R: usize, pending: ?Pending, mtp: bool) !Pending {
    const b = &x.b;
    const g = f.g;
    try f.putLayer(layer, "h_in", b.h, R * g.wide() * 2);
    var pend = pending;
    const sp = f.splitOf(x, R);
    if (layer.ple) |*p| {
        if (pend) |pp| {
            try f.writebackRows(b.h, x, R, pp, sp);
            pend = null;
        }
        // the n-gram block runs on every row (its conv reads the rows before each): the halves gathered first
        if (sp) |q| try f.gatherRows(b.h, q, g.wide() * 2, .bf16, 2);
        try f.mark(.hc_writeback);
        try f.pleBlock(p, segs, x, R);
        try f.mark(.ple);
        try f.putLayer(layer, "ple_emb", b.ple_emb, R * g.ple_dim * 2);
        try f.putLayer(layer, "h_ple", b.h, R * g.wide() * 2);
    }
    try f.hcBlock(&layer.attn_hc, x, R, b.h, pend, b.inj_a, sp);
    try f.putLayer(layer, "hc_a", b.mixed, R * g.hidden * 2);
    f.pf_mixer = prefetchHc(&layer.mlp_hc);
    const br = if (layer.linear) try f.gdnBlock(layer, segs, x, R) else try f.attnBlock(layer, segs, x, R, mtp);
    try f.putBranch(layer, if (layer.linear) "gdn" else "attn", br, R);
    try f.hcBlock(&layer.mlp_hc, x, R, b.h, .{ .branch = br, .inject = b.inj_a }, b.inj_m, sp);
    try f.putLayer(layer, "hc_m", b.mixed, R * g.hidden * 2);
    const moe = try f.moeBlock(&layer.moe, x, R, mtp);
    try f.putLayer(layer, "router_logits", b.moe_logits, R * (g.experts + 1) * 4);
    try f.putLayer(layer, "router_pick", b.moe_pick, R * (g.topKFor(mtp) + 1) * 4);
    try f.putLayer(layer, "router_wts", b.moe_wts, R * (g.topKFor(mtp) + 1) * 4);
    try f.putBranch(layer, "moe_y", moe, R);
    return .{ .branch = moe, .inject = b.inj_m };
}

/// ple_block: h += the n-gram embedding branch through the stream's conv tail (rows staged by `stage`).
pub fn pleBlock(f: *Forward, p: *const W.Ple, segs: []const Seg, x: *const Bufs, R: usize) !void {
    const b = &x.b;
    const g = f.g;
    const scale: f32 = if (f.w.table) |t| (if (t.fp8) 1.0 else t.scale) else 1.0;
    const table = f.w.table.?;
    if (table.gpu) |gp| {
        // two ranks: this rank's heads embedded, both halves gathered in head order, then interleaved
        const cm = f.comm orelse return error.TwoRanksNeedComm;
        const half = gp.heads * p.ngram.dims;
        try f.t.pleEmbedBf16(b.ple_v, f.sc.ple_half, f.sc.xs_half, R, gp.heads, p.ngram.dims, scale);
        try cm.allGather(f.sc.ple_half, f.sc.ple_all, R * half, .bf16, f.s.handle);
        try cm.allGather(f.sc.xs_half, f.sc.xs_all, R * half / 32, .f32, f.s.handle);
        for (0..cm.world) |r| {
            try f.th.slotCopy(f.sc.ple_all + r * R * half * 2, half * 2, b.ple_emb + r * half * 2, cm.world * half * 2, half * 2, R);
            try f.th.slotCopy(f.sc.xs_all + r * R * (half / 32) * 4, (half / 32) * 4, b.xs_ple + r * (half / 32) * 4, cm.world * (half / 32) * 4, (half / 32) * 4, R);
        }
    } else try f.t.pleEmbedBf16(b.ple_v, b.ple_emb, b.xs_ple, R, p.ngram.heads, p.ngram.dims, scale);
    try f.mm(b.ple_emb, g.ple_dim, p.key, b.ple_keys, false, R);
    try f.mm(b.ple_emb, g.ple_dim, p.value, b.ple_vals, false, R);
    try f.t.pleGate(b.ple_keys, b.ple_vals, b.h, p.norm_key, p.norm_query, b.ple_gated, b.ple_pss, R, g.hidden, g.streams, f.eps);
    const Wd = g.wide();
    for (segs) |sg| {
        const o = sg.a0;
        try f.t.pleConv(b.ple_gated + o * Wd * 2, b.ple_pss + o * g.streams * 4, p.norm_conv, sg.seq.st.ple_tail, p.conv, b.h + o * Wd * 2, b.h + o * Wd * 2, b.ple_nrow + o * Wd * 2, sg.a1 - sg.a0, g.hidden, g.streams, g.ple_kernel, g.ngram_size, f.eps);
    }
}

/// gdn_block: the projection, then the decode chain, or in a prompt chunk front, chain and back committed at once.
pub fn gdnBlock(f: *Forward, layer: *const W.Layer, segs: []const Seg, x: *const Bufs, R: usize) !tri.Branch {
    const b = &x.b;
    const g = f.g;
    const gd = &layer.gdn.?;
    const li = f.sub[@intCast(layer.index)];
    const pw = g.gdnWidth();
    const nv = g.nv;
    if (b.prefill) {
        // projection over every row, then each prompt's own chain with its own conv taps and state, as alone
        try f.projFace(.gdn_proj, b.mixed, g.hidden, gd.proj8, gd.proj, b.proj, pw, R);
        const C = g.gdnConv();
        for (segs) |sg| {
            const st = &sg.seq.st;
            const o = sg.a0;
            const n = sg.a1 - sg.a0;
            if (sg.seq.held != 0) return error.UnfoldedRows; // a prompt starts from reset or restore
            const p = b.proj + o * pw * 2;
            const qo = o * g.nk * state.gdn_dk * 4;
            const vo = o * nv * state.gdn_dv * 2;
            const so = o * nv * 4;
            try f.th.fill64(b.conv_ptr, f.conv(st, li), 1);
            try f.ops.gdnFront(nv, p, b.conv_ptr, b.sid, b.windows, n, gd.conv, gd.a_log, gd.dt_bias, f.sc.gq + qo, f.sc.gk + qo, f.sc.gv + vo, f.sc.gg + so, f.sc.gb + so);
            try f.mark(.gdn_front);
            // the chunk-invariant chain into the scratch state, then over the sequence's one state
            try f.ops.gdnPrefill(true, f.sc.gq + qo, f.sc.gk + qo, f.sc.gv + vo, f.sc.gg + so, f.sc.gb + so, f.rec(st, li), f.sc.gdn_t, f.sc.gy + vo, n, g.nk, nv);
            try f.th.copy(f.rec(st, li), f.sc.gdn_t, nv * state.gdn_dv * state.gdn_dk * 4);
            try f.mark(.gdn_chain);
            try f.ops.gdnBack(nv, f.sc.gy + vo, p, gd.norm, f.eps, b.gout + vo, b.gxs + o * (nv * state.gdn_dv / 32) * 4, n);
            try f.mark(.gdn_back);
            try f.t.shiftWindows(f.conv(st, li), p, n, (g.conv_kernel - 1) * C, b.rows * pw, pw, 1, C, g.conv_kernel - 1);
            try f.mark(.gdn_shift);
        }
        return f.outProj(x, b.gout, gd.out, gd.out8, R);
    }
    const proj = b.proj + li * b.rows * pw * 2;
    if (gd.proj8 != null) try f.projFace(.gdn_proj, b.mixed, g.hidden, gd.proj8, gd.proj, proj, pw, R) else {
        try f.mm(b.mixed, g.hidden, gd.proj, proj, false, R);
        try f.mark(.gdn_proj);
    }
    // every stream's rows in one launch a step (gdn_multi.block), each tree folding its last kept rows first
    const r = &(f.round orelse return error.NoRoundScratch);
    const n = segs.len;
    const p: usize = f.cur_par;
    var most: usize = 1;
    for (segs) |sg| most = @max(most, sg.a1 - sg.a0);
    const at = r.layer(p, li);
    const prev = r.layer(1 - p, li);
    try f.ops.gdnFront(nv, proj, r.conv + li * n * 8, r.sid, r.win, R, gd.conv, gd.a_log, gd.dt_bias, r.q, at.k, at.v, at.g, at.beta);
    try f.mark(.gdn_front);
    try f.ops.gdnTree(r.q, at.k, at.v, at.g, at.beta, 0, r.state + li * n * 8, r.starts, r.starts, R, 0, n, most, f.sc.gy, g.nk, nv, state.gdn_dv, .{ .k = prev.k, .v = prev.v, .g = prev.g, .beta = prev.beta, .rows = r.held, .row_stride = @intCast(Round.held_stride), .counts = r.counts, .count_stride = 1 }, 0, 0);
    try f.mark(.gdn_chain);
    try f.ops.gdnBack(nv, f.sc.gy, proj, gd.norm, f.eps, b.gout, b.gxs, R);
    try f.mark(.gdn_back);
    return f.outProj(x, b.gout, gd.out, gd.out8, R);
}
