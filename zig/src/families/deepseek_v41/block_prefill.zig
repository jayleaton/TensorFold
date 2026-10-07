//! A prefill segment on the prod fast-prefill path (TF_DSV41_PREFILL_KERNELS=fast with prod-knobs.env), emitted as
//! calls (calls.zig) by block.zig's Emitter: every launch with its arguments and its tensors named by role, in the
//! Python engine's order (forward._run / _layers, blocks.attention's serial path, prefill_moe.forward, mhc_pf). M1
//! checks the calls against the capture's prefill phase launch by launch; the forward issues the same calls.
//!
//! What differs from a decode window (block.zig):
//! - a dense projection is prefill_mm.matmul: upstream's rot_in into the workspace, W_q unpacked from the words
//!   (dense3's lanes, upstream's strips, or the shared expert's stored tiles), the Triton `_gemm`, in column blocks
//!   whose W_q fits the 64 MiB workspace;
//! - an mHC site is mhc_pf.run + `_finish_k` from 256 rows (Triton `_site` below, and for the post-only / final sites);
//! - the SWA rows go through the staging ring (forward.staging) when the slot's ring cannot hold the segment;
//! - the indexer materialises its scores (`_scores`, `_keys`, the candidate source's `_block_keys`; in row blocks
//!   under the index budget) and torch picks the top-k (glue);
//! - the attention core is the fused prefill kernel (`_fused`), group-major for wo_a's groups (TF_DSV41_PF_COPIES);
//! - the routed experts are x3gm (rot, then gate/up and down once a K2 width present, upstream's combine), the shared
//!   expert three GEMMs, the partial bf16(routed + shared) before its exchange;
//! - the head is the decode path's linear over every row, 128 rows a launch.
//!
//! Roles as block.zig's ("s." persistent, "w." the segment, "L." a layer), plus the workspace "s.pf.*", the staging
//! ring "s.stage.*" and x3gm's pointer tables "s.L<i>.gm.*". The attention's exchange lands in "L.recv.attn": the
//! caching allocator hands each layer another storage for it.
//!
//! Glue steps (no captured launch; the forward runs them): positions, embed, ix_w (the head weights' fp64 -> bf16 ->
//! fp32 casts), proj (a compressor projection to fp32, ratio 2: [kv | gate], then [carry | rows]), zeros, stage_in /
//! stage_out, block_pos (a row block's first position), cand_keys, topk (index.top_positions), counts
//! (visible_counts), exchange, gm_picks (kit weights, the shared slot masked, the routed slots' contiguous copies),
//! gm_ticket, gm_plan (x3gm.plan of one width's picks), swiglu (the shared expert's clamped SiLU product), moe_sum
//! (bf16(routed + shared)), engram_rows, trace, kit_logits.
//!
//! A short segment (at most 16 rows: a prompt's ragged tail, or a whole short prompt) is Python's mix (pod 24's
//! captures at 1, 12 and 16 rows): the prefill projections, indexer scores, fused attention, x3gm experts and shared
//! GEMMs as above, but every kernel that has a row path for at most 16 rows takes it, as in a decode window
//! (block.zig, block_moe.zig): mhc_cuda for every site, post-only and the final norm (no `_finish_k`), the L2 prefetch
//! (o / f / x sites), attn_cuda's top-k (a Reindex layer scoring over its candidate blocks, no `_keys` or torch top-k)
//! and the narrow router. The exchanges land in the decode window's "w.recv.*" that mhc_cuda reads.
//!
//! Not taken here (error.Unsupported): the stream top-k of full-mode layers 2 / 8 / 14 at TF_DSV41_INDEX_STREAM_MIN
//! visible keys or more, uniform (not ragged) expert stacks, gate / up sharing their sign vector (x3gm shx).

const std = @import("std");
const dk = @import("dsv41_kernels");
const calls = @import("calls.zig");
const block = @import("block.zig");
const config = @import("config.zig");
const moe = @import("block_moe.zig");
const Config = config.Config;
const E = block.Emitter;
const Arg = calls.Arg;
const Dt = calls.Dt;
const dense = dk.dense;

// prod knobs and the Python engine's constants
const ws_limit: i64 = 64 << 20; // TF_DSV41_PREFILL_WS_MIB: a workspace W_q's bytes at most (column blocks)
const had_scale: f64 = 0.08838834764831845; // exl3.prefill.HAD_SCALE (the literal: 1 / sqrt(128) rounds the other way)
const pf_rows: i64 = 256; // TF_DSV41_MHC_PF_ROWS / TF_DSV41_MHC_ROWS: mhc_pf and the wide `_site` tile from here
const mhc_bm: i64 = 64; // TF_DSV41_MHC_BM
const split_rows: i64 = 32; // TF_DSV41_MHC_SPLIT_ROWS: out-of-place boundaries, a program a stream
const index_budget: i64 = 256 << 20; // TF_DSV41_INDEX_BUDGET_MIB: scores + sort keys a launch
const stream_min: i64 = 4096; // TF_DSV41_INDEX_STREAM_MIN
const bmq: i64 = 32; // TF_DSV41_PREFILL_ATTN_BMQ
const kt: i64 = 32; // csa2.attn.KT
const router_chunk: i64 = 2048; // router_gemv.CHUNK: rows a route launch
const gm_bm: i64 = 64; // x3gm: members a pass of the gate/up (cfg 1) and down (cfg 0) tiles below 3,072 rows
const gm_widths: u32 = 7; // x3gm.MAX_WIDTHS: the ticket's down half starts there

fn size(dt: Dt) i64 {
    return switch (dt) {
        .bf16, .f16, .i16 => 2,
        .f32, .i32 => 4,
        .f64, .i64 => 8,
        .i8, .u8, .bool => 1,
    };
}

fn cdiv(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

/// How a matrix's words are kept: dense3's lanes, upstream's strips, or the checkpoint's tiles as stored (the shared
/// expert: prefill_moe.shared_linears, layout "stored").
const Form = enum { lanes, strips, stored };

/// Where a matmul's [m, N] result goes: `role`'s rows of `stride` elements from byte `offset`.
const Out = struct { role: calls.Role, dt: Dt, stride: i64, offset: i64 = 0 };

/// One prefill segment: the emitter, the SWA ring its rows go through, and what the segment's layers left (layer
/// 20's candidates).
const Seg = struct {
    e: *E,
    /// the SWA ring the layers store into and attend over: the slot's (2 x window rows) or the staging ring
    staged: bool,
    ring: i64,
    /// the candidate source ran in this segment (reindex layers read its candidates)
    cand: bool = false,
    /// at most 16 rows: Python's short segment (the module doc)
    short: bool = false,

    // -----------------------------------------------------------------------------------------------------------
    // dense EXL3 projections (prefill_mm.matmul)

    /// y = x [m, K] @ W [K, N] (`w`'s prefix): per column block, upstream's rot_in, the W_q unpack, `_gemm`.
    fn matmul(s: *Seg, w: []const u8, form: Form, x: Arg, k: i64, n: i64, y: Out) !void {
        const e = s.e;
        const m = e.n;
        const k2: i64 = try e.w.k2(w);
        const tw = 4 * k2;
        const nb = @divExact(n, 128);
        // prefill_mm.column_blocks: whole 128-column blocks whose W_q fits the workspace (strips / lanes only)
        const per: i64 = if (k * n * 2 <= ws_limit or form == .stored) nb else @max(1, @divFloor(ws_limit, k * 2 * 128));
        const st = dense.strides(@intCast(k), @intCast(k2));
        var b0: i64 = 0;
        while (b0 < nb) : (b0 += per) {
            const b1 = @min(nb, b0 + per);
            const c0 = 128 * b0;
            const nc = 128 * (b1 - b0);
            const xh = try e.buf("s.pf.xh", .{}, .f16, &.{ m, k });
            try e.ext("tensorfold_exl3_linear_v4.rot_in", &.{ x, try e.weight("{s}.suh", .{w}, .f16, &.{k}), xh });
            const wq = try e.buf("s.pf.w", .{}, .f16, &.{ k, nc });
            const blk_words = @divExact(k, 16) * 8 * tw; // words of a 128-column block
            switch (form) {
                .lanes => try e.ext("tf_dsv41_dense3_v1.unpack", &.{
                    try e.view(.{ .buf = try e.fmt("s.{s}.lanes", .{w}) }, .i32, &.{ b1 - b0, @divExact(k, 16), 8, tw }, &.{ blk_words, 8 * tw, tw, 1 }, b0 * blk_words * 4),
                    wq, .{ .i = st[0] }, .{ .i = st[1] }, .{ .i = k2 },
                }),
                .strips => try e.ext("tensorfold_exl3_linear_v4.unpack", &.{
                    // one block of 128 columns: the stored trellis is the strips (block.zig's upstream)
                    if (n == 128)
                        try e.view(.{ .weight = try e.fmt("{s}.trellis", .{w}) }, .i32, &.{ 1, @divExact(k, 16), 8, tw }, &.{ 8 * tw, 8 * tw, tw, 1 }, 0)
                    else
                        try e.view(.{ .buf = try e.fmt("s.{s}.T", .{w}) }, .i32, &.{ b1 - b0, @divExact(k, 16), 8, tw }, &.{ blk_words, 8 * tw, tw, 1 }, b0 * blk_words * 4),
                    wq, .{ .i = st[0] }, .{ .i = st[1] }, .{ .i = k2 }, .{ .i = 2 },
                }),
                // Exl3Linear.strides of the stored layout: (words between k tiles, between 16-column tiles)
                .stored => try e.ext("tensorfold_exl3_linear_v4.unpack", &.{
                    try e.view(.{ .weight = try e.fmt("{s}.trellis", .{w}) }, .i32, &.{ @divExact(k, 16), @divExact(n, 16), tw }, &.{ @divExact(n, 16) * tw, tw, 1 }, 0),
                    wq, .{ .i = @divExact(n, 16) * tw }, .{ .i = 8 * tw }, .{ .i = k2 }, .{ .i = 2 },
                }),
            }
            const svh = try e.view(.{ .weight = try e.fmt("{s}.svh", .{w}) }, .f16, &.{nc}, &.{1}, c0 * 2);
            const out = try e.view(y.role, y.dt, &.{ m, nc }, &.{ y.stride, 1 }, y.offset + c0 * size(y.dt));
            // exl3.prefill.tiles: BM 128, BK 32, group 8 (the shape's alone)
            try e.triton("_gemm", .{ cdiv(m, 128) * @divExact(nc, 128), 1, 1 }, &.{
                .{ .name = "X", .arg = xh },                                   .{ .name = "W", .arg = wq },
                .{ .name = "H", .arg = try e.buf("s.pf.had", .{}, .bf16, &.{ 128, 128 }) }, .{ .name = "SVH", .arg = svh },
                .{ .name = "BIAS", .arg = svh },                               .{ .name = "OUT", .arg = out },
                .{ .name = "M", .arg = .{ .i = m } },                          .{ .name = "o_stride", .arg = .{ .i = y.stride } },
                .{ .name = "K", .arg = .{ .i = k } },                          .{ .name = "N", .arg = .{ .i = nc } },
                .{ .name = "BM", .arg = .{ .i = 128 } },                       .{ .name = "BK", .arg = .{ .i = 32 } },
                .{ .name = "GROUP", .arg = .{ .i = 8 } },                      .{ .name = "HAS_BIAS", .arg = .{ .b = false } },
                .{ .name = "SCALE", .arg = .{ .f = had_scale } },
            });
        }
    }

    /// A projection into a whole contiguous buffer.
    fn proj(s: *Seg, w: []const u8, form: Form, x: Arg, k: i64, n: i64, y: Arg) !void {
        return s.matmul(w, form, x, k, n, .{ .role = y.t.role, .dt = y.t.dt, .stride = n, .offset = y.t.offset });
    }

    // -----------------------------------------------------------------------------------------------------------
    // mHC (mhc.py with mhc_pf)

    fn coefs(e: *E, set: u1) ![3]Arg {
        return .{
            try e.buf("w.c{d}.pre", .{set}, .f32, &.{ e.n, 4 }),
            try e.buf("w.c{d}.post", .{set}, .f32, &.{ e.n, 4 }),
            try e.buf("w.c{d}.comb", .{set}, .f32, &.{ e.n, 16 }),
        };
    }

    /// DSpark's tap (forward._layers): the new streams' mean at a boundary entering a target layer.
    fn tapOf(e: *E, L: u32, which: []const u8, mode: moe.Mode) !?Arg {
        if (!e.o.taps or mode != .boundary or !std.mem.eql(u8, which, "hc_attn")) return null;
        const t = e.cfg.dspark_targets.items();
        const j = std.mem.indexOfScalar(u32, t, L) orelse return null;
        const D: i64 = e.cfg.hidden;
        return try e.view(.{ .buf = "w.taps" }, .bf16, &.{ e.n, D }, &.{ @as(i64, @intCast(t.len)) * D, 1 }, @as(i64, @intCast(j)) * D * 2);
    }

    /// The exchange a post reads: the attention's partials at the FFN site, the MoE's anywhere else. Both are
    /// layer-scoped: Python's allocator gives each exchange its own storage (pod 18's capture moved the MoE's between
    /// layers); the role stays one buffer here, read by the next layer's first site.
    fn gathered(e: *E, which: []const u8) !Arg {
        const D: i64 = e.cfg.hidden;
        if (std.mem.eql(u8, which, "hc_ffn")) return e.buf("L.recv.attn", .{}, .bf16, &.{ e.o.world, e.n, D });
        return e.buf("L.recv.moe", .{}, .bf16, &.{ e.o.world, e.n, D });
    }

    /// One mixing site of layer L (`which`: hc_attn / hc_ffn): mhc.site / mhc.boundary, then `_finish_k` (a short
    /// segment: mhc_cuda alone).
    fn site(s: *Seg, L: u32, which: []const u8, mode: moe.Mode) !void {
        const e = s.e;
        if (s.short) return moe.mhc(e, L, which, mode);
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const post = mode == .boundary;
        const x = try e.streams();
        const prev = try coefs(e, e.coef);
        const next = try coefs(e, 1 - e.coef);
        const b0 = try e.empty(.bf16);
        const f0 = try e.empty(.f32);
        const fnw = try e.buf("s.L{d}.{s}.fn16", .{ L, which }, .bf16, &.{ 6 * @as(i64, e.cfg.hc_mult), 4 * D });
        const part = try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 });
        const c = try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D });
        const tap = try tapOf(e, L, which, mode);
        // the forward's second streams buffer (mhc.out_of_place): a boundary of at most 32 rows writes it
        const spare = post and n <= split_rows;
        const xout = if (spare) try e.spare() else x;
        if (n >= pf_rows) {
            try e.ext("tf_dsv41_mhc_pf_v1.run", &.{
                x,                            if (post) xout else b0,
                if (post) try gathered(e, which) else b0, if (post) prev[1] else f0,
                if (post) prev[2] else f0,    if (mode != .site1) prev[0] else f0,
                fnw,                          part,
                c,                            tap orelse b0,
                .{ .i = n },                  .{ .i = @intFromEnum(mode) },
            });
        } else try s.siteTriton(x, if (post) xout else null, if (post) try gathered(e, which) else null, prev, if (mode != .site1) prev[0] else null, fnw, part, c, tap, if (mode == .site1) 1 else 2, true);
        try s.finish(.{ L, which }, next, try e.weight("L{d}.{s}", .{ L, if (std.mem.eql(u8, which, "hc_attn")) "attn_norm" else "ffn_norm" }, .f32, &.{D}), part, c);
        e.coef = 1 - e.coef;
        if (spare) e.cur = 1 - e.cur;
    }

    /// mhc._launch's Triton `_site`: `xout` / `g` with the post, `pre` with collapse 2, `fnw` with the mix.
    fn siteTriton(s: *Seg, x: Arg, xout: ?Arg, g: ?Arg, prev: [3]Arg, pre: ?Arg, fnw: ?Arg, part: Arg, c: Arg, tap: ?Arg, collapse: i64, mix: bool) !void {
        const e = s.e;
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const post = g != null;
        const bm: i64 = if (n >= pf_rows) mhc_bm else 16;
        // a program a stream for small mixing windows, never in place with a post
        const split = mix and n <= split_rows and !(post and std.mem.eql(u8, xout.?.t.role.buf, x.t.role.buf));
        try e.triton("_site", .{ cdiv(n, bm), 40, if (split) 4 else 1 }, &.{
            .{ .name = "X", .arg = x },                                    .{ .name = "x_stride", .arg = .{ .i = 4 * D } },
            .{ .name = "XOUT", .arg = xout orelse x },                     .{ .name = "G", .arg = g orelse x },
            .{ .name = "g_rank", .arg = .{ .i = if (post) n * D else 0 } }, .{ .name = "POST", .arg = if (post) prev[1] else x },
            .{ .name = "COMB", .arg = if (post) prev[2] else x },          .{ .name = "PRE", .arg = pre orelse x },
            .{ .name = "FN", .arg = fnw orelse x },                        .{ .name = "PART", .arg = part },
            .{ .name = "C", .arg = c },                                    .{ .name = "TAP", .arg = tap orelse x },
            .{ .name = "tap_stride", .arg = .{ .i = if (tap) |t| t.t.stride[0] else 0 } }, .{ .name = "R", .arg = .{ .i = n } },
            .{ .name = "D", .arg = .{ .i = D } },                          .{ .name = "NB", .arg = .{ .i = 40 } },
            .{ .name = "BM", .arg = .{ .i = bm } },                        .{ .name = "BK", .arg = .{ .i = 16 } },
            .{ .name = "WORLD", .arg = .{ .i = if (post) e.o.world else 1 } }, .{ .name = "POST_ON", .arg = .{ .b = post } },
            .{ .name = "COLLAPSE", .arg = .{ .i = collapse } },           .{ .name = "MIX", .arg = .{ .b = mix } },
            .{ .name = "TAP_ON", .arg = .{ .b = tap != null } },           .{ .name = "SPLIT", .arg = .{ .b = split } },
        });
    }

    /// mhc._finish: the partials' sums, the next coefficients (a layer's site) and the normed input into "w.out".
    fn finish(s: *Seg, hc: ?struct { u32, []const u8 }, next: [3]Arg, nw: Arg, part: Arg, c: Arg) !void {
        const e = s.e;
        const D: i64 = e.cfg.hidden;
        const coef = hc != null;
        const base = if (hc) |h| try e.weight("L{d}.{s}.base", .{ h[0], h[1] }, .f32, &.{6 * @as(i64, e.cfg.hc_mult)}) else part;
        const scale = if (hc) |h| try e.weight("L{d}.{s}.scale", .{ h[0], h[1] }, .f32, &.{3}) else part;
        try e.triton("_finish_k", .{ e.n, 1, 1 }, &.{
            .{ .name = "PART", .arg = part },                              .{ .name = "BASE", .arg = base },
            .{ .name = "SCALE", .arg = scale },                            .{ .name = "PRE", .arg = if (coef) next[0] else part },
            .{ .name = "POST", .arg = if (coef) next[1] else part },       .{ .name = "COMB", .arg = if (coef) next[2] else part },
            .{ .name = "C", .arg = c },                                    .{ .name = "NW", .arg = nw },
            .{ .name = "OUT", .arg = try e.buf("w.out", .{}, .bf16, &.{ e.n, D }) }, .{ .name = "eps", .arg = .{ .f = E.dec(e.cfg.eps) } },
            .{ .name = "hc_eps", .arg = .{ .f = E.dec(e.cfg.hc_eps) } },  .{ .name = "post_alpha", .arg = .{ .f = E.dec(e.cfg.hc_post_alpha) } },
            .{ .name = "D", .arg = .{ .i = D } },                          .{ .name = "NB", .arg = .{ .i = 40 } },
            .{ .name = "ITERS", .arg = .{ .i = e.cfg.hc_sinkhorn_iters } }, .{ .name = "COEF", .arg = .{ .b = coef } },
            .{ .name = "BLOCK", .arg = .{ .i = @intCast(std.math.gcd(@as(u64, 1024), @as(u64, @intCast(D)))) } }, .{ .name = "UNROLL", .arg = .{ .b = true } },
            .{ .name = "NORM_FIRST", .arg = .{ .b = true } },
        });
    }

    /// mhc.post_only: the last exchange's post into `xout` (the streams in place, or the trace's copy).
    fn postOnly(s: *Seg, xout: Arg) !void {
        const e = s.e;
        if (s.short) return moe.postOnly(e, xout);
        const prev = try coefs(e, e.coef);
        const x = try e.streams();
        try s.siteTriton(x, xout, try gathered(e, ""), prev, null, null, x, x, null, 0, false);
    }

    /// mhc.final: the last post, collapse 2, the final norm; then the head's columns over every row.
    fn final(s: *Seg) !void {
        const e = s.e;
        if (s.short) return moe.finish(e); // mhc_cuda's final, then the same head
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const x = try e.streams();
        const prev = try coefs(e, e.coef);
        const part = try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 });
        const c = try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D });
        try s.siteTriton(x, x, try gathered(e, ""), prev, prev[0], null, part, c, null, 2, false);
        try s.finish(null, prev, try e.weight("norm", .{}, .f32, &.{D}), part, c);
        // the head: the decode path's linear (prefill_mm.fast off), upstream's 128 rows a launch
        const V: i64 = e.cfg.vocab / e.o.world;
        var a: i64 = 0;
        while (a < n) : (a += 128) {
            const m = @min(n - a, 128);
            e.n = m;
            defer e.n = n;
            try e.upstream("head", try e.view(.{ .buf = "w.out" }, .bf16, &.{ m, D }, &.{ D, 1 }, a * D * 2), D, V, try e.view(.{ .buf = "w.logits" }, .f32, &.{ m, V }, &.{ V, 1 }, a * V * 4));
        }
        try e.glue("kit_logits", &.{try e.buf("w.logits", .{}, .f32, &.{ n, V })});
    }

    // -----------------------------------------------------------------------------------------------------------
    // attention (blocks.attention's serial path, _segment with PF_COPIES)

    fn pos(e: *E) !Arg {
        return e.buf("w.pos", .{}, .i32, &.{1});
    }

    fn rope(e: *E, x: Arg, heads: i64, dim: i64) !void {
        const n = e.n;
        const xv = try e.view(x.t.role, .bf16, &.{ n, heads, dim }, &.{ heads * dim, dim, 1 }, 0);
        try e.triton("_rope", .{ n, @divExact(heads, 4), 1 }, &.{
            .{ .name = "X", .arg = xv },                               .{ .name = "x_rs", .arg = .{ .i = heads * dim } },
            .{ .name = "x_hs", .arg = .{ .i = dim } },                 .{ .name = "OUT", .arg = xv },
            .{ .name = "o_rs", .arg = .{ .i = heads * dim } },         .{ .name = "o_hs", .arg = .{ .i = dim } },
            .{ .name = "CS", .arg = try e.ropeTable() },              .{ .name = "cs_rs", .arg = .{ .i = 64 } },
            .{ .name = "POS", .arg = try e.buf("w.pos64", .{}, .i64, &.{n}) }, .{ .name = "H", .arg = .{ .i = heads } },
            .{ .name = "NOPE", .arg = .{ .i = dim - 64 } },            .{ .name = "NB", .arg = .{ .i = 64 } },
            .{ .name = "HALF", .arg = .{ .i = 32 } },                  .{ .name = "BH", .arg = .{ .i = 4 } },
            .{ .name = "INV", .arg = .{ .b = false } },                .{ .name = "COPY", .arg = .{ .b = false } },
            .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }

    /// The SWA ring this segment's layer L stores into (values, scales): the staging ring or the slot's.
    fn swa(s: *Seg, L: u32) ![2]Arg {
        const e = s.e;
        if (s.staged) return .{ try e.buf("s.stage.v", .{}, .u8, &.{ s.ring, 576 }), try e.buf("s.stage.s", .{}, .u8, &.{ s.ring, 8 }) };
        return slotRing(e, L);
    }

    fn slotRing(e: *E, L: u32) ![2]Arg {
        const r = 2 * @as(i64, e.cfg.window);
        return .{ try e.buf("s.L{d}.swa.v", .{L}, .u8, &.{ r, 576 }), try e.buf("s.L{d}.swa.s", .{L}, .u8, &.{ r, 8 }) };
    }

    /// A KV source's compressed pool (values, scales).
    fn pool(e: *E, L: u32) ![2]Arg {
        const rows = @divFloor(e.o.limit, e.cfg.compressRatio(L)) + 1;
        return .{ try e.buf("s.L{d}.comp.v", .{L}, .u8, &.{ rows, 576 }), try e.buf("s.L{d}.comp.s", .{L}, .u8, &.{ rows, 8 }) };
    }

    fn indexKeys(e: *E, L: u32) !Arg {
        const r: i64 = e.cfg.compressRatio(L);
        return e.buf("s.L{d}.ik", .{L}, .u8, &.{ @divFloor(e.o.limit, r) + 1, @as(i64, e.cfg.index_dim) + 4 });
    }

    /// compress.kv_store: rows into the SWA ring (ratio 0) or a compressed pool.
    fn kvStore(s: *Seg, lat: Arg, rows: [2]Arg, ratio: i64) !void {
        const e = s.e;
        try e.triton("_kv_store", .{ e.n, 1, 1 }, &.{
            .{ .name = "LAT", .arg = lat },                          .{ .name = "l_stride", .arg = .{ .i = e.cfg.head_dim } },
            .{ .name = "CS", .arg = try e.ropeTable() },             .{ .name = "cs_stride", .arg = .{ .i = 64 } },
            .{ .name = "V", .arg = rows[0] },                        .{ .name = "S", .arg = rows[1] },
            .{ .name = "v_stride", .arg = .{ .i = 576 } },           .{ .name = "s_stride", .arg = .{ .i = 8 } },
            .{ .name = "POS", .arg = try pos(e) },                   .{ .name = "RATIO", .arg = .{ .i = ratio } },
            .{ .name = "RING", .arg = .{ .i = if (ratio == 0) s.ring else 1 } }, .{ .name = "PT", .arg = .none },
            .{ .name = "PSH", .arg = .{ .i = 0 } },                  .{ .name = "SL", .arg = .none },
            .{ .name = "PTS", .arg = .{ .i = 0 } },                  .{ .name = "ROWS", .arg = .{ .b = false } },
            .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }

    /// A KV source's compressor (blocks._compress): pooled rows normed into its pool, the index keys from them.
    fn compress(s: *Seg, L: u32, projection: Arg) !void {
        const e = s.e;
        const n = e.n;
        const hd: i64 = e.cfg.head_dim;
        const ID: i64 = e.cfg.index_dim;
        const ratio: i64 = e.cfg.compressRatio(L);
        // ratio 2: [the slot's carry (position start - 1) | the rows]
        const buf = if (ratio == 2) try e.buf("L.cbuf", .{}, .f32, &.{ n + 1, 2 * hd }) else projection;
        if (ratio == 2) try e.glue("proj_carry", &.{ try e.buf("s.L{d}.carry", .{L}, .f32, &.{ 1, 2 * hd }), projection, buf });
        const lat = try e.buf("L.lat", .{}, .bf16, &.{ n, hd });
        try e.glue("zeros", &.{lat});
        try e.triton("_pool_norm", .{ n, 1, 1 }, &.{
            .{ .name = "BUF", .arg = buf },                                                   .{ .name = "b_stride", .arg = .{ .i = ratio * hd } },
            .{ .name = "W", .arg = try e.weight("L{d}.attn.comp_norm", .{L}, .f32, &.{hd}) }, .{ .name = "LAT", .arg = lat },
            .{ .name = "POS", .arg = try pos(e) },                                            .{ .name = "n", .arg = .{ .i = n } },
            .{ .name = "EPS", .arg = .{ .f = E.dec(e.cfg.eps) } },                            .{ .name = "RATIO", .arg = .{ .i = ratio } },
            .{ .name = "SL", .arg = .none },                                                  .{ .name = "PREV", .arg = .none },
            .{ .name = "OFF", .arg = .{ .i = 0 } },                                           .{ .name = "ROWS", .arg = .{ .b = false } },
            .{ .name = "PDL", .arg = .{ .b = false } },                                       .{ .name = "CARRY", .arg = .none },
            .{ .name = "c_stride", .arg = .{ .i = 0 } },                                      .{ .name = "SPLIT", .arg = .{ .b = false } },
        });
        try s.kvStore(lat, try pool(e, L), ratio);
        const kp = try e.buf("L.ix_k", .{}, .bf16, &.{ n, ID });
        try s.proj(try e.fmt("L{d}.attn.ix_wk", .{L}), .strips, lat, hd, ID, kp);
        try e.triton("_index_k", .{ n, 1, 1 }, &.{
            .{ .name = "KP", .arg = kp },                                                    .{ .name = "k_stride", .arg = .{ .i = ID } },
            .{ .name = "W", .arg = try e.weight("L{d}.attn.ix_knorm", .{L}, .f32, &.{ID}) }, .{ .name = "CS", .arg = try e.ropeTable() },
            .{ .name = "cs_stride", .arg = .{ .i = 64 } },                                   .{ .name = "IK", .arg = try indexKeys(e, L) },
            .{ .name = "POS", .arg = try pos(e) },                                           .{ .name = "EPS", .arg = .{ .f = E.dec(e.cfg.eps) } },
            .{ .name = "RATIO", .arg = .{ .i = ratio } },                                    .{ .name = "PT", .arg = .none },
            .{ .name = "PSH", .arg = .{ .i = 0 } },                                          .{ .name = "KFP8", .arg = .{ .b = true } },
            .{ .name = "SL", .arg = .none },                                                 .{ .name = "PTS", .arg = .{ .i = 0 } },
            .{ .name = "ROWS", .arg = .{ .b = false } },                                     .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }

    /// The segment's selection of index layer `src` (reuse layers read the latest source's).
    fn selection(e: *E, src: u32) ![2]Arg {
        return .{ try e.buf("w.sel{d}", .{src}, .i32, &.{ e.n, e.cfg.index_topk }), try e.buf("w.cnt{d}", .{src}, .i32, &.{e.n}) };
    }

    /// Rows [a, a + rows) of a row-major [n, cols] role.
    fn rowsOf(e: *E, x: Arg, dt: Dt, a: i64, rows: i64, cols: i64) !Arg {
        return e.view(x.t.role, dt, &.{ rows, cols }, &.{ cols, 1 }, a * cols * size(dt));
    }

    /// index.scores over rows [a, a + rows) of the segment: dense (key i = position i) or over `keys`' positions
    /// (`cbs` > 0: `keys` are candidate blocks of `cbs` positions, a short segment's Reindex layer).
    fn scores(s: *Seg, keys_of: u32, ratio: i64, a: i64, rows: i64, p: Arg, nk: i64, keys: ?Arg, cbs: i64) !Arg {
        const e = s.e;
        const IH: i64 = e.cfg.index_heads;
        const ID: i64 = e.cfg.index_dim;
        const out = try e.buf("L.ix.scores", .{}, .f32, &.{ rows, nk });
        try e.triton("_scores", .{ rows, cdiv(nk, 64), 1 }, &.{
            .{ .name = "QI", .arg = try e.view(.{ .buf = "L.qi" }, .bf16, &.{ rows, IH, ID }, &.{ IH * ID, ID, 1 }, a * IH * ID * 2) },
            .{ .name = "W", .arg = try e.view(.{ .buf = "L.ix_w" }, .f32, &.{ rows, IH }, &.{ IH, 1 }, a * IH * 4) },
            .{ .name = "w_stride", .arg = .{ .i = IH } },               .{ .name = "IK", .arg = try indexKeys(e, keys_of) },
            .{ .name = "OUT", .arg = out },                             .{ .name = "POS", .arg = p },
            .{ .name = "KEYS", .arg = keys orelse out },                .{ .name = "k_stride", .arg = .{ .i = if (cbs > 0) keys.?.t.shape[1] else nk } },
            .{ .name = "NK", .arg = .{ .i = nk } },                     .{ .name = "o_stride", .arg = .{ .i = nk } },
            .{ .name = "RATIO", .arg = .{ .i = ratio } },               .{ .name = "H", .arg = .{ .i = IH } },
            .{ .name = "D", .arg = .{ .i = ID } },                      .{ .name = "BP", .arg = .{ .i = 64 } },
            .{ .name = "WS", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(IH))) } },
            .{ .name = "SCALE", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(ID))) } },
            .{ .name = "GATHER", .arg = .{ .b = keys != null } },       .{ .name = "PT", .arg = .none },
            .{ .name = "PSH", .arg = .{ .i = 0 } },                     .{ .name = "KFP8", .arg = .{ .b = true } },
            .{ .name = "SL", .arg = .none },                            .{ .name = "PTS", .arg = .{ .i = 0 } },
            .{ .name = "ROWS", .arg = .{ .b = false } },                .{ .name = "CBS", .arg = .{ .i = cbs } },
        });
        return out;
    }

    /// index.sort_keys: a unique int64 key a score (`posn`: the gathered keys' positions).
    fn sortKeys(s: *Seg, sc: Arg, rows: i64, nk: i64, posn: ?Arg) !Arg {
        const e = s.e;
        const k = try e.buf("L.ix.keys", .{}, .i64, &.{ rows, nk });
        try e.triton("_keys", .{ rows, cdiv(nk, 1024), 1 }, &.{
            .{ .name = "S", .arg = sc },                    .{ .name = "s_stride", .arg = .{ .i = nk } },
            .{ .name = "K", .arg = k },                     .{ .name = "k_stride", .arg = .{ .i = nk } },
            .{ .name = "POSN", .arg = posn orelse sc },     .{ .name = "p_stride", .arg = .{ .i = nk } },
            .{ .name = "NK", .arg = .{ .i = nk } },         .{ .name = "BLOCK", .arg = .{ .i = 1024 } },
            .{ .name = "GATHER", .arg = .{ .b = posn != null } },
        });
        return k;
    }

    /// index.candidate_blocks: the blocks' keys from the scores, then torch's top-k (glue) into rows [a, ..) of the
    /// candidates.
    fn candidates(s: *Seg, sc: Arg, a: i64, rows: i64, nk: i64, p: Arg, ratio: i64) !void {
        const e = s.e;
        const cbs: i64 = e.cfg.candidate_block_size;
        const cb: i64 = e.cfg.candidate_blocks;
        const nb = cdiv(nk, cbs);
        const k = try e.buf("L.ix.bkeys", .{}, .i64, &.{ rows, nb });
        try e.triton("_block_keys", .{ rows, cdiv(nb, 256), 1 }, &.{
            .{ .name = "S", .arg = sc },                  .{ .name = "s_stride", .arg = .{ .i = nk } },
            .{ .name = "K", .arg = k },                   .{ .name = "k_stride", .arg = .{ .i = nb } },
            .{ .name = "POS", .arg = p },                 .{ .name = "NB", .arg = .{ .i = nb } },
            .{ .name = "RATIO", .arg = .{ .i = ratio } }, .{ .name = "BS", .arg = .{ .i = cbs } },
            .{ .name = "TB", .arg = .{ .i = 256 } },      .{ .name = "ROWS", .arg = .{ .b = false } },
        });
        try e.glue("topk", &.{ k, try rowsOf(e, try e.buf("w.cand", .{}, .i32, &.{ e.n, cb }), .i32, a, rows, cb), .{ .i = cb } });
    }

    /// Row blocks of the materialised indexer (backend._row_blocks): `per_row` bytes a row under the index budget.
    fn step(e: *E, per_row: i64) i64 {
        return if (e.n > 16) @max(16, @divFloor(index_budget, @max(1, per_row))) else e.n;
    }

    /// An index layer's selection (blocks._select over backend.select / select_cand / reindex).
    fn select(s: *Seg, L: u32) !void {
        const e = s.e;
        const n = e.n;
        const md = e.cfg.mode(L);
        const src = e.cfg.kvSource(L).?;
        const ratio: i64 = e.cfg.compressRatio(L);
        const topk: i64 = e.cfg.index_topk;
        const cbs: i64 = e.cfg.candidate_block_size;
        const cb: i64 = e.cfg.candidate_blocks;
        const sel = try selection(e, L);
        if (s.short) {
            // the whole segment's scores, then attn_cuda's top-k (its job list picks a source's candidate blocks; a
            // Reindex layer scores over them) and visible counts
            const cand = try e.buf("w.cand", .{}, .i32, &.{ n, cb });
            if (md == .reindex and e.cfg.usesCandidates(L) and s.cand) {
                const sc = try s.scores(src, ratio, 0, n, try pos(e), cb * cbs, cand, cbs);
                return e.topk(L, sc, cand, cb * cbs, ratio);
            }
            const nvis = @max(@divFloor(e.end(), ratio), 1);
            const sc = try s.scores(src, ratio, 0, n, try pos(e), nvis, null, 0);
            try e.topk(L, sc, cand, nvis, ratio);
            if (e.cfg.isCandidateSource(L)) s.cand = true;
            return;
        }
        if (md == .reindex and e.cfg.usesCandidates(L) and s.cand) {
            // backend.reindex: the row's own scores over its candidates' positions, in row blocks under the budget
            const NK = cb * cbs;
            const whole = n * NK * 16 <= index_budget;
            const st = if (whole) n else step(e, NK * 16);
            var a: i64 = 0;
            var j: usize = 0;
            while (a < n) : ({
                a += st;
                j += 1;
            }) {
                const rows = @min(st, n - a);
                const p = if (whole) try pos(e) else try s.blockPos(j, a);
                const ck = try e.buf("L.ix.ckeys", .{}, .i32, &.{ rows, NK });
                try e.glue("cand_keys", &.{ try rowsOf(e, try e.buf("w.cand", .{}, .i32, &.{ n, cb }), .i32, a, rows, cb), ck, .{ .i = cbs } });
                const sc = try s.scores(src, ratio, a, rows, p, NK, ck, 0);
                try e.glue("topk", &.{ try s.sortKeys(sc, rows, NK, ck), try rowsOf(e, sel[0], .i32, a, rows, topk), .{ .i = topk } });
            }
        } else {
            const cand = e.cfg.isCandidateSource(L);
            const nvis = @max(@divFloor(e.end(), ratio), 1);
            // backend.select's stream path (full-mode layers without the candidates' scores): not ported
            if (!cand and n > 16 and nvis >= stream_min and @popCount(topk) == 1) return error.Unsupported;
            const whole = n * nvis * 12 <= index_budget;
            const st = if (whole) n else step(e, nvis * 12);
            var a: i64 = 0;
            var j: usize = 0;
            while (a < n) : ({
                a += st;
                j += 1;
            }) {
                const rows = @min(st, n - a);
                const p = if (whole) try pos(e) else try s.blockPos(j, a);
                const nk = if (whole) nvis else @max(@divFloor(e.start + a + rows, ratio), 1);
                const sc = try s.scores(src, ratio, a, rows, p, nk, null, 0);
                try e.glue("topk", &.{ try s.sortKeys(sc, rows, nk, null), try rowsOf(e, sel[0], .i32, a, rows, topk), .{ .i = topk } });
                if (cand) try s.candidates(sc, a, rows, nk, p, ratio);
            }
            if (cand) s.cand = true;
        }
        try e.glue("counts", &.{ sel[0], sel[1], try e.buf("w.pos64", .{}, .i64, &.{n}), .{ .i = ratio } });
    }

    /// Win.sub's first position (int32 [1]) of the row block starting at row `a`.
    fn blockPos(s: *Seg, j: usize, a: i64) !Arg {
        const e = s.e;
        const p = try e.buf("L.ix.pos{d}", .{j}, .i32, &.{1});
        try e.glue("block_pos", &.{ p, .{ .i = e.start + a } });
        return p;
    }

    fn attention(s: *Seg, L: u32) !void {
        const e = s.e;
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const md = e.cfg.mode(L);
        const ratio: i64 = e.cfg.compressRatio(L);
        const H: i64 = e.cfg.heads / e.o.world;
        const hd: i64 = e.cfg.head_dim;
        const IH: i64 = e.cfg.index_heads;
        const ID: i64 = e.cfg.index_dim;
        const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
        const indexer = md == .full or md == .reindex;
        const own_kv = md == .full;
        // the x projections (wq_a, wkv, a KV source's compressor), each its own matmul
        const qa = try e.buf("L.qa", .{}, .bf16, &.{ n, e.cfg.q_lora });
        const kva = try e.buf("L.kva", .{}, .bf16, &.{ n, hd });
        try s.proj(try e.fmt("L{d}.attn.wq_a", .{L}), .lanes, x, D, e.cfg.q_lora, qa);
        try s.proj(try e.fmt("L{d}.attn.wkv", .{L}), .lanes, x, D, hd, kva);
        var projection: Arg = undefined;
        if (own_kv) {
            // blocks._projection: fp32 rows (ratio 2: [kv | gate], kept for the commit's carry)
            const cw = try e.buf("L.comp", .{}, .bf16, &.{ n, hd });
            try s.proj(try e.fmt("L{d}.attn.comp_wkv", .{L}), .lanes, x, D, hd, cw);
            if (ratio == 2) {
                const cg = try e.buf("L.comp_gate", .{}, .bf16, &.{ n, hd });
                try s.proj(try e.fmt("L{d}.attn.comp_wgate", .{L}), .lanes, x, D, hd, cg);
                projection = try e.buf("w.L{d}.proj", .{L}, .f32, &.{ n, 2 * hd });
                try e.glue("proj", &.{ cw, cg, projection });
            } else {
                projection = try e.buf("L.proj", .{}, .f32, &.{ n, hd });
                try e.glue("proj", &.{ cw, projection });
            }
        }
        const qn = try e.buf("L.qn", .{}, .bf16, &.{ n, e.cfg.q_lora });
        const kv = try e.buf("L.kv", .{}, .bf16, &.{ n, hd });
        try e.triton("_rms2", .{ n, 2, 1 }, &.{
            .{ .name = "XA", .arg = qa },                                                      .{ .name = "xa_rs", .arg = .{ .i = e.cfg.q_lora } },
            .{ .name = "WA", .arg = try e.weight("L{d}.attn.q_norm", .{L}, .f32, &.{e.cfg.q_lora}) }, .{ .name = "OA", .arg = qn },
            .{ .name = "oa_rs", .arg = .{ .i = e.cfg.q_lora } },                              .{ .name = "KA", .arg = .{ .i = e.cfg.q_lora } },
            .{ .name = "inv_ka", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(e.cfg.q_lora)) } }, .{ .name = "XB", .arg = kva },
            .{ .name = "xb_rs", .arg = .{ .i = hd } },                                        .{ .name = "WB", .arg = try e.weight("L{d}.attn.kv_norm", .{L}, .f32, &.{hd}) },
            .{ .name = "OB", .arg = kv },                                                     .{ .name = "ob_rs", .arg = .{ .i = hd } },
            .{ .name = "KB", .arg = .{ .i = hd } },                                           .{ .name = "inv_kb", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(hd)) } },
            .{ .name = "eps", .arg = .{ .f = E.dec(e.cfg.eps) } },                            .{ .name = "BK", .arg = .{ .i = 512 } },
            .{ .name = "NARROW", .arg = .{ .b = true } },                                     .{ .name = "PDL", .arg = .{ .b = false } },
        });
        const q = try e.buf("L.q", .{}, .bf16, &.{ n, H * hd });
        try s.proj(try e.fmt("L{d}.attn.wq_b", .{L}), .lanes, qn, e.cfg.q_lora, H * hd, q);
        const qi = try e.buf("L.qi", .{}, .bf16, &.{ n, IH * ID });
        if (indexer) try s.proj(try e.fmt("L{d}.attn.ix_wq_b", .{L}), .lanes, qn, e.cfg.q_lora, IH * ID, qi);
        if (s.short) try e.prefetchO(L);
        try rope(e, q, H, hd);
        if (indexer) {
            try rope(e, qi, IH, ID);
            // the head weights: `_plain`'s fp64 sums, cast to bf16 then fp32 by torch
            const w64 = try e.buf("L.ix_w64", .{}, .f64, &.{ n, IH });
            try e.triton("_plain", .{ n, IH, 1 }, &.{
                .{ .name = "X", .arg = x },                                                      .{ .name = "x_rs", .arg = .{ .i = D } },
                .{ .name = "W", .arg = try e.weight("L{d}.attn.ix_wp", .{L}, .f32, &.{ IH, D }) }, .{ .name = "w_rs", .arg = .{ .i = D } },
                .{ .name = "OUT", .arg = w64 },                                                  .{ .name = "K", .arg = .{ .i = D } },
                .{ .name = "N", .arg = .{ .i = IH } },                                           .{ .name = "BK", .arg = .{ .i = 256 } },
                .{ .name = "PDL", .arg = .{ .b = false } },                                      .{ .name = "BF16", .arg = .{ .b = false } },
            });
            try e.glue("ix_w", &.{ w64, try e.buf("L.ix_w", .{}, .f32, &.{ n, IH }) });
        }
        // blocks._segment: a segment the slot's ring cannot hold runs on the staging ring, the window before it in
        const rows = try s.swa(L);
        const slot = try slotRing(e, L);
        const W: i64 = e.cfg.window;
        const ring0 = 2 * W;
        if (s.staged) try e.glue("stage_in", &.{ rows[0], rows[1], .{ .i = s.ring }, slot[0], slot[1], .{ .i = ring0 }, .{ .i = @max(0, e.start - W + 1) }, .{ .i = e.start } });
        try s.kvStore(kv, rows, 0);
        if (own_kv) try s.compress(L, projection);
        if (indexer) try s.select(L);
        // the fused prefill core over the SWA ring and (a compressed layer) the source's pool at the latest selection
        const groups: i64 = e.cfg.o_groups / e.o.world;
        if (@mod(H, bmq) != 0 or @mod(H, groups) != 0) return error.Unsupported;
        const hg = @divExact(H, groups);
        const o = try e.buf("L.o", .{}, .bf16, &.{ groups, n, hg * hd });
        const lo = try e.buf("w.lo", .{}, .i32, &.{n});
        var cv = rows[0];
        var cs = rows[1];
        var tok = lo;
        var cnt = lo;
        var t_stride: i64 = 0;
        if (ratio > 0) {
            const p = try pool(e, e.cfg.kvSource(L).?);
            cv = p[0];
            cs = p[1];
            const sel = try selection(e, e.cfg.indexSource(L).?);
            tok = sel[0];
            cnt = sel[1];
            t_stride = e.cfg.index_topk;
        }
        try e.triton("_fused", .{ n, @divExact(H, bmq), 1 }, &.{
            .{ .name = "Q", .arg = try e.view(q.t.role, .bf16, &.{ n, H, hd }, &.{ H * hd, hd, 1 }, 0) },
            .{ .name = "CV", .arg = cv },                         .{ .name = "CSC", .arg = cs },
            .{ .name = "cvs", .arg = .{ .i = 576 } },             .{ .name = "css", .arg = .{ .i = 8 } },
            .{ .name = "TOK", .arg = tok },                       .{ .name = "t_stride", .arg = .{ .i = t_stride } },
            .{ .name = "CNT", .arg = cnt },                       .{ .name = "SV", .arg = rows[0] },
            .{ .name = "SSC", .arg = rows[1] },                   .{ .name = "svs", .arg = .{ .i = 576 } },
            .{ .name = "sss", .arg = .{ .i = 8 } },               .{ .name = "LO", .arg = lo },
            .{ .name = "POS", .arg = try pos(e) },                .{ .name = "SINK", .arg = try e.weight("L{d}.attn.sink", .{L}, .f32, &.{H}) },
            .{ .name = "CS", .arg = try e.ropeTable() },          .{ .name = "cs_stride", .arg = .{ .i = 64 } },
            .{ .name = "OUT", .arg = o },                         .{ .name = "H", .arg = .{ .i = H } },
            .{ .name = "KT", .arg = .{ .i = kt } },               .{ .name = "BMQ", .arg = .{ .i = bmq } },
            // Python's D ** -0.5 (libm pow, correctly rounded): sqrt of the exact 1 / 512 rounds the same way
            .{ .name = "SCALE", .arg = .{ .f = @sqrt(1.0 / @as(f64, @floatFromInt(hd))) } }, .{ .name = "RING", .arg = .{ .i = s.ring } },
            .{ .name = "WINDOW", .arg = .{ .i = W } },            .{ .name = "COMP", .arg = .{ .b = ratio > 0 } },
            .{ .name = "PT", .arg = .none },                      .{ .name = "PSH", .arg = .{ .i = 0 } },
            .{ .name = "HG", .arg = .{ .i = hg } },               .{ .name = "OR", .arg = .{ .i = n } },
        });
        if (s.staged) try e.glue("stage_out", &.{ slot[0], slot[1], .{ .i = ring0 }, rows[0], rows[1], .{ .i = s.ring }, .{ .i = @max(0, e.end() - W) }, .{ .i = e.end() } });
        // _out_pf: each wo_a group's GEMM over its contiguous slice into its columns of z, then wo_b (bf16 partial)
        const ol: i64 = e.cfg.o_lora;
        const gk = hg * hd;
        const z: Out = .{ .role = .{ .buf = "L.z" }, .dt = .bf16, .stride = groups * ol };
        var g: i64 = 0;
        while (g < groups) : (g += 1) {
            var zg = z;
            zg.offset = g * ol * 2;
            try s.matmul(try e.fmt("L{d}.attn.wo_a.{d}", .{ L, g }), .lanes, try e.view(o.t.role, .bf16, &.{ n, gk }, &.{ gk, 1 }, g * n * gk * 2), gk, ol, zg);
        }
        try s.proj(try e.fmt("L{d}.attn.wo_b", .{L}), .lanes, try e.buf("L.z", .{}, .bf16, &.{ n, groups * ol }), groups * ol, D, try e.buf("L.part", .{}, .bf16, &.{ n, D }));
    }

    // -----------------------------------------------------------------------------------------------------------
    // MoE (prefill_moe.forward: router, x3gm's ragged launches, the shared expert's GEMMs)

    fn moeBlock(s: *Seg, L: u32) !void {
        const e = s.e;
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const ex = e.cfg.expertsOf(L);
        const topk: i64 = ex.topk;
        const slots: i64 = topk + 1;
        const Et: i64 = ex.count;
        const I: i64 = e.cfg.expert_width / e.o.world;
        const P = n * topk;
        const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
        const pick = try e.buf("L.pick", .{}, .i32, &.{ n, slots });
        const wts = try e.buf("L.wts", .{}, .f32, &.{ n, slots });
        // router_gemv.route over every row (one CHUNK), prefill's config (8 warps, 2 experts a warp); at most 16 rows
        // the narrow kernel (TF_DSV41_RG_NARROW 2, as a decode window's)
        const tile: i64 = for ([_]i64{ 1, 2, 4, 8, 16 }) |t| {
            if (t >= @min(n, 16)) break t;
        } else 16;
        const cfg: [2]i64 = if (n <= 16) .{ 2, 0 } else .{ 8, 2 };
        try e.ext("tf_dsv41_router_gemv_v2.route", &.{
            x,                                                        try e.weight("L{d}.moe.gate", .{L}, .bf16, &.{ Et, D }),
            try e.weight("L{d}.moe.bias", .{L}, .f32, &.{Et}),       pick,
            wts,                                                      try e.buf("L.logits", .{}, .f32, &.{ n, Et }),
            try e.buf("s.router.cnt", .{}, .i32, &.{128}),           .{ .i = topk },
            .{ .i = slots },                                          .{ .f = E.dec(e.cfg.routed_scale) },
            .{ .i = tile },                                           .{ .i = cfg[0] },
            .{ .i = cfg[1] },                                         .{ .b = true },
            .{ .b = false },                                          try e.empty(.i32),
            try e.empty(.i32),                                        try e.t(.empty, .i32, &.{ 0, 1 }),
            .{ .i = 0 },                                              .{ .b = false },
        });
        if (e.o.r1) {
            // the R1 twin's route always takes the prune list (router_gemv.route: list(prune or [])); prefill's is
            // empty (the prune runs after it, prefill_moe's rule)
            const last = &e.out.items[e.out.items.len - 1];
            const more = try e.a.alloc(calls.Named, last.args.len + 1);
            @memcpy(more[0..last.args.len], last.args);
            more[last.args.len] = .{ .arg = .{ .list = &.{} } };
            last.args = more;
        }
        // the kit's fp16 weights, the shared slot masked, the routed slots' contiguous copies (one row: torch's
        // .contiguous() keeps the [1, topk] view of the [1, slots] rows, the same bytes)
        const rs: i64 = if (n == 1) slots else topk;
        const pk = try e.view(.{ .buf = "L.gm.pick" }, .i32, &.{ n, topk }, &.{ rs, 1 }, 0);
        const w6 = try e.view(.{ .buf = "L.gm.wts" }, .f32, &.{ n, topk }, &.{ rs, 1 }, 0);
        try e.glue("gm_picks", &.{ pick, wts, pk, w6 });
        // x3gm._run_ragged: one row block (X / Y in the experts' scratch z, Xd in its xg), rot, a gate/up launch a
        // gate width present, a down launch a down width present, upstream's combine
        const widths = e.w.gm.get(L) orelse return error.NoWidth;
        const rows: i64 = e.o.expert_rows;
        const zb = rows * (topk + 1) * @max(4 * 2 * I, D); // the scratch z's floats (block_moe)
        if (P * D > zb or P * I > rows * (topk + 1) * D) return error.Unsupported; // x3gm.Buffers: rows halve
        const pre = try e.fmt("L{d}.moe", .{L});
        const xg = try e.view(.{ .buf = "s.ex.z" }, .f16, &.{ P, D }, &.{ D, 1 }, 0);
        const xu = try e.view(.{ .buf = "s.ex.z" }, .f16, &.{ P, D }, &.{ D, 1 }, P * D * 2);
        const xd = try e.view(.{ .buf = "s.ex.xg" }, .f16, &.{ P, I }, &.{ I, 1 }, 0);
        const y = try e.view(.{ .buf = "s.ex.z" }, .f32, &.{ P, D }, &.{ D, 1 }, 0);
        try e.ext("tf_dsv41_x3gm_v1.rot", &.{
            x,  .{ .i = D }, pk, try e.weight("{s}.w1.suh", .{pre}, .f16, &.{ Et, D }), try e.weight("{s}.w3.suh", .{pre}, .f16, &.{ Et, D }),
            xg, xu,          .{ .i = D }, .{ .i = topk }, .{ .i = P }, .{ .i = 2 },
        });
        const ticket = try e.buf("L.gm.ticket", .{}, .i32, &.{2 * gm_widths});
        try e.glue("gm_ticket", &.{ticket});
        const tmax = @divFloor(P, gm_bm) + Et + 1;
        const lim: f64 = E.dec(e.cfg.swiglu_limit);
        for (0..2) |down| {
            var j: i64 = 0;
            for (0..32) |kk| {
                const k2: i64 = @intCast(kk);
                if (widths[down] >> @intCast(kk) & 1 == 0) continue;
                const tag = if (down == 1) "d" else "g";
                const plan = [5]Arg{
                    try e.buf("L.gm.{s}{d}.order", .{ tag, j }, .i32, &.{P}),
                    try e.buf("L.gm.{s}{d}.pe", .{ tag, j }, .i32, &.{tmax}),
                    try e.buf("L.gm.{s}{d}.poff", .{ tag, j }, .i32, &.{tmax}),
                    try e.buf("L.gm.{s}{d}.pcnt", .{ tag, j }, .i32, &.{tmax}),
                    try e.buf("L.gm.{s}{d}.npass", .{ tag, j }, .i32, &.{1}),
                };
                try e.glue("gm_plan", &.{ pk, plan[0], plan[1], plan[2], plan[3], plan[4], .{ .i = k2 }, .{ .b = down == 1 }, .{ .i = gm_bm }, .{ .i = Et } });
                const t = try e.view(ticket.t.role, .i32, &.{1}, &.{1}, (j + if (down == 1) @as(i64, gm_widths) else 0) * 4);
                if (down == 0) {
                    try e.ext("tf_dsv41_x3gm_v1.gateup", &.{
                        xg,                                                  xu,
                        try e.buf("s.L{d}.gm.tg", .{L}, .i64, &.{Et}),      try e.buf("s.L{d}.gm.tu", .{L}, .i64, &.{Et}),
                        plan[0],                                             plan[1],
                        plan[2],                                             plan[3],
                        plan[4],                                             t,
                        try e.weight("{s}.w1.svh", .{pre}, .f16, &.{ Et, I }), try e.weight("{s}.w3.svh", .{pre}, .f16, &.{ Et, I }),
                        try e.weight("{s}.w2.suh", .{pre}, .f16, &.{ Et, I }), xd,
                        .{ .i = D },                                         .{ .i = I },
                        .{ .i = k2 },                                        .{ .b = false },
                        .{ .i = 1 },                                         .{ .b = true },
                        .{ .f = lim },                                       .{ .b = true },
                    });
                } else {
                    try e.ext("tf_dsv41_x3gm_v1.down", &.{
                        xd,          try e.buf("s.L{d}.gm.td", .{L}, .i64, &.{Et}), plan[0],     plan[1],
                        plan[2],     plan[3],                                       plan[4],     t,
                        try e.weight("{s}.w2.svh", .{pre}, .f16, &.{ Et, D }),      y,           .{ .i = I },
                        .{ .i = D }, .{ .i = k2 },                                  .{ .i = 0 }, .{ .b = true },
                        .{ .b = true },
                    });
                }
                j += 1;
            }
        }
        const routed = try e.buf("L.routed", .{}, .f32, &.{ n, D });
        try e.ext("tensorfold_exl3_experts_v1.combine", &.{ y, w6, routed, .{ .i = n }, .{ .i = D }, .{ .i = topk } });
        // the shared expert: gate and up GEMMs, the clamped SiLU product (torch), the down GEMM
        const part = try e.buf("L.moe16", .{}, .bf16, &.{ n, D });
        if (e.cfg.shared_experts > 0) {
            const sh = try e.fmt("{s}.shared.0", .{pre});
            const g = try e.buf("L.sh.g", .{}, .f32, &.{ n, I });
            const u = try e.buf("L.sh.u", .{}, .f32, &.{ n, I });
            const act = try e.buf("L.sh.act", .{}, .f32, &.{ n, I });
            const sy = try e.buf("L.sh.y", .{}, .f32, &.{ n, D });
            try s.proj(try e.fmt("{s}.w1", .{sh}), .stored, x, D, I, g);
            try s.proj(try e.fmt("{s}.w3", .{sh}), .stored, x, D, I, u);
            try e.glue("swiglu", &.{ g, u, act, .{ .f = lim } });
            try s.proj(try e.fmt("{s}.w2", .{sh}), .stored, act, I, D, sy);
            try e.glue("moe_sum", &.{ routed, sy, part });
        } else try e.glue("moe_sum", &.{ routed, part });
    }

    /// Engram before the layer's first site (blocks.engram): wkv over the hashed rows (a prefill matmul), the fusion.
    fn engram(s: *Seg, L: u32) !void {
        const e = s.e;
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const K: i64 = @as(i64, e.cfg.hashCols()) * e.cfg.engram_head_dim;
        const N: i64 = (@as(i64, e.cfg.engram_max_ngram) + 1) * D;
        const kv = try e.buf("L.engram.kv", .{}, .bf16, &.{ n, N });
        const rws = try e.buf("L.engram.rows", .{}, .bf16, &.{ n, K });
        try e.glue("engram_rows", &.{ rws, .{ .i = L } });
        try s.proj(try e.fmt("L{d}.engram.wkv", .{L}), .strips, rws, K, N, kv);
        const qk = try e.weight("L{d}.engram.qk", .{L}, .f32, &.{ e.cfg.hc_mult, D });
        try e.triton("_fuse_dec", .{ n, 1, 1 }, &.{
            .{ .name = "X", .arg = try e.streams() },                               .{ .name = "x_stride", .arg = .{ .i = 4 * D } },
            .{ .name = "KV", .arg = kv },                                           .{ .name = "kv_stride", .arg = .{ .i = N } },
            .{ .name = "QK", .arg = qk },                                           .{ .name = "KEEP", .arg = qk },
            .{ .name = "GATE", .arg = qk },                                         .{ .name = "eps", .arg = .{ .f = E.dec(e.cfg.eps) } },
            .{ .name = "clamp", .arg = .{ .f = E.dec(e.cfg.engram_gate_clamp) } }, .{ .name = "sqrt_d", .arg = .{ .f = @sqrt(@as(f64, @floatFromInt(D))) } },
            .{ .name = "D", .arg = .{ .i = D } },                                   .{ .name = "BK", .arg = .{ .i = 64 } },
            .{ .name = "UB", .arg = .{ .i = 1024 } },                               .{ .name = "HAS_KEEP", .arg = .{ .b = false } },
            .{ .name = "HAS_GATE", .arg = .{ .b = false } },
        });
    }

    // -----------------------------------------------------------------------------------------------------------
    // the segment (forward._run / _layers / _finish)

    /// Where an exchange (`which`: attn / moe) lands: the layer's (`gathered`), or a short segment's "w.recv.*"
    /// (block_moe.mhc's).
    fn recv(s: *Seg, comptime which: []const u8) !Arg {
        const e = s.e;
        return e.buf("{s}.recv." ++ which, .{if (s.short) "w" else "L"}, .bf16, &.{ e.o.world, e.n, e.cfg.hidden });
    }

    fn run(s: *Seg, layers: []const u32, head: bool) !void {
        const e = s.e;
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        try e.glue("positions", &.{ try pos(e), try e.buf("w.pos64", .{}, .i64, &.{n}), try e.buf("w.lo", .{}, .i32, &.{n}) });
        try e.glue("embed", &.{ try e.streams(), try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, n, D }) });
        for (layers, 0..) |L, i| {
            if (i > 0) e.begin = .layer;
            e.layer_ratio = e.cfg.compressRatio(L);
            if (i == 0 or e.cfg.isEngram(L)) {
                // the run's first block, or an Engram block: the last exchange's post alone, Engram, then a site that
                // collapses with the previous coefficients' pre
                if (i > 0) try s.postOnly(try e.streams());
                if (e.cfg.isEngram(L)) try s.engram(L);
                try s.site(L, "hc_attn", if (i == 0) .site1 else .site2);
            } else try s.site(L, "hc_attn", .boundary);
            try s.attention(L);
            if (s.short) try e.prefetchF(L);
            try e.glue("exchange", &.{ try e.buf("L.part", .{}, .bf16, &.{ n, D }), try s.recv("attn") });
            try s.site(L, "hc_ffn", .boundary);
            try s.moeBlock(L);
            if (s.short and i + 1 < layers.len) try e.prefetchX(layers[i + 1]);
            try e.glue("exchange", &.{ try e.buf("L.moe16", .{}, .bf16, &.{ n, D }), try s.recv("moe") });
            if (e.o.trace) {
                const y = try e.buf("w.trace", .{}, .bf16, &.{ n, 4 * D });
                try s.postOnly(y);
                try e.glue("trace", &.{ y, .{ .i = L } });
            }
        }
        if (head) try s.final();
    }
};

/// The calls of a prefill segment of `n` rows from position `start` over `layers` (ascending, consecutive), and the
/// final norm + head when `head` (arena-owned by `a`).
pub fn emitPrefill(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, n: i64, start: i64, head: bool) ![]calls.Call {
    if (n < 1 or n > o.prefill_rows or n > router_chunk) return error.Unsupported;
    var e: E = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = n, .start = start };
    const W: i64 = cfg.window;
    // forward.staging: the slot's ring holds 2 x window rows; a longer segment goes through the staging ring
    const staged = n + W - 1 > 2 * W;
    const stage: i64 = @intCast(std.math.ceilPowerOfTwo(u64, @intCast(o.prefill_rows + W - 1)) catch return error.Unsupported);
    const ring = if (staged) stage else 2 * W;
    if (ring < n + W - 1) return error.Unsupported; // attn_pf: the ring must hold the rows and their windows
    var s: Seg = .{ .e = &e, .staged = staged, .ring = ring, .short = n <= 16 };
    try s.run(layers, head);
    return e.out.items;
}
