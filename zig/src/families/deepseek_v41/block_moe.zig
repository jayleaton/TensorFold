//! The prod decode window's mHC sites (mhc_cuda), MoE (router GEMV, top-p prune, upstream group / rot_in, x3ld,
//! epilogues), Engram and head, emitted as calls (calls.zig) by block.zig's Emitter.

const std = @import("std");
const calls = @import("calls.zig");
const block = @import("block.zig");
const E = block.Emitter;
const Arg = calls.Arg;

/// mhc_cuda.run modes: a sublayer's site without post-processing (the window's first), a boundary between sublayers
/// (post of the last, collapse 2, mix), the final norm (post, collapse 2, no mix).
pub const Mode = enum(i64) { boundary = 0, site1 = 1, final = 3 };

fn coefs(e: *E, set: u1) ![3]Arg {
    return .{
        try e.buf("w.c{d}.pre", .{set}, .f32, &.{ e.n, 4 }),
        try e.buf("w.c{d}.post", .{set}, .f32, &.{ e.n, 4 }),
        try e.buf("w.c{d}.comb", .{set}, .f32, &.{ e.n, 16 }),
    };
}

/// One mHC site of layer L (`which`: hc_attn / hc_ffn), or the final norm (L null).
pub fn mhc(e: *E, layer: ?u32, which: []const u8, mode: Mode) !void {
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const post = mode != .site1;
    const mix = mode != .final;
    const f0 = try e.empty(.f32);
    const b0 = try e.empty(.bf16);
    const x = try e.streams();
    const prev = try coefs(e, e.coef);
    const next = try coefs(e, 1 - e.coef);
    var nw: Arg = undefined;
    var hc = [3]Arg{ b0, f0, f0 };
    if (layer) |L| {
        hc = .{
            try e.buf("s.L{d}.{s}.fn16", .{ L, which }, .bf16, &.{ 6 * @as(i64, e.cfg.hc_mult), 4 * D }),
            try e.weight("L{d}.{s}.base", .{ L, which }, .f32, &.{6 * @as(i64, e.cfg.hc_mult)}),
            try e.weight("L{d}.{s}.scale", .{ L, which }, .f32, &.{3}),
        };
        nw = try e.weight("L{d}.{s}", .{ L, if (std.mem.eql(u8, which, "hc_attn")) "attn_norm" else "ffn_norm" }, .f32, &.{D});
    } else nw = try e.weight("norm", .{}, .f32, &.{D});
    const xout: Arg = switch (mode) {
        .site1 => b0,
        .final => x,
        .boundary => try e.spare(),
    };
    try e.ext("tf_dsv41_mhc_cuda_v1.run", &.{
        x,                                                               xout,
        // the exchange that ran last: the attention's partials at the FFN site, the MoE's at the next attention / final
        if (post) try e.buf("w.recv.{s}", .{if (std.mem.eql(u8, which, "hc_ffn")) "attn" else "moe"}, .bf16, &.{ e.o.world, n, D }) else b0, if (post) prev[1] else f0,
        if (post) prev[2] else f0,                                       if (post) prev[0] else f0,
        hc[0],                                                           hc[1],
        hc[2],                                                           try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 }),
        try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D }),                    b0,
        nw,                                                              try e.buf("w.out", .{}, .bf16, &.{ n, D }),
        if (mix) next[0] else f0,                                        if (mix) next[1] else f0,
        if (mix) next[2] else f0,                                        try e.buf("w.mhc.cnt", .{}, .i32, &.{4}),
        .{ .i = n },                                                     .{ .i = @intFromEnum(mode) },
        .{ .f = E.dec(e.cfg.eps) },                                      .{ .f = E.dec(e.cfg.hc_eps) },
        .{ .f = E.dec(e.cfg.hc_post_alpha) },                            .{ .i = e.cfg.hc_sinkhorn_iters },
    });
    if (mix) e.coef = 1 - e.coef;
    if (mode == .boundary) e.cur = 1 - e.cur;
}

/// Router GEMV, the top-p prune and upstream's group (a DSpark block's router groups itself), then the routed + shared
/// experts: upstream rot_in, x3ld gate/up, the gate/up epilogue, x3ld down, down_combine (this rank's fp32 partial).
pub fn moe(e: *E, L: u32) !void {
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const backbone = e.cfg.isBackbone(L);
    const ex = e.cfg.expertsOf(L);
    const slots: i64 = ex.topk + 1;
    const Et: i64 = ex.count + 1;
    const I: i64 = e.cfg.expert_width / e.o.world;
    const P = n * slots;
    const rows: i64 = if (backbone) e.o.expert_rows else e.o.dspark_rows;
    const sc: []const u8 = if (backbone) "s.ex" else "s.dx";
    const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
    const pick = try e.buf("L.pick", .{}, .i32, &.{ n, slots });
    const wts = try e.buf("L.wts", .{}, .f32, &.{ n, slots });
    const uids = try e.buf("{s}.ids", .{sc}, .i32, &.{P});
    const ucount = try e.buf("{s}.count", .{sc}, .i32, &.{1});
    const members = try e.buf("{s}.members", .{sc}, .i32, &.{ P, n });
    const rows_pad = std.math.ceilPowerOfTwo(u64, @intCast(n)) catch unreachable;
    const scale: f64 = E.dec(e.cfg.routed_scale);
    try e.ext("tf_dsv41_router_gemv_v2.route", &.{
        x,                                                        try e.weight("L{d}.moe.gate", .{L}, .bf16, &.{ ex.count, D }),
        try e.weight("L{d}.moe.bias", .{L}, .f32, &.{ex.count}), pick,
        wts,                                                      try e.buf("L.logits", .{}, .f32, &.{ n, ex.count }),
        try e.buf("s.router.cnt", .{}, .i32, &.{128}),           .{ .i = ex.topk },
        .{ .i = slots },                                          .{ .f = scale },
        .{ .i = @intCast(rows_pad) },                             .{ .i = 2 },
        .{ .i = 0 },                                              .{ .b = true },
        .{ .b = !backbone },                                      if (backbone) try e.empty(.i32) else uids,
        if (backbone) try e.empty(.i32) else ucount,              if (backbone) try e.t(.empty, .i32, &.{ 0, 1 }) else members,
        .{ .i = if (backbone) 0 else Et },                        .{ .b = !backbone },
    });
    if (backbone) {
        try e.triton("_prune", .{ std.math.divCeil(i64, n, 16) catch unreachable, 1, 1 }, &.{
            .{ .name = "PICK", .arg = pick },               .{ .name = "WTS", .arg = wts },
            .{ .name = "R", .arg = .{ .i = n } },           .{ .name = "stride", .arg = .{ .i = slots } },
            .{ .name = "MIN_W", .arg = .{ .f = 0.0 } },     .{ .name = "TOPP", .arg = .{ .f = e.o.expert_topp } },
            .{ .name = "SCALE", .arg = .{ .f = scale } },   .{ .name = "DROP", .arg = .{ .i = Et } },
            .{ .name = "MIN_K", .arg = .{ .i = 3 } },       .{ .name = "K", .arg = .{ .i = ex.topk } },
            .{ .name = "KP", .arg = .{ .i = 8 } },          .{ .name = "DUP", .arg = .{ .b = false } },
            .{ .name = "BR", .arg = .{ .i = 16 } },         .{ .name = "ORIG", .arg = .{ .b = false } },
        });
        try e.ext("tensorfold_exl3_experts_v1.group", &.{ pick, uids, ucount, members, .{ .i = n }, .{ .i = slots }, .{ .i = Et } });
    }
    const pre = try e.fmt("s.L{d}.ex", .{L});
    const xg = try e.buf("{s}.xg", .{sc}, .f16, &.{ rows * slots, D });
    const xu = try e.buf("{s}.xu", .{sc}, .f16, &.{ rows * slots, D });
    const xd = try e.buf("{s}.xd", .{sc}, .f16, &.{ rows * slots, I });
    const z = try e.buf("{s}.z", .{sc}, .f32, &.{rows * slots * @max(4 * 2 * I, D)});
    try e.ext("tensorfold_exl3_experts_v1.rot_in", &.{
        x,  .{ .i = D }, pick, try e.buf("{s}.suh_g", .{pre}, .f16, &.{ Et, D }), try e.buf("{s}.suh_u", .{pre}, .f16, &.{ Et, D }),
        xg, xu,          .{ .i = n }, .{ .i = D }, .{ .i = slots }, .{ .i = Et },
    });
    const r = e.w.experts.get(L) orelse return error.NoWidth;
    const gu = try block.k2Range(r[0][0], r[0][1]);
    const dn = try block.k2Range(r[1][0], r[1][1]);
    try e.ext("tf_dsv41_x3ld_v1.grouped", &.{
        xg,                                                  xu,
        try e.buf("{s}.tp_g", .{pre}, .i64, &.{Et}),        try e.buf("{s}.tp_u", .{pre}, .i64, &.{Et}),
        try e.buf("{s}.k2_g", .{pre}, .i32, &.{Et}),        try e.buf("{s}.k2_u", .{pre}, .i32, &.{Et}),
        uids,                                                ucount,
        members,                                             z,
        .{ .i = 2 },                                         .{ .i = D },
        .{ .i = I },                                         .{ .i = P },
        .{ .i = 4 },                                         .{ .i = slots },
        .{ .i = 2 },                                         .{ .i = 8 },
        .{ .i = 1 },                                         .{ .i = 0 },
        .{ .i = gu[0] },                                     .{ .i = gu[1] },
        .{ .b = false },
    });
    try e.ext("tensorfold_exl3_experts_v1.gateup_epilogue", &.{
        z,                                                pick,
        try e.buf("{s}.svh_g", .{pre}, .f16, &.{ Et, I }), try e.buf("{s}.svh_u", .{pre}, .f16, &.{ Et, I }),
        try e.buf("{s}.suh_d", .{pre}, .f16, &.{ Et, I }), xd,
        .{ .i = n },                                      .{ .i = P },
        .{ .i = I },                                      .{ .i = 4 },
        .{ .i = slots },                                  .{ .i = Et },
        .{ .f = E.dec(e.cfg.swiglu_limit) },              .{ .i = 1 },
    });
    const tp_d = try e.buf("{s}.tp_d", .{pre}, .i64, &.{Et});
    const k2_d = try e.buf("{s}.k2_d", .{pre}, .i32, &.{Et});
    try e.ext("tf_dsv41_x3ld_v1.grouped", &.{
        xd,          xd,          tp_d,        tp_d,        k2_d,        k2_d,            uids,            ucount,
        members,     z,           .{ .i = 1 }, .{ .i = I }, .{ .i = D }, .{ .i = P },     .{ .i = 1 },     .{ .i = slots },
        .{ .i = 2 }, .{ .i = 8 }, .{ .i = 1 }, .{ .i = 0 }, .{ .i = dn[0] }, .{ .i = dn[1] }, .{ .b = false },
    });
    try e.ext("tensorfold_exl3_experts_v1.down_combine", &.{
        z,                                                pick,
        try e.buf("{s}.svh_d", .{pre}, .f16, &.{ Et, D }), try e.buf("{s}.y", .{sc}, .f32, &.{ rows * slots, D }),
        // after a prune, kit_weights' fp16 rounding (a pointwise op the forward runs) gives new weights
        if (backbone) try e.buf("L.wts16", .{}, .f32, &.{ n, slots }) else wts, try e.buf("L.moe", .{}, .f32, &.{ n, D }),
        .{ .i = n },                                      .{ .i = P },
        .{ .i = D },                                      .{ .i = 1 },
        .{ .i = slots },                                  .{ .i = Et },
    });
}

/// Engram (layers 1 and 14), before the layer's first mHC site: wkv over the hashed rows, the gated fusion into the
/// streams.
pub fn engram(e: *E, L: u32) !void {
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const K: i64 = @as(i64, e.cfg.hashCols()) * e.cfg.engram_head_dim;
    const N: i64 = (@as(i64, e.cfg.engram_max_ngram) + 1) * D;
    const kv = try e.buf("L.engram.kv", .{}, .bf16, &.{ n, N });
    try e.upstream(try e.fmt("L{d}.engram.wkv", .{L}), try e.buf("L.engram.rows", .{}, .bf16, &.{ n, K }), K, N, kv);
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

/// The final norm (mHC final) and this rank's head columns (fp32 logits).
pub fn finish(e: *E) !void {
    try mhc(e, null, "", .final);
    const D: i64 = e.cfg.hidden;
    const V: i64 = e.cfg.vocab / e.o.world;
    try e.upstream("head", try e.buf("w.out", .{}, .bf16, &.{ e.n, D }), D, V, try e.buf("w.logits", .{}, .f32, &.{ e.n, V }));
}
