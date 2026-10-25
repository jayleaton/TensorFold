//! The device half of a forward step: kernels, buffers and `Device.emit`; UNEXERCISED (compiled, never launched).
const std = @import("std");
const cuda = @import("cuda");
const plan = @import("forward_plan.zig");
const weights = @import("weights.zig");
const ffn = @import("ffn.zig");
const rows_cfg = @import("rows.zig");
const iface = @import("iface.zig");
const rows_plan = @import("rows_plan.zig");
const sparse = cuda.minimax_m3_sparse;
const rk = cuda.minimax_m3_rows;
const Op = plan.Op;
const Buf = plan.Buf;
const Ref = plan.Ref;

pub const Error = error{ NoLink, BadOp, ScoreScratch };

/// Every kernel module a step launches from.
pub const Modules = struct {
    dense: cuda.exl3.Dense,
    experts: cuda.exl3_experts.Experts,
    gm: cuda.exl3_gm.Experts, // the prompt chunks' routed experts
    norms: cuda.norms.Norms,
    embed: cuda.embed_rows.EmbedRows,
    act: cuda.minimax_m3_act.MiniMaxAct,
    rows: rk.Rows,
    proj: cuda.minimax_m3_proj.Proj,
    sparse: sparse.Sparse,
    scatter: cuda.rank_scatter.RankScatter,
    dspark: cuda.minimax_m3_dspark.Dspark,

    pub fn load(d: *const cuda.Driver) !Modules {
        var m: Modules = undefined;
        m.dense = try cuda.exl3.Dense.load(d);
        errdefer m.dense.unload();
        try m.dense.usePromptTiles();
        m.experts = try cuda.exl3_experts.Experts.load(d);
        errdefer m.experts.unload();
        m.gm = try cuda.exl3_gm.Experts.load(d);
        errdefer m.gm.unload();
        m.norms = try cuda.norms.Norms.load(d);
        errdefer m.norms.unload();
        m.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer m.embed.unload();
        m.act = try cuda.minimax_m3_act.MiniMaxAct.load(d);
        errdefer m.act.unload();
        m.rows = try rk.Rows.load(d);
        errdefer m.rows.unload();
        m.proj = try cuda.minimax_m3_proj.Proj.load(d);
        errdefer m.proj.unload();
        m.sparse = try sparse.Sparse.load(d);
        errdefer m.sparse.unload();
        m.scatter = try cuda.rank_scatter.RankScatter.load(d);
        errdefer m.scatter.unload();
        m.dspark = try cuda.minimax_m3_dspark.Dspark.load(d);
        return m;
    }

    pub fn deinit(m: *Modules) void {
        m.dspark.unload();
        m.scatter.unload();
        m.sparse.unload();
        m.proj.unload();
        m.rows.unload();
        m.act.unload();
        m.embed.unload();
        m.norms.unload();
        m.gm.unload();
        m.experts.unload();
        m.dense.unload();
        m.* = undefined;
    }
};

/// How the two ranks meet: the group's all-gather, and NCCL for the row-share form of a prompt chunk's sum.
pub const Link = struct { group: *const cuda.tp_group.Group, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm };

/// The routed experts' layer shape, for sizing the scratch before any weights exist.
fn shapeLayer(g: *const plan.Geometry) cuda.exl3_experts.Layer {
    return .{ .gate = 0, .up = 0, .down = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .experts = g.experts, .dims = g.hidden, .width = g.moe_w, .codebook = .mul1 };
}

/// Rows of one rank's share of `rows` rows, at most (the padded slot of the share form).
fn shareRows(rows: usize, world: usize) usize {
    return (rows + world - 1) / world;
}

/// Buffers that are not rows of a step: scratch of the kernels and the exchange.
pub const Extra = enum { experts, gathered, scatter, share, shares, bmax, sel, status, gm };

/// Rows of one prompt-experts call the buffers hold: a prompt chunk's, at most the planner's call.
pub fn gmRows(g: *const plan.Geometry) usize {
    return @min(g.max_rows, plan.prompt_expert_rows);
}

/// Bytes of the prompt experts' scratch Buffers holds (its `gm`), for admission before any device exists.
pub fn promptExpertBytes(g: *const plan.Geometry) usize {
    return cuda.exl3_gm.scratchBytes(shapeLayer(g), gmRows(g), g.topk);
}

pub const Buffers = struct {
    pub const slots = plan.bufs.len + std.enums.values(Extra).len;

    mem: [slots]cuda.DeviceBuffer,
    n: usize,
    kvh: u32,

    fn take(b: *Buffers, d: *const cuda.Driver, bytes: usize) !void {
        b.mem[b.n] = try cuda.DeviceBuffer.alloc(d, bytes);
        b.n += 1;
    }

    /// `score_bytes` bounds the block maxima (rows_plan.Plan.scratch); a call's stride is its own block count.
    pub fn init(d: *const cuda.Driver, g: *const plan.Geometry, score_bytes: u64) !Buffers {
        var b: Buffers = .{ .mem = undefined, .n = 0, .kvh = g.kvh };
        errdefer b.deinit();
        for (plan.bufs) |id| try b.take(d, g.rowBytes(id) * g.capacity(id));
        const world: usize = g.world;
        const slot_rows = shareRows(g.max_rows, world);
        const experts = try cuda.exl3_experts.scratchBytes(shapeLayer(g), plan.expert_rows, g.topk);
        const wide: usize = @min(g.max_rows, 128);
        try b.take(d, experts);
        try b.take(d, world * wide * g.hidden * 4);
        try b.take(d, world * slot_rows * g.hidden * 4);
        try b.take(d, slot_rows * g.hidden * 4);
        try b.take(d, world * slot_rows * g.hidden * 4);
        try b.take(d, score_bytes);
        try b.take(d, @as(usize, g.max_rows) * g.kvh * 16 * 4);
        try b.take(d, 4);
        try b.take(d, promptExpertBytes(g));
        return b;
    }

    pub fn deinit(b: *Buffers) void {
        for (b.mem[0..b.n]) |*m| m.free();
        b.n = 0;
    }

    pub fn ptr(b: *const Buffers, id: Buf) u64 {
        return b.mem[@backingInt(id)].ptr;
    }

    pub fn extra(b: *const Buffers, e: Extra) *const cuda.DeviceBuffer {
        return &b.mem[plan.bufs.len + @backingInt(e)];
    }

    /// The scratch of `rows` rows whose last position is at most `max_pos`, packed at that call's stride.
    pub fn scratch(b: *const Buffers, rows: u32, max_pos: u32) Error!sparse.Scratch {
        if (rows_plan.need(b.kvh, rows, max_pos) > b.extra(.bmax).len) return error.ScoreScratch;
        return .{ .bmax = b.extra(.bmax).ptr, .sel = b.extra(.sel).ptr, .bm_stride = rows_plan.blocks(max_pos) };
    }
};

/// Where a call's tapped rows go: the taps buffer and the streams' rows of the call.
pub const Taps = struct {
    base: u64 = 0, // device address of row 0 of the taps buffer (0: the forward does not tap)
    prompt: bool = false,
    segs: [iface.max_shared]iface.Seg = undefined, // rows of each stream in the batch (res rows row0 ...)
    n: u32 = 0,
};

/// The pool the paged kernels address (the engine's arena), and where this call's block tables are.
pub const Pages = struct { cache: rk.Cache, layer_bytes: u64, mirror: *rk.Mirror, slot: u32 = 0, pos0: u32 = 0 };

pub const Device = struct {
    d: *const cuda.Driver,
    st: cuda.Stream,
    m: *Modules,
    w: *const weights.Weights,
    g: *const plan.Geometry,
    b: *Buffers,
    pages: Pages,
    link: ?Link,
    opts: rk.Opts,
    sw: ffn.Switches = .{},
    shared: bool = false, // this call's rows carry their own positions and table offsets
    taps: Taps = .{},

    fn at(dv: *const Device, r: Ref) u64 {
        return dv.b.ptr(r.buf) + @as(u64, r.row) * dv.g.rowBytes(r.buf);
    }

    fn layer(dv: *const Device, l: u32) *const weights.Layer {
        return &dv.w.layers[l];
    }

    fn paged(dv: *const Device, l: u32) !sparse.Paged {
        var c = dv.pages.cache;
        c.layer_off = @as(u64, l) * dv.pages.layer_bytes;
        return if (dv.shared) dv.pages.mirror.shared(c) else try dv.pages.mirror.single(c, dv.pages.slot);
    }

    fn rowsOf(dv: *const Device, n: u32) sparse.Rows {
        return if (dv.shared) .{ .n = n, .pos = dv.b.ptr(.pos) } else .{ .n = n, .pos0 = @intCast(dv.pages.pos0) };
    }

    fn zeroBytes(dv: *const Device, p: u64, bytes: usize) !void {
        try dv.d.check(dv.d.api.cuMemsetD8Async(p, 0, bytes, dv.st.handle), "cuMemsetD8Async");
    }

    fn copy(dv: *const Device, dst: u64, src: u64, bytes: usize) !void {
        try dv.d.check(dv.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, dv.st.handle), "cuMemcpyDtoDAsync");
    }

    fn weightOf(dv: *const Device, w: plan.Lin, l: u32) cuda.exl3.Weight {
        const ly = dv.layer(l);
        return switch (w) {
            .q => ly.q,
            .k => ly.k,
            .v => ly.v,
            .o => ly.o,
            .dense_gate => ly.ffn.dense.gate,
            .dense_up => ly.ffn.dense.up,
            .dense_down => ly.ffn.dense.down,
            .shared_gate => ly.ffn.moe.shared.gate,
            .shared_up => ly.ffn.moe.shared.up,
            .shared_down => ly.ffn.moe.shared.down,
            .head => dv.w.head,
        };
    }

    fn normOf(dv: *const Device, n: plan.Norm, l: u32) u64 {
        return switch (n) {
            .input => dv.layer(l).input_ln,
            .post => dv.layer(l).post_ln,
            .final => dv.w.final_norm,
        };
    }

    /// Launches one op on the stream.
    pub fn emit(dv: *Device, op: Op) !void {
        const s = dv.st;
        const m = dv.m;
        const hidden = dv.g.hidden;
        switch (op) {
            .stage => |o| {
                dv.shared = o.shared;
                try dv.zeroBytes(dv.b.extra(.status).ptr, 4);
            },
            .embed => |o| try m.embed.bf16(s, dv.b.ptr(.ids), dv.w.embed, dv.b.ptr(.emb), o.rows, hidden),
            .zero => |o| try dv.zeroBytes(dv.b.ptr(o.buf), @as(usize, o.rows) * dv.g.rowBytes(o.buf)),
            .norm_add => |o| {
                const kind: cuda.norms.Kind = if (o.x_f32) .f32 else .bf16;
                const wt = dv.normOf(o.norm, o.layer);
                try m.norms.addRms(s, dv.at(o.x), kind, dv.b.ptr(.res), null, wt, dv.at(o.y), .bf16, o.rows, hidden, dv.g.cfg.eps);
            },
            .linear => |o| {
                const w = dv.weightOf(o.w, o.layer);
                const wide = o.w == .o or o.w == .dense_down or o.w == .shared_down or o.w == .head;
                const ydt: cuda.exl3.DType = if (wide) .f32 else .bf16;
                if (o.w == .head and o.rows > 1 and w.n != dv.g.slot) {
                    // The kernel writes row r at stride w.n; the readers (gather, pick) use the largest rank's width.
                    const xb = dv.g.rowBytes(o.x.buf);
                    const yb = dv.g.rowBytes(o.y.buf);
                    var r: u32 = 0;
                    while (r < o.rows) : (r += 1) {
                        try m.dense.decode(s, w, dv.at(o.x) + @as(u64, r) * xb, .bf16, 1, dv.at(o.y) + @as(u64, r) * yb, ydt);
                    }
                } else if (o.prompt) {
                    try m.dense.prompt(s, w, dv.at(o.x), .bf16, o.rows, dv.at(o.y), ydt);
                } else try m.dense.decode(s, w, dv.at(o.x), .bf16, o.rows, dv.at(o.y), ydt);
            },
            .index_proj => |o| {
                const ly = dv.layer(o.layer);
                const ot: cuda.minimax_m3_proj.OutType = if (dv.opts.flags.index_fp16) .f16 else .bf16;
                const q: cuda.minimax_m3_proj.Part = .{ .w = ly.index_q_proj, .n = dv.g.iq_w, .out = dv.b.ptr(.iq_raw), .out_ld = dv.g.iq_w };
                const k: cuda.minimax_m3_proj.Part = .{ .w = ly.index_k_proj, .n = dv.g.ik_w, .out = dv.b.ptr(.ik_raw), .out_ld = dv.g.ik_w };
                try m.proj.project(s, dv.b.ptr(.a), hidden, false, hidden, q, k, o.rows, ot);
            },
            .qk_rope => |o| try dv.qkRope(o.layer, o.rows),
            .score => |o| {
                const scale: f32 = @floatCast(std.math.pow(f64, 128, -0.5));
                try m.sparse.score(s, dv.b.ptr(.iqn), dv.rowsOf(o.rows), try dv.paged(o.layer), try dv.b.scratch(o.rows, o.max_pos), o.max_pos, scale);
            },
            .select => |o| try m.sparse.select(s, dv.rowsOf(o.rows), dv.g.kvh, try dv.b.scratch(o.rows, o.max_pos), .{}),
            .attend => |o| {
                const scale: f32 = @floatCast(std.math.pow(f64, 128, -0.5));
                try m.sparse.attend(s, dv.b.ptr(.qn), dv.rowsOf(o.rows), try dv.paged(o.layer), try dv.b.scratch(o.rows, 0), dv.b.ptr(.attn), scale); // reads only the selections
            },
            .router => |o| try m.rows.routerLogits(s, rows_cfg.router, dv.b.ptr(.a), hidden, false, dv.layer(o.layer).ffn.moe.router, dv.b.ptr(.logits_r), o.rows),
            .route => |o| {
                const moe = dv.layer(o.layer).ffn.moe;
                const rp = try ffn.routePlan(dv.sw);
                try m.act.route(s, dv.b.ptr(.logits_r), moe.bias, rp.route, dv.b.ptr(.pick), dv.b.ptr(.wts), o.rows);
            },
            .experts => |o| try dv.experts(o.layer, o.row0, o.rows, o.prompt),
            .swiglu => |o| {
                const p = ffn.actParams(dv.sw);
                const ld: usize = o.n;
                try m.act.swigluoai(s, .bf16, dv.b.ptr(.gate), ld, dv.b.ptr(.up), ld, dv.b.ptr(.act), ld, o.rows, o.n, p);
            },
            .add2 => |o| {
                // tf_rank_sum as a two-term fp32 add: peers slot 0 is `a`, own (rank 1) is `b`, one rounded add each
                try m.scatter.rankSum(s, dv.b.ptr(o.a), dv.b.ptr(o.b), 2, 1, @as(usize, o.rows) * hidden, dv.b.ptr(o.out));
            },
            .exchange => |o| try dv.sum(o),
            .tap => |o| try dv.tap(o.slot, o.rows),
            .logits_gather => |o| {
                const link = dv.link orelse return error.NoLink;
                const bytes = @as(usize, o.rows) * dv.g.slot * 4;
                try link.group.allGather(s, dv.b.ptr(.hlog), dv.b.ptr(.hgather), bytes);
            },
            .local_pick => |o| try m.proj.pick(s, dv.b.ptr(.hlog), dv.g.slot, dv.g.own_vocab, dv.g.vocab_lo, o.rows, dv.b.ptr(.pairs)),
            .pick_gather => |o| {
                // [rank][row] pairs of (logit bits, id), the layout of the logits gather with a slot of one pair
                const link = dv.link orelse return error.NoLink;
                try link.group.allGather(s, dv.b.ptr(.pairs), dv.b.ptr(.pairs_all), @as(usize, o.rows) * 8);
            },
        }
    }

    /// Routed experts of rows row0..: a prompt call on x3gm when the layer's widths allow, else decode-kernel calls.
    fn experts(dv: *const Device, li: u32, row0: u32, rows: u32, prompt: bool) !void {
        const moe = dv.layer(li).ffn.moe;
        const l = moe.experts;
        const hidden = dv.g.hidden;
        const w = moe.prompt orelse cuda.exl3_gm.Widths{};
        if (prompt and moe.prompt != null and cuda.exl3_gm.eligible(l, w) and rows <= gmRows(dv.g)) {
            const sc = cuda.exl3_gm.scratch(dv.b.extra(.gm).ptr, l, gmRows(dv.g), dv.g.topk);
            const x = dv.at(.{ .buf = .a, .row = row0 });
            return dv.m.gm.run(dv.d, dv.st, l, w, x, hidden, dv.at(.{ .buf = .pick, .row = row0 }), dv.at(.{ .buf = .wts, .row = row0 }), rows, dv.at(.{ .buf = .routed, .row = row0 }), sc);
        }
        const sc = try cuda.exl3_experts.scratch(dv.b.extra(.experts).ptr, l, plan.expert_rows, dv.g.topk);
        var r: u32 = row0;
        while (r < row0 + rows) : (r += plan.expert_rows) {
            const n = @min(plan.expert_rows, row0 + rows - r);
            const x = dv.at(.{ .buf = .a, .row = r });
            try dv.m.experts.run(dv.st, l, x, hidden, .bf16, dv.at(.{ .buf = .pick, .row = r }), dv.at(.{ .buf = .wts, .row = r }), n, dv.at(.{ .buf = .routed, .row = r }), sc);
        }
    }

    fn qkRope(dv: *const Device, l: u32, n: u32) !void {
        const ly = dv.layer(l);
        const g = dv.g;
        const heads: rk.Heads = .{ .nq = g.nq, .niq = g.niq, .kvh = g.kvh, .index_key = true };
        const b = dv.b;
        const in: rk.Inputs = .{
            .q = b.ptr(.q),
            .q_ld = g.q_w,
            .k = b.ptr(.k),
            .k_ld = g.kv_w,
            .v = b.ptr(.v),
            .v_ld = g.kv_w,
            .iq = b.ptr(.iq_raw),
            .iq_ld = g.iq_w,
            .ik = b.ptr(.ik_raw),
            .ik_ld = g.ik_w,
        };
        const nm: rk.Norms = .{ .q = ly.q_norm, .k = ly.k_norm, .iq = ly.index_q_norm, .ik = ly.index_k_norm };
        const out: rk.Outputs = .{ .q = b.ptr(.qn), .q_ld = g.q_w, .iq = b.ptr(.iqn), .iq_ld = g.iq_w };
        const pg = try dv.paged(l);
        try dv.m.rows.qkRope(dv.st, heads, dv.opts, in, nm, out, dv.rowsOf(n), pg, @intCast(dv.pages.mirror.max_pages), b.extra(.status).ptr);
    }

    /// A block's partial sums added in rank order into `dst` on every rank, in tp.zig's gather or share form.
    fn sum(dv: *const Device, o: anytype) !void {
        const link = dv.link orelse return error.NoLink;
        const g = dv.g;
        const world: usize = g.world;
        const rank: usize = g.rank;
        const width: usize = g.hidden;
        const rows: usize = o.rows;
        if (o.form == .gather_sum) {
            const slot = rows * width * 4;
            try link.group.allGather(dv.st, dv.b.ptr(o.src), dv.b.extra(.gathered).ptr, slot);
            return dv.m.scatter.rankSum(dv.st, dv.b.extra(.gathered).ptr, dv.b.ptr(o.src), world, rank, rows * width, dv.b.ptr(o.dst));
        }
        const slot = shareRows(rows, world) * width * 4;
        try dv.m.scatter.run(link.lib, link.comm, world, rank, dv.st, dv.b.ptr(o.src), rows, width, dv.b.extra(.scatter).ptr, dv.b.extra(.share).ptr);
        try link.group.allGather(dv.st, dv.b.extra(.share).ptr, dv.b.extra(.shares).ptr, slot);
        for (0..world) |r| {
            const rs = cuda.rank_scatter.share(rows, world, r);
            const bytes = (rs[1] - rs[0]) * width * 4;
            if (bytes > 0) try dv.copy(dv.b.ptr(o.dst) + rs[0] * width * 4, dv.b.extra(.shares).ptr + r * slot, bytes);
        }
    }

    /// The tapped layer's output rows as bf16: a prompt chunk's into the chunk rows, a window's into its slot's rows.
    fn tap(dv: *const Device, slot: u32, rows: u32) !void {
        if (dv.taps.base == 0) return;
        const hidden = dv.g.hidden;
        const ld: usize = @as(usize, dv.g.n_taps) * hidden;
        const res = dv.b.ptr(.res);
        if (dv.taps.prompt) {
            try dv.m.dspark.tap(dv.st, res, dv.taps.base, ld, slot, hidden, rows);
            return;
        }
        for (dv.taps.segs[0..dv.taps.n]) |sg| {
            const dst = dv.taps.base + @as(u64, iface.tapWindowRow(sg.slot)) * ld * 2;
            try dv.m.dspark.tap(dv.st, res + @as(u64, sg.row0) * hidden * 4, dst, ld, slot, hidden, sg.rows);
        }
    }

    /// After the step: a launch that found no page for a row left bits in the status word.
    pub fn status(dv: *const Device) !i32 {
        var out: [4]u8 = undefined;
        try dv.b.extra(.status).download(0, &out);
        try dv.st.synchronize();
        return std.mem.readInt(i32, &out, .little);
    }
};
