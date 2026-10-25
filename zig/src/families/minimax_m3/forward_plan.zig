//! One MiniMax-M3.1-Flash forward step as data: a pure planner (no GPU) the executor and the tests share.
const std = @import("std");
const Config = @import("config.zig").Config;
const shard = @import("shard.zig");
const tp = @import("tp.zig");
const iface = @import("iface.zig");

pub const Error = error{ BadRows, TooManyRows, BadThin };

/// Routed-expert rows one launch takes: its scratch is sized for them, a longer chunk runs in several calls.
pub const expert_rows: u32 = 256;
/// Most rows whose logits the head keeps at once (a decode or verify window; a prompt chunk keeps one row).
pub const head_rows: u32 = 128;

/// Device buffers a step's ops name; the executor maps each to an allocation (rows are the first index).
pub const Buf = enum {
    ids, // int32 token ids
    pos, // int32 positions (a shared window's rows)
    row_tab, // int32 table offsets (a shared window's rows)
    emb, // bf16 embedding rows
    res, // fp32 residual stream (decision R1: fp32, the T0 default has no rounding)
    a, // bf16 normed input of the current block
    q,
    k,
    v, // bf16 projections
    iq_raw,
    ik_raw, // fp16 index projections
    qn,
    iqn, // bf16 normed and rotated q and index q
    attn, // bf16 attention output
    part, // fp32 this rank's partial sum of a block
    routed, // fp32 routed experts' output
    shr, // fp32 shared expert's output
    sum, // fp32 the ranks' partials added in rank order
    gate,
    up,
    act, // bf16 dense and shared MLP rows
    logits_r, // fp32 router logits
    pick, // int32 routed picks
    wts, // fp32 routed weights
    hlog, // fp32 this rank's head logits
    hgather, // fp32 every rank's head logits, [rank][row][slot]
    pairs, // (logit, id) of this rank's best
    pairs_all, // every rank's pair
};

pub const bufs = std.enums.values(Buf);

/// A buffer and the first row an op reads or writes in it.
pub const Ref = struct { buf: Buf, row: u32 = 0 };

pub const Norm = enum { input, post, final };

/// The matrices a `linear` op names (layer picks the record; head and the layer's dense or shared MLP are by kind).
pub const Lin = enum { q, k, v, o, dense_gate, dense_up, dense_down, shared_gate, shared_up, shared_down, head };

pub const Form = tp.Form;

pub const Op = union(enum) {
    stage: struct { rows: u32, shared: bool }, // ids, positions and row tables go to the device; status cleared
    embed: struct { rows: u32 },
    zero: struct { buf: Buf, rows: u32 },
    norm_add: struct { x: Ref, x_f32: bool, norm: Norm, layer: u32, y: Ref, rows: u32 }, // res += x; y = rms(res) w
    linear: struct { w: Lin, layer: u32, x: Ref, y: Ref, rows: u32, prompt: bool },
    index_proj: struct { layer: u32, rows: u32 }, // fp16 weights: a -> iq_raw, ik_raw
    qk_rope: struct { layer: u32, rows: u32 }, // norm + rope; q, index q to scratch; k, v, index key into pages
    score: struct { layer: u32, rows: u32, max_pos: u32 },
    select: struct { layer: u32, rows: u32 },
    attend: struct { layer: u32, rows: u32 },
    router: struct { layer: u32, rows: u32 }, // fp32 gate times a
    route: struct { layer: u32, rows: u32 },
    experts: struct { layer: u32, row0: u32, rows: u32 },
    swiglu: struct { layer: u32, shared: bool, n: u32, rows: u32 }, // act = swigluoai(gate, up)
    add2: struct { a: Buf, b: Buf, out: Buf, rows: u32 }, // out = a + b in fp32 (routed + shared)
    exchange: struct { form: Form, site: tp.Site, src: Buf, dst: Buf, rows: u32 },
    tap: struct { slot: u32, rows: u32 }, // bf16 copy of res (the layer's output) into column slot `slot` of the taps buffer
    logits_gather: struct { rows: u32 },
    local_pick: struct { rows: u32 },
    pick_gather: struct { rows: u32 },
};

/// The kernels' classes, one per launch of the T0 model's order (see forward_test.zig).
pub const Stage = enum {
    stage,
    embed,
    zero,
    norm_in,
    proj_q,
    proj_k,
    proj_v,
    index_proj,
    qk_rope,
    score,
    select,
    attend,
    proj_o,
    exchange_attn,
    norm_post,
    dense_gate,
    dense_up,
    dense_act,
    dense_down,
    router,
    route,
    experts,
    shared_gate,
    shared_up,
    shared_act,
    shared_down,
    combine,
    exchange_ffn,
    tap,
    norm_final,
    head,
    gather,
    pick,
};

pub fn stageOf(op: Op) Stage {
    return switch (op) {
        .stage => .stage,
        .embed => .embed,
        .zero => .zero,
        .norm_add => |o| switch (o.norm) {
            .input => .norm_in,
            .post => .norm_post,
            .final => .norm_final,
        },
        .linear => |o| switch (o.w) {
            .q => .proj_q,
            .k => .proj_k,
            .v => .proj_v,
            .o => .proj_o,
            .dense_gate => .dense_gate,
            .dense_up => .dense_up,
            .dense_down => .dense_down,
            .shared_gate => .shared_gate,
            .shared_up => .shared_up,
            .shared_down => .shared_down,
            .head => .head,
        },
        .index_proj => .index_proj,
        .qk_rope => .qk_rope,
        .score => .score,
        .select => .select,
        .attend => .attend,
        .router => .router,
        .route => .route,
        .experts => .experts,
        .swiglu => |o| if (o.shared) .shared_act else .dense_act,
        .add2 => .combine,
        .exchange => |o| if (o.site == .attn_out) .exchange_attn else .exchange_ffn,
        .tap => .tap,
        .logits_gather, .pick_gather => .gather,
        .local_pick => .pick,
    };
}

/// Most layers the dspark draft may tap.
pub const max_taps = 8;

/// The rank's widths: what each buffer holds per row and which matrices exist.
pub const Geometry = struct {
    cfg: *const Config,
    hidden: u32,
    layers: u32, // layers run (a thin model: the first ones)
    world: u32,
    rank: u32,
    max_rows: u32,
    nq: u32, // this rank's query heads, index heads, kv heads
    niq: u32,
    kvh: u32,
    q_w: u32,
    kv_w: u32,
    iq_w: u32,
    ik_w: u32,
    dense_w: u32,
    shared_w: u32,
    moe_w: u32,
    experts: u32,
    topk: u32,
    slot: u32, // logits a rank's gather slot holds (the largest share)
    own_vocab: u32, // this rank's head rows
    vocab_lo: u32, // first vocabulary id of this rank's rows
    tap_ids: [max_taps]u32 = @splat(0), // the dspark draft's layers (ascending) whose output rows are copied; none: no taps
    n_taps: u32 = 0,

    pub fn of(c: *const Config, s: shard.Share, layers: u32, max_rows: u32) !Geometry {
        if (layers == 0 or layers > c.layers) return error.BadThin;
        const lg = try tp.LogitsGather.of(c, s.world);
        return .{
            .cfg = c,
            .hidden = c.hidden,
            .layers = layers,
            .world = @intCast(s.world),
            .rank = @intCast(s.rank),
            .max_rows = max_rows,
            .nq = @intCast(s.q_heads.len()),
            .niq = @intCast(s.index_heads.len()),
            .kvh = @intCast(s.kv_heads.len()),
            .q_w = @intCast(s.q_heads.len() * c.head_dim),
            .kv_w = @intCast(s.kv_heads.len() * c.head_dim),
            .iq_w = @intCast(s.index_heads.len() * c.index.dim),
            .ik_w = c.index.dim,
            .dense_w = @intCast(s.dense.len()),
            .shared_w = @intCast(s.shared.len()),
            .moe_w = @intCast(s.moe.len()),
            .experts = c.experts,
            .topk = c.topk,
            .slot = @intCast(lg.slot),
            .own_vocab = @intCast(s.vocab.len()),
            .vocab_lo = @intCast(s.vocab.lo),
        };
    }

    /// Which tap slot layer `layer` fills, if the draft reads it.
    pub fn tapSlot(g: *const Geometry, layer: u32) ?u32 {
        for (g.tap_ids[0..g.n_taps], 0..) |id, t| if (id == layer) return @intCast(t);
        return null;
    }

    pub fn isMoe(g: *const Geometry, layer: u32) bool {
        return g.cfg.isMoe(layer);
    }

    /// Rows a buffer holds at most: the ids and activations by `max_rows`, the head's by `head_rows`.
    pub fn capacity(g: *const Geometry, b: Buf) u32 {
        return switch (b) {
            .hlog, .hgather, .pairs, .pairs_all => head_rows,
            else => g.max_rows,
        };
    }

    /// Bytes one row of `b` takes.
    pub fn rowBytes(g: *const Geometry, b: Buf) usize {
        const rows_w: usize = @max(g.dense_w, g.shared_w);
        return switch (b) {
            .ids, .pos, .row_tab => 4,
            .emb, .a => g.hidden * 2,
            .res, .part, .routed, .shr, .sum => g.hidden * 4,
            .q, .qn, .attn => g.q_w * 2,
            .k, .v => g.kv_w * 2,
            .iq_raw, .iqn => g.iq_w * 2,
            .ik_raw => g.ik_w * 2,
            .gate, .up, .act => rows_w * 2,
            .logits_r => g.experts * 4,
            .pick, .wts => g.topk * 4,
            .hlog => g.slot * 4,
            .hgather => g.world * g.slot * 4,
            .pairs => 8,
            .pairs_all => g.world * 8,
        };
    }
};

/// What one call computes: its rows, how its streams sit in the batch and which rows the head reads.
pub const Head = enum { all, last, none };
pub const HeadMode = enum { gather, picks };

pub const Shape = struct {
    rows: u32,
    prompt: bool = false, // a prompt chunk of one stream
    shared: bool = false, // several streams' windows in one batch (positions and table offsets per row)
    max_pos: u32 = 0, // the largest position among the rows
    head: Head = .all,
    mode: HeadMode = .gather,
};

pub fn check(g: *const Geometry, sh: Shape) Error!void {
    if (sh.rows == 0) return error.BadRows;
    if (sh.rows > g.max_rows) return error.TooManyRows;
    if (sh.shared and (sh.prompt or sh.rows > iface.shared_rows)) return error.TooManyRows;
    if (sh.head == .all and sh.rows > head_rows) return error.TooManyRows;
}

/// The shipped model's launches for `sh`, in order, into `sink.emit(op)`.
pub fn plan(g: *const Geometry, sh: Shape, sink: anytype) !void {
    try check(g, sh);
    const n = sh.rows;
    try sink.emit(.{ .stage = .{ .rows = n, .shared = sh.shared } });
    try sink.emit(.{ .embed = .{ .rows = n } });
    try sink.emit(.{ .zero = .{ .buf = .res, .rows = n } });
    try sink.emit(.{ .norm_add = .{ .x = .{ .buf = .emb }, .x_f32 = false, .norm = .input, .layer = 0, .y = .{ .buf = .a }, .rows = n } });
    var l: u32 = 0;
    while (l < g.layers) : (l += 1) {
        try attention(g, sh, l, sink);
        try mlp(g, sh, l, sink);
    }
    try head(g, sh, sink);
}

fn lin(sink: anytype, w: Lin, l: u32, x: Ref, y: Ref, n: u32, prompt: bool) !void {
    try sink.emit(.{ .linear = .{ .w = w, .layer = l, .x = x, .y = y, .rows = n, .prompt = prompt } });
}

/// The ranks' partial sums of a block: none on one rank, else the exchange; returns the buffer holding the sum.
fn exchange(g: *const Geometry, sh: Shape, site: tp.Site, sink: anytype) !Buf {
    if (g.world == 1) return .part;
    try sink.emit(.{ .exchange = .{ .form = tp.formFor(sh.rows), .site = site, .src = .part, .dst = .sum, .rows = sh.rows } });
    return .sum;
}

/// The Gemma norm after the sum: the next norm of the stream, the layer's own post norm or the final one.
fn normAdd(sink: anytype, x: Buf, norm: Norm, layer: u32, n: u32) !void {
    try sink.emit(.{ .norm_add = .{ .x = .{ .buf = x }, .x_f32 = true, .norm = norm, .layer = layer, .y = .{ .buf = .a }, .rows = n } });
}

fn attention(g: *const Geometry, sh: Shape, l: u32, sink: anytype) !void {
    const n = sh.rows;
    const a: Ref = .{ .buf = .a };
    try lin(sink, .q, l, a, .{ .buf = .q }, n, sh.prompt);
    try lin(sink, .k, l, a, .{ .buf = .k }, n, sh.prompt);
    try lin(sink, .v, l, a, .{ .buf = .v }, n, sh.prompt);
    try sink.emit(.{ .index_proj = .{ .layer = l, .rows = n } });
    try sink.emit(.{ .qk_rope = .{ .layer = l, .rows = n } });
    try sink.emit(.{ .score = .{ .layer = l, .rows = n, .max_pos = sh.max_pos } });
    try sink.emit(.{ .select = .{ .layer = l, .rows = n } });
    try sink.emit(.{ .attend = .{ .layer = l, .rows = n } });
    try lin(sink, .o, l, .{ .buf = .attn }, .{ .buf = .part }, n, sh.prompt);
    try normAdd(sink, try exchange(g, sh, .attn_out, sink), .post, l, n);
}

fn swiglu(sink: anytype, l: u32, n: u32, width: u32, shared: bool) !void {
    try sink.emit(.{ .swiglu = .{ .layer = l, .shared = shared, .n = width, .rows = n } });
}

fn mlp(g: *const Geometry, sh: Shape, l: u32, sink: anytype) !void {
    const n = sh.rows;
    const a: Ref = .{ .buf = .a };
    if (!g.isMoe(l)) {
        try lin(sink, .dense_gate, l, a, .{ .buf = .gate }, n, sh.prompt);
        try lin(sink, .dense_up, l, a, .{ .buf = .up }, n, sh.prompt);
        try swiglu(sink, l, n, g.dense_w, false);
        try lin(sink, .dense_down, l, .{ .buf = .act }, .{ .buf = .part }, n, sh.prompt);
    } else {
        try sink.emit(.{ .router = .{ .layer = l, .rows = n } });
        try sink.emit(.{ .route = .{ .layer = l, .rows = n } });
        var at: u32 = 0;
        while (at < n) : (at += expert_rows) {
            try sink.emit(.{ .experts = .{ .layer = l, .row0 = at, .rows = @min(expert_rows, n - at) } });
        }
        try lin(sink, .shared_gate, l, a, .{ .buf = .gate }, n, sh.prompt);
        try lin(sink, .shared_up, l, a, .{ .buf = .up }, n, sh.prompt);
        try swiglu(sink, l, n, g.shared_w, true);
        try lin(sink, .shared_down, l, .{ .buf = .act }, .{ .buf = .shr }, n, sh.prompt);
        try sink.emit(.{ .add2 = .{ .a = .routed, .b = .shr, .out = .part, .rows = n } });
    }
    const next: Norm = if (l + 1 < g.layers) .input else .final;
    try normAdd(sink, try exchange(g, sh, .ffn_out, sink), next, if (next == .input) l + 1 else l, n);
    // the layer's output is in `res` now: the draft's tap
    if (g.tapSlot(l)) |t| try sink.emit(.{ .tap = .{ .slot = t, .rows = n } });
}

/// The final norm ran at the end of the last layer; the head cuts this rank's rows and gathers or merges them.
fn head(g: *const Geometry, sh: Shape, sink: anytype) !void {
    if (sh.head == .none) return;
    const n: u32 = if (sh.head == .all) sh.rows else 1;
    const x: Ref = .{ .buf = .a, .row = if (sh.head == .all) 0 else sh.rows - 1 };
    try lin(sink, .head, 0, x, .{ .buf = .hlog }, n, false);
    switch (sh.mode) {
        .gather => if (g.world > 1) try sink.emit(.{ .logits_gather = .{ .rows = n } }),
        .picks => {
            try sink.emit(.{ .local_pick = .{ .rows = n } });
            if (g.world > 1) try sink.emit(.{ .pick_gather = .{ .rows = n } });
        },
    }
}

/// An op list for tests: owns the ops it was given.
pub const Recorder = struct {
    gpa: std.mem.Allocator,
    ops: std.ArrayList(Op) = .empty,

    pub fn emit(r: *Recorder, op: Op) !void {
        try r.ops.append(r.gpa, op);
    }

    pub fn deinit(r: *Recorder) void {
        r.ops.deinit(r.gpa);
    }
};
