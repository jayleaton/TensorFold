//! Flash Next CUDA engine from TensorFold qwen4_exp/cuda decode.py + engine.py (Ash Hart, vcruz305); ranks in step.
const std = @import("std");
const envs = @import("cuda_env.zig");
const cuda = @import("cuda");
const lanes = @import("lanes");
const media_mod = @import("cuda_media.zig");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const tops = @import("cuda_torch_ops.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const cfgs = @import("cuda_config.zig");
const comms = @import("cuda_comm.zig");
const rope = @import("cuda_rope.zig");
const sampler = @import("cuda_sampler.zig");
const decode = @import("cuda_decode.zig");
const fwd = @import("cuda_forward.zig");
const fp8 = @import("cuda_fp8.zig");
const int4 = @import("cuda_int4.zig");
const mtp = @import("cuda_mtp.zig");
const vmm = @import("cuda_vmm.zig");
const nucleus = @import("cuda_nucleus.zig");
const profs = @import("cuda_prof.zig");
const prompt_mm = @import("cuda_prompt.zig");
const moep = @import("cuda_moe_prompt.zig");
const eng_init = @import("cuda_engine_init.zig");
const eng_prompt = @import("cuda_engine_prompt.zig");
const eng_rounds = @import("cuda_engine_rounds.zig");
const eng_graphs = @import("cuda_engine_graphs.zig");

const Allocator = std.mem.Allocator;

/// MTP drafts a round by default (Python DEPTH) and their chain's confidence (CONFIDENCE).
pub const default_depth = 6;
pub const default_confidence = 0.70;

/// Served draft stop rule: running product >= 0.4 (encoded -0.4); TF_FLASHNEXT_CONFIDENCE overrides; drafts only.
pub const served_confidence = -0.4;

/// Served hybrid: running product while at most this many streams are live, per-draft wide_confidence above.
pub const product_streams_default = 2;
pub const wide_confidence = 0.5;

pub fn productStreams() usize {
    const v = envs.get("TF_FLASHNEXT_PRODUCT_STREAMS") orelse return product_streams_default;
    return std.fmt.parseInt(usize, v, 10) catch product_streams_default;
}

pub fn confidenceSetting() f64 {
    const v = envs.get("TF_FLASHNEXT_CONFIDENCE") orelse return served_confidence;
    const x = std.fmt.parseFloat(f64, v) catch return served_confidence;
    return if (x >= -1 and x <= 1) x else served_confidence;
}
/// The most candidates a main-head draw reads back (top_k + MARGIN); larger top_k values are refused.
pub const max_k = 256;

pub const Options = struct {
    context: usize = 262144,
    mtp: bool = true,
    /// CUDA graphs of the decode windows and MTP steps (Python graphs.py; eager gives the same bits)
    graphs: bool = true,
    rank: u32 = 0,
    world: u32 = 1,
    comm: ?*const comms.Comm = null,
    /// null below 262,144 (cuda_rope.zig, W9)
    yarn: ?rope.Yarn = null,
    /// MTP drafts a round at most: the decode buffers hold depth + 1 rows (at least 8, Python max(8, depth + 1))
    depth: usize = default_depth,
    /// streams a shared round may hold (verifyMany): the decode buffers hold streams x (depth + 1) rows
    streams: usize = 1,
    /// bytes the sequences' caches may take on this rank; null: what is free once loaded, less `reserve_gib`
    kv_budget: ?usize = null,
    /// memory kept free beside the caches in GiB; TENSORFOLD_MEMORY_RESERVE_GIB overrides it in the CLI and the server
    reserve_gib: f64 = 12,
    prefill_rows: usize = state.prefill_rows,
    /// caches in reserved address space grown in place, indexer keys in a ring (cuda_vmm.zig); false: copy growth
    vmm: bool = true,
    /// bytes kept free beside the reserve for the vision helper's workspace (rank 0 with --vision)
    vision_reserve: usize = 0,
    /// the attention caches' format (--kv-dtype): bf16 (exact, the default) or FP8 rows (cuda_state.KvDtype)
    kv: state.KvDtype = .bf16,
};

/// Rows the indexer-key ring holds at least: a prompt chunk's rows and the 3 before them that a pool reads.
pub const ring_min_rows = 4096;

pub const Seq = fwd.Seq;

/// A captured decode window or MTP step (Python Graphs.main/mtp keys); `media` separates image/video sequences' graphs.
pub const GraphKey = struct { seq: usize, version: u32, mtp: bool, rows: u32, parity: u32, ctx: usize, media: u64 = 0 };

/// A shared round's (or shared MTP step's) graph and the sequences whose buffers it bakes in.
const MultiGraph = struct { exec: cuda.graph.Exec, seqs: []usize };

/// Shared-round graphs kept at most; a composition is captured the second time it is seen.
pub const max_multi_graphs = 192;

/// The state of a class of captures after failures: rounds left to run eagerly, and the failures in a row.
pub const CaptureFault = struct { pause: u32 = 0, streak: u32 = 0, total: u64 = 0 };
/// TF_FLASHNEXT_MULTI_GRAPHS: most streams a shared step is captured for; wider steps rarely repeat and run eagerly.
pub const multi_graphs_default = 4;

/// Graphs._bucket: the attention bound a captured window replays at, a power of two from 8192 up to the capacity.
pub fn bucket(end: usize, capacity: usize) usize {
    const p = std.math.ceilPowerOfTwo(usize, @max(end, 1)) catch end;
    return @min(capacity, @max(8192, p));
}

pub fn seconds(io: std.Io, since: std.Io.Timestamp) f64 {
    const now = std.Io.Clock.awake.now(io);
    return @as(f64, @floatFromInt(now.toNanoseconds() - since.toNanoseconds())) / 1e9;
}

/// TF_FLASHNEXT_MTP_TOPK: the MTP layer's routed experts a row (drafts only; 0 or unset: the config's top_k).
fn mtpTopK(top_k: u32) usize {
    const v = envs.get("TF_FLASHNEXT_MTP_TOPK") orelse return 0;
    const n = std.fmt.parseInt(usize, v, 10) catch return 0;
    // the kernel set routes the checkpoints' own values (10, INT4-AutoRound's 5); 10 is the only other one accepted
    if (n != 10 or n == top_k) {
        if (n != top_k) std.log.warn("TF_FLASHNEXT_MTP_TOPK={d} ignored: the kernel set has top-5 and top-10 routing", .{n});
        return 0;
    }
    return n;
}

/// The shapes one rank computes with (cuda_state.Geometry and cuda_triton.Dims) from the config's and the split.
pub fn geometry(c: *const cfgs.Config, world: u32, has_mtp: bool) !state.Geometry {
    const sh = try c.shares(world);
    return .{
        .hidden = c.hidden,
        .streams = c.streams,
        .low = c.low,
        .heads = sh.heads,
        .kv_heads = sh.kv_heads,
        .head_dim = c.head_dim,
        .index_heads = c.index_heads,
        .index_dim = c.index_dim,
        .index_budget = c.index_budget,
        .index_ratio = c.index_ratio,
        .nk = sh.nk,
        .nv = sh.nv,
        .conv_kernel = c.conv_kernel,
        .experts = c.experts,
        .top_k = c.top_k,
        .moe_width = sh.moe_width,
        .ple_dim = c.ple_dim,
        .heads_per_ngram = c.heads_per_ngram,
        .ngram_size = c.ngram_size,
        .ple_kernel = c.ple_kernel,
        .head_n = c.vocab / world,
        .linear_layers = c.count(.linear),
        .attention_layers = c.count(.attention),
        .mtp = has_mtp,
        .world = world,
        .shared_width = sh.shared_width,
        .int4ar = c.int4ar(),
        .mtp_top_k = mtpTopK(c.top_k),
    };
}

pub fn dims(c: *const cfgs.Config, g: state.Geometry) tri.Dims {
    return .{
        .hidden = g.hidden,
        .streams = g.streams,
        .low = g.low,
        .heads = g.heads,
        .kv_heads = g.kv_heads,
        .head_dim = g.head_dim,
        .index_heads = g.index_heads,
        .index_dim = g.index_dim,
        .rotary_half = c.rope.rotary_dim / 2,
        .experts = g.experts,
        .top_k = g.top_k,
        .moe_width = g.moe_width,
        .vocab = g.head_n,
        .key_heads = g.nk,
        .value_heads = g.nv,
        .gdn_dim = c.dk,
        .ngram_heads = g.pleHeads(),
        .ngram_dim = g.pleHeadDim(),
        .ple_taps = g.ple_kernel,
        .ngram_size = g.ngram_size,
        .eps = @floatCast(c.eps),
    };
}

pub fn aligned(sizes: []const usize) usize {
    var t: usize = 0;
    for (sizes) |n| t += std.mem.alignForward(usize, n, 256);
    return t;
}

/// Two ranks' prompt chunk rows (TF_FLASHNEXT_PREFILL_ROWS overrides; any chunking gives the same bits); one GPU: 2048.
pub const tp2_prefill_rows: usize = 4096;

pub const Engine = struct {
    gpa: Allocator,
    io: std.Io,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    k: kern.Kernels,
    torch: tops.Ops,
    set: cuda.aot.Set,
    c: cfgs.Config,
    w: weights.Weights,
    g: state.Geometry,
    dims: tri.Dims,
    buf: fwd.Bufs,
    mbuf: ?fwd.Bufs = null,
    pbuf: fwd.Bufs,
    f: fwd.Forward,
    draws: ?mtp.Draws = null,
    /// decode D5: the MTP head's matrices in 4-bit (TF_FLASHNEXT_MTP_Q4=1), drafts only
    mtpq: ?@import("cuda_mtp_q4.zig").Table = null,
    /// host copy of the draft ids (this rank's), column -> token
    draft_host: []u32 = &.{},
    /// sampling scratch, the prompt's last logits and gathered candidates, the draft read-back
    extra: state.Arena,
    main_sample: tops.Torch.Sample,
    tp: ?mtp.TpScratch = null,
    /// top_k-off draws' device scratch (one row of masses)
    nsc: nucleus.Scratch = undefined,
    last_logits: u64,
    last_cand: u64,
    /// two ranks: the last shared round's gathered candidates on the host, read once for every window's draws
    cand_host: std.ArrayList(f32) = .empty,
    cand_round: u64 = 0,
    cand_have: u64 = std.math.maxInt(u64),
    /// TF_FLASHNEXT_DRAW_ONCE=0: each window's and draft's draws read the candidates again (the old way)
    draw_once: bool = true,
    /// generateMany's served hybrid: running product while at most this many streams are live, `wide_confidence` above
    product_streams: usize = std.math.maxInt(usize),
    /// shared steps of at most this many streams are captured (TF_FLASHNEXT_MULTI_GRAPHS)
    multi_graphs: usize = multi_graphs_default,
    /// the Triton variants each launch site picked (the forward's launches skip `find`'s scan)
    memo: tri.Memo = undefined,
    /// the rows the context caches hold (Python max_len: the window plus depth + 1 speculative rows)
    max_len: usize,
    depth: usize,
    rank: u32,
    world: u32,
    prefill_rows: usize,
    /// a last chunk takes up to this many extra rows, not a chunk of its own (same bits); TF_FLASHNEXT_PREFILL_TAIL
    prefill_tail: usize = 0,
    sampling: ?lanes.Sampling = null,
    /// prompt chunks absorb into the MTP head (Python prefill's `mtp`; the serial reference prefills without)
    absorbing: bool = true,
    own: *Seq = undefined,
    bound: *Seq = undefined,
    rows: usize = 0,
    load_seconds: f64 = 0,
    graphs_on: bool = true,
    /// image and video sequences through graphs too (keyed by their attachment); TF_FLASHNEXT_MEDIA_GRAPHS=0: eager
    media_graphs: bool = true,
    /// the last shared round's streams and their rows (verifyMany)
    round_segs: []fwd.Seg = &.{},
    round_n: usize = 0,
    times: Times = .{},
    budget: fwd.Budget = .{ .limit = std.math.maxInt(usize) },
    vm: ?vmm.Vmm = null,
    /// the fill whose chunk holds the prompt buffers between slices (one at a time; prefill waits for it)
    pfill: ?*const Fill = null,
    graphs: std.AutoHashMapUnmanaged(GraphKey, cuda.graph.Exec) = .empty,
    captures: usize = 0,
    /// a failed capture runs that round eagerly (same bits) and pauses captures; TF_FLASHNEXT_GRAPH_LOG/_FAIL test
    solo_fault: CaptureFault = .{},
    multi_fault: CaptureFault = .{},
    instantiated: u64 = 0,
    graph_log: bool = false,
    inject_every: u64 = 0,
    /// shared rounds' and shared MTP steps' graphs by composition (multiKey), and compositions seen once
    multi: std.AutoHashMapUnmanaged(u64, MultiGraph) = .empty,
    seen: std.AutoHashMapUnmanaged(u64, void) = .empty,
    /// TF_FLASHNEXT_PROFILE: each prompt chunk's GPU time by part (cuda_prof.zig)
    prof: ?*profs.Prof = null,
    /// TF_FLASHNEXT_PROMPT_EXPERTS (default on): prompt chunks' experts read weights once a call
    px: ?moep.Prompt4 = null,
    /// TF_FLASHNEXT_PREFILL_DIGEST: each prompt chunk's final streams (every row) and last logits hashed and printed
    digest: bool = false,
    /// INT4-AutoRound: the block-FP8 lane matmul (cuda_fp8.zig) and the GPTQ int4 kernels (cuda_int4.zig)
    k8: ?fp8.Kernels = null,
    k4: ?int4.Kernels = null,
    /// with `digest`: each chunk's (streams, logits) sha256 prefixes kept here too (the CLI's media gate)
    digests: ?*std.ArrayList([2][16]u8) = null,
    /// TF_FLASHNEXT_POISON_PADS: prompt-buffer rows past a chunk's end get rank-dependent bytes (none may be read)
    poison: bool = false,
    /// TF_FLASHNEXT_PASS_TIMES=1: prefillWith logs its parts' wall times (the stream drained at each edge)
    pass_times: bool = false,
    /// TF_FLASHNEXT_QSA_FAST (default on): fn_qsa_scores for the prompt indexer's scores
    qsa_fast: ?prompt_mm.QsaScores = null,

    pub const init = eng_init.init;
    pub const initWindows = eng_init.initWindows;

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        var it = e.graphs.valueIterator();
        while (it.next()) |g| g.deinit();
        e.graphs.deinit(e.gpa);
        e.graphs = .empty;
        e.dropMulti(null);
        e.multi.deinit(e.gpa);
        e.multi = .empty;
        e.seen.deinit(e.gpa);
        e.seen = .empty;
        e.freeSeq(e.own);
        e.gpa.free(e.draft_host);
        e.gpa.free(e.round_segs);
        e.extra.buf.free();
        if (e.prof) |p| p.deinit(e.gpa);
        e.cand_host.deinit(e.gpa);
        e.memo.deinit();
        if (e.px) |*q| q.deinit();
        if (e.k8) |*q| q.deinit();
        if (e.k4) |*q| q.deinit();
        if (e.mtpq) |*t| t.deinit();
        if (e.qsa_fast) |*q| q.deinit();
        if (e.f.cs) |*cs| {
            cs.synchronize() catch {};
            cs.deinit();
            for (&e.f.cs_ev) |*ev| ev.deinit();
        }
        e.f.deinit();
        e.pbuf.deinit();
        if (e.mbuf) |*m| m.deinit();
        e.buf.deinit();
        e.w.deinit();
        e.set.deinit();
        e.torch.deinit();
        e.k.deinit();
        e.stream.deinit();
        if (e.vm) |*v| v.deinit();
        e.c.deinit();
        e.gpa.destroy(e);
    }

    pub const newSeq = eng_init.newSeq;
    pub const mapSeq = eng_init.mapSeq;
    pub const freeSeq = eng_init.freeSeq;
    pub const ringBytes = eng_init.ringBytes;
    pub const seqBytes = eng_init.seqBytes;
    pub const fits = eng_init.fits;

    pub fn bind(e: *Engine, s: *Seq) void {
        e.bound = s;
    }

    /// The sampling rule `sample`, `sampleDraft` and `drawDraft` apply (null: greedy).
    pub fn setSampling(e: *Engine, s: ?lanes.Sampling) void {
        e.sampling = if (s) |x| (if (x.temperature > 0) x else null) else null;
    }

    pub fn ops(e: *Engine) *fwd.Forward {
        return &e.f;
    }

    pub const prefill = eng_prompt.prefill;
    pub const Snapshot = eng_prompt.Snapshot;
    pub const PrefillOptions = eng_prompt.PrefillOptions;
    pub const Prefilled = eng_prompt.Prefilled;
    pub const snapshotBytes = eng_prompt.snapshotBytes;
    pub const ikcTail = eng_prompt.ikcTail;
    pub const snapshot = eng_prompt.snapshot;
    pub const freeSnapshot = eng_prompt.freeSnapshot;
    pub const restore = eng_prompt.restore;
    pub const copyPrefix = eng_prompt.copyPrefix;
    pub const prefillWith = eng_prompt.prefillWith;
    pub const chunkEnd = eng_prompt.chunkEnd;
    pub const passLap = eng_prompt.passLap;
    pub const ManyPrompt = eng_prompt.ManyPrompt;
    pub const manyFit = eng_prompt.manyFit;
    pub const prefillMany = eng_prompt.prefillMany;
    pub const Fill = eng_prompt.Fill;
    pub const fillBegin = eng_prompt.fillBegin;
    pub const fillFree = eng_prompt.fillFree;
    pub const fillStep = eng_prompt.fillStep;
    pub const poisonPads = eng_prompt.poisonPads;
    pub const chunkDigest = eng_prompt.chunkDigest;
    pub const lastStreams = eng_prompt.lastStreams;

    pub const read = eng_rounds.read;
    pub const draw = eng_rounds.draw;
    pub const chooseFrom = eng_rounds.chooseFrom;
    pub const drawRows = eng_rounds.drawRows;
    pub const nucleusRows = eng_rounds.nucleusRows;
    pub const Window = eng_rounds.Window;
    pub const verifyMany = eng_rounds.verifyMany;
    pub const sampleWindow = eng_rounds.sampleWindow;
    pub const commitWindow = eng_rounds.commitWindow;
    pub const DraftSource = eng_rounds.DraftSource;
    pub const DraftReq = eng_rounds.DraftReq;
    pub const draftMany = eng_rounds.draftMany;
    pub const Request = eng_rounds.Request;
    pub const Times = eng_rounds.Times;
    pub const ManyOptions = eng_rounds.ManyOptions;
    pub const hybrid = eng_rounds.hybrid;
    pub const generateMany = eng_rounds.generateMany;
    pub const generateManyWith = eng_rounds.generateManyWith;

    pub const dropGraphs = eng_graphs.dropGraphs;
    pub const dropGraphsOf = eng_graphs.dropGraphsOf;
    pub const graphable = eng_graphs.graphable;
    pub const parity = eng_graphs.parity;
    pub const body = eng_graphs.body;
    pub const graphed = eng_graphs.graphed;
    pub const capBegin = eng_graphs.capBegin;
    pub const capEnd = eng_graphs.capEnd;
    pub const capNote = eng_graphs.capNote;
    pub const capLog = eng_graphs.capLog;
    pub const warmGraphs = eng_graphs.warmGraphs;
    pub const dropMulti = eng_graphs.dropMulti;
    pub const multiKey = eng_graphs.multiKey;
    pub const runMulti = eng_graphs.runMulti;
    pub const mtpForward = eng_graphs.mtpForward;

    // -- cuda_decode's E interface on the bound sequence ------------------------------------------------------

    pub fn pos(e: *Engine) u64 {
        return e.bound.st.pos;
    }

    /// A window's rows at pos .. pos + tokens.len - 1 (main head logits of every row).
    pub fn forward(e: *Engine, tokens: []const u32) !void {
        const R = tokens.len;
        try e.f.stage(e.bound, &e.buf, tokens);
        const par = e.parity();
        if (e.graphable(R) and par != null) {
            try e.graphed(false, R, bucket(e.bound.st.pos + R, e.bound.st.capacity), par.?);
        } else _ = try e.f.compute(e.bound, &e.buf, R, true);
        e.rows = R;
    }

    /// The target's draws of the window's first `rows` rows, keyed at first_position + r.
    pub fn sample(e: *Engine, rows: usize, first_position: u64, out: []u32) !void {
        try e.draw(e.buf.b.logits, e.buf.b.cand_all, rows, first_position, out);
    }

    pub fn commit(e: *Engine, rows: usize, keep: usize) !void {
        try e.f.commit(e.bound, &e.buf, rows, keep);
    }

    /// decode.absorb: the MTP cache drops the last chain's drafts and takes the kept rows' streams and next tokens.
    pub fn absorb(e: *Engine, source: decode.Source, next_tokens: []const u32) !void {
        const mb = &(e.mbuf orelse return error.NoMtpHead);
        const st = &e.bound.st;
        if (st.mtp_drafted != 0) {
            try e.f.setMtpLen(st, st.mtp_len - st.mtp_drafted);
            st.mtp_drafted = 0;
        }
        const streams = switch (source) {
            .prefill_last => e.lastStreams(),
            .window => e.buf.b.streams,
        };
        try e.mtpForward(mb, next_tokens, streams);
        try e.f.setMtpLen(st, st.mtp_len + next_tokens.len);
    }

    /// One more MTP step on draft `token`, from the head's output row `from_row` (decode.draft's chain).
    pub fn chain(e: *Engine, token: u32, from_row: usize) !void {
        const mb = &(e.mbuf orelse return error.NoMtpHead);
        const st = &e.bound.st;
        const prev = mb.b.streams + from_row * e.g.wide() * 2;
        try e.mtpForward(mb, &.{token}, prev);
        try e.f.setMtpLen(st, st.mtp_len + 1);
        st.mtp_drafted += 1;
    }

    pub fn sampleDraft(e: *Engine, position: u64) !decode.Draw {
        const d = &(e.draws orelse return error.NoMtpHead);
        const got = try d.sampleDraft(&e.f, &e.mbuf.?, e.mbuf.?.b.logits, position, e.sampling);
        return .{ .token = @intCast(got.token), .prob = got.prob };
    }

    pub fn drawDraft(e: *Engine, position: u64) !u32 {
        const d = &(e.draws orelse return error.NoMtpHead);
        return @intCast(try d.drawDraft(&e.f, &e.mbuf.?, e.mbuf.?.b.logits, position, e.sampling));
    }

    pub fn isEos(e: *Engine, token: u32) bool {
        return e.c.isEos(token);
    }

    // -- the oracle checks' entry points ----------------------------------------------------------------------

    /// The teacher's MTP step: mtp_forward([next], buf.streams[:1]) at mtp_len, then mtp_len + 1; the draft head's row.
    pub fn teacherMtp(e: *Engine, next: u32) !u64 {
        const mb = &(e.mbuf orelse return error.NoMtpHead);
        const st = &e.bound.st;
        try e.mtpForward(mb, &.{next}, e.buf.b.streams);
        try e.f.setMtpLen(st, st.mtp_len + 1);
        return mb.b.logits;
    }

    pub fn reset(e: *Engine) !void {
        try e.f.reset(e.bound);
    }
};

test "graph buckets follow Graphs._bucket" {
    try std.testing.expectEqual(@as(usize, 8192), bucket(1, 262151));
    try std.testing.expectEqual(@as(usize, 8192), bucket(8192, 262151));
    try std.testing.expectEqual(@as(usize, 16384), bucket(8193, 262151));
    try std.testing.expectEqual(@as(usize, 262144), bucket(200000, 262151));
    try std.testing.expectEqual(@as(usize, 262151), bucket(262150, 262151));
    try std.testing.expectEqual(@as(usize, 4096), bucket(10, 4096));
}

test "geometry and dims follow the config's split" {
    // the shapes the Triton wrappers' fixtures were recorded with
    const d = tri.Dims{};
    try std.testing.expectEqual(@as(usize, 13952), d.attnWidth());
    try std.testing.expectEqual(@as(usize, 16480), d.projWidth());
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Engine);
}
