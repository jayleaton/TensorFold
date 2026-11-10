//! Flash Next CUDA forward, Python forward.py/mtp.py by Ash Hart (ashhart) and vcruz305 (TensorFold), same launches.
const std = @import("std");
const envs = @import("cuda_env.zig");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const tops = @import("cuda_torch_ops.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const cfgs = @import("cuda_config.zig");
const comms = @import("cuda_comm.zig");
const vmm = @import("cuda_vmm.zig");
const profs = @import("cuda_prof.zig");
const prompt = @import("cuda_prompt.zig");
const moep = @import("cuda_moe_prompt.zig");
const fp8 = @import("cuda_fp8.zig");
const int4 = @import("cuda_int4.zig");
const fw_round = @import("cuda_forward_round.zig");
const fw_ops = @import("cuda_forward_ops.zig");
const fw_dump = @import("cuda_forward_dump.zig");
const fw_stage = @import("cuda_forward_stage.zig");
const fw_layer = @import("cuda_forward_layer.zig");
const fw_hc = @import("cuda_forward_hc.zig");
const fw_attn = @import("cuda_forward_attn.zig");
const fw_moe = @import("cuda_forward_moe.zig");

const Allocator = std.mem.Allocator;
const W = weights;

/// A pending write-back: the branch `_hc_writeback` adds next and its inject gates (Python's (mode, ...) tuple).
pub const Pending = struct { branch: tri.Branch, inject: u64 };

/// One window's buffers plus Python Buffers' extras: the experts' slot dtype and the attention scratch's geometry.
pub const Bufs = struct {
    b: state.Buffers,
    y_f32: bool,
    attn: tri.AttnGeometry,

    pub fn deinit(x: *Bufs) void {
        x.b.deinit();
    }
};

/// A sequence's images and video frames (Python image_rows.attach): device rotary table [len, 3] i32 and features.
pub const SeqMedia = struct {
    table: cuda.DeviceBuffer,
    len: usize,
    feats: ?cuda.DeviceBuffer = null,
    rows: []u32 = &.{},
    width: usize,
    /// bytes of the engine's budget the table and features hold
    charged: usize = 0,
    /// this attachment's number (the engine's count of attachments): graphs that bake in the table key on it
    epoch: u64 = 0,

    /// The features and their rows released (Python image_rows.finish); the table stays.
    pub fn finish(m: *SeqMedia, gpa: Allocator) usize {
        var freed: usize = 0;
        if (m.feats) |*b| {
            freed = b.len;
            b.free();
        }
        m.feats = null;
        gpa.free(m.rows);
        m.rows = &.{};
        return freed;
    }

    pub fn deinit(m: *SeqMedia, gpa: Allocator) void {
        _ = m.finish(gpa);
        m.table.free();
    }
};

/// One sequence: its committed caches (cuda_state.State) and the window `stage` staged for `commit`.
pub const Seq = struct {
    st: state.State,
    /// the staged window's tokens (the n-gram history takes the kept ones at commit)
    window: std.ArrayList(i64) = .empty,
    staged: bool = false,
    /// the prompt's last row's final streams [1, S*D] bf16 (decode.Engine.last_streams), the first absorb's input
    last_streams: ?cuda.DeviceBuffer = null,
    /// bytes this sequence holds of the engine's Budget
    charged: usize = 0,
    /// its caches in reserved address space (cuda_vmm.zig); empty: State's own allocation (grown by copying)
    regions: std.ArrayList(vmm.Region) = .empty,
    /// rows the indexer-key ring holds (0: no ring)
    ring_rows: usize = 0,
    /// last round's kept rows not yet folded into DeltaNet states: [held_a0, held_a0 + held) of parity held_par
    held: usize = 0,
    held_a0: usize = 0,
    held_par: u1 = 0,
    /// the prompt's images and video frames (null: text)
    media: ?SeqMedia = null,

    /// The sequence's rotary table, when its prompt has images or video.
    pub fn rope(s: *const Seq) ?tri.Tri.Rope {
        const m = &(s.media orelse return null);
        return .{ .table = m.table.ptr, .delta = s.st.rope_delta_dev, .length = m.len };
    }

    pub fn deinit(s: *Seq, gpa: Allocator) void {
        if (s.media) |*m| m.deinit(gpa);
        s.media = null;
        for (s.regions.items) |*r| r.deinit(gpa);
        s.regions.deinit(gpa);
        if (s.last_streams) |*b| b.free();
        s.window.deinit(gpa);
        s.st.deinit();
    }
};

/// The rank's cache memory (Python MemoryGate): fixed state plus caches grown in steps; past budget, error.NoRoom.
pub const Budget = struct {
    limit: usize,
    used: usize = 0,
    /// MemAvailable a growth step must leave, checked against the kernel's count at each step; 0: unchecked
    floor: usize = 0,

    /// Whether `bytes` more leave MemAvailable at or above the floor (true when it cannot be read).
    pub fn leaves(b: *const Budget, bytes: usize) bool {
        if (b.floor == 0) return true;
        const avail = memAvailable() orelse return true;
        return avail >= b.floor + bytes;
    }

    pub fn room(b: *const Budget) usize {
        return b.limit -| b.used;
    }
};

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn close(fd: c_int) c_int;
const c_open = open;
const c_read = read;
const c_close = close;

/// /proc/meminfo's MemAvailable in bytes, or null.
pub fn memAvailable() ?usize {
    const fd = c_open("/proc/meminfo", 0);
    if (fd < 0) return null;
    defer _ = c_close(fd);
    var buf: [4096]u8 = undefined;
    const got = c_read(fd, &buf, buf.len);
    if (got <= 0) return null;
    const n: usize = @intCast(got);
    var it = std.mem.tokenizeScalar(u8, buf[0..n], '\n');
    while (it.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var f = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \t");
        const kb = std.fmt.parseInt(usize, f.next() orelse return null, 10) catch return null;
        return kb * 1024;
    };
    return null;
}

/// A stream's rows [a0, a1) of a window shared by several streams (Python's Seg = (State, a0, a1)).
pub const Seg = struct { seq: *Seq, a0: usize, a1: usize };

/// One stream's window of a shared round: its tokens at its own seq.st.pos.
pub const Window = struct { seq: *Seq, tokens: []const u32 };

/// Scratch Python's wrappers allocate per call from torch's caching allocator, sized for the widest window.
pub const Scratch = struct {
    arena: state.Arena,
    part: u64,
    dn: u64, // [rows, low + streams] fp32
    shared_g: u64, // [rows, 2 NI] bf16
    shared_y: u64, // [rows, D] fp32 or bf16
    gq: u64, // [rows, nk, 128] fp32
    gk: u64,
    gv: u64, // [rows, nv, 128] bf16
    gg: u64, // [rows, nv] fp32
    gb: u64,
    gy: u64, // [rows, nv, 128] bf16
    kv_scale: u64, // a bf16 cache's one-element fp16 scales (kvcache.KVCache)
    invalid: u64, // gatherRows' bad-index flag
    vote: u64, // two ranks' growth agreement (this rank's word, then both)
    gdn_t: u64, // a prompt chunk's DeltaNet state of one layer, copied back over the sequence's
    // two ranks, n-gram table on the GPU: head row ids, this rank's embedding and group sums, both ranks' gathered
    ngram_ids: u64, // [rows, heads] int64
    ple_half: u64, // [rows, heads / world * dh] bf16
    xs_half: u64, // [rows, heads / world * dh / 32] fp32
    ple_all: u64, // [world, rows, heads / world * dh] bf16
    xs_all: u64,
    // INT4-AutoRound: a projection's block-FP8 and bf16 columns before copying, the shared expert's gate|up and SwiGLU
    p8: u64, // [rows, the widest block-FP8 face] bf16
    pb: u64, // [rows, the widest bf16 part] bf16
    sh_g: u64, // [rows, 2 shared width] bf16
    sh_a: u64, // [rows, shared width] bf16

    qsa_pos: u64, // [1] int32: a row-split indexer block's first own row position (cuda_prompt.zig)

    fn sizes(g: state.Geometry, rows: usize, part: usize) [24]usize {
        const ni = g.moe_width;
        const two = g.world > 1;
        const emb: usize = if (two) rows * g.ple_dim * 2 else 0;
        const xs: usize = if (two) rows * (g.ple_dim / 32) * 4 else 0;
        const on4 = g.int4ar;
        const face8 = @max(g.gdnConv() + g.nv * state.gdn_dv, g.heads * 2 * g.head_dim + 2 * g.kv_heads * g.head_dim);
        const faceb = @max(2 * g.nv, (g.index_heads + 1) * g.index_dim);
        const sw = g.sharedWidth();
        return .{ part, rows * (g.low + g.streams) * 4, rows * 2 * ni * 2, rows * g.hidden * 4, rows * g.nk * state.gdn_dk * 4, rows * g.nk * state.gdn_dk * 4, rows * g.nv * state.gdn_dv * 2, rows * g.nv * 4, rows * g.nv * 4, rows * g.nv * state.gdn_dv * 2, 256, 256, 256, g.nv * state.gdn_dv * state.gdn_dk * 4, if (two) rows * g.pleHeads() * 8 else 0, emb, xs, emb, xs, 256, if (on4) rows * face8 * 2 else 0, if (on4) rows * faceb * 2 else 0, if (on4) rows * 2 * sw * 2 else 0, if (on4) rows * sw * 2 else 0 };
    }

    pub fn init(d: *const cuda.Driver, g: state.Geometry, rows: usize, part: usize) !Scratch {
        const sz = sizes(g, rows, part);
        var total: usize = 0;
        for (sz) |n| total += std.mem.alignForward(usize, n, 256);
        var s: Scratch = undefined;
        s.arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, total) };
        errdefer s.arena.buf.free();
        try s.arena.buf.fill8(0, null);
        const ptrs = [_]*u64{ &s.part, &s.dn, &s.shared_g, &s.shared_y, &s.gq, &s.gk, &s.gv, &s.gg, &s.gb, &s.gy, &s.kv_scale, &s.invalid, &s.vote, &s.gdn_t, &s.ngram_ids, &s.ple_half, &s.xs_half, &s.ple_all, &s.xs_all, &s.qsa_pos, &s.p8, &s.pb, &s.sh_g, &s.sh_a };
        for (ptrs, sz) |p, n| p.* = s.arena.take(n);
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        s.arena.buf.free();
    }
};

/// The most rows the K-serial NVFP4 kernel takes for `n` columns (where it beats `_fp4mm` on GB10).
pub fn fp4SerialRows(n: usize) usize {
    return if (n <= 640) 128 else if (n <= 1280) 64 else 0;
}

pub const multi_ptrs = fw_round.multi_ptrs;

pub const Side = struct { s: cuda.Stream, fork: cuda.Event, join: cuda.Event };

pub const Round = fw_round.Round;

/// The split-K partials the widest window's matmuls need (bf16.matmul and nvfp4.matmul allocate them per call).
pub fn partBytes(g: state.Geometry, rows: usize) usize {
    const D = g.hidden;
    const Wd = g.wide();
    const b16 = [_][2]usize{
        .{ g.low + g.streams, Wd },                .{ g.low, Wd },        .{ Wd, g.low },               .{ g.gdnWidth(), D },
        .{ D, g.nv * state.gdn_dv },               .{ g.attnWidth(), D }, .{ D, g.heads * g.head_dim }, .{ Wd, g.ple_dim },
        .{ D, g.ple_dim },                         .{ D, D },             .{ g.head_n, D },
        // INT4-AutoRound's bf16 parts beside block FP8: DeltaNet b|a, the indexer
                    .{ 2 * g.nv, D },
        .{ (g.index_heads + 1) * g.index_dim, D },
    };
    var most: usize = 0;
    for (b16) |nk| most = @max(most, tri.b16PartBytes(rows * (if (nk[0] == D and nk[1] == D) g.streams else 1), nk[0], nk[1]));
    most = @max(most, tri.fp4PartBytes(rows, 2 * g.moe_width, D));
    most = @max(most, tri.fp4PartBytes(rows, D, g.moe_width));
    return @max(most, 256);
}

pub const Dump = fw_dump.Dump;

/// Two ranks' prompt glue split by rows (rh = ceil(R/2) to 16), partials exchanged; bits equal the unsplit path's.
pub const Split = struct {
    rank: usize,
    /// rows a rank runs: ceil(R / 2) rounded up to 16 (the second rank's last ones pad rows past R)
    rh: usize,
    /// this rank's first row and the peer's
    o: usize,
    po: usize,
};

/// Rows from which prompt chunks split the glue (smaller chunks: little to gain).
pub const split_min_rows = 64;

/// How a forward reaches the GPU: the stream, the three kernel sets, the weights and the shapes of this rank.
pub const Forward = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    ops: kern.Ops,
    t: tri.Tri,
    th: tops.Torch,
    w: *const W.Weights,
    c: *const cfgs.Config,
    dims: tri.Dims,
    g: state.Geometry,
    sc: Scratch,
    comm: ?*const comms.Comm = null,
    /// media attachments made (SeqMedia.epoch)
    media_epoch: u64 = 0,
    /// pinned staging: token ids (i32), then the n-gram rows (bf16)
    pin: cuda.HostBuffer,
    pin_rows: usize,
    staged: cuda.Event,
    /// each decoder layer's index among the DeltaNet layers or among the attention layers
    sub: [cfgs.max_layers]usize = @splat(0),
    dump: ?*Dump = null,
    /// a decode window's attention keys bound (a graph's context bucket, Python compute's `context`); null: exact
    context: ?usize = null,
    /// a shared round's graph: each segment's keys bound (its context bucket), parallel to the segments
    contexts: ?[]const usize = null,
    /// the sequences' memory (null: unbounded)
    budget: ?*Budget = null,
    /// decode rounds' DeltaNet scratch, the parity the last decode stage wrote, and every live sequence
    round: ?Round = null,
    cur_par: u1 = 1,
    seqs: std.ArrayList(*Seq) = .empty,
    /// two ranks: the n-gram table's GPU gather of this rank's heads (fn_pack.cu)
    gather: ?W.Gather = null,
    eps: f32,
    vocab_offset: i64 = 0,
    /// the sampling scratch (cuda_engine.zig sizes it for the head rows); `candidates` reads it at two ranks
    sample: ?tops.Torch.Sample = null,
    /// TF_FLASHNEXT_PROFILE: the prompt chunk's parts timed (cuda_prof.zig); null: no events
    prof: ?*profs.Prof = null,
    /// TF_FLASHNEXT_COUNT_EXPERTS=1: decode MoE layers' distinct routed experts and rows, counted since last report
    count_experts: bool = false,
    /// the shared expert's stream (initSide): it runs beside the routed experts, joined before their sum
    side: ?Side = null,
    /// `_fp4mm` SK 1 launches of 9 to fp4SerialRows(n) rows on fn_ops' K-serial kernel (fp4Serial)
    fp4_serial: bool = false,
    /// a shared round of at least this many streams runs attention in one launch a kernel (attn_multi; 0 off)
    attn_multi: usize = 0,
    round_multi: bool = false,
    experts_seen: [2]struct { layers: usize = 0, distinct: usize = 0, rows: usize = 0 } = .{ .{}, .{} },
    /// prompt split-K matmuls run their slices in one program (cuda_prompt.zig, same bits); false: Python's kernels
    prompt_mm: bool = false,
    /// prompt chunks' routed experts read each expert's weights once a call (cuda_moe_prompt.zig, the same bits)
    px: ?*const moep.Prompt4 = null,
    /// two ranks, prompt chunks: the hyper-connection glue split by rows (`Split`; same bits either way)
    glue_split: bool = false,
    /// INT4-AutoRound: the block-FP8 lane matmul and the GPTQ int4 kernels (cuda_engine loads them)
    k8: ?*const fp8.Kernels = null,
    k4: ?*const int4.Kernels = null,
    /// INT4-AutoRound: a projection's block-FP8 columns stored straight into its rows (fn_qmmf_ld), not copied
    fp8_ld: bool = true,
    /// TF_FLASHNEXT_L2PF=1 (cuda_comm.zig): weights read first by the kernels after the mixer's and MoE's gathers
    pf_mixer: comms.Prefetch = .{},
    pf_moe: comms.Prefetch = .{},
    /// decode D5 (TF_FLASHNEXT_MTP_Q4=1): the MTP head's matrices in 4-bit, run by `mmx` (drafts only)
    mtp_q4: ?*const @import("cuda_mtp_q4.zig").Table = null,
    /// on unless TF_FLASHNEXT_GLUE_FUSE=0: prompt read-outs' up projection and mix in one pass (`_hc_up_mix`)
    up_mix: bool = false,
    /// on unless TF_FLASHNEXT_WB_NORM=0: prompt chunks' write-back and norm in one pass (`_hc_wb_norm`)
    wb_norm: bool = false,
    /// two ranks, split prompt glue: the exchanges run on this stream beside the next rows' work (TF_FLASHNEXT_OVERLAP)
    cs: ?cuda.Stream = null,
    cs_ev: [4]cuda.Event = undefined,
    cs_i: usize = 0,
    /// comm stream overlap (TF_FLASHNEXT_OVERLAP bits): 1 the mixers' out_proj, 2 the MoE in two row halves
    overlap: u8 = 1,
    /// two ranks, prompt chunks: indexer rows split per rank from this many keys (TF_FLASHNEXT_QSA_SPLIT_KEYS; 0 never)
    qsa_split_keys: usize = 65536,
    /// prompt indexer kernels: `_scores_rows` rows a program (0: Python's), `_select_tiles` from this many blocks
    qsa_rt: usize = 0,
    qsa_tiles_from: usize = 8192,
    /// fn_qsa_scores.cu for the prompt indexer's scores (`_scores`' bits); null: Python's `_scores`
    qsa_fast: ?*const prompt.QsaScores = null,

    pub fn init(gpa: Allocator, d: *const cuda.Driver, s: cuda.Stream, ops: kern.Ops, t: tri.Tri, th: tops.Torch, w: *const W.Weights, c: *const cfgs.Config, g: state.Geometry, dims: tri.Dims, max_rows: usize) !Forward {
        var f: Forward = .{ .gpa = gpa, .d = d, .s = s, .ops = ops, .t = t, .th = th, .w = w, .c = c, .dims = dims, .g = g, .sc = undefined, .pin = undefined, .pin_rows = max_rows, .staged = undefined, .eps = @floatCast(c.eps), .vocab_offset = w.vocab_offset };
        f.sc = try Scratch.init(d, g, max_rows, partBytes(g, max_rows));
        errdefer f.sc.deinit();
        f.pin = try cuda.HostBuffer.alloc(d, max_rows * 4 + max_rows * g.pleHeads() * g.pleHeadDim() * 2 + 256);
        errdefer f.pin.free();
        f.staged = try cuda.Event.init(d, false);
        try f.staged.record(s);
        var lin: usize = 0;
        var att: usize = 0;
        for (w.layers, 0..) |l, i| {
            if (l.linear) {
                f.sub[i] = lin;
                lin += 1;
            } else {
                f.sub[i] = att;
                att += 1;
            }
        }
        if (lin != g.linear_layers or att != g.attention_layers) return error.LayerCountsDiffer;
        var ple: usize = 0;
        for (w.layers) |l| ple += @intFromBool(l.ple != null);
        if (ple > 1) return error.UnsupportedModel; // stage's rows are one PLE layer's (Python stages each at the same rows)
        if (w.table) |tab| if (tab.gpu != null) {
            f.gather = try W.Gather.init(d);
        };
        return f;
    }

    /// The decode rounds' DeltaNet scratch for `rows` rows of up to `streams` streams.
    pub fn initRound(f: *Forward, rows: usize, streams: usize) !void {
        f.round = try Round.init(f.d, f.g, rows, streams);
    }

    pub fn register(f: *Forward, seq: *Seq) !void {
        try f.seqs.append(f.gpa, seq);
    }

    pub fn unregister(f: *Forward, seq: *Seq) void {
        for (f.seqs.items, 0..) |q, i| if (q == seq) {
            _ = f.seqs.swapRemove(i);
            return;
        };
    }

    /// A second stream for the latency-bound shared expert beside the routed experts (same kernels, same bits).
    pub fn initSide(f: *Forward) !void {
        var s = try cuda.Stream.init(f.d, true);
        errdefer s.deinit();
        var fork = try cuda.Event.init(f.d, false);
        errdefer fork.deinit();
        const join = try cuda.Event.init(f.d, false);
        f.side = .{ .s = s, .fork = fork, .join = join };
    }

    pub fn deinit(f: *Forward) void {
        if (f.side) |*sd| {
            sd.join.deinit();
            sd.fork.deinit();
            sd.s.deinit();
        }
        if (f.round) |*r| r.deinit();
        f.seqs.deinit(f.gpa);
        if (f.gather) |*g| g.deinit();
        f.staged.deinit();
        f.pin.free();
        f.sc.deinit();
    }

    pub const flush = fw_round.flush;
    pub const roundTables = fw_round.roundTables;
    pub const multiTables = fw_round.multiTables;
    pub const uploadRound = fw_round.uploadRound;
    pub const splitOf = fw_ops.splitOf;
    pub const gatherRows = fw_ops.gatherRows;
    pub const nextEvent = fw_ops.nextEvent;
    pub const exchangeAsync = fw_ops.exchangeAsync;
    pub const exchangeWait = fw_ops.exchangeWait;
    pub const mark = fw_ops.mark;
    pub const mm = fw_ops.mm;
    pub const mmAt = fw_ops.mmAt;
    pub const denseAt = fw_ops.denseAt;
    pub const projFace = fw_ops.projFace;
    pub const headMm = fw_ops.headMm;
    pub const fp4On = fw_ops.fp4On;
    pub const fp4At = fw_ops.fp4At;
    pub const fp4Serial = fw_ops.fp4Serial;
    pub const mmx = fw_ops.mmx;
    pub const mmxAt = fw_ops.mmxAt;
    pub const setPos = fw_ops.setPos;
    pub const setMtpLen = fw_ops.setMtpLen;
    pub const rec = fw_ops.rec;
    pub const conv = fw_ops.conv;
    pub const upload = fw_ops.upload;
    pub const put = fw_dump.put;
    pub const putLayer = fw_dump.putLayer;
    pub const putBranch = fw_dump.putBranch;
    pub const reset = fw_stage.reset;
    pub const dropMedia = fw_stage.dropMedia;
    pub const attachMedia = fw_stage.attachMedia;
    pub const finishMedia = fw_stage.finishMedia;
    pub const injectMedia = fw_stage.injectMedia;
    pub const grow = fw_stage.grow;
    pub const agree = fw_stage.agree;
    pub const growMapped = fw_stage.growMapped;
    pub const stageIds = fw_stage.stageIds;
    pub const stageMany = fw_stage.stageMany;
    pub const stage = fw_stage.stage;
    pub const commit = fw_stage.commit;
    pub const commitAt = fw_stage.commitAt;
    pub const computeSegs = fw_layer.computeSegs;
    pub const prefetchOf = fw_layer.prefetchOf;
    pub const prefetchHc = fw_layer.prefetchHc;
    pub const compute = fw_layer.compute;
    pub const forward = fw_layer.forward;
    pub const layerForward = fw_layer.layerForward;
    pub const pleBlock = fw_layer.pleBlock;
    pub const gdnBlock = fw_layer.gdnBlock;
    pub const writeback = fw_hc.writeback;
    pub const writebackRows = fw_hc.writebackRows;
    pub const pssRow = fw_hc.pssRow;
    pub const hcBlock = fw_hc.hcBlock;
    pub const fusesNorm = fw_hc.fusesNorm;
    pub const readout = fw_hc.readout;
    pub const readoutAt = fw_hc.readoutAt;
    pub const outProj = fw_hc.outProj;
    pub const outProjX = fw_hc.outProjX;
    pub const finish = fw_hc.finish;
    pub const finishSegs = fw_hc.finishSegs;
    pub const candidates = fw_hc.candidates;
    pub const attnBlock = fw_attn.attnBlock;
    pub const qsaBlock = fw_attn.qsaBlock;
    pub const qsaBlockOpt = fw_attn.qsaBlockOpt;
    pub const qsaScoresSelect = fw_attn.qsaScoresSelect;
    pub const attnMulti = fw_attn.attnMulti;
    pub const moeBlock = fw_moe.moeBlock;
    pub const countExperts = fw_moe.countExperts;
    pub const moeRows = fw_moe.moeRows;
    pub const moeRowsInt4 = fw_moe.moeRowsInt4;
    pub const int4Down = fw_moe.int4Down;
};

test "the partial scratch covers the widest split-K matmul of a prompt chunk" {
    const g: state.Geometry = .{ .hidden = 2560, .streams = 4, .low = 320, .heads = 24, .kv_heads = 2, .head_dim = 256, .index_heads = 4, .index_dim = 128, .index_budget = 2048, .index_ratio = 4, .nk = 16, .nv = 48, .conv_kernel = 4, .experts = 512, .top_k = 10, .moe_width = 640, .ple_dim = 2560, .heads_per_ngram = 8, .ngram_size = 3, .ple_kernel = 4, .head_n = 248320, .linear_layers = 36, .attention_layers = 12, .mtp = true, .world = 1 };
    // the MTP head's fc_h over a chunk's 4 streams a row: 4 K slices of [4 * 2048, 2560] fp32
    try std.testing.expectEqual(@as(usize, 4 * 4 * 2048 * 2560 * 4), partBytes(g, 2048));
    // the MTP head's fc_h at 8 rows takes 32 stream rows of [2560, 2560]
    try std.testing.expect(partBytes(g, 8) >= tri.b16PartBytes(32, 2560, 2560));
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Forward);
    std.testing.refAllDecls(Dump);
    std.testing.refAllDecls(Scratch);
}
