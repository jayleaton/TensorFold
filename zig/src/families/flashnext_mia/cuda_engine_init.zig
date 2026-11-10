//! Flash Next engine setup: Engine.init (weights, buffers, scratch, memory budget) and sequence caches.
const std = @import("std");
const envs = @import("cuda_env.zig");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const tops = @import("cuda_torch_ops.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const cfgs = @import("cuda_config.zig");
const rope = @import("cuda_rope.zig");
const fwd = @import("cuda_forward.zig");
const fp8 = @import("cuda_fp8.zig");
const int4 = @import("cuda_int4.zig");
const vmm = @import("cuda_vmm.zig");
const profs = @import("cuda_prof.zig");
const prompt_mm = @import("cuda_prompt.zig");
const moep = @import("cuda_moe_prompt.zig");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const Allocator = std.mem.Allocator;
const max_k = engine.max_k;
const Options = engine.Options;
const ring_min_rows = engine.ring_min_rows;
const Seq = engine.Seq;
const multi_graphs_default = engine.multi_graphs_default;
const seconds = engine.seconds;
const geometry = engine.geometry;
const dims = engine.dims;
const aligned = engine.aligned;
const tp2_prefill_rows = engine.tp2_prefill_rows;

/// /proc/meminfo's MemAvailable in bytes, or null.
fn memAvailable(io: std.Io) ?usize {
    var buf: [4096]u8 = undefined;
    const text = std.Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf) catch return null;
    var it = std.mem.tokenizeScalar(u8, text, '\n');
    while (it.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var f = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \t");
        const kb = std.fmt.parseInt(usize, f.next() orelse return null, 10) catch return null;
        return kb * 1024;
    };
    return null;
}

/// cfgs.Yarn (the config's form) from cuda_rope's.
fn yarnConfig(y: rope.Yarn) cfgs.Yarn {
    return .{ .factor = y.factor, .original_max_position_embeddings = y.original, .beta_fast = y.beta_fast, .beta_slow = y.beta_slow, .attention_factor = if (y.attention_factor == 0) null else y.attention_factor, .truncate = y.truncate };
}

/// Field-wise maximum of sampling scratch sizes.
fn sampleSizes(f: *const tops.Functions, cases: []const [3]usize) ![7]usize {
    var most: [7]usize = @splat(0);
    for (cases) |c| {
        const s = try tops.Torch.Sample.bytes(f, c[0], c[1], c[2]);
        for (&most, s) |*m, v| m.* = @max(m.*, v);
    }
    return most;
}

fn takeSample(a: *state.Arena, sz: [7]usize) tops.Torch.Sample {
    return .{ .f32 = a.take(sz[0]), .vals = a.take(sz[1]), .idx = a.take(sz[2]), .topk = a.take(sz[3]), .lse = a.take(sz[4]), .lse_scratch = a.take(sz[5]), .col = a.take(sz[6]) };
}

/// A non-blocking stream at `priority` (lower is higher; the driver clamps to its range).
fn streamWithPriority(d: *const cuda.Driver, priority: c_int) !cuda.Stream {
    if (priority == 0) return cuda.Stream.init(d, true);
    var lib = std.DynLib.open("libcuda.so.1") catch return cuda.Stream.init(d, true);
    defer lib.close();
    const F = *const fn (*cuda.abi.Stream, c_uint, c_int) callconv(.c) cuda.abi.Result;
    const f = lib.lookup(F, "cuStreamCreateWithPriority") orelse return cuda.Stream.init(d, true);
    var s: cuda.abi.Stream = null;
    try d.check(f(&s, cuda.abi.stream_non_blocking, priority), "cuStreamCreateWithPriority");
    return .{ .d = d, .handle = s };
}

fn promptRows(default: usize) !usize {
    const v = envs.get("TF_FLASHNEXT_PREFILL_ROWS") orelse return default;
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadPrefillRows;
    if (n < 16 or n > 16384) return error.BadPrefillRows;
    return n;
}

pub fn init(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels_dir: []const u8, o: Options) !*Engine {
    try envs.startup();
    const t0 = std.Io.Clock.awake.now(io);
    if (o.world < 1 or o.world > 2 or o.rank >= o.world) return error.BadRank;
    if (o.world > 1 and o.comm == null) return error.TwoRanksNeedComm;
    if (o.depth > state.max_rows - 1) return error.DepthPastWindow;
    const e = try gpa.create(Engine);
    errdefer gpa.destroy(e);
    e.* = .{ .gpa = gpa, .io = io, .ctx = ctx, .stream = undefined, .k = undefined, .torch = undefined, .set = undefined, .c = undefined, .w = undefined, .g = undefined, .dims = undefined, .buf = undefined, .pbuf = undefined, .f = undefined, .extra = undefined, .main_sample = undefined, .last_logits = 0, .last_cand = 0, .max_len = 0, .depth = if (o.mtp) o.depth else 0, .rank = o.rank, .world = o.world, .prefill_rows = try promptRows(if (o.world == 2 and o.prefill_rows == state.prefill_rows) tp2_prefill_rows else o.prefill_rows), .graphs_on = o.graphs };
    const d = ctx.d;
    e.c = try cfgs.Config.read(gpa, io, dir, .{ .yarn = if (o.yarn) |y| yarnConfig(y) else null });
    errdefer e.c.deinit();
    e.stream = try cuda.Stream.init(d, true);
    errdefer e.stream.deinit();
    e.k = try kern.Kernels.load(ctx);
    errdefer e.k.deinit();
    e.torch = try tops.Ops.load(ctx);
    errdefer e.torch.deinit();
    e.set = try cuda.aot.Set.load(gpa, io, d, ctx.device, kernels_dir);
    errdefer e.set.deinit();
    // TF_FLASHNEXT_INT4AR_FAST=1: INT4-AutoRound's lossy fast-fp8/ variant over the checkpoint (opt-in)
    var overlay_buf: [1024]u8 = undefined;
    const overlay: ?[]const u8 = if (e.c.int4ar() and envs.get("TF_FLASHNEXT_INT4AR_FAST") != null and !envs.off("TF_FLASHNEXT_INT4AR_FAST")) try std.fmt.bufPrint(&overlay_buf, "{s}/fast-fp8", .{dir}) else null;
    e.w = try weights.load(gpa, io, dir, &e.c, .{ .rank = o.rank, .world = o.world, .mtp = o.mtp, .draft_head = o.mtp, .mode = .device, .driver = d, .overlay = overlay });
    errdefer e.w.deinit();
    if (o.mtp and e.w.mtp == null) return error.NoMtpHead;
    e.g = try geometry(&e.c, o.world, e.w.mtp != null);
    e.g.kv = o.kv;
    e.dims = dims(&e.c, e.g);
    // Python: cache_slots = the window + depth + 1 (capacity.admit), every buffer's attention geometry at it
    e.max_len = o.context + e.depth + 1;
    const rows = @max(8, @max(1, o.streams) * (e.depth + 1));
    e.round_segs = try gpa.alloc(fwd.Seg, @max(1, o.streams));
    errdefer gpa.free(e.round_segs);
    const ag = tri.AttnGeometry.init(e.max_len, e.g.index_budget, e.g.index_ratio);
    e.buf = .{ .b = try state.Buffers.init(d, e.g, rows, .{ .moe_prefill = true, .capacity = e.max_len }), .y_f32 = false, .attn = ag };
    errdefer e.buf.deinit();
    if (e.w.mtp != null) e.mbuf = .{ .b = try state.Buffers.init(d, e.g, rows, .{ .capacity = e.max_len }), .y_f32 = true, .attn = ag };
    errdefer if (e.mbuf) |*m| m.deinit();
    // default 512: a last chunk takes up to 512 more rows instead of its own chunk (exact: chunk invariance)
    if (e.prefill_rows + 512 <= 16384) e.prefill_tail = 512;
    if (envs.get("TF_FLASHNEXT_PREFILL_TAIL")) |v| {
        e.prefill_tail = std.fmt.parseInt(usize, v, 10) catch return error.BadPrefillTail;
        if (e.prefill_tail % 16 != 0 or e.prefill_rows + e.prefill_tail > 16384) return error.BadPrefillTail;
    }
    e.pbuf = .{ .b = try state.Buffers.init(d, e.g, e.prefill_rows + e.prefill_tail, .{ .prefill = true, .capacity = e.max_len }), .y_f32 = false, .attn = tri.AttnGeometry.init(e.max_len, e.g.index_budget, e.g.index_ratio) };
    errdefer e.pbuf.deinit();
    try e.initWindows();
    const th = e.torch.on(e.stream);
    e.memo = .{ .gpa = gpa };
    errdefer e.memo.deinit();
    e.f = try fwd.Forward.init(gpa, d, e.stream, .{ .k = &e.k, .s = e.stream }, .{ .set = &e.set, .s = e.stream, .memo = &e.memo }, th, &e.w, &e.c, e.g, e.dims, e.prefill_rows + e.prefill_tail);
    errdefer e.f.deinit();
    e.f.comm = o.comm;
    // decode rounds' DeltaNet scratch: every stream's window rows, two rounds' kept inputs
    try e.f.initRound(rows, @max(1, o.streams));
    if (e.c.int4ar()) {
        e.k8 = try fp8.Kernels.load(ctx);
        e.k4 = try int4.Kernels.load(ctx);
        e.f.k8 = &e.k8.?;
        e.f.k4 = &e.k4.?;
    }
    errdefer if (e.k8) |*q| q.deinit();
    errdefer if (e.k4) |*q| q.deinit();
    // prompt matmuls without split-K partials when the kernel set has them (TF_FLASHNEXT_PROMPT_MM=0: off)
    e.f.prompt_mm = !envs.off(prompt_mm.env) and prompt_mm.available(&e.set);
    e.draw_once = !envs.off("TF_FLASHNEXT_DRAW_ONCE");
    // the shared expert beside the routed experts on a second stream (TF_FLASHNEXT_SHARED_SIDE=0: in line)
    if (!envs.off("TF_FLASHNEXT_SHARED_SIDE")) try e.f.initSide();
    // INT4-AutoRound's projections: block-FP8 columns stored in place (TF_FLASHNEXT_FP8_LD=0: scratch + copy)
    e.f.fp8_ld = !envs.off("TF_FLASHNEXT_FP8_LD");
    // shared expert's single-slice gate/up on the K-serial kernel (same bits); TF_FLASHNEXT_FP4_SERIAL=0 off
    e.f.fp4_serial = !envs.off("TF_FLASHNEXT_FP4_SERIAL");
    e.graph_log = envs.get("TF_FLASHNEXT_GRAPH_LOG") != null;
    if (envs.get("TF_FLASHNEXT_GRAPH_FAIL")) |v| e.inject_every = std.fmt.parseInt(u64, v, 10) catch 0;
    if (envs.get("TF_FLASHNEXT_MULTI_GRAPHS")) |v| e.multi_graphs = std.fmt.parseInt(usize, v, 10) catch multi_graphs_default;
    // shared rounds past captured widths run attention in one launch: TF_FLASHNEXT_ATTN_MULTI=N from N streams
    if (tri.Tri.multiAvailable(&e.set)) {
        const want: usize = if (envs.get("TF_FLASHNEXT_ATTN_MULTI")) |v| (std.fmt.parseInt(usize, v, 10) catch 0) else e.multi_graphs + 1;
        if (want > 0) e.f.attn_multi = @max(want, e.multi_graphs + 1, 2);
        // the kernel set's attn_multi kernels read bf16 caches: an FP8 cache keeps the one-stream kernels
        if (e.g.kv != .bf16) e.f.attn_multi = 0;
    }
    if (envs.get("TF_FLASHNEXT_MEDIA_GRAPHS")) |v| e.media_graphs = !std.mem.eql(u8, v, "0");
    e.f.count_experts = if (envs.get("TF_FLASHNEXT_COUNT_EXPERTS")) |v| std.mem.eql(u8, v, "1") else false;
    if (!envs.off(moep.env)) {
        e.px = try moep.Prompt4.init(ctx);
        if (e.px.?.gu2) try e.px.?.reserveGate(d, (e.prefill_rows + e.prefill_tail) * e.g.slots(), e.g.moe_width);
        e.f.px = &e.px.?;
    }
    errdefer if (e.px) |*q| q.deinit();
    // two ranks: prompt glue split by rows (TF_FLASHNEXT_GLUE_SPLIT=0: every rank runs every row)
    e.f.glue_split = o.world == 2 and !envs.off("TF_FLASHNEXT_GLUE_SPLIT");
    e.digest = envs.get("TF_FLASHNEXT_PREFILL_DIGEST") != null;
    // NCCL connects point-to-point channels on first use, so one prompt-sized exchange runs now on both ranks
    if (e.f.glue_split) if (o.comm) |cm| {
        const n = @min(e.prefill_rows / 2, 2048) * e.g.hidden;
        try cm.exchange(e.pbuf.b.part_branch, e.pbuf.b.g_branch, n, .f32, e.stream.handle);
        try cm.exchange(e.pbuf.b.part_moe, e.pbuf.b.g_moe, n, .f32, e.stream.handle);
        try e.stream.synchronize();
    };
    // both ranks loaded: drop the checkpoint's pages from the page cache (TF_FLASHNEXT_DROP_PAGES=0 keeps them)
    if (!envs.off("TF_FLASHNEXT_DROP_PAGES")) weights.dropPages(gpa, io, dir);
    e.poison = envs.get("TF_FLASHNEXT_POISON_PADS") != null;
    e.pass_times = envs.get("TF_FLASHNEXT_PASS_TIMES") != null;
    e.f.up_mix = !envs.off("TF_FLASHNEXT_GLUE_FUSE") and prompt_mm.upMixAvailable(&e.set);
    e.f.wb_norm = !envs.off("TF_FLASHNEXT_WB_NORM") and prompt_mm.wbNormAvailable(&e.set);
    // the split glue's exchanges on their own stream beside the other rows' work (TF_FLASHNEXT_OVERLAP=0: in line)
    if (!envs.off("TF_FLASHNEXT_QSA_FAST")) {
        e.qsa_fast = try prompt_mm.QsaScores.init(d);
        e.f.qsa_fast = &e.qsa_fast.?;
    }
    errdefer if (e.qsa_fast) |*q| q.deinit();
    if (envs.get("TF_FLASHNEXT_QSA_RT")) |v| {
        e.f.qsa_rt = std.fmt.parseInt(usize, v, 10) catch 0;
    }
    if (envs.get("TF_FLASHNEXT_QSA_TILES_FROM")) |v| {
        e.f.qsa_tiles_from = std.fmt.parseInt(usize, v, 10) catch std.math.maxInt(usize);
    }
    if (envs.get("TF_FLASHNEXT_QSA_SPLIT_KEYS")) |v| {
        e.f.qsa_split_keys = std.fmt.parseInt(usize, v, 10) catch 0;
    }
    // measured ~1% (research/R2-measured.md): off unless asked
    if (e.f.glue_split and envs.get("TF_FLASHNEXT_OVERLAP") != null and !envs.off("TF_FLASHNEXT_OVERLAP")) {
        if (envs.get("TF_FLASHNEXT_OVERLAP")) |v| e.f.overlap = std.fmt.parseInt(u8, v, 10) catch 1;
        const prio: c_int = if (envs.get("TF_FLASHNEXT_COMM_PRIORITY")) |v| (std.fmt.parseInt(c_int, v, 10) catch 0) else 0;
        e.f.cs = try streamWithPriority(d, prio);
        for (&e.f.cs_ev) |*ev| ev.* = try cuda.Event.init(d, false);
        // CTAs the persistent expert kernels leave free for NCCL's (the grids loop over units: same bits)
        if (envs.get("TF_FLASHNEXT_EXPERT_SMS_FREE")) |v| {
            const free = std.fmt.parseInt(usize, v, 10) catch 0;
            const sms: usize = @intCast(e.k.sms);
            for (&e.k.nvfp4_blocks) |*n| n.* = n.* / sms * (sms - @min(free, sms - 1));
            if (e.px) |*q| q.sms = sms - @min(free, sms - 1);
        }
    }
    errdefer if (e.f.cs) |*cs| {
        cs.deinit();
        for (&e.f.cs_ev) |*ev| ev.deinit();
    };
    if (profs.wanted(envs.get(profs.env))) {
        e.prof = try profs.Prof.init(gpa, d, 16384);
    }
    errdefer if (e.prof) |p| p.deinit(gpa);
    if (o.rank == 0) std.debug.print("prompt path: {d}-row chunks (the last up to {d} more), glue {s}{s}, matmul K slices {s}, experts {s}{s}\n", .{ e.prefill_rows, e.prefill_tail, if (e.f.glue_split) "split by rows" else "on every row", if (e.f.cs != null) " (exchanges overlapped)" else "", if (e.f.prompt_mm) "in one program (_b16mm_ks/_fp4mm_ks)" else "split with _reduce", if (e.px != null) "on 128-pair items (fn_experts_prompt)" else "on 16-pair items", if (e.prof != null) ", profiled" else "" });
    if (e.w.mtp != null) if (envs.get("TF_FLASHNEXT_MTP_Q4")) |v| if (!std.mem.eql(u8, v, "0")) {
        const mse = envs.is("TF_FLASHNEXT_MTP_Q4_MSE", "1");
        e.mtpq = try @import("cuda_mtp_q4.zig").Table.build(gpa, d, &e.w, v, mse);
        e.f.mtp_q4 = &e.mtpq.?;
        std.log.info("MTP head in 4-bit for the drafts: {d} matrices, {d:.1} -> {d:.1} MB", .{ e.mtpq.?.n, @as(f64, @floatFromInt(e.mtpq.?.bytes_bf16)) / 1e6, @as(f64, @floatFromInt(e.mtpq.?.bytes_q4)) / 1e6 });
    };
    errdefer if (e.mtpq) |*t| t.deinit();
    // sampling scratch: the main head's rows and the draft head's one row
    const draft_n: usize = if (e.w.draft_head) |q| q.n else e.g.head_n;
    const main_sz = try sampleSizes(&e.torch.f, &.{ .{ state.max_rows, e.g.head_n, max_k }, .{ 1, e.g.head_n, max_k }, .{ rows, e.g.head_n, state.cand }, .{ 1, draft_n, state.cand }, .{ rows, draft_n, state.cand } });
    const draft_sz = try sampleSizes(&e.torch.f, &.{.{ 1, draft_n, draft_n }});
    const pack = state.max_rows * (2 * max_k + 1) * 4;
    const fixed = [_]usize{ e.g.head_n * 2, o.world * (2 * state.cand + 1) * 4, (2 * draft_n + 1) * 4, if (o.world > 1) pack else 0, if (o.world > 1) o.world * pack else 0, @max(e.g.head_n, draft_n) * 8, 8, 8 };
    e.extra = .{ .buf = try cuda.DeviceBuffer.alloc(d, aligned(&main_sz) + aligned(&draft_sz) + aligned(&fixed)) };
    errdefer e.extra.buf.free();
    try e.extra.buf.fill8(0, null);
    e.main_sample = takeSample(&e.extra, main_sz);
    e.f.sample = e.main_sample;
    const draft_s = takeSample(&e.extra, draft_sz);
    e.last_logits = e.extra.take(fixed[0]);
    e.last_cand = e.extra.take(fixed[1]);
    const draft_out = e.extra.take(fixed[2]);
    if (o.world > 1) e.tp = .{ .s = e.main_sample, .pack = e.extra.take(fixed[3]), .all = e.extra.take(fixed[4]), .max_rows = state.max_rows, .max_k = max_k };
    e.nsc = .{ .mass = e.extra.take(fixed[5]), .top = e.extra.take(fixed[6]), .sum = e.extra.take(fixed[7]) };
    if (e.w.mtp != null) {
        if (e.w.draft_head != null) e.draft_host = try weights.draftIds(gpa, weights.draft_vocab, e.c.vocab, o.rank, o.world);
        e.draws = .{ .gpa = gpa, .ids = if (e.w.draft_head != null) e.draft_host else null, .columns = draft_n, .world = o.world, .s = draft_s, .out = draft_out, .max_k = draft_n, .tp = e.tp, .nsc = e.nsc };
    }
    errdefer gpa.free(e.draft_host);
    if (o.vmm) {
        e.vm = vmm.Vmm.init(ctx) catch |err| blk: {
            std.log.warn("CUDA VMM unavailable ({s}): caches grow by copying", .{@errorName(err)});
            break :blk null;
        };
        if (e.vm) |v| std.log.info("caches in reserved address space ({d} KiB granularity), indexer keys in a ring", .{v.granularity >> 10});
    }
    // the sequences' memory: what is free now (MemAvailable: GB10's memory is unified), less the reserve
    try e.stream.synchronize();
    // the floor the server and the recipe set (TENSORFOLD_MEMORY_RESERVE_GIB) wins over the option
    var reserve_gib = o.reserve_gib;
    if (envs.get("TENSORFOLD_MEMORY_RESERVE_GIB")) |v| reserve_gib = std.fmt.parseFloat(f64, v) catch return error.BadMemoryReserve;
    const reserve: usize = @intFromFloat(reserve_gib * (1 << 30));
    // MemAvailable counts the mapped n-gram table's reclaimable pages; the device's free count leaves them out
    const free = memAvailable(io) orelse (try ctx.memInfo()).free;
    e.budget = .{ .limit = o.kv_budget orelse (free -| reserve -| o.vision_reserve) };
    if (o.vision_reserve > 0) std.log.info("vision: {d:.2} GiB kept free for the image tower's workspace", .{@as(f64, @floatFromInt(o.vision_reserve)) / (1 << 30)});
    // two ranks decide growth together: both take the smaller budget
    if (o.comm) |cm| {
        const v: cuda.DeviceBuffer = .{ .d = d, .ptr = e.f.sc.vote, .len = 8 };
        try v.upload(0, std.mem.asBytes(&@as(u64, e.budget.limit)));
        try cm.allGather(e.f.sc.vote, e.f.sc.vote + 64, 1, .u64, e.stream.handle);
        try e.stream.synchronize();
        var both: [2]u64 = undefined;
        const r: cuda.DeviceBuffer = .{ .d = d, .ptr = e.f.sc.vote + 64, .len = 16 };
        try r.download(0, std.mem.sliceAsBytes(&both));
        e.budget.limit = @min(both[0], both[1]);
    }
    e.f.budget = &e.budget;
    e.budget.floor = @intFromFloat(@min(reserve_gib, 10) * (1 << 30));
    std.log.info("sequence memory: {d:.2} GiB ({d:.2} GiB free, {d:.1} GiB kept free); {s} KV cache, {d} bytes a position a layer", .{ @as(f64, @floatFromInt(e.budget.limit)) / (1 << 30), @as(f64, @floatFromInt(free)) / (1 << 30), reserve_gib, @tagName(e.g.kv), e.g.kRow() + e.g.vRow() });
    e.own = try e.newSeq(e.max_len);
    e.bound = e.own;
    try e.stream.synchronize();
    e.load_seconds = seconds(io, t0);
    return e;
}

/// The prompt buffers' DeltaNet taps: row r reads [conv state (3) | rows] at r .. r + 3 (Python Buffers.windows).
pub fn initWindows(e: *Engine) !void {
    const b = &e.pbuf.b;
    const taps = e.g.conv_kernel;
    const host = try e.gpa.alloc(i32, b.rows * taps);
    defer e.gpa.free(host);
    for (0..b.rows) |r| for (0..taps) |j| {
        host[r * taps + j] = @intCast(r + j);
    };
    const buf: cuda.DeviceBuffer = .{ .d = e.ctx.d, .ptr = b.windows, .len = host.len * 4 };
    try buf.upload(0, std.mem.sliceAsBytes(host));
    const sid: cuda.DeviceBuffer = .{ .d = e.ctx.d, .ptr = b.sid, .len = b.rows * 4 };
    try sid.fill32(0, null);
}

// -- sequences --------------------------------------------------------------------------------------------

/// A new empty sequence whose caches grow up to `limit_rows` (at most the engine's max_len).
pub fn newSeq(e: *Engine, limit_rows: usize) !*Seq {
    const limit = @min(limit_rows, e.max_len);
    const s = try e.gpa.create(Seq);
    errdefer e.gpa.destroy(s);
    // caches start at one growth step and grow with the stream (State.ensure), up to `limit`
    const first = @min(limit, state.grow_step);
    if (!e.fits(first)) return error.NoRoom;
    s.* = .{ .st = try state.State.init(e.ctx.d, e.g, if (e.vm != null) 1 else first, limit) };
    errdefer s.deinit(e.gpa);
    s.charged = s.st.fixed.buf.len + (if (s.st.ctx) |c| c.buf.len else 0) + e.g.wide() * 2;
    e.budget.used += s.charged;
    errdefer e.budget.used -|= s.charged;
    if (e.vm) |*v| try e.mapSeq(s, v, first);
    try e.f.register(s);
    errdefer e.f.unregister(s);
    s.last_streams = try cuda.DeviceBuffer.alloc(e.ctx.d, e.g.wide() * 2);
    try e.f.reset(s);
    return s;
}

/// A sequence's caches in reserved address space: per layer KV and pooled ranges mapped to `first`, a key ring.
pub fn mapSeq(e: *Engine, s: *Seq, v: *const vmm.Vmm, first: usize) !void {
    const g = e.g;
    const st = &s.st;
    if (st.ctx) |*c| {
        s.charged -= c.buf.len;
        e.budget.used -= c.buf.len;
        c.buf.free();
        st.ctx = null;
    }
    const ikc_row = g.index_dim * 2;
    const chunk = e.ringBytes(v);
    const layers = g.attention_layers + @intFromBool(g.mtp);
    const limit = st.limit;
    for (0..layers) |i| {
        const sizes = [4]usize{ limit * g.kRow(), limit * g.vRow(), std.mem.alignForward(usize, limit * ikc_row, chunk), g.blocks(limit) * ikc_row };
        for (sizes) |b| {
            const r = try vmm.Region.reserve(v, b);
            try s.regions.append(e.gpa, r);
        }
        const r = s.regions.items[4 * i ..][0..4];
        const ring = try r[2].mapRing(e.gpa, chunk);
        try e.f.th.zero(r[2].base, ring);
        s.charged += ring;
        e.budget.used += ring;
        st.kc[i] = r[0].base;
        st.vc[i] = r[1].base;
        st.ikc[i] = r[2].base;
        st.pooled[i] = r[3].base;
    }
    s.ring_rows = chunk / ikc_row;
    st.capacity = 0;
    try e.f.grow(s, first);
}

pub fn freeSeq(e: *Engine, s: *Seq) void {
    if (e.bound == s and s != e.own) e.bound = e.own;
    e.budget.used -|= s.charged;
    e.f.unregister(s);
    e.dropGraphs(@intFromPtr(s), null);
    e.dropMulti(@intFromPtr(s));
    s.deinit(e.gpa);
    e.gpa.destroy(s);
}

/// One cache layer's indexer-key ring bytes: chunk rows plus 3, at least ring_min_rows, a power of two.
pub fn ringBytes(e: *const Engine, v: *const vmm.Vmm) usize {
    const rows = std.math.ceilPowerOfTwo(usize, @max(ring_min_rows, e.prefill_rows + e.g.index_ratio)) catch unreachable;
    return v.up(@max(rows * e.g.index_dim * 2, v.granularity));
}

/// Bytes a new sequence of `rows` context rows takes (fixed state plus caches at that growth step), as newSeq does.
pub fn seqBytes(e: *const Engine, rows: usize) usize {
    const g = e.g;
    const cap = @min(e.max_len, std.mem.alignForward(usize, @max(rows, 1), state.grow_step));
    const fixed = state.State.fixedBytes(g) + g.wide() * 2;
    const v = if (e.vm) |*x| x else return fixed + state.State.cacheBytes(g, cap) + 64 * 256;
    const layer = e.ringBytes(v) + v.up(cap * g.kRow()) + v.up(cap * g.vRow()) + v.up(g.blocks(cap) * g.index_dim * 2);
    return fixed + (g.attention_layers + @intFromBool(g.mtp)) * layer;
}

/// Whether a new sequence that will hold `rows` rows fits beside the live ones now (admission).
pub fn fits(e: *const Engine, rows: usize) bool {
    return e.budget.used + e.seqBytes(rows) <= e.budget.limit;
}
