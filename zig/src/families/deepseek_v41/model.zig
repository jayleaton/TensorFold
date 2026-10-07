//! One rank of the whole model for serving (M4 on the GPU): the backbone, the head and (with drafts) DSpark loaded at
//! TP=N from the pack, the forward over the tp runtime's session (rank 0 leads, the others follow its plan ops), the
//! GPU target and DSpark's pass. The gates' drivers (m2b.zig) load the same way from a reference's state; a served
//! slot starts empty and prefills through the target.
//!
//! The assets a host builds once next to the pack (TF_DSV41_ASSETS):
//! - `rope.bin`: the rope tables (tools/zig/dsv41_rope_tables.py, torch's cos / sin on that host);
//! - `engram-host.bin`: Engram's token map and hash multipliers (tools/zig/dsv41_engram_host.py);
//! - `aot/`: the Triton kernels by variant (any M1 capture's or M2b reference's AOT set on that GPU, merged with a
//!   set P capture's for DSpark: tools/zig/dsv41_aot_merge.py).
//! The kernel knobs: TF_DSV41_EXPERT_TOPP (prod 0.85; unset: off), TF_DSV41_R1=1 (prod-perf1's decode path).
//! M5 (kv_state.zig): TF_DSV41_POOL_TOKENS puts the KV families in the paged pool (its measured boot check first),
//! TF_DSV41_KV_SPLIT=1 shards the comp rows over the ranks; every rank must set the same.

const std = @import("std");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const tp = @import("tp");
const Config = @import("config.zig").Config;
const Pack = @import("pack.zig").Pack;
const plan = @import("plan.zig");
const named = @import("named.zig");
const load = @import("load.zig");
const buffers = @import("buffers.zig");
const forms = @import("forms.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const m2a = @import("m2a.zig");
const target_mod = @import("target.zig");
const dspark_emit = @import("dspark_emit.zig");
const dspark_gpu = @import("dspark_gpu.zig");
const kvs = @import("kv_state.zig");
const sg = @import("sessions_gpu.zig");
const slots_mod = @import("slots.zig");
const batch = @import("batch.zig");
const samp = @import("sampling_gpu.zig");

pub const Options = struct {
    drafts: bool = true,
    /// rows a prefill window takes (the decode buckets the plan sizes for) when the prompt runs as decode windows
    prefill_rows: u32 = 16,
    /// the prompt through the forward's own prefill (2,048-row segments) instead: TF_DSV41_OWN_PREFILL (default on)
    own_prefill: bool = true,
    /// the backbone layers to load and run (null: all; a gate runs a reference's prefix, e.g. 0-24)
    layers: ?[]const u32 = null,
};

/// TF_DSV41_LAYERS=<lo>-<hi>: load and run only that backbone prefix (a pod gate's model; null: every layer)
pub fn layersEnv(gpa: std.mem.Allocator) !?[]const u32 {
    const v = std.mem.span(std.c.getenv("TF_DSV41_LAYERS") orelse return null);
    var it = std.mem.tokenizeScalar(u8, v, '-');
    const lo = try std.fmt.parseInt(u32, it.next() orelse return error.BadLayers, 10);
    const hi = try std.fmt.parseInt(u32, it.next() orelse return error.BadLayers, 10);
    if (hi < lo) return error.BadLayers;
    const ls = try gpa.alloc(u32, hi - lo + 1);
    for (ls, 0..) |*l, i| l.* = lo + @as(u32, @intCast(i));
    return ls;
}

/// TF_DSV41_OWN_PREFILL=0: prompts as 16-row decode windows (every rank must agree: the plans differ)
pub fn ownPrefillEnv() bool {
    const v = std.c.getenv("TF_DSV41_OWN_PREFILL") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    tcfg: tp.Config,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    kernels: dk.Kernels,
    session: tp.Session,
    cfg: Config,
    pack: Pack,
    pl: plan.Plan,
    weights: load.Weights,
    aot: cuda.aot.Set,
    widths: @import("block.zig").Widths,
    runner: run.Runner,
    f: fwd.Forward,
    host: eh.Host,
    lbuf: [eh.max_layers]u32 = undefined,
    gt: target_mod.GpuTarget,
    pass: ?*dspark_gpu.GpuPass = null,
    /// M5: the session store over the paged pool (with TF_DSV41_POOL_TOKENS)
    sessions: ?*sg.Sessions = null,
    /// live slots (TF_DSV41_SLOTS; > 1: stacked state and row-mode windows, batch.zig)
    slots: u32 = 1,
    slot_set: ?*slots_mod.SlotSet = null,
    rows: ?*batch.Batch = null,
    /// the keyed sampler, every rank with TF_DSV41_SAMPLING=1 (requests with T > 0; followers run its gathers in follow)
    sampler: ?*samp.GpuSampler = null,

    pub fn rank(m: *const Model) u32 {
        return m.tcfg.rank;
    }

    /// Loads this rank (TF_TP_RANK / WORLD / DEVICE / MASTER / PORT) of `pack_dir` with `assets` (the module doc).
    pub fn open(gpa: std.mem.Allocator, io: std.Io, pack_dir: []const u8, assets: []const u8, o: Options) !*Model {
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.arena = std.heap.ArenaAllocator.init(gpa);
        errdefer m.arena.deinit();
        const a = m.arena.allocator();
        const cwd = std.Io.Dir.cwd();
        m.tcfg = try tp.Config.fromEnv();
        m.driver = try cuda.Driver.open();
        m.ctx = try cuda.Context.init(&m.driver, @intCast(m.tcfg.device));
        m.stream = try cuda.Stream.init(&m.driver, true);
        m.kernels = try dk.Kernels.load(&m.ctx);
        // TF_TP_KERNEL_IMAGE: the mailbox / RoCE kernels (TF_COMM_BACKEND=roce: the hybrid); unset: NCCL alone
        const image: []const u8 = if (std.c.getenv("TF_TP_KERNEL_IMAGE")) |p|
            try cwd.readFileAllocOptions(io, std.mem.span(p), a, .limited(1 << 26), .@"16", null)
        else
            &.{};
        try m.session.open(&m.driver, m.tcfg, image);
        const comm = m.session.comm();
        const world: u32 = @intCast(comm.world());
        m.cfg = try Config.read(gpa, io, pack_dir);
        m.pack = try Pack.open(gpa, io, pack_dir);
        const backbone = if (o.layers) |ls| try a.dupe(u32, ls) else blk: {
            const all = try a.alloc(u32, m.cfg.layers);
            for (all, 0..) |*l, i| l.* = @intCast(i);
            break :blk all;
        };
        var ds: [8]u32 = undefined;
        const blocks = if (o.drafts) try std.mem.concat(a, u32, &.{ backbone, dspark_emit.blocks(&m.cfg, &ds) }) else backbone;
        m.pl = try plan.build(gpa, &m.cfg, &m.pack, .{ .rank = m.tcfg.rank, .world = world, .blocks = blocks, .top = true, .dspark = o.drafts });
        m.weights = .{ .gpa = gpa };
        for (m.pl.layers) |*l| {
            var b: named.Builder = .{ .gpa = gpa, .io = io, .pack = &m.pack };
            defer b.deinit();
            try b.layer(l, m.cfg.o_groups, m.tcfg.rank, world);
            try m.weights.upload(&m.driver, &b);
        }
        {
            var b: named.Builder = .{ .gpa = gpa, .io = io, .pack = &m.pack };
            defer b.deinit();
            try b.top(&m.pl);
            try b.dspark(&m.pl);
            try m.weights.upload(&m.driver, &b);
        }
        m.aot = try cuda.aot.Set.load(gpa, io, &m.driver, m.ctx.device, try std.fs.path.join(a, &.{ assets, "aot" }));
        m.widths = try m2a.widths(a, &m.weights, &m.pl);
        m.runner = .{ .gpa = gpa, .d = &m.driver, .stream = m.stream, .kernels = &m.kernels, .triton = &m.aot, .weights = &m.weights };
        const topp: ?f64 = if (std.c.getenv("TF_DSV41_EXPERT_TOPP")) |v| try std.fmt.parseFloat(f64, std.mem.span(v)) else null;
        const r1 = if (std.c.getenv("TF_DSV41_R1")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        m.f = fwd.Forward.init(gpa, &m.cfg, &m.widths, .{ .world = world, .expert_topp = topp, .r1 = r1, .taps = o.drafts }, &m.runner, comm);
        m.f.layers = backbone;
        m.slots = try slots_mod.countFromEnv();
        if (try kvs.optionsFromEnv(@intCast(m.f.opts.limit), world)) |ko_| {
            var ko = ko_;
            ko.slots = m.slots;
            try kvs.bootCheck(&m.ctx, &m.cfg, ko, m.tcfg.rank, world);
            m.f.kv = try kvs.Kv.init(gpa, &m.driver, &m.ctx, &m.cfg, ko, &m.runner, comm);
            m.f.opts.pool = m.f.kv.?.options();
        }
        // the buffer plan: every window up to the prefill rows (a verify window is at most a block + 1), and DSpark's
        var buckets: [64]u32 = undefined;
        const nb = @max(o.prefill_rows, m.cfg.dspark_block + 1);
        for (buckets[0..nb], 0..) |*b, i| b.* = @intCast(i + 1);
        var bp = try m.f.plan(buckets[0..nb]);
        if (o.drafts) try dspark_gpu.GpuPass.plan(m.f.arena.allocator(), &m.f, &bp, nb);
        if (o.own_prefill) try m.f.planPrefill(&bp, @intCast(m.f.opts.prefill_rows));
        const rows_cap = try batch.capFromEnv();
        if (m.slots > 1) try batch.Batch.plan(&m.f, &bp, m.slots, rows_cap);
        if (m.f.kv) |k| try k.bind(&m.runner, &bp);
        try m.runner.bind(&bp);
        const src: forms.Source = .{ .d = &m.driver, .weights = &m.weights, .plan = &m.pl };
        for (bp.sizes.keys()) |role| {
            if (buffers.scopeOf(role) != .persistent) continue;
            var ra = std.heap.ArenaAllocator.init(gpa);
            defer ra.deinit();
            const bytes = (try forms.build(ra.allocator(), &src, role)) orelse continue;
            try cuda.DeviceBuffer.upload(.{ .d = &m.driver, .ptr = m.runner.addressOf(role).?, .len = bytes.len }, 0, bytes);
        }
        const rope = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ assets, "rope.bin" }), a, .limited(1 << 28));
        const half = (rope.len - 16) / 2;
        for ([_][]const u8{ "s.rope.main", "s.rope.comp" }, 0..) |role, i| if (m.runner.addressOf(role)) |p|
            try cuda.DeviceBuffer.upload(.{ .d = &m.driver, .ptr = p, .len = half }, 0, rope[16 + i * half ..][0..half]);
        const ehb = try cwd.readFileAllocOptions(io, try std.fs.path.join(a, &.{ assets, "engram-host.bin" }), a, .limited(1 << 26), .@"4", null);
        m.host = try eh.Host.load(ehb, m.cfg.engram_heads, m.cfg.engram_pad, m.cfg.engram_vocab, &m.lbuf);
        m.f.engram = &m.host;
        if (world > 1) m.f.link = &m.session.plan;
        if (m.f.kv != null and m.slots == 1) m.sessions = try sg.Sessions.init(gpa, io, &m.f, &bp, try sg.optionsFromEnv());
        m.gt = .{ .f = &m.f, .prefill_rows = o.prefill_rows, .max_rows = nb, .own_prefill = o.own_prefill, .sessions = m.sessions };
        if (o.drafts) m.pass = try dspark_gpu.GpuPass.init(gpa, &m.f, nb);
        if (m.slots > 1) try m.openRows(rows_cap, &bp);
        // TF_DSV41_SAMPLING=1: keyed sampling (T > 0) on every rank; off: such requests are refused (error.Sampling)
        m.sampler = if (samp.enabled()) try samp.GpuSampler.init(gpa, &m.f) else null;
        m.gt.sampler = m.sampler;
        try m.driver.check(m.driver.api.cuCtxSynchronize(), "cuCtxSynchronize");
        try comm.barrier();
        return m;
    }

    /// Several live slots (slots.zig / batch.zig): the stacked roles' bases, the row runner, the followers' row ops;
    /// one-slot graphs off (row windows have their own), sessions not opened (one slot's roles only).
    fn openRows(m: *Model, cap: u32, bp: *const buffers.Plan) !void {
        std.log.scoped(.dsv41).info("slots: {d} live, row windows of up to {d} rows (sessions off with several slots)", .{ m.slots, cap });
        const ss = try slots_mod.SlotSet.init(m.gpa, &m.f, m.slots);
        errdefer ss.deinit();
        try ss.bindBases(bp);
        m.slot_set = ss;
        m.rows = try batch.Batch.init(m.gpa, &m.f, ss, cap);
        m.f.ext = m.rows.?.ext();
        m.f.graphs_read = true;
        m.gt.rows = m.rows;
    }

    pub fn close(m: *Model) void {
        const gpa = m.gpa;
        if (m.rows) |b| b.deinit();
        if (m.slot_set) |ss| ss.deinit();
        if (m.pass) |p| p.deinit();
        if (m.sampler) |x| x.deinit();
        if (m.sessions) |x| x.deinit();
        if (m.f.kv) |k| k.deinit();
        m.f.deinit();
        m.runner.deinit();
        m.aot.deinit();
        m.weights.deinit();
        m.pl.deinit();
        m.pack.deinit();
        m.session.close();
        m.kernels.deinit();
        m.stream.deinit();
        m.ctx.deinit();
        m.driver.close();
        m.arena.deinit();
        gpa.destroy(m);
    }
};
