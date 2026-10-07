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

pub const Options = struct {
    drafts: bool = true,
    /// rows a prefill window takes (the decode buckets the plan sizes for)
    prefill_rows: u32 = 16,
};

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
        const backbone = try a.alloc(u32, m.cfg.layers);
        for (backbone, 0..) |*l, i| l.* = @intCast(i);
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
        // the buffer plan: every window up to the prefill rows (a verify window is at most a block + 1), and DSpark's
        var buckets: [64]u32 = undefined;
        const nb = @max(o.prefill_rows, m.cfg.dspark_block + 1);
        for (buckets[0..nb], 0..) |*b, i| b.* = @intCast(i + 1);
        var bp = try m.f.plan(buckets[0..nb]);
        if (o.drafts) try dspark_gpu.GpuPass.plan(m.f.arena.allocator(), &m.f, &bp, nb);
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
        m.gt = .{ .f = &m.f, .prefill_rows = o.prefill_rows, .max_rows = nb };
        if (o.drafts) m.pass = try dspark_gpu.GpuPass.init(gpa, &m.f, nb);
        try m.driver.check(m.driver.api.cuCtxSynchronize(), "cuCtxSynchronize");
        try comm.barrier();
        return m;
    }

    pub fn close(m: *Model) void {
        const gpa = m.gpa;
        if (m.pass) |p| p.deinit();
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
