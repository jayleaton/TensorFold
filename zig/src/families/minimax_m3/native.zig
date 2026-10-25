//! MiniMax-M3.1-Flash behind the native CUDA host (GLM's contract): `open` checks, joins the ranks, builds the engine.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const lanes = @import("lanes");
const iface = @import("iface.zig");
const ranks = @import("ranks.zig");
const budget = @import("budget.zig");
const config = @import("config.zig");
const load = @import("load.zig");
const pool = @import("pool.zig");
const shard = @import("shard.zig");
const pages = @import("pages.zig");
const weights_device = @import("weights_device.zig");
const engine = @import("engine.zig");
const engine_dev = @import("engine_dev.zig");
const forward_plan = @import("forward_plan.zig");
const forward_exec = @import("forward_exec.zig");
const follow = @import("follow.zig");
const backend = @import("backend.zig");
const dspark_config = @import("dspark_config.zig");
const dspark_folder = @import("dspark_folder.zig");
const dspark_drafter = @import("dspark_drafter.zig");

pub const model_type = config.model_type;
pub const formats: []const []const u8 = &.{"exl3"};
// The plan says 65,536 to 131,072 tokens; the host's memory plan and `admission` decide what really fits.
pub const default_context: i64 = 65536;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = iface.prefill_rows;
pub const window_rows: u32 = iface.window_rows;
pub const explains_open = true;
// --drafter takes the dspark draft (the checkpoint's own dspark/ folder, or a folder of one): there is no MTP head
pub const takes_drafter = true;
// Decision M2: the backend owns a radix prompt cache over the pool's pages, so the host builds no snapshot store.
pub const owns_prompt_cache = iface.decisions.owns_prompt_cache;

/// The server's reading of this rank's room after its load (the budget streams are admitted by); null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };

pub const Options = struct {
    context: usize,
    drafts: bool, // read by the host; this family has no head, so it drafts by copies only
    segments: usize = 1,
    slide: bool = false,
    lanes: u32 = 1,
    lanes_fixed: bool = false,
    room: ?Room = null,
    grant: u64 = 0, // the host's room before the load, net of its reserve (0: unknown, budget unchecked); decision M10
    kv8: bool = false, // --kv-cache fp8: K and V pages as e4m3 rows with an fp32 scale each; the index keys stay bf16 (docs/fp8-kv.md)
    drafter: ?[]const u8 = null, // --drafter: the dspark draft ("dspark": the checkpoint's own folder, else a folder); every rank gets it
    drafter_bits: u8 = 4, // --drafter-bits: read and ignored (the draft is mxfp8 as shipped)
};

/// The smallest room among the other ranks and whose it is.
pub const Peer = struct { bytes: u64, rank: usize };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize = 0, // every stream is paged: its table row is the layout's fixed part
    free_streams: u32 = iface.decisions.free_streams,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    layout: ?kv.Layout = null,
    peer_room: ?Peer = null,
};

/// What `attach` builds on and `open` owns: the page state of rank 0 and, past `open`, its engine, link and lanes.
pub const Owned = struct {
    gpa: std.mem.Allocator,
    layout: kv.Layout,
    max_pages: usize,
    window: u64 = window_rows,
    pages: ?*pages.Pages = null,
    up: bool = false, // the engine below exists (a test builds the page state alone)
    group: cuda.tp_group.Group = undefined, // the ranks' communicator; two ranks add RoCE writes for the main stream
    e: *engine.Engine = undefined,
    wire: engine_dev.NcclWire = undefined,
    link: follow.Link = undefined,
    bits: engine_dev.Bits = undefined,
    io: std.Io = undefined,
    lanes: backend.Lanes = undefined,
    draft_dir: ?[]u8 = null, // the dspark folder in use (gpa-owned), null: no draft
    draft_cfg: dspark_config.Config = dspark_config.flash,
    draft_stream_bytes: usize = 0, // device bytes a stream holds for the draft on a rank (ring and pass scratch)
};

/// Slots for `context` plus a verify window, whole pages: the longest sequence a stream holds.
pub fn capacity(context: usize) usize {
    return std.mem.alignForward(usize, context + window_rows, pool.page_tokens);
}

/// Bytes this rank's weights take, for the host's admission: the plan's exact slot bytes, else the whole checkpoint.
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64) u64 {
    const p = ranks.place(&ranks.environment) catch return total;
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return total;
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24)) catch return total;
    const c = config.parse(a, text) catch return total;
    var cat = load.Catalog.open(a, io, dir) catch return total;
    defer cat.deinit();
    var plan = load.Plan.build(a, &cat, &c, shard.Share.of(&c, p.rank, p.world) catch return total, .{}) catch return total;
    defer plan.deinit();
    return plan.slotBytes();
}

/// The pool behind the engine: `streams` paged sequences and `count` pages, made now (decision M1: all streams page).
pub fn attach(p: *anyopaque, streams: u32, count: usize) !void {
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.pages) |old| {
        own.lanes.pages = null;
        old.destroy();
        own.pages = null;
    }
    const made = try pages.Pages.create(own.gpa, own.layout, streams, count, own.max_pages, own.window);
    errdefer made.destroy();
    if (own.up) { // the arena and tables on every rank first, then rank 0's pushes reach its own device table
        try own.lanes.x.attach(streams, count);
        made.tables.upload = own.e.uploadHook();
        own.lanes.pages = made;
        try own.lanes.x.idle(); // no stream yet: the other ranks sleep on their lines until the first
    }
    own.pages = made;
}

/// The prompt cache's limits from the host's plan: pages and snapshot records (0 and 0: off).
pub fn cache_limit(p: *anyopaque, count: usize, snaps: usize) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.pages) |pg| pg.limit(count, snaps);
}

/// Whether a request's caches fit now (the lane host's admission, on its engine thread).
pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const own: *Owned = @ptrCast(@alignCast(p));
    const pg = own.pages orelse return false;
    if (!pg.fits(prompt, max_tokens)) return false;
    return !own.up or own.e.floorAdmits(); // the memory floor's target on every rank (alone, the host admits anyway)
}

/// The words of the last budget refusal (single-threaded start-up), for `explain`.
var detail_buf: [1024]u8 = undefined;
var detail_len: usize = 0;

/// What the host-side checks of `open` conclude for this rank at `context`: its place, plan and the budget's verdict.
pub const Admission = struct { place: ranks.Place, weights: u64, verdict: ?budget.Verdict, fingerprint_weights: [32]u8 };

/// What the dspark draft adds to a rank's budget: weights, module, absorb chunk, ring snapshots, a ring a stream.
pub fn draftCost(dcfg: *const dspark_config.Config, share: shard.Share) dspark_config.Cost {
    var cost = dspark_config.cost(dcfg, share, window_rows, prompt_rows, false);
    cost.runtime += @as(u64, backend.max_snaps) * dspark_config.ringBytes(dcfg, share, window_rows);
    return cost;
}

pub fn admission(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const config.Config, p: ranks.Place, o: Options) !Admission {
    return admissionWith(gpa, io, dir, c, p, o, null);
}

/// `admission` with the dspark draft's cost (spent before any KV) in the budget.
pub fn admissionWith(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const config.Config, p: ranks.Place, o: Options, draft: ?dspark_config.Cost) !Admission {
    var cat = try load.Catalog.open(gpa, io, dir);
    defer cat.deinit();
    const share = try shard.Share.of(c, p.rank, p.world);
    var plan = try load.Plan.build(gpa, &cat, c, share, .{});
    defer plan.deinit();
    const dims = (try pool.Dims.of(c, share)).withKv8(o.kv8);
    var out: Admission = .{ .place = p, .weights = plan.slotBytes(), .verdict = null, .fingerprint_weights = try weights_device.digest(gpa, &cat) };
    if (o.grant == 0) return out;
    const terms = try promptTerms(scoreTerms(if (draft) |dcost| budget.withDraft(budget.hosted(), dcost) else budget.hosted(), dims), c, share);
    out.verdict = budget.check(dims, .{ .available = o.grant, .weights = out.weights, .terms = terms, .lanes = o.lanes, .lanes_fixed = o.lanes_fixed, .window = window_rows, .max_positions = c.max_positions }, o.context);
    return out;
}

/// `t` with the block-score scratch the forward makes (forward_exec.Buffers' bmax): its rows, this rank's kv heads.
pub fn scoreTerms(t: budget.Terms, dims: pool.Dims) budget.Terms {
    var u = t;
    u.prefill_rows = iface.max_rows;
    u.index_heads = dims.kv_heads;
    return u;
}

/// `t` with the prompt experts' x3gm scratch (forward_exec.Buffers' gm) on its workspace, at this rank's share.
pub fn promptTerms(t: budget.Terms, c: *const config.Config, share: shard.Share) !budget.Terms {
    const g = try forward_plan.Geometry.of(c, share, c.layers, iface.max_rows);
    var u = t;
    u.workspace += forward_exec.promptExpertBytes(&g);
    return u;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = kernels; // our own kernels: a captured set is not used
    const p = try ranks.admit(&ranks.environment);
    // the ranks' lines before any refusal (cuda/tp_watch.zig): a rank that refuses, fails or dies stops every rank
    cuda.tp_watch.exitCoherently(); // every stop exits 70: the recipe's watchdog restarts the pair
    if (p.world > 1) try cuda.tp_watch.start(io, p.rank, p.world);
    var problem: []const u8 = "";
    var set = ranks.Settings.read(&ranks.environment, &problem) catch |err| {
        std.log.err("{s}", .{problem});
        return err;
    };
    set.kv8 = o.kv8; // the ranks' fingerprint carries it: a rank started without --kv-cache fp8 stops the others
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    const c = try config.parse(gpa, text);
    var draft_dir: ?[]u8 = null;
    errdefer if (draft_dir) |dd| gpa.free(dd);
    var draft_cfg: dspark_config.Config = dspark_config.flash;
    var draft_cost: ?dspark_config.Cost = null;
    if (o.drafter) |value| if (o.drafts) { // --no-drafts leaves the draft unloaded
        draft_dir = dspark_folder.resolve(gpa, io, dir, value) catch |err| switch (err) {
            error.NoDraft => return refuse(dspark_folder.words(&detail_buf, dir, value), error.DrafterRefused),
            else => return err,
        };
        draft_cfg = try loadDraftConfig(gpa, io, draft_dir.?, &c, p);
        draft_cost = draftCost(&draft_cfg, try shard.Share.of(&c, p.rank, p.world));
    };
    const a = try admissionWith(gpa, io, dir, &c, p, o, draft_cost);
    if (a.verdict) |v| switch (v) {
        .refused => |r| {
            const words = std.fmt.bufPrint(&detail_buf, "{f}", .{r}) catch detail_buf[0..0];
            detail_len = words.len;
            return error.BudgetRefused;
        },
        .ok => |cap| std.log.info("minimax budget: {d} tokens a stream, {d} pages, {d:.2} GiB of KV room, {s} K/V (the host's plan has the last word)", .{ cap.context, cap.pages, @as(f64, @floatFromInt(cap.pool_bytes)) / (1 << 30), if (o.kv8) "fp8" else "bf16" }),
    };
    return openRank(gpa, io, ctx, dir, &c, o, p, set, draft_dir, draft_cfg) catch |err| {
        cuda.tp_watch.fail(@errorName(err)); // the other ranks stop with this rank's error
        return err;
    };
}

/// `open` once the ranks' lines are up: NCCL, the engine and the link, then a follower's loop or rank 0's lanes.
fn openRank(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, c: *const config.Config, o: Options, p: ranks.Place, set: ranks.Settings, draft_dir: ?[]u8, draft_cfg: dspark_config.Config) !Loaded {
    const share = try shard.Share.of(c, p.rank, p.world);
    const dims = (try pool.Dims.of(c, share)).withKv8(o.kv8);
    const cap = capacity(o.context);
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .layout = dims.layout(iface.maxPages(cap)), .max_pages = iface.maxPages(cap), .io = io, .draft_dir = draft_dir, .draft_cfg = draft_cfg };
    errdefer if (own.draft_dir) |dd| gpa.free(dd);
    engine_dev.setNcclDefaults();
    var lib = try cuda.nccl.Library.open();
    errdefer lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId"); // NCCL_COMM_ID: the same id on every rank
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(p.world), id, @intCast(p.rank)), "ncclCommInitRank");
    own.group = .{ .lib = lib, .comm = comm, .rank = p.rank, .world = p.world, .coherent = true }; // RoCE marker and fail-both: opt-in
    errdefer _ = own.group.lib.api.ncclCommDestroy(own.group.comm);
    if (p.world > 1) try engine_dev.warmPeers(ctx.d, &own.group);
    own.e = try engine.Engine.init(gpa, io, ctx.d, ctx, dir, &own.group.lib, own.group.comm, p.rank, p.world, .{ .capacity = cap, .settings = set, .drafter = draft_dir });
    errdefer own.e.deinit();
    const stream = own.e.p.stream.?;
    const roce_bytes = @as(usize, if (set.roce_kb == 0) 4096 else set.roce_kb) * 1024;
    try own.group.openRoce(gpa, io, ctx.d, stream, roce_bytes); // both ranks bring it up or neither does
    errdefer if (own.group.fast) |f| f.deinit();
    own.e.setGroup(&own.group);
    own.wire = try engine_dev.NcclWire.init(ctx.d, &own.group.lib, own.group.comm, stream, follow.Link.capacityFor(cap));
    errdefer own.wire.deinit();
    own.link = try follow.Link.init(gpa, own.wire.wire(), engine_dev.hooksWith(&own.io), cap);
    errdefer own.link.deinit();
    // Rank 0 admits by the smallest room of the ranks.
    const rooms = try gpa.alloc(u64, p.world);
    defer gpa.free(rooms);
    const mine: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    try own.e.p.comm.gatherBytes(std.mem.asBytes(&mine), std.mem.sliceAsBytes(rooms));
    var peer: ?Peer = null;
    for (rooms[1..], 1..) |r, q| if (peer == null or r < peer.?.bytes) {
        peer = .{ .bytes = r, .rank = q };
    };
    // every rank loaded and agreed (the fingerprint) before the first step; the digest names what they agreed on
    if (p.world > 1) try cuda.tp_watch.ready(io, own.e.checkpoint_digest);
    if (p.rank != 0) { // a follower never serves: it runs rank 0's steps, then leaves the process
        follow.follow(own.e, &own.link) catch |err| {
            std.log.err("rank {d}: following rank 0 stopped ({s})", .{ p.rank, @errorName(err) });
            std.c._exit(cuda.tp_watch.coherent_exit); // follow told the other ranks; the stream may hold kernels waiting on a gone rank
        };
        std.log.info("rank {d}: rank 0 ended; leaving", .{p.rank});
        std.process.exit(0);
    }
    own.link.asleep = cuda.tp_watch.open(); // the other ranks begin asleep (follow): the first step wakes them
    own.bits = .{ .gpa = gpa, .words = (c.vocab + 31) / 32 };
    own.lanes = .{ .gpa = gpa, .x = .{ .e = own.e, .link = &own.link }, .masks = own.bits.interface() };
    if (own.draft_dir != null) {
        own.draft_stream_bytes = @intCast(draftCost(&own.draft_cfg, share).per_lane);
        own.lanes.tap_ids = own.draft_cfg.tapIds(); // the layers the forward taps for the draft
        own.lanes.tap_hidden = own.draft_cfg.hidden;
    }
    own.up = true;
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(), // (with a draft: no copy branch beside a drafted window, which would be a tree)
        .rows = window_rows,
        .stream_bytes = 0, // every stream pages: its table row is the layout's fixed part
        .free_streams = iface.decisions.free_streams,
        .ctx = own,
        .deinit = close,
        .layout = own.layout,
        .peer_room = peer,
    };
}

fn close(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    own.lanes.stopped = true; // no snapshot step may follow the end step
    own.lanes.x.end() catch {}; // the other ranks leave on the end step, so their lines close on purpose from here
    own.lanes.deinit();
    if (own.pages) |pg| pg.destroy();
    own.link.deinit();
    own.wire.deinit();
    own.bits.deinit();
    own.e.deinit();
    own.group.close();
    if (own.draft_dir) |dd| own.gpa.free(dd);
    own.gpa.destroy(own);
}

/// The words of the last refusal at start-up, then the error: `explain` shows them.
fn refuse(words: []const u8, err: anyerror) anyerror {
    detail_len = words.len;
    if (words.ptr != &detail_buf) @memcpy(detail_buf[0..words.len], words);
    return err;
}

/// The draft's config.json in `dir`, parsed and checked against the model and the ranks; refusals in words.
fn loadDraftConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const config.Config, p: ranks.Place) !dspark_config.Config {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    const dcfg = dspark_config.parse(gpa, text) catch |err| {
        const words = std.fmt.bufPrint(&detail_buf, "--drafter: {s} is not a draft this engine runs ({s}): it takes the pack's DSparkMiniMaxDraftModel", .{ dir, @errorName(err) }) catch detail_buf[0..0];
        return refuse(words, error.DrafterRefused);
    };
    if (dspark_config.admit(&dcfg, c, .{ .world = p.world, .window_rows = window_rows })) |why| return refuse(why.text(), error.DrafterRefused);
    return dcfg;
}

/// --drafter on the lane host: the dspark draft (loaded with the engine on every rank) behind the core's Drafted.
pub fn openDrafter(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8) !struct { drafter: lanes.drafter.Drafter, ctx: *anyopaque, close: *const fn (*anyopaque) void, reserve: *const fn (*anyopaque, u32) anyerror!void } {
    _ = io;
    _ = dir; // the folder was resolved and loaded in `open`, on every rank
    _ = bits;
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.draft_dir == null) return error.NoDraft;
    const d = try dspark_drafter.Draft.init(gpa, &own.lanes, own.draft_cfg, .{ .stream_bytes = own.draft_stream_bytes });
    return .{ .drafter = d.drafter(), .ctx = d, .close = closeDrafter, .reserve = reserveDrafter };
}

fn reserveDrafter(_: *anyopaque, _: u32) anyerror!void {} // the rings are made at `attach`, for the host's streams

fn closeDrafter(p: *anyopaque) void {
    const d: *dspark_drafter.Draft = @ptrCast(@alignCast(p));
    d.deinit();
}

/// The server's words for this family's refusals.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    inline for (@typeInfo(ranks.Error).error_set.error_names.?) |name| {
        if (err == @field(anyerror, name)) return ranks.refusal(@field(ranks.Error, name));
    }
    return switch (err) {
        error.EngineNotBuilt => "MiniMax-M3.1-Flash is registered and its checkpoint, ranks and memory budget were checked, but this build has no forward pass for it yet (the engine, prefill and lanes backend are not in this tree)",
        error.ForwardNotBuilt => "MiniMax-M3.1-Flash's ranks agreed and its engine loop is built, but this build has no forward or prompt pass to run on it (forward.zig and prefill.zig are not in this tree)",
        error.VisionUnsupported => "MiniMax-M3.1-Flash on CUDA is text only: this pack has no vision tower, send the request without images",
        error.RankLost => "another rank of MiniMax-M3.1-Flash stopped: every rank ends, start them again",
        error.BudgetRefused, error.DrafterRefused => detail_buf[0..detail_len],
        error.MemoryFloor => "a rank's MemAvailable less TF_MM3_FLOOR_TRANSIENT_GIB is under TF_MM3_FLOOR_HARD_GIB (rank 0's log names the rank): the prompt stopped before its next forward; free memory on that host, or serve a smaller --context or --parallel",
        error.RankMismatch => "the ranks' settings differ (each rank's log names the one): start them alike",
        error.BadRank => "TF_MM3_WORLD is larger than the checkpoint can be cut for (4 kv heads, 128-unit widths)",
        else => null,
    };
}

comptime {
    iface.conformsNative(@This());
}

test "capacity keeps a verify window past the context, in whole pages" {
    try std.testing.expectEqual(@as(usize, 65664), capacity(65536));
    try std.testing.expectEqual(@as(usize, 128), capacity(1));
    try std.testing.expect(capacity(100000) % pool.page_tokens == 0 and capacity(100000) >= 100000 + window_rows);
}

test "attach makes the pool, fits asks it, the cache limit reaches the store, and a family without pages refuses" {
    const gpa = std.testing.allocator;
    const l: kv.Layout = .{ .fixed_bytes = 4096, .token_bytes = 1024, .page_tokens = pool.page_tokens, .snap_bytes = pool.snap_bytes, .all_paged = true };
    var own: Owned = .{ .gpa = gpa, .layout = l, .max_pages = 64 };
    try std.testing.expect(!fits(&own, 10, 10));
    try attach(&own, 2, 100);
    defer own.pages.?.destroy();
    try std.testing.expect(fits(&own, 1000, 100));
    try std.testing.expect(!fits(&own, 100 * 128, 0));
    cache_limit(&own, 40, 8);
    try attach(&own, 3, 50);
    try std.testing.expectEqual(@as(usize, 50), own.pages.?.pages.total());
}

test "explain words every named refusal and leaves foreign errors to the host" {
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoRank).?, "TF_MM3_RANK") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoCommId).?, "NCCL_COMM_ID") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.EngineNotBuilt).?, "no forward pass") != null);
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
}

test "the draft's cost: its weights, the module and absorb chunk with eight ring snapshots, and a ring a stream" {
    const share = try shard.Share.of(&config.flash, 0, 2);
    const base = dspark_config.cost(&dspark_config.flash, share, window_rows, prompt_rows, false);
    const cost = draftCost(&dspark_config.flash, share);
    try std.testing.expectEqual(base.weights, cost.weights);
    try std.testing.expectEqual(base.per_lane, cost.per_lane);
    try std.testing.expectEqual(base.runtime + 8 * 21_053_440, cost.runtime);
    const terms = budget.withDraft(budget.hosted(), cost);
    try std.testing.expectEqual(cost.weights + cost.runtime + 3 * cost.per_lane, budget.draftBytes(terms, 3));
    try std.testing.expect(budget.draftBytes(budget.hosted(), 3) == 0);
}

test "the budget prices the block-score scratch the forward makes: its rows and this rank's kv heads" {
    const share = try shard.Share.of(&config.flash, 0, 2);
    const dims = try pool.Dims.of(&config.flash, share);
    const t = scoreTerms(budget.hosted(), dims);
    const context: u64 = 131072;
    try std.testing.expectEqual(@as(u64, iface.max_rows) * dims.kv_heads * (context / pool.page_tokens) * 4, budget.scoreBytes(t, context));
    try std.testing.expect(budget.scoreBytes(t, context) < budget.scoreBytes(budget.hosted(), context));
}

test "the budget prices the prompt experts' scratch the forward makes at this rank's share" {
    const share = try shard.Share.of(&config.flash, 0, 2);
    const g = try forward_plan.Geometry.of(&config.flash, share, config.flash.layers, iface.max_rows);
    const t = try promptTerms(budget.hosted(), &config.flash, share);
    try std.testing.expectEqual(budget.hosted().workspace + forward_exec.promptExpertBytes(&g), t.workspace);
    try std.testing.expect(forward_exec.promptExpertBytes(&g) > 400_000_000); // 2,048 rows, top 4, width 6144 and 1536
}

test "a floor refusal is worded for the server" {
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.MemoryFloor).?, "TF_MM3_FLOOR_HARD_GIB") != null);
}

test "a refused draft is worded for the server, and --drafter is taken" {
    const words = "--drafter: nothing here";
    try std.testing.expectEqual(@as(anyerror, error.DrafterRefused), refuse(words, error.DrafterRefused));
    try std.testing.expectEqualStrings(words, explain(null, error.DrafterRefused).?);
    try std.testing.expect(takes_drafter and @hasDecl(@This(), "openDrafter"));
}

test "the family's declarations are the ones the host reads" {
    try std.testing.expectEqualStrings("minimax_m3_vl", model_type);
    try std.testing.expectEqualStrings("exl3", formats[0]);
    try std.testing.expect(owns_prompt_cache and explains_open);
    try std.testing.expectEqual(@as(u32, 2048), prompt_rows);
}
