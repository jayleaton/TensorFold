//! MiaAI-Lab's Flash Next engine (flashnext_mia) for qwen4_exp checkpoints, opened only with TF_FLASHNEXT_ENGINE=mia.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const mia = @import("flashnext_mia");
const lane_backend = @import("cuda_bound.zig");
const windows = @import("cuda_window.zig");
const setting = @import("cuda_device.zig").setting;
const Allocator = std.mem.Allocator;
const F = mia.native;

/// The context a lane thread needs current: the lane host steps rounds on its own thread, CUDA binds per thread.
threadlocal var bound: ?*const cuda.Context = null;

/// Whether `o` asks for this engine: a qwen4_exp checkpoint with TF_FLASHNEXT_ENGINE=mia (unset: upstream's flashnext).
pub fn wanted(o: api.Open) bool {
    return std.mem.eql(u8, o.model_type, F.model_type) and mia.selected();
}

/// One loaded model behind the lane host: the family, its lane core and its kept prompt states.
const Host = struct {
    gpa: Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    family: *anyopaque,
    release: *const fn (*anyopaque) void,
    follow_fn: ?*const fn (*anyopaque) anyerror!void = null, // rank 1's loop (two ranks)
    inner: lanes.backend.Backend,
    vtable: lanes.backend.Backend.VTable,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    store: ?api.prompt_cache.Store = null, // kept prompt states when the family's Loaded.cache has a budget
    store_vt: api.prompt_cache.Snapshots.VTable = undefined,

    fn close(p: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(p));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        h.ctx.makeCurrent() catch {};
        if (h.store) |*st| st.deinit(); // its states go back to the family before the family goes
        h.release(h.family);
        h.ctx.deinit();
        h.driver.close();
        h.gpa.destroy(h);
    }

    fn halt(p: *anyopaque, reason: []const u8) void {
        const h: *Host = @ptrCast(@alignCast(p));
        h.host.halt(reason);
    }

    fn follow(p: *anyopaque) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(p));
        try h.ctx.makeCurrent();
        return h.follow_fn.?(h.family);
    }

    pub fn bind(p: *anyopaque) *Host {
        const h: *Host = @ptrCast(@alignCast(p));
        if (bound != &h.ctx) {
            h.ctx.makeCurrent() catch |e| std.log.err("cuCtxSetCurrent on the lane thread: {s}", .{@errorName(e)});
            bound = &h.ctx;
        }
        return h;
    }

    /// The family's backend, each entry it gives called with the context current (cuda_bound.zig).
    fn backend(h: *Host) lanes.backend.Backend {
        h.vtable = lane_backend.vtable(Host, h.inner.vtable.*);
        return .{ .ptr = h, .round_observer = h.inner.round_observer, .vtable = &h.vtable, .first_before_draft = h.inner.first_before_draft, .penalties = h.inner.penalties, .chunk_invariant = h.inner.chunk_invariant };
    }
};

/// Frees a host whose family never opened or is already released: its context, driver and memory.
fn discard(h: *Host) void {
    bound = null;
    h.ctx.deinit();
    h.driver.close();
    h.gpa.destroy(h);
}

/// The kernel set: TENSORFOLD_CUDA_KERNELS, else share/tensorfold/cuda/sm<capability> beside the binary.
fn kernelDir(a: Allocator, io: std.Io, capability: u32) ![]const u8 {
    if (std.c.getenv("TENSORFOLD_CUDA_KERNELS")) |dir| return a.dupe(u8, std.mem.span(dir));
    const exe = try std.process.executableDirPathAlloc(io, a);
    return std.fs.path.join(a, &.{ exe, "..", "share", "tensorfold", "cuda", try std.fmt.allocPrint(a, "sm{d}", .{capability}) });
}

/// The engine for `o.dir` on this family, or null with `problem` set.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    // a window past config.json's with YaRN: the family's modelWindow says how far
    const native = F.modelWindow(a, io, o.dir);
    const window = windows.contextWindow(o.context, native, F.default_context) catch {
        problem.* = try std.fmt.allocPrint(a, "--context {d} exceeds this model's {d}-token window", .{ o.context orelse 0, native });
        return null;
    };
    if (o.vision != null) {
        problem.* = "--vision: TF_FLASHNEXT_ENGINE=mia serves text only; unset it to serve images with upstream's engine";
        return null;
    }
    if (o.tp != 1 and o.tp != 2) {
        problem.* = try std.fmt.allocPrint(a, "--tp {d}: TF_FLASHNEXT_ENGINE=mia runs on one or two ranks", .{o.tp});
        return null;
    }
    const device = try setting(a, o.device, "TF_CUDA_DEVICE", 0, problem) orelse return null;
    const h = try gpa.create(Host);
    h.gpa = gpa;
    h.driver = cuda.Driver.open() catch |e| {
        problem.* = try std.fmt.allocPrint(a, "no CUDA driver ({s})", .{@errorName(e)});
        gpa.destroy(h); // a null return runs no errdefer
        return null;
    };
    h.ctx = cuda.Context.init(&h.driver, @intCast(device)) catch |e| {
        h.driver.close();
        gpa.destroy(h);
        return e;
    };
    bound = &h.ctx;
    errdefer discard(h);
    const kernels = try kernelDir(a, io, try h.ctx.capability());
    var fo: F.Options = .{ .context = @intCast(window), .drafts = o.drafts };
    fo.kv_dtype = if (o.kv8) .fp8 else .bf16; // --kv-cache fp8 (her --kv-dtype fp8)
    fo.parallel = o.lanes; // --parallel: streams one shared forward holds
    fo.prompt_cache_gib = o.prompt_cache_gib;
    fo.tp = o.tp;
    fo.rank = o.rank;
    fo.master = o.master;
    fo.master_port = o.master_port;
    const loaded = F.open(gpa, io, &h.ctx, o.dir, kernels, fo) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "TF_FLASHNEXT_ENGINE=mia cannot load {s} with kernels {s} ({s})", .{ o.dir, kernels, @errorName(e) });
        discard(h); // a null return runs no errdefer
        return null;
    };
    errdefer loaded.deinit(loaded.ctx);
    h.family = loaded.ctx;
    h.release = loaded.deinit;
    h.follow_fn = loaded.follow;
    h.inner = loaded.backend;
    h.store = null;
    h.cfg = try lanes.Config.init(gpa, loaded.facts, loaded.rows, loaded.rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = F.prefill_step });
    if (loaded.cache) |c| if (c.budget > 0) {
        // the family keeps and restores the states; the store picks them (exact reuse, core/prompt_cache.zig)
        h.store_vt = .{ .bytes = c.bytes, .save = c.save, .restore = c.restore, .drop = c.drop };
        const rules: api.prompt_cache.Rules = .{ .lookahead = F.cache_rules.lookahead, .planned = F.cache_rules.planned };
        h.store = api.prompt_cache.Store.init(gpa, .{ .ptr = c.ptr, .vtable = &h.store_vt }, rules, c.budget);
        h.host.cache = &h.store.?;
    };
    errdefer if (h.store) |*st| st.deinit();
    try h.host.start();
    const follower: ?*const fn (*anyopaque) anyerror!void = if (h.follow_fn != null) Host.follow else null;
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h, .halt = Host.halt, .follow = follower };
}
