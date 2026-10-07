//! M4 on the GPU: DeepSeek-V4.1's engine for the native server (tensorfold-dsv41, `native_engines`), the lanes engine
//! (core/lanes + draft/lanes.zig) over the GPU target and DSpark's GPU pass in a LaneHost on rank 0; the other ranks
//! run `tf-dsv41-m1 follow PACK ASSETS`, which loads the same rank and replays rank 0's forward operations until the
//! server stops. One slot (GpuTarget's): the host serves one stream at a time.
//!
//! TF_DSV41_ASSETS: the assets dir (model.zig). TF_DSV41_LAYERS=<lo>-<hi>: a backbone prefix only (a pod gate's model). TF_DSV41_TRACE_TOKENS=<file>: every finished reply's prompt and reply
//! ids appended as a JSON line, for the gate that `tf-dsv41-m1 generate` replays the same prompt through the same
//! engine without HTTP and must emit the same ids.

const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const model = @import("model.zig");
const ln = @import("draft/lanes.zig");
const drive = @import("draft/drive.zig");
const costs_mod = @import("draft/costs.zig");
const tree = @import("draft/tree.zig");
const calib_env = @import("draft/calib_env.zig");
const m3 = @import("m3.zig");
const ph = @import("phases.zig");
const timed = @import("phases_timed.zig");
const Allocator = std.mem.Allocator;

pub const backends: []const []const u8 = &.{"cuda"};
pub const families: []const api.Family = &.{.{ .model_type = "deepseek_v41", .formats = &.{"exl3"} }};

pub fn chip(_: Allocator) ?[]const u8 {
    return null;
}

/// The engine for `o.dir`, or null with `problem` set (the native_engines contract).
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (!std.mem.eql(u8, o.model_type, "deepseek_v41")) {
        problem.* = try std.fmt.allocPrint(a, "this build serves deepseek_v41 checkpoints only, not {s}", .{o.model_type});
        return null;
    }
    const assets = std.mem.span(std.c.getenv("TF_DSV41_ASSETS") orelse {
        problem.* = "TF_DSV41_ASSETS: the dir with rope.bin, engram-host.bin and aot/ (model.zig)";
        return null;
    });
    const s = Served.open(gpa, io, o.dir, assets, o.drafts, o.context) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the DeepSeek-V4.1 CUDA engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
    return .{ .engine = s.engine(), .close = Served.close, .ctx = s };
}

pub const Served = struct {
    gpa: Allocator,
    io: std.Io,
    m: *model.Model,
    x: *ln.Lanes,
    nd: drive.NoDraft = .{},
    costs: costs_mod.Costs,
    arena: std.heap.ArenaAllocator,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    rec: Recorder,
    /// TF_DSV41_PHASES (phases.zig): the round's phases around the target and the pass
    phases: ?*ph.Phases = null,
    tt: timed.TimedTarget = undefined,
    tpass: timed.TimedPass = undefined,

    /// Rank 0's engine over a loaded model (`drafts`: DSpark's pass, else serial windows through the same lanes).
    pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, assets: []const u8, drafts: bool, context: ?i64) !*Served {
        const m = try model.Model.open(gpa, io, dir, assets, .{ .drafts = drafts, .own_prefill = model.ownPrefillEnv(), .layers = try model.layersEnv(gpa) });
        errdefer m.close();
        if (m.rank() != 0) return error.NotRankZero; // the other ranks run `tf-dsv41-m1 follow`
        const s = try gpa.create(Served);
        errdefer gpa.destroy(s);
        s.gpa = gpa;
        s.io = io;
        s.m = m;
        s.nd = .{};
        // draft pricing: prod's cached calibration (TF_DSV41_CALIB / _CALIB_DIR / _CALIB_FILE: Python's calib-*.json,
        // shared as rank 0 shares it) and Python's depth knobs; without an entry for this engine, today's defaults
        const cal = try calib_env.load(gpa, io, calib_env.Knobs.fromEnv(), .{ .slots = 1, .world = m.f.comm.world(), .dspark = m.pass != null, .context = @intCast(@min(context orelse m.f.opts.limit, m.f.opts.limit)) }, 64);
        defer if (cal.path) |p| gpa.free(p);
        s.costs = cal.costs;
        std.log.info("dsv41 draft costs: {t}{s}{s}, verify 1 / 16 rows {d:.3} / {d:.3} ms, pass {d:.3} ms", .{ cal.source, if (cal.path != null) " " else "", cal.path orelse "", s.costs.windowMs(1), s.costs.windowMs(16), s.costs.draft });
        var pass = if (m.pass) |p| p.pass() else s.nd.pass();
        var tgt = m.gt.target();
        s.phases = try ph.Phases.fromEnv(gpa);
        if (s.phases) |x| {
            x.sync = .{ .ctx = m, .run = syncStream };
            m.f.phases = x;
            s.tt = .{ .inner = tgt, .p = x, .gpa = gpa, .io = io };
            s.tpass = .{ .inner = pass, .p = x };
            tgt = s.tt.target();
            pass = s.tpass.pass();
        }
        // draft trees (TF_DSV41_TREE / _DUP_MS / _ROWS, Python's knobs; off by default): first-position siblings
        const trees = if (m.pass != null) try tree.Settings.fromEnv() else tree.Settings{};
        s.x = try ln.Lanes.init(gpa, tgt, pass, s.costs, .{ .shape = m3.shapeOf(&m.f), .slots = 1, .siblings = trees.siblings, .sib_rows = trees.sib_rows, .dup = trees.dup, .depth = try calib_env.depthSettings(&calib_env.env) });
        s.arena = std.heap.ArenaAllocator.init(gpa);
        s.cfg = try lanes.Config.init(gpa, try s.x.model(s.arena.allocator(), 64), 16, 15);
        // drafts sized from the cost tables, not the wall clock: a request drafts (and so verifies) the same whatever the
        // load or the requests before it, so HTTP == CLI replay holds (TF_DSV41_DRAFT_TIMED=1: the measured pricing)
        s.cfg.timed = if (std.c.getenv("TF_DSV41_DRAFT_TIMED")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        s.clock = .{ .io = io };
        s.core = lanes.Engine.init(gpa, &s.cfg, s.x.backend(), s.clock.clock());
        s.x.attach(&s.core);
        const window: u32 = @intCast(@min(context orelse m.f.opts.limit, m.f.opts.limit));
        s.host = api.LaneHost.init(gpa, io, &s.core, .{ .name = "dsv41-cuda", .lanes = 1, .context_window = window, .prefill_step = m.gt.prefill_rows });
        // the lanes run on LaneHost's thread: the model's CUDA context must be current there
        s.host.on_thread = .{ .ctx = m, .run = bindContext };
        try s.host.start();
        s.rec = Recorder.init(gpa, io, s.host.engine());
        return s;
    }

    fn syncStream(ctx: *anyopaque) anyerror!void {
        const m: *model.Model = @ptrCast(@alignCast(ctx));
        try m.stream.synchronize();
    }

    fn bindContext(ctx: *anyopaque) anyerror!void {
        const m: *model.Model = @ptrCast(@alignCast(ctx));
        try m.ctx.makeCurrent();
    }

    pub fn engine(s: *Served) api.Engine {
        return s.rec.engine();
    }

    pub fn close(ctx: *anyopaque) void {
        const s: *Served = @ptrCast(@alignCast(ctx));
        const gpa = s.gpa;
        s.host.stop();
        s.m.f.stop() catch {}; // the followers' loops end
        if (s.phases) |x| {
            x.flush(gpa, s.io);
            s.m.f.phases = null;
            gpa.destroy(x);
        }
        s.rec.deinit();
        s.core.deinit();
        s.cfg.deinit(gpa);
        s.arena.deinit();
        s.x.deinit();
        s.costs.deinit(gpa);
        s.m.close();
        gpa.destroy(s);
    }
};

/// The engine as served, with each finished reply's prompt and reply ids appended to TF_DSV41_TRACE_TOKENS (when set).
pub const Recorder = struct {
    gpa: Allocator,
    io: std.Io,
    inner: api.Engine,
    path: ?[]const u8,
    mutex: std.Io.Mutex = .init,
    live: std.AutoHashMapUnmanaged(api.Id, *Rec) = .empty,
    /// every recorded line so far (the file is rewritten whole: one reply at a time)
    log: std.ArrayList(u8) = .empty,

    /// `sampling`: the request's own (seed included), so `tf-dsv41-m1 generate` replays a sampled reply as it was asked
    const Rec = struct { r: *Recorder, sink: api.Sink, prompt: []u32, sampling: ?api.Sampling = null, out: std.ArrayList(u32) = .empty, drafted: u64 = 0 };

    pub fn init(gpa: Allocator, io: std.Io, inner: api.Engine) Recorder {
        return .{ .gpa = gpa, .io = io, .inner = inner, .path = if (std.c.getenv("TF_DSV41_TRACE_TOKENS")) |p| std.mem.span(p) else null };
    }

    pub fn deinit(r: *Recorder) void {
        var it = r.live.valueIterator();
        while (it.next()) |x| r.drop(x.*);
        r.live.deinit(r.gpa);
        r.log.deinit(r.gpa);
    }

    fn drop(r: *Recorder, x: *Rec) void {
        r.gpa.free(x.prompt);
        x.out.deinit(r.gpa);
        r.gpa.destroy(x);
    }

    pub fn engine(r: *Recorder) api.Engine {
        if (r.path == null) return r.inner;
        return .{ .ctx = r, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn self(ctx: *anyopaque) *Recorder {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(ctx: *anyopaque) api.Info {
        return self(ctx).inner.info();
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        self(ctx).inner.cancel(id);
    }
    fn status(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
        self(ctx).inner.status(out, stream_tokens);
    }
    fn memory(ctx: *anyopaque, reset_peak: bool) ?api.Memory {
        return self(ctx).inner.memory(reset_peak);
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const r = self(ctx);
        const x = r.gpa.create(Rec) catch return error.Busy;
        x.* = .{ .r = r, .sink = sink, .sampling = request.sampling, .prompt = r.gpa.dupe(u32, request.prompt) catch {
            r.gpa.destroy(x);
            return error.Busy;
        } };
        {
            r.mutex.lockUncancelable(r.io);
            defer r.mutex.unlock(r.io);
            r.live.put(r.gpa, id, x) catch {
                r.drop(x);
                return error.Busy;
            };
        }
        return r.inner.submit(id, request, .{ .ctx = x, .event = event });
    }

    fn event(ctx: *anyopaque, id: api.Id, ev: *const api.Event) void {
        const x: *Rec = @ptrCast(@alignCast(ctx));
        const r = x.r;
        switch (ev.*) {
            .tokens => |t| x.out.appendSlice(r.gpa, t) catch {},
            .finished => |fin| {
                x.drafted = fin.stats.drafted;
                r.write(x);
            },
            else => {},
        }
        x.sink.event(x.sink.ctx, id, ev);
        if (ev.* == .finished) {
            r.mutex.lockUncancelable(r.io);
            defer r.mutex.unlock(r.io);
            _ = r.live.remove(id);
            r.drop(x);
        }
    }

    fn write(r: *Recorder, x: *const Rec) void {
        var line: std.Io.Writer.Allocating = .init(r.gpa);
        defer line.deinit();
        std.json.Stringify.value(.{ .prompt = x.prompt, .tokens = x.out.items, .drafted = x.drafted, .sampling = x.sampling }, .{}, &line.writer) catch return;
        line.writer.writeByte('\n') catch return;
        r.log.appendSlice(r.gpa, line.written()) catch return;
        std.Io.Dir.cwd().writeFile(r.io, .{ .sub_path = r.path.?, .data = r.log.items }) catch {};
    }
};
