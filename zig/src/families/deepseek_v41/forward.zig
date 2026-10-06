//! M2's forward: a slot's decode windows over the whole model, block.zig's program (launches + glue) run by run.zig
//! on this rank, the exchanges over the TP Collective. One slot, contiguous state (the prod path's: no page tables);
//! the window's calls are emitted once a row bucket and reused (the same launches every window of that size: graphs
//! capture them in R4). Glue steps whose kernel is not in dsv41_kernels yet fail with GlueNotBuilt (named), so a run
//! says exactly what is missing.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const calls = @import("calls.zig");
const block = @import("block.zig");
const buffers = @import("buffers.zig");
const run = @import("run.zig");
const Config = @import("config.zig").Config;

pub const Error = error{ GlueNotBuilt, BadWindow };

/// One slot's position, Engram tail and pending window (what keep() commits).
pub const Slot = struct {
    pos: u64 = 0,
    /// the window in flight: its first position, its rows
    pending: ?struct { start: u64, n: u32 } = null,
};

pub const Forward = struct {
    gpa: std.mem.Allocator,
    cfg: *const Config,
    widths: *const block.Widths,
    opts: block.Options,
    runner: *run.Runner,
    comm: tp.collective.Collective,
    /// the window's calls by row count, emitted once (arena-owned)
    arena: std.heap.ArenaAllocator,
    programs: std.AutoHashMapUnmanaged(u32, []const calls.Call) = .empty,
    slot: Slot = .{},
    /// pinned staging of the window's token ids, and their device copy (int64, the embedding step reads them)
    host: ?cuda.HostBuffer = null,
    dev_ids: ?cuda.DeviceBuffer = null,
    /// a bf16 [rows, hidden] send buffer for the exchanges whose partial is not bf16 already (embedding, MoE)
    send: ?cuda.DeviceBuffer = null,
    /// the window's token ids (the embedding step reads them)
    ids: []const u32 = &.{},

    pub fn init(gpa: std.mem.Allocator, cfg: *const Config, widths: *const block.Widths, opts: block.Options, runner: *run.Runner, comm: tp.collective.Collective) Forward {
        return .{ .gpa = gpa, .cfg = cfg, .widths = widths, .opts = opts, .runner = runner, .comm = comm, .arena = std.heap.ArenaAllocator.init(gpa) };
    }

    pub fn deinit(f: *Forward) void {
        if (f.host) |*h| h.free();
        if (f.dev_ids) |*b| b.free();
        if (f.send) |*b| b.free();
        f.programs.deinit(f.gpa);
        f.arena.deinit();
    }

    /// The backbone's layers, then the head.
    fn backbone(f: *Forward, a: std.mem.Allocator) ![]const u32 {
        const ls = try a.alloc(u32, f.cfg.layers);
        for (ls, 0..) |*l, i| l.* = @intCast(i);
        return ls;
    }

    /// The buffer plan of every row bucket the forward runs (bind the runner with it once, before the first window).
    pub fn plan(f: *Forward, buckets: []const u32) !buffers.Plan {
        var p: buffers.Plan = .{ .a = f.arena.allocator() };
        for (buckets) |n| try p.add(try f.program(n));
        return p;
    }

    /// The calls of an `n`-row window (positions are the glue's; the program does not change with them).
    pub fn program(f: *Forward, n: u32) ![]const calls.Call {
        if (f.programs.get(n)) |cs| return cs;
        const a = f.arena.allocator();
        // the emitter's positions only size the indexer's scores at the window's end; the largest the slot reaches
        const cs = try block.emit(a, f.cfg, f.widths, f.opts, try f.backbone(a), n, f.opts.limit - n, true);
        try f.programs.put(f.gpa, n, cs);
        return cs;
    }

    /// One decode window: `ids` at the slot's position. This rank's logits stay in the runner's "w.logits".
    pub fn window(f: *Forward, ids: []const u32) !void {
        if (ids.len == 0 or f.slot.pending != null) return error.BadWindow;
        const cs = try f.program(@intCast(ids.len));
        f.ids = ids;
        f.slot.pending = .{ .start = f.slot.pos, .n = @intCast(ids.len) };
        f.runner.glue = .{ .ctx = f, .run = glueFn };
        try f.runner.window(cs);
    }

    /// Commits the pending window's first row + `accepted` drafts (Python's Forward.keep).
    pub fn keep(f: *Forward, accepted: u32) !void {
        const p = f.slot.pending orelse return error.BadWindow;
        if (accepted >= p.n) return error.BadWindow;
        // the ratio-2 KV sources' carry: fp32 of the accepted row of the window's compressor projection
        const r = f.runner;
        var nb: [64]u8 = undefined;
        for (f.cfg.kv_sources.items()) |L| if (f.cfg.compressRatio(L) == 2) {
            const w = 2 * @as(usize, f.cfg.head_dim);
            const src = r.addressOf(try std.fmt.bufPrint(&nb, "w.L{d}.comp", .{L})) orelse return error.Unbound;
            const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L})) orelse return error.Unbound;
            try r.kernels.others(r.stream).carry(src, w, 0, accepted, w, dst);
        };
        f.slot.pos = p.start + accepted + 1;
        f.slot.pending = null;
    }

    fn glueMissing(name: []const u8) Error {
        std.log.scoped(.dsv41).warn("glue step {s}: its kernel is not built yet (dsq-zig-kern-1's glue list)", .{name});
        return error.GlueNotBuilt;
    }

    fn numel(t: calls.Tensor) usize {
        var n: usize = 1;
        for (t.shape) |x| n *= @intCast(x);
        return n;
    }

    fn sendBuffer(f: *Forward, r: *run.Runner, bytes: usize) !u64 {
        if (f.send == null or f.send.?.len < bytes) {
            if (f.send) |*b| b.free();
            f.send = try cuda.DeviceBuffer.alloc(r.d, bytes);
        }
        return f.send.?.ptr;
    }

    fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
        const f: *Forward = @ptrCast(@alignCast(ctx));
        const step = c.name["glue.".len..];
        const s = r.stream.handle;
        const k = r.kernels.others(r.stream);
        const p = f.slot.pending.?;
        const D: usize = f.cfg.hidden;
        const W: usize = f.comm.world();
        if (std.mem.eql(u8, step, "positions")) {
            const t = [_]calls.Tensor{ c.args[0].arg.t, c.args[1].arg.t, c.args[2].arg.t };
            return k.positions(@intCast(p.start), p.n, try r.tensorAddr(t[0]), try r.tensorAddr(t[1]), try r.tensorAddr(t[2]));
        }
        if (std.mem.eql(u8, step, "embed")) {
            // this rank's vocabulary rows (others +0), their all-sum (the window's first exchange), x4 into the streams
            const n: usize = p.n;
            if (f.host == null or f.host.?.bytes.len < 8 * n) {
                if (f.host) |*h| h.free();
                f.host = try cuda.HostBuffer.alloc(r.d, @max(8 * n, 4096));
            }
            if (f.dev_ids == null or f.dev_ids.?.len < 8 * n) {
                if (f.dev_ids) |*b| b.free();
                f.dev_ids = try cuda.DeviceBuffer.alloc(r.d, @max(8 * n, 256));
            }
            const h = f.host.?.bytes;
            for (f.ids, 0..) |id, i| std.mem.writeInt(i64, h[8 * i ..][0..8], id, .little);
            try r.d.check(r.d.api.cuMemcpyHtoDAsync_v2(f.dev_ids.?.ptr, h.ptr, 8 * n, s), "cuMemcpyHtoDAsync");
            const embed = r.weights.get("embed") orelse return error.MissingWeight;
            const V: usize = f.cfg.vocab / W;
            const rows = try f.sendBuffer(r, 2 * n * D);
            try k.embedSend(f.dev_ids.?.ptr, n, embed.ptr, @intCast(f.comm.rank() * V), V, D, rows);
            const recv = try r.tensorAddr(c.args[1].arg.t);
            try f.comm.allGather(rows, recv, n * D, .bf16, s);
            return k.embedSum(recv, W, n, D, try r.tensorAddr(c.args[0].arg.t));
        }
        if (std.mem.eql(u8, step, "exchange")) {
            // attention's bf16 partial gathered into the receive buffer, rank 0's first
            const send = c.args[0].arg.t;
            return f.comm.allGather(try r.tensorAddr(send), try r.tensorAddr(c.args[1].arg.t), numel(send), .bf16, s);
        }
        if (std.mem.eql(u8, step, "exchange_f32")) {
            // the MoE's fp32 partial rounded to bf16, then gathered
            const part = c.args[0].arg.t;
            const n = numel(part);
            const half = try f.sendBuffer(r, 2 * n);
            try k.castBf16(try r.tensorAddr(part), half, n);
            return f.comm.allGather(half, try r.tensorAddr(c.args[1].arg.t), n, .bf16, s);
        }
        if (std.mem.eql(u8, step, "kit_weights"))
            return k.kitWeights(try r.tensorAddr(c.args[0].arg.t), try r.tensorAddr(c.args[1].arg.t), numel(c.args[0].arg.t));
        if (std.mem.eql(u8, step, "kit_logits")) return k.kitLogits(try r.tensorAddr(c.args[0].arg.t), numel(c.args[0].arg.t));
        return glueMissing(step); // engram_rows: the host hashing (NgramHasher) and the rows' source (EngramSource)
    }
};

/// M2a's peer: world 2, rank 0 alone on one GPU, the peer's half of every all-gather zeros (dsv41_m1_capture.py's
/// ZeroPeer, the Python reference's). Only the all-gather the forward uses is implemented.
pub const ZeroPeer = struct {
    d: *const cuda.Driver,

    const C = tp.collective;

    fn rank(_: *anyopaque) u32 {
        return 0;
    }
    fn world(_: *anyopaque) u32 {
        return 2;
    }
    fn allGather(ptr: *anyopaque, send: C.DevicePtr, recv: C.DevicePtr, count: usize, t: C.DType, stream: C.Stream) C.Error!void {
        const z: *ZeroPeer = @ptrCast(@alignCast(ptr));
        const n = count * t.size();
        z.d.check(z.d.api.cuMemcpyDtoDAsync_v2(recv, send, n, stream), "cuMemcpyDtoDAsync") catch return error.BackendFailed;
        z.d.check(z.d.api.cuMemsetD8Async(recv + n, 0, n, stream), "cuMemsetD8Async") catch return error.BackendFailed;
    }
    fn unsupported(_: *anyopaque, _: C.DevicePtr, _: C.DevicePtr, _: usize, _: C.DType, _: C.Op, _: C.Stream) C.Error!void {
        return error.Unsupported;
    }
    fn exchange(_: *anyopaque, _: C.DevicePtr, _: C.DevicePtr, _: usize, _: C.DType, _: u32, _: C.Stream) C.Error!void {
        return error.Unsupported;
    }
    fn p2p(_: *anyopaque, _: C.DevicePtr, _: usize, _: C.DType, _: u32, _: C.Stream) C.Error!void {
        return error.Unsupported;
    }
    fn none(_: *anyopaque) C.Error!void {}
    fn abort(_: *anyopaque) void {}

    const vtable: C.Collective.VTable = .{
        .kind = .host,
        .rank = rank,
        .world = world,
        .all_reduce = unsupported,
        .all_gather = allGather,
        .exchange = exchange,
        .send = p2p,
        .recv = p2p,
        .barrier = none,
        .check = none,
        .abort = abort,
    };

    pub fn collective(z: *ZeroPeer) C.Collective {
        return .{ .ptr = z, .vtable = &vtable };
    }
};
