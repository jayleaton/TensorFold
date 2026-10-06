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
    /// pinned staging of the window's positions (pos, pos64, lo)
    host: ?cuda.HostBuffer = null,
    /// the window's token ids (the embedding step reads them)
    ids: []const u32 = &.{},

    pub fn init(gpa: std.mem.Allocator, cfg: *const Config, widths: *const block.Widths, opts: block.Options, runner: *run.Runner, comm: tp.collective.Collective) Forward {
        return .{ .gpa = gpa, .cfg = cfg, .widths = widths, .opts = opts, .runner = runner, .comm = comm, .arena = std.heap.ArenaAllocator.init(gpa) };
    }

    pub fn deinit(f: *Forward) void {
        if (f.host) |*h| h.free();
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
        for (f.cfg.kv_sources.items()) |L| if (f.cfg.compressRatio(L) == 2) return glueMissing("carry");
        f.slot.pos = p.start + accepted + 1;
        f.slot.pending = null;
    }

    fn glueMissing(name: []const u8) Error {
        std.log.scoped(.dsv41).warn("glue step {s}: its kernel is not built yet (dsq-zig-kern-1's glue list)", .{name});
        return error.GlueNotBuilt;
    }

    fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
        const f: *Forward = @ptrCast(@alignCast(ctx));
        const step = c.name["glue.".len..];
        const s = r.stream.handle;
        if (std.mem.eql(u8, step, "positions")) {
            const n = f.slot.pending.?.n;
            const start = f.slot.pending.?.start;
            const bytes = 4 + 8 * n + 4 * n;
            if (f.host == null or f.host.?.bytes.len < bytes) {
                if (f.host) |*h| h.free();
                f.host = try cuda.HostBuffer.alloc(r.d, @max(bytes, 4096));
            }
            const h = f.host.?.bytes;
            std.mem.writeInt(i32, h[0..4], @intCast(start), .little);
            for (0..n) |i| std.mem.writeInt(i64, h[4 + 8 * i ..][0..8], @intCast(start + i), .little);
            @memset(h[4 + 8 * n ..][0 .. 4 * n], 0);
            const dsts = [_][]const u8{ "w.pos", "w.pos64", "w.lo" };
            const offs = [_]usize{ 0, 4, 4 + 8 * n };
            const lens = [_]usize{ 4, 8 * n, 4 * n };
            for (dsts, offs, lens) |role, o, len| {
                const dst = r.addressOf(role) orelse return error.Unbound;
                try r.d.check(r.d.api.cuMemcpyHtoDAsync_v2(dst, h[o..].ptr, len, s), "cuMemcpyHtoDAsync");
            }
            return;
        }
        if (std.mem.eql(u8, step, "exchange")) {
            // attention's bf16 partial gathered into the receive buffer, rank 0's first
            const send = c.args[0].arg.t;
            const recv = c.args[1].arg.t;
            var count: usize = 1;
            for (send.shape) |x| count *= @intCast(x);
            return f.comm.allGather(try r.tensorAddr(send), try r.tensorAddr(recv), count, .bf16, s);
        }
        return glueMissing(step);
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
