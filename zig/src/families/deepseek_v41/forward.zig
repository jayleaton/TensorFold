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
const eh = @import("engram_host.zig");

pub const Error = error{ GlueNotBuilt, BadWindow };

/// One slot's position, Engram tail and pending window (what keep() commits).
pub const Slot = struct {
    pos: u64 = 0,
    /// the last max_ngram - 1 committed ids (Engram's lookback; fewer at the sequence start)
    tail: [eh.max_ngram]u32 = undefined,
    tail_len: usize = 0,
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
    /// Engram's hasher (tools/zig/dsv41_engram_host.py's tables) and the rows' source; M2a: the synthetic rows
    engram: ?*const eh.Host = null,
    gathered: ?cuda.DeviceBuffer = null,
    /// Engram's own pinned staging (the ids' staging may still be in flight: one buffer a producer)
    rows_host: ?cuda.HostBuffer = null,
    /// a bf16 [rows, hidden] send buffer for the exchanges whose partial is not bf16 already (embedding, MoE)
    send: ?cuda.DeviceBuffer = null,
    /// the window's token ids (the embedding step reads them)
    ids: []const u32 = &.{},
    /// the backbone layers the forward runs (all of them unless a gate runs a prefix: M2a's 0-24)
    layers: ?[]const u32 = null,
    /// M2a: each block's traced streams (block.Options.trace) as SHA-256 hex by layer, for the last window
    trace: ?*std.AutoArrayHashMapUnmanaged(u32, [64]u8) = null,

    pub fn init(gpa: std.mem.Allocator, cfg: *const Config, widths: *const block.Widths, opts: block.Options, runner: *run.Runner, comm: tp.collective.Collective) Forward {
        return .{ .gpa = gpa, .cfg = cfg, .widths = widths, .opts = opts, .runner = runner, .comm = comm, .arena = std.heap.ArenaAllocator.init(gpa) };
    }

    pub fn deinit(f: *Forward) void {
        if (f.host) |*h| h.free();
        if (f.dev_ids) |*b| b.free();
        if (f.send) |*b| b.free();
        if (f.gathered) |*b| b.free();
        if (f.rows_host) |*b| b.free();
        f.programs.deinit(f.gpa);
        f.arena.deinit();
    }

    /// The backbone's layers, then the head.
    fn backbone(f: *Forward, a: std.mem.Allocator) ![]const u32 {
        if (f.layers) |ls| return ls;
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

    /// The calls of an `n`-row window at the slot's last position: the largest every role gets (the buffer plan sizes
    /// from it). A window's own calls depend on its position (the indexer's key count), so window() emits them anew.
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
        // the window's calls at its position (the indexer's scores and top-k grow with the cache: NK = end / ratio)
        var wa = std.heap.ArenaAllocator.init(f.gpa);
        defer wa.deinit();
        const cs = try block.emit(wa.allocator(), f.cfg, f.widths, f.opts, try f.backbone(wa.allocator()), @intCast(ids.len), @intCast(f.slot.pos), true);
        f.ids = ids;
        f.slot.pending = .{ .start = f.slot.pos, .n = @intCast(ids.len) };
        f.runner.glue = .{ .ctx = f, .run = glueFn };
        try f.runner.window(cs);
    }

    /// Greedy choices of the last window's rows (Python's forward.greedy): each rank's max a row (the lowest column on
    /// ties; -0.0 == +0.0), the float64 (value, global id) pairs all-gathered, the higher value (the lower rank on ties).
    pub fn greedy(f: *Forward, out: []u32) !void {
        const r = f.runner;
        const n = out.len;
        const W: usize = f.comm.world();
        const V: usize = f.cfg.vocab / W;
        const host = try f.gpa.alloc(f32, n * V);
        defer f.gpa.free(host);
        try r.stream.synchronize();
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = r.addressOf("w.logits") orelse return error.Unbound, .len = 4 * n * V }, 0, std.mem.sliceAsBytes(host));
        const pairs = try f.gpa.alloc(f64, 2 * n * (W + 1));
        defer f.gpa.free(pairs);
        for (0..n) |row| {
            const vals = host[row * V ..][0..V];
            var best: usize = 0;
            for (vals, 0..) |x, i| if (x > vals[best]) {
                best = i;
            };
            pairs[2 * row] = vals[best];
            pairs[2 * row + 1] = @floatFromInt(best + f.comm.rank() * V);
        }
        var dev = try cuda.DeviceBuffer.alloc(r.d, 8 * pairs.len);
        defer dev.free();
        try cuda.DeviceBuffer.upload(dev, 0, std.mem.sliceAsBytes(pairs[0 .. 2 * n]));
        try f.comm.allGather(dev.ptr, dev.ptr + 16 * n, 2 * n, .f64, r.stream.handle);
        try r.stream.synchronize();
        const all = pairs[2 * n ..];
        try cuda.DeviceBuffer.download(dev, 16 * n, std.mem.sliceAsBytes(all));
        for (out, 0..) |*o, row| {
            var bk: usize = 0;
            for (1..W) |k| if (all[2 * (k * n + row)] > all[2 * (bk * n + row)]) {
                bk = k;
            };
            o.* = @intFromFloat(all[2 * (bk * n + row) + 1]);
        }
    }

    /// Commits the pending window's first row + `accepted` drafts (Python's Forward.keep).
    pub fn keep(f: *Forward, accepted: u32) !void {
        const p = f.slot.pending orelse return error.BadWindow;
        if (accepted >= p.n) return error.BadWindow;
        // the ratio-2 KV sources' carry: fp32 of the accepted row of the window's compressor projection
        const r = f.runner;
        var nb: [64]u8 = undefined;
        var la = std.heap.ArenaAllocator.init(f.gpa);
        defer la.deinit();
        for (try f.backbone(la.allocator())) |L| if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2) {
            const w = 2 * @as(usize, f.cfg.head_dim);
            const src = r.addressOf(try std.fmt.bufPrint(&nb, "w.L{d}.comp", .{L})) orelse return error.Unbound;
            const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L})) orelse return error.Unbound;
            try r.kernels.others(r.stream).carry(src, w, 0, accepted, w, dst);
        };
        // Engram's tail: (tail + the committed ids)[-(max_ngram - 1):]
        if (f.engram) |h| {
            const keep_n = h.max_ngram - 1;
            var all: [2 * eh.max_ngram + 128]u32 = undefined;
            var m: usize = 0;
            for (f.slot.tail[0..f.slot.tail_len]) |t| {
                all[m] = t;
                m += 1;
            }
            const take = @min(accepted + 1, all.len - m);
            for (f.ids[0..take]) |t| {
                all[m] = t;
                m += 1;
            }
            const k = @min(m, keep_n);
            @memcpy(f.slot.tail[0..k], all[m - k .. m]);
            f.slot.tail_len = k;
        }
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
            try r.stream.synchronize(); // the previous window's copy out of this staging has finished
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
        if (std.mem.eql(u8, step, "engram_rows")) return f.engramRows(r, c);
        if (std.mem.eql(u8, step, "trace")) {
            const map = f.trace orelse return;
            const t = c.args[0].arg.t;
            const host = try f.gpa.alloc(u8, 2 * numel(t));
            defer f.gpa.free(host);
            try r.stream.synchronize();
            try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(t), .len = host.len }, 0, host);
            var sha: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(host, &sha, .{});
            return map.put(f.gpa, @intCast(c.args[1].arg.i), std.fmt.bytesToHex(sha, .lower));
        }
        return glueMissing(step);
    }

    /// A window's Engram rows of layer L: hashes of its ids (with the slot's tail), this rank's heads' rows (M2a: the
    /// synthetic index rows), the all-gather and the column order of forward.gather_cols.
    fn engramRows(f: *Forward, r: *run.Runner, c: *const calls.Call) !void {
        const h = f.engram orelse return glueMissing("engram_rows (no Engram tables)");
        const L: u32 = @intCast(c.args[1].arg.i);
        const li = std.mem.indexOfScalar(u32, h.layer_ids, L) orelse return error.NotEngram;
        const n = f.ids.len;
        const C = h.cols();
        const W = f.comm.world();
        const dim: usize = f.cfg.engram_head_dim;
        const sh = h.headShard(f.comm.rank(), W);
        const a = f.gpa;
        const hashes = try a.alloc(i64, n * h.layer_ids.len * C);
        defer a.free(hashes);
        h.hash(f.ids, f.slot.tail[0..f.slot.tail_len], hashes);
        const idx = try a.alloc(i64, n * sh[1]);
        defer a.free(idx);
        for (0..n) |t| for (0..sh[1]) |j| {
            idx[t * sh[1] + j] = hashes[(t * h.layer_ids.len + li) * C + sh[0] + j];
        };
        const cols = sh[1] * dim; // this rank's columns of a row
        const bytes = 2 * n * cols;
        // an earlier copy out of this staging (the previous Engram layer's) must have left it before it is rewritten
        try r.stream.synchronize();
        if (f.rows_host == null or f.rows_host.?.bytes.len < bytes) {
            if (f.rows_host) |*hb| hb.free();
            f.rows_host = try cuda.HostBuffer.alloc(r.d, @max(bytes, 4096));
        }
        eh.indexRows(idx, dim, @alignCast(std.mem.bytesAsSlice(u16, f.rows_host.?.bytes[0..bytes])));
        const send = try f.sendBuffer(r, bytes);
        const s = r.stream.handle;
        try r.d.check(r.d.api.cuMemcpyHtoDAsync_v2(send, f.rows_host.?.bytes.ptr, bytes, s), "cuMemcpyHtoDAsync");
        if (f.gathered == null or f.gathered.?.len < W * bytes) {
            if (f.gathered) |*g| g.free();
            f.gathered = try cuda.DeviceBuffer.alloc(r.d, W * bytes);
        }
        try f.comm.allGather(send, f.gathered.?.ptr, n * cols, .bf16, s);
        return r.kernels.others(r.stream).gatherCols(f.gathered.?.ptr, W, n, cols, try r.tensorAddr(c.args[0].arg.t));
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
