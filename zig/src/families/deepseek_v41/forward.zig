//! M2's forward: a slot's decode windows over the whole model, block.zig's program (launches + glue) run by run.zig
//! on this rank, the exchanges over the TP Collective. One slot: contiguous state (the gates' reference), or the paged
//! KV pool with split KV and sessions (M5, `kv`: kv_state.zig);
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
const graphs = @import("graphs.zig");
const pf = @import("forward_prefill.zig");
const kvs = @import("kv_state.zig");
const er = @import("engram_rows.zig");
const greedy_host = @import("greedy.zig");
const ph = @import("phases.zig");

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
    /// the window's token ids (the embedding step reads them; window() copies them into ids_own)
    ids: []const u32 = &.{},
    ids_own: std.ArrayList(u32) = .empty,
    /// the backbone layers the forward runs (all of them unless a gate runs a prefix: M2a's 0-24)
    layers: ?[]const u32 = null,
    /// TP: rank 0 leads (sends each forward operation to the others over the plan link); the others `follow`
    link: ?*tp.planlink.PlanLink = null,
    /// a follower's drafter (the leader's lanes ask its pass; every rank runs it)
    drafter: ?Drafter = null,
    /// M2a: each block's traced streams (block.Options.trace) as SHA-256 hex by layer, for the last window
    trace: ?*std.AutoArrayHashMapUnmanaged(u32, [64]u8) = null,
    /// the largest row bucket plan() sized the buffers for (a graphed window never has more rows)
    planned_rows: u32 = 0,
    /// R4: decode windows through CUDA graphs (graphs.zig), TF_DSV41_GRAPHS=1 on every rank; null until the first
    /// window reads the knob, then off (eager, the gated path) unless it is set
    graphed: ?*Graphed = null,
    graphs_read: bool = false,
    /// the graphs' default when TF_DSV41_GRAPHS is unset (model.zig: on for M4's served model; the gates' drivers: off)
    graphs_default: bool = false,
    /// R3: Engram's local shards (TF_DSV41_ENGRAM_DIR, engram_rows.zig), opened on first use; unset: M2a's index rows
    engram_rows: ?*er.Rows = null,
    engram_opened: bool = false,
    /// R3's read-ahead at the DSpark pass (TF_DSV41_ENGRAM_PREFETCH=0: rows read when the window needs them; same bits)
    engram_prefetch: bool = true,
    /// TF_DSV41_PHASES (phases.zig): the window path's timers
    phases: ?*ph.Phases = null,
    /// greedy's pinned logits / pairs staging and device pairs, kept across windows
    pick_host: ?cuda.HostBuffer = null,
    pick_dev: ?cuda.DeviceBuffer = null,
    /// the prefill's state (forward_prefill.zig)
    prefill_state: pf.State = .{},
    /// rows of "w.logits" the last prefill left (its last segment's, or its tail's): greedy takes out[0..last_rows]
    last_rows: u32 = 0,
    /// rank 0 inside a prefill: its windows and keeps are the followers' own (they got the whole prompt), not sent
    quiet: bool = false,
    /// M5: the KV families in the paged pool (null: the slot's contiguous buffers); opts.pool is its emitter view
    kv: ?*kvs.Kv = null,
    /// M5: the session store (sessions_gpu.zig): the ids each keep commits, and the operations a follower runs
    sess: ?Sess = null,
    /// the keyed sampler (sampling_gpu.zig): a follower runs rank 0's sampled choices too (their gathers pair up)
    sampler: ?Sampler = null,

    pub fn init(gpa: std.mem.Allocator, cfg: *const Config, widths: *const block.Widths, opts: block.Options, runner: *run.Runner, comm: tp.collective.Collective) Forward {
        return .{ .gpa = gpa, .cfg = cfg, .widths = widths, .opts = opts, .runner = runner, .comm = comm, .arena = std.heap.ArenaAllocator.init(gpa) };
    }

    pub fn deinit(f: *Forward) void {
        if (f.graphed) |g| {
            g.deinit();
            f.gpa.destroy(g);
        }
        if (f.host) |*h| h.free();
        if (f.dev_ids) |*b| b.free();
        if (f.send) |*b| b.free();
        if (f.gathered) |*b| b.free();
        if (f.rows_host) |*b| b.free();
        f.prefill_state.deinit(f.gpa);
        if (f.engram_rows) |x| x.close();
        if (f.pick_host) |*b| b.free();
        if (f.pick_dev) |*b| b.free();
        f.programs.deinit(f.gpa);
        f.ids_own.deinit(f.gpa);
        f.arena.deinit();
    }

    /// The backbone's layers, then the head.
    pub fn backbone(f: *Forward, a: std.mem.Allocator) ![]const u32 {
        if (f.layers) |ls| return ls;
        const ls = try a.alloc(u32, f.cfg.layers);
        for (ls, 0..) |*l, i| l.* = @intCast(i);
        return ls;
    }

    /// The buffer plan of every row bucket the forward runs (bind the runner with it once, before the first window).
    pub fn plan(f: *Forward, buckets: []const u32) !buffers.Plan {
        var p: buffers.Plan = .{ .a = f.arena.allocator() };
        for (buckets) |n| try p.add(try f.program(n));
        for (buckets) |n| f.planned_rows = @max(f.planned_rows, n);
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

    /// Adds the prefill program's roles (block_prefill.zig, segments of up to `rows` rows) to `p`, before bind.
    /// Shared roles take the larger size; the decode roles stay.
    pub fn planPrefill(f: *Forward, p: *buffers.Plan, rows: u32) !void {
        return pf.plan(f, p, rows);
    }

    /// The prompt `ids` from the slot's position (forward_prefill.zig): segments of opts.prefill_rows rows, each run
    /// then kept whole; a tail of at most 16 rows as decode windows. After it the position has moved by ids.len,
    /// nothing is pending, and "w.logits" holds `last_rows` rows (the caller's greedy(out[0..last_rows]), last row).
    pub fn prefill(f: *Forward, ids: []const u32) !void {
        if (f.slot.pending != null) return error.BadWindow;
        if (ids.len == 0) return;
        if (f.leads()) {
            // [op, total, offset, ids...] frames (planlink takes up to 64M words a frame; a prompt is a few)
            const frame: usize = 1 << 20;
            const msg = try f.gpa.alloc(i64, 3 + @min(frame, ids.len));
            defer f.gpa.free(msg);
            var at: usize = 0;
            while (at < ids.len) : (at += frame) {
                const m = @min(frame, ids.len - at);
                msg[0] = op_prefill;
                msg[1] = @intCast(ids.len);
                msg[2] = @intCast(at);
                for (ids[at .. at + m], msg[3 .. 3 + m]) |t, *y| y.* = t;
                try f.link.?.send(msg[0 .. 3 + m]);
            }
        }
        f.quiet = true;
        defer f.quiet = false;
        return pf.prompt(f, ids);
    }

    /// One decode window: `ids` at the slot's position. This rank's logits stay in the runner's "w.logits".
    pub fn window(f: *Forward, ids_in: []const u32) !void {
        // the forward's own copy: keep() reads the ids (Engram's tail) after the caller's slice may be gone (lanes
        // hands a stack row or a scratch arena reset before the keep)
        if (ids_in.ptr != f.ids_own.items.ptr) {
            f.ids_own.clearRetainingCapacity();
            try f.ids_own.appendSlice(f.gpa, ids_in);
        }
        const ids = f.ids_own.items;
        if (ids.len == 0 or f.slot.pending != null) return error.BadWindow;
        // the pool's pages for the window's positions, their table entries up before the launches (every rank alike)
        if (f.kv) |kx| try kx.reserve(f.slot.pos + ids.len);
        if (f.leads()) {
            var msg: [2 + 128]i64 = undefined;
            if (ids.len > 128) return error.BadWindow;
            msg[0] = op_window;
            msg[1] = @intCast(ids.len);
            for (ids, 0..) |t, i| msg[2 + i] = t;
            try f.link.?.send(msg[0 .. 2 + ids.len]);
        }
        const tf = ph.start(f.phases, .forward);
        defer tf.stop();
        if (try f.graphedWindow(ids)) return;
        // the window's calls at its position (the indexer's scores and top-k grow with the cache: NK = end / ratio)
        var wa = std.heap.ArenaAllocator.init(f.gpa);
        defer wa.deinit();
        const cs = try block.emit(wa.allocator(), f.cfg, f.widths, f.opts, try f.backbone(wa.allocator()), @intCast(ids.len), @intCast(f.slot.pos), true);
        f.ids = ids;
        f.slot.pending = .{ .start = f.slot.pos, .n = @intCast(ids.len) };
        f.runner.glue = .{ .ctx = f, .run = glueFn };
        try f.runner.timedWindow(cs);
    }

    /// Greedy choices of the last window's rows (Python's forward.greedy): each rank's max a row (the lowest column on
    /// ties; -0.0 == +0.0), the float64 (value, global id) pairs all-gathered, the higher value (the lower rank on ties).
    pub fn greedy(f: *Forward, out: []u32) !void {
        if (f.leads()) try f.link.?.send(&.{ op_greedy, @intCast(out.len) });
        const tc = ph.start(f.phases, .candidates);
        defer tc.stop();
        const r = f.runner;
        const n = out.len;
        const W: usize = f.comm.world();
        const V: usize = f.cfg.vocab / W;
        // pinned: the logits [n, V], then the pairs [2n] and every rank's [W][2n] (reused, no allocation a window)
        const lb = std.mem.alignForward(usize, 4 * n * V, 8);
        const need = lb + 16 * n * (W + 1);
        if (f.pick_host == null or f.pick_host.?.bytes.len < need) {
            if (f.pick_host) |*b| b.free();
            f.pick_host = try cuda.HostBuffer.alloc(r.d, @max(need, 1 << 20));
        }
        if (f.pick_dev == null or f.pick_dev.?.len < 16 * n * (W + 1)) {
            if (f.pick_dev) |*b| b.free();
            f.pick_dev = try cuda.DeviceBuffer.alloc(r.d, @max(16 * n * (W + 1), 4096));
        }
        const hb = f.pick_host.?.bytes;
        const host: []f32 = @alignCast(std.mem.bytesAsSlice(f32, hb[0 .. 4 * n * V]));
        const pairs: []f64 = @alignCast(std.mem.bytesAsSlice(f64, hb[lb..need]));
        const dev = f.pick_dev.?;
        const logits = r.addressOf("w.logits") orelse return error.Unbound;
        try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(host.ptr, logits, 4 * n * V, r.stream.handle), "cuMemcpyDtoHAsync");
        try r.stream.synchronize();
        for (0..n) |row| {
            const vals = host[row * V ..][0..V];
            const best = greedy_host.argmax(vals);
            pairs[2 * row] = vals[best];
            pairs[2 * row + 1] = @floatFromInt(best + f.comm.rank() * V);
        }
        try dev.uploadAsync(0, std.mem.sliceAsBytes(pairs[0 .. 2 * n]), r.stream.handle);
        try f.comm.allGather(dev.ptr, dev.ptr + 16 * n, 2 * n, .f64, r.stream.handle);
        const all = pairs[2 * n ..];
        try dev.downloadAsync(16 * n, std.mem.sliceAsBytes(all), r.stream.handle);
        try r.stream.synchronize();
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
        if (f.leads()) try f.link.?.send(&.{ op_keep, accepted });
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
        if (f.sess) |x| try x.commit(x.ptr, f.ids[0 .. accepted + 1]);
    }

    const op_window: i64 = 1;
    const op_greedy: i64 = 2;
    const op_keep: i64 = 3;
    const op_stop: i64 = 4;
    const op_ds_ingest: i64 = 5;
    const op_ds_propose: i64 = 6;
    const op_prefill: i64 = 7;
    const op_sess_save: i64 = 8;
    const op_sess_restore: i64 = 9;
    const op_release: i64 = 10;
    const op_ds_reset: i64 = 11;
    /// 12-15: dsq-zig-samp-1 (the keyed sampler); 12: [op, row0, n, T, top_k, top_p, min_p]
    pub const op_sample: i64 = 12;

    /// The sampler's side of a follower (sampling_gpu.zig): rank 0's op_sample message, run on this rank.
    pub const Sampler = struct {
        ptr: *anyopaque,
        follow: *const fn (ptr: *anyopaque, msg: []const i64) anyerror!void,
    };

    /// Rank 0: a sampled choice's message to the followers (before its gathers).
    pub fn sendSample(f: *Forward, ints: []const i64) !void {
        if (f.leads()) try f.link.?.send(ints);
    }
    const op_tree_drop: i64 = 20;

    /// The session store's side of the forward (sessions_gpu.zig): `commit` hears every committed id (keep, prefill
    /// segments); `save` / `restore` are the leader's operations the followers run in its order.
    pub const Sess = struct {
        ptr: *anyopaque,
        commit: *const fn (ptr: *anyopaque, ids: []const u32) anyerror!void,
        save: *const fn (ptr: *anyopaque) anyerror!void,
        restore: *const fn (ptr: *anyopaque, id: u32) anyerror!void,
        /// the slot empty: its pages back, the committed ids cleared
        reset: *const fn (ptr: *anyopaque) anyerror!void,
    };

    /// Drops the pending window on every rank without committing a row (draft/branches.zig: a tree's chain whose branch
    /// lost; the next chain rewrites its positions before any read, as a rejected draft's).
    pub fn drop(f: *Forward) !void {
        if (f.slot.pending == null) return error.BadWindow;
        if (f.leads()) try f.link.?.send(&.{op_tree_drop});
        f.slot.pending = null;
    }

    /// Frees the slot for the next stream on every rank (the Spark window's fix: a follower kept its position, so the
    /// next request ran rank 1 from a stale slot): position 0, no Engram tail, no pending window, the pool's pages back.
    pub fn release(f: *Forward) !void {
        if (f.leads()) try f.link.?.send(&.{op_release});
        if (f.sess) |x| return x.reset(x.ptr);
        if (f.kv) |k| try k.release();
        f.slot = .{};
    }

    /// Rank 0: a session operation to the followers (before running it).
    pub fn sendSess(f: *Forward, op: enum { save, restore }, id: u32) !void {
        if (f.leads()) try f.link.?.send(&.{ if (op == .save) op_sess_save else op_sess_restore, id });
    }

    /// The drafter's operations a follower runs (dspark_gpu.zig): its collectives pair up with the leader's.
    pub const Drafter = struct {
        ptr: *anyopaque,
        ingest: *const fn (ptr: *anyopaque, start: u64, row0: u32, n: u32, avail: u32) anyerror!void,
        propose: *const fn (ptr: *anyopaque, anchor: u32, start: u64) anyerror!void,
        /// the slot's draft context restarts (a new request): every rank's, or their candidates differ
        reset: *const fn (ptr: *anyopaque) void,
    };

    pub fn dsResetOp() [1]i64 {
        return .{op_ds_reset};
    }

    /// Rank 0: a drafter operation to the followers (before running it).
    pub fn sendDrafter(f: *Forward, ints: []const i64) !void {
        if (f.leads()) try f.link.?.send(ints);
        // R3: the pass's anchor is the next window's row 0: its Engram rows are read while the pass runs
        if (ints.len == 3 and ints[0] == op_ds_propose) f.prefetchEngram(@intCast(ints[2]), &.{@intCast(ints[1])});
    }

    pub fn dsIngestOp(start: u64, row0: u32, n: u32, avail: u32) [5]i64 {
        return .{ op_ds_ingest, @intCast(start), row0, n, avail };
    }

    pub fn dsProposeOp(anchor: u32, start: u64) [3]i64 {
        return .{ op_ds_propose, anchor, @intCast(start) };
    }

    fn leads(f: *const Forward) bool {
        return f.link != null and !f.quiet and f.comm.rank() == 0;
    }

    /// Rank 0: the followers' loops end (after the last operation they were sent).
    pub fn stop(f: *Forward) !void {
        if (f.leads()) try f.link.?.send(&.{op_stop});
    }

    /// Ranks > 0: run the leader's operations as they arrive, in its order (every collective then pairs up), until
    /// `stop`. The window's token ids come with it; greedy's choice is computed here too (its all-gather needs every
    /// rank) and dropped. A prefill's prompt comes in frames; the last one starts it.
    pub fn follow(f: *Forward) !void {
        var buf: std.ArrayList(i64) = .empty;
        defer buf.deinit(f.gpa);
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(f.gpa);
        var prompt: std.ArrayList(u32) = .empty;
        defer prompt.deinit(f.gpa);
        // greedy's rows: a window's, or a prefill's last segment (up to prefill_rows)
        var picks: std.ArrayList(u32) = .empty;
        defer picks.deinit(f.gpa);
        while (true) {
            const msg = try f.link.?.recvGrow(f.gpa, &buf);
            if (msg.len == 0) return error.BadPlan;
            switch (msg[0]) {
                op_window => {
                    const n: usize = @intCast(msg[1]);
                    ids.clearRetainingCapacity();
                    for (msg[2 .. 2 + n]) |t| try ids.append(f.gpa, @intCast(t));
                    try f.window(ids.items);
                },
                op_greedy => {
                    try picks.resize(f.gpa, @intCast(msg[1]));
                    try f.greedy(picks.items);
                },
                op_prefill => {
                    const total: usize = @intCast(msg[1]);
                    const at: usize = @intCast(msg[2]);
                    if (at == 0) prompt.clearRetainingCapacity();
                    if (at != prompt.items.len) return error.BadPlan;
                    for (msg[3..]) |t| try prompt.append(f.gpa, @intCast(t));
                    if (prompt.items.len > total) return error.BadPlan;
                    if (prompt.items.len == total) try f.prefill(prompt.items);
                },
                op_keep => try f.keep(@intCast(msg[1])),
                op_release => try f.release(),
                op_tree_drop => try f.drop(),
                op_stop => return,
                op_sess_save => {
                    const x = f.sess orelse return error.BadPlan;
                    try x.save(x.ptr);
                },
                op_sess_restore => {
                    const x = f.sess orelse return error.BadPlan;
                    try x.restore(x.ptr, @intCast(msg[1]));
                },
                op_ds_ingest => {
                    const d = f.drafter orelse return error.BadPlan;
                    try d.ingest(d.ptr, @intCast(msg[1]), @intCast(msg[2]), @intCast(msg[3]), @intCast(msg[4]));
                },
                op_ds_reset => {
                    const d = f.drafter orelse return error.BadPlan;
                    d.reset(d.ptr);
                },
                op_ds_propose => {
                    const d = f.drafter orelse return error.BadPlan;
                    f.prefetchEngram(@intCast(msg[2]), &.{@intCast(msg[1])});
                    try d.propose(d.ptr, @intCast(msg[1]), @intCast(msg[2]));
                },
                op_sample => {
                    const x = f.sampler orelse return error.BadPlan;
                    try x.follow(x.ptr, msg);
                },
                else => return error.BadPlan,
            }
        }
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

    pub fn sendBuffer(f: *Forward, r: *run.Runner, bytes: usize) !u64 {
        if (f.send == null or f.send.?.len < bytes) {
            if (f.send) |*b| b.free();
            f.send = try cuda.DeviceBuffer.alloc(r.d, bytes);
        }
        return f.send.?.ptr;
    }

    /// The decode window's glue (a prefill segment's handler hands it the steps both have).
    pub fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
        const f: *Forward = @ptrCast(@alignCast(ctx));
        const step = c.name["glue.".len..];
        if (std.mem.startsWith(u8, step, "kx_")) return (f.kv orelse return error.GlueNotBuilt).glue(r, c, step);
        const s = r.stream.handle;
        const k = r.kernels.others(r.stream);
        const p = f.slot.pending.?;
        const D: usize = f.cfg.hidden;
        const W: usize = f.comm.world();
        const g = f.activeGraph();
        if (std.mem.eql(u8, step, "positions")) {
            const t = [_]calls.Tensor{ c.args[0].arg.t, c.args[1].arg.t, c.args[2].arg.t };
            // graphed: the start staged in the statics (one graph serves every window of its key)
            if (g) |gr| return k.positionsDev(gr.statics.startDev(), p.n, try r.tensorAddr(t[0]), try r.tensorAddr(t[1]), try r.tensorAddr(t[2]));
            return k.positions(@intCast(p.start), p.n, try r.tensorAddr(t[0]), try r.tensorAddr(t[1]), try r.tensorAddr(t[2]));
        }
        if (g != null and std.mem.eql(u8, step, "embed")) {
            // graphed: the ids staged before the launch; no host step inside the window
            const n: usize = p.n;
            const embed = r.weights.get("embed") orelse return error.MissingWeight;
            const V: usize = f.cfg.vocab / W;
            const rows = try f.sendBuffer(r, 2 * n * D);
            try k.embedSend(g.?.statics.idsDev(), n, embed.ptr, @intCast(f.comm.rank() * V), V, D, rows);
            const recv = try r.tensorAddr(c.args[1].arg.t);
            try f.comm.allGather(rows, recv, n * D, .bf16, s);
            return k.embedSum(recv, W, n, D, try r.tensorAddr(c.args[0].arg.t));
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
        if (std.mem.eql(u8, step, "engram_rows")) {
            if (g) |gr| return f.engramGathered(gr, r, c);
            return f.engramRows(r, c);
        }
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
        const n = f.ids.len;
        const W = f.comm.world();
        const cols = f.engramCols(h); // this rank's columns of a row
        const bytes = 2 * n * cols;
        // an earlier copy out of this staging (the previous Engram layer's) must have left it before it is rewritten
        try r.stream.synchronize();
        if (f.rows_host == null or f.rows_host.?.bytes.len < bytes) {
            if (f.rows_host) |*hb| hb.free();
            f.rows_host = try cuda.HostBuffer.alloc(r.d, @max(bytes, 4096));
        }
        try f.fillEngramRows(h, L, @alignCast(std.mem.bytesAsSlice(u16, f.rows_host.?.bytes[0..bytes])));
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

    /// This rank's columns of an Engram row (its heads' share).
    fn engramCols(f: *const Forward, h: *const eh.Host) usize {
        return h.headShard(f.comm.rank(), f.comm.world())[1] * @as(usize, f.cfg.engram_head_dim);
    }

    /// Layer L's rows of the window's ids (with the slot's tail): the hashes, this rank's heads' index rows.
    fn fillEngramRows(f: *Forward, h: *const eh.Host, L: u32, out: []u16) !void {
        const tw = ph.start(f.phases, .@"engram.wait");
        defer tw.stop();
        return h.windowRows(f.gpa, try f.engramSource(h), f.ids, f.slot.tail[0..f.slot.tail_len], L, f.comm.rank(), f.comm.world(), f.cfg.engram_head_dim, out);
    }

    /// R3: the Engram rows' source, opened on first use: TF_DSV41_ENGRAM_DIR's shards on every rank alike (null:
    /// M2a's index rows, the gates' references). TF_DSV41_ENGRAM_CACHE: records a layer (65,536).
    fn engramSource(f: *Forward, h: *const eh.Host) !?eh.Source {
        if (!f.engram_opened) {
            f.engram_opened = true;
            if (std.c.getenv("TF_DSV41_ENGRAM_PREFETCH")) |v| f.engram_prefetch = !std.mem.eql(u8, std.mem.span(v), "0");
            if (std.c.getenv("TF_DSV41_ENGRAM_DIR")) |dir| {
                const cap: u32 = if (std.c.getenv("TF_DSV41_ENGRAM_CACHE")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 65536;
                f.engram_rows = try er.Rows.open(f.gpa, std.mem.span(dir), f.comm.rank(), f.comm.world(), h.layer_ids, cap);
                std.log.scoped(.dsv41).info("engram: rows from {s} (io_uring {s}, {d} records a layer cached)", .{ std.mem.span(dir), if (f.engram_rows.?.ring != null) "on" else "off", cap });
            } else std.log.scoped(.dsv41).warn("engram: TF_DSV41_ENGRAM_DIR unset: synthetic index rows (M2a's), not the model's table", .{});
        }
        return if (f.engram_rows) |x| x.source() else null;
    }

    /// R3: the coming window's rows `ids` at `start` (row 0: the pending token) read ahead on this rank; its lookback
    /// is the slot's tail plus the rows of the window in flight that `start` implies are kept. A wrong guess costs a
    /// read at the window, never a bit.
    pub fn prefetchEngram(f: *Forward, start: u64, ids: []const u32) void {
        const h = f.engram orelse return;
        const src = (f.engramSource(h) catch return) orelse return;
        if (!f.engram_prefetch) return;
        const tp_ = ph.start(f.phases, .prefetch);
        defer tp_.stop();
        var buf: [2 * eh.max_ngram + 128]u32 = undefined;
        const pend: ?[]const u32 = if (f.slot.pending) |p| f.ids[0..p.n] else null;
        const lb = eh.lookbackAt(f.slot.tail[0..f.slot.tail_len], f.slot.pos, pend, start, h.max_ngram - 1, &buf) orelse return;
        h.prefetch(f.gpa, src, ids, lb, f.comm.rank(), f.comm.world());
    }

    /// Graphed: layer L's staged rows (copied before the launch) all-gathered, then forward.gather_cols' order.
    fn engramGathered(f: *Forward, g: *Graphed, r: *run.Runner, c: *const calls.Call) !void {
        const L: u32 = @intCast(c.args[1].arg.i);
        const st = g.stageOf(L) orelse return error.NotEngram;
        const n = f.ids.len;
        try f.comm.allGather(st.dev.ptr, f.gathered.?.ptr, n * st.cols, .bf16, r.stream.handle);
        return r.kernels.others(r.stream).gatherCols(f.gathered.?.ptr, f.comm.world(), n, st.cols, try r.tensorAddr(c.args[0].arg.t));
    }

    fn activeGraph(f: *Forward) ?*Graphed {
        const g = f.graphed orelse return null;
        return if (g.active) g else null;
    }

    /// The graphs' state, made on the first window when TF_DSV41_GRAPHS is set (both ranks must set it alike).
    fn graphState(f: *Forward) !?*Graphed {
        if (!f.graphs_read) {
            f.graphs_read = true;
            const s = try graphs.Settings.fromEnvOr(f.graphs_default);
            if (s.on and f.planned_rows > 0) {
                const g = try f.gpa.create(Graphed);
                errdefer f.gpa.destroy(g);
                try g.init(f, s);
                f.graphed = g;
                std.log.scoped(.dsv41).info("graphs: on, windows of up to {d} rows, context buckets of {d}", .{ g.rows_max, s.bucket });
            }
        }
        return f.graphed;
    }

    /// R4: the window through its key's graph (captured on first use), its inputs staged first. False: not
    /// graphable here (graphs off, more rows than planned, past the slot's limit, a traced program), run eagerly.
    fn graphedWindow(f: *Forward, ids: []const u32) !bool {
        const g = (try f.graphState()) orelse return false;
        // a traced window (M2a / M2b's first step) is never graphed, and its program is not the key's: it is never
        // emitted or cached here (pod 16: a cached traced program kept every later window eager)
        if (f.opts.trace) return false;
        const n: u32 = @intCast(ids.len);
        const limit: u64 = @intCast(f.opts.limit);
        if (n > g.rows_max or f.slot.pos + n > limit) return false;
        const key: graphs.Key = .{ .rows = n, .ctx = graphs.bucketOf(f.slot.pos + n - 1, g.settings.bucket, g.settings.grow) };
        const cs = try g.program(f, key);
        if (graphs.audit(cs) != null) return false;
        const r = f.runner;
        f.ids = ids;
        f.slot.pending = .{ .start = f.slot.pos, .n = n };
        // inputs: one copy of the ids and start, then each Engram layer's rows (hashed here, on the host)
        {
            const ts = ph.start(f.phases, .@"graph.stage");
            defer ts.stop();
            try g.statics.stage(r.stream, ids, f.slot.pos);
            if (f.engram) |h| for (g.stages.items) |*st| {
                try f.fillEngramRows(h, st.layer, try st.begin(n));
                try st.commit(r.stream, n);
            };
        }
        g.active = true;
        defer g.active = false;
        r.glue = .{ .ctx = f, .run = glueFn };
        g.current = cs;
        const t0 = ph.nowNs();
        const outcome = try g.cache.run(key, r.stream, f.fingerprint(g), .{ .ctx = f, .run = graphBody });
        if (f.phases) |x| x.add(if (outcome == .captured) .@"graph.capture" else .@"graph.replay", ph.nowNs() - t0);
        if (outcome == .captured and g.cache.stats.captured == 1)
            std.log.scoped(.dsv41).info("graphs: first capture ({d} rows, context bucket {d})", .{ key.rows, key.ctx });
        return true;
    }

    fn graphBody(ctx: *anyopaque, stream: cuda.Stream) anyerror!void {
        const f: *Forward = @ptrCast(@alignCast(ctx));
        _ = stream; // the runner's stream: the one the cache captures
        try f.runner.window(f.graphed.?.current);
    }

    /// The addresses a graphed window bakes in beyond the runner's bound roles: any change drops every graph.
    fn fingerprint(f: *const Forward, g: *const Graphed) u64 {
        var words: [64]u64 = undefined;
        var m: usize = 0;
        for (f.runner.owned.items) |b| {
            if (m + 2 > words.len) break;
            words[m] = b.ptr;
            words[m + 1] = b.len;
            m += 2;
        }
        for ([_]?cuda.DeviceBuffer{ f.runner.scratch, f.send, f.gathered, g.statics.dev }) |b| if (b) |x| {
            if (m + 1 > words.len) break;
            words[m] = x.ptr;
            m += 1;
        };
        for (g.stages.items) |st| if (m < words.len) {
            words[m] = st.dev.ptr;
            m += 1;
        };
        return graphs.gc.fingerprintOf(words[0..m]);
    }
};

/// The graphed path's state: the cache over the runner's stream, the window statics, the Engram stages, the
/// programs by key (emitted at the bucket's end: the indexer scores the bucket's keys).
pub const Graphed = struct {
    gpa: std.mem.Allocator,
    settings: graphs.Settings,
    rows_max: u32,
    engine: graphs.gc.CudaEngine,
    agreement: graphs.Agreement,
    cache: graphs.Cache,
    statics: graphs.Statics,
    stages: std.ArrayList(graphs.EngramStage) = .empty,
    programs: std.AutoHashMapUnmanaged(graphs.Key, []const calls.Call) = .empty,
    current: []const calls.Call = &.{},
    active: bool = false,

    fn init(g: *Graphed, f: *Forward, s: graphs.Settings) !void {
        const r = f.runner;
        const rows = @min(s.rows_max, f.planned_rows);
        g.* = .{
            .gpa = f.gpa,
            .settings = s,
            .rows_max = rows,
            .engine = .{ .d = r.d },
            .agreement = try graphs.Agreement.init(r.d, f.comm, r.stream),
            .cache = undefined,
            .statics = undefined,
        };
        errdefer g.agreement.deinit();
        g.statics = try graphs.Statics.init(r.d, rows);
        errdefer g.statics.deinit();
        g.cache = graphs.Cache.init(f.gpa, g.engine.engine(), .{ .on = true, .max = s.max });
        g.cache.agree = g.agreement.agree();
        g.cache.room = roomFn;
        room_floor = s.floor_gib;
        // every buffer a graph bakes in, at the largest window: the exchanges' send, Engram's gathered rows
        const D: usize = f.cfg.hidden;
        const W: usize = f.comm.world();
        // at the planned rows, so an eager window of more rows than a graph takes moves nothing (the fingerprint)
        const most: usize = f.planned_rows;
        _ = try f.sendBuffer(r, 2 * most * D);
        if (f.engram) |h| {
            var la = std.heap.ArenaAllocator.init(f.gpa);
            defer la.deinit();
            const cols = f.engramCols(h);
            for (try f.backbone(la.allocator())) |L| if (std.mem.indexOfScalar(u32, h.layer_ids, L) != null) {
                try g.stages.append(f.gpa, try graphs.EngramStage.init(r.d, L, rows, cols));
            };
            const bytes = W * 2 * most * cols;
            if (f.gathered == null or f.gathered.?.len < bytes) {
                if (f.gathered) |*b| b.free();
                f.gathered = try cuda.DeviceBuffer.alloc(r.d, bytes);
            }
        }
    }

    fn deinit(g: *Graphed) void {
        const cs = g.cache.stats;
        std.log.scoped(.dsv41).info("graphs: {d} captured, {d} replayed, {d} eager, {d} keys held, {d} evicted, {d} dropped{s}", .{ cs.captured, cs.replayed, cs.eager, g.cache.count(), cs.evicted, cs.dropped, if (g.cache.failed) " (off after a failed capture)" else "" });
        g.cache.deinit();
        for (g.stages.items) |*st| st.deinit();
        g.stages.deinit(g.gpa);
        g.programs.deinit(g.gpa);
        g.statics.deinit();
        g.agreement.deinit();
    }

    fn stageOf(g: *Graphed, L: u32) ?*graphs.EngramStage {
        for (g.stages.items) |*st| if (st.layer == L) return st;
        return null;
    }

    /// The key's calls, emitted once (forward's arena) at the start that puts the window's end at its bucket's end.
    fn program(g: *Graphed, f: *Forward, key: graphs.Key) ![]const calls.Call {
        if (g.programs.get(key)) |cs| return cs;
        const a = f.arena.allocator();
        const start = graphs.emitStart(key.rows, key.ctx, g.settings, @intCast(f.opts.limit));
        const cs = try block.emit(a, f.cfg, f.widths, f.opts, try f.backbone(a), key.rows, start, true);
        try g.programs.put(g.gpa, key, cs);
        return cs;
    }
};

var room_floor: f64 = 5.0;
fn roomFn() bool {
    return graphs.roomAbove(room_floor);
}

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
