//! DSpark's drafting pass on the GPU: draft/iface.zig's `Pass` over dspark_emit.zig's launches, run by the forward's
//! runner on every rank (the leader sends each ingest / pass to the followers: forward.Drafter). One slot.
//!
//! - **ingest**: the committed rows' taps ("w.taps", the target's last window) into the three blocks' rings. A run
//!   whose rows are not on the device (a slot resumed from another engine's state) restarts the context after them
//!   (Python's restore without DSpark rows: `valid`).
//! - **propose**: the statics staged as Python's DraftInputs (ids, positions, the context list, lo / hi, rows), the
//!   pass, then the host: each rank's best K head columns (dspark.candidates), the ranks' lists gathered (rank order,
//!   Python's gather_cols), the Markov chain and confidences (dspark.chain), as drafter.chain_host. Drafts only
//!   propose (verification is exact): this rule equals the `_chain` kernel's at T = 0 up to fp32 summation order.
//!
//! Glue the pass needs: the embedding of its ids, q's fp32 copy before `_rope` (glue.cu widen_f32), the fp32
//! partials' exchanges, and "ds_out" (the host's step).

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("calls.zig");
const run = @import("run.zig");
const buffers = @import("buffers.zig");
const fwd = @import("forward.zig");
const emit = @import("dspark_emit.zig");
const iface = @import("draft/iface.zig");
const dspark = @import("draft/dspark.zig");
const dsd = @import("dspark_dev.zig");

pub const Error = error{ Slots, NotAPrefix, NoWeight, CandidateWidth, BadCandidates };

pub const GpuPass = struct {
    gpa: std.mem.Allocator,
    f: *fwd.Forward,
    shape: dspark.Shape,
    /// candidates a rank (drafter.k: min(64, 128 / world))
    k: u32,
    /// the most rows an ingest takes at once (its buffers are planned for it)
    ingest_rows: u32,
    markov: dspark.Markov,
    w1: []u16,
    w2: []u16,
    conf: ?[]f32,
    arena: std.heap.ArenaAllocator,
    pass_calls: []const calls.Call = &.{},
    /// where the pass's host step ("glue.ds_out") starts: `begin` launches the calls before it, `collect` the rest
    head: usize = 0,
    /// the ask a `begin` launched (TF_DSV41_SPEC_DRAFT, draft/spec.zig), until its `collect`
    begun: ?iface.Ask = null,
    /// the first position whose ring rows are this slot's
    valid: u64 = 0,
    st: emit.Statics,
    h64: []i64,
    h32: []i32,
    send: ?cuda.DeviceBuffer = null,
    gathered: ?cuda.DeviceBuffer = null,
    /// the last pass's results on the host: gathered candidates / base logits [n][W k], head hidden [n][D]
    cand: []i32,
    cval: []f32,
    hid: []f32,
    /// TF_DSV41_DRAFT_GRAPHS (dspark_dev.zig): candidates on the device, the pass graphed; null: the host step
    dev: ?*dsd.Dev = null,
    /// the last eager pass's logits (a check pass compares the device candidates with the host's on them)
    last_logits: u64 = 0,
    last_v: usize = 0,

    /// The buffer roles DSpark adds to the forward's plan (call before Runner.bind).
    pub fn plan(a: std.mem.Allocator, f: *const fwd.Forward, p: *buffers.Plan, ingest_rows: u32) !void {
        try p.add(try emit.emitIngest(a, f.cfg, f.widths, f.opts, ingest_rows, 0));
        try p.add(try emit.emitPass(a, f.cfg, f.widths, f.opts, f.cfg.dspark_block));
    }

    pub fn init(gpa: std.mem.Allocator, f: *fwd.Forward, ingest_rows: u32) !*GpuPass {
        const cfg = f.cfg;
        const r = f.runner;
        const world = f.comm.world();
        const x = try gpa.create(GpuPass);
        errdefer gpa.destroy(x);
        const n: i64 = cfg.dspark_block;
        const k: u32 = @min(64, 128 / world);
        const rank = cfg.dspark_markov_rank;
        const V: usize = cfg.vocab;
        const w1 = try gpa.alloc(u16, V * rank);
        const w2 = try gpa.alloc(u16, V * rank);
        try fetch(r, "dspark.markov.w1", std.mem.sliceAsBytes(w1));
        try fetch(r, "dspark.markov.w2", std.mem.sliceAsBytes(w2));
        var conf: ?[]f32 = null;
        if (r.weights.get("dspark.conf")) |t| {
            conf = try gpa.alloc(f32, t.len / 4);
            try fetch(r, "dspark.conf", std.mem.sliceAsBytes(conf.?));
        }
        const st: emit.Statics = .{ .n = n, .window = cfg.window };
        const W: usize = world * k;
        x.* = .{
            .gpa = gpa,
            .f = f,
            .shape = .{ .block = cfg.dspark_block, .noise = cfg.dspark_noise_token, .window = cfg.window, .rank = rank, .hidden = cfg.hidden, .candidates = world * k },
            .k = k,
            .ingest_rows = ingest_rows,
            .markov = .{ .w1 = w1, .w2 = w2, .rank = rank, .conf = conf },
            .w1 = w1,
            .w2 = w2,
            .conf = conf,
            .arena = std.heap.ArenaAllocator.init(gpa),
            .st = st,
            .h64 = try gpa.alloc(i64, @intCast(st.len64())),
            .h32 = try gpa.alloc(i32, @intCast(st.len32())),
            .cand = try gpa.alloc(i32, @as(usize, @intCast(n)) * W),
            .cval = try gpa.alloc(f32, @as(usize, @intCast(n)) * W),
            .hid = try gpa.alloc(f32, @as(usize, @intCast(n)) * cfg.hidden),
        };
        x.pass_calls = try emit.emitPass(x.arena.allocator(), cfg, f.widths, f.opts, n);
        x.head = x.pass_calls.len;
        for (x.pass_calls, 0..) |c, i| if (std.mem.eql(u8, c.name, "glue.ds_out")) {
            x.head = i;
        };
        f.drafter = .{ .ptr = x, .ingest = followIngest, .propose = followPropose, .reset = followReset };
        const mode = try dsd.modeFromEnv();
        if (mode != .host) {
            if (r.kernels.others(r.stream).f.ds_cands == null) {
                std.log.scoped(.dsv41).warn("draft graphs: TF_DSV41_DRAFT_GRAPHS set, but the glue fatbin has no ds_cands kernel: the host step", .{});
            } else {
                // every buffer a graphed pass bakes in, at its largest: the embedding's and exchanges' send
                _ = try x.device(&x.send, 2 * @as(usize, @intCast(n)) * cfg.hidden);
                x.dev = try dsd.Dev.init(gpa, r.d, f.comm, r.stream, mode, k, cfg.hidden, @intCast(n), conf != null, @intCast(8 * st.len64() + 4 * st.len32()), 1);
            }
        }
        return x;
    }

    pub fn deinit(x: *GpuPass) void {
        const gpa = x.gpa;
        x.f.drafter = null;
        if (x.dev) |d| d.deinit();
        if (x.send) |*b| b.free();
        if (x.gathered) |*b| b.free();
        gpa.free(x.w1);
        gpa.free(x.w2);
        if (x.conf) |c| gpa.free(c);
        gpa.free(x.h64);
        gpa.free(x.h32);
        gpa.free(x.cand);
        gpa.free(x.cval);
        gpa.free(x.hid);
        x.arena.deinit();
        gpa.destroy(x);
    }

    fn fetch(r: *run.Runner, name: []const u8, out: []u8) !void {
        const t = r.weights.get(name) orelse return error.NoWeight;
        if (t.len < out.len) return error.NoWeight;
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = t.ptr, .len = out.len }, 0, out);
    }

    pub fn pass(x: *GpuPass) iface.Pass {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: iface.Pass.VTable = .{ .ingest = ingest, .propose = propose, .reset = reset, .markov = heads, .begin = begin, .collect = collect };

    fn self(p: *anyopaque) *GpuPass {
        return @ptrCast(@alignCast(p));
    }

    /// A new request's draft context, on every rank: the followers' `valid` decides their half of the candidates, so a
    /// follower that kept the last request's (the Spark window: the same request drafted 207 vs 199) drafts differently.
    fn reset(p: *anyopaque, slot: u32) void {
        _ = slot;
        const x = self(p);
        if (x.begun != null) std.log.err("dspark: a reset while a speculative pass is begun", .{});
        const op = fwd.Forward.dsResetOp();
        x.f.sendDrafter(&op) catch |e| std.log.err("dspark: the draft reset did not reach the followers ({t})", .{e});
        x.valid = 0;
    }

    fn followReset(p: *anyopaque) void {
        self(p).valid = 0;
    }

    fn heads(p: *anyopaque) ?dspark.Markov {
        return self(p).markov;
    }

    /// `rows`: the taps' rows (the window's or prompt's row indices, lanes already kept the last `window`), landing at
    /// positions start .. ; a chain's run is contiguous.
    fn ingest(p: *anyopaque, slot: u32, start: u64, taps: iface.Taps, rows: []const u32) anyerror!void {
        const x = self(p);
        if (slot != 0) return error.Slots;
        if (x.begun != null) return error.PassBegun;
        if (rows.len == 0) return;
        for (rows, 0..) |r, i| if (r != rows[0] + i) return error.NotAPrefix;
        const n: u32 = @intCast(rows.len);
        const ops = fwd.Forward.dsIngestOp(start, rows[0], n, taps.rows);
        try x.f.sendDrafter(&ops);
        try x.doIngest(start, rows[0], n, taps.rows);
    }

    fn followIngest(p: *anyopaque, start: u64, row0: u32, n: u32, avail: u32) anyerror!void {
        return self(p).doIngest(start, row0, n, avail);
    }

    fn doIngest(x: *GpuPass, start: u64, row0: u32, n: u32, avail: u32) !void {
        if (n == 0) return;
        if (row0 + n > avail or n > x.ingest_rows) {
            // rows not on the device (a resumed slot's prompt): the context restarts after them
            x.valid = start + n;
            return;
        }
        const r = x.f.runner;
        try r.stream.synchronize();
        const pos: i32 = @intCast(start);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.pos") orelse return error.Unbound, .len = 4 }, 0, std.mem.asBytes(&pos), r.stream.handle);
        var la = std.heap.ArenaAllocator.init(x.gpa);
        defer la.deinit();
        const cs = try emit.emitIngest(la.allocator(), x.f.cfg, x.f.widths, x.f.opts, n, row0);
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(cs);
    }

    fn propose(p: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        if (asks.len != 1) return error.Slots;
        if (x.begun != null) return error.PassBegun;
        const a = asks[0];
        const ops = fwd.Forward.dsProposeOp(a.anchor, a.start);
        try x.f.sendDrafter(&ops);
        try x.runPass(a.anchor, a.start);
        return x.finish(a, out);
    }

    /// The speculative pass (draft/spec.zig): the followers get the same op_ds_propose and run the whole pass; this
    /// rank launches the pass up to its host step and returns, so the GPU drafts while the lanes commit. Until
    /// `collect`, only work without collectives may follow on the stream (the target's keep): the host step's gather
    /// pairs with the followers' next one.
    fn begin(p: *anyopaque, asks: []const iface.Ask) anyerror!void {
        const x = self(p);
        if (asks.len != 1) return error.Slots;
        if (x.begun != null) return error.PassBegun;
        const a = asks[0];
        const ops = fwd.Forward.dsProposeOp(a.anchor, a.start);
        try x.f.sendDrafter(&ops);
        if (x.devOn()) {
            // the whole pass, its candidates and their copy back: nothing in it waits for the host
            try x.devLaunch(a.anchor, a.start);
            x.begun = a;
            return;
        }
        try x.stagePass(a.anchor, a.start);
        try x.f.runner.window(x.pass_calls[0..x.head]);
        x.begun = a;
    }

    fn collect(p: *anyopaque, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        const a = x.begun orelse return error.NoPassBegun;
        x.begun = null;
        if (out.len != 1) return error.Slots;
        if (x.devOn()) {
            try x.devCollect(true);
            return x.finish(a, out);
        }
        const r = x.f.runner;
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(x.pass_calls[x.head..]);
        return x.finish(a, out);
    }

    /// The host's chain over the pass's gathered candidates (the last `runPass` or `collect`).
    fn finish(x: *GpuPass, a: iface.Ask, out: []iface.Proposal) !void {
        // the chain (Python's chain_host): greedy at T = 0, keyed sampling at the ask's parameters
        const c: usize = x.f.comm.world() * x.k;
        var scratch = std.heap.ArenaAllocator.init(x.gpa);
        defer scratch.deinit();
        try dspark.chain(scratch.allocator(), x.markov, x.cand, x.cval, c, a.anchor, a.params, if (x.conf != null) x.hid else null, out[0].drafts, out[0].conf);
        // trees (lanes' siblings): the gathered candidates and base logits [block][world x k], Python's last_tree
        if (out[0].cand) |cd| {
            if (cd.len != x.cand.len) return error.CandidateWidth;
            @memcpy(cd, x.cand);
        }
        if (out[0].base) |b| {
            if (b.len != x.cval.len) return error.CandidateWidth;
            @memcpy(b, x.cval);
        }
    }

    fn followPropose(p: *anyopaque, anchor: u32, start: u64) anyerror!void {
        return self(p).runPass(anchor, start);
    }

    /// The pass on this rank: statics staged, the launches, the host's candidates (every rank: the gather pairs up).
    fn runPass(x: *GpuPass, anchor: u32, start: u64) !void {
        if (x.devOn()) {
            try x.devLaunch(anchor, start);
            return x.devCollect(x.f.comm.rank() == 0 or x.f.link == null);
        }
        try x.stagePass(anchor, start);
        try x.f.runner.window(x.pass_calls);
    }

    fn devOn(x: *const GpuPass) bool {
        const d = x.dev orelse return false;
        return d.on();
    }

    /// The device path: the statics from pinned staging (no sync), the pass (its key's graph once the checks passed),
    /// the event the host waits on.
    fn devLaunch(x: *GpuPass, anchor: u32, start: u64) !void {
        const d = x.dev.?;
        const st = x.st;
        const sv = try d.statics(0, @intCast(st.len64()), @intCast(st.len32()));
        x.fillStatics(anchor, start, sv[0], sv[1]);
        const r = x.f.runner;
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i64") orelse return error.Unbound, .len = 8 * sv[0].len }, 0, std.mem.sliceAsBytes(sv[0]), r.stream.handle);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i32") orelse return error.Unbound, .len = 4 * sv[1].len }, 0, std.mem.sliceAsBytes(sv[1]), r.stream.handle);
        r.glue = .{ .ctx = x, .run = glueFn };
        const extra = [_]u64{ if (x.send) |b| b.ptr else 0, if (x.send) |b| b.len else 0 };
        try d.issue(0, dsd.fingerprint(r, d, &extra), .{ .ctx = x, .run = passBody });
        try d.mark();
    }

    fn passBody(ctx: *anyopaque, stream: cuda.Stream) anyerror!void {
        const x: *GpuPass = @ptrCast(@alignCast(ctx));
        _ = stream; // the runner's stream
        try x.f.runner.window(x.pass_calls);
    }

    /// The device pass's results on the host (`unpack`: rank 0's chain reads them), a check pass's comparison, and
    /// the ranks' verdict after the checks.
    fn devCollect(x: *GpuPass, unpack: bool) !void {
        const d = x.dev.?;
        try d.wait();
        if (d.checked < dsd.check_passes) try d.check(x.last_logits, x.last_v, @intCast(x.st.n), 0);
        if (unpack) try d.unpack(@intCast(x.st.n), 0, x.f.cfg.vocab, x.cand, x.cval, if (x.conf != null) x.hid else null);
        try d.settle();
    }

    /// The pass's statics (Python's DraftInputs) on the device, and its glue in place.
    fn stagePass(x: *GpuPass, anchor: u32, start: u64) !void {
        x.fillStatics(anchor, start, x.h64, x.h32);
        const r = x.f.runner;
        try r.stream.synchronize();
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i64") orelse return error.Unbound, .len = 8 * x.h64.len }, 0, std.mem.sliceAsBytes(x.h64), r.stream.handle);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i32") orelse return error.Unbound, .len = 4 * x.h32.len }, 0, std.mem.sliceAsBytes(x.h32), r.stream.handle);
        r.glue = .{ .ctx = x, .run = glueFn };
    }

    /// The statics (Python's DraftInputs, one slot) into `h64` / `h32`.
    fn fillStatics(x: *GpuPass, anchor: u32, start: u64, h64: []i64, h32: []i32) void {
        const st = x.st;
        const n: usize = @intCast(st.n);
        const W: usize = @intCast(st.window);
        @memset(h64, 0);
        @memset(h32, 0);
        const ring: u64 = @intCast(emit.ringRows(x.f.cfg));
        for (0..n) |i| {
            h64[@intCast(st.ids() + @as(i64, @intCast(i)))] = if (i == 0) anchor else x.shape.noise;
            h64[@intCast(st.positions() + @as(i64, @intCast(i)))] = @intCast(start + i);
            h64[@intCast(st.readRows() + @as(i64, @intCast(i)))] = 0; // slot 0's ring
            h64[@intCast(st.writeRows() + @as(i64, @intCast(i)))] = 0;
        }
        const ctx = dspark.Context.of(x.shape, start, x.valid);
        h32[@intCast(st.start())] = @intCast(start);
        const tok = h32[@intCast(st.tokens())..][0 .. n * W];
        @memset(tok, -1);
        for (0..n) |i| for (0..ctx.count) |j| {
            tok[i * W + j] = @intCast((ctx.first + j) % ring);
        };
        @memset(h32[@intCast(st.counts())..][0..n], @intCast(ctx.count));
        @memset(h32[@intCast(st.lo())..][0..n], @intCast(start));
        @memset(h32[@intCast(st.hi())..][0..n], @intCast(start + n - 1));
        h32[@intCast(st.anchor())] = @intCast(anchor);
    }

    fn device(x: *GpuPass, slot: *?cuda.DeviceBuffer, bytes: usize) !u64 {
        if (slot.* == null or slot.*.?.len < bytes) {
            if (slot.*) |*b| b.free();
            slot.* = try cuda.DeviceBuffer.alloc(x.f.runner.d, bytes);
        }
        return slot.*.?.ptr;
    }

    fn numel(t: calls.Tensor) usize {
        var n: usize = 1;
        for (t.shape) |d| n *= @intCast(d);
        return n;
    }

    fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
        const x: *GpuPass = @ptrCast(@alignCast(ctx));
        const step = c.name["glue.".len..];
        const s = r.stream.handle;
        const k = r.kernels.others(r.stream);
        const comm = x.f.comm;
        const D: usize = x.f.cfg.hidden;
        const Wd: usize = comm.world();
        if (std.mem.eql(u8, step, "ds_embed")) {
            // the block's ids (staged) through this rank's vocabulary rows, their all-sum, x hc_mult into the streams
            const n: usize = @intCast(x.st.n);
            const V: usize = x.f.cfg.vocab / Wd;
            const embed = r.weights.get("embed") orelse return error.MissingWeight;
            const rows = try x.device(&x.send, 2 * n * D);
            try k.embedSend(try r.tensorAddr(c.args[2].arg.t), n, embed.ptr, @intCast(comm.rank() * V), V, D, rows);
            const recv = try r.tensorAddr(c.args[1].arg.t);
            try comm.allGather(rows, recv, n * D, .bf16, s);
            return k.embedSum(recv, Wd, n, D, try r.tensorAddr(c.args[0].arg.t));
        }
        if (std.mem.eql(u8, step, "exchange_f32")) {
            const part = c.args[0].arg.t;
            const n = numel(part);
            const half = try x.device(&x.send, 2 * n);
            try k.castBf16(try r.tensorAddr(part), half, n);
            return comm.allGather(half, try r.tensorAddr(c.args[1].arg.t), n, .bf16, s);
        }
        if (std.mem.eql(u8, step, "widen")) // q.to(fp32): exact
            return k.widenF32(try r.tensorAddr(c.args[0].arg.t), try r.tensorAddr(c.args[1].arg.t), numel(c.args[0].arg.t));
        if (std.mem.eql(u8, step, "ds_out")) {
            if (x.dev) |dv| if (dv.on()) {
                x.last_logits = try r.tensorAddr(c.args[0].arg.t);
                x.last_v = @intCast(c.args[0].arg.t.shape[1]);
                return dv.launch(r, c.args[0].arg.t, c.args[1].arg.t, @intCast(x.st.n), 0);
            };
            return x.hostStep(r, c.args[0].arg.t, c.args[1].arg.t);
        }
        return error.GlueNotBuilt;
    }

    /// The host's step after the head: this rank's best k columns a row (as token ids), every rank's lists gathered in
    /// rank order (value bits then ids, gather_cols' layout), and the head hidden for the confidences.
    fn hostStep(x: *GpuPass, r: *run.Runner, logits_t: calls.Tensor, hid_t: calls.Tensor) !void {
        const comm = x.f.comm;
        const n: usize = @intCast(x.st.n);
        const V: usize = @intCast(logits_t.shape[1]);
        const D: usize = x.f.cfg.hidden;
        const k: usize = x.k;
        const Wd: usize = comm.world();
        const gpa = x.gpa;
        const logits = try gpa.alloc(f32, n * V);
        defer gpa.free(logits);
        const hid16 = try gpa.alloc(u16, n * D);
        defer gpa.free(hid16);
        try r.stream.synchronize();
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(logits_t), .len = 4 * logits.len }, 0, std.mem.sliceAsBytes(logits));
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(hid_t), .len = 2 * hid16.len }, 0, std.mem.sliceAsBytes(hid16));
        for (hid16, x.hid) |b, *h| h.* = dspark.bf16(b);
        // packed [n][2k] fp32: values, then ids as bits
        const packed_ = try gpa.alloc(f32, n * 2 * k);
        defer gpa.free(packed_);
        const ids = try gpa.alloc(i32, k);
        defer gpa.free(ids);
        const id0: u32 = @intCast(comm.rank() * V);
        for (0..n) |i| {
            const row = packed_[i * 2 * k ..][0 .. 2 * k];
            try dspark.candidates(gpa, logits[i * V ..][0..V], id0, ids, row[0..k]);
            for (ids, row[k..]) |id, *o| o.* = @bitCast(id);
        }
        const bytes = 4 * packed_.len;
        const src = try x.device(&x.send, bytes);
        const dst = try x.device(&x.gathered, bytes * Wd);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = src, .len = bytes }, 0, std.mem.sliceAsBytes(packed_), r.stream.handle);
        try comm.allGather(src, dst, packed_.len, .f32, r.stream.handle); // bits only (ids as fp32 bit patterns)
        try r.stream.synchronize();
        const all = try gpa.alloc(f32, packed_.len * Wd);
        defer gpa.free(all);
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = dst, .len = 4 * all.len }, 0, std.mem.sliceAsBytes(all));
        // [W][n][2k] -> per row: rank 0's k, rank 1's k, ...
        const c = Wd * k;
        for (0..Wd) |w| for (0..n) |i| {
            const src_row = all[(w * n + i) * 2 * k ..][0 .. 2 * k];
            @memcpy(x.cval[i * c + w * k ..][0..k], src_row[0..k]);
            for (src_row[k..], x.cand[i * c + w * k ..][0..k]) |v, *o| o.* = @bitCast(v);
        };
        try checkCandidates(x.cand, x.f.cfg.vocab);
    }
};

/// The gathered candidates are token ids (or -1): anything else is a gather that read its source before the upload
/// landed, or a mis-paired collective. Named here on every rank, before the chain indexes the Markov tables with it.
pub fn checkCandidates(cand: []const i32, vocab: usize) error{BadCandidates}!void {
    for (cand, 0..) |id, i| if (id < -1 or id >= @as(i64, @intCast(vocab))) {
        std.log.err("dspark: gathered candidate {d} at {d} is not a token id", .{ id, i });
        return error.BadCandidates;
    };
}
