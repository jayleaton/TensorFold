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

pub const Error = error{ Slots, NotAPrefix, NoWeight };

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
            .shape = .{ .block = cfg.dspark_block, .noise = cfg.dspark_noise_token, .window = cfg.window, .rank = rank, .hidden = cfg.hidden, .candidates = k },
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
        f.drafter = .{ .ptr = x, .ingest = followIngest, .propose = followPropose, .reset = followReset };
        return x;
    }

    pub fn deinit(x: *GpuPass) void {
        const gpa = x.gpa;
        x.f.drafter = null;
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

    const vtable: iface.Pass.VTable = .{ .ingest = ingest, .propose = propose, .reset = reset, .markov = heads };

    fn self(p: *anyopaque) *GpuPass {
        return @ptrCast(@alignCast(p));
    }

    /// A new request's draft context, on every rank: the followers' `valid` decides their half of the candidates, so a
    /// follower that kept the last request's (the Spark window: the same request drafted 207 vs 199) drafts differently.
    fn reset(p: *anyopaque, slot: u32) void {
        _ = slot;
        const x = self(p);
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
        try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = r.addressOf("w.ds.pos") orelse return error.Unbound, .len = 4 }, 0, std.mem.asBytes(&pos));
        var la = std.heap.ArenaAllocator.init(x.gpa);
        defer la.deinit();
        const cs = try emit.emitIngest(la.allocator(), x.f.cfg, x.f.widths, x.f.opts, n, row0);
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(cs);
    }

    fn propose(p: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        if (asks.len != 1) return error.Slots;
        const a = asks[0];
        const ops = fwd.Forward.dsProposeOp(a.anchor, a.start);
        try x.f.sendDrafter(&ops);
        try x.runPass(a.anchor, a.start);
        // the chain (Python's chain_host): greedy at T = 0, keyed sampling at the ask's parameters
        const c: usize = x.f.comm.world() * x.k;
        var scratch = std.heap.ArenaAllocator.init(x.gpa);
        defer scratch.deinit();
        try dspark.chain(scratch.allocator(), x.markov, x.cand, x.cval, c, a.anchor, a.params, if (x.conf != null) x.hid else null, out[0].drafts, out[0].conf);
        if (out[0].cand) |cd| @memcpy(cd, x.cand[0..cd.len]);
        if (out[0].base) |b| @memcpy(b, x.cval[0..b.len]);
    }

    fn followPropose(p: *anyopaque, anchor: u32, start: u64) anyerror!void {
        return self(p).runPass(anchor, start);
    }

    /// The pass on this rank: statics staged, the launches, the host's candidates (every rank: the gather pairs up).
    fn runPass(x: *GpuPass, anchor: u32, start: u64) !void {
        const st = x.st;
        const n: usize = @intCast(st.n);
        const W: usize = @intCast(st.window);
        @memset(x.h64, 0);
        @memset(x.h32, 0);
        const ring: u64 = @intCast(emit.ringRows(x.f.cfg));
        for (0..n) |i| {
            x.h64[@intCast(st.ids() + @as(i64, @intCast(i)))] = if (i == 0) anchor else x.shape.noise;
            x.h64[@intCast(st.positions() + @as(i64, @intCast(i)))] = @intCast(start + i);
            x.h64[@intCast(st.readRows() + @as(i64, @intCast(i)))] = 0; // slot 0's ring
            x.h64[@intCast(st.writeRows() + @as(i64, @intCast(i)))] = 0;
        }
        const ctx = dspark.Context.of(x.shape, start, x.valid);
        x.h32[@intCast(st.start())] = @intCast(start);
        const tok = x.h32[@intCast(st.tokens())..][0 .. n * W];
        @memset(tok, -1);
        for (0..n) |i| for (0..ctx.count) |j| {
            tok[i * W + j] = @intCast((ctx.first + j) % ring);
        };
        @memset(x.h32[@intCast(st.counts())..][0..n], @intCast(ctx.count));
        @memset(x.h32[@intCast(st.lo())..][0..n], @intCast(start));
        @memset(x.h32[@intCast(st.hi())..][0..n], @intCast(start + n - 1));
        x.h32[@intCast(st.anchor())] = @intCast(anchor);
        const r = x.f.runner;
        try r.stream.synchronize();
        try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = r.addressOf("w.ds.i64") orelse return error.Unbound, .len = 8 * x.h64.len }, 0, std.mem.sliceAsBytes(x.h64));
        try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = r.addressOf("w.ds.i32") orelse return error.Unbound, .len = 4 * x.h32.len }, 0, std.mem.sliceAsBytes(x.h32));
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(x.pass_calls);
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
        if (std.mem.eql(u8, step, "ds_out")) return x.hostStep(r, c.args[0].arg.t, c.args[1].arg.t);
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
        try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = src, .len = bytes }, 0, std.mem.sliceAsBytes(packed_));
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
    }
};
