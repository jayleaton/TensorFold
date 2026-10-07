//! DSpark drafting over several live slots on the GPU (TF_DSV41_SLOT_DRAFTS=1 with TF_DSV41_SLOTS > 1): draft/iface.zig's
//! `Pass` as Python's Drafter runs it in row mode (G8), every rank alike (the leader sends ops 40-42 first):
//!
//! - **state a slot**: every block's ring is S rings back to back (Python's `stacked`), a slot's rows at slot x ring;
//!   `valid` a slot (the first position whose ring rows are its own). The Markov / confidence heads and the weights are
//!   the one-slot pass's (dspark_gpu.zig).
//! - **ingest**: a slot's committed taps rows into its rings: the row window's rows in "w.taps", or batch.zig's stash
//!   when a later forward of the round overwrote them; the prompt's rows from the one-slot prefill.
//! - **propose**: the asks sorted by slot (Python's order), split into passes of at most `group` slots
//!   (dspark_rows.groups); each pass one launch set over its slots' rows, then the host's candidates, the ranks'
//!   gather and each slot's chain, as the one-slot pass does.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("calls.zig");
const run = @import("run.zig");
const buffers = @import("buffers.zig");
const fwd = @import("forward.zig");
const emit = @import("dspark_emit.zig");
const rows = @import("dspark_rows.zig");
const gpu = @import("dspark_gpu.zig");
const batch = @import("batch.zig");
const iface = @import("draft/iface.zig");
const dspark = @import("draft/dspark.zig");

pub const Error = error{ Slots, NotAPrefix, CandidateWidth, Unbound };

pub const SlotPass = struct {
    gpa: std.mem.Allocator,
    f: *fwd.Forward,
    one: *gpu.GpuPass,
    /// live slots (the rings' count)
    nslots: u32,
    /// slots a pass at most
    group: u32,
    valid: []u64,
    arena: std.heap.ArenaAllocator,
    /// a pass's launches by its slot count - 1
    programs: [][]const calls.Call,
    h64: []i64,
    h32: []i32,
    /// the last pass's host results, [rows][W k] / [rows][D]; the host step's scratch
    cand: []i32,
    cval: []f32,
    hid: []f32,
    logits: []f32,
    hid16: []u16,
    packed_: []f32,
    all: []f32,
    ids: []i32,
    send: ?cuda.DeviceBuffer = null,
    gathered: ?cuda.DeviceBuffer = null,
    prev_ext: ?fwd.Forward.Ext = null,
    /// batch.zig's taps stash (base, bytes; 0, 0: none)
    stash: [2]u64,
    stats: struct { passes: u64 = 0, slots: u64 = 0 } = .{},

    /// The roles several-slot passes add to the forward's plan (before Runner.bind): the stacked rings, each pass
    /// size's statics and rows, the ingests' slot views.
    pub fn plan(a: std.mem.Allocator, f: *const fwd.Forward, p: *buffers.Plan, nslots: u32, group: u32) !void {
        const n: i64 = f.cfg.dspark_block;
        for (1..group + 1) |k| try p.add(try emit.emitPassSlots(a, f.cfg, f.widths, f.opts, n, @intCast(k), nslots));
    }

    /// On every rank after the one-slot pass and the row windows (`prev`: the forward's extension so far, slots').
    pub fn init(gpa: std.mem.Allocator, f: *fwd.Forward, one: *gpu.GpuPass, b: *const batch.Batch, nslots: u32, group: u32) !*SlotPass {
        const x = try gpa.create(SlotPass);
        errdefer gpa.destroy(x);
        const n: usize = f.cfg.dspark_block;
        const R = n * group;
        const c: usize = f.comm.world() * one.k;
        const lay = emit.Layout.of(group, @intCast(n), f.cfg.window);
        const V: usize = f.cfg.vocab / f.comm.world();
        x.* = .{
            .gpa = gpa,
            .f = f,
            .one = one,
            .nslots = nslots,
            .group = group,
            .valid = try gpa.alloc(u64, nslots),
            .arena = std.heap.ArenaAllocator.init(gpa),
            .programs = undefined,
            .h64 = try gpa.alloc(i64, @intCast(lay.len64)),
            .h32 = try gpa.alloc(i32, @intCast(lay.len32)),
            .cand = try gpa.alloc(i32, R * c),
            .cval = try gpa.alloc(f32, R * c),
            .hid = try gpa.alloc(f32, R * f.cfg.hidden),
            .logits = try gpa.alloc(f32, R * V),
            .hid16 = try gpa.alloc(u16, R * f.cfg.hidden),
            .packed_ = try gpa.alloc(f32, R * 2 * one.k),
            .all = try gpa.alloc(f32, R * 2 * c),
            .ids = try gpa.alloc(i32, one.k),
            .prev_ext = f.ext,
            .stash = if (b.taps_stash) |t| .{ t.ptr, t.len } else .{ 0, 0 },
        };
        @memset(x.valid, 0);
        const a = x.arena.allocator();
        x.programs = try a.alloc([]const calls.Call, group);
        for (x.programs, 1..) |*pg, k| pg.* = try emit.emitPassSlots(a, f.cfg, f.widths, f.opts, @intCast(n), @intCast(k), nslots);
        f.ext = .{ .ptr = x, .run = followFn };
        std.log.scoped(.dsv41).info("slot drafts: {d} slots, DSpark passes of up to {d} slots ({d} rows)", .{ nslots, group, R });
        return x;
    }

    pub fn deinit(x: *SlotPass) void {
        const gpa = x.gpa;
        x.f.ext = x.prev_ext;
        if (x.send) |*b| b.free();
        if (x.gathered) |*b| b.free();
        inline for (.{ "valid", "h64", "h32", "cand", "cval", "hid", "logits", "hid16", "packed_", "all", "ids" }) |name| gpa.free(@field(x, name));
        x.arena.deinit();
        gpa.destroy(x);
    }

    pub fn pass(x: *SlotPass) iface.Pass {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: iface.Pass.VTable = .{ .ingest = ingest, .propose = propose, .reset = reset, .markov = heads };

    fn self(p: *anyopaque) *SlotPass {
        return @ptrCast(@alignCast(p));
    }

    fn leads(x: *const SlotPass) bool {
        return x.f.link != null and !x.f.quiet and x.f.comm.rank() == 0;
    }

    fn sendOp(x: *SlotPass, msg: []const i64) !void {
        if (x.leads()) try x.f.link.?.send(msg);
    }

    fn heads(p: *anyopaque) ?dspark.Markov {
        return self(p).one.markov;
    }

    fn reset(p: *anyopaque, slot: u32) void {
        const x = self(p);
        x.sendOp(&.{ rows.op_reset, slot }) catch |e| std.log.err("slot drafts: the draft reset did not reach the followers ({t})", .{e});
        if (slot < x.nslots) x.valid[slot] = 0;
    }

    // -------------------------------------------------------------------------------------------------------------
    // ingest

    fn ingest(p: *anyopaque, slot: u32, start: u64, taps: iface.Taps, rws: []const u32) anyerror!void {
        const x = self(p);
        if (slot >= x.nslots) return error.Slots;
        if (rws.len == 0) return;
        for (rws, 0..) |r, i| if (r != rws[0] + i) return error.NotAPrefix;
        if (taps.map != null) return error.NotAPrefix; // trees are refused with several slots (batch.zig)
        const n: u32 = @intCast(rws.len);
        const r = x.f.runner;
        const at = try rows.locate(taps.address, taps.rows, 2 * @as(u64, taps.stride), rws[0], n, x.one.ingest_rows, r.addressOf("w.taps") orelse return error.Unbound, x.stash);
        try x.sendOp(&.{ rows.op_ingest, slot, @intCast(start), at.src, at.skip, n });
        try x.doIngest(slot, start, at, n);
    }

    fn doIngest(x: *SlotPass, slot: u32, start: u64, at: rows.TapsAt, n: u32) !void {
        if (at.src < 0) {
            x.valid[slot] = start + n; // the rows are not on the device: the context restarts after them
            return;
        }
        const r = x.f.runner;
        try r.stream.synchronize();
        const pos: i32 = @intCast(start);
        try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = r.addressOf("w.ds.pos") orelse return error.Unbound, .len = 4 }, 0, std.mem.asBytes(&pos));
        var la = std.heap.ArenaAllocator.init(x.gpa);
        defer la.deinit();
        const sl: emit.Slots = .{ .ring_slots = x.nslots, .slot = slot, .taps_role = if (at.src == 1) batch.taps_stash_role else "w.taps" };
        const cs = try emit.emitIngestAt(la.allocator(), x.f.cfg, x.f.widths, x.f.opts, n, at.skip, sl);
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(cs);
    }

    // -------------------------------------------------------------------------------------------------------------
    // propose

    fn bySlot(asks: []const iface.Ask, a: usize, b: usize) bool {
        return asks[a].slot < asks[b].slot;
    }

    /// Every ask's drafts: the asks sorted by slot (Python's order), one pass a group (dspark_rows.groups).
    fn propose(p: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        if (asks.len == 0) return;
        if (asks.len > x.nslots) return error.Slots;
        var order: [16]usize = undefined;
        for (order[0..asks.len], 0..) |*o, i| o.* = i;
        std.sort.insertion(usize, order[0..asks.len], asks, bySlot);
        var gs: [16][2]usize = undefined;
        const ng = rows.groups(asks.len, x.group, &gs);
        const n: usize = x.f.cfg.dspark_block;
        const c: usize = x.f.comm.world() * x.one.k;
        const D: usize = x.f.cfg.hidden;
        var scratch = std.heap.ArenaAllocator.init(x.gpa);
        defer scratch.deinit();
        for (gs[0..ng]) |g| {
            var ms: [16]rows.Member = undefined;
            const idx = order[g[0]..g[1]];
            for (idx, ms[0..idx.len]) |i, *m| {
                const a = asks[i];
                if (a.slot >= x.nslots) return error.Slots;
                m.* = .{ .slot = a.slot, .anchor = a.anchor, .start = a.start, .valid = x.valid[a.slot] };
            }
            var buf: [2 + 3 * 16]i64 = undefined;
            try x.sendOp(try rows.proposeMsg(ms[0..idx.len], &buf));
            try x.runGroup(ms[0..idx.len]);
            // each slot's chain (Python's chain_host) over its own rows of the pass
            for (idx, 0..) |i, j| {
                const a = asks[i];
                const o = out[i];
                const cd = x.cand[j * n * c ..][0 .. n * c];
                const cv = x.cval[j * n * c ..][0 .. n * c];
                try dspark.chain(scratch.allocator(), x.one.markov, cd, cv, c, a.anchor, a.params, if (x.one.conf != null) x.hid[j * n * D ..][0 .. n * D] else null, o.drafts, o.conf);
                if (o.cand) |t| {
                    if (t.len != cd.len) return error.CandidateWidth;
                    @memcpy(t, cd);
                }
                if (o.base) |t| {
                    if (t.len != cv.len) return error.CandidateWidth;
                    @memcpy(t, cv);
                }
            }
        }
    }

    /// One pass over `ms` on this rank: the statics staged, the launches, the host's candidates (every rank).
    fn runGroup(x: *SlotPass, ms: []const rows.Member) !void {
        const f = x.f;
        const n: u32 = f.cfg.dspark_block;
        const lay = emit.Layout.of(@intCast(ms.len), n, f.cfg.window);
        rows.stage(lay, n, f.cfg.window, @intCast(emit.ringRows(f.cfg)), x.one.shape.noise, ms, x.h64, x.h32);
        const r = f.runner;
        try r.stream.synchronize();
        try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = r.addressOf("w.ds.i64") orelse return error.Unbound, .len = 8 * @as(usize, @intCast(lay.len64)) }, 0, std.mem.sliceAsBytes(x.h64[0..@intCast(lay.len64)]));
        try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = r.addressOf("w.ds.i32") orelse return error.Unbound, .len = 4 * @as(usize, @intCast(lay.end32)) }, 0, std.mem.sliceAsBytes(x.h32[0..@intCast(lay.end32)]));
        x.stats.passes += 1;
        x.stats.slots += ms.len;
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(x.programs[ms.len - 1]);
    }

    // -------------------------------------------------------------------------------------------------------------
    // followers

    fn followFn(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const x: *SlotPass = @ptrCast(@alignCast(ptr));
        switch (msg[0]) {
            rows.op_reset => {
                if (msg.len != 2) return error.BadPlan;
                const s: usize = @intCast(msg[1]);
                if (s < x.nslots) x.valid[s] = 0;
            },
            rows.op_ingest => {
                if (msg.len != 6 or msg[1] < 0 or msg[1] >= x.nslots) return error.BadPlan;
                try x.doIngest(@intCast(msg[1]), @intCast(msg[2]), .{ .src = msg[3], .skip = msg[4] }, @intCast(msg[5]));
            },
            rows.op_propose => {
                var ms: [16]rows.Member = undefined;
                const got = try rows.proposeOf(msg, x.valid, ms[0..x.group]);
                try x.runGroup(got);
            },
            else => {
                const e = x.prev_ext orelse return error.BadPlan;
                try e.run(e.ptr, msg);
            },
        }
    }

    // -------------------------------------------------------------------------------------------------------------
    // glue

    fn device(x: *SlotPass, slot: *?cuda.DeviceBuffer, bytes: usize) !u64 {
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

    /// dspark_gpu.zig's glue steps at the pass's rows (`R`: every slot's).
    fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
        const x: *SlotPass = @ptrCast(@alignCast(ctx));
        const step = c.name["glue.".len..];
        const s = r.stream.handle;
        const k = r.kernels.others(r.stream);
        const comm = x.f.comm;
        const D: usize = x.f.cfg.hidden;
        const Wd: usize = comm.world();
        if (std.mem.eql(u8, step, "ds_embed")) {
            const n: usize = numel(c.args[2].arg.t);
            const V: usize = x.f.cfg.vocab / Wd;
            const embed = r.weights.get("embed") orelse return error.MissingWeight;
            const rws = try x.device(&x.send, 2 * n * D);
            try k.embedSend(try r.tensorAddr(c.args[2].arg.t), n, embed.ptr, @intCast(comm.rank() * V), V, D, rws);
            const recv = try r.tensorAddr(c.args[1].arg.t);
            try comm.allGather(rws, recv, n * D, .bf16, s);
            return k.embedSum(recv, Wd, n, D, try r.tensorAddr(c.args[0].arg.t));
        }
        if (std.mem.eql(u8, step, "exchange_f32")) {
            const part = c.args[0].arg.t;
            const n = numel(part);
            const half = try x.device(&x.send, 2 * n);
            try k.castBf16(try r.tensorAddr(part), half, n);
            return comm.allGather(half, try r.tensorAddr(c.args[1].arg.t), n, .bf16, s);
        }
        if (std.mem.eql(u8, step, "widen"))
            return k.widenF32(try r.tensorAddr(c.args[0].arg.t), try r.tensorAddr(c.args[1].arg.t), numel(c.args[0].arg.t));
        if (std.mem.eql(u8, step, "ds_out")) return x.hostStep(r, c.args[0].arg.t, c.args[1].arg.t);
        return error.GlueNotBuilt;
    }

    /// dspark_gpu.zig's host step over the pass's rows: each rank's best k a row, the ranks' lists gathered, the head
    /// hidden for the confidences.
    fn hostStep(x: *SlotPass, r: *run.Runner, logits_t: calls.Tensor, hid_t: calls.Tensor) !void {
        const comm = x.f.comm;
        const R: usize = @intCast(logits_t.shape[0]);
        const V: usize = @intCast(logits_t.shape[1]);
        const D: usize = x.f.cfg.hidden;
        const k: usize = x.one.k;
        const Wd: usize = comm.world();
        const logits = x.logits[0 .. R * V];
        const hid16 = x.hid16[0 .. R * D];
        try r.stream.synchronize();
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(logits_t), .len = 4 * logits.len }, 0, std.mem.sliceAsBytes(logits));
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(hid_t), .len = 2 * hid16.len }, 0, std.mem.sliceAsBytes(hid16));
        for (hid16, x.hid[0 .. R * D]) |b, *h| h.* = dspark.bf16(b);
        const packed_ = x.packed_[0 .. R * 2 * k];
        const id0: u32 = @intCast(comm.rank() * V);
        for (0..R) |i| {
            const row = packed_[i * 2 * k ..][0 .. 2 * k];
            try dspark.candidates(x.gpa, logits[i * V ..][0..V], id0, x.ids, row[0..k]);
            for (x.ids, row[k..]) |id, *o| o.* = @bitCast(id);
        }
        const bytes = 4 * packed_.len;
        const src = try x.device(&x.send, bytes);
        const dst = try x.device(&x.gathered, bytes * Wd);
        try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = src, .len = bytes }, 0, std.mem.sliceAsBytes(packed_));
        try comm.allGather(src, dst, packed_.len, .f32, r.stream.handle);
        try r.stream.synchronize();
        const all = x.all[0 .. packed_.len * Wd];
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = dst, .len = 4 * all.len }, 0, std.mem.sliceAsBytes(all));
        rows.ungather(all, Wd, R, k, x.cand, x.cval);
    }
};
