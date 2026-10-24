//! Host tests of the forward planner and the prompt driver: op order against the T0 model, widths, dataflow, ranks.
const std = @import("std");
const plan = @import("forward_plan.zig");
const exec = @import("forward_exec.zig");
const fwd = @import("forward.zig");
const prefill = @import("prefill.zig");
const iface = @import("iface.zig");
const tp = @import("tp.zig");
const shard = @import("shard.zig");
const pool = @import("pool.zig");
const Op = plan.Op;
const Stage = plan.Stage;
const Buf = plan.Buf;
const flash = &@import("config.zig").flash;

fn geometry(rank: usize, world: usize, layers: u32, max_rows: u32) !plan.Geometry {
    return plan.Geometry.of(flash, try shard.Share.of(flash, rank, world), layers, max_rows);
}

fn record(g: *const plan.Geometry, sh: plan.Shape, r: *plan.Recorder) !void {
    try plan.plan(g, sh, r);
}

// ---------------------------------------------------------------- T0 order

// What ref.zig's layerStep does, event by event (attention, then the feed-forward), with TP2's rank sum added.
const Ev = enum { norm, q, k, v, rope_q, rope_k, index_q, index_k, rope_iq, rope_ik, cache, scores, pick, attend, o, rank_sum, residual, router, route, experts, w1, w3, act, w2, combine };

const t0_attention = [_]Ev{ .norm, .q, .k, .v, .rope_q, .rope_k, .index_q, .index_k, .rope_iq, .rope_ik, .cache, .scores, .pick, .attend, .o, .rank_sum, .residual };
const t0_dense = [_]Ev{ .norm, .w1, .w3, .act, .w2, .rank_sum, .residual };
const t0_moe = [_]Ev{ .norm, .router, .route, .experts, .w1, .w3, .act, .w2, .combine, .rank_sum, .residual };

// The device stages one T0 event list becomes: the fused norm+residual, one qk_rope after the index projections.
fn deviceStages(a: std.mem.Allocator, events: []const Ev, moe: bool, attn: bool, world: u32, out: *std.ArrayList(Stage)) !void {
    var rope = false;
    for (events) |e| switch (e) {
        .q => try out.append(a, .proj_q),
        .k => try out.append(a, .proj_k),
        .v => try out.append(a, .proj_v),
        .index_q => try out.append(a, .index_proj),
        .index_k, .rope_q, .rope_k, .rope_iq, .cache => {},
        .rope_ik => if (!rope) {
            try out.append(a, .qk_rope);
            rope = true;
        },
        .scores => try out.append(a, .score),
        .pick => try out.append(a, .select),
        .attend => try out.append(a, .attend),
        .o => try out.append(a, .proj_o),
        .rank_sum => if (world > 1) try out.append(a, if (attn) .exchange_attn else .exchange_ffn),
        .residual, .norm => {},
        .router => try out.append(a, .router),
        .route => try out.append(a, .route),
        .experts => try out.append(a, .experts),
        .w1 => try out.append(a, if (moe) .shared_gate else .dense_gate),
        .w3 => try out.append(a, if (moe) .shared_up else .dense_up),
        .act => try out.append(a, if (moe) .shared_act else .dense_act),
        .w2 => try out.append(a, if (moe) .shared_down else .dense_down),
        .combine => try out.append(a, .combine),
    };
}

// The expected stages of a whole model: the layers' events with the fused norms placed where residual + norm meet.
fn expected(a: std.mem.Allocator, layers: u32, world: u32, out: *std.ArrayList(Stage)) !void {
    try out.appendSlice(a, &.{ .stage, .embed, .zero, .norm_in });
    for (0..layers) |l| {
        const moe = flash.isMoe(@intCast(l));
        try deviceStages(a, &t0_attention, false, true, world, out);
        try out.append(a, .norm_post);
        try deviceStages(a, if (moe) &t0_moe else &t0_dense, moe, false, world, out);
        // the exchange of the ffn site is the rank_sum event after w2, which deviceStages maps by position
        try out.append(a, if (l + 1 < layers) .norm_in else .norm_final);
    }
    try out.append(a, .head);
}

fn stages(a: std.mem.Allocator, ops: []const Op) ![]Stage {
    const out = try a.alloc(Stage, ops.len);
    for (ops, out) |op, *s| s.* = plan.stageOf(op);
    return out;
}

test "the planner's order is the T0 model's order, dense and MoE layers, one rank and two" {
    const a = std.testing.allocator;
    for ([_]u32{ 1, 2 }) |world| {
        const g = try geometry(0, world, 5, 16);
        var r: plan.Recorder = .{ .gpa = a };
        defer r.deinit();
        try record(&g, .{ .rows = 3 }, &r);
        var want: std.ArrayList(Stage) = .empty;
        defer want.deinit(a);
        try expected(a, 5, world, &want);
        // T0 has no logits gather of its own: take it off the end
        const got = try stages(a, r.ops.items);
        defer a.free(got);
        const n = if (world > 1) got.len - 1 else got.len;
        try std.testing.expectEqualSlices(Stage, want.items, got[0..n]);
        if (world > 1) try std.testing.expectEqual(Stage.gather, got[n]);
    }
}

// ---------------------------------------------------------------- dataflow

const Set = std.EnumSet(Tag);
const Tag = enum { ids, pos, row_tab, emb, res, a, q, k, v, iq_raw, ik_raw, qn, iqn, attn, part, routed, shr, sum, gate, up, act, logits_r, pick, wts, hlog, hgather, pairs, pairs_all, pages, bmax, sel };

fn tag(b: Buf) Tag {
    return std.meta.stringToEnum(Tag, @tagName(b)).?;
}

const Flow = struct {
    r: [8]Tag = undefined,
    nr: usize = 0,
    w: [4]Tag = undefined,
    nw: usize = 0,

    fn of(reads: []const Tag, writes: []const Tag) Flow {
        var f: Flow = .{};
        @memcpy(f.r[0..reads.len], reads);
        f.nr = reads.len;
        @memcpy(f.w[0..writes.len], writes);
        f.nw = writes.len;
        return f;
    }
};

fn flow(op: Op, shared: bool) Flow {
    return switch (op) {
        .stage => Flow.of(&.{}, &.{ .ids, .pos, .row_tab }),
        .embed => Flow.of(&.{.ids}, &.{.emb}),
        .zero => |o| Flow.of(&.{}, &.{tag(o.buf)}),
        .norm_add => |o| Flow.of(&.{ tag(o.x.buf), .res }, &.{ .res, tag(o.y.buf) }),
        .linear => |o| Flow.of(&.{tag(o.x.buf)}, &.{tag(o.y.buf)}),
        .index_proj => Flow.of(&.{.a}, &.{ .iq_raw, .ik_raw }),
        .qk_rope => if (shared) Flow.of(&.{ .q, .k, .v, .iq_raw, .ik_raw, .pos, .row_tab }, &.{ .qn, .iqn, .pages }) else Flow.of(&.{ .q, .k, .v, .iq_raw, .ik_raw }, &.{ .qn, .iqn, .pages }),
        .score => if (shared) Flow.of(&.{ .iqn, .pages, .pos, .row_tab }, &.{.bmax}) else Flow.of(&.{ .iqn, .pages }, &.{.bmax}),
        .select => Flow.of(&.{.bmax}, &.{.sel}),
        .attend => Flow.of(&.{ .qn, .sel, .pages }, &.{.attn}),
        .router => Flow.of(&.{.a}, &.{.logits_r}),
        .route => Flow.of(&.{.logits_r}, &.{ .pick, .wts }),
        .experts => Flow.of(&.{ .a, .pick, .wts }, &.{.routed}),
        .swiglu => Flow.of(&.{ .gate, .up }, &.{.act}),
        .add2 => |o| Flow.of(&.{ tag(o.a), tag(o.b) }, &.{tag(o.out)}),
        .exchange => |o| Flow.of(&.{tag(o.src)}, &.{tag(o.dst)}),
        .tap => Flow.of(&.{.res}, &.{}),
        .logits_gather => Flow.of(&.{.hlog}, &.{.hgather}),
        .local_pick => Flow.of(&.{.hlog}, &.{.pairs}),
        .pick_gather => Flow.of(&.{.pairs}, &.{.pairs_all}),
    };
}

fn dataflowHolds(ops: []const Op, shared: bool) !void {
    var written: Set = .{};
    for (ops, 0..) |op, i| {
        const f = flow(op, shared);
        for (f.r[0..f.nr]) |t| if (!written.contains(t)) {
            std.debug.print("op {d} ({t}) reads {t} before anything wrote it\n", .{ i, plan.stageOf(op), t });
            return error.ReadBeforeWrite;
        };
        for (f.w[0..f.nw]) |t| written.insert(t);
    }
}

test "every op reads only what an earlier op wrote, at every width, shared or not, one rank or two" {
    const a = std.testing.allocator;
    for ([_]u32{ 1, 2 }) |world| for ([_]u32{ 1, 2, 3, 16, 17, 64, 128 }) |n| for ([_]bool{ false, true }) |shared| {
        if (shared and n > iface.shared_rows) continue;
        const g = try geometry(@intCast(world - 1), world, 4, 128);
        var r: plan.Recorder = .{ .gpa = a };
        defer r.deinit();
        try record(&g, .{ .rows = n, .shared = shared, .max_pos = 4000 }, &r);
        try dataflowHolds(r.ops.items, shared);
    };
}

// ---------------------------------------------------------------- widths

test "every width the contract names plans the same stages; only the rows differ" {
    const a = std.testing.allocator;
    const g = try geometry(0, 2, 5, 128);
    var base: ?[]Stage = null;
    defer if (base) |b| a.free(b);
    for ([_]u32{ 1, 2, 3, 16, 17, 64, 128 }) |n| {
        var r: plan.Recorder = .{ .gpa = a };
        defer r.deinit();
        try record(&g, .{ .rows = n, .max_pos = 300 }, &r);
        const s = try stages(a, r.ops.items);
        if (base) |b| {
            defer a.free(s);
            try std.testing.expectEqualSlices(Stage, b, s);
        } else base = s;
        for (r.ops.items) |op| switch (op) {
            .linear => |o| try std.testing.expect(o.rows == n and !o.prompt),
            .exchange => |o| {
                try std.testing.expectEqual(plan.Form.gather_sum, o.form);
                try std.testing.expectEqual(n, o.rows);
            },
            .experts => |o| try std.testing.expect(o.rows == n and !o.prompt),
            .score => |o| try std.testing.expectEqual(@as(u32, 300), o.max_pos),
            else => {},
        };
    }
    var r: plan.Recorder = .{ .gpa = a };
    defer r.deinit();
    try std.testing.expectError(error.TooManyRows, record(&g, .{ .rows = 129 }, &r));
    try std.testing.expectError(error.TooManyRows, record(&g, .{ .rows = iface.shared_rows + 1, .shared = true }, &r));
    try std.testing.expectError(error.BadRows, record(&g, .{ .rows = 0 }, &r));
    try std.testing.expect(iface.window_rows <= 128 and iface.shared_rows <= 128);
}

test "a prompt chunk: prompt kernels, one head row, the share-form sum, expert calls covering the rows" {
    const a = std.testing.allocator;
    const g = try geometry(1, 2, 4, iface.prefill_rows);
    var r: plan.Recorder = .{ .gpa = a };
    defer r.deinit();
    try record(&g, .{ .rows = iface.prefill_rows, .prompt = true, .max_pos = 4095, .head = .last }, &r);
    var covered: u32 = 0;
    var exchanges: usize = 0;
    var heads: usize = 0;
    for (r.ops.items) |op| switch (op) {
        .linear => |o| {
            if (o.w == .head) {
                heads += 1;
                try std.testing.expectEqual(@as(u32, 1), o.rows);
                try std.testing.expectEqual(iface.prefill_rows - 1, o.x.row);
                try std.testing.expect(!o.prompt);
            } else try std.testing.expect(o.prompt and o.rows == iface.prefill_rows);
        },
        .experts => |o| {
            try std.testing.expectEqual(covered, o.row0);
            try std.testing.expect(o.prompt and o.rows <= plan.prompt_expert_rows);
            covered += o.rows;
        },
        .exchange => |o| {
            exchanges += 1;
            try std.testing.expectEqual(plan.Form.scatter_gather, o.form);
        },
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 1), heads);
    try std.testing.expectEqual(iface.prefill_rows, covered); // one MoE layer of four
    try std.testing.expectEqual(@as(usize, 8), exchanges);
    var none: plan.Recorder = .{ .gpa = a };
    defer none.deinit();
    try record(&g, .{ .rows = 128, .prompt = true, .head = .none }, &none);
    try std.testing.expectEqual(Stage.norm_final, plan.stageOf(none.ops.items[none.ops.items.len - 1]));
    try std.testing.expectError(error.TooManyRows, record(&g, .{ .rows = 256, .head = .all }, &none));
    try std.testing.expectError(error.TooManyRows, record(&g, .{ .rows = 3, .prompt = true, .shared = true }, &none));
}

// ---------------------------------------------------------------- ranks and the TP plan

test "both ranks launch the same sequence, and the exchanges are tp.zig's plan" {
    const a = std.testing.allocator;
    const g0 = try geometry(0, 2, 6, 16);
    const g1 = try geometry(1, 2, 6, 16);
    var r0: plan.Recorder = .{ .gpa = a };
    defer r0.deinit();
    var r1: plan.Recorder = .{ .gpa = a };
    defer r1.deinit();
    try record(&g0, .{ .rows = 5 }, &r0);
    try record(&g1, .{ .rows = 5 }, &r1);
    try std.testing.expectEqual(r0.ops.items.len, r1.ops.items.len);
    for (r0.ops.items, r1.ops.items) |x, y| try std.testing.expect(std.meta.eql(x, y));
    var sites: std.ArrayList(tp.Site) = .empty;
    defer sites.deinit(a);
    for (r0.ops.items) |op| if (op == .exchange) try sites.append(a, op.exchange.site);
    try std.testing.expectEqual(tp.tokenExchanges(flash, 6), sites.items.len);
    for (0..6) |l| {
        const lp = tp.layerPlan(flash, @intCast(l));
        for (lp, 0..) |c, i| try std.testing.expectEqual(c.site, sites.items[2 * l + i]);
    }
    var one: plan.Recorder = .{ .gpa = a };
    defer one.deinit();
    const solo = try geometry(0, 1, 6, 16);
    try record(&solo, .{ .rows = 5 }, &one);
    for (one.ops.items) |op| try std.testing.expect(op != .exchange and op != .logits_gather);
}

test "a MoE layer adds the routed and shared experts into one partial before its single exchange" {
    const a = std.testing.allocator;
    const g = try geometry(0, 2, 4, 16);
    var r: plan.Recorder = .{ .gpa = a };
    defer r.deinit();
    try record(&g, .{ .rows = 4 }, &r);
    var in_layer3 = false;
    var exch: usize = 0;
    var adds: usize = 0;
    for (r.ops.items) |op| {
        if (op == .router and op.router.layer == 3) in_layer3 = true;
        if (!in_layer3) continue;
        if (op == .add2) {
            adds += 1;
            try std.testing.expectEqual(Buf.routed, op.add2.a);
            try std.testing.expectEqual(Buf.shr, op.add2.b);
            try std.testing.expectEqual(Buf.part, op.add2.out);
        }
        if (op == .exchange) exch += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), adds);
    try std.testing.expectEqual(@as(usize, 1), exch); // layer 3's attention exchange came before its router
}

test "the head: padded gather bytes match tp.LogitsGather, or a pick merge" {
    const a = std.testing.allocator;
    const g = try geometry(0, 2, 3, 128);
    const lg = try tp.LogitsGather.of(flash, 2);
    try std.testing.expectEqual(@as(usize, lg.slot), g.slot);
    try std.testing.expectEqual(lg.bytes(16), @as(usize, g.world) * 16 * g.slot * 4);
    var r: plan.Recorder = .{ .gpa = a };
    defer r.deinit();
    try record(&g, .{ .rows = 16 }, &r);
    const last = r.ops.items[r.ops.items.len - 1];
    try std.testing.expectEqual(@as(u32, 16), last.logits_gather.rows);
    var p: plan.Recorder = .{ .gpa = a };
    defer p.deinit();
    try record(&g, .{ .rows = 1, .mode = .picks }, &p);
    try std.testing.expectEqual(Stage.pick, plan.stageOf(p.ops.items[p.ops.items.len - 2]));
    try std.testing.expectEqual(Stage.gather, plan.stageOf(p.ops.items[p.ops.items.len - 1]));
    const out = fwd.Forward.out(&g, .{ .rows = 16, .head = .last });
    try std.testing.expectEqual(@as(u32, 1), out.rows);
    try std.testing.expectEqual(@as(u32, 0), fwd.Forward.out(&g, .{ .rows = 16, .head = .none }).rows);
}

test "geometry: this rank's widths for the shipped model" {
    const g = try geometry(0, 2, 60, 2048);
    try std.testing.expectEqual(@as(u32, 32), g.nq);
    try std.testing.expectEqual(@as(u32, 2), g.niq);
    try std.testing.expectEqual(@as(u32, 2), g.kvh);
    try std.testing.expectEqual(@as(u32, 4096), g.q_w);
    try std.testing.expectEqual(@as(u32, 256), g.kv_w);
    try std.testing.expectEqual(@as(u32, 6144), g.dense_w);
    try std.testing.expectEqual(@as(u32, 1536), g.shared_w);
    try std.testing.expectEqual(@as(u32, 1536), g.moe_w);
    try std.testing.expectEqual(@as(u32, 782 * 128), g.slot);
    try std.testing.expectError(error.BadThin, geometry(0, 2, 61, 16));
    var total: usize = 0;
    for (plan.bufs) |b| total += g.rowBytes(b) * g.capacity(b);
    try std.testing.expect(total < (2 << 30)); // the row buffers of a 2048-row chunk stay under 2 GiB
}

test "the fp32 residual is never rounded: partials, the sum and the residual are fp32 in every norm" {
    const a = std.testing.allocator;
    const g = try geometry(0, 2, 4, 16);
    var r: plan.Recorder = .{ .gpa = a };
    defer r.deinit();
    try record(&g, .{ .rows = 2 }, &r);
    var norms: usize = 0;
    for (r.ops.items) |op| if (op == .norm_add) {
        if (norms > 0) {
            try std.testing.expect(op.norm_add.x_f32);
            try std.testing.expectEqual(Buf.sum, op.norm_add.x.buf);
        }
        norms += 1;
    };
    try std.testing.expectEqual(@as(usize, 4 * 2 + 1), norms);
}

// ---------------------------------------------------------------- the call and the commit

test "a call's shape: one stream, a shared window, and what is refused" {
    const ids = [_]i32{ 1, 2, 3, 4, 5 };
    const s = try fwd.Forward.shape(.{ .ids = &ids, .pos0 = 300 });
    try std.testing.expect(!s.shared and s.rows == 5 and s.max_pos == 304);
    const pos = [_]i32{ 10, 11, 12, 700, 701 };
    const segs = [_]iface.Seg{ .{ .slot = 0, .row0 = 0, .rows = 3 }, .{ .slot = 1, .row0 = 3, .rows = 2 } };
    const sh = try fwd.Forward.shape(.{ .ids = &ids, .segs = &segs, .positions = &pos });
    try std.testing.expect(sh.shared and sh.max_pos == 701);
    try std.testing.expectError(error.BadCall, fwd.Forward.shape(.{ .ids = &ids, .segs = &segs }));
    try std.testing.expectError(error.BadCall, fwd.Forward.shape(.{ .ids = &.{} }));
    try std.testing.expectError(error.BadCall, fwd.Forward.shape(.{ .ids = &ids, .segs = segs[0..1], .positions = &pos }));
    try std.testing.expectEqual(@as(usize, 316), try fwd.Forward.commit(300, 16, 16, false));
    try std.testing.expectEqual(@as(usize, 303), try fwd.Forward.commit(300, 16, 3, false));
    try std.testing.expectError(error.BadCommit, fwd.Forward.commit(300, 16, 0, false));
    try std.testing.expectError(error.BadCommit, fwd.Forward.commit(300, 16, 17, false));
    try std.testing.expectError(error.BadCommit, fwd.Forward.commit(0, 128, 100, true));
}

// ---------------------------------------------------------------- the prompt driver

const Fake = struct {
    cap: usize,
    pos: usize = 0,
    log: std.ArrayList([3]usize) = .empty, // start, rows, head (0 none, 1 last)
    marks: std.ArrayList(u32) = .empty,
    gates: usize = 0,
    drew: ?usize = null,
    cancel_after: ?usize = null,

    pub fn capacity(f: *Fake) usize {
        return f.cap;
    }
    pub fn gate(f: *Fake, c: ?iface.Cancel) !void {
        f.gates += 1;
        if (prefill.cancelled(c)) return error.Cancelled;
    }
    pub fn chunk(f: *Fake, ids: []const i32, start: usize, head: plan.Head) !void {
        try std.testing.expectEqual(f.pos, start);
        try f.log.append(std.testing.allocator, .{ start, ids.len, @intFromBool(head == .last) });
    }
    pub fn first(f: *Fake, len: usize) !u32 {
        f.drew = len;
        return 4242;
    }
    pub fn advance(f: *Fake, rows: usize) !void {
        f.pos += rows;
    }
    fn deinit(f: *Fake) void {
        f.log.deinit(std.testing.allocator);
        f.marks.deinit(std.testing.allocator);
    }
};

fn always(_: ?*anyopaque) bool {
    return true;
}

fn onMark(ctx: *anyopaque, at: u32) void {
    const f: *Fake = @ptrCast(@alignCast(ctx));
    f.marks.append(std.testing.allocator, at) catch unreachable;
}

test "a prompt pass: whole-page chunks, the head only on the last, marks on chunk starts, one first token" {
    var ids: [5000]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i);
    var f: Fake = .{ .cap = 70000 };
    defer f.deinit();
    const at = [_]u32{ 1024, 3200, 4992 };
    const marks: iface.Marks = .{ .at = &at, .ctx = &f, .call = onMark };
    try std.testing.expectEqual(@as(u32, 4242), try prefill.prefill(&f, &ids, null, null, marks));
    var total: usize = 0;
    for (f.log.items, 0..) |c, i| {
        try std.testing.expectEqual(total, c[0]);
        try std.testing.expectEqual(@as(usize, 0), c[0] % pool.page_tokens);
        try std.testing.expect(c[1] <= iface.prefill_rows);
        try std.testing.expectEqual(@as(usize, @intFromBool(i + 1 == f.log.items.len)), c[2]);
        total += c[1];
    }
    try std.testing.expectEqual(@as(usize, 5000), total);
    try std.testing.expectEqual(@as(usize, 5000), f.pos);
    try std.testing.expectEqual(@as(?usize, 5000), f.drew);
    try std.testing.expectEqualSlices(u32, &at, f.marks.items);
    try std.testing.expectEqual(f.log.items.len, f.gates);
    // 0-1024, 1024-3072, 3072-3200, 3200-4992, 4992-5000
    try std.testing.expectEqual(@as(usize, 5), f.log.items.len);
}

test "a resumed pass starts at the adopted pages and ends where a fresh one does" {
    var ids: [4100]u32 = undefined;
    @memset(&ids, 9);
    var fresh: Fake = .{ .cap = 70000 };
    defer fresh.deinit();
    _ = try prefill.prefill(&fresh, &ids, null, null, null);
    var res: Fake = .{ .cap = 70000, .pos = 2048 };
    defer res.deinit();
    _ = try prefill.prefill(&res, &ids, .{ .at = 2048 }, null, null);
    try std.testing.expectEqual(fresh.pos, res.pos);
    try std.testing.expectEqual(@as(usize, 2048), res.log.items[0][0]);
    try std.testing.expectEqual(@as(?usize, 4100), res.drew);
    var bad: Fake = .{ .cap = 70000 };
    defer bad.deinit();
    try std.testing.expectError(error.BadResume, prefill.prefill(&bad, &ids, .{ .at = 100 }, null, null));
    try std.testing.expectError(error.BadResume, prefill.prefill(&bad, &ids, .{ .at = 4100 }, null, null));
    try std.testing.expectError(error.BadResume, prefill.prefill(&bad, &ids, .{ .at = 0 }, null, null));
    var tight: Fake = .{ .cap = 4110 };
    defer tight.deinit();
    try std.testing.expectError(error.PromptTooLong, prefill.prefill(&tight, ids[0..4100], null, null, null));
}

test "a follower's chunk with the next id, a bad chunk, a cancel and an over-long prompt" {
    var ids: [300]u32 = undefined;
    @memset(&ids, 1);
    var f: Fake = .{ .cap = 1000 };
    defer f.deinit();
    try std.testing.expectError(error.BadChunk, prefill.prefillChunk(&f, ids[0..129], 0, 300, null, null, null));
    try std.testing.expectEqual(@as(?u32, null), try prefill.prefillChunk(&f, ids[0..128], 0, 300, null, null, null));
    try std.testing.expectEqual(@as(usize, 128), f.log.items[0][1]);
    var long: [3000]u32 = undefined;
    @memset(&long, 3);
    var w: Fake = .{ .cap = 5000 };
    defer w.deinit();
    try std.testing.expectEqual(@as(?u32, null), try prefill.prefillChunk(&w, long[0..2049], 0, 3000, null, null, null));
    try std.testing.expectEqual(@as(usize, 2048), w.log.items[0][1]);
    try std.testing.expectError(error.BadChunk, prefill.prefillChunk(&f, ids[0..10], 100, 300, null, null, null));
    try std.testing.expectError(error.BadChunk, prefill.prefillChunk(&f, ids[0..0], 128, 300, null, null, null));
    try std.testing.expectError(error.BadChunk, prefill.prefillChunk(&f, ids[0..20], 256, 270, null, null, null));
    try std.testing.expectEqual(@as(?u32, 4242), try prefill.prefillChunk(&f, ids[128..300], 128, 300, null, null, null));
    var g: Fake = .{ .cap = 300 };
    defer g.deinit();
    try std.testing.expectError(error.PromptTooLong, prefill.prefill(&g, &ids, null, null, null));
    const stop: iface.Cancel = .{ .ptr = null, .check = always };
    var h: Fake = .{ .cap = 1000 };
    defer h.deinit();
    try std.testing.expectError(error.Cancelled, prefill.prefill(&h, &ids, null, stop, null));
    try std.testing.expectEqual(@as(usize, 0), h.log.items.len);
    var it: prefill.Chunks = .{ .at = 0, .len = 5000 };
    var n: usize = 0;
    while (it.next()) |c| n += c.end - c.start;
    try std.testing.expectEqual(@as(usize, 5000), n);
}

// ---------------------------------------------------------------- the device half compiles

test "the executor, buffers and call path compile against the CUDA wrappers (no launch here)" {
    _ = &exec.Device.emit;
    _ = &exec.Buffers.init;
    _ = &exec.Modules.load;
    _ = &fwd.Forward.init;
    _ = &fwd.Forward.run;
    _ = &fwd.Forward.address;
    _ = &exec.Device.status;
}

// ---------------------------------------------------------------- the dspark draft's taps

test "taps add one op after each tapped layer's closing norm and nothing else: the step is the same with them off" {
    const a = std.testing.allocator;
    for ([_]u32{ 1, 2 }) |world| for ([_]bool{ false, true }) |shared| {
        var off = try geometry(0, world, 6, 64);
        var on = off;
        on.tap_ids[0] = 1;
        on.tap_ids[1] = 3;
        on.tap_ids[2] = 5; // the last layer: its closing norm is the final one
        on.n_taps = 3;
        const sh: plan.Shape = .{ .rows = 7, .shared = shared, .max_pos = 100 };
        var r0: plan.Recorder = .{ .gpa = a };
        defer r0.deinit();
        var r1: plan.Recorder = .{ .gpa = a };
        defer r1.deinit();
        try record(&off, sh, &r0);
        try record(&on, sh, &r1);
        try std.testing.expectEqual(r0.ops.items.len + 3, r1.ops.items.len);
        var at: usize = 0;
        var seen: u32 = 0;
        for (r1.ops.items) |op| {
            if (op == .tap) {
                try std.testing.expectEqual(seen, op.tap.slot);
                try std.testing.expectEqual(@as(u32, 7), op.tap.rows);
                seen += 1;
            } else {
                try std.testing.expect(plan.stageOf(op) == plan.stageOf(r0.ops.items[at]));
                at += 1;
            }
        }
        try std.testing.expectEqual(@as(u32, 3), seen);
        // each tap follows a norm_add that closes a layer (input of the next layer, or the final norm)
        for (r1.ops.items, 0..) |op, i| if (op == .tap) {
            const before = r1.ops.items[i - 1];
            try std.testing.expect(before == .norm_add and (before.norm_add.norm == .input or before.norm_add.norm == .final));
            try std.testing.expect(before.norm_add.x.buf == .sum or before.norm_add.x.buf == .part);
        };
        try std.testing.expectEqual(@as(?u32, 1), on.tapSlot(3));
        try std.testing.expectEqual(@as(?u32, null), on.tapSlot(2));
        try std.testing.expectEqual(@as(?u32, null), off.tapSlot(3));
    };
}

test "a thin model that does not reach a tapped layer emits no tap for it" {
    const a = std.testing.allocator;
    var g = try geometry(0, 1, 2, 16);
    g.tap_ids[0] = 1;
    g.tap_ids[1] = 59;
    g.n_taps = 2;
    var r: plan.Recorder = .{ .gpa = a };
    defer r.deinit();
    try record(&g, .{ .rows = 3 }, &r);
    var taps: usize = 0;
    for (r.ops.items) |op| taps += @intFromBool(op == .tap);
    try std.testing.expectEqual(@as(usize, 1), taps);
}
