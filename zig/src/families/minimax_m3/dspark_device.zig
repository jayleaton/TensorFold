//! The dspark draft on one rank's device (engine_ops.Draft): K/V rings over the target's taps, block pass, chain.
const std = @import("std");
const cuda = @import("cuda");
const iface = @import("iface.zig");
const ops = @import("engine_ops.zig");
const exec = @import("forward_exec.zig");
const plan = @import("forward_plan.zig");
const weights = @import("weights.zig");
const ffn = @import("ffn.zig");
const dc = @import("dspark_config.zig");
const dw = @import("dspark_weights.zig");
const shard = @import("shard.zig");

const ds = cuda.minimax_m3_dspark;
const qmmf = cuda.qmmf;
const DeviceBuffer = cuda.DeviceBuffer;

/// Candidates a rank offers per drafted row: the head's best (dspark_ref.Config.candidates).
pub const cand = 64;
/// Rows of the block (the anchor and the noise rows); the pass is sized for this many.
pub const max_block = 8;
/// Snapshots of a ring the device keeps (backend.max_snaps).
pub const max_snaps = @import("backend.zig").max_snaps;

/// One stream's ring: `lo` is the first position whose slot still holds its own row, `next` one past the last absorbed.
pub const Meta = struct { lo: u32 = 0, next: u32 = 0 };

/// What the host knows of a pass before it launches: the ring window it reads.
pub fn passWindow(m: Meta, pos: u32, ring_rows: u32) u32 {
    // rows absorbed up to the pending position only: a ring that lags or leads has a gap, so it is read as empty
    if (m.next != pos) return pos;
    return @max(m.lo, pos -| ring_rows);
}

/// The ring after absorbing positions [start, start + rows): it continues, or starts over at `start` after a gap.
pub fn afterAbsorb(m: Meta, start: u32, rows: u32, ring_rows: u32) Meta {
    var r = m;
    if (r.next == r.lo or start != r.next) r = .{ .lo = start, .next = start };
    r.next = start + rows;
    r.lo = @max(r.lo, r.next -| ring_rows);
    return r;
}

pub const Device = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    st: cuda.Stream,
    cfg: dc.Config,
    model: dw.Model,
    lane: qmmf.Lane,
    k: *const ds.Dspark,
    m: *exec.Modules,
    tw: *const weights.Weights, // the target's embedding and head
    g: *const plan.Geometry,
    link: ?exec.Link,
    world: u32,
    rank: u32,
    // dimensions of this rank
    nq: u32,
    kvh: u32,
    ffn_w: u32,
    ring_rows: u32,
    ring_bytes: usize,
    // per-stream state
    rings: []DeviceBuffer = &.{},
    meta: []Meta = &.{},
    snaps: [max_snaps]DeviceBuffer = undefined,
    snap_meta: [max_snaps]Meta = @splat(.{}),
    snap_made: bool = false,
    results: DeviceBuffer = undefined, // [slots][max_block] of {i32 token, f32 conf}
    result_slots: u32 = 0,
    // scratch
    sc: Scratch = undefined,

    const Scratch = struct {
        buf: DeviceBuffer,
        fc_out: u64,
        hn: u64,
        k_raw: u64,
        v_raw: u64,
        ids: u64,
        emb: u64,
        res: u64,
        a: u64,
        q_raw: u64,
        qn: u64,
        kn: u64,
        vn: u64,
        attn: u64,
        part: u64,
        sum: u64,
        gathered: u64,
        gate: u64,
        up: u64,
        act: u64,
        post: u64,
        logits: u64,
        cand_local: u64,
        cand_all: u64,
        mm_part: u64,
    };

    /// Bytes of one scratch field laid out in `Scratch`'s order.
    fn layout(dv: *const Device, assign: ?*Scratch) usize {
        const h: usize = dv.cfg.hidden;
        const rows = iface.prefill_rows;
        const b = max_block;
        const kvw: usize = @as(usize, dv.kvh) * 128;
        const qw: usize = @as(usize, dv.nq) * 128;
        const w: usize = dv.world;
        const sizes = [_]usize{
            rows * h * 2, rows * h * 2, rows * kvw * 2, rows * kvw * 2, // fc_out, hn, k_raw, v_raw
            b * 4, b * h * 2, b * h * 4, b * h * 2, // ids, emb, res, a
            b * qw * 2, b * qw * 2, b * kvw * 2, b * kvw * 2, // q_raw, qn, kn, vn
            b * qw * 2, b * h * 4, b * h * 4, w * b * h * 4, // attn, part, sum, gathered
            b * dv.ffn_w * 2, b * dv.ffn_w * 2, b * dv.ffn_w * 2, b * h * 2, // gate, up, act, post
            b * @as(usize, dv.g.slot) * 4, b * cand * 8, w * b * cand * 8, dv.mmPartBytes(), // logits, cand_local, cand_all, mm_part
        };
        var total: usize = 0;
        var ptrs: [sizes.len]u64 = undefined;
        for (sizes, &ptrs) |sz, *p| {
            total = std.mem.alignForward(usize, total, 256);
            p.* = total;
            total += sz;
        }
        if (assign) |s| {
            const base = s.buf.ptr;
            inline for (@typeInfo(Scratch).@"struct".field_names[1..], 0..) |name, i| @field(s, name) = base + ptrs[i];
        }
        return total;
    }

    /// Split-K partials the lane matmul needs for the largest call it makes here (0 with clusters).
    fn mmPartBytes(dv: *const Device) usize {
        var most: usize = 16;
        const calls = [_]usize{ max_block, 255 };
        for (dv.model.layers) |ly| for ([_]qmmf.Weight{ ly.q, ly.k, ly.v, ly.o, ly.gate, ly.up, ly.down }) |wt| {
            for (calls) |mm| most = @max(most, dv.lane.partBytes(mm, wt));
        };
        for (calls) |mm| most = @max(most, dv.lane.partBytes(mm, dv.model.fc));
        return most;
    }

    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, stream: cuda.Stream, dir: []const u8, cfg: dc.Config, share: shard.Share, m: *exec.Modules, tw: *const weights.Weights, g: *const plan.Geometry, link: ?exec.Link) !*Device {
        const dv = try gpa.create(Device);
        errdefer gpa.destroy(dv);
        const cap = (try ctx.capability()) / 10;
        const rings = dc.ringRows(&cfg, iface.window_rows);
        dv.* = .{
            .gpa = gpa,
            .d = d,
            .st = stream,
            .cfg = cfg,
            .model = try dw.loadModel(gpa, io, d, stream, dir, &cfg, share),
            .lane = undefined,
            .k = &m.dspark,
            .m = m,
            .tw = tw,
            .g = g,
            .link = link,
            .world = g.world,
            .rank = g.rank,
            .nq = @intCast(share.q_heads.len()),
            .kvh = @intCast(share.kv_heads.len()),
            .ffn_w = @intCast(share.dense.len()),
            .ring_rows = @intCast(rings),
            .ring_bytes = @intCast(dc.ringBytes(&cfg, share, iface.window_rows)),
        };
        errdefer dv.model.deinit();
        dv.lane = try qmmf.Lane.load(d, cap);
        errdefer dv.lane.unload();
        const total = dv.layout(null);
        dv.sc.buf = try DeviceBuffer.alloc(d, total);
        errdefer dv.sc.buf.free();
        _ = dv.layout(&dv.sc);
        try dv.sc.buf.fill8(0, stream.handle);
        for (&dv.snaps) |*b| b.* = .{ .d = d, .ptr = 0, .len = 0 };
        var made: usize = 0;
        errdefer for (dv.snaps[0..made]) |*b| b.free();
        for (&dv.snaps) |*b| {
            b.* = try DeviceBuffer.alloc(d, dv.ring_bytes);
            made += 1;
        }
        dv.snap_made = true;
        try stream.synchronize();
        return dv;
    }

    pub fn interface(dv: *Device) ops.Draft {
        return .{ .ptr = dv, .vtable = &.{ .attach = attach, .begin = begin, .absorb = absorb, .pass = pass, .result = result, .snap_save = snapSave, .snap_restore = snapRestore, .snap_drop = snapDrop, .deinit = deinit } };
    }

    /// Device bytes the draft holds on this rank besides the snapshots: weights, scratch, and a ring per stream.
    pub fn bytes(dv: *const Device) u64 {
        return dv.model.bytes + dv.sc.buf.len;
    }

    fn of(p: *anyopaque) *Device {
        return @ptrCast(@alignCast(p));
    }

    fn freeRings(dv: *Device) void {
        for (dv.rings) |*r| r.free();
        dv.gpa.free(dv.rings);
        dv.gpa.free(dv.meta);
        dv.rings = &.{};
        dv.meta = &.{};
        if (dv.result_slots > 0) dv.results.free();
        dv.result_slots = 0;
    }

    fn attach(p: *anyopaque, slots: u32) anyerror!void {
        const dv = of(p);
        dv.freeRings();
        dv.rings = try dv.gpa.alloc(DeviceBuffer, slots);
        errdefer dv.gpa.free(dv.rings);
        dv.meta = try dv.gpa.alloc(Meta, slots);
        errdefer dv.gpa.free(dv.meta);
        var made: usize = 0;
        errdefer for (dv.rings[0..made]) |*r| r.free();
        for (dv.rings) |*r| {
            r.* = try DeviceBuffer.alloc(dv.d, dv.ring_bytes);
            made += 1;
        }
        @memset(dv.meta, .{});
        dv.results = try DeviceBuffer.alloc(dv.d, @as(usize, slots) * max_block * 8);
        dv.result_slots = slots;
        try dv.results.fill8(0, dv.st.handle);
    }

    fn begin(p: *anyopaque, slot: u32) void {
        const dv = of(p);
        if (slot < dv.meta.len) dv.meta[slot] = .{};
    }

    /// K (kind 0) or V (kind 1) ring of layer `l` in stream `slot`'s buffer.
    pub fn ringAt(dv: *const Device, base: u64, l: usize, kind: usize) u64 {
        const one: u64 = @as(u64, dv.ring_rows) * dv.kvh * 128 * 2;
        return base + (@as(u64, l) * 2 + kind) * one;
    }

    fn heads(dv: *const Device) ds.Heads {
        return .{ .nq = dv.nq, .kvh = dv.kvh, .rot = dv.cfg.rotary_dim, .theta = dv.cfg.rope_theta, .eps = dv.cfg.eps, .gemma = true };
    }

    /// Tap rows [row0, row0 + rows): fc, hidden_norm, then each layer's K and V into the slot's ring.
    fn absorb(p: *anyopaque, slot: u32, start: u32, rows: u32, base: u64, row0: u32) anyerror!void {
        const dv = of(p);
        if (slot >= dv.meta.len or rows == 0 or rows > iface.prefill_rows or base == 0) return error.BadAbsorb;
        const c = &dv.cfg;
        const h: u32 = c.hidden;
        const row_bytes: u64 = @as(u64, c.fcIn()) * 2;
        const x = base + @as(u64, row0) * row_bytes;
        const sc = &dv.sc;
        try dv.lane.matmul(dv.st, x, c.fcIn(), rows, dv.model.fc, sc.fc_out, sc.mm_part);
        try dv.k.norm(dv.st, sc.fc_out, false, dv.model.hidden_norm, true, sc.hn, rows, h, c.eps);
        const ring = dv.rings[slot].ptr;
        for (dv.model.layers, 0..) |ly, l| {
            try dv.lane.matmul(dv.st, sc.hn, h, rows, ly.k, sc.k_raw, sc.mm_part);
            try dv.lane.matmul(dv.st, sc.hn, h, rows, ly.v, sc.v_raw, sc.mm_part);
            var kh = dv.heads();
            kh.nq = 0; // (ring_put reads no query heads)
            try dv.k.ringPut(dv.st, kh, sc.k_raw, sc.v_raw, ly.k_norm, dv.ringAt(ring, l, 0), dv.ringAt(ring, l, 1), dv.ring_rows, rows, start);
        }
        dv.meta[slot] = afterAbsorb(dv.meta[slot], start, rows, dv.ring_rows);
    }

    /// `src` f32 [rows][hidden] partial sums of both ranks added in rank order into `dst` (src itself on one rank).
    fn exchange(dv: *const Device, src: u64, dst: u64, rows: u32) !u64 {
        const link = dv.link orelse return src;
        const n: usize = @as(usize, rows) * dv.cfg.hidden;
        try link.group.allGather(dv.st, src, dv.sc.gathered, n * 4);
        try dv.m.scatter.rankSum(dv.st, dv.sc.gathered, src, dv.world, dv.rank, n, dst);
        return dst;
    }

    fn pass(p: *anyopaque, req: ops.DraftPass) anyerror!void {
        const dv = of(p);
        if (req.slot >= dv.meta.len) return error.BadPass;
        try dv.block(req);
        try dv.heads_(req);
    }

    /// The block's layers and the final norm: `sc.post` [block][hidden] bf16 (the part of a pass that needs no head).
    pub fn block(dv: *Device, req: ops.DraftPass) !void {
        const c = &dv.cfg;
        const sc = &dv.sc;
        const rows: u32 = c.block;
        const h: u32 = c.hidden;
        const n: usize = @as(usize, rows) * h;
        const lo = passWindow(dv.meta[req.slot], req.pos, dv.ring_rows);
        const ring = dv.rings[req.slot].ptr;
        // the block's ids: the anchor, then the noise token
        var ids: [max_block]u32 = @splat(c.mask_token);
        ids[0] = req.anchor;
        try dv.sc.buf.uploadAsync(sc.ids - dv.sc.buf.ptr, std.mem.sliceAsBytes(ids[0..rows]), dv.st.handle);
        try dv.m.embed.bf16(dv.st, sc.ids, dv.tw.embed, sc.emb, rows, h);
        try dv.k.add(dv.st, 0, sc.emb, false, sc.res, n);
        const attn: ds.Attn = .{ .nq = dv.nq, .kvh = dv.kvh, .ring_rows = dv.ring_rows, .window = c.window, .scale = @floatCast(std.math.pow(f64, 128, -0.5)) };
        const act_params = ffn.actParams(.{});
        for (dv.model.layers, 0..) |ly, l| {
            try dv.k.norm(dv.st, sc.res, true, ly.input_ln, true, sc.a, rows, h, c.eps);
            try dv.lane.matmul(dv.st, sc.a, h, rows, ly.q, sc.q_raw, sc.mm_part);
            try dv.lane.matmul(dv.st, sc.a, h, rows, ly.k, sc.k_raw, sc.mm_part);
            try dv.lane.matmul(dv.st, sc.a, h, rows, ly.v, sc.v_raw, sc.mm_part);
            try dv.k.blockHeads(dv.st, dv.heads(), sc.q_raw, sc.k_raw, sc.v_raw, ly.q_norm, ly.k_norm, sc.qn, sc.kn, sc.vn, rows, req.pos);
            try dv.k.blockAttn(dv.st, attn, sc.qn, dv.ringAt(ring, l, 0), dv.ringAt(ring, l, 1), sc.kn, sc.vn, sc.attn, rows, req.pos, lo);
            try dv.lane.matmulF32(dv.st, sc.attn, @as(usize, dv.nq) * 128, rows, ly.o, sc.part, sc.mm_part);
            const o_sum = try dv.exchange(sc.part, sc.sum, rows);
            try dv.k.add(dv.st, sc.res, o_sum, true, sc.res, n);
            try dv.k.norm(dv.st, sc.res, true, ly.post_ln, true, sc.a, rows, h, c.eps);
            try dv.lane.matmul(dv.st, sc.a, h, rows, ly.gate, sc.gate, sc.mm_part);
            try dv.lane.matmul(dv.st, sc.a, h, rows, ly.up, sc.up, sc.mm_part);
            try dv.m.act.swigluoai(dv.st, .bf16, sc.gate, dv.ffn_w, sc.up, dv.ffn_w, sc.act, dv.ffn_w, rows, dv.ffn_w, act_params);
            try dv.lane.matmulF32(dv.st, sc.act, dv.ffn_w, rows, ly.down, sc.part, sc.mm_part);
            const d_sum = try dv.exchange(sc.part, sc.sum, rows);
            try dv.k.add(dv.st, sc.res, d_sum, true, sc.res, n);
        }
        try dv.k.norm(dv.st, sc.res, true, dv.model.final_norm, true, sc.post, rows, h, c.eps);
    }

    /// The head, the best candidates, the chain and the confidences of the block `block` left.
    fn heads_(dv: *Device, req: ops.DraftPass) !void {
        const c = &dv.cfg;
        const sc = &dv.sc;
        const rows: u32 = c.block;
        const h: u32 = c.hidden;
        // the target's head over this rank's vocabulary rows, the best candidates, then the chain on both ranks alike
        try dv.m.dense.decode(dv.st, dv.tw.head, sc.post, .bf16, rows, sc.logits, .f32);
        // (the dense kernel writes a row w.n logits after the last: the head weight's own width is the stride here)
        try dv.k.topC(dv.st, sc.logits, dv.tw.head.n, dv.g.own_vocab, dv.g.vocab_lo, rows, cand, sc.cand_local);
        var all = sc.cand_local;
        if (dv.link) |link| {
            try link.group.allGather(dv.st, sc.cand_local, sc.cand_all, @as(usize, rows) * cand * 8);
            all = sc.cand_all;
        }
        const out = dv.results.ptr + @as(u64, req.slot) * max_block * 8;
        try dv.k.chain(dv.st, .{ .hidden = h, .rank = c.markov_rank, .cand = cand, .world = dv.world, .rows = rows }, req.anchor, all, sc.post, dv.model.markov_w1, dv.model.markov_w2, dv.model.conf_w, dv.model.conf_b, out);
    }

    fn result(p: *anyopaque, slot: u32, out: *ops.DraftResult) anyerror!void {
        const dv = of(p);
        if (slot >= dv.result_slots) return error.BadPass;
        try dv.st.synchronize();
        var raw: [max_block * 8]u8 = undefined;
        try dv.results.download(@as(usize, slot) * max_block * 8, &raw);
        out.* = .{ .n = dv.cfg.block };
        for (0..dv.cfg.block) |i| {
            out.tokens[i] = std.mem.readInt(i32, raw[8 * i ..][0..4], .little);
            out.conf[i] = @bitCast(std.mem.readInt(u32, raw[8 * i + 4 ..][0..4], .little));
        }
    }

    fn snapSave(p: *anyopaque, slot: u32, id: u32) anyerror!void {
        const dv = of(p);
        if (slot >= dv.meta.len or id >= max_snaps) return error.BadSnap;
        try dv.d.check(dv.d.api.cuMemcpyDtoDAsync_v2(dv.snaps[id].ptr, dv.rings[slot].ptr, dv.ring_bytes, dv.st.handle), "cuMemcpyDtoDAsync");
        dv.snap_meta[id] = dv.meta[slot];
    }

    fn snapRestore(p: *anyopaque, slot: u32, id: u32) anyerror!void {
        const dv = of(p);
        if (slot >= dv.meta.len or id >= max_snaps) return error.BadSnap;
        try dv.d.check(dv.d.api.cuMemcpyDtoDAsync_v2(dv.rings[slot].ptr, dv.snaps[id].ptr, dv.ring_bytes, dv.st.handle), "cuMemcpyDtoDAsync");
        dv.meta[slot] = dv.snap_meta[id];
    }

    fn snapDrop(_: *anyopaque, _: u32) void {}

    fn deinit(p: *anyopaque) void {
        const dv = of(p);
        dv.st.synchronize() catch {};
        dv.freeRings();
        if (dv.snap_made) for (&dv.snaps) |*b| b.free();
        dv.sc.buf.free();
        dv.lane.unload();
        dv.model.deinit();
        dv.gpa.destroy(dv);
    }
};

test "an absorb continues the ring or starts it over at a gap; the window is what the ring still holds" {
    const rr: u32 = 4112;
    var m: Meta = .{};
    m = afterAbsorb(m, 0, 100, rr);
    try std.testing.expectEqual(Meta{ .lo = 0, .next = 100 }, m);
    m = afterAbsorb(m, 100, 5, rr);
    try std.testing.expectEqual(Meta{ .lo = 0, .next = 105 }, m);
    m = afterAbsorb(m, 2048, 10, rr); // a gap (a resumed prompt): the ring starts over there
    try std.testing.expectEqual(Meta{ .lo = 2048, .next = 2058 }, m);
    m = afterAbsorb(.{ .lo = 0, .next = 4100 }, 4100, 16, rr); // the oldest rows are overwritten
    try std.testing.expectEqual(Meta{ .lo = 4, .next = 4116 }, m);
    // a pass reads the window up to the pending position only if the ring ends exactly there
    try std.testing.expectEqual(@as(u32, 0), passWindow(.{ .lo = 0, .next = 105 }, 105, rr));
    try std.testing.expectEqual(@as(u32, 2048), passWindow(.{ .lo = 2048, .next = 2058 }, 2058, rr));
    try std.testing.expectEqual(@as(u32, 300), passWindow(.{ .lo = 0, .next = 100 }, 300, rr)); // the ring lags: read as empty
    try std.testing.expectEqual(@as(u32, 9000 - 4112), passWindow(.{ .lo = 0, .next = 9000 }, 9000, rr));
}
