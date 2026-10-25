//! The dspark draft's weights on one rank's device: the plan's slots streamed in, matrices and `fc` repacked for qmmf.
const std = @import("std");
const cuda = @import("cuda");
const load = @import("load.zig");
const shard = @import("shard.zig");
const weights_device = @import("weights_device.zig");
const dc = @import("dspark_config.zig");
const dl = @import("dspark_load.zig");
const dn = @import("dspark_names.zig");

const qmmf = cuda.qmmf;
const mxfp8 = cuda.mxfp8;

/// One draft layer on this rank: its seven matrices packed for the lane matmul, its four small tensors as stored.
pub const Layer = struct {
    q: qmmf.Weight,
    k: qmmf.Weight,
    v: qmmf.Weight,
    o: qmmf.Weight,
    gate: qmmf.Weight,
    up: qmmf.Weight,
    down: qmmf.Weight,
    input_ln: u64,
    post_ln: u64,
    q_norm: u64,
    k_norm: u64,
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    cfg: dc.Config,
    layers: []Layer,
    fc: qmmf.Weight,
    hidden_norm: u64,
    final_norm: u64,
    markov_w1: u64,
    markov_w2: u64,
    conf_w: u64,
    conf_b: u64,
    bufs: []cuda.DeviceBuffer, // every allocation, for `deinit`
    bytes: u64, // device bytes held

    pub fn deinit(m: *Model) void {
        for (m.bufs) |*b| b.free();
        m.gpa.free(m.bufs);
        m.gpa.free(m.layers);
        m.* = undefined;
    }

    /// The matrix `proj` of layer `l`.
    pub fn lin(m: *const Model, l: usize, p: dn.Lin) qmmf.Weight {
        const ly = &m.layers[l];
        return switch (p) {
            .q => ly.q,
            .k => ly.k,
            .v => ly.v,
            .o => ly.o,
            .gate => ly.gate,
            .up => ly.up,
            .down => ly.down,
        };
    }
};

/// The sink onto per-slot device buffers (weights_device.zig's, which keeps its own private).
const Upload = struct {
    bufs: []const cuda.DeviceBuffer,
    stream: cuda.Stream,
    since: u64 = 0,

    fn put(ctx: *anyopaque, slot: u32, offset: u64, bytes: []const u8) anyerror!void {
        const u: *Upload = @ptrCast(@alignCast(ctx));
        try u.bufs[slot].uploadAsync(@intCast(offset), bytes, u.stream.handle);
        u.since += bytes.len;
        if (u.since >= weights_device.sync_every) {
            try u.stream.synchronize();
            u.since = 0;
        }
    }
};

/// The load.Plan view of a draft plan: the streamer reads its slots and items (the draft has no EXL3 matrices).
fn streamerPlan(p: *const dl.Plan) load.Plan {
    return .{ .arena = std.heap.ArenaAllocator.init(std.heap.page_allocator), .share = p.share, .slots = p.slots, .items = p.items, .matrices = &.{}, .plain = &.{}, .moe = &.{}, .ignored = p.ignored };
}

/// Plans rank `share`'s part of the draft in `dir`, streams it to the device and repacks. `cat` is the draft folder's.
pub fn loadModel(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, dir: []const u8, cfg: *const dc.Config, share: shard.Share) !Model {
    var cat = try load.Catalog.open(gpa, io, dir);
    defer cat.deinit();
    var plan = try dl.Plan.build(gpa, &cat, cfg, share, .{ .markov_cut = false });
    defer plan.deinit();
    const slots = try gpa.alloc(cuda.DeviceBuffer, plan.slots.len);
    defer gpa.free(slots);
    var made: usize = 0;
    errdefer for (slots[0..made]) |*b| b.free();
    for (plan.slots, slots) |s, *b| {
        b.* = try cuda.DeviceBuffer.alloc(d, @intCast(s.bytes));
        made += 1;
    }
    var up: Upload = .{ .bufs = slots, .stream = stream };
    var drop: weights_device.CacheDrop = .{ .cat = &cat };
    const sp = streamerPlan(&plan);
    try weights_device.feed(gpa, &cat, &sp, .{ .ctx = &up, .put = Upload.put }, drop.hook());
    try stream.synchronize();

    // what stays: the bf16 tensors' slots; what is repacked: the matrices' (their raw slots are freed once packed)
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    errdefer {
        for (keep.items) |*b| b.free();
        keep.deinit(gpa);
    }
    const raw_used = try gpa.alloc(bool, slots.len);
    defer gpa.free(raw_used);
    @memset(raw_used, false);

    var out: Model = .{ .gpa = gpa, .cfg = cfg.*, .layers = &.{}, .fc = undefined, .hidden_norm = 0, .final_norm = 0, .markov_w1 = 0, .markov_w2 = 0, .conf_w = 0, .conf_b = 0, .bufs = &.{}, .bytes = 0 };
    out.layers = try gpa.alloc(Layer, cfg.layers);
    errdefer gpa.free(out.layers);

    // plain tensors keep their slot buffers
    const T = struct {
        fn ptr(pl: *const dl.Plan, sl: []cuda.DeviceBuffer, used: []bool, role: dn.Plain, layer: u32) !u64 {
            const t = pl.tensor(role, layer) orelse return error.MissingTensor;
            used[t.slot] = true;
            return sl[t.slot].ptr;
        }
    };
    out.hidden_norm = try T.ptr(&plan, slots, raw_used, .hidden_norm, 0);
    out.final_norm = try T.ptr(&plan, slots, raw_used, .final_norm, 0);
    out.markov_w1 = try T.ptr(&plan, slots, raw_used, .markov_w1, 0);
    out.markov_w2 = try T.ptr(&plan, slots, raw_used, .markov_w2, 0);
    out.conf_w = try T.ptr(&plan, slots, raw_used, .conf_w, 0);
    out.conf_b = try T.ptr(&plan, slots, raw_used, .conf_b, 0);
    for (out.layers, 0..) |*ly, l| {
        ly.input_ln = try T.ptr(&plan, slots, raw_used, .input_ln, @intCast(l));
        ly.post_ln = try T.ptr(&plan, slots, raw_used, .post_ln, @intCast(l));
        ly.q_norm = try T.ptr(&plan, slots, raw_used, .q_norm, @intCast(l));
        ly.k_norm = try T.ptr(&plan, slots, raw_used, .k_norm, @intCast(l));
    }

    // fc: bf16 [hidden, n_taps x hidden] packed in the lane matmul's bf16 order
    {
        const t = plan.tensor(.fc, 0) orelse return error.MissingTensor;
        const n: usize = t.rows;
        const k: usize = t.cols;
        const raw = try gpa.alloc(u16, n * k);
        defer gpa.free(raw);
        try slots[t.slot].download(0, std.mem.sliceAsBytes(raw));
        const packed_ = try gpa.alloc(u16, mxfp8.padded(n) * k);
        defer gpa.free(packed_);
        qmmf.packBf16(packed_, raw, n, k);
        const b = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(packed_));
        try keep.append(gpa, b);
        out.fc = .{ .mode = .bf16, .codes = b.ptr, .scales = 0, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(mxfp8.padded(n)) };
        raw_used[t.slot] = false;
    }

    // the mxfp8 matrices
    for (out.layers, 0..) |*ly, l| {
        inline for (.{ .{ "q", dn.Lin.q }, .{ "k", dn.Lin.k }, .{ "v", dn.Lin.v }, .{ "o", dn.Lin.o }, .{ "gate", dn.Lin.gate }, .{ "up", dn.Lin.up }, .{ "down", dn.Lin.down } }) |pair| {
            const m = plan.mat(pair[1], @intCast(l)) orelse return error.MissingTensor;
            const n: usize = m.n;
            const k: usize = m.k;
            if (k % qmmf.group != 0) return error.BadShape;
            const codes = try gpa.alloc(u8, n * k);
            defer gpa.free(codes);
            const scales = try gpa.alloc(u8, n * (k / mxfp8.group));
            defer gpa.free(scales);
            try slots[m.codes].download(0, codes);
            try slots[m.scales].download(0, scales);
            const pc = try gpa.alloc(u8, mxfp8.padded(n) * k);
            defer gpa.free(pc);
            const ps = try gpa.alloc(u8, mxfp8.padded(n) * (k / mxfp8.group));
            defer gpa.free(ps);
            mxfp8.packCodes(pc, codes, n, k);
            mxfp8.packScales(ps, scales, n, k);
            const bc = try cuda.DeviceBuffer.fromHost(d, pc);
            try keep.append(gpa, bc);
            const bs = try cuda.DeviceBuffer.fromHost(d, ps);
            try keep.append(gpa, bs);
            @field(ly, pair[0]) = mxfp8.weight(bc.ptr, bs.ptr, n, k);
        }
    }
    // the raw slots of matrices and fc are done with; the bf16 tensors' slots stay
    try keep.ensureUnusedCapacity(gpa, slots.len);
    made = 0;
    for (slots, raw_used) |*b, used| {
        if (used) keep.appendAssumeCapacity(b.*) else b.free();
    }
    var total: u64 = 0;
    for (keep.items) |b| total += b.len;
    out.bufs = try keep.toOwnedSlice(gpa);
    out.bytes = total;
    return out;
}

test "the packed forms: every matrix of a layer has the lane matmul's alignment" {
    // n is a multiple of 128 (padded rows) and k of 64 for the shipped shapes, on both ranks
    const c = dc.flash;
    for ([_]usize{ 0, 1 }) |r| {
        const s = try shard.Share.of(&@import("config.zig").flash, r, 2);
        const q = s.q_heads.len() * c.head_dim;
        const kv = s.kv_heads.len() * c.head_dim;
        for ([_]usize{ q, kv, s.dense.len(), c.hidden }) |n| try std.testing.expectEqual(@as(usize, 0), n % 128);
        for ([_]usize{ c.hidden, q, s.dense.len(), c.fcIn() }) |k| try std.testing.expectEqual(@as(usize, 0), k % qmmf.group);
    }
}
