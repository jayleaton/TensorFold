//! tf-cuda-test mm3-dspark-pass [DIR]: the dspark absorb and block pass on the GPU against dspark_ref in float64.
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("minimax_cuda");
const check = @import("check.zig");

const dc = mm.dspark_config;
const dn = mm.dspark_names;
const dd = mm.dspark_device;
const ref = mm.dspark_ref;

const layers = 2;
const mask_token = 3;
const anchor: u32 = 5;
const first_pos: u32 = 4100; // the ring wraps (4112 slots) inside the first absorb
const n_first = 24;
const n_second = 8;

const Entry = struct { name: []const u8, dtype: []const u8, shape: [2]usize, rank: usize, bytes: usize, at: usize = 0 };

/// The synthetic folder's tensors: names and shapes of the real pack at `layers` layers.
const Image = struct {
    gpa: std.mem.Allocator,
    entries: std.ArrayList(Entry) = .empty,
    names: std.ArrayList([]u8) = .empty,
    total: usize = 0,
    data: []u8 = &.{},
    head: usize = 0,

    fn add(im: *Image, name: []const u8, dtype: []const u8, esize: usize, shape: []const usize) !void {
        const owned = try im.gpa.dupe(u8, name);
        try im.names.append(im.gpa, owned);
        var e: Entry = .{ .name = owned, .dtype = dtype, .shape = .{ shape[0], if (shape.len > 1) shape[1] else 1 }, .rank = shape.len, .bytes = esize * shape[0] * (if (shape.len > 1) shape[1] else 1) };
        e.at = im.total;
        im.total += e.bytes;
        try im.entries.append(im.gpa, e);
    }

    fn deinit(im: *Image) void {
        for (im.names.items) |n| im.gpa.free(n);
        im.names.deinit(im.gpa);
        im.entries.deinit(im.gpa);
        im.gpa.free(im.data);
    }

    fn slice(im: *const Image, name: []const u8) []u8 {
        for (im.entries.items) |e| if (std.mem.eql(u8, e.name, name)) return im.data[im.head + e.at ..][0..e.bytes];
        unreachable;
    }

    /// The header, then the data allocated: `slice` hands out each tensor's bytes.
    fn build(im: *Image) !void {
        var h: std.Io.Writer.Allocating = .init(im.gpa);
        defer h.deinit();
        try h.writer.writeAll("{\"__metadata__\":{\"format\":\"pt\"}");
        for (im.entries.items) |e| {
            try h.writer.print(",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d}", .{ e.name, e.dtype, e.shape[0] });
            if (e.rank > 1) try h.writer.print(",{d}", .{e.shape[1]});
            try h.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ e.at, e.at + e.bytes });
        }
        try h.writer.writeAll("}");
        const header = h.written();
        im.head = 8 + header.len;
        im.data = try im.gpa.alloc(u8, im.head + im.total);
        std.mem.writeInt(u64, im.data[0..8], header.len, .little);
        @memcpy(im.data[8..][0..header.len], header);
    }
};

fn bf16(v: f32) u16 {
    return mm_toBf16(v);
}

fn mm_toBf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn fromBf16(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

/// The draft's config for `layers` layers and mask token 3, from the shipped one (every copy of a setting alike).
fn config(gpa: std.mem.Allocator) ![]u8 {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, dc.fixture, .{});
    defer parsed.deinit();
    patch(parsed.value);
    return std.json.Stringify.valueAlloc(gpa, parsed.value, .{});
}

fn patch(v: std.json.Value) void {
    switch (v) {
        .object => |o| {
            for (o.keys(), o.values()) |k, *x| {
                if (std.mem.eql(u8, k, "num_hidden_layers") and x.* == .integer and x.integer == 5) x.* = .{ .integer = layers };
                if (std.mem.eql(u8, k, "layer_types") and x.* == .array and x.array.items.len == 5) x.array.shrinkRetainingCapacity(layers);
                if (x.* == .integer and x.integer == 200058) x.* = .{ .integer = mask_token };
                patch(x.*);
            }
        },
        .array => |a| for (a.items) |x| patch(x),
        else => {},
    }
}

fn fillBf16(rng: std.Random, out: []u8, scale: f32, bias: f32) void {
    for (0..out.len / 2) |i| std.mem.writeInt(u16, out[2 * i ..][0..2], bf16(bias + (rng.float(f32) - 0.5) * 2 * scale), .little);
}

/// The bf16 values of little-endian bytes (a tensor inside the image is not aligned for u16).
fn bf16List(gpa: std.mem.Allocator, bytes: []const u8) ![]f64 {
    const out = try gpa.alloc(f64, bytes.len / 2);
    for (out, 0..) |*o, i| o.* = fromBf16(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
    return out;
}

/// e4m3 codes of magnitude 0.5 to 5 and ue8m0 scales 2^-9 .. 2^-5: weights of about 0.01 to 0.1.
fn fillCodes(rng: std.Random, out: []u8) void {
    for (out) |*c| c.* = (if (rng.boolean()) @as(u8, 0x80) else 0) | (0x30 + rng.uintLessThan(u8, 0x15));
}

fn fillScales(rng: std.Random, out: []u8) void {
    for (out) |*s| s.* = 118 + rng.uintLessThan(u8, 5);
}

fn dequant(gpa: std.mem.Allocator, codes: []const u8, scales: []const u8, n: usize, k: usize) ![]f64 {
    const w = try gpa.alloc(f64, n * k);
    ref.dequant(codes, scales, n, k, 32, 127, w);
    return w;
}

fn tapOf(gpa: std.mem.Allocator, h: []const u16) ![]f64 {
    const out = try gpa.alloc(f64, h.len);
    for (out, h) |*o, x| o.* = fromBf16(x);
    return out;
}

/// Relative L2 error of `got` against `want` and the largest element error over the rms of `want`.
const Err = struct { l2: f64, worst: f64 };

fn err(got: []const f64, want: []const f64) Err {
    var num: f64 = 0;
    var den: f64 = 0;
    for (got, want) |g, w| {
        num += (g - w) * (g - w);
        den += w * w;
    }
    const rms = @sqrt(den / @as(f64, @floatFromInt(want.len)));
    var worst: f64 = 0;
    for (got, want) |g, w| worst = @max(worst, @abs(g - w) / @max(rms, 1e-30));
    return .{ .l2 = @sqrt(num / @max(den, 1e-300)), .worst = worst };
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const base: []const u8 = if (rest.len > 0) rest[0] else "dspark-synth";
    const text = try config(gpa);
    defer gpa.free(text);
    const cfg = try dc.parse(gpa, text);
    if (cfg.layers != layers or cfg.mask_token != mask_token) return error.BadSyntheticConfig;
    const share = try mm.shard.Share.of(&mm.config.flash, 0, 1);
    var prng = std.Random.DefaultPrng.init(0x6d33_d5a1);
    const rng = prng.random();

    // ---- the folder
    var im: Image = .{ .gpa = gpa };
    defer im.deinit();
    var nb: dn.Buf = undefined;
    const h: usize = cfg.hidden;
    const qw: usize = cfg.qWidth();
    const kvw: usize = cfg.kvWidth();
    const ffn: usize = cfg.ffn;
    const plain = [_]struct { role: dn.Plain, shape: []const usize }{
        .{ .role = .fc, .shape = &.{ h, cfg.fcIn() } },
        .{ .role = .hidden_norm, .shape = &.{h} },
        .{ .role = .final_norm, .shape = &.{h} },
        .{ .role = .markov_w1, .shape = &.{ cfg.vocab, cfg.markov_rank } },
        .{ .role = .markov_w2, .shape = &.{ cfg.vocab, cfg.markov_rank } },
        .{ .role = .conf_w, .shape = &.{ 1, cfg.confIn() } },
        .{ .role = .conf_b, .shape = &.{1} },
    };
    for (plain) |p| try im.add(dn.plain(&nb, p.role, 0), "BF16", 2, p.shape);
    for (0..layers) |l| {
        for ([_]struct { role: dn.Plain, n: usize }{ .{ .role = .input_ln, .n = h }, .{ .role = .post_ln, .n = h }, .{ .role = .q_norm, .n = 128 }, .{ .role = .k_norm, .n = 128 } }) |p|
            try im.add(dn.plain(&nb, p.role, @intCast(l)), "BF16", 2, &.{p.n});
        for ([_]struct { p: dn.Lin, n: usize, k: usize }{ .{ .p = .q, .n = qw, .k = h }, .{ .p = .k, .n = kvw, .k = h }, .{ .p = .v, .n = kvw, .k = h }, .{ .p = .o, .n = h, .k = qw }, .{ .p = .gate, .n = ffn, .k = h }, .{ .p = .up, .n = ffn, .k = h }, .{ .p = .down, .n = h, .k = ffn } }) |m| {
            try im.add(dn.lin(&nb, m.p, @intCast(l), .weight), "F8_E4M3", 1, &.{ m.n, m.k });
            try im.add(dn.lin(&nb, m.p, @intCast(l), .weight_scale_inv), "U8", 1, &.{ m.n, m.k / 32 });
        }
    }
    try im.build();
    for (plain) |p| {
        const s = im.slice(dn.plain(&nb, p.role, 0));
        switch (p.role) {
            .fc => fillBf16(rng, s, 0.01, 0),
            .hidden_norm, .final_norm => fillBf16(rng, s, 0.05, 0.03),
            else => @memset(s, 0),
        }
    }
    for (0..layers) |l| {
        for ([_]dn.Plain{ .input_ln, .post_ln, .q_norm, .k_norm }) |role| fillBf16(rng, im.slice(dn.plain(&nb, role, @intCast(l))), 0.1, 0);
        for ([_]dn.Lin{ .q, .k, .v, .o, .gate, .up, .down }) |p| {
            fillCodes(rng, im.slice(dn.lin(&nb, p, @intCast(l), .weight)));
            fillScales(rng, im.slice(dn.lin(&nb, p, @intCast(l), .weight_scale_inv)));
        }
    }
    const dir = try std.fmt.allocPrint(gpa, "{s}/draft", .{base});
    defer gpa.free(dir);
    try std.Io.Dir.cwd().createDirPath(io, dir);
    defer std.Io.Dir.cwd().deleteTree(io, base) catch {};
    const shard_path = try std.fmt.allocPrint(gpa, "{s}/model-dspark-00001.safetensors", .{dir});
    defer gpa.free(shard_path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = shard_path, .data = im.data });

    // ---- the target side the draft borrows: kernels, the embedding rows of two ids, a geometry
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var modules = try mm.forward.Modules.load(gpu.d);
    defer modules.deinit();
    const geometry = try mm.forward.Geometry.of(&mm.config.flash, share, 2, mm.iface.max_rows);
    const embed_host = try gpa.alloc(u16, 8 * h);
    defer gpa.free(embed_host);
    for (embed_host) |*e| e.* = bf16((rng.float(f32) - 0.5) * 2);
    var embed = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(embed_host));
    defer embed.free();
    const tw: mm.weights.Weights = .{ .embed = embed.ptr, .final_norm = 0, .head = std.mem.zeroes(cuda.exl3.Weight), .layers = &.{} };

    const dev = try dd.Device.open(gpa, io, gpu.d, gpu.ctx, stream, dir, cfg, share, &modules, &tw, &geometry, null);
    var iface_ = dev.interface();
    defer iface_.vtable.deinit(dev);
    std.debug.print("INFO draft on the device: {d:.2} GiB\n", .{@as(f64, @floatFromInt(dev.bytes())) / (1 << 30)});
    try iface_.attach(1);

    // ---- the taps of n rows (bf16 values the target would have left), absorbed in two goes
    const n_all = n_first + n_second;
    const fcin = cfg.fcIn();
    const feats16 = try gpa.alloc(u16, n_all * fcin);
    defer gpa.free(feats16);
    for (feats16) |*e| e.* = bf16(@as(f32, @floatCast(rng.floatNorm(f64))));
    var taps = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(feats16));
    defer taps.free();
    var t0 = check.now(io);
    try iface_.absorb(0, first_pos, n_first, taps.ptr, 0);
    try stream.synchronize();
    std.debug.print("INFO absorb of {d} rows: {d:.1} ms\n", .{ n_first, @as(f64, @floatFromInt(check.now(io) - t0)) / 1e6 });
    try iface_.absorb(0, first_pos + n_first, n_second, taps.ptr, n_first);
    const pos = first_pos + n_all;
    t0 = check.now(io);
    try dev.block(.{ .slot = 0, .anchor = anchor, .pos = pos });
    try stream.synchronize();
    std.debug.print("INFO block pass ({d} layers, 7 rows, {d} context rows): {d:.1} ms\n", .{ layers, n_all, @as(f64, @floatFromInt(check.now(io) - t0)) / 1e6 });

    // ---- float64 reference over the same bytes
    const rl = try gpa.alloc(ref.Layer, layers);
    defer gpa.free(rl);
    var rc: ref.Config = .{ .layers = layers, .mask_token = mask_token, .sw = .{ .base = .{ .bf16_norm_out = true, .bf16_proj_out = true, .bf16_rope_out = true, .bf16_attn_out = true, .bf16_act_out = true } } };
    rc.vocab = 8;
    const norm = bf16List;
    for (rl, 0..) |*r, l| {
        const L: u32 = @intCast(l);
        r.input_norm = try norm(gpa, im.slice(dn.plain(&nb, .input_ln, L)));
        r.post_norm = try norm(gpa, im.slice(dn.plain(&nb, .post_ln, L)));
        r.q_norm = try norm(gpa, im.slice(dn.plain(&nb, .q_norm, L)));
        r.k_norm = try norm(gpa, im.slice(dn.plain(&nb, .k_norm, L)));
        inline for (.{ .{ "q", dn.Lin.q, qw, h }, .{ "k", dn.Lin.k, kvw, h }, .{ "v", dn.Lin.v, kvw, h }, .{ "o", dn.Lin.o, h, qw }, .{ "gate", dn.Lin.gate, ffn, h }, .{ "up", dn.Lin.up, ffn, h }, .{ "down", dn.Lin.down, h, ffn } }) |m| {
            const w = try dequant(gpa, im.slice(dn.lin(&nb, m[1], L, .weight)), im.slice(dn.lin(&nb, m[1], L, .weight_scale_inv)), m[2], m[3]);
            @field(r, m[0]) = .{ .n = m[2], .k = m[3], .w = w };
        }
    }
    const fc_w = try norm(gpa, im.slice(dn.plain(&nb, .fc, 0)));
    const embed_w = try tapOf(gpa, embed_host);
    const feats = try tapOf(gpa, feats16);
    const rd: ref.Draft = .{
        .cfg = rc,
        .fc = .{ .n = h, .k = fcin, .w = fc_w },
        .hidden_norm = try norm(gpa, im.slice(dn.plain(&nb, .hidden_norm, 0))),
        .final_norm = try norm(gpa, im.slice(dn.plain(&nb, .final_norm, 0))),
        .layers = rl,
        .markov_w1 = &.{},
        .markov_w2 = &.{},
        .conf_w = &.{},
        .conf_b = 0,
        .embed = embed_w,
        .head = .{ .n = 0, .k = h, .w = &.{} },
    };
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const cx = try ref.buildCtx(&rd, arena.allocator(), feats, first_pos, n_all);
    const blk = try ref.blockForward(&rd, arena.allocator(), &cx, anchor, pos);

    // ---- compare: the ring (every layer's K and V, slot = position mod ring rows), then the block's final rows
    var worst_ring: Err = .{ .l2 = 0, .worst = 0 };
    const ring_host = try gpa.alloc(u8, dev.ring_bytes);
    defer gpa.free(ring_host);
    try dev.rings[0].download(0, ring_host);
    const want_kv = try gpa.alloc(f64, n_all * kvw);
    defer gpa.free(want_kv);
    const got_kv = try gpa.alloc(f64, n_all * kvw);
    defer gpa.free(got_kv);
    for (0..layers) |l| for ([_]usize{ 0, 1 }) |kind| {
        const base_off = (@as(usize, dev.ringAt(0, l, kind))); // offset of the layer's K or V ring in the buffer
        const rows_b = ring_host[base_off..][0 .. @as(usize, dev.ring_rows) * kvw * 2];
        for (0..n_all) |j| {
            const slot = (first_pos + j) % dev.ring_rows;
            for (0..kvw) |e| got_kv[j * kvw + e] = fromBf16(std.mem.readInt(u16, rows_b[2 * (slot * kvw + e) ..][0..2], .little));
        }
        @memcpy(want_kv, if (kind == 0) cx.k[l][0 .. n_all * kvw] else cx.v[l][0 .. n_all * kvw]);
        const e = err(got_kv, want_kv);
        std.debug.print("INFO ring layer {d} {s}: rel L2 {e:.2}, worst element {e:.2} of rms\n", .{ l, if (kind == 0) "K" else "V", e.l2, e.worst });
        worst_ring.l2 = @max(worst_ring.l2, e.l2);
        worst_ring.worst = @max(worst_ring.worst, e.worst);
    };
    var post_raw: [7 * 6144 * 2]u8 = undefined;
    const post_buf: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = dev.sc.post, .len = post_raw.len };
    try post_buf.download(0, &post_raw);
    const got_post = try bf16List(gpa, &post_raw);
    defer gpa.free(got_post);
    const e_post = err(got_post, blk.post);
    std.debug.print("INFO block rows after final_norm: rel L2 {e:.2}, worst element {e:.2} of rms\n", .{ e_post.l2, e_post.worst });
    try check.expect(worst_ring.l2 < 0.02, "mm3-dspark-pass: the K/V ring differs from float64 (rel L2 {e:.2})", .{worst_ring.l2});
    try check.expect(e_post.l2 < 0.03 and e_post.worst < 0.25, "mm3-dspark-pass: the block rows differ from float64 (rel L2 {e:.2}, worst {e:.2})", .{ e_post.l2, e_post.worst });
    check.pass("mm3-dspark-pass: ring rel L2 {e:.2}, block rows rel L2 {e:.2} against float64 ({d} layers, ring wrap, two absorbs)", .{ worst_ring.l2, e_post.l2, layers });
}
