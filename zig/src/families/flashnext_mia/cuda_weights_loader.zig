//! The Flash Next weight loader: the device sink, the checkpoint reader and the per-layer dispatch.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfgs = @import("cuda_config.zig");
const lay = @import("cuda_layouts.zig");
const ropes = @import("cuda_rope.zig");
const int4 = @import("cuda_int4.zig");
const wt = @import("cuda_weights.zig");
const wblk = @import("cuda_weights_blocks.zig");
const wfp8 = @import("cuda_weights_fp8.zig");
const wexp = @import("cuda_weights_experts.zig");
const wple = @import("cuda_weights_ple.zig");

const Config = cfgs.Config;
const st = core.safetensors;
const Tensor = st.Tensor;
const Options = wt.Options;
const Weights = wt.Weights;
const Named = wt.Named;
const Rows = wt.Rows;
const Layer = wt.Layer;

// ---- the sink -----------------------------------------------------------------------------------------------------

const Out = struct {
    gpa: std.mem.Allocator,
    w: *Weights,
    cur: ?Cur = null,

    const Cur = struct { buf: cuda.DeviceBuffer, at: usize, h: std.crypto.hash.sha2.Sha256 };

    /// A buffer of `len` bytes named `name`, its bytes to follow in order through `put`.
    pub fn begin(o: *Out, name: []const u8, dtype: []const u8, shape: []const usize) !u64 {
        std.debug.assert(o.cur == null);
        var len: usize = dsize(dtype);
        for (shape) |d| len *= d;
        var b: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = len };
        if (o.w.mode == .device) {
            b = try cuda.DeviceBuffer.alloc(o.w.driver.?, len);
            errdefer b.free();
            try o.w.buffers.append(o.gpa, b);
        }
        var n: Named = .{ .name = try o.gpa.dupe(u8, name), .ptr = b.ptr, .len = len, .dtype = dtype, .rank = @intCast(shape.len) };
        @memcpy(n.shape[0..shape.len], shape);
        errdefer o.gpa.free(n.name);
        try o.w.named.append(o.gpa, n);
        o.w.bytes += len;
        o.cur = .{ .buf = b, .at = 0, .h = .init(.{}) };
        return b.ptr;
    }

    pub fn put(o: *Out, bytes: []const u8) !void {
        const c = &o.cur.?;
        if (c.at + bytes.len > c.buf.len) return error.BufferOverrun;
        switch (o.w.mode) {
            .device => try c.buf.upload(c.at, bytes),
            .hash => c.h.update(bytes),
            .count => {},
        }
        c.at += bytes.len;
    }

    pub fn end(o: *Out) !void {
        var c = o.cur.?;
        o.cur = null;
        if (c.at != c.buf.len) {
            std.log.err("{s}: {d} of {d} bytes written", .{ o.w.named.items[o.w.named.items.len - 1].name, c.at, c.buf.len });
            return error.BufferShort;
        }
        if (o.w.mode == .hash) {
            var s: [32]u8 = undefined;
            c.h.final(&s);
            o.w.named.items[o.w.named.items.len - 1].sha = s;
        }
    }

    pub fn whole(o: *Out, name: []const u8, dtype: []const u8, shape: []const usize, bytes: []const u8) !u64 {
        const p = try o.begin(name, dtype, shape);
        try o.put(bytes);
        try o.end();
        return p;
    }

    pub fn ours(o: *Out) void {
        o.w.named.items[o.w.named.items.len - 1].oracle = false;
    }
};

fn dsize(dtype: []const u8) usize {
    const two = [_][]const u8{ "bfloat16", "uint16", "int16", "float16" };
    const four = [_][]const u8{ "float32", "int32", "uint32" };
    for (two) |t| if (std.mem.eql(u8, t, dtype)) return 2;
    for (four) |t| if (std.mem.eql(u8, t, dtype)) return 4;
    if (std.mem.eql(u8, dtype, "int64")) return 8;
    return 1;
}

// ---- the loader ---------------------------------------------------------------------------------------------------

pub const Loader = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    ck: *core.Checkpoint,
    c: *const Config,
    o: Options,
    out: Out,
    prefix: []const u8 = "",
    mbase: []const u8 = "model.",
    around_one: bool = true,
    /// the shard paths in the checkpoint's file order (Checkpoint.openModel opens shardFiles' list)
    paths: []const [:0]u8 = &.{},
    /// the rope (YaRN folds its attention factor into the rotated norms' gammas: yarn.py apply)
    rope: ropes.Rope,
    host: std.ArrayList(u8) = .empty,
    ring: [16][256]u8 = undefined,
    ring_at: usize = 0,
    /// mapped ranges read since the last release
    touched: std.ArrayList(struct { file: usize, bytes: []const u8 }) = .empty,
    /// safetensors beside the checkpoint's index that only the n-gram table reads (INT4-AutoRound's ple-table/)
    extra: std.ArrayList(st.File) = .empty,
    extra_paths: std.ArrayList([:0]u8) = .empty,
    /// Options.overlay's files
    over: std.ArrayList(st.File) = .empty,

    pub fn deinit(L: *Loader) void {
        L.host.deinit(L.gpa);
        L.touched.deinit(L.gpa);
        for (L.extra.items) |*f| f.close(L.io);
        L.extra.deinit(L.gpa);
        for (L.extra_paths.items) |x| L.gpa.free(x);
        L.extra_paths.deinit(L.gpa);
        for (L.over.items) |*f| f.close(L.io);
        L.over.deinit(L.gpa);
    }

    /// Options.overlay's *.safetensors (an index there is not read: its names are the files' own).
    pub fn openOverlay(L: *Loader, dir: []const u8) !void {
        var d = try std.Io.Dir.cwd().openDir(L.io, dir, .{ .iterate = true });
        defer d.close(L.io);
        var it = d.iterate();
        while (try it.next(L.io)) |e| {
            if (e.kind != .file and e.kind != .sym_link) continue;
            if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
            const path = try std.fs.path.join(L.gpa, &.{ dir, e.name });
            defer L.gpa.free(path);
            try L.over.append(L.gpa, try st.File.open(L.gpa, L.io, path));
            std.log.info("overlay {s}", .{path});
        }
    }

    /// The .safetensors files in `dir`/ple-table when present (only the table's tensors in them are read).
    pub fn openExtra(L: *Loader, dir: []const u8) !void {
        const sub = try std.fs.path.join(L.gpa, &.{ dir, "ple-table" });
        defer L.gpa.free(sub);
        var d = std.Io.Dir.cwd().openDir(L.io, sub, .{ .iterate = true }) catch return;
        defer d.close(L.io);
        var it = d.iterate();
        while (try it.next(L.io)) |e| {
            if (e.kind != .file and e.kind != .sym_link) continue;
            if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
            const path = try std.fmt.allocPrintSentinel(L.gpa, "{s}/{s}", .{ sub, e.name }, 0);
            errdefer L.gpa.free(path);
            try L.extra.append(L.gpa, try st.File.open(L.gpa, L.io, path));
            try L.extra_paths.append(L.gpa, path);
        }
    }

    pub fn extraIndex(L: *Loader, key: []const u8) ?usize {
        for (L.extra.items, 0..) |*f, i| if (f.names.contains(key)) return i;
        return null;
    }

    pub fn hasExtra(L: *Loader, key: []const u8) bool {
        return L.extraIndex(key) != null;
    }

    /// The INT4-AutoRound lm_head (GPTQ int4, groups of 128) by vocab columns; returns the bytes for the draft head.
    pub fn int4Head(L: *Loader, w: *Weights, vl: usize) !int4.Source {
        const c = L.c;
        const d: usize = c.hidden;
        const v: usize = c.vocab;
        const qw = try L.expect("lm_head.qweight", .i32, &.{ d / 8, v });
        const sc = try L.expect("lm_head.scales", .f16, &.{ d / int4.checkpoint_group, v });
        const qz = try L.expect("lm_head.qzeros", .i32, &.{ d / int4.checkpoint_group, v / 8 });
        if (!int4.zerosAreSymmetric(qz.bytes)) {
            std.log.err("lm_head: zero points other than 8 (AutoGPTQ v1 qzeros 7); the engine reads symmetric GPTQ", .{});
            return error.UnsupportedQuantization;
        }
        const src: int4.Source = .{ .qweight = qw.bytes, .scales = sc.bytes, .n_full = v, .k_full = d };
        if (vl % 8 != 0) return error.UnexpectedTensor;
        const n0 = L.o.rank * vl;
        const words = try L.gpa.alloc(u32, int4.wordsOf(vl, d));
        defer L.gpa.free(words);
        const scales = try L.gpa.alloc(u16, int4.scalesOf(vl, d, 128));
        defer L.gpa.free(scales);
        const per: usize = 256; // n8 tiles a job
        const tiles = vl / 8;
        const Ctx = struct {
            src: int4.Source,
            n0: usize,
            d: usize,
            tiles: usize,
            words: []u32,
            scales: []u16,
            failed: *std.atomic.Value(bool),
            fn run(q: @This(), job: usize) void {
                const t0 = job * per;
                const t1 = @min(t0 + per, q.tiles);
                const sl: int4.Slice = .{ .n0 = q.n0 + t0 * 8, .n = (t1 - t0) * 8, .k0 = 0, .k = q.d, .gs = 128 };
                int4.packWords(q.src, sl, q.words[t0 * q.d ..][0 .. (t1 - t0) * q.d]) catch q.failed.store(true, .monotonic);
                int4.packScales(q.src, sl, q.scales[t0 * 8 * (q.d / 128) ..][0 .. (t1 - t0) * 8 * (q.d / 128)]) catch q.failed.store(true, .monotonic);
            }
        };
        var failed = std.atomic.Value(bool).init(false);
        parallel(L.o.threads, (tiles + per - 1) / per, Ctx{ .src = src, .n0 = n0, .d = d, .tiles = tiles, .words = words, .scales = scales, .failed = &failed }, Ctx.run);
        if (failed.load(.monotonic)) return error.UnexpectedTensor;
        var m: int4.Mat = .{ .n = @intCast(vl), .k = @intCast(d), .gs = 128 };
        m.w = try L.out.whole("head4.weight", "int32", &.{words.len}, std.mem.sliceAsBytes(words));
        L.out.ours();
        m.s = try L.out.whole("head4.scales", "float16", &.{scales.len}, std.mem.sliceAsBytes(scales));
        L.out.ours();
        w.head = .{ .weight = 0, .n = @intCast(vl), .k = @intCast(d) };
        w.head4 = m;
        return src;
    }

    pub fn staging(L: *Loader, bytes: usize) ![]u8 {
        try L.host.resize(L.gpa, bytes);
        return L.host.items;
    }

    pub fn has(L: *Loader, name: []const u8) bool {
        for (L.over.items) |*f| if (f.names.contains(name)) return true;
        for (L.ck.files.items) |*f| if (f.names.contains(name)) return true;
        return false;
    }

    /// The tensor `name` (after the prefix), marked used, pages released after the layer; an overlay's tensor wins.
    pub fn get(L: *Loader, name: []const u8) !Tensor {
        var buf: [256]u8 = undefined;
        const full = try std.fmt.bufPrint(&buf, "{s}{s}", .{ L.prefix, name });
        for (L.over.items) |*f| if (f.get(full)) |t| {
            for (L.ck.files.items) |*cf| if (cf.names.getKey(full)) |k| try L.ck.used.put(L.gpa, k, {});
            return t;
        };
        for (L.ck.files.items, 0..) |*f, i| if (f.get(full)) |t| {
            try L.ck.used.put(L.gpa, f.names.getKey(full).?, {});
            try L.touched.append(L.gpa, .{ .file = i, .bytes = t.bytes });
            return t;
        };
        std.log.err("checkpoint has no tensor {s}", .{full});
        return error.MissingTensor;
    }

    pub fn expect(L: *Loader, name: []const u8, dtype: st.DType, shape: []const usize) !Tensor {
        const t = try L.get(name);
        if (!t.is(dtype, shape)) {
            std.log.err("{s}{s}: {t} {any}, expected {t} {any}", .{ L.prefix, name, t.dtype, t.shape[0..t.rank], dtype, shape });
            return error.UnexpectedTensor;
        }
        return t;
    }

    /// A bf16 linear [n, k] (`_plain`: quantized bytes where real values are read are refused).
    pub fn linear(L: *Loader, name: []const u8, n: usize, k: usize) !Tensor {
        var b: [256]u8 = undefined;
        const wn = try std.fmt.bufPrint(&b, "{s}.weight", .{name});
        const t = try L.get(wn);
        if (t.dtype != .bf16) {
            std.log.err("{s}{s}: {t} weights; this engine reads Flash Next's non-expert linears as bf16", .{ L.prefix, wn, t.dtype });
            return error.UnsupportedQuantization;
        }
        if (!t.is(.bf16, &.{ n, k })) {
            std.log.err("{s}{s}: {any}, expected [{d}, {d}]", .{ L.prefix, wn, t.shape[0..t.rank], n, k });
            return error.UnexpectedTensor;
        }
        return t;
    }

    /// Drop the pages of what the last layer read (madvise, then fadvise: the page cache never holds both copies).
    pub fn release(L: *Loader) void {
        defer L.touched.clearRetainingCapacity();
        if (!L.o.release_pages) return;
        const page: usize = 64 << 10;
        for (L.touched.items) |t| {
            const f = &L.ck.files.items[t.file];
            const base = @intFromPtr(f.map.memory.ptr);
            const lo = std.mem.alignForward(usize, @intFromPtr(t.bytes.ptr), page);
            const hi = std.mem.alignBackward(usize, @intFromPtr(t.bytes.ptr) + t.bytes.len, page);
            if (hi <= lo) continue;
            const p: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(lo);
            std.posix.madvise(p, hi - lo, std.posix.MADV.DONTNEED) catch {};
            _ = std.os.linux.fadvise(f.file.handle, @intCast(lo - base), @intCast(hi - lo), std.os.linux.POSIX_FADV.DONTNEED);
        }
    }

    /// A checkpoint name in a ring of 16 buffers: valid for the next 15 calls, so it may be an argument of the next.
    pub fn nameOf(L: *Loader, comptime fmt: []const u8, args: anytype) ![]const u8 {
        L.ring_at = (L.ring_at + 1) % L.ring.len;
        return std.fmt.bufPrint(&L.ring[L.ring_at], fmt, args);
    }

    // -- small tensors --

    /// cscale: a norm's weight as fp32, plus one where the checkpoint stores them centred (around zero).
    pub fn cscale(L: *Loader, path: []const u8, name: []const u8, n: usize) !u64 {
        return L.cscaleRope(path, name, n, false);
    }

    /// cscale, with `rotated` folding YaRN's attention factor into rotary dims (cuda_rope.foldGamma).
    pub fn cscaleRope(L: *Loader, path: []const u8, name: []const u8, n: usize, rotated: bool) !u64 {
        const t = try L.expect(name, .bf16, &.{n});
        const host: []f32 = @alignCast(std.mem.bytesAsSlice(f32, try L.staging(n * 4)));
        for (host, 0..) |*h, i| {
            const v = lay.bf16ToF32(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
            h.* = if (L.around_one) v else 1.0 + v;
        }
        if (rotated) try ropes.foldGamma(host, L.rope.rotary_dim, L.rope.scale());
        return L.out.whole(path, "float32", &.{n}, L.host.items);
    }

    /// `.float()` of a bf16 vector's elements [lo, hi).
    pub fn widen(L: *Loader, path: []const u8, t: Tensor, lo: usize, hi: usize) !u64 {
        if (t.dtype != .bf16) return error.UnexpectedTensor;
        const host = std.mem.bytesAsSlice(f32, try L.staging((hi - lo) * 4));
        for (host, lo..) |*h, i| h.* = lay.bf16ToF32(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
        return L.out.whole(path, "float32", &.{hi - lo}, L.host.items);
    }

    // -- faces: bf16 rows stacked in order --

    pub const Part = struct { t: Tensor, rows: []const [2]usize = &.{}, cols: ?[2]usize = null };

    /// Several linears of one input as one [n, k] bf16 matrix (stack_b16); `rows` ranges per part, `cols` the TP split.
    pub fn face(L: *Loader, path: []const u8, parts: []const Part) !Rows {
        const k_all = parts[0].t.dim(1);
        const k = if (parts[0].cols) |c| c[1] - c[0] else k_all;
        var n: usize = 0;
        for (parts) |p| {
            if (p.t.dtype != .bf16 or p.t.dim(1) != k_all) return error.UnexpectedTensor;
            if (p.rows.len == 0) {
                n += p.t.dim(0);
            } else for (p.rows) |r| {
                n += r[1] - r[0];
            }
        }
        const ptr = try L.out.begin(path, "bfloat16", &.{ n, k });
        for (parts) |p| {
            const all = [_][2]usize{.{ 0, p.t.dim(0) }};
            const ranges = if (p.rows.len == 0) &all else p.rows;
            for (ranges) |r| {
                if (p.cols) |c| {
                    const host = try L.staging((r[1] - r[0]) * k * 2);
                    for (r[0]..r[1], 0..) |row, i| @memcpy(host[i * k * 2 ..][0 .. k * 2], p.t.bytes[(row * k_all + c[0]) * 2 ..][0 .. k * 2]);
                    try L.out.put(host);
                } else {
                    try L.out.put(p.t.bytes[r[0] * k_all * 2 .. r[1] * k_all * 2]);
                }
            }
        }
        try L.out.end();
        return .{ .weight = ptr, .n = @intCast(n), .k = @intCast(k) };
    }

    pub fn b16(L: *Loader, path: []const u8, name: []const u8, n: usize, k: usize) !Rows {
        return L.face(path, &.{.{ .t = try L.linear(name, n, k) }});
    }

    pub const hc = wblk.hc;
    pub const gdn = wblk.gdn;
    pub const attention = wblk.attention;
    pub const moe = wblk.moe;
    pub const fp4Table = wblk.fp4Table;
    pub const isFp8 = wfp8.isFp8;
    pub const fp8Face = wfp8.fp8Face;
    pub const gdn8 = wfp8.gdn8;
    pub const attention8 = wfp8.attention8;
    pub const moe8 = wfp8.moe8;
    pub const int4Experts = wexp.int4Experts;
    pub const packRouted = wexp.packRouted;
    pub const mtpExperts = wexp.mtpExperts;
    pub const ple = wple.ple;
    pub const openTable = wple.openTable;
    pub const tableToGpu = wple.tableToGpu;

    pub fn layer(L: *Loader, w: *Weights, path: []const u8, i: i32, base: []const u8, kind: cfgs.LayerType, with_ple: bool) !Layer {
        var lw: Layer = .{ .index = i, .linear = kind == .linear };
        var pb: [192]u8 = undefined;
        var nb: [192]u8 = undefined;
        lw.attn_hc = try L.hc(try std.fmt.bufPrint(&pb, "{s}.attn_hc", .{path}), try std.fmt.bufPrint(&nb, "{s}.attn_hyper_connection", .{base}), true);
        lw.mlp_hc = try L.hc(try std.fmt.bufPrint(&pb, "{s}.mlp_hc", .{path}), try std.fmt.bufPrint(&nb, "{s}.mlp_hyper_connection", .{base}), true);
        // INT4-AutoRound: block-FP8 mixers, GPTQ int4 experts with the block-FP8 shared expert (the MTP layer is bf16)
        const int4ar = L.c.int4ar() and i >= 0;
        if (kind == .linear) {
            const nm = try std.fmt.bufPrint(&nb, "{s}.linear_attn", .{base});
            lw.gdn = if (int4ar and try L.isFp8(try L.nameOf("{s}.in_proj_qkv", .{nm}))) try L.gdn8(try std.fmt.bufPrint(&pb, "{s}.gdn", .{path}), nm) else try L.gdn(try std.fmt.bufPrint(&pb, "{s}.gdn", .{path}), nm);
        } else {
            const nm = try std.fmt.bufPrint(&nb, "{s}.self_attn", .{base});
            lw.attn = if (int4ar and try L.isFp8(try L.nameOf("{s}.q_proj", .{nm}))) try L.attention8(try std.fmt.bufPrint(&pb, "{s}.attn", .{path}), nm) else try L.attention(try std.fmt.bufPrint(&pb, "{s}.attn", .{path}), nm);
        }
        lw.moe = if (int4ar) try L.moe8(try std.fmt.bufPrint(&pb, "{s}.moe", .{path}), try std.fmt.bufPrint(&nb, "{s}.mlp", .{base})) else try L.moe(try std.fmt.bufPrint(&pb, "{s}.moe", .{path}), try std.fmt.bufPrint(&nb, "{s}.mlp", .{base}), i < 0);
        if (with_ple and i >= 0) if (L.c.pleIndex(@intCast(i))) |pi| {
            lw.ple = try L.ple(try std.fmt.bufPrint(&pb, "{s}.ple", .{path}), try std.fmt.bufPrint(&nb, "{s}.ple", .{base}), pi, w);
        };
        return lw;
    }
};

/// `f(ctx, i)` for i in [0, n) over up to `threads` threads (the caller's among them).
pub fn parallel(threads: u32, n: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize) void) void {
    var next = std.atomic.Value(usize).init(0);
    const W = struct {
        fn go(c: @TypeOf(ctx), counter: *std.atomic.Value(usize), total: usize) void {
            while (true) {
                const i = counter.fetchAdd(1, .monotonic);
                if (i >= total) return;
                f(c, i);
            }
        }
    };
    var pool: [64]?std.Thread = @splat(null);
    const extra = @min(@as(usize, threads), pool.len, n) -| 1;
    for (pool[0..extra]) |*t| t.* = std.Thread.spawn(.{}, W.go, .{ ctx, &next, n }) catch null;
    W.go(ctx, &next, n);
    for (pool[0..extra]) |t| if (t) |th| th.join();
}
