//! Flash Next weights in Python's GPU layouts, TP rank-sliced; from weights.py et al. by Ash Hart, tournierjc et al.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfgs = @import("cuda_config.zig");
const lay = @import("cuda_layouts.zig");
const ngram = @import("cuda_ngram.zig");
const ropes = @import("cuda_rope.zig");
const fp8 = @import("cuda_fp8.zig");
const int4 = @import("cuda_int4.zig");
const wl = @import("cuda_weights_loader.zig");

const Config = cfgs.Config;
const st = core.safetensors;
const Tensor = st.Tensor;

const Loader = wl.Loader;
const parallel = wl.parallel;

pub const Mode = enum { device, hash, count };

/// The default draft vocabulary (qwen4_exp/cuda/draft_vocab.txt, 79,591 ids), embedded.
pub const draft_vocab = @embedFile("cuda_draft_vocab.txt");

/// What the loader takes beside the checkpoint and its Config.
pub const Options = struct {
    rank: u32 = 0,
    world: u32 = 1,
    mtp: bool = true,
    /// the MTP drafts' head over the default draft vocabulary (Python engine draft_vocab "default")
    draft_head: bool = true,
    /// the n-gram table's rows on the GPU (a rank's heads); null: on the GPU at two ranks, host-mapped at one
    ngram_on_gpu: ?bool = null,
    mode: Mode = .device,
    /// the CUDA driver (its context current) for `.device`
    driver: ?*const cuda.Driver = null,
    /// a directory whose safetensors override the checkpoint's same-named tensors (INT4-AutoRound's fast-fp8/)
    overlay: ?[]const u8 = null,
    threads: u32 = 16,
    /// drop the checkpoint's pages once a layer's tensors are on the device (Python's reader.release)
    release_pages: bool = true,
};

/// A bf16 matrix [n, k] as stored (Python bf16._Routed, its B16 at `.b`: named "<path>.b.weight").
pub const Rows = struct { weight: u64 = 0, n: u32 = 0, k: u32 = 0 };
/// An FP4 table in the pattern form: bf16 bits [n/64][k/64][64][64], fp32 scales [k/16, n], fp32 scale2 [n].
pub const Fp4 = struct { weight: u64 = 0, scale: u64 = 0, scale2: u64 = 0, n: u32 = 0, k: u32 = 0 };
pub const Expert4 = struct { gu: Fp4 = .{}, down: Fp4 = .{} };
/// Experts4: up [E, NI/32, D/32, 2, 144] int32, down [E, D/32, NI/32, 1, 144], fp32 up_scale [E, 2], down_scale [E, 1].
pub const Experts4 = struct { up: u64 = 0, down: u64 = 0, up_scale: u64 = 0, down_scale: u64 = 0, count: u32 = 0, width: u32 = 0, dims: u32 = 0 };
/// nvfp4_moe.MoE4 with MoEW's router: [E + 1, D] bf16, the shared expert's gate row last.
pub const MoE = struct { router: u64 = 0, routed: Experts4 = .{}, shared: Expert4 = .{}, int4: ?int4.Experts = null, shared8: ?Shared8 = null };
/// INT4-AutoRound's fast-fp8 variant: `down8` / `up8` block FP8 (then `down` holds only the bf16 inject rows).
pub const Hc = struct {
    down: Rows = .{},
    up: Rows = .{},
    scale: u64 = 0,
    inject: bool = false,
    down8: ?fp8.Linear = null,
    up8: ?fp8.Linear = null,

    /// The down projection's outputs: the low-rank rows, then the inject rows.
    pub fn downN(h: Hc) u32 {
        return h.down.n + if (h.down8) |d| d.n else 0;
    }
};
/// INT4-AutoRound: `proj8` leading block-FP8 projection columns, `proj` the bf16 rows after; `out8`/`o8` replace out/o.
pub const Gdn = struct { proj: Rows = .{}, conv: u64 = 0, a_log: u64 = 0, dt_bias: u64 = 0, norm: u64 = 0, out: Rows = .{}, proj8: ?fp8.Linear = null, out8: ?fp8.Linear = null };
pub const Attn = struct { proj: Rows = .{}, q_scale: u64 = 0, k_scale: u64 = 0, iq_scale: u64 = 0, ik_scale: u64 = 0, o: Rows = .{}, proj8: ?fp8.Linear = null, o8: ?fp8.Linear = null };
/// INT4-AutoRound's healed shared expert: block FP8 gate|up rows [2 w, D] and down [D, w] (w its rank's width).
pub const Shared8 = struct { gu: fp8.Linear, down: fp8.Linear, width: u32 };
pub const Ple = struct { key: Rows = .{}, value: Rows = .{}, norm_key: u64 = 0, norm_query: u64 = 0, norm_conv: u64 = 0, conv: u64 = 0, ngram: ngram.NGram };
pub const Layer = struct { index: i32, linear: bool, attn_hc: Hc = .{}, mlp_hc: Hc = .{}, gdn: ?Gdn = null, attn: ?Attn = null, moe: MoE = .{}, ple: ?Ple = null };
pub const Mtp = struct { norm_e: u64 = 0, norm_h: u64 = 0, fc_e: Rows = .{}, fc_h: Rows = .{}, layer: Layer, mixer: Hc = .{} };
/// qmm.Q4 in the lane matmul's frag layout: int32 [npad/64][k/32][8][32], bf16 scales and biases [k/32, npad].
pub const Q4 = struct { weight: u64 = 0, scales: u64 = 0, biases: u64 = 0, n: u32 = 0, k: u32 = 0, npad: u32 = 0 };

/// A device buffer by the Python engine's dotted name (`oracle` false: ours alone, no Python counterpart).
pub const Named = struct {
    name: []u8,
    ptr: u64,
    len: usize,
    dtype: []const u8,
    shape: [6]usize = @splat(1),
    rank: u8,
    sha: ?[32]u8 = null,
    oracle: bool = true,

    /// "AxB:dtype", as the oracle's weights.json ".shape" entries.
    pub fn shapeText(self: Named, buf: []u8) ![]const u8 {
        var w: std.Io.Writer = .fixed(buf);
        for (self.shape[0..self.rank], 0..) |d, i| try w.print("{s}{d}", .{ if (i == 0) "" else "x", d });
        try w.print(":{s}", .{self.dtype});
        return buf[0..w.end];
    }
};

/// The FP8 (or bf16) n-gram table: mapped shards with a host gather, or on the GPU a rank's heads' rows with the LUT.
pub const NgramTable = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    files: std.ArrayList(st.File) = .empty,
    shards: std.ArrayList([]const u8) = .empty,
    /// each shard's first global row, then the total
    starts: std.ArrayList(u64) = .empty,
    rows: u64 = 0,
    width: u32 = 0,
    fp8: bool = true,
    scale: f32 = 1,
    lut: [256]u16 = @splat(0),
    /// on the GPU: rows [base, base + count) as stored, the LUT ([256] bf16 bits), heads [head0, head0 + heads)
    gpu: ?struct { rows: u64, lut: u64, base: u64, count: u64, head0: u32, heads: u32 } = null,

    pub fn deinit(t: *NgramTable) void {
        for (t.files.items) |*f| f.close(t.io);
        t.files.deinit(t.gpa);
        t.shards.deinit(t.gpa);
        t.starts.deinit(t.gpa);
        t.* = undefined;
    }

    fn row(t: *const NgramTable, id: u64) []const u8 {
        // shards are equal but for the last: a division finds the shard, a step corrects it
        var s: usize = @intCast(@min(id / (t.starts.items[1] - t.starts.items[0]), t.shards.items.len - 1));
        while (t.starts.items[s] > id) s -= 1;
        while (t.starts.items[s + 1] <= id) s += 1;
        const bytes: usize = if (t.fp8) t.width else 2 * t.width;
        return t.shards.items[s][@intCast((id - t.starts.items[s]) * bytes)..][0..bytes];
    }

    /// Rows `ids` (global, as NGram.ids gives them) -> `out[ids.len][width]` bf16 bits, Python's gather exactly.
    pub fn gather(t: *const NgramTable, ids: []const i64, out: []u16) !void {
        if (out.len != ids.len * t.width) return error.NgramOutLength;
        for (ids, 0..) |id, i| {
            if (id < 0 or id >= t.rows) return error.NgramIdOutOfRange;
            const src = t.row(@intCast(id));
            const dst = out[i * t.width ..][0..t.width];
            if (t.fp8) {
                for (dst, src) |*d, c| d.* = t.lut[c];
            } else {
                for (dst, 0..) |*d, j| d.* = std.mem.readInt(u16, src[2 * j ..][0..2], .little);
            }
        }
    }
};

/// fn_pack.cu gather: global row ids int64 [tokens][ids_stride] -> bf16 [tokens][heads * width] for the rank's heads.
pub const Gather = struct {
    module: cuda.Module,
    fp8: cuda.Function,
    bf16: cuda.Function,

    pub fn init(d: *const cuda.Driver) !Gather {
        var m = try cuda.Module.load(d, cuda.kernels.fn_pack);
        errdefer m.unload();
        return .{ .module = m, .fp8 = try m.function("fn_ngram_gather"), .bf16 = try m.function("fn_ngram_gather_bf16") };
    }

    pub fn deinit(g: *Gather) void {
        g.module.unload();
        g.* = undefined;
    }

    pub fn run(g: Gather, s: cuda.Stream, t: *const NgramTable, ids: u64, ids_stride: u32, tokens: u32, out: u64) !void {
        const gpu = t.gpu orelse return error.NgramTableNotOnGpu;
        if (tokens == 0) return;
        var a: cuda.Args = .{};
        a.add(gpu.rows);
        if (t.fp8) a.add(gpu.lut);
        a.add(ids);
        for ([_]u32{ ids_stride, gpu.head0, gpu.heads }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(@as(c_longlong, @intCast(gpu.base)));
        a.add(@as(c_longlong, @intCast(gpu.count)));
        a.add(@as(c_int, @intCast(t.width)));
        a.add(@as(c_int, @intCast(tokens)));
        a.add(out);
        const total: u64 = @as(u64, tokens) * gpu.heads * t.width;
        try cuda.launch.launch(if (t.fp8) g.fp8 else g.bf16, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    mode: Mode,
    driver: ?*const cuda.Driver,
    config: *const Config,
    rank: u32,
    world: u32,
    /// the bf16 token table [V, D] (Python `embed` is a 1-tuple on NVFP4 checkpoints: "embed.0")
    embed: u64 = 0,
    layers: []Layer = &.{},
    mixer: Hc = .{},
    /// this rank's vocabulary rows of lm_head, from `vocab_offset` (INT4-AutoRound: `weight` 0, the rows in `head4`)
    head: Rows = .{},
    /// INT4-AutoRound: the GPTQ int4 lm_head's vocabulary columns of this rank
    head4: ?int4.Mat = null,
    vocab_offset: u32 = 0,
    inv_freq: u64 = 0,
    mtp: ?Mtp = null,
    draft_head: ?Q4 = null,
    /// int64 draft ids (this rank's share), in draft-head row order
    draft_ids: u64 = 0,
    draft_count: u32 = 0,
    around_one: bool = true,
    table: ?NgramTable = null,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    named: std.ArrayList(Named) = .empty,
    /// device bytes the buffers hold
    bytes: u64 = 0,
    /// tensors left unread on purpose (the vision tower, the activation scales the engine never applies)
    skipped: u64 = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        for (w.named.items) |n| w.gpa.free(n.name);
        w.buffers.deinit(w.gpa);
        w.named.deinit(w.gpa);
        w.gpa.free(w.layers);
        if (w.table) |*t| t.deinit();
        w.* = undefined;
    }

    /// The named buffer, for tests and checks.
    pub fn find(w: *const Weights, name: []const u8) ?Named {
        for (w.named.items) |n| if (std.mem.eql(u8, n.name, name)) return n;
        return null;
    }
};

/// One (name, sha256) the coordinator's check-weights compares with the oracle's weights.json.
pub const Digest = struct { name: []const u8, sha256: [64]u8, len: usize, shape: []u8, oracle: bool };

pub fn freeDigests(gpa: std.mem.Allocator, ds: []Digest) void {
    for (ds) |d| gpa.free(d.shape);
    gpa.free(ds);
}

/// sha256 of every buffer: hashed at load, else downloaded a chunk at a time; names borrow the Weights' strings.
pub fn digests(gpa: std.mem.Allocator, w: *const Weights) ![]Digest {
    const out = try gpa.alloc(Digest, w.named.items.len);
    var done: usize = 0;
    errdefer {
        for (out[0..done]) |d| gpa.free(d.shape);
        gpa.free(out);
    }
    var host: []u8 = &.{};
    defer gpa.free(host);
    for (w.named.items, out) |n, *d| {
        var sum: [32]u8 = undefined;
        if (n.sha) |s| {
            sum = s;
        } else {
            if (w.mode != .device) return error.NoDigests;
            if (host.len == 0) host = try gpa.alloc(u8, 256 << 20);
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            const buf: cuda.DeviceBuffer = .{ .d = w.driver.?, .ptr = n.ptr, .len = n.len };
            var at: usize = 0;
            while (at < n.len) {
                const step = @min(host.len, n.len - at);
                try buf.download(at, host[0..step]);
                h.update(host[0..step]);
                at += step;
            }
            h.final(&sum);
        }
        var sb: [96]u8 = undefined;
        d.* = .{ .name = n.name, .sha256 = std.fmt.bytesToHex(sum, .lower), .len = n.len, .shape = try gpa.dupe(u8, try n.shapeText(&sb)), .oracle = n.oracle };
        done += 1;
    }
    return out;
}

/// Python's norms_around_one over hc_norm means: true when scales (around one), false when centred; else refused.
pub fn aroundOne(means: []const f64) !bool {
    if (means.len == 0) return true;
    var above: usize = 0;
    for (means) |m| above += @intFromBool(m > 0.5);
    const frac = @as(f64, @floatFromInt(above)) / @as(f64, @floatFromInt(means.len));
    var sorted: [cfgs.max_layers]f64 = undefined;
    @memcpy(sorted[0..means.len], means);
    std.mem.sort(f64, sorted[0..means.len], {}, std.sort.asc(f64));
    const n = means.len;
    const median = if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
    if (frac >= 0.9 and median >= 0.75 and median <= 1.5) return true;
    if (frac <= 0.1 and median >= -0.5 and median <= 0.25) return false;
    return error.NormsUnclear;
}

/// Draft vocabulary ids below `vocab`, sorted unique (np.unique), then rank's share of `world` (np.array_split).
pub fn draftIds(gpa: std.mem.Allocator, text: []const u8, vocab: u32, rank: u32, world: u32) ![]u32 {
    var all: std.ArrayList(u32) = .empty;
    defer all.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, " \n\r\t,");
    while (it.next()) |tok| {
        const v = try std.fmt.parseInt(i64, tok, 10);
        if (v >= 0 and v < vocab) try all.append(gpa, @intCast(v));
    }
    std.mem.sort(u32, all.items, {}, std.sort.asc(u32));
    var n: usize = 0;
    for (all.items, 0..) |v, i| if (i == 0 or v != all.items[n - 1]) {
        all.items[n] = v;
        n += 1;
    };
    const len = n / world;
    const extra = n % world;
    const start = rank * len + @min(rank, extra);
    const size = len + @intFromBool(rank < extra);
    return gpa.dupe(u32, all.items[start..][0..size]);
}

/// Mark every tensor under `prefix` read on purpose (the vision tower), counting them.
fn skip(ck: *core.Checkpoint, gpa: std.mem.Allocator, prefix: []const u8) !usize {
    var n: usize = 0;
    for (ck.files.items) |*f| for (f.names.keys()) |k| if (std.mem.startsWith(u8, k, prefix)) {
        try ck.used.put(gpa, k, {});
        n += 1;
    };
    return n;
}

/// Drops the checkpoint's shards from the page cache (POSIX_FADV_DONTNEED) so GPU allocation does not wait for reclaim.
pub fn dropPages(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) void {
    const paths = core.checkpoint.shardFiles(gpa, io, dir) catch return;
    defer core.checkpoint.freeShardFiles(gpa, paths);
    for (paths) |path| {
        var f = std.Io.Dir.cwd().openFile(io, path, .{}) catch continue;
        defer f.close(io);
        _ = std.os.linux.fadvise(f.handle, 0, 0, std.os.linux.POSIX_FADV.DONTNEED);
    }
}

/// weights.load of the ModelOpt NVFP4 checkpoint: this rank's buffers, leftovers refused (vision, input_scale aside).
pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const Config, o: Options) !Weights {
    var why: cfgs.Why = .{};
    c.check(&why) catch |e| {
        std.log.err("{s}: {s}", .{ dir, why.text() });
        return e;
    };
    if (o.rank >= o.world) return error.BadRank;
    _ = try c.shares(o.world);
    if (o.mode == .device and o.driver == null) return error.NoDriver;
    var w: Weights = .{ .gpa = gpa, .mode = o.mode, .driver = o.driver, .config = c, .rank = o.rank, .world = o.world };
    errdefer w.deinit();
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const paths = try core.checkpoint.shardFiles(gpa, io, dir);
    defer core.checkpoint.freeShardFiles(gpa, paths);
    if (paths.len != ck.files.items.len) return error.CheckpointChanged;
    var L: Loader = .{ .gpa = gpa, .io = io, .ck = &ck, .c = c, .o = o, .out = .{ .gpa = gpa, .w = &w }, .paths = paths, .rope = ropeOf(c) };
    defer L.deinit();
    L.prefix = if (L.has("language_model.model.embed_tokens.weight")) "language_model." else "";
    try L.openExtra(dir);
    if (o.overlay) |ov| try L.openOverlay(ov);
    L.mbase = if (L.has(try L.nameOf("{s}model.language_model.embed_tokens.weight", .{L.prefix}))) "model.language_model." else "model.";

    // norms_around_one over every layer's attn hyper-connection norm
    var means: [cfgs.max_layers]f64 = undefined;
    for (0..c.layers) |i| {
        const t = try L.expect(try L.nameOf("{s}layers.{d}.attn_hyper_connection.hc_norm.weight", .{ L.mbase, i }), .bf16, &.{c.streams * c.hidden});
        var sum: f64 = 0;
        for (0..t.numel()) |j| sum += lay.bf16ToF32(std.mem.readInt(u16, t.bytes[2 * j ..][0..2], .little));
        means[i] = sum / @as(f64, @floatFromInt(t.numel()));
    }
    L.around_one = aroundOne(means[0..c.layers]) catch |e| {
        std.log.err("{s}: cannot tell how the norm weights are stored (hc_norm means around neither 1 nor 0)", .{dir});
        return e;
    };
    w.around_one = L.around_one;

    const embed = try L.linear(try L.nameOf("{s}embed_tokens", .{L.mbase}), c.vocab, c.hidden);
    w.embed = try L.out.whole("embed.0", "bfloat16", &.{ c.vocab, c.hidden }, embed.bytes);
    L.release();

    w.layers = try gpa.alloc(Layer, c.layers);
    for (0..c.layers) |i| {
        var pb: [64]u8 = undefined;
        var bb: [96]u8 = undefined;
        w.layers[i] = try L.layer(&w, try std.fmt.bufPrint(&pb, "layers.{d}", .{i}), @intCast(i), try std.fmt.bufPrint(&bb, "{s}layers.{d}", .{ L.mbase, i }), c.layer_types[i], true);
        L.release();
    }
    w.mixer = try L.hc("mixer", try L.nameOf("{s}hyper_connection_mixer", .{L.mbase}), false);

    // lm_head: this rank's vocabulary rows (bf16; a block-FP8 head is refused), INT4-AutoRound's GPTQ int4 columns
    if (L.has(try L.nameOf("{s}lm_head.weight_scale_inv", .{L.prefix}))) return error.UnsupportedQuantization;
    const vl = c.vocab / o.world;
    w.vocab_offset = o.rank * vl;
    var head_src: ?int4.Source = null;
    var head_bf16: ?Tensor = null;
    if (c.int4ar()) {
        head_src = try L.int4Head(&w, vl);
    } else {
        const head = try L.linear("lm_head", c.vocab, c.hidden);
        head_bf16 = head;
        w.head = try L.face("head.b.weight", &.{.{ .t = head, .rows = &.{.{ o.rank * vl, (o.rank + 1) * vl }} }});
    }

    // the draft head: the lm_head's draft rows quantized 4-bit (groups of 32), packed for the lane matmul
    if (o.draft_head) {
        const ids = try draftIds(gpa, draft_vocab, c.vocab, o.rank, o.world);
        defer gpa.free(ids);
        const k = c.hidden;
        const n = ids.len;
        const rows = try gpa.alloc(u16, n * k);
        defer gpa.free(rows);
        if (head_src) |src| {
            // the int4 head's draft rows dequantized ((q - 8) * scale, exact in fp32) and rounded to bf16
            const Dq = struct {
                src: int4.Source,
                ids: []const u32,
                rows: []u16,
                k: usize,
                fn run(q: @This(), chunk: usize) void {
                    const lo = chunk * 1024;
                    const hi = @min(lo + 1024, q.ids.len);
                    for (lo..hi) |r| for (0..q.k) |j| {
                        q.rows[r * q.k + j] = lay.bf16Rne(int4.weight(q.src, q.ids[r], j));
                    };
                }
            };
            parallel(o.threads, (n + 1023) / 1024, Dq{ .src = src, .ids = ids, .rows = rows, .k = k }, Dq.run);
        } else {
            const head = head_bf16.?;
            for (ids, 0..) |id, r| for (0..k) |j| {
                rows[r * k + j] = std.mem.readInt(u16, head.bytes[(id * k + j) * 2 ..][0..2], .little);
            };
        }
        const words = try gpa.alloc(u32, n * k / 8);
        defer gpa.free(words);
        const sc = try gpa.alloc(u16, n * k / 32);
        defer gpa.free(sc);
        const bi = try gpa.alloc(u16, n * k / 32);
        defer gpa.free(bi);
        const Q = struct {
            rows: []const u16,
            words: []u32,
            sc: []u16,
            bi: []u16,
            k: usize,
            fn run(q: @This(), chunk: usize) void {
                const lo = chunk * 1024;
                const hi = @min(lo + 1024, q.rows.len / q.k);
                lay.quantize4(q.rows[lo * q.k .. hi * q.k], hi - lo, q.k, q.words[lo * q.k / 8 .. hi * q.k / 8], q.sc[lo * q.k / 32 .. hi * q.k / 32], q.bi[lo * q.k / 32 .. hi * q.k / 32]);
            }
        };
        parallel(o.threads, (n + 1023) / 1024, Q{ .rows = rows, .words = words, .sc = sc, .bi = bi, .k = k }, Q.run);
        const npad = (n + 127) / 128 * 128;
        const kg = k / 32;
        const frag = try gpa.alloc(u32, npad / 64 * kg * 8 * 32);
        defer gpa.free(frag);
        lay.packQ4(words, n, k, frag);
        var q: Q4 = .{ .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) };
        q.weight = try L.out.whole("draft_head.weight", "int32", &.{ npad / 64, kg, 8, 32, 1 }, std.mem.sliceAsBytes(frag));
        const major = try gpa.alloc(u16, kg * npad);
        defer gpa.free(major);
        lay.majorQ4(sc, n, kg, major);
        q.scales = try L.out.whole("draft_head.scales", "bfloat16", &.{ kg, npad }, std.mem.sliceAsBytes(major));
        lay.majorQ4(bi, n, kg, major);
        q.biases = try L.out.whole("draft_head.biases", "bfloat16", &.{ kg, npad }, std.mem.sliceAsBytes(major));
        w.draft_head = q;
        const wide = try gpa.alloc(i64, n);
        defer gpa.free(wide);
        for (ids, wide) |id, *x| x.* = id;
        w.draft_ids = try L.out.whole("draft_ids", "int64", &.{n}, std.mem.sliceAsBytes(wide));
        w.draft_count = @intCast(n);
    }
    L.release();

    // inv_freq: theta ** (-i / half) in fp64, stored fp32; YaRN's ramp when on (cuda_rope.zig, the patched Python)
    var inv: [ropes.max_half]f32 = undefined;
    const half = L.rope.half();
    try L.rope.invFreq(inv[0..half]);
    w.inv_freq = try L.out.whole("inv_freq", "float32", &.{half}, std.mem.sliceAsBytes(inv[0..half]));

    if (o.mtp and L.has(try L.nameOf("{s}mtp.fc_embedding.weight", .{L.prefix}))) {
        const sd = c.streams * c.hidden;
        var m: Mtp = .{ .layer = undefined };
        m.norm_e = try L.cscale("mtp.norm_e", "mtp.pre_fc_norm_embedding.weight", c.hidden);
        m.norm_h = try L.cscale("mtp.norm_h", "mtp.pre_fc_norm_hidden.weight", sd);
        m.fc_e = try L.b16("mtp.fc_e.b.weight", "mtp.fc_embedding", c.hidden, c.hidden);
        m.fc_h = try L.b16("mtp.fc_h.b.weight", "mtp.fc_hidden", c.hidden, c.hidden);
        m.layer = try L.layer(&w, "mtp.layer", -1, "mtp.layers.0", .attention, false);
        m.mixer = try L.hc("mtp.mixer", "mtp.hyper_connection_mixer", false);
        w.mtp = m;
        L.release();
    } else if (L.has(try L.nameOf("{s}mtp.fc_embedding.weight", .{L.prefix}))) {
        w.skipped += try skip(&ck, gpa, "mtp.");
    }
    // the vision tower: rank 0's later (work/PLAN.md); read on purpose as nothing now
    w.skipped += try skip(&ck, gpa, "model.visual.");
    if (ck.unused() != 0) return error.UnusedCheckpointTensors;
    return w;
}

/// The rope the weights fold: cuda_config's fields mapped 1:1 onto cuda_rope's (attention_factor null -> 0: derived).
pub fn ropeOf(c: *const Config) ropes.Rope {
    var r: ropes.Rope = .{ .theta = c.rope.theta, .rotary_dim = c.rope.rotary_dim };
    if (c.rope.kind == .yarn) if (c.rope.yarn) |y| {
        r.yarn = .{ .factor = y.factor, .original = y.original_max_position_embeddings, .beta_fast = y.beta_fast, .beta_slow = y.beta_slow, .attention_factor = y.attention_factor orelse 0, .truncate = y.truncate };
    };
    return r;
}

test {
    _ = lay;
    _ = cfgs;
    _ = &load;
    _ = &digests;
}

test "the default draft vocabulary: 79,591 ids, split as np.array_split" {
    const a = std.testing.allocator;
    const all = try draftIds(a, draft_vocab, 248320, 0, 1);
    defer a.free(all);
    try std.testing.expectEqual(@as(usize, 79591), all.len);
    try std.testing.expectEqual(@as(u32, 0), all[0]);
    try std.testing.expectEqual(@as(u32, 248076), all[all.len - 1]);
    const r0 = try draftIds(a, draft_vocab, 248320, 0, 2);
    defer a.free(r0);
    const r1 = try draftIds(a, draft_vocab, 248320, 1, 2);
    defer a.free(r1);
    try std.testing.expectEqual(@as(usize, 39796), r0.len);
    try std.testing.expectEqual(@as(usize, 39795), r1.len);
    try std.testing.expectEqualSlices(u32, all[0..39796], r0);
    try std.testing.expectEqualSlices(u32, all[39796..], r1);
    const small = try draftIds(a, "5 3 3 9 1", 6, 0, 1);
    defer a.free(small);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 5 }, small);
}

test "norms around one or centred, as reader.norms_around_one" {
    try std.testing.expect(try aroundOne(&.{ 1.0, 0.98, 1.1, 0.9 }));
    try std.testing.expect(!try aroundOne(&.{ 0.01, -0.02, 0.1, 0.0 }));
    try std.testing.expectError(error.NormsUnclear, aroundOne(&.{ 0.5, 0.6, 0.4, 0.55 }));
}

test "ropeOf maps cuda_config's YaRN onto cuda_rope's" {
    var why: cfgs.Why = .{};
    var c = try cfgs.parse(std.testing.allocator, @embedFile("fixtures_cuda_config.json"), null, .{ .yarn = .{ .factor = 4, .original_max_position_embeddings = 262144 } }, &why);
    defer c.deinit();
    const r = ropeOf(&c);
    try std.testing.expectEqual(@as(u32, 64), r.rotary_dim);
    try std.testing.expectEqual(@as(u64, 262144), r.yarn.?.original);
    try std.testing.expectEqual(@as(u32, 0x3f91be9c), @as(u32, @bitCast(r.scale())));
    var plain = try cfgs.parse(std.testing.allocator, @embedFile("fixtures_cuda_config.json"), null, .{}, &why);
    defer plain.deinit();
    try std.testing.expectEqual(@as(?ropes.Yarn, null), ropeOf(&plain).yarn);
    // without YaRN the frequencies are layouts.invFreq's (Python's stored fp32)
    var a: [32]f32 = undefined;
    var b: [32]f32 = undefined;
    try ropeOf(&plain).invFreq(&a);
    lay.invFreq(1e7, 32, &b);
    try std.testing.expectEqualSlices(f32, &b, &a);
}
