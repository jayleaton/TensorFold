//! Triton launch bindings `Tri`: their argument types, split-K shapes, the launch recorder and the variant memo.

const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
const Dims = @import("cuda_triton.zig").Dims;

const p = aot.ptr;

const bf16 = "*bf16";
const f32p = "*fp32";
const f16p = "*fp16";
const i32p = "*i32";
const fp8p = "*fp8e4nv"; // kv8.py FP8 cache rows (torch.float8_e4m3fn)

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn cb(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

fn u(x: usize) u32 {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// triton.next_power_of_2
pub fn pow2(n: usize) usize {
    return std.math.ceilPowerOfTwo(usize, @max(n, 1)) catch unreachable;
}

/// One launch as `aot.Set.run` takes it, copied (names and types are static strings) for the tests.
pub const Recorder = struct {
    launches: [64]Saved = undefined,
    n: usize = 0,

    pub const Saved = struct {
        name: []const u8,
        grid: [3]u32,
        args: [24]aot.Arg = undefined,
        nargs: usize,
        consts: [16]aot.Const = undefined,
        nconsts: usize,
    };

    fn add(r: *Recorder, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        if (r.n == r.launches.len) return error.TooManyLaunches;
        const s = &r.launches[r.n];
        s.* = .{ .name = name, .grid = grid, .nargs = args.len, .nconsts = consts.len };
        @memcpy(s.args[0..args.len], args);
        @memcpy(s.consts[0..consts.len], consts);
        r.n += 1;
    }
};

/// What `_hc_writeback` adds to the streams before their read-out (glue.hc_writeback's `mode`).
pub const Branch = union(enum) {
    /// mode 0: no pending branch (the first layer); the streams' squared sums only
    none,
    /// mode 1: a block's bf16 output [R, D]
    bf16: u64,
    /// mode 2: the MoE's slots y [R, slots, D] (fp32 in the MTP head's buffers, bf16 in the main ones) and weights
    moe: struct { y: u64, y_f32: bool, wts: u64, slots: usize },
    /// mode 3: every rank's fp32 partials [world, R, D], summed rank 0 first (TP=2)
    ranks: struct { part: u64, world: usize },
    /// mode 4: a matmul's fp32 K slices [SK, R, D]
    slices: struct { part: u64, sk: usize },
};

/// attention.AttnScratch's geometry for a buffer's capacity: partial chunks, the sparse key lists, the block scores.
pub const AttnGeometry = struct {
    nch: usize,
    qsa: bool,
    idw: usize,
    nb: usize,
    budget: usize,
    ratio: usize,

    pub const chunk = 512;
    /// _select holds a row's block scores in registers up to this many
    pub const select_regs = 32768;

    pub fn init(capacity: usize, budget: usize, ratio: usize) AttnGeometry {
        return .{ .nch = cdiv(@min(capacity, budget + ratio - 1), chunk), .qsa = capacity > budget, .idw = budget + ratio, .nb = cdiv(capacity, ratio), .budget = budget, .ratio = ratio };
    }

    /// The chunks a launch covers for `context` keys (null: the whole scratch), as attention.attention bounds them.
    pub fn chunks(g: AttnGeometry, context: ?usize) usize {
        var keys = context orelse g.nch * chunk;
        if (g.qsa) keys = @min(keys, (g.budget / g.ratio + 1) * g.ratio - 1);
        return @min(g.nch, cdiv(keys, chunk));
    }

    /// qsa_rows' scored blocks for `context` keys.
    pub fn blocks(g: AttnGeometry, context: ?usize) usize {
        const c = context orelse return g.nb;
        return @min(g.nb, @max(1, cdiv(c, g.ratio)));
    }
};

/// AttnScratch's device buffers: chunk partials, each row's key list, its length and whether it is sparse, scores.
pub const AttnScratch = struct { po: u64, pm: u64, pl: u64, ids: u64, nk: u64, sparse: u64, scores: u64 };

/// One layer's attention caches: k/v (bf16, int8/int4 codes) and fp16 scales; `fp8` holds kv8.py rows, `bits` stays 0.
pub const Cache = struct {
    k: u64,
    v: u64,
    ks: u64,
    vs: u64,
    bits: usize = 0,
    fp8: bool = false,

    fn codes(c: Cache) []const u8 {
        return switch (c.bits) {
            0 => bf16,
            8 => "*i8",
            else => "*u8",
        };
    }
};

/// An NVFP4 matmul's table (nvfp4.FP4): bf16 patterns and fp32 scales, or the checkpoint's codes and e4m3 scales.
pub const Fp4 = struct { weight: u64, scale: u64, scale2: u64, codes: bool = false };

/// bf16.split_k: K slices by the weight's shape alone, a power of two of whole 64-blocks.
pub fn b16SplitK(n: usize, k: usize) usize {
    const tiles = cdiv(n, 64);
    const blocks = k / 64;
    var sk: usize = 1;
    while (sk < 32 and tiles * sk < 160 and blocks % (sk * 2) == 0 and blocks / (sk * 2) >= 1) sk *= 2;
    return sk;
}

/// The fp32 scratch bf16.matmul's split K needs for `m` rows (0 without a split).
pub fn b16PartBytes(m: usize, n: usize, k: usize) usize {
    const sk = b16SplitK(n, k);
    return if (sk > 1) sk * m * n * 4 else 0;
}

/// nvfp4.split_for: the shapes tuned at up to 16 rows keep their K slices at every row count, else split_k.
pub fn fp4SplitK(n: usize, k: usize) usize {
    if (fp4Tuned(n, k)) |t| return t.sk;
    const tiles = cdiv(n, 64);
    const blocks = k / 16;
    var sk: usize = 1;
    while (sk < 32 and tiles * sk < 160 and blocks % (sk * 2) == 0 and blocks / (sk * 2) >= 4) sk *= 2;
    return sk;
}

pub fn fp4PartBytes(m: usize, n: usize, k: usize) usize {
    const sk = fp4SplitK(n, k);
    return if (sk > 1) sk * m * n * 4 else 0;
}

const Fp4Tuned = struct { sk: usize, gpi: usize };

/// nvfp4.SHAPES16 (the warps and stages it also sets are not constexprs).
fn fp4Tuned(n: usize, k: usize) ?Fp4Tuned {
    if (k == 2560 and (n == 640 or n == 1280)) return .{ .sk = 1, .gpi = 4 };
    if (n == 2560 and k == 640) return .{ .sk = 8, .gpi = 2 };
    return null;
}

/// nvfp4.bucket / moe._tile: 16 to 128 rows, then 128-row tiles.
fn bucket(m: usize) usize {
    for ([_]usize{ 16, 32, 64, 128 }) |b| if (m <= b) return b;
    return 128;
}

/// nvfp4.gpi_for
fn gpiFor(per: usize, want: usize) usize {
    for ([_]usize{ want, 8, 4, 2, 1 }) |g| if (g <= want and per % g == 0) return g;
    return 1;
}

/// Memo of `aot.Set.find`'s pick per (function, constexprs, specialization), so repeated launches skip the scan.
pub const Memo = struct {
    gpa: std.mem.Allocator,
    map: std.AutoHashMapUnmanaged(u64, usize) = .empty,

    pub fn deinit(m: *Memo) void {
        m.map.deinit(m.gpa);
    }

    fn key(name: []const u8, args: []const aot.Arg, consts: []const aot.Const) u64 {
        var h = std.hash.Wyhash.init(0x7472693a);
        h.update(name);
        for (consts) |c| {
            h.update(&.{0xc0});
            h.update(c.name);
            if (c.int) |x| h.update(std.mem.asBytes(&x));
            if (c.f32) |x| h.update(std.mem.asBytes(&@as(u32, @bitCast(x))));
            h.update(&.{ @intFromBool(c.int != null), @intFromBool(c.f32 != null) });
        }
        for (args) |a| {
            h.update(&.{0xa0});
            h.update(a.name);
            switch (a.value) {
                .ptr => |x| {
                    h.update(x.ty);
                    h.update(&.{ 1, @intFromBool(x.addr % 16 == 0) });
                },
                .i32 => |x| h.update(&.{ 2, @intFromBool(x == 1), @intFromBool(@mod(x, 16) == 0) }),
                .f32 => h.update(&.{3}),
                .u64 => |x| h.update(&.{ 4, @intFromBool(x % 16 == 0) }),
            }
        }
        return h.final();
    }
};

pub const Tri = struct {
    set: ?*const aot.Set,
    s: cuda.Stream = undefined,
    /// tests: record each launch instead of running it
    rec: ?*Recorder = null,
    /// the variants already picked (null: `find` every launch)
    memo: ?*Memo = null,

    fn run(t: Tri, name: []const u8, grid: [3]usize, args: []const aot.Arg, consts: []const aot.Const) !void {
        const g: [3]u32 = .{ u(grid[0]), u(grid[1]), u(grid[2]) };
        if (t.rec) |r| return r.add(name, g, args, consts);
        const set = t.set.?;
        const m = t.memo orelse return set.run(t.s, name, g, args, consts);
        const k = Memo.key(name, args, consts);
        const at = m.map.get(k) orelse blk: {
            const v = try set.find(name, args, consts);
            const i = (@intFromPtr(v) - @intFromPtr(set.variants.ptr)) / @sizeOf(@TypeOf(set.variants[0]));
            try m.map.put(m.gpa, k, i);
            break :blk i;
        };
        // aot.Set.run's launch: runtime arguments in the variant's order
        const v = &set.variants[at];
        var packed_args: cuda.launch.Args = .{};
        for (v.spec.params) |prm| {
            const a = for (args) |x| {
                if (std.mem.eql(u8, x.name, prm.name)) break x;
            } else return error.MissingTritonArgument;
            switch (a.value) {
                .ptr => |x| packed_args.add(x.addr),
                .i32 => |x| packed_args.add(x),
                .f32 => |x| packed_args.add(x),
                .u64 => |x| packed_args.add(x),
            }
        }
        try v.kernel.launchOn(.{ .x = g[0], .y = g[1], .z = g[2] }, t.s, &packed_args, .{}, &.{});
    }

    // -- embedding, hyper-connections, norms ---------------------------------------------------------------

    /// exl3_mm.embed: ids [R] int32 -> out [R, copies * d] bf16, the unquantized table's row in each stream.
    pub fn embed(t: Tri, ids: u64, table: u64, out: u64, rows: usize, d: usize, copies: usize) !void {
        try t.run("_embed", .{ rows, d / 256, 1 }, &.{ p("IDS", i32p, ids), p("T", bf16, table), p("OUT", bf16, out) }, &.{ ci("D", d), ci("S", copies), ci("BLOCK", 256) });
    }

    /// glue.hc_writeback: branch into streams h [R, S*D], squared sums per 256 dims into pss; `inject` [R, S] gates.
    pub fn hcWriteback(t: Tri, h: u64, hout: u64, pss: u64, inject: ?u64, branch: Branch, rows: usize, d: usize, streams: usize) !void {
        return t.hcWritebackNorm(h, hout, pss, inject, branch, rows, d, streams, null);
    }

    /// The scale (fp32 [S*D]), the normed rows [R, S*D] bf16 and eps of `_hc_wb_norm`.
    pub const NormOut = struct { scale: u64, normed: u64, eps: f32 };

    /// hcWriteback plus, with `norm`, hcNormed's rows in one launch, a row a program (prompt_mm._hc_wb_norm).
    pub fn hcWritebackNorm(t: Tri, h: u64, hout: u64, pss: u64, inject: ?u64, branch: Branch, rows: usize, d: usize, streams: usize, norm: ?NormOut) !void {
        const block = 256;
        var br = p("BR", bf16, h);
        var y = p("Y", bf16, h);
        var wts = p("WTS", bf16, h);
        var mode: usize = 0;
        var top: usize = 1;
        var slots: usize = 1;
        var world: usize = 1;
        switch (branch) {
            .none => {},
            .bf16 => |b| {
                mode = 1;
                br = p("BR", bf16, b);
            },
            .moe => |m| {
                mode = 2;
                y = p("Y", if (m.y_f32) f32p else bf16, m.y);
                wts = p("WTS", f32p, m.wts);
                top = m.slots - 1;
                slots = m.slots;
            },
            .ranks => |r| {
                mode = 3;
                br = p("BR", f32p, r.part);
                world = r.world;
            },
            .slices => |s| {
                mode = 4;
                br = p("BR", f32p, s.part);
                world = s.sk;
            },
        }
        if (mode != 0 and inject == null) return error.MissingInject;
        const inj = p("INJ", bf16, inject orelse h);
        const consts = [_]aot.Const{ ci("D", d), ci("S", streams), ci("MODE", mode), ci("TOPK", top), ci("SLOTS", slots), ci("BLOCK", block), ci("WORLD", world) };
        if (norm) |nm| {
            try t.run("_hc_wb_norm", .{ rows, 1, 1 }, &.{ p("H", bf16, h), p("HOUT", bf16, hout), p("PSS", f32p, pss), br, inj, y, wts, int("RS", rows * d), p("SCALE", f32p, nm.scale), p("NORMED", bf16, nm.normed), aot.float("eps", nm.eps) }, &consts);
            return;
        }
        try t.run("_hc_writeback", .{ rows, d / block, 1 }, &.{ p("H", bf16, h), p("HOUT", bf16, hout), p("PSS", f32p, pss), br, inj, y, wts, int("RS", rows * d) }, &consts);
    }

    /// glue.hc_normed: normed [R, S*D] = bf16(h * rinv_s * scale) and its 32-group sums xs, rinv from pss.
    pub fn hcNormed(t: Tri, h: u64, pss: u64, scale: u64, normed: u64, xs: u64, rows: usize, d: usize, streams: usize, eps: f32) !void {
        const block = 512;
        try t.run("_hc_normed", .{ rows, streams * d / block, 1 }, &.{ p("H", bf16, h), p("PSS", f32p, pss), p("SCALE", f32p, scale), p("NORMED", bf16, normed), p("XS", f32p, xs), aot.float("eps", eps) }, &.{ ci("D", d), ci("S", streams), ci("NC", d / 256), ci("BLOCK", block) });
    }

    /// glue.hc_act: fp32 dn [R, ndn] -> act [R, low] bf16 (SiLU) and group sums; `inject` gates when ndn = low + S.
    pub fn hcAct(t: Tri, dn: u64, act: u64, xs: u64, inject: ?u64, rows: usize, ndn: usize, streams: usize, low: usize) !void {
        try t.run("_hc_act", .{ rows, 1, 1 }, &.{ p("DN", f32p, dn), p("ACT", bf16, act), p("XS", f32p, xs), p("INJ", bf16, inject orelse act) }, &.{ ci("S", streams), ci("LOW", low), ci("LOWP", pow2(low)), cb("HAS_INJ", inject != null), ci("NDN", ndn) });
    }

    /// glue.hc_mix: mixed [R, D] = bf16(sum_s bf16(sigmoid(up_s) * normed_s) / S) and its group sums.
    pub fn hcMix(t: Tri, up: u64, normed: u64, mixed: u64, xs: u64, rows: usize, d: usize, streams: usize) !void {
        try t.run("_hc_mix", .{ rows, d / 256, 1 }, &.{ p("UP", bf16, up), p("NORMED", bf16, normed), p("MIXED", bf16, mixed), p("XS", f32p, xs) }, &.{ ci("D", d), ci("S", streams), ci("BLOCK", 256) });
    }

    /// glue.rmsnorm: each `group` run (null: whole row) normalized times fp32 w, bf16 out [R, d] and its 32-group sums.
    pub fn rmsnorm(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, xs: u64, rows: usize, d: usize, group: ?usize, eps: f32) !void {
        const g = group orelse d;
        try t.run("_rmsnorm", .{ rows, d / g, 1 }, &.{ p("X", bf16, x), p("W", f32p, w), p("OUT", bf16, out), p("XS", f32p, xs), aot.float("eps", eps), int("x_stride", x_stride) }, &.{ ci("D", d), ci("G", g), ci("BLOCK", pow2(g)) });
    }

    /// mtp.mtp_compute's input: out [R, S*D] = bf16(e + hs) per stream.
    pub fn addStreams(t: Tri, e: u64, hs: u64, out: u64, rows: usize, d: usize, streams: usize) !void {
        try t.run("_add_streams", .{ rows, d / 256, 1 }, &.{ p("E", bf16, e), p("HS", bf16, hs), p("OUT", bf16, out) }, &.{ ci("D", d), ci("S", streams), ci("BLOCK", 256) });
    }

    // -- attention -----------------------------------------------------------------------------------------

    pub const Prep = struct { p: u64, pos0: u64, qw: u64, kw: u64, iw: u64, inv: u64, q: u64, cache: Cache, iq: u64, ikc: u64 };

    /// Image/video rotary table (image_rows.attach): rows below `length` use the (t, h, w) table, later pos + delta.
    pub const Rope = struct { table: u64, delta: u64, length: usize };

    /// The multimodal rotary sections' h and w counts (rope_parameters.mrope_section [11, 11, 10]).
    pub const rope_s1 = 11;
    pub const rope_s2 = 10;

    /// glue.attn_prep on text positions (MODE 0): q, k, indexer heads normed and rotated, k/v stored at POS0 + r.
    pub fn attnPrep(t: Tri, a: Prep, rows: usize, d: Dims) !void {
        return t.attnPrepRope(a, rows, d, null);
    }

    /// glue.attn_prep: text positions without `rope` (MODE 0), else the image prompt's table (MODE 2).
    pub fn attnPrepRope(t: Tri, a: Prep, rows: usize, d: Dims, rope: ?Rope) !void {
        const pw = d.attnWidth();
        const table = if (rope) |r| r.table else a.pos0;
        const delta = if (rope) |r| r.delta else a.pos0;
        const length: usize = if (rope) |r| r.length else 0;
        const mode: usize = if (rope != null) 2 else 0;
        if (a.cache.fp8) {
            // kv8._attn_prep8: _attn_prep's arithmetic, keys and values stored as FP8 rows with their scales
            try t.run("_attn_prep8", .{ rows, d.heads + d.kv_heads + d.index_heads + 1, 1 }, &.{ p("P", bf16, a.p), p("POS0", i32p, a.pos0), p("QW", f32p, a.qw), p("KW", f32p, a.kw), p("IW", f32p, a.iw), p("INV", f32p, a.inv), p("Q", bf16, a.q), p("KC", fp8p, a.cache.k), p("VC", fp8p, a.cache.v), p("KS", f32p, a.cache.k), p("IQ", bf16, a.iq), p("IKC", bf16, a.ikc), p("ROPE", i32p, table), p("DELTA", i32p, delta), int("length", length), aot.float("eps", d.eps) }, &.{ ci("PW", pw), ci("NQ", d.heads), ci("NKV", d.kv_heads), ci("HD", d.head_dim), ci("NI", d.index_heads), ci("IHD", d.index_dim), ci("HALF", d.rotary_half), ci("MODE", mode), ci("S1", rope_s1), ci("S2", rope_s2) });
            return;
        }
        try t.run("_attn_prep", .{ rows, d.heads + d.kv_heads + d.index_heads + 1, 1 }, &.{ p("P", bf16, a.p), p("POS0", i32p, a.pos0), p("QW", f32p, a.qw), p("KW", f32p, a.kw), p("IW", f32p, a.iw), p("INV", f32p, a.inv), p("Q", bf16, a.q), p("KC", a.cache.codes(), a.cache.k), p("VC", a.cache.codes(), a.cache.v), p("KS", f16p, a.cache.ks), p("VS", f16p, a.cache.vs), p("IQ", bf16, a.iq), p("IKC", bf16, a.ikc), p("ROPE", i32p, table), p("DELTA", i32p, delta), int("length", length), aot.float("eps", d.eps) }, &.{ ci("PW", pw), ci("NQ", d.heads), ci("NKV", d.kv_heads), ci("HD", d.head_dim), ci("NI", d.index_heads), ci("IHD", d.index_dim), ci("HALF", d.rotary_half), ci("BITS", a.cache.bits), ci("MODE", mode), ci("S1", rope_s1), ci("S2", rope_s2) });
    }

    /// attention.attention: queries [R, H, D] over 512-key chunks merged in chunk order; `context` bounds chunks.
    pub fn attention(t: Tri, q: u64, cache: Cache, pos0: u64, sc: AttnScratch, out: u64, rows: usize, g: AttnGeometry, context: ?usize, d: Dims) !void {
        const h = d.heads;
        const hk = d.kv_heads;
        const grp = h / hk;
        if (grp > 16) return error.TooManyQueryHeads;
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(d.head_dim), -0.5));
        const ty = cache.codes();
        if (cache.fp8) {
            // kv8._chunks8: the codes to bf16, each key's s_k on its dots and s_v on its probabilities
            try t.run("_chunks8", .{ rows, hk, g.chunks(context) }, &.{ p("Q", bf16, q), p("KC", fp8p, cache.k), p("VC", fp8p, cache.v), p("KS", f32p, cache.k), p("POS0", i32p, pos0), p("PO", f32p, sc.po), p("PM", f32p, sc.pm), p("PL", f32p, sc.pl), p("IDS", i32p, sc.ids), p("NKR", i32p, sc.nk), p("SPR", i32p, sc.sparse) }, &.{ ci("H", h), ci("HK", hk), ci("D", d.head_dim), ci("G", grp), ci("CH", AttnGeometry.chunk), ci("NCH", g.nch), aot.cf("SCALE", scale), ci("IDW", g.idw), cb("QSA", g.qsa) });
        } else try t.run("_chunks", .{ rows, hk, g.chunks(context) }, &.{ p("Q", bf16, q), p("KC", ty, cache.k), p("VC", ty, cache.v), p("KSC", f16p, cache.ks), p("VSC", f16p, cache.vs), p("POS0", i32p, pos0), p("PO", f32p, sc.po), p("PM", f32p, sc.pm), p("PL", f32p, sc.pl), p("IDS", i32p, sc.ids), p("NKR", i32p, sc.nk), p("SPR", i32p, sc.sparse) }, &.{ ci("H", h), ci("HK", hk), ci("D", d.head_dim), ci("G", grp), ci("CH", AttnGeometry.chunk), ci("NCH", g.nch), aot.cf("SCALE", scale), ci("IDW", g.idw), cb("QSA", g.qsa), ci("BITS", cache.bits) });
        try t.run("_merge", .{ rows, hk, 1 }, &.{ p("PO", f32p, sc.po), p("PM", f32p, sc.pm), p("PL", f32p, sc.pl), p("POS0", i32p, pos0), p("OUT", bf16, out), p("NKR", i32p, sc.nk), p("SPR", i32p, sc.sparse) }, &.{ ci("H", h), ci("HK", hk), ci("D", d.head_dim), ci("G", grp), ci("CH", AttnGeometry.chunk), ci("NCH", g.nch), cb("QSA", g.qsa), ci("BITS", cache.bits) });
    }

    /// attention.qsa_pool on text positions: complete RATIO-key blocks pooled, normed (k_layernorm) and rotated.
    pub fn qsaPool(t: Tri, ikc: u64, pooled: u64, pos0: u64, w: u64, inv: u64, rows: usize, g: AttnGeometry, d: Dims) !void {
        return t.qsaPoolRope(ikc, pooled, pos0, w, inv, rows, g, d, null);
    }

    /// attention.qsa_pool: text positions without `rope` (MODE 0), else the image prompt's table (MODE 2).
    pub fn qsaPoolRope(t: Tri, ikc: u64, pooled: u64, pos0: u64, w: u64, inv: u64, rows: usize, g: AttnGeometry, d: Dims, rope: ?Rope) !void {
        const grid: [3]usize = .{ rows / g.ratio + 2, 1, 1 };
        const consts = [_]aot.Const{ ci("DI", d.index_dim), ci("HALF", d.rotary_half), ci("RATIO", g.ratio), ci("MODE", if (rope != null) 2 else 0), ci("S1", rope_s1), ci("S2", rope_s2) };
        if (rope) |r| return t.run("_pool", grid, &.{ p("IKC", bf16, ikc), p("POOLED", bf16, pooled), p("POS0", i32p, pos0), p("W", f32p, w), p("INV", f32p, inv), aot.float("eps", d.eps), int("R", rows), p("ROPE", i32p, r.table), p("DELTA", i32p, r.delta), int("length", r.length) }, &consts);
        try t.run("_pool", grid, &.{ p("IKC", bf16, ikc), p("POOLED", bf16, pooled), p("POS0", i32p, pos0), p("W", f32p, w), p("INV", f32p, inv), aot.float("eps", d.eps), int("R", rows), int("length", 0) }, &consts);
    }

    /// attention.qsa_rows: each sparse row's block scores, then its TOP blocks and tail into the scratch's key lists.
    pub fn qsaRows(t: Tri, iq: u64, pooled: u64, pos0: u64, sc: AttnScratch, rows: usize, g: AttnGeometry, context: ?usize, d: Dims) !void {
        const top = g.budget / g.ratio;
        const blocks = g.blocks(context);
        try t.run("_scores", .{ rows, cdiv(blocks, 64), 1 }, &.{ p("IQ", bf16, iq), p("POOLED", bf16, pooled), p("POS0", i32p, pos0), p("SC", f32p, sc.scores), int("NB", g.nb) }, &.{ ci("HI", d.index_heads), ci("DI", d.index_dim), ci("RATIO", g.ratio), ci("TOP", top), ci("BB", 64) });
        const lists = [_]aot.Arg{ p("SC", f32p, sc.scores), p("POS0", i32p, pos0), p("IDS", i32p, sc.ids), p("NKR", i32p, sc.nk), p("SPR", i32p, sc.sparse), int("NB", g.nb) };
        const width = pow2(blocks);
        if (width <= AttnGeometry.select_regs) {
            try t.run("_select", .{ rows, 1, 1 }, &lists, &.{ ci("RATIO", g.ratio), ci("TOP", top), ci("IDW", g.idw), ci("BLOCK", width) });
        } else {
            try t.run("_select_tiles", .{ rows, 1, 1 }, &lists, &.{ ci("RATIO", g.ratio), ci("TOP", top), ci("IDW", g.idw), ci("TB", if (rows >= 64) 4096 else 8192) });
        }
    }

    /// attention.qsa_select (a decode window): pool the blocks the window completes, then score and select.
    pub fn qsaSelect(t: Tri, iq: u64, ikc: u64, pooled: u64, pos0: u64, w: u64, inv: u64, sc: AttnScratch, rows: usize, g: AttnGeometry, context: ?usize, d: Dims) !void {
        return t.qsaSelectRope(iq, ikc, pooled, pos0, w, inv, sc, rows, g, context, d, null);
    }

    /// qsaSelect with an image prompt's rotary table (null: text).
    pub fn qsaSelectRope(t: Tri, iq: u64, ikc: u64, pooled: u64, pos0: u64, w: u64, inv: u64, sc: AttnScratch, rows: usize, g: AttnGeometry, context: ?usize, d: Dims, rope: ?Rope) !void {
        try t.qsaPoolRope(ikc, pooled, pos0, w, inv, rows, g, d, rope);
        try t.qsaRows(iq, pooled, pos0, sc, rows, g, context, d);
    }

    // -- a shared round's attention in one launch a kernel (attn_multi.py) ------------------------------------

    /// attn_multi.Step's tables: row position and stream, stream first position and count, a layer's [6][N] pointers.
    pub const Multi = struct { posr: u64, sid: u64, first: u64, counts: u64, n: usize };

    /// Whether the kernel set holds attn_multi's kernels (aot/w6-ks1 and later).
    pub fn multiAvailable(set: *const aot.Set) bool {
        return set.smallestConst("_chunks_multi", "NCH", 0) != null and set.smallestConst("_prep_multi", "PW", 0) != null and
            set.smallestConst("_merge_multi", "NCH", 0) != null and set.smallestConst("_pool_multi", "DI", 0) != null;
    }

    /// attn_multi._prep_multi: every row's attn_prep through its stream's caches (text positions, bf16 caches).
    pub fn prepMulti(t: Tri, m: Multi, cp: u64, pa: u64, qw: u64, kw: u64, iw: u64, inv: u64, q: u64, iq: u64, rows: usize, d: Dims) !void {
        try t.run("_prep_multi", .{ rows, d.heads + d.kv_heads + d.index_heads + 1, 1 }, &.{ p("P", bf16, pa), p("POSR", i32p, m.posr), p("SID", i32p, m.sid), p("CP", "*i64", cp), p("VP", "*i64", cp), p("QW", f32p, qw), p("KW", f32p, kw), p("IW", f32p, iw), p("INV", f32p, inv), p("Q", bf16, q), p("IQ", bf16, iq), aot.float("eps", d.eps), int("N", m.n) }, &.{ ci("PW", d.attnWidth()), ci("NQ", d.heads), ci("NKV", d.kv_heads), ci("HD", d.head_dim), ci("NI", d.index_heads), ci("IHD", d.index_dim), ci("HALF", d.rotary_half), ci("BITS", 0), cb("VISION", false), ci("S1", 11), ci("S2", 10) });
    }

    /// attn_multi._pool_multi: each stream's RATIO-key blocks its rows complete, pooled (`most` the most rows).
    pub fn poolMulti(t: Tri, m: Multi, cp: u64, w: u64, inv: u64, most: usize, g: AttnGeometry, d: Dims) !void {
        try t.run("_pool_multi", .{ m.n, most / g.ratio + 2, 1 }, &.{ p("CP", "*i64", cp), p("VP", "*i64", cp), p("P0", i32p, m.first), p("RS", i32p, m.counts), p("W", f32p, w), p("INV", f32p, inv), aot.float("eps", d.eps), int("N", m.n) }, &.{ ci("DI", d.index_dim), ci("HALF", d.rotary_half), ci("RATIO", g.ratio), cb("VISION", false), ci("S1", 11), ci("S2", 10) });
    }

    /// attn_multi._chunks_multi + _merge_multi: every row over its stream's caches merged into out; `keys` the row max.
    pub fn attentionMulti(t: Tri, m: Multi, cp: u64, q: u64, sc: AttnScratch, out: u64, rows: usize, g: AttnGeometry, keys: usize, d: Dims) !void {
        const h = d.heads;
        const hk = d.kv_heads;
        const grp = h / hk;
        if (grp > 16) return error.TooManyQueryHeads;
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(d.head_dim), -0.5));
        const top = g.budget / g.ratio;
        try t.run("_chunks_multi", .{ rows, hk, g.chunks(keys) }, &.{ p("Q", bf16, q), p("CP", "*i64", cp), p("POSR", i32p, m.posr), p("SID", i32p, m.sid), p("PO", f32p, sc.po), p("PM", f32p, sc.pm), p("PL", f32p, sc.pl), p("IDS", i32p, sc.ids), p("NKR", i32p, sc.nk), int("N", m.n) }, &.{ ci("H", h), ci("HK", hk), ci("D", d.head_dim), ci("G", grp), ci("CH", AttnGeometry.chunk), ci("NCH", g.nch), aot.cf("SCALE", scale), ci("IDW", g.idw), cb("QSA", g.qsa), ci("BITS", 0), ci("RATIO", g.ratio), ci("TOP", top) });
        try t.run("_merge_multi", .{ rows, hk, 1 }, &.{ p("PO", f32p, sc.po), p("PM", f32p, sc.pm), p("PL", f32p, sc.pl), p("POSR", i32p, m.posr), p("OUT", bf16, out), p("NKR", i32p, sc.nk) }, &.{ ci("H", h), ci("HK", hk), ci("D", d.head_dim), ci("G", grp), ci("CH", AttnGeometry.chunk), ci("NCH", g.nch), cb("QSA", g.qsa), ci("BITS", 0), ci("RATIO", g.ratio), ci("TOP", top) });
    }

    /// glue.attn_gate: out [R, H*D] = bf16(o * sigmoid(gate)), the gate from pa's [q | gate] pairs, and group sums.
    pub fn attnGate(t: Tri, o: u64, pa: u64, out: u64, xs: u64, rows: usize, d: Dims) !void {
        try t.run("_attn_gate", .{ rows, d.heads, 1 }, &.{ p("O", bf16, o), p("P", bf16, pa), p("OUT", bf16, out), p("XS", f32p, xs) }, &.{ ci("PW", d.attnWidth()), ci("NQ", d.heads), ci("HD", d.head_dim) });
    }

    // -- the n-gram embedding (PLE) ------------------------------------------------------------------------

    /// glue.ple_embed_bf16: gathered rows -> out [R, heads * dh] and group sums; `scale` 1.0 for the FP8 table (LUT).
    pub fn pleEmbedBf16(t: Tri, v: u64, out: u64, xs: u64, rows: usize, heads: usize, dh: usize, scale: f32) !void {
        try t.run("_ple_embed_bf16", .{ rows, heads, 1 }, &.{ p("V", bf16, v), p("OUT", bf16, out), p("XS", f32p, xs) }, &.{ ci("HEADS", heads), ci("DH", dh), aot.cf("SCALE", scale) });
    }

    /// glue.ple_gate: each stream's gate from its normed key and query, gated values [R, S*D] and squared sums [R, S].
    pub fn pleGate(t: Tri, keys: u64, vals: u64, h: u64, norm_key: u64, norm_query: u64, gated: u64, pss: u64, rows: usize, d: usize, streams: usize, eps: f32) !void {
        try t.run("_ple_gate", .{ rows, 1, 1 }, &.{ p("KEYS", bf16, keys), p("VALS", bf16, vals), p("H", bf16, h), p("NK", f32p, norm_key), p("NQ", f32p, norm_query), p("GATED", bf16, gated), p("PSS", f32p, pss), aot.float("eps", eps) }, &.{ ci("D", d), ci("S", streams), ci("BLOCK", 512) });
    }

    /// glue.ple_conv (one stream): dilated conv over [tail; rows], SiLU, residual adds into hout, normed rows to nrow.
    pub fn pleConv(t: Tri, gated: u64, pss: u64, norm_conv: u64, tail: u64, conv_w: u64, h: u64, hout: u64, nrow: u64, rows: usize, d: usize, streams: usize, taps: usize, dilation: usize, eps: f32) !void {
        try t.run("_ple_conv", .{ rows, streams * d / 512, 1 }, &.{ p("GATED", bf16, gated), p("PSS", f32p, pss), p("NC", f32p, norm_conv), p("TAIL", bf16, tail), p("CW", bf16, conv_w), p("H", bf16, h), p("HOUT", bf16, hout), p("NROW", bf16, nrow), aot.float("eps", eps), int("R", rows) }, &.{ ci("D", d), ci("S", streams), ci("TAPS", taps), ci("DIL", dilation), ci("BLOCK", 512) });
    }

    // -- matmuls -------------------------------------------------------------------------------------------

    /// bf16.matmul: x [M, K] @ w [N, K].T -> out [M, N] (bf16 or fp32); K slices summed in order by `_reduce`.
    pub fn b16mm(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, fp32: bool, part: u64, m: usize, n: usize, k: usize) !void {
        if (k % 64 != 0) return error.KNotBlocked;
        const sk = b16SplitK(n, k);
        const bm: usize = if (m > 128) 128 else 16;
        const oty = if (fp32) f32p else bf16;
        const split = sk > 1;
        try t.run("_b16mm", .{ cdiv(m, bm), cdiv(n, 64), sk }, &.{ p("X", bf16, x), p("W", bf16, w), p("OUT", oty, out), p("PART", if (split) f32p else oty, if (split) part else out), int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", sk), ci("BM", bm), ci("BLOCK_N", 64), ci("BK", 64), cb("F32", fp32) });
        if (split) try t.reduce(part, out, fp32, m * n, sk);
    }

    /// bf16._reduce and nvfp4._reduce: `total` fp32 sums of `sk` slices in order, one rounding for bf16 out.
    fn reduce(t: Tri, part: u64, out: u64, fp32: bool, total: usize, sk: usize) !void {
        try t.run("_reduce", .{ cdiv(total, 1024), 1, 1 }, &.{ p("PART", f32p, part), p("OUT", if (fp32) f32p else bf16, out), int("total", total) }, &.{ ci("SK", sk), ci("BLOCK", 1024), cb("F32", fp32) });
    }

    /// nvfp4.matmul at defaults (shared expert): x [M, K] bf16 @ fp.T -> out [M, N], K slices summed in order.
    pub fn fp4mm(t: Tri, x: u64, x_stride: usize, fp: Fp4, out: u64, fp32: bool, part: u64, m: usize, n: usize, k: usize) !void {
        if (n % 64 != 0) return error.NNotTiled;
        const bm = bucket(m);
        var want: usize = switch (bm) {
            16 => 4,
            32, 64 => 2,
            else => 1,
        };
        if (bm == 16) if (fp4Tuned(n, k)) |tuned| {
            want = tuned.gpi;
        };
        const sk = fp4SplitK(n, k);
        const gpi = gpiFor((k / 16) / sk, want);
        const oty = if (fp32) f32p else bf16;
        const split = sk > 1;
        try t.run("_fp4mm", .{ cdiv(m, bm), n / 64, sk }, &.{ p("X", bf16, x), p("W", if (fp.codes) "*u8" else "*u16", fp.weight), p("S", if (fp.codes) "*u8" else f32p, fp.scale), p("S2", f32p, fp.scale2), p("OUT", oty, out), p("PART", if (split) f32p else oty, if (split) part else out), int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", sk), ci("BM", bm), ci("SBN", 64), ci("BLOCK_N", 64), ci("GPI", gpi), cb("F32", fp32), cb("PACKED", fp.codes) });
        if (split) try t.reduce(part, out, fp32, m * n, sk);
    }

    // -- MoE -----------------------------------------------------------------------------------------------

    /// moe.router: logits [M, E + 1] fp32 = x [M, D] . rows (the router's E rows, then the shared expert's gate).
    pub fn router(t: Tri, x: u64, x_stride: usize, rows_w: u64, out: u64, m: usize, d: usize, ne: usize) !void {
        const bm = bucket(m);
        const be: usize, const bk: usize = if (bm == 16) .{ 32, 256 } else .{ 64, 64 };
        try t.run("_router", .{ cdiv(m, bm), cdiv(ne, be), 1 }, &.{ p("X", bf16, x), p("W", bf16, rows_w), p("OUT", f32p, out), int("M", m), int("x_stride", x_stride) }, &.{ ci("D", d), ci("NE", ne), ci("BM", bm), ci("BLOCK_E", be), ci("BK", bk) });
    }

    /// moe.select_rows: each row's top_k experts and weights, then the shared expert as slot top_k (pick, wts).
    pub fn topkRows(t: Tri, logits: u64, pick: u64, wts: u64, rows: usize, experts: usize, top_k: usize) !void {
        try t.run("_topk_rows", .{ rows, 1, 1 }, &.{ p("L", f32p, logits), p("PICK", i32p, pick), p("WTS", f32p, wts) }, &.{ ci("NE", experts), ci("NL", experts + 1), ci("TOPK", top_k), ci("SLOTS", top_k + 1), ci("BLOCK", pow2(experts + 1)), ci("SLOTP", pow2(top_k + 1)) });
    }

    /// glue.moe_partial (TP=2): a rank's fp32 MoE share sum_k w_k y_k + w_s y_s [R, D], gathered for mode 3.
    pub fn moePartial(t: Tri, y: u64, y_f32: bool, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
        try t.run("_moe_partial", .{ rows, d / 256, 1 }, &.{ p("Y", if (y_f32) f32p else bf16, y), p("WTS", f32p, wts), p("OUT", f32p, out) }, &.{ ci("D", d), ci("TOPK", slots - 1), ci("SLOTS", slots), ci("BLOCK", 256) });
    }

    // -- commit --------------------------------------------------------------------------------------------

    /// forward.shift_windows: old [L, T, C] in place takes rows keep .. keep + T of [old; new rows].
    pub fn shiftWindows(t: Tri, old: u64, new: u64, keep: usize, old_l: usize, new_l: usize, new_row: usize, layers: usize, channels: usize, taps: usize) !void {
        try t.run("_shift_windows", .{ layers, cdiv(channels, 256), 1 }, &.{ p("OLD", bf16, old), p("NEW", bf16, new), int("keep", keep), int("OLD_L", old_l), int("NEW_L", new_l), int("NEW_ROW", new_row) }, &.{ ci("C", channels), ci("T", taps), ci("TP", pow2(taps)), ci("BLOCK", 256) });
    }
};
