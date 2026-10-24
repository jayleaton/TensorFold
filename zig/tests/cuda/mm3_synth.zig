//! A synthetic MiniMax-M3.1-Flash on one GPU (world 1, no checkpoint): seeded EXL3 codes under the production forward.
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("minimax_cuda");

const fw = mm.forward;
const iface = mm.iface;
const weights = mm.weights;
const Config = mm.config.Config;
const exl3 = cuda.exl3;
const rk = cuda.minimax_m3_rows;
const Pool = cuda.minimax_m3_sparse.geom.Pool;

/// Half-bit widths of the synthetic pack: 3-bit experts, 4-bit attention and dense/shared MLPs, a 6-bit head.
pub const expert_k2: u32 = 6;
pub const linear_halves: u32 = 8;
pub const head_halves: u32 = 12;

pub const Options = struct {
    layers: u32 = 1,
    slots: u32 = 2,
    capacity_tokens: u32 = 4096, // positions a slot holds, rounded up to whole pages
    seed: u64 = 0x6d33_5e7d,
    kv8: bool = false,
    first_moe: ?u32 = null, // layers from this one on are routed (null: the config's own moe_layer_freq)
    score_mib: u32 = 64, // the block-score scratch bound (rows_plan; 0: unbounded)
};

pub const Error = error{ BadOptions, PageStatus, BadSlot };

/// What one matrix's trellis takes in bytes at `halves` half-bits a value, stored layout.
fn trellisBytes(k: usize, n: usize, halves: usize) usize {
    return (k / 16) * (n / 16) * halves * 4 * 4;
}

/// Uploads synthetic tensors and keeps every buffer it made; codes are windows into one shared noise buffer.
const Maker = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    rng: std.Random,
    owned: *std.ArrayList(cuda.DeviceBuffer),
    noise: cuda.DeviceBuffer = undefined,

    fn keep(mk: *Maker, b: cuda.DeviceBuffer) !u64 {
        errdefer {
            var x = b;
            x.free();
        }
        try mk.owned.append(mk.gpa, b);
        return b.ptr;
    }

    fn upload(mk: *Maker, bytes: []const u8) !u64 {
        return mk.keep(try cuda.DeviceBuffer.fromHost(mk.d, bytes));
    }

    /// `len` random bytes on the device, filled from the host in 64 MiB pieces.
    fn randomBytes(mk: *Maker, len: usize) !cuda.DeviceBuffer {
        var b = try cuda.DeviceBuffer.alloc(mk.d, len);
        errdefer b.free();
        const piece = try mk.gpa.alloc(u8, @min(len, 64 << 20));
        defer mk.gpa.free(piece);
        var at: usize = 0;
        while (at < len) : (at += piece.len) {
            const n = @min(piece.len, len - at);
            mk.rng.bytes(piece[0..n]);
            try b.upload(at, piece[0..n]);
        }
        return b;
    }

    /// A trellis of `bytes`: a 256-byte aligned window at a random offset of the noise buffer.
    fn codes(mk: *Maker, bytes: usize) u64 {
        std.debug.assert(bytes <= mk.noise.len);
        const span = (mk.noise.len - bytes) / 256;
        return mk.noise.ptr + 256 * mk.rng.uintLessThan(usize, span + 1);
    }

    fn f16s(mk: *Maker, n: usize, comptime fill: anytype, arg: f32) !u64 {
        const h = try mk.gpa.alloc(u16, n);
        defer mk.gpa.free(h);
        for (h) |*v| v.* = @bitCast(@as(f16, @floatCast(fill(mk.rng, arg))));
        return mk.upload(std.mem.sliceAsBytes(h));
    }

    fn f32s(mk: *Maker, n: usize, comptime fill: anytype, arg: f32) !u64 {
        const h = try mk.gpa.alloc(f32, n);
        defer mk.gpa.free(h);
        for (h) |*v| v.* = fill(mk.rng, arg);
        return mk.upload(std.mem.sliceAsBytes(h));
    }

    /// One EXL3 matrix [k] -> [n]: sign `suh`, `svh` of 1/sqrt(k) so a unit-rms row stays near unit rms.
    fn matrix(mk: *Maker, k: u32, n: u32, halves: u32) !exl3.Weight {
        const st = exl3.strides(k, n, halves, false);
        return .{
            .trellis = mk.codes(trellisBytes(k, n, halves)),
            .suh = try mk.f16s(k, sign, 1),
            .svh = try mk.f16s(n, sign, 1 / @sqrt(@as(f32, @floatFromInt(k)))),
            .k = k,
            .n = n,
            .halves = halves,
            .codebook = .mul1,
            .stride_k = st.k,
            .stride_nb = st.nb,
        };
    }

    fn norm(mk: *Maker, n: usize) !u64 {
        return mk.f32s(n, around, 1);
    }

    /// A routed layer's experts at `expert_k2` and its packed tables; `attachTables` wires them once Weights exist.
    fn experts(mk: *Maker, l: u32, e: u32, dims: u32, width: u32, tabs: *std.ArrayList(Pending)) !cuda.exl3_experts.Layer {
        const bytes = trellisBytes(dims, width, expert_k2);
        const ptrs = try mk.gpa.alloc(u64, 3 * e);
        defer mk.gpa.free(ptrs);
        for (ptrs) |*p| p.* = mk.codes(bytes);
        const k2 = try mk.gpa.alloc(i32, e);
        defer mk.gpa.free(k2);
        @memset(k2, @intCast(expert_k2));
        const t: weights.Tables = .{ .layer = @intCast(l), .gate = ptrs[0..e], .up = ptrs[e .. 2 * e], .down = ptrs[2 * e ..], .gate_k2 = k2, .up_k2 = k2, .down_k2 = k2 };
        const packed_ = try weights.packTables(mk.gpa, t);
        defer mk.gpa.free(packed_);
        const base = try mk.upload(packed_);
        try tabs.append(mk.gpa, .{ .layer = @intCast(l), .experts = e, .ptrs = weights.tablePtrs(base, e) });
        const du: f32 = 1 / @sqrt(@as(f32, @floatFromInt(dims)));
        const wu: f32 = 1 / @sqrt(@as(f32, @floatFromInt(width)));
        return .{
            .gate = 0,
            .up = 0,
            .down = 0,
            .gate_k2 = 0,
            .up_k2 = 0,
            .down_k2 = 0,
            .suh_g = try mk.f16s(@as(usize, e) * dims, sign, 1),
            .suh_u = try mk.f16s(@as(usize, e) * dims, sign, 1),
            .svh_g = try mk.f16s(@as(usize, e) * width, sign, du),
            .svh_u = try mk.f16s(@as(usize, e) * width, sign, du),
            .suh_d = try mk.f16s(@as(usize, e) * width, sign, 1),
            .svh_d = try mk.f16s(@as(usize, e) * dims, sign, wu),
            .experts = e,
            .dims = dims,
            .width = width,
            .codebook = .mul1,
            .uniform_k2 = expert_k2,
        };
    }

    fn layer(mk: *Maker, g: *const fw.Geometry, l: u32, tabs: *std.ArrayList(Pending)) !weights.Layer {
        const h = g.hidden;
        const proj: f32 = @sqrt(3 / @as(f32, @floatFromInt(h)));
        var ly: weights.Layer = .{
            .input_ln = try mk.norm(h),
            .post_ln = try mk.norm(h),
            .q_norm = try mk.norm(g.q_w),
            .k_norm = try mk.norm(g.q_w),
            .index_q_norm = try mk.norm(g.q_w),
            .index_k_norm = try mk.norm(g.q_w),
            .index_q_proj = try mk.f16s(@as(usize, g.iq_w) * h, uniform, proj),
            .index_k_proj = try mk.f16s(@as(usize, g.ik_w) * h, uniform, proj),
            .q = try mk.matrix(h, g.q_w, linear_halves),
            .k = try mk.matrix(h, g.kv_w, linear_halves),
            .v = try mk.matrix(h, g.kv_w, linear_halves),
            .o = try mk.matrix(g.q_w, h, linear_halves),
            .ffn = undefined,
        };
        if (!g.isMoe(l)) {
            ly.ffn = .{ .dense = try mk.mlp(h, g.dense_w) };
            return ly;
        }
        ly.ffn = .{ .moe = .{
            .router = try mk.f32s(@as(usize, g.experts) * h, uniform, proj),
            .bias = try mk.f32s(g.experts, uniform, 0.05),
            .experts = try mk.experts(l, g.experts, h, g.moe_w, tabs),
            .shared = try mk.mlp(h, g.shared_w),
        } };
        return ly;
    }

    fn mlp(mk: *Maker, h: u32, w: u32) !weights.Dense {
        return .{ .gate = try mk.matrix(h, w, linear_halves), .up = try mk.matrix(h, w, linear_halves), .down = try mk.matrix(w, h, linear_halves) };
    }

    /// bf16 [vocab, hidden]: a random sign and mantissa at magnitude [2^-6, 2^-5), four values per draw.
    fn embed(mk: *Maker, rows: usize, hidden: usize) !u64 {
        const n = rows * hidden;
        var b = try cuda.DeviceBuffer.alloc(mk.d, n * 2);
        errdefer b.free();
        const piece = try mk.gpa.alloc(u16, @min(n, 32 << 20));
        defer mk.gpa.free(piece);
        var at: usize = 0;
        while (at < n) : (at += piece.len) {
            const len = @min(piece.len, n - at);
            var i: usize = 0;
            while (i < len) : (i += 4) {
                var r = mk.rng.int(u64);
                for (piece[i..@min(i + 4, len)]) |*v| {
                    v.* = @as(u16, @truncate(r & 0x807f)) | ((127 - 6) << 7);
                    r >>= 16;
                }
            }
            try b.upload(at * 2, std.mem.sliceAsBytes(piece[0..len]));
        }
        return mk.keep(b);
    }
};

const Pending = struct { layer: u32, experts: u32, ptrs: weights.TablePtrs };

fn sign(r: std.Random, scale: f32) f32 {
    return if (r.boolean()) scale else -scale;
}

fn uniform(r: std.Random, scale: f32) f32 {
    return (r.float(f32) * 2 - 1) * scale;
}

fn around(r: std.Random, center: f32) f32 {
    return center + (r.float(f32) * 2 - 1) * 0.05;
}

/// The lowest id of the largest logit.
pub fn argmax(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |v, i| {
        if (v > row[best]) best = i;
    }
    return @intCast(best);
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    st: cuda.Stream,
    o: Options,
    cfg: Config,
    geo: fw.Geometry,
    mods: fw.Modules,
    bufs: fw.Buffers,
    f: fw.Forward,
    w: weights.Weights,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    pool: Pool,
    max_pages: u32,
    plan: mm.rows_plan.Plan, // the block-score scratch bound (64 MiB) the forward's buffers are sized by
    arena: cuda.DeviceBuffer,
    mirror: rk.Mirror,
    table: []i32, // host copy of the block tables: slot s's block b is page table[s * max_pages + b]
    last: []f32, // the last prompt chunk's head row (own_vocab logits)
    weight_bytes: u64 = 0,
    chunks: u32 = 0, // prompt chunks run so far

    /// The model, its pages (a seeded permutation of page ids over the slots) and the forward over them.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, st: cuda.Stream, o: Options) !*Model {
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = .{ .gpa = gpa, .d = d, .st = st, .o = o, .cfg = undefined, .geo = undefined, .mods = undefined, .bufs = undefined, .f = undefined, .w = undefined, .pool = undefined, .max_pages = 0, .plan = undefined, .arena = undefined, .mirror = undefined, .table = &.{}, .last = &.{} };
        errdefer m.freeOwned();
        m.cfg = try mm.config.parse(gpa, mm.config.fixture);
        if (o.layers == 0 or o.layers > m.cfg.layers or o.slots == 0 or o.capacity_tokens == 0) return error.BadOptions;
        if (o.first_moe) |first| {
            m.cfg.moe = .empty;
            for (first..m.cfg.layers) |l| m.cfg.moe.set(l);
        }
        const share = try mm.shard.Share.of(&m.cfg, 0, 1);
        m.geo = try fw.Geometry.of(&m.cfg, share, o.layers, iface.max_rows);
        var prng = std.Random.DefaultPrng.init(o.seed);
        const layers = try gpa.alloc(weights.Layer, o.layers);
        errdefer gpa.free(layers);
        try m.makeWeights(prng.random(), layers);
        m.max_pages = @intCast(iface.maxPages(o.capacity_tokens));
        try m.makePages(prng.random());
        errdefer m.freePages();
        m.mods = try fw.Modules.load(d);
        errdefer m.mods.deinit();
        m.plan = mm.rows_plan.Plan.of(m.geo.kvh, m.max_pages, o.score_mib);
        m.bufs = try fw.Buffers.init(d, &m.geo, m.plan.scratch);
        errdefer m.bufs.deinit();
        const pages: fw.Pages = .{ .cache = .{ .pool = m.arena.ptr, .page_bytes = m.pool.pageBytes(), .layer_off = 0, .kvh = m.geo.kvh, .kv8 = o.kv8 }, .layer_bytes = m.pool.layerBytes(), .mirror = &m.mirror };
        m.f = try fw.Forward.init(gpa, d, st, &m.mods, &m.w, &m.geo, &m.bufs, pages, null);
        errdefer m.f.deinit(gpa);
        m.last = try gpa.alloc(f32, m.geo.own_vocab);
        return m;
    }

    fn makeWeights(m: *Model, rng: std.Random, layers: []weights.Layer) !void {
        var tabs: std.ArrayList(Pending) = .empty;
        defer tabs.deinit(m.gpa);
        var mk: Maker = .{ .gpa = m.gpa, .d = m.d, .rng = rng, .owned = &m.owned };
        const g = &m.geo;
        mk.noise = try mk.randomBytes(trellisBytes(g.hidden, g.own_vocab, head_halves) + (64 << 20));
        _ = try mk.keep(mk.noise);
        for (layers, 0..) |*ly, l| ly.* = try mk.layer(g, @intCast(l), &tabs);
        m.w = .{ .embed = try mk.embed(m.cfg.vocab, g.hidden), .final_norm = try mk.norm(g.hidden), .head = try mk.matrix(g.hidden, g.own_vocab, head_halves), .layers = layers };
        for (tabs.items) |t| {
            const fake: weights.Tables = .{ .layer = @intCast(t.layer), .gate = &.{}, .up = &.{}, .down = &.{}, .gate_k2 = &.{}, .up_k2 = &.{}, .down_k2 = &.{} };
            weights.attachTables(&m.w, fake, t.ptrs, .{});
        }
        for (m.owned.items) |b| m.weight_bytes += b.len;
    }

    fn makePages(m: *Model, rng: std.Random) !void {
        m.pool = .{ .kvh = m.geo.kvh, .layers = m.o.layers, .kv8 = m.o.kv8 };
        const n: usize = @as(usize, m.o.slots) * m.max_pages;
        m.table = try m.gpa.alloc(i32, n);
        errdefer m.gpa.free(m.table);
        for (m.table, 0..) |*t, i| t.* = @intCast(i);
        rng.shuffle(i32, m.table);
        m.arena = try cuda.DeviceBuffer.alloc(m.d, n * m.pool.pageBytes());
        errdefer m.arena.free();
        try m.arena.fill8(0, null);
        m.mirror = try rk.Mirror.init(m.d, m.o.slots, m.max_pages, iface.shared_rows);
        errdefer m.mirror.deinit();
        try m.mirror.putAll(m.table);
    }

    fn freePages(m: *Model) void {
        m.mirror.deinit();
        m.arena.free();
        m.gpa.free(m.table);
    }

    fn freeOwned(m: *Model) void {
        for (m.owned.items) |*b| b.free();
        m.owned.deinit(m.gpa);
    }

    pub fn deinit(m: *Model) void {
        m.st.synchronize() catch {};
        m.gpa.free(m.last);
        m.f.deinit(m.gpa);
        m.bufs.deinit();
        m.mods.deinit();
        m.freePages();
        m.gpa.free(m.w.layers);
        m.freeOwned();
        m.gpa.destroy(m);
    }

    /// The forward, for callers that drive `run` themselves (graph mode, timing loops).
    pub fn fwd(m: *Model) *fw.Forward {
        return &m.f;
    }

    pub fn geometry(m: *const Model) *const fw.Geometry {
        return &m.geo;
    }

    /// Waits for the stream and refuses a call that found no page for a row.
    pub fn settle(m: *Model) !void {
        const s = try m.f.dev.status();
        if (s != 0) {
            std.debug.print("FAIL mm3.synth: page status 0x{x}\n", .{s});
            return error.PageStatus;
        }
    }

    /// `rows` head rows of `out` copied to the host, own_vocab logits each (caller frees).
    fn rowsOf(m: *Model, out: fw.Out) ![]f32 {
        const v: usize = m.geo.own_vocab;
        const stride: usize = m.geo.slot;
        const raw = try m.gpa.alloc(f32, out.rows * stride);
        defer m.gpa.free(raw);
        const src: cuda.DeviceBuffer = .{ .d = m.d, .ptr = m.f.address(out), .len = raw.len * 4, .borrowed = true };
        try src.download(0, std.mem.sliceAsBytes(raw));
        const rows = try m.gpa.alloc(f32, out.rows * v);
        for (0..out.rows) |r| @memcpy(rows[r * v ..][0..v], raw[r * stride ..][0..v]);
        return rows;
    }

    /// A prompt on `slot` in `chunk_rows`-row chunks (page multiple; 0: the plan's rows); the last row's token.
    pub fn prefill(m: *Model, slot: u32, ids: []const i32, chunk_rows: u32) !u32 {
        if (slot >= m.o.slots or ids.len == 0 or chunk_rows % 128 != 0 or chunk_rows > iface.prefill_rows) return error.BadOptions;
        if (ids.len > @as(usize, m.max_pages) * 128) return error.BadOptions;
        var at: usize = 0;
        while (at < ids.len) {
            const end = if (chunk_rows == 0) m.plan.chunkEnd(at, ids.len, null) else @min(ids.len, at + chunk_rows);
            m.chunks += 1;
            const last = end == ids.len;
            if (!m.plan.fits(end - at, end - 1)) return error.BadOptions;
            const out = try m.f.run(.{ .ids = ids[at..end], .slot = slot, .pos0 = @intCast(at), .prompt = true, .head = if (last) .last else .none });
            try m.settle();
            if (last) {
                const row = try m.rowsOf(out);
                defer m.gpa.free(row);
                @memcpy(m.last, row);
            }
            at = end;
        }
        return argmax(m.last);
    }

    /// The last prompt chunk's head row, valid until the next `prefill`.
    pub fn lastLogits(m: *const Model) []const f32 {
        return m.last;
    }

    /// One stream's window: `ids` at positions pos .. pos + len - 1 of `slot`; every row's logits (caller frees).
    pub fn decode(m: *Model, slot: u32, pos: u32, ids: []const i32) ![]f32 {
        if (slot >= m.o.slots) return error.BadSlot;
        const out = try m.f.run(.{ .ids = ids, .slot = slot, .pos0 = pos, .head = .all });
        try m.settle();
        return m.rowsOf(out);
    }

    /// Several streams' windows in one forward: rows in `segs` order, each at its own position (caller frees).
    pub fn shared(m: *Model, segs: []const iface.Seg, ids: []const i32, positions: []const i32) ![]f32 {
        for (segs) |s| if (s.slot >= m.o.slots) return error.BadSlot;
        const out = try m.f.run(.{ .ids = ids, .segs = segs, .positions = positions, .head = .all });
        try m.settle();
        return m.rowsOf(out);
    }

    /// Bytes one token takes in kvBytes' copy (every layer: K and V rows, the index key, fp8's scales).
    pub fn kvTokenBytes(m: *const Model) usize {
        return m.pool.tokenBytes();
    }

    /// `slot`'s first `tokens` positions in logical order, per layer: K[h][t], V[h][t], IK[t], then kv8's Ks, Vs.
    pub fn kvBytes(m: *Model, slot: u32, tokens: usize) ![]u8 {
        if (slot >= m.o.slots or tokens > @as(usize, m.max_pages) * 128) return error.BadSlot;
        try m.st.synchronize();
        const all = try m.gpa.alloc(u8, m.arena.len);
        defer m.gpa.free(all);
        try m.arena.download(0, all);
        const p = m.pool;
        const out = try m.gpa.alloc(u8, tokens * p.tokenBytes());
        const row = 128 * p.valueBytes();
        const ik = 128 * 2;
        var at: usize = 0;
        const put = struct {
            fn f(o: []u8, i: *usize, src: []const u8) void {
                @memcpy(o[i.*..][0..src.len], src);
                i.* += src.len;
            }
        }.f;
        for (0..p.layers) |l| {
            for (0..2) |which| for (0..p.kvh) |h| for (0..tokens) |t| {
                const off = p.kv(m.page(slot, t), l, which, h, t % 128);
                put(out, &at, all[off..][0..row]);
            };
            for (0..tokens) |t| put(out, &at, all[p.ik(m.page(slot, t), l, t % 128)..][0..ik]);
            if (p.kv8) for (0..2) |which| for (0..p.kvh) |h| for (0..tokens) |t| {
                put(out, &at, all[p.scale(m.page(slot, t), l, which, h, t % 128)..][0..4]);
            };
        }
        std.debug.assert(at == out.len);
        return out;
    }

    /// The page holding position `pos` of `slot`.
    pub fn page(m: *const Model, slot: u32, pos: usize) usize {
        return @intCast(m.table[@as(usize, slot) * m.max_pages + pos / 128]);
    }
};
