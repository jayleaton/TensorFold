//! MiniMax-M3.1-Flash forced-block top-16 and selected-block GQA attention under the contract, over NaN-poisoned pages.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const mr = core.minimax_ref;
const sp = cuda.minimax_m3_sparse;
const geom = sp.geom;

/// Local kv heads, layers in the test pool, and the layer read (the other is NaN).
const kvh = 2;
const layers = 2;
const layer = 1;
/// The bf16 pool: the host's shadow of every cell, and the device pool unless the config is fp8.
const pool: geom.Pool = .{ .kvh = kvh, .layers = layers };
/// The fp8 pool the device reads when `Cfg.kv8` (docs/fp8-kv.md): e4m3 K/V rows with an fp32 scale each.
const pool8: geom.Pool = .{ .kvh = kvh, .layers = layers, .kv8 = true };

fn zero(_: c.Shape) usize {
    return 0;
}

pub const Variant = struct { name: []const u8, sw: mr.Switches };

pub const variants = [_]Variant{
    .{ .name = "cuda minimax_m3 tf_mm3_select, own block forced, ties to the lower block", .sw = .{} },
    .{ .name = "cuda minimax_m3 tf_mm3_select, init block 0 inclusive, trailing window 2, ties to the higher block", .sw = .{ .sparse_init_block = 0, .init_mode = .inclusive, .sparse_local_block = 2, .local_mode = .trailing_window, .tie = .higher_block } },
};

pub fn rulesOf(sw: mr.Switches) geom.Rules {
    return .{
        .init_n = @intCast(sw.sparse_init_block),
        .init_inclusive = sw.init_mode == .inclusive,
        .local_n = @intCast(sw.sparse_local_block),
        .local_trailing = sw.local_mode == .trailing_window,
        .higher_tie = sw.tie == .higher_block,
    };
}

pub fn selectSpec(v: Variant) c.Spec {
    return .{
        .name = v.name,
        .accum = .none,
        .depth = zero,
        .out = .raw,
        .nan = .{ .stated = "a NaN block score ranks as -inf, as minimax_ref.topk; forced blocks are +inf; ties by the switch's block order" },
        // rows = query rows, n = blocks of the first row (its position's block + 1), k unused
        .tiles = .{ .{}, .{ .size = geom.topk }, .{} },
        .served = &.{
            .{ .rows = 8, .n = 17, .k = 0 },
            .{ .rows = 1, .n = 1024, .k = 0 },
            .{ .rows = 4, .n = 16, .k = 0 },
            .{ .rows = 3, .n = 3, .k = 0 },
            .{ .rows = 2, .n = 2, .k = 0 },
        },
    };
}

pub const Select = struct {
    rig: rig_.Rig,
    f: *const sp.Sparse,
    v: Variant,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    bm: std.ArrayList(f32) = .empty,
    pos: std.ArrayList(i32) = .empty,
    want: std.ArrayList(i32) = .empty,
    ptr: [3]u64 = @splat(0),

    pub fn deinit(self: *Select) void {
        const a = self.rig.gpu.gpa;
        self.bm.deinit(a);
        self.pos.deinit(a);
        self.want.deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Select) g.Kernel {
        return .{ .spec = selectSpec(self.v), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Select);

    /// Row r's block count: the first row's, one fewer for odd rows (never below 1).
    fn blocksOf(self: *const Select, r: usize) usize {
        return @max(1, self.shape.n - (r % 2) * @min(1, self.shape.n - 1));
    }

    pub fn prepare(self: *Select, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n > 4096) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        const stride = shape.n;
        try self.bm.resize(a, rows * kvh * stride);
        try self.pos.resize(a, rows);
        try self.want.resize(a, rows * kvh * geom.topk);
        // small integer scores make ties on both sides of the cut
        for (self.bm.items) |*x| x.* = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 9))) - 4);
        if (fill == .nonfinite) {
            const special = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), -0.0, 0.0 };
            for (self.bm.items, 0..) |*x, i| if (i % 7 == 3) {
                x.* = special[(i / 7) % special.len];
            };
        }
        var scratch: [4096]f64 = undefined;
        var wide: [4096]f64 = undefined;
        var rids: [geom.topk]usize = undefined;
        for (0..rows) |r| {
            const nb = self.blocksOf(r);
            self.pos.items[r] = @intCast((nb - 1) * geom.block + (r * 37) % geom.block);
            for (0..kvh) |h| {
                const row = self.bm.items[(r * kvh + h) * stride ..][0..nb];
                for (wide[0..nb], row) |*d, s| d.* = s;
                const cnt = mr.selectBlocks(wide[0..nb], nb - 1, geom.topk, self.v.sw, &scratch, &rids);
                const out = self.want.items[(r * kvh + h) * geom.topk ..][0..geom.topk];
                for (out, 0..) |*o, i| o.* = if (i < cnt) @intCast(rids[i]) else -1;
            }
        }
        self.ptr[0] = try self.rig.input(std.mem.sliceAsBytes(self.bm.items));
        self.ptr[1] = try self.rig.input(std.mem.sliceAsBytes(self.pos.items));
        self.ptr[2] = try self.rig.output(self.want.items.len * 4);
        return true;
    }

    pub fn run(self: *Select) !void {
        const sc: sp.Scratch = .{ .bmax = self.ptr[0], .sel = self.ptr[2], .bm_stride = self.shape.n };
        try self.f.select(self.rig.s, .{ .n = self.shape.rows, .pos = self.ptr[1] }, kvh, sc, rulesOf(self.v.sw));
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Select) usize {
        return self.want.items.len * 4;
    }

    pub fn reference(self: *Select, want: []f64, mag: []f64) !void {
        for (want, mag, std.mem.sliceAsBytes(self.want.items)) |*w, *m, byte| {
            w.* = @floatFromInt(byte);
            m.* = 0;
        }
    }

    pub fn check(self: *Select, at: g.Gate) !?[]const u8 {
        _ = at;
        if (!try self.rig.intact()) return "selection changed a buffer guard";
        const inputs = try self.rig.inputBytes();
        const bm = std.mem.sliceAsBytes(self.bm.items);
        const pos = std.mem.sliceAsBytes(self.pos.items);
        if (inputs.len != bm.len + pos.len or !std.mem.eql(u8, bm, inputs[0..bm.len]) or !std.mem.eql(u8, pos, inputs[bm.len..])) return "selection changed its scores or positions";
        const got = try self.rig.outputBytes();
        return if (std.mem.eql(u8, std.mem.sliceAsBytes(self.want.items), got)) null else "selected block ids differ from minimax_ref.selectBlocks (ascending, -1 padded)";
    }
};

/// A decode window (positions given) or a prompt chunk (positions k, k + 1, ...).
pub const Cfg = struct { name: []const u8, prompt: bool, kv8: bool = false };

pub const cfgs = [_]Cfg{
    .{ .name = "cuda minimax_m3 tf_mm3_attn decode, 32 q heads over 2 kv heads, paged", .prompt = false },
    .{ .name = "cuda minimax_m3 tf_mm3_attn prompt chunk, 32 q heads over 2 kv heads, paged", .prompt = true },
    .{ .name = "cuda minimax_m3 tf_mm3_attn decode, fp8 K/V pages (kv8), 32 q heads over 2 kv heads, paged", .prompt = false, .kv8 = true },
    .{ .name = "cuda minimax_m3 tf_mm3_attn prompt chunk, fp8 K/V pages (kv8), 32 q heads over 2 kv heads, paged", .prompt = true, .kv8 = true },
};

fn keyChain(s: c.Shape) usize {
    return @min(s.k, geom.topk * geom.block) + 32 + 32;
}

pub fn attnSpec(f: Cfg) c.Spec {
    return .{
        .name = f.name,
        .accum = .f32,
        .depth = keyChain,
        // the fp8 reader scales the dot and the weight: two more fp32 roundings in a score and one in a weight
        .ops = if (f.kv8) 3 else 1,
        .out = .bf16,
        // rows, the 32 query heads, and the positions' bound (the last row's position + 1): edges at a block boundary
        .tiles = .{ .{}, .{}, .{ .size = geom.block, .min = 1 } },
        .served = &.{ .{ .rows = 16, .n = 32, .k = 3000 }, .{ .rows = 4, .n = 32, .k = 2048 }, .{ .rows = 2, .n = 32, .k = 129 }, .{ .rows = 8, .n = 32, .k = 128 } },
        .sums = .{ .why = attn.why },
    };
}

/// attn.mirrored with e4m3-exact values and a +-448 in every row (scale 1); the 1 residue pairs are integers 9..16.
fn mirrored8(rng: std.Random, out: []f32, e: usize) void {
    const h = out.len / 2;
    for (0..h) |i| {
        const v: f32 = if (i < 8) @floatFromInt(9 + rng.uintLessThan(u32, 8)) else @floatCast(mr.e4m3Value(@intCast(0x10 + rng.uintLessThan(u32, 0x6f))));
        out[i] = v;
        out[h + i] = (if (i < 8) @as(f32, 1) else 0) - v;
    }
    if (out.len % 2 == 1) out[out.len - 1] = 0;
    if (e >= 8) for (out, 0..) |*y, i| {
        if (i % 120 == e - 8) y.* = if (i % 2 == 0) 448 else -448;
    };
}

pub const Attn = struct {
    rig: rig_.Rig,
    f: Cfg,
    s: *const sp.Sparse,
    r: ?attn.Ref = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    pos: std.ArrayList(i32) = .empty,
    sel: std.ArrayList(i32) = .empty,
    table: std.ArrayList(i32) = .empty,
    q: std.ArrayList(u16) = .empty,
    cache: std.ArrayList(u16) = .empty,
    /// fp8 only: the pool bytes the device reads, and the float64 value each bf16 cell of `cache` decodes to.
    img: std.ArrayList(u8) = .empty,
    dec: std.ArrayList(f64) = .empty,
    dev: [5]u64 = @splat(0),
    out: u64 = 0,
    blocks_max: usize = 0,

    pub fn deinit(self: *Attn) void {
        const a = self.rig.gpu.gpa;
        if (self.r) |*r| r.deinit();
        inline for (.{ "pos", "sel", "table", "q", "cache", "img", "dec" }) |n| @field(self, n).deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Attn) g.Kernel {
        return .{ .spec = attnSpec(self.f), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Attn);

    fn posOf(self: *const Attn, r: usize) usize {
        return @intCast(self.pos.items[r]);
    }

    fn cell(self: *Attn, page: usize, which: usize, h: usize, t: usize, e: usize) *u16 {
        return &self.cache.items[(pool.kv(page, layer, which, h, t) >> 1) + e];
    }

    fn rowOf(self: *Attn, page: usize, which: usize, h: usize, t: usize) []u16 {
        return self.cache.items[pool.kv(page, layer, which, h, t) >> 1 ..][0..geom.d];
    }

    pub fn prepare(self: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != kvh * geom.group or shape.rows == 0 or shape.k == 0 or shape.k + shape.rows > 8192) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const n = shape.rows;
        // positions: a decode window's spread over [0, k), the last at k - 1; a prompt chunk's k + i
        try self.pos.resize(a, n);
        for (self.pos.items, 0..) |*p, i| p.* = @intCast(if (self.f.prompt) shape.k + i else if (fill == .stress or i == n - 1) shape.k - 1 else (i * 7919) % shape.k);
        var top: usize = 0;
        for (self.pos.items) |p| top = @max(top, @as(usize, @intCast(p)));
        const nb = geom.blocks(top);
        self.blocks_max = nb;
        // the pool's pages in a shuffled order; the table maps block -> page id
        const pages = nb + 3;
        try self.table.resize(a, nb);
        var perm = try a.alloc(i32, pages);
        defer a.free(perm);
        for (perm, 0..) |*x, i| x.* = @intCast(i);
        rng.shuffle(i32, perm);
        @memcpy(self.table.items, perm[0..nb]);
        // the cache: the other layer and every token not yet written are NaN, the read layer random
        try self.cache.resize(a, pages * pool.pageBytes() / 2);
        @memset(self.cache.items, c.toBf16(std.math.nan(f32)));
        for (0..nb) |b| for (0..geom.block) |t| {
            const page: usize = @intCast(self.table.items[b]);
            if (b * geom.block + t > top) continue;
            for (0..2) |w| for (0..kvh) |h| for (0..geom.d) |e| {
                self.cell(page, w, h, t, e).* = attn.normal(rng);
            };
            const ik = (pool.ik(page, layer, t) >> 1);
            @memset(self.cache.items[ik..][0..geom.d], 0);
        };
        // selected lists: every block when a row has at most 16, else its own block and 15 others; ascending, -1 padded
        try self.sel.resize(a, n * kvh * geom.topk);
        for (0..n) |r| for (0..kvh) |h| {
            const own = geom.blocks(self.posOf(r)) - 1;
            const out = self.sel.items[(r * kvh + h) * geom.topk ..][0..geom.topk];
            @memset(out, -1);
            if (own + 1 <= geom.topk) {
                for (out[0 .. own + 1], 0..) |*o, i| o.* = @intCast(i);
            } else {
                var chosen: [geom.topk]i32 = undefined;
                chosen[0] = @intCast(own);
                var m: usize = 1;
                while (m < geom.topk) {
                    const cand: i32 = @intCast(rng.uintLessThan(usize, own));
                    if (std.mem.indexOfScalar(i32, chosen[0..m], cand) == null) {
                        chosen[m] = cand;
                        m += 1;
                    }
                }
                std.mem.sort(i32, &chosen, {}, std.sort.asc(i32));
                @memcpy(out, &chosen);
            }
        };
        try self.q.resize(a, n * shape.n * geom.d);
        for (self.q.items) |*e| e.* = attn.normal(rng);
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(a, geom.d, geom.topk * geom.block);
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        if (self.f.kv8) try self.quantize(nb);
        const poolb = if (self.f.kv8) self.img.items else std.mem.sliceAsBytes(self.cache.items);
        for ([_][]const u8{ std.mem.sliceAsBytes(self.q.items), std.mem.sliceAsBytes(self.sel.items), std.mem.sliceAsBytes(self.pos.items), std.mem.sliceAsBytes(self.table.items), poolb }, &self.dev) |bytes, *p| p.* = try self.rig.input(bytes);
        self.out = try self.rig.output(n * shape.n * geom.d * 2);
        return true;
    }

    /// The fp8 pool image from the bf16 cells by the host rule, every other byte NaN; `dec` is what cells read back as.
    fn quantize(self: *Attn, nb: usize) !void {
        const a = self.rig.gpu.gpa;
        try self.img.resize(a, self.cache.items.len * 2 / pool.pageBytes() * pool8.pageBytes());
        @memset(self.img.items, 0xff);
        try self.dec.resize(a, self.cache.items.len);
        @memset(self.dec.items, std.math.nan(f64));
        var top: usize = 0;
        for (self.pos.items) |p| top = @max(top, @as(usize, @intCast(p)));
        for (0..nb) |b| for (0..geom.block) |t| {
            const page: usize = @intCast(self.table.items[b]);
            if (b * geom.block + t > top) continue;
            for (0..2) |w| for (0..kvh) |h| {
                var x: [geom.d]f32 = undefined;
                for (&x, 0..) |*o, e| o.* = c.fromBf16(self.cell(page, w, h, t, e).*);
                const at = pool8.kv(page, layer, w, h, t);
                const sc = mr.quantRowF32(&x, self.img.items[at..][0..geom.d]);
                std.mem.writeInt(u32, self.img.items[pool8.scale(page, layer, w, h, t)..][0..4], @bitCast(sc), .little);
                for (0..geom.d) |e| self.dec.items[(pool.kv(page, layer, w, h, t) >> 1) + e] = mr.e4m3Value(self.img.items[at + e]) * @as(f64, sc);
            };
        };
    }

    /// Row r's key list for kv head h: the tokens of its selected blocks up to its position, as (block, token).
    fn key(self: *const Attn, r: usize, h: usize, i: usize) struct { b: usize, t: usize } {
        const ids = self.sel.items[(r * kvh + h) * geom.topk ..][0..geom.topk];
        const p = self.posOf(r);
        var left = i;
        for (ids) |b| {
            if (b < 0) break;
            const bb: usize = @intCast(b);
            const valid = @min(geom.block, p + 1 - bb * geom.block);
            if (left < valid) return .{ .b = bb, .t = left };
            left -= valid;
        }
        unreachable;
    }

    fn keysOf(self: *const Attn, r: usize) usize {
        const p = self.posOf(r);
        var n: usize = 0;
        for (self.sel.items[(r * kvh) * geom.topk ..][0..geom.topk]) |b| {
            if (b < 0) break;
            n += @min(geom.block, p + 1 - @as(usize, @intCast(b)) * geom.block);
        }
        return n;
    }

    /// NaN query; +inf / -inf values in one column; a NaN key; +inf at the first key.
    fn plant(self: *Attn) void {
        const n = self.shape.rows;
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        self.q.items[3] = nan;
        if (self.f.kv8) {
            // a bad value poisons its whole row: plant at the last rows' own keys
            const l = n - 1;
            const kv = self.key(l, 0, self.keysOf(l) - 1);
            self.cell(@intCast(self.table.items[kv.b]), 1, 0, kv.t, 7).* = inf;
            if (n > 2) {
                const kk = self.key(l - 1, 1, self.keysOf(l - 1) - 1);
                self.cell(@intCast(self.table.items[kk.b]), 0, 1, kk.t, 9).* = nan;
            }
            return;
        }
        const k0 = self.key(0, 0, self.keysOf(0) - 1);
        self.cell(@intCast(self.table.items[k0.b]), 1, 0, k0.t, 7).* = inf;
        if (n > 1) {
            const k1 = self.key(1, 1, self.keysOf(1) - 1);
            self.cell(@intCast(self.table.items[k1.b]), 1, 1, k1.t, 9).* = ninf;
        }
        if (n > 3) {
            const k3 = self.key(3, 0, 0);
            self.cell(@intCast(self.table.items[k3.b]), 0, 0, k3.t, 0).* = nan;
        }
        if (n > 4) {
            const k4 = self.key(4, 1, 0);
            self.cell(@intCast(self.table.items[k4.b]), 1, 1, k4.t, 13).* = inf;
        }
    }

    /// Gate 4: every key alike (p = 1), values mirrored over the keys row 0 sees, 0 elsewhere. All rows share one list.
    fn stress(self: *Attn, rng: std.Random) !void {
        const n = self.shape.rows;
        // one list for every row: row 0's
        for (1..n) |r| @memcpy(self.sel.items[r * kvh * geom.topk ..][0 .. kvh * geom.topk], self.sel.items[0 .. kvh * geom.topk]);
        const a = self.rig.gpu.gpa;
        const key0 = try a.alloc(u16, kvh * geom.d);
        defer a.free(key0);
        for (key0) |*e| e.* = attn.normal(rng);
        for (self.table.items) |pg| for (0..geom.block) |t| for (0..kvh) |h| {
            @memcpy(self.rowOf(@intCast(pg), 0, h, t), key0[h * geom.d ..][0..geom.d]);
            @memset(self.rowOf(@intCast(pg), 1, h, t), 0);
        };
        const nk = self.keysOf(0);
        const col = try a.alloc(f32, nk);
        defer a.free(col);
        for (0..kvh) |h| for (0..geom.d) |e| {
            if (self.f.kv8) mirrored8(rng, col, e) else attn.mirrored(rng, col);
            for (col, 0..) |y, i| {
                const k = self.key(0, h, i);
                self.cell(@intCast(self.table.items[k.b]), 1, h, k.t, e).* = c.toBf16(y);
            }
        };
        // a row's own-block tokens past row 0's are zero values with equal keys: p = 1 each, V = 0
    }

    fn paged(self: *const Attn) sp.Paged {
        const dev = if (self.f.kv8) pool8 else pool;
        return .{ .pool = self.dev[4], .page_bytes = dev.pageBytes(), .layer_off = dev.layerOff(layer), .kvh = kvh, .tables = self.dev[3], .kv8 = self.f.kv8 };
    }

    pub fn run(self: *Attn) !void {
        const n = self.shape.rows;
        const rows: sp.Rows = if (self.f.prompt) .{ .n = n, .pos0 = @intCast(self.shape.k) } else .{ .n = n, .pos = self.dev[2] };
        const sc: sp.Scratch = .{ .bmax = 0, .sel = self.dev[1], .bm_stride = 0 };
        const scale32: f32 = @floatCast(self.scale());
        try self.s.attend(self.rig.s, self.dev[0], rows, self.paged(), sc, self.out, scale32);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Attn) usize {
        return self.shape.rows * self.shape.n * geom.d;
    }

    /// Output head i (row i / 32, head i % 32): the keys of the row's selected blocks up to its position.
    pub fn load(self: *Attn, i: usize, hd: *attn.Head) !void {
        const r = i / self.shape.n;
        const head = i % self.shape.n;
        const h = head / geom.group;
        for (hd.q, 0..) |*x, e| x.* = c.fromBf16(self.q.items[i * geom.d + e]);
        const nk = self.keysOf(r);
        for (0..nk) |j| {
            const k = self.key(r, h, j);
            const page: usize = @intCast(self.table.items[k.b]);
            for (0..geom.d) |e| {
                if (self.f.kv8) {
                    hd.k[j * geom.d + e] = self.dec.items[(pool.kv(page, layer, 0, h, k.t) >> 1) + e];
                    hd.v[j * geom.d + e] = self.dec.items[(pool.kv(page, layer, 1, h, k.t) >> 1) + e];
                } else {
                    hd.k[j * geom.d + e] = c.fromBf16(self.cell(page, 0, h, k.t, e).*);
                    hd.v[j * geom.d + e] = c.fromBf16(self.cell(page, 1, h, k.t, e).*);
                }
            }
        }
        hd.n = nk;
    }

    fn heads(self: *Attn) usize {
        return self.shape.rows * self.shape.n;
    }

    fn scale(_: *const Attn) f64 {
        return std.math.pow(f64, geom.d, -0.5);
    }

    pub fn check(self: *Attn, gate: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.heads(), self.scale(), try self.rig.outputBytes(), gate);
    }

    pub fn reference(self: *Attn, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.heads(), self.scale(), want, mag);
    }

    pub fn simulate(self: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.heads(), self.scale(), acc, order, out);
    }
};
