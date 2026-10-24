//! MiniMax-M3.1-Flash tf_mm3_index (bf16 and fp8 pools) under the contract, against minimax_ref, then select.
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

/// A rank's index heads (= local kv heads), layers in the test pool, and the layer read (the other is NaN).
const kvh = 2;
const layers = 2;
const layer = 1;

pub const Cfg = struct { name: []const u8, prompt: bool, kv8: bool };

pub const cfgs = [_]Cfg{
    .{ .name = "cuda minimax_m3 tf_mm3_index decode, fp8 pages (kv8), 2 index heads, paged", .prompt = false, .kv8 = true },
    .{ .name = "cuda minimax_m3 tf_mm3_index prompt chunk, fp8 pages (kv8), 2 index heads, paged", .prompt = true, .kv8 = true },
    .{ .name = "cuda minimax_m3 tf_mm3_index decode, bf16 pages, 2 index heads, paged", .prompt = false, .kv8 = false },
    .{ .name = "cuda minimax_m3 tf_mm3_index prompt chunk, bf16 pages, 2 index heads, paged", .prompt = true, .kv8 = false },
};

/// A score: one fp32 product and 3 adds in a lane, then the 5-step xor tree.
fn depth(_: c.Shape) usize {
    return 9;
}

pub fn spec(f: Cfg) c.Spec {
    return .{
        .name = f.name,
        .accum = .f32,
        .depth = depth,
        .ops = 1, // the scale
        .out = .f32,
        .nan = .{ .stated = "fmax drops a NaN key score; a block whose scores are all NaN is -inf; blocks past a row's own stay unwritten" },
        // rows, index heads, and the positions' bound (the last row's position + 1): edges at a block boundary
        .tiles = .{ .{}, .{}, .{ .size = geom.block, .min = 1 } },
        .served = &.{
            .{ .rows = 8, .n = kvh, .k = 4100 },
            .{ .rows = 16, .n = kvh, .k = 8192 },
            .{ .rows = 1, .n = kvh, .k = 2049 },
            .{ .rows = 64, .n = kvh, .k = 2048 },
        },
    };
}

/// Running worst figures over the shapes gate, printed by `report`.
const Stats = struct { q: f64 = 0, abs: f64 = 0, rel: f64 = 0, sparse: usize = 0, lists: usize = 0, ties: usize = 0 };

pub const Index = struct {
    rig: rig_.Rig,
    f: Cfg,
    s: *const sp.Sparse,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    fill: g.Fill = .random,
    pos: std.ArrayList(i32) = .empty,
    table: std.ArrayList(i32) = .empty,
    iq: std.ArrayList(u16) = .empty,
    /// The index keys of the read layer by logical token (block * 128 + t), bf16, through the last written page.
    ik: std.ArrayList(u16) = .empty,
    img: std.ArrayList(u8) = .empty,
    want: std.ArrayList(f64) = .empty,
    mag: std.ArrayList(f64) = .empty,
    dev: [4]u64 = @splat(0),
    out: u64 = 0,
    sel: u64 = 0,
    nb: usize = 0,
    st: Stats = .{},

    pub fn deinit(self: *Index) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ "pos", "table", "iq", "ik", "img", "want", "mag" }) |n| @field(self, n).deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Index) g.Kernel {
        return .{ .spec = spec(self.f), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Index);

    fn pool(self: *const Index) geom.Pool {
        return .{ .kvh = kvh, .layers = layers, .kv8 = self.f.kv8 };
    }

    fn posOf(self: *const Index, r: usize) usize {
        return @intCast(self.pos.items[r]);
    }

    fn scale() f32 {
        return @floatCast(std.math.pow(f64, geom.d, -0.5));
    }

    pub fn prepare(self: *Index, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != kvh or shape.rows == 0 or shape.k == 0 or shape.k + shape.rows > 16384) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        self.fill = fill;
        const n = shape.rows;
        // positions: a decode window spread over [0, k), the last at k - 1 (stress: all there); a prompt's k + i
        try self.pos.resize(a, n);
        for (self.pos.items, 0..) |*p, i| p.* = @intCast(if (self.f.prompt) shape.k + i else if (fill == .stress or i == n - 1) shape.k - 1 else (i * 7919) % shape.k);
        var top: usize = 0;
        for (self.pos.items) |p| top = @max(top, @as(usize, @intCast(p)));
        const nb = geom.blocks(top);
        self.nb = nb;
        const pages = nb + 3;
        try self.table.resize(a, nb);
        const perm = try a.alloc(i32, pages);
        defer a.free(perm);
        for (perm, 0..) |*x, i| x.* = @intCast(i);
        rng.shuffle(i32, perm);
        @memcpy(self.table.items, perm[0..nb]);
        // index keys: random up to the last position; past it, decoys of +-100 that win any max they wrongly enter
        try self.ik.resize(a, nb * geom.block * geom.d);
        for (0..nb * geom.block) |j| {
            const row = self.ik.items[j * geom.d ..][0..geom.d];
            if (j > top) {
                @memset(row, c.toBf16(if (j % 2 == 0) 100 else -100));
            } else for (row) |*e| {
                e.* = attn.normal(rng);
            }
        }
        try self.iq.resize(a, n * kvh * geom.d);
        for (self.iq.items) |*e| e.* = attn.normal(rng);
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => self.stress(rng, top),
        }
        // the pool image: every byte NaN (bf16, fp32 and e4m3 alike), then the read layer's index keys
        const p = self.pool();
        try self.img.resize(a, pages * p.pageBytes());
        @memset(self.img.items, 0xff);
        for (0..nb) |b| for (0..geom.block) |t| {
            const page: usize = @intCast(self.table.items[b]);
            const src = std.mem.sliceAsBytes(self.ik.items[(b * geom.block + t) * geom.d ..][0..geom.d]);
            @memcpy(self.img.items[p.ik(page, layer, t)..][0..src.len], src);
        };
        try self.computeWant();
        for ([_][]const u8{ std.mem.sliceAsBytes(self.iq.items), std.mem.sliceAsBytes(self.pos.items), std.mem.sliceAsBytes(self.table.items), self.img.items }, &self.dev) |bytes, *d| d.* = try self.rig.input(bytes);
        self.out = try self.rig.output(n * kvh * nb * 4);
        self.sel = try self.rig.scratch(n * kvh * geom.topk * 4);
        return true;
    }

    /// NaN in one query element (every score NaN: -inf blocks), +inf in another, a NaN key, a key with both infinities.
    fn plant(self: *Index) void {
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        self.iq.items[0 * geom.d + 3] = nan; // row 0, head 0
        self.iq.items[(1 * kvh + 1) * geom.d + 17] = inf; // row 1, head 1
        const top = self.posOf(self.shape.rows - 1);
        const keys = top + 1;
        self.ik.items[(keys / 3) * geom.d + 9] = nan;
        const both = (2 * keys) / 3;
        self.ik.items[both * geom.d + 5] = inf;
        self.ik.items[both * geom.d + 6] = c.toBf16(-std.math.inf(f32));
        self.ik.items[(keys - 1) * geom.d + 40] = inf;
    }

    /// Gate 4: mirrored integers 129..255 (score q[0] * scale, sums to 2^21): fp32 exact, fp16 overflows, bf16 loses.
    fn stress(self: *Index, rng: std.Random, top: usize) void {
        const h = geom.d / 2;
        for (0..self.shape.rows * kvh) |i| {
            const q = self.iq.items[i * geom.d ..][0..geom.d];
            for (0..h) |e| {
                q[e] = c.toBf16(@floatFromInt(129 + rng.uintLessThan(u32, 127)));
                q[h + e] = q[e];
            }
        }
        for (0..top + 1) |j| {
            const k = self.ik.items[j * geom.d ..][0..geom.d];
            for (0..h) |e| {
                const v: f32 = @floatFromInt(129 + rng.uintLessThan(u32, 127));
                k[e] = c.toBf16(v);
                k[h + e] = c.toBf16(-v + @as(f32, if (e == 0) 1 else 0));
            }
        }
    }

    /// Float64 block maxima by the stated rule; `mag` is the largest finite score's sum of |terms| (its error's scale).
    fn computeWant(self: *Index) !void {
        const a = self.rig.gpu.gpa;
        const n = self.shape.rows * kvh * self.nb;
        try self.want.resize(a, n);
        try self.mag.resize(a, n);
        const sc: f64 = scale();
        for (0..self.shape.rows) |r| for (0..kvh) |hd| {
            const q = self.iq.items[(r * kvh + hd) * geom.d ..][0..geom.d];
            const p = self.posOf(r);
            for (0..self.nb) |b| {
                const i = (r * kvh + hd) * self.nb + b;
                if (b > p / geom.block) {
                    self.want.items[i] = std.math.nan(f64);
                    self.mag.items[i] = 0;
                    continue;
                }
                var best = -std.math.inf(f64);
                var m: f64 = 0;
                for (b * geom.block..@min((b + 1) * geom.block, p + 1)) |j| {
                    const k = self.ik.items[j * geom.d ..][0..geom.d];
                    var s: f64 = 0;
                    var abs: f64 = 0;
                    for (q, k) |qe, ke| {
                        const t = @as(f64, c.fromBf16(qe)) * @as(f64, c.fromBf16(ke));
                        s += t;
                        abs += @abs(t);
                    }
                    s *= sc;
                    if (std.math.isNan(s)) continue;
                    if (s > best) best = s;
                    if (std.math.isFinite(s)) m = @max(m, abs * sc);
                }
                self.want.items[i] = best;
                self.mag.items[i] = m;
            }
        };
    }

    fn paged(self: *const Index) sp.Paged {
        const p = self.pool();
        return .{ .pool = self.dev[3], .page_bytes = p.pageBytes(), .layer_off = p.layerOff(layer), .kvh = kvh, .tables = self.dev[2], .kv8 = self.f.kv8 };
    }

    fn rows(self: *const Index) sp.Rows {
        return if (self.f.prompt) .{ .n = self.shape.rows, .pos0 = @intCast(self.shape.k) } else .{ .n = self.shape.rows, .pos = self.dev[1] };
    }

    fn scratch(self: *const Index) sp.Scratch {
        return .{ .bmax = self.out, .sel = self.sel, .bm_stride = self.nb };
    }

    pub fn run(self: *Index) !void {
        try self.s.score(self.rig.s, self.dev[0], self.rows(), self.paged(), self.scratch(), self.posOf(self.shape.rows - 1), scale());
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Index) usize {
        return self.shape.rows * kvh * self.nb;
    }

    pub fn reference(self: *Index, want: []f64, mag: []f64) !void {
        @memcpy(want, self.want.items);
        @memcpy(mag, self.mag.items);
    }

    pub fn simulate(self: *Index, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const terms = try a.alloc(f64, geom.d);
        defer a.free(terms);
        const sc: f64 = scale();
        for (0..self.shape.rows) |r| for (0..kvh) |hd| {
            const q = self.iq.items[(r * kvh + hd) * geom.d ..][0..geom.d];
            const p = self.posOf(r);
            for (0..self.nb) |b| {
                const i = (r * kvh + hd) * self.nb + b;
                if (b > p / geom.block) {
                    out[i] = std.math.nan(f64);
                    continue;
                }
                var best = -std.math.inf(f64);
                for (b * geom.block..@min((b + 1) * geom.block, p + 1)) |j| {
                    const k = self.ik.items[j * geom.d ..][0..geom.d];
                    for (terms, q, k) |*t, qe, ke| t.* = @as(f64, c.fromBf16(qe)) * @as(f64, c.fromBf16(ke));
                    const s = contract.rows.sumIn(acc, order, terms) * sc;
                    if (!std.math.isNan(s) and s > best) best = s;
                }
                out[i] = best;
            }
        };
    }

    /// Visible blocks within their float64 bound (or class), the rest unwritten, inputs and guards intact; then select.
    pub fn check(self: *Index, at: g.Gate) !?[]const u8 {
        if (!try self.rig.intact()) return "scoring changed a buffer guard";
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const sp_ = spec(self.f);
        for (self.want.items, self.mag.items, 0..) |w, m, i| {
            const bits = std.mem.readInt(u32, got[4 * i ..][0..4], .little);
            const v: f64 = @as(f32, @bitCast(bits));
            if (std.math.isNan(w)) {
                if (bits != 0xffff_ffff) return "a block past a row's own was written";
                continue;
            }
            if (c.classOf(v) != c.classOf(w)) return "a block maximum's class differs from float64's (stated NaN rule)";
            if (!std.math.isFinite(w)) continue;
            const lim = c.bound(sp_, self.shape, w, m);
            const err = @abs(v - w);
            if (!(err <= lim)) {
                std.debug.print("INDEX MISS {s} at {d}x{d}x{d} cell {d}: got {e:.9} want {e:.9} err {e:.3} bound {e:.3}\n", .{ self.f.name, self.shape.rows, self.shape.n, self.shape.k, i, v, w, err, lim });
                return "a block maximum leaves its float64 bound";
            }
            if (at == .shapes) {
                self.st.q = @max(self.st.q, err / lim);
                self.st.abs = @max(self.st.abs, err);
                self.st.rel = @max(self.st.rel, err / @max(@abs(w), 1e-30));
            }
        }
        const inputs = try self.rig.inputBytes();
        const parts = [_][]const u8{ std.mem.sliceAsBytes(self.iq.items), std.mem.sliceAsBytes(self.pos.items), std.mem.sliceAsBytes(self.table.items), self.img.items };
        var off: usize = 0;
        for (parts) |p| {
            if (off + p.len > inputs.len or !std.mem.eql(u8, p, inputs[off..][0..p.len])) return "scoring changed an input";
            off += p.len;
        }
        if (at == .shapes) return self.selection(got);
        return null;
    }

    /// tf_mm3_select equals selectBlocks over the device maxima, and over float64's but for near ties at the cut.
    fn selection(self: *Index, got: []const u8) !?[]const u8 {
        try self.s.select(self.rig.s, self.rows(), kvh, self.scratch(), .{});
        const lists = try self.rig.bytes(self.sel, self.shape.rows * kvh * geom.topk * 4);
        const sel = try self.rig.gpu.gpa.dupe(u8, lists);
        defer self.rig.gpu.gpa.free(sel);
        var dev_s: [1024]f64 = undefined;
        var ref_s: [1024]f64 = undefined;
        var work: [1024]f64 = undefined;
        var ids: [geom.topk]usize = undefined;
        var rids: [geom.topk]usize = undefined;
        const sp_ = spec(self.f);
        for (0..self.shape.rows) |r| for (0..kvh) |hd| {
            const nb = geom.blocks(self.posOf(r));
            if (nb > dev_s.len) return "more blocks than the check holds";
            const base = (r * kvh + hd) * self.nb;
            for (0..nb) |b| {
                dev_s[b] = @as(f32, @bitCast(std.mem.readInt(u32, got[4 * (base + b) ..][0..4], .little)));
                ref_s[b] = self.want.items[base + b];
            }
            const cnt = mr.selectBlocks(dev_s[0..nb], nb - 1, geom.topk, .{}, &work, &ids);
            const out = sel[(r * kvh + hd) * geom.topk * 4 ..][0 .. geom.topk * 4];
            for (0..geom.topk) |i| {
                const v = std.mem.readInt(i32, out[4 * i ..][0..4], .little);
                const w: i32 = if (i < cnt) @intCast(ids[i]) else -1;
                if (v != w) return "tf_mm3_select over the scored maxima differs from selectBlocks over them";
            }
            self.st.lists += 1;
            if (nb <= geom.topk) continue;
            self.st.sparse += 1;
            if (std.mem.indexOfScalar(usize, ids[0..cnt], nb - 1) == null) return "a sparse row's own block is not selected";
            _ = mr.selectBlocks(ref_s[0..nb], nb - 1, geom.topk, .{}, &work, &rids);
            if (std.mem.eql(usize, ids[0..cnt], rids[0..cnt])) continue;
            // the blocks only one side picked: float64 must not separate them by more than their bounds
            var lo = std.math.inf(f64);
            var hi = -std.math.inf(f64);
            var slack: f64 = 0;
            for (ids[0..cnt]) |b| if (std.mem.indexOfScalar(usize, rids[0..cnt], b) == null) {
                lo = @min(lo, ref_s[b]);
                slack += c.bound(sp_, self.shape, ref_s[b], self.mag.items[base + b]);
            };
            for (rids[0..cnt]) |b| if (std.mem.indexOfScalar(usize, ids[0..cnt], b) == null) {
                hi = @max(hi, ref_s[b]);
                slack += c.bound(sp_, self.shape, ref_s[b], self.mag.items[base + b]);
            };
            if (!(hi - lo <= slack)) return "the selected blocks differ from float64's beyond a near tie";
            self.st.ties += 1;
        };
        return null;
    }

    pub fn report(self: *const Index) void {
        std.debug.print("RESULT {s}: worst {d:.3} of the bound, max abs err {e:.3}, max rel err {e:.3}; select lists {d} ({d} sparse, > 16 blocks), {d} near-tie cuts\n", .{ self.f.name, self.st.q, self.st.abs, self.st.rel, self.st.lists, self.st.sparse, self.st.ties });
    }
};

test "the served shapes are sparse past 2048 keys and fit the launch" {
    for (spec(cfgs[0]).served) |s| {
        try geom.check(s.rows, s.n);
        try std.testing.expect(s.k + s.rows <= 16384);
    }
    try std.testing.expect(geom.blocks(8191) > geom.topk and geom.blocks(4099) > geom.topk);
}
