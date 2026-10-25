//! MiniMax-M3.1-Flash row kernels under the contract on CUDA (run on spark2, GB10 sm_121).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const Dense = contract.dense.Dense;
const mr = core.minimax_ref;
const rk = cuda.minimax_m3_rows;
const geom = rk.geom;

const experts = 128;
const hidden = 6144;

fn chain(s: c.Shape) usize {
    return s.k / 128 + 3 + 5;
}

pub const Form = enum { bf16, f32 };

pub fn routerSpec(f: Form) c.Spec {
    return .{
        .name = if (f == .bf16) "cuda minimax_m3_rows tf_mm3_router_bf16" else "cuda minimax_m3_rows tf_mm3_router_f32",
        .accum = .f32,
        .depth = chain,
        .out = .f32,
        .replaces = if (f == .f32) "tf_mm3_router_bf16" else null,
        // 4 rows a warp, 4 experts a block, 128 values a step of the reduction
        .tiles = .{ .{ .size = geom.router_rows }, .{ .size = geom.router_warps, .step = 1, .min = 1 }, .{ .size = geom.router_chunk, .step = geom.router_chunk, .min = geom.router_chunk } },
        .served = &.{ .{ .rows = 1, .n = experts, .k = hidden }, .{ .rows = 5, .n = experts, .k = hidden }, .{ .rows = 17, .n = experts, .k = hidden }, .{ .rows = 64, .n = experts, .k = hidden } },
    };
}

pub const Logits = struct {
    rig: rig_.Rig,
    form: Form,
    k: *const rk.Rows,
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    base: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Logits) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Logits) g.Kernel {
        return .{ .spec = routerSpec(self.form), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Logits);

    pub fn prepare(self: *Logits, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % geom.router_chunk != 0) return false;
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 1);
        const w = try a.alloc(f32, shape.n * shape.k);
        defer a.free(w);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6d6d3372);
        const rng = prng.random();
        const k = shape.k;
        for (0..shape.n) |j| for (0..k) |i| {
            // gate 4: integers mirrored into the second half, so fp32 sums are exact in any order
            const v: f32 = if (fill != .stress) (rng.float(f32) - 0.5) * 0.1 else if (i < k / 2) @floatFromInt(1 + rng.uintLessThan(u32, 7)) else -w[j * k + i - k / 2];
            w[j * k + i] = v;
            d.w[j * k + i] = v;
        };
        if (self.form == .f32) {
            const xf = try a.alloc(f32, d.x.len);
            defer a.free(xf);
            for (xf, d.x) |*o, h| o.* = c.fromBf16(h);
            self.x = try self.rig.input(std.mem.sliceAsBytes(xf));
        } else self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(w));
        self.out = try self.rig.output(shape.rows * shape.n * 4);
        if (self.form == .f32) self.base = try self.rig.scratch(shape.rows * shape.n * 4);
        return true;
    }

    fn launch(self: *Logits, x_f32: bool, x: u64, out: u64) !void {
        const sh = self.d.?.shape;
        try self.k.routerLogits(self.rig.s, .{ .experts = @intCast(sh.n), .k = @intCast(sh.k) }, x, sh.k, x_f32, self.w, out, sh.rows);
    }

    pub fn run(self: *Logits) !void {
        try self.launch(self.form == .f32, self.x, self.out);
        try self.rig.s.synchronize();
    }

    /// The bf16 form on the same values.
    pub fn replaced(self: *Logits) ![]const u8 {
        const d = self.d.?;
        const xb = try self.rig.gpu.gpa.alloc(u8, d.x.len * 2);
        defer self.rig.gpu.gpa.free(xb);
        @memcpy(xb, std.mem.sliceAsBytes(d.x));
        const dev = try self.rig.scratch(xb.len);
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = dev, .len = xb.len };
        try b.upload(0, xb);
        try self.launch(false, dev, self.base);
        return self.rig.bytes(self.base, self.count() * 4);
    }

    pub fn count(self: *Logits) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Logits, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Logits, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};

const nq = 32;
const niq = 2;
const kvh = 2;

/// The test pool: one layer of the bf16 layout, or of the fp8 layout (K and V as e4m3 rows with an fp32 scale each).
fn poolOf(kv8: bool) cuda.minimax_m3_sparse.geom.Pool {
    return .{ .kvh = kvh, .layers = 1, .kv8 = kv8 };
}

/// A decode window (positions given) or a prompt chunk (positions k, k + 1, ...); `kv8` writes the fp8 pool.
pub const Cfg = struct { name: []const u8, prompt: bool, kv8: bool = false };

pub const cfgs = [_]Cfg{
    .{ .name = "cuda minimax_m3_rows tf_mm3_rows decode, 32 q + 2 index heads, 2 kv heads, paged", .prompt = false },
    .{ .name = "cuda minimax_m3_rows tf_mm3_rows prompt chunk, 32 q + 2 index heads, 2 kv heads, paged", .prompt = true },
    .{ .name = "cuda minimax_m3_rows tf_mm3_rows decode, fp8 K/V pages (kv8), 32 q + 2 index heads, 2 kv heads, paged", .prompt = false, .kv8 = true },
    .{ .name = "cuda minimax_m3_rows tf_mm3_rows prompt chunk, fp8 K/V pages (kv8), 32 q + 2 index heads, 2 kv heads, paged", .prompt = true, .kv8 = true },
};

fn normDepth(_: c.Shape) usize {
    return 4 + 5 + 3;
}

pub fn rowsSpec(f: Cfg) c.Spec {
    return .{
        .name = f.name,
        .accum = .f32,
        .depth = normDepth,
        // div, sqrt, add, div, two products, the rotation's three, the weight add
        .ops = 14,
        .out = .bf16,
        // rows, q heads, and the positions' bound (the last row's position + 1): edges at a page boundary
        .tiles = .{ .{}, .{}, .{ .size = geom.block, .min = 1 } },
        .served = &.{ .{ .rows = 8, .n = nq, .k = 3000 }, .{ .rows = 5, .n = nq, .k = 129 }, .{ .rows = 16, .n = nq, .k = 128 }, .{ .rows = 6, .n = nq, .k = 1_000_003 } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "a norm's sum of squares is not a dot the fp16 and bf16 accumulators can be made to miss" },
    };
}

pub const Writer = struct {
    rig: rig_.Rig,
    f: Cfg,
    k: *const rk.Rows,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    arena: std.heap.ArenaAllocator,
    pos: []i32 = &.{},
    table: []i32 = &.{},
    /// Decode: row r's table starts at r * blocks, so rows at one position own distinct pages; none for a prompt chunk.
    row_tab: []i32 = &.{},
    blocks: usize = 0,
    q: []u16 = &.{},
    kk: []u16 = &.{},
    vv: []u16 = &.{},
    iq: []u16 = &.{},
    ik: []u16 = &.{},
    w: [4][128]f32 = undefined,
    dev: [13]u64 = @splat(0),
    pages: usize = 0,

    pub fn deinit(self: *Writer) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Writer) g.Kernel {
        return .{ .spec = rowsSpec(self.f), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Writer);

    fn fp16(v: f32) u16 {
        const h: f16 = @floatCast(v);
        return @bitCast(h);
    }

    fn fp16v(h: u16) f32 {
        const x: f16 = @bitCast(h);
        return x;
    }

    pub fn prepare(self: *Writer, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != nq or shape.rows == 0 or shape.k == 0 or shape.k + shape.rows > 1_100_000) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const n = shape.rows;
        self.pos = try a.alloc(i32, n);
        for (self.pos, 0..) |*p, i| p.* = @intCast(if (self.f.prompt) shape.k + i else if (i == n - 1) shape.k - 1 else (i * 7919) % shape.k);
        // Prompt chunk: one stream, shuffled page ids. Decode: a stream a row (row_tab), as positions repeat at k 1.
        var lo: usize = std.math.maxInt(usize);
        var hi: usize = 0;
        for (self.pos) |p| {
            lo = @min(lo, @as(usize, @intCast(p)) / geom.block);
            hi = @max(hi, @as(usize, @intCast(p)) / geom.block);
        }
        const blocks = hi + 1;
        self.blocks = blocks;
        if (self.f.prompt) {
            self.pages = (hi - lo + 1) + 2;
            self.table = try a.alloc(i32, blocks);
            self.row_tab = &.{};
            @memset(self.table, -1);
            const perm = try a.alloc(i32, self.pages);
            for (perm, 0..) |*x, i| x.* = @intCast(i);
            rng.shuffle(i32, perm);
            for (lo..blocks) |b| self.table[b] = perm[b - lo];
        } else {
            self.pages = n + 2;
            self.table = try a.alloc(i32, n * blocks);
            self.row_tab = try a.alloc(i32, n);
            @memset(self.table, -1);
            const perm = try a.alloc(i32, self.pages);
            for (perm, 0..) |*x, i| x.* = @intCast(i);
            rng.shuffle(i32, perm);
            for (self.pos, self.row_tab, 0..) |p, *t, r| {
                t.* = @intCast(r * blocks);
                self.table[r * blocks + @as(usize, @intCast(p)) / geom.block] = perm[r];
            }
        }
        // x in [-4, 4); gate 4 (stress) uses small integers so the sum of squares is exact
        inline for (.{ .{ "q", n * nq * 128 }, .{ "kk", n * kvh * 128 }, .{ "vv", n * kvh * 128 } }) |p| {
            @field(self, p[0]) = try a.alloc(u16, p[1]);
            for (@field(self, p[0])) |*e| e.* = c.toBf16(if (fill == .stress) @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -3, 3))) else (rng.float(f32) - 0.5) * 8);
        }
        self.iq = try a.alloc(u16, n * niq * 128);
        self.ik = try a.alloc(u16, n * 128);
        for (self.iq) |*e| e.* = fp16(if (fill == .stress) @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -3, 3))) else (rng.float(f32) - 0.5) * 8);
        for (self.ik) |*e| e.* = fp16(if (fill == .stress) @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -3, 3))) else (rng.float(f32) - 0.5) * 8);
        for (&self.w) |*row| {
            for (row) |*e| e.* = 1 + (rng.float(f32) - 0.5);
        }
        if (fill == .nonfinite) {
            self.q[5] = c.toBf16(std.math.nan(f32));
            if (n > 1) self.q[(1 * nq + 3) * 128 + 70] = c.toBf16(std.math.inf(f32));
            if (n > 2) self.iq[(2 * niq + 1) * 128 + 9] = 0x7c00;
        }
        const lay = poolOf(self.f.kv8).pageBytes();
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.kk));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.vv));
        self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(self.iq));
        self.dev[4] = try self.rig.input(std.mem.sliceAsBytes(self.ik));
        self.dev[5] = try self.rig.input(std.mem.sliceAsBytes(self.w[0..]));
        self.dev[6] = try self.rig.input(std.mem.sliceAsBytes(self.table));
        self.dev[7] = try self.rig.input(std.mem.sliceAsBytes(self.pos));
        self.dev[8] = try self.rig.output(self.q.len * 2);
        self.dev[9] = try self.rig.output(self.iq.len * 2);
        self.dev[10] = try self.rig.scratch(self.pages * lay);
        self.dev[11] = try self.rig.scratch(4);
        self.dev[12] = if (self.f.prompt) 0 else try self.rig.input(std.mem.sliceAsBytes(self.row_tab));
        return true;
    }

    fn opts(self: *const Writer) rk.Opts {
        return .{ .flags = .{ .kv8 = self.f.kv8 } };
    }

    pub fn run(self: *Writer) !void {
        const n = self.shape.rows;
        const rows: cuda.minimax_m3_sparse.Rows = if (self.f.prompt) .{ .n = n, .pos0 = self.pos[0] } else .{ .n = n, .pos = self.dev[7] };
        const p: cuda.minimax_m3_sparse.Paged = .{ .pool = self.dev[10], .page_bytes = poolOf(self.f.kv8).pageBytes(), .layer_off = 0, .kvh = kvh, .tables = self.dev[6], .row_tab = self.dev[12], .kv8 = self.f.kv8 };
        const status = self.dev[11];
        const in: rk.Inputs = .{ .q = self.dev[0], .q_ld = nq * 128, .k = self.dev[1], .k_ld = kvh * 128, .v = self.dev[2], .v_ld = kvh * 128, .iq = self.dev[3], .iq_ld = niq * 128, .ik = self.dev[4], .ik_ld = 128 };
        const nm: rk.Norms = .{ .q = self.dev[5], .k = self.dev[5] + 512, .iq = self.dev[5] + 1024, .ik = self.dev[5] + 1536 };
        const out: rk.Outputs = .{ .q = self.dev[8], .q_ld = nq * 128, .iq = self.dev[9], .iq_ld = niq * 128 };
        try self.k.qkRope(self.rig.s, .{}, self.opts(), in, nm, out, rows, p, @intCast(self.blocks), status);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Writer) usize {
        return self.q.len + self.iq.len;
    }

    /// Row r's head i of the q scratch (kind 0) or the index query (kind 1): float64 norm, rotation, terms' magnitude.
    fn head(self: *const Writer, kind: usize, r: usize, i: usize, y: *[128]f64, m: *[128]f64) void {
        const p: usize = @intCast(self.pos[r]);
        const src = if (kind == 0) self.q[(r * nq + i) * 128 ..][0..128] else self.iq[(r * niq + i) * 128 ..][0..128];
        var x: [128]f64 = undefined;
        var w: [128]f64 = undefined;
        for (&x, &w, src, self.w[kind * 2][0..]) |*o, *ww, s, wv| {
            o.* = if (kind == 0) c.fromBf16(s) else fp16v(s);
            ww.* = wv;
        }
        var n: [128]f64 = undefined;
        mr.gemmaNorm(&x, &w, 1e-6, false, &n);
        y.* = n;
        mr.rope(y, 64, p, 1e7, .neox_half);
        for (m, n, 0..) |*o, v, d| o.* = if (d < 32) @abs(n[d]) + @abs(n[d + 32]) else if (d < 64) @abs(n[d]) + @abs(n[d - 32]) else @abs(v);
    }

    /// The scratch outputs if each norm's sum of squares (the kernel's only sums) ran in `acc` in `order`.
    pub fn simulate(self: *Writer, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..self.shape.rows) |r| {
            for (0..nq + niq) |h| {
                const kind: usize = if (h < nq) 0 else 1;
                const i = if (h < nq) h else h - nq;
                const p: usize = @intCast(self.pos[r]);
                const src = if (kind == 0) self.q[(r * nq + i) * 128 ..][0..128] else self.iq[(r * niq + i) * 128 ..][0..128];
                var x: [128]f64 = undefined;
                var sq: [128]f64 = undefined;
                for (&x, &sq, src) |*o, *t, s| {
                    o.* = if (kind == 0) c.fromBf16(s) else fp16v(s);
                    t.* = o.* * o.*;
                }
                const ss = rows_.sumIn(acc, order, &sq);
                const inv = 1 / @sqrt(ss / 128 + 1e-6);
                var y: [128]f64 = undefined;
                for (&y, x, self.w[kind * 2][0..]) |*o, v, wv| o.* = v * inv * wv;
                mr.rope(&y, 64, p, 1e7, .neox_half);
                const at = if (kind == 0) (r * nq + i) * 128 else self.q.len + (r * niq + i) * 128;
                @memcpy(out[at..][0..128], &y);
            }
        }
    }

    pub fn reference(self: *Writer, want: []f64, mag: []f64) !void {
        var y: [128]f64 = undefined;
        var m: [128]f64 = undefined;
        for (0..self.shape.rows) |r| {
            for (0..nq) |i| {
                self.head(0, r, i, &y, &m);
                @memcpy(want[(r * nq + i) * 128 ..][0..128], &y);
                @memcpy(mag[(r * nq + i) * 128 ..][0..128], &m);
            }
            for (0..niq) |i| {
                self.head(1, r, i, &y, &m);
                @memcpy(want[self.q.len + (r * niq + i) * 128 ..][0..128], &y);
                @memcpy(mag[self.q.len + (r * niq + i) * 128 ..][0..128], &m);
            }
        }
    }

    /// The pool holds each row's K, V (exactly) and IK in its page, every other byte zero, and no status bit is set.
    pub fn check(self: *Writer, at: g.Gate) !?[]const u8 {
        _ = at;
        if (!try self.rig.intact()) return "the writer changed a buffer guard";
        const st = try self.rig.bytes(self.dev[11], 4);
        if (std.mem.readInt(u32, st[0..4], .little) != 0) return "the writer reported a missing page";
        if (self.f.kv8) return self.checkKv8();
        const pool = poolOf(false);
        const bytes = try self.rig.bytes(self.dev[10], self.pages * pool.pageBytes());
        const cells = try self.arena.allocator().alloc(u16, bytes.len / 2);
        @memcpy(std.mem.sliceAsBytes(cells), bytes);
        var touched = try self.arena.allocator().alloc(bool, cells.len);
        @memset(touched, false);
        for (0..self.shape.rows) |r| {
            const p: usize = @intCast(self.pos[r]);
            const page: usize = @intCast(self.table[(if (self.f.prompt) 0 else r * self.blocks) + p / geom.block]);
            const t = p % geom.block;
            for (0..kvh) |h| {
                const k0 = pool.kv(page, 0, 0, h, t) / 2;
                const v0 = pool.kv(page, 0, 1, h, t) / 2;
                @memset(touched[k0..][0..128], true);
                @memset(touched[v0..][0..128], true);
                if (!std.mem.eql(u16, cells[v0..][0..128], self.vv[(r * kvh + h) * 128 ..][0..128])) return "v was not stored as it came";
                if (!self.rowNear(cells[k0..][0..128], self.kk[(r * kvh + h) * 128 ..][0..128], false, 1, p)) return "k differs from the float64 norm and rotation";
            }
            const ik0 = pool.ik(page, 0, t) / 2;
            @memset(touched[ik0..][0..128], true);
            if (!self.rowNear(cells[ik0..][0..128], self.ik[r * 128 ..][0..128], true, 3, p)) return "the index key differs from the float64 norm and rotation";
        }
        for (cells, touched) |v, t| if (!t and v != 0) return "the writer wrote outside the rows' token slots";
        return null;
    }

    /// The fp8 pool: v exactly as the host rule gives it, k within e4m3's half step of the float64 norm and rope.
    fn checkKv8(self: *Writer) !?[]const u8 {
        const pool = poolOf(true);
        const bytes = try self.rig.bytes(self.dev[10], self.pages * pool.pageBytes());
        var touched = try self.arena.allocator().alloc(bool, bytes.len);
        @memset(touched, false);
        for (0..self.shape.rows) |r| {
            const p: usize = @intCast(self.pos[r]);
            const page: usize = @intCast(self.table[(if (self.f.prompt) 0 else r * self.blocks) + p / geom.block]);
            const t = p % geom.block;
            for (0..kvh) |h| {
                const k0 = pool.kv(page, 0, 0, h, t);
                const v0 = pool.kv(page, 0, 1, h, t);
                const ks0 = pool.scale(page, 0, 0, h, t);
                const vs0 = pool.scale(page, 0, 1, h, t);
                @memset(touched[k0..][0..128], true);
                @memset(touched[v0..][0..128], true);
                @memset(touched[ks0..][0..4], true);
                @memset(touched[vs0..][0..4], true);
                var v32: [128]f32 = undefined;
                for (&v32, self.vv[(r * kvh + h) * 128 ..][0..128]) |*o, s| o.* = c.fromBf16(s);
                var want: [128]u8 = undefined;
                const want_scale = mr.quantRowF32(&v32, &want);
                if (!std.mem.eql(u8, bytes[v0..][0..128], &want)) return "v was not stored as the e4m3 row of its bf16 value";
                if (std.mem.readInt(u32, bytes[vs0..][0..4], .little) != @as(u32, @bitCast(want_scale))) return "v's scale is not amax / 448";
                const ks: f32 = @bitCast(std.mem.readInt(u32, bytes[ks0..][0..4], .little));
                if (!self.keyNear(bytes[k0..][0..128], ks, self.kk[(r * kvh + h) * 128 ..][0..128], p)) return "k differs from the float64 norm and rotation beyond e4m3's half step";
            }
            const ik0 = pool.ik(page, 0, t);
            @memset(touched[ik0..][0..256], true);
            var cells: [128]u16 = undefined;
            @memcpy(std.mem.sliceAsBytes(&cells), bytes[ik0..][0..256]);
            if (!self.rowNear(&cells, self.ik[r * 128 ..][0..128], true, 3, p)) return "the index key differs from the float64 norm and rotation";
        }
        for (bytes, touched) |v, t| if (!t and v != 0) return "the writer wrote outside the rows' token slots";
        return null;
    }

    /// A stored key row (codes and scale) against the float64 norm and rotation of `src`.
    fn keyNear(self: *const Writer, codes: []const u8, scale: f32, src: []const u16, p: usize) bool {
        var x: [128]f64 = undefined;
        var w: [128]f64 = undefined;
        for (&x, &w, src, self.w[1][0..]) |*o, *ww, s, wv| {
            o.* = c.fromBf16(s);
            ww.* = wv;
        }
        var y: [128]f64 = undefined;
        mr.gemmaNorm(&x, &w, 1e-6, false, &y);
        mr.rope(&y, 64, p, 1e7, .neox_half);
        var amax: f64 = 0;
        for (y) |v| amax = @max(amax, @abs(v));
        if (!std.math.isFinite(scale) or !(@abs(@as(f64, scale) - amax / 448) <= amax / 448 * 0x1p-6 + 1e-30)) return false;
        for (codes, y) |code, want| {
            const v = mr.e4m3Value(code) * @as(f64, scale);
            // half an e4m3 step (2^-4 relative, 2^-10 scales below), the bf16 rounding and the norm's error
            if (!(@abs(v - want) <= @abs(want) * (0x1p-4 + 0x1p-7) + @as(f64, scale) * 0x1p-9 + 1e-4)) return false;
        }
        return true;
    }

    fn rowNear(self: *const Writer, got: []const u16, src: []const u16, fp: bool, wi: usize, p: usize) bool {
        var x: [128]f64 = undefined;
        var w: [128]f64 = undefined;
        for (&x, &w, src, self.w[wi][0..]) |*o, *ww, s, wv| {
            o.* = if (fp) fp16v(s) else c.fromBf16(s);
            ww.* = wv;
        }
        var y: [128]f64 = undefined;
        mr.gemmaNorm(&x, &w, 1e-6, false, &y);
        mr.rope(&y, 64, p, 1e7, .neox_half);
        for (got, y) |h, want| {
            const v: f64 = c.fromBf16(h);
            if (!(@abs(v - want) <= @abs(want) * 0x1p-7 + 1e-4)) return false;
        }
        return true;
    }
};

test "the writer's served shapes fit the launch and the router's are multiples of its chunk" {
    for (rowsSpec(cfgs[0]).served) |s| {
        _ = try geom.rowsLaunch(s.rows);
        try geom.checkRows(64, s.n * 128, s.n * 128);
    }
    for (routerSpec(.bf16).served) |s| {
        _ = try geom.routerLaunch(s.rows, s.n);
        try geom.checkRouter(s.k, s.k);
    }
    try std.testing.expectEqual(@as(?[]const u8, "tf_mm3_router_bf16"), routerSpec(.f32).replaces);
}
