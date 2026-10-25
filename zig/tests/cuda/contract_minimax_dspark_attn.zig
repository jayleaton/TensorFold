//! The dspark block attention under the contract on CUDA.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");
const common = @import("contract_minimax_dspark_common.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const mr = core.minimax_ref;
const ds = cuda.minimax_m3_dspark;
const hidden = common.hidden;
const eps32 = common.eps32;
const eps = common.eps;
const rot = common.rot;
const theta = common.theta;
const zeroDepth = common.zeroDepth;
const bf = common.bf;
const isNanBf = common.isNanBf;
const nq = common.nq;
const kvh = common.kvh;

// ---- block attention

pub const ring_rows = 4112;
pub const group = nq / kvh; // 16 q heads a kv head
pub const base_pos = 100;

pub fn keyChain(s: c.Shape) usize {
    return s.k + 16 + 32 + 32;
}

pub const AttnCfg = struct { name: []const u8, windowed: bool };

pub const attn_cfgs = [_]AttnCfg{
    .{ .name = "cuda minimax_m3_dspark tf_mm3_ds_attn, 32 q heads over 2 kv heads, whole context", .windowed = false },
    .{ .name = "cuda minimax_m3_dspark tf_mm3_ds_attn, 32 q heads over 2 kv heads, 300-key window", .windowed = true },
};

pub fn attnSpec(f: AttnCfg) c.Spec {
    return .{
        .name = f.name,
        // the windowed rows see different keys, so gate 4's identical-key construction needs the whole-context form
        .accum = if (f.windowed) .none else .f32,
        .depth = keyChain,
        .ops = 1,
        .out = .bf16,
        // rows, the 32 query heads, and the context keys (edges at the 128-key chunk)
        .tiles = .{ .{}, .{}, .{ .size = 128, .min = 0 } },
        .served = &.{ .{ .rows = 7, .n = nq, .k = 4096 }, .{ .rows = 7, .n = nq, .k = 129 }, .{ .rows = 16, .n = nq, .k = 1000 }, .{ .rows = 1, .n = nq, .k = 0 }, .{ .rows = 7, .n = nq, .k = 4111 } },
        .sums = .{ .why = attn.why },
    };
}

pub const Attn = struct {
    rig: rig_.Rig,
    f: AttnCfg,
    k: *const ds.Dspark,
    arena: std.heap.ArenaAllocator,
    r: ?attn.Ref = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    q: []u16 = &.{},
    ring_k: []u16 = &.{},
    ring_v: []u16 = &.{},
    bk: []u16 = &.{},
    bv: []u16 = &.{},
    p: usize = 0,
    lo: usize = 0,
    window: usize = 0,
    dev: [6]u64 = @splat(0),

    pub fn deinit(self: *Attn) void {
        if (self.r) |*r| r.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Attn) g.Kernel {
        return .{ .spec = attnSpec(self.f), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Attn);

    fn lowest(self: *const Attn, row: usize) usize {
        const w0 = if (self.p + row + 1 > self.window) self.p + row + 1 - self.window else 0;
        return @max(self.lo, w0);
    }

    fn kvAt(self: *Attn, which: usize, kp: usize, grp: usize) []u16 {
        const arr = if (which == 0) self.ring_k else self.ring_v;
        return arr[((kp % ring_rows) * kvh + grp) * 128 ..][0..128];
    }

    pub fn prepare(self: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != nq or shape.rows == 0 or shape.rows > 16 or shape.k > ring_rows) return false;
        if (self.f.windowed and fill == .stress) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const n = shape.rows;
        if (self.f.windowed) {
            self.p = shape.k;
            self.lo = 0;
            self.window = 300;
        } else {
            self.p = shape.k + base_pos;
            self.lo = base_pos;
            self.window = 1 << 20;
        }
        const first = @max(self.lo, if (self.p + 1 > self.window) self.p + 1 - self.window else 0);
        const nan = c.toBf16(std.math.nan(f32));
        self.ring_k = try a.alloc(u16, ring_rows * kvh * 128);
        self.ring_v = try a.alloc(u16, ring_rows * kvh * 128);
        @memset(self.ring_k, nan);
        @memset(self.ring_v, nan);
        for (first..self.p) |kp| for (0..kvh) |gq| {
            for (self.kvAt(0, kp, gq)) |*e| e.* = attn.normal(rng);
            for (self.kvAt(1, kp, gq)) |*e| e.* = attn.normal(rng);
        };
        self.q = try a.alloc(u16, n * nq * 128);
        self.bk = try a.alloc(u16, n * kvh * 128);
        self.bv = try a.alloc(u16, n * kvh * 128);
        for (self.q) |*e| e.* = attn.normal(rng);
        for (self.bk) |*e| e.* = attn.normal(rng);
        for (self.bv) |*e| e.* = attn.normal(rng);
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(self.rig.gpu.gpa, 128, ring_rows + 16);
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(first),
            .stress => try self.stress(rng, first),
        }
        const bytes = [_][]const u8{ std.mem.sliceAsBytes(self.q), std.mem.sliceAsBytes(self.ring_k), std.mem.sliceAsBytes(self.ring_v), std.mem.sliceAsBytes(self.bk), std.mem.sliceAsBytes(self.bv) };
        for (bytes, self.dev[0..5]) |b, *p| p.* = try self.rig.input(b);
        self.dev[5] = try self.rig.output(n * nq * 128 * 2);
        return true;
    }

    /// NaN query; +inf and -inf values; a NaN key.
    fn plant(self: *Attn, first: usize) void {
        const n = self.shape.rows;
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        self.q[3] = nan;
        if (self.p > first) self.kvAt(1, self.p - 1, 0)[7] = inf;
        self.bv[(n - 1) * kvh * 128 + 128 + 9] = c.toBf16(-std.math.inf(f32));
        if (n > 3) self.bk[0] = nan;
    }

    /// Gate 4: every key alike (p = 1), values mirrored over the keys all rows see, which are the same for every row.
    fn stress(self: *Attn, rng: std.Random, first: usize) !void {
        const al = self.arena.allocator();
        const nctx = self.p - first;
        const nk = nctx + self.shape.rows;
        const key0 = try al.alloc(u16, kvh * 128);
        for (key0) |*e| e.* = attn.normal(rng);
        for (0..kvh) |gq| {
            for (first..self.p) |kp| @memcpy(self.kvAt(0, kp, gq), key0[gq * 128 ..][0..128]);
            for (0..self.shape.rows) |r| @memcpy(self.bk[(r * kvh + gq) * 128 ..][0..128], key0[gq * 128 ..][0..128]);
        }
        const col = try al.alloc(f32, nk);
        for (0..kvh) |gq| for (0..128) |e| {
            small(rng, col);
            for (col, 0..) |y, i| {
                const cell = if (i < nctx) &self.kvAt(1, first + i, gq)[e] else &self.bv[((i - nctx) * kvh + gq) * 128 + e];
                cell.* = c.toBf16(y);
            }
        };
    }

    pub fn run(self: *Attn) !void {
        const d = self.dev;
        const a: ds.Attn = .{ .nq = nq, .kvh = kvh, .ring_rows = ring_rows, .window = @intCast(@min(self.window, 1 << 30)), .scale = @floatCast(self.scale()) };
        try self.k.blockAttn(self.rig.s, a, d[0], d[1], d[2], d[3], d[4], d[5], @intCast(self.shape.rows), @intCast(self.p), @intCast(self.lo));
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Attn) usize {
        return self.shape.rows * nq * 128;
    }

    fn scale(_: *const Attn) f64 {
        return std.math.pow(f64, 128, -0.5);
    }

    /// Output head i (row i / 32, head i % 32): its row's visible context keys, then every block key.
    pub fn load(self: *Attn, i: usize, hd: *attn.Head) !void {
        const row = i / nq;
        const head = i % nq;
        const gq = head / group;
        for (hd.q, 0..) |*x, e| x.* = c.fromBf16(self.q[i * 128 + e]);
        var n: usize = 0;
        for (self.lowest(row)..self.p) |kp| {
            for (0..128) |e| {
                hd.k[n * 128 + e] = c.fromBf16(self.kvAt(0, kp, gq)[e]);
                hd.v[n * 128 + e] = c.fromBf16(self.kvAt(1, kp, gq)[e]);
            }
            n += 1;
        }
        for (0..self.shape.rows) |j| {
            for (0..128) |e| {
                hd.k[n * 128 + e] = c.fromBf16(self.bk[(j * kvh + gq) * 128 + e]);
                hd.v[n * 128 + e] = c.fromBf16(self.bv[(j * kvh + gq) * 128 + e]);
            }
            n += 1;
        }
        hd.n = n;
    }

    pub fn check(self: *Attn, gate: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.count() / 128, self.scale(), try self.rig.outputBytes(), gate);
    }

    pub fn reference(self: *Attn, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.count() / 128, self.scale(), want, mag);
    }

    pub fn simulate(self: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.count() / 128, self.scale(), acc, order, out);
    }
};

/// Gate 4's values: odd integers, every other one times 4, the second half cancelling the first; sums stay under 2^24.
pub fn small(rng: std.Random, out: []f32) void {
    const h = out.len / 2;
    for (0..h) |i| {
        const v: f32 = @floatFromInt((1 + 2 * rng.uintLessThan(u32, 128)) * @as(u32, if (i % 2 == 1) 4 else 1));
        out[i] = v;
        out[h + i] = (if (i < 8) @as(f32, 1) else 0) - v;
    }
    if (out.len % 2 == 1) out[out.len - 1] = 0;
}
