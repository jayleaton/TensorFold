//! The dspark block-heads and K/V ring writers under the contract on CUDA.
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

// ---- per-head norm + RoPE: a draft block and the K/V ring

pub fn headDepth(_: c.Shape) usize {
    return 4 + 5 + 3;
}

pub fn headMag(n: *const [128]f64, v: f64, d: usize) f64 {
    return if (d < 32) @abs(n[d]) + @abs(n[d + 32]) else if (d < 64) @abs(n[d]) + @abs(n[d - 32]) else @abs(v);
}

/// One head's float64 norm and rotation; kind 0 normed with the q weights, 1 with the k weights.
pub fn headRef(x16: []const u16, w16: []const u16, gemma: bool, pos: usize, y: *[128]f64, m: *[128]f64) void {
    var x: [128]f64 = undefined;
    var w: [128]f64 = undefined;
    for (&x, &w, x16, w16) |*o, *ww, s, wv| {
        o.* = c.fromBf16(s);
        ww.* = c.fromBf16(wv);
    }
    var n: [128]f64 = undefined;
    mr.gemmaNorm(&x, &w, eps, gemma, &n);
    y.* = n;
    mr.rope(y, rot, pos, theta, .neox_half);
    for (m, n, 0..) |*o, v, d| o.* = headMag(&n, v, d);
}

pub fn blockSpec(gemma_on: bool) c.Spec {
    return .{
        .name = if (gemma_on) "cuda minimax_m3_dspark tf_mm3_ds_block_heads (1 + w), 32 q + 2 kv heads" else "cuda minimax_m3_dspark tf_mm3_ds_block_heads (w), 32 q + 2 kv heads",
        .accum = .f32,
        .depth = headDepth,
        // div, sqrt, add, div, two products, the rotation's three, the weight add
        .ops = 14,
        .out = .bf16,
        // rows, q heads, and the first position (the rotation angle grows with it)
        .tiles = .{ .{}, .{}, .{ .size = 128, .min = 0 } },
        .served = &.{ .{ .rows = 7, .n = nq, .k = 3000 }, .{ .rows = 1, .n = nq, .k = 0 }, .{ .rows = 16, .n = nq, .k = 129 }, .{ .rows = 7, .n = nq, .k = 1_000_003 } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "a norm's sum of squares is not a dot the fp16 and bf16 accumulators can be made to miss" },
    };
}

pub const Block = struct {
    rig: rig_.Rig,
    gemma: bool,
    k: *const ds.Dspark,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    q: []u16 = &.{},
    kk: []u16 = &.{},
    vv: []u16 = &.{},
    w: [2][128]u16 = undefined,
    dev: [8]u64 = @splat(0),

    pub fn deinit(self: *Block) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Block) g.Kernel {
        return .{ .spec = blockSpec(self.gemma), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Block);

    pub fn prepare(self: *Block, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != nq or shape.rows == 0 or shape.rows > 16 or shape.k + shape.rows > 1_100_000) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const n = shape.rows;
        inline for (.{ .{ "q", n * nq * 128 }, .{ "kk", n * kvh * 128 }, .{ "vv", n * kvh * 128 } }) |p| {
            @field(self, p[0]) = try a.alloc(u16, p[1]);
            for (@field(self, p[0])) |*e| e.* = c.toBf16(if (fill == .stress) @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -3, 3))) else (rng.float(f32) - 0.5) * 8);
        }
        for (&self.w) |*row| {
            for (row) |*e| e.* = bf(rng, 0.25);
        }
        if (fill == .nonfinite) {
            self.q[5] = c.toBf16(std.math.nan(f32));
            if (n > 1) self.q[(1 * nq + 3) * 128 + 70] = c.toBf16(std.math.inf(f32));
            if (n > 2) self.kk[(2 * kvh + 1) * 128 + 9] = c.toBf16(-std.math.inf(f32));
            if (n > 3) self.vv[3 * kvh * 128 + 4] = c.toBf16(std.math.nan(f32));
        }
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.kk));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.vv));
        self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(self.w[0][0..]));
        self.dev[4] = try self.rig.input(std.mem.sliceAsBytes(self.w[1][0..]));
        self.dev[5] = try self.rig.output(self.q.len * 2);
        self.dev[6] = try self.rig.output(self.kk.len * 2);
        self.dev[7] = try self.rig.output(self.vv.len * 2);
        return true;
    }

    fn heads(self: *const Block) ds.Heads {
        return .{ .nq = nq, .kvh = kvh, .rot = rot, .theta = theta, .eps = eps32, .gemma = self.gemma };
    }

    pub fn run(self: *Block) !void {
        const d = self.dev;
        try self.k.blockHeads(self.rig.s, self.heads(), d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], @intCast(self.shape.rows), @intCast(self.shape.k));
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Block) usize {
        return self.q.len + 2 * self.kk.len;
    }

    pub fn reference(self: *Block, want: []f64, mag: []f64) !void {
        var y: [128]f64 = undefined;
        var m: [128]f64 = undefined;
        const nrow = self.shape.rows;
        for (0..nrow) |r| {
            const pos = self.shape.k + r;
            for (0..nq) |h| {
                headRef(self.q[(r * nq + h) * 128 ..][0..128], &self.w[0], self.gemma, pos, &y, &m);
                @memcpy(want[(r * nq + h) * 128 ..][0..128], &y);
                @memcpy(mag[(r * nq + h) * 128 ..][0..128], &m);
            }
            for (0..kvh) |h| {
                headRef(self.kk[(r * kvh + h) * 128 ..][0..128], &self.w[1], self.gemma, pos, &y, &m);
                @memcpy(want[self.q.len + (r * kvh + h) * 128 ..][0..128], &y);
                @memcpy(mag[self.q.len + (r * kvh + h) * 128 ..][0..128], &m);
                for (0..128) |d| {
                    want[self.q.len + self.kk.len + (r * kvh + h) * 128 + d] = c.fromBf16(self.vv[(r * kvh + h) * 128 + d]);
                    mag[self.q.len + self.kk.len + (r * kvh + h) * 128 + d] = 0;
                }
            }
        }
    }

    pub fn simulate(self: *Block, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..self.shape.rows) |r| {
            const pos = self.shape.k + r;
            for (0..nq + kvh) |s| {
                const is_q = s < nq;
                const h = if (is_q) s else s - nq;
                const src = if (is_q) self.q[(r * nq + h) * 128 ..][0..128] else self.kk[(r * kvh + h) * 128 ..][0..128];
                const wv = if (is_q) &self.w[0] else &self.w[1];
                var x: [128]f64 = undefined;
                var sq: [128]f64 = undefined;
                for (&x, &sq, src) |*o, *t, v| {
                    o.* = c.fromBf16(v);
                    t.* = o.* * o.*;
                }
                const ss = rows_.sumIn(acc, order, &sq);
                const inv = 1 / @sqrt(ss / 128 + eps);
                var y: [128]f64 = undefined;
                for (&y, x, wv) |*o, v, w16| o.* = v * inv * (c.fromBf16(w16) + @as(f64, if (self.gemma) 1 else 0));
                mr.rope(&y, rot, pos, theta, .neox_half);
                const at = if (is_q) (r * nq + h) * 128 else self.q.len + (r * kvh + h) * 128;
                @memcpy(out[at..][0..128], &y);
            }
            for (0..kvh) |h| for (0..128) |d| {
                out[self.q.len + self.kk.len + (r * kvh + h) * 128 + d] = c.fromBf16(self.vv[(r * kvh + h) * 128 + d]);
            };
        }
    }
};

pub const ring_rows_test = 40;

pub const ring_spec: c.Spec = .{
    .name = "cuda minimax_m3_dspark tf_mm3_ds_ring_put, 2 kv heads, 40-slot ring",
    .accum = .none,
    .depth = zeroDepth,
    .out = .raw,
    .nan = .{ .stated = "a NaN or infinity in k keeps its class through the norm and the rotation; v is stored as it came" },
    // rows, (unused), first position
    .served = &.{ .{ .rows = 7, .n = kvh, .k = 3000 }, .{ .rows = 16, .n = kvh, .k = 5 }, .{ .rows = 1, .n = kvh, .k = 39 }, .{ .rows = 8, .n = kvh, .k = 1_000_003 } },
    .sums = .{ .f16 = 0, .bf16 = 0, .why = "the norm's sum of squares is checked against float64 by the K check; this kernel declares no sums" },
};

pub const Ring = struct {
    rig: rig_.Rig,
    k: *const ds.Dspark,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    kk: []u16 = &.{},
    vv: []u16 = &.{},
    w: [128]u16 = undefined,
    dev: [5]u64 = @splat(0),
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Ring) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Ring) g.Kernel {
        return .{ .spec = ring_spec, .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Ring);

    pub fn prepare(self: *Ring, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != kvh or shape.rows == 0 or shape.rows > 16) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const n = shape.rows;
        self.kk = try a.alloc(u16, n * kvh * 128);
        self.vv = try a.alloc(u16, n * kvh * 128);
        for (self.kk) |*e| e.* = c.toBf16((rng.float(f32) - 0.5) * 8);
        for (self.vv) |*e| e.* = c.toBf16((rng.float(f32) - 0.5) * 8);
        for (&self.w) |*e| e.* = bf(rng, 0.25);
        if (fill == .nonfinite) {
            self.kk[7] = c.toBf16(std.math.nan(f32));
            if (n > 1) self.kk[(1 * kvh + 1) * 128 + 70] = c.toBf16(std.math.inf(f32));
            if (n > 2) self.vv[2 * kvh * 128 + 9] = c.toBf16(-std.math.inf(f32));
        }
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.kk));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.vv));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.w[0..]));
        self.dev[3] = try self.rig.output(ring_rows_test * kvh * 128 * 2);
        self.dev[4] = try self.rig.output(ring_rows_test * kvh * 128 * 2);
        return true;
    }

    pub fn run(self: *Ring) !void {
        const d = self.dev;
        const h: ds.Heads = .{ .nq = 0, .kvh = kvh, .rot = rot, .theta = theta, .eps = eps32, .gemma = true };
        try self.k.ringPut(self.rig.s, h, d[0], d[1], d[2], d[3], d[4], ring_rows_test, @intCast(self.shape.rows), @intCast(self.shape.k));
        try self.rig.s.synchronize();
    }

    pub fn count(_: *Ring) usize {
        return ring_rows_test * kvh * 128 * 2;
    }

    pub fn reference(self: *Ring, want: []f64, mag: []f64) !void {
        _ = self;
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn near(got: u16, want: f64) bool {
        const v: f64 = c.fromBf16(got);
        if (std.math.isNan(want)) return std.math.isNan(v);
        if (std.math.isInf(want)) return v == want;
        return @abs(v - want) <= @abs(want) * 0x1p-7 + 1e-4;
    }

    pub fn check(self: *Ring, at: g.Gate) !?[]const u8 {
        _ = at;
        const out = try self.rig.outputBytes();
        const half = ring_rows_test * kvh * 128 * 2;
        var written: [ring_rows_test]bool = @splat(false);
        for (0..self.shape.rows) |r| {
            const slot = (self.shape.k + r) % ring_rows_test;
            written[slot] = true;
            for (0..kvh) |h| {
                var y: [128]f64 = undefined;
                var m: [128]f64 = undefined;
                headRef(self.kk[(r * kvh + h) * 128 ..][0..128], &self.w, true, self.shape.k + r, &y, &m);
                for (0..128) |d| {
                    const at_e = (slot * kvh + h) * 128 + d;
                    const gk = std.mem.readInt(u16, out[2 * at_e ..][0..2], .little);
                    const gv = std.mem.readInt(u16, out[half + 2 * at_e ..][0..2], .little);
                    if (!near(gk, y[d])) return std.fmt.bufPrint(&self.msg, "row {d} head {d} dim {d}: k {e} where float64 gives {e}", .{ r, h, d, c.fromBf16(gk), y[d] }) catch "ring k differs";
                    if (gv != self.vv[(r * kvh + h) * 128 + d] and !isNanBf(gv)) return "ring v was not stored as it came";
                    if (isNanBf(self.vv[(r * kvh + h) * 128 + d]) != isNanBf(gv)) return "ring v NaN class differs";
                }
            }
        }
        for (0..ring_rows_test) |slot| if (!written[slot]) {
            for (0..kvh * 128) |e| {
                const i = slot * kvh * 128 + e;
                if (std.mem.readInt(u16, out[2 * i ..][0..2], .little) != 0xffff or std.mem.readInt(u16, out[half + 2 * i ..][0..2], .little) != 0xffff) return "the ring write touched a slot of another position";
            }
        };
        if (!try self.rig.intact()) return "the ring write changed a buffer guard";
        return null;
    }
};
