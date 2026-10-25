//! The dspark tap, add and norm kernels under the contract on CUDA.
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

// ---- tap

pub const tap_spec: c.Spec = .{
    .name = "cuda minimax_m3_dspark tf_mm3_ds_tap",
    .accum = .none,
    .depth = zeroDepth,
    .out = .raw,
    .nan = .{ .stated = "a non-finite fp32 value keeps its class in bf16; the destination's other columns are untouched" },
    // rows, hidden, tap slots; the destination row holds all slots
    .served = &.{ .{ .rows = 7, .n = 6144, .k = 5 }, .{ .rows = 1, .n = 6144, .k = 5 }, .{ .rows = 64, .n = 6144, .k = 5 }, .{ .rows = 5, .n = 1000, .k = 2 } },
    .sums = .{ .f16 = 0, .bf16 = 0, .why = "a tap rounds and copies; it sums nothing" },
};

pub const Tap = struct {
    rig: rig_.Rig,
    k: *const ds.Dspark,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    src: []f32 = &.{},
    dev: [2]u64 = .{ 0, 0 },
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Tap) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Tap) g.Kernel {
        return .{ .spec = tap_spec, .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Tap);

    fn slot(self: *const Tap) u32 {
        return @intCast(self.shape.rows % self.shape.k);
    }

    pub fn prepare(self: *Tap, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        self.src = try a.alloc(f32, shape.rows * shape.n);
        for (self.src) |*v| v.* = rng.floatNorm(f32) * 100;
        if (fill == .nonfinite) {
            self.src[0] = std.math.nan(f32);
            if (shape.rows > 1) self.src[shape.n + 1] = std.math.inf(f32);
            if (shape.rows > 2) self.src[2 * shape.n + shape.n - 1] = -std.math.inf(f32);
        }
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.src));
        self.dev[1] = try self.rig.output(shape.rows * shape.k * shape.n * 2);
        return true;
    }

    pub fn run(self: *Tap) !void {
        const s = self.shape;
        try self.k.tap(self.rig.s, self.dev[0], self.dev[1], s.k * s.n, self.slot(), @intCast(s.n), @intCast(s.rows));
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Tap) usize {
        return self.shape.rows * self.shape.k * self.shape.n;
    }

    pub fn reference(self: *Tap, want: []f64, mag: []f64) !void {
        _ = self;
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Bits, not tolerances: the slot's columns are the fp32 values rounded to bf16, every other column is still the
    pub fn check(self: *Tap, at: g.Gate) !?[]const u8 {
        _ = at;
        const s = self.shape;
        const out = try self.rig.outputBytes();
        const ld = s.k * s.n;
        const lo = self.slot() * s.n;
        for (0..s.rows) |r| for (0..ld) |col| {
            const got = std.mem.readInt(u16, out[2 * (r * ld + col) ..][0..2], .little);
            if (col >= lo and col < lo + s.n) {
                const v = self.src[r * s.n + col - lo];
                if (std.math.isNan(v)) {
                    if (!isNanBf(got)) return "a NaN tap did not stay NaN";
                } else if (got != c.toBf16(v)) return std.fmt.bufPrint(&self.msg, "row {d} column {d}: {x} where bf16 of the value is {x}", .{ r, col - lo, got, c.toBf16(v) }) catch "tap bits differ";
            } else if (got != 0xffff) return "a tap wrote outside its slot's columns";
        };
        if (!try self.rig.intact()) return "the tap changed a buffer guard";
        return null;
    }
};

// ---- add

pub const AddForm = enum { bf16, f32, bf16_zero };

pub fn addSpec(f: AddForm) c.Spec {
    return .{
        .name = switch (f) {
            .bf16 => "cuda minimax_m3_dspark tf_mm3_ds_add_bf16",
            .f32 => "cuda minimax_m3_dspark tf_mm3_ds_add_f32",
            .bf16_zero => "cuda minimax_m3_dspark tf_mm3_ds_add_bf16 (no first operand)",
        },
        .accum = .none,
        .depth = zeroDepth,
        .out = .f32,
        .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
        .served = &.{ .{ .rows = 1, .n = 7 * hidden, .k = 1 }, .{ .rows = 1, .n = 1 << 20, .k = 1 }, .{ .rows = 1, .n = 1, .k = 1 }, .{ .rows = 1, .n = 1_000_003, .k = 1 } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "one fp32 add of two values, no sum of many" },
    };
}

pub const Add = struct {
    rig: rig_.Rig,
    form: AddForm,
    k: *const ds.Dspark,
    arena: std.heap.ArenaAllocator,
    n: usize = 0,
    a: []f32 = &.{},
    b: []f32 = &.{},
    dev: [3]u64 = @splat(0),

    pub fn deinit(self: *Add) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Add) g.Kernel {
        return .{ .spec = addSpec(self.form), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Add);

    pub fn prepare(self: *Add, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const al = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.n = shape.n;
        self.a = try al.alloc(f32, shape.n);
        self.b = try al.alloc(f32, shape.n);
        const bits = try al.alloc(u16, shape.n);
        for (self.a, self.b, bits) |*x, *y, *h| {
            x.* = if (self.form == .bf16_zero) 0 else rng.floatNorm(f32) * 50;
            if (self.form == .f32) {
                y.* = rng.floatNorm(f32) * 3;
            } else {
                h.* = bf(rng, 3);
                y.* = c.fromBf16(h.*);
            }
        }
        if (fill == .nonfinite) {
            if (self.form == .bf16_zero) self.b[0] = std.math.nan(f32) else self.a[0] = std.math.nan(f32);
            self.b[1 % shape.n] = std.math.inf(f32);
            if (shape.n > 3 and self.form != .bf16_zero) {
                self.a[2] = -std.math.inf(f32);
                self.b[2] = std.math.inf(f32);
            }
            if (self.form != .f32) for (self.b, bits) |y, *h| {
                h.* = c.toBf16(y);
            };
        }
        self.dev[0] = if (self.form == .bf16_zero) 0 else try self.rig.input(std.mem.sliceAsBytes(self.a));
        self.dev[1] = if (self.form == .f32) try self.rig.input(std.mem.sliceAsBytes(self.b)) else try self.rig.input(std.mem.sliceAsBytes(bits));
        self.dev[2] = try self.rig.output(shape.n * 4);
        return true;
    }

    pub fn run(self: *Add) !void {
        try self.k.add(self.rig.s, self.dev[0], self.dev[1], self.form == .f32, self.dev[2], self.n);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Add) usize {
        return self.n;
    }

    pub fn reference(self: *Add, want: []f64, mag: []f64) !void {
        for (want, mag, self.a, self.b) |*w, *m, x, y| {
            w.* = @as(f64, x) + y;
            m.* = 0;
        }
    }
};

// ---- norm

pub const NormForm = enum { bf16_gemma, f32_gemma, f32_plain };

pub fn normDepth(s: c.Shape) usize {
    return s.k / 256 + 1 + 5 + 7;
}

pub fn normSpec(f: NormForm) c.Spec {
    return .{
        .name = switch (f) {
            .bf16_gemma => "cuda minimax_m3_dspark tf_mm3_ds_norm_bf16 (1 + w)",
            .f32_gemma => "cuda minimax_m3_dspark tf_mm3_ds_norm_f32 (1 + w)",
            .f32_plain => "cuda minimax_m3_dspark tf_mm3_ds_norm_f32 (w)",
        },
        .accum = .f32,
        .depth = normDepth,
        // div, sqrt, add, div, two products, the weight add
        .ops = 7,
        .out = .bf16,
        .replaces = if (f == .f32_gemma) "tf_mm3_ds_norm_bf16" else null,
        // rows are CTAs; the row is strided over 256 threads
        .tiles = .{ .{}, .{}, .{ .size = 256, .step = 1, .min = 1 } },
        .served = &.{ .{ .rows = 7, .n = hidden, .k = hidden }, .{ .rows = 1, .n = hidden, .k = hidden }, .{ .rows = 17, .n = hidden, .k = hidden }, .{ .rows = 2048, .n = 1024, .k = 1024 } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "a norm's sum of squares is not a dot the fp16 and bf16 accumulators can be made to miss" },
    };
}

pub const Norm = struct {
    rig: rig_.Rig,
    form: NormForm,
    k: *const ds.Dspark,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    x: []u16 = &.{},
    w: []u16 = &.{},
    dev: [3]u64 = @splat(0),
    base: u64 = 0,

    pub fn deinit(self: *Norm) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Norm) g.Kernel {
        return .{ .spec = normSpec(self.form), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Norm);

    fn gemma(self: *const Norm) bool {
        return self.form != .f32_plain;
    }

    pub fn prepare(self: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const k = shape.k;
        self.x = try a.alloc(u16, shape.rows * k);
        self.w = try a.alloc(u16, k);
        for (self.w) |*v| v.* = bf(rng, 0.25);
        for (self.x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 64 == 7) @as(f32, 20) else 1));
        if (fill == .stress) {
            // gate 4: integer terms, squares exact in fp32 whatever the order
            for (0..shape.rows) |r| for (0..k) |i| {
                self.x[r * k + i] = c.toBf16(@floatFromInt(rng.intRangeAtMost(i32, -32, 32)));
            };
        } else if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                self.x[p[0] * k + p[1]] = c.toBf16(v);
            };
        }
        if (self.form == .bf16_gemma) {
            self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.x));
        } else {
            const xf = try a.alloc(f32, self.x.len);
            for (xf, self.x) |*o, h| o.* = c.fromBf16(h);
            self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(xf));
        }
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.w));
        self.dev[2] = try self.rig.output(shape.rows * k * 2);
        if (self.form == .f32_gemma) self.base = try self.rig.scratch(shape.rows * k * 2);
        return true;
    }

    pub fn run(self: *Norm) !void {
        const s = self.shape;
        try self.k.norm(self.rig.s, self.dev[0], self.form != .bf16_gemma, self.dev[1], self.gemma(), self.dev[2], @intCast(s.rows), @intCast(s.k), eps32);
        try self.rig.s.synchronize();
    }

    /// The bf16-input form on the same values.
    pub fn replaced(self: *Norm) ![]const u8 {
        const s = self.shape;
        const xb = std.mem.sliceAsBytes(self.x);
        const dev = try self.rig.scratch(xb.len);
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = dev, .len = xb.len };
        try b.upload(0, xb);
        try self.k.norm(self.rig.s, dev, false, self.dev[1], true, self.base, @intCast(s.rows), @intCast(s.k), eps32);
        return self.rig.bytes(self.base, s.rows * s.k * 2);
    }

    pub fn count(self: *Norm) usize {
        return self.shape.rows * self.shape.k;
    }

    fn row(self: *const Norm, r: usize, x: []f64, w: []f64) void {
        const k = self.shape.k;
        for (x, w, 0..) |*xv, *wv, i| {
            xv.* = c.fromBf16(self.x[r * k + i]);
            wv.* = c.fromBf16(self.w[i]);
        }
    }

    pub fn reference(self: *Norm, want: []f64, mag: []f64) !void {
        const k = self.shape.k;
        const a = self.rig.gpu.gpa;
        const buf = try a.alloc(f64, 2 * k);
        defer a.free(buf);
        for (0..self.shape.rows) |r| {
            self.row(r, buf[0..k], buf[k..]);
            mr.gemmaNorm(buf[0..k], buf[k..], eps, self.gemma(), want[r * k ..][0..k]);
            for (mag[r * k ..][0..k], want[r * k ..][0..k]) |*m, y| m.* = @abs(y) / 2;
        }
    }

    pub fn simulate(self: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const k = self.shape.k;
        const a = self.rig.gpu.gpa;
        const buf = try a.alloc(f64, 3 * k);
        defer a.free(buf);
        for (0..self.shape.rows) |r| {
            self.row(r, buf[0..k], buf[k .. 2 * k]);
            for (buf[2 * k ..][0..k], buf[0..k]) |*t, v| t.* = v * v;
            const ss = rows_.sumIn(acc, order, buf[2 * k ..][0..k]);
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(k)) + eps);
            const add: f64 = if (self.gemma()) 1 else 0;
            for (0..k) |i| out[r * k + i] = buf[i] * inv * (buf[k + i] + add);
        }
    }
};
