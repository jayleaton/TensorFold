//! minimax_m3_act.cu's swigluoai under the contract on CUDA: float64 reference core/minimax_ref.zig `swigluoai`, NaN
//! propagating as float64 does, and two planted controls (a clamp by fminf, which drops a NaN, and a lost tail value).
//! Written for the GB10 run; nothing here has run on a GPU yet.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const mr = core.minimax_ref;
const act = cuda.minimax_m3_act;

pub const Which = enum { bf16, bf16_symmetric_no_bias, f32, ctl_fmin, ctl_tail };

fn noSums(_: c.Shape) usize {
    return 0;
}

/// expf (2 ulp), the add, the division and the products.
const act_ops = 8;

// Layers 0-2's dense MLP (12288) and the shared expert (3072), whole and cut in two for the two-rank split.
const served = &[_]c.Shape{ .{ .rows = 1, .n = 3072, .k = 1 }, .{ .rows = 1, .n = 12288, .k = 1 }, .{ .rows = 17, .n = 1536, .k = 1 }, .{ .rows = 64, .n = 6144, .k = 1 } };
const tile_n: c.Tile = .{ .size = 256, .step = 1, .min = 1 }; // one thread a value, 256 a block; the guard is the edge
const tile_rows: c.Tile = .{ .size = 16 };

pub fn spec(which: Which) c.Spec {
    const name = switch (which) {
        .bf16 => "cuda minimax_m3_act tf_mm3_swigluoai_bf16",
        .bf16_symmetric_no_bias => "cuda minimax_m3_act tf_mm3_swigluoai_bf16 (symmetric gate clamp, no +1)",
        .f32 => "cuda minimax_m3_act tf_mm3_swigluoai_f32",
        .ctl_fmin => "control: swigluoai clamped by fminf",
        .ctl_tail => "control: swigluoai losing a tail value",
    };
    return .{ .name = name, .accum = .none, .depth = noSums, .ops = act_ops, .out = .bf16, .tiles = .{ tile_rows, tile_n, .{} }, .served = served };
}

pub fn control(which: Which) ?g.Gate {
    return switch (which) {
        .ctl_fmin => .nonfinite,
        .ctl_tail => .shapes,
        else => null,
    };
}

fn params(which: Which) act.Params {
    return if (which == .bf16_symmetric_no_bias) .{ .symmetric_gate_clamp = true, .up_bias = 0 } else .{};
}

pub const Mm3Act = struct {
    rig: rig_.Rig,
    which: Which,
    k: *const act.MiniMaxAct,
    ctl: ?cuda.Function = null,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    gate: []f32 = &.{},
    up: []f32 = &.{},
    gate_ptr: u64 = 0,
    up_ptr: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Mm3Act) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Mm3Act);

    pub fn kernel(self: *Mm3Act) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = control(self.which) };
    }

    fn isF32(self: *const Mm3Act) bool {
        return self.which == .f32;
    }

    pub fn prepare(self: *Mm3Act, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        const n = shape.rows * shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.gate = try a.alloc(f32, n);
        self.up = try a.alloc(f32, n);
        // gates past +-7 (the clamp), a few huge ones; ups past +-7 on both sides
        for (self.gate, self.up, 0..) |*gv, *uv, i| {
            gv.* = (rng.float(f32) * 2 - 1) * (if (i % 17 == 3) @as(f32, 30) else 9);
            uv.* = (rng.float(f32) * 2 - 1) * (if (i % 13 == 5) @as(f32, 20) else 8);
        }
        if (fill == .nonfinite) {
            const w = shape.n;
            const at = [_][2]usize{ .{ 0, w / 3 }, .{ 1, w / 2 }, .{ 2, w - 1 }, .{ 3, 0 }, .{ 3, w - 1 } };
            const nan = std.math.nan(f32);
            const inf = std.math.inf(f32);
            // gate: NaN, +inf, -inf; up: row 3 holds +inf and -inf; one more NaN in an up
            const put_g = [_]f32{ nan, inf, -inf, 1, 1 };
            const put_u = [_]f32{ 1, 1, 1, inf, -inf };
            for (at, put_g, put_u, 0..) |p, pg, pu, i| if (p[0] < shape.rows) {
                self.gate[p[0] * w + p[1]] = pg;
                self.up[p[0] * w + p[1]] = pu;
                if (i == 0 and shape.rows > 4) self.up[4 * w + w / 4] = nan;
            };
        }
        // the inputs are bf16 values (or fp32 in the f32 form), so the reference reads exactly what the kernel reads
        if (!self.isF32()) {
            for (self.gate) |*v| v.* = c.fromBf16(c.toBf16(v.*));
            for (self.up) |*v| v.* = c.fromBf16(c.toBf16(v.*));
            const gb = try a.alloc(u16, n);
            const ub = try a.alloc(u16, n);
            for (gb, self.gate) |*o, v| o.* = c.toBf16(v);
            for (ub, self.up) |*o, v| o.* = c.toBf16(v);
            self.gate_ptr = try self.rig.input(std.mem.sliceAsBytes(gb));
            self.up_ptr = try self.rig.input(std.mem.sliceAsBytes(ub));
        } else {
            self.gate_ptr = try self.rig.input(std.mem.sliceAsBytes(self.gate));
            self.up_ptr = try self.rig.input(std.mem.sliceAsBytes(self.up));
        }
        self.out = try self.rig.output(n * 2);
        return true;
    }

    pub fn run(self: *Mm3Act) !void {
        const s = self.rig.s;
        const r = self.shape.rows;
        const n = self.shape.n;
        const p = params(self.which);
        if (self.ctl) |f| {
            var args: cuda.Args = .{};
            args.add(self.gate_ptr);
            args.add(self.up_ptr);
            args.add(self.out);
            args.add(@as(c_int, @intCast(n)));
            args.add(p.alpha);
            args.add(p.limit);
            args.add(p.up_bias);
            args.add(@as(c_int, if (self.which == .ctl_fmin) 0 else 1));
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(r), .y = @intCast((n + 255) / 256), .z = 1 }, .block = .{ .x = 256 } }, s, &args);
        } else {
            try self.k.swigluoai(s, if (self.isF32()) .f32 else .bf16, self.gate_ptr, n, self.up_ptr, n, self.out, n, r, n, p);
        }
        try s.synchronize();
    }

    pub fn count(self: *Mm3Act) usize {
        return self.shape.rows * self.shape.n;
    }

    /// Float64 swigluoai of the exact inputs, with the alpha the kernel is given (an fp32 value).
    pub fn reference(self: *Mm3Act, want: []f64, mag: []f64) !void {
        const p = params(self.which);
        const sw: mr.Switches = .{ .gate_clamp = if (p.symmetric_gate_clamp) .symmetric else .upper_only, .up_plus_one = p.up_bias != 0 };
        for (want, mag, self.gate, self.up) |*o, *m, gv, uv| {
            o.* = mr.swigluoai(gv, uv, p.alpha, p.limit, sw);
            m.* = @abs(o.*);
        }
    }
};

test "the controls are the two named faults, each at its gate" {
    try std.testing.expectEqual(@as(?g.Gate, .nonfinite), control(.ctl_fmin));
    try std.testing.expectEqual(@as(?g.Gate, .shapes), control(.ctl_tail));
    for ([_]Which{ .bf16, .bf16_symmetric_no_bias, .f32 }) |w| try std.testing.expectEqual(@as(?g.Gate, null), control(w));
}

test "the served shapes sit inside the launch's limits and the dense widths are 128 multiples" {
    for (served) |s| {
        try std.testing.expectEqual(@as(usize, 0), s.n % 128);
        try act.check(s.rows, s.n, s.n, s.n, s.n);
    }
}
