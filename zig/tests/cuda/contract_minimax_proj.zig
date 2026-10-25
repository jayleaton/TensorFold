//! MiniMax-M3.1-Flash fp16 index projection and pick under the contract on CUDA, float64 references.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const px = cuda.minimax_m3_proj;

const hidden = 6144;
const iq_w = 256; // a rank's two index heads
const ik_w = 128;

fn chain(s: c.Shape) usize {
    return s.k / 128 + 3 + 5;
}

/// bf16 or fp32 input to an fp16 or bf16 output; the fp32 form must equal the bf16 form's bits.
pub const Form = enum { bf16_f16, f32_f16, bf16_bf16 };

pub fn projSpec(f: Form) c.Spec {
    return .{
        .name = switch (f) {
            .bf16_f16 => "cuda minimax_m3_proj tf_mm3_proj_bf16 (fp16 out)",
            .f32_f16 => "cuda minimax_m3_proj tf_mm3_proj_f32 (fp16 out)",
            .bf16_bf16 => "cuda minimax_m3_proj tf_mm3_proj_bf16 (bf16 out)",
        },
        .accum = .f32,
        .depth = chain,
        .out = if (f == .bf16_bf16) .bf16 else .f16,
        .replaces = if (f == .f32_f16) "tf_mm3_proj_bf16" else null,
        // 4 rows a warp, 4 outputs a block, 128 values a step of the reduction
        .tiles = .{ .{ .size = px.row_group }, .{ .size = px.warps, .step = 1, .min = 1 }, .{ .size = px.chunk, .step = px.chunk, .min = px.chunk } },
        .served = &.{
            .{ .rows = 1, .n = iq_w + ik_w, .k = hidden },
            .{ .rows = 5, .n = iq_w + ik_w, .k = hidden },
            .{ .rows = 17, .n = iq_w + ik_w, .k = hidden },
            .{ .rows = 64, .n = iq_w + ik_w, .k = hidden },
            .{ .rows = 1, .n = iq_w, .k = hidden },
        },
    };
}

fn half(h: u16) f32 {
    const v: f16 = @bitCast(h);
    return v;
}

fn fp16(v: f32) u16 {
    const h: f16 = @floatCast(v);
    return @bitCast(h);
}

/// n outputs: the first (n + 1) / 2 are the query matrix, the rest the key matrix (n = 1: the key part is empty).
pub const Proj = struct {
    rig: rig_.Rig,
    form: Form,
    k: *const px.Proj,
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    base: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Proj) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Proj) g.Kernel {
        return .{ .spec = projSpec(self.form), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Proj);

    pub fn prepare(self: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % px.chunk != 0) return false;
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 1);
        const w = try a.alloc(u16, shape.n * shape.k);
        defer a.free(w);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6d6d3370);
        const rng = prng.random();
        const k = shape.k;
        for (0..shape.n) |j| for (0..k) |i| {
            // gate 4: integers mirrored into the second half, so fp32 sums are exact in any order
            const v: f32 = if (fill != .stress) (rng.float(f32) - 0.5) * 0.1 else if (i < k / 2) @floatFromInt(1 + rng.uintLessThan(u32, 7)) else -half(w[j * k + i - k / 2]);
            w[j * k + i] = fp16(v);
            d.w[j * k + i] = half(w[j * k + i]);
        };
        if (self.form == .f32_f16) {
            const xf = try a.alloc(f32, d.x.len);
            defer a.free(xf);
            for (xf, d.x) |*o, h| o.* = c.fromBf16(h);
            self.x = try self.rig.input(std.mem.sliceAsBytes(xf));
        } else self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(w));
        self.out = try self.rig.output(shape.rows * shape.n * 2);
        if (self.form == .f32_f16) self.base = try self.rig.scratch(shape.rows * shape.n * 2);
        return true;
    }

    fn launch(self: *Proj, x_f32: bool, x: u64, out: u64) !void {
        const sh = self.d.?.shape;
        const n0 = (sh.n + 1) / 2;
        const n1 = sh.n - n0;
        const a: px.Part = .{ .w = self.w, .n = @intCast(n0), .out = out, .out_ld = sh.n };
        const b: px.Part = .{ .w = self.w + n0 * sh.k * 2, .n = @intCast(n1), .out = out + n0 * 2, .out_ld = sh.n };
        try self.k.project(self.rig.s, x, sh.k, x_f32, @intCast(sh.k), a, b, sh.rows, if (self.form == .bf16_bf16) .bf16 else .f16);
    }

    pub fn run(self: *Proj) !void {
        try self.launch(self.form == .f32_f16, self.x, self.out);
        try self.rig.s.synchronize();
    }

    /// The bf16-input form on the same values.
    pub fn replaced(self: *Proj) ![]const u8 {
        const d = self.d.?;
        const xb = std.mem.sliceAsBytes(d.x);
        const dev = try self.rig.scratch(xb.len);
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = dev, .len = xb.len };
        try b.upload(0, xb);
        try self.launch(false, dev, self.base);
        return self.rig.bytes(self.base, self.count() * 2);
    }

    pub fn count(self: *Proj) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Proj, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};

pub const pick_spec: c.Spec = .{
    .name = "cuda minimax_m3_proj tf_mm3_pick",
    .accum = .none,
    .depth = pickDepth,
    .out = .f32,
    .nan = .{ .stated = "a NaN logit ranks last; all NaN picks the lowest id; larger logit wins, ties go to the lowest id" },
    // a row a CTA; 256 threads stride the slice, so the edges are the thread count and its neighbours
    .tiles = .{ .{}, .{}, .{ .size = px.pick_threads, .step = 1, .min = 1 } },
    .served = &.{
        .{ .rows = 1, .n = 1, .k = pick_slot },
        .{ .rows = 5, .n = 1, .k = pick_slot },
        .{ .rows = 128, .n = 1, .k = pick_slot },
        .{ .rows = 1, .n = 1, .k = pick_slot - 128 },
    },
    .sums = .{ .f16 = 0, .bf16 = 0, .why = "a pick compares and sums nothing" },
};

fn pickDepth(_: c.Shape) usize {
    return 1;
}

const pick_slot = 782 * 128; // the largest rank slice of the 200,064-token vocabulary
const pick_pad = 64; // decoy logits after the slice in every row
const pick_id0: u32 = 782 * 128; // rank 1's first id

pub const Pick = struct {
    rig: rig_.Rig,
    k: *const px.Proj,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    logits: []f32 = &.{},
    input: u64 = 0,
    output: u64 = 0,
    probe: u64 = 0,

    pub fn deinit(self: *Pick) void {
        self.rig.gpu.gpa.free(self.logits);
        self.rig.deinit();
    }

    pub fn kernel(self: *Pick) g.Kernel {
        return .{ .spec = pick_spec, .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Pick);

    fn ld(self: *const Pick) usize {
        return self.shape.k + pick_pad;
    }

    /// Row patterns by r % 8 (random, ties, NaNs, infinities, the maximum last); the padding is +inf, a decoy.
    pub fn prepare(self: *Pick, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 128 or shape.n != 1 or shape.k == 0 or shape.k > pick_slot) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.logits);
        self.logits = &.{};
        self.shape = shape;
        self.logits = try a.alloc(f32, shape.rows * self.ld());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const k = shape.k;
        const nan = std.math.nan(f32);
        for (0..shape.rows) |r| {
            const row = self.logits[r * self.ld() ..][0..self.ld()];
            @memset(row[k..], std.math.inf(f32));
            for (row[0..k], 0..) |*v, i| v.* = switch (r % 8) {
                0 => rng.floatNorm(f32) * 3,
                1 => @floatFromInt(rng.intRangeAtMost(i32, -3, 3)),
                2 => 1.5,
                3 => if (i % 5 == 0) nan else rng.floatNorm(f32),
                4 => nan,
                5 => if (i == k / 3 or i == k - 1) std.math.inf(f32) else -std.math.inf(f32),
                6 => -std.math.inf(f32),
                else => if (i + 1 == k) 100 else rng.floatNorm(f32),
            };
            if (fill == .stress) @memset(row[0..k], 0); // ties everywhere: id 0
        }
        self.input = try self.rig.input(std.mem.sliceAsBytes(self.logits));
        self.output = try self.rig.output(self.count() * 4);
        self.probe = try self.rig.scratch(8);
        return true;
    }

    fn launch(self: *Pick, logits: u64, out: u64, rows: usize) !void {
        try self.k.pick(self.rig.s, logits, self.ld(), @intCast(self.shape.k), pick_id0, rows, out);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Pick) !void {
        try self.launch(self.input, self.output, self.shape.rows);
    }

    pub fn count(self: *Pick) usize {
        return self.shape.rows * 2;
    }

    const Best = struct { logit: f32, id: u32 };

    /// The oracle: a linear scan with the stated order.
    fn expected(row: []const f32) Best {
        var best: Best = .{ .logit = row[0], .id = 0 };
        for (row[1..], 1..) |v, i| {
            const better = if (std.math.isNan(v)) false else if (std.math.isNan(best.logit)) true else v > best.logit;
            if (better) best = .{ .logit = v, .id = @intCast(i) };
        }
        return .{ .logit = best.logit, .id = pick_id0 + best.id };
    }

    pub fn reference(self: *Pick, want: []f64, mag: []f64) !void {
        for (0..self.shape.rows) |r| {
            const e = expected(self.logits[r * self.ld() ..][0..self.shape.k]);
            want[2 * r] = e.logit;
            want[2 * r + 1] = @as(f32, @bitCast(e.id));
            mag[2 * r] = 0;
            mag[2 * r + 1] = 0;
        }
    }

    /// Bits, not tolerances: the winning logit's bits (any NaN for an all-NaN row), its id, the guards, the inputs.
    pub fn check(self: *Pick, at: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const actual = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(actual);
        for (0..self.shape.rows) |r| {
            const e = expected(self.logits[r * self.ld() ..][0..self.shape.k]);
            const got_bits = std.mem.readInt(u32, actual[8 * r ..][0..4], .little);
            const got_id = std.mem.readInt(u32, actual[8 * r + 4 ..][0..4], .little);
            const got: f32 = @bitCast(got_bits);
            if (std.math.isNan(e.logit)) {
                if (!std.math.isNan(got)) return "an all-NaN row did not return NaN";
            } else if (got_bits != @as(u32, @bitCast(e.logit))) return "the winning logit's bits differ from the scan's";
            if (got_id != e.id) return "the winning id differs from the lowest-id scan";
            if (at == .shapes) {
                try self.launch(self.input + r * self.ld() * 4, self.probe, 1);
                if (!std.mem.eql(u8, actual[8 * r ..][0..8], try self.rig.bytes(self.probe, 8))) return "a batched row differs from the same row picked alone";
            }
        }
        if (!try self.rig.intact()) return "the pick changed a buffer guard";
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(self.logits), try self.rig.inputBytes())) return "the pick changed the logits";
        return null;
    }
};

/// tf-cuda-test mm3-proj-contract: these kernels alone through gates 1 to 4.
pub fn run(gpu: @import("check.zig").Gpu) !void {
    const check = @import("check.zig");
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var k = try px.Proj.load(gpu.d);
    defer k.unload();
    var projs: [3]Proj = undefined;
    for (std.enums.values(Form), &projs) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = f, .k = &k };
    defer for (&projs) |*x| x.deinit();
    var pick: Pick = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &k };
    defer pick.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    for (&projs) |*x| try kernels.append(a, x.kernel());
    try kernels.append(a, pick.kernel());
    var t: g.Tally = .{};
    try g.runAll(a, kernels.items, 0x6d33, &t);
    std.debug.print("SUMMARY mm3-proj-contract {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings\n", .{ t.kernels, t.passed, t.failed, t.known });
    try check.expect(t.failed == 0, "mm3-proj-contract: {d} gate failures", .{t.failed});
}

test "the served shapes fit the launches, and the pick's slice is rank 0's largest share" {
    for (projSpec(.bf16_f16).served) |s| {
        _ = try px.projLaunch(s.rows, s.n);
        try px.checkProj(s.k, s.k);
    }
    for (pick_spec.served) |s| _ = try px.pickLaunch(s.rows, s.k, s.k + pick_pad);
    try std.testing.expectEqual(@as(usize, 100_096), pick_slot);
    try std.testing.expectEqual(@as(?[]const u8, "tf_mm3_proj_bf16"), projSpec(.f32_f16).replaces);
}
