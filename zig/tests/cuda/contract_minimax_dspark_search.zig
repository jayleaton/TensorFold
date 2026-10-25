//! The dspark top-C and Markov chain kernels under the contract on CUDA.
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

// ---- top-C

pub const slot_n = 782 * 128; // the largest rank slice of the 200,064-token vocabulary
pub const pad = 64;
pub const id0: u32 = slot_n;

pub fn orderKey(v: f32) u32 {
    if (std.math.isNan(v)) return 0;
    var u: u32 = @bitCast(v);
    if (u == 0x8000_0000) u = 0;
    return if (u & 0x8000_0000 != 0) ~u else u | 0x8000_0000;
}

pub const topc_spec: c.Spec = .{
    .name = "cuda minimax_m3_dspark tf_mm3_ds_topc",
    .accum = .none,
    .depth = zeroDepth,
    .out = .raw,
    .nan = .{ .stated = "a NaN ranks below every number; -0 equals +0; larger wins, ties go to the lower id; fewer than c valid pads {-inf, -1}" },
    // rows, valid logits, candidates a row
    .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
    .served = &.{ .{ .rows = 7, .n = slot_n, .k = 64 }, .{ .rows = 1, .n = 5000, .k = 64 }, .{ .rows = 16, .n = 3000, .k = 128 }, .{ .rows = 7, .n = 40, .k = 64 }, .{ .rows = 8, .n = slot_n - 128, .k = 1 } },
    .sums = .{ .f16 = 0, .bf16 = 0, .why = "a selection compares and sums nothing" },
};

pub const TopC = struct {
    rig: rig_.Rig,
    k: *const ds.Dspark,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    logits: []f32 = &.{},
    dev: [2]u64 = .{ 0, 0 },
    msg: [200]u8 = undefined,

    pub fn deinit(self: *TopC) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *TopC) g.Kernel {
        return .{ .spec = topc_spec, .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(TopC);

    fn ld(self: *const TopC) usize {
        return self.shape.n + pad;
    }

    pub fn prepare(self: *TopC, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16 or shape.n == 0 or shape.n > slot_n or shape.k == 0 or shape.k > 128) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        self.logits = try a.alloc(f32, shape.rows * self.ld());
        const nan = std.math.nan(f32);
        const inf = std.math.inf(f32);
        for (0..shape.rows) |r| {
            const row = self.logits[r * self.ld() ..][0..self.ld()];
            @memset(row[shape.n..], inf); // decoys after the slice
            for (row[0..shape.n], 0..) |*v, i| v.* = switch (r % 8) {
                0 => rng.floatNorm(f32) * 3,
                1 => @floatFromInt(rng.intRangeAtMost(i32, -3, 3)),
                2 => 1.5,
                3 => if (i % 5 == 0) nan else rng.floatNorm(f32),
                4 => nan,
                5 => if (i % 7 == 0) inf else if (i % 2 == 0) -0.0 else 0.0,
                6 => -inf,
                else => if (i + 1 == shape.n) 100 else if (i % 3 == 0) -0.0 else @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -2, 2))),
            };
            if (fill == .nonfinite and r == 0) row[shape.n / 2] = inf;
        }
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.logits));
        self.dev[1] = try self.rig.output(shape.rows * shape.k * 8);
        return true;
    }

    pub fn run(self: *TopC) !void {
        const s = self.shape;
        try self.k.topC(self.rig.s, self.dev[0], self.ld(), @intCast(s.n), id0, @intCast(s.rows), @intCast(s.k), self.dev[1]);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *TopC) usize {
        return self.shape.rows * self.shape.k * 2;
    }

    pub fn reference(self: *TopC, want: []f64, mag: []f64) !void {
        _ = self;
        @memset(want, 0);
        @memset(mag, 0);
    }

    const Less = struct {
        row: []const f32,
        fn before(ctx: @This(), a: u32, b: u32) bool {
            const ka = orderKey(ctx.row[a]);
            const kb = orderKey(ctx.row[b]);
            return ka > kb or (ka == kb and a < b);
        }
    };

    /// The oracle: every index sorted by (key desc, index asc), the first c.
    pub fn check(self: *TopC, at: g.Gate) !?[]const u8 {
        _ = at;
        const s = self.shape;
        const al = self.arena.allocator();
        const out = try self.rig.outputBytes();
        const idx = try al.alloc(u32, s.n);
        defer al.free(idx);
        for (0..s.rows) |r| {
            const row = self.logits[r * self.ld() ..][0..s.n];
            for (idx, 0..) |*x, i| x.* = @intCast(i);
            std.mem.sort(u32, idx, Less{ .row = row }, Less.before);
            for (0..s.k) |j| {
                const o = (r * s.k + j) * 8;
                const got_bits = std.mem.readInt(u32, out[o..][0..4], .little);
                const got_id = std.mem.readInt(u32, out[o + 4 ..][0..4], .little);
                if (j < s.n) {
                    const want = row[idx[j]];
                    if (got_id != id0 + idx[j]) return std.fmt.bufPrint(&self.msg, "row {d} rank {d}: id {d} where the sorted order gives {d}", .{ r, j, got_id, id0 + idx[j] }) catch "topc id differs";
                    if (got_bits != @as(u32, @bitCast(want)) and !(std.math.isNan(want) and std.math.isNan(@as(f32, @bitCast(got_bits))))) return "a candidate's logit bits differ from its source";
                } else if (got_bits != @as(u32, @bitCast(-std.math.inf(f32))) or got_id != 0xffff_ffff) return "padding is not {-inf, -1}";
            }
        }
        if (!try self.rig.intact()) return "the top-C changed a buffer guard";
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(self.logits), try self.rig.inputBytes())) return "the top-C changed the logits";
        return null;
    }
};

// ---- Markov chain + confidence

pub const vocab_t = 3000;
pub const rank = 256;
pub const cand = 64;

pub const chain_spec: c.Spec = .{
    .name = "cuda minimax_m3_dspark tf_mm3_ds_chain",
    .accum = .none,
    .depth = zeroDepth,
    .out = .raw,
    .nan = .{ .stated = "a NaN score loses to every number, an equal score goes to the lower id, no valid candidate gives token -1; a NaN input gives NaN confidence" },
    // rows, candidates a rank, ranks
    .served = &.{ .{ .rows = 7, .n = cand, .k = 2 }, .{ .rows = 7, .n = cand, .k = 1 }, .{ .rows = 16, .n = 8, .k = 2 }, .{ .rows = 1, .n = 3, .k = 2 } },
    .sums = .{ .f16 = 0, .bf16 = 0, .why = "scores and confidence are checked against float64 with the fp32 chain's bound; no sums are declared" },
};

pub const Chain = struct {
    rig: rig_.Rig,
    k: *const ds.Dspark,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    cands: []u32 = &.{}, // [world][rows][cand] x {logit bits, id}
    post: []u16 = &.{},
    w1: []u16 = &.{},
    w2: []u16 = &.{},
    cw: []u16 = &.{},
    cb: [1]u16 = .{0},
    anchor: u32 = 17,
    dev: [6]u64 = @splat(0),
    out: u64 = 0,
    msg: [240]u8 = undefined,

    pub fn deinit(self: *Chain) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Chain) g.Kernel {
        return .{ .spec = chain_spec, .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Chain);

    fn ent(self: *const Chain, w: usize, r: usize, i: usize) *[2]u32 {
        const s = self.shape;
        return @ptrCast(self.cands[(((w * s.rows + r) * s.n) + i) * 2 ..][0..2]);
    }

    pub fn prepare(self: *Chain, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const world = shape.k;
        if (shape.rows == 0 or shape.rows > 16 or shape.n == 0 or world == 0 or world > 2 or world * shape.n > 256 or shape.n * world > vocab_t / 2) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const rows = shape.rows;
        self.cands = try a.alloc(u32, world * rows * shape.n * 2);
        self.post = try a.alloc(u16, rows * hidden);
        self.w1 = try a.alloc(u16, vocab_t * rank);
        self.w2 = try a.alloc(u16, vocab_t * rank);
        self.cw = try a.alloc(u16, hidden + rank);
        for (self.w1) |*e| e.* = bf(rng, 0.25);
        for (self.w2) |*e| e.* = bf(rng, 0.25);
        for (self.post) |*e| e.* = bf(rng, 1);
        for (self.cw) |*e| e.* = bf(rng, 0.02);
        self.cb[0] = bf(rng, 0.5);
        const perm = try a.alloc(u32, vocab_t);
        for (0..world) |w| for (0..rows) |r| {
            for (perm, 0..) |*p, i| p.* = @intCast(i);
            rng.shuffle(u32, perm);
            var m: usize = 0;
            for (perm) |id| { // rank w's ids are its own half of the vocabulary
                if (id / (vocab_t / 2) != w and world == 2) continue;
                if (m == shape.n) break;
                const e = self.ent(w, r, m);
                e[0] = @bitCast(rng.floatNorm(f32) * 3);
                e[1] = id;
                m += 1;
            }
            // padding at the end of a rank's list; one row with no candidate at all
            const invalid: usize = if (r % 3 == 2) @min(5, shape.n - 1) else 0;
            for (shape.n - invalid..shape.n) |i| {
                const e = self.ent(w, r, i);
                e[0] = @bitCast(-std.math.inf(f32));
                e[1] = 0xffff_ffff;
            }
            if (r == 4) for (0..shape.n) |i| {
                const e = self.ent(w, r, i);
                e[0] = @bitCast(-std.math.inf(f32));
                e[1] = 0xffff_ffff;
            };
        };
        // row 2: two candidates of rank 0 with equal scores that beat the rest: the lower id must win
        if (rows > 2 and shape.n >= 8) {
            const ea = self.ent(0, 2, 0);
            const eb = self.ent(0, 2, 1);
            @memcpy(self.w2[eb[1] * rank ..][0..rank], self.w2[ea[1] * rank ..][0..rank]);
            ea[0] = @bitCast(@as(f32, 50));
            eb[0] = @bitCast(@as(f32, 50));
        }
        if (fill == .nonfinite) {
            self.ent(0, 0, 3)[0] = @bitCast(std.math.nan(f32));
            if (rows > 1) self.ent(0, 1, 2)[0] = @bitCast(std.math.inf(f32));
            if (rows > 3) self.post[3 * hidden + 11] = c.toBf16(std.math.nan(f32));
        }
        const bytes = [_][]const u8{ std.mem.sliceAsBytes(self.cands), std.mem.sliceAsBytes(self.post), std.mem.sliceAsBytes(self.w1), std.mem.sliceAsBytes(self.w2), std.mem.sliceAsBytes(self.cw), std.mem.sliceAsBytes(self.cb[0..]) };
        for (bytes, &self.dev) |b, *p| p.* = try self.rig.input(b);
        self.out = try self.rig.output(rows * 8);
        return true;
    }

    pub fn run(self: *Chain) !void {
        const s = self.shape;
        const d = self.dev;
        try self.k.chain(self.rig.s, .{ .hidden = hidden, .rank = rank, .cand = @intCast(s.n), .world = @intCast(s.k), .rows = @intCast(s.rows) }, self.anchor, d[0], d[1], d[2], d[3], d[4], d[5], self.out);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Chain) usize {
        return self.shape.rows * 2;
    }

    pub fn reference(self: *Chain, want: []f64, mag: []f64) !void {
        _ = self;
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn bias(self: *const Chain, prev: usize, id: usize) f64 {
        var s: f64 = 0;
        for (self.w1[prev * rank ..][0..rank], self.w2[id * rank ..][0..rank]) |x, y| s += @as(f64, c.fromBf16(x)) * c.fromBf16(y);
        return s;
    }

    /// Follows the kernel's own previous tokens: each token must score within the fp32 chain's error of the best, ties
    pub fn check(self: *Chain, at: g.Gate) !?[]const u8 {
        _ = at;
        const s = self.shape;
        const out = try self.rig.outputBytes();
        var prev: usize = self.anchor;
        for (0..s.rows) |r| {
            const tok = std.mem.readInt(i32, out[8 * r ..][0..4], .little);
            const conf: f32 = @bitCast(std.mem.readInt(u32, out[8 * r + 4 ..][0..4], .little));
            // float64 scores of the valid candidates
            var best_id: i64 = -1;
            var best_z: f64 = std.math.nan(f64);
            var low_id: i64 = -1;
            var found: ?f64 = null;
            for (0..s.k) |w| for (0..s.n) |i| {
                const e = self.ent(w, r, i);
                if (e[1] == 0xffff_ffff) continue;
                const z = @as(f64, @as(f32, @bitCast(e[0]))) + self.bias(prev, e[1]);
                const id: i64 = e[1];
                if (low_id < 0 or id < low_id) low_id = id;
                if (tok == id) found = z;
                const better = if (std.math.isNan(z)) false else if (std.math.isNan(best_z)) true else z > best_z or (z == best_z and id < best_id);
                if (better) {
                    best_z = z;
                    best_id = id;
                }
            };
            if (low_id < 0) {
                if (tok != -1) return "no valid candidate but the token is not -1";
            } else if (std.math.isNan(best_z)) {
                if (tok != low_id) return "every score is NaN: the lowest id must win";
            } else {
                const z = found orelse return std.fmt.bufPrint(&self.msg, "row {d}: token {d} is not a candidate", .{ r, tok }) catch "token is not a candidate";
                const tol = 2e-4 * (1 + @abs(best_z));
                if (!(z == best_z or z >= best_z - tol)) return std.fmt.bufPrint(&self.msg, "row {d}: token {d} scores {e} where the best scores {e}", .{ r, tok, z, best_z }) catch "token is not the best";
                // an exact tie must go to the lower id
                var ties: usize = 0;
                for (0..s.k) |w| for (0..s.n) |i| {
                    const e = self.ent(w, r, i);
                    if (e[1] == 0xffff_ffff) continue;
                    const zz = @as(f64, @as(f32, @bitCast(e[0]))) + self.bias(prev, e[1]);
                    if (zz == best_z) ties += 1;
                };
                if (ties > 1 and tok != best_id) return "an exact tie did not go to the lower id";
            }
            // the confidence of this token's chain position uses the kernel's previous token
            var sum: f64 = c.fromBf16(self.cb[0]);
            var mg: f64 = 0;
            for (0..hidden) |j| {
                const t = @as(f64, c.fromBf16(self.post[r * hidden + j])) * c.fromBf16(self.cw[j]);
                sum += t;
                mg += @abs(t);
            }
            for (0..rank) |j| {
                const t = @as(f64, c.fromBf16(self.w1[prev * rank + j])) * c.fromBf16(self.cw[hidden + j]);
                sum += t;
                mg += @abs(t);
            }
            const want = mr.sigmoid(sum);
            if (std.math.isNan(want)) {
                if (!std.math.isNan(conf)) return "a NaN input must give NaN confidence";
            } else if (!(@abs(conf - want) <= 0.25 * 1e-5 * mg + 2e-6)) return std.fmt.bufPrint(&self.msg, "row {d}: confidence {e} where float64 gives {e}", .{ r, conf, want }) catch "confidence differs";
            prev = if (tok < 0) self.anchor else @intCast(tok);
        }
        if (!try self.rig.intact()) return "the chain changed a buffer guard";
        return null;
    }
};
