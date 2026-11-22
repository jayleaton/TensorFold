//! Flash Next's checkpoint-math SwiGLU experts under the contract on CUDA: ck_experts.cu's skipping pair, every form.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_ck_swiglu_ref.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Ck = cuda.ck.Ck;
const grouped = cuda.grouped;

/// Gate-up to down's NVFP4 pair rows, then fp32 down: the 64-column decode form, its 32-column twin, the prompt tile.
pub const Kind = enum { up, up_few, up_prompt, down, down_few, down_prompt };

/// A planted control's fault: alpha read a slot late, the next expert's global scale, the skipped expert computed.
pub const Fault = enum { none, alpha_late, next_g, write_skip };

/// The planted controls: real launches, each given one fault, which gate 3 must catch.
pub const controls = [_]struct { Kind, Fault }{ .{ .up, .alpha_late }, .{ .up_prompt, .next_g }, .{ .down, .write_skip }, .{ .down_prompt, .write_skip } };

pub const experts = 6; // routed experts; id `experts` is the skipped one, packed too so a fault could compute it
pub const ids = experts + 1;
pub const skip: i32 = experts;
pub const slots = 11; // Flash Next's pairs a token row
const decode_rule = "float64's classes from NaN and infinite alphas and a NaN block scale; skipped rows untouched";
const twin_rule = "as the decode form gives them: NaN and infinite alphas, a NaN block scale; skipped rows untouched";

pub const Form = enum { decode, few, prompt };

pub fn formOf(kind: Kind) Form {
    return switch (kind) {
        .up, .down => .decode,
        .up_few, .down_few => .few,
        .up_prompt, .down_prompt => .prompt,
    };
}

pub fn isUp(kind: Kind) bool {
    return kind == .up or kind == .up_few or kind == .up_prompt;
}

/// A 64-input mma step a link, after the products inside it.
pub fn stepDepth(s: c.Shape) usize {
    return s.k / 64 + 64;
}

/// Output o's gate column in the packed gate-up weights (16q + c: gate 8q + c for c < 8); up's is 8 past it.
pub fn gateCol(o: usize) usize {
    return o / 8 * 16 + o % 8;
}

fn shapes(comptime n: usize, comptime k: usize, comptime rows: []const usize) [rows.len]c.Shape {
    var out: [rows.len]c.Shape = undefined;
    for (rows, &out) |r, *s| s.* = .{ .rows = r, .n = n, .k = k };
    return out;
}

/// Flash Next's routed experts (dims 2,560, width 640): gate-up 1,280 packed columns from 2,560, down 2,560 from 640.
const served = struct {
    const up_decode = shapes(1280, 2560, &.{ 1, 16, 32, 64 });
    const up_few = shapes(1280, 2560, &.{ 1, 16, 32 });
    const up_prompt = shapes(1280, 2560, &.{ 128, 129, 300 });
    const down_decode = shapes(2560, 640, &.{ 1, 16, 32, 64 });
    const down_few = shapes(2560, 640, &.{ 1, 16, 32 });
    const down_prompt = shapes(2560, 640, &.{ 128, 129, 300 });
};

pub fn spec(kind: Kind, fault: Fault) c.Spec {
    const tiles = [3]c.Tile{ .{ .size = 16 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 256, .step = 64, .min = 64 } };
    // a prompt block: a plan item of 128 pairs (11 token rows and a part) by 128 columns
    const prompt_tiles = [3]c.Tile{ .{ .size = cuda.ck.prompt_tile / slots }, .{ .size = 128, .step = 64, .min = 64 }, .{ .size = 256, .step = 64, .min = 64 } };
    const up = isUp(kind);
    const list: []const c.Shape = switch (kind) {
        .up => &served.up_decode,
        .up_few => &served.up_few,
        .up_prompt => &served.up_prompt,
        .down => &served.down_decode,
        .down_few => &served.down_few,
        .down_prompt => &served.down_prompt,
    };
    const name: []const u8 = if (fault == .none) switch (kind) {
        .up => "cuda ck swiglu up tf_ck4_swiglu_nvfp4 (NVFP4 pair rows out, decode form)",
        .up_few => "cuda ck swiglu up tf_ck4_swiglu_nvfp4_few (NVFP4 pair rows out)",
        .up_prompt => "cuda ck swiglu up tf_ck4_swiglu_nvfp4_prompt (NVFP4 pair rows out)",
        .down => "cuda ck swiglu down tf_ck4_down_f32_skip (decode form)",
        .down_few => "cuda ck swiglu down tf_ck4_down_f32_skip_few",
        .down_prompt => "cuda ck swiglu down tf_ck4_down_f32_skip_prompt",
    } else switch (kind) {
        .up => "control: tf_ck4_swiglu_nvfp4 reading alpha a slot late",
        .up_prompt => "control: tf_ck4_swiglu_nvfp4_prompt under the next expert's global scale",
        .down => "control: tf_ck4_down_f32_skip computing the skipped expert",
        .down_prompt => "control: tf_ck4_down_f32_skip_prompt computing the skipped expert",
        else => "control: tf_ck4 swiglu (unplanned)",
    };
    const decode = formOf(kind) == .decode;
    const replaces: ?[]const u8 = if (decode) null else if (up) "tf_ck4_swiglu_nvfp4" else "tf_ck4_down_f32_skip";
    return .{ .name = name, .accum = .f32, .depth = stepDepth, .ops = if (up) 0 else 1, .out = if (up) .raw else .f32, .replaces = replaces, .nan = .{ .stated = if (decode) decode_rule else twin_rule }, .tiles = if (formOf(kind) == .prompt) prompt_tiles else tiles, .served = list };
}

/// shape.rows token rows, `slots` pairs each over `ids` ids; gate-up reads a token's row, down a pair's.
pub const Experts = struct {
    rig: rig_.Rig,
    kind: Kind,
    fault: Fault = .none,
    ck: *const Ck,
    router: grouped.Router,
    shape: c.Shape = undefined,
    x: []u8 = &.{}, // NVFP4 rows: codes, then block scales
    xv: []f64 = &.{}, // x's values
    w: []u8 = &.{}, // [ids][n][k/2] codes; gate-up's n columns are packed as gateCol says
    ws: []u8 = &.{}, // [ids][n][k/16] block scales
    alpha: [2 * ids]f32 = undefined, // gate-up: [ids][2] gate's then up's; down: [ids]
    act: [ids]f32 = undefined, // gate-up: each expert's down input scale; its rows' quantizer takes 1 / it
    picks: []i32 = &.{},
    members: []usize = &.{}, // pairs by expert, `first` apart
    first: [ids + 1]usize = undefined,
    xd: u64 = 0,
    wd: [2]u64 = undefined,
    ad: u64 = 0,
    gd: u64 = 0,
    plan: grouped.Plan = undefined, // 16 pairs an item: the decode form's and the 32-column form's
    wide: grouped.Plan = undefined, // cuda.ck.prompt_tile pairs an item: the prompt tile's
    out: u64 = 0,
    base: u64 = 0, // the decode form's bytes, 0xff wherever it writes nothing
    msg: [192]u8 = undefined,

    pub fn deinit(self: *Experts) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Experts) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.xv);
        a.free(self.w);
        a.free(self.ws);
        a.free(self.picks);
        a.free(self.members);
        self.x = &.{};
        self.xv = &.{};
        self.w = &.{};
        self.ws = &.{};
        self.picks = &.{};
        self.members = &.{};
    }

    pub fn kernel(self: *Experts) g.Kernel {
        return .{ .spec = spec(self.kind, self.fault), .ctx = self, .vt = &vt, .control = if (self.fault == .none) null else .shapes };
    }

    const vt = g.hooks(Experts);

    pub fn up(self: *const Experts) bool {
        return isUp(self.kind);
    }

    pub fn pairs(self: *const Experts) usize {
        return self.shape.rows * slots;
    }

    fn xRows(self: *const Experts) usize {
        return if (self.up()) self.shape.rows else self.pairs();
    }

    fn per(self: *const Experts) usize {
        return if (self.up()) 2 else 1;
    }

    fn items(self: *const Experts, tile: usize) usize {
        return grouped.maxItems(self.pairs(), ids, tile);
    }

    fn outBytes(self: *const Experts) usize {
        return if (self.up()) cuda.ck.rowsBytes(self.pairs(), self.shape.n / 2) else self.pairs() * self.shape.n * 4;
    }

    /// Expert e's pairs.
    pub fn of(self: *const Experts, e: usize) []const usize {
        const lo = self.first[e];
        const hi = self.first[e + 1];
        return self.members[lo..hi];
    }

    /// Pair p's input values: its token's row for gate-up, its own for down.
    pub fn row(self: *const Experts, p: usize) []const f64 {
        const k = self.shape.k;
        return self.xv[(if (self.up()) p / slots else p) * k ..][0..k];
    }

    /// Column j of expert e's weights, as values.
    pub fn weights(self: *const Experts, e: usize, j: usize, out: []f64) void {
        for (out, 0..) |*o, i| o.* = value(self.w, self.ws, e * self.shape.n + j, self.shape.k, i);
    }

    pub fn say(self: *Experts, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&self.msg, fmt, args);
    }

    /// A plan's buffers for `tile` pairs an item, routed from the picks over every id.
    fn route(self: *Experts, picks: u64, tile: usize, zeros: []const u8) !grouped.Plan {
        const blocks = (self.pairs() + 1023) / 1024;
        const p: grouped.Plan = .{ .members = try self.rig.input(zeros[0 .. self.pairs() * 4]), .items = try self.rig.input(zeros[0 .. self.items(tile) * 12]), .counts = try self.rig.input(zeros[0..8]), .rank = try self.rig.scratch(self.pairs() * 4), .hist = try self.rig.scratch(blocks * ids * 4) };
        try self.router.route(self.rig.s, picks, self.pairs(), ids, tile, p);
        return p;
    }

    /// Random codes and scales, every id picked; non-finite: planted alphas and scale; stress: mirrored, none skipped.
    pub fn prepare(self: *Experts, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const form = formOf(self.kind);
        if (shape.rows == 0 or (form == .few and shape.rows > cuda.ck.few_tokens) or shape.n % 64 != 0 or shape.n == 0 or shape.k % 64 != 0 or shape.k == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const n = shape.n;
        const k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(u8, cuda.ck.rowsBytes(self.xRows(), k));
        self.w = try a.alloc(u8, ids * n * k / 2);
        self.ws = try a.alloc(u8, ids * n * k / 16);
        self.picks = try a.alloc(i32, self.pairs());
        self.members = try a.alloc(usize, self.pairs());
        rng.bytes(self.x[0 .. self.xRows() * k / 2]);
        rng.bytes(self.w);
        for (self.x[self.xRows() * k / 2 ..]) |*s| s.* = 0x20 + rng.uintLessThan(u8, 0x28);
        for (self.ws) |*s| s.* = 0x20 + rng.uintLessThan(u8, 0x28);
        for (&self.alpha) |*v| v.* = 0.005 + rng.float(f32) * 0.03;
        for (&self.act, 0..) |*v, e| v.* = ([_]f32{ 2, 4, 8 })[e % 3];
        const routed: usize = if (fill == .stress) experts else ids;
        for (self.picks, 0..) |*p, i| p.* = @intCast(if (i < routed) i else rng.uintLessThan(usize, routed));
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => self.stress(rng),
        }
        self.group();
        try self.values();
        self.xd = try self.rig.input(self.x);
        const packed_bytes = Ck.packedBytes(ids, n, k);
        const zeros = try a.alloc(u8, @max(packed_bytes[0], self.pairs() * 4, self.items(16) * 12));
        defer a.free(zeros);
        @memset(zeros, 0);
        self.wd = .{ try self.rig.input(zeros[0..packed_bytes[0]]), try self.rig.input(zeros[0..packed_bytes[1]]) };
        const src = [2]u64{ try self.rig.scratch(self.w.len), try self.rig.scratch(self.ws.len) };
        for (src, [2][]const u8{ self.w, self.ws }) |p, h| try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = p, .len = h.len }).upload(0, h);
        try self.ck.pack(self.rig.s, src[0], src[1], ids, n, k, 0, k, self.wd[0], self.wd[1]);
        self.ad = try self.rig.input(std.mem.sliceAsBytes(self.alpha[0 .. ids * self.per()]));
        var gs: [ids]f32 = undefined;
        for (&gs, self.act) |*v, s| v.* = cuda.ck.inverse(s);
        self.gd = try self.rig.input(std.mem.sliceAsBytes(&gs));
        const picks = try self.rig.input(std.mem.sliceAsBytes(self.picks));
        self.plan = try self.route(picks, 16, zeros);
        if (form == .prompt) self.wide = try self.route(picks, cuda.ck.prompt_tile, zeros);
        try self.rig.s.synchronize();
        self.out = try self.rig.output(self.outBytes());
        self.base = try self.rig.scratch(self.outBytes());
        return true;
    }

    /// Gate 1: expert 1's gate (down: its) alpha NaN, expert 2's up +Inf, expert 3's gate -Inf, a NaN block scale.
    fn plant(self: *Experts) void {
        const p = self.per();
        self.alpha[p] = std.math.nan(f32);
        self.alpha[2 * p + p - 1] = std.math.inf(f32);
        self.alpha[3 * p] = -std.math.inf(f32);
        self.ws[0] = 0x7f; // expert 0's first column (gate-up: output 0's gate), first block
    }

    /// Gate 4: block scales 16 or 64+ (x) and 8 (w), values 4 and 6 mirrored (8 of x's a step lower), alphas 1.
    fn stress(self: *Experts, rng: std.Random) void {
        const k = self.shape.k;
        const xs = self.x[self.xRows() * k / 2 ..];
        if (self.up()) {
            @memset(xs, 0x58);
        } else {
            // down's x scales 64 to 120, mirrored by block: 32-lane chains pass fp16's 65,504 and outgrow bf16's 8 bits
            for (0..self.xRows()) |r| for (0..k / 32) |b| {
                const scale: u8 = 0x68 + rng.uintLessThan(u8, 8);
                xs[r * k / 16 + b] = scale;
                xs[r * k / 16 + k / 32 + b] = scale;
            };
        }
        @memset(self.ws, 0x50);
        @memset(&self.alpha, 1);
        // the sums are 4,096 to 12,288, so SwiGLU's products stay inside e4m3 block scales under these
        for (&self.act, 0..) |*v, e| v.* = if (e % 2 == 0) 65536 else 131072;
        for (0..self.xRows()) |r| for (0..k / 2) |i| {
            const v: u8 = 6 + rng.uintLessThan(u8, 2);
            put(self.x[r * k / 2 ..], i, v);
            put(self.x[r * k / 2 ..], k / 2 + i, if (i < 8) v - 1 else v);
        };
        for (0..ids * self.shape.n) |j| for (0..k / 2) |i| {
            const v: u8 = 6 + rng.uintLessThan(u8, 2);
            put(self.w[j * k / 2 ..], i, v);
            put(self.w[j * k / 2 ..], k / 2 + i, v | 8);
        };
    }

    fn put(codes: []u8, i: usize, v: u8) void {
        const b = &codes[i / 2];
        b.* = if (i % 2 == 0) (b.* & 0xf0) | v else (b.* & 0x0f) | v << 4;
    }

    fn value(codes: []const u8, scales: []const u8, r: usize, k: usize, i: usize) f64 {
        const b = codes[r * k / 2 + i / 2];
        const nib = if (i % 2 == 0) b & 0xf else b >> 4;
        return core.quant.nvfp4.e2m1(nib) * core.quant.fp8.e4m3(scales[r * k / 16 + i / 16]);
    }

    /// The pairs grouped by expert.
    fn group(self: *Experts) void {
        @memset(&self.first, 0);
        for (self.picks) |p| self.first[@as(usize, @intCast(p)) + 1] += 1;
        for (0..ids) |e| self.first[e + 1] += self.first[e];
        var at = self.first;
        for (self.picks, 0..) |p, i| {
            const e: usize = @intCast(p);
            self.members[at[e]] = i;
            at[e] += 1;
        }
    }

    fn values(self: *Experts) !void {
        const k = self.shape.k;
        self.xv = try self.rig.gpu.gpa.alloc(f64, self.xRows() * k);
        const xs = self.x[self.xRows() * k / 2 ..];
        for (0..self.xRows()) |r| for (0..k) |i| {
            self.xv[r * k + i] = value(self.x, xs, r, k, i);
        };
    }

    /// ck.zig's view of the packed experts; `faulty` applies this entry's planted fault.
    fn layer(self: *const Experts, faulty: bool) cuda.ck.Swiglu {
        const u = self.up();
        const f = if (faulty) self.fault else .none;
        const late: u64 = if (f == .alpha_late) 4 else 0;
        const next: u64 = if (f == .next_g) 4 else 0;
        return .{ .up = self.wd[0], .up_s = self.wd[1], .down = self.wd[0], .down_s = self.wd[1], .up_alpha = self.ad + late, .down_alpha = self.ad, .down_g = self.gd + next, .up_act = 1, .limit = 0, .experts = ids, .width = if (u) self.shape.n / 2 else self.shape.k, .dims = if (u) self.shape.k else self.shape.n, .skip = if (f == .write_skip) -1 else skip };
    }

    pub fn run(self: *Experts) !void {
        const l = self.layer(true);
        const form = formOf(self.kind);
        if (form == .decode) {
            try self.decode(self.out, l);
        } else {
            const prompt = form == .prompt;
            const p = if (prompt) self.wide else self.plan;
            const n = self.items(if (prompt) cuda.ck.prompt_tile else 16);
            if (self.up()) {
                try self.ck.swigluUp(self.rig.s, l, self.xd, self.shape.rows, slots, p, self.pairs(), self.out, n, prompt);
            } else try self.ck.swigluDown(self.rig.s, l, self.xd, self.pairs(), self.shape.rows, p, self.out, n, prompt);
        }
        try self.rig.s.synchronize();
    }

    /// The 64-column decode form at any rows (Ck.swigluUp and swigluDown take it only from 33 token rows).
    fn decode(self: *Experts, out: u64, l: cuda.ck.Swiglu) !void {
        const u = self.up();
        const n = self.shape.n;
        const k = self.shape.k;
        const nrows = self.xRows();
        var a: cuda.Args = .{};
        a.add(self.xd);
        a.add(self.xd + nrows * k / 2);
        a.add(@as(i32, if (u) slots else 0));
        a.add(if (u) l.up else l.down);
        a.add(if (u) l.up_s else l.down_s);
        a.add(if (u) l.up_alpha else l.down_alpha);
        a.add(@as(i32, @intCast(k)));
        a.add(@as(i32, @intCast(n)));
        for ([_]u64{ self.plan.items, self.plan.counts, self.plan.members, out }) |v| a.add(v);
        a.add(if (u) out + self.pairs() * n / 4 else @as(u64, 0));
        a.add(if (u) l.down_g else @as(u64, 0));
        a.add(l.limit);
        a.add(l.skip);
        const j = @intFromBool(!u);
        const grid = @min((self.items(16) * (n / Ck.warp_columns[0]) + 3) / 4, self.ck.swiglu_resident[0][j]);
        if (grid < 1) return;
        try cuda.launch.launch(self.ck.swiglu_fns[0][j], .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = 128 } }, self.rig.s, &a);
    }

    /// The decode form's bytes on the prepared inputs, over an output of 0xff bytes.
    pub fn replaced(self: *Experts) ![]const u8 {
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.base, .len = self.outBytes() }).fill8(0xff, self.rig.s.handle);
        try self.decode(self.base, self.layer(false));
        return self.rig.bytes(self.base, self.outBytes());
    }

    /// The decode forms' gates 1 and 3 against float64; the others' gate 1: decode's bytes, a non-finite routed output.
    pub fn check(self: *Experts, at: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        if (formOf(self.kind) == .decode) {
            if (self.up()) return ref.up(self, got, at);
            return ref.down(self, got, at);
        }
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |i| return try self.say("byte {d} of {d} differs from {s}'s", .{ i, got.len, spec(self.kind, .none).replaces.? });
        if (try self.untouched(got)) |why| return why;
        if (at != .nonfinite or self.planted(got)) return null;
        return try self.say("no routed output is non-finite: the planted values prove nothing", .{});
    }

    /// Null when every row of the skipped expert's pairs still holds the output's 0xff bytes.
    pub fn untouched(self: *Experts, got: []const u8) !?[]const u8 {
        const n = self.shape.n;
        const no = n / 2;
        for (self.of(experts)) |p| {
            const codes = std.mem.allEqual(u8, if (self.up()) got[p * no / 2 ..][0 .. no / 2] else got[p * n * 4 ..][0 .. n * 4], 0xff);
            const clean = codes and (!self.up() or std.mem.allEqual(u8, got[self.pairs() * no / 2 + p * no / 16 ..][0 .. no / 16], 0xff));
            if (!clean) return try self.say("pair {d} picks the skipped expert, and its row was written", .{p});
        }
        return null;
    }

    /// Some routed output is non-finite: a NaN block scale (gate-up), a non-finite value (down).
    fn planted(self: *const Experts, got: []const u8) bool {
        const n = self.shape.n;
        const no = n / 2;
        for (0..experts) |e| for (self.of(e)) |p| {
            if (self.up()) {
                for (got[self.pairs() * no / 2 + p * no / 16 ..][0 .. no / 16]) |b| if ((b & 0x7f) == 0x7f) return true;
            } else for (0..n) |j| if (c.classOf(c.read(.f32, got, p * n + j)) != .finite) return true;
        };
        return false;
    }

    /// Outputs: gate-up's NVFP4 bytes, or down's values.
    pub fn count(self: *Experts) usize {
        return if (self.up()) self.outBytes() else self.pairs() * self.shape.n;
    }

    pub fn reference(self: *Experts, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        if (self.up()) return self.sums(.none, .sequential, want);
        const a = self.rig.gpu.gpa;
        const n = self.shape.n;
        const w = try a.alloc(f64, self.shape.k);
        defer a.free(w);
        @memset(want, std.math.nan(f64)); // the skipped rows keep the output's NaN
        for (0..experts) |e| for (0..n) |j| {
            self.weights(e, j, w);
            for (self.of(e)) |p| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (self.row(p), w) |x, y| {
                    sum += x * y;
                    abs += @abs(x * y);
                }
                want[p * n + j] = sum * self.alpha[e];
                mag[p * n + j] = @abs(self.alpha[e]) * abs;
            }
        };
    }

    pub fn simulate(self: *Experts, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.sums(acc, order, out);
    }

    /// Every output with each sum in `acc` (none: float64); gate-up's as the bytes quant4 makes of its bf16 rows.
    fn sums(self: *Experts, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const n = self.shape.n;
        const k = self.shape.k;
        const buf = try a.alloc(f64, 3 * k);
        defer a.free(buf);
        const t = buf[0..k];
        const wg = buf[k .. 2 * k];
        const wu = buf[2 * k ..];
        if (!self.up()) {
            @memset(out, std.math.nan(f64));
            for (0..experts) |e| for (0..n) |j| {
                self.weights(e, j, wg);
                for (self.of(e)) |p| {
                    for (t, self.row(p), wg) |*o, x, y| o.* = x * y;
                    out[p * n + j] = contract.rows.sumIn(acc, order, t) * self.alpha[e];
                }
            };
            return;
        }
        const no = n / 2;
        const rows = try a.alloc(u16, self.pairs() * no);
        defer a.free(rows);
        @memset(rows, 0);
        for (0..experts) |e| for (0..no) |o| {
            self.weights(e, gateCol(o), wg);
            self.weights(e, gateCol(o) + 8, wu);
            for (self.of(e)) |p| {
                for (t, self.row(p), wg) |*v, x, y| v.* = x * y;
                const sg = contract.rows.sumIn(acc, order, t);
                for (t, self.row(p), wu) |*v, x, y| v.* = x * y;
                rows[p * no + o] = self.swiglu(e, sg, contract.rows.sumIn(acc, order, t));
            }
        };
        try self.quantize(rows, out);
    }

    /// ck_experts.cu's epilogue on fp32 sums: bf16(sum * alpha) for gate and up, bf16(silu(gate)) * up to bf16.
    fn swiglu(self: *const Experts, e: usize, sg: f64, su: f64) u16 {
        const gv = c.fromBf16(c.toBf16(@as(f32, @floatCast(sg)) * self.alpha[2 * e]));
        const uv = c.fromBf16(c.toBf16(@as(f32, @floatCast(su)) * self.alpha[2 * e + 1]));
        const s = c.fromBf16(c.toBf16(gv / (1 + @exp(-gv))));
        return c.toBf16(s * uv);
    }

    /// tf_ck_quant4's bytes of each routed pair's bf16 row under its expert's global scale; others stay 0xff.
    fn quantize(self: *Experts, rows: []const u16, out: []f64) !void {
        const no = self.shape.n / 2;
        const m = self.pairs();
        const total = self.outBytes();
        const src = try self.rig.scratch(rows.len * 2);
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = src, .len = rows.len * 2 }).upload(0, std.mem.sliceAsBytes(rows));
        const dst = try self.rig.scratch(total);
        @memset(out, 0xff);
        for (0..experts) |e| {
            try self.ck.quant4(self.rig.s, src, no, m, no, self.act[e], dst);
            const got = try self.rig.bytes(dst, total);
            for (self.of(e)) |p| {
                const codes = p * no / 2;
                const scales = m * no / 2 + p * no / 16;
                for (out[codes..][0 .. no / 2], got[codes..][0 .. no / 2]) |*o, b| o.* = @floatFromInt(b);
                for (out[scales..][0 .. no / 16], got[scales..][0 .. no / 16]) |*o, b| o.* = @floatFromInt(b);
            }
        }
    }
};

/// Every kind and the planted controls on one Ck (none without one: the FP4 mma needs sm_120a).
pub const Group = struct {
    a: std.mem.Allocator,
    entries: [kinds.len + controls.len]Experts = undefined,
    n: usize = 0,
    const kinds = std.enums.values(Kind);

    pub fn create(a: std.mem.Allocator, gpu: check_.Gpu, s: cuda.Stream, ck: ?*const Ck, router: grouped.Router, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        self.* = .{ .a = a };
        errdefer self.destroy();
        const q = ck orelse return self;
        for (kinds) |kind| try self.add(.{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .ck = q, .router = router }, kernels);
        for (controls) |ctl| try self.add(.{ .rig = .{ .gpu = gpu, .s = s }, .kind = ctl[0], .fault = ctl[1], .ck = q, .router = router }, kernels);
        return self;
    }

    fn add(self: *Group, x: Experts, kernels: *std.ArrayList(g.Kernel)) !void {
        self.entries[self.n] = x;
        self.n += 1;
        try kernels.append(self.a, self.entries[self.n - 1].kernel());
    }

    pub fn destroy(self: *Group) void {
        for (self.entries[0..self.n]) |*x| x.deinit();
        self.a.destroy(self);
    }
};
