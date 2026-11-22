//! --precision checkpoint: routed NVFP4 experts in the checkpoint's math (NVFP4 rows on the FP4 mma, cuda/ck.zig).
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const routed = @import("cuda_modelopt_experts.zig");
const threads = @import("cuda_modelopt_threads.zig");

const ck = cuda.ck;

/// The FP4 mma's compute capability (sm_120a, sm_121a).
pub const major = 12;

/// --precision: the checkpoint's activation math, or bf16 rows against the stored weights (the default).
pub const Precision = enum { checkpoint, full };

/// Whether a load runs the checkpoint's math: only when asked (full stays the default) and on Blackwell.
pub fn wanted(capability_major: u32, asked: ?Precision) !bool {
    const p = asked orelse return false;
    if (p == .full) return false;
    if (capability_major != major) return error.CheckpointMathNeedsBlackwell;
    return true;
}

/// A layer's fp32 factors and its gate and up input scale, beside the words and tiles routed.Owned holds.
pub const Factors = struct {
    up_alpha: cuda.DeviceBuffer, // [E][2]
    down_alpha: cuda.DeviceBuffer, // [E]
    down_g: cuda.DeviceBuffer, // [E]
    up_act: f32,

    pub fn free(self: *Factors) void {
        self.up_alpha.free();
        self.down_alpha.free();
        self.down_g.free();
    }
};

/// ck.zig's view of a checkpoint-math layer; `skip` is the plan's id past the rank's experts.
pub fn view(o: *const routed.Owned, f: *const Factors) ck.Swiglu {
    return .{ .up = o.up.ptr, .up_s = o.up_scale.ptr, .down = o.down.ptr, .down_s = o.down_scale.ptr, .up_alpha = f.up_alpha.ptr, .down_alpha = f.down_alpha.ptr, .down_g = f.down_g.ptr, .up_act = f.up_act, .limit = 0, .experts = o.count, .width = o.width, .dims = o.dims, .skip = @intCast(o.count) };
}

/// Packed row c's source: gate and up interleave by 8 (16q + c: gate 8q + c, then up 8q + c - 8), or row c.
fn source(parts: []const threads.Plane, c: usize) struct { threads.Plane, usize } {
    if (parts.len == 1) return .{ parts[0], c };
    const w = c % 16;
    return .{ parts[w / 8], c / 16 * 8 + w % 8 };
}

/// One expert's rows in tf_ck_pack4's layout: words [n/64][k/64][8][32][2], scale tiles [n/64][k/64][64][4].
pub fn packRows(words: []u8, tiles: []u8, parts: []const threads.Plane, n: usize, k: usize) void {
    const kg = k / 64;
    std.debug.assert(n % 64 == 0 and k % 64 == 0 and words.len == n * k / 2 and tiles.len == n * k / 16);
    for (0..n / 64) |tile| for (0..kg) |step| for (0..8) |j| for (0..32) |lane| {
        const part, const r = source(parts, tile * 64 + j * 8 + lane / 4);
        const row = part.codes[r * (k / 2) + step * 32 ..];
        const at = ((((tile * kg + step) * 8 + j) * 64) + lane * 2) * 4;
        const t = lane % 4;
        @memcpy(words[at..][0..4], row[4 * t ..][0..4]);
        @memcpy(words[at + 4 ..][0..4], row[16 + 4 * t ..][0..4]);
    };
    for (0..n / 64) |tile| for (0..kg) |g| for (0..64) |c| {
        const part, const r = source(parts, tile * 64 + c);
        @memcpy(tiles[((tile * kg + g) * 64 + c) * 4 ..][0..4], part.scales[r * (k / 16) + g * 4 ..][0..4]);
    };
}

/// ck.zig's layout for threads.loadLayout: gate-up words, down words, gate-up scale tiles, down scale tiles.
const Layout = struct {
    pub const parts = 4;

    pub fn bytes(width: usize, dims: usize) ![parts]usize {
        const u = ck.Ck.packedBytes(1, 2 * width, dims);
        const d = ck.Ck.packedBytes(1, dims, width);
        return .{ u[0], d[0], u[1], d[1] };
    }

    pub fn pack(out: [parts][]u8, g: threads.Plane, u: threads.Plane, o: threads.Plane, width: usize, dims: usize) void {
        packRows(out[0], out[2], &.{ g, u }, 2 * width, dims);
        packRows(out[1], out[3], &.{o}, dims, width);
    }
};

fn inputScale(store: anytype, name: []const u8) !f32 {
    var buf: [512]u8 = undefined;
    const key = try std.fmt.bufPrint(&buf, "{s}.input_scale", .{name});
    if (!store.has(key)) return error.PrecisionNotBuilt;
    const t = try store.get(key); // held until the caller's one release: each release drops the shards' pages
    if (t.dtype != .f32 or t.bytes.len != 4) return error.UnexpectedTensor;
    const v: f32 = @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
    if (!std.math.isFinite(v) or v <= 0) return error.UnsupportedQuantization;
    return v;
}

/// One GPU's experts packed for ck.zig on the threaded loader, with every input scale and factor (no W4A16 copy).
pub fn load(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, world: usize, dims: usize, width: usize, count: usize) !routed.Owned {
    if (world != 1) return error.CheckpointMathOneRank;
    if (dims % 64 != 0 or width % 64 != 0) return error.UnexpectedTensor;
    // scales first: a checkpoint without them is refused before any expert is read
    const factors = try a.alloc(f32, 7 * count);
    defer a.free(factors);
    const up_alpha = factors[0 .. 2 * count];
    const down_alpha = factors[2 * count ..][0..count];
    const down_g = factors[3 * count ..][0..count];
    const up_w = factors[4 * count ..][0 .. 2 * count]; // the weights' globals, as threads.loadLayout places them
    const down_w = factors[6 * count ..][0..count];
    var buf: [512]u8 = undefined;
    errdefer store.release();
    const act = try inputScale(store, try std.fmt.bufPrint(&buf, "{s}.experts.0.gate_proj", .{prefix}));
    for (0..count) |e| {
        // one quantization of the layer's rows serves every expert's gate and up
        for ([_][]const u8{ "gate_proj", "up_proj" }) |proj|
            if (try inputScale(store, try std.fmt.bufPrint(&buf, "{s}.experts.{d}.{s}", .{ prefix, e, proj })) != act) return error.UnsupportedQuantization;
        const d_act = try inputScale(store, try std.fmt.bufPrint(&buf, "{s}.experts.{d}.down_proj", .{ prefix, e }));
        down_alpha[e] = d_act;
        down_g[e] = ck.inverse(d_act);
    }
    store.release();
    const n = try threads.configured();
    const direct = comptime @hasDecl(std.meta.Child(@TypeOf(store)), "sources");
    var bufs = if (direct and core.direct_gather.enabled())
        try threads.loadLayout(Layout, a, d, store, prefix, dims, width, count, n, threads.staging_bytes, true, up_w, down_w)
    else
        try threads.loadLayout(Layout, a, d, store, prefix, dims, width, count, n, threads.staging_bytes, false, up_w, down_w);
    errdefer for (&bufs) |*b| b.free();
    for (up_alpha, up_w) |*al, w| al.* = ck.alpha(act, w);
    for (down_alpha, down_w) |*al, w| al.* = ck.alpha(al.*, w);
    var ua = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(up_alpha));
    errdefer ua.free();
    var da = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(down_alpha));
    errdefer da.free();
    const dg = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(down_g));
    return .{ .up = bufs[0], .down = bufs[1], .up_scale = bufs[2], .down_scale = bufs[3], .width = width, .dims = dims, .count = count, .ck = .{ .up_alpha = ua, .down_alpha = da, .down_g = dg, .up_act = act } };
}

/// The routed pairs of an n-row pass: the layer's rows to NVFP4 once, SwiGLU up to NVFP4 pair rows, fp32 down into y.
pub fn run(self: anytype, owned: *const routed.Owned, f: *const Factors, picks: u64, n: usize, pairs: usize, plan: cuda.grouped.Plan) !void {
    const s = self.s;
    const k = &(self.k.ck orelse return error.PrecisionNotBuilt);
    const l = view(owned, f);
    const prompt = n >= ck.prompt_tokens;
    const pt: usize = if (prompt) ck.prompt_tile else 16;
    try self.k.router.route(self.stream, picks, pairs, owned.count + 1, pt, plan);
    const items = cuda.grouped.maxItems(pairs, owned.count + 1, pt);
    // moe_act holds down's NVFP4 pair rows, then the layer's NVFP4 rows (both far under its bf16 pair rows)
    const act = s.moe_act;
    const rows = act + std.mem.alignForward(usize, ck.rowsBytes(pairs, owned.width), 256);
    try k.quant4(self.stream, s.mixed, wt.dims, n, wt.dims, l.up_act, rows);
    try k.swigluUp(self.stream, l, rows, n, wt.slots, plan, pairs, act, items, prompt);
    try k.swigluDown(self.stream, l, act, pairs, n, plan, s.y, items, prompt);
}

test "gate and up interleave by 8 and the words take tf_ck_pack4's order" {
    const n = 64;
    const k = 64;
    var gc: [32 * k / 2]u8 = undefined;
    var uc: [32 * k / 2]u8 = undefined;
    var gs: [32 * k / 16]u8 = undefined;
    var us: [32 * k / 16]u8 = undefined;
    for (&gc, 0..) |*b, i| b.* = @truncate(i);
    for (&uc, 0..) |*b, i| b.* = @truncate(i + 128);
    for (&gs, 0..) |*b, i| b.* = @truncate(i);
    for (&us, 0..) |*b, i| b.* = @truncate(i + 200);
    var words: [n * k / 2]u8 = undefined;
    var tiles: [n * k / 16]u8 = undefined;
    packRows(&words, &tiles, &.{ .{ .codes = &gc, .scales = &gs }, .{ .codes = &uc, .scales = &us } }, n, k);
    // lane 4 of tile j 1 is packed row 9: up row 1; its t 0 word is bytes 0..3 of that row, then bytes 16..19
    try std.testing.expectEqualSlices(u8, uc[32..36], words[(64 + 8) * 4 ..][0..4]);
    try std.testing.expectEqualSlices(u8, uc[48..52], words[(64 + 8) * 4 + 4 ..][0..4]);
    // packed row 16 is gate row 8: its four scales open tile row 16
    try std.testing.expectEqualSlices(u8, gs[8 * 4 ..][0..4], tiles[16 * 4 ..][0..4]);
    try std.testing.expectEqual(false, try wanted(12, null));
    try std.testing.expectEqual(false, try wanted(8, .full));
    try std.testing.expectError(error.CheckpointMathNeedsBlackwell, wanted(9, .checkpoint));
}
