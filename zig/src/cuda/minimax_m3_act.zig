//! minimax_m3_act.cu: MiniMax-M3.1-Flash's pointwise swigluoai and router; unverified switches are launch arguments.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

const threads = 256;

/// act_mode values shared with the fused epilogues (exl3_experts.Layer.act_mode, exl3_block_exl3.Act.mode).
pub const mode_upper_clamp: i32 = 2;
pub const mode_symmetric_clamp: i32 = 3;

/// swigluoai(g, u) = clamp(g) * sigmoid(alpha * clamp(g)) * (clamp(u) + up_bias); `limit` <= 0 turns the clamps off.
pub const Params = struct {
    alpha: f32 = 1.702,
    limit: f32 = 7.0,
    up_bias: f32 = 1.0,
    symmetric_gate_clamp: bool = false,

    pub fn mode(p: Params) i32 {
        return if (p.symmetric_gate_clamp) mode_symmetric_clamp else mode_upper_clamp;
    }
};

pub const Input = enum { bf16, f32 };

/// The launch's checks, apart from the device: n and every stride positive, rows within the grid.
pub fn check(rows: usize, n: usize, g_ld: usize, u_ld: usize, o_ld: usize) error{RowShape}!void {
    if (n == 0 or g_ld < n or u_ld < n or o_ld < n) return error.RowShape;
    if (rows > std.math.maxInt(i32) or n > std.math.maxInt(i32)) return error.RowShape;
    if ((n + threads - 1) / threads > 65535) return error.RowShape;
}

pub const MiniMaxAct = struct {
    mod: Module,
    bf16_fn: Function,
    f32_fn: Function,
    route_fn: Function,

    pub fn load(d: *const Driver) !MiniMaxAct {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.minimax_m3_act);
        errdefer mod.unload();
        return .{ .mod = mod, .bf16_fn = try mod.function("tf_mm3_swigluoai_bf16"), .f32_fn = try mod.function("tf_mm3_swigluoai_f32"), .route_fn = try mod.function("tf_mm3_route_warp") };
    }

    pub fn unload(g: *MiniMaxAct) void {
        g.mod.unload();
    }

    /// out[r, :n] = bf16(swigluoai(gate, up)) at element strides g_ld, u_ld, o_ld; `in` the gate/up type.
    pub fn swigluoai(g: *const MiniMaxAct, s: Stream, in: Input, gate: u64, g_ld: usize, up: u64, u_ld: usize, out: u64, o_ld: usize, rows: usize, n: usize, p: Params) !void {
        try check(rows, n, g_ld, u_ld, o_ld);
        if (rows == 0) return;
        const eb: usize = if (in == .bf16) 2 else 4;
        if (gate % eb != 0 or up % eb != 0 or out % 2 != 0) return error.RowShape;
        var a: launch_.Args = .{};
        a.add(gate);
        a.add(@as(i64, @intCast(g_ld)));
        a.add(up);
        a.add(@as(i64, @intCast(u_ld)));
        a.add(out);
        a.add(@as(i64, @intCast(o_ld)));
        a.add(@as(i32, @intCast(n)));
        a.add(p.alpha);
        a.add(p.limit);
        a.add(p.up_bias);
        a.add(p.mode());
        const f = if (in == .bf16) g.bf16_fn else g.f32_fn;
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((n + threads - 1) / threads), .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// Top `topk` of fp32 logits by sigmoid(logit) + bias into int32 picks and fp32 weights (sigmoid [/ sum] * scale).
    pub fn route(g: *const MiniMaxAct, s: Stream, logits: u64, bias: ?u64, r: Route, pick: u64, wts: u64, rows: usize) !void {
        try routeCheck(r, rows);
        if (rows == 0) return;
        var a: launch_.Args = .{};
        a.add(logits);
        a.add(bias orelse 0);
        a.add(pick);
        a.add(wts);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(r.experts)));
        a.add(@as(i32, @intCast(r.topk)));
        a.add(@as(i32, @intFromBool(r.normalize)));
        a.add(r.scale);
        try launch_.launch(g.route_fn, .{ .grid = .{ .x = @intCast((rows + 7) / 8), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }
};

/// The router's launch limits (a warp a row, 4 values a lane).
pub const route_most_experts = 128;
pub const route_most_topk = 32;

pub const Route = struct {
    experts: usize = 128,
    topk: usize = 4,
    normalize: bool = true,
    scale: f32 = 2.0,
};

pub fn routeCheck(r: Route, rows: usize) error{RouteOutOfRange}!void {
    if (r.experts == 0 or r.experts > route_most_experts or r.topk == 0 or r.topk > route_most_topk or r.topk > r.experts) return error.RouteOutOfRange;
    if (rows > std.math.maxInt(i32) - 7) return error.RouteOutOfRange;
}

test "route checks" {
    try routeCheck(.{}, 8);
    try std.testing.expectError(error.RouteOutOfRange, routeCheck(.{ .experts = 129 }, 1));
    try std.testing.expectError(error.RouteOutOfRange, routeCheck(.{ .topk = 0 }, 1));
    try std.testing.expectError(error.RouteOutOfRange, routeCheck(.{ .experts = 4, .topk = 5 }, 1));
}

test "launch checks refuse empty widths, short strides and oversized grids" {
    try check(4, 3072, 3072, 6144, 3072);
    try std.testing.expectError(error.RowShape, check(4, 0, 1, 1, 1));
    try std.testing.expectError(error.RowShape, check(4, 3072, 3071, 3072, 3072));
    try std.testing.expectError(error.RowShape, check(4, 3072, 3072, 3072, 3071));
    try std.testing.expectError(error.RowShape, check(4, 256 * 65536, 256 * 65536, 256 * 65536, 256 * 65536));
}

test "params map to the act_mode the epilogues share" {
    try std.testing.expectEqual(@as(i32, 2), (Params{}).mode());
    try std.testing.expectEqual(@as(i32, 3), (Params{ .symmetric_gate_clamp = true }).mode());
}
