//! MiniMax-M3.1-Flash's FFN wiring (engine-port-plan.md T4): which kernel parameters the unverified T0 switches become,
//! the router's launch plan, and the layer descriptor for the fused swigluoai epilogues. Host-testable; no GPU work here.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const mr = core.minimax_ref;
const act = cuda.minimax_m3_act;

pub const Switches = mr.Switches;
pub const Layer = cuda.exl3_experts.Layer;

/// The checkpoint's swigluoai constants (REFERENCE, plan section 2); the switches below are UNVERIFIED.
pub const alpha: f32 = 1.702;
pub const limit: f32 = 7.0;
pub const routed_scale: f32 = 2.0;
pub const experts: usize = 128;
pub const topk: usize = 4;

/// The kernel parameters for a set of switches: the gate clamp's sides and the "+1" are arguments, never constants.
pub fn actParams(sw: Switches) act.Params {
    return .{
        .alpha = alpha,
        .limit = limit,
        .up_bias = if (sw.up_plus_one) 1.0 else 0.0,
        .symmetric_gate_clamp = sw.gate_clamp == .symmetric,
    };
}

/// Set a routed-experts layer to swigluoai. The loader builds `gate` from w1 and `up` from w3; when the switch says w1 is
/// the linear side instead (`w1_is_gate` false), the two descriptors swap so the kernels' "gate" is w3.
pub fn configureLayer(l: *Layer, sw: Switches) void {
    const p = actParams(sw);
    l.act_mode = p.mode();
    l.alpha = p.alpha;
    l.up_bias = p.up_bias;
    l.limit = p.limit;
    if (!sw.w1_is_gate) {
        std.mem.swap(u64, &l.gate, &l.up);
        std.mem.swap(u64, &l.gate_k2, &l.up_k2);
        std.mem.swap(u64, &l.suh_g, &l.suh_u);
        std.mem.swap(u64, &l.svh_g, &l.svh_u);
    }
}

/// For a dense layer or the shared expert, the (gate, up) of the pointwise kernel given w1/w3 or gate_proj/up_proj outputs.
pub fn pickGateUp(sw: Switches, w1: u64, w3: u64) struct { gate: u64, up: u64 } {
    return if (sw.w1_is_gate) .{ .gate = w1, .up = w3 } else .{ .gate = w3, .up = w1 };
}

/// What a router launch needs, from the switches: the kernel's own normalize/scale, and the scale to apply after routed +
/// shared are added (1 unless the switch says the factor multiplies both).
pub const RoutePlan = struct {
    route: act.Route,
    post_scale: f32,
};

pub fn routePlan(sw: Switches) error{Unsupported}!RoutePlan {
    // Both kernels weigh by sigmoid(logit) alone; sigmoid(logit) + bias as the weight is not implemented.
    if (sw.router_weights_use_bias) return error.Unsupported;
    const total = mr.totalScale(sw, routed_scale);
    return .{
        .route = .{
            .experts = experts,
            .topk = topk,
            .normalize = sw.router_norm != .scale_only,
            .scale = if (sw.router_norm == .normalize_only or sw.scale_on == .routed_plus_shared) 1.0 else routed_scale,
        },
        .post_scale = @floatCast(total),
    };
}

/// The same plan as moe_glue.zig's Route, for tf_moe_topk_sigmoid (normalize 1, scale 2.0 by default). That kernel ranks by
/// logit + bias, so it matches the model only where the correction bias is zero (see `moeGlueRanksLikeTheModel`); the
/// engine uses `act.MiniMaxAct.route` (tf_mm3_route_warp), which ranks by sigmoid(logit) + bias.
pub fn moeGlueRoute(p: RoutePlan) cuda.moe_glue.Route {
    return .{ .experts = p.route.experts, .topk = p.route.topk, .shared = false, .normalize = p.route.normalize, .scale = p.route.scale };
}

/// moe_glue.cu's pick for one row (float64 emulation of its ranking: logit + bias, lower id on ties), for the host test.
fn moeGluePick(logits: []const f64, bias: []const f64, ids: []usize) void {
    for (ids, 0..) |*id, k| {
        var best: usize = 0;
        var found = false;
        for (logits, bias, 0..) |l, b, i| {
            if (std.mem.indexOfScalar(usize, ids[0..k], i) != null) continue;
            if (!found or l + b > logits[best] + bias[best]) best = i;
            found = true;
        }
        id.* = best;
    }
}

/// Float32 mirror of minimax_m3_act.cuh's swigluoai (same operations, same order), so the formula is checked on the host.
fn swigluoaiF32(g_in: f32, u_in: f32, a: f32, lim: f32, bias: f32, sym: bool) f32 {
    var g = g_in;
    var u = u_in;
    if (lim > 0) {
        g = if (g > lim) lim else if (sym and g < -lim) -lim else g;
        u = if (u > lim) lim else if (u < -lim) -lim else u;
    }
    const sg = g / (1.0 + @exp(-(a * g)));
    return sg * (u + bias);
}

test "switches become kernel arguments, not constants" {
    const d = actParams(.{});
    try std.testing.expectEqual(@as(i32, 2), d.mode());
    try std.testing.expectEqual(@as(f32, 1.0), d.up_bias);
    try std.testing.expectEqual(@as(f32, 1.702), d.alpha);
    try std.testing.expectEqual(@as(f32, 7.0), d.limit);
    const s = actParams(.{ .gate_clamp = .symmetric, .up_plus_one = false });
    try std.testing.expectEqual(@as(i32, 3), s.mode());
    try std.testing.expectEqual(@as(f32, 0.0), s.up_bias);
}

test "configureLayer sets swigluoai and swaps gate and up when w1 is not the gate" {
    var l: Layer = .{ .gate = 1, .up = 2, .down = 3, .gate_k2 = 11, .up_k2 = 12, .down_k2 = 13, .suh_g = 21, .suh_u = 22, .svh_g = 31, .svh_u = 32, .suh_d = 41, .svh_d = 42, .experts = 128, .dims = 6144, .width = 1536, .codebook = .mul1 };
    try std.testing.expectEqual(@as(i32, 0), l.act_mode); // the silu default is untouched until configured
    configureLayer(&l, .{});
    try std.testing.expectEqual(@as(i32, 2), l.act_mode);
    try std.testing.expectEqual(@as(u64, 1), l.gate);
    try std.testing.expectEqual(@as(f32, 7.0), l.limit);
    configureLayer(&l, .{ .w1_is_gate = false, .up_plus_one = false });
    try std.testing.expectEqual(@as(u64, 2), l.gate);
    try std.testing.expectEqual(@as(u64, 1), l.up);
    try std.testing.expectEqual(@as(u64, 12), l.gate_k2);
    try std.testing.expectEqual(@as(u64, 32), l.svh_g);
    try std.testing.expectEqual(@as(u64, 21), l.suh_u);
    try std.testing.expectEqual(@as(f32, 0.0), l.up_bias);
    const pg = pickGateUp(.{ .w1_is_gate = false }, 7, 9);
    try std.testing.expectEqual(@as(u64, 9), pg.gate);
}

test "the route plan is normalize 1, scale 2.0 by default and follows the switches" {
    const p = try routePlan(.{});
    try std.testing.expect(p.route.normalize);
    try std.testing.expectEqual(@as(f32, 2.0), p.route.scale);
    try std.testing.expectEqual(@as(f32, 1.0), p.post_scale);
    const g = moeGlueRoute(p);
    try std.testing.expectEqual(@as(usize, 128), g.experts);
    try std.testing.expectEqual(@as(usize, 4), g.topk);
    try std.testing.expect(!g.shared);
    const both = try routePlan(.{ .scale_on = .routed_plus_shared });
    try std.testing.expectEqual(@as(f32, 1.0), both.route.scale);
    try std.testing.expectEqual(@as(f32, 2.0), both.post_scale);
    const only = try routePlan(.{ .router_norm = .scale_only });
    try std.testing.expect(!only.route.normalize);
    try std.testing.expectEqual(@as(f32, 2.0), only.route.scale);
    try std.testing.expectError(error.Unsupported, routePlan(.{ .router_weights_use_bias = true }));
    try act.routeCheck(p.route, 8);
}

test "moe_glue's logit + bias ranking differs from sigmoid(logit) + bias once the bias is not zero" {
    const logits = [_]f64{ 2.0, 3.0, -1.0, 0.0 };
    var scratch: [4]f64 = undefined;
    var ids: [1]usize = undefined;
    var w: [1]f64 = undefined;
    var model: [1]usize = undefined;
    // no bias: the same pick
    const zero = [_]f64{ 0, 0, 0, 0 };
    mr.routerPick(&logits, &zero, 2.0, .{}, &scratch, &model, &w);
    moeGluePick(&logits, &zero, &ids);
    try std.testing.expectEqual(model[0], ids[0]);
    // bias 0.1 on expert 0: sigmoid(2) + 0.1 = 0.981 beats sigmoid(3) = 0.953, but 2.1 < 3
    const bias = [_]f64{ 0.1, 0, 0, 0 };
    mr.routerPick(&logits, &bias, 2.0, .{}, &scratch, &model, &w);
    moeGluePick(&logits, &bias, &ids);
    try std.testing.expectEqual(@as(usize, 0), model[0]);
    try std.testing.expectEqual(@as(usize, 1), ids[0]);
}

test "the kernel's swigluoai formula in fp32 tracks the float64 reference, NaN and infinities included" {
    const sw: Switches = .{};
    const gs = [_]f32{ -40, -12.5, -3, -0.25, 0, 0.5, 2, 6.99, 7, 7.5, 30, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };
    const us = [_]f32{ -100, -7.5, -7, -1, 0, 0.3, 1, 6.5, 7, 9, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };
    for (gs) |g| for (us) |u| {
        const want = mr.swigluoai(g, u, alpha, limit, sw);
        const got = swigluoaiF32(g, u, alpha, limit, 1.0, false);
        try std.testing.expectEqual(std.math.isNan(want), std.math.isNan(got));
        if (std.math.isNan(want)) continue;
        try std.testing.expectEqual(std.math.isInf(want), std.math.isInf(got));
        if (std.math.isInf(want)) continue;
        const tol = @abs(want) * 1e-5 + 1e-6;
        try std.testing.expect(@abs(@as(f64, got) - want) <= tol);
    };
    // the unverified switches: symmetric clamp and no "+1" change the value, as the reference says
    const sym: Switches = .{ .gate_clamp = .symmetric, .up_plus_one = false };
    try std.testing.expect(@abs(@as(f64, swigluoaiF32(-12.5, 0.3, alpha, limit, 0.0, true)) - mr.swigluoai(-12.5, 0.3, alpha, limit, sym)) < 1e-6);
    // below -limit the sigmoid has already shrunk the gate, so the clamp's side moves the value by little but not by zero
    try std.testing.expect(mr.swigluoai(-12.5, 0.3, alpha, limit, sym) != mr.swigluoai(-12.5, 0.3, alpha, limit, .{ .up_plus_one = false }));
    try std.testing.expect(mr.swigluoai(-12.5, 0.3, alpha, limit, .{ .gate_clamp = .symmetric }) != mr.swigluoai(-12.5, 0.3, alpha, limit, .{}));
}
