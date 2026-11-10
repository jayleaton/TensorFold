//! Flash Next MoE block: router, routed NVFP4 or INT4 experts and the shared expert, ranks gathered.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const moep = @import("cuda_moe_prompt.zig");
const fp8 = @import("cuda_fp8.zig");
const int4 = @import("cuda_int4.zig");
const W = @import("cuda_weights.zig");
const fwd = @import("cuda_forward.zig");
const Forward = fwd.Forward;
const Bufs = fwd.Bufs;

/// moe_block (nvfp4_moe.moe): router, routed and shared SwiGLU and down; one GPU pending, two ranks gathered.
pub fn moeBlock(f: *Forward, m: *const W.MoE, x: *const Bufs, R: usize, mtp: bool) !tri.Branch {
    const b = &x.b;
    const g = f.g;
    const D = g.hidden;
    const E = g.experts;
    const top = g.topKFor(mtp);
    const slots = top + 1;
    const es: usize = if (x.y_f32) 4 else 2;
    try f.t.router(b.mixed, D, m.router, b.moe_logits, R, D, E + 1);
    try f.mark(.router);
    try f.t.topkRows(b.moe_logits, b.moe_pick, b.moe_wts, R, E, top);
    if (f.count_experts and f.prof != null and !b.prefill) try f.countExperts(b.moe_pick, R, slots, E);
    if (f.splitOf(x, R)) |q| if (f.cs != null and f.overlap & 2 != 0) {
        // the peer's rows first: their experts and partial, sent while this rank's own rows run
        const yrow = slots * D * es;
        try f.moeRows(m, x, q.po, @min(q.rh, R - q.po), top);
        try f.t.moePartial(b.moe_y + q.po * yrow, x.y_f32, b.moe_wts + q.po * slots * 4, b.part_moe, q.rh, D, slots);
        try f.mark(.moe_partial);
        try f.exchangeAsync(b.part_moe, b.g_moe + (1 - q.rank) * q.rh * D * 4, q.rh * D, .f32);
        try f.moeRows(m, x, q.o, @min(q.rh, R - q.o), top);
        try f.t.moePartial(b.moe_y + q.o * yrow, x.y_f32, b.moe_wts + q.o * slots * 4, b.g_moe + q.rank * q.rh * D * 4, q.rh, D, slots);
        try f.mark(.moe_partial);
        try f.exchangeWait();
        try f.mark(.moe_gather);
        return .{ .ranks = .{ .part = b.g_moe, .world = 2 } };
    };
    try f.moeRows(m, x, 0, R, top);
    if (f.splitOf(x, R)) |q| {
        // the peer's rows' partial first, sent while this rank's own rows' partial runs
        const yrow = slots * D * es;
        try f.t.moePartial(b.moe_y + q.po * yrow, x.y_f32, b.moe_wts + q.po * slots * 4, b.part_moe, q.rh, D, slots);
        try f.exchangeAsync(b.part_moe, b.g_moe + (1 - q.rank) * q.rh * D * 4, q.rh * D, .f32);
        try f.t.moePartial(b.moe_y + q.o * yrow, x.y_f32, b.moe_wts + q.o * slots * 4, b.g_moe + q.rank * q.rh * D * 4, q.rh, D, slots);
        try f.mark(.moe_partial);
        try f.exchangeWait();
        try f.mark(.moe_gather);
        return .{ .ranks = .{ .part = b.g_moe, .world = 2 } };
    }
    if (f.comm) |cm| {
        try f.t.moePartial(b.moe_y, x.y_f32, b.moe_wts, b.part_moe, R, D, slots);
        try f.mark(.moe_partial);
        try cm.gatherPartialsThen(b.part_moe, b.g_moe, R, D, f.s.handle, f.pf_moe);
        try f.mark(.moe_gather);
        return .{ .ranks = .{ .part = b.g_moe, .world = cm.world } };
    }
    return .{ .moe = .{ .y = b.moe_y, .y_f32 = x.y_f32, .wts = b.moe_wts, .slots = slots } };
}

/// TF_FLASHNEXT_COUNT_EXPERTS: this layer's distinct routed experts over its R rows' picks (the shared slot apart).
pub fn countExperts(f: *Forward, picks: u64, R: usize, slots: usize, E: usize) !void {
    const host = try f.gpa.alloc(i32, R * slots);
    defer f.gpa.free(host);
    try f.s.synchronize();
    const buf: cuda.DeviceBuffer = .{ .d = f.d, .ptr = picks, .len = host.len * 4 };
    try buf.download(0, std.mem.sliceAsBytes(host));
    var seen = try std.DynamicBitSet.initEmpty(f.gpa, E + 1);
    defer seen.deinit();
    for (host) |x| if (x >= 0 and x < E) seen.set(@intCast(x));
    const c = &f.experts_seen[@intFromBool(f.prof.?.mtp)];
    c.layers += 1;
    c.distinct += seen.count();
    c.rows += R;
}

/// The experts, shared expert and slots of rows [a0, a0 + n), `top` routed experts a row.
pub fn moeRows(f: *Forward, m: *const W.MoE, x: *const Bufs, a0: usize, n: usize, top: usize) !void {
    if (m.int4) |ex4| return f.moeRowsInt4(m, ex4, x, a0, n, top);
    const b = &x.b;
    const g = f.g;
    const D = g.hidden;
    const E = g.experts;
    const slots = top + 1;
    const ni = g.moe_width;
    const es: usize = if (x.y_f32) 4 else 2;
    const R = n;
    const pick = b.moe_pick + a0 * slots * 4;
    const xin = b.mixed + a0 * D * 2;
    const act = b.moe_act + a0 * slots * ni * 2;
    const y = b.moe_y + a0 * slots * D * es;
    const plan: kern.Plan = .{ .members = b.plan_members, .items = b.plan_items, .counts = b.plan_counts, .rank = b.plan_rank, .hist = b.plan_hist };
    // prompt chunks: gate/up and down on cuda_moe_prompt's items from their row counts (the same bits)
    const gu_px: ?*const moep.Prompt4 = if (f.px) |q| (if (R >= @max(moep.min_rows, q.gu_rows)) q else null) else null;
    const dn_px: ?*const moep.Prompt4 = if (f.px) |q| (if (R >= @max(moep.min_rows, q.down_rows)) q else null) else null;
    if (gu_px) |q| try q.plan(f.ops, pick, R * slots, E + 1, plan) else try f.ops.plan(pick, R * slots, E + 1, kern.plan_tile, plan);
    const ex: kern.Experts4 = .{ .up = m.routed.up, .down = m.routed.down, .up_scale = m.routed.up_scale, .down_scale = m.routed.down_scale, .width = m.routed.width, .dims = m.routed.dims };
    if (ex.width != ni or ex.dims != D) return error.ExpertShape;
    const skip: c_int = @intCast(E);
    try f.mark(.topk_plan);
    const gu = m.shared.gu;
    const dn = m.shared.down;
    const shared_act = act + top * ni * 2;
    if (f.side) |sd| {
        // the shared expert on the side stream: its slot of moe_act and moe_y, which the routed kernels skip
        try sd.fork.record(f.s);
        try sd.s.wait(sd.fork);
        var t2 = f.t;
        t2.s = sd.s;
        var th2 = f.th;
        th2.s = sd.s;
        try f.fp4On(t2, th2, xin, D, .{ .weight = gu.weight, .scale = gu.scale, .scale2 = gu.scale2 }, f.sc.shared_g, false, R, gu.n, gu.k);
        try th2.sharedSwiglu(f.sc.shared_g, shared_act, R, ni, slots * ni);
        try f.fp4On(t2, th2, shared_act, slots * ni, .{ .weight = dn.weight, .scale = dn.scale, .scale2 = dn.scale2 }, f.sc.shared_y, x.y_f32, R, dn.n, dn.k);
        try th2.slotCopy(f.sc.shared_y, D * es, y + top * D * es, slots * D * es, D * es, R);
        try sd.join.record(sd.s);
    }
    if (gu_px) |q| try q.gateUp(f.s, xin, D, ex, plan, slots, E + 1, act, R, skip) else try f.ops.nvfp4GateUp(xin, D, ex, plan, slots, E + 1, act, R, skip);
    try f.mark(.experts_gate_up);
    if (f.side == null) {
        try f.fp4At(.shared_gate_up, xin, D, .{ .weight = gu.weight, .scale = gu.scale, .scale2 = gu.scale2 }, f.sc.shared_g, false, R, gu.n, gu.k);
        try f.th.sharedSwiglu(f.sc.shared_g, shared_act, R, ni, slots * ni);
        try f.mark(.shared_swiglu);
    }
    if ((gu_px == null) != (dn_px == null)) {
        if (dn_px) |q| try q.plan(f.ops, pick, R * slots, E + 1, plan) else try f.ops.plan(pick, R * slots, E + 1, kern.plan_tile, plan);
    }
    if (dn_px) |q| try q.down(f.s, act, ni, ex, plan, slots, E + 1, y, x.y_f32, R, skip) else try f.ops.nvfp4Down(act, ni, ex, plan, slots, E + 1, y, x.y_f32, R, skip);
    try f.mark(.experts_down);
    if (f.side) |sd| {
        try f.s.wait(sd.join);
        try f.mark(.shared_down);
    } else {
        try f.fp4At(.shared_down, shared_act, slots * ni, .{ .weight = dn.weight, .scale = dn.scale, .scale2 = dn.scale2 }, f.sc.shared_y, x.y_f32, R, dn.n, dn.k);
        try f.th.slotCopy(f.sc.shared_y, D * es, y + top * D * es, slots * D * es, D * es, R);
        try f.mark(.slot_copy);
    }
}

/// INT4-AutoRound experts of rows [a0, a0 + n): int4 gate/up and down, block-FP8 shared expert into its slot.
pub fn moeRowsInt4(f: *Forward, m: *const W.MoE, ex4: int4.Experts, x: *const Bufs, a0: usize, n: usize, top: usize) !void {
    const b = &x.b;
    const g = f.g;
    const D = g.hidden;
    const E = g.experts;
    const slots = top + 1;
    const ni = g.moe_width;
    const es: usize = if (x.y_f32) 4 else 2;
    const R = n;
    const pick = b.moe_pick + a0 * slots * 4;
    const xin = b.mixed + a0 * D * 2;
    const act = b.moe_act + a0 * slots * ni * 2;
    const y = b.moe_y + a0 * slots * D * es;
    const plan: kern.Plan = .{ .members = b.plan_members, .items = b.plan_items, .counts = b.plan_counts, .rank = b.plan_rank, .hist = b.plan_hist };
    const k4 = f.k4 orelse return error.NoInt4Kernels;
    const k8 = f.k8 orelse return error.NoFp8Kernels;
    if (ex4.width != ni or ex4.dims != D) return error.ExpertShape;
    const skip: c_int = @intCast(E);
    // items of 16 pairs (int4.tileFor; the prompt tile is opt-in, the same bits)
    const tile = int4.tileFor(R * slots);
    try f.ops.plan(pick, R * slots, E + 1, tile, plan);
    try f.mark(.topk_plan);
    const sh = m.shared8 orelse return error.NoSharedExpert;
    if (f.side) |sd| {
        // block-FP8 shared expert on the side stream (own scratch and moe_y slot): same bits, beside routed
        try sd.fork.record(f.s);
        try sd.s.wait(sd.fork);
        var th2 = f.th;
        th2.s = sd.s;
        try fp8.matmul(k8, sd.s, xin, D, sh.gu, f.sc.sh_g, false, R);
        try th2.sharedSwiglu(f.sc.sh_g, f.sc.sh_a, R, sh.width, sh.width);
        try fp8.matmul(k8, sd.s, f.sc.sh_a, sh.width, sh.down, f.sc.shared_y, x.y_f32, R);
        try th2.slotCopy(f.sc.shared_y, D * es, y + top * D * es, slots * D * es, D * es, R);
        try sd.join.record(sd.s);
        try int4.gateUp(k4, f.s, xin, D, ex4, plan, slots, E + 1, act, R, skip, tile);
        try f.mark(.experts_gate_up);
        try f.int4Down(k4, pick, act, ex4, plan, slots, E, y, x.y_f32, R, skip, tile);
        try f.mark(.experts_down);
        try f.s.wait(sd.join);
        try f.mark(.shared_down);
        return;
    }
    try int4.gateUp(k4, f.s, xin, D, ex4, plan, slots, E + 1, act, R, skip, tile);
    try f.mark(.experts_gate_up);
    try fp8.matmul(k8, f.s, xin, D, sh.gu, f.sc.sh_g, false, R);
    try f.th.sharedSwiglu(f.sc.sh_g, f.sc.sh_a, R, sh.width, sh.width);
    try f.mark(.shared_swiglu);
    try f.int4Down(k4, pick, act, ex4, plan, slots, E, y, x.y_f32, R, skip, tile);
    try f.mark(.experts_down);
    try fp8.matmul(k8, f.s, f.sc.sh_a, sh.width, sh.down, f.sc.shared_y, x.y_f32, R);
    try f.mark(.shared_down);
    try f.th.slotCopy(f.sc.shared_y, D * es, y + top * D * es, slots * D * es, D * es, R);
    try f.mark(.slot_copy);
}

/// The routed int4 down: prompt calls on int4_prompt_kernel from a 64-pair plan (same bytes), else int4_kernel.
pub fn int4Down(f: *Forward, k4: *const int4.Kernels, pick: u64, act: u64, ex4: int4.Experts, plan: kern.Plan, slots: usize, E: usize, y: u64, y_f32: bool, R: usize, skip: c_int, tile: usize) !void {
    if (int4.promptDown(R * slots)) {
        try f.ops.plan(pick, R * slots, E + 1, int4.prompt_down_tile, plan);
        return int4.downPrompt(k4, f.s, act, ex4.width, ex4, plan, slots, E + 1, y, y_f32, R, skip);
    }
    return int4.down(k4, f.s, act, ex4.width, ex4, plan, slots, E + 1, y, y_f32, R, skip, tile);
}
