//! All MiniMax kernels (act, select, attn, rows, router, proj, pick) through gates 1-4; no controls, no limits.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const act_k = @import("contract_minimax_act.zig");
const sparse_k = @import("contract_minimax_sparse.zig");
const rows_k = @import("contract_minimax_rows.zig");
const proj_k = @import("contract_minimax_proj.zig");
const ds_k = @import("contract_minimax_dspark.zig");

const g = core.contract.gates;

pub fn run(gpu: @import("check.zig").Gpu) !void {
    const check = @import("check.zig");
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);

    var act = try cuda.minimax_m3_act.MiniMaxAct.load(gpu.d);
    defer act.unload();
    var acts: [std.enums.values(act_k.Which).len]act_k.Mm3Act = undefined;
    var n_acts: usize = 0;
    for (std.enums.values(act_k.Which)) |w| {
        if (act_k.control(w) != null) continue;
        acts[n_acts] = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .k = &act, .arena = .init(a) };
        n_acts += 1;
    }
    defer for (acts[0..n_acts]) |*x| x.deinit();
    for (acts[0..n_acts]) |*x| try kernels.append(a, x.kernel());

    var sparse = try cuda.minimax_m3_sparse.Sparse.load(gpu.d);
    defer sparse.unload();
    var selects: [sparse_k.variants.len]sparse_k.Select = undefined;
    for (sparse_k.variants, &selects) |v, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = &sparse, .v = v };
    defer for (&selects) |*x| x.deinit();
    for (&selects) |*x| try kernels.append(a, x.kernel());
    var attns: [sparse_k.cfgs.len]sparse_k.Attn = undefined;
    for (sparse_k.cfgs, &attns) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .s = &sparse };
    defer for (&attns) |*x| x.deinit();
    for (&attns) |*x| try kernels.append(a, x.kernel());

    var rows = try cuda.minimax_m3_rows.Rows.load(gpu.d);
    defer rows.unload();
    var logits: [2]rows_k.Logits = undefined;
    for (std.enums.values(rows_k.Form), &logits) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = f, .k = &rows };
    defer for (&logits) |*x| x.deinit();
    for (&logits) |*x| try kernels.append(a, x.kernel());
    var writers: [rows_k.cfgs.len]rows_k.Writer = undefined;
    for (rows_k.cfgs, &writers) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .k = &rows, .arena = .init(a) };
    defer for (&writers) |*x| x.deinit();
    for (&writers) |*x| try kernels.append(a, x.kernel());

    var proj = try cuda.minimax_m3_proj.Proj.load(gpu.d);
    defer proj.unload();
    var projs: [3]proj_k.Proj = undefined;
    for (std.enums.values(proj_k.Form), &projs) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = f, .k = &proj };
    defer for (&projs) |*x| x.deinit();
    for (&projs) |*x| try kernels.append(a, x.kernel());
    var pick: proj_k.Pick = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &proj };
    defer pick.deinit();
    try kernels.append(a, pick.kernel());

    var dspark = try ds_k.Holder.open(gpu, s, &kernels);
    defer dspark.close();

    var t: g.Tally = .{};
    try g.runAll(a, kernels.items, 0x6d33, &t);
    std.debug.print("SUMMARY mm3-contract {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings\n", .{ t.kernels, t.passed, t.failed, t.known });
    try check.expect(t.failed == 0, "mm3-contract: {d} gate failures", .{t.failed});
}
