//! The MiniMax-M3 writers registered into kernel-contract's list, at the same two places and in the same order.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Gpu = @import("check.zig").Gpu;
const act_k = @import("contract_minimax_act.zig");
const sparse_k = @import("contract_minimax_sparse.zig");
const rows_k = @import("contract_minimax_rows.zig");
const proj_k = @import("contract_minimax_proj.zig");
const ds_k = @import("contract_minimax_dspark.zig");

const g = core.contract.gates;
const act_ns = cuda.minimax_m3_act;

/// The act module and its writers (controls included); the writers point into it, so it lives on the heap.
pub const Act = struct {
    gpa: std.mem.Allocator,
    k: act_ns.MiniMaxAct,
    acts: [std.enums.values(act_k.Which).len]act_k.Mm3Act,

    pub fn open(gpu: Gpu, s: cuda.Stream, ctl_mod: cuda.Module, list: *std.ArrayList(g.Kernel)) !*Act {
        const a = gpu.gpa;
        const r = try a.create(Act);
        errdefer a.destroy(r);
        r.gpa = a;
        r.k = try act_ns.MiniMaxAct.load(gpu.d);
        errdefer r.k.unload();
        for (std.enums.values(act_k.Which), &r.acts) |w, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .k = &r.k, .arena = .init(a), .ctl = if (act_k.control(w) != null) try ctl_mod.function("tf_ctl_mm3_act") else null };
        errdefer for (&r.acts) |*x| x.deinit();
        for (&r.acts) |*x| try list.append(a, x.kernel());
        return r;
    }

    pub fn close(r: *Act) void {
        for (&r.acts) |*x| x.deinit();
        r.k.unload();
        r.gpa.destroy(r);
    }
};

/// The sparse, rows, proj, pick and dspark writers; the writers point into it, so it lives on the heap.
pub const Rest = struct {
    gpa: std.mem.Allocator,
    sparse: cuda.minimax_m3_sparse.Sparse,
    selects: [sparse_k.variants.len]sparse_k.Select,
    attns: [sparse_k.cfgs.len]sparse_k.Attn,
    rows: cuda.minimax_m3_rows.Rows,
    logits: [2]rows_k.Logits,
    writers: [rows_k.cfgs.len]rows_k.Writer,
    proj: cuda.minimax_m3_proj.Proj,
    projs: [3]proj_k.Proj,
    pick: proj_k.Pick,
    ds: ds_k.Holder,

    pub fn open(gpu: Gpu, s: cuda.Stream, list: *std.ArrayList(g.Kernel)) !*Rest {
        const a = gpu.gpa;
        const r = try a.create(Rest);
        errdefer a.destroy(r);
        r.gpa = a;
        r.sparse = try cuda.minimax_m3_sparse.Sparse.load(gpu.d);
        errdefer r.sparse.unload();
        for (sparse_k.variants, &r.selects) |v, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = &r.sparse, .v = v };
        errdefer for (&r.selects) |*x| x.deinit();
        for (&r.selects) |*x| try list.append(a, x.kernel());
        for (sparse_k.cfgs, &r.attns) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .s = &r.sparse };
        errdefer for (&r.attns) |*x| x.deinit();
        for (&r.attns) |*x| try list.append(a, x.kernel());
        r.rows = try cuda.minimax_m3_rows.Rows.load(gpu.d);
        errdefer r.rows.unload();
        for (std.enums.values(rows_k.Form), &r.logits) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = f, .k = &r.rows };
        errdefer for (&r.logits) |*x| x.deinit();
        for (&r.logits) |*x| try list.append(a, x.kernel());
        for (rows_k.cfgs, &r.writers) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .k = &r.rows, .arena = .init(a) };
        errdefer for (&r.writers) |*x| x.deinit();
        for (&r.writers) |*x| try list.append(a, x.kernel());
        r.proj = try cuda.minimax_m3_proj.Proj.load(gpu.d);
        errdefer r.proj.unload();
        for (std.enums.values(proj_k.Form), &r.projs) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = f, .k = &r.proj };
        errdefer for (&r.projs) |*x| x.deinit();
        for (&r.projs) |*x| try list.append(a, x.kernel());
        r.pick = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &r.proj };
        errdefer r.pick.deinit();
        try list.append(a, r.pick.kernel());
        r.ds = try ds_k.Holder.open(gpu, s, list);
        return r;
    }

    pub fn close(r: *Rest) void {
        r.ds.close();
        r.pick.deinit();
        for (&r.projs) |*x| x.deinit();
        r.proj.unload();
        for (&r.writers) |*x| x.deinit();
        for (&r.logits) |*x| x.deinit();
        r.rows.unload();
        for (&r.attns) |*x| x.deinit();
        for (&r.selects) |*x| x.deinit();
        r.sparse.unload();
        r.gpa.destroy(r);
    }
};
