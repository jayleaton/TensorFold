//! A rank's weights as the kernels take them, described from load.Plan and each slot's device base; host only.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const load = @import("load.zig");
const names = @import("names.zig");
const ffn = @import("ffn.zig");
const Config = @import("config.zig").Config;
const exl3 = cuda.exl3;
const st = core.safetensors;

pub const Error = error{ MissingTensor, BadDtype, BadShape, BadLayer, OutOfMemory };

pub const Dense = struct { gate: exl3.Weight, up: exl3.Weight, down: exl3.Weight };

pub const Moe = struct {
    router: u64, // fp32 [experts, hidden] (an f16 or bf16 checkpoint's widened by the loader)
    bias: u64, // fp32 [experts], e_score_correction_bias (widened like the router)
    experts: cuda.exl3_experts.Layer,
    shared: Dense,
};

pub const Ffn = union(enum) { dense: Dense, moe: Moe };

/// One layer. Norm weights are fp32 `1 + w` (the Gemma fold ran in the streamer); index projections are fp16.
pub const Layer = struct {
    input_ln: u64 = 0,
    post_ln: u64 = 0,
    q_norm: u64 = 0,
    k_norm: u64 = 0,
    index_q_norm: u64 = 0,
    index_k_norm: u64 = 0,
    index_q_proj: u64 = 0, // fp16 [this rank's index heads * index dim, hidden]
    index_k_proj: u64 = 0, // fp16 [index dim, hidden], replicated
    q: exl3.Weight,
    k: exl3.Weight,
    v: exl3.Weight,
    o: exl3.Weight,
    ffn: Ffn,
};

pub const Weights = struct {
    embed: u64 = 0, // bf16 [vocab, hidden], replicated
    final_norm: u64 = 0,
    head: exl3.Weight,
    layers: []Layer,
};

/// One routed layer's per-expert tables as the experts kernel reads them, before they are on the device.
pub const Tables = struct {
    layer: u16,
    gate: []u64, // w1: trellis pointer of each expert
    up: []u64, // w3
    down: []u64, // w2
    gate_k2: []const i32, // 2 * bits of each expert's trellis
    up_k2: []const i32,
    down_k2: []const i32,
};

/// The device pointers of the six tables once uploaded, in `Tables` order.
pub const TablePtrs = struct { gate: u64, up: u64, down: u64, gate_k2: u64, up_k2: u64, down_k2: u64 };

pub const Described = struct { weights: Weights, tables: []Tables };

/// The kernel's record of one dense matrix: slot base plus the matrix' byte offsets, strides for its rank-local shape.
pub fn weight(m: load.Matrix, bases: []const u64) exl3.Weight {
    const s = exl3.strides(m.k, m.n, m.halves, false);
    return .{
        .trellis = bases[m.trellis] + m.trellis_off,
        .suh = bases[m.suh] + m.suh_off,
        .svh = bases[m.svh] + m.svh_off,
        .k = m.k,
        .n = m.n,
        .halves = m.halves,
        .codebook = .mul1,
        .stride_k = s.k,
        .stride_nb = s.nb,
    };
}

fn matrix(plan: *const load.Plan, bases: []const u64, proj: names.Proj, layer: u32) Error!exl3.Weight {
    const m = plan.matrix(proj, layer, 0) orelse return error.MissingTensor;
    return weight(m.*, bases);
}

/// The half-bit width every expert of all three projections shares, else 0 (layer 17's 4-bit w1/w3 with a 3-bit w2).
pub fn uniformK2(m: *const load.MoeLayer) u32 {
    var width: ?u32 = null;
    for ([_]*const load.MoeProj{ &m.w1, &m.w3, &m.w2 }) |p| {
        var seen: ?u32 = null;
        for (p.histogram, 0..) |count, k2| {
            if (count == 0) continue;
            if (seen != null) return 0;
            seen = @intCast(k2);
        }
        const only = seen orelse return 0;
        if (width != null and width.? != only) return 0;
        width = only;
    }
    return width orelse 0;
}

fn tables(a: std.mem.Allocator, m: *const load.MoeLayer, bases: []const u64) Error!Tables {
    const n = m.w1.halves.len;
    var out: Tables = .{ .layer = m.layer, .gate = undefined, .up = undefined, .down = undefined, .gate_k2 = m.w1.halves, .up_k2 = m.w3.halves, .down_k2 = m.w2.halves };
    if (m.w3.halves.len != n or m.w2.halves.len != n) return error.BadShape;
    inline for (.{ .{ "gate", &m.w1 }, .{ "up", &m.w3 }, .{ "down", &m.w2 } }) |pair| {
        const p: *const load.MoeProj = pair[1];
        const ptrs = try a.alloc(u64, n);
        for (ptrs, p.trellis_off) |*d, off| d.* = bases[p.trellis_slot] + off;
        @field(out, pair[0]) = ptrs;
    }
    return out;
}

fn moeLayer(a: std.mem.Allocator, plan: *const load.Plan, c: *const Config, bases: []const u64, l: u32, pending: *std.ArrayList(Tables)) Error!Moe {
    const m = plan.moeLayer(l) orelse return error.MissingTensor;
    if (m.w1.k != m.w3.k or m.w1.n != m.w3.n or m.w2.k != m.w1.n or m.w2.n != m.w1.k) return error.BadShape;
    try pending.append(a, try tables(a, m, bases));
    return .{
        .router = 0,
        .bias = 0,
        .shared = .{
            .gate = try matrix(plan, bases, .shared_gate, l),
            .up = try matrix(plan, bases, .shared_up, l),
            .down = try matrix(plan, bases, .shared_down, l),
        },
        .experts = .{
            .gate = 0,
            .up = 0,
            .down = 0,
            .gate_k2 = 0,
            .up_k2 = 0,
            .down_k2 = 0,
            .suh_g = bases[m.w1.suh_slot],
            .suh_u = bases[m.w3.suh_slot],
            .svh_g = bases[m.w1.svh_slot],
            .svh_u = bases[m.w3.svh_slot],
            .suh_d = bases[m.w2.suh_slot],
            .svh_d = bases[m.w2.svh_slot],
            .experts = c.experts,
            .dims = m.w1.k,
            .width = m.w1.n,
            .codebook = .mul1,
            .uniform_k2 = uniformK2(m),
            .shared_input = false, // decision M7: no check of the gate/up scales yet
        },
    };
}

/// `bases[slot]` is slot's device address. `layers` is how many layers the plan holds (a thin model: the first ones).
pub fn describe(a: std.mem.Allocator, plan: *const load.Plan, c: *const Config, bases: []const u64, layers: usize) Error!Described {
    if (layers > c.layers) return error.BadLayer;
    const ids = try a.alloc(u16, layers);
    for (ids, 0..) |*id, i| id.* = @intCast(i);
    return describeLayers(a, plan, c, bases, ids);
}

/// Layers `ids` (ascending) of a plan cut to them; `layers` has max(ids) + 1 entries, the others undefined and unread.
pub fn describeLayers(a: std.mem.Allocator, plan: *const load.Plan, c: *const Config, bases: []const u64, ids: []const u16) Error!Described {
    if (bases.len != plan.slots.len) return error.BadLayer;
    for (ids, 0..) |l, i| if (l >= c.layers or (i > 0 and l <= ids[i - 1])) return error.BadLayer;
    const layers: usize = if (ids.len == 0) 0 else @as(usize, ids[ids.len - 1]) + 1;
    const out = try a.alloc(Layer, layers);
    var pending: std.ArrayList(Tables) = .empty;
    for (ids) |id| {
        const l: u32 = id;
        const o = &out[id];
        o.* = .{
            .q = try matrix(plan, bases, .q, l),
            .k = try matrix(plan, bases, .k, l),
            .v = try matrix(plan, bases, .v, l),
            .o = try matrix(plan, bases, .o, l),
            .ffn = if (c.isMoe(l)) .{ .moe = try moeLayer(a, plan, c, bases, l, &pending) } else .{ .dense = .{
                .gate = try matrix(plan, bases, .dense_gate, l),
                .up = try matrix(plan, bases, .dense_up, l),
                .down = try matrix(plan, bases, .dense_down, l),
            } },
        };
    }
    var w: Weights = .{ .head = try matrix(plan, bases, .head, 0), .layers = out };
    var seen_embed = false;
    for (plan.plain) |p| {
        const at = bases[p.slot];
        try checkPlain(p);
        switch (p.role) {
            .embed => {
                w.embed = at;
                seen_embed = true;
            },
            .final_norm => w.final_norm = at,
            else => {
                if (std.mem.indexOfScalar(u16, ids, p.layer) == null) continue;
                const o = &out[p.layer];
                switch (p.role) {
                    .input_ln => o.input_ln = at,
                    .post_ln => o.post_ln = at,
                    .q_norm => o.q_norm = at,
                    .k_norm => o.k_norm = at,
                    .index_q_norm => o.index_q_norm = at,
                    .index_k_norm => o.index_k_norm = at,
                    .index_q_proj => o.index_q_proj = at,
                    .index_k_proj => o.index_k_proj = at,
                    .router => if (o.ffn == .moe) {
                        o.ffn.moe.router = at;
                    } else return error.BadLayer,
                    .router_bias => if (o.ffn == .moe) {
                        o.ffn.moe.bias = at;
                    } else return error.BadLayer,
                    .embed, .final_norm => unreachable,
                }
            },
        }
    }
    if (!seen_embed) return error.MissingTensor;
    return .{ .weights = w, .tables = try pending.toOwnedSlice(a) };
}

/// The dtype each plain role must hold on the device: norms folded to fp32, index projections fp16, router fp32.
pub fn checkPlain(p: load.PlainTensor) Error!void {
    const want: st.DType = switch (p.role) {
        .embed => .bf16,
        .index_q_proj, .index_k_proj => .f16,
        .router, .router_bias => .f32,
        else => .f32,
    };
    if (p.dtype != want) return error.BadDtype;
}

/// Sets layer `t.layer`'s table pointers and applies decision M5's switches (which may swap gate and up).
pub fn attachTables(w: *Weights, t: Tables, ptrs: TablePtrs, sw: ffn.Switches) void {
    const e = &w.layers[t.layer].ffn.moe.experts;
    e.gate = ptrs.gate;
    e.up = ptrs.up;
    e.down = ptrs.down;
    e.gate_k2 = ptrs.gate_k2;
    e.up_k2 = ptrs.up_k2;
    e.down_k2 = ptrs.down_k2;
    ffn.configureLayer(e, sw);
}

/// One buffer per routed layer holds its six tables: three int64 pointer tables, then three int32 width tables.
pub fn packTables(a: std.mem.Allocator, t: Tables) Error![]u8 {
    const n = t.gate.len;
    const out = try a.alloc(u8, n * (3 * 8 + 3 * 4));
    var at: usize = 0;
    for ([_][]const u64{ t.gate, t.up, t.down }) |p| for (p) |v| {
        std.mem.writeInt(u64, out[at..][0..8], v, .little);
        at += 8;
    };
    for ([_][]const i32{ t.gate_k2, t.up_k2, t.down_k2 }) |p| for (p) |v| {
        std.mem.writeInt(i32, out[at..][0..4], v, .little);
        at += 4;
    };
    return out;
}

/// Where the six tables of `packTables` sit when the buffer starts at device address `base`, for `n` experts.
pub fn tablePtrs(base: u64, n: usize) TablePtrs {
    return .{ .gate = base, .up = base + 8 * n, .down = base + 16 * n, .gate_k2 = base + 24 * n, .up_k2 = base + 28 * n, .down_k2 = base + 32 * n };
}

test {
    _ = @import("weights_test.zig");
}
