//! The bf16 faces of the NVFP4 checkpoint's blocks: hyper-connections, DeltaNet, attention, MoE, FP4 tables.
const std = @import("std");
const core = @import("core");
const lay = @import("cuda_layouts.zig");
const wt = @import("cuda_weights.zig");
const wl = @import("cuda_weights_loader.zig");

const Tensor = core.safetensors.Tensor;
const Loader = wl.Loader;
const Part = Loader.Part;
const Hc = wt.Hc;
const Gdn = wt.Gdn;
const Attn = wt.Attn;
const MoE = wt.MoE;
const Fp4 = wt.Fp4;

// -- the blocks --

/// hc_nvfp4: down (and the block inject rows) stacked, up, the norm's fp32 scale.
pub fn hc(L: *Loader, path: []const u8, name: []const u8, inject: bool) !Hc {
    const c = L.c;
    const sd = c.streams * c.hidden;
    var h: Hc = .{ .inject = inject };
    if (try L.isFp8(try L.nameOf("{s}.input_mix_weight_down", .{name}))) {
        // the fast-fp8 variant: down and up on block FP8, the inject rows bf16 beside them
        var pb: [192]u8 = undefined;
        h.down8 = try L.fp8Face(try std.fmt.bufPrint(&pb, "{s}.down8", .{path}), &.{.{ .name = try L.nameOf("{s}.input_mix_weight_down", .{name}) }});
        if (h.down8.?.n != c.low or h.down8.?.k != sd) return error.UnexpectedTensor;
        if (inject) {
            h.down = try L.b16(try std.fmt.bufPrint(&pb, "{s}.inject.b.weight", .{path}), try L.nameOf("{s}.block_inject_weight", .{name}), c.streams, sd);
            L.out.ours();
        } else h.down = .{ .n = 0, .k = @intCast(sd) };
        h.up8 = try L.fp8Face(try std.fmt.bufPrint(&pb, "{s}.up8", .{path}), &.{.{ .name = try L.nameOf("{s}.input_mix_weight_up", .{name}) }});
        h.up = .{ .n = @intCast(sd), .k = @intCast(c.low) };
        h.scale = try L.cscale(try std.fmt.bufPrint(&pb, "{s}.scale", .{path}), try L.nameOf("{s}.hc_norm.weight", .{name}), sd);
        return h;
    }
    var parts: [2]Part = undefined;
    parts[0] = .{ .t = try L.linear(try L.nameOf("{s}.input_mix_weight_down", .{name}), c.low, sd) };
    if (inject) parts[1] = .{ .t = try L.linear(try L.nameOf("{s}.block_inject_weight", .{name}), c.streams, sd) };
    var pb: [192]u8 = undefined;
    h.down = try L.face(try std.fmt.bufPrint(&pb, "{s}.down.b.weight", .{path}), parts[0 .. @as(usize, 1) + @intFromBool(inject)]);
    h.up = try L.b16(try std.fmt.bufPrint(&pb, "{s}.up.b.weight", .{path}), try L.nameOf("{s}.input_mix_weight_up", .{name}), sd, c.low);
    h.scale = try L.cscale(try std.fmt.bufPrint(&pb, "{s}.scale", .{path}), try L.nameOf("{s}.hc_norm.weight", .{name}), sd);
    return h;
}

/// gdn_nvfp4: the rank's q, k, v, z, b, a rows and conv taps; fp32 A_log, dt_bias; out_proj's rank head columns.
pub fn gdn(L: *Loader, path: []const u8, name: []const u8) !Gdn {
    const c = L.c;
    const r: usize = L.o.rank;
    const kl = c.nk / L.o.world;
    const vl = c.nv / L.o.world;
    const qk = kl * c.dk;
    const vv = vl * c.dv;
    const channels = [3][2]usize{ .{ r * qk, (r + 1) * qk }, .{ c.nk * c.dk + r * qk, c.nk * c.dk + (r + 1) * qk }, .{ 2 * c.nk * c.dk + r * vv, 2 * c.nk * c.dk + (r + 1) * vv } };
    const z = [1][2]usize{.{ r * vv, (r + 1) * vv }};
    const ab = [1][2]usize{.{ r * vl, (r + 1) * vl }};
    var g: Gdn = .{};
    var pb: [192]u8 = undefined;
    g.proj = try L.face(try std.fmt.bufPrint(&pb, "{s}.proj.b.weight", .{path}), &.{
        .{ .t = try L.linear(try L.nameOf("{s}.in_proj_qkv", .{name}), c.convDim(), c.hidden), .rows = &channels },
        .{ .t = try L.linear(try L.nameOf("{s}.in_proj_z", .{name}), c.nv * c.dv, c.hidden), .rows = &z },
        .{ .t = try L.linear(try L.nameOf("{s}.in_proj_b", .{name}), c.nv, c.hidden), .rows = &ab },
        .{ .t = try L.linear(try L.nameOf("{s}.in_proj_a", .{name}), c.nv, c.hidden), .rows = &ab },
    });
    const conv = try L.expect(try L.nameOf("{s}.conv1d.weight", .{name}), .bf16, &.{ c.convDim(), 1, c.conv_kernel });
    const row = c.conv_kernel * 2;
    g.conv = try L.out.begin(try std.fmt.bufPrint(&pb, "{s}.conv", .{path}), "bfloat16", &.{ 2 * qk + vv, c.conv_kernel });
    for (channels) |ch| try L.out.put(conv.bytes[ch[0] * row .. ch[1] * row]);
    try L.out.end();
    g.a_log = try L.widen(try std.fmt.bufPrint(&pb, "{s}.a_log", .{path}), try L.expect(try L.nameOf("{s}.A_log", .{name}), .bf16, &.{c.nv}), r * vl, (r + 1) * vl);
    g.dt_bias = try L.widen(try std.fmt.bufPrint(&pb, "{s}.dt_bias", .{path}), try L.expect(try L.nameOf("{s}.dt_bias", .{name}), .bf16, &.{c.nv}), r * vl, (r + 1) * vl);
    const norm = try L.expect(try L.nameOf("{s}.norm.weight", .{name}), .bf16, &.{c.dv});
    g.norm = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.norm", .{path}), "bfloat16", &.{c.dv}, norm.bytes);
    const out_t = try L.linear(try L.nameOf("{s}.out_proj", .{name}), c.hidden, c.nv * c.dv);
    g.out = try L.face(try std.fmt.bufPrint(&pb, "{s}.out.b.weight", .{path}), &.{.{ .t = out_t, .cols = .{ r * vv, (r + 1) * vv } }});
    return g;
}

/// attention_nvfp4: the rank's q|gate rows, k and v rows, the indexer's rows whole; o_proj's columns; fp32 norms.
pub fn attention(L: *Loader, path: []const u8, name: []const u8) !Attn {
    const c = L.c;
    const r: usize = L.o.rank;
    const hd = c.head_dim;
    const hl = c.heads / L.o.world;
    const kl = c.kv_heads / L.o.world;
    const q = [1][2]usize{.{ r * hl * 2 * hd, (r + 1) * hl * 2 * hd }};
    const kv = [1][2]usize{.{ r * kl * hd, (r + 1) * kl * hd }};
    var a: Attn = .{};
    var pb: [192]u8 = undefined;
    const idx_rows = c.index_heads * c.index_dim + c.index_dim * c.index_kv_heads;
    a.proj = try L.face(try std.fmt.bufPrint(&pb, "{s}.proj.b.weight", .{path}), &.{
        .{ .t = try L.linear(try L.nameOf("{s}.q_proj", .{name}), c.heads * 2 * hd, c.hidden), .rows = &q },
        .{ .t = try L.linear(try L.nameOf("{s}.k_proj", .{name}), c.kv_heads * hd, c.hidden), .rows = &kv },
        .{ .t = try L.linear(try L.nameOf("{s}.v_proj", .{name}), c.kv_heads * hd, c.hidden), .rows = &kv },
        .{ .t = try L.linear(try L.nameOf("{s}.indexer.index_qk_proj", .{name}), idx_rows, c.hidden) },
    });
    a.q_scale = try L.cscaleRope(try std.fmt.bufPrint(&pb, "{s}.q_scale", .{path}), try L.nameOf("{s}.q_norm.weight", .{name}), hd, true);
    a.k_scale = try L.cscaleRope(try std.fmt.bufPrint(&pb, "{s}.k_scale", .{path}), try L.nameOf("{s}.k_norm.weight", .{name}), hd, true);
    a.iq_scale = try L.cscaleRope(try std.fmt.bufPrint(&pb, "{s}.iq_scale", .{path}), try L.nameOf("{s}.indexer.q_layernorm.weight", .{name}), c.index_dim, true);
    a.ik_scale = try L.cscaleRope(try std.fmt.bufPrint(&pb, "{s}.ik_scale", .{path}), try L.nameOf("{s}.indexer.k_layernorm.weight", .{name}), c.index_dim, true);
    const o_t = try L.linear(try L.nameOf("{s}.o_proj", .{name}), c.hidden, c.heads * hd);
    a.o = try L.face(try std.fmt.bufPrint(&pb, "{s}.o.b.weight", .{path}), &.{.{ .t = o_t, .cols = .{ r * hl * hd, (r + 1) * hl * hd } }});
    return a;
}

/// moe_nvfp4: router with shared gate row, shared FP4 expert at half width, NVFP4 or re-quantized (MTP) experts.
pub fn moe(L: *Loader, path: []const u8, name: []const u8, mtp: bool) !MoE {
    const c = L.c;
    const r: usize = L.o.rank;
    const world: usize = L.o.world;
    const e = c.experts;
    const d = c.hidden;
    const lo = r * c.moe_width / world;
    const hi = (r + 1) * c.moe_width / world;
    const gs = c.nvfp4_group;
    const ni = hi - lo;
    var m: MoE = .{};
    var pb: [192]u8 = undefined;

    const router = try L.linear(try L.nameOf("{s}.gate", .{name}), e, d);
    const sgate = try L.linear(try L.nameOf("{s}.shared_expert_gate", .{name}), 1, d);
    m.router = (try L.face(try std.fmt.bufPrint(&pb, "{s}.router", .{path}), &.{ .{ .t = router }, .{ .t = sgate } })).weight;

    // shared expert: gate/up rows [lo, hi), down's columns; MTP uses its own width (INT4-AutoRound's main is wider)
    const sw = if (mtp and c.int4ar()) (try L.get(try L.nameOf("{s}.shared_expert.gate_proj.weight", .{name}))).dim(0) else c.shared_width;
    const sg = try L.linear(try L.nameOf("{s}.shared_expert.gate_proj", .{name}), sw, d);
    const su = try L.linear(try L.nameOf("{s}.shared_expert.up_proj", .{name}), sw, d);
    const sdn = try L.linear(try L.nameOf("{s}.shared_expert.down_proj", .{name}), d, sw);
    m.shared.gu = try L.fp4Table(try std.fmt.bufPrint(&pb, "{s}.experts.shared.gu", .{path}), &.{ sg, su }, lo, hi, null);
    m.shared.down = try L.fp4Table(try std.fmt.bufPrint(&pb, "{s}.experts.shared.down", .{path}), &.{sdn}, 0, d, .{ lo / gs * gs, hi / gs * gs });

    m.routed = .{ .count = e, .width = @intCast(ni), .dims = d };
    var b2: [192]u8 = undefined;
    if (mtp) {
        try L.mtpExperts(path, name, &m.routed, lo, hi);
    } else {
        if (L.has(try L.nameOf("{s}{s}.experts.0.gate_proj.weight_scale_inv", .{ L.prefix, name }))) return error.UnsupportedQuantization;
        const up = try L.packRouted(try std.fmt.bufPrint(&b2, "{s}.experts.routed_experts.up", .{path}), name, &.{ "gate_proj", "up_proj" }, ni, d, lo, false);
        m.routed.up = up.blocks;
        const down = try L.packRouted(try std.fmt.bufPrint(&b2, "{s}.experts.routed_experts.down", .{path}), name, &.{"down_proj"}, d, ni, lo, true);
        m.routed.down = down.blocks;
        m.routed.up_scale = try L.out.whole(try std.fmt.bufPrint(&b2, "{s}.experts.routed_experts.up_scale", .{path}), "float32", &.{ e, 2 }, std.mem.sliceAsBytes(up.scale2));
        m.routed.down_scale = try L.out.whole(try std.fmt.bufPrint(&b2, "{s}.experts.routed_experts.down_scale", .{path}), "float32", &.{ e, 1 }, std.mem.sliceAsBytes(down.scale2));
        L.gpa.free(up.scale2);
        L.gpa.free(down.scale2);
    }
    return m;
}

/// fp4_from_bf16 of rows [lo, hi) of each part stacked (columns `cols` when given): bf16 bits tiled, unit scales.
pub fn fp4Table(L: *Loader, path: []const u8, parts: []const Tensor, lo: usize, hi: usize, cols: ?[2]usize) !Fp4 {
    const k_all = parts[0].dim(1);
    const k0 = if (cols) |cc| cc[0] else 0;
    const k = if (cols) |cc| cc[1] - cc[0] else k_all;
    const n = parts.len * (hi - lo);
    // gather the rows (whole when no column range) into one grid, then tile
    const grid = try L.gpa.alloc(u16, n * k);
    defer L.gpa.free(grid);
    for (parts, 0..) |p, i| for (lo..hi, 0..) |row, j| {
        const src = std.mem.bytesAsSlice(u16, p.bytes[(row * k_all + k0) * 2 ..][0 .. k * 2]);
        for (grid[(i * (hi - lo) + j) * k ..][0..k], 0..) |*g, x| g.* = std.mem.littleToNative(u16, @as(*align(1) const u16, @ptrCast(&src[x])).*);
    };
    const tiled = std.mem.bytesAsSlice(u16, try L.staging(n * k * 2));
    lay.tileBits(grid, k, 0, n, k, @alignCast(tiled));
    var pb: [192]u8 = undefined;
    var f: Fp4 = .{ .n = @intCast(n), .k = @intCast(k) };
    f.weight = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.weight", .{path}), "uint16", &.{ n / 64, k / 64, 64, 64 }, L.host.items);
    const ones = std.mem.bytesAsSlice(f32, try L.staging(@max(k / 16, 1) * n * 4));
    @memset(ones, 1.0);
    f.scale = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.scale", .{path}), "float32", &.{ k / 16, n }, L.host.items);
    f.scale2 = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.scale2", .{path}), "float32", &.{n}, L.host.items[0 .. n * 4]);
    return f;
}
