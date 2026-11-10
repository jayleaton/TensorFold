//! INT4-AutoRound's block-FP8 faces: DeltaNet, attention and the MoE with its block-FP8 shared expert.
const std = @import("std");
const core = @import("core");
const fp8 = @import("cuda_fp8.zig");
const wt = @import("cuda_weights.zig");
const wl = @import("cuda_weights_loader.zig");

const Tensor = core.safetensors.Tensor;
const Loader = wl.Loader;
const Gdn = wt.Gdn;
const Attn = wt.Attn;
const MoE = wt.MoE;

// -- INT4-AutoRound: block-FP8 faces, GPTQ int4 experts --------------------------------------------------

/// One block-FP8 linear of a face: name (e4m3 weight, fp32 128x128 scale_inv), row ranges and column range.
const Part8 = struct { name: []const u8, rows: []const [2]usize = &.{}, cols: ?[2]usize = null };

pub fn isFp8(L: *Loader, name: []const u8) !bool {
    var b: [256]u8 = undefined;
    return L.has(try std.fmt.bufPrint(&b, "{s}{s}.weight_scale_inv", .{ L.prefix, name }));
}

/// Fp8BlockLinear.from_rows of stacked rows (fp32 scale per 64 inputs) in FP8G fragment order: ".w8", ".bs".
pub fn fp8Face(L: *Loader, path: []const u8, parts: []const Part8) !fp8.Linear {
    var n: usize = 0;
    var k: usize = 0;
    var ts: [4]Tensor = undefined;
    var invs: [4]Tensor = undefined;
    if (parts.len > ts.len) return error.UnexpectedTensor;
    for (parts, 0..) |p, i| {
        var b: [2][256]u8 = undefined;
        ts[i] = try L.get(try std.fmt.bufPrint(&b[0], "{s}.weight", .{p.name}));
        invs[i] = try L.get(try std.fmt.bufPrint(&b[1], "{s}.weight_scale_inv", .{p.name}));
        const t = ts[i];
        if (t.dtype != .f8_e4m3 or t.rank != 2 or invs[i].dtype != .f32 or !invs[i].is(.f32, &.{ (t.dim(0) + 127) / 128, (t.dim(1) + 127) / 128 })) {
            std.log.err("{s}{s}: {t} {any} with scales {t} {any}; expected e4m3 rows and fp32 128x128-block scales", .{ L.prefix, p.name, t.dtype, t.shape[0..t.rank], invs[i].dtype, invs[i].shape[0..invs[i].rank] });
            return error.UnexpectedTensor;
        }
        const kk = if (p.cols) |c| c[1] - c[0] else t.dim(1);
        if (i == 0) k = kk else if (kk != k) return error.UnexpectedTensor;
        if (p.rows.len == 0) n += t.dim(0) else for (p.rows) |r| {
            n += r[1] - r[0];
        }
    }
    if (k % 64 != 0) return error.UnexpectedTensor;
    const kg = k / 64;
    const codes = try L.gpa.alloc(u8, n * k);
    defer L.gpa.free(codes);
    const cols = try L.gpa.alloc(f32, n * kg);
    defer L.gpa.free(cols);
    var at: usize = 0;
    for (parts, 0..) |p, i| {
        const t = ts[i];
        const pn = t.dim(0);
        const pk = t.dim(1);
        const invf = try L.gpa.alloc(f32, invs[i].numel());
        defer L.gpa.free(invf);
        for (invf, 0..) |*v, j| v.* = @bitCast(std.mem.readInt(u32, invs[i].bytes[4 * j ..][0..4], .little));
        const full = try L.gpa.alloc(f32, pn * (pk / 64));
        defer L.gpa.free(full);
        if (pk % 128 == 0) try fp8.columnScales(invf, pn, pk, full) else {
            // a partial last block of inputs (the fast-fp8 hyper-connections' 320): its scale for its 64s too
            const kb = (pk + 127) / 128;
            for (0..pn) |row| for (0..pk / 64) |j| {
                full[row * (pk / 64) + j] = invf[(row / 128) * kb + (j * 64) / 128];
            };
        }
        const c0 = if (p.cols) |c| c[0] else 0;
        if (c0 % 64 != 0) return error.UnexpectedTensor;
        const all = [_][2]usize{.{ 0, pn }};
        const ranges = if (p.rows.len == 0) &all else p.rows;
        for (ranges) |r| for (r[0]..r[1]) |row| {
            @memcpy(codes[at * k ..][0..k], t.bytes[row * pk + c0 ..][0..k]);
            @memcpy(cols[at * kg ..][0..kg], full[row * (pk / 64) + c0 / 64 ..][0..kg]);
            at += 1;
        };
    }
    const npad = fp8.npadOf(n);
    const w8 = try L.staging(npad * k);
    try fp8.fragmentOrder(codes, n, k, npad, w8);
    var pb: [192]u8 = undefined;
    var l: fp8.Linear = .{ .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) };
    l.w8 = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.w8", .{path}), "uint8", &.{npad * k}, w8);
    L.out.ours();
    const bs = std.mem.bytesAsSlice(f32, try L.staging(npad * kg * 4));
    try fp8.tileScales(cols, n, k, npad, @alignCast(bs));
    l.bs = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.bs", .{path}), "float32", &.{ npad / 64, kg, 64 }, L.host.items);
    L.out.ours();
    return l;
}

/// INT4-AutoRound DeltaNet: q, k, v, z block FP8 then bf16 b, a rows; out_proj's rank head columns block FP8.
pub fn gdn8(L: *Loader, path: []const u8, name: []const u8) !Gdn {
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
    g.proj8 = try L.fp8Face(try std.fmt.bufPrint(&pb, "{s}.proj8", .{path}), &.{
        .{ .name = try L.nameOf("{s}.in_proj_qkv", .{name}), .rows = &channels },
        .{ .name = try L.nameOf("{s}.in_proj_z", .{name}), .rows = &z },
    });
    L.out.ours();
    g.proj = try L.face(try std.fmt.bufPrint(&pb, "{s}.proj.b.weight", .{path}), &.{
        .{ .t = try L.linear(try L.nameOf("{s}.in_proj_b", .{name}), c.nv, c.hidden), .rows = &ab },
        .{ .t = try L.linear(try L.nameOf("{s}.in_proj_a", .{name}), c.nv, c.hidden), .rows = &ab },
    });
    L.out.ours();
    const conv = try L.expect(try L.nameOf("{s}.conv1d.weight", .{name}), .bf16, &.{ c.convDim(), 1, c.conv_kernel });
    const row = c.conv_kernel * 2;
    g.conv = try L.out.begin(try std.fmt.bufPrint(&pb, "{s}.conv", .{path}), "bfloat16", &.{ 2 * qk + vv, c.conv_kernel });
    for (channels) |ch| try L.out.put(conv.bytes[ch[0] * row .. ch[1] * row]);
    try L.out.end();
    g.a_log = try L.widen(try std.fmt.bufPrint(&pb, "{s}.a_log", .{path}), try L.expect(try L.nameOf("{s}.A_log", .{name}), .bf16, &.{c.nv}), r * vl, (r + 1) * vl);
    g.dt_bias = try L.widen(try std.fmt.bufPrint(&pb, "{s}.dt_bias", .{path}), try L.expect(try L.nameOf("{s}.dt_bias", .{name}), .bf16, &.{c.nv}), r * vl, (r + 1) * vl);
    const norm = try L.expect(try L.nameOf("{s}.norm.weight", .{name}), .bf16, &.{c.dv});
    g.norm = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.norm", .{path}), "bfloat16", &.{c.dv}, norm.bytes);
    g.out8 = try L.fp8Face(try std.fmt.bufPrint(&pb, "{s}.out8", .{path}), &.{.{ .name = try L.nameOf("{s}.out_proj", .{name}), .cols = .{ r * vv, (r + 1) * vv } }});
    g.out = .{ .n = @intCast(c.hidden), .k = @intCast(vv) };
    return g;
}

/// INT4-AutoRound attention: rank's q|gate, k, v block FP8, bf16 indexer rows; o_proj block FP8; fp32 norms.
pub fn attention8(L: *Loader, path: []const u8, name: []const u8) !Attn {
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
    a.proj8 = try L.fp8Face(try std.fmt.bufPrint(&pb, "{s}.proj8", .{path}), &.{
        .{ .name = try L.nameOf("{s}.q_proj", .{name}), .rows = &q },
        .{ .name = try L.nameOf("{s}.k_proj", .{name}), .rows = &kv },
        .{ .name = try L.nameOf("{s}.v_proj", .{name}), .rows = &kv },
    });
    a.proj = try L.face(try std.fmt.bufPrint(&pb, "{s}.proj.b.weight", .{path}), &.{
        .{ .t = try L.linear(try L.nameOf("{s}.indexer.index_qk_proj", .{name}), idx_rows, c.hidden) },
    });
    L.out.ours();
    a.q_scale = try L.cscaleRope(try std.fmt.bufPrint(&pb, "{s}.q_scale", .{path}), try L.nameOf("{s}.q_norm.weight", .{name}), hd, true);
    a.k_scale = try L.cscaleRope(try std.fmt.bufPrint(&pb, "{s}.k_scale", .{path}), try L.nameOf("{s}.k_norm.weight", .{name}), hd, true);
    a.iq_scale = try L.cscaleRope(try std.fmt.bufPrint(&pb, "{s}.iq_scale", .{path}), try L.nameOf("{s}.indexer.q_layernorm.weight", .{name}), c.index_dim, true);
    a.ik_scale = try L.cscaleRope(try std.fmt.bufPrint(&pb, "{s}.ik_scale", .{path}), try L.nameOf("{s}.indexer.k_layernorm.weight", .{name}), c.index_dim, true);
    a.o8 = try L.fp8Face(try std.fmt.bufPrint(&pb, "{s}.o8", .{path}), &.{.{ .name = try L.nameOf("{s}.o_proj", .{name}), .cols = .{ r * hl * hd, (r + 1) * hl * hd } }});
    a.o = .{ .n = @intCast(c.hidden), .k = @intCast(hl * hd) };
    return a;
}

/// INT4-AutoRound MoE: bf16 router with shared gate row, block-FP8 shared expert, GPTQ int4 experts; rank halves.
pub fn moe8(L: *Loader, path: []const u8, name: []const u8) !MoE {
    const c = L.c;
    const r: usize = L.o.rank;
    const world: usize = L.o.world;
    const e = c.experts;
    const d = c.hidden;
    var m: MoE = .{};
    var pb: [192]u8 = undefined;
    const router = try L.linear(try L.nameOf("{s}.gate", .{name}), e, d);
    const sgate = try L.linear(try L.nameOf("{s}.shared_expert_gate", .{name}), 1, d);
    m.router = (try L.face(try std.fmt.bufPrint(&pb, "{s}.router", .{path}), &.{ .{ .t = router }, .{ .t = sgate } })).weight;
    const sw = c.shared_width;
    const slo = r * sw / world;
    const shi = (r + 1) * sw / world;
    const srows = [1][2]usize{.{ slo, shi }};
    const gu = try L.fp8Face(try std.fmt.bufPrint(&pb, "{s}.shared8.gu", .{path}), &.{
        .{ .name = try L.nameOf("{s}.shared_expert.gate_proj", .{name}), .rows = &srows },
        .{ .name = try L.nameOf("{s}.shared_expert.up_proj", .{name}), .rows = &srows },
    });
    const dn = try L.fp8Face(try std.fmt.bufPrint(&pb, "{s}.shared8.down", .{path}), &.{.{ .name = try L.nameOf("{s}.shared_expert.down_proj", .{name}), .cols = .{ slo, shi } }});
    m.shared8 = .{ .gu = gu, .down = dn, .width = @intCast(shi - slo) };
    m.int4 = try L.int4Experts(path, name, r * c.moe_width / world, (r + 1) * c.moe_width / world);
    m.routed = .{ .count = e, .width = @intCast(m.int4.?.width), .dims = d };
    return m;
}
