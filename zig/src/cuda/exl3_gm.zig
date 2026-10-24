//! Prompt-sized routed EXL3 experts on the tensor units (exl3_gm.cu): plan, rotation, x3gm gate/up and down, combine.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const ex = @import("exl3_experts.zig");
const x3 = @import("exl3_block_x3gm.zig");
const block = @import("exl3_block_exl3.zig");
const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const k2s = x3.gm_k2s;
const ngu = x3.gu_tiles.len;
const ndn = x3.dn_tiles.len;

/// One layer's widths: each projection's K2 when every expert shares it, else 0 (each expert's own, from the tables).
pub const Widths = struct { gu: u32 = 0, down: u32 = 0 };

/// Configuration ids of x3gm.cu's gate/up (GM_GU*) and down (GM_DN*) tile tables.
pub const Tiles = struct { gu: usize, dn: usize };

/// Members an expert on average -> tiles that keep the flushed chains in registers; TF_EXL3_GM=gu,dn forces both.
pub const tune = [_]x3.TuneRow{ .{ .members = 0, .gu = .{ 0, 0 }, .dn = 0 }, .{ .members = 48, .gu = .{ 0, 0 }, .dn = 2 } };

const sym = struct {
    const rot = "_ZN10dsv41_x3gm10rot_kernelI13__nv_bfloat16Li2EEEvPKT_iPKiPK6__halfS9_PS7_SA_iii";
    const plan = "_ZN15dsv41_x3gm_plan11plan_kernelEPKiiiiPiS2_S2_S2_S2_";
    const combine = "tf_exl3_gm_combine";
};

/// A resolved x3gm instance: its threads, dynamic shared bytes and CTAs an SM (0: does not fit this GPU).
pub const Gm = struct { f: Function, threads: u32, smem: u32, per_sm: u32 };

/// Whether x3gm keeps `l`'s formula: mul1, 128 blocks, < 512 experts, swigluoai or fp32 SiLU with a limit, its widths.
pub fn eligible(l: ex.Layer, w: Widths) bool {
    if (l.codebook != .mul1 or l.gu_stride != 0 or l.experts < 1 or l.experts >= x3.plan_max_experts) return false;
    if (l.dims == 0 or l.width == 0 or l.dims % 128 != 0 or l.width % 128 != 0) return false;
    const act = l.act_mode == 2 or l.act_mode == 3 or (l.act_mode == 1 and l.limit > 0);
    if (!act) return false;
    if ((w.gu == 0) != (w.down == 0)) return false; // v2 runs both projections from the tables
    return w.gu == 0 or (x3.gmK2Index(w.gu) != null and x3.gmK2Index(w.down) != null);
}

/// x3gm's act_mode: 0 its fp32 SiLU (the experts' mode 1), else the experts' swigluoai mode.
fn actMode(l: ex.Layer) c_int {
    return if (l.act_mode == 1) 0 else l.act_mode;
}

/// Pass tables a plan writes: pass -> expert, first member, members; the smallest pass (64 members) bounds them.
const Passes = struct { pe: u64, poff: u64, pcnt: u64, npass: u64 };

/// One call's buffers, laid out in scratch()'s order, each 256-byte aligned.
pub const Scratch = struct {
    xg: u64, // fp16 [P, D] rotated gate inputs
    xu: u64, // fp16 [P, D] rotated up inputs
    xd: u64, // fp16 [P, I] down inputs
    y: u64, // fp32 [P, D] per-slot outputs
    order: u64, // int32 [P] pairs by expert
    pass: [2]Passes, // gate/up's plan, down's plan
    ticket: u64, // int32 [2], zeroed before each launch
    rows: usize,
    slots: usize,
};

fn align256(n: usize) usize {
    return (n + 255) / 256 * 256;
}

fn sizes(l: ex.Layer, rows: usize, slots: usize) [6]usize {
    const p = rows * slots;
    const passes = x3.planPasses(p, l.experts, 64);
    return .{ align256(p * l.dims * 2), align256(p * l.width * 2), align256(p * l.dims * 4), align256(p * 4), align256(passes * 4), 256 };
}

/// Bytes of scratch() for `rows` rows of `slots` routed slots.
pub fn scratchBytes(l: ex.Layer, rows: usize, slots: usize) usize {
    const n = sizes(l, rows, slots);
    return 2 * n[0] + n[1] + n[2] + n[3] + 2 * (3 * n[4] + n[5]) + n[5];
}

pub fn scratch(base: u64, l: ex.Layer, rows: usize, slots: usize) Scratch {
    const n = sizes(l, rows, slots);
    var at = base;
    const take = struct {
        fn f(a: *u64, bytes: usize) u64 {
            const r = a.*;
            a.* = r + bytes;
            return r;
        }
    }.f;
    var sc: Scratch = undefined;
    sc.xg = take(&at, n[0]);
    sc.xu = take(&at, n[0]);
    sc.xd = take(&at, n[1]);
    sc.y = take(&at, n[2]);
    sc.order = take(&at, n[3]);
    for (&sc.pass) |*ps| ps.* = .{ .pe = take(&at, n[4]), .poff = take(&at, n[4]), .pcnt = take(&at, n[4]), .npass = take(&at, n[5]) };
    sc.ticket = take(&at, n[5]);
    sc.rows = rows;
    sc.slots = slots;
    return sc;
}

/// TF_EXL3_GM=gu,dn: both configuration ids forced, else null.
pub fn forced() ?Tiles {
    const v = std.mem.span(std.c.getenv("TF_EXL3_GM") orelse return null);
    var it = std.mem.splitScalar(u8, v, ',');
    const gu = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
    const dn = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
    if (gu >= ngu or dn >= ndn) return null;
    return .{ .gu = gu, .dn = dn };
}

/// The tiles of a call: the table's (by members an expert), forced ones, and v2's own tiles for ragged layers.
pub fn tiles(l: ex.Layer, w: Widths, rows: usize, slots: usize, force: ?Tiles) Tiles {
    if (w.gu == 0) {
        const gu: usize = if (force) |f| (if (f.gu < x3.v2_gu.len) f.gu else 0) else 0;
        return .{ .gu = gu, .dn = 0 };
    }
    const t = x3.gmTuned(&tune, if (force) |f| f.gu else null, if (force) |f| f.dn else null, rows, slots, l.experts, false, w.gu, w.down);
    return .{ .gu = t[0], .dn = t[1] };
}

fn deviceAttribute(d: *const Driver, a: anytype) !u32 {
    var dev: @import("abi.zig").Device = undefined;
    try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
    var v: c_int = 0;
    try d.check(d.api.cuDeviceGetAttribute(&v, a, dev), "cuDeviceGetAttribute");
    return @intCast(v);
}

fn resolve(m: Module, name: [:0]const u8, shape: x3.GmShape, optin: u32) !Gm {
    const f = try m.function(name);
    const ok = try block.fitShared(f, name, shape.smem, optin);
    return .{ .f = f, .threads = shape.threads, .smem = shape.smem, .per_sm = if (ok) try f.occupancy(shape.threads, shape.smem) else 0 };
}

fn i(n: usize) c_int {
    return @intCast(n);
}

pub const Experts = struct {
    mod: Module,
    plan: Function,
    rot: Function,
    combine: Function,
    gu: [k2s.len][ngu]Gm,
    dn: [k2s.len][ndn]Gm,
    gu2: [x3.v2_gu.len]Gm,
    dn2: Gm,
    sms: usize,
    force: ?Tiles = null,

    pub fn load(d: *const Driver) !Experts {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var e: Experts = undefined;
        e.mod = try Module.load(d, kernels.exl3_gm);
        errdefer e.mod.unload();
        const optin = try deviceAttribute(d, .max_shared_memory_per_block_optin);
        e.sms = try deviceAttribute(d, .multiprocessor_count);
        e.plan = try e.mod.function(sym.plan);
        e.rot = try e.mod.function(sym.rot);
        e.combine = try e.mod.function(sym.combine);
        inline for (k2s, 0..) |k2, ki| {
            inline for (0..ngu) |c| e.gu[ki][c] = try resolve(e.mod, x3.gmGuSymbol(2, k2, c), x3.GmShape.of(2, 2, k2, x3.gu_tiles[c]), optin);
            inline for (0..ndn) |c| e.dn[ki][c] = try resolve(e.mod, x3.gmDnSymbol(k2, c), x3.GmShape.of(1, 1, k2, x3.dn_tiles[c]), optin);
        }
        inline for (x3.v2_gu, 0..) |c, ci| e.gu2[ci] = try resolve(e.mod, x3.gm2GuSymbol(2, c), x3.gm2Shape(2, 2, x3.gu_tiles[c]), optin);
        e.dn2 = try resolve(e.mod, x3.gm2DnSymbol(0), x3.gm2Shape(1, 1, x3.dn_tiles[0]), optin);
        e.force = forced();
        return e;
    }

    pub fn unload(e: *Experts) void {
        e.mod.unload();
    }

    fn pick(e: *const Experts, w: Widths, t: Tiles) struct { gu: Gm, dn: Gm } {
        if (w.gu == 0) return .{ .gu = e.gu2[t.gu], .dn = e.dn2 };
        return .{ .gu = e.gu[x3.gmK2Index(w.gu).?][t.gu], .dn = e.dn[x3.gmK2Index(w.down).?][t.dn] };
    }

    fn gm(e: *const Experts, s: Stream, g: Gm, a: x3.GmArgs) !void {
        if (g.per_sm == 0) return error.Unsupported;
        var args: launch_.Args = .{};
        args.add(a);
        try launch_.launch(g.f, .{ .grid = .{ .x = @intCast(e.sms * g.per_sm) }, .block = .{ .x = g.threads }, .shared = g.smem }, s, &args);
    }

    fn planOf(e: *const Experts, s: Stream, pk: u64, pairs: usize, experts: usize, bm: usize, order: u64, ps: Passes) !void {
        var a: launch_.Args = .{};
        a.add(pk);
        for ([_]usize{ pairs, experts, bm }) |v| a.add(i(v));
        for ([_]u64{ order, ps.pe, ps.poff, ps.pcnt, ps.npass }) |v| a.add(v);
        try launch_.launch(e.plan, .{ .grid = .{ .x = 1 }, .block = .{ .x = 512 } }, s, &a);
    }

    /// The routed experts of `rows` bf16 rows into fp32 `out` [rows, D]: Experts.run's formula on the tensor units.
    pub fn run(e: *const Experts, d: *const Driver, s: Stream, l: ex.Layer, w: Widths, x: u64, x_stride: usize, pk: u64, wts: u64, rows: usize, out: u64, sc: Scratch) !void {
        if (!eligible(l, w) or rows < 1 or rows > sc.rows or sc.slots < 1 or sc.slots > 32 or x_stride < l.dims) return error.Exl3Shape;
        const t = tiles(l, w, rows, sc.slots, e.force);
        const g = e.pick(w, t);
        if (g.gu.per_sm == 0 or g.dn.per_sm == 0) return error.Unsupported;
        try x3.gmCheck(.gu, t.gu, l.dims, l.width);
        try x3.gmCheck(.dn, if (w.gu == 0) 0 else t.dn, l.width, l.dims);
        const p = rows * sc.slots;
        const bm_gu = x3.gmMembers(.gu, t.gu);
        const bm_dn = x3.gmMembers(.dn, if (w.gu == 0) 0 else t.dn);
        try e.planOf(s, pk, p, l.experts, bm_gu, sc.order, sc.pass[0]);
        const dn_pass = if (bm_dn == bm_gu) sc.pass[0] else sc.pass[1];
        if (bm_dn != bm_gu) try e.planOf(s, pk, p, l.experts, bm_dn, sc.order, sc.pass[1]);

        var r: launch_.Args = .{};
        r.add(x);
        r.add(i(x_stride));
        for ([_]u64{ pk, l.suh_g, l.suh_u, sc.xg, sc.xu }) |v| r.add(v);
        const items = p * (l.dims / 128);
        for ([_]usize{ l.dims, sc.slots, items }) |v| r.add(i(v));
        try launch_.launch(e.rot, .{ .grid = .{ .x = @intCast((items + 7) / 8) }, .block = .{ .x = 256 } }, s, &r);
        try d.check(d.api.cuMemsetD8Async(sc.ticket, 0, 8, s.handle), "cuMemsetD8Async");

        const pg = sc.pass[0];
        try e.gm(s, g.gu, .{ .x0 = sc.xg, .x1 = sc.xu, .tp0 = l.gate, .tp1 = l.up, .k2e = l.gate_k2, .order = sc.order, .pe = pg.pe, .poff = pg.poff, .pcnt = pg.pcnt, .npass = pg.npass, .ticket = sc.ticket, .sv0 = l.svh_g, .sv1 = l.svh_u, .sd = l.suh_d, .xd = sc.xd, .K = i(l.dims), .N = i(l.width), .limit = l.limit, .act_mode = actMode(l), .alpha = l.alpha, .up_bias = l.up_bias });
        try e.gm(s, g.dn, .{ .x0 = sc.xd, .x1 = sc.xd, .tp0 = l.down, .tp1 = l.down, .k2e = l.down_k2, .order = sc.order, .pe = dn_pass.pe, .poff = dn_pass.poff, .pcnt = dn_pass.pcnt, .npass = dn_pass.npass, .ticket = sc.ticket + 4, .sv0 = l.svh_d, .y = sc.y, .K = i(l.width), .N = i(l.dims) });

        var c: launch_.Args = .{};
        for ([_]u64{ sc.y, wts, out }) |v| c.add(v);
        c.add(i(l.dims));
        c.add(i(sc.slots));
        try launch_.launch(e.combine, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((l.dims + 255) / 256) }, .block = .{ .x = 256 } }, s, &c);
    }
};

fn layerOf(act: i32, limit: f32) ex.Layer {
    return .{ .gate = 0, .up = 0, .down = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .experts = 128, .dims = 6144, .width = 1536, .codebook = .mul1, .act_mode = act, .limit = limit };
}

test "x3gm takes the formulas it keeps and the widths it builds" {
    try std.testing.expect(eligible(layerOf(2, 7), .{ .gu = 6, .down = 6 }));
    try std.testing.expect(eligible(layerOf(3, 7), .{ .gu = 6, .down = 8 }));
    try std.testing.expect(eligible(layerOf(2, 7), .{}));
    try std.testing.expect(!eligible(layerOf(0, 7), .{ .gu = 8, .down = 8 })); // GLM's bf16-rounded SiLU
    try std.testing.expect(!eligible(layerOf(1, 0), .{ .gu = 8, .down = 8 })); // x3gm would clamp the gate at 0
    try std.testing.expect(!eligible(layerOf(2, 7), .{ .gu = 12, .down = 12 })); // 6 bits: not an x3gm width
    try std.testing.expect(!eligible(layerOf(2, 7), .{ .gu = 6, .down = 0 }));
    var mcg = layerOf(2, 7);
    mcg.codebook = .mcg;
    try std.testing.expect(!eligible(mcg, .{ .gu = 8, .down = 8 }));
}

test "the scratch layout is 256-aligned and inside scratchBytes" {
    const l = layerOf(2, 7);
    const sc = scratch(0x10000, l, 2048, 4);
    for ([_]u64{ sc.xg, sc.xu, sc.xd, sc.y, sc.order, sc.pass[0].pe, sc.pass[1].npass, sc.ticket }) |p| try std.testing.expectEqual(@as(u64, 0), p % 256);
    try std.testing.expect(sc.ticket + 8 <= 0x10000 + scratchBytes(l, 2048, 4));
    try std.testing.expectEqual(@as(u64, 0x10000 + 2048 * 4 * 6144 * 2), sc.xu);
}

test "ragged layers run v2's tiles; uniform ones the table's" {
    const l = layerOf(2, 7);
    try std.testing.expectEqual(Tiles{ .gu = 0, .dn = 0 }, tiles(l, .{}, 2048, 4, null));
    try std.testing.expectEqual(Tiles{ .gu = 0, .dn = 2 }, tiles(l, .{ .gu = 6, .down = 6 }, 2048, 4, null)); // 64 members
    try std.testing.expectEqual(Tiles{ .gu = 0, .dn = 0 }, tiles(l, .{ .gu = 6, .down = 6 }, 256, 4, null)); // 8 members
    try std.testing.expectEqual(Tiles{ .gu = 5, .dn = 4 }, tiles(l, .{ .gu = 6, .down = 6 }, 256, 4, .{ .gu = 5, .dn = 4 }));
}
