//! A decode window on the prod kernel path (prod.env: dense3, x3seg, the CUDA mHC / attention / top-k, router GEMV, x3ld,
//! L2 prefetch), emitted as calls (calls.zig): every launch with its arguments and its tensors named by role, in the
//! Python engine's order. M1 checks the calls against the Python engine's capture launch by launch (structure on the
//! host, bits on the GPU); M2's forward issues the same calls.
//!
//! Roles: "s." persists (weights' run-time forms, KV pools, scratch), "w." lives for the window (streams, positions, mHC
//! coefficients, the exchange buffer, the index selections reuse layers read), "L." for one layer.

const std = @import("std");
const dk = @import("dsv41_kernels");
const calls = @import("calls.zig");
const config = @import("config.zig");
const moe = @import("block_moe.zig");
const Config = config.Config;
const Arg = calls.Arg;
const Named = calls.Named;
const Dt = calls.Dt;
const dense = dk.dense;

/// The widths the loader picked: K2 of each dense group (by our weight prefix) and each routed layer's expert K2 range
/// (gate/up, down; the shared expert included). From the plan in a run; the M1 check reads the capture's.
pub const Widths = struct {
    dense: std.StringHashMapUnmanaged(u32) = .empty,
    experts: std.AutoHashMapUnmanaged(u32, [2][2]u32) = .empty,

    pub fn k2(w: *const Widths, name: []const u8) !u32 {
        return w.dense.get(name) orelse error.NoWidth;
    }
};

/// Run-time sizes the Python engine fixes at load (its scratch and pools), and the prod L2 prefetch settings.
pub const Options = struct {
    /// TP ranks (a boot constant): every per-rank width below is the model's / world; the plan checked divisibility
    world: u32 = 2,
    /// KV positions a slot holds (pool rows = limit / ratio + 1)
    limit: i64 = 4096,
    /// rope table rows (cos / sin pairs of 32)
    rope_rows: i64 = 6144,
    /// rows the routed experts' scratch holds: the backbone's and a DSpark block's
    expert_rows: i64 = 1024,
    dspark_rows: i64 = 64,
    /// TF_DSV41_L2PF_MB (bytes), _CHUNK_KB, _PACE_GBPS, _PACE_CTAS; the GPU's SMs (a segments launch's grid bound)
    l2pf_budget: u64 = 12 << 20,
    l2pf_chunk: i64 = 32768,
    l2pf_pace_gbps: f64 = 150,
    l2pf_pace_ctas: i64 = 2,
    sms: i64 = 188,
    /// TF_DSV41_EXPERT_TOPP (prod: 0.85; lossy)
    expert_topp: f64 = 0.85,
};

pub const Emitter = struct {
    a: std.mem.Allocator,
    cfg: *const Config,
    w: *const Widths,
    o: Options,
    /// rows this window and the position of its first
    n: i64,
    start: i64,
    out: std.ArrayList(calls.Call) = .empty,
    /// which of the two stream buffers holds the streams, and which coefficient set the last boundary wrote
    cur: u1 = 0,
    coef: u1 = 0,
    begin: calls.Begin = .window,
    /// the current layer's compression ratio (picks its rope table)
    layer_ratio: u32 = 0,

    pub fn fmt(e: *Emitter, comptime f: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(e.a, f, args);
    }

    /// A contiguous tensor of `role`.
    pub fn t(e: *Emitter, role: calls.Role, dt: Dt, shape: []const i64) !Arg {
        const s = try e.a.dupe(i64, shape);
        return .{ .t = .{ .role = role, .dt = dt, .shape = s, .stride = try calls.contiguous(e.a, s) } };
    }

    /// A view of `role`: its own strides and a byte offset.
    pub fn view(e: *Emitter, role: calls.Role, dt: Dt, shape: []const i64, stride: []const i64, offset: i64) !Arg {
        return .{ .t = .{ .role = role, .dt = dt, .shape = try e.a.dupe(i64, shape), .stride = try e.a.dupe(i64, stride), .offset = offset } };
    }

    pub fn buf(e: *Emitter, comptime f: []const u8, args: anytype, dt: Dt, shape: []const i64) !Arg {
        return e.t(.{ .buf = try e.fmt(f, args) }, dt, shape);
    }

    pub fn weight(e: *Emitter, comptime f: []const u8, args: anytype, dt: Dt, shape: []const i64) !Arg {
        return e.t(.{ .weight = try e.fmt(f, args) }, dt, shape);
    }

    pub fn empty(e: *Emitter, dt: Dt) !Arg {
        return e.t(.empty, dt, &.{0});
    }

    pub fn ints(e: *Emitter, xs: []const i64) !Arg {
        const l = try e.a.alloc(Arg, xs.len);
        for (xs, l) |x, *y| y.* = .{ .i = x };
        return .{ .list = l };
    }

    fn push(e: *Emitter, c: calls.Call) !void {
        var x = c;
        x.begin = e.begin;
        e.begin = .none;
        try e.out.append(e.a, x);
    }

    pub fn ext(e: *Emitter, name: []const u8, args: []const Arg) !void {
        const l = try e.a.alloc(Named, args.len);
        for (args, l) |x, *y| y.* = .{ .arg = x };
        try e.push(.{ .triton = false, .name = name, .args = l });
    }

    pub fn triton(e: *Emitter, name: []const u8, grid: [3]i64, args: []const Named) !void {
        try e.push(.{ .triton = true, .name = name, .grid = grid, .args = try e.a.dupe(Named, args) });
    }

    /// The Python engine's float of a config value: the JSON number (f64), not the f32 we keep.
    pub fn dec(x: f32) f64 {
        var b: [64]u8 = undefined;
        const s = std.fmt.bufPrint(&b, "{e}", .{x}) catch return x;
        return std.fmt.parseFloat(f64, s) catch x;
    }

    /// Positions after this window: the last row's + 1.
    pub fn end(e: *const Emitter) i64 {
        return e.start + e.n;
    }

    pub fn streams(e: *Emitter) !Arg {
        return e.buf("w.x{d}", .{e.cur}, .bf16, &.{ e.n, 4 * @as(i64, e.cfg.hidden) });
    }

    pub fn spare(e: *Emitter) !Arg {
        return e.buf("w.x{d}", .{1 - e.cur}, .bf16, &.{ e.n, 4 * @as(i64, e.cfg.hidden) });
    }

    // -----------------------------------------------------------------------------------------------------------
    // dense EXL3 projections

    /// A trellis's run-time words (int32 [N/128, K/16, 8, 4 K2]): upstream's strips "s.<weight>.T", or dense3's lanes
    /// layout of them "s.<weight>.lanes" (prepare.strips, prepare.lanes).
    pub fn strips(e: *Emitter, w: []const u8, k: i64, n: i64) !Arg {
        return e.words(w, k, n, "T");
    }

    fn words(e: *Emitter, w: []const u8, k: i64, n: i64, form: []const u8) !Arg {
        const k2: i64 = try e.w.k2(w);
        return e.t(.{ .buf = try e.fmt("s.{s}.{s}", .{ w, form }) }, .i32, &.{ @divExact(n, 128), @divExact(k, 16), 8, 4 * k2 });
    }

    /// Bytes of a group's trellis (= its strips).
    pub fn trellisBytes(e: *Emitter, w: []const u8, k: i64, n: i64) !u64 {
        const k2: u64 = try e.w.k2(w);
        return @as(u64, @intCast(@divExact(k, 16) * @divExact(n, 16))) * 8 * k2 * 2;
    }

    /// One projection of a fused group: weight prefix, input, output (a whole buffer; `col` its first column), K x N.
    pub const Proj = struct { w: []const u8, x: Arg, y: Arg, col: usize = 0, k: usize, n: usize };

    /// x3seg rot_in, then dense3 a width (fused_proj.run): the group's projections in one rot_in launch, one linear
    /// launch for each K2 in first-appearance order. `tag` names the group's scratch.
    pub fn fused(e: *Emitter, tag: []const u8, ps: []const Proj) !void {
        const m: usize = @intCast(e.n);
        const segs = try e.a.alloc(dense.Seg, ps.len);
        for (ps, segs, 0..) |p, *s, i| {
            const pl = dense.plan(p.k, p.n);
            s.* = .{ .k = p.k, .n = p.n, .k2 = try e.w.k2(p.w), .sk = pl[0], .wk = pl[1], .out = i, .col = p.col };
        }
        const lay = try dense.layout(segs, m);
        const xs = try e.a.alloc(Arg, ps.len);
        const suh = try e.a.alloc(Arg, ps.len);
        const offs = try e.a.alloc(i64, ps.len);
        for (ps, xs, suh, offs, 0..) |p, *x, *s, *o, i| {
            x.* = p.x;
            s.* = try e.weight("{s}.suh", .{p.w}, .f16, &.{@intCast(p.k)});
            o.* = @intCast(lay.xh_off[i]);
        }
        const xh = try e.buf("L.{s}.xh", .{tag}, .f16, &.{@intCast(lay.xh_total)});
        try e.ext("tf_dsv41_x3seg_v3.rot_in", &.{ .{ .list = xs }, .{ .list = suh }, xh, try e.ints(offs), .{ .b = false } });
        const z: Arg = if (lay.z_total > 0) try e.buf("L.{s}.z", .{tag}, .f32, &.{@intCast(lay.z_total)}) else .none;
        const counters = try e.buf("L.{s}.cnt", .{tag}, .i32, &.{@intCast(lay.c_total)});
        for (lay.launches[0..lay.n_launches]) |l| {
            const T = try e.a.alloc(Arg, l.count);
            const svh = try e.a.alloc(Arg, l.count);
            const ys = try e.a.alloc(Arg, l.count);
            var meta: std.ArrayList(i64) = .empty;
            for (l.segs[0..l.count], 0..) |si, j| {
                const s = segs[si];
                const p = ps[si];
                T[j] = try e.words(p.w, @intCast(p.k), @intCast(p.n), "lanes");
                svh[j] = try e.weight("{s}.svh", .{p.w}, .f16, &.{@intCast(p.n)});
                ys[j] = p.y;
                const st = dense.strides(s.k, s.k2);
                try meta.appendSlice(e.a, &.{ @intCast(s.k), @intCast(s.n), @intCast(s.sk), @intCast(s.wk), st[0], st[1], @intCast(lay.xh_off[si]), @intCast(lay.z_off[si]), @intCast(lay.c_off[si]), @intCast(s.col), @intCast(l.p0[j]) });
            }
            try e.ext("tf_dsv41_dense3_v1.linear", &.{
                xh,                    .{ .list = T },                 .{ .list = svh },          .{ .list = ys },
                z,                     counters,                       try e.ints(meta.items),    .{ .i = @intCast(l.programs) },
                .{ .i = l.k2 },        .{ .i = @intCast(l.wk) },       .{ .i = if (l.wk == 8) 1 else 3 }, .{ .b = false },
            });
        }
    }

    /// Upstream's rot_in + linear (exl3 linear_v4): one projection, its trellis as strips unless one block of 128
    /// columns (then the stored trellis is the strips).
    pub fn upstream(e: *Emitter, w: []const u8, x: Arg, k: i64, n: i64, y: Arg) !void {
        const k2: i64 = try e.w.k2(w);
        const xh = try e.buf("L.{s}.xh", .{w}, .f16, &.{ e.n, k });
        try e.ext("tensorfold_exl3_linear_v4.rot_in", &.{ x, try e.weight("{s}.suh", .{w}, .f16, &.{k}), xh });
        const pl = dense.plan(@intCast(k), @intCast(n));
        const st = dense.strides(@intCast(k), @intCast(k2));
        const T = if (n == 128)
            try e.view(.{ .weight = try e.fmt("{s}.trellis", .{w}) }, .i32, &.{ 1, @divExact(k, 16), 8, 4 * k2 }, &.{ 32 * k2, 32 * k2, 4 * k2, 1 }, 0)
        else
            try e.strips(w, k, n);
        if (pl[0] != 1) return error.Unsupported; // no upstream linear on the prod decode path splits K
        try e.ext("tensorfold_exl3_linear_v4.linear", &.{
            xh, T, .{ .i = st[0] }, .{ .i = st[1] }, try e.weight("{s}.svh", .{w}, .f16, &.{n}), .none, y, .none,
            try e.buf("L.{s}.cnt", .{w}, .i32, &.{8 * @divExact(n, 128)}),
            .{ .i = k2 }, .{ .i = 2 }, .{ .i = @intCast(pl[0]) }, .{ .i = @intCast(pl[1]) },
        });
    }

    // -----------------------------------------------------------------------------------------------------------
    // L2 prefetch (l2pf.py): the table rows only (the addresses are the device's; the replay skips these launches)

    /// l2pf.take: spans of `sizes` in order within the budget, the last one's head.
    fn take(e: *const Emitter, sizes: []const u64) i64 {
        var left = e.o.l2pf_budget;
        var rows: i64 = 0;
        for (sizes) |s| {
            if (left < 16) break;
            const n = @min(left, s) / 16 * 16;
            if (n == 0) continue;
            rows += 1;
            left -= n;
        }
        return rows;
    }

    fn linSizes(e: *Emitter, out: *std.ArrayList(u64), ws: []const []const u8, ks: []const i64, ns: []const i64) !void {
        for (ks, ns) |k, n| try out.appendSlice(e.a, &.{ @intCast(2 * k), @intCast(2 * n) });
        for (ws, ks, ns) |w, k, n| try out.append(e.a, try e.trellisBytes(w, k, n));
    }

    /// Site "o" (after wq_b): wo_a's groups, then wo_b; l2pf segments.
    pub fn prefetchO(e: *Emitter, L: u32) !void {
        const D: i64 = e.cfg.hidden;
        const g = e.cfg.o_groups / e.o.world;
        const ki: i64 = @divExact(@as(i64, e.cfg.heads) * e.cfg.head_dim, @as(i64, e.cfg.o_groups));
        var sizes: std.ArrayList(u64) = .empty;
        const ws = try e.a.alloc([]const u8, g);
        const ks = try e.a.alloc(i64, g);
        const ns = try e.a.alloc(i64, g);
        for (ws, ks, ns, 0..) |*w, *k, *n, i| {
            w.* = try e.fmt("L{d}.attn.wo_a.{d}", .{ L, i });
            k.* = ki;
            n.* = e.cfg.o_lora;
        }
        try e.linSizes(&sizes, ws, ks, ns);
        try e.linSizes(&sizes, &.{try e.fmt("L{d}.attn.wo_b", .{L})}, &.{@as(i64, g) * e.cfg.o_lora}, &.{D});
        var total: u64 = 0;
        var left = e.o.l2pf_budget;
        for (sizes.items) |s| {
            if (left < 16) break;
            const n = @min(left, s) / 16 * 16;
            total += n;
            left -= n;
        }
        const pieces = std.math.divCeil(u64, total, @intCast(e.o.l2pf_chunk)) catch 1;
        const grid: i64 = @max(1, @min(e.o.sms, @as(i64, @intCast(std.math.divCeil(u64, pieces, 128) catch 1))));
        const rows = e.take(sizes.items);
        try e.ext("tf_dsv41_l2pf_v1.segments", &.{ .{ .opaque_table = (try e.buf("s.L{d}.pf.o", .{L}, .i64, &.{ rows, 2 })).t }, .{ .i = grid }, .{ .i = 128 }, .{ .i = e.o.l2pf_chunk }, .{ .b = true } });
    }

    fn paced(e: *Emitter, role: []const u8, rows: i64) !void {
        const ns: i64 = @max(1, @as(i64, @intFromFloat(@round(@as(f64, @floatFromInt(e.o.l2pf_chunk)) / e.o.l2pf_pace_gbps))));
        try e.ext("tf_dsv41_l2pace_v1.paced", &.{ .{ .opaque_table = (try e.buf("{s}", .{role}, .i64, &.{ rows, 2 })).t }, .{ .i = e.o.l2pf_pace_ctas }, .{ .i = e.o.l2pf_chunk }, .{ .i = ns }, .{ .i = 0 } });
    }

    /// Site "f" (after the attention sublayer): the router's gate, then the shared expert's gate and up trellises.
    pub fn prefetchF(e: *Emitter, L: u32) !void {
        const D: i64 = e.cfg.hidden;
        const I: i64 = e.cfg.expert_width / e.o.world;
        const ex = e.cfg.expertsOf(L);
        const sh = try e.fmt("L{d}.moe.shared.0.w1", .{L});
        const k2: u64 = try e.w.k2(sh);
        const shared: u64 = @as(u64, @intCast(@divExact(D, 16) * @divExact(I, 16))) * 8 * k2 * 2;
        const sizes = [_]u64{ @as(u64, ex.count) * @as(u64, @intCast(D)) * 2, shared, shared };
        try e.paced(try e.fmt("s.L{d}.pf.f", .{L}), e.take(&sizes));
    }

    /// Site "x" of the block before `nxt` (after its MoE): `nxt`'s x group, then its q group.
    pub fn prefetchX(e: *Emitter, nxt: u32) !void {
        var sizes: std.ArrayList(u64) = .empty;
        const xg = try e.xGroup(nxt);
        try e.linSizes(&sizes, xg.ws, xg.ks, xg.ns);
        const qg = try e.qGroup(nxt);
        try e.linSizes(&sizes, qg.ws, qg.ks, qg.ns);
        try e.paced(try e.fmt("s.L{d}.pf.x", .{nxt}), e.take(sizes.items));
    }

    const Group = struct { ws: []const []const u8, ks: []const i64, ns: []const i64 };

    /// The x group (fused_proj): wq_a, wkv, then a KV source's compressor projections.
    fn xGroup(e: *Emitter, L: u32) !Group {
        const D: i64 = e.cfg.hidden;
        const ratio = e.cfg.compressRatio(L);
        const comp: usize = if (e.cfg.isKvSource(L)) (if (ratio == 2) 2 else 1) else 0;
        const ws = try e.a.alloc([]const u8, 2 + comp);
        const ns = try e.a.alloc(i64, 2 + comp);
        const names = [_][]const u8{ "wq_a", "wkv", "comp_wkv", "comp_wgate" };
        for (ws, ns, 0..) |*w, *n, i| {
            w.* = try e.fmt("L{d}.attn.{s}", .{ L, names[i] });
            n.* = if (i == 0) e.cfg.q_lora else e.cfg.head_dim;
        }
        const ks = try e.a.alloc(i64, ws.len);
        @memset(ks, D);
        return .{ .ws = ws, .ks = ks, .ns = ns };
    }

    /// The q group: wq_b, and an indexer's ix_wq_b.
    fn qGroup(e: *Emitter, L: u32) !Group {
        const ix = switch (e.cfg.mode(L)) {
            .full, .reindex => true,
            else => false,
        };
        const ws = try e.a.alloc([]const u8, if (ix) 2 else 1);
        const ns = try e.a.alloc(i64, ws.len);
        ws[0] = try e.fmt("L{d}.attn.wq_b", .{L});
        ns[0] = @as(i64, e.cfg.heads / e.o.world) * e.cfg.head_dim;
        if (ix) {
            ws[1] = try e.fmt("L{d}.attn.ix_wq_b", .{L});
            ns[1] = @as(i64, e.cfg.index_heads) * e.cfg.index_dim;
        }
        const ks = try e.a.alloc(i64, ws.len);
        @memset(ks, e.cfg.q_lora);
        return .{ .ws = ws, .ks = ks, .ns = ns };
    }

    // -----------------------------------------------------------------------------------------------------------
    // attention

    fn rope(e: *Emitter, x: Arg, heads: i64, dim: i64, nope: i64) !void {
        const n = e.n;
        const xv = try e.view(x.t.role, .bf16, &.{ n, heads, dim }, &.{ heads * dim, dim, 1 }, 0);
        try e.triton("_rope", .{ n, @divExact(heads, 4), 1 }, &.{
            .{ .name = "X", .arg = xv },                               .{ .name = "x_rs", .arg = .{ .i = heads * dim } },
            .{ .name = "x_hs", .arg = .{ .i = dim } },                 .{ .name = "OUT", .arg = xv },
            .{ .name = "o_rs", .arg = .{ .i = heads * dim } },         .{ .name = "o_hs", .arg = .{ .i = dim } },
            .{ .name = "CS", .arg = try e.ropeTable() },              .{ .name = "cs_rs", .arg = .{ .i = 64 } },
            .{ .name = "POS", .arg = try e.buf("w.pos64", .{}, .i64, &.{n}) }, .{ .name = "H", .arg = .{ .i = heads } },
            .{ .name = "NOPE", .arg = .{ .i = nope } },                .{ .name = "NB", .arg = .{ .i = 64 } },
            .{ .name = "HALF", .arg = .{ .i = 32 } },                  .{ .name = "BH", .arg = .{ .i = 4 } },
            .{ .name = "INV", .arg = .{ .b = false } },                .{ .name = "COPY", .arg = .{ .b = false } },
            .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }

    /// The rope table (cos / sin) a layer uses: the compressed layers' (compress_theta) or the SWA layers'.
    pub fn ropeTable(e: *Emitter) !Arg {
        return e.buf("s.rope.{s}", .{if (e.layer_ratio > 0) "comp" else "main"}, .f32, &.{ e.o.rope_rows, 64 });
    }

    fn pos(e: *Emitter) !Arg {
        return e.buf("w.pos", .{}, .i32, &.{1});
    }

    /// The KV store (_kv_store): the SWA ring (ratio 0) or a compressed pool.
    fn kvStore(e: *Emitter, lat: Arg, src: u32, ratio: i64) !void {
        const rows = try e.pool(src, ratio);
        try e.triton("_kv_store", .{ e.n, 1, 1 }, &.{
            .{ .name = "LAT", .arg = lat },                          .{ .name = "l_stride", .arg = .{ .i = e.cfg.head_dim } },
            .{ .name = "CS", .arg = try e.ropeTable() },             .{ .name = "cs_stride", .arg = .{ .i = 64 } },
            .{ .name = "V", .arg = rows[0] },                        .{ .name = "S", .arg = rows[1] },
            .{ .name = "v_stride", .arg = .{ .i = 576 } },           .{ .name = "s_stride", .arg = .{ .i = 8 } },
            .{ .name = "POS", .arg = try e.pos() },                  .{ .name = "RATIO", .arg = .{ .i = ratio } },
            .{ .name = "RING", .arg = .{ .i = if (ratio == 0) 2 * @as(i64, e.cfg.window) else 1 } }, .{ .name = "PT", .arg = .none },
            .{ .name = "PSH", .arg = .{ .i = 0 } },                  .{ .name = "SL", .arg = .none },
            .{ .name = "PTS", .arg = .{ .i = 0 } },                  .{ .name = "ROWS", .arg = .{ .b = false } },
            .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }

    /// A layer's KV rows (fp8 latent + rope, 576 B) and scales (8 B): the SWA ring (ratio 0) or the compressed pool.
    fn pool(e: *Emitter, L: u32, ratio: i64) ![2]Arg {
        const rows: i64 = if (ratio == 0) 2 * @as(i64, e.cfg.window) else @divFloor(e.o.limit, ratio) + 1;
        const kind = if (ratio == 0) "swa" else "comp";
        return .{ try e.buf("s.L{d}.{s}.v", .{ L, kind }, .u8, &.{ rows, 576 }), try e.buf("s.L{d}.{s}.s", .{ L, kind }, .u8, &.{ rows, 8 }) };
    }

    fn indexKeys(e: *Emitter, L: u32) !Arg {
        const r: i64 = e.cfg.compressRatio(L);
        return e.buf("s.L{d}.ik", .{L}, .u8, &.{ @divFloor(e.o.limit, r) + 1, @as(i64, e.cfg.index_dim) + 4 });
    }

    /// topk.plan: (cluster CTAs, entries a thread) for `nk` entries.
    fn topkPlan(nk: i64, cl_in: ?i64) [2]i64 {
        const cl = cl_in orelse @max(1, @min(8, std.math.divCeil(i64, nk, 4096) catch 1));
        var ept = @max(1, std.math.divCeil(i64, nk, cl * 512) catch 1);
        ept += 1 - @mod(ept, 2);
        return .{ cl, ept };
    }

    /// The window's index selection of layer `src` (reuse layers read the source's).
    fn selection(e: *Emitter, src: u32) ![2]Arg {
        return .{ try e.buf("w.sel{d}", .{src}, .i32, &.{ e.n, e.cfg.index_topk }), try e.buf("w.cnt{d}", .{src}, .i32, &.{e.n}) };
    }

    pub fn attention(e: *Emitter, L: u32) !void {
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const md = e.cfg.mode(L);
        const ratio: i64 = e.cfg.compressRatio(L);
        const kv_src = e.cfg.kvSource(L);
        const H: i64 = e.cfg.heads / e.o.world;
        const hd: i64 = e.cfg.head_dim;
        const IH: i64 = e.cfg.index_heads;
        const ID: i64 = e.cfg.index_dim;
        const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
        const indexer = md == .full or md == .reindex;
        const own_kv = md == .full;
        // the indexer's head weights (a bf16 GEMV over the normed input)
        const iw = try e.buf("L.ix_w", .{}, .bf16, &.{ n, IH });
        if (indexer) try e.triton("_plain", .{ n, IH, 1 }, &.{
            .{ .name = "X", .arg = x },                                                      .{ .name = "x_rs", .arg = .{ .i = D } },
            .{ .name = "W", .arg = try e.weight("L{d}.attn.ix_wp", .{L}, .f32, &.{ IH, D }) }, .{ .name = "w_rs", .arg = .{ .i = D } },
            .{ .name = "OUT", .arg = iw },                                                   .{ .name = "K", .arg = .{ .i = D } },
            .{ .name = "N", .arg = .{ .i = IH } },                                           .{ .name = "BK", .arg = .{ .i = 256 } },
            .{ .name = "PDL", .arg = .{ .b = false } },                                      .{ .name = "BF16", .arg = .{ .b = true } },
        });
        // x group: wq_a, wkv (+ the compressor's on a KV source)
        const xg = try e.xGroup(L);
        const qa = try e.buf("L.qa", .{}, .bf16, &.{ n, e.cfg.q_lora });
        const kv = try e.buf("L.kv", .{}, .bf16, &.{ n, hd });
        const cw: i64 = if (ratio == 2) 2 * hd else hd;
        const comp = try e.buf("L.comp", .{}, .bf16, &.{ n, cw });
        var ps: [4]Proj = undefined;
        for (xg.ws, 0..) |w, i| ps[i] = .{ .w = w, .x = x, .k = @intCast(D), .n = @intCast(xg.ns[i]), .y = switch (i) {
            0 => qa,
            1 => kv,
            else => comp,
        }, .col = if (i == 3) @intCast(hd) else 0 };
        try e.fused("x", ps[0..xg.ws.len]);
        const qn = try e.buf("L.qn", .{}, .bf16, &.{ n, e.cfg.q_lora });
        const kvn = try e.buf("L.kvn", .{}, .bf16, &.{ n, hd });
        try e.triton("_rms2", .{ n, 2, 1 }, &.{
            .{ .name = "XA", .arg = qa },                                                      .{ .name = "xa_rs", .arg = .{ .i = e.cfg.q_lora } },
            .{ .name = "WA", .arg = try e.weight("L{d}.attn.q_norm", .{L}, .f32, &.{e.cfg.q_lora}) }, .{ .name = "OA", .arg = qn },
            .{ .name = "oa_rs", .arg = .{ .i = e.cfg.q_lora } },                              .{ .name = "KA", .arg = .{ .i = e.cfg.q_lora } },
            .{ .name = "inv_ka", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(e.cfg.q_lora)) } }, .{ .name = "XB", .arg = kv },
            .{ .name = "xb_rs", .arg = .{ .i = hd } },                                        .{ .name = "WB", .arg = try e.weight("L{d}.attn.kv_norm", .{L}, .f32, &.{hd}) },
            .{ .name = "OB", .arg = kvn },                                                    .{ .name = "ob_rs", .arg = .{ .i = hd } },
            .{ .name = "KB", .arg = .{ .i = hd } },                                           .{ .name = "inv_kb", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(hd)) } },
            .{ .name = "eps", .arg = .{ .f = dec(e.cfg.eps) } },                              .{ .name = "BK", .arg = .{ .i = 512 } },
            .{ .name = "NARROW", .arg = .{ .b = true } },                                     .{ .name = "PDL", .arg = .{ .b = false } },
        });
        if (!indexer) try e.kvStore(kvn, L, 0);
        if (own_kv) {
            // the compressor: pooled rows, normed, into the pool; the indexer's keys from the same rows
            const lat = try e.buf("L.lat", .{}, .bf16, &.{ n, hd });
            try e.triton("_pool_norm", .{ n, 1, 1 }, &.{
                .{ .name = "BUF", .arg = comp },                                                  .{ .name = "b_stride", .arg = .{ .i = cw } },
                .{ .name = "W", .arg = try e.weight("L{d}.attn.comp_norm", .{L}, .f32, &.{hd}) }, .{ .name = "LAT", .arg = lat },
                .{ .name = "POS", .arg = try e.pos() },                                           .{ .name = "n", .arg = .{ .i = n } },
                .{ .name = "EPS", .arg = .{ .f = dec(e.cfg.eps) } },                              .{ .name = "RATIO", .arg = .{ .i = ratio } },
                .{ .name = "SL", .arg = .none },                                                  .{ .name = "PREV", .arg = .none },
                .{ .name = "OFF", .arg = .{ .i = if (ratio == 2) 1 else 0 } },                    .{ .name = "ROWS", .arg = .{ .b = false } },
                .{ .name = "PDL", .arg = .{ .b = false } },
                .{ .name = "CARRY", .arg = if (ratio == 2) try e.buf("s.L{d}.carry", .{L}, .f32, &.{ 1, cw }) else .none },
                .{ .name = "c_stride", .arg = .{ .i = if (ratio == 2) cw else 0 } },              .{ .name = "SPLIT", .arg = .{ .b = ratio == 2 } },
            });
            try e.kvStore(lat, L, ratio);
            const kp = try e.buf("L.ix_k", .{}, .bf16, &.{ n, ID });
            try e.upstream(try e.fmt("L{d}.attn.ix_wk", .{L}), lat, hd, ID, kp);
            try e.triton("_index_k", .{ n, 1, 1 }, &.{
                .{ .name = "KP", .arg = kp },                                                    .{ .name = "k_stride", .arg = .{ .i = ID } },
                .{ .name = "W", .arg = try e.weight("L{d}.attn.ix_knorm", .{L}, .f32, &.{ID}) }, .{ .name = "CS", .arg = try e.ropeTable() },
                .{ .name = "cs_stride", .arg = .{ .i = 64 } },                                   .{ .name = "IK", .arg = try e.indexKeys(L) },
                .{ .name = "POS", .arg = try e.pos() },                                          .{ .name = "EPS", .arg = .{ .f = dec(e.cfg.eps) } },
                .{ .name = "RATIO", .arg = .{ .i = ratio } },                                    .{ .name = "PT", .arg = .none },
                .{ .name = "PSH", .arg = .{ .i = 0 } },                                          .{ .name = "KFP8", .arg = .{ .b = true } },
                .{ .name = "SL", .arg = .none },                                                 .{ .name = "PTS", .arg = .{ .i = 0 } },
                .{ .name = "ROWS", .arg = .{ .b = false } },                                     .{ .name = "PDL", .arg = .{ .b = false } },
            });
        }
        if (indexer) try e.index(L, qn, iw);
        // q group: wq_b
        const q = try e.buf("L.q", .{}, .bf16, &.{ n, H * hd });
        try e.fused("q", &.{.{ .w = try e.fmt("L{d}.attn.wq_b", .{L}), .x = qn, .y = q, .k = e.cfg.q_lora, .n = @intCast(H * hd) }});
        try e.prefetchO(L);
        try e.rope(q, H, hd, hd - 64);
        if (indexer) try e.kvStore(kvn, L, 0);
        // attention over the SWA ring and (a compressed layer) the source's pool at the source's selection
        const o = try e.buf("L.o", .{}, .bf16, &.{ n, H, hd });
        const swa = try e.pool(L, 0);
        var cv = try e.empty(.u8);
        var cs = try e.empty(.u8);
        var tok = try e.empty(.i32);
        var cnt = try e.empty(.i32);
        if (ratio > 0) {
            const p = try e.pool(kv_src.?, e.cfg.compressRatio(kv_src.?));
            cv = p[0];
            cs = p[1];
            const sel = try e.selection(e.cfg.indexSource(L).?);
            tok = sel[0];
            cnt = sel[1];
        }
        const rows_scr = n * 5 * H;
        try e.ext("tf_dsv41_attn_cuda_v1.attn", &.{
            try e.view(q.t.role, .bf16, &.{ n, H, hd }, &.{ H * hd, hd, 1 }, 0), cv, cs, tok, cnt, swa[0], swa[1],
            try e.buf("w.lo", .{}, .i32, &.{n}),                                 try e.empty(.i32),
            try e.pos(),                              try e.empty(.i64),
            try e.empty(.i32),                                                  try e.weight("L{d}.attn.sink", .{L}, .f32, &.{H}),
            try e.ropeTable(),                                                  try e.buf("s.attn.po", .{}, .f32, &.{rows_scr * 512}),
            try e.buf("s.attn.pm", .{}, .f32, &.{rows_scr}),                    try e.buf("s.attn.pl", .{}, .f32, &.{rows_scr}),
            o,                                                                  try e.buf("s.attn.ticket", .{}, .i32, &.{256}),
            .{ .i = 2 * @as(i64, e.cfg.window) },                               .{ .i = 0 },
            .{ .i = 0 },                                                        .{ .i = 4 },
        });
        // o: wo_a's groups over the heads' columns, then wo_b (this rank's partial)
        const g: usize = e.cfg.o_groups / e.o.world;
        const gk: i64 = @divExact(H * hd, @as(i64, @intCast(g)));
        const oa = try e.buf("L.oa", .{}, .bf16, &.{ n, @as(i64, @intCast(g)) * e.cfg.o_lora });
        var po: [8]Proj = undefined;
        for (po[0..g], 0..) |*p, i| p.* = .{
            .w = try e.fmt("L{d}.attn.wo_a.{d}", .{ L, i }),
            .x = try e.view(o.t.role, .bf16, &.{ n, gk }, &.{ H * hd, 1 }, @as(i64, @intCast(i)) * gk * 2),
            .y = oa,
            .col = i * e.cfg.o_lora,
            .k = @intCast(gk),
            .n = e.cfg.o_lora,
        };
        try e.fused("o", po[0..g]);
        try e.fused("wo_b", &.{.{ .w = try e.fmt("L{d}.attn.wo_b", .{L}), .x = oa, .y = try e.buf("L.part", .{}, .bf16, &.{ n, D }), .k = g * e.cfg.o_lora, .n = @intCast(D) }});
    }

    /// The indexer: query projection and rope, scores over the keys, top-k (a candidate source also picks candidate
    /// blocks for the Reindex layers; a Reindex layer scores only those).
    fn index(e: *Emitter, L: u32, qn: Arg, iw: Arg) !void {
        const n = e.n;
        const IH: i64 = e.cfg.index_heads;
        const ID: i64 = e.cfg.index_dim;
        const qi = try e.buf("L.qi", .{}, .bf16, &.{ n, IH * ID });
        try e.fused("qi", &.{.{ .w = try e.fmt("L{d}.attn.ix_wq_b", .{L}), .x = qn, .y = qi, .k = e.cfg.q_lora, .n = @intCast(IH * ID) }});
        try e.rope(qi, IH, ID, ID - 64);
        const reindex = e.cfg.mode(L) == .reindex;
        const src = if (reindex) e.cfg.kvSource(L).? else L;
        const ratio: i64 = e.cfg.compressRatio(src);
        const cbs: i64 = e.cfg.candidate_block_size;
        const nk: i64 = if (reindex) @as(i64, e.cfg.candidate_blocks) * cbs else @divFloor(e.end(), ratio);
        const scores = try e.buf("L.scores", .{}, .f32, &.{ n, nk });
        const cand = try e.buf("w.cand", .{}, .i32, &.{ n, e.cfg.candidate_blocks });
        const keys = if (reindex) cand else scores;
        const kst: i64 = if (reindex) e.cfg.candidate_blocks else nk;
        try e.triton("_scores", .{ n, std.math.divCeil(i64, nk, 64) catch unreachable, 1 }, &.{
            .{ .name = "QI", .arg = try e.view(qi.t.role, .bf16, &.{ n, IH, ID }, &.{ IH * ID, ID, 1 }, 0) },
            .{ .name = "W", .arg = iw },                                .{ .name = "w_stride", .arg = .{ .i = IH } },
            .{ .name = "IK", .arg = try e.indexKeys(src) },             .{ .name = "OUT", .arg = scores },
            .{ .name = "POS", .arg = try e.pos() },                     .{ .name = "KEYS", .arg = keys },
            .{ .name = "k_stride", .arg = .{ .i = kst } },              .{ .name = "NK", .arg = .{ .i = nk } },
            .{ .name = "o_stride", .arg = .{ .i = nk } },               .{ .name = "RATIO", .arg = .{ .i = ratio } },
            .{ .name = "H", .arg = .{ .i = IH } },                      .{ .name = "D", .arg = .{ .i = ID } },
            .{ .name = "BP", .arg = .{ .i = 64 } },                     .{ .name = "WS", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(IH))) } },
            .{ .name = "SCALE", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(ID))) } }, .{ .name = "GATHER", .arg = .{ .b = reindex } },
            .{ .name = "PT", .arg = .none },                            .{ .name = "PSH", .arg = .{ .i = 0 } },
            .{ .name = "KFP8", .arg = .{ .b = true } },                 .{ .name = "SL", .arg = .none },
            .{ .name = "PTS", .arg = .{ .i = 0 } },                     .{ .name = "ROWS", .arg = .{ .b = false } },
            .{ .name = "CBS", .arg = .{ .i = if (reindex) cbs else 0 } },
        });
        // attn_cuda topk: up to two jobs in one launch (the list repeats a single job)
        const sel = try e.selection(L);
        const k: i64 = e.cfg.index_topk;
        const pl = topkPlan(nk, null);
        var jobs: [2][8]Arg = undefined;
        jobs[0] = .{ scores, if (reindex) cand else try e.empty(.i32), sel[0], sel[1], .{ .i = nk }, .{ .i = k }, .{ .i = if (reindex) 3 else 0 }, .{ .i = pl[1] } };
        const blocks = e.cfg.isCandidateSource(L);
        jobs[1] = jobs[0];
        if (blocks) {
            const nb = std.math.divCeil(i64, nk, cbs) catch unreachable;
            jobs[1] = .{ scores, try e.empty(.i32), cand, try e.view(cand.t.role, .i32, &.{0}, &.{e.cfg.candidate_blocks}, 0), .{ .i = nb }, .{ .i = e.cfg.candidate_blocks }, .{ .i = 2 }, .{ .i = topkPlan(nb, pl[0])[1] } };
        }
        var args: std.ArrayList(Arg) = .empty;
        try args.appendSlice(e.a, &jobs[0]);
        try args.appendSlice(e.a, &jobs[1]);
        try args.appendSlice(e.a, &.{ .{ .i = if (blocks) 2 else 1 }, try e.pos(), .{ .i = ratio }, .{ .i = if (blocks or reindex) cbs else 1 }, .{ .i = n }, .{ .i = pl[0] } });
        try e.ext("tf_dsv41_attn_cuda_v1.topk", args.items);
    }

    // -----------------------------------------------------------------------------------------------------------
    // the window

    /// Every launch of a decode window of `n` rows from position `start` over `layers` (ascending, consecutive), and
    /// the head when `head` (the window ends the model).
    pub fn window(e: *Emitter, layers: []const u32, head: bool) !void {
        for (layers, 0..) |L, i| {
            if (i > 0) e.begin = .layer;
            e.layer_ratio = e.cfg.compressRatio(L);
            if (e.cfg.isEngram(L)) try moe.engram(e, L);
            try moe.mhc(e, L, "hc_attn", if (i == 0) .site1 else .boundary);
            try e.attention(L);
            try e.prefetchF(L);
            try moe.mhc(e, L, "hc_ffn", .boundary);
            try moe.moe(e, L);
            if (i + 1 < layers.len) try e.prefetchX(layers[i + 1]);
        }
        if (head) try moe.finish(e);
    }
};

/// The window's calls (arena-owned by `a`).
pub fn emit(a: std.mem.Allocator, cfg: *const Config, w: *const Widths, o: Options, layers: []const u32, n: i64, start: i64, head: bool) ![]calls.Call {
    var e: Emitter = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = n, .start = start };
    try e.window(layers, head);
    return e.out.items;
}

/// exl3 expert_loads.k2_range: the instance covering a layer's half-bit widths [lo, hi].
pub fn k2Range(lo: u32, hi: u32) ![2]i64 {
    const ranges = [_][2]u32{ .{ 8, 8 }, .{ 2, 10 }, .{ 2, 12 } };
    for (ranges) |r| if (r[0] <= lo and hi <= r[1]) return .{ r[0], r[1] };
    return error.NoInstance;
}
