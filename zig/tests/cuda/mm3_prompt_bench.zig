//! MiniMax-M3 prompt kernels against the served ones at TP2 rank 0's shapes: routed experts and dense rows, timed.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const cand = @import("exl3_candidate_tests.zig");
const Gpu = check.Gpu;
const Rig = cand.Rig;
const st = core.safetensors;
const ex = cuda.exl3_experts;
const gmk = cuda.exl3_gm;

const E = 128;
const slots = 4;
const D = 6144;
const I = 1536; // rank 0's rows of each expert's 3072
const I_full = 3072;
const old_rows = 256; // forward_plan.expert_rows
const new_rows = 2048; // forward_plan.prompt_expert_rows
const lm = "language_model.model.";

/// A routed-expert layer of the pack and the gate/up and down bits its synthetic stand-in takes.
const Spec = struct { layer: u32, gu: u32, down: u32 };
const specs = [_]Spec{ .{ .layer = 20, .gu = 3, .down = 3 }, .{ .layer = 3, .gu = 4, .down = 4 }, .{ .layer = 18, .gu = 3, .down = 4 } };

const Axis = enum { col, row };

/// A stored matrix (K x N) and rank 0's cut of it: output columns or input rows [0, keep).
const Mat = struct {
    k: usize,
    n: usize,
    axis: Axis,
    keep: usize,

    fn localK(m: Mat) usize {
        return if (m.axis == .row) m.keep else m.k;
    }
    fn localN(m: Mat) usize {
        return if (m.axis == .col) m.keep else m.n;
    }
};
const gate_mat: Mat = .{ .k = D, .n = I_full, .axis = .col, .keep = I };
const down_mat: Mat = .{ .k = I_full, .n = D, .axis = .row, .keep = I };

/// A dense projection at rank 0's shape, its synthetic bits, output type and aggregate group.
const Proj = struct { class: []const u8, name: []const u8, m: Mat, bits: u32, f32_out: bool, group: usize };
const groups = [_][]const u8{ "attn_qkvo", "dense_mlp", "shared_expert" };
const projs = [_]Proj{
    .{ .class = "q_proj", .name = "layers.0.self_attn.q_proj", .m = .{ .k = D, .n = 8192, .axis = .col, .keep = 4096 }, .bits = 6, .f32_out = false, .group = 0 },
    .{ .class = "k_proj", .name = "layers.0.self_attn.k_proj", .m = .{ .k = D, .n = 512, .axis = .col, .keep = 256 }, .bits = 8, .f32_out = false, .group = 0 },
    .{ .class = "v_proj", .name = "layers.0.self_attn.v_proj", .m = .{ .k = D, .n = 512, .axis = .col, .keep = 256 }, .bits = 8, .f32_out = false, .group = 0 },
    .{ .class = "o_proj", .name = "layers.0.self_attn.o_proj", .m = .{ .k = 8192, .n = D, .axis = .row, .keep = 4096 }, .bits = 6, .f32_out = true, .group = 0 },
    .{ .class = "mlp_gate", .name = "layers.0.mlp.gate_proj", .m = .{ .k = D, .n = 12288, .axis = .col, .keep = 6144 }, .bits = 6, .f32_out = false, .group = 1 },
    .{ .class = "mlp_up", .name = "layers.0.mlp.up_proj", .m = .{ .k = D, .n = 12288, .axis = .col, .keep = 6144 }, .bits = 6, .f32_out = false, .group = 1 },
    .{ .class = "mlp_down", .name = "layers.0.mlp.down_proj", .m = .{ .k = 12288, .n = D, .axis = .row, .keep = 6144 }, .bits = 6, .f32_out = true, .group = 1 },
    .{ .class = "shared_gate", .name = "layers.20.block_sparse_moe.shared_experts.gate_proj", .m = gate_mat, .bits = 6, .f32_out = false, .group = 2 },
    .{ .class = "shared_up", .name = "layers.20.block_sparse_moe.shared_experts.up_proj", .m = gate_mat, .bits = 6, .f32_out = false, .group = 2 },
    .{ .class = "shared_down", .name = "layers.20.block_sparse_moe.shared_experts.down_proj", .m = down_mat, .bits = 6, .f32_out = true, .group = 2 },
};

/// The pack's weight map and up to four mapped shards; a tensor's bytes are copied before the next lookup.
const Pack = struct {
    a: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    text: []u8,
    json: std.json.Parsed(std.json.Value),
    map: std.json.ObjectMap,
    open: [4]?Open = @splat(null),
    next: usize = 0,

    const Open = struct { shard: []const u8, file: st.File };

    fn init(gpu: Gpu, dir: []const u8) !Pack {
        const a = gpu.gpa;
        const path = try std.fmt.allocPrint(a, "{s}/model.safetensors.index.json", .{dir});
        defer a.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(st.index.max_bytes));
        errdefer a.free(text);
        const json = try std.json.parseFromSlice(std.json.Value, a, text, .{});
        errdefer json.deinit();
        return .{ .a = a, .io = gpu.io, .dir = dir, .text = text, .json = json, .map = try st.index.weightMap(json.value) };
    }

    fn deinit(p: *Pack) void {
        p.drop();
        p.json.deinit();
        p.a.free(p.text);
    }

    /// Closes every shard and returns its pages (unified memory needs the room).
    fn drop(p: *Pack) void {
        for (&p.open) |*slot| {
            if (slot.*) |*o| {
                o.file.release();
                o.file.close(p.io);
            }
            slot.* = null;
        }
    }

    fn get(p: *Pack, name: []const u8) !st.Tensor {
        const shard = (p.map.get(name) orelse return error.TensorNotInIndex).string;
        for (&p.open) |*slot| {
            if (slot.*) |*o| if (std.mem.eql(u8, o.shard, shard)) return o.file.get(name) orelse return error.TensorNotInShard;
        }
        const slot = &p.open[p.next];
        p.next = (p.next + 1) % p.open.len;
        if (slot.*) |*o| {
            o.file.release();
            o.file.close(p.io);
            slot.* = null;
        }
        const path = try std.fmt.allocPrint(p.a, "{s}/{s}", .{ p.dir, shard });
        defer p.a.free(path);
        slot.* = .{ .shard = shard, .file = try st.File.open(p.a, p.io, path) };
        return slot.*.?.file.get(name) orelse return error.TensorNotInShard;
    }
};

fn readScale(p: *Pack, a: std.mem.Allocator, name: []const u8, part_name: []const u8, whole: usize, keep: usize) ![]f16 {
    var buf: [192]u8 = undefined;
    const t = try p.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ name, part_name }));
    if (t.dtype != .f16 or t.bytes.len != 2 * whole) return error.UnsupportedScale;
    const out = try a.alloc(f16, keep);
    @memcpy(std.mem.sliceAsBytes(out), t.bytes[0 .. 2 * keep]);
    return out;
}

/// Rank 0's share of one stored matrix: the cut's trellis tiles (each K-tile row's run), suh and svh cut to match.
fn readPart(p: *Pack, a: std.mem.Allocator, name: []const u8, m: Mat) !cand.Host {
    var buf: [192]u8 = undefined;
    const t = try p.get(try std.fmt.bufPrint(&buf, "{s}.trellis", .{name}));
    if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != m.k / 16 or t.dim(1) != m.n / 16 or t.dim(2) % 16 != 0) return error.UnsupportedTrellis;
    const tile = 2 * t.dim(2);
    const kt = m.localK() / 16;
    const per = m.localN() / 16 * tile;
    const row = t.dim(1) * tile;
    const trellis = try a.alloc(i16, kt * per / 2);
    errdefer a.free(trellis);
    const dst = std.mem.sliceAsBytes(trellis);
    for (0..kt) |r| @memcpy(dst[r * per ..][0..per], t.bytes[r * row ..][0..per]);
    const bits: u32 = @intCast(t.dim(2) / 16);
    const suh = try readScale(p, a, name, "suh", m.k, m.localK());
    errdefer a.free(suh);
    const svh = try readScale(p, a, name, "svh", m.n, m.localN());
    return .{ .k = m.localK(), .n = m.localN(), .bits = bits, .book = .mul1, .trellis = trellis, .suh = suh, .svh = svh };
}

fn part(pack: ?*Pack, a: std.mem.Allocator, r: std.Random, name: []const u8, m: Mat, bits: u32) !cand.Host {
    if (pack) |p| return readPart(p, a, name, m);
    return cand.randomMatrix(a, r, .{ .k = m.localK(), .n = m.localN(), .bits = bits, .book = .mul1 });
}

/// One layer's rank-0 routed experts on the device, laid out as exl3_experts.Layer documents.
const Routed = struct {
    rig: Rig,
    layer: ex.Layer,
    bits: [3]u32,
    pack: bool,

    fn widths(b: *const Routed) gmk.Widths {
        return .{ .gu = 2 * b.bits[0], .down = 2 * b.bits[2] };
    }
};

fn buildExperts(gpu: Gpu, pack: ?*Pack, spec: Spec, r: std.Random) !Routed {
    const a = gpu.gpa;
    var rig: Rig = .{ .gpu = gpu };
    errdefer rig.deinit();
    const kinds = [3][]const u8{ "w1", "w3", "w2" };
    const mats = [3]Mat{ gate_mat, gate_mat, down_mat };
    const synth = [3]u32{ spec.gu, spec.gu, spec.down };
    // scales in Layer order: suh_g, svh_g, suh_u, svh_u, suh_d, svh_d
    const lens = [6]usize{ D, I, D, I, I, D };
    var scales: [6][]f16 = undefined;
    var made: usize = 0;
    defer for (scales[0..made]) |s| a.free(s);
    for (&scales, lens) |*s, n| {
        s.* = try a.alloc(f16, E * n);
        made += 1;
    }
    var tables: [3][E]i64 = undefined;
    var k2: [3][E]i32 = undefined;
    var base: [3]u64 = undefined;
    var per: [3]usize = undefined;
    var bits: [3]u32 = undefined;
    var buf: [192]u8 = undefined;
    for (0..E) |e| for (0..3) |j| {
        const name = try std.fmt.bufPrint(&buf, lm ++ "layers.{d}.block_sparse_moe.experts.{d}.{s}", .{ spec.layer, e, kinds[j] });
        const h = try part(pack, a, r, name, mats[j], synth[j]);
        defer cand.freeHost(a, h);
        if (e == 0) {
            bits[j] = h.bits;
            per[j] = 2 * h.trellis.len;
            base[j] = try rig.zeros(E * per[j]);
        }
        if (h.bits != bits[j] or 2 * h.trellis.len != per[j]) return error.MixedWidths;
        const at = base[j] + @as(u64, e * per[j]);
        const dst: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = at, .len = per[j] };
        try dst.upload(0, std.mem.sliceAsBytes(h.trellis));
        tables[j][e] = @intCast(at);
        k2[j][e] = @intCast(2 * h.bits);
        @memcpy(scales[2 * j][e * h.k ..][0..h.k], h.suh);
        @memcpy(scales[2 * j + 1][e * h.n ..][0..h.n], h.svh);
    };
    if (bits[0] != bits[1]) return error.MixedWidths;
    var tp: [3]u64 = undefined;
    var kp: [3]u64 = undefined;
    var sp: [6]u64 = undefined;
    for (0..3) |j| {
        tp[j] = try rig.put(&tables[j]);
        kp[j] = try rig.put(&k2[j]);
    }
    for (scales, &sp) |s, *p| p.* = try rig.put(s);
    return .{ .rig = rig, .bits = bits, .pack = pack != null, .layer = .{
        .gate = tp[0],
        .up = tp[1],
        .down = tp[2],
        .gate_k2 = kp[0],
        .up_k2 = kp[1],
        .down_k2 = kp[2],
        .suh_g = sp[0],
        .svh_g = sp[1],
        .suh_u = sp[2],
        .svh_u = sp[3],
        .suh_d = sp[4],
        .svh_d = sp[5],
        .experts = E,
        .dims = D,
        .width = I,
        .codebook = .mul1,
        .act_mode = 2,
        .limit = 7.0,
        .alpha = 1.702,
        .up_bias = 1.0,
    } };
}

/// The pack's layer, or synthetic words and scales of the spec's widths when there is no pack or a read fails.
fn loadExperts(gpu: Gpu, pack: ?*Pack, spec: Spec, r: std.Random) !Routed {
    if (pack) |p| {
        defer p.drop();
        if (buildExperts(gpu, p, spec, r)) |b| return b else |err| std.debug.print("NOTE experts layer {d}: pack read failed ({s}), synthetic weights\n", .{ spec.layer, @errorName(err) });
    }
    return buildExperts(gpu, null, spec, r);
}

const Inputs = struct { x: u64, pick: u64, wts: u64, out: u64 };

/// Distinct experts a row, uniform or Zipf(0.8) by expert id (rejection over the CDF), weights summing to 1.
fn route(r: std.Random, pick: []i32, wts: []f32, zipf: bool) void {
    var cdf: [E]f64 = undefined;
    var acc: f64 = 0;
    for (&cdf, 0..) |*c, e| {
        acc += if (zipf) 1.0 / std.math.pow(f64, @floatFromInt(e + 1), 0.8) else 1.0;
        c.* = acc;
    }
    for (0..pick.len / slots) |row| {
        const p = pick[row * slots ..][0..slots];
        var n: usize = 0;
        while (n < slots) {
            const u = r.float(f64) * acc;
            var e: usize = 0;
            while (e + 1 < E and cdf[e] < u) e += 1;
            const ei: i32 = @intCast(e);
            if (std.mem.indexOfScalar(i32, p[0..n], ei) != null) continue;
            p[n] = ei;
            n += 1;
        }
        const w = wts[row * slots ..][0..slots];
        var sum: f32 = 0;
        for (w) |*v| {
            v.* = 0.05 + r.float(f32);
            sum += v.*;
        }
        for (w) |*v| v.* /= sum;
    }
}

/// One timed routed-experts pass over `rows` rows: the served kernels in 256-row calls, or x3gm in 2048-row calls.
const RoutedJob = struct {
    path: enum { old, new },
    old: *const ex.Experts,
    gm: *const gmk.Experts,
    d: *const cuda.Driver,
    s: cuda.Stream,
    l: ex.Layer,
    w: gmk.Widths = .{},
    in: Inputs,
    rows: usize = 0,
    osc: ex.Scratch,
    nsc: gmk.Scratch,

    fn call(j: *const RoutedJob) !void {
        const step: usize = if (j.path == .old) old_rows else new_rows;
        var r: usize = 0;
        while (r < j.rows) : (r += step) {
            const n = @min(step, j.rows - r);
            const at: u64 = r;
            const x = j.in.x + at * D * 2;
            const pk = j.in.pick + at * slots * 4;
            const wt = j.in.wts + at * slots * 4;
            const out = j.in.out + at * D * 4;
            switch (j.path) {
                .old => try j.old.run(j.s, j.l, x, D, .bf16, pk, wt, n, out, j.osc),
                .new => try j.gm.run(j.d, j.s, j.l, j.w, x, D, pk, wt, n, out, j.nsc),
            }
        }
    }
};

/// One timed dense prompt call: the 128-row decode blocks, or pfdense once usePromptTiles loaded it.
const DenseJob = struct {
    de: *cuda.exl3.Dense,
    s: cuda.Stream,
    w: cuda.exl3.Weight,
    x: u64,
    rows: usize,
    y: u64,
    ydt: cuda.exl3.DType,

    fn call(j: *const DenseJob) !void {
        try j.de.prompt(j.s, j.w, j.x, .bf16, j.rows, j.y, j.ydt);
    }
};

/// The median ms of `reps` event-timed calls after two warm-ups.
fn timeIt(s: cuda.Stream, d: *const cuda.Driver, job: anytype, reps: usize) !f64 {
    var t0 = try cuda.Event.init(d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(d, true);
    defer t1.deinit();
    for (0..2) |_| try job.call();
    var ms: [64]f64 = undefined;
    const n = @min(@max(reps, 1), ms.len);
    for (0..n) |i| {
        try t0.record(s);
        try job.call();
        try t1.record(s);
        try t1.synchronize();
        ms[i] = try cuda.Event.elapsedMs(t0, t1);
    }
    return check.median(ms[0..n]);
}

fn tokS(rows: usize, ms: f64) f64 {
    return @as(f64, @floatFromInt(rows)) * 1000.0 / ms;
}

fn report(class: []const u8, rows: usize, routing: []const u8, old: f64, new: f64, tail: []const u8) void {
    std.debug.print("BENCH {s} rows={d} routing={s} old_ms={d:.3} new_ms={d:.3} old_tok_s={d:.0} new_tok_s={d:.0} speedup={d:.2}x {s}\n", .{ class, rows, routing, old, new, tokS(rows, old), tokS(rows, new), old / new, tail });
}

fn benchExperts(gpu: Gpu, s: cuda.Stream, pack: ?*Pack, rows: []const usize, reps: usize, in: Inputs, pb: cuda.DeviceBuffer, wb: cuda.DeviceBuffer, r: std.Random) !void {
    const a = gpu.gpa;
    var old = try ex.Experts.load(gpu.d);
    defer old.unload();
    var gm = try gmk.Experts.load(gpu.d);
    defer gm.unload();
    const max_rows = std.mem.max(usize, rows);
    const pick = try a.alloc(i32, max_rows * slots);
    defer a.free(pick);
    const wts = try a.alloc(f32, max_rows * slots);
    defer a.free(wts);
    for (specs) |spec| {
        var b = try loadExperts(gpu, pack, spec, r);
        defer b.rig.deinit();
        var cb: [32]u8 = undefined;
        const class = try std.fmt.bufPrint(&cb, "experts_l{d}", .{spec.layer});
        const ob = try ex.scratchBytes(b.layer, old_rows, slots);
        const nb = gmk.scratchBytes(b.layer, new_rows, slots);
        std.debug.print("MEM {s} old_scratch_bytes={d} new_scratch_bytes={d}\n", .{ class, ob, nb });
        var osb = try cuda.DeviceBuffer.alloc(gpu.d, ob);
        defer osb.free();
        var nsb = try cuda.DeviceBuffer.alloc(gpu.d, nb);
        defer nsb.free();
        const base: RoutedJob = .{
            .path = .old,
            .old = &old,
            .gm = &gm,
            .d = gpu.d,
            .s = s,
            .l = b.layer,
            .in = in,
            .osc = try ex.scratch(osb.ptr, b.layer, old_rows, slots),
            .nsc = gmk.scratch(nsb.ptr, b.layer, new_rows, slots),
        };
        var tb: [96]u8 = undefined;
        for ([_]bool{ false, true }) |zipf| {
            route(r, pick, wts, zipf);
            try pb.upload(0, std.mem.sliceAsBytes(pick));
            try wb.upload(0, std.mem.sliceAsBytes(wts));
            for (rows) |R| {
                var jo = base;
                jo.rows = R;
                var jn = jo;
                jn.path = .new;
                jn.w = b.widths();
                var j2 = jn;
                j2.w = .{};
                const t_old = try timeIt(s, gpu.d, &jo, reps);
                const t_new = try timeIt(s, gpu.d, &jn, reps);
                const t_v2 = try timeIt(s, gpu.d, &j2, reps);
                const tail = try std.fmt.bufPrint(&tb, "v2_ms={d:.3} weights={s} bits={d}/{d}/{d}", .{ t_v2, if (b.pack) "pack" else "synthetic", b.bits[0], b.bits[1], b.bits[2] });
                report(class, R, if (zipf) "zipf" else "uniform", t_old, t_new, tail);
            }
        }
    }
}

/// The pack's share of a dense matrix, or random words of the class's bits when there is no pack or a read fails.
fn densePart(pack: ?*Pack, a: std.mem.Allocator, r: std.Random, p: Proj, from_pack: *bool) !cand.Host {
    var buf: [192]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, lm ++ "{s}", .{p.name});
    from_pack.* = false;
    if (pack) |pk| {
        if (readPart(pk, a, name, p.m)) |h| {
            from_pack.* = true;
            return h;
        } else |err| std.debug.print("NOTE {s}: pack read failed ({s}), synthetic weights\n", .{ p.class, @errorName(err) });
    }
    return cand.randomMatrix(a, r, .{ .k = p.m.localK(), .n = p.m.localN(), .bits = p.bits, .book = .mul1 });
}

fn benchDense(gpu: Gpu, s: cuda.Stream, pack: ?*Pack, rows: []const usize, reps: usize, in: Inputs, r: std.Random) !void {
    const a = gpu.gpa;
    var rig: Rig = .{ .gpu = gpu };
    defer rig.deinit();
    var ws: [projs.len]cuda.exl3.Weight = undefined;
    var from_pack: [projs.len]bool = undefined;
    for (projs, &ws, &from_pack) |p, *w, *fp| {
        const h = try densePart(pack, a, r, p, fp);
        defer cand.freeHost(a, h);
        const halves = 2 * h.bits;
        const at = cuda.exl3.strides(h.k, h.n, halves, false);
        w.* = .{ .trellis = try rig.put(h.trellis), .suh = try rig.put(h.suh), .svh = try rig.put(h.svh), .k = @intCast(h.k), .n = @intCast(h.n), .halves = halves, .codebook = .mul1, .stride_k = at.k, .stride_nb = at.nb };
    }
    if (pack) |p| p.drop();
    var old = try cuda.exl3.Dense.load(gpu.d);
    defer old.unload();
    var new = try cuda.exl3.Dense.load(gpu.d);
    defer new.unload();
    try new.usePromptTiles();
    var tb: [64]u8 = undefined;
    for (rows) |R| {
        var sums: [groups.len][2]f64 = @splat(.{ 0, 0 });
        for (projs, ws, from_pack) |p, w, fp| {
            const jo: DenseJob = .{ .de = &old, .s = s, .w = w, .x = in.x, .rows = R, .y = in.out, .ydt = if (p.f32_out) .f32 else .bf16 };
            var jn = jo;
            jn.de = &new;
            const t_old = try timeIt(s, gpu.d, &jo, reps);
            const t_new = try timeIt(s, gpu.d, &jn, reps);
            sums[p.group][0] += t_old;
            sums[p.group][1] += t_new;
            const tail = try std.fmt.bufPrint(&tb, "weights={s} bits={d} k={d} n={d}", .{ if (fp) "pack" else "synthetic", w.halves / 2, w.k, w.n });
            report(p.class, R, "none", t_old, t_new, tail);
        }
        for (groups, sums) |g, t| report(g, R, "none", t[0], t[1], "sum=gemms");
    }
}

/// `tf-cuda-test mm3-prompt-bench PACK_DIR [rows_csv] [reps]`: old against new prompt kernels, alternating.
pub fn run(gpu: check.Gpu, pack_dir: []const u8, rest: []const []const u8) !void {
    const a = gpu.gpa;
    const rows_arg = if (rest.len > 0) rest[0] else "2048,8192,32768";
    var list: [16]usize = undefined;
    var nrows: usize = 0;
    var it = std.mem.tokenizeScalar(u8, rows_arg, ',');
    while (it.next()) |t| {
        if (nrows == list.len) return error.TooManyRowCounts;
        list[nrows] = try std.fmt.parseInt(usize, t, 10);
        if (list[nrows] == 0) return error.BadRowCount;
        nrows += 1;
    }
    if (nrows == 0) return error.BadRowCount;
    const reps: usize = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 5;
    const rows = list[0..nrows];
    const max_rows = std.mem.max(usize, rows);
    var name_buf: [256]u8 = undefined;
    std.debug.print("mm3-prompt-bench: GPU {s}, {d} SMs; rows {s}, {d} reps (median after 2 warm-ups), TP2 rank 0 shapes\n", .{ try gpu.ctx.name(&name_buf), try gpu.ctx.attribute(.multiprocessor_count), rows_arg, reps });

    var pack_v: ?Pack = Pack.init(gpu, pack_dir) catch |err| blk: {
        std.debug.print("NOTE pack index unreadable ({s}): every matrix synthetic\n", .{@errorName(err)});
        break :blk null;
    };
    defer if (pack_v) |*p| p.deinit();
    const pack: ?*Pack = if (pack_v) |*p| p else null;

    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var xb = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * D * 2);
    defer xb.free();
    var pb = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * slots * 4);
    defer pb.free();
    var wb = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * slots * 4);
    defer wb.free();
    var ob = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * D * 4);
    defer ob.free();
    var prng = std.Random.DefaultPrng.init(0x3A11_F00D);
    const r = prng.random();
    var row: usize = 0;
    while (row < max_rows) : (row += new_rows) {
        const n = @min(new_rows, max_rows - row);
        const x = try cand.randomRows(a, r, n, D);
        defer a.free(x);
        try xb.upload(row * D * 2, std.mem.sliceAsBytes(x));
    }
    const in: Inputs = .{ .x = xb.ptr, .pick = pb.ptr, .wts = wb.ptr, .out = ob.ptr };
    try benchExperts(gpu, s, pack, rows, reps, in, pb, wb, r);
    try benchDense(gpu, s, pack, rows, reps, in, r);
    check.pass("mm3-prompt-bench: {d} expert layers x 2 routings and {d} dense projections at {d} row counts", .{ specs.len, projs.len, nrows });
}
