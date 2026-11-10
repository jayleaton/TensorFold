//! Routed experts: GPTQ int4 packing, the NVFP4 main-layer packing and the MTP layer's re-quantized experts.
const std = @import("std");
const lay = @import("cuda_layouts.zig");
const int4 = @import("cuda_int4.zig");
const wt = @import("cuda_weights.zig");
const wl = @import("cuda_weights_loader.zig");

const Loader = wl.Loader;
const Experts4 = wt.Experts4;
const parallel = wl.parallel;

/// GPTQ int4 experts for fn_int4.cu: gate/up rows [lo, hi) per expert, down's columns [lo, hi) (64-groups if cut).
pub fn int4Experts(L: *Loader, path: []const u8, name: []const u8, lo: usize, hi: usize) !int4.Experts {
    const c = L.c;
    const e = c.experts;
    const d = c.hidden;
    const full = c.moe_width;
    const ni = hi - lo;
    const gs_down: usize = if (lo % int4.checkpoint_group != 0 or ni % int4.checkpoint_group != 0) 64 else 128;
    const uw = int4.wordsOf(ni, d);
    const us = int4.scalesOf(ni, d, 128);
    const dw = int4.wordsOf(d, ni);
    const ds = int4.scalesOf(d, ni, gs_down);
    const up_w = try L.gpa.alloc(u32, e * 2 * uw);
    defer L.gpa.free(up_w);
    const up_s = try L.gpa.alloc(u16, e * 2 * us);
    defer L.gpa.free(up_s);
    const dn_w = try L.gpa.alloc(u32, e * dw);
    defer L.gpa.free(dn_w);
    const dn_s = try L.gpa.alloc(u16, e * ds);
    defer L.gpa.free(dn_s);
    const Job = struct { src: int4.Source, sl: int4.Slice, w: []u32, s: []u16 };
    const jobs = try L.gpa.alloc(Job, e * 3);
    defer L.gpa.free(jobs);
    for (0..e) |x| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
        const down = m == 2;
        var b: [3][256]u8 = undefined;
        const kf = if (down) full else d;
        const nf = if (down) d else full;
        const qw = try L.expect(try std.fmt.bufPrint(&b[0], "{s}.experts.{d}.{s}.qweight", .{ name, x, proj }), .i32, &.{ kf / 8, nf });
        const sc = try L.expect(try std.fmt.bufPrint(&b[1], "{s}.experts.{d}.{s}.scales", .{ name, x, proj }), .f16, &.{ kf / int4.checkpoint_group, nf });
        const qz = try L.expect(try std.fmt.bufPrint(&b[2], "{s}.experts.{d}.{s}.qzeros", .{ name, x, proj }), .i32, &.{ kf / int4.checkpoint_group, nf / 8 });
        if (!int4.zerosAreSymmetric(qz.bytes)) {
            std.log.err("{s}{s}: zero points other than 8 (AutoGPTQ v1 qzeros 7); the engine reads symmetric GPTQ", .{ L.prefix, b[2][0 .. std.mem.indexOfScalar(u8, &b[2], 0) orelse 0] });
            return error.UnsupportedQuantization;
        }
        const src: int4.Source = .{ .qweight = qw.bytes, .scales = sc.bytes, .n_full = nf, .k_full = kf };
        jobs[x * 3 + m] = if (down)
            .{ .src = src, .sl = .{ .n0 = 0, .n = d, .k0 = lo, .k = ni, .gs = gs_down }, .w = dn_w[x * dw ..][0..dw], .s = dn_s[x * ds ..][0..ds] }
        else
            .{ .src = src, .sl = .{ .n0 = lo, .n = ni, .k0 = 0, .k = d, .gs = 128 }, .w = up_w[(x * 2 + m) * uw ..][0..uw], .s = up_s[(x * 2 + m) * us ..][0..us] };
    };
    const Ctx = struct {
        jobs: []const Job,
        failed: *std.atomic.Value(bool),
        fn run(ctx: @This(), i: usize) void {
            const j = ctx.jobs[i];
            int4.packWords(j.src, j.sl, j.w) catch ctx.failed.store(true, .monotonic);
            int4.packScales(j.src, j.sl, j.s) catch ctx.failed.store(true, .monotonic);
        }
    };
    var failed = std.atomic.Value(bool).init(false);
    parallel(L.o.threads, jobs.len, Ctx{ .jobs = jobs, .failed = &failed }, Ctx.run);
    if (failed.load(.monotonic)) return error.UnexpectedTensor;
    var pb: [192]u8 = undefined;
    var x: int4.Experts = .{ .count = @intCast(e), .width = @intCast(ni), .dims = @intCast(d), .gs_down = @intCast(gs_down) };
    x.up = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.int4.up", .{path}), "int32", &.{ e, 2, uw }, std.mem.sliceAsBytes(up_w));
    L.out.ours();
    x.up_s = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.int4.up_scales", .{path}), "float16", &.{ e, 2, us }, std.mem.sliceAsBytes(up_s));
    L.out.ours();
    x.down = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.int4.down", .{path}), "int32", &.{ e, dw }, std.mem.sliceAsBytes(dn_w));
    L.out.ours();
    x.down_s = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.int4.down_scales", .{path}), "float16", &.{ e, ds }, std.mem.sliceAsBytes(dn_s));
    L.out.ours();
    return x;
}

const Packed = struct { blocks: u64, scale2: []f32 };

/// Main-layer routed experts as shipped (U8 codes, e4m3 scales, fp32 scale_2) packed by `_pack` at the rank slice.
pub fn packRouted(L: *Loader, path: []const u8, name: []const u8, projs: []const []const u8, n: usize, k: usize, lo: usize, down: bool) !Packed {
    const c = L.c;
    const e = c.experts;
    const m_count = projs.len;
    const full_n = if (down) n else c.moe_width;
    const full_k = if (down) c.moe_width else k;
    const cell = (n / 32) * (k / 32) * m_count * lay.words_a_block;
    const host = std.mem.bytesAsSlice(u32, try L.staging(e * cell * 4));
    const s2 = try L.gpa.alloc(f32, e * m_count);
    errdefer L.gpa.free(s2);
    const Job = struct { src: lay.Fp4Rows, e: usize, m: usize };
    const jobs = try L.gpa.alloc(Job, e * m_count);
    defer L.gpa.free(jobs);
    for (0..e) |x| for (projs, 0..) |proj, m| {
        var b: [3][256]u8 = undefined;
        const w = try L.expect(try std.fmt.bufPrint(&b[0], "{s}.experts.{d}.{s}.weight", .{ name, x, proj }), .u8, &.{ full_n, full_k / 2 });
        const s = try L.expect(try std.fmt.bufPrint(&b[1], "{s}.experts.{d}.{s}.weight_scale", .{ name, x, proj }), .f8_e4m3, &.{ full_n, full_k / 16 });
        const t2 = try L.expect(try std.fmt.bufPrint(&b[2], "{s}.experts.{d}.{s}.weight_scale_2", .{ name, x, proj }), .f32, &.{});
        s2[x * m_count + m] = @bitCast(std.mem.readInt(u32, t2.bytes[0..4], .little));
        // the activations' input_scale: ModelOpt's W4A4 calibration; the engine runs bf16 activations
        _ = try L.expect(try std.fmt.bufPrint(&b[2], "{s}.experts.{d}.{s}.input_scale", .{ name, x, proj }), .f32, &.{});
        L.out.w.skipped += 1;
        jobs[x * m_count + m] = .{ .src = .{ .words = w.bytes, .word_pitch = full_k / 2, .scales = s.bytes, .scale_pitch = full_k / 16, .row0 = if (down) 0 else lo, .k0 = if (down) lo else 0 }, .e = x, .m = m };
    };
    const Ctx = struct {
        jobs: []const Job,
        out: []u32,
        cell: usize,
        n: usize,
        k: usize,
        m_count: usize,
        fn run(ctx: @This(), i: usize) void {
            const j = ctx.jobs[i];
            lay.packExpert(j.src, ctx.n, ctx.k, ctx.out[j.e * ctx.cell ..][0..ctx.cell], j.m, ctx.m_count);
        }
    };
    parallel(L.o.threads, jobs.len, Ctx{ .jobs = jobs, .out = @alignCast(host), .cell = cell, .n = n, .k = k, .m_count = m_count }, Ctx.run);
    const ptr = try L.out.whole(path, "int32", &.{ e, n / 32, k / 32, m_count, lay.words_a_block }, L.host.items);
    return .{ .blocks = ptr, .scale2 = s2 };
}

/// MTP FP8_PB_WO experts: dequantized to bf16, rank-sliced, ModelOpt-quantized and packed (moe4_from_bf16).
pub fn mtpExperts(L: *Loader, path: []const u8, name: []const u8, x: *Experts4, lo: usize, hi: usize) !void {
    const c = L.c;
    const e = c.experts;
    const d = c.hidden;
    const w = c.moe_width;
    const ni = hi - lo;
    // FP8_PB_WO (bf16 or fp32 block scales), or bf16 as stored (INT4-AutoRound's MTP layer)
    const Src = struct { codes: []const u8, inv: []const u8, kind: enum { fp8_bf16, fp8_f32, bf16 } };
    const srcs = try L.gpa.alloc(Src, e * 3);
    defer L.gpa.free(srcs);
    for (0..e) |i| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
        var b: [2][256]u8 = undefined;
        const shape = if (m == 2) [2]usize{ d, w } else [2]usize{ w, d };
        const wname = try std.fmt.bufPrint(&b[0], "{s}.experts.{d}.{s}.weight", .{ name, i, proj });
        if (L.c.int4ar() and (try L.get(wname)).dtype == .bf16) {
            const codes = try L.expect(wname, .bf16, &shape);
            srcs[i * 3 + m] = .{ .codes = codes.bytes, .inv = &.{}, .kind = .bf16 };
            continue;
        }
        const codes = try L.expect(wname, .f8_e4m3, &shape);
        const iname = try std.fmt.bufPrint(&b[1], "{s}.experts.{d}.{s}.weight_scale_inv", .{ name, i, proj });
        const inv_t = try L.get(iname);
        const inv = try L.expect(iname, if (L.c.int4ar() and inv_t.dtype == .f32) .f32 else .bf16, &.{ (shape[0] + 127) / 128, shape[1] / 128 });
        srcs[i * 3 + m] = .{ .codes = codes.bytes, .inv = inv.bytes, .kind = if (inv.dtype == .f32) .fp8_f32 else .fp8_bf16 };
    };
    const up_cell = (ni / 32) * (d / 32) * 2 * lay.words_a_block;
    const dn_cell = (d / 32) * (ni / 32) * lay.words_a_block;
    const up = try L.gpa.alloc(u32, e * up_cell);
    defer L.gpa.free(up);
    const dn = try L.gpa.alloc(u32, e * dn_cell);
    defer L.gpa.free(dn);
    const us = try L.gpa.alloc(f32, e * 2);
    defer L.gpa.free(us);
    const ds = try L.gpa.alloc(f32, e);
    defer L.gpa.free(ds);
    const Ctx = struct {
        srcs: []const Src,
        up: []u32,
        dn: []u32,
        us: []f32,
        ds: []f32,
        d: usize,
        w: usize,
        lo: usize,
        ni: usize,
        up_cell: usize,
        dn_cell: usize,
        failed: *std.atomic.Value(bool),
        fn run(ctx: @This(), job: usize) void {
            ctx.one(job) catch ctx.failed.store(true, .monotonic);
        }
        fn one(ctx: @This(), job: usize) !void {
            const i = job / 3;
            const m = job % 3;
            const s = ctx.srcs[job];
            const n = if (m == 2) ctx.d else ctx.w;
            const k = if (m == 2) ctx.w else ctx.d;
            const gpa = std.heap.page_allocator;
            const deq = try gpa.alloc(u16, n * k);
            defer gpa.free(deq);
            switch (s.kind) {
                .bf16 => for (deq, 0..) |*v, j| {
                    v.* = std.mem.readInt(u16, s.codes[2 * j ..][0..2], .little);
                },
                .fp8_bf16 => {
                    const inv16 = try gpa.alloc(u16, s.inv.len / 2);
                    defer gpa.free(inv16);
                    for (inv16, 0..) |*v, j| v.* = std.mem.readInt(u16, s.inv[2 * j ..][0..2], .little);
                    lay.dequantFp8(s.codes, inv16, n, k, deq);
                },
                .fp8_f32 => {
                    // Python weight_bf16: (codes.float() * scale per 64 inputs).to(bf16)
                    const kb = k / 128;
                    for (0..n) |r| for (0..k) |j| {
                        const sc: f32 = @bitCast(std.mem.readInt(u32, s.inv[4 * ((r / 128) * kb + j / 128) ..][0..4], .little));
                        deq[r * k + j] = lay.bf16Rne(lay.e4m3ToF32(s.codes[r * k + j]) * sc);
                    };
                },
            }
            // the rank's slice: gate/up rows [lo, lo + ni), down's columns [lo, lo + ni)
            const rows = if (m == 2) deq else deq[ctx.lo * k .. (ctx.lo + ctx.ni) * k];
            const sn = if (m == 2) n else ctx.ni;
            const sk = if (m == 2) ctx.ni else k;
            const k0 = if (m == 2) ctx.lo else 0;
            const g = if (m == 2) globalCols(rows, k, k0, sn, sk) else lay.fp4Global(rows);
            const words = try gpa.alloc(u8, sn * sk / 2);
            defer gpa.free(words);
            const scales = try gpa.alloc(u8, sn * sk / 16);
            defer gpa.free(scales);
            lay.quantizeFp4(rows, k, k0, sn, sk, g, words, scales);
            const src: lay.Fp4Rows = .{ .words = words, .word_pitch = sk / 2, .scales = scales, .scale_pitch = sk / 16 };
            if (m == 2) {
                lay.packExpert(src, sn, sk, ctx.dn[i * ctx.dn_cell ..][0..ctx.dn_cell], 0, 1);
                ctx.ds[i] = g;
            } else {
                lay.packExpert(src, sn, sk, ctx.up[i * ctx.up_cell ..][0..ctx.up_cell], m, 2);
                ctx.us[i * 2 + m] = g;
            }
        }
    };
    var failed = std.atomic.Value(bool).init(false);
    parallel(L.o.threads, e * 3, Ctx{ .srcs = srcs, .up = up, .dn = dn, .us = us, .ds = ds, .d = d, .w = w, .lo = lo, .ni = ni, .up_cell = up_cell, .dn_cell = dn_cell, .failed = &failed }, Ctx.run);
    if (failed.load(.monotonic)) return error.OutOfMemory;
    var pb: [192]u8 = undefined;
    x.up = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.experts.routed_experts.up", .{path}), "int32", &.{ e, ni / 32, d / 32, 2, lay.words_a_block }, std.mem.sliceAsBytes(up));
    x.down = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.experts.routed_experts.down", .{path}), "int32", &.{ e, d / 32, ni / 32, 1, lay.words_a_block }, std.mem.sliceAsBytes(dn));
    x.up_scale = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.experts.routed_experts.up_scale", .{path}), "float32", &.{ e, 2 }, std.mem.sliceAsBytes(us));
    x.down_scale = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.experts.routed_experts.down_scale", .{path}), "float32", &.{ e, 1 }, std.mem.sliceAsBytes(ds));
}

/// fp4Global over columns [k0, k0 + sk) of n rows of `pitch` (down's TP slice).
fn globalCols(rows: []const u16, pitch: usize, k0: usize, n: usize, sk: usize) f32 {
    var amax: f32 = 0;
    for (0..n) |r| for (rows[r * pitch + k0 ..][0..sk]) |b| {
        amax = @max(amax, @abs(lay.bf16ToF32(b)));
    };
    const inv: f32 = @as(f32, 1.0) / @as(f32, 6.0 * 448.0);
    return @max(amax * inv, @as(f32, @floatCast(@as(f64, 1e-30))));
}
