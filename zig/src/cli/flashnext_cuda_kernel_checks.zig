//! Flash Next CUDA CLI kernel checks without weights: prompt matmuls, experts, indexer, FP4, glue.
const std = @import("std");
const cuda = @import("cuda");
const flashnext = @import("flashnext_mia");
const cli = @import("flashnext_cuda.zig");

const Allocator = std.mem.Allocator;
const Options = cli.Options;

/// `mm-check`: no checkpoint; the kernel set's in-program K-slice matmuls against the split kernels + _reduce.
pub fn mmCheck(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, kernels: []const u8, o: Options) !u8 {
    const d = ctx.d;
    var set = try cuda.aot.Set.load(gpa, io, d, ctx.device, kernels);
    defer set.deinit();
    if (!flashnext.prompt.available(&set)) {
        std.debug.print("the kernel set {s} has no _b16mm_ks / _fp4mm_ks\n", .{kernels});
        return 1;
    }
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const t: flashnext.triton.Tri = .{ .set = &set, .s = stream };
    var rows: std.ArrayList(usize) = .empty;
    defer rows.deinit(gpa);
    if (o.rows.len > 0) {
        var it = std.mem.tokenizeScalar(u8, o.rows, ',');
        while (it.next()) |r| try rows.append(gpa, try std.fmt.parseInt(usize, r, 10));
    } else try rows.appendSlice(gpa, &.{ 129, 200, 974, 2048, 2049, 4096 });
    const tp: u32 = 3; // TP=1 and TP=2 shapes
    const ok = try flashnext.prompt.check(gpa, d, t, tp, rows.items);
    for (flashnext.prompt.cases) |c| if (c.tp & tp != 0) {
        try flashnext.prompt.bench(gpa, d, t, c, 2048, 10);
        try flashnext.prompt.bench(gpa, d, t, c, 4096, 10);
    };
    std.debug.print("{s} mm-check: in-program K slices {s} Python's split kernels + _reduce\n", .{ if (ok) "PASS" else "FAIL", if (ok) "byte-equal to" else "DIFFER from" });
    return if (ok) 0 else 1;
}

/// `experts-check`: no checkpoint; fn_experts_prompt.cu against nvfp4_expert_kernel (TP=2 and TP=1 widths).
pub fn expertsCheck(gpa: Allocator, ctx: *const cuda.Context, o: Options) !u8 {
    var k = try flashnext.kernels.Kernels.load(ctx);
    defer k.deinit();
    var p = try flashnext.moe_prompt.Prompt4.init(ctx);
    defer p.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    const ops: flashnext.kernels.Ops = .{ .k = &k, .s = stream };
    var rows: std.ArrayList(usize) = .empty;
    defer rows.deinit(gpa);
    if (o.rows.len > 0) {
        var it = std.mem.tokenizeScalar(u8, o.rows, ',');
        while (it.next()) |r| try rows.append(gpa, try std.fmt.parseInt(usize, r, 10));
    } else try rows.appendSlice(gpa, &.{ 129, 974, 2048, 4096 });
    var ok = true;
    // the plan's item size and row tiles a warp: every choice must give the same bytes; the first is the default
    const cfgs = [_][3]usize{ .{ p.tile, p.t_gu, p.t_down }, .{ 64, 2, 4 }, .{ 48, 3, 2 }, .{ 16, 2, 2 } };
    for (cfgs, 0..) |c, ci| {
        p.tile = c[0];
        p.t_gu = c[1];
        p.t_down = c[2];
        ok = (try flashnext.moe_prompt.check(gpa, ctx, ops, &p, 320, rows.items, false)) and ok;
        if (ci == 0) {
            ok = (try flashnext.moe_prompt.check(gpa, ctx, ops, &p, 320, rows.items, true)) and ok;
            ok = (try flashnext.moe_prompt.check(gpa, ctx, ops, &p, 640, rows.items, false)) and ok;
            ok = (try flashnext.moe_prompt.checkWith(gpa, ctx, ops, &p, 320, rows.items, true, true)) and ok;
        }
    }
    // the 4-CTA variants (TF_FLASHNEXT_PROMPT_EXPERTS_OCC) on the default tile
    p.tile = 32;
    p.t_gu = 2;
    p.t_down = 2;
    p.use_occ = true;
    ok = (try flashnext.moe_prompt.check(gpa, ctx, ops, &p, 320, rows.items, false)) and ok;
    p.use_occ = false;
    // gate/up in two one-matrix passes (TF_FLASHNEXT_PROMPT_EXPERTS_GU2), plain, special values and TP=1 width
    p.gu2 = true;
    ok = (try flashnext.moe_prompt.check(gpa, ctx, ops, &p, 320, rows.items, false)) and ok;
    ok = (try flashnext.moe_prompt.checkWith(gpa, ctx, ops, &p, 320, rows.items, true, true)) and ok;
    ok = (try flashnext.moe_prompt.check(gpa, ctx, ops, &p, 640, rows.items, false)) and ok;
    p.gu2 = false;
    std.debug.print("{s} experts-check: the prompt expert kernel {s} Python's on 16-pair items\n", .{ if (ok) "PASS" else "FAIL", if (ok) "byte-equal to" else "DIFFER from" });
    return if (ok) 0 else 1;
}

/// `qsa-check`: no checkpoint; the prompt indexer's row-tiled scoring and tiled selection against Python's kernels.
pub fn qsaCheck(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, kernels: []const u8) !u8 {
    var set = try cuda.aot.Set.load(gpa, io, ctx.d, ctx.device, kernels);
    defer set.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    var fast = try flashnext.prompt.QsaScores.init(ctx.d);
    defer fast.deinit();
    const ok = try flashnext.prompt.qsaCheck(gpa, ctx.d, .{ .set = &set, .s = stream }, &fast);
    // timing: Python's _scores against fn_qsa_scores on 256 rows at 128k and 1M-like key counts
    try flashnext.prompt.qsaBench(ctx.d, .{ .set = &set, .s = stream }, &fast);
    std.debug.print("{s} qsa-check\n", .{if (ok) "PASS" else "FAIL"});
    return if (ok) 0 else 1;
}

/// `fp4-check`: no checkpoint; fn_ops' K-serial NVFP4 kernel against Triton's `_fp4mm` at one K slice, bytes and time.
pub fn fp4Check(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, kernels: []const u8, o: Options) !u8 {
    var set = try cuda.aot.Set.load(gpa, io, ctx.d, ctx.device, kernels);
    defer set.deinit();
    var ops = try flashnext.torch_ops.Ops.load(ctx);
    defer ops.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    const t: flashnext.triton.Tri = .{ .set = &set, .s = stream };
    const th = ops.on(stream);
    var rows: std.ArrayList(usize) = .empty;
    defer rows.deinit(gpa);
    if (o.rows.len > 0) {
        var it = std.mem.tokenizeScalar(u8, o.rows, ',');
        while (it.next()) |r| try rows.append(gpa, try std.fmt.parseInt(usize, r, 10));
    } else try rows.appendSlice(gpa, &flashnext.fp4_serial.default_rows);
    const ok = try flashnext.fp4_serial.check(gpa, ctx.d, t, th, rows.items);
    try flashnext.fp4_serial.bench(gpa, ctx.d, t, th, &.{ 1, 8, 36, 48, 64, 113, 128, 224 }, 50);
    std.debug.print("{s} fp4-check: the K-serial kernel {s} _fp4mm at one K slice\n", .{ if (ok) "PASS" else "FAIL", if (ok) "byte-equal to" else "DIFFERS from" });
    return if (ok) 0 else 1;
}

/// `glue-check`: no checkpoint; the fused hyper-connection up+mix and write-back+norm against their unfused kernels.
pub fn glueCheck(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, kernels: []const u8) !u8 {
    var set = try cuda.aot.Set.load(gpa, io, ctx.d, ctx.device, kernels);
    defer set.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    const t: flashnext.triton.Tri = .{ .set = &set, .s = stream };
    const ok = (try flashnext.prompt.glueCheck(gpa, ctx.d, t)) and (try flashnext.prompt.wbNormCheck(gpa, ctx.d, t));
    std.debug.print("{s} glue-check\n", .{if (ok) "PASS" else "FAIL"});
    return if (ok) 0 else 1;
}
