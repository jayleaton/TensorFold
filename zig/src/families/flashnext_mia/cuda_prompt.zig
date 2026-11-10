//! Prompt matmuls without split-K partials (`_b16mm_ks`, `_fp4mm_ks`): slice totals in order, `_reduce`'s bits.
const std = @import("std");
const envs = @import("cuda_env.zig");
const cuda = @import("cuda");
const aot = cuda.aot;
const tri = @import("cuda_triton.zig");
const mm_check = @import("cuda_prompt_check.zig");
const glue = @import("cuda_prompt_glue.zig");

pub const env = "TF_FLASHNEXT_PROMPT_MM";

const bf16 = "*bf16";
const f32p = "*fp32";

/// Rows from which the in-program slices run: the wrappers' 128-row bucket (bf16.matmul's bm = 128 if m > 128).
pub const min_rows = 129;
/// `_b16mm_ks`'s row tiles a band (tools/zig/flashnext_mia/flashnext_prompt_spec.py GROUP).
const group = 8;

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn cb(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

fn run(t: tri.Tri, name: []const u8, grid: [3]usize, args: []const aot.Arg, consts: []const aot.Const) !void {
    const g: [3]u32 = .{ @intCast(grid[0]), @intCast(grid[1]), @intCast(grid[2]) };
    try t.set.?.run(t.s, name, g, args, consts);
}

/// Whether the kernel set holds the in-program kernels (an older set: Python's split path everywhere).
pub fn available(set: *const aot.Set) bool {
    return set.smallestConst("_b16mm_ks", "SK", 0) != null and set.smallestConst("_fp4mm_ks", "SK", 0) != null;
}

/// Whether a bf16 matmul of `m` rows takes `_b16mm_ks`.
pub fn b16Takes(on: bool, m: usize, n: usize, k: usize) bool {
    return on and m >= min_rows and k % 64 == 0 and tri.b16SplitK(n, k) > 1;
}

/// Whether an NVFP4-table matmul of `m` rows takes `_fp4mm_ks`.
pub fn fp4Takes(on: bool, m: usize, n: usize, k: usize) bool {
    return on and m >= min_rows and n % 64 == 0 and tri.fp4SplitK(n, k) > 1;
}

/// bf16.matmul's bits for m >= min_rows and split K: x [M, K] (rows `x_stride` apart) @ w [N, K].T -> out [M, N].
pub fn b16(t: tri.Tri, x: u64, x_stride: usize, w: u64, out: u64, fp32: bool, m: usize, n: usize, k: usize) !void {
    const sk = tri.b16SplitK(n, k);
    if (m < min_rows or sk < 2 or k % 64 != 0) return error.NotAPromptSplit;
    try run(t, "_b16mm_ks", .{ cdiv(m, 128) * cdiv(n, 64), 1, 1 }, &.{ aot.ptr("X", bf16, x), aot.ptr("W", bf16, w), aot.ptr("OUT", if (fp32) f32p else bf16, out), int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", sk), ci("BM", 128), ci("BLOCK_N", 64), ci("BK", 64), cb("F32", fp32), ci("GROUP", group) });
}

/// nvfp4.matmul's bits for m >= min_rows and split K (the shared expert's tables).
pub fn fp4(t: tri.Tri, x: u64, x_stride: usize, fp: tri.Fp4, out: u64, fp32: bool, m: usize, n: usize, k: usize) !void {
    const sk = tri.fp4SplitK(n, k);
    if (m < min_rows or sk < 2 or n % 64 != 0) return error.NotAPromptSplit;
    const gpi = gpiFor((k / 16) / sk, 1); // nvfp4.CONFIG[128]: one block a step
    try run(t, "_fp4mm_ks", .{ cdiv(m, 128), n / 64, 1 }, &.{ aot.ptr("X", bf16, x), aot.ptr("W", if (fp.codes) "*u8" else "*u16", fp.weight), aot.ptr("S", if (fp.codes) "*u8" else f32p, fp.scale), aot.ptr("S2", f32p, fp.scale2), aot.ptr("OUT", if (fp32) f32p else bf16, out), int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", sk), ci("BM", 128), ci("SBN", 64), ci("BLOCK_N", 64), ci("GPI", gpi), cb("F32", fp32), cb("PACKED", fp.codes) });
}

fn gpiFor(per: usize, want: usize) usize {
    for ([_]usize{ want, 8, 4, 2, 1 }) |g| if (g <= want and per % g == 0) return g;
    return 1;
}

/// `_b16mm` alone into the partials (the profile's split of a matmul from its `_reduce`; Tri.b16mm's launch).
pub fn b16Slices(t: tri.Tri, x: u64, x_stride: usize, w: u64, part: u64, fp32: bool, m: usize, n: usize, k: usize) !void {
    const sk = tri.b16SplitK(n, k);
    const bm: usize = if (m > 128) 128 else 16;
    try run(t, "_b16mm", .{ cdiv(m, bm), cdiv(n, 64), sk }, &.{ aot.ptr("X", bf16, x), aot.ptr("W", bf16, w), aot.ptr("OUT", if (fp32) f32p else bf16, part), aot.ptr("PART", f32p, part), int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", sk), ci("BM", bm), ci("BLOCK_N", 64), ci("BK", 64), cb("F32", fp32) });
}

/// `_reduce` of `sk` slices (Tri.reduce).
pub fn reduce(t: tri.Tri, part: u64, out: u64, fp32: bool, total: usize, sk: usize) !void {
    try run(t, "_reduce", .{ cdiv(total, 1024), 1, 1 }, &.{ aot.ptr("PART", f32p, part), aot.ptr("OUT", if (fp32) f32p else bf16, out), int("total", total) }, &.{ ci("SK", sk), ci("BLOCK", 1024), cb("F32", fp32) });
}

pub const cases = mm_check.cases;
pub const check = mm_check.check;
pub const bench = mm_check.bench;
pub const qsaCheck = mm_check.qsaCheck;
pub const qsaBench = mm_check.qsaBench;
pub const glueCheck = glue.glueCheck;
pub const wbNormCheck = glue.wbNormCheck;

// -- the indexer's scoring and selection of a prompt block, by rows ---------------------------------------------

/// attention.qsa_rows' `_scores` alone: rows [0, rows) of the block starting at `pos0`, scores into `sc.scores`.
pub fn qsaScores(t: tri.Tri, iq: u64, pooled: u64, pos0: u64, sc: tri.AttnScratch, rows: usize, g: tri.AttnGeometry, context: ?usize, d: tri.Dims) !void {
    const blocks = g.blocks(context);
    try run(t, "_scores", .{ rows, cdiv(blocks, 64), 1 }, &.{ aot.ptr("IQ", bf16, iq), aot.ptr("POOLED", bf16, pooled), aot.ptr("POS0", "*i32", pos0), aot.ptr("SC", f32p, sc.scores), int("NB", g.nb) }, &.{ ci("HI", d.index_heads), ci("DI", d.index_dim), ci("RATIO", g.ratio), ci("TOP", g.budget / g.ratio), ci("BB", 64) });
}

/// Tri.qsaRows' second launch: `_select` (or `_select_tiles` past the registers) of the scored rows.
pub fn qsaSelect(t: tri.Tri, pos0: u64, sc: tri.AttnScratch, rows: usize, g: tri.AttnGeometry, context: ?usize) !void {
    const top = g.budget / g.ratio;
    const blocks = g.blocks(context);
    const lists = [_]aot.Arg{ aot.ptr("SC", f32p, sc.scores), aot.ptr("POS0", "*i32", pos0), aot.ptr("IDS", "*i32", sc.ids), aot.ptr("NKR", "*i32", sc.nk), aot.ptr("SPR", "*i32", sc.sparse), int("NB", g.nb) };
    var width: usize = 1;
    while (width < blocks) width *= 2;
    if (width <= tri.AttnGeometry.select_regs) {
        try run(t, "_select", .{ rows, 1, 1 }, &lists, &.{ ci("RATIO", g.ratio), ci("TOP", top), ci("IDW", g.idw), ci("BLOCK", width) });
    } else {
        try run(t, "_select_tiles", .{ rows, 1, 1 }, &lists, &.{ ci("RATIO", g.ratio), ci("TOP", top), ci("IDW", g.idw), ci("TB", if (rows >= 64) 4096 else 8192) });
    }
}

/// The scratch of rows [r, ..) of a block: scores, key lists, lengths and flags at their rows.
pub fn scratchAt(sc: tri.AttnScratch, g: tri.AttnGeometry, r: usize) tri.AttnScratch {
    var s = sc;
    s.scores = sc.scores + r * g.nb * 4;
    s.ids = sc.ids + r * g.idw * 4;
    s.nk = sc.nk + r * 4;
    s.sparse = sc.sparse + r * 4;
    return s;
}

/// `_scores_rows`: `_scores` for `rt` rows a program (each block tile's pooled keys loaded once for them).
pub fn qsaScoresRows(t: tri.Tri, iq: u64, pooled: u64, pos0: u64, sc: tri.AttnScratch, rows: usize, g: tri.AttnGeometry, context: ?usize, d: tri.Dims, rt: usize) !void {
    const blocks = g.blocks(context);
    try run(t, "_scores_rows", .{ cdiv(rows, rt), cdiv(blocks, 64), 1 }, &.{ aot.ptr("IQ", bf16, iq), aot.ptr("POOLED", bf16, pooled), aot.ptr("POS0", "*i32", pos0), aot.ptr("SC", f32p, sc.scores), int("NB", g.nb), int("ROWS", rows) }, &.{ ci("HI", d.index_heads), ci("DI", d.index_dim), ci("RATIO", g.ratio), ci("TOP", g.budget / g.ratio), ci("BB", 64), ci("RT", rt) });
}

/// fn_qsa_scores.cu: `_scores`' bits from 32-row x 64-block tiles (the prompt indexer's scoring).
pub const QsaScores = struct {
    mod: cuda.Module,
    fs: [4]cuda.Function, // (rows, blocks) a thread: (2, 4), (4, 4), (2, 8), (4, 8)
    pick: usize = 0,

    const tiles = [4][2]usize{ .{ 2, 4 }, .{ 4, 4 }, .{ 2, 8 }, .{ 4, 8 } };
    const names = [4][:0]const u8{ "fn_qsa_scores", "fn_qsa_scores_r4b4", "fn_qsa_scores_r2b8", "fn_qsa_scores_r4b8" };

    fn shared(i: usize) usize {
        // tile 1 keeps its keys widened to fp32
        return (16 * tiles[i][1] * @as(usize, if (i == 1) 2 else 1) + 16 * tiles[i][0] * 4) * 128 * 2;
    }

    pub fn init(d: *const cuda.Driver) !QsaScores {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var m = try cuda.Module.load(d, cuda.kernels.fn_qsa_scores);
        errdefer m.unload();
        var q: QsaScores = .{ .mod = m, .fs = undefined };
        for (&q.fs, names, 0..) |*f, n, i| {
            f.* = try m.function(n);
            try f.allowDynamicShared(@intCast(shared(i)));
        }
        if (envs.get("TF_FLASHNEXT_QSA_TILE")) |v| q.pick = @min(3, std.fmt.parseInt(usize, v, 10) catch 0);
        return q;
    }

    pub fn deinit(q: *QsaScores) void {
        q.mod.unload();
    }

    pub fn run(q: *const QsaScores, s: cuda.Stream, iq: u64, pooled: u64, pos0: u64, sc: tri.AttnScratch, rows: usize, g: tri.AttnGeometry, context: ?usize) !void {
        const blocks = g.blocks(context);
        var a: cuda.Args = .{};
        for ([_]u64{ iq, pooled, pos0, sc.scores }) |v| a.add(v);
        for ([_]usize{ g.nb, rows, g.ratio, g.budget / g.ratio }) |v| a.add(@as(c_int, @intCast(v)));
        const t = tiles[q.pick];
        try cuda.launch.launch(q.fs[q.pick], .{ .grid = .{ .x = @intCast(cdiv(rows, 16 * t[0])), .y = @intCast(cdiv(blocks, 16 * t[1])), .z = 1 }, .block = .{ .x = 256 }, .shared = @intCast(shared(q.pick)) }, s, &a);
    }
};

/// `_select_tiles` for any row length (Python's own choice past `_select`'s registers; the same lists).
pub fn qsaSelectTiles(t: tri.Tri, pos0: u64, sc: tri.AttnScratch, rows: usize, g: tri.AttnGeometry) !void {
    const lists = [_]aot.Arg{ aot.ptr("SC", f32p, sc.scores), aot.ptr("POS0", "*i32", pos0), aot.ptr("IDS", "*i32", sc.ids), aot.ptr("NKR", "*i32", sc.nk), aot.ptr("SPR", "*i32", sc.sparse), int("NB", g.nb) };
    try run(t, "_select_tiles", .{ rows, 1, 1 }, &lists, &.{ ci("RATIO", g.ratio), ci("TOP", g.budget / g.ratio), ci("IDW", g.idw), ci("TB", if (rows >= 64) 4096 else 8192) });
}

// -- the hyper-connection read-out's up projection and mix in one pass ---------------------------------------------

pub fn upMixAvailable(set: *const aot.Set) bool {
    return set.smallestConst("_hc_up_mix", "BM", 0) != null;
}

/// Whether the kernel set holds `_hc_wb_norm` (an older set: `_hc_writeback` and `_hc_normed`).
pub fn wbNormAvailable(set: *const aot.Set) bool {
    return set.smallestConst("_hc_wb_norm", "MODE", 0) != null;
}

/// `_hc_up_mix`: mixed [M, D] = hc_mix(b16mm(act, up), normed) without writing the up rows or 32-group sums.
pub fn upMix(t: tri.Tri, act: u64, w: u64, normed: u64, mixed: u64, m: usize, d: usize, streams: usize, k: usize, bm: usize) !void {
    try run(t, "_hc_up_mix", .{ cdiv(m, bm), d / 64, 1 }, &.{ aot.ptr("ACT", bf16, act), aot.ptr("W", bf16, w), aot.ptr("NORMED", bf16, normed), aot.ptr("MIXED", bf16, mixed), int("M", m) }, &.{ ci("D", d), ci("S", streams), ci("K", k), ci("BM", bm), ci("BD", 64), ci("BK", 64) });
}
