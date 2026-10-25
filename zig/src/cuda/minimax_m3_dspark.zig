//! MiniMax-M3.1-Flash dspark draft kernels (minimax_m3_dspark.cu): taps, residual add, Gemma norm, per-head norm +
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const memory = @import("memory.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = @import("stream.zig").Stream;

pub const head_dim = 128;
pub const threads = 256; // every kernel but the merge and the merge's 128
pub const max_rot = 128;
pub const keys = 128; // keys one attention CTA scores
pub const k_row = 130; // padded K tile row, bf16 elements
pub const part_floats = 132; // m, l, 2 pad, 128 accumulators
pub const max_rows = 16; // a draft block or a verify window
pub const max_queries = 256; // heads of a kv group x rows
pub const topc_max = 128;
pub const chain_max_cand = 256;
pub const part_cap = 16 << 20; // bytes of attention partials the module holds
pub const attn_shared: u32 = keys * k_row * 2 + keys * head_dim * 2 + 2 * 8 * head_dim * 4;

pub const Error = error{ Empty, TooManyRows, BadStride, Align, BadShape, TooMuchContext, Scratch };

pub const Launch = struct { grid: [3]u32, block: u32 };

fn cfg(l: Launch, shared: u32) launch_.Config {
    return .{ .grid = .{ .x = l.grid[0], .y = l.grid[1], .z = l.grid[2] }, .block = .{ .x = l.block }, .shared = shared };
}

fn aligned(p: u64, to: u64) bool {
    return p % to == 0;
}

// ---- geometry

/// One CTA a row.
pub fn rowLaunch(rows: usize) Error!Launch {
    if (rows == 0) return error.Empty;
    if (rows > 65535) return error.TooManyRows;
    return .{ .grid = .{ @intCast(rows), 1, 1 }, .block = threads };
}

pub fn addLaunch(n: usize) Error!Launch {
    if (n == 0) return error.Empty;
    const blocks = @min((n + threads - 1) / threads, 1 << 16);
    return .{ .grid = .{ @intCast(blocks), 1, 1 }, .block = threads };
}

/// What an attention call splits into: the first context key, context chunks of 128 keys, and chunks in all (the
pub const Split = struct { c0: u32, ctx_chunks: u32, chunks: u32, queries: u32 };

pub fn attnSplit(a: Attn, rows: u32, p: u32, lo: u32) Error!Split {
    if (rows == 0) return error.Empty;
    if (rows > max_rows) return error.TooManyRows;
    if (a.kvh == 0 or a.nq == 0 or a.nq % a.kvh != 0 or a.ring_rows == 0 or a.window == 0) return error.BadShape;
    const rep = a.nq / a.kvh;
    const q = rep * rows;
    if (q > max_queries) return error.BadShape;
    const w0: u32 = if (p + 1 > a.window) p + 1 - a.window else 0;
    const c0 = @max(lo, w0);
    const nctx: u32 = if (p > c0) p - c0 else 0;
    if (nctx > a.ring_rows) return error.TooMuchContext;
    const ctx_chunks = (nctx + keys - 1) / keys;
    return .{ .c0 = c0, .ctx_chunks = ctx_chunks, .chunks = ctx_chunks + 1, .queries = q };
}

pub fn attnPartBytes(a: Attn, s: Split) usize {
    return @as(usize, a.kvh) * s.chunks * s.queries * part_floats * 4;
}

pub fn topcLaunch(rows: usize, n: usize, ld: usize, c: usize) Error!Launch {
    if (rows == 0 or n == 0 or c == 0) return error.Empty;
    if (rows > 65535) return error.TooManyRows;
    if (c > topc_max) return error.BadShape;
    if (ld < n or n > std.math.maxInt(i32)) return error.BadStride;
    return .{ .grid = .{ @intCast(rows), 1, 1 }, .block = threads };
}

pub const Heads = struct { nq: u32, kvh: u32, rot: u32, theta: f64, eps: f32, gemma: bool };

pub const Attn = struct { nq: u32, kvh: u32, ring_rows: u32, window: u32, scale: f32 };

pub const Chain = struct { hidden: u32, rank: u32, cand: u32, world: u32, rows: u32 };

pub fn checkChain(c: Chain) Error!void {
    if (c.rows == 0 or c.cand == 0 or c.world == 0) return error.Empty;
    if (c.rows > max_rows or c.world * c.cand > chain_max_cand) return error.TooManyRows;
    if (c.hidden == 0 or c.hidden % 128 != 0 or c.rank == 0 or c.rank % 32 != 0 or c.rank > 1024) return error.BadShape;
}

fn checkHeads(h: Heads, q_heads: u32) Error!void {
    if (h.kvh == 0 or h.rot == 0 or h.rot % 2 != 0 or h.rot > max_rot) return error.BadShape;
    if (q_heads == 0 and h.nq != 0) return error.BadShape;
}

pub const Dspark = struct {
    mod: Module,
    tap_fn: Function,
    add_bf16: Function,
    add_f32: Function,
    norm_bf16: Function,
    norm_f32: Function,
    block_heads_fn: Function,
    ring_put_fn: Function,
    attn_fn: Function,
    merge_fn: Function,
    topc_fn: Function,
    chain_fn: Function,
    part: memory.DeviceBuffer, // attention partials, part_cap bytes

    pub fn load(d: *const Driver) !Dspark {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var m = try Module.load(d, kernels.minimax_m3_dspark);
        errdefer m.unload();
        var part = try memory.DeviceBuffer.alloc(d, part_cap);
        errdefer part.free();
        const attn = try m.function("tf_mm3_ds_attn");
        try attn.allowDynamicShared(attn_shared);
        return .{
            .mod = m,
            .tap_fn = try m.function("tf_mm3_ds_tap"),
            .add_bf16 = try m.function("tf_mm3_ds_add_bf16"),
            .add_f32 = try m.function("tf_mm3_ds_add_f32"),
            .norm_bf16 = try m.function("tf_mm3_ds_norm_bf16"),
            .norm_f32 = try m.function("tf_mm3_ds_norm_f32"),
            .block_heads_fn = try m.function("tf_mm3_ds_block_heads"),
            .ring_put_fn = try m.function("tf_mm3_ds_ring_put"),
            .attn_fn = attn,
            .merge_fn = try m.function("tf_mm3_ds_attn_merge"),
            .topc_fn = try m.function("tf_mm3_ds_topc"),
            .chain_fn = try m.function("tf_mm3_ds_chain"),
            .part = part,
        };
    }

    pub fn unload(g: *Dspark) void {
        g.part.free();
        g.mod.unload();
    }

    /// dst bf16 [rows][dst_ld], columns [slot * hidden, (slot + 1) * hidden) = bf16(src f32 [rows][hidden]).
    pub fn tap(g: *const Dspark, st: Stream, src: u64, dst: u64, dst_ld: usize, slot: u32, hidden: u32, rows: u32) !void {
        const l = try rowLaunch(rows);
        if (hidden == 0 or dst_ld < (@as(usize, slot) + 1) * hidden) return error.BadStride;
        if (src == 0 or dst == 0 or !aligned(src, 4) or !aligned(dst, 2)) return error.Align;
        var args: launch_.Args = .{};
        args.add(src);
        args.add(dst);
        args.add(@as(i64, @intCast(dst_ld)));
        args.add(@as(i32, @intCast(slot)));
        args.add(@as(i32, @intCast(hidden)));
        args.add(@as(i32, @intCast(rows)));
        try launch_.launch(g.tap_fn, cfg(l, 0), st, &args);
    }

    /// dst f32 [n] = (a == 0 ? 0 : a[i]) + b[i]; b is bf16 or f32; a may alias dst.
    pub fn add(g: *const Dspark, st: Stream, a: u64, b: u64, b_f32: bool, dst: u64, n: usize) !void {
        const l = try addLaunch(n);
        if (b == 0 or dst == 0 or !aligned(a, 4) or !aligned(dst, 4) or !aligned(b, if (b_f32) 4 else 2)) return error.Align;
        var args: launch_.Args = .{};
        args.add(a);
        args.add(b);
        args.add(dst);
        args.add(@as(i64, @intCast(n)));
        try launch_.launch(if (b_f32) g.add_f32 else g.add_bf16, cfg(l, 0), st, &args);
    }

    /// y bf16 [rows][hidden] = x * (1 / sqrt(mean(x^2) + eps)) * (gemma ? 1 + w : w); x f32 or bf16, w bf16 [hidden].
    pub fn norm(g: *const Dspark, st: Stream, x: u64, x_f32: bool, w: u64, gemma: bool, y: u64, rows: u32, hidden: u32, eps: f32) !void {
        const l = try rowLaunch(rows);
        if (hidden == 0) return error.BadStride;
        if (x == 0 or w == 0 or y == 0 or !aligned(x, if (x_f32) 4 else 2) or !aligned(w, 2) or !aligned(y, 2)) return error.Align;
        var args: launch_.Args = .{};
        args.add(x);
        args.add(w);
        args.add(y);
        args.add(@as(i32, @intCast(rows)));
        args.add(@as(i32, @intCast(hidden)));
        args.add(eps);
        args.add(@as(i32, @intFromBool(gemma)));
        try launch_.launch(if (x_f32) g.norm_f32 else g.norm_bf16, cfg(l, 0), st, &args);
    }

    /// q [rows][nq * 128], k, v [rows][kvh * 128] -> qn, kn (per-head norm, RoPE at pos0 + row) and vn (a copy).
    pub fn blockHeads(g: *const Dspark, st: Stream, h: Heads, q: u64, k: u64, v: u64, qw: u64, kw: u64, qn: u64, kn: u64, vn: u64, rows: u32, pos0: u32) !void {
        const l = try rowLaunch(rows);
        try checkHeads(h, h.nq);
        if (h.nq == 0) return error.BadShape;
        for ([_]u64{ q, k, v, qw, kw, qn, kn, vn }) |p| if (p == 0 or !aligned(p, 8)) return error.Align;
        var args: launch_.Args = .{};
        for ([_]u64{ q, k, v, qw, kw, qn, kn, vn }) |p| args.add(p);
        args.add(@as(i32, @intCast(rows)));
        args.add(@as(i32, @intCast(h.nq)));
        args.add(@as(i32, @intCast(h.kvh)));
        args.add(@as(i32, @intCast(h.rot)));
        args.add(h.eps);
        args.add(@as(i32, @intFromBool(h.gemma)));
        args.add(h.theta);
        args.add(@as(i32, @intCast(pos0)));
        try launch_.launch(g.block_heads_fn, cfg(l, 0), st, &args);
    }

    /// Absorbed rows k, v [rows][kvh * 128] -> ring slot (pos0 + row) % ring_rows: norm + RoPE'd k and v as it came.
    pub fn ringPut(g: *const Dspark, st: Stream, h: Heads, k: u64, v: u64, kw: u64, ring_k: u64, ring_v: u64, ring_rows: u32, rows: u32, pos0: u32) !void {
        const l = try rowLaunch(rows);
        try checkHeads(h, 0);
        if (ring_rows == 0) return error.BadShape;
        for ([_]u64{ k, v, kw, ring_k, ring_v }) |p| if (p == 0 or !aligned(p, 8)) return error.Align;
        var args: launch_.Args = .{};
        for ([_]u64{ k, v, kw, ring_k, ring_v }) |p| args.add(p);
        args.add(@as(i32, @intCast(rows)));
        args.add(@as(i32, @intCast(h.kvh)));
        args.add(@as(i32, @intCast(h.rot)));
        args.add(h.eps);
        args.add(@as(i32, @intFromBool(h.gemma)));
        args.add(h.theta);
        args.add(@as(i32, @intCast(pos0)));
        args.add(@as(i32, @intCast(ring_rows)));
        try launch_.launch(g.ring_put_fn, cfg(l, 0), st, &args);
    }

    /// Row i (position p + i) attends ring positions max(lo, p + i + 1 - window) <= kp < p and all `rows` block rows.
    pub fn blockAttn(g: *const Dspark, st: Stream, a: Attn, qn: u64, ring_k: u64, ring_v: u64, kn: u64, vn: u64, out: u64, rows: u32, p: u32, lo: u32) !void {
        const s = try attnSplit(a, rows, p, lo);
        if (attnPartBytes(a, s) > part_cap) return error.Scratch;
        for ([_]u64{ qn, ring_k, ring_v, kn, vn }) |x| if (x == 0 or !aligned(x, 16)) return error.Align;
        if (out == 0 or !aligned(out, 2)) return error.Align;
        var args: launch_.Args = .{};
        for ([_]u64{ qn, ring_k, ring_v, kn, vn, g.part.ptr }) |x| args.add(x);
        args.add(@as(i32, @intCast(a.nq)));
        args.add(@as(i32, @intCast(a.kvh)));
        args.add(@as(i32, @intCast(a.ring_rows)));
        args.add(@as(i32, @intCast(a.window)));
        args.add(a.scale);
        args.add(@as(i32, @intCast(rows)));
        args.add(@as(i32, @intCast(p)));
        args.add(@as(i32, @intCast(lo)));
        args.add(@as(i32, @intCast(s.c0)));
        args.add(@as(i32, @intCast(s.chunks)));
        try launch_.launch(g.attn_fn, cfg(.{ .grid = .{ a.kvh, s.chunks, 1 }, .block = threads }, attn_shared), st, &args);
        var m: launch_.Args = .{};
        m.add(g.part.ptr);
        m.add(out);
        m.add(@as(i32, @intCast(a.nq)));
        m.add(@as(i32, @intCast(a.kvh)));
        m.add(@as(i32, @intCast(rows)));
        m.add(@as(i32, @intCast(s.chunks)));
        try launch_.launch(g.merge_fn, cfg(.{ .grid = .{ a.kvh * s.queries, 1, 1 }, .block = head_dim }, 0), st, &m);
    }

    /// The c best of each row's first n logits: out [rows][c] of {f32 logit, i32 id0 + index}; padding {-inf, -1}.
    pub fn topC(g: *const Dspark, st: Stream, logits: u64, ld: usize, n: u32, id0: u32, rows: u32, c: u32, out: u64) !void {
        const l = try topcLaunch(rows, n, ld, c);
        if (logits == 0 or out == 0 or !aligned(logits, 4) or !aligned(out, 8) or @as(u64, id0) + n > std.math.maxInt(i32)) return error.Align;
        var args: launch_.Args = .{};
        args.add(logits);
        args.add(@as(i64, @intCast(ld)));
        args.add(@as(i32, @intCast(n)));
        args.add(@as(i32, @intCast(id0)));
        args.add(@as(i32, @intCast(c)));
        args.add(out);
        try launch_.launch(g.topc_fn, cfg(l, 0), st, &args);
    }

    /// The Markov chain over gathered candidates and the confidence head: out [rows] of {i32 token, f32 conf}.
    pub fn chain(g: *const Dspark, st: Stream, c: Chain, anchor: u32, cands: u64, post: u64, w1: u64, w2: u64, conf_w: u64, conf_b: u64, out: u64) !void {
        try checkChain(c);
        for ([_]u64{ cands, post, conf_w, out }) |x| if (x == 0 or !aligned(x, 8)) return error.Align;
        for ([_]u64{ w1, w2, conf_b }) |x| if (x == 0 or !aligned(x, 2)) return error.Align;
        var args: launch_.Args = .{};
        for ([_]u64{ cands, post, w1, w2, conf_w, conf_b, out }) |x| args.add(x);
        args.add(@as(i32, @intCast(c.hidden)));
        args.add(@as(i32, @intCast(c.rank)));
        args.add(@as(i32, @intCast(c.cand)));
        args.add(@as(i32, @intCast(c.world)));
        args.add(@as(i32, @intCast(c.rows)));
        args.add(anchor);
        try launch_.launch(g.chain_fn, cfg(.{ .grid = .{ 1, 1, 1 }, .block = threads }, 0), st, &args);
    }
};

test "launch geometry: a CTA a row, a flat add, one CTA per row of top-C" {
    const l = try rowLaunch(7);
    try std.testing.expectEqual([3]u32{ 7, 1, 1 }, l.grid);
    try std.testing.expectError(error.Empty, rowLaunch(0));
    try std.testing.expectError(error.TooManyRows, rowLaunch(70000));
    try std.testing.expectEqual(@as(u32, 24), (try addLaunch(24 * 256)).grid[0]);
    try std.testing.expectEqual(@as(u32, 1 << 16), (try addLaunch(1 << 30)).grid[0]);
    try std.testing.expectError(error.Empty, addLaunch(0));
    const t = try topcLaunch(7, 100096, 100160, 64);
    try std.testing.expectEqual([3]u32{ 7, 1, 1 }, t.grid);
    try std.testing.expectError(error.BadShape, topcLaunch(7, 100, 100, 129));
    try std.testing.expectError(error.BadStride, topcLaunch(7, 100, 99, 8));
    try std.testing.expectError(error.Empty, topcLaunch(7, 0, 0, 8));
}

test "attention splits the context into 128-key chunks after the window's start, and the block is the last chunk" {
    const a: Attn = .{ .nq = 32, .kvh = 2, .ring_rows = 4112, .window = 4096, .scale = 0.0883883 };
    // 5000 rows of context, a 4096 window: row 0 starts at 5000 + 1 - 4096
    const s = try attnSplit(a, 7, 5000, 0);
    try std.testing.expectEqual(@as(u32, 905), s.c0);
    try std.testing.expectEqual(@as(u32, 32), s.ctx_chunks); // 4095 keys
    try std.testing.expectEqual(@as(u32, 33), s.chunks);
    try std.testing.expectEqual(@as(u32, 16 * 7), s.queries);
    // a short prompt: everything from lo
    const t = try attnSplit(a, 7, 300, 100);
    try std.testing.expectEqual(@as(u32, 100), t.c0);
    try std.testing.expectEqual(@as(u32, 2), t.ctx_chunks);
    // nothing in the ring yet: the block alone
    const u = try attnSplit(a, 7, 40, 40);
    try std.testing.expectEqual(@as(u32, 0), u.ctx_chunks);
    try std.testing.expectEqual(@as(u32, 1), u.chunks);
    // more context than the ring holds is refused
    try std.testing.expectError(error.TooMuchContext, attnSplit(.{ .nq = 32, .kvh = 2, .ring_rows = 256, .window = 4096, .scale = 1 }, 7, 5000, 0));
    try std.testing.expectError(error.TooManyRows, attnSplit(a, 17, 10, 0));
    try std.testing.expectError(error.BadShape, attnSplit(.{ .nq = 33, .kvh = 2, .ring_rows = 8, .window = 8, .scale = 1 }, 1, 1, 0));
    // the partials of the shipped shape fit the module's scratch with room for 16 rows
    try std.testing.expect(attnPartBytes(a, s) < part_cap);
    const wide = try attnSplit(a, 16, 5000, 0);
    try std.testing.expect(attnPartBytes(a, wide) < part_cap);
    try std.testing.expectEqual(@as(u32, 74240), attn_shared);
}

test "chain shapes: hidden and rank whole, candidates of both ranks fit one CTA" {
    try checkChain(.{ .hidden = 6144, .rank = 256, .cand = 64, .world = 2, .rows = 7 });
    try std.testing.expectError(error.BadShape, checkChain(.{ .hidden = 6000, .rank = 256, .cand = 64, .world = 2, .rows = 7 }));
    try std.testing.expectError(error.TooManyRows, checkChain(.{ .hidden = 6144, .rank = 256, .cand = 129, .world = 2, .rows = 7 }));
    try std.testing.expectError(error.Empty, checkChain(.{ .hidden = 6144, .rank = 256, .cand = 64, .world = 0, .rows = 7 }));
}
