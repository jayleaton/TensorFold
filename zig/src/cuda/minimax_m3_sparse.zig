//! MiniMax-M3.1-Flash block-sparse attention launches: index scoring + block maxima, forced-block top-16, selected-block
//! GQA attention (zig/kernels/cuda/minimax_m3_{index,select,attn}.cu). K/V/IK live in 128-token pages addressed by page
//! ids from block tables (geometry in minimax_m3_geom.zig). Decode rows come with positions; a prompt chunk is `pos == 0`
//! and `pos0`. Never run on a GPU in this tree's history: see docs for what is checked.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
pub const geom = @import("minimax_m3_geom.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// The paged cache one launch reads: the pool's base, its page and layer strides, local kv heads and the block tables.
pub const Paged = struct {
    pool: u64,
    page_bytes: u64,
    layer_off: u64,
    kvh: u32,
    tables: u64,
    /// int32 [rows] offsets into `tables`, or 0 when every row uses the table at its start (one stream).
    row_tab: u64 = 0,
    /// The pool is the fp8 layout (geom.Pool.kv8): e4m3 K/V rows with fp32 scales; false is the bf16 pool.
    kv8: bool = false,
};

/// Rows of a call: device int32 positions [n], or 0 and `pos0` (rows pos0, pos0 + 1, ... of one stream).
pub const Rows = struct { n: usize, pos: u64 = 0, pos0: i32 = 0 };

pub const Scratch = struct {
    /// fp32 [rows, kvh, bm_stride] and int32 [rows, kvh, 16] device buffers.
    bmax: u64,
    sel: u64,
    bm_stride: usize,
};

pub const Sparse = struct {
    index_mod: Module,
    select_mod: Module,
    attn_mod: Module,
    index_fn: Function,
    select_fn: Function,
    attn_fn: Function,

    pub fn load(d: *const Driver) !Sparse {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var im = try Module.load(d, kernels.minimax_m3_index);
        errdefer im.unload();
        var sm = try Module.load(d, kernels.minimax_m3_select);
        errdefer sm.unload();
        var am = try Module.load(d, kernels.minimax_m3_attn);
        errdefer am.unload();
        return .{
            .index_mod = im,
            .select_mod = sm,
            .attn_mod = am,
            .index_fn = try im.function("tf_mm3_index"),
            .select_fn = try sm.function("tf_mm3_select"),
            .attn_fn = try am.function("tf_mm3_attn"),
        };
    }

    pub fn unload(s: *Sparse) void {
        s.index_mod.unload();
        s.select_mod.unload();
        s.attn_mod.unload();
    }

    fn cfg(l: geom.Launch) launch_.Config {
        return .{ .grid = .{ .x = l.grid[0], .y = l.grid[1], .z = l.grid[2] }, .block = .{ .x = l.block }, .shared = l.dynamic_shared };
    }

    /// iq [rows, kvh, 128] bf16 -> scratch.bmax; `max_pos` bounds the rows' positions (the grid's block count).
    pub fn score(s: *const Sparse, st: Stream, iq: u64, rows: Rows, p: Paged, sc: Scratch, max_pos: usize, scale: f32) !void {
        try geom.check(rows.n, p.kvh);
        if (geom.blocks(max_pos) > sc.bm_stride) return error.TableTooShort;
        var a: launch_.Args = .{};
        a.add(iq);
        a.add(rows.pos);
        a.add(rows.pos0);
        a.add(p.row_tab);
        a.add(p.tables);
        a.add(p.pool);
        a.add(@as(i64, @intCast(p.page_bytes)));
        a.add(@as(i64, @intCast(p.layer_off)));
        a.add(@as(i32, @intCast(p.kvh)));
        a.add(sc.bmax);
        a.add(@as(i32, @intCast(sc.bm_stride)));
        a.add(scale);
        a.add(@as(i32, @intFromBool(p.kv8)));
        try launch_.launch(s.index_fn, cfg(geom.indexLaunch(rows.n, p.kvh, max_pos)), st, &a);
    }

    /// scratch.bmax -> scratch.sel: ascending block ids per (row, kv head), -1 padded, by the forcing and tie rules.
    pub fn select(s: *const Sparse, st: Stream, rows: Rows, kvh: usize, sc: Scratch, rules: geom.Rules) !void {
        try geom.check(rows.n, kvh);
        var a: launch_.Args = .{};
        a.add(sc.bmax);
        a.add(rows.pos);
        a.add(rows.pos0);
        a.add(sc.sel);
        a.add(@as(i32, @intCast(kvh)));
        a.add(@as(i32, @intCast(sc.bm_stride)));
        a.add(rules.init_n);
        a.add(@as(i32, @intFromBool(rules.init_inclusive)));
        a.add(rules.local_n);
        a.add(@as(i32, @intFromBool(rules.local_trailing)));
        a.add(@as(i32, @intFromBool(rules.higher_tie)));
        try launch_.launch(s.select_fn, cfg(geom.selectLaunch(rows.n, kvh)), st, &a);
    }

    /// q [rows, kvh * 16, 128] bf16 and scratch.sel -> out [rows, kvh * 16, 128] bf16.
    pub fn attend(s: *const Sparse, st: Stream, q: u64, rows: Rows, p: Paged, sc: Scratch, out: u64, scale: f32) !void {
        try geom.check(rows.n, p.kvh);
        var a: launch_.Args = .{};
        a.add(q);
        a.add(sc.sel);
        a.add(rows.pos);
        a.add(rows.pos0);
        a.add(p.row_tab);
        a.add(p.tables);
        a.add(p.pool);
        a.add(@as(i64, @intCast(p.page_bytes)));
        a.add(@as(i64, @intCast(p.layer_off)));
        a.add(@as(i32, @intCast(p.kvh)));
        a.add(out);
        a.add(scale);
        a.add(@as(i32, @intFromBool(p.kv8)));
        try launch_.launch(s.attn_fn, cfg(geom.attnLaunch(rows.n, p.kvh)), st, &a);
    }

    /// The three launches of one layer's sparse attention, in order, on one stream.
    pub fn run(s: *const Sparse, st: Stream, iq: u64, q: u64, rows: Rows, p: Paged, sc: Scratch, rules: geom.Rules, max_pos: usize, out: u64) !void {
        const scale: f32 = @floatCast(std.math.pow(f64, geom.d, -0.5));
        try s.score(st, iq, rows, p, sc, max_pos, scale);
        try s.select(st, rows, p.kvh, sc, rules);
        try s.attend(st, q, rows, p, sc, out, scale);
    }
};

test {
    _ = geom;
}
