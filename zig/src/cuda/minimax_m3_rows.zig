//! MiniMax-M3.1-Flash row kernels (minimax_m3_rows.cu): norm and RoPE into paged K/V/IK, fp32 router, table mirror.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const memory = @import("memory.zig");
const sp = @import("minimax_m3_sparse.zig");
pub const geom = @import("minimax_m3_rows_geom.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// Local head counts of a rank: query heads, index query heads, kv heads, and whether the index key is written.
pub const Heads = struct { nq: u32 = 32, niq: u32 = 2, kvh: u32 = 2, index_key: bool = true };

/// Norm and rotation arguments; `w_add` (q, k) and `w_add_index` are 1 for raw Gemma weights, 0 for the folded `1 + w`.
pub const Opts = struct {
    rot: u32 = 64,
    theta: f64 = 1e7,
    eps: f32 = 1e-6,
    w_add: f32 = 0,
    w_add_index: f32 = 0,
    flags: geom.Flags = .{},
};

/// Device inputs: row-major projections at element strides; the index inputs are fp16 (or bf16 per `flags.index_fp16`).
pub const Inputs = struct {
    q: u64,
    q_ld: usize,
    k: u64,
    k_ld: usize,
    v: u64,
    v_ld: usize,
    iq: u64 = 0,
    iq_ld: usize = 0,
    ik: u64 = 0,
    ik_ld: usize = 0,
};

/// fp32 [128] norm weights on the device; 0 when that norm is off.
pub const Norms = struct { q: u64 = 0, k: u64 = 0, iq: u64 = 0, ik: u64 = 0 };

/// bf16 scratch rows for q [rows, nq * 128] and the index query [rows, niq * 128].
pub const Outputs = struct { q: u64, q_ld: usize, iq: u64 = 0, iq_ld: usize = 0 };

pub const Router = struct { experts: u32 = 128, k: u32 = 6144 };

pub const Rows = struct {
    mod: Module,
    rows_fn: Function,
    bf16_fn: Function,
    f32_fn: Function,

    pub fn load(d: *const Driver) !Rows {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var m = try Module.load(d, kernels.minimax_m3_rows);
        errdefer m.unload();
        return .{
            .mod = m,
            .rows_fn = try m.function("tf_mm3_rows"),
            .bf16_fn = try m.function("tf_mm3_router_bf16"),
            .f32_fn = try m.function("tf_mm3_router_f32"),
        };
    }

    pub fn unload(g: *Rows) void {
        g.mod.unload();
    }

    fn cfg(l: geom.Launch) launch_.Config {
        return .{ .grid = .{ .x = l.grid[0], .y = l.grid[1], .z = l.grid[2] }, .block = .{ .x = l.block } };
    }

    /// Norm and rotate the heads of `rows` into scratch and the pages `p` names; `status` (int32 or 0) gets bits.
    pub fn qkRope(g: *const Rows, st: Stream, h: Heads, o: Opts, in: Inputs, nm: Norms, out: Outputs, rows: sp.Rows, p: sp.Paged, max_pages: u32, status: u64) !void {
        const launch = try geom.rowsLaunch(rows.n);
        if (p.kvh != h.kvh or h.nq == 0 or h.nq % h.kvh != 0) return error.Heads;
        try geom.checkRows(o.rot, h.nq * geom.d, in.q_ld);
        try geom.checkRows(o.rot, h.nq * geom.d, out.q_ld);
        try geom.checkRows(o.rot, h.kvh * geom.d, in.k_ld);
        try geom.checkRows(o.rot, h.kvh * geom.d, in.v_ld);
        if (h.niq > 0) {
            try geom.checkRows(o.rot, h.niq * geom.d, in.iq_ld);
            try geom.checkRows(o.rot, h.niq * geom.d, out.iq_ld);
            if (in.iq == 0 or out.iq == 0) return error.Heads;
        }
        if (h.index_key) try geom.checkRows(o.rot, geom.d, in.ik_ld);
        if (h.index_key and in.ik == 0) return error.Heads;
        const f = o.flags;
        if ((f.norm_qk and (nm.q == 0 or nm.k == 0)) or (f.norm_index and ((h.niq > 0 and nm.iq == 0) or (h.index_key and nm.ik == 0)))) return error.Heads;
        if (p.tables == 0 or p.pool == 0 or max_pages == 0) return error.Heads;
        if (f.kv8 != p.kv8) return error.Heads; // the flag and the pool's layout must say the same thing
        var a: launch_.Args = .{};
        a.add(in.q);
        a.add(@as(i64, @intCast(in.q_ld)));
        a.add(in.k);
        a.add(@as(i64, @intCast(in.k_ld)));
        a.add(in.v);
        a.add(@as(i64, @intCast(in.v_ld)));
        a.add(in.iq);
        a.add(@as(i64, @intCast(in.iq_ld)));
        a.add(in.ik);
        a.add(@as(i64, @intCast(in.ik_ld)));
        for ([_]u64{ nm.q, nm.k, nm.iq, nm.ik }) |w| a.add(w);
        a.add(out.q);
        a.add(@as(i64, @intCast(out.q_ld)));
        a.add(out.iq);
        a.add(@as(i64, @intCast(out.iq_ld)));
        a.add(rows.pos);
        a.add(rows.pos0);
        a.add(p.row_tab);
        a.add(p.tables);
        a.add(@as(i32, @intCast(max_pages)));
        a.add(p.pool);
        a.add(@as(i64, @intCast(p.page_bytes)));
        a.add(@as(i64, @intCast(p.layer_off)));
        a.add(@as(i32, @intCast(rows.n)));
        a.add(@as(i32, @intCast(h.nq)));
        a.add(@as(i32, @intCast(h.niq)));
        a.add(@as(i32, @intCast(h.kvh)));
        a.add(@as(i32, @intFromBool(h.index_key)));
        a.add(f.bits());
        a.add(@as(i32, @intCast(o.rot)));
        a.add(o.eps);
        a.add(o.w_add);
        a.add(o.w_add_index);
        a.add(o.theta);
        a.add(status);
        try launch_.launch(g.rows_fn, cfg(launch), st, &a);
    }

    /// logits [rows, experts] fp32 = x [rows, k] times w [experts, k] fp32; x bf16, or fp32 when `x_f32`.
    pub fn routerLogits(g: *const Rows, st: Stream, r: Router, x: u64, x_ld: usize, x_f32: bool, w: u64, out: u64, rows: usize) !void {
        const launch = try geom.routerLaunch(rows, r.experts);
        try geom.checkRouter(r.k, x_ld);
        if (x % 16 != 0 or w % 16 != 0 or x_ld % 4 != 0) return error.Heads;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(x_ld)));
        a.add(w);
        a.add(out);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(r.experts)));
        a.add(@as(i32, @intCast(r.k)));
        try launch_.launch(if (x_f32) g.f32_fn else g.bf16_fn, cfg(launch), st, &a);
    }
};

/// The pool the paged kernels address: base, page and layer strides, local kv heads.
pub const Cache = struct { pool: u64, page_bytes: u64, layer_off: u64, kvh: u32, kv8: bool = false };

/// Device copy of the block tables (int32 page ids, -1 padded, `max_pages` a slot) and the per-row offsets T5 reads.
pub const Mirror = struct {
    tables: memory.DeviceBuffer,
    row_tab: memory.DeviceBuffer,
    slots: usize,
    max_pages: usize,
    max_rows: usize,

    pub fn init(d: *const Driver, slots: usize, max_pages: usize, max_rows: usize) !Mirror {
        _ = try geom.entryOffset(slots, max_pages);
        var t = try memory.DeviceBuffer.alloc(d, slots * max_pages * @sizeOf(i32));
        errdefer t.free();
        try t.fill32(0xffffffff, null);
        const rt = try memory.DeviceBuffer.alloc(d, max_rows * @sizeOf(i32));
        return .{ .tables = t, .row_tab = rt, .slots = slots, .max_pages = max_pages, .max_rows = max_rows };
    }

    pub fn deinit(m: *Mirror) void {
        m.tables.free();
        m.row_tab.free();
    }

    /// Entries [from, from + entries.len) of `slot`'s row, copied to the device once they have landed.
    pub fn put(m: *Mirror, slot: usize, from: usize, entries: []const i32) !void {
        const r = try geom.entryBytes(m.slots, m.max_pages, slot, from, entries.len);
        try m.tables.upload(r.off, std.mem.sliceAsBytes(entries));
    }

    /// kv pool Tables' `Upload.call`: pass the Mirror as its ctx.
    pub fn call(ctx: *anyopaque, slot: u32, from: usize, entries: []const i32) anyerror!void {
        const m: *Mirror = @ptrCast(@alignCast(ctx));
        try m.put(slot, from, entries);
    }

    /// The whole host table (`slots * max_pages` entries) in one copy, e.g. after a restore.
    pub fn putAll(m: *Mirror, rows: []const i32) !void {
        if (rows.len != m.slots * m.max_pages) return error.TableFull;
        try m.tables.upload(0, std.mem.sliceAsBytes(rows));
    }

    /// The window's per-row offsets: `host` (one entry a row) is filled from `segs` and queued on `st`.
    pub fn setRows(m: *Mirror, st: Stream, host: []i32, segs: []const geom.Seg) !void {
        if (host.len == 0 or host.len > m.max_rows) return error.TooManyRows;
        try geom.fillRowTab(host, segs, m.slots, m.max_pages);
        try m.row_tab.uploadAsync(0, std.mem.sliceAsBytes(host), st.handle);
    }

    /// Several streams' rows in one call: every row reads the table at its own `row_tab` offset.
    pub fn shared(m: *const Mirror, c: Cache) sp.Paged {
        return .{ .pool = c.pool, .page_bytes = c.page_bytes, .layer_off = c.layer_off, .kvh = c.kvh, .tables = m.tables.ptr, .row_tab = m.row_tab.ptr, .kv8 = c.kv8 };
    }

    /// One stream's rows (a prompt chunk): the table pointer is the slot's row and `row_tab` is 0.
    pub fn single(m: *const Mirror, c: Cache, slot: usize) !sp.Paged {
        const r = try geom.entryBytes(m.slots, m.max_pages, slot, 0, 0);
        return .{ .pool = c.pool, .page_bytes = c.page_bytes, .layer_off = c.layer_off, .kvh = c.kvh, .tables = m.tables.ptr + r.off, .kv8 = c.kv8 };
    }
};

test {
    _ = geom;
}
