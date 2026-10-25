//! MiniMax-M3.1-Flash row kernels, host only: table mirror arithmetic, paged addressing, host twins, mutants, records.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const pool_mod = @import("mm3_pool");
const plan = @import("mm3_plan");
const twin = @import("minimax_rows_twin.zig");

const mr = core.minimax_ref;
const c = core.contract.spec;
const g = cuda.minimax_m3_rows.geom;
const geom = cuda.minimax_m3_sparse.geom;
const testing = std.testing;
const registry = core.contract.registry;

test {
    _ = plan;
    _ = cuda.minimax_m3_rows;
}

fn must(ok: bool, src: std.builtin.SourceLocation) !void {
    if (!ok) {
        std.debug.print("expected true at line {d}\n", .{src.line});
        return error.TestExpectedTrue;
    }
}

fn bf16v(rng: std.Random, n: usize, sd: f32) ![]f32 {
    const v = try testing.allocator.alloc(f32, n);
    for (v) |*x| x.* = c.fromBf16(c.toBf16(rng.floatNorm(f32) * sd));
    return v;
}

fn fp16v(rng: std.Random, n: usize, sd: f32) ![]f32 {
    const v = try testing.allocator.alloc(f32, n);
    for (v) |*x| {
        const h: f16 = @floatCast(rng.floatNorm(f32) * sd);
        x.* = h;
    }
    return v;
}

fn weights(rng: std.Random) [128]f32 {
    var w: [128]f32 = undefined;
    for (&w) |*x| x.* = 1 + rng.floatNorm(f32) * 0.3;
    return w;
}

test "block table offsets: slot * max_pages entries, int32 overflow refused" {
    try testing.expectEqual(@as(i32, 0), try g.entryOffset(0, 1024));
    try testing.expectEqual(@as(i32, 3 * 1024), try g.entryOffset(3, 1024));
    try testing.expectEqual(@as(i32, std.math.maxInt(i32)), try g.entryOffset(std.math.maxInt(i32), 1));
    try testing.expectError(error.Overflow, g.entryOffset(1 << 20, 1 << 12));
    try testing.expectError(error.Overflow, g.entryOffset(std.math.maxInt(usize), 2));
    const r = try g.entryBytes(4, 1024, 2, 10, 5);
    try testing.expectEqual(@as(usize, (2 * 1024 + 10) * 4), r.off);
    try testing.expectEqual(@as(usize, 20), r.len);
    try testing.expectError(error.BadSlot, g.entryBytes(4, 1024, 4, 0, 1));
    try testing.expectError(error.TableFull, g.entryBytes(4, 1024, 0, 1020, 5));
    _ = try g.entryBytes(4, 1024, 3, 1020, 4);
}

test "row_tab: segments tile the rows, gaps and bad slots are refused" {
    var out: [7]i32 = undefined;
    try g.fillRowTab(&out, &.{ .{ .slot = 2, .rows = 3 }, .{ .slot = 0, .rows = 1 }, .{ .slot = 1, .rows = 3 } }, 3, 64);
    try testing.expectEqualSlices(i32, &.{ 128, 128, 128, 0, 64, 64, 64 }, &out);
    try testing.expectError(error.Gap, g.fillRowTab(&out, &.{.{ .slot = 0, .rows = 6 }}, 3, 64));
    try testing.expectError(error.Gap, g.fillRowTab(&out, &.{ .{ .slot = 0, .rows = 4 }, .{ .slot = 1, .rows = 4 } }, 3, 64));
    try testing.expectError(error.BadSlot, g.fillRowTab(&out, &.{.{ .slot = 3, .rows = 7 }}, 3, 64));
}

/// A device copy kept as bytes, written through the same ranges Mirror.put uses.
const FakeDevice = struct {
    bytes: []u8,
    slots: usize,
    max_pages: usize,
    fail: bool = false,

    fn call(ctx: *anyopaque, slot: u32, from: usize, entries: []const i32) anyerror!void {
        const self: *FakeDevice = @ptrCast(@alignCast(ctx));
        const r = try g.entryBytes(self.slots, self.max_pages, slot, from, entries.len);
        if (self.fail) return error.UploadFailed;
        @memcpy(self.bytes[r.off..][0..r.len], std.mem.sliceAsBytes(entries));
    }
};

test "the byte ranges keep a device copy equal to the host Tables, through writes, a failed upload and a clear" {
    const slots = 3;
    const max_pages = 40;
    var t = try pool_mod.Tables.init(testing.allocator, slots, max_pages);
    defer t.deinit();
    const bytes = try testing.allocator.alloc(u8, slots * max_pages * 4);
    defer testing.allocator.free(bytes);
    @memset(bytes, 0xff);
    var dev: FakeDevice = .{ .bytes = bytes, .slots = slots, .max_pages = max_pages };
    t.upload = .{ .ctx = &dev, .call = FakeDevice.call };
    try t.write(1, &.{ 70, 71, 72 }, 0);
    try t.write(2, &.{ 5, 9 }, 0);
    try t.write(1, &.{ 73, 74 }, 3);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(t.rows), bytes);
    dev.fail = true;
    try testing.expectError(error.UploadFailed, t.write(2, &.{ 11, 12 }, 2));
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(t.rows), bytes);
    dev.fail = false;
    try testing.expectError(error.TableFull, t.write(0, &.{ 1, 2 }, 39));
    // a restore pushes whole rows, then the device agrees with every slot's entries
    for (0..slots) |s| try FakeDevice.call(&dev, @intCast(s), 0, t.row(@intCast(s)));
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(t.rows), bytes);
}

test "row_tab offsets index the real Tables: tables[row_tab[r] + pos / 128] is the page Tables.page names" {
    const max_pages = 40;
    var t = try pool_mod.Tables.init(testing.allocator, 4, max_pages);
    defer t.deinit();
    try t.write(0, &.{ 4, 5, 6 }, 0);
    try t.write(1, &.{ 20, 21 }, 0);
    try t.write(3, &.{ 31, 30, 29, 28 }, 0);
    var rt: [9]i32 = undefined;
    try g.fillRowTab(&rt, &.{ .{ .slot = 3, .rows = 4 }, .{ .slot = 0, .rows = 2 }, .{ .slot = 1, .rows = 3 } }, 4, max_pages);
    const pos = [_]i32{ 0, 127, 128, 511, 300, 5, 255, 200, 129 };
    try g.checkCall(t.rows, max_pages, &rt, &pos, 0, 9);
    for (rt, pos, 0..) |off, p, r| {
        const slot: u32 = @intCast(@divExact(off, max_pages));
        const want = t.page(slot, @intCast(@divTrunc(p, 128))).?;
        try testing.expectEqual(want, try g.pageOf(t.rows, max_pages, @intCast(off), @intCast(p)));
        _ = r;
    }
    // slot 1 owns two pages: its row at position 256 has no page, and positions past the row are another error
    try testing.expectError(error.NoPage, g.checkCall(t.rows, max_pages, &rt, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 256 }, 0, 9));
    try testing.expectError(error.PastTable, g.pageOf(t.rows, max_pages, 0, max_pages * 128));
    // a prompt chunk of one stream: no row_tab, the table pointer is the slot's row (entry offset slot * max_pages)
    const slot3 = t.rows[3 * max_pages ..];
    try g.checkCall(slot3, max_pages, null, null, 384, 100);
    try testing.expectError(error.NoPage, g.checkCall(slot3, max_pages, null, null, 384, 129));
    // padded rows (negative positions) are skipped
    try g.checkCall(t.rows, max_pages, null, &.{ -1, 5 }, 0, 2);
    try testing.expectEqual(@as(usize, 1), g.pagesNeeded(127));
    try testing.expectEqual(@as(usize, 2), g.pagesNeeded(128));
}

test "kernel addressing equals the pool's page map and the T5 geometry for every (page, layer, K/V/IK, head, token)" {
    const kvh = 2;
    const layers = 3;
    const gp: geom.Pool = .{ .kvh = kvh, .layers = layers };
    const dims: pool_mod.Dims = .{ .layers = layers, .kv_heads = kvh, .head_dim = 128, .index_dim = 128 };
    const map = dims.map();
    try testing.expectEqual(@as(u64, gp.pageBytes()), dims.pageBytes());
    for ([_]usize{ 0, 1, 6 }) |page| for (0..layers) |ly| for ([_]usize{ 0, 1, 77, 127 }) |t| {
        for (0..kvh) |h| {
            const k = page * gp.pageBytes() + gp.layerOff(ly) + twin.elem(kvh, 0, h, t) * 2;
            const v = page * gp.pageBytes() + gp.layerOff(ly) + twin.elem(kvh, 1, h, t) * 2;
            try testing.expectEqual(gp.kv(page, ly, 0, h, t), k);
            try testing.expectEqual(gp.kv(page, ly, 1, h, t), v);
            try testing.expectEqual(map.page(@intCast(page)) + map.k(@intCast(ly), @intCast(h)) + t * 256, k);
            try testing.expectEqual(map.page(@intCast(page)) + map.v(@intCast(ly), @intCast(h)) + t * 256, v);
        }
        const ik = page * gp.pageBytes() + gp.layerOff(ly) + twin.ikElem(kvh, t) * 2;
        try testing.expectEqual(gp.ik(page, ly, t), ik);
        try testing.expectEqual(map.page(@intCast(page)) + map.ik(@intCast(ly)) + t * 256, ik);
    };
}

var tol_rel: f64 = 0x1p-8;

fn near(got: u16, want: f64) bool {
    const v: f64 = c.fromBf16(got);
    return @abs(v - want) <= @abs(want) * tol_rel + 2e-5;
}

/// The float64 reference of one head: Gemma norm with weight w + add (if on), bf16 rounding (if asked), partial RoPE.
fn refHead(x: []const f32, w: []const f32, norm_on: bool, add: f64, rope_on: bool, pos: usize, o: twin.Opts) [128]f64 {
    var xs: [128]f64 = undefined;
    for (&xs, x[0..128]) |*d, s| d.* = s;
    if (norm_on) {
        var ws: [128]f64 = undefined;
        for (&ws, w[0..128]) |*d, v| d.* = v + add;
        mr.gemmaNorm(&xs, &ws, o.eps, false, &xs);
        mr.roundAll(o.flags.round_norm, &xs);
    }
    if (rope_on) mr.rope(&xs, o.rot, pos, o.theta, if (o.flags.gptj) .gptj_interleaved else .neox_half);
    return xs;
}

const fx_kvh = 2;
const fx_pages = 6;
const n_pages = 9;
const fx_layers = 2;
const layer = 1;
const pool_geom: geom.Pool = .{ .kvh = fx_kvh, .layers = fx_layers };

/// Random inputs for `rows` rows of a rank (32 q, 2 index, 2 kv heads), a shuffled table and a NaN-poisoned pool.
const Fix = struct {
    rows: usize,
    h: twin.Heads = .{},
    q: []f32,
    k: []f32,
    v: []f32,
    iq: []f32,
    ik: []f32,
    nm: twin.Norms,
    pool: []u16,
    tables: []i32,
    q_out: []u16,
    iq_out: []u16,

    fn init(rng: std.Random, rows: usize, index_fp16: bool) !Fix {
        const a = testing.allocator;
        const h: twin.Heads = .{};
        const pool = try a.alloc(u16, n_pages * pool_geom.pageBytes() / 2);
        @memset(pool, c.toBf16(std.math.nan(f32)));
        const tables = try a.alloc(i32, 2 * fx_pages);
        @memset(tables, -1);
        var perm: [n_pages]i32 = undefined;
        for (&perm, 0..) |*x, i| x.* = @intCast(i);
        rng.shuffle(i32, &perm);
        @memcpy(tables[fx_pages .. fx_pages + 5], perm[0..5]);
        @memcpy(tables[0..3], perm[5..8]);
        const mk = if (index_fp16) &fp16v else &bf16v;
        return .{
            .rows = rows,
            .q = try bf16v(rng, rows * h.nq * 128, 2),
            .k = try bf16v(rng, rows * fx_kvh * 128, 2),
            .v = try bf16v(rng, rows * fx_kvh * 128, 2),
            .iq = try mk(rng, rows * h.niq * 128, 2),
            .ik = try mk(rng, rows * 128, 2),
            .nm = .{ .q = weights(rng), .k = weights(rng), .iq = weights(rng), .ik = weights(rng) },
            .pool = pool,
            .tables = tables,
            .q_out = try a.alloc(u16, rows * h.nq * 128),
            .iq_out = try a.alloc(u16, rows * h.niq * 128),
        };
    }

    fn deinit(f: *Fix) void {
        inline for (.{ "q", "k", "v", "iq", "ik", "pool", "tables", "q_out", "iq_out" }) |n| testing.allocator.free(@field(f, n));
    }

    fn target(f: *Fix, row_tab: ?[]const i32, pos: ?[]const i32, pos0: i32) twin.Target {
        return .{ .q = f.q_out, .iq = f.iq_out, .pool = f.pool, .tables = f.tables, .max_pages = fx_pages, .row_tab = row_tab, .pos = pos, .pos0 = pos0, .page_bytes = pool_geom.pageBytes(), .layer_off = pool_geom.layerOff(layer) };
    }

    fn inputs(f: *const Fix) twin.Inputs {
        return .{ .q = f.q, .k = f.k, .v = f.v, .iq = f.iq, .ik = f.ik };
    }

    /// The pool's row for (page id, K/V, head, token).
    fn cell(f: *const Fix, page: usize, which: usize, head: usize, t: usize) []const u16 {
        return f.pool[pool_geom.kv(page, layer, which, head, t) / 2 ..][0..128];
    }

    fn ikCell(f: *const Fix, page: usize, t: usize) []const u16 {
        return f.pool[pool_geom.ik(page, layer, t) / 2 ..][0..128];
    }
};

fn expectRow(got: []const u16, want: [128]f64) !void {
    for (got, want) |x, y| try testing.expect(near(x, y));
}

/// Every output of a call against the float64 reference: scratch q and iq, and K, V and IK in their pages.
fn checkCall(f: *const Fix, o: twin.Opts, pos: []const i32, row_tab: ?[]const i32) !void {
    const fl = o.flags;
    tol_rel = if (fl.round_norm) 0x1p-6 else 0x1p-8; // a value near a bf16 rounding boundary may round either way in the middle
    defer tol_rel = 0x1p-8;
    for (0..f.rows) |r| {
        const p: usize = @intCast(pos[r]);
        const off: usize = if (row_tab) |t| @intCast(t[r]) else 0;
        const page: usize = @intCast(f.tables[off + p / 128]);
        for (0..f.h.nq) |i| try expectRow(f.q_out[(r * f.h.nq + i) * 128 ..][0..128], refHead(f.q[(r * f.h.nq + i) * 128 ..], &f.nm.q, fl.norm_qk, o.w_add, true, p, o));
        for (0..f.h.niq) |i| try expectRow(f.iq_out[(r * f.h.niq + i) * 128 ..][0..128], refHead(f.iq[(r * f.h.niq + i) * 128 ..], &f.nm.iq, fl.norm_index, o.w_add_index, fl.index_rope, p, o));
        for (0..fx_kvh) |i| {
            try expectRow(f.cell(page, 0, i, p % 128), refHead(f.k[(r * fx_kvh + i) * 128 ..], &f.nm.k, fl.norm_qk, o.w_add, true, p, o));
            for (f.cell(page, 1, i, p % 128), f.v[(r * fx_kvh + i) * 128 ..][0..128]) |x, y| try testing.expectEqual(c.toBf16(y), x);
        }
        try expectRow(f.ikCell(page, p % 128), refHead(f.ik[r * 128 ..], &f.nm.ik, fl.norm_index, o.w_add_index, fl.index_rope, p, o));
    }
}

fn okCall(f: *const Fix, o: twin.Opts, pos: []const i32, row_tab: ?[]const i32) bool {
    checkCall(f, o, pos, row_tab) catch return false;
    return true;
}

test "a decode window across a page edge and a prompt chunk: every output matches the float64 reference" {
    var prng = std.Random.DefaultPrng.init(21);
    const rng = prng.random();
    // slot 0 (table row 0) holds 3 pages, slot 1 (row at entry 6) holds 5; the window mixes both streams
    var f = try Fix.init(rng, 5, true);
    defer f.deinit();
    const pos = [_]i32{ 126, 127, 128, 129, 300 };
    const row_tab = [_]i32{ 0, 0, 0, 0, fx_pages };
    const o: twin.Opts = .{};
    try testing.expectEqual(@as(i32, 0), twin.call(5, f.h, o, f.inputs(), &f.nm, f.target(&row_tab, &pos, 0), .none));
    try checkCall(&f, o, &pos, &row_tab);
    // a prompt chunk of slot 1: positions pos0 + r, one table, the same rows' bits as the window gave
    var p = try Fix.init(rng, 5, true);
    defer p.deinit();
    try testing.expectEqual(@as(i32, 0), twin.call(5, p.h, o, p.inputs(), &p.nm, p.target(null, null, 125), .none));
    try checkCall(&p, o, &.{ 125, 126, 127, 128, 129 }, null);
}

/// The weights as the loader would store them raw: the folded 1 + w minus 1, for a kernel that adds 1 itself.
fn rawNorms(f: *Fix) void {
    for (&f.nm.q, &f.nm.k, &f.nm.iq, &f.nm.ik) |*a, *b, *cc, *d| {
        a.* -= 1;
        b.* -= 1;
        cc.* -= 1;
        d.* -= 1;
    }
}

test "the switches: GPT-J layout, no index rope, bf16 norm rounding, raw weights with the offset, bf16 index inputs" {
    var prng = std.Random.DefaultPrng.init(22);
    const rng = prng.random();
    const variants = [_]twin.Opts{
        .{ .flags = .{ .gptj = true } },
        .{ .flags = .{ .index_rope = false } },
        .{ .flags = .{ .round_norm = true } },
        .{ .w_add = 1, .w_add_index = 1 },
        .{ .flags = .{ .index_fp16 = false } },
        .{ .flags = .{ .norm_qk = false, .norm_index = false } },
    };
    for (variants) |o| {
        var f = try Fix.init(rng, 3, o.flags.index_fp16);
        defer f.deinit();
        if (o.w_add == 1) rawNorms(&f);
        const pos = [_]i32{ 3, 130, 4 * 128 + 127 };
        const rt = [_]i32{ fx_pages, fx_pages, fx_pages };
        try testing.expectEqual(@as(i32, 0), twin.call(3, f.h, o, f.inputs(), &f.nm, f.target(&rt, &pos, 0), .none));
        try checkCall(&f, o, &pos, &rt);
    }
}

test "a row's bits do not depend on the window: alone, with neighbours, or at another row index" {
    var prng = std.Random.DefaultPrng.init(23);
    const rng = prng.random();
    var f = try Fix.init(rng, 4, true);
    defer f.deinit();
    const o: twin.Opts = .{};
    const pos = [_]i32{ 200, 7, 515, 129 };
    const rt = [_]i32{ fx_pages, fx_pages, fx_pages, fx_pages };
    _ = twin.call(4, f.h, o, f.inputs(), &f.nm, f.target(&rt, &pos, 0), .none);
    const whole_q = try testing.allocator.dupe(u16, f.q_out);
    defer testing.allocator.free(whole_q);
    const whole_pool = try testing.allocator.dupe(u16, f.pool);
    defer testing.allocator.free(whole_pool);
    // row 2 alone: its inputs moved to row 0 of a one-row call
    var one = f;
    one.rows = 1;
    one.q = f.q[2 * 32 * 128 ..][0 .. 32 * 128];
    one.k = f.k[2 * fx_kvh * 128 ..][0 .. fx_kvh * 128];
    one.v = f.v[2 * fx_kvh * 128 ..][0 .. fx_kvh * 128];
    one.iq = f.iq[2 * 2 * 128 ..][0 .. 2 * 128];
    one.ik = f.ik[2 * 128 ..][0..128];
    one.q_out = try testing.allocator.alloc(u16, 32 * 128);
    defer testing.allocator.free(one.q_out);
    one.iq_out = try testing.allocator.alloc(u16, 2 * 128);
    defer testing.allocator.free(one.iq_out);
    one.pool = try testing.allocator.dupe(u16, f.pool);
    defer testing.allocator.free(one.pool);
    _ = twin.call(1, one.h, o, one.inputs(), &one.nm, one.target(&.{fx_pages}, &.{515}, 0), .none);
    try testing.expectEqualSlices(u16, whole_q[2 * 32 * 128 ..][0 .. 32 * 128], one.q_out);
    try testing.expectEqualSlices(u16, whole_pool, one.pool);
}

test "pages: a missing page or a position past the table sets the status bit and writes nothing to the pool" {
    var prng = std.Random.DefaultPrng.init(24);
    const rng = prng.random();
    var f = try Fix.init(rng, 3, true);
    defer f.deinit();
    const poison = try testing.allocator.dupe(u16, f.pool);
    defer testing.allocator.free(poison);
    // slot 0 has pages for blocks 0..2: block 3 has none; block 6 is past the 6-entry row
    const rt = [_]i32{ 0, 0, 0 };
    const st = twin.call(3, f.h, .{}, f.inputs(), &f.nm, f.target(&rt, &.{ 3 * 128, 6 * 128, -1 }, 0), .none);
    try testing.expectEqual(g.status_no_page | g.status_past_table, st);
    try testing.expectEqualSlices(u16, poison, f.pool);
    // q still lands for the rows with a position, and a padded row (-1) writes nothing at all
    try expectRow(f.q_out[0..128], refHead(f.q[0..128], &f.nm.q, true, 0, true, 3 * 128, .{}));
}

test "writer mutants: each planted fault fails the reference check the real twin passes" {
    var prng = std.Random.DefaultPrng.init(25);
    const rng = prng.random();
    const pos = [_]i32{ 1_000_003 % 640, 640 - 1, 130 };
    const rt = [_]i32{ fx_pages, fx_pages, fx_pages };
    for (std.enums.values(twin.Mutant)) |m| {
        if (m == .split_by_rows or m == .reorder_chains or m == .f32_angle) continue;
        var f = try Fix.init(rng, 3, true);
        defer f.deinit();
        var o: twin.Opts = .{};
        if (m == .no_plus_one) {
            o.w_add = 1;
            o.w_add_index = 1;
            rawNorms(&f);
        }
        _ = twin.call(3, f.h, o, f.inputs(), &f.nm, f.target(&rt, &pos, 0), m);
        try testing.expectEqual(m == .none, okCall(&f, o, &pos, &rt));
    }
}

test "rows aliasing one (page, slot) cannot all be checked: the check flags the planted alias, distinct streams pass" {
    // The GPU contract once put 8 decode rows at position 0 behind one table: one slot, one survivor; see it.
    var prng = std.Random.DefaultPrng.init(26);
    const rng = prng.random();
    const pos = [_]i32{ 0, 0 };
    var f = try Fix.init(rng, 2, true);
    defer f.deinit();
    const alias = [_]i32{ 0, 0 };
    _ = twin.call(2, f.h, .{}, f.inputs(), &f.nm, f.target(&alias, &pos, 0), .none);
    try testing.expect(!okCall(&f, .{}, &pos, &alias));
    // the same positions in two streams (two tables, two pages): each row's cells are its own
    var h = try Fix.init(rng, 2, true);
    defer h.deinit();
    const own = [_]i32{ 0, fx_pages };
    h.tables[fx_pages] = if (h.tables[0] == 0) 1 else 0;
    _ = twin.call(2, h.h, .{}, h.inputs(), &h.nm, h.target(&own, &pos, 0), .none);
    try testing.expect(okCall(&h, .{}, &pos, &own));
}

test "angles: float64 sincos rounded once; an fp32 angle is off by whole bf16 steps at a million positions" {
    const pos = 1_048_573;
    var worst: f64 = 0;
    var worst_bad: f64 = 0;
    for (0..32) |i| {
        const inv = std.math.pow(f64, 1e7, -2 * @as(f64, @floatFromInt(i)) / 64);
        const ang = @as(f64, pos) * inv;
        const good = twin.angle(pos, i, 64, 1e7, .none);
        const bad = twin.angle(pos, i, 64, 1e7, .f32_angle);
        worst = @max(worst, @abs(@as(f64, good[0]) - @cos(ang)), @abs(@as(f64, good[1]) - @sin(ang)));
        worst_bad = @max(worst_bad, @abs(@as(f64, bad[0]) - @cos(ang)), @abs(@as(f64, bad[1]) - @sin(ang)));
    }
    try testing.expect(worst < 1e-7);
    try testing.expect(worst_bad > 1e-3);
}

const hidden = 6144;
const experts = 128;

/// tf_mm3_router_* over a call: every (row, expert) is its own fixed-order dot, whichever tile it falls in.
fn routerCall(x: []const f32, w: []const f32, rows: usize, m: twin.Mutant, out: []f32) void {
    for (0..rows) |r| for (0..experts) |e| {
        out[r * experts + e] = twin.router(x[r * hidden ..][0..hidden], w[e * hidden ..][0..hidden], hidden, rows, m);
    };
}

fn wide(a: std.mem.Allocator, v: []const f32) ![]f64 {
    const o = try a.alloc(f64, v.len);
    for (o, v) |*d, s| d.* = s;
    return o;
}

test "router: fp32 logits track the float64 dot within the fp32 chain bound, for 1 to 17 rows" {
    var prng = std.Random.DefaultPrng.init(31);
    const rng = prng.random();
    const x = try bf16v(rng, 17 * hidden, 1.5);
    defer testing.allocator.free(x);
    const w = try testing.allocator.alloc(f32, experts * hidden);
    defer testing.allocator.free(w);
    for (w) |*v| v.* = rng.floatNorm(f32) * 0.03;
    const out = try testing.allocator.alloc(f32, 17 * experts);
    defer testing.allocator.free(out);
    routerCall(x, w, 17, .none, out);
    // depth: 48 fma steps a lane chain, the 4-chain sum and the 5-level tree
    const u: f64 = 0x1p-24;
    for (0..17) |r| for (0..experts) |e| {
        var sum: f64 = 0;
        var mag: f64 = 0;
        for (x[r * hidden ..][0..hidden], w[e * hidden ..][0..hidden]) |a, b| {
            sum += @as(f64, a) * b;
            mag += @abs(@as(f64, a) * b);
        }
        try testing.expect(@abs(out[r * experts + e] - sum) <= c.gamma(48 + 3 + 5, u) * mag);
    };
}

test "router: a row's bits are the same in a call of 1, 3, 4, 5 or 17 rows; a split-K by row count is caught" {
    var prng = std.Random.DefaultPrng.init(32);
    const rng = prng.random();
    const x = try bf16v(rng, 17 * hidden, 2);
    defer testing.allocator.free(x);
    const w = try testing.allocator.alloc(f32, experts * hidden);
    defer testing.allocator.free(w);
    for (w) |*v| v.* = rng.floatNorm(f32) * 0.05;
    const full = try testing.allocator.alloc(f32, 17 * experts);
    defer testing.allocator.free(full);
    routerCall(x, w, 17, .none, full);
    var part: [17 * experts]f32 = undefined;
    for ([_]usize{ 1, 3, 4, 5 }) |n| {
        routerCall(x[0 .. n * hidden], w, n, .none, part[0 .. n * experts]);
        try testing.expectEqualSlices(f32, full[0 .. n * experts], part[0 .. n * experts]);
    }
    // the same rows through a tail window: row 16 alone equals row 16 of 17
    routerCall(x[16 * hidden ..], w, 1, .none, part[0..experts]);
    try testing.expectEqualSlices(f32, full[16 * experts ..][0..experts], part[0..experts]);
    // the planted faults: a depth that follows the row count, and another order of the lane chains
    routerCall(x, w, 17, .split_by_rows, &part);
    var one: [experts]f32 = undefined;
    routerCall(x[0..hidden], w, 1, .split_by_rows, &one);
    try testing.expect(!std.mem.eql(f32, part[0..experts], one[0..]));
    routerCall(x, w, 17, .reorder_chains, &part);
    try testing.expect(!std.mem.eql(f32, full[0 .. 17 * experts], part[0 .. 17 * experts]));
}

test "router: NaN and infinities reach the logit as float64 arithmetic gives them" {
    var prng = std.Random.DefaultPrng.init(33);
    const rng = prng.random();
    const x = try bf16v(rng, 4 * hidden, 1);
    defer testing.allocator.free(x);
    const w = try testing.allocator.alloc(f32, experts * hidden);
    defer testing.allocator.free(w);
    for (w) |*v| v.* = rng.floatNorm(f32) * 0.05;
    x[0 * hidden + 77] = std.math.nan(f32);
    x[1 * hidden + 5000] = std.math.inf(f32);
    x[2 * hidden + 6143] = -std.math.inf(f32);
    x[2 * hidden + 3] = std.math.inf(f32);
    const out = try testing.allocator.alloc(f32, 4 * experts);
    defer testing.allocator.free(out);
    routerCall(x, w, 4, .none, out);
    for (0..4) |r| {
        const xr = try wide(testing.allocator, x[r * hidden ..][0..hidden]);
        defer testing.allocator.free(xr);
        for (0..experts) |e| {
            var sum: f64 = 0;
            for (xr, w[e * hidden ..][0..hidden]) |a, b| sum += a * b;
            try testing.expectEqual(c.classOf(sum), c.classOf(out[r * experts + e]));
        }
    }
}

test "gate 5: the three launches are registered, match the geometry, and fit every chip" {
    const want = [_]struct { sym: []const u8, threads: u32, static: u32 }{
        .{ .sym = "tf_mm3_rows", .threads = g.rows_threads, .static = 4608 },
        .{ .sym = "tf_mm3_router_bf16", .threads = g.router_threads, .static = 0 },
        .{ .sym = "tf_mm3_router_f32", .threads = g.router_threads, .static = 0 },
    };
    var mine: [3]registry.CudaRecord = undefined;
    for (want, &mine) |w, *m| {
        const rec = for (registry.cuda) |r| {
            if (std.mem.eql(u8, r.launch.symbol, w.sym)) break r;
        } else return error.RecordMissing;
        try testing.expectEqualStrings("minimax_m3_rows", rec.launch.image);
        try testing.expectEqual(w.threads, rec.launch.threads);
        try testing.expectEqual(@as(u32, 0), rec.launch.dynamic);
        for (rec.usage) |u| {
            try testing.expectEqual(w.static, u.static);
            try testing.expectEqual(w.threads, u.bound);
            try testing.expect(u.regs * w.threads <= 65536);
        }
        m.* = rec;
    }
    try testing.expectEqual(@as(usize, 0), try registry.checkCuda(testing.allocator, &mine, true));
    // shared bytes the writer declares: 64 cos/sin pairs of fp32 plus 8 staged heads of 128 fp32
    try testing.expectEqual(@as(u32, 2 * 64 * 4 + 8 * 128 * 4), mine_static());
    // a planted control: the same launch asking for more threads than its bound must be reported
    var over = mine[0];
    over.launch.threads = 512;
    try testing.expect(try registry.checkCuda(testing.allocator, &.{over}, true) > 0);
}

fn mine_static() u32 {
    return 2 * g.max_rot_pairs * 4 + g.rows_warps * g.d * 4;
}
