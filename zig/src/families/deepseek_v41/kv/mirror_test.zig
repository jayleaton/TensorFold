//! TF_DSV41_KV_MIRROR on a host model of W ranks: mirror rows equal split KV's gathered rows; its fill, the stores.

const std = @import("std");
const testing = std.testing;
const mirror = @import("mirror.zig");
const split = @import("split.zig");
const layout = @import("layout.zig");
const Geometry = mirror.Geometry;

test "kv mirror: the knob unset, blank or 0 is off; positions round up to whole pages of every residue" {
    try testing.expectEqual(@as(u64, 0), try mirror.parse(null, 256, 2));
    try testing.expectEqual(@as(u64, 0), try mirror.parse(" ", 256, 2));
    try testing.expectEqual(@as(u64, 0), try mirror.parse("0", 256, 2));
    try testing.expectEqual(@as(u64, 512), try mirror.parse("1", 256, 2));
    try testing.expectEqual(@as(u64, 2048), try mirror.parse(" 2048 ", 256, 2));
    try testing.expectEqual(@as(u64, 2560), try mirror.parse("2049", 256, 2));
    try testing.expectEqual(@as(u64, 2304), try mirror.parse("2049", 256, 3));
    for ([_][]const u8{ "-1", "on", "1e3", "2048 rows", "0x800", "99999999999999" }) |v|
        try testing.expectError(error.BadMirror, mirror.parse(v, 256, 2));
}

test "kv mirror: each slot's run is owner-major, every page past it the discard page; the release's bytes" {
    const g = Geometry.of(4096, 256, 3, 2);
    try testing.expectEqual(@as(u32, 16), g.pages);
    try testing.expectEqual(@as(u32, 49), g.tensorPages());
    var seen: [48]bool = @splat(false);
    for (0..3) |s| for (0..16) |p| {
        const m = g.pageOf(@intCast(s), @intCast(p));
        try testing.expect(m >= s * 16 and m < (s + 1) * 16 and !seen[m]);
        seen[m] = true;
    };
    try testing.expectEqual(@as(u32, 16 + 8 + 1), g.pageOf(1, 3)); // owner 1, turn 1
    try testing.expectEqual(g.discard(), g.pageOf(2, 16));
    var tab: [3 * 20]i32 = undefined;
    g.table(20, &tab);
    try testing.expectEqual(@as(i32, 48), tab[2 * 20 + 19]);
    try testing.expectEqual(@as(i32, 32), tab[2 * 20]);
    // the release's comp families (three ratio 2, one ratio 1): 4096 x 1,460 B a slot, plus the discard pages
    var l = try layout.Layout.fromConfig(layout.Release{}, .{ .split = true });
    const one = Geometry.of(4096, 256, 1, 2);
    const want: u64 = 4 * 20 + 17 * (3 * 128 + 256) * 584;
    try testing.expectEqual(want, one.bytes(l.families(), 256, 20));
}

/// W ranks of split storage for `slots` slots of `pts` logical pages (page rows 1 << psh), rows of seeded bytes.
const Model = struct {
    world: u32,
    rank_rows: [3][]u8,
    /// [rank][slots x pts] split tables: a rank's local page, or its discard page for the others' pages
    tables: [3][]i32,
    /// every slot's logical rows, [slots, pts << psh, rb]
    rows: []u8,
    slots: u32,
    pts: u32,
    psh: u5,
    rb: u32,

    fn init(a: std.mem.Allocator, rnd: std.Random, world: u32, slots: u32, pts: u32, psh: u5, rb: u32) !Model {
        var m: Model = .{ .world = world, .rank_rows = undefined, .tables = undefined, .rows = undefined, .slots = slots, .pts = pts, .psh = psh, .rb = rb };
        const q = @as(usize, 1) << psh;
        m.rows = try a.alloc(u8, @as(usize, slots) * pts * q * rb);
        rnd.bytes(m.rows);
        const owned = @as(usize, slots) * ((pts + world - 1) / world);
        for (0..world) |o| {
            // local pages in a shuffled order, then the discard page (junk: the others' rows land there)
            m.rank_rows[o] = try a.alloc(u8, (owned + 1) * q * rb);
            rnd.bytes(m.rank_rows[o]);
            m.tables[o] = try a.alloc(i32, @as(usize, slots) * pts);
            const order = try a.alloc(u32, owned);
            for (order, 0..) |*x, i| x.* = @intCast(i);
            rnd.shuffle(u32, order);
            var next: usize = 0;
            for (0..slots) |s| for (0..pts) |p| {
                const at = s * pts + p;
                if (p % world != o) {
                    m.tables[o][at] = @intCast(owned);
                    continue;
                }
                const lp = order[next];
                next += 1;
                m.tables[o][at] = @intCast(lp);
                @memcpy(m.rank_rows[o][lp * q * rb ..][0 .. q * rb], m.rows[at * q * rb ..][0 .. q * rb]);
            };
        }
        return m;
    }

    fn row(m: *const Model, s: usize, t: usize) []const u8 {
        const q = @as(usize, 1) << m.psh;
        return m.rows[(s * m.pts * q + t) * m.rb ..][0..m.rb];
    }
};

/// The mirror `_kv_store` leaves: every slot's logical row through the mirror table (rows past it on the discard page).
fn stored(a: std.mem.Allocator, m: *const Model, g: Geometry) ![]u8 {
    const q = @as(usize, 1) << m.psh;
    const out = try a.alloc(u8, g.tensorPages() * q * m.rb);
    @memset(out, 0);
    for (0..m.slots) |s| for (0..m.pts * q) |t| {
        const pg = g.pageOf(@intCast(s), @intCast(t >> m.psh));
        if (pg == g.discard()) continue;
        @memcpy(out[(pg * q + t % q) * m.rb ..][0..m.rb], m.row(s, t));
    };
    return out;
}

test "kv mirror: a restore's fill (each rank's run gathered, all-gathered) is the stores' mirror, byte for byte" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x5151);
    for ([_][4]u32{ .{ 2, 3, 10, 7 }, .{ 2, 2, 8, 8 }, .{ 3, 2, 7, 7 } }) |c| {
        const world = c[0];
        const m = try Model.init(a, prng.random(), world, c[1], c[2], @intCast(c[3]), 40);
        const g = Geometry.of(@as(u64, 2 * world) * 256, 256, m.slots, world);
        const want = try stored(a, &m, g);
        var hk: split.HostKernels = .{};
        const kern = hk.kernels();
        const rows = g.fillRows(m.psh);
        const run_bytes = rows * world * m.rb;
        for (0..m.slots) |s| {
            const recv = try a.alloc(u8, run_bytes);
            for (0..world) |o| {
                const local = try a.alloc(u32, g.pages);
                for (local, 0..) |*x, p| x.* = @intCast(m.tables[o][s * m.pts + p]);
                const phys = try a.alloc(u32, rows);
                g.fillPhys(@intCast(o), m.psh, local, phys);
                // the gather into send, then the all-gather: owner o's rows at o x rows
                const send = recv[o * rows * m.rb ..][0 .. rows * m.rb];
                try kern.vtable.gather(kern.ptr, @intFromPtr(m.rank_rows[o].ptr), m.rb, @intFromPtr(phys.ptr), rows, @intFromPtr(send.ptr), null);
            }
            try testing.expectEqualSlices(u8, want[s * run_bytes ..][0..run_bytes], recv);
        }
    }
}

test "kv mirror: every selected row read through the mirror table equals split KV's gathered row (dense and packed)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x7171);
    const rnd = prng.random();
    for ([_]u32{ 2, 3 }) |world| for ([_]bool{ false, true }) |stacked| {
        const m = try Model.init(a, rnd, world, 4, 12, 7, 584);
        const g = Geometry.of(@as(u64, 4 * world) * 256, 256, m.slots, world);
        const mir = try stored(a, &m, g);
        var mt: [4 * 12]i32 = undefined;
        g.table(m.pts, &mt);
        const R: u32 = 6;
        const K: u32 = 64;
        const n = R * K;
        // selections inside the mirror: visible rows only (the emitter's condition), -1 padding
        const visible = g.pages << m.psh;
        const sel = try a.alloc(i32, n);
        for (sel) |*x| x.* = if (rnd.uintLessThan(u32, 6) == 0) -1 else @intCast(rnd.uintLessThan(u32, visible));
        const rslot = try a.alloc(i32, R);
        for (rslot) |*x| x.* = if (stacked) @intCast(rnd.uintLessThan(u32, m.slots)) else 0;
        var hk: split.HostKernels = .{};
        const kern = hk.kernels();
        for ([_]bool{ false, true }) |packed_| {
            const recv = try a.alloc(u8, world * n * m.rb);
            const tok = try a.alloc(i32, n);
            for (0..world) |o| {
                const send = try a.alloc(u8, n * m.rb);
                const lens = try a.alloc(i32, world);
                const d: split.DenseArgs = .{ .sel = @intFromPtr(sel.ptr), .rows = R, .k = K, .table = @intFromPtr(m.tables[o].ptr), .pts = if (stacked) m.pts else 0, .rslot = if (stacked) @intFromPtr(rslot.ptr) else 0, .psh = m.psh, .base = @intFromPtr(m.rank_rows[o].ptr), .row_bytes = m.rb, .world = world, .send = @intFromPtr(send.ptr), .tok = @intFromPtr(tok.ptr) };
                if (packed_) try kern.vtable.pack.?(kern.ptr, split.PackArgs.of(d, @intCast(o), @intFromPtr(lens.ptr)), null) else try kern.vtable.dense(kern.ptr, d, null);
                // the all-gather: rank o's shard at o x R x K rows (a packed shard's own rows first)
                @memcpy(recv[o * n * m.rb ..][0 .. n * m.rb], send);
            }
            var compared: usize = 0;
            for (0..n) |i| {
                try testing.expectEqual(sel[i] < 0, tok[i] < 0);
                if (sel[i] < 0) continue;
                const t: u32 = @intCast(sel[i]);
                const s: usize = @intCast(rslot[i / K]);
                const q = @as(u32, 1) << m.psh;
                // the attention's address through a page table: (pt[slot x pts + t >> psh] << psh) + (t & mask)
                const phys = (@as(usize, @intCast(mt[s * m.pts + (t >> m.psh)])) << m.psh) + (t & (q - 1));
                const got = recv[@as(usize, @intCast(tok[i])) * m.rb ..][0..m.rb];
                try testing.expectEqualSlices(u8, got, mir[phys * m.rb ..][0..m.rb]);
                try testing.expectEqualSlices(u8, m.row(s, t), got);
                compared += 1;
            }
            try testing.expect(compared > n / 2);
        }
    };
}
