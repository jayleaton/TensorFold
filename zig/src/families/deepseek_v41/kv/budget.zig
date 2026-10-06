//! The KV side of the priced memory floor (memory.py budget / load_check / prefill_price / Floor.check, with split KV and the
//! streamed NVMe tier): how big a pool a rank can boot with, what a prompt or an NVMe resume costs while it runs, and whether an
//! admission fits both ranks' pages and memory. Pure arithmetic: the engine passes measured or planned figures.

const std = @import("std");
const sessions = @import("sessions");

pub const GiB: f64 = @floatFromInt(@as(u64, 1) << 30);

/// A slot's bounded state outside the pool (state.py: 43 SWA rings x 256 x 584, 3 ratio-2 carries, Engram lookback, DSpark taps).
pub const slot_bytes: u64 = 43 * 256 * 584 + 3 * 1024 * 4 + 3 * 4 + 3 * 5120 * 2 * 128;

pub const Knobs = struct {
    /// TF_DSV41_PREFILL_ROWS and TF_DSV41_INDEX_BUDGET_MIB
    prefill_rows: u32 = 2048,
    index_budget_mib: u32 = 256,
    /// G16's fitted terms (memory.py PF_*)
    pf_chunk_gib: f64 = 0.533,
    pf_host_gib: f64 = 0.6,
    pf_other_gib: f64 = 0.0,
    pf_index_gib: f64 = 1.067,
    index_positions: u64 = 32768,
    /// split KV's union receive cap (TF_DSV41_KV_SPLIT_UNION_MIB); 0 when unsplit
    union_cap: u64 = 0,
    /// the NVMe tier's staging set: a streamed park's or restore's whole host transient
    staging_bytes: u64 = 128 << 20,
    hard_gib: f64 = 4.0,
    target_gib: f64 = 5.0,
};

/// A rank's boot budget in GiB (memory.py Budget): what is allocated before the pool, and what must stay free.
pub const Boot = struct {
    available: f64 = 112.0,
    weights: f64,
    slots: u32 = 4,
    /// the planned workspace (graphs 1.2, prefill 1.3, select 0.25, drafting 0.35, slack 0.9)
    workspace: f64 = 4.0,
    runtime: f64 = 1.0,
    /// the RAM tier's bounded state (TF_DSV41_SESSION_RAM_MIB)
    session_ram: f64 = 0.25,

    pub fn fixed(b: Boot, k: Knobs) f64 {
        const rings = @as(f64, @floatFromInt(b.slots * slot_bytes)) / GiB;
        const staging = @as(f64, @floatFromInt(k.staging_bytes)) / GiB;
        const unions = 1.5 * @as(f64, @floatFromInt(k.union_cap)) / GiB;
        return b.weights + rings + b.workspace + b.runtime + b.session_ram + staging + unions;
    }

    /// The floor left with a pool of pool_bytes: under the hard floor the boot refuses.
    pub fn floor(b: Boot, k: Knobs, pool_bytes: u64) f64 {
        return b.available - b.fixed(k) - @as(f64, @floatFromInt(pool_bytes)) / GiB;
    }

    /// The most pool tokens that keep the hard floor, in whole pages and a multiple of world pages (a split pool's residues).
    pub fn poolTokens(b: Boot, k: Knobs, layout: sessions.Layout, world: u32) u64 {
        const spare = b.available - b.fixed(k) - k.hard_gib;
        if (spare <= 0) return 0;
        const per_page: f64 = @floatFromInt(layout.page);
        const page_bytes = layout.bytesPerToken(world) * per_page;
        const pages: u64 = @intFromFloat(@floor(spare * GiB / page_bytes));
        return pages / world * world * layout.page;
    }
};

/// A prompt's prefill transient in bytes (memory.py Price): once (the window's buffers) and index (grows to index_positions).
pub const Price = struct {
    once: u64 = 0,
    index: u64 = 0,
    start: u64 = 0,
    end: u64 = 0,

    pub fn total(p: Price) u64 {
        return p.once + p.index;
    }

    /// What has not materialised once the prompt's state holds done tokens (the floor's reservation for a prompt still prefilling).
    pub fn outstanding(p: Price, done: u64) u64 {
        if (p.total() == 0) return 0;
        const once = if (done <= p.start) p.once else 0;
        const span = p.end - p.start;
        const left = p.end -| @max(done, p.start);
        return once + if (span > 0) @as(u64, @intFromFloat(@as(f64, @floatFromInt(p.index)) * @as(f64, @floatFromInt(left)) / @as(f64, @floatFromInt(span)))) else 0;
    }
};

/// Prefilling prompt tokens [cached, n - 1): memory.py prefill_price, plus split KV's union buffers (receive cap + half for the send).
pub fn prefillPrice(k: Knobs, n: u64, cached: u64) Price {
    const todo = (n -| 1) -| cached;
    if (todo == 0) return .{};
    const w = @as(f64, @floatFromInt(@min(k.prefill_rows, todo))) / 2048.0;
    var once = (k.pf_chunk_gib + k.pf_host_gib) * w + k.pf_other_gib;
    once += 1.5 * @as(f64, @floatFromInt(k.union_cap)) / GiB;
    const end = @min(n - 1, k.index_positions);
    const share = @as(f64, @floatFromInt(end -| cached)) / @as(f64, @floatFromInt(k.index_positions));
    const index = k.pf_index_gib * @as(f64, @floatFromInt(k.index_budget_mib)) / 256.0 * share;
    return .{ .once = @intFromFloat(once * GiB), .index = @intFromFloat(index * GiB), .start = cached, .end = @max(cached, end) };
}

/// An NVMe resume's host transient: the staging set (streamed), or nothing when the bytes are already in host memory.
pub fn restorePrice(k: Knobs, from_disk: bool) u64 {
    return if (from_disk) k.staging_bytes else 0;
}

pub const Verdict = enum { admit, wait_pages, wait_memory };

pub const Ask = struct {
    /// pages the request needs reserved (prompt + reply + slack) less those a RAM resume shares
    pages: u32,
    /// the prompt's price, an NVMe resume's price, and what admitted prompts have not materialised yet
    price: u64,
    restore: u64 = 0,
    outstanding: u64 = 0,
};

/// Whether a request may start: pages any residue can give (Pool.available), and on every rank usable less the transient stays at the hard floor.
/// usable: each rank's usable bytes (rank 0's own view, the peers' last reports); returns the binding rank too.
pub fn admit(k: Knobs, pool: *const sessions.Pool, ask: Ask, usable: []const u64) struct { verdict: Verdict, rank: u32 } {
    if (pool.available() < ask.pages) return .{ .verdict = .wait_pages, .rank = 0 };
    const want = ask.price + ask.restore + ask.outstanding + @as(u64, @intFromFloat(k.hard_gib * GiB));
    var tight: u32 = 0;
    for (usable, 0..) |u, r| if (u < usable[tight]) {
        tight = @intCast(r);
    };
    if (usable[tight] < want) return .{ .verdict = .wait_memory, .rank = tight };
    return .{ .verdict = .admit, .rank = tight };
}

const layout_mod = @import("layout.zig");

test "the boot budget: Mia's pool cap of LONG-CONTEXT.md 2, v2's, and split KV's pool from the same memory" {
    var l = try layout_mod.Layout.fromConfig(layout_mod.Release{}, .{});
    const k: Knobs = .{ .staging_bytes = 0 };
    // memory.py load_check with mia29 refuses pools above 3.23 GiB = 1.94M tokens (111.75 - 99.48 - 4 slots - 4.0 - 1.0 - 4)
    const mia: Boot = .{ .weights = 99.48 };
    const t = mia.poolTokens(k, l.pool(), 1);
    try std.testing.expect(t > 1_930_000 and t < 1_945_000);
    try std.testing.expect(mia.floor(k, t * 1790) >= 4.0);
    // the same memory with split KV over two ranks holds 1,790 / 1,060 times the tokens (less the union buffers it prices)
    var s = try layout_mod.Layout.fromConfig(layout_mod.Release{}, .{ .split = true });
    const ks: Knobs = .{ .staging_bytes = 0, .union_cap = 128 << 20 };
    const ts = mia.poolTokens(ks, s.pool(), 2);
    const ratio = @as(f64, @floatFromInt(ts)) / @as(f64, @floatFromInt(t));
    const spare = mia.available - mia.fixed(k) - k.hard_gib;
    const want = (spare - 0.1875) / spare * 1790.0 / 1060.0; // 1.59: the union's 192 MiB comes off first
    try std.testing.expectApproxEqAbs(want, ratio, 1e-3);
    try std.testing.expectEqual(@as(u64, 0), ts % (2 * 256));
}

test "prefill and restore prices, admission against both ranks and residues" {
    const k: Knobs = .{};
    const p = prefillPrice(k, 1_000_000, 0); // a 1M prompt is priced as a 32K one: 1.133 + 1.067 GiB at 2,048 rows
    const p32 = prefillPrice(k, 32_769, 0);
    try std.testing.expectEqual(p.total(), p32.total());
    try std.testing.expectApproxEqAbs(@as(f64, 1.133 + 1.067), @as(f64, @floatFromInt(p.total())) / GiB, 1e-6);
    try std.testing.expectEqual(@as(u64, 0), prefillPrice(k, 100, 99).total());
    try std.testing.expectEqual(p.index / 2, p.outstanding(16384) - 0);
    const ks: Knobs = .{ .union_cap = 128 << 20 };
    try std.testing.expectEqual(@as(u64, 192 << 20), prefillPrice(ks, 5000, 0).once - prefillPrice(k, 5000, 0).once);
    const fams = [_]sessions.Family{.{ .name = "comp", .ratio = 1, .row_bytes = 584, .split = true }};
    var pool = try sessions.Pool.init(std.testing.allocator, .{ .families = &fams }, 16 * 256, 2, 0);
    defer pool.deinit();
    const g: u64 = 1 << 30;
    const ask: Ask = .{ .pages = 10, .price = p.total(), .restore = restorePrice(k, true) };
    try std.testing.expectEqual(Verdict.admit, admit(k, &pool, ask, &.{ 9 * g, 8 * g }).verdict);
    const r = admit(k, &pool, ask, &.{ 9 * g, 6 * g }); // rank 1 binds: 6 - 2.2 - 0.125 < 4
    try std.testing.expect(r.verdict == .wait_memory and r.rank == 1);
    try std.testing.expectEqual(Verdict.wait_pages, admit(k, &pool, .{ .pages = 17, .price = 0 }, &.{ 9 * g, 9 * g }).verdict);
}
