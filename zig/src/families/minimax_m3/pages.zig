//! The host half of the engine's page state: page ids, the radix prompt cache, the block tables (device half: T8).
const std = @import("std");
const kv = @import("core").kv_pool;
const pool = @import("pool.zig");
const state = @import("state.zig");

pub const Pages = struct {
    gpa: std.mem.Allocator,
    pages: kv.Pool,
    cache: state.Cache,
    tables: pool.Tables,
    layout: kv.Layout,
    max_pages: usize,
    window: u64,

    /// `streams` slots of at most `max_pages` pages over a pool of `count` pages; `window` rows past a reply.
    pub fn create(gpa: std.mem.Allocator, layout: kv.Layout, streams: u32, count: usize, max_pages: usize, window: u64) !*Pages {
        const p = try gpa.create(Pages);
        errdefer gpa.destroy(p);
        p.gpa = gpa;
        p.layout = layout;
        p.max_pages = max_pages;
        p.window = window;
        p.pages = try kv.Pool.init(gpa, count, layout.pageBytes());
        errdefer p.pages.deinit();
        p.tables = try pool.Tables.init(gpa, streams, max_pages);
        errdefer p.tables.deinit();
        try p.cache.init(gpa, &p.pages, layout, state.rules);
        return p;
    }

    pub fn destroy(p: *Pages) void {
        const gpa = p.gpa;
        p.cache.deinit();
        p.tables.deinit();
        p.pages.deinit();
        gpa.destroy(p);
    }

    /// Whether a request's prompt, reply and last window can be promised now (the store's idle pages counted).
    pub fn fits(p: *Pages, prompt: u32, max_tokens: u32) bool {
        const need = p.cache.need(prompt, max_tokens, p.window);
        return need <= p.max_pages and need <= p.pages.total() and p.pages.fits(need);
    }

    /// The prompt cache may hold `count` pages (half the pool at most) and `snaps` records; 0 and 0 switches it off.
    pub fn limit(p: *Pages, count: usize, snaps: usize) void {
        p.cache.limit(@min(count, p.pages.total() / 2), snaps);
    }
};

test "pages: the cache is limited to half the pool, a request that cannot fit is refused, and teardown is clean" {
    const gpa = std.testing.allocator;
    const l: kv.Layout = .{ .fixed_bytes = 4096, .token_bytes = 1024, .page_tokens = pool.page_tokens, .snap_bytes = pool.snap_bytes, .all_paged = true };
    const p = try Pages.create(gpa, l, 2, 64, 32, 16);
    defer p.destroy();
    try std.testing.expect(p.fits(1000, 100)); // 9 pages
    try std.testing.expect(!p.fits(4000, 200)); // 33 pages: past the table row
    try std.testing.expect(!p.fits(70 * 128, 0)); // past the pool
    p.limit(1000, 8);
    try std.testing.expect(p.pages.idle());
    var s: state.Stream = .{ .slot = 1 };
    var sink = p.tables.sink(1);
    const prompt: [3000]u32 = @splat(7);
    const marks = try p.cache.admit(gpa, &s, &prompt, 0, 100, p.window, sink.table(), p.max_pages);
    defer gpa.free(marks);
    try std.testing.expectEqual(@as(usize, 25), s.lease.ids.items.len);
    try std.testing.expectEqual(@as(?u32, s.lease.ids.items[3]), p.tables.page(1, 3));
    p.cache.end(&s, &p.tables);
    try std.testing.expect(p.pages.idle());
    try std.testing.expectEqual(@as(?u32, null), p.tables.page(1, 3));
}
