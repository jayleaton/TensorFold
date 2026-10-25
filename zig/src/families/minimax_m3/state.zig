//! MiniMax-M3.1-Flash stream state and prompt cache on the page pool; a snapshot is page references, never a copy.
const std = @import("std");
const api = @import("engine_api");
const kv = @import("core").kv_pool;
const pool = @import("pool.zig");
const radix = api.prompt_radix;

pub const Error = error{ ContextTooLong, LaneMemory };

/// Prompt-cache rules: marks and resumes on 128-row pages; kernels are row-invariant, so the store is not `planned`.
pub const rules: radix.Rules = .{ .page = pool.page_tokens, .min_gap = 1024, .min_prompt = 2048, .tail = true };

/// A live stream: the lease (its page ids in token order, the block table's entries), its slot and rows filled.
pub const Stream = struct {
    lease: kv.Lease = .{},
    slot: u32 = 0,
    /// Rows whose K, V and IK are in the pages: the next row written is `pos`.
    pos: u32 = 0,
};

/// What a kept snapshot is: the position, and with a dspark draft the id of its saved K/V ring.
const Saved = struct { at: u32, ring: ?u32 = null };

/// The dspark draft's ring snapshots: `save` may refuse (null: that snapshot keeps no ring).
pub const RingHook = struct {
    ctx: *anyopaque,
    save: *const fn (ctx: *anyopaque, slot: u32, at: u32) ?u32,
    restore: *const fn (ctx: *anyopaque, slot: u32, id: u32) void,
    drop: *const fn (ctx: *anyopaque, id: u32) void,
};

/// The prompt cache over a pool: a radix store of page references and the snapshots' records.
pub const Cache = struct {
    gpa: std.mem.Allocator,
    pages: *kv.Pool,
    layout: kv.Layout,
    store: radix.Store,
    /// Snapshot records alive; zero at rest after `clear`.
    live: usize = 0,
    /// Fails the next restore or save (tests of the store's fallbacks).
    fail_restore: bool = false,
    /// The draft's ring snapshots (null: no draft).
    ring: ?RingHook = null,

    /// `self` must stay where it is: the pool's reclaim and the store's family point into it.
    pub fn init(c: *Cache, gpa: std.mem.Allocator, pages: *kv.Pool, layout: kv.Layout, r: radix.Rules) !void {
        c.* = .{ .gpa = gpa, .pages = pages, .layout = layout, .store = undefined };
        c.store = try radix.Store.init(gpa, c.snapshots(), radix.pagesOf(kv.Pool, pages, pages.page_bytes), r);
        pages.reclaim = .{ .ctx = &c.store, .call = reclaim };
    }

    pub fn deinit(c: *Cache) void {
        c.clear();
        c.store.deinit();
        c.pages.reclaim = null;
    }

    /// The store may hold `pages` pages and `snaps` snapshots; 0 and 0 clears it.
    pub fn limit(c: *Cache, pages: usize, snaps: usize) void {
        c.store.limit(.{ .pages = pages, .snaps = snaps });
    }

    /// A half of the pool's pages and the plan's snapshot slots: kept pages are evicted when a stream asks for them.
    pub fn limitTo(c: *Cache, snaps: usize) void {
        c.limit(c.pages.total() / 2, snaps);
    }

    pub fn clear(c: *Cache) void {
        c.limit(0, 0);
    }

    fn reclaim(ctx: *anyopaque, want: usize) bool {
        const s: *radix.Store = @ptrCast(@alignCast(ctx));
        return s.reclaim(want);
    }

    fn of(ptr: *anyopaque) *Cache {
        return @ptrCast(@alignCast(ptr));
    }

    fn snapshots(c: *Cache) radix.Snapshots {
        return .{ .ptr = c, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }

    fn bytes(_: *anyopaque, _: u32) u64 {
        return pool.snap_bytes;
    }

    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!radix.Saved {
        const c = of(ptr);
        const rec = try c.gpa.create(Saved);
        var ring: ?u32 = null;
        if (c.ring) |h| if (owner) |o| {
            ring = h.save(h.ctx, @as(*Stream, @ptrCast(@alignCast(o))).slot, at);
        };
        rec.* = .{ .at = at, .ring = ring };
        c.live += 1;
        return rec;
    }

    /// The stream continues at the snapshot's position; its pages come from the store (`begin` adopts them).
    fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: radix.Saved) anyerror!void {
        const c = of(ptr);
        if (c.fail_restore) return error.RestoreFailed;
        const rec: *Saved = @ptrCast(@alignCast(saved));
        if (owner) |o| {
            const st: *Stream = @ptrCast(@alignCast(o));
            st.pos = rec.at;
            if (c.ring) |h| if (rec.ring) |id| h.restore(h.ctx, st.slot, id);
        }
    }

    fn drop(ptr: *anyopaque, saved: radix.Saved) void {
        const c = of(ptr);
        const rec: *Saved = @ptrCast(@alignCast(saved));
        if (c.ring) |h| if (rec.ring) |id| h.drop(h.ctx, id);
        c.gpa.destroy(rec);
        c.live -= 1;
    }

    /// Pages a request reserves: prompt, reply and one last verify window.
    pub fn need(c: *const Cache, prompt: usize, max_tokens: u64, window: u64) usize {
        return c.layout.need(prompt, max_tokens, window);
    }

    /// Admits `s`: resumes the longest kept prefix and takes the rest's pages; returns the marks where to call `keep`.
    pub fn admit(c: *Cache, a: std.mem.Allocator, s: *Stream, prompt: []const u32, history: u32, max_new: u64, window: u64, table: ?kv.Table, max_pages: usize) ![]const u32 {
        const want = c.need(prompt.len, max_new, window);
        if (want > max_pages or want > c.pages.total()) return error.ContextTooLong;
        var adopt: std.ArrayList(u32) = .empty;
        defer adopt.deinit(c.gpa);
        const plan = try c.store.begin(a, prompt, history, &.{}, s, &adopt);
        errdefer a.free(plan.marks);
        const l = &s.lease;
        errdefer c.pages.finish(l);
        try c.pages.adopt(l, adopt.items);
        try c.pages.promise(l, want -| adopt.items.len);
        try c.pages.cover(l, want, table);
        s.pos = plan.from;
        return plan.marks;
    }

    /// The pass filled `at` rows (a mark): the store keeps a snapshot and the stream adopts the store's matching pages.
    pub fn keep(c: *Cache, s: *Stream, prompt: []const u32, at: u32, table: ?kv.Table) !void {
        const n = at / pool.page_tokens;
        if (n == 0 or n > s.lease.ids.items.len) return;
        const path = try c.gpa.alloc(u32, n);
        defer c.gpa.free(path);
        const kept = c.store.keep(prompt, at, s, s.lease.ids.items[0..n], path);
        for (path[0..kept.shared], 0..) |id, i| c.pages.share(&s.lease, i, id);
        if (kept.shared > 0) if (table) |t| try t.write(t.ctx, s.lease.ids.items[0..kept.shared], 0);
    }

    /// The stream ended: its pages and promise go back (kept pages stay) and its block-table row, if given, goes to -1.
    pub fn end(c: *Cache, s: *Stream, tables: ?*pool.Tables) void {
        c.pages.finish(&s.lease);
        if (tables) |t| t.clear(s.slot);
        s.pos = 0;
    }
};
