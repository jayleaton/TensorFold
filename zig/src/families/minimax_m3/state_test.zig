//! The stream state and prompt cache on pages: admission, eviction, snapshot save and restore, refusals.
const std = @import("std");
const kv = @import("core").kv_pool;
const pool = @import("pool.zig");
const state = @import("state.zig");
const pool_test = @import("pool_test.zig");

const gpa = std.testing.allocator;
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;

const small: @import("engine_api").prompt_radix.Rules = .{ .page = 128, .min_gap = 128, .min_prompt = 0, .tail = true };

/// A pool of `n` pages, its block tables and the cache on both.
const Rig = struct {
    pages: kv.Pool,
    tables: pool.Tables,
    cache: state.Cache,
    max_pages: usize,

    fn start(r: *Rig, n: usize, max_pages: usize) !void {
        const d = try pool_test.dims(0);
        r.max_pages = max_pages;
        r.pages = try kv.Pool.init(gpa, n, d.pageBytes());
        r.tables = try pool.Tables.init(gpa, 4, max_pages);
        try r.cache.init(gpa, &r.pages, d.layout(max_pages), small);
    }

    fn stop(r: *Rig) void {
        r.cache.deinit();
        r.tables.deinit();
        r.pages.deinit();
    }

    fn admit(r: *Rig, s: *state.Stream, prompt: []const u32, history: u32, max_new: u64) !void {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var sink = r.tables.sink(s.slot);
        const marks = try r.cache.admit(arena.allocator(), s, prompt, history, max_new, 16, sink.table(), r.max_pages);
        for (marks) |m| try r.cache.keep(s, prompt, m, sink.table()); // the prefill reaching each mark
    }

    fn end(r: *Rig, s: *state.Stream) void {
        r.cache.end(s, &r.tables);
    }
};

fn tokens(comptime n: usize, seed: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*t, i| t.* = seed * 100_000 + @as(u32, @intCast(i));
    return out;
}

test "pages for prompt, reply and the verify window" {
    var r: Rig = undefined;
    try r.start(8, 8);
    defer r.stop();
    try expectEqual(@as(usize, 1), r.cache.need(100, 12, 16)); // 128 rows exactly
    try expectEqual(@as(usize, 2), r.cache.need(100, 13, 16));
    try expectEqual(@as(usize, 3), r.cache.need(300, 50, 16)); // 366 rows
    try expectEqual(@as(usize, 1024), r.cache.need(131_000, 0, 72)); // 131,072 rows
    var s: state.Stream = .{};
    const p = tokens(300, 1);
    try r.admit(&s, &p, 0, 50);
    try expectEqual(@as(usize, 3), s.lease.ids.items.len);
    for (s.lease.ids.items, 0..) |id, b| try expectEqual(@as(?u32, id), r.tables.page(0, b)); // the block table is the lease
    try expectEqual(@as(?u32, null), r.tables.page(0, 3));
    try expectEqual(@as(usize, 5), r.pages.available()); // three taken, none promised
    r.end(&s);
    try expect(r.pages.idle());
}

test "a kept prompt resumes from page references: the pages are the first stream's, nothing is copied" {
    var r: Rig = undefined;
    try r.start(16, 16);
    defer r.stop();
    r.cache.limit(8, 4);
    var a: state.Stream = .{ .slot = 0 };
    const p1 = tokens(600, 1);
    try r.admit(&a, &p1, 0, 40);
    const first_pages = try gpa.dupe(u32, a.lease.ids.items);
    defer gpa.free(first_pages);
    try expect(r.cache.live > 0); // a snapshot record was kept
    r.end(&a);
    // the store alone holds the kept prefix now; the rest is free
    const c = r.pages.census();
    try expect(c.kept > 0 and c.leased == 0 and c.promised == 0);
    try expectEqual(c.total, c.free + c.kept);
    // a later turn of the same conversation: the first 512 rows are the kept pages
    var p2: [700]u32 = undefined;
    @memcpy(p2[0..600], &p1);
    for (p2[600..], 0..) |*t, i| t.* = 777_000 + @as(u32, @intCast(i));
    var b: state.Stream = .{ .slot = 1 };
    try r.admit(&b, &p2, 600, 40);
    try expect(b.pos > 0 and b.pos % 128 == 0 and b.pos <= 600);
    const shared = b.pos / 128;
    try expectEqualSlices(first_pages[0..shared], b.lease.ids.items[0..shared]);
    try expectEqual(@as(usize, 1), r.pages.leased[b.lease.ids.items[0]]);
    try expect(r.pages.holders(b.lease.ids.items[0]) == 2); // the stream and the store
    try expect(r.cache.store.tree.counts.hits == 1);
    // the stream's block table starts with the shared pages
    for (b.lease.ids.items, 0..) |id, k| try expectEqual(@as(?u32, id), r.tables.page(1, k));
    r.end(&b);
    r.cache.clear();
    try expect(r.pages.idle());
    try expectEqual(@as(usize, r.pages.total()), r.pages.nfree);
    try expectEqual(@as(usize, 0), r.cache.live);
}

fn expectEqualSlices(a: []const u32, b: []const u32) !void {
    try std.testing.expectEqualSlices(u32, a, b);
}

test "a restore that fails prefills from the start and drops the snapshot" {
    var r: Rig = undefined;
    try r.start(16, 16);
    defer r.stop();
    r.cache.limit(8, 4);
    var a: state.Stream = .{};
    const p1 = tokens(600, 2);
    try r.admit(&a, &p1, 0, 8);
    r.end(&a);
    const live = r.cache.live;
    try expect(live > 0);
    r.cache.fail_restore = true;
    var b: state.Stream = .{};
    try r.admit(&b, &p1, 0, 8);
    try expectEqual(@as(u32, 0), b.pos);
    try expect(r.cache.store.tree.counts.failed >= 1);
    try expect(r.cache.live < live + 1);
    r.end(&b);
    r.cache.clear();
    try expect(r.pages.idle());
    try expectEqual(@as(usize, 0), r.cache.live);
}

test "eviction: a stream that needs the pool takes the kept pages back" {
    var r: Rig = undefined;
    try r.start(8, 8);
    defer r.stop();
    r.cache.limit(8, 4);
    var a: state.Stream = .{};
    const p1 = tokens(600, 3);
    try r.admit(&a, &p1, 0, 8); // 5 pages
    r.end(&a);
    const kept = r.pages.census().kept;
    try expect(kept >= 3);
    try expect(r.pages.available() < 8);
    // an unrelated prompt of 7 pages: only possible if the store gives pages back
    var b: state.Stream = .{};
    const p2 = tokens(850, 4);
    try r.admit(&b, &p2, 0, 0);
    try expectEqual(@as(usize, 7), b.lease.ids.items.len);
    try expect(r.cache.store.tree.counts.evicted > 0);
    r.end(&b);
    r.cache.clear();
    try expect(r.pages.idle());
    try expectEqual(@as(usize, 8), r.pages.nfree);
}

test "refusals: past the table row or the pool is final, a full pool is LaneMemory and leaves nothing leased" {
    var r: Rig = undefined;
    try r.start(6, 4);
    defer r.stop();
    var s: state.Stream = .{};
    const long = tokens(700, 5); // 6 pages: past the 4-entry table
    try std.testing.expectError(error.ContextTooLong, r.admit(&s, &long, 0, 0));
    try expect(r.pages.idle());
    var big: state.Stream = .{};
    const four = tokens(480, 6); // 4 pages
    try r.admit(&big, &four, 0, 0);
    var t: state.Stream = .{ .slot = 1 };
    const three = tokens(300, 7); // 3 pages, 2 free
    try std.testing.expectError(error.LaneMemory, r.admit(&t, &three, 0, 0));
    try expectEqual(@as(usize, 2), r.pages.available());
    try expectEqual(@as(usize, 0), r.pages.promised);
    try expectEqual(@as(usize, 0), t.lease.ids.items.len);
    // the first stream ends and the second runs
    r.end(&big);
    try r.admit(&t, &three, 0, 0);
    r.end(&t);
    try expect(r.pages.idle());
}

test "a failed table upload gives the pages back" {
    var r: Rig = undefined;
    try r.start(8, 2); // a 2-entry row
    defer r.stop();
    var s: state.Stream = .{};
    const p = tokens(200, 8); // 2 pages fit the row
    try r.admit(&s, &p, 0, 0);
    r.end(&s);
    r.tables.upload = .{ .ctx = &r, .call = struct {
        fn call(_: *anyopaque, _: u32, _: usize, _: []const i32) anyerror!void {
            return error.UploadFailed;
        }
    }.call };
    const q = tokens(200, 9);
    try std.testing.expectError(error.UploadFailed, r.admit(&s, &q, 0, 0));
    try expect(r.pages.idle());
    try expectEqual(@as(?u32, null), r.tables.page(0, 0));
}

test "random admits, ends and limits leave every page free or the store's" {
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for (0..40) |seed| {
        var r: Rig = undefined;
        try r.start(12 + seed % 5, 12);
        defer r.stop();
        r.cache.limit(6, 4);
        var live: [4]?state.Stream = @splat(null);
        var convs: [3][1000]u32 = undefined;
        for (&convs, 0..) |*c, k| for (c, 0..) |*t, i| {
            t.* = if (i < 256) @intCast(i) else @intCast(100_000 * (k + 1) + i);
        };
        var lens: [3]u32 = @splat(0);
        for (0..60) |_| {
            const k = rnd.uintLessThan(usize, 3);
            const slot = rnd.uintLessThan(usize, 4);
            if (live[slot]) |*s| {
                r.end(s);
                live[slot] = null;
            } else if (rnd.uintLessThan(u8, 4) != 0) {
                const hist = lens[k];
                lens[k] = @min(900, hist + 130 + rnd.uintLessThan(u32, 200));
                var s: state.Stream = .{ .slot = @intCast(slot) };
                r.admit(&s, convs[k][0..lens[k]], hist, rnd.uintLessThan(u64, 60)) catch |e| switch (e) {
                    error.LaneMemory, error.ContextTooLong => continue,
                    else => return e,
                };
                live[slot] = s;
            }
            // every page: holders are the live leases and the store; free pages are the rest
            var refs: [20]u32 = @splat(0);
            var promised: usize = 0;
            for (live) |m| if (m) |s| {
                promised += s.lease.promised;
                for (s.lease.ids.items) |id| refs[id] += 1;
            };
            for (r.pages.leased, 0..) |l, id| try expect(l == refs[id]);
            try expectEqual(promised, r.pages.promised);
        }
        for (&live) |*m| if (m.*) |*s| r.end(s);
        r.cache.clear();
        try expect(r.pages.idle());
        try expectEqual(r.pages.total(), r.pages.nfree);
        try expectEqual(@as(usize, 0), r.cache.live);
    }
}
