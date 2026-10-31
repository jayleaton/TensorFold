//! TF_DSV41_KV_MIRROR on the emitters: off unchanged; inside the mirror replicated reads, no exchange; stores always.

const std = @import("std");
const testing = std.testing;
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const calls = @import("block_calls.zig");
const check = @import("calls_check.zig");
const rowmode = @import("rowmode.zig");
const aot_needs = @import("aot_needs.zig");
const Config = @import("config.zig").Config;
const kvs = @import("cuda_kv.zig");
const env = @import("sys").env;

const attn = "tf_dsv41_attn_cuda_v1.attn";
const limit: i64 = 8192;
const pages: i64 = @divExact(limit, block.Pool.page);
const mirror: i64 = 2048;
const slots: i64 = 3;

fn splitPool(m: i64) block.Pool {
    const mp: i64 = if (m > 0) slots * @divExact(m, block.Pool.page) + 1 else 0;
    return .{ .comp_pages = @divExact(pages, 2) + 2, .ik_pages = pages + 1, .pts = pages, .split = true, .mirror = m, .mirror_pages = mp };
}

/// KV_SPLIT=0's pool at the same tensor pages (the replicated path the mirror's reads take)
fn pagedPool() block.Pool {
    return .{ .comp_pages = @divExact(pages, 2) + 2, .ik_pages = pages + 1, .pts = pages };
}

fn widths(a: std.mem.Allocator, cfg: *const Config) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    for (0..cfg.layers) |L| try w.gm.put(a, @intCast(L), .{ (1 << 4) | (1 << 6) | (1 << 8), (1 << 4) | (1 << 10) });
    try w.dense.put(a, "dspark.main_proj", try w.k2("L40.attn.wkv"));
    return w;
}

fn roleOf(x: calls.Arg) []const u8 {
    return switch (x) {
        .t, .opaque_table => |t| switch (t.role) {
            .buf, .weight => |r| r,
            .empty => "",
        },
        else => "",
    };
}

fn argKey(a: std.mem.Allocator, out: *std.ArrayList(u8), x: calls.Arg) !void {
    switch (x) {
        .t, .opaque_table => |t| {
            try out.print(a, " {s}:{t}@{d}[", .{ roleOf(x), t.dt, t.offset });
            for (t.shape, t.stride) |n, s| try out.print(a, "{d}/{d},", .{ n, s });
            try out.append(a, ']');
        },
        .list => |l| {
            try out.appendSlice(a, " (");
            for (l) |y| try argKey(a, out, y);
            try out.append(a, ')');
        },
        .i => |v| try out.print(a, " {d}", .{v}),
        .f => |v| try out.print(a, " {e}", .{v}),
        .b => |v| try out.print(a, " {any}", .{v}),
        .none => try out.appendSlice(a, " none"),
    }
}

/// A call as compared here: name, grid, scope, streams, marks and every argument but those at `skip`.
fn key(a: std.mem.Allocator, c: calls.Call, skip: []const usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.print(a, "{s} {any} {t} side={any} fork={any} join={any} mark={d} wait={d} dside={any} djoin={any}", .{ c.name, c.grid, c.begin, c.side, c.fork, c.join, c.mark, c.wait, c.defer_side, c.defer_join });
    for (c.args, 0..) |x, i| {
        if (std.mem.indexOfScalar(usize, skip, i) != null) continue;
        try out.print(a, " {s}=", .{x.name});
        try argKey(a, &out, x.arg);
    }
    return out.items;
}

fn at(c: calls.Call, name: []const u8) !usize {
    for (c.args, 0..) |x, i| if (std.mem.eql(u8, x.name, name)) return i;
    return error.NoArgument;
}

fn compStore(c: calls.Call) !bool {
    if (!std.mem.eql(u8, c.name, "_kv_store")) return false;
    return c.args[try at(c, "RATIO")].arg.i > 0;
}

fn isKx(c: calls.Call) bool {
    return c.glue and std.mem.startsWith(u8, c.name, "glue.kx_");
}

fn compAttn(c: calls.Call) bool {
    return std.mem.eql(u8, c.name, attn) and roleOf(c.args[1].arg).len > 0;
}

/// The mirror's store right after the pool's: the same launch into "s.kv.mc.L<src>" through "s.kv.mt".
fn checkStore(a: std.mem.Allocator, ps: calls.Call, ms: calls.Call, p: block.Pool) !void {
    const v = try at(ms, "V");
    const s = try at(ms, "S");
    const pt = try at(ms, "PT");
    try testing.expectEqualStrings(try key(a, ps, &.{ v, s, pt }), try key(a, ms, &.{ v, s, pt }));
    const comp = roleOf(ps.args[v].arg);
    try testing.expect(std.mem.startsWith(u8, comp, "s.kv.comp.L"));
    const mc = roleOf(ms.args[v].arg);
    try testing.expect(std.mem.startsWith(u8, mc, "s.kv.mc.L"));
    try testing.expectEqualStrings(comp["s.kv.comp.L".len..], mc["s.kv.mc.L".len..]);
    try testing.expectEqualStrings(mc, roleOf(ms.args[s].arg));
    const ratio = ms.args[try at(ms, "RATIO")].arg.i;
    try testing.expectEqual(p.mirror_pages * @divExact(block.Pool.page, ratio), ms.args[v].arg.t.shape[0]);
    try testing.expectEqual(@as(i64, 0), ms.args[v].arg.t.offset);
    try testing.expectEqual(@as(i64, 576), ms.args[s].arg.t.offset);
    try testing.expectEqualStrings("s.kv.ct", roleOf(ps.args[pt].arg));
    try testing.expectEqualStrings("s.kv.mt", roleOf(ms.args[pt].arg));
    try testing.expectEqualSlices(i64, &.{p.pts}, ms.args[pt].arg.t.shape);
}

/// `cs` without the mirror's stores, each checked against the pool store before it; their count in `stores`.
fn withoutStores(a: std.mem.Allocator, cs: []const calls.Call, p: block.Pool, stores: *usize) ![]const calls.Call {
    var out: std.ArrayList(calls.Call) = .empty;
    var i: usize = 0;
    while (i < cs.len) : (i += 1) {
        try out.append(a, cs[i]);
        if (!try compStore(cs[i])) continue;
        try testing.expect(i + 1 < cs.len);
        try checkStore(a, cs[i], cs[i + 1], p);
        stores.* += 1;
        i += 1;
    }
    return out.items;
}

/// A compressed layer's attention inside the mirror: KV_SPLIT=0's read (the selection, paged) of the mirror's rows.
fn checkRead(a: std.mem.Allocator, off: calls.Call, on: calls.Call, paged: calls.Call, p: block.Pool) !void {
    try testing.expectEqualStrings("w.kx.recv", roleOf(off.args[1].arg));
    try testing.expectEqualStrings(try key(a, off, &.{ 1, 2, 3, 11, 20 }), try key(a, on, &.{ 1, 2, 3, 11, 20 }));
    try testing.expectEqualStrings(try key(a, paged, &.{ 1, 2, 11 }), try key(a, on, &.{ 1, 2, 11 }));
    const comp = roleOf(paged.args[1].arg);
    const mc = roleOf(on.args[1].arg);
    try testing.expect(std.mem.startsWith(u8, comp, "s.kv.comp.L") and std.mem.startsWith(u8, mc, "s.kv.mc.L"));
    try testing.expectEqualStrings(comp["s.kv.comp.L".len..], mc["s.kv.mc.L".len..]);
    try testing.expectEqualStrings(mc, roleOf(on.args[2].arg));
    try testing.expectEqual(@as(i64, 0), on.args[1].arg.t.offset);
    try testing.expectEqual(@as(i64, 576), on.args[2].arg.t.offset);
    const rows = on.args[1].arg.t.shape[0];
    try testing.expect(rows == p.mirror_pages * 128 or rows == p.mirror_pages * 256);
    try testing.expectEqualStrings("s.kv.pt", roleOf(paged.args[11].arg));
    try testing.expectEqualStrings("s.kv.mt", roleOf(on.args[11].arg));
    try testing.expect(on.args[20].arg.i > 0);
}

test "TF_DSV41_KV_MIRROR: inside the mirror no exchange and KV_SPLIT=0's reads; past it today's calls plus the stores" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(a, &cfg);
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const served: block.Options = .{ .limit = limit, .r1 = true, .index_bound = true, .expert_topp = 0.85, .branches = true, .mhc_defer = true, .kv_norm_store = true };
    var rows = served;
    rows.rows = true;
    var reads: usize = 0;
    for ([_]block.Options{ .{ .limit = limit }, served, rows }) |base| for ([_]i64{ 1, 6, 16, 24, 64 }) |n| for ([_]i64{ 300, 1025, 2048, 2049, 4096 }) |end| {
        var o = base;
        o.pool = splitPool(0);
        var on = base;
        on.pool = splitPool(mirror);
        var pg = base;
        pg.pool = pagedPool();
        const off_cs = try block.emit(a, &cfg, &w, o, &backbone, n, end - n, true);
        const on_cs = try block.emit(a, &cfg, &w, on, &backbone, n, end - n, true);
        const paged = try block.emit(a, &cfg, &w, pg, &backbone, n, end - n, true);
        // the knob without split KV changes nothing (replicated KV has no exchange to skip)
        var pgm = pg;
        pgm.pool.?.mirror = mirror;
        pgm.pool.?.mirror_pages = splitPool(mirror).mirror_pages;
        const paged_m = try block.emit(a, &cfg, &w, pgm, &backbone, n, end - n, true);
        try testing.expectEqual(paged.len, paged_m.len);
        for (paged, paged_m) |x, y| try testing.expectEqualStrings(try key(a, x, &.{}), try key(a, y, &.{}));
        // off: no mirror role anywhere, an exchange a compressed index layer
        var kx: usize = 0;
        for (off_cs) |c| {
            kx += @intFromBool(isKx(c));
            for (c.args) |x| try testing.expect(!std.mem.startsWith(u8, roleOf(x.arg), "s.kv.m"));
        }
        try testing.expect(kx > 0);
        var stores: usize = 0;
        const inner = try withoutStores(a, on_cs, on.pool.?, &stores);
        try testing.expect(stores > 0);
        if (end > mirror) {
            try testing.expectEqual(off_cs.len, inner.len);
            for (off_cs, inner) |x, y| try testing.expectEqualStrings(try key(a, x, &.{}), try key(a, y, &.{}));
            continue;
        }
        reads += 1;
        // off less its exchanges (each one's join on the call after it) equals on up to the compressed reads
        var cut: std.ArrayList(calls.Call) = .empty;
        var carry: ?calls.Call = null;
        for (off_cs) |c| {
            if (isKx(c)) {
                carry = c;
                continue;
            }
            var x = c;
            if (carry) |k| {
                x.join = x.join or k.join;
                x.fork = x.fork or k.fork;
                carry = null;
            }
            try cut.append(a, x);
        }
        try testing.expectEqual(off_cs.len - kx, inner.len);
        try testing.expectEqual(paged.len, inner.len);
        var attns: usize = 0;
        for (cut.items, inner, paged) |x, y, z| {
            try testing.expect(!isKx(y));
            if (compAttn(y)) {
                try checkRead(a, x, y, z, on.pool.?);
                attns += 1;
                continue;
            }
            try testing.expectEqualStrings(try key(a, x, &.{}), try key(a, y, &.{}));
            // KV_SPLIT=0's calls but for the pool store's split table
            const pt = [1]usize{if (try compStore(y)) try at(y, "PT") else y.args.len};
            try testing.expectEqualStrings(try key(a, z, &pt), try key(a, y, &pt));
        }
        try testing.expect(attns >= kx);
    };
    try testing.expectEqual(@as(usize, 3 * 5 * 3), reads);
}

test "TF_DSV41_KV_MIRROR: row mode stacks the mirror table as the page tables, in the stores and the reads" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(a, &cfg);
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    var o: block.Options = .{ .limit = limit, .r1 = true, .branches = true, .mhc_defer = true, .rows = true };
    o.pool = splitPool(mirror);
    const cs = try rowmode.transform(a, try block.emit(a, &cfg, &w, o, &backbone, 6, 1994, true), 6, .{ .slots = slots, .rmax = 64, .pts = pages });
    var stores: usize = 0;
    var reads: usize = 0;
    for (cs) |c| {
        for (c.args) |x| if (std.mem.eql(u8, roleOf(x.arg), "s.kv.mt")) try testing.expectEqualSlices(i64, &.{ slots, pages }, x.arg.t.shape);
        if (try compStore(c) and std.mem.startsWith(u8, roleOf(c.args[try at(c, "V")].arg), "s.kv.mc.L")) {
            try testing.expectEqual(pages, c.args[try at(c, "PTS")].arg.i);
            try testing.expect(c.args[try at(c, "ROWS")].arg.b);
            stores += 1;
        }
        if (compAttn(c)) {
            try testing.expectEqualStrings("s.kv.mt", roleOf(c.args[11].arg));
            try testing.expectEqual(pages, c.args[21].arg.i);
            reads += 1;
        }
    }
    try testing.expect(stores > 0 and reads > 0);
}

test "TF_DSV41_KV_MIRROR: a prefill segment also stores into the mirror and otherwise emits today's calls" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(a, &cfg);
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    for ([_][2]i64{ .{ 0, 300 }, .{ 1700, 600 } }) |sn| {
        var o: block.Options = .{ .limit = limit };
        o.pool = splitPool(0);
        var on = o;
        on.pool = splitPool(mirror);
        const off_cs = try block_prefill.emitPrefill(a, &cfg, &w, o, &backbone, sn[1], sn[0], true);
        const on_cs = try block_prefill.emitPrefill(a, &cfg, &w, on, &backbone, sn[1], sn[0], true);
        var stores: usize = 0;
        const inner = try withoutStores(a, on_cs, on.pool.?, &stores);
        try testing.expect(stores > 0);
        try testing.expectEqual(off_cs.len, inner.len);
        for (off_cs, inner) |x, y| try testing.expectEqualStrings(try key(a, x, &.{}), try key(a, y, &.{}));
    }
}

test "TF_DSV41_KV_MIRROR: the mirror's store reduces to its pool store's Triton variant (no AOT asset added)" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(struct { kernels: aot_needs.Sigs }, gpa, @embedFile("fixtures/aot-sigs.json"), .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const n: aot_needs.Needs = .{ .gpa = gpa, .sigs = &parsed.value.kernels };
    const cfg: Config = .{};
    const w = try widths(a, &cfg);
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    var o: block.Options = .{ .limit = limit };
    o.pool = splitPool(mirror);
    const decode = try block.emit(a, &cfg, &w, o, &backbone, 6, 1000, true);
    const prefill = try block_prefill.emitPrefill(a, &cfg, &w, o, &backbone, 300, 0, true);
    var pairs: usize = 0;
    for ([_][]const calls.Call{ decode, prefill }) |cs| for (cs[0 .. cs.len - 1], cs[1..]) |*x, *y| {
        if (!try compStore(x.*) or !try compStore(y.*)) continue;
        if (!std.mem.startsWith(u8, roleOf(y.args[try at(y.*, "V")].arg), "s.kv.mc.L")) continue;
        try testing.expectEqualStrings(try n.reduce(a, x), try n.reduce(a, y));
        pairs += 1;
    };
    try testing.expect(pairs > 0);
}

test "TF_DSV41_KV_MIRROR: the pool takes it with split KV only, at most the slot, in whole pages of every residue" {
    const names = [_][:0]const u8{ "TF_DSV41_POOL_TOKENS", "TF_DSV41_KV_SPLIT", "TF_DSV41_KV_MIRROR" };
    defer for (names) |k| {
        _ = env.unsetenv(k);
    };
    _ = env.setenv("TF_DSV41_POOL_TOKENS", "3000000", 1);
    _ = env.setenv("TF_DSV41_KV_SPLIT", "1", 1);
    _ = env.unsetenv("TF_DSV41_KV_MIRROR");
    try testing.expectEqual(@as(u64, 0), (try kvs.optionsFromEnv(1 << 20, 2)).?.mirror);
    _ = env.setenv("TF_DSV41_KV_MIRROR", "0", 1);
    try testing.expectEqual(@as(u64, 0), (try kvs.optionsFromEnv(1 << 20, 2)).?.mirror);
    _ = env.setenv("TF_DSV41_KV_MIRROR", "2000", 1);
    try testing.expectEqual(@as(u64, 2048), (try kvs.optionsFromEnv(1 << 20, 2)).?.mirror);
    try testing.expectEqual(@as(u64, 1024), (try kvs.optionsFromEnv(1000, 2)).?.mirror);
    _ = env.setenv("TF_DSV41_KV_MIRROR", "lots", 1);
    try testing.expectError(error.BadMirror, kvs.optionsFromEnv(1 << 20, 2));
    _ = env.setenv("TF_DSV41_KV_MIRROR", "2048", 1);
    _ = env.setenv("TF_DSV41_KV_SPLIT", "0", 1);
    try testing.expectError(error.MirrorNeedsSplit, kvs.optionsFromEnv(1 << 20, 2));
}
