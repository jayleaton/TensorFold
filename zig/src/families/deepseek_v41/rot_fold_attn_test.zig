//! TF_DSV41_ROT_FOLD (3b): the decode windows with the merge's fold on and off, and its rows against rot_in's.

const std = @import("std");
const testing = std.testing;
const block = @import("block.zig");
const block_attn = @import("block_attn.zig");
const calls = @import("block_calls.zig");
const check = @import("calls_check.zig");
const p3a = @import("rot_fold_test.zig");
const ar = @import("dsv41_kernels").attn_rot;
const Config = @import("config.zig").Config;

const oneKey = p3a.oneKey;
const key = p3a.key;
const firstRole = p3a.firstRole;

fn isAttn(c: calls.Call) bool {
    return std.mem.eql(u8, c.name, "tf_dsv41_attn_cuda_v1.attn");
}

/// An x3seg rotation whose first input is a view of `role` (wo_a's: L.o; wo_b's: L.oa).
fn rotInOf(c: calls.Call, role: []const u8) bool {
    if (!std.mem.eql(u8, c.name, "tf_dsv41_x3seg_v3.rot_in")) return false;
    const r = firstRole(c.args[0].arg) orelse return false;
    return std.mem.eql(u8, r, role);
}

const Counts = struct { attn: usize = 0, wo_a: usize = 0, wo_b: usize = 0, linears: usize = 0 };

/// `on` (mask m) is `off` less the rot_ins m drops, R1's attention and wo_a's linears gaining arguments; nothing else
fn compareWindows(a: std.mem.Allocator, off: []const calls.Call, on: []const calls.Call, m: u3) !Counts {
    var k: Counts = .{};
    var j: usize = 0;
    var rows: ?[3][]const u8 = null; // the attention's appended (suh, xh, offsets), for the rot_in they replace
    for (off) |c| {
        if (rotInOf(c, "L.o") and m & 2 != 0) {
            // exactly the rotation H runs: the same suh list, the same L.o.xh and the same row offsets
            const want = rows orelse return error.NoFoldedAttention;
            try testing.expectEqualStrings(try oneKey(a, c.args[1].arg), want[0]);
            try testing.expectEqualStrings(try oneKey(a, c.args[2].arg), want[1]);
            try testing.expectEqualStrings(try oneKey(a, c.args[3].arg), want[2]);
            try testing.expectEqual(@as(usize, 5), c.args.len); // no folded norm
            k.wo_a += 1;
            rows = null;
            continue;
        }
        if (rotInOf(c, "L.oa") and m & 1 != 0) {
            k.wo_b += 1;
            continue;
        }
        try testing.expect(j < on.len);
        const d = on[j];
        j += 1;
        if (isAttn(c) and m & 2 != 0) {
            k.attn += 1;
            try testing.expectEqual(@as(usize, 25), c.args.len);
            try testing.expectEqual(@as(usize, 28), d.args.len);
            try testing.expectEqualStrings(try key(a, c, 25), try key(a, d, 25));
            rows = .{ try oneKey(a, d.args[25].arg), try oneKey(a, d.args[26].arg), try oneKey(a, d.args[27].arg) };
            continue;
        }
        if (p3a.writes(c, "L.oa") and m & 1 != 0) {
            k.linears += 1;
            try testing.expectEqual(@as(usize, 15), d.args.len);
            try testing.expectEqualStrings(try key(a, c, 12), try key(a, d, 12));
            continue;
        }
        try testing.expectEqualStrings(try key(a, c, c.args.len), try key(a, d, d.args.len));
    }
    try testing.expectEqual(on.len, j);
    try testing.expect(rows == null);
    return k;
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    return w;
}

test "TF_DSV41_ROT_FOLD: bit 1 drops each layer's wo_a rot_in and only adds its arguments to R1's attention" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 8192;
    const pages = @divExact(limit, block.Pool.page);
    var served: block.Options = .{ .limit = limit, .r1 = true, .index_bound = true, .expert_topp = 0.85, .branches = true, .mhc_defer = true, .kv_norm_store = true };
    served.pool = .{ .comp_pages = @divExact(pages, 2) + 2, .ik_pages = pages + 1, .pts = pages, .split = true };
    var rows = served;
    rows.rows = true;
    for ([_]block.Options{ served, rows }) |o| for ([_]i64{ 1, 2, 3, 5, 16, 17, 64 }) |n| {
        const off_cs = try block.emit(a, &cfg, &w, o, &backbone, n, 300, true);
        // flag off: H's calls (every layer's wo_a rot_in, every attention at R1's 25 arguments)
        var rot_ins: usize = 0;
        for (off_cs) |c| {
            rot_ins += @intFromBool(rotInOf(c, "L.o"));
            if (isAttn(c)) try testing.expectEqual(@as(usize, 25), c.args.len);
        }
        try testing.expectEqual(backbone.len, rot_ins);
        for ([_]u3{ 2, 3 }) |m| {
            var on = o;
            on.rot_fold = m;
            const on_cs = try block.emit(a, &cfg, &w, on, &backbone, n, 300, true);
            const k = try compareWindows(a, off_cs, on_cs, m);
            try testing.expectEqual(backbone.len, k.attn);
            try testing.expectEqual(backbone.len, k.wo_a);
            // bits 0 and 1 together: wo_b's rot_in goes too, and wo_a's linears (which read L.o.xh) write its rows
            try testing.expectEqual(if (m & 1 != 0) backbone.len else 0, k.wo_b);
            try testing.expectEqual(if (m & 1 != 0) backbone.len else 0, k.linears);
            try testing.expectEqual(off_cs.len - k.wo_a - k.wo_b, on_cs.len);
        }
    };
}

test "TF_DSV41_ROT_FOLD: without R1 bit 1 emits H's window exactly (attn_kernel has no fold)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    for ([_]i64{ 1, 5, 64 }) |n| for ([_]u3{ 0, 1 }) |base| {
        const off_cs = try block.emit(a, &cfg, &w, .{ .limit = 8192, .rot_fold = base }, &backbone, n, 300, true);
        const on_cs = try block.emit(a, &cfg, &w, .{ .limit = 8192, .rot_fold = base | 2 }, &backbone, n, 300, true);
        try testing.expectEqual(off_cs.len, on_cs.len);
        for (off_cs, on_cs) |c, d| try testing.expectEqualStrings(try key(a, c, c.args.len), try key(a, d, d.args.len));
    };
}

test "TF_DSV41_ROT_FOLD: attnFold takes R1 with bit 1 and whole 128-column groups of 512-wide heads only" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    var cfg: Config = .{};
    var e: block.Emitter = .{ .a = a, .cfg = &cfg, .w = &w, .o = .{ .rot_fold = 2, .r1 = true }, .n = 4, .start = 0 };
    try testing.expect(block_attn.attnFold(&e, 4, 4096)); // TP2's rank: 32 heads, 4 groups
    try testing.expect(block_attn.attnFold(&e, 8, 4096)); // TP1: 64 heads, 8 groups
    try testing.expect(!block_attn.attnFold(&e, 4, 4096 + 64)); // a group boundary inside a merge warp's block
    try testing.expect(!block_attn.attnFold(&e, 9, 4096)); // past the kernel's 8 groups
    for ([_]u3{ 0, 1, 4, 5 }) |m| {
        e.o.rot_fold = m;
        try testing.expect(!block_attn.attnFold(&e, 4, 4096));
    }
    e.o.rot_fold = 3;
    try testing.expect(block_attn.attnFold(&e, 4, 4096));
    e.o.r1 = false; // attn_kernel's merge: H's rot_in stays
    try testing.expect(!block_attn.attnFold(&e, 4, 4096));
    e.o.r1 = true;
    cfg.head_dim = 256;
    try testing.expect(!block_attn.attnFold(&e, 4, 2048));
}

// -- the merge's rotation against rot_in's, on the host -----------------------------------------------------------

const Rot = *const fn (x: *const [32][4]f32, s: *const [32][4]f32, v: *[32][4]f32) void;

/// rot_in as seg_rot_in_kernel's SASS runs it (attn_rot.cuh rot128): the scale fused into the first butterfly.
fn rot128(x: *const [32][4]f32, s: *const [32][4]f32, v: *[32][4]f32) void {
    for (x, s, v) |xl, sl, *l| {
        const p1 = xl[1] * sl[1];
        const p3 = xl[3] * sl[3];
        const a = @mulAdd(f32, xl[0], sl[0], p1);
        const b = @mulAdd(f32, xl[0], sl[0], -p1);
        const c = @mulAdd(f32, xl[2], sl[2], p3);
        const d = @mulAdd(f32, xl[2], sl[2], -p3);
        l.* = .{ a + c, b + d, a - c, b - d };
    }
    var m: usize = 1;
    while (m < 32) : (m <<= 1) {
        const old = v.*;
        for (0..32) |lane| for (0..4) |j| {
            const o = old[lane ^ m][j];
            v[lane][j] = if (lane & m != 0) o - old[lane][j] else old[lane][j] + o;
        };
    }
    for (v) |*l| for (l) |*y| {
        y.* *= p3a.had;
    };
}

/// The source's order (v *= suh, then fwht128, no contraction); bf16 * f16 is exact in fp32, so it matches rot128.
fn rotSource(x: *const [32][4]f32, s: *const [32][4]f32, v: *[32][4]f32) void {
    for (x, s, v) |xl, sl, *l| l.* = .{ xl[0] * sl[0], xl[1] * sl[1], xl[2] * sl[2], xl[3] * sl[3] };
    p3a.fwht128(v);
    for (v) |*l| for (l) |*y| {
        y.* *= p3a.had;
    };
}

/// H: wo_a's x3seg rot_in, group s reading o's columns from s gk (row stride heads 512), xh at offs[s] (stride gk).
fn rotIns(o: []const u16, heads: usize, rows: usize, gk: usize, suh: []const []const u16, offs: []const usize, xh: []u16, f: Rot) void {
    const ld = heads * ar.head_dim;
    for (suh, offs, 0..) |sg, off, s| for (0..rows) |row| for (0..gk / 128) |blk| {
        var x: [32][4]f32 = undefined;
        var sc: [32][4]f32 = undefined;
        var v: [32][4]f32 = undefined;
        for (0..32) |lane| for (0..4) |j| {
            const k = blk * 128 + 4 * lane + j;
            x[lane][j] = p3a.widenBf16(o[row * ld + s * gk + k]);
            sc[lane][j] = p3a.widenF16(sg[k]);
        };
        f(&x, &sc, &v);
        for (0..32) |lane| for (0..4) |j| {
            xh[off + row * gk + blk * 128 + 4 * lane + j] = p3a.f16Bits(v[lane][j]);
        };
    };
}

/// The fold: rot_cols over split_kernel P's merge units (thread tid, unit tid + ub NT), through rotIn's pointers.
fn foldRows(o: []const u16, heads: usize, rows: usize, gk: usize, suh: []const []const u16, offs: []const usize, xh: []u16, p: usize) !void {
    const base: u64 = 0x100000;
    const tag: u64 = 1 << 32; // suh pointers name their group: (s + 1) << 32
    var sp: [ar.max_groups]u64 = undefined;
    for (sp[0..suh.len], 0..) |*x, s| x.* = tag * @as(u64, s + 1);
    const ro = try ar.rotIn(heads, sp[0..suh.len], gk, base, offs);
    const ld: usize = @intCast(ro.gk);
    const nc = ar.head_dim / p;
    const units = 16 * (nc / 4) / 256;
    for (0..rows) |r| for (0..heads / 16) |hb| for (0..p) |k| for (0..units) |ub| for (0..8) |warp| {
        var x: [32][4]f32 = undefined;
        var sc: [32][4]f32 = undefined;
        var v: [32][4]f32 = undefined;
        var at: [32]usize = undefined;
        for (0..32) |lane| {
            const u = 32 * warp + lane + ub * 256;
            const h = hb * 16 + u / (nc / 4);
            const d = nc * k + (u % (nc / 4)) * 4;
            const col = h * ar.head_dim + d;
            const s = col / ld;
            const j = col - s * ld;
            const g: usize = @intCast(ro.suh[s] / tag - 1);
            for (0..4) |e| {
                x[lane][e] = p3a.widenBf16(o[(r * heads + h) * ar.head_dim + d + e]);
                sc[lane][e] = p3a.widenF16(suh[g][j + e]);
            }
            at[lane] = @as(usize, @intCast((ro.xh[s] - base) / 2)) + r * ld + j;
        }
        rot128(&x, &sc, &v);
        for (0..32) |lane| for (0..4) |e| {
            xh[at[lane] + e] = p3a.f16Bits(v[lane][e]);
        };
    };
}

const Shape = struct { heads: usize, groups: usize };

/// TP2's rank (32 heads in 4 groups), TP1 (64 in 8), one head tile in 2 groups, and 8 groups ending mid-tile.
const shapes = [_]Shape{ .{ .heads = 32, .groups = 4 }, .{ .heads = 64, .groups = 8 }, .{ .heads = 16, .groups = 2 }, .{ .heads = 32, .groups = 8 } };

fn randomO(r: std.Random, o: []u16) void {
    for (o) |*v| v.* = switch (r.uintLessThan(u8, 6)) {
        0 => p3a.bf16Bits(@bitCast(p3a.y_edges[r.uintLessThan(usize, p3a.y_edges.len)])),
        1 => r.int(u16), // any bf16, NaN payloads included
        2 => p3a.bf16Bits(@floatCast(@as(f64, @floatFromInt(r.int(i16))) * 64.0)), // past fp16 once rotated
        else => p3a.bf16Bits(@floatCast(r.floatNorm(f64) * 0.25)),
    };
}

const Case = struct { o: []u16, suh: [][]u16, offs: []usize, gk: usize, len: usize };

fn makeCase(a: std.mem.Allocator, r: std.Random, sh: Shape, rows: usize) !Case {
    const gk = sh.heads * ar.head_dim / sh.groups;
    const o = try a.alloc(u16, rows * sh.heads * ar.head_dim);
    randomO(r, o);
    const suh = try a.alloc([]u16, sh.groups);
    const offs = try a.alloc(usize, sh.groups);
    for (suh, offs, 0..) |*s, *off, i| {
        s.* = try a.alloc(u16, gk);
        p3a.randomSuh(r, s.*);
        off.* = i * rows * gk; // fusedRot's layout (block_attn.woaRows)
    }
    return .{ .o = o, .suh = suh, .offs = offs, .gk = gk, .len = sh.groups * rows * gk };
}

test "rotation twin: the folded merge writes rot_in's exact wo_a rows at P 1 and 4 (random and edge values)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x3b_a77e);
    const r = prng.random();
    for (shapes) |sh| for ([_]usize{ 1, 2, 3 }) |rows| {
        const c = try makeCase(a, r, sh, rows);
        const want = try a.alloc(u16, c.len);
        rotIns(c.o, sh.heads, rows, c.gk, c.suh, c.offs, want, &rot128);
        for ([_]usize{ 1, 4 }) |p| {
            const got = try a.alloc(u16, c.len);
            @memset(got, 0xdead); // every element must be written
            try foldRows(c.o, sh.heads, rows, c.gk, c.suh, c.offs, got, p);
            try testing.expectEqualSlices(u16, want, got);
        }
    };
}

test "rotation twin controls: contraction is exact on bf16 inputs, a mid-block start and an off-grid group fail" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xc0_47e2);
    const r = prng.random();
    const sh = shapes[0];
    const c = try makeCase(a, r, sh, 2);
    for (c.o) |*v| v.* = p3a.bf16Bits(@floatCast(r.floatNorm(f64))); // finite
    const want = try a.alloc(u16, c.len);
    rotIns(c.o, sh.heads, 2, c.gk, c.suh, c.offs, want, &rot128);
    // bf16 * f16 products are exact, so rot_in's fused scale matches the source; full fp32 inputs tell them apart
    var x: [32][4]f32 = undefined;
    var sc: [32][4]f32 = undefined;
    var v: [32][4]f32 = undefined;
    var u: [32][4]f32 = undefined;
    var differs = false;
    for (0..c.o.len / 128) |blk| {
        for (0..32) |lane| for (0..4) |j| {
            x[lane][j] = p3a.widenBf16(c.o[128 * blk + 4 * lane + j]);
            sc[lane][j] = p3a.widenF16(c.suh[0][(4 * lane + j) % c.suh[0].len]);
        };
        rot128(&x, &sc, &v);
        rotSource(&x, &sc, &u);
        try testing.expectEqualSlices(u8, std.mem.asBytes(&u), std.mem.asBytes(&v));
        for (&x) |*l| for (l) |*y| {
            y.* = r.float(f32) * 2 - 1;
        };
        rot128(&x, &sc, &v);
        rotSource(&x, &sc, &u);
        if (!std.mem.eql(u8, std.mem.asBytes(&v), std.mem.asBytes(&u))) differs = true;
    }
    try testing.expect(differs);
    // a block 64 columns into a head: not one of rot_in's blocks
    for (0..32) |lane| for (0..4) |j| {
        x[lane][j] = p3a.widenBf16(c.o[64 + 4 * lane + j]);
        sc[lane][j] = p3a.widenF16(c.suh[0][64 + 4 * lane + j]);
    };
    rot128(&x, &sc, &v);
    var shifted: [128]u16 = undefined;
    for (0..32) |lane| for (0..4) |j| {
        shifted[4 * lane + j] = p3a.f16Bits(v[lane][j]);
    };
    try testing.expect(!std.mem.eql(u16, want[64..192], &shifted));
    // groups off the 128-column grid are refused before any launch
    const sp = [_]u64{ 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000, 0x8000 };
    const offs: [8]usize = @splat(0);
    try testing.expectError(error.Unsupported, ar.rotIn(9, &sp, 576, 0x100000, &offs));
}
