//! MiniMax-M3.1-Flash sparse attention, host only: paged addressing, launch and memory arithmetic at block boundaries, the
//! kernels' host twins against the float64 reference (core/minimax_ref.zig), planted controls, and the gate-5 launch records.
//! No GPU ran any of this: the CUDA kernels are compile-checked only.
const std = @import("std");
const core = @import("core");
const geom = @import("mm3_geom");
const twin = @import("minimax_sparse_twin.zig");

const mr = core.minimax_ref;
const bf = core.bf16_round;
const testing = std.testing;
const registry = core.contract.registry;

const gib: usize = 1 << 30;
const boundary = [_]usize{ 127, 128, 129, 2047, 2048, 2049, 131071 };

test "token bytes: 76,800 a rank for 2 of 4 kv heads over 60 layers, 2,304 a layer for all 4" {
    const rank: geom.Pool = .{ .kvh = 2, .layers = 60 };
    try testing.expectEqual(@as(usize, 76_800), rank.tokenBytes());
    try testing.expectEqual(@as(usize, 9_830_400), rank.pageBytes());
    try testing.expectEqual(@as(usize, 163_840), rank.layerBytes());
    const full: geom.Pool = .{ .kvh = 4, .layers = 1 };
    try testing.expectEqual(@as(usize, 2304), full.tokenBytes());
}

test "plan section 5: cache sizes and the tokens a 9 to 10 GiB budget holds" {
    const rank: geom.Pool = .{ .kvh = 2, .layers = 60 };
    // 32k = 2.34 GiB, 128k = 9.375 GiB, 256k = 18.75 GiB, 1M = 75 GiB
    try testing.expectEqual(@as(usize, 32768 * 76_800), rank.bytesFor(32768));
    try testing.expectEqual(@as(usize, 131072 * 76_800), rank.bytesFor(131072));
    try testing.expectEqual(@as(usize, 9 * gib + gib * 3 / 8), rank.bytesFor(131072));
    try testing.expectEqual(@as(usize, 75 * gib), rank.bytesFor(1 << 20));
    try testing.expectEqual(@as(usize, 983 * 128), rank.tokensFitting(9 * gib)); // 125,824
    try testing.expectEqual(@as(usize, 1092 * 128), rank.tokensFitting(10 * gib)); // 139,776
    try testing.expect(rank.tokensFitting(9 * gib) < 131072 and rank.tokensFitting(10 * gib) > 131072);
    try testing.expectEqual(@as(usize, 1), geom.Pool.pagesFor(1));
    try testing.expectEqual(@as(usize, 1), geom.Pool.pagesFor(128));
    try testing.expectEqual(@as(usize, 2), geom.Pool.pagesFor(129));
    try testing.expectEqual(@as(usize, 128 * 76_800), rank.bytesFor(1));
}

test "plan sections 4-T5 and 5: scratch sizes and the IK bytes scoring reads each token" {
    try testing.expectEqual(@as(usize, 134_217_728), geom.bmaxBytes(8192, 4, 1024)); // 128 MiB at 128k, all 4 heads
    try testing.expectEqual(@as(usize, 67_108_864), geom.bmaxBytes(8192, 2, 1024)); // a rank's 2 heads
    try testing.expectEqual(@as(usize, 8192 * 2 * 16 * 4), geom.selBytes(8192, 2));
    try testing.expectEqual(@as(usize, 32 << 20), geom.indexReadBytes(131071)); // 32 MiB a layer
    try testing.expectEqual(@as(usize, 60 * (32 << 20)), 60 * geom.indexReadBytes(131071)); // 1.875 GiB a token
    try testing.expectEqual(@as(usize, 256), geom.indexReadBytes(0));
}

test "paged addressing: every (page, layer, K/V/IK, head, token) row is a distinct, gapless 256-byte row of the pool" {
    const p: geom.Pool = .{ .kvh = 2, .layers = 3 };
    const pages = 3;
    const rows = pages * p.pageBytes() / 256;
    const seen = try testing.allocator.alloc(bool, rows);
    defer testing.allocator.free(seen);
    @memset(seen, false);
    for (0..pages) |pg| for (0..p.layers) |ly| for (0..128) |t| {
        for (0..2) |w| for (0..p.kvh) |h| mark(seen, p.kv(pg, ly, w, h, t));
        mark(seen, p.ik(pg, ly, t));
    };
    for (seen) |s| try testing.expect(s);
    try testing.expectEqual(@as(usize, p.layerBytes()), p.layerOff(1));
    // the three planes of a layer are contiguous: K heads, V heads, IK
    try testing.expectEqual(p.kv(0, 0, 0, 0, 0) + 2 * 128 * 256, p.kv(0, 0, 1, 0, 0));
    try testing.expectEqual(p.kv(0, 0, 1, 0, 0) + 2 * 128 * 256, p.ik(0, 0, 0));
}

fn mark(seen: []bool, byte: usize) void {
    std.debug.assert(byte % 256 == 0);
    std.debug.assert(!seen[byte / 256]);
    seen[byte / 256] = true;
}

test "block boundaries: blocks, own-block keys, selected blocks, keys attended, steps and launch grids" {
    const want = [_]struct { nb: usize, own: usize, sel: usize, keys: usize, steps: usize }{
        .{ .nb = 1, .own = 128, .sel = 1, .keys = 128, .steps = 4 }, // 127: block 0 full
        .{ .nb = 2, .own = 1, .sel = 2, .keys = 129, .steps = 5 }, // 128: block 1 holds one key
        .{ .nb = 2, .own = 2, .sel = 2, .keys = 130, .steps = 5 },
        .{ .nb = 16, .own = 128, .sel = 16, .keys = 2048, .steps = 64 }, // 2047: the last dense row
        .{ .nb = 17, .own = 1, .sel = 16, .keys = 1921, .steps = 61 }, // 2048: the first row that selects
        .{ .nb = 17, .own = 2, .sel = 16, .keys = 1922, .steps = 61 },
        .{ .nb = 1024, .own = 128, .sel = 16, .keys = 2048, .steps = 64 }, // 128k - 1
    };
    for (boundary, want) |pos, w| {
        try testing.expectEqual(w.nb, geom.blocks(pos));
        try testing.expectEqual(w.own, geom.ownKeys(pos));
        try testing.expectEqual(w.sel, geom.selected(pos));
        try testing.expectEqual(w.keys, geom.keysAttended(pos));
        try testing.expectEqual(w.steps, geom.attnSteps(pos));
        const il = geom.indexLaunch(3, 2, pos);
        try testing.expectEqual([3]u32{ @intCast(w.nb), 2, 3 }, il.grid);
        try testing.expectEqual(@as(u32, 128), il.block);
    }
    const sl = geom.selectLaunch(5, 2);
    try testing.expectEqual([3]u32{ 2, 5, 1 }, sl.grid);
    try testing.expectEqual(@as(u32, 256), sl.block);
    const al = geom.attnLaunch(5, 2);
    try testing.expectEqual([3]u32{ 5, 2, 1 }, al.grid);
    try testing.expectEqual(@as(u32, 256), al.block);
    try testing.expectError(error.TooManyRows, geom.check(65536, 2));
    try testing.expectError(error.Empty, geom.check(0, 2));
    try geom.check(8192, 4);
    // the scoring grid covers a window's furthest row: the block count is the max over its rows
    try testing.expectEqual(@as(u32, 2), geom.indexLaunch(2, 4, 128).grid[0]);
}

fn bf16v(rng: std.Random, n: usize, sd: f32) ![]f32 {
    const v = try testing.allocator.alloc(f32, n);
    for (v) |*x| x.* = bf.round32(rng.floatNorm(f32) * sd);
    return v;
}

fn widen(x: []const f32) ![]f64 {
    const o = try testing.allocator.alloc(f64, x.len);
    for (o, x) |*d, s| d.* = s;
    return o;
}

const scale64 = 0.08838834764831845; // 128^-0.5
const scale32: f32 = @floatCast(scale64);

fn rulesOf(sw: mr.Switches) geom.Rules {
    return .{
        .init_n = @intCast(sw.sparse_init_block),
        .init_inclusive = sw.init_mode == .inclusive,
        .local_n = @intCast(sw.sparse_local_block),
        .local_trailing = sw.local_mode == .trailing_window,
        .higher_tie = sw.tie == .higher_block,
    };
}

/// Scores, selects and attends one row at `pos` with the twins, and checks every stage against float64.
fn row(pos: usize, seed: u64, compact: bool) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const n = pos + 1;
    const iq = try bf16v(rng, 128, 1);
    defer testing.allocator.free(iq);
    const ik = try bf16v(rng, n * 128, 1);
    defer testing.allocator.free(ik);
    const iq64 = try widen(iq);
    defer testing.allocator.free(iq64);
    const ik64 = try widen(ik);
    defer testing.allocator.free(ik64);
    const nb = geom.blocks(pos);
    const t_bm = try testing.allocator.alloc(f32, nb);
    defer testing.allocator.free(t_bm);
    const r_bm = try testing.allocator.alloc(f64, nb);
    defer testing.allocator.free(r_bm);
    try testing.expectEqual(nb, twin.blockMax(iq, ik, pos, scale32, .none, t_bm));
    try testing.expectEqual(nb, mr.blockScores(iq64, ik64, pos, 128, scale64, 128, r_bm));
    for (t_bm, r_bm) |t, r| try testing.expectApproxEqAbs(r, @as(f64, t), 1e-5 * (1 + @abs(r)));

    // select on the twin's own fp32 maxima so the tie order is comparable bit for bit
    const sw: mr.Switches = .{};
    var ids: [geom.topk]i32 = undefined;
    twin.select(t_bm, pos, rulesOf(sw), .none, &ids);
    const tb64 = try widen(t_bm);
    defer testing.allocator.free(tb64);
    var scratch: [1024]f64 = undefined;
    var rids: [geom.topk]usize = undefined;
    const cnt = mr.selectBlocks(tb64, nb - 1, geom.topk, sw, &scratch, &rids);
    try testing.expectEqual(geom.selected(pos), cnt);
    for (0..cnt) |i| try testing.expectEqual(@as(i32, @intCast(rids[i])), ids[i]);
    for (ids[cnt..]) |x| try testing.expectEqual(@as(i32, -1), x);
    try testing.expectEqual(@as(i32, @intCast(nb - 1)), ids[cnt - 1]); // the own block is selected, and is the last

    // attention over the selected keys
    const q = try bf16v(rng, geom.group * 128, 1);
    defer testing.allocator.free(q);
    const q64 = try widen(q);
    defer testing.allocator.free(q64);
    var keys: [geom.topk * 128]usize = undefined;
    var o64: [geom.group * 128]f64 = undefined;
    var o32: [geom.group * 128]f32 = undefined;
    if (!compact) {
        const kk = try bf16v(rng, n * 128, 1);
        defer testing.allocator.free(kk);
        const vv = try bf16v(rng, n * 128, 1);
        defer testing.allocator.free(vv);
        const k64 = try widen(kk);
        defer testing.allocator.free(k64);
        const v64 = try widen(vv);
        defer testing.allocator.free(v64);
        mr.attendBlocks(q64, k64, v64, rids[0..cnt], pos, 128, 128, geom.group, &keys, &o64);
        twin.attend(q, kk, vv, ids[0..cnt], pos, scale32, .none, &o32);
        try closeBf16(&o64, &o32);
        // fixed split: the same row alone, again, and padded with -1 entries gives the same bits
        var again: [geom.group * 128]f32 = undefined;
        twin.attend(q, kk, vv, &ids, pos, scale32, .none, &again);
        try testing.expectEqualSlices(f32, &o32, &again);
    } else {
        // 128k - 1: only the 16 selected blocks matter; gather them in order and attend at the compacted position
        const kk = try bf16v(rng, geom.topk * 128 * 128, 1);
        defer testing.allocator.free(kk);
        const vv = try bf16v(rng, geom.topk * 128 * 128, 1);
        defer testing.allocator.free(vv);
        const k64 = try widen(kk);
        defer testing.allocator.free(k64);
        const v64 = try widen(vv);
        defer testing.allocator.free(v64);
        const cpos = (geom.topk - 1) * 128 + pos % 128;
        var cids: [geom.topk]i32 = undefined;
        var crids: [geom.topk]usize = undefined;
        for (&cids, &crids, 0..) |*c, *r, i| {
            c.* = @intCast(i);
            r.* = i;
        }
        mr.attendBlocks(q64, k64, v64, &crids, cpos, 128, 128, geom.group, &keys, &o64);
        twin.attend(q, kk, vv, &cids, cpos, scale32, .none, &o32);
        try closeBf16(&o64, &o32);
    }
}

fn closeBf16(want: []const f64, got: []const f32) !void {
    var big: f64 = 0;
    for (want) |w| big = @max(big, @abs(w));
    for (want, got) |w, g| try testing.expectApproxEqAbs(w, @as(f64, g), 0.01 * big + 1e-6);
}

test "twins of index, select and attention match float64 at pos 127, 128, 129, 2047, 2048, 2049" {
    for (boundary[0..6], 0..) |pos, i| try row(pos, 100 + i, false);
}

test "twins match float64 at pos 131071 (1024 blocks): scores and selection on the whole cache, attention on the selected blocks" {
    try row(131071, 7, true);
}

test "select twin equals the reference for every forcing, local and tie switch over tie-heavy, NaN and infinite scores" {
    var prng = std.Random.DefaultPrng.init(55);
    const rng = prng.random();
    var scratch: [1024]f64 = undefined;
    var trials: usize = 0;
    for ([_]mr.InitBlocks{ .count, .inclusive }) |im| for ([_]mr.LocalBlocks{ .own_block_if_positive, .trailing_window }) |lm| {
        for ([_]mr.Tie{ .lower_block, .higher_block }) |tie| for (0..3) |init_n| for (0..4) |local_n| {
            const sw: mr.Switches = .{ .init_mode = im, .local_mode = lm, .tie = tie, .sparse_init_block = init_n, .sparse_local_block = local_n };
            for (0..40) |_| {
                const nb = 17 + rng.uintLessThan(usize, 90);
                const pos = (nb - 1) * 128 + rng.uintLessThan(usize, 128);
                var bm: [128]f32 = undefined;
                const palette = [_]f32{ -1, 0, 0.5, 0.5, 1, 1, 2, nan32, inf32, -inf32 };
                for (bm[0..nb]) |*x| x.* = palette[rng.uintLessThan(usize, palette.len)];
                var got: [geom.topk]i32 = undefined;
                twin.select(bm[0..nb], pos, rulesOf(sw), .none, &got);
                var b64: [128]f64 = undefined;
                for (b64[0..nb], bm[0..nb]) |*d, s| d.* = s;
                var rids: [geom.topk]usize = undefined;
                _ = mr.selectBlocks(b64[0..nb], nb - 1, geom.topk, sw, &scratch, &rids);
                for (rids, got) |a, b| try testing.expectEqual(@as(i32, @intCast(a)), b);
                for (got[1..], got[0 .. got.len - 1]) |a, b| try testing.expect(a > b); // strictly ascending
                trials += 1;
            }
        };
    };
    try testing.expectEqual(@as(usize, 2 * 2 * 2 * 3 * 4 * 40), trials);
}

const nan32 = std.math.nan(f32);
const inf32 = std.math.inf(f32);

test "forcing semantics are launch parameters: block 0 forced by inclusive init, own block off when local is 0" {
    var bm: [20]f32 = undefined;
    for (&bm, 0..) |*x, i| x.* = @floatFromInt(i); // block 19 (own) scores highest, block 0 lowest
    var out: [geom.topk]i32 = undefined;
    const pos = 19 * 128 + 5;
    twin.select(&bm, pos, .{}, .none, &out); // default: no init, own forced: 4..19
    try testing.expectEqual(@as(i32, 4), out[0]);
    twin.select(&bm, pos, .{ .init_n = 0, .init_inclusive = true }, .none, &out); // block 0 forced: 0, 5..19
    try testing.expectEqual(@as(i32, 0), out[0]);
    try testing.expectEqual(@as(i32, 5), out[1]);
    bm[19] = -5; // own block scores lowest: forced when local_n > 0, dropped when 0
    twin.select(&bm, pos, .{}, .none, &out);
    try testing.expectEqual(@as(i32, 19), out[15]);
    twin.select(&bm, pos, .{ .local_n = 0 }, .none, &out);
    try testing.expect(out[15] != 19);
}

fn must(ok: bool, src: std.builtin.SourceLocation) !void {
    if (!ok) {
        std.debug.print("control check failed at line {d}\n", .{src.line});
        return error.TestUnexpectedResult;
    }
}

fn padded(head: []const i32) [geom.topk]i32 {
    var o: [geom.topk]i32 = @splat(-1);
    @memcpy(o[0..head.len], head);
    return o;
}

fn differs(a: []const i32, b: []const i32) bool {
    return !std.mem.eql(i32, a, b);
}

test "planted controls: each mutant is caught by the check that the real twin passes" {
    var prng = std.Random.DefaultPrng.init(9);
    const rng = prng.random();
    var scratch: [1024]f64 = undefined;
    // no_forcing and tie_flipped: caught against the reference on tie-heavy rows; the real twin never differs
    var caught = [_]bool{ false, false };
    for (0..400) |_| {
        const nb = 17 + rng.uintLessThan(usize, 30);
        const pos = (nb - 1) * 128 + rng.uintLessThan(usize, 128);
        var bm: [64]f32 = undefined;
        for (bm[0..nb]) |*x| x.* = @floatFromInt(rng.uintLessThan(u8, 4)); // ties; the own block rarely wins on score
        var b64: [64]f64 = undefined;
        for (b64[0..nb], bm[0..nb]) |*d, s| d.* = s;
        var rids: [geom.topk]usize = undefined;
        _ = mr.selectBlocks(b64[0..nb], nb - 1, geom.topk, .{}, &scratch, &rids);
        var want: [geom.topk]i32 = undefined;
        for (&want, rids) |*w, r| w.* = @intCast(r);
        var got: [geom.topk]i32 = undefined;
        twin.select(bm[0..nb], pos, .{}, .none, &got);
        try must(!differs(&want, &got), @src());
        twin.select(bm[0..nb], pos, .{}, .no_forcing, &got);
        caught[0] = caught[0] or differs(&want, &got);
        twin.select(bm[0..nb], pos, .{}, .tie_flipped, &got);
        caught[1] = caught[1] or differs(&want, &got);
    }
    try must(caught[0] and caught[1], @src());

    // drop_tail: pos 128 puts one key in block 1; dropping it turns a finite maximum into -inf
    const ik = try bf16v(rng, 129 * 128, 1);
    defer testing.allocator.free(ik);
    const iq = try bf16v(rng, 128, 1);
    defer testing.allocator.free(iq);
    var good: [2]f32 = undefined;
    var bad: [2]f32 = undefined;
    _ = twin.blockMax(iq, ik, 128, scale32, .none, &good);
    _ = twin.blockMax(iq, ik, 128, scale32, .drop_tail, &bad);
    try must(std.math.isFinite(good[1]) and bad[1] == -inf32, @src());

    // nan_poison: one NaN key in block 0; fmax drops it (the reference's rule), the mutant yields NaN
    ik[5 * 128 + 3] = nan32;
    _ = twin.blockMax(iq, ik, 127, scale32, .none, good[0..1]);
    _ = twin.blockMax(iq, ik, 127, scale32, .nan_poison, bad[0..1]);
    try must(std.math.isFinite(good[0]) and std.math.isNan(bad[0]), @src());
    var i64s: [128 * 128]f64 = undefined;
    for (&i64s, ik[0 .. 128 * 128]) |*d, s| d.* = s;
    var q64: [128]f64 = undefined;
    for (&q64, iq) |*d, s| d.* = s;
    var r0: [1]f64 = undefined;
    _ = mr.blockScores(&q64, &i64s, 127, 128, scale64, 128, &r0);
    try must(std.math.isFinite(r0[0]), @src());

    // window_split: a 16-key tile for short rows changes bits; the fixed 32-key twin repeats exactly
    const pos = 4 * 128 + 9;
    const k = try bf16v(rng, (pos + 1) * 128, 1);
    defer testing.allocator.free(k);
    const v = try bf16v(rng, (pos + 1) * 128, 1);
    defer testing.allocator.free(v);
    const q = try bf16v(rng, geom.group * 128, 1);
    defer testing.allocator.free(q);
    const ids = padded(&.{ 0, 1, 2, 3, 4 });
    var a: [geom.group * 128]f32 = undefined;
    var b: [geom.group * 128]f32 = undefined;
    var c: [geom.group * 128]f32 = undefined;
    twin.attendRaw(q, k, v, &ids, pos, scale32, .none, &a);
    twin.attendRaw(q, k, v, &ids, pos, scale32, .none, &b);
    twin.attendRaw(q, k, v, &ids, pos, scale32, .window_split, &c);
    try testing.expectEqualSlices(f32, &a, &b);
    try must(!std.mem.eql(f32, &a, &c), @src());
}

test "attention propagates a NaN query like the float64 reference" {
    var prng = std.Random.DefaultPrng.init(3);
    const rng = prng.random();
    const pos = 300;
    const k = try bf16v(rng, (pos + 1) * 128, 1);
    defer testing.allocator.free(k);
    const v = try bf16v(rng, (pos + 1) * 128, 1);
    defer testing.allocator.free(v);
    const q = try bf16v(rng, geom.group * 128, 1);
    defer testing.allocator.free(q);
    q[7] = nan32; // head 0
    const ids = padded(&.{ 0, 1, 2 });
    var o: [geom.group * 128]f32 = undefined;
    twin.attend(q, k, v, &ids, pos, scale32, .none, &o);
    for (o[0..128]) |x| try testing.expect(std.math.isNan(x));
    for (o[128..256]) |x| try testing.expect(std.math.isFinite(x));
}

test "gate 5: the three launches are registered, match the geometry, and fit every chip" {
    const want = [_]struct { sym: []const u8, image: []const u8, threads: u32, static: u32 }{
        .{ .sym = "tf_mm3_index", .image = "minimax_m3_index", .threads = geom.index_threads, .static = 16 },
        .{ .sym = "tf_mm3_select", .image = "minimax_m3_select", .threads = geom.select_threads, .static = 136 },
        .{ .sym = "tf_mm3_attn", .image = "minimax_m3_attn", .threads = geom.attn_threads, .static = geom.attn_static_shared },
    };
    var mine: [3]registry.CudaRecord = undefined;
    for (want, &mine) |w, *m| {
        const rec = for (registry.cuda) |r| {
            if (std.mem.eql(u8, r.launch.symbol, w.sym)) break r;
        } else return error.RecordMissing;
        try testing.expectEqualStrings(w.image, rec.launch.image);
        try testing.expectEqual(w.threads, rec.launch.threads);
        try testing.expectEqual(@as(u32, 0), rec.launch.dynamic);
        for (rec.usage) |u| {
            try testing.expectEqual(w.static, u.static);
            try testing.expectEqual(w.threads, u.bound);
            try testing.expect(u.regs * w.threads <= 65536); // a block's registers fit the file
        }
        m.* = rec;
        var found = false;
        for (registry.all_cuda) |r| found = found or std.mem.eql(u8, r.launch.symbol, w.sym);
        try testing.expect(found);
    }
    try testing.expectEqual(@as(usize, 0), try registry.checkCuda(testing.allocator, &mine, true));
    // shared bytes the attention kernel declares: 8 KiB q + 8.5 KiB K tile + 8 KiB V tile + 2 KiB p
    try testing.expectEqual(@as(usize, 27_136), geom.attn_static_shared);
    // a planted control: the same launch asking for more threads than its bound must be reported
    var over = mine[2];
    over.launch.threads = 512;
    try testing.expect(try registry.checkCuda(testing.allocator, &.{over}, true) > 0);
}
