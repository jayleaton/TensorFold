//! Token draws from the forward's gathered fp32 logits on rank 0's host, by the lane core's keyed rule; others do none.
const std = @import("std");
const ops = @import("engine_ops.zig");
const shard = @import("shard.zig");
const iface = @import("iface.zig");
const gpu_rule = @import("lanes").gpu_rule;

const Cand = gpu_rule.Cand;

/// How the device's gathered rows reach the host: `out.len` bytes at device address `addr`, after the stream drained.
pub const Fetch = struct {
    ptr: *anyopaque,
    call: *const fn (ptr: *anyopaque, addr: u64, out: []u8) anyerror!void,
};

/// Reusable memory of a draw: a whole row and the candidates' heap.
pub const Scratch = struct {
    row: []f32,
    heap: []Cand,

    pub fn init(gpa: std.mem.Allocator, vocab: usize) !Scratch {
        const row = try gpa.alloc(f32, vocab);
        errdefer gpa.free(row);
        return .{ .row = row, .heap = try gpa.alloc(Cand, gpu_rule.candidates) };
    }

    pub fn deinit(s: *Scratch, gpa: std.mem.Allocator) void {
        gpa.free(s.row);
        gpa.free(s.heap);
    }
};

/// A logit the draw cannot take: NaN or +inf (-inf is a masked token).
fn bad(v: f32) bool {
    return std.math.isNan(v) or v == std.math.inf(f32);
}

/// gpu_rule.sample's draw without sorting the whole vocabulary: the candidates come from a bounded heap.
pub fn sample(s: *Scratch, logits: []const f32, rule: ?ops.Sampling, position: u32) !u32 {
    for (logits) |v| if (bad(v)) return error.NonFiniteLogits;
    const st = rule orelse {
        const at = gpu_rule.argmax(logits);
        return if (logits[at] == -std.math.inf(f32)) error.NoAllowedToken else at;
    };
    const r = gpu_rule.rule(st);
    const cap: usize = if (r.top_k == 0 or r.top_k > gpu_rule.candidates) gpu_rule.candidates else r.top_k;
    const n: usize = @min(@min(logits.len, gpu_rule.candidates), cap);
    var m = -std.math.inf(f32);
    for (logits) |l| m = @max(m, l * r.inv_t);
    if (m == -std.math.inf(f32)) return error.NoAllowedToken;
    var lanes: [1024]f32 = @splat(0.0);
    var heap_n: usize = 0;
    for (logits, 0..) |l, i| {
        const c: Cand = .{ .v = l * r.inv_t, .i = @intCast(i) };
        lanes[i % 1024] += @exp(c.v - m);
        heapPush(s.heap[0..n], &heap_n, c);
    }
    const all = s.heap[0..heap_n];
    std.mem.sort(Cand, all, {}, gpu_rule.better);
    var norm = gpu_rule.lanesSum(&lanes);
    if (r.top_k != 0) {
        norm = 0.0;
        for (all) |c| norm += @exp(c.v - m);
    }
    var keep: usize = all.len;
    if (r.top_p > 0.0 and r.top_p < 1.0) {
        var cum: f32 = 0.0;
        for (all, 0..) |c, j| {
            cum += @exp(c.v - m) / norm;
            if (cum >= r.top_p) {
                keep = j + 1;
                break;
            }
        }
    }
    if (r.min_log > -std.math.inf(f32)) {
        const floor = m + r.min_log;
        var j: usize = 0;
        while (j < keep and all[j].v >= floor) j += 1;
        keep = j;
    }
    var best: usize = 0;
    var best_score = -std.math.inf(f32);
    for (all[0..keep], 0..) |c, t| {
        const score = c.v - @log(-@log(gpu_rule.uniform24(st.seed, position, c.i)));
        if (t == 0 or score > best_score) {
            best = t;
            best_score = score;
        }
    }
    return all[best].i;
}

/// A min-heap on the kernel's candidate order (the worst candidate at the root) holding the best `heap.len`.
fn heapPush(heap: []Cand, n: *usize, c: Cand) void {
    if (n.* < heap.len) {
        var i = n.*;
        n.* += 1;
        heap[i] = c;
        while (i > 0) { // sift up while the child is worse than its parent
            const p = (i - 1) / 2;
            if (!gpu_rule.better({}, heap[p], heap[i])) break;
            std.mem.swap(Cand, &heap[p], &heap[i]);
            i = p;
        }
        return;
    }
    if (!gpu_rule.better({}, c, heap[0])) return;
    heap[0] = c;
    var i: usize = 0;
    while (true) {
        const l = 2 * i + 1;
        const r = l + 1;
        var w = i;
        if (l < heap.len and gpu_rule.better({}, heap[w], heap[l])) w = l;
        if (r < heap.len and gpu_rule.better({}, heap[w], heap[r])) w = r;
        if (w == i) return;
        std.mem.swap(Cand, &heap[w], &heap[i]);
        i = w;
    }
}

/// A row of grammar words (bit t % 32 of word t / 32: token t allowed) applied to a whole row; the rest become -inf.
pub fn mask(row: []f32, words: []const u32) void {
    for (row, 0..) |*v, t| {
        if (words[t / 32] >> @intCast(t % 32) & 1 == 0) v.* = -std.math.inf(f32);
    }
}

/// Rank 0's draw over the gathered logits `[rank][row][slot]` of the last pass; another rank's draw returns at once.
pub const HostDrawer = struct {
    gpa: std.mem.Allocator,
    rank: usize,
    spans: []shard.Span, // every rank's vocabulary span
    slot: usize, // logits a rank's slot holds per row (the largest share)
    vocab: usize,
    fetch: Fetch,
    scratch: Scratch,
    gathered: []f32, // world * total * slot, for the largest pass

    pub fn init(gpa: std.mem.Allocator, rank: usize, spans: []const shard.Span, vocab: usize, fetch: Fetch) !*HostDrawer {
        const d = try gpa.create(HostDrawer);
        errdefer gpa.destroy(d);
        const own = try gpa.dupe(shard.Span, spans);
        errdefer gpa.free(own);
        var slot: usize = 0;
        for (spans) |sp| slot = @max(slot, sp.len());
        var scratch = try Scratch.init(gpa, vocab);
        errdefer scratch.deinit(gpa);
        const gathered: []f32 = if (rank == 0) try gpa.alloc(f32, spans.len * iface.shared_rows * slot) else &.{};
        d.* = .{ .gpa = gpa, .rank = rank, .spans = own, .slot = slot, .vocab = vocab, .fetch = fetch, .scratch = scratch, .gathered = gathered };
        return d;
    }

    pub fn deinit(d: *HostDrawer) void {
        if (d.rank == 0) d.gpa.free(d.gathered);
        d.scratch.deinit(d.gpa);
        d.gpa.free(d.spans);
        d.gpa.destroy(d);
    }

    pub fn drawer(d: *HostDrawer) ops.Drawer {
        return .{ .ptr = d, .draw_fn = drawFn };
    }

    fn drawFn(ptr: *anyopaque, req: ops.Draw) anyerror!void {
        const d: *HostDrawer = @ptrCast(@alignCast(ptr));
        if (d.rank != 0 or req.rows == 0) return;
        const world = d.spans.len;
        if (req.total > iface.shared_rows or req.row0 + req.rows > req.total or req.out.len < req.rows or req.first == 0) return error.WindowTooWide;
        const g = d.gathered[0 .. world * req.total * d.slot];
        try d.fetch.call(d.fetch.ptr, req.base, std.mem.sliceAsBytes(g));
        const words = (d.vocab + 31) / 32;
        for (0..req.rows) |k| {
            for (d.spans, 0..) |sp, q| @memcpy(d.scratch.row[sp.lo..sp.hi], g[(q * req.total + req.row0 + k) * d.slot ..][0..sp.len()]);
            if (req.bits != 0) {
                const all: [*]const u32 = @ptrFromInt(req.bits);
                mask(d.scratch.row, all[k * words ..][0..words]);
            }
            req.out[k] = try sample(&d.scratch, d.scratch.row, req.rule, @intCast(req.first + k));
        }
    }
};

fn randomRow(r: std.Random, row: []f32, levels: u32) void {
    const noise: f32 = if (levels > 1000) 1.0 else 0.0;
    for (row) |*v| v.* = @as(f32, @floatFromInt(r.uintLessThan(u32, levels))) * 0.25 - 4.0 + r.float(f32) * noise;
}

test "the bounded-heap draw is the lane core's draw, with ties, every rule and masked tokens" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    const vocab = 4099;
    var s = try Scratch.init(gpa, vocab);
    defer s.deinit(gpa);
    const row = try gpa.alloc(f32, vocab);
    defer gpa.free(row);
    const rules = [_]?ops.Sampling{
        null,
        .{ .seed = 1, .temperature = 0.8, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 2, .temperature = 1.0, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 3, .temperature = 0.3, .top_k = 1, .top_p = 0.5, .min_p = 0.05 },
        .{ .seed = 4, .temperature = 1.5, .top_k = 5000, .top_p = 0.9, .min_p = 0.01 },
    };
    for (0..30) |case| {
        randomRow(r, row, if (case % 2 == 0) 40 else 100_000); // many exact ties, then few
        if (case % 3 == 0) for (row, 0..) |*v, t| {
            if (t % 5 != 0) v.* = -std.math.inf(f32); // a grammar left every fifth token
        };
        for (rules) |rule| {
            const pos: u32 = @intCast(100 + case);
            const want = try gpu_rule.sample(gpa, row, rule, pos, null);
            try std.testing.expectEqual(want, try sample(&s, row, rule, pos));
        }
    }
}

test "a non-finite logit is refused and a grammar keeps only its tokens" {
    const gpa = std.testing.allocator;
    var s = try Scratch.init(gpa, 64);
    defer s.deinit(gpa);
    const row = try gpa.alloc(f32, 64);
    defer gpa.free(row);
    @memset(row, 0);
    row[7] = 5;
    try std.testing.expectEqual(@as(u32, 7), try sample(&s, row, null, 1));
    var words = [_]u32{ 1 << 3 | 1 << 8, 1 << 5 }; // tokens 3, 8 and 37 allowed
    mask(row, &words);
    try std.testing.expectEqual(@as(u32, 3), try sample(&s, row, null, 1)); // 7 is masked; ties go to the lowest id
    var none = [_]u32{ 0, 0 };
    mask(row, &none);
    try std.testing.expectError(error.NoAllowedToken, sample(&s, row, null, 1));
    try std.testing.expectError(error.NoAllowedToken, sample(&s, row, .{ .seed = 1, .temperature = 1 }, 1));
    row[0] = std.math.nan(f32);
    try std.testing.expectError(error.NonFiniteLogits, sample(&s, row, null, 1));
}

test "rank 0 draws from the gathered slots of every rank, the other rank draws nothing" {
    const gpa = std.testing.allocator;
    const spans = [_]shard.Span{ .{ .lo = 0, .hi = 6 }, .{ .lo = 6, .hi = 10 } }; // slot 6: the second rank pads two
    const Dev = struct {
        data: [2 * 3 * 6]f32 = @splat(0), // [rank][row][slot] for three rows
        fn fetch(p: *anyopaque, addr: u64, out: []u8) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(p));
            try std.testing.expectEqual(@as(u64, 0xabc), addr);
            @memcpy(out, std.mem.sliceAsBytes(self.data[0 .. out.len / 4]));
        }
    };
    var dev: Dev = .{};
    dev.data[(0 * 3 + 1) * 6 + 2] = 9; // rank 0, row 1: token 2 wins
    dev.data[(1 * 3 + 2) * 6 + 3] = 9; // rank 1, row 2: token 6 + 3 = 9 wins
    const f: Fetch = .{ .ptr = &dev, .call = Dev.fetch };
    const d0 = try HostDrawer.init(gpa, 0, &spans, 10, f);
    defer d0.deinit();
    var out: [2]u32 = undefined;
    try d0.drawer().draw(.{ .base = 0xabc, .total = 3, .row0 = 1, .rows = 2, .first = 50, .rule = null, .bits = 0, .out = &out });
    try std.testing.expectEqualSlices(u32, &.{ 2, 9 }, &out);
    const d1 = try HostDrawer.init(gpa, 1, &spans, 10, f);
    defer d1.deinit();
    out = .{ 77, 77 };
    try d1.drawer().draw(.{ .base = 0xabc, .total = 3, .row0 = 1, .rows = 2, .first = 50, .rule = null, .bits = 0, .out = &out });
    try std.testing.expectEqualSlices(u32, &.{ 77, 77 }, &out);
    var words = [_]u32{ 1 << 9, 1 << 9 }; // only token 9 allowed in either row
    try d0.drawer().draw(.{ .base = 0xabc, .total = 3, .row0 = 0, .rows = 2, .first = 50, .rule = null, .bits = @intFromPtr(&words), .out = &out });
    try std.testing.expectEqualSlices(u32, &.{ 9, 9 }, &out);
}
