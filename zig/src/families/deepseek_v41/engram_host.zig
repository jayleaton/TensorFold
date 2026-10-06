//! Engram's host side (Python engram_host.py): the table layout (prime bucket sizes and row offsets of every layer,
//! order and head), the n-gram hashes of a window's tokens, the heads a rank owns, and M2a's synthetic rows. The token
//! map and the hash multipliers come precomputed per pack (tools/zig/dsv41_engram_host.py: the tokenizer normalizer
//! and numpy's PCG64 are not worth porting); everything else is computed here as Python computes it (int64 wrapping
//! products, XOR, floored modulo).

const std = @import("std");

pub const dead_id: i64 = -1;
pub const max_layers = 4;
pub const max_ngram = 8;
pub const max_cols = 64;

pub const Host = struct {
    layer_ids: []const u32,
    max_ngram: u32,
    heads: u32,
    /// token id -> compressed id
    token_map: []const u32,
    pad_id: i64,
    mult: [max_layers][max_ngram]i64 = undefined,
    primes: [max_layers][max_cols]i64 = undefined,
    offsets: [max_layers][max_cols]i64 = undefined,

    pub fn cols(h: *const Host) u32 {
        return (h.max_ngram - 1) * h.heads;
    }

    /// Prime bucket sizes: for each layer and order the next unused primes above `vocab_size - 1` (Python's order).
    pub fn layout(h: *Host, vocab_size: u64) void {
        var seen: [max_layers * max_cols]i64 = undefined;
        var nseen: usize = 0;
        for (0..h.layer_ids.len) |li| {
            var col: usize = 0;
            for (0..h.max_ngram - 1) |_| {
                var cur: i64 = @intCast(vocab_size - 1);
                for (0..h.heads) |_| {
                    cur += 1;
                    while (!isPrime(@intCast(cur)) or std.mem.indexOfScalar(i64, seen[0..nseen], cur) != null) cur += 1;
                    seen[nseen] = cur;
                    nseen += 1;
                    h.primes[li][col] = cur;
                    col += 1;
                }
            }
            var acc: i64 = 0;
            for (0..col) |c| {
                h.offsets[li][c] = acc;
                acc += h.primes[li][c];
            }
        }
    }

    /// Table rows [t][layers][cols] of `ids` with `lookback` (the ids just before, oldest first; fewer than
    /// max_ngram - 1: the sequence starts there). Ids past the map are image positions (dead).
    pub fn hash(h: *const Host, ids: []const u32, lookback: []const u32, out: []i64) void {
        const nb = lookback.len;
        const L = h.layer_ids.len;
        const C = h.cols();
        std.debug.assert(out.len == ids.len * L * C);
        for (0..ids.len) |t| for (0..L) |li| {
            var blocked = false;
            var rolling: i64 = 0;
            for (0..h.max_ngram) |s| {
                const j = @as(i64, @intCast(t + nb)) - @as(i64, @intCast(s));
                const before = j < 0;
                const v: i64 = if (before) h.pad_id else h.src(if (j < @as(i64, @intCast(nb))) lookback[@intCast(j)] else ids[@as(usize, @intCast(j)) - nb]);
                blocked = blocked or before or v == dead_id;
                const value = if (blocked) h.pad_id else v;
                rolling ^= value *% h.mult[li][s];
                if (s == 0) continue;
                for (0..h.heads) |hh| {
                    const c = (s - 1) * h.heads + hh;
                    out[(t * L + li) * C + c] = @mod(rolling, h.primes[li][c]) + h.offsets[li][c];
                }
            }
        };
    }

    fn src(h: *const Host, id: u32) i64 {
        return if (id >= h.token_map.len) dead_id else h.token_map[id];
    }

    /// (first head, heads) rank `rank` of `world` owns: complete heads, TP-major (head_shard).
    pub fn headShard(h: *const Host, rank: u32, world: u32) [2]u32 {
        const n = h.cols();
        const part = (n + world - 1) / world;
        const h0 = @min(rank * part, n);
        return .{ h0, @min(h0 + part, n) - h0 };
    }

    /// Loads tools/zig/dsv41_engram_host.py's file (the map borrowed from `bytes`, which must outlive the host).
    pub fn load(bytes: []const u8, heads: u32, pad_token: u32, vocab_size: u64, layer_buf: []u32) !Host {
        if (bytes.len < 24 or !std.mem.eql(u8, bytes[0..8], "DSV41EH1")) return error.BadEngramHost;
        const L = std.mem.readInt(u32, bytes[8..12], .little);
        const M = std.mem.readInt(u32, bytes[12..16], .little);
        const V = std.mem.readInt(u32, bytes[16..20], .little);
        if (L > max_layers or M > max_ngram or (M - 1) * heads > max_cols or L > layer_buf.len) return error.BadEngramHost;
        var at: usize = 24;
        for (0..L) |i| layer_buf[i] = std.mem.readInt(u32, bytes[at + 4 * i ..][0..4], .little);
        at += 4 * L;
        var h: Host = .{ .layer_ids = layer_buf[0..L], .max_ngram = M, .heads = heads, .token_map = &.{}, .pad_id = 0 };
        for (0..L) |li| for (0..M) |s| {
            h.mult[li][s] = std.mem.readInt(i64, bytes[at..][0..8], .little);
            at += 8;
        };
        if (bytes.len - at < 4 * @as(usize, V)) return error.BadEngramHost;
        h.token_map = @alignCast(std.mem.bytesAsSlice(u32, bytes[at..][0 .. 4 * V]));
        h.pad_id = h.token_map[pad_token];
        h.layout(vocab_size);
        return h;
    }
};

/// Python engram_host._is_prime: small primes, then Miller-Rabin with bases 2, 7, 61 (exact below 4.7e9).
pub fn isPrime(n: u64) bool {
    if (n < 2) return false;
    for ([_]u64{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 }) |p| if (n % p == 0) return n == p;
    var d = n - 1;
    var r: u32 = 0;
    while (d % 2 == 0) : (r += 1) d /= 2;
    for ([_]u64{ 2, 7, 61 }) |a| {
        var x = powMod(a, d, n);
        if (x == 1 or x == n - 1) continue;
        var ok = false;
        var i: u32 = 1;
        while (i < r) : (i += 1) {
            x = @intCast(@as(u128, x) * x % n);
            if (x == n - 1) {
                ok = true;
                break;
            }
        }
        if (!ok) return false;
    }
    return true;
}

fn powMod(b0: u64, e0: u64, m: u64) u64 {
    var b: u128 = b0 % m;
    var e = e0;
    var r: u128 = 1;
    while (e > 0) : (e >>= 1) {
        if (e & 1 == 1) r = r * b % m;
        b = b * b % m;
    }
    return @intCast(r);
}

/// M2a's synthetic rows (dsv41_m1_capture.IndexRows): row i, value j = (((i * 2654435761 + j * 40503) % 511) - 255) / 1024,
/// fp32 then bf16 (exact: multiples of 1/1024 within 8 bits), into `out` [rows.len][dim] as bf16 bits.
pub fn indexRows(rows: []const i64, dim: usize, out: []u16) void {
    for (rows, 0..) |i, r| for (0..dim) |j| {
        const v = @mod(i *% 2654435761 +% @as(i64, @intCast(j)) * 40503, 511) - 255;
        const f: f32 = @as(f32, @floatFromInt(v)) / 1024.0;
        out[r * dim + j] = @truncate(@as(u32, @bitCast(f)) >> 16);
    };
}

const testing = std.testing;

test "the n-gram hashes: Python's NgramHasher on the release config and tokenizer (fixtures/engram-hash-ref.bin)" {
    const a = testing.allocator;
    const fx: []const u8 = @embedFile("fixtures/engram-hash-ref.bin");
    const L = std.mem.readInt(u32, fx[8..12], .little);
    const M = std.mem.readInt(u32, fx[12..16], .little);
    const V = std.mem.readInt(u32, fx[16..20], .little);
    var at: usize = 24 + 4 * L + 8 * L * M;
    // a full-size map holding only the entries the fixture's ids use
    const map = try a.alloc(u32, V);
    defer a.free(map);
    @memset(map, 0);
    const used = std.mem.readInt(u32, fx[at..][0..4], .little);
    at += 4;
    for (0..used) |_| {
        map[std.mem.readInt(u32, fx[at..][0..4], .little)] = std.mem.readInt(u32, fx[at + 4 ..][0..4], .little);
        at += 8;
    }
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    try file.appendSlice(a, fx[0 .. 24 + 4 * L + 8 * L * M]);
    try file.appendSlice(a, std.mem.sliceAsBytes(map));
    const bytes = try a.alignedAlloc(u8, .@"4", file.items.len);
    defer a.free(bytes);
    @memcpy(bytes, file.items);
    var ids_buf: [max_layers]u32 = undefined;
    const h = try Host.load(bytes, 8, 2, 16_000_000, &ids_buf);
    // Python's EngramLayout: the first and last primes, the last head's offset
    try testing.expectEqual(@as(i64, 16_000_057), h.primes[0][0]);
    try testing.expectEqual(@as(i64, 16_000_889), h.primes[1][23]);
    try testing.expectEqual(@as(i64, 368_005_705), h.offsets[0][23]);
    const cases = std.mem.readInt(u32, fx[at..][0..4], .little);
    at += 4;
    for (0..cases) |_| {
        const n = std.mem.readInt(u32, fx[at..][0..4], .little);
        const nb = std.mem.readInt(u32, fx[at + 4 ..][0..4], .little);
        at += 8;
        const ids_c = try a.alloc(u32, n);
        defer a.free(ids_c);
        for (ids_c, 0..) |*x, i| x.* = std.mem.readInt(u32, fx[at + 4 * i ..][0..4], .little);
        at += 4 * n;
        const lb_c = try a.alloc(u32, nb);
        defer a.free(lb_c);
        for (lb_c, 0..) |*x, i| x.* = std.mem.readInt(u32, fx[at + 4 * i ..][0..4], .little);
        at += 4 * nb;
        const got = try a.alloc(i64, n * L * h.cols());
        defer a.free(got);
        h.hash(ids_c, lb_c, got);
        for (got, 0..) |g, i| try testing.expectEqual(std.mem.readInt(i64, fx[at + 8 * i ..][0..8], .little), g);
        at += 8 * got.len;
    }
    try testing.expectEqual(fx.len, at);
    try testing.expectEqual([2]u32{ 12, 12 }, h.headShard(1, 2));
}
