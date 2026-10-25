//! MiniMax-M3.1-Flash's paged KV on core/kv_pool.zig: page geometry, Layout and block tables (a page is a block).
const std = @import("std");
const kv = @import("core").kv_pool;
const Config = @import("config.zig").Config;
const shard = @import("shard.zig");

/// Tokens a page holds: the sparse attention's block (`sparse_attention_config` block size, checked in `Dims.of`).
pub const page_tokens: u32 = 128;
/// bf16 K, V and IK; with `Dims.kv8` K and V are one byte a value (e4m3) beside an fp32 scale a row, IK stays bf16.
const elem: u64 = 2;
/// An fp8 K or V row's scale (one per token and kv head, fp32).
pub const scale_bytes: u64 = 4;
/// A kept prompt snapshot is a record of a position (the pages are the store's references): its nominal slot.
pub const snap_bytes: u64 = 64;

pub const Error = error{ BlockIsNotPage, TableFull, BadSlot };

/// What a rank caches a token and layer: its kv heads' K and V, and the one shared index key head.
pub const Dims = struct {
    layers: u32,
    kv_heads: u32, // this rank's
    head_dim: u32,
    index_dim: u32,
    /// `--kv-cache fp8`: K and V as e4m3 rows with an fp32 scale each (docs/fp8-kv.md). Index keys stay bf16.
    kv8: bool = false,

    /// The same dims with the fp8 cache on or off.
    pub fn withKv8(d: Dims, on: bool) Dims {
        var r = d;
        r.kv8 = on;
        return r;
    }

    /// Rank `s.rank`'s dims. Index keys are replicated: every rank scores its own index heads over the whole IK.
    pub fn of(c: *const Config, s: shard.Share) Error!Dims {
        if (c.index.block != page_tokens) return error.BlockIsNotPage;
        return .{ .layers = c.layers, .kv_heads = @intCast(s.kv_heads.len()), .head_dim = c.head_dim, .index_dim = c.index.dim };
    }

    /// Bytes a token adds in one layer: K and V of the rank's heads, then the index key (and fp8's two scales a head).
    pub fn layerToken(d: Dims) u64 {
        if (d.kv8) return 2 * @as(u64, d.kv_heads) * (d.head_dim + scale_bytes) + @as(u64, d.index_dim) * elem;
        return (2 * @as(u64, d.kv_heads) * d.head_dim + d.index_dim) * elem;
    }

    /// Bytes a token adds across every layer (76,800 for the shipped model at two ranks; 47,040 with fp8 K and V).
    pub fn tokenBytes(d: Dims) u64 {
        return d.layerToken() * d.layers;
    }

    pub fn pageBytes(d: Dims) u64 {
        return d.tokenBytes() * page_tokens;
    }

    /// Bytes of one stream's block-table row of `max_pages` entries.
    pub fn tableBytes(max_pages: u64) u64 {
        return max_pages * @sizeOf(i32);
    }

    /// The layout for streams of at most `max_pages` pages: the table row is their fixed part, every stream pages.
    pub fn layout(d: Dims, max_pages: u64) kv.Layout {
        return .{ .fixed_bytes = tableBytes(max_pages), .token_bytes = d.tokenBytes(), .page_tokens = page_tokens, .snap_bytes = snap_bytes, .all_paged = true };
    }

    /// Per-layer layout of a page: `K[kvh][128 tok][head_dim]`, `V[kvh][128][head_dim]`, `IK[128][index_dim]`.
    pub fn map(d: Dims) PageMap {
        return .{ .d = d };
    }
};

pub const PageMap = struct {
    d: Dims,

    fn layerBytes(m: PageMap) u64 {
        return m.d.layerToken() * page_tokens;
    }
    fn valueBytes(m: PageMap) u64 {
        return if (m.d.kv8) 1 else elem;
    }
    fn headBytes(m: PageMap) u64 {
        return @as(u64, page_tokens) * m.d.head_dim * m.valueBytes();
    }
    fn kvBytes(m: PageMap) u64 {
        return m.d.kv_heads * m.headBytes();
    }

    /// Byte offset of kv head `kvh`'s first key at `layer`; row `t` adds `t * head_dim * 2` (`* 1` with fp8).
    pub fn k(m: PageMap, layer: u32, kvh: u32) u64 {
        std.debug.assert(layer < m.d.layers and kvh < m.d.kv_heads);
        return layer * m.layerBytes() + kvh * m.headBytes();
    }

    pub fn v(m: PageMap, layer: u32, kvh: u32) u64 {
        std.debug.assert(layer < m.d.layers and kvh < m.d.kv_heads);
        return layer * m.layerBytes() + m.kvBytes() + kvh * m.headBytes();
    }

    /// fp8 only: the 128 fp32 scales of K head `kvh` (row t's at `t * 4`) / of V head `kvh`, after the index keys.
    pub fn kScale(m: PageMap, layer: u32, kvh: u32) u64 {
        std.debug.assert(m.d.kv8 and layer < m.d.layers and kvh < m.d.kv_heads);
        return layer * m.layerBytes() + 2 * m.kvBytes() + page_tokens * m.d.index_dim * elem + @as(u64, kvh) * page_tokens * scale_bytes;
    }
    pub fn vScale(m: PageMap, layer: u32, kvh: u32) u64 {
        return m.kScale(layer, kvh) + @as(u64, m.d.kv_heads) * page_tokens * scale_bytes;
    }

    /// The block's index keys at `layer`: 128 rows of `index_dim`, contiguous.
    pub fn ik(m: PageMap, layer: u32) u64 {
        std.debug.assert(layer < m.d.layers);
        return layer * m.layerBytes() + 2 * m.kvBytes();
    }

    /// Byte offset of a page's start in the pool's arena.
    pub fn page(m: PageMap, id: u32) u64 {
        return @as(u64, id) * m.d.pageBytes();
    }
};

/// Page id and row in it of key position `pos`.
pub fn locate(pos: u64) struct { block: usize, row: u32 } {
    return .{ .block = @intCast(pos / page_tokens), .row = @intCast(pos % page_tokens) };
}

/// Pushes `[from, from + len)` of a slot's row to the device (null: the host mirror is the table, as in tests).
pub const Upload = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, slot: u32, from: usize, entries: []const i32) anyerror!void,
};

/// Block tables: int32 page ids, a `max_pages` row a slot, -1 past the end; block `b` of a stream is page `row[b]`.
pub const Tables = struct {
    gpa: std.mem.Allocator,
    rows: []i32,
    max_pages: usize,
    slots: u32,
    upload: ?Upload = null,

    pub fn init(gpa: std.mem.Allocator, slots: u32, max_pages: usize) !Tables {
        const rows = try gpa.alloc(i32, @as(usize, slots) * max_pages);
        @memset(rows, -1);
        return .{ .gpa = gpa, .rows = rows, .max_pages = max_pages, .slots = slots };
    }

    pub fn deinit(t: *Tables) void {
        t.gpa.free(t.rows);
        t.* = undefined;
    }

    pub fn row(t: *const Tables, slot: u32) []const i32 {
        return t.rows[@as(usize, slot) * t.max_pages ..][0..t.max_pages];
    }

    /// The page holding key block `block` of `slot`; null past the stream's pages.
    pub fn page(t: *const Tables, slot: u32, block: usize) ?u32 {
        if (block >= t.max_pages) return null;
        const id = t.row(slot)[block];
        return if (id < 0) null else @intCast(id);
    }

    /// Writes `ids` at entry `from` of `slot`'s row; a failed upload leaves the row as it was.
    pub fn write(t: *Tables, slot: u32, ids: []const u32, from: usize) anyerror!void {
        if (slot >= t.slots) return error.BadSlot;
        if (from + ids.len > t.max_pages) return error.TableFull;
        const dst = t.rows[@as(usize, slot) * t.max_pages + from ..][0..ids.len];
        var saved: [64]i32 = undefined;
        const keep = if (ids.len <= saved.len) saved[0..ids.len] else try t.gpa.alloc(i32, ids.len);
        defer if (ids.len > saved.len) t.gpa.free(keep);
        @memcpy(keep, dst);
        for (dst, ids) |*d, id| d.* = @intCast(id);
        if (t.upload) |u| u.call(u.ctx, slot, from, dst) catch |e| {
            @memcpy(dst, keep);
            return e;
        };
    }

    /// The slot's row back to -1 (its stream ended).
    pub fn clear(t: *Tables, slot: u32) void {
        @memset(t.rows[@as(usize, slot) * t.max_pages ..][0..t.max_pages], -1);
    }

    /// `kv.Table` for pool.cover: the slot's entries reach this row as pages are taken.
    pub fn sink(t: *Tables, slot: u32) Sink {
        return .{ .t = t, .slot = slot };
    }

    pub const Sink = struct {
        t: *Tables,
        slot: u32,

        pub fn table(s: *Sink) kv.Table {
            return .{ .ctx = s, .write = put };
        }
        fn put(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
            const s: *Sink = @ptrCast(@alignCast(ctx));
            // cover passes the whole table and `from`: write only the new entries
            try s.t.write(s.slot, ids[from..], from);
        }
    };
};

test {
    _ = @import("pool_test.zig");
    _ = @import("budget.zig");
    _ = @import("state.zig");
    _ = @import("state_test.zig");
}
