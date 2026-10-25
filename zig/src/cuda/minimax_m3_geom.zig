//! MiniMax-M3.1-Flash sparse attention: paged addressing, launch geometry and sizing (mirrors minimax_m3_sparse.cuh).
const std = @import("std");

pub const d = 128; // head_dim and index_head_dim
pub const block = 128; // keys a selection block = tokens a page
pub const topk = 16; // blocks a row attends
pub const group = 16; // query heads a kv head
pub const bf16_bytes = 2;

pub const index_threads = 128;
pub const select_threads = 256;
pub const attn_threads = 256;
/// Static shared bytes of the attention kernel: sq 8 + sk 8.5 + sv 8 + sp 2 KiB (plus 1 KiB reserved).
pub const attn_static_shared = (group * d + group * 32) * 4 + (32 * 136 + 32 * d) * bf16_bytes;
/// Scores a block at 1 key a warp step: the index kernel's reduction tree (32 lanes, offsets 16..1) is fixed.
pub const max_grid_z = 65535;

/// Bytes of one fp32 scale a K or V row holds in the fp8 layout.
pub const scale_bytes = 4;
/// Largest finite e4m3 value: a row's scale is its largest magnitude over this.
pub const e4m3_max = 448;

/// One rank's cache: `kvh` local kv heads and its layers; `kv8` is the e4m3 layout of docs/fp8-kv.md (scales after IK).
pub const Pool = struct {
    kvh: usize,
    layers: usize,
    kv8: bool = false,

    /// Bytes one page holds for one layer: K and V for each local head, plus the shared index keys (and fp8's scales).
    pub fn layerBytes(p: Pool) usize {
        if (p.kv8) return block * (2 * p.kvh * d + d * bf16_bytes + 2 * p.kvh * scale_bytes);
        return block * (2 * p.kvh * d + d) * bf16_bytes;
    }
    pub fn pageBytes(p: Pool) usize {
        return p.layers * p.layerBytes();
    }
    pub fn tokenBytes(p: Pool) usize {
        return p.pageBytes() / block;
    }
    pub fn layerOff(p: Pool, layer: usize) usize {
        return layer * p.layerBytes();
    }
    /// Bytes a K or V value takes.
    pub fn valueBytes(p: Pool) usize {
        return if (p.kv8) 1 else bf16_bytes;
    }
    /// Byte offset from the pool base of K (which 0) or V (which 1), local head h, token t of a page, in a layer.
    pub fn kv(p: Pool, page: usize, layer: usize, which: usize, h: usize, t: usize) usize {
        return page * p.pageBytes() + p.layerOff(layer) + ((which * p.kvh + h) * block + t) * d * p.valueBytes();
    }
    /// Byte offset inside a layer region of the index keys (256 bytes a token in both layouts).
    pub fn ikInLayer(p: Pool) usize {
        return 2 * p.kvh * block * d * p.valueBytes();
    }
    pub fn ik(p: Pool, page: usize, layer: usize, t: usize) usize {
        return page * p.pageBytes() + p.layerOff(layer) + p.ikInLayer() + t * d * bf16_bytes;
    }
    /// Byte offset of the fp32 scale of K (which 0) or V (which 1), head h, token t (fp8 layout only).
    pub fn scale(p: Pool, page: usize, layer: usize, which: usize, h: usize, t: usize) usize {
        std.debug.assert(p.kv8);
        return page * p.pageBytes() + p.layerOff(layer) + p.ikInLayer() + block * d * bf16_bytes + ((which * p.kvh + h) * block + t) * scale_bytes;
    }
    pub fn pagesFor(tokens: usize) usize {
        return (tokens + block - 1) / block;
    }
    /// Cache bytes of a stream holding `tokens` (whole pages).
    pub fn bytesFor(p: Pool, tokens: usize) usize {
        return pagesFor(tokens) * p.pageBytes();
    }
    /// Most tokens (a whole number of pages) that fit in `bytes`.
    pub fn tokensFitting(p: Pool, bytes: usize) usize {
        return bytes / p.pageBytes() * block;
    }
};

/// Blocks a row at `pos` can see (the last is its own and may be partial).
pub fn blocks(pos: usize) usize {
    return pos / block + 1;
}
/// Valid keys of the row's own block (causal): pos % 128 + 1.
pub fn ownKeys(pos: usize) usize {
    return pos % block + 1;
}
/// Blocks a row attends: all when it has at most 16, else 16.
pub fn selected(pos: usize) usize {
    return @min(topk, blocks(pos));
}
/// Keys a row attends when its own block is among the selected (always so with the default forcing).
pub fn keysAttended(pos: usize) usize {
    return if (blocks(pos) <= topk) pos + 1 else (topk - 1) * block + ownKeys(pos);
}
/// 32-key steps the attention kernel runs for that row.
pub fn attnSteps(pos: usize) usize {
    const full = (selected(pos) - 1) * (block / 32);
    return full + (ownKeys(pos) + 31) / 32;
}

pub const Launch = struct { grid: [3]u32, block: u32, dynamic_shared: u32 = 0 };

/// Scoring: a CTA per (block, kv head, row); `max_pos` is the largest position of the rows.
pub fn indexLaunch(rows: usize, kvh: usize, max_pos: usize) Launch {
    return .{ .grid = .{ @intCast(blocks(max_pos)), @intCast(kvh), @intCast(rows) }, .block = index_threads };
}
pub fn selectLaunch(rows: usize, kvh: usize) Launch {
    return .{ .grid = .{ @intCast(kvh), @intCast(rows), 1 }, .block = select_threads };
}
pub fn attnLaunch(rows: usize, kvh: usize) Launch {
    return .{ .grid = .{ @intCast(rows), @intCast(kvh), 1 }, .block = attn_threads };
}

/// fp32 block maxima [rows, kvh, stride]; the stride is the block-table length (>= blocks(max_pos)).
pub fn bmaxBytes(rows: usize, kvh: usize, stride: usize) usize {
    return rows * kvh * stride * 4;
}
/// int32 selected-block lists [rows, kvh, 16].
pub fn selBytes(rows: usize, kvh: usize) usize {
    return rows * kvh * topk * 4;
}
/// IK bytes one scoring pass reads from the cache for a row at `pos` (one shared key head, 256 B a token).
pub fn indexReadBytes(pos: usize) usize {
    return (pos + 1) * d * bf16_bytes;
}

pub const LaunchError = error{ TooManyRows, KvHeads, Empty };

pub fn check(rows: usize, kvh: usize) LaunchError!void {
    if (rows == 0) return error.Empty;
    if (rows > max_grid_z) return error.TooManyRows;
    if (kvh == 0 or kvh > 4) return error.KvHeads;
}

/// The select kernel's launch arguments; defaults are minimax_ref.zig's Switches defaults (UNVERIFIED upstream).
pub const Rules = struct {
    init_n: i32 = 0,
    init_inclusive: bool = false,
    local_n: i32 = 1,
    local_trailing: bool = false,
    higher_tie: bool = false,
};
