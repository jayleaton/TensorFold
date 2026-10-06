//! Split KV's row exchange (kvsplit.py, LONG-CONTEXT.md 7.2): after a selection, every rank sends the selected comp rows it owns and one
//! all-gather (tp's Collective) brings each rank all of them; the attention then reads the received rows unpaged with remapped tokens.
//!
//! Logical page k of a slot lives on rank k % W, so a row's owner is its logical page's residue. The core reads the same bytes in the same
//! list order as from the replicated pool: split == replicated, bit for bit.
//! - dense (decode windows, row mode: fixed shapes, inside CUDA graphs): entry (r, j) of the [R, K] selection becomes token owner*R*K + r*K + j;
//! - union (prefill segments, host-synced): the sorted unique rows, each owner's list padded to the longest; token owner*M + index in its list;
//!   halves of the segment's rows while send + receive pass the cap (rows are independent in every attention kernel).

const std = @import("std");
const tp = @import("tp");
const DevicePtr = tp.collective.DevicePtr;
const Stream = tp.collective.Stream;

/// One selection's dense gather: [R, K] logical rows of a comp family -> this rank's owned rows into send, and the remapped tokens.
pub const DenseArgs = extern struct {
    /// int32 [R, K] logical rows (-1: none)
    sel: u64,
    rows: u32,
    k: u32,
    /// int32 split page tables (Slot.localTableAt): one [pages] table, or row mode's stacked [S, pts]
    table: u64,
    pts: u32,
    /// int32 [R]: each row's slot in the stacked tables (0: every row reads table row 0)
    rslot: u64,
    /// log2 of rows a page
    psh: u32,
    /// the family's local tensor and its row bytes
    base: u64,
    row_bytes: u32,
    world: u32,
    /// out: [R x K, row bytes] and int32 [R, K]
    send: u64,
    tok: u64,
};

/// The device work around the all-gather: the dense gather + remap and a gather by row list (the kernel agent's .cu; HostKernels below).
pub const Kernels = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        dense: *const fn (ptr: *anyopaque, a: DenseArgs, stream: Stream) anyerror!void,
        /// send[i] = base row phys[i] (uint32 [n])
        gather: *const fn (ptr: *anyopaque, base: u64, row_bytes: u32, phys: u64, n: u32, send: u64, stream: Stream) anyerror!void,
    };
};

/// The kernels on host memory (addresses are host pointers): the reference the device kernels must equal, and the host ranks' path.
pub const HostKernels = struct {
    pub fn kernels(h: *HostKernels) Kernels {
        return .{ .ptr = h, .vtable = &.{ .dense = dense, .gather = gather } };
    }

    fn dense(_: *anyopaque, a: DenseArgs, _: Stream) anyerror!void {
        const n = @as(usize, a.rows) * a.k;
        const sel: [*]const i32 = @ptrFromInt(a.sel);
        const tab: [*]const i32 = @ptrFromInt(a.table);
        const rslot: ?[*]const i32 = if (a.rslot == 0) null else @ptrFromInt(a.rslot);
        const base: [*]const u8 = @ptrFromInt(a.base);
        const send: [*]u8 = @ptrFromInt(a.send);
        const tok: [*]i32 = @ptrFromInt(a.tok);
        const mask = (@as(u32, 1) << @intCast(a.psh)) - 1;
        for (0..n) |i| {
            const t: u32 = @intCast(@max(sel[i], 0));
            const row = i / a.k;
            const s: usize = if (rslot) |rs| @intCast(rs[row]) else 0;
            const local: u32 = @intCast(tab[s * a.pts + (t >> @intCast(a.psh))]);
            const phys = (@as(usize, local) << @intCast(a.psh)) | (t & mask);
            @memcpy(send[i * a.row_bytes ..][0..a.row_bytes], base[phys * a.row_bytes ..][0..a.row_bytes]);
            tok[i] = if (sel[i] < 0) -1 else @intCast(ownerOf(t, a.psh, a.world) * n + i);
        }
    }

    fn gather(_: *anyopaque, base: u64, rb: u32, phys: u64, n: u32, send: u64, _: Stream) anyerror!void {
        const ph: [*]const u32 = @ptrFromInt(phys);
        const b: [*]const u8 = @ptrFromInt(base);
        const s: [*]u8 = @ptrFromInt(send);
        for (0..n) |i| @memcpy(s[i * rb ..][0..rb], b[@as(usize, ph[i]) * rb ..][0..rb]);
    }
};

/// The rank that stores logical row t of a split family: its logical page's residue.
pub fn ownerOf(t: u32, psh: u32, world: u32) u32 {
    return (t >> @intCast(psh)) % world;
}

/// Send and receive bytes of a dense exchange of R x K entries.
pub fn denseBytes(rows: u32, k: u32, row_bytes: u32, world: u32) struct { send: u64, recv: u64 } {
    const s = @as(u64, rows) * k * row_bytes;
    return .{ .send = s, .recv = s * world };
}

/// A prefill union block: rows [a, b) of the segment, m rows an owner (padded), this rank's local rows to send, the block's tokens.
pub const Block = struct {
    a: u32,
    b: u32,
    m: u32,
    phys: []u32,
    tokens: []i32,

    pub fn deinit(blk: *Block, gpa: std.mem.Allocator) void {
        gpa.free(blk.phys);
        gpa.free(blk.tokens);
    }
};

pub const UnionIn = struct {
    /// [rows, k] logical rows (-1: none), host
    sel: []const i32,
    k: u32,
    /// the slot's split table (local page by logical page), host
    table: []const u32,
    psh: u32,
    world: u32,
    rank: u32,
    row_bytes: u32,
    /// send + receive bytes a block may take (TF_DSV41_KV_SPLIT_UNION_MIB)
    cap: u64,
};

/// Plans a segment's union exchange: blocks of rows whose (W + 1) x M rows fit the cap; O(n log n) in the block's entries.
/// Ranks a split exchange plans for (the session pool's bound: sessions.pool.max_world).
pub const max_world = 8;

pub fn planUnion(gpa: std.mem.Allocator, in: UnionIn, out: *std.ArrayList(Block)) !void {
    if (in.world < 1 or in.world > max_world or in.rank >= in.world) return error.BadWorld;
    const rows: u32 = @intCast(in.sel.len / in.k);
    try planRange(gpa, in, 0, rows, out);
}

fn planRange(gpa: std.mem.Allocator, in: UnionIn, a: u32, b: u32, out: *std.ArrayList(Block)) !void {
    const part = in.sel[@as(usize, a) * in.k .. @as(usize, b) * in.k];
    var uniq: std.ArrayList(u32) = .empty;
    defer uniq.deinit(gpa);
    try uniq.ensureTotalCapacity(gpa, part.len);
    for (part) |t| if (t >= 0) uniq.appendAssumeCapacity(@intCast(t));
    std.mem.sort(u32, uniq.items, {}, std.sort.asc(u32));
    var n: usize = 0;
    for (uniq.items, 0..) |t, i| if (i == 0 or t != uniq.items[i - 1]) {
        uniq.items[n] = t;
        n += 1;
    };
    uniq.shrinkRetainingCapacity(n);
    // stable partition by owner: each owner's rows stay sorted, so a row's index in its list is a binary search
    var sizes: [max_world]u32 = @splat(0);
    for (uniq.items) |t| sizes[ownerOf(t, in.psh, in.world)] += 1;
    var m: u32 = 1;
    for (sizes[0..in.world]) |sz| m = @max(m, sz);
    if (@as(u64, in.world + 1) * m * in.row_bytes > in.cap and b - a > 1) {
        const h = a + (b - a) / 2;
        try planRange(gpa, in, a, h, out);
        return planRange(gpa, in, h, b, out);
    }
    const lists = try gpa.alloc(u32, uniq.items.len);
    defer gpa.free(lists);
    var at: [max_world]u32 = undefined;
    var acc: u32 = 0;
    for (0..in.world) |r| {
        at[r] = acc;
        acc += sizes[r];
    }
    const starts = at;
    for (uniq.items) |t| {
        const r = ownerOf(t, in.psh, in.world);
        lists[at[r]] = t;
        at[r] += 1;
    }
    var blk: Block = .{ .a = a, .b = b, .m = m, .phys = try gpa.alloc(u32, m), .tokens = try gpa.alloc(i32, part.len) };
    errdefer blk.deinit(gpa);
    const mask = (@as(u32, 1) << @intCast(in.psh)) - 1;
    const mine = lists[starts[in.rank]..][0..sizes[in.rank]];
    for (blk.phys, 0..) |*p, i| p.* = if (i < mine.len) (in.table[mine[i] >> @intCast(in.psh)] << @intCast(in.psh)) | (mine[i] & mask) else 0; // padding: any stored row
    for (part, blk.tokens) |t, *o| {
        if (t < 0) {
            o.* = -1;
            continue;
        }
        const u: u32 = @intCast(t);
        const r = ownerOf(u, in.psh, in.world);
        const list = lists[starts[r]..][0..sizes[r]];
        const idx = std.sort.binarySearch(u32, list, u, orderU32).?;
        o.* = @intCast(r * m + idx);
    }
    try out.append(gpa, blk);
}

fn orderU32(a: u32, b: u32) std.math.Order {
    return std.math.order(a, b);
}

/// The exchange itself: kernels to fill the send buffer, the communicator's all-gather to fill the receive buffer.
pub const Exchange = struct {
    comm: tp.Collective,
    kernels: Kernels,

    /// Dense: send [R x K, rb] gathered and tokens remapped, then recv [W, R x K, rb] (persistent buffers: graph-capturable).
    pub fn dense(x: Exchange, a: DenseArgs, recv: DevicePtr, stream: Stream) !void {
        try x.kernels.vtable.dense(x.kernels.ptr, a, stream);
        try x.comm.allGather(a.send, recv, @as(usize, a.rows) * a.k * a.row_bytes, .u8, stream);
    }

    /// One union block: this rank's M rows (phys: the block's list on the device) to send, recv [W, M, rb].
    pub fn unionBlock(x: Exchange, base: u64, row_bytes: u32, phys: DevicePtr, m: u32, send: DevicePtr, recv: DevicePtr, stream: Stream) !void {
        try x.kernels.vtable.gather(x.kernels.ptr, base, row_bytes, phys, m, send, stream);
        try x.comm.allGather(send, recv, @as(usize, m) * row_bytes, .u8, stream);
    }
};
