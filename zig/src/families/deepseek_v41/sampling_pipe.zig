//! A window's keyed choices from the ranks' logits, as the Python engine makes them (``slots.cand_gather``,
//! ``vsample.merge``, ``nucleus.finish``, ``batch._choose``): every rank runs the same steps, so its collectives pair up,
//! and every rank decides alike (the decisions read only gathered data and the sampling); rank 0's tokens are used.
//!
//! 1. Each rank's best k = min(count, V) of each row (``pick.top``: value descending, ties to the lower id), packed
//!    fp32 [n, 2k] (values, then global ids as int32 bits), all-gathered in rank order.
//! 2. Nucleus rows (T > 0, top_k 0, 0 < top_p < 1): each rank's float64 (max, sum exp(v / T - max / T)) a row
//!    (``nucleus.row_stats``), all-gathered.
//! 3. ``serve/sampling.zig``'s ``Picker`` on the merged rows: the token, or ``.full_row`` (a nucleus row its candidates
//!    cannot decide). Those rows, and every row when k exceeds the device top-k's bound, are gathered whole (each rank's
//!    vocabulary slice) and picked again from all of them (``exact_sampling.choose_rows``).
//!
//! The device side sits behind `Device` (sampling_gpu.zig: topk_keys.cu + sampling.cu; `HostDevice`: the reference
//! the host tests run, addresses being host pointers as in ``tp.host``).

const std = @import("std");
const tp = @import("tp");
const smp = @import("dsv41_serve").sampling;

pub const Sampling = smp.Sampling;

/// The device top-k's bounds (topk_keys.cu KMAX, pick.COLS): a larger k, or a wider slice (one rank), gathers whole
/// rows instead (the same tokens: the merged rows are cut to `count` either way).
pub const max_k: usize = 1024;
pub const max_cols: usize = 1 << 16;

/// The device side of the steps: `lg` is this rank's fp32 logits [n, C] (row stride C), addresses device addresses.
pub const Device = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Each row's best k (value desc, then the lower column), packed fp32 [n, 2k] at `out` with ids column + id0.
        top: *const fn (ptr: *anyopaque, lg: u64, n: usize, C: usize, k: usize, id0: u32, out: u64) anyerror!void,
        /// float64 [n, 2] at `out`: each row's (max, sum exp(v / t - max / t)).
        stats: *const fn (ptr: *anyopaque, lg: u64, n: usize, C: usize, t: f64, out: u64) anyerror!void,
        /// Scratch slot `slot` of at least `bytes` (grown, kept).
        scratch: *const fn (ptr: *anyopaque, slot: u32, bytes: usize) anyerror!u64,
        /// The device bytes at `src` into `dst`, after the stream's work.
        download: *const fn (ptr: *anyopaque, src: u64, dst: []u8) anyerror!void,
        /// The stream the collectives run on.
        stream: *const fn (ptr: *anyopaque) tp.collective.Stream,
    };
};

/// The candidates' count of a sampling (``Batcher._decode``) at `nucleus` (TF_DSV41_NUCLEUS, 0: off).
pub fn countOf(s: Sampling, vocab: u32, nucleus: u32) u32 {
    return smp.candidateCount(s, vocab, nucleus);
}

/// T > 0: a keyed choice (T <= 0 is greedy: forward.greedy).
pub fn sampled(s: ?Sampling) bool {
    return if (s) |x| x.temperature > 0 else false;
}

/// TF_DSV41_NUCLEUS (``nucleus.candidates``): candidates a nucleus row, default 512, 0 = off.
pub fn nucleusEnv() !u32 {
    const v = std.c.getenv("TF_DSV41_NUCLEUS") orelse return smp.nucleus_count;
    const s = std.mem.span(v);
    if (s.len == 0) return smp.nucleus_count;
    return std.fmt.parseInt(u32, s, 10) catch error.BadNucleus;
}

/// One rank's sampler: the host scratch, sized once for the whole vocabulary (picking allocates nothing).
pub const Pipe = struct {
    gpa: std.mem.Allocator,
    comm: tp.collective.Collective,
    /// this rank's vocabulary slice (the head's columns): global ids rank x V + column
    V: u32,
    nucleus: u32,
    picker: smp.Picker,
    values: []f32,
    ids: []u32,
    host: std.array_list.Aligned(u8, .@"16") = .empty,
    full: std.ArrayList(u32) = .empty,
    picks: std.ArrayList(smp.Pick) = .empty,
    /// rows taken whole in the last call (the nucleus fallback or k > max_k): the tests' and logs' count
    whole_rows: usize = 0,

    pub fn init(gpa: std.mem.Allocator, comm: tp.collective.Collective, V: u32, nucleus: u32) !Pipe {
        const vocab: usize = @as(usize, V) * comm.world();
        var picker = try smp.Picker.init(gpa, vocab);
        errdefer picker.deinit();
        const values = try gpa.alloc(f32, vocab);
        errdefer gpa.free(values);
        return .{ .gpa = gpa, .comm = comm, .V = V, .nucleus = nucleus, .picker = picker, .values = values, .ids = try gpa.alloc(u32, vocab) };
    }

    pub fn deinit(p: *Pipe) void {
        p.picker.deinit();
        p.gpa.free(p.values);
        p.gpa.free(p.ids);
        p.host.deinit(p.gpa);
        p.full.deinit(p.gpa);
        p.picks.deinit(p.gpa);
    }

    fn hostBytes(p: *Pipe, bytes: usize) ![]align(16) u8 {
        try p.host.resize(p.gpa, bytes);
        return p.host.items;
    }

    /// The keyed choices of rows [0, n) of `lg` (fp32 [n, V]) under `s` (T > 0), row i keyed at `positions[i]`
    /// (followers: null, their tokens are dropped). Every rank calls it with the same n and sampling.
    pub fn choose(p: *Pipe, dev: Device, lg: u64, n: usize, s: Sampling, positions: ?[]const u64, out: []u32) !void {
        std.debug.assert(out.len >= n and n > 0);
        const W: usize = p.comm.world();
        const V: usize = p.V;
        const count: usize = countOf(s, @intCast(V * W), p.nucleus);
        const k = @min(count, V);
        p.whole_rows = 0;
        try p.picks.resize(p.gpa, n);
        if (k > max_k or V > max_cols) {
            // every row whole (top_k 0 without the nucleus path, a top_k past the device bound, or one rank's 129,280)
            p.full.clearRetainingCapacity();
            for (0..n) |r| try p.full.append(p.gpa, @intCast(r));
            return p.wholeRows(dev, lg, s, count, positions, out);
        }
        const nuc = smp.isNucleus(s) and p.nucleus > 0;
        const vt = dev.vtable;
        const stream = vt.stream(dev.ptr);
        const row_f32 = 2 * k;
        const packed_ = try vt.scratch(dev.ptr, 0, 4 * n * row_f32);
        const gathered = try vt.scratch(dev.ptr, 1, 4 * W * n * row_f32);
        try vt.top(dev.ptr, lg, n, V, k, @intCast(p.comm.rank() * V), packed_);
        try p.comm.allGather(packed_, gathered, n * row_f32, .f32, stream);
        var stats_dev: u64 = 0;
        if (nuc) {
            const st = try vt.scratch(dev.ptr, 2, 16 * n);
            stats_dev = try vt.scratch(dev.ptr, 3, 16 * W * n);
            try vt.stats(dev.ptr, lg, n, V, s.temperature, st);
            try p.comm.allGather(st, stats_dev, 2 * n, .f64, stream);
        }
        const cand_bytes = 4 * W * n * row_f32;
        const stat_bytes: usize = if (nuc) 16 * W * n else 0;
        const host = try p.hostBytes(cand_bytes + stat_bytes);
        try vt.download(dev.ptr, gathered, host[0..cand_bytes]);
        if (nuc) try vt.download(dev.ptr, stats_dev, host[cand_bytes..][0..stat_bytes]);
        const cand: smp.Candidates = .{ .data = @alignCast(std.mem.bytesAsSlice(f32, host[0..cand_bytes])), .world = @intCast(W), .rows = @intCast(n), .k = @intCast(k) };
        const all_stats: []const f64 = if (nuc) @alignCast(std.mem.bytesAsSlice(f64, host[cand_bytes..][0..stat_bytes])) else &.{};
        var stats: [tp.host.max_ranks][2]f64 = undefined;
        p.full.clearRetainingCapacity();
        for (0..n) |r| {
            const m = cand.row(@intCast(r), p.values, p.ids);
            // rank w's (m, s) of row r: [W][n][2]
            if (nuc) for (stats[0..W], 0..) |*st, w| {
                st.* = .{ all_stats[2 * (w * n + r)], all_stats[2 * (w * n + r) + 1] };
            };
            const row: smp.Row = .{ .values = p.values[0..m], .ids = p.ids[0..m], .stats = if (nuc) stats[0..W] else &.{}, .count = count };
            switch (p.picker.pick(row, if (positions) |ps| ps[r] else 0, s)) {
                .token => |t| out[r] = t,
                .full_row => try p.full.append(p.gpa, @intCast(r)),
            }
        }
        if (p.full.items.len > 0) try p.wholeRows(dev, lg, s, 0, positions, out);
    }

    /// The rows in `full`, each gathered whole (every rank's vocabulary slice, rank order) and picked from all of its
    /// tokens (``count`` = their cut; 0: all of them, the nucleus fallback's ``fw.vocab``).
    fn wholeRows(p: *Pipe, dev: Device, lg: u64, s: Sampling, count: usize, positions: ?[]const u64, out: []u32) !void {
        const W: usize = p.comm.world();
        const V: usize = p.V;
        const vt = dev.vtable;
        const recv = try vt.scratch(dev.ptr, 4, 4 * W * V);
        const host = try p.hostBytes(4 * W * V);
        const vals: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, host));
        for (p.ids[0 .. W * V], 0..) |*id, i| id.* = @intCast(i); // rank order: global id = the index
        for (p.full.items) |r| {
            try p.comm.allGather(lg + 4 * @as(u64, r) * V, recv, V, .f32, vt.stream(dev.ptr));
            try vt.download(dev.ptr, recv, host);
            @memcpy(p.values[0 .. W * V], vals);
            const row: smp.Row = .{ .values = p.values[0 .. W * V], .ids = p.ids[0 .. W * V], .count = count };
            const t = p.picker.pick(row, if (positions) |ps| ps[r] else 0, s);
            out[r] = t.token; // no statistics: never .full_row
            p.whole_rows += 1;
        }
    }
};

/// The reference device: `lg` and every address are host pointers (``tp.host``'s ranks share one address space).
pub const HostDevice = struct {
    gpa: std.mem.Allocator,
    bufs: [8][]align(16) u8 = @splat(&.{}),
    keys: std.ArrayList(u64) = .empty,

    pub fn deinit(h: *HostDevice) void {
        for (h.bufs) |b| if (b.len > 0) h.gpa.free(b);
        h.keys.deinit(h.gpa);
    }

    pub fn device(h: *HostDevice) Device {
        return .{ .ptr = h, .vtable = &.{ .top = top, .stats = stats, .scratch = scratch, .download = download, .stream = stream } };
    }

    fn self_(p: *anyopaque) *HostDevice {
        return @ptrCast(@alignCast(p));
    }

    fn f32s(a: u64, n: usize) []f32 {
        return @as([*]f32, @ptrFromInt(a))[0..n];
    }

    /// ``pick.keys``' order on any fp32 values: larger value (-0.0 as +0.0), then the lower column.
    fn key(v: f32, col: usize) u64 {
        const bits: u32 = @bitCast(v + 0.0);
        const mono: u32 = if (bits & 0x8000_0000 == 0) bits | 0x8000_0000 else ~bits;
        return @as(u64, mono) << 32 | (0xFFFF_FFFF - @as(u32, @intCast(col)));
    }

    fn top(p: *anyopaque, lg: u64, n: usize, C: usize, k: usize, id0: u32, out: u64) anyerror!void {
        const h = self_(p);
        try h.keys.resize(h.gpa, C);
        const o = f32s(out, n * 2 * k);
        for (0..n) |r| {
            const row = f32s(lg + 4 * r * C, C);
            for (h.keys.items, row, 0..) |*kk, v, c| kk.* = key(v, c);
            std.sort.pdq(u64, h.keys.items, {}, std.sort.desc(u64));
            for (h.keys.items[0..k], 0..) |kk, j| {
                const c = 0xFFFF_FFFF - @as(u32, @truncate(kk));
                o[r * 2 * k + j] = row[c];
                o[r * 2 * k + k + j] = @bitCast(c + id0);
            }
        }
    }

    fn stats(_: *anyopaque, lg: u64, n: usize, C: usize, t: f64, out: u64) anyerror!void {
        const o = @as([*]f64, @ptrFromInt(out))[0 .. 2 * n];
        for (0..n) |r| {
            const row = f32s(lg + 4 * r * C, C);
            var m = -std.math.inf(f64);
            for (row) |v| m = @max(m, @as(f64, v));
            const mt = m / t;
            var s: f64 = 0;
            for (row) |v| s += @exp(@as(f64, v) / t - mt);
            o[2 * r] = m;
            o[2 * r + 1] = s;
        }
    }

    fn scratch(p: *anyopaque, slot: u32, bytes: usize) anyerror!u64 {
        const h = self_(p);
        if (h.bufs[slot].len < bytes) {
            if (h.bufs[slot].len > 0) h.gpa.free(h.bufs[slot]);
            h.bufs[slot] = try h.gpa.alignedAlloc(u8, .@"16", bytes);
        }
        return @intFromPtr(h.bufs[slot].ptr);
    }

    fn download(_: *anyopaque, src: u64, dst: []u8) anyerror!void {
        @memcpy(dst, @as([*]const u8, @ptrFromInt(src))[0..dst.len]);
    }

    fn stream(_: *anyopaque) tp.collective.Stream {
        return null;
    }
};
