//! Flash Next decode rounds: the DeltaNet and attn_multi tables a shared round uploads, and folding kept rows.
const std = @import("std");
const cuda = @import("cuda");
const state = @import("cuda_state.zig");
const fwd = @import("cuda_forward.zig");
const Forward = fwd.Forward;
const Seq = fwd.Seq;
const Seg = fwd.Seg;

/// attn_multi.PTRS: a stream's pointers a layer (keys, values, key scales, value scales, index keys, pooled).
pub const multi_ptrs = 6;

pub const Round = struct {
    arena: state.Arena,
    pin: cuda.HostBuffer,
    rows: usize,
    streams: usize,
    lin: usize,
    q: u64,
    kv: [2][4]u64, // per parity: k, v, g, beta of every layer ([lin][rows][...])
    sizes: [4]usize, // a layer's k, v, g, beta bytes
    sid: u64,
    win: u64,
    starts: u64,
    held: u64,
    counts: u64,
    conv: u64,
    state: u64,
    rtable: u64,
    rrows: u64,
    rcount: u64,
    host_bytes: usize,
    /// attn_multi.Step's tables: row positions, stream first positions and rows, attention cache pointers (pinned)
    att: usize,
    posr: u64,
    first: u64,
    acounts: u64,
    aptr: u64,
    apin: cuda.HostBuffer,

    pub const held_stride = state.max_rows;

    pub const Layer = struct { k: u64, v: u64, g: u64, beta: u64 };

    pub fn layer(r: *const Round, p: usize, li: usize) Layer {
        return .{ .k = r.kv[p][0] + li * r.sizes[0], .v = r.kv[p][1] + li * r.sizes[1], .g = r.kv[p][2] + li * r.sizes[2], .beta = r.kv[p][3] + li * r.sizes[3] };
    }

    pub fn init(d: *const cuda.Driver, g: state.Geometry, rows: usize, streams: usize) !Round {
        const lin = g.linear_layers;
        const sz = [4]usize{ rows * g.nk * state.gdn_dk * 4, rows * g.nv * state.gdn_dv * 2, rows * g.nv * 4, rows * g.nv * 4 };
        const tables = [_]usize{ rows * 4, rows * 16, (streams + 1) * 4, streams * held_stride * 4, streams * 4, lin * streams * 8, lin * streams * 8, 5 * lin * 8, held_stride * 4, 4 };
        var total: usize = std.mem.alignForward(usize, rows * g.nk * state.gdn_dk * 4, 256);
        for (0..2) |_| for (sz) |b| {
            total += std.mem.alignForward(usize, lin * b, 256);
        };
        for (tables) |b| total += std.mem.alignForward(usize, b, 256);
        for ([_]usize{ rows * 4, streams * 4, streams * 4, g.attention_layers * multi_ptrs * streams * 8 }) |b| total += std.mem.alignForward(usize, b, 256);
        var r: Round = undefined;
        r.arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, total) };
        errdefer r.arena.buf.free();
        try r.arena.buf.fill8(0, null);
        r.rows = rows;
        r.streams = streams;
        r.lin = lin;
        r.sizes = sz;
        r.q = r.arena.take(rows * g.nk * state.gdn_dk * 4);
        for (0..2) |p| for (0..4) |j| {
            r.kv[p][j] = r.arena.take(lin * sz[j]);
        };
        const ptrs = [_]*u64{ &r.sid, &r.win, &r.starts, &r.held, &r.counts, &r.conv, &r.state, &r.rtable, &r.rrows, &r.rcount };
        for (ptrs, tables) |q, b| q.* = r.arena.take(b);
        r.host_bytes = 0;
        for (tables) |b| r.host_bytes += std.mem.alignForward(usize, b, 64);
        r.pin = try cuda.HostBuffer.alloc(d, r.host_bytes);
        errdefer r.pin.free();
        r.att = g.attention_layers;
        const at = r.attSizes();
        r.posr = r.arena.take(at[0]);
        r.first = r.arena.take(at[1]);
        r.acounts = r.arena.take(at[2]);
        r.aptr = r.arena.take(at[3]);
        r.apin = try cuda.HostBuffer.alloc(d, attHost(at));
        return r;
    }

    /// posr, first, counts and the pointer tables' bytes.
    fn attSizes(r: *const Round) [4]usize {
        return .{ r.rows * 4, r.streams * 4, r.streams * 4, r.att * multi_ptrs * r.streams * 8 };
    }

    fn attHost(at: [4]usize) usize {
        var n: usize = 0;
        for (at) |b| n += std.mem.alignForward(usize, b, 64);
        return n;
    }

    pub fn deinit(r: *Round) void {
        r.apin.free();
        r.pin.free();
        r.arena.buf.free();
    }
};

/// The sequence's kept rows folded into its DeltaNet states now (gdn.replay in place, every layer at once).
pub fn flush(f: *Forward, seq: *Seq) !void {
    if (seq.held == 0) return;
    const r = &(f.round orelse return error.NoRoundScratch);
    const g = f.g;
    try f.staged.synchronize();
    const base = r.host_bytes - std.mem.alignForward(usize, 4, 64) - std.mem.alignForward(usize, Round.held_stride * 4, 64) - std.mem.alignForward(usize, 5 * r.lin * 8, 64);
    const tbl = std.mem.bytesAsSlice(u64, r.pin.bytes[base..][0 .. 5 * r.lin * 8]);
    for (0..r.lin) |li| {
        const l = r.layer(seq.held_par, li);
        tbl[4 * li + 0] = l.k;
        tbl[4 * li + 1] = l.v;
        tbl[4 * li + 2] = l.g;
        tbl[4 * li + 3] = l.beta;
        tbl[4 * r.lin + li] = f.rec(&seq.st, li);
    }
    const rows_at = base + std.mem.alignForward(usize, 5 * r.lin * 8, 64);
    const rows = std.mem.bytesAsSlice(i32, r.pin.bytes[rows_at..][0 .. Round.held_stride * 4]);
    for (0..seq.held) |j| rows[j] = @intCast(seq.held_a0 + j);
    const count_at = rows_at + std.mem.alignForward(usize, Round.held_stride * 4, 64);
    std.mem.bytesAsSlice(i32, r.pin.bytes[count_at..][0..4])[0] = @intCast(seq.held);
    try f.uploadRound(r.rtable, base, 5 * r.lin * 8);
    try f.uploadRound(r.rrows, rows_at, Round.held_stride * 4);
    try f.uploadRound(r.rcount, count_at, 4);
    try f.ops.gdnTreeReplay(r.rtable, r.lin, 1, r.rrows, Round.held_stride, r.rcount, 1, 0, g.nk, g.nv, state.gdn_dv);
    try f.staged.record(f.s);
    seq.held = 0;
}

/// A decode round's DeltaNet tables (Python gdn_multi.Tables) writing parity `p`; kept rows in p fold first.
pub fn roundTables(f: *Forward, segs: []const Seg, p: u1) !void {
    const r = &(f.round orelse return error.NoRoundScratch);
    const n = segs.len;
    const R = segs[n - 1].a1;
    if (n > r.streams or R > r.rows) return error.WindowPastBuffers;
    for (f.seqs.items) |q| if (q.held > 0 and q.held_par == p) try f.flush(q);
    try f.staged.synchronize();
    var at: usize = 0;
    const take = struct {
        fn next(a: *usize, bytes: usize) usize {
            const o = a.*;
            a.* += std.mem.alignForward(usize, bytes, 64);
            return o;
        }
    }.next;
    const o_sid = take(&at, r.rows * 4);
    const o_win = take(&at, r.rows * 16);
    const o_starts = take(&at, (r.streams + 1) * 4);
    const o_held = take(&at, r.streams * Round.held_stride * 4);
    const o_counts = take(&at, r.streams * 4);
    const o_conv = take(&at, r.lin * r.streams * 8);
    const o_state = take(&at, r.lin * r.streams * 8);
    const b = r.pin.bytes;
    const sid = std.mem.bytesAsSlice(i32, b[o_sid..][0 .. R * 4]);
    const win = std.mem.bytesAsSlice(i32, b[o_win..][0 .. R * 16]);
    const starts = std.mem.bytesAsSlice(i32, b[o_starts..][0 .. (n + 1) * 4]);
    const held = std.mem.bytesAsSlice(i32, b[o_held..][0 .. n * Round.held_stride * 4]);
    const counts = std.mem.bytesAsSlice(i32, b[o_counts..][0 .. n * 4]);
    const convs = std.mem.bytesAsSlice(u64, b[o_conv..][0 .. r.lin * n * 8]);
    const st_ptrs = std.mem.bytesAsSlice(u64, b[o_state..][0 .. r.lin * n * 8]);
    for (segs, 0..) |sg, i| {
        for (sg.a0..sg.a1) |row| {
            sid[row] = @intCast(i);
            for (0..4) |tap| {
                const j = row - sg.a0 + tap;
                win[row * 4 + tap] = @intCast(if (j < 3) j else sg.a0 + j);
            }
        }
        starts[i] = @intCast(sg.a0);
        const q = sg.seq;
        const folding = q.held > 0 and q.held_par != p;
        counts[i] = if (folding) @intCast(q.held) else 0;
        if (folding) for (0..q.held) |j| {
            held[i * Round.held_stride + j] = @intCast(q.held_a0 + j);
        };
        for (0..r.lin) |li| {
            convs[li * n + i] = f.conv(&q.st, li);
            st_ptrs[li * n + i] = f.rec(&q.st, li);
        }
    }
    starts[n] = @intCast(R);
    try f.uploadRound(r.sid, o_sid, R * 4);
    try f.uploadRound(r.win, o_win, R * 16);
    try f.uploadRound(r.starts, o_starts, (n + 1) * 4);
    try f.uploadRound(r.held, o_held, n * Round.held_stride * 4);
    try f.uploadRound(r.counts, o_counts, n * 4);
    try f.uploadRound(r.conv, o_conv, r.lin * n * 8);
    try f.uploadRound(r.state, o_state, r.lin * n * 8);
    // image or video streams take the per-stream launches (same bits as attn_multi's VISION variant)
    const media = for (segs) |sg| {
        if (sg.seq.media != null) break true;
    } else false;
    f.round_multi = f.attn_multi != 0 and n >= f.attn_multi and !media;
    if (f.round_multi) try f.multiTables(r, segs);
    // the kept rows ride in this round's trees: the streams hold nothing until they commit again
    for (segs) |sg| sg.seq.held = 0;
    f.cur_par = p;
    try f.staged.record(f.s);
}

/// attn_multi.Step for `segs` (positions before commit), uploaded from the round's own pinned buffer.
pub fn multiTables(f: *Forward, r: *Round, segs: []const Seg) !void {
    const n = segs.len;
    const R = segs[n - 1].a1;
    const at = r.attSizes();
    var o: [4]usize = undefined;
    var off: usize = 0;
    for (at, &o) |b, *x| {
        x.* = off;
        off += std.mem.alignForward(usize, b, 64);
    }
    const hb = r.apin.bytes;
    const posr = std.mem.bytesAsSlice(i32, hb[o[0]..][0 .. R * 4]);
    const first = std.mem.bytesAsSlice(i32, hb[o[1]..][0 .. n * 4]);
    const counts = std.mem.bytesAsSlice(i32, hb[o[2]..][0 .. n * 4]);
    const ptrs = std.mem.bytesAsSlice(u64, hb[o[3]..][0 .. r.att * multi_ptrs * n * 8]);
    for (segs, 0..) |sg, s| {
        const st = &sg.seq.st;
        for (sg.a0..sg.a1) |row| posr[row] = @intCast(st.pos + row - sg.a0);
        first[s] = @intCast(st.pos);
        counts[s] = @intCast(sg.a1 - sg.a0);
        for (0..r.att) |ai| {
            const t = ptrs[ai * multi_ptrs * n ..][0 .. multi_ptrs * n];
            const vals = [multi_ptrs]u64{ st.kc[ai], st.vc[ai], f.sc.kv_scale, f.sc.kv_scale, st.ikc[ai], st.pooled[ai] };
            for (vals, 0..) |v, j| t[j * n + s] = v;
        }
    }
    // the device tables at the round's first n rows / streams (the pointer tables a layer [6][n] apart)
    for ([_]u64{ r.posr, r.first, r.acounts, r.aptr }, o, [_]usize{ R * 4, n * 4, n * 4, r.att * multi_ptrs * n * 8 }) |dst, src, bytes|
        try f.d.check(f.d.api.cuMemcpyHtoDAsync_v2(dst, r.apin.bytes[src..].ptr, bytes, f.s.handle), "cuMemcpyHtoDAsync");
}

/// `upload` from the round tables' own pinned buffer.
pub fn uploadRound(f: *Forward, dst: u64, at: usize, bytes: usize) !void {
    if (bytes == 0) return;
    const r = &(f.round orelse return error.NoRoundScratch);
    try f.d.check(f.d.api.cuMemcpyHtoDAsync_v2(dst, r.pin.bytes[at..].ptr, bytes, f.s.handle), "cuMemcpyHtoDAsync");
}
