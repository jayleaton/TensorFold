//! The forward's own prefill: a prompt in segments of `opts.prefill_rows` (2,048, Python's pchunk) through
//! block_prefill.zig's program, run eagerly by run.zig with this file's glue handler, each segment then committed as a
//! keep of all its rows. Python's Forward.prompt: the head (and kit_logits) only on the last segment.
//!
//! - A segment of at most 16 rows (the prompt's ragged tail, or a whole short prompt) is block_prefill's short segment:
//!   Python's mix of prefill projections and the decode kernels' row paths, committed the same way.
//! - The commit (keep's equivalent, Python's Forward.keep(slot, n - 1)): each ratio-2 KV source's carry is its fp32
//!   projection's last row ("w.L<i>.proj", the f32 of the bf16 rows, so the same bits as decode's carry kernel), the
//!   Engram tail is (tail + the segment's ids)[-(max_ngram - 1):], the position moves by the rows, nothing pending.
//!   The rings, pools and index keys the segment's kernels already wrote.
//! - After a prefill the last step's logits are in "w.logits", `Forward.last_rows` rows of them (the last segment's):
//!   greedy takes `out[0..last_rows]` and the caller reads the last row.
//! - Run once before the first segment: the persistent tables no loader form builds, the prefill GEMM's Hadamard
//!   ("s.pf.had", exl3.prefill's bf16 +-1) and x3gm's pointer tables ("s.L<i>.gm.t{g,u,d}": the routed experts'
//!   trellis addresses, the prefix of forms' "s.L<i>.ex.tp_*" before the shared expert's), and the exchange / Engram
//!   buffers grown to a segment (so a graphed decode window captured after it keeps its addresses).
//! - Long prompts (longpf.zig): past TF_DSV41_INDEX_STREAM_MIN visible keys the full-mode layers select with the
//!   stream top-k (glue "stream_merge"); under split KV a union past the cap runs its attentions in row blocks
//!   (longpf.window).
//! - CED replay (TF_DSV41_PREFILL=replay, ced.zig): each segment is the encoder pass (block_prefill.emitReplay
//!   `.encoder`, no head) whose glue "ced_stash" keeps its last rows entering the decoder in the ring; `finish` then
//!   runs the decoder over the prompt's tail from the ring (glue "ced_load", windows from R0) and commits nothing.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("calls.zig");
const buffers = @import("buffers.zig");
const run = @import("run.zig");
const block_prefill = @import("block_prefill.zig");
const longpf = @import("longpf.zig");
const vision_rows = @import("vision_rows.zig");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const ced = @import("ced.zig");
const prod_knobs = @import("prod_knobs.zig");
const Forward = fwd.Forward;

const log = std.log.scoped(.dsv41);

pub const Error = error{ NoExpertTable, NoK2Table, BadGlue, StashGap };

/// The prefill's own state on the forward (forward.zig: `prefill_state`).
pub const State = struct {
    /// the persistent tables and grown buffers are in place (ready())
    ready: bool = false,
    /// the prompt's own copy (a caller's slice may alias the window's ids)
    ids: std.ArrayList(u32) = .empty,
    /// gm_plan's width-masked picks (int32 [P])
    masked: ?cuda.DeviceBuffer = null,
    /// the segment's layers and the MoE blocks seen: gm_ticket starts each layer's MoE, gm_plan reads its K2 table
    layers: []const u32 = &.{},
    moe: usize = 0,
    layer: u32 = 0,
    /// TF_DSV41_PREFETCH_AHEAD (prod_knobs.zig): the next segment's Engram reads issued before this one runs
    ahead: bool = false,
    /// long prompts' buffers (longpf.zig)
    long: longpf.State = .{},
    /// TF_DSV41_PREFILL (prod_knobs.apply): `.replay` = CED (ced.zig), segments are encoder passes and `finish` runs
    /// the decoder; `.full`: every layer over every row
    mode: prod_knobs.PrefillMode = .full,
    /// the decoder replay's ids (the glue reads them through Forward.ids while it runs)
    tail_ids: [ced.keep]u32 = undefined,
    /// what "w.taps" holds after the prompt: positions [taps_at, taps_at + taps_rows) at rows 0 .. (the last full
    /// segment's, or the decoder replay's; an encoder segment writes none): the DSpark hand-off (ced.handoff)
    taps_at: u64 = 0,
    taps_rows: u32 = 0,

    pub fn deinit(s: *State, gpa: std.mem.Allocator) void {
        s.long.deinit(gpa);
        if (s.masked) |*b| b.free();
        s.ids.deinit(gpa);
    }
};

/// Adds the prefill program's roles at `rows` (the largest segment) to `p`: at start 0 and at the last start its
/// emitter takes (the indexer's scores grow with the keys), a 17-row segment's (the small-segment paths: the Triton
/// mHC site, the out-of-place boundary streams) and a 16-row short segment's (the decode kernels' roles).
pub fn plan(f: *Forward, p: *buffers.Plan, rows: u32) !void {
    if (rows <= 16) return error.BadWindow;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    const layers = try f.backbone(la.allocator());
    const sizes = [_]u32{ rows, 17, 16 };
    for (sizes[if (rows > 17) 0 else 1..]) |n| {
        const top = try lastStart(f, layers, n, .whole);
        for ([_]i64{ 0, top }) |start| {
            var ea = std.heap.ArenaAllocator.init(f.gpa);
            defer ea.deinit();
            try p.add(try block_prefill.emitPrefill(ea.allocator(), f.cfg, f.widths, vision_rows.planOptions(f), layers, n, start, true));
        }
    }
    if (f.prefill_state.mode == .replay) try planReplay(f, p, rows, layers);
}

/// CED replay's programs (ced.zig): an encoder segment (the whole segment's roles up to the decoder's first layer, and
/// the stash ring), and the decoder pass at the row counts where its paths change (a short segment up to 16 rows, the
/// selection's attn_cuda top-k up to 64, the materialised selection past it, the replay's 127), at 0 and its last start.
fn planReplay(f: *Forward, p: *buffers.Plan, rows: u32, layers: []const u32) !void {
    const enc = try encoderLayers(f, layers);
    const dec = try decoderLayers(f, layers);
    var ea = std.heap.ArenaAllocator.init(f.gpa);
    defer ea.deinit();
    try p.add(try block_prefill.emitReplay(ea.allocator(), f.cfg, f.widths, vision_rows.planOptions(f), enc, rows, 0, .encoder));
    for ([_]u32{ 16, 17, 32, 33, 64, 65, ced.keep }) |m| {
        if (m > rows) continue;
        const top = try lastStart(f, dec, m, .decoder);
        for ([_]i64{ 0, top }) |start| {
            var da = std.heap.ArenaAllocator.init(f.gpa);
            defer da.deinit();
            try p.add(try block_prefill.emitReplay(da.allocator(), f.cfg, f.widths, vision_rows.planOptions(f), dec, m, start, .decoder));
        }
    }
}

/// The encoder pass's layers: the forward's up to the decoder's first (ced.decoderStart), inclusive.
fn encoderLayers(f: *Forward, layers: []const u32) ![]const u32 {
    const d = try ced.decoderStart(f.cfg);
    const i = std.mem.indexOfScalar(u32, layers, d) orelse return error.NoDecoder;
    return layers[0 .. i + 1];
}

/// The decoder pass's layers: the forward's from the decoder's first on.
fn decoderLayers(f: *Forward, layers: []const u32) ![]const u32 {
    const d = try ced.decoderStart(f.cfg);
    const i = std.mem.indexOfScalar(u32, layers, d) orelse return error.NoDecoder;
    return layers[i..];
}

/// Whether an `n`-row segment (`part` of it) at `start` is on the prefill path (block_prefill's emitter takes it).
fn emits(f: *Forward, layers: []const u32, n: u32, start: i64, part: block_prefill.Part) !bool {
    var ea = std.heap.ArenaAllocator.init(f.gpa);
    defer ea.deinit();
    _ = block_prefill.emitReplay(ea.allocator(), f.cfg, f.widths, f.opts, layers, n, start, part) catch |e| switch (e) {
        error.Unsupported => return false,
        else => return e,
    };
    return true;
}

/// The last start an `n`-row segment emits at (the stream top-k's threshold moves with the segment's end). The decoder
/// pass ends a row before the prompt's last token: its last start is one lower.
fn lastStart(f: *Forward, layers: []const u32, n: u32, part: block_prefill.Part) !i64 {
    var hi: i64 = f.opts.limit - n - @intFromBool(part == .decoder);
    if (hi < 0) return error.BadWindow;
    if (try emits(f, layers, n, hi, part)) return hi;
    if (!try emits(f, layers, n, 0, part)) {
        log.warn("prefill: a {d}-row segment is not on the prefill path at any position (block_prefill: Unsupported)", .{n});
        return error.Unsupported;
    }
    var lo: i64 = 0;
    while (hi - lo > 1) {
        const mid = lo + @divFloor(hi - lo, 2);
        if (try emits(f, layers, n, mid, part)) lo = mid else hi = mid;
    }
    log.info("prefill: {d}-row segments emit up to start {d}", .{ n, lo });
    return lo;
}

/// The prompt `ids` from the slot's position (forward.prefill: the leader has sent it to the followers).
pub fn prompt(f: *Forward, ids_in: []const u32) !void {
    const st = &f.prefill_state;
    if (f.slot.pending != null) return error.BadWindow;
    if (ids_in.len == 0) return;
    if (f.slot.pos + ids_in.len > @as(u64, @intCast(f.opts.limit))) return error.BadWindow;
    if (ids_in.ptr != st.ids.items.ptr) {
        st.ids.clearRetainingCapacity();
        try st.ids.appendSlice(f.gpa, ids_in);
    }
    const ids = st.ids.items;
    st.taps_rows = 0;
    try ready(f);
    const seg: usize = @intCast(f.opts.prefill_rows);
    var at: usize = 0;
    while (at < ids.len) {
        const n = @min(seg, ids.len - at);
        if (st.ahead and at + n < ids.len) readAhead(f, ids, at, n, @min(seg, ids.len - at - n));
        try segment(f, ids[at .. at + n], at + n == ids.len and st.mode == .full);
        at += n;
    }
}

/// prefetch.py's one segment ahead: the reads of the segment after `ids[at .. at + n]` (its `next` rows, with the
/// lookback they will see: the slot's tail and this segment's ids) start before this segment runs, so they are in
/// flight while it does; its glue then waits for them (engram_rows keeps an unread bulk batch). A wrong guess costs a
/// read, never a bit: the rows are the table's whatever the path.
fn readAhead(f: *Forward, ids: []const u32, at: usize, n: usize, next: usize) void {
    const h = f.engram orelse return;
    if (!f.engram_prefetch) return;
    const src = (f.engramSource(h) catch return) orelse return;
    const keep: usize = h.max_ngram - 1;
    const b = at + n;
    var buf: [2 * eh.max_ngram]u32 = undefined;
    const lb: []const u32 = if (n >= keep) ids[b - keep .. b] else blk: {
        const old = @min(f.slot.tail_len, keep - n);
        @memcpy(buf[0..old], f.slot.tail[f.slot.tail_len - old .. f.slot.tail_len]);
        @memcpy(buf[old .. old + n], ids[at..b]);
        break :blk buf[0 .. old + n];
    };
    h.prefetch(f.gpa, src, ids[b .. b + next], lb, f.comm.rank(), f.comm.world());
}

/// One segment at the slot's position, run eagerly, then committed.
fn segment(f: *Forward, ids: []const u32, head: bool) !void {
    const st = &f.prefill_state;
    const n: u32 = @intCast(ids.len);
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    // CED replay: the encoder pass (layers up to the decoder's first, its site and compressor, the stash; no head)
    const replay = st.mode == .replay;
    const layers = if (replay) try encoderLayers(f, try f.backbone(a)) else try f.backbone(a);
    const start = f.slot.pos;
    if (f.kv) |kx| try kx.reserve(start + n);
    const o = vision_rows.segmentOptions(f, ids);
    const emitted = if (replay) block_prefill.emitReplay(a, f.cfg, f.widths, o, layers, n, @intCast(start), .encoder) else block_prefill.emitPrefill(a, f.cfg, f.widths, o, layers, n, @intCast(start), head);
    const cs = emitted catch |e| {
        if (e == error.Unsupported) log.warn("prefill: the {d}-row segment at {d} is not on the prefill path (block_prefill: Unsupported)", .{ n, start });
        return e;
    };
    // the glue reads the ids (embedding, Engram) and the pending window (positions)
    f.ids = ids;
    f.slot.pending = .{ .start = start, .n = n };
    errdefer f.slot.pending = null;
    st.layers = layers;
    st.moe = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, start, cs);
    try commit(f, n);
    // replay.encode: the stash's ids follow its rows (a segment that does not continue it starts it over)
    if (replay) f.slot.ced.extend(start, ids);
    f.last_rows = if (replay) 0 else n;
    st.taps_at = start;
    st.taps_rows = if (replay) 0 else n;
}

/// CED's decoder replay (replay.finish; Forward.finishPrompt on every rank): layers D .. over the prompt's prefilled
/// tail [R0, end) from the stash, end = the slot's position, every window from R0. Nothing is committed: the slot
/// stays at `end`, the decoder's SWA rings hold [R0, end), and the prompt's last token is the next window's row.
pub fn finish(f: *Forward) !void {
    const st = &f.prefill_state;
    if (st.mode != .replay) return;
    if (f.slot.pending != null) return error.BadWindow;
    const end = f.slot.pos;
    const t = ced.tailOf(end);
    if (t.m == 0) return;
    if (!f.slot.ced.covers(t.r0, end)) {
        log.err("CED replay of [{d}, {d}), but the stash holds [{d}, {d})", .{ t.r0, end, f.slot.ced.lo(), f.slot.ced.hi });
        return error.StashGap;
    }
    try ready(f);
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const layers = try decoderLayers(f, try f.backbone(a));
    const ids = st.tail_ids[0..t.m];
    f.slot.ced.tail(t.r0, end, ids);
    // the tail's image positions route with gate.bias_vl (replay._image); no decoder layer reads Engram
    const cs = try block_prefill.emitReplay(a, f.cfg, f.widths, vision_rows.segmentOptions(f, ids), layers, t.m, @intCast(t.r0), .decoder);
    f.ids = ids;
    f.slot.pending = .{ .start = t.r0, .n = t.m };
    defer f.slot.pending = null;
    st.layers = layers;
    st.moe = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, t.r0, cs);
    f.last_rows = 0;
    // the drafter's context rows: the tail's taps (replay.finish's, which slots.finish_prompt hands to `replayed`)
    st.taps_at = t.r0;
    st.taps_rows = t.m;
}

/// ced_stash (`store`): the segment's last min(n, ring) rows of each tensor (the streams, the normed input, the
/// coefficients) into the ring, position p at row p % ring; ced_load: the ring's rows of the pass's positions into the
/// tensors' rows 0 .. n.
fn stashCopy(f: *Forward, r: *run.Runner, c: *const calls.Call, store: bool) !void {
    const p = f.slot.pending orelse return error.BadGlue;
    const x = c.args;
    if (x.len != 2 * ced.roles.len) return error.BadGlue;
    const k: u64 = @min(p.n, ced.ring);
    const a0: u64 = p.start + p.n - k;
    var sp: [2]ced.Span = undefined;
    for (0..ced.roles.len) |j| {
        const t = x[j].arg.t;
        const rg = x[ced.roles.len + j].arg.t;
        const row = dim(t, 1) * buffers.dtSize(t.dt);
        if (dim(rg, 1) * buffers.dtSize(rg.dt) != row or ld(t) != dim(t, 1) or dim(rg, 0) != ced.ring) return error.BadGlue;
        const ta = try r.tensorAddr(t);
        const ra = try r.tensorAddr(rg);
        for (ced.spans(a0, a0 + k, &sp)) |s| {
            const tr = ta + (a0 - p.start + s.at) * row;
            const rr = ra + @as(u64, s.row) * row;
            if (store) try copy(r, rr, tr, s.n * row) else try copy(r, tr, rr, s.n * row);
        }
    }
}

/// Keep of all the pending segment's rows: the ratio-2 carries, the Engram tail, the position.
fn commit(f: *Forward, n: u32) !void {
    const p = f.slot.pending orelse return error.BadWindow;
    const r = f.runner;
    var nb: [64]u8 = undefined;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    const row: usize = 4 * 2 * @as(usize, f.cfg.head_dim); // fp32 [kv | gate]
    for (try f.backbone(la.allocator())) |L| if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2) {
        const src = r.addressOf(try std.fmt.bufPrint(&nb, "w.L{d}.proj", .{L})) orelse return error.Unbound;
        const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L})) orelse return error.Unbound;
        try copy(r, dst, src + (n - 1) * row, row);
    };
    if (f.engram) |h| tailOf(&f.slot, f.ids[0..n], h.max_ngram - 1);
    f.slot.pos = p.start + n;
    f.slot.pending = null;
    if (f.sess) |x| try x.commit(x.ptr, f.ids[0..n]);
}

/// Engram's tail after committing `ids`: (tail + ids)[-keep:].
fn tailOf(s: *fwd.Slot, ids: []const u32, keep: usize) void {
    if (ids.len >= keep) {
        @memcpy(s.tail[0..keep], ids[ids.len - keep ..]);
        s.tail_len = keep;
        return;
    }
    const old = @min(s.tail_len, keep - ids.len); // the old tail's last ones that stay
    std.mem.copyForwards(u32, s.tail[0..old], s.tail[s.tail_len - old .. s.tail_len]);
    @memcpy(s.tail[old .. old + ids.len], ids);
    s.tail_len = old + ids.len;
}

/// Once: the tables no loader form builds and the buffers a segment's exchanges / Engram rows need.
fn ready(f: *Forward) !void {
    const st = &f.prefill_state;
    if (st.ready) return;
    const r = f.runner;
    // exl3.prefill's Hadamard: H[i, j] = (-1)^popcount(i & j), bf16
    if (r.addressOf("s.pf.had")) |p| {
        const h = try f.gpa.alloc(u16, 128 * 128);
        defer f.gpa.free(h);
        for (0..128) |i| for (0..128) |j| {
            h[i * 128 + j] = if (@popCount(i & j) & 1 == 1) 0xBF80 else 0x3F80;
        };
        try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = p, .len = h.len * 2 }, 0, std.mem.sliceAsBytes(h));
    }
    // x3gm's pointer tables: the routed experts' trellis addresses (forms' tables hold the shared expert's last)
    var nb: [64]u8 = undefined;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    for (try f.backbone(la.allocator())) |L| for ([_][2][]const u8{ .{ "tg", "tp_g" }, .{ "tu", "tp_u" }, .{ "td", "tp_d" } }) |t| {
        const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.gm.{s}", .{ L, t[0] })) orelse continue;
        const src = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.ex.{s}", .{ L, t[1] })) orelse return error.NoExpertTable;
        try copy(r, dst, src, 8 * @as(usize, f.cfg.expertsOf(L).count));
    };
    // x3gm v2's width tables (gm2pf.zig, x3gm.Ragged.k2g / k2d): the routed experts' K2s, then a 0 sentinel
    for (try f.backbone(la.allocator())) |L| for ([_][2][]const u8{ .{ "k2g", "k2_g" }, .{ "k2d", "k2_d" } }) |t| {
        const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.gm.{s}", .{ L, t[0] })) orelse continue;
        const src = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.ex.{s}", .{ L, t[1] })) orelse return error.NoK2Table;
        const Et: usize = f.cfg.expertsOf(L).count;
        try copy(r, dst, src, 4 * Et);
        try zero(r, dst + 4 * Et, 4);
    };
    // a segment's exchange send and Engram rows at their largest (decode's would grow them on the first segment)
    const rows: usize = @intCast(f.opts.prefill_rows);
    const W: usize = f.comm.world();
    _ = try f.sendBuffer(r, 2 * rows * f.cfg.hidden);
    if (f.engram) |h| {
        const cols = h.headShard(f.comm.rank(), f.comm.world())[1] * @as(usize, f.cfg.engram_head_dim);
        _ = try grow(&f.gathered, r.d, W * 2 * rows * cols);
        for (0..h.layer_ids.len) |li| _ = try f.rows_stage[li].take(r.d, 2 * rows * cols);
    }
    _ = try f.ids_stage.take(r.d, 8 * rows);
    try r.stream.synchronize();
    st.ready = true;
}

fn grow(b: *?cuda.DeviceBuffer, d: *const cuda.Driver, bytes: usize) !u64 {
    if (b.* == null or b.*.?.len < bytes) {
        if (b.*) |*x| x.free();
        b.* = try cuda.DeviceBuffer.alloc(d, @max(bytes, 256));
    }
    return b.*.?.ptr;
}

fn copy(r: *run.Runner, dst: u64, src: u64, bytes: usize) !void {
    if (bytes == 0) return;
    try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, r.stream.handle), "cuMemcpyDtoDAsync");
}

fn zero(r: *run.Runner, dst: u64, bytes: usize) !void {
    if (bytes == 0) return;
    try r.d.check(r.d.api.cuMemsetD8Async(dst, 0, bytes, r.stream.handle), "cuMemsetD8Async");
}

fn numel(t: calls.Tensor) usize {
    var n: usize = 1;
    for (t.shape) |x| n *= @intCast(x);
    return n;
}

fn bytesOf(t: calls.Tensor) usize {
    return numel(t) * buffers.dtSize(t.dt);
}

fn dim(t: calls.Tensor, i: usize) usize {
    return @intCast(t.shape[i]);
}

fn ld(t: calls.Tensor) usize {
    return @intCast(t.stride[0]);
}

/// A prefill segment's glue steps (block_prefill.zig's module doc); the ones a decode window has too (positions,
/// embed, exchange, kit_logits, engram_rows, trace, and the materialised selection's topk / counts / block_pos) are the
/// decode glue's.
fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
    const f: *Forward = @ptrCast(@alignCast(ctx));
    const st = &f.prefill_state;
    const step = c.name["glue.".len..];
    const k = r.kernels.others(r.stream);
    const x = c.args;
    const eql = std.mem.eql;
    if (std.mem.startsWith(u8, step, "kx_")) return (f.kv orelse return error.BadGlue).glue(r, c, step);
    if (eql(u8, step, "proj_carry")) {
        // torch.cat([the slot's carry, the projection]): position start - 1's row, then the segment's
        const carry = x[0].arg.t;
        const buf = try r.tensorAddr(x[2].arg.t);
        try copy(r, buf, try r.tensorAddr(carry), bytesOf(carry));
        return copy(r, buf + bytesOf(carry), try r.tensorAddr(x[1].arg.t), bytesOf(x[1].arg.t));
    }
    if (eql(u8, step, "stream_merge")) return longpf.merge(&st.long, f.gpa, r, c);
    if (eql(u8, step, "ced_stash") or eql(u8, step, "ced_load")) return stashCopy(f, r, c, eql(u8, step, "ced_stash"));
    if (eql(u8, step, "zeros")) return zero(r, try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t));
    if (eql(u8, step, "image_keep")) return vision_rows.keep(f, r, c);
    if (eql(u8, step, "image_split") or eql(u8, step, "image_merge")) return vision_rows.splitRows(f, r, c, eql(u8, step, "image_merge"));
    if (eql(u8, step, "gm_ticket_again")) return zero(r, try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t)); // vision.moe's text call: the same layer
    if (eql(u8, step, "cand_keys")) {
        const blocks = x[0].arg.t;
        return k.candKeys(try r.tensorAddr(blocks), ld(blocks), dim(blocks, 0), dim(blocks, 1), @intCast(x[2].arg.i), try r.tensorAddr(x[1].arg.t));
    }
    if (eql(u8, step, "proj")) {
        // blocks._projection: f32(kv), or ratio 2 [f32(kv) | f32(gate)]
        const kv = x[0].arg.t;
        const gate: u64 = if (x.len == 3) try r.tensorAddr(x[1].arg.t) else 0;
        return k.widenCat(try r.tensorAddr(kv), gate, dim(kv, 0), dim(kv, 1), try r.tensorAddr(x[x.len - 1].arg.t));
    }
    if (eql(u8, step, "ix_w")) return k.f64Bf16F32(try r.tensorAddr(x[0].arg.t), try r.tensorAddr(x[1].arg.t), numel(x[0].arg.t));
    if (eql(u8, step, "stage_in") or eql(u8, step, "stage_out")) {
        // blocks.stage_rows(dst, dst_ring, src, src_ring, lo, hi), a tensor at a time (values, scales)
        const dst_ring: usize = @intCast(x[2].arg.i);
        const src_ring: usize = @intCast(x[5].arg.i);
        const lo: u64 = @intCast(x[6].arg.i);
        const hi: u64 = @intCast(x[7].arg.i);
        for (0..2) |j| {
            const dst = x[j].arg.t;
            const src = x[3 + j].arg.t;
            const row = dim(dst, 1) * buffers.dtSize(dst.dt);
            if (dim(src, 1) * buffers.dtSize(src.dt) != row) return error.BadGlue;
            try k.ringCopy(try r.tensorAddr(src), src_ring, try r.tensorAddr(dst), dst_ring, row, lo, hi);
        }
        return;
    }
    if (eql(u8, step, "gm_picks")) {
        const pick = x[0].arg.t;
        return k.gmPicks(try r.tensorAddr(pick), try r.tensorAddr(x[1].arg.t), dim(pick, 1), dim(x[2].arg.t, 1), dim(pick, 0), try r.tensorAddr(x[2].arg.t), try r.tensorAddr(x[3].arg.t));
    }
    if (eql(u8, step, "gm_ticket")) {
        // a layer's MoE starts here: the layer gm_plan's K2 tables are of
        if (st.moe >= st.layers.len) return error.BadGlue;
        st.layer = st.layers[st.moe];
        st.moe += 1;
        return zero(r, try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t));
    }
    if (eql(u8, step, "gm_plan")) {
        // x3gm._run_ragged: the picks of one width (the others past the table), then x3gm.plan's passes
        const pk = x[0].arg.t;
        const P = numel(pk);
        const k2: u32 = @intCast(x[6].arg.i);
        const down = x[7].arg.b;
        const bm: usize = @intCast(x[8].arg.i);
        const Et: usize = @intCast(x[9].arg.i);
        // k2 0 (x3gm._run_v2's one plan, gm2pf.zig): every pick, no width mask
        if (k2 == 0) return r.kernels.experts(r.stream).gmPlan(try r.tensorAddr(pk), P, Et, bm, try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[2].arg.t), try r.tensorAddr(x[3].arg.t), try r.tensorAddr(x[4].arg.t), try r.tensorAddr(x[5].arg.t));
        var nb: [64]u8 = undefined;
        // forms' "s.L<i>.ex.k2_{g,d}": each routed expert's K2 (gate = up), the shared expert's past Et (never read)
        const tab = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.ex.k2_{s}", .{ st.layer, if (down) "d" else "g" })) orelse return error.NoK2Table;
        const masked = try grow(&st.masked, r.d, 4 * P);
        try k.widthMask(try r.tensorAddr(pk), P, tab, Et, k2, masked);
        const t = [5]u64{ try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[2].arg.t), try r.tensorAddr(x[3].arg.t), try r.tensorAddr(x[4].arg.t), try r.tensorAddr(x[5].arg.t) };
        return r.kernels.experts(r.stream).gmPlan(masked, P, Et, bm, t[0], t[1], t[2], t[3], t[4]);
    }
    if (eql(u8, step, "swiglu")) {
        // the shared expert's silu(min(g, limit)) * clamp(u, +-limit), fp32
        const g = x[0].arg.t;
        const u = x[1].arg.t;
        const act = x[2].arg.t;
        return k.swiglu(try r.tensorAddr(g), ld(g), try r.tensorAddr(u), ld(u), try r.tensorAddr(act), ld(act), dim(g, 0), dim(g, 1), @floatCast(x[3].arg.f));
    }
    if (eql(u8, step, "moe_sum")) {
        // bf16(routed + shared): one RNE of the fp32 sum; no shared expert: the routed partial's cast
        const routed = x[0].arg.t;
        if (x.len == 3) return k.addBf16(try r.tensorAddr(routed), try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[2].arg.t), numel(routed));
        return k.castBf16(try r.tensorAddr(routed), try r.tensorAddr(x[1].arg.t), numel(routed));
    }
    return Forward.glueFn(ctx, r, c);
}
