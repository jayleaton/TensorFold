//! Flash Next sequences: reset, images and video, cache growth, staging windows and commit.
const std = @import("std");
const cuda = @import("cuda");
const state = @import("cuda_state.zig");
const fwd = @import("cuda_forward.zig");
const Forward = fwd.Forward;
const Bufs = fwd.Bufs;
const Seq = fwd.Seq;
const Seg = fwd.Seg;
const Window = fwd.Window;

/// State.reset: an empty sequence in the same buffers.
pub fn reset(f: *Forward, seq: *Seq) !void {
    const st = &seq.st;
    const g = f.g;
    try f.th.zero(st.conv, g.linear_layers * (g.conv_kernel - 1) * g.gdnConv() * 2);
    try f.th.zero(st.rec, g.linear_layers * g.nv * state.gdn_dv * state.gdn_dk * 4);
    st.cur = @splat(0);
    seq.held = 0;
    try f.th.zero(st.ple_tail, (g.ple_kernel - 1) * g.ngram_size * g.wide() * 2);
    for (f.w.layers) |l| if (l.ple) |p| p.ngram.initialHistory(st.history[0 .. p.ngram.n - 1]);
    seq.staged = false;
    try f.setPos(st, 0);
    st.rope_delta = 0;
    try f.th.fill32(st.rope_delta_dev, 0, 1);
    st.mtp_drafted = 0;
    try f.setMtpLen(st, 0);
    f.dropMedia(seq);
}

/// Release the sequence's images and video frames to the budget, draining the stream first.
pub fn dropMedia(f: *Forward, seq: *Seq) void {
    const m = &(seq.media orelse return);
    f.s.synchronize() catch {};
    if (f.budget) |b| b.used -|= m.charged;
    seq.charged -|= m.charged;
    m.deinit(f.gpa);
    seq.media = null;
}

/// Python image_rows.attach; two ranks agree validity and budget first, then rank 0 broadcasts the media by NCCL.
pub fn attachMedia(f: *Forward, seq: *Seq, media: *const @import("cuda_media.zig").Media, prompt_len: usize) !void {
    const lead = if (f.comm) |cm| cm.rank == 0 else true;
    var valid = true;
    media.check(prompt_len) catch {
        valid = false;
    };
    if (lead and (media.follower_rows != 0 or media.features.len != media.featureBytes())) valid = false;
    f.dropMedia(seq);
    const k = media.rowCount();
    const table_bytes = prompt_len * 3 * 4;
    const feat_bytes = media.featureBytes();
    const total = std.mem.alignForward(usize, table_bytes, 256) + std.mem.alignForward(usize, feat_bytes, 256) + std.mem.alignForward(usize, k * 4, 256);
    const fits = if (f.budget) |b| b.used + total <= b.limit and b.leaves(total) else true;
    var table: ?cuda.DeviceBuffer = null;
    var feats: ?cuda.DeviceBuffer = null;
    var rows_dev: ?cuda.DeviceBuffer = null;
    defer if (rows_dev) |*b| b.free();
    errdefer {
        if (table) |*b| b.free();
        if (feats) |*b| b.free();
    }
    var allocated = false;
    if (valid and fits) blk: {
        table = cuda.DeviceBuffer.alloc(f.d, table_bytes) catch break :blk;
        feats = cuda.DeviceBuffer.alloc(f.d, @max(feat_bytes, 1)) catch break :blk;
        rows_dev = cuda.DeviceBuffer.alloc(f.d, @max(k * 4, 1)) catch break :blk;
        allocated = true;
    }
    // an agreed refusal: both ranks return it before any collective of the prompt, so the request fails alone
    if (!try f.agree(valid and fits and allocated)) {
        if (table) |*b| b.free();
        if (feats) |*b| b.free();
        table = null;
        feats = null;
        return error.MediaRefused;
    }
    const rows = try f.gpa.alloc(u32, k);
    errdefer f.gpa.free(rows);
    try f.s.synchronize();
    if (lead) {
        try table.?.upload(0, std.mem.sliceAsBytes(media.positions));
        try feats.?.upload(0, media.features);
        try rows_dev.?.upload(0, std.mem.sliceAsBytes(media.rows));
    }
    if (f.comm) |cm| {
        try cm.broadcast(table.?.ptr, table_bytes, 0, f.s.handle);
        try cm.broadcast(rows_dev.?.ptr, k * 4, 0, f.s.handle);
        try cm.broadcast(feats.?.ptr, feat_bytes, 0, f.s.handle);
        try f.s.synchronize();
    }
    try rows_dev.?.download(0, std.mem.sliceAsBytes(rows));
    f.media_epoch += 1;
    seq.media = .{ .table = table.?, .len = prompt_len, .feats = feats.?, .rows = rows, .width = media.width, .charged = total, .epoch = f.media_epoch };
    table = null;
    feats = null;
    if (f.budget) |b| b.used += total;
    seq.charged += total;
    seq.st.rope_delta = @intCast(media.delta);
    try f.th.fill32(seq.st.rope_delta_dev, @bitCast(@as(i32, @intCast(media.delta))), 1);
}

/// Python image_rows.finish: the features end with the prompt (the table stays for decode).
pub fn finishMedia(f: *Forward, seq: *Seq) void {
    const m = &(seq.media orelse return);
    if (m.feats == null) return;
    f.s.synchronize() catch {};
    const freed = m.finish(f.gpa);
    if (f.budget) |b| b.used -|= freed;
    seq.charged -|= freed;
    m.charged -|= freed;
}

/// Python image_rows.embed: placeholder rows of the chunk get their feature row in each of the S residual streams.
pub fn injectMedia(f: *Forward, seq: *Seq, x: *const Bufs, R: usize) !void {
    const m = &(seq.media orelse return);
    const feats = m.feats orelse return;
    const p0 = seq.st.pos;
    const D = f.g.hidden;
    const S = f.g.streams;
    if (m.width != D) return error.MediaWidth;
    const row_bytes = D * 2;
    // the first media row at or past p0
    var i = std.sort.lowerBound(u32, m.rows, @as(u32, @intCast(p0)), struct {
        fn order(a: u32, b: u32) std.math.Order {
            return std.math.order(a, b);
        }
    }.order);
    while (i < m.rows.len and m.rows[i] < p0 + R) {
        // a run of consecutive prompt rows
        var j = i + 1;
        while (j < m.rows.len and m.rows[j] == m.rows[j - 1] + 1 and m.rows[j] < p0 + R) j += 1;
        const t = m.rows[i] - p0;
        for (0..S) |sidx| try f.th.slotCopy(feats.ptr + i * row_bytes, row_bytes, x.b.h + (t * S + sidx) * row_bytes, S * row_bytes, row_bytes, j - i);
        i = j;
    }
}

// -- staging ----------------------------------------------------------------------------------------------

/// Grow a sequence's caches to `rows` rows within the budget (old and new fit together); drains the stream first.
pub fn grow(f: *Forward, seq: *Seq, rows: usize) !void {
    const st = &seq.st;
    if (rows <= st.capacity) return;
    if (rows > st.limit) return error.ContextPastWindow;
    const target = @min(st.limit, std.mem.alignForward(usize, rows, state.grow_step));
    if (seq.regions.items.len > 0) return f.growMapped(seq, target);
    const now = if (st.ctx) |c| c.buf.len else 0;
    const next = state.State.cacheBytes(f.g, target) + 64 * 256;
    if (!try f.agree(if (f.budget) |b| b.used + next <= b.limit and b.leaves(next) else true)) return error.NoRoom;
    try f.s.synchronize();
    _ = try st.ensure(rows);
    const after = if (st.ctx) |c| c.buf.len else 0;
    if (f.budget) |b| {
        b.used = b.used - now + after;
        seq.charged = seq.charged - now + after;
    }
}

/// Two ranks: a growth goes ahead only when both may take it, so both refuse together and stay in step.
pub fn agree(f: *Forward, ok: bool) !bool {
    const cm = f.comm orelse return ok;
    const mine: u64 = @intFromBool(ok);
    const v: cuda.DeviceBuffer = .{ .d = f.d, .ptr = f.sc.vote, .len = 8 };
    try f.s.synchronize();
    try v.upload(0, std.mem.asBytes(&mine));
    try cm.allGather(f.sc.vote, f.sc.vote + 64, 1, .u64, f.s.handle);
    var both: [2]u64 = undefined;
    try f.s.synchronize();
    const buf: cuda.DeviceBuffer = .{ .d = f.d, .ptr = f.sc.vote + 64, .len = 16 };
    try buf.download(0, std.mem.sliceAsBytes(&both));
    return both[0] != 0 and both[1] != 0;
}

/// grow in place: more zeroed physical memory under each reserved range, no copy, no pointer moves.
pub fn growMapped(f: *Forward, seq: *Seq, target: usize) !void {
    const st = &seq.st;
    const g = f.g;
    const k_row = g.kRow();
    const v_row = g.vRow();
    const pooled = g.blocks(target) * g.index_dim * 2;
    var need: usize = 0;
    const layers = seq.regions.items.len / 4;
    for (0..layers) |i| {
        const r = seq.regions.items[4 * i ..][0..4];
        for ([_]usize{ 0, 1, 3 }, [_]usize{ target * k_row, target * v_row, pooled }) |j, bytes| need += r[j].v.up(bytes) -| r[j].mapped;
    }
    if (!try f.agree(if (f.budget) |b| b.used + need <= b.limit and b.leaves(need) else true)) return error.NoRoom;
    var added: usize = 0;
    for (0..layers) |i| {
        const r = seq.regions.items[4 * i ..][0..4];
        for ([_]usize{ 0, 1, 3 }, [_]usize{ target * k_row, target * v_row, pooled }) |j, bytes| {
            const before = r[j].mapped;
            const got = try r[j].growTo(f.gpa, bytes);
            if (got > 0) try f.th.zero(r[j].base + before, got);
            added += got;
        }
    }
    st.capacity = target;
    if (f.budget) |b| b.used += added;
    seq.charged += added;
}

/// Token ids into the window's `ids` through the pinned buffer (after the last staging's copies are done).
pub fn stageIds(f: *Forward, x: *const Bufs, tokens: []const u32) !void {
    if (tokens.len > x.b.rows or tokens.len > f.pin_rows) return error.WindowPastBuffers;
    try f.staged.synchronize();
    const ids = std.mem.bytesAsSlice(i32, f.pin.bytes[0 .. tokens.len * 4]);
    for (ids, tokens) |*a, t| a.* = @intCast(t);
    try f.upload(x.b.ids, 0, tokens.len * 4);
    try f.staged.record(f.s);
}

/// forward.stage: every window's token ids and n-gram rows into one buffer; `segs` gets each stream's rows.
pub fn stageMany(f: *Forward, windows: []const Window, x: *const Bufs, segs: []Seg) !usize {
    var R: usize = 0;
    if (segs.len < windows.len) return error.SegsTooFew;
    for (windows, 0..) |w, i| {
        if (w.tokens.len == 0) return error.EmptyWindow;
        for (windows[0..i]) |o| if (o.seq == w.seq) return error.StreamTwiceInRound;
        segs[i] = .{ .seq = w.seq, .a0 = R, .a1 = R + w.tokens.len };
        R += w.tokens.len;
    }
    if (R > x.b.rows or R > f.pin_rows) return error.WindowPastBuffers;
    for (windows) |w| {
        const st = &w.seq.st;
        if (st.pos + w.tokens.len > st.capacity) try f.grow(w.seq, st.pos + w.tokens.len);
        try w.seq.window.resize(f.gpa, w.tokens.len);
        for (w.seq.window.items, w.tokens) |*a, t| a.* = t;
    }
    // a prompt chunk runs from the committed DeltaNet state: last round's kept rows folded in first
    if (x.b.prefill) for (windows) |w| try f.flush(w.seq);
    // the previous step's copies out of the pinned buffer are done
    try f.staged.synchronize();
    const ids = std.mem.bytesAsSlice(i32, f.pin.bytes[0 .. R * 4]);
    for (segs[0..windows.len], windows) |sg, w| for (ids[sg.a0..sg.a1], w.tokens) |*a, t| {
        a.* = @intCast(t);
    };
    try f.upload(x.b.ids, 0, R * 4);
    const rows_at = std.mem.alignForward(usize, f.pin_rows * 4, 256);
    for (f.w.layers) |l| if (l.ple) |p| {
        const heads = p.ngram.heads;
        const nid = try f.gpa.alloc(i64, R * heads);
        defer f.gpa.free(nid);
        for (segs[0..windows.len]) |sg| try p.ngram.ids(sg.seq.st.history[0 .. p.ngram.n - 1], sg.seq.window.items, nid[sg.a0 * heads .. sg.a1 * heads]);
        const table = &(f.w.table orelse return error.NoNgramTable);
        if (table.gpu != null) {
            // the ids on the device, then the rank's heads' rows gathered there (W4's Gather, the host bits)
            @memcpy(std.mem.bytesAsSlice(i64, f.pin.bytes[rows_at..][0 .. R * heads * 8]), nid);
            try f.upload(f.sc.ngram_ids, rows_at, R * heads * 8);
            try f.gather.?.run(f.s, table, f.sc.ngram_ids, heads, @intCast(R), x.b.ple_v);
            continue;
        }
        const out = std.mem.bytesAsSlice(u16, f.pin.bytes[rows_at..][0 .. R * heads * table.width * 2]);
        try table.gather(nid, @alignCast(out));
        try f.upload(x.b.ple_v, rows_at, R * heads * table.width * 2);
    };
    for (windows) |w| w.seq.staged = true;
    try f.staged.record(f.s);
    if (!x.b.prefill and f.round != null and f.g.linear_layers > 0) try f.roundTables(segs[0..windows.len], 1 - f.cur_par);
    return R;
}

/// forward.stage for one stream (its rows from 0).
pub fn stage(f: *Forward, seq: *Seq, x: *const Bufs, tokens: []const u32) !void {
    var one: [1]Seg = undefined;
    _ = try f.stageMany(&.{.{ .seq = seq, .tokens = tokens }}, x, &one);
}

// -- commit -----------------------------------------------------------------------------------------------

/// forward.commit: keep the first `keep` rows: replays, conv shift, n-gram history and tail, pos.
pub fn commit(f: *Forward, seq: *Seq, x: *const Bufs, R: usize, keep: usize) !void {
    return f.commitAt(seq, x, R, keep, 0);
}

/// commit(..., at): the stream ran the window's rows [at, at + R) of a shared round.
pub fn commitAt(f: *Forward, seq: *Seq, x: *const Bufs, R: usize, keep: usize, at: usize) !void {
    const b = &x.b;
    const g = f.g;
    const st = &seq.st;
    if (keep < 1 or keep > R or (b.prefill and keep != R)) return error.BadKeep;
    if (!b.prefill) {
        // the kept rows wait in the round scratch; the stream's next tree (or a flush) folds them in
        seq.held = keep;
        seq.held_a0 = at;
        seq.held_par = f.cur_par;
        const C = g.gdnConv();
        const pw = g.gdnWidth();
        try f.t.shiftWindows(st.conv, b.proj + at * pw * 2, keep, (g.conv_kernel - 1) * C, b.rows * pw, pw, g.linear_layers, C, g.conv_kernel - 1);
    }
    if (seq.staged) {
        for (f.w.layers) |l| if (l.ple) |p| p.ngram.advance(st.history[0 .. p.ngram.n - 1], seq.window.items[0..keep]);
        seq.staged = false;
        const Wd = g.wide();
        const taps = (g.ple_kernel - 1) * g.ngram_size;
        try f.t.shiftWindows(st.ple_tail, b.ple_nrow + at * Wd * 2, keep, taps * Wd, b.rows * Wd, Wd, 1, Wd, taps);
    }
    try f.setPos(st, st.pos + keep);
}
