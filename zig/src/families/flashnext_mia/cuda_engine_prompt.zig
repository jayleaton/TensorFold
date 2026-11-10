//! Flash Next engine prompts: prefill, kept snapshots and resumes, burst prefill and sliced fills.
const std = @import("std");
const envs = @import("cuda_env.zig");
const cuda = @import("cuda");
const lanes = @import("lanes");
const media_mod = @import("cuda_media.zig");
const state = @import("cuda_state.zig");
const fwd = @import("cuda_forward.zig");
const mtp = @import("cuda_mtp.zig");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const Seq = engine.Seq;
const seconds = engine.seconds;
const aligned = engine.aligned;

// -- the prompt -------------------------------------------------------------------------------------------

/// decode.prefill (fresh): the prompt's chunks committed in order, each absorbed by the MTP head; the first draw.
pub fn prefill(e: *Engine, prompt: []const u32, sampling: ?lanes.Sampling) !u32 {
    return (try e.prefillWith(prompt, sampling, .{})).first;
}

/// A kept prompt state (Python State.snapshot + MTP tail) at `pos`; a resume needs those cache rows in place.
pub const Snapshot = struct {
    pos: usize,
    mtp_len: usize,
    history: [3]i64,
    tokens: []u32,
    mem: cuda.DeviceBuffer,
    rec: u64,
    conv: u64,
    ple_tail: u64,
    tail: u64,
    /// each cache layer's indexer keys of its incomplete block (a ring may overwrite them before a resume)
    ikc: u64,
};

pub const PrefillOptions = struct {
    /// keep the state at this many tokens (0 < keep_at <= prompt.len; Python entry_end is len - 1)
    keep_at: ?usize = null,
    /// resume from a kept state of this sequence (the prompt extends its tokens)
    resume_from: ?*const Snapshot = null,
    /// the prompt's images and video frames (Python prefill(..., vision=)): fresh prompts only, nothing kept
    media: ?*const media_mod.Media = null,
};

pub const Prefilled = struct { first: u32, kept: ?*Snapshot = null };

pub fn snapshotBytes(e: *const Engine) [5]usize {
    const g = e.g;
    return .{ g.linear_layers * g.nv * state.gdn_dv * state.gdn_dk * 4, g.linear_layers * (g.conv_kernel - 1) * g.gdnConv() * 2, (g.ple_kernel - 1) * g.ngram_size * g.wide() * 2, g.wide() * 2, (g.attention_layers + 1) * g.index_ratio * g.index_dim * 2 };
}

/// Cache layer i's incomplete-block rows [first, end) at the committed rows (pos, or mtp_len for the MTP layer).
pub fn ikcTail(e: *const Engine, i: usize, pos_rows: usize, mtp_rows: usize) [2]usize {
    const end = if (e.g.mtp and i == e.g.attention_layers) mtp_rows else pos_rows;
    return .{ end / e.g.index_ratio * e.g.index_ratio, end };
}

/// The bound sequence's committed state as a Snapshot (State.snapshot); `mtp_len` and the tail row given.
pub fn snapshot(e: *Engine, seq: *Seq, mtp_len: usize, tail: u64) !*Snapshot {
    const st = &seq.st;
    try e.f.flush(seq); // the kept rows folded in: the state is the committed one
    const sz = e.snapshotBytes();
    const total = aligned(&sz);
    if (e.budget.used + total > e.budget.limit) return error.NoRoom;
    const snap = try e.gpa.create(Snapshot);
    errdefer e.gpa.destroy(snap);
    var a: state.Arena = .{ .buf = try cuda.DeviceBuffer.alloc(e.ctx.d, total) };
    errdefer a.buf.free();
    snap.* = .{ .pos = st.pos, .mtp_len = mtp_len, .history = st.history, .tokens = &.{}, .mem = a.buf, .rec = a.take(sz[0]), .conv = a.take(sz[1]), .ple_tail = a.take(sz[2]), .tail = a.take(sz[3]), .ikc = a.take(sz[4]) };
    const row = e.g.index_dim * 2;
    for (0..e.g.attention_layers + @intFromBool(e.g.mtp)) |i| {
        const t = e.ikcTail(i, st.pos, mtp_len);
        try e.f.th.copy(snap.ikc + i * e.g.index_ratio * row, st.ikc[i] + t[0] * row, (t[1] - t[0]) * row);
    }
    try e.f.th.copy(snap.rec, st.rec, sz[0]);
    try e.f.th.copy(snap.conv, st.conv, sz[1]);
    try e.f.th.copy(snap.ple_tail, st.ple_tail, sz[2]);
    try e.f.th.copy(snap.tail, tail, sz[3]);
    e.budget.used += total;
    return snap;
}

pub fn freeSnapshot(e: *Engine, snap: *Snapshot) void {
    e.stream.synchronize() catch {};
    e.budget.used -|= snap.mem.len;
    snap.mem.free();
    e.gpa.free(snap.tokens);
    e.gpa.destroy(snap);
}

/// State.restore: the snapshot's state into `seq` (its caches' first snap.pos rows must hold the prompt's).
pub fn restore(e: *Engine, seq: *Seq, snap: *const Snapshot) !void {
    const st = &seq.st;
    const sz = e.snapshotBytes();
    if (snap.pos > st.capacity) return error.CachesTooShort;
    try e.f.th.copy(st.rec, snap.rec, sz[0]);
    st.cur = @splat(0);
    seq.held = 0;
    try e.f.th.copy(st.conv, snap.conv, sz[1]);
    try e.f.th.copy(st.ple_tail, snap.ple_tail, sz[2]);
    const row = e.g.index_dim * 2;
    for (0..e.g.attention_layers + @intFromBool(e.g.mtp)) |i| {
        const t = e.ikcTail(i, snap.pos, snap.mtp_len);
        try e.f.th.copy(st.ikc[i] + t[0] * row, snap.ikc + i * e.g.index_ratio * row, (t[1] - t[0]) * row);
    }
    st.history = snap.history;
    seq.staged = false;
    try e.f.setPos(st, snap.pos);
    st.rope_delta = 0;
    try e.f.th.fill32(st.rope_delta_dev, 0, 1);
    st.mtp_drafted = 0;
    try e.f.setMtpLen(st, snap.mtp_len);
}

/// State.copy_prefix: `src`'s first `rows` cache rows and `mtp_rows` MTP rows into `dst`, to resume src's snapshot.
pub fn copyPrefix(e: *Engine, dst: *Seq, src: *Seq, rows: usize, mtp_rows: usize) !void {
    if (dst == src) return error.SameSequence;
    if (rows > src.st.pos or mtp_rows > src.st.mtp_len) return error.PrefixPastSource;
    try e.f.grow(dst, @max(rows, mtp_rows));
    const g = e.g;
    for (0..g.attention_layers + @intFromBool(g.mtp)) |i| {
        const n = if (g.mtp and i == g.attention_layers) mtp_rows else rows;
        // whole rows: an FP8 key row carries its position's key and value scales
        try e.f.th.copy(dst.st.kc[i], src.st.kc[i], n * g.kRow());
        try e.f.th.copy(dst.st.vc[i], src.st.vc[i], n * g.vRow());
        // the indexer keys a later pool reads: the incomplete block's rows (a ring holds no older ones)
        const first_ikc = if (dst.ring_rows > 0 or src.ring_rows > 0) n / g.index_ratio * g.index_ratio else 0;
        try e.f.th.copy(dst.st.ikc[i] + first_ikc * g.index_dim * 2, src.st.ikc[i] + first_ikc * g.index_dim * 2, (n - first_ikc) * g.index_dim * 2);
        try e.f.th.copy(dst.st.pooled[i], src.st.pooled[i], (n / g.index_ratio) * g.index_dim * 2);
    }
}

/// decode.prefill with kept states: chunks (one ends at kept `keep_at`; same bits) or a kept-state resume; draw.
pub fn prefillWith(e: *Engine, prompt: []const u32, sampling: ?lanes.Sampling, o: PrefillOptions) !Prefilled {
    if (prompt.len == 0) return error.EmptyPrompt;
    if (e.pfill != null) return error.PromptBuffersBusy; // a sliced fill holds the prompt buffers mid-chunk
    if (prompt.len + e.depth + 1 > e.bound.st.limit) return error.ContextPastWindow;
    if (o.keep_at) |k| if (k == 0 or k > prompt.len) return error.BadKeepPoint;
    // an image prompt prefills from its start and keeps no token-only snapshot (Python decode.prefill)
    if (o.media != null and (o.keep_at != null or o.resume_from != null)) return error.MediaPromptKept;
    e.setSampling(sampling);
    const seq = e.bound;
    const Wd = e.g.wide();
    const pb = &e.pbuf;
    const use_mtp = e.absorbing and e.w.mtp != null;
    var start: usize = 0;
    var t_pw: std.Io.Timestamp = undefined;
    var pw: [4]f64 = @splat(0); // restore, the chunks' forwards, their MTP absorbs and commits, the draw
    var pw_chunks: usize = 0;
    if (e.pass_times) {
        try e.stream.synchronize();
        t_pw = std.Io.Clock.awake.now(e.io);
    }
    if (o.resume_from) |snap| {
        if (!(snap.pos > 0 and snap.pos < prompt.len)) return error.ResumeMustExtend;
        if (!std.mem.eql(u32, snap.tokens, prompt[0..snap.pos])) return error.ResumeOtherPrompt;
        try e.restore(seq, snap);
        if (use_mtp) {
            _ = try mtp.forward(&e.f, seq, pb, prompt[snap.pos .. snap.pos + 1], snap.tail);
            try e.f.setMtpLen(&seq.st, seq.st.mtp_len + 1);
        }
        start = snap.pos;
    } else try e.f.reset(seq);
    if (o.media) |m| try e.f.attachMedia(seq, m, prompt.len);
    errdefer e.f.finishMedia(seq);
    if (e.pass_times) pw[0] += try e.passLap(&t_pw);
    var kept: ?*Snapshot = null;
    errdefer if (kept) |k| e.freeSnapshot(k);
    var chunk: usize = 0;
    while (start < prompt.len) : (chunk += 1) {
        var end = e.chunkEnd(start, prompt.len);
        if (o.keep_at) |k| if (start < k and k < end) {
            end = k;
        };
        const R = end - start;
        const final = end == prompt.len;
        if (e.f.dump) |dm| dm.setChunk(chunk);
        const t_chunk = std.Io.Clock.awake.now(e.io);
        if (e.prof) |p| {
            // prompt chunks only (decode windows may be graph captures)
            try e.stream.synchronize();
            e.f.prof = p;
            p.mtp = false;
            try p.mark(e.stream, .start);
        }
        defer e.f.prof = null;
        if (e.poison) try e.poisonPads(R);
        const lg = try e.f.forward(seq, pb, prompt[start..end], final);
        if (e.pass_times) {
            pw[1] += try e.passLap(&t_pw);
            pw_chunks += 1;
        }
        if (final) {
            try e.f.th.copy(e.last_logits, lg.?, e.g.head_n * 2);
            if (e.world > 1) try e.f.th.copy(e.last_cand, pb.b.cand_all, e.world * (2 * state.cand + 1) * 4);
        }
        try e.f.th.copy(seq.last_streams.?.ptr, pb.b.streams + (R - 1) * Wd * 2, Wd * 2);
        if (e.digest) try e.chunkDigest(chunk, R, if (final) lg.? else null);
        const nxt = prompt[start + 1 .. @min(end + 1, prompt.len)];
        if (use_mtp and nxt.len > 0) {
            if (e.prof) |p| {
                try p.mark(e.stream, .other);
                p.mtp = true;
            }
            _ = try mtp.forward(&e.f, seq, pb, nxt, pb.b.streams);
            try e.f.setMtpLen(&seq.st, seq.st.mtp_len + nxt.len);
            if (e.prof) |p| {
                try p.mark(e.stream, .other);
                p.mtp = false;
            }
        }
        try e.f.commit(seq, pb, R, R);
        if (o.keep_at) |k| if (end == k and kept == null) {
            // the MTP head's length one short: a resume absorbs the next prompt's own token at row k - 1
            const snap = try e.snapshot(seq, if (use_mtp and !final) seq.st.mtp_len - 1 else seq.st.mtp_len, seq.last_streams.?.ptr);
            kept = snap;
            snap.tokens = try e.gpa.dupe(u32, prompt[0..k]);
        };
        if (e.prof) |p| {
            try p.mark(e.stream, .commit);
            try e.stream.synchronize();
            try p.chunk(e.rank, R, seconds(e.io, t_chunk) * 1e3);
        }
        if (e.pass_times) pw[2] += try e.passLap(&t_pw);
        start = end;
    }
    if (e.prof) |p| p.summary(e.rank, "prefill");
    if (e.f.dump) |dm| dm.setChunk(null);
    e.f.finishMedia(seq);
    var first: [1]u32 = undefined;
    try e.draw(e.last_logits, e.last_cand, 1, prompt.len, &first);
    if (e.pass_times) {
        pw[3] = try e.passLap(&t_pw);
        if (e.rank == 0) std.log.info("prefillWith {d} tokens{s}{s}: restore {d:.1} ms, {d} forwards {d:.1} ms, absorbs + commits {d:.1} ms, draw {d:.1} ms", .{ prompt.len, if (o.resume_from != null) " (resumed)" else "", if (o.keep_at != null) " (kept)" else "", pw[0], pw_chunks, pw[1], pw[2], pw[3] });
    }
    return .{ .first = first[0], .kept = kept };
}

/// Where a prompt chunk from `start` ends: prefill_rows on, or the prompt's end if at most prefill_tail further.
pub fn chunkEnd(e: *const Engine, start: usize, len: usize) usize {
    const end = @min(start + e.prefill_rows, len);
    return if (len - end <= e.prefill_tail) len else end;
}

/// ms since `last` with the stream drained; `last` moves to now.
pub fn passLap(e: *Engine, last: *std.Io.Timestamp) !f64 {
    try e.stream.synchronize();
    const now = std.Io.Clock.awake.now(e.io);
    const ns = last.durationTo(now).nanoseconds;
    last.* = now;
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// One prompt of a burst (prefillMany).
pub const ManyPrompt = struct { seq: *Seq, prompt: []const u32, sampling: ?lanes.Sampling };

/// Whether prefillMany takes these prompts in one pass: 2..state.max_rows whole prompts that fit the buffers.
pub fn manyFit(e: *const Engine, prompts: []const ManyPrompt) bool {
    if (prompts.len < 2 or prompts.len > state.max_rows or e.pfill != null) return false;
    var rows: usize = 0;
    for (prompts) |p| {
        if (p.prompt.len == 0 or p.prompt.len + e.depth + 1 > p.seq.st.limit) return false;
        rows += p.prompt.len;
    }
    return rows <= e.prefill_rows;
}

/// Several prompts in one pass: row-wise steps once, state per prompt; each one equals its own prefillWith.
pub fn prefillMany(e: *Engine, prompts: []const ManyPrompt, firsts: []u32) !void {
    if (!e.manyFit(prompts)) return error.BadBurst;
    if (firsts.len < prompts.len) return error.BadBurst;
    const n = prompts.len;
    const pb = &e.pbuf;
    const Wd = e.g.wide();
    const use_mtp = e.absorbing and e.w.mtp != null;
    if (e.rank == 0 and envs.get("TF_FLASHNEXT_BURST_LOG") != null) {
        var rows: usize = 0;
        for (prompts) |p| rows += p.prompt.len;
        std.debug.print("burst prefill: {d} prompts, {d} rows in one pass\n", .{ n, rows });
    }
    var windows: [state.max_rows]fwd.Window = undefined;
    var segs: [state.max_rows]fwd.Seg = undefined;
    for (prompts, 0..) |p, i| {
        try e.f.reset(p.seq);
        windows[i] = .{ .seq = p.seq, .tokens = p.prompt };
    }
    _ = try e.f.stageMany(windows[0..n], pb, &segs);
    const lg = (try e.f.computeSegs(segs[0..n], pb, true)).?;
    for (prompts, segs[0..n]) |p, sg| try e.f.th.copy(p.seq.last_streams.?.ptr, pb.b.streams + (sg.a1 - 1) * Wd * 2, Wd * 2);
    if (use_mtp) {
        // each prompt's MTP rows: its next tokens over its rows' streams (a one-row prompt has none)
        var inputs: [state.max_rows]mtp.Input = undefined;
        var m: usize = 0;
        for (prompts, segs[0..n]) |p, sg| {
            if (p.prompt.len < 2) continue;
            inputs[m] = .{ .seq = p.seq, .next = p.prompt[1..], .streams = pb.b.streams + sg.a0 * Wd * 2 };
            m += 1;
        }
        if (m > 0) {
            var msegs: [state.max_rows]fwd.Seg = undefined;
            _ = try mtp.stageMany(&e.f, inputs[0..m], pb, &msegs);
            _ = try mtp.computeSegs(&e.f, msegs[0..m], pb);
            for (inputs[0..m]) |in| try e.f.setMtpLen(&in.seq.st, in.seq.st.mtp_len + in.next.len);
        }
    }
    for (prompts, segs[0..n]) |p, sg| try e.f.commitAt(p.seq, pb, sg.a1 - sg.a0, sg.a1 - sg.a0, sg.a0);
    for (prompts, 0..) |p, i| {
        const s: ?lanes.Sampling = if (p.sampling) |x| (if (x.temperature > 0) x else null) else null;
        try e.drawRows(lg, pb.b.cand_all, n, i, 1, p.prompt.len, s, firsts[i .. i + 1]);
    }
}

/// A prompt filling in slices between decode rounds: the chunk in progress, its next layer, its pending write-back.
pub const Fill = struct {
    seq: *Seq,
    prompt: []u32,
    sampling: ?lanes.Sampling,
    keep_at: ?usize,
    use_mtp: bool,
    start: usize,
    end: usize = 0,
    chunk: usize = 0,
    layer: usize = 0,
    open: bool = false,
    pending: ?fwd.Pending = null,
    kept: ?*Snapshot = null,
    done: bool = false,
    first: u32 = 0,
};

/// prefillWith's start (reset, or restore and the MTP tail absorbed) for a prompt that then fills in slices.
pub fn fillBegin(e: *Engine, seq: *Seq, prompt: []const u32, sampling: ?lanes.Sampling, o: PrefillOptions) !*Fill {
    if (prompt.len == 0) return error.EmptyPrompt;
    if (prompt.len + e.depth + 1 > seq.st.limit) return error.ContextPastWindow;
    if (o.keep_at) |k| if (k == 0 or k > prompt.len) return error.BadKeepPoint;
    if (o.media != null and (o.keep_at != null or o.resume_from != null)) return error.MediaPromptKept;
    const fl = try e.gpa.create(Fill);
    errdefer e.gpa.destroy(fl);
    fl.* = .{ .seq = seq, .prompt = try e.gpa.dupe(u32, prompt), .sampling = if (sampling) |x| (if (x.temperature > 0) x else null) else null, .keep_at = o.keep_at, .use_mtp = e.absorbing and e.w.mtp != null, .start = 0 };
    errdefer e.gpa.free(fl.prompt);
    if (o.resume_from) |snap| {
        if (!(snap.pos > 0 and snap.pos < prompt.len)) return error.ResumeMustExtend;
        if (!std.mem.eql(u32, snap.tokens, prompt[0..snap.pos])) return error.ResumeOtherPrompt;
        try e.restore(seq, snap);
        if (fl.use_mtp) {
            _ = try mtp.forward(&e.f, seq, &e.pbuf, prompt[snap.pos .. snap.pos + 1], snap.tail);
            try e.f.setMtpLen(&seq.st, seq.st.mtp_len + 1);
        }
        fl.start = snap.pos;
    } else try e.f.reset(seq);
    if (o.media) |m| try e.f.attachMedia(seq, m, prompt.len);
    return fl;
}

pub fn fillFree(e: *Engine, fl: *Fill) void {
    if (e.pfill == fl) e.pfill = null;
    if (fl.kept) |k| e.freeSnapshot(k);
    e.gpa.free(fl.prompt);
    e.gpa.destroy(fl);
}

/// Up to `layers` more layer passes of the filling prompt, as prefillWith calls them; true once `first` is drawn.
pub fn fillStep(e: *Engine, fl: *Fill, layers: usize) !bool {
    if (fl.done) return true;
    const pb = &e.pbuf;
    const seq = fl.seq;
    const Wd = e.g.wide();
    const n_layers = e.w.layers.len;
    var budget = @max(layers, 1);
    while (budget > 0) {
        if (!fl.open) {
            if (e.pfill != null and e.pfill.? != fl) return error.PromptBuffersBusy;
            e.pfill = fl;
            var end = e.chunkEnd(fl.start, fl.prompt.len);
            if (fl.keep_at) |k| if (fl.start < k and k < end) {
                end = k;
            };
            fl.end = end;
            try e.f.stage(seq, pb, fl.prompt[fl.start..end]);
            try e.f.t.embed(pb.b.ids, e.w.embed, pb.b.h, end - fl.start, e.g.hidden, e.g.streams);
            try e.f.injectMedia(seq, pb, end - fl.start);
            fl.layer = 0;
            fl.pending = null;
            fl.open = true;
        }
        const R = fl.end - fl.start;
        const segs = [1]fwd.Seg{.{ .seq = seq, .a0 = 0, .a1 = R }};
        if (fl.layer < n_layers) {
            fl.pending = try e.f.layerForward(&e.w.layers[fl.layer], &segs, pb, R, fl.pending, false);
            fl.layer += 1;
            budget -= 1;
            continue;
        }
        // the chunk's finish, MTP absorb and commit (as prefillWith's loop body)
        const final = fl.end == fl.prompt.len;
        const lg = try e.f.finish(&e.w.mixer, pb, R, fl.pending.?, final);
        if (final) {
            try e.f.th.copy(e.last_logits, lg.?, e.g.head_n * 2);
            if (e.world > 1) try e.f.th.copy(e.last_cand, pb.b.cand_all, e.world * (2 * state.cand + 1) * 4);
        }
        try e.f.th.copy(seq.last_streams.?.ptr, pb.b.streams + (R - 1) * Wd * 2, Wd * 2);
        const nxt = fl.prompt[fl.start + 1 .. @min(fl.end + 1, fl.prompt.len)];
        if (fl.use_mtp and nxt.len > 0) {
            _ = try mtp.forward(&e.f, seq, pb, nxt, pb.b.streams);
            try e.f.setMtpLen(&seq.st, seq.st.mtp_len + nxt.len);
        }
        try e.f.commit(seq, pb, R, R);
        if (fl.keep_at) |k| if (fl.end == k and fl.kept == null) {
            const snap = try e.snapshot(seq, if (fl.use_mtp and !final) seq.st.mtp_len - 1 else seq.st.mtp_len, seq.last_streams.?.ptr);
            fl.kept = snap;
            snap.tokens = try e.gpa.dupe(u32, fl.prompt[0..k]);
        };
        fl.start = fl.end;
        fl.open = false;
        e.pfill = null;
        fl.chunk += 1;
        if (fl.start == fl.prompt.len) {
            e.f.finishMedia(seq);
            var first: [1]u32 = undefined;
            try e.drawRows(e.last_logits, e.last_cand, 1, 0, 1, fl.prompt.len, fl.sampling, &first);
            fl.first = first[0];
            fl.done = true;
            return true;
        }
        budget -= 1;
    }
    return false;
}

/// TF_FLASHNEXT_POISON_PADS: rows [R, rows) of the prompt buffers a row split's pad rows touch.
pub fn poisonPads(e: *Engine, R: usize) !void {
    const b = &e.pbuf.b;
    const g = e.g;
    const v: u8 = if (e.rank == 0) 0xFF else 0x7F;
    const qrow = g.heads * g.head_dim * 2;
    const list = [_][2]u64{
        .{ b.h, g.wide() * 2 },                 .{ b.streams, g.wide() * 2 },  .{ b.mixed, g.hidden * 2 },
        .{ b.gout, g.nv * state.gdn_dv * 2 },   .{ b.gated, qrow },            .{ b.attn_o, qrow },
        .{ b.moe_y, g.slots() * g.hidden * 2 }, .{ b.moe_wts, g.slots() * 4 }, .{ b.moe_act, g.slots() * g.moe_width * 2 },
        .{ b.inj_a, g.streams * 2 },            .{ b.inj_m, g.streams * 2 },   .{ b.pss, (g.hidden / 256) * g.streams * 4 },
    };
    if (R >= b.rows) return;
    for (list) |it| {
        const buf: cuda.DeviceBuffer = .{ .d = e.ctx.d, .ptr = it[0] + R * it[1], .len = (b.rows - R) * it[1] };
        try buf.fill8(v, e.stream.handle);
    }
}

/// TF_FLASHNEXT_PREFILL_DIGEST: sha256 of a chunk's final streams (all R rows) and of its last logits.
pub fn chunkDigest(e: *Engine, chunk: usize, R: usize, logits: ?u64) !void {
    const Wd = e.g.wide();
    const n = R * Wd * 2;
    const host = try e.gpa.alloc(u8, @max(n, e.g.head_n * 2));
    defer e.gpa.free(host);
    try e.read(e.pbuf.b.streams, host[0..n]);
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(host[0..n], &d, .{});
    std.debug.print("digest rank {d} chunk {d}: {d} rows, streams {s}", .{ e.rank, chunk, R, std.fmt.bytesToHex(d, .lower)[0..16] });
    var kept: [2][16]u8 = .{ std.fmt.bytesToHex(d, .lower)[0..16].*, @splat('-') };
    if (logits) |lg| {
        try e.read(lg, host[0 .. e.g.head_n * 2]);
        std.crypto.hash.sha2.Sha256.hash(host[0 .. e.g.head_n * 2], &d, .{});
        std.debug.print(" logits {s}", .{std.fmt.bytesToHex(d, .lower)[0..16]});
        kept[1] = std.fmt.bytesToHex(d, .lower)[0..16].*;
    }
    std.debug.print("\n", .{});
    if (e.digests) |list| try list.append(e.gpa, kept);
}

/// The prompt's last row's streams (device, [1, S*D] bf16), what the first MTP absorb reads.
pub fn lastStreams(e: *Engine) u64 {
    return e.bound.last_streams.?.ptr;
}
