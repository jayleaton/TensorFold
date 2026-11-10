//! Flash Next engine rounds: the main head's draws, shared verify rounds, MTP drafting and generateMany.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const media_mod = @import("cuda_media.zig");
const state = @import("cuda_state.zig");
const sampler = @import("cuda_sampler.zig");
const decode = @import("cuda_decode.zig");
const fwd = @import("cuda_forward.zig");
const mtp = @import("cuda_mtp.zig");
const nucleus = @import("cuda_nucleus.zig");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const Allocator = std.mem.Allocator;
const default_confidence = engine.default_confidence;
const wide_confidence = engine.wide_confidence;
const max_k = engine.max_k;
const Seq = engine.Seq;
const seconds = engine.seconds;
const ManyPrompt = Engine.ManyPrompt;
const Fill = Engine.Fill;

// -- draws ------------------------------------------------------------------------------------------------

pub fn read(e: *Engine, ptr: u64, host: []u8) !void {
    try e.stream.synchronize();
    const buf: cuda.DeviceBuffer = .{ .d = e.ctx.d, .ptr = ptr, .len = host.len };
    try buf.download(0, host);
}

/// Engine.sample over `rows` rows of the main head's `logits` (one rank) or gathered `cand_all` (two ranks).
pub fn draw(e: *Engine, logits: u64, cand_all: u64, rows: usize, first_position: u64, out: []u32) !void {
    return e.drawRows(logits, cand_all, rows, 0, rows, first_position, e.sampling, out);
}

/// choose_gathered over one stream's rows [row0, row0 + rows) of the host copy `all` [world][total][2 CAND + 1].
pub fn chooseFrom(e: *Engine, all: []const f32, total: usize, row0: usize, rows: usize, first_position: u64, s: ?lanes.Sampling, out: []u32) !void {
    const width = 2 * state.cand + 1;
    // this window's rows of every rank's block
    const got = try e.gpa.alloc(f32, e.world * rows * width);
    defer e.gpa.free(got);
    for (0..e.world) |r| @memcpy(got[r * rows * width ..][0 .. rows * width], all[(r * total + row0) * width ..][0 .. rows * width]);
    var positions: [state.max_rows]u64 = undefined;
    var draws: [state.max_rows]sampler.Draw = undefined;
    for (0..rows) |r| positions[r] = first_position + r;
    try sampler.chooseGathered(e.gpa, got, e.world, rows, positions[0..rows], s, false, draws[0..rows]);
    for (out[0..rows], draws[0..rows]) |*o, dr| o.* = @intCast(dr.token);
}

pub fn drawRows(e: *Engine, logits_base: u64, cand_all: u64, total: usize, row0: usize, rows: usize, first_position: u64, s: ?lanes.Sampling, out: []u32) !void {
    const V = e.g.head_n;
    if (rows > state.max_rows) return error.WindowPastRows;
    const logits = logits_base + row0 * V * 2;
    if (e.world > 1) {
        if (!sampler.gatheredFits(s)) {
            if (s.?.top_k == 0) return e.nucleusRows(logits, rows, first_position, s.?, out);
            return mtp.tpSampleRows(&e.f, e.tp.?, logits, rows, V, null, first_position, s.?, out, null);
        }
        const all = try e.gpa.alloc(f32, e.world * total * (2 * state.cand + 1));
        defer e.gpa.free(all);
        try e.read(cand_all, std.mem.sliceAsBytes(all));
        return e.chooseFrom(all, total, row0, rows, first_position, s, out);
    }
    const ms = e.main_sample;
    if (s == null) {
        try e.f.th.argmax(logits, V, V, ms.col, rows);
        var ids: [state.max_rows]i32 = undefined;
        try e.read(ms.col, std.mem.sliceAsBytes(ids[0..rows]));
        for (out[0..rows], ids[0..rows]) |*o, x| o.* = @intCast(x);
        return;
    }
    if (s.?.top_k == 0) return e.nucleusRows(logits, rows, first_position, s.?, out);
    const k = mtp.topK(V, s.?);
    if (k > max_k) return error.TopKPastScratch;
    try e.f.th.sampleTopk(logits, rows, V, k, ms);
    const vals = try e.gpa.alloc(f32, rows * k);
    defer e.gpa.free(vals);
    const cols = try e.gpa.alloc(i64, rows * k);
    defer e.gpa.free(cols);
    try e.read(ms.vals, std.mem.sliceAsBytes(vals));
    try e.read(ms.idx, std.mem.sliceAsBytes(cols));
    const ids = try e.gpa.alloc(u64, k);
    defer e.gpa.free(ids);
    for (0..rows) |r| {
        for (ids, cols[r * k ..][0..k]) |*t, c| t.* = @intCast(c);
        const got = try sampler.drawRow(e.gpa, vals[r * k ..][0..k], ids, first_position + r, s, null);
        out[r] = @intCast(got.token);
    }
}

/// sampling.nucleus_rows (top_k off) over the main head's rows (this rank's vocabulary shard).
pub fn nucleusRows(e: *Engine, logits: u64, rows: usize, first_position: u64, s: lanes.Sampling, out: []u32) !void {
    var positions: [state.max_rows]u64 = undefined;
    for (0..rows) |r| positions[r] = first_position + r;
    try nucleus.sampleRows(&e.f, e.nsc, logits, rows, e.g.head_n, null, e.f.vocab_offset, positions[0..rows], s, out[0..rows], null);
}

// -- shared rounds: several streams' windows in one forward (Python forward's segs; MultiDecoder.round) --

pub const Window = fwd.Window;

/// One forward over every stream's window (row-wise layers once, state per stream), then sample, commit, draftMany.
pub fn verifyMany(e: *Engine, windows: []const Window) !void {
    if (windows.len == 0) return error.EmptyRound;
    try e.f.mark(.gap_verify);
    e.cand_round +%= 1;
    if (windows.len > e.round_segs.len) return error.TooManyStreams;
    if (windows.len == 1) {
        const keep = e.bound;
        e.bound = windows[0].seq;
        defer e.bound = keep;
        try e.forward(windows[0].tokens);
        e.round_segs[0] = .{ .seq = windows[0].seq, .a0 = 0, .a1 = windows[0].tokens.len };
        e.round_n = 1;
        return;
    }
    const R = try e.f.stageMany(windows, &e.buf, e.round_segs);
    e.round_n = windows.len;
    try e.runMulti(false, e.round_segs[0..windows.len], &e.buf);
    e.rows = R;
}

/// The target's draws of window `i`'s rows, keyed at its seq.pos + 1 + r with sampling `s` (its stream's).
pub fn sampleWindow(e: *Engine, i: usize, s: ?lanes.Sampling, out: []u32) !void {
    if (i >= e.round_n) return error.NoSuchWindow;
    const sg = e.round_segs[i];
    const smp: ?lanes.Sampling = if (s) |x| (if (x.temperature > 0) x else null) else null;
    const total = e.round_segs[e.round_n - 1].a1;
    if (e.world > 1 and e.draw_once and sampler.gatheredFits(smp)) {
        // the round's candidates read once (one wait), every window's draws from the host copy
        if (e.cand_have != e.cand_round) {
            try e.cand_host.resize(e.gpa, e.world * total * (2 * state.cand + 1));
            try e.read(e.buf.b.cand_all, std.mem.sliceAsBytes(e.cand_host.items));
            e.cand_have = e.cand_round;
        }
        return e.chooseFrom(e.cand_host.items, total, sg.a0, sg.a1 - sg.a0, sg.seq.st.pos + 1, smp, out);
    }
    try e.drawRows(e.buf.b.logits, e.buf.b.cand_all, total, sg.a0, sg.a1 - sg.a0, sg.seq.st.pos + 1, smp, out);
}

/// Keep window `i`'s first `keep` rows (its stream's commit, at its rows of the shared window).
pub fn commitWindow(e: *Engine, i: usize, keep: usize) !void {
    if (i >= e.round_n) return error.NoSuchWindow;
    const sg = e.round_segs[i];
    try e.f.commitAt(sg.seq, &e.buf, sg.a1 - sg.a0, keep, sg.a0);
}

/// Where a stream's absorb reads the main model's final streams: the prompt's last row or the window's first rows.
pub const DraftSource = union(enum) { prefill_last, window: usize };

/// One stream's MTP drafting (Python _draft_all): absorb `next`, chain up to `count` drafts into `out`, `drafted`.
pub const DraftReq = struct {
    seq: *Seq,
    source: DraftSource,
    next: []const u32,
    count: usize,
    sampling: ?lanes.Sampling = null,
    confidence: f64 = default_confidence,
    out: []u32,
    drafted: usize = 0,
};

/// MultiDecoder._draft_all: one absorb step for all streams, then a step a depth; drafts equal each solo chain's.
pub fn draftMany(e: *Engine, reqs: []DraftReq) !void {
    const mb = &(e.mbuf orelse return error.NoMtpHead);
    const d = &(e.draws orelse return error.NoMtpHead);
    if (reqs.len == 0) return;
    // profiling: the host gap before the head, then its time under the MTP head by draft level
    try e.f.mark(.gap_draft);
    if (e.f.prof) |p| {
        p.mtp = true;
        p.level = 0;
    }
    defer if (e.f.prof) |p| {
        p.mtp = false;
        p.level = 0;
    };
    if (reqs.len > e.round_segs.len) return error.TooManyStreams;
    const Wd = e.g.wide();
    var inputs: [64]mtp.Input = undefined;
    var segs: [64]fwd.Seg = undefined;
    var active: [64]struct { req: usize, row: usize } = undefined;
    if (reqs.len > inputs.len) return error.TooManyStreams;
    for (reqs, 0..) |*r, i| {
        r.drafted = 0;
        const st = &r.seq.st;
        if (st.mtp_drafted != 0) {
            try e.f.setMtpLen(st, st.mtp_len - st.mtp_drafted);
            st.mtp_drafted = 0;
        }
        const streams = switch (r.source) {
            .prefill_last => r.seq.last_streams.?.ptr,
            .window => |w| blk: {
                if (w >= e.round_n or e.round_segs[w].seq != r.seq) return error.NoSuchWindow;
                break :blk e.buf.b.streams + e.round_segs[w].a0 * Wd * 2;
            },
        };
        inputs[i] = .{ .seq = r.seq, .next = r.next, .streams = streams };
    }
    _ = try mtp.stageMany(&e.f, inputs[0..reqs.len], mb, &segs);
    try e.runMulti(true, segs[0..reqs.len], mb);
    const logits = mb.b.logits;
    var n_active: usize = 0;
    for (reqs, 0..) |*r, i| {
        try e.f.setMtpLen(&r.seq.st, r.seq.st.mtp_len + r.next.len);
        const room = @min(r.count, r.out.len);
        if (room > 0) {
            active[n_active] = .{ .req = i, .row = segs[i].a1 - 1 };
            n_active += 1;
        }
    }
    // each stream's stop rule (cuda_decode.Stop: per draft, running product or none), as its solo chain's
    var stops: [64]decode.Stop = undefined;
    for (reqs, 0..) |r, i| stops[i] = .{ .c = r.confidence };
    var lg = logits;
    var rows = reqs.len; // the head's rows of the step just run, active[k] at row k
    var row_of: [64]usize = undefined; // active k's row in that step
    for (0..n_active) |k| row_of[k] = active[k].req;
    var j: usize = 0;
    while (n_active > 0) : (j += 1) {
        var next: [64]struct { req: usize, row: usize, tok: u32 } = undefined;
        var n_next: usize = 0;
        // two ranks: the step's gathered candidates read once (one wait) for every stream's draw
        var cands: ?[]f32 = null;
        defer if (cands) |c| e.gpa.free(c);
        if (e.world > 1 and e.draw_once) {
            const c = try e.gpa.alloc(f32, e.world * rows * (2 * state.cand + 1));
            cands = c;
            try e.read(mb.b.cand_all, std.mem.sliceAsBytes(c));
        }
        for (active[0..n_active], 0..) |a, k| {
            const r = &reqs[a.req];
            const smp: ?lanes.Sampling = if (r.sampling) |x| (if (x.temperature > 0) x else null) else null;
            const got = if (cands != null and sampler.gatheredFits(smp))
                try d.gatheredFrom(cands.?, row_of[k], rows, r.seq.st.pos + 1 + j, smp)
            else
                try d.sampleDraftRow(&e.f, mb, lg, row_of[k], rows, r.seq.st.pos + 1 + j, smp);
            const t = stops[a.req].take(j, got.prob);
            if (!t.propose) continue;
            r.out[r.drafted] = @intCast(got.token);
            r.drafted += 1;
            if (t.more and j + 1 < @min(r.count, r.out.len)) {
                next[n_next] = .{ .req = a.req, .row = a.row, .tok = @intCast(got.token) };
                n_next += 1;
            }
        }
        try e.f.mark(.draft_sample);
        if (n_next == 0) break;
        if (e.f.prof) |p| p.level = @intCast(@min(j + 1, 255));
        var toks: [64][1]u32 = undefined;
        for (next[0..n_next], 0..) |nx, k| {
            toks[k] = .{nx.tok};
            inputs[k] = .{ .seq = reqs[nx.req].seq, .next = &toks[k], .streams = mb.b.streams + nx.row * Wd * 2 };
        }
        _ = try mtp.stageMany(&e.f, inputs[0..n_next], mb, &segs);
        try e.runMulti(true, segs[0..n_next], mb);
        lg = mb.b.logits;
        rows = n_next;
        for (next[0..n_next], 0..) |nx, k| {
            const st = &reqs[nx.req].seq.st;
            try e.f.setMtpLen(st, st.mtp_len + 1);
            st.mtp_drafted += 1;
            active[k] = .{ .req = nx.req, .row = segs[k].a0 };
            row_of[k] = k;
        }
        n_active = n_next;
    }
}

/// One request of a reference multi-stream run (generateMany) and the round it arrives at (later ones fill slices).
pub const Request = struct {
    prompt: []const u32,
    max_tokens: usize,
    sampling: ?lanes.Sampling = null,
    drafts: bool = true,
    /// the chains' confidence rule (Python's 0.70 for the oracle gates; the server's is confidenceSetting())
    confidence: f64 = default_confidence,
    stop_eos: bool = true,
    arrive: usize = 0,
    /// the prompt's images and video frames (null: text)
    media: ?*const media_mod.Media = null,
    tokens: std.ArrayList(u32) = .empty,
    rounds: usize = 0,
    accepted: usize = 0,
};

/// Seconds generateMany spent in each part of its last run (the stream synchronized at each boundary).
pub const Times = struct { prefill: f64 = 0, verify: f64 = 0, verify_host: f64 = 0, sample: f64 = 0, draft: f64 = 0, fill: f64 = 0, rounds: usize = 0 };

pub const ManyOptions = struct {
    /// layers a filling prompt runs between two rounds (0: an arriving prompt prefills at once)
    fill_layers: usize = 0,
    /// the prompts present at the start prefill together, one pass over each group that fits (prefillMany)
    burst: bool = false,
};

/// A request's stop rule this round: a running product (c < 0) only while at most product_streams are live.
pub fn hybrid(e: *const Engine, c: f64, live: []const bool) f64 {
    if (c >= 0) return c;
    var n: usize = 0;
    for (live) |l| n += @intFromBool(l);
    return if (n <= e.product_streams) c else wide_confidence;
}

pub fn generateMany(e: *Engine, gpa: Allocator, reqs: []Request) !void {
    return e.generateManyWith(gpa, reqs, .{});
}

/// Requests decoded in shared rounds (Python MultiDecoder.round order); each one's tokens equal its solo run's.
pub fn generateManyWith(e: *Engine, gpa: Allocator, reqs: []Request, mo: ManyOptions) !void {
    e.times = .{};
    var t = std.Io.Clock.awake.now(e.io);
    const n = reqs.len;
    if (n > e.round_segs.len) return error.TooManyStreams;
    const seqs = try gpa.alloc(?*Seq, n);
    defer gpa.free(seqs);
    @memset(seqs, null);
    defer for (seqs) |sq| if (sq) |x| e.freeSeq(x);
    const fills = try gpa.alloc(?*Fill, n);
    defer gpa.free(fills);
    @memset(fills, null);
    defer for (fills) |f| if (f) |x| e.fillFree(x);
    const counts = try gpa.alloc(usize, n);
    defer gpa.free(counts);
    const live = try gpa.alloc(bool, n);
    defer gpa.free(live);
    @memset(live, false);
    const finished = try gpa.alloc(bool, n);
    defer gpa.free(finished);
    @memset(finished, false);
    const drafts = try gpa.alloc([state.max_rows]u32, n);
    defer gpa.free(drafts);
    const n_drafts = try gpa.alloc(usize, n);
    defer gpa.free(n_drafts);
    @memset(n_drafts, 0);
    const keep_bound = e.bound;
    defer e.bound = keep_bound;
    var reqs_d: [64]DraftReq = undefined;
    var map: [64]usize = undefined;
    for (reqs, 0..) |*r, i| {
        const room = @as(i64, @intCast(e.max_len)) - @as(i64, @intCast(r.prompt.len)) - @as(i64, @intCast(e.depth)) - 1;
        if (room < 1) return error.PromptPastWindow;
        counts[i] = @max(1, @min(r.max_tokens, @as(usize, @intCast(room))));
        r.tokens.clearRetainingCapacity();
    }
    // a prompt's first draw: the stream goes live, its first drafts from the prompt's last row follow
    const Start = struct {
        fn first(eng: *Engine, gp: Allocator, rq: *Request, tok: u32, cnt: usize, lv: *bool) !void {
            try rq.tokens.append(gp, tok);
            lv.* = !((rq.stop_eos and eng.isEos(tok)) or cnt <= 1);
        }
    };
    var nd: usize = 0;
    const pre = try gpa.alloc(?u32, n);
    defer gpa.free(pre);
    @memset(pre, null);
    if (mo.burst) {
        // greedy groups of consecutive prompts of one head mode that fit one pass; a lone one prefills as usual
        var i: usize = 0;
        while (i < n) {
            if (reqs[i].media != null or !(reqs[i].arrive == 0 or mo.fill_layers == 0)) {
                i += 1;
                continue;
            }
            const absorbing = reqs[i].drafts and e.depth > 0;
            var items: [state.max_rows]ManyPrompt = undefined;
            var at: [state.max_rows]usize = undefined;
            var m: usize = 0;
            var rows: usize = 0;
            var j = i;
            while (j < n and m < state.max_rows and reqs[j].media == null and (reqs[j].arrive == 0 or mo.fill_layers == 0) and (reqs[j].drafts and e.depth > 0) == absorbing and rows + reqs[j].prompt.len <= e.prefill_rows) : (j += 1) {
                seqs[j] = try e.newSeq(e.max_len);
                items[m] = .{ .seq = seqs[j].?, .prompt = reqs[j].prompt, .sampling = reqs[j].sampling };
                at[m] = j;
                rows += reqs[j].prompt.len;
                m += 1;
            }
            if (m >= 2 and e.manyFit(items[0..m])) {
                var firsts: [state.max_rows]u32 = undefined;
                e.absorbing = absorbing;
                try e.prefillMany(items[0..m], &firsts);
                for (at[0..m], firsts[0..m]) |k, f| pre[k] = f;
            }
            i = @max(j, i + 1);
        }
    }
    for (reqs, 0..) |*r, i| if (r.arrive == 0 or mo.fill_layers == 0) {
        if (seqs[i] == null) seqs[i] = try e.newSeq(e.max_len);
        e.bound = seqs[i].?;
        e.absorbing = r.drafts and e.depth > 0;
        const first = if (pre[i]) |f| f else (try e.prefillWith(r.prompt, r.sampling, .{ .media = r.media })).first;
        try Start.first(e, gpa, r, first, counts[i], &live[i]);
        finished[i] = !live[i];
        if (live[i] and r.drafts and e.depth > 0) {
            reqs_d[nd] = .{ .seq = seqs[i].?, .source = .prefill_last, .next = r.tokens.items[0..1], .count = @min(e.depth, (counts[i] - 1) -| 1), .sampling = r.sampling, .confidence = e.hybrid(r.confidence, live), .out = &drafts[i] };
            map[nd] = i;
            nd += 1;
        }
    };
    try e.stream.synchronize();
    e.times.prefill += seconds(e.io, t);
    t = std.Io.Clock.awake.now(e.io);
    // TF_FLASHNEXT_PROFILE: the rounds' GPU time by part (eager: graphs off while profiling)
    const t_rounds = t;
    e.f.prof = e.prof;
    defer e.f.prof = null;
    if (e.prof) |p| p.n = 0;
    try e.draftMany(reqs_d[0..nd]);
    for (reqs_d[0..nd], map[0..nd]) |dr, i| n_drafts[i] = dr.drafted;
    var windows: [64]Window = undefined;
    var widx: [64]usize = undefined;
    var toks: [64][state.max_rows]u32 = undefined;
    var sampled: [state.max_rows]u32 = undefined;
    var kept: [64][state.max_rows]u32 = undefined;
    var keeps: [64]usize = undefined;
    var round: usize = 0;
    while (true) : (round += 1) {
        // arrivals: the oldest arrived prompt starts filling once no other fill holds the prompt buffers
        if (mo.fill_layers > 0) {
            var busy = false;
            for (fills) |f| busy = busy or f != null;
            if (!busy) for (reqs, 0..) |*r, i| if (r.arrive > 0 and r.arrive <= round and seqs[i] == null) {
                seqs[i] = try e.newSeq(e.max_len);
                e.absorbing = r.drafts and e.depth > 0;
                fills[i] = try e.fillBegin(seqs[i].?, r.prompt, r.sampling, .{ .media = r.media });
                break;
            };
        }
        var nw: usize = 0;
        for (reqs, 0..) |r, i| if (live[i]) {
            toks[i][0] = r.tokens.items[r.tokens.items.len - 1];
            @memcpy(toks[i][1 .. 1 + n_drafts[i]], drafts[i][0..n_drafts[i]]);
            windows[nw] = .{ .seq = seqs[i].?, .tokens = toks[i][0 .. 1 + n_drafts[i]] };
            widx[nw] = i;
            nw += 1;
        };
        var waiting = false;
        for (reqs, 0..) |_, i| waiting = waiting or (!finished[i] and !live[i]);
        if (nw == 0 and !waiting) break;
        if (nw > 0) {
            try e.stream.synchronize();
            e.times.draft += seconds(e.io, t);
            t = std.Io.Clock.awake.now(e.io);
            try e.verifyMany(windows[0..nw]);
            e.times.verify_host += seconds(e.io, t);
            try e.stream.synchronize();
            e.times.verify += seconds(e.io, t);
            t = std.Io.Clock.awake.now(e.io);
            e.times.rounds += 1;
            for (0..nw) |w| {
                const i = widx[w];
                const r = &reqs[i];
                const rows = windows[w].tokens.len;
                try e.sampleWindow(w, r.sampling, sampled[0..rows]);
                try e.f.mark(.sample);
                var keep: usize = 1;
                for (windows[w].tokens[1..rows], 0..) |dt, k| {
                    if (sampled[k] != dt or (r.stop_eos and e.isEos(sampled[k]))) break;
                    keep += 1;
                }
                try e.commitWindow(w, keep);
                try e.f.mark(.commit);
                @memcpy(kept[i][0..keep], sampled[0..keep]);
                keeps[i] = keep;
                r.rounds += 1;
                r.accepted += keep - 1;
                try r.tokens.appendSlice(gpa, sampled[0..keep]);
                const last = r.tokens.items[r.tokens.items.len - 1];
                if (r.tokens.items.len >= counts[i] or (r.stop_eos and e.isEos(last))) {
                    live[i] = false;
                    finished[i] = true;
                }
            }
            try e.stream.synchronize();
            if (e.f.prof) |p| try p.drain();
            e.times.sample += seconds(e.io, t);
            t = std.Io.Clock.awake.now(e.io);
        }
        nd = 0;
        for (0..nw) |w| {
            const i = widx[w];
            n_drafts[i] = 0;
            const r = &reqs[i];
            if (!live[i] or !r.drafts or e.depth == 0) continue;
            // room 0 (one token left): the head still absorbs the kept rows, drafting nothing
            const room = @min(e.depth, (counts[i] - r.tokens.items.len) -| 1);
            reqs_d[nd] = .{ .seq = seqs[i].?, .source = .{ .window = w }, .next = kept[i][0..keeps[i]], .count = room, .sampling = r.sampling, .confidence = e.hybrid(r.confidence, live), .out = &drafts[i] };
            map[nd] = i;
            nd += 1;
        }
        // filling prompts: a slice each; one that ends drafts from its last row with the others
        const tf = std.Io.Clock.awake.now(e.io);
        for (reqs, 0..) |*r, i| if (fills[i]) |fl| {
            if (!try e.fillStep(fl, mo.fill_layers)) continue;
            try Start.first(e, gpa, r, fl.first, counts[i], &live[i]);
            finished[i] = !live[i];
            e.fillFree(fl);
            fills[i] = null;
            n_drafts[i] = 0;
            if (live[i] and r.drafts and e.depth > 0) {
                reqs_d[nd] = .{ .seq = seqs[i].?, .source = .prefill_last, .next = r.tokens.items[0..1], .count = @min(e.depth, (counts[i] - 1) -| 1), .sampling = r.sampling, .confidence = e.hybrid(r.confidence, live), .out = &drafts[i] };
                map[nd] = i;
                nd += 1;
            }
        };
        try e.stream.synchronize();
        e.times.fill += seconds(e.io, tf);
        try e.draftMany(reqs_d[0..nd]);
        for (reqs_d[0..nd], map[0..nd]) |dr, i| n_drafts[i] = dr.drafted;
    }
    for (reqs, counts) |*r, c| r.tokens.shrinkRetainingCapacity(@min(r.tokens.items.len, c));
    if (e.prof) |p| {
        try e.stream.synchronize();
        try p.rounds(e.rank, "decode", e.times.rounds, seconds(e.io, t_rounds) * 1000.0);
        for (&e.f.experts_seen, [_][]const u8{ "main", "MTP head" }) |*c, what| {
            if (c.layers > 0) std.debug.print("rounds rank {d} decode   {s}: {d:.1} distinct routed experts a MoE layer over {d:.1} rows\n", .{ e.rank, what, @as(f64, @floatFromInt(c.distinct)) / @as(f64, @floatFromInt(c.layers)), @as(f64, @floatFromInt(c.rows)) / @as(f64, @floatFromInt(c.layers)) });
            c.* = .{};
        }
    }
}
