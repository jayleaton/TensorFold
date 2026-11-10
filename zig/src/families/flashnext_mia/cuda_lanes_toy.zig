//! Flash Next lanes' toy engine for tests: several sequences, a hashed target and a head that guesses it mostly.
const std = @import("std");
const lanes = @import("lanes");
const media_mod = @import("cuda_media.zig");
const decode = @import("cuda_decode.zig");
const cl = @import("cuda_lanes.zig");

const max_rows = cl.max_rows;
const max_streams = cl.max_streams;
const default_confidence = cl.default_confidence;

/// Toy target: draws from a hash of the sequence, window rows and position; the head guesses it right mostly.
pub const Toy = struct {
    pub const Seq = struct {
        tokens: std.ArrayList(u32) = .empty,
        sampling: ?lanes.Sampling = null,
        head: std.ArrayList(u32) = .empty, // the committed tokens the head absorbed, then its chain
        absorbed: usize = 0,
        limit: usize,
    };

    pub const Window = struct { seq: *Seq, tokens: []const u32 };
    pub const DraftSource = union(enum) { prefill_last, window: usize };
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
    const Held = struct { seq: *Seq, tokens: [max_rows]u32, n: usize };

    gpa: std.mem.Allocator,
    stream: ?*anyopaque = null, // the engine's stream (cuda_native's records run on it)
    budget: struct { limit: usize } = .{ .limit = std.math.maxInt(usize) },
    round_windows: [max_streams]Held = undefined,
    round_n: usize = 0,
    rounds: usize = 0,
    widest: usize = 0,
    snaps: usize = 0,
    passes: usize = 0,
    resumed: usize = 0,
    fills: usize = 0,
    bound: ?*Seq = null,
    window: [max_rows]u32 = undefined,
    rows: usize = 0,
    seqs: usize = 0,
    forwards: usize = 0,
    head_err: u64 = 1,

    pub fn next(done: []const u32, extra: []const u32, s: ?lanes.Sampling, position: u64) u32 {
        var h = std.hash.Wyhash.init(11);
        h.update(std.mem.sliceAsBytes(done));
        h.update(std.mem.sliceAsBytes(extra));
        if (s) |smp| {
            h.update(std.mem.asBytes(&smp.seed));
            h.update(std.mem.asBytes(&position));
        }
        return @intCast(h.final() % 50);
    }

    /// As cuda_engine's Engine.init, for cuda_native.openOn's tests.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: anytype, dir: []const u8, kernels: []const u8, o: anytype) !*Toy {
        _ = .{ io, ctx, dir, kernels, o.context, o.mtp, o.rank, o.world, o.comm, o.yarn, o.depth, o.streams };
        const t = try gpa.create(Toy);
        t.* = .{ .gpa = gpa };
        return t;
    }
    pub fn deinit(t: *Toy) void {
        t.gpa.destroy(t);
    }

    /// A sequence's caches at `rows` rows (cuda_engine's seqBytes): 100 bytes a row here.
    pub fn seqBytes(_: *const Toy, rows: usize) usize {
        return rows * 100;
    }

    pub fn newSeq(t: *Toy, limit: usize) !*Seq {
        const s = try t.gpa.create(Seq);
        s.* = .{ .limit = limit };
        t.seqs += 1;
        return s;
    }
    pub fn freeSeq(t: *Toy, s: *Seq) void {
        if (t.bound == s) t.bound = null;
        s.tokens.deinit(t.gpa);
        s.head.deinit(t.gpa);
        t.gpa.destroy(s);
        t.seqs -= 1;
    }
    pub fn bind(t: *Toy, s: *Seq) void {
        t.bound = s;
    }
    fn seq(t: *Toy) *Seq {
        return t.bound.?;
    }
    pub fn prefill(t: *Toy, prompt: []const u32, s: ?lanes.Sampling) !u32 {
        return (try t.prefillWith(prompt, s, .{})).first;
    }

    // -- kept prompt states (cuda_engine's Snapshot, prefillWith, copyPrefix, freeSnapshot) --

    pub const Snapshot = struct { pos: usize, mtp_len: usize, tokens: []u32 };
    pub const PrefillOptions = struct { keep_at: ?usize = null, resume_from: ?*const Snapshot = null, media: ?*const media_mod.Media = null };
    pub const Prefilled = struct { first: u32, kept: ?*Snapshot = null };

    /// The prompt from 0 or a kept state, keeping keep_at's state; first draw; the head holds all but the last row.
    pub fn prefillWith(t: *Toy, prompt: []const u32, s: ?lanes.Sampling, o: PrefillOptions) !Prefilled {
        const q = t.seq();
        if (prompt.len == 0 or prompt.len > q.limit) return error.ToyMisuse;
        if (o.resume_from) |snap| {
            if (snap.pos >= prompt.len or !std.mem.eql(u32, snap.tokens, prompt[0..snap.pos])) return error.ResumeOtherPrompt;
            if (q.tokens.items.len < snap.pos or !std.mem.eql(u32, q.tokens.items[0..snap.pos], prompt[0..snap.pos])) return error.CachesNotThePrompts;
            q.tokens.shrinkRetainingCapacity(snap.pos);
            t.resumed += 1;
        } else q.tokens.clearRetainingCapacity();
        q.sampling = s;
        try q.tokens.appendSlice(t.gpa, prompt[q.tokens.items.len..]);
        // the head takes every prompt row but the last (its row waits for the first token)
        q.head.clearRetainingCapacity();
        try q.head.appendSlice(t.gpa, prompt[0 .. prompt.len - 1]);
        q.absorbed = prompt.len - 1;
        t.rows = 0;
        var kept: ?*Snapshot = null;
        if (o.keep_at) |k| {
            if (k == 0 or k > prompt.len) return error.BadKeepPoint;
            const snap = try t.gpa.create(Snapshot);
            snap.* = .{ .pos = k, .mtp_len = k - 1, .tokens = try t.gpa.dupe(u32, prompt[0..k]) };
            t.snaps += 1;
            kept = snap;
        }
        t.passes += 1;
        return .{ .first = next(q.tokens.items, &.{}, s, prompt.len), .kept = kept };
    }

    /// cuda_engine's Fill: a prompt filling in layer slices; here `left` layers, then prefillWith's effect.
    pub const Fill = struct {
        seq: *Seq,
        prompt: []u32,
        sampling: ?lanes.Sampling,
        opts: PrefillOptions,
        left: usize = 5,
        kept: ?*Snapshot = null,
        first: u32 = 0,
        done: bool = false,
    };

    pub fn fillBegin(t: *Toy, q: *Seq, prompt: []const u32, sampling: ?lanes.Sampling, o: PrefillOptions) !*Fill {
        if (prompt.len == 0 or prompt.len > q.limit) return error.ToyMisuse;
        const fl = try t.gpa.create(Fill);
        fl.* = .{ .seq = q, .prompt = try t.gpa.dupe(u32, prompt), .sampling = sampling, .opts = o };
        t.fills += 1;
        return fl;
    }

    pub fn fillStep(t: *Toy, fl: *Fill, layers: usize) !bool {
        if (fl.done) return true;
        fl.left -|= layers;
        if (fl.left > 0) return false;
        // the slices' effect: the same pass prefillWith makes, on this fill's sequence
        const keep = t.bound;
        t.bound = fl.seq;
        defer t.bound = keep;
        const r = try t.prefillWith(fl.prompt, fl.sampling, fl.opts);
        fl.kept = r.kept;
        fl.first = r.first;
        fl.done = true;
        return true;
    }

    pub fn fillFree(t: *Toy, fl: *Fill) void {
        if (fl.kept) |k| t.freeSnapshot(k);
        t.gpa.free(fl.prompt);
        t.gpa.destroy(fl);
        t.fills -= 1;
    }

    pub fn copyPrefix(t: *Toy, dst: *Seq, src: *Seq, rows: usize, mtp_rows: usize) !void {
        _ = mtp_rows;
        if (rows > src.tokens.items.len) return error.PrefixPastSource;
        dst.tokens.clearRetainingCapacity();
        try dst.tokens.appendSlice(t.gpa, src.tokens.items[0..rows]);
    }

    pub fn freeSnapshot(t: *Toy, snap: *Snapshot) void {
        t.gpa.free(snap.tokens);
        t.gpa.destroy(snap);
        t.snaps -= 1;
    }
    pub fn pos(t: *Toy) u64 {
        return t.seq().tokens.items.len;
    }
    pub fn forward(t: *Toy, tokens: []const u32) !void {
        if (t.seq().tokens.items.len + tokens.len > t.seq().limit) return error.SeqFull;
        @memcpy(t.window[0..tokens.len], tokens);
        t.rows = tokens.len;
        t.forwards += 1;
    }
    pub fn sample(t: *Toy, rows: usize, first: u64, out: []u32) !void {
        const q = t.seq();
        if (first != q.tokens.items.len + 1 or rows != t.rows) return error.ToyMisuse;
        for (0..rows) |i| out[i] = next(q.tokens.items, t.window[0 .. i + 1], q.sampling, first + i);
    }
    pub fn commit(t: *Toy, rows: usize, keep: usize) !void {
        if (keep < 1 or keep > rows or rows != t.rows) return error.ToyMisuse;
        try t.seq().tokens.appendSlice(t.gpa, t.window[0..keep]);
        t.rows = 0;
    }
    pub fn absorb(t: *Toy, source: decode.Source, next_tokens: []const u32) !void {
        const q = t.seq();
        // the head drops its last chain and must take exactly the rows committed since it last absorbed
        q.head.shrinkRetainingCapacity(q.absorbed);
        const behind = q.tokens.items.len - q.absorbed;
        if (behind != next_tokens.len) return error.HeadOutOfStep;
        if (source == .prefill_last and behind != 1) return error.HeadOutOfStep;
        try q.head.appendSlice(t.gpa, q.tokens.items[q.absorbed..]);
        q.absorbed = q.tokens.items.len;
        // the head's input after the kept rows is the token the target drew after the last of them
        try q.head.append(t.gpa, next_tokens[next_tokens.len - 1]);
    }
    pub fn chain(t: *Toy, token: u32, _: usize) !void {
        try t.seq().head.append(t.gpa, token);
    }
    // -- the shared rounds (cuda_engine's verifyMany / sampleWindow / commitWindow / draftMany) --

    pub fn verifyMany(t: *Toy, windows: []const Window) !void {
        if (windows.len == 0 or windows.len > max_streams) return error.ToyMisuse;
        for (windows, 0..) |w, i| {
            if (w.seq.tokens.items.len + w.tokens.len > w.seq.limit) return error.SeqFull;
            t.round_windows[i] = .{ .seq = w.seq, .tokens = undefined, .n = w.tokens.len };
            @memcpy(t.round_windows[i].tokens[0..w.tokens.len], w.tokens);
        }
        t.round_n = windows.len;
        t.rounds += 1;
        t.forwards += 1;
        t.widest = @max(t.widest, windows.len);
    }
    pub fn sampleWindow(t: *Toy, i: usize, s: ?lanes.Sampling, out: []u32) !void {
        if (i >= t.round_n) return error.ToyMisuse;
        const w = t.round_windows[i];
        const done = w.seq.tokens.items;
        for (0..w.n) |r| out[r] = next(done, w.tokens[0 .. r + 1], s, done.len + 1 + r);
    }
    pub fn commitWindow(t: *Toy, i: usize, keep: usize) !void {
        if (i >= t.round_n) return error.ToyMisuse;
        const w = &t.round_windows[i];
        if (keep < 1 or keep > w.n) return error.ToyMisuse;
        try w.seq.tokens.appendSlice(t.gpa, w.tokens[0..keep]);
        w.n = 0; // committed once
    }
    pub fn draftMany(t: *Toy, reqs: []DraftReq) !void {
        for (reqs) |*r| {
            const q = r.seq;
            q.head.shrinkRetainingCapacity(q.absorbed);
            const behind = q.tokens.items.len - q.absorbed;
            if (behind != r.next.len) return error.HeadOutOfStep;
            switch (r.source) {
                .prefill_last => if (behind != 1) return error.HeadOutOfStep,
                .window => |w| if (w >= t.round_n or t.round_windows[w].seq != q) return error.ToyMisuse,
            }
            try q.head.appendSlice(t.gpa, q.tokens.items[q.absorbed..]);
            q.absorbed = q.tokens.items.len;
            try q.head.append(t.gpa, r.next[r.next.len - 1]);
            r.drafted = 0;
            const pos0 = q.tokens.items.len + 1;
            var stop: decode.Stop = .{ .c = r.confidence };
            for (0..@min(r.count, r.out.len)) |j| {
                const g = t.guessOn(q, r.sampling, pos0 + j);
                const take = stop.take(j, g.prob);
                if (!take.propose) break;
                r.out[r.drafted] = g.token;
                r.drafted += 1;
                if (!take.more) break;
                try q.head.append(t.gpa, g.token);
            }
        }
    }

    fn guess(t: *Toy, position: u64) decode.Draw {
        return t.guessOn(t.seq(), t.seq().sampling, position);
    }

    fn guessOn(t: *Toy, q: *Seq, sampling: ?lanes.Sampling, position: u64) decode.Draw {
        const h_items = q.head.items;
        const want = next(h_items[0..q.absorbed], h_items[q.absorbed..], sampling, position);
        var h = std.hash.Wyhash.init(t.head_err);
        h.update(std.mem.asBytes(&position));
        h.update(std.mem.sliceAsBytes(h_items[q.absorbed..]));
        const r = h.final() % 10;
        return if (r < 7) .{ .token = want, .prob = if (r == 0) 0.6 else 0.9 } else .{ .token = (want + 1) % 50, .prob = if (r == 9) 0.5 else 0.8 };
    }
    pub fn sampleDraft(t: *Toy, position: u64) !decode.Draw {
        return t.guess(position);
    }
    pub fn drawDraft(t: *Toy, position: u64) !u32 {
        return t.guess(position).token;
    }
    pub fn isEos(_: *Toy, token: u32) bool {
        return token == 49;
    }
};
