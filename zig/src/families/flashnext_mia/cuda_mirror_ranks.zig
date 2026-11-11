//! The two ranks: rank 0's Leader records each engine call into batches, rank 1's Follower replays them.
const std = @import("std");
const lanes = @import("lanes");
const media_mod = @import("cuda_media.zig");
const be = lanes.backend;
const Stream = lanes.Stream;
const Allocator = std.mem.Allocator;
const codec = @import("cuda_mirror.zig");
const Call = codec.Call;
const StreamSpec = codec.StreamSpec;
const WindowCall = codec.WindowCall;
const DraftCall = codec.DraftCall;
const keyOf = codec.keyOf;
const decode = codec.decode;
const Batch = codec.Batch;
const Calls = codec.Calls;
const Channel = codec.Channel;
const Watch = @import("cuda_mirror_watch.zig").Watch;

fn windowCalls(a: Allocator, ws: []const be.Window) ![]WindowCall {
    const out = try a.alloc(WindowCall, ws.len);
    for (out, ws) |*o, w| o.* = .{
        .key = keyOf(w.stream),
        .pending = w.pending,
        .held = w.held,
        .tokens = w.tokens,
        .parents = w.parents,
        .positions = w.positions,
        .early = w.early,
    };
    return out;
}

// -- rank 0 -------------------------------------------------------------------------------------------------------

/// Rank 0's backend: each call goes to rank 1 first, then `inner`; non-collective calls batch with the next one.
pub const Leader = struct {
    gpa: Allocator,
    inner: be.Backend,
    channel: Channel,
    batch: Batch = .{},
    arena: std.heap.ArenaAllocator,
    vtable: be.Backend.VTable = undefined,
    open: std.AutoHashMapUnmanaged(u64, void) = .empty,
    watch: ?*Watch = null,
    sends: u64 = 0,
    calls: u64 = 0,
    stopped: bool = false,
    /// a send failed: rank 1 is gone, so every later call fails rather than run out of step
    broken: bool = false,

    pub fn init(gpa: Allocator, inner: be.Backend, channel: Channel) Leader {
        return .{ .gpa = gpa, .inner = inner, .channel = channel, .arena = .init(gpa) };
    }

    pub fn deinit(self: *Leader) void {
        self.stop();
        self.batch.deinit(self.gpa);
        self.open.deinit(self.gpa);
        self.arena.deinit();
    }

    /// Rank 1 leaves its loop (once, after the calls still in the batch; the channel stays open for the caller).
    pub fn stop(self: *Leader) void {
        if (self.stopped) return;
        if (!self.broken) {
            self.flush() catch {};
            self.channel.stop() catch {};
        }
        self.stopped = true;
    }

    fn flush(self: *Leader) !void {
        if (self.batch.count == 0) return;
        defer self.batch.clear();
        self.sends += 1;
        self.channel.send(self.batch.bytes()) catch |e| {
            self.broken = true;
            return e;
        };
    }

    /// `call` into the batch; a call with collectives sends the batch now (rank 1 must join them).
    fn send(self: *Leader, call: Call, collective: bool) !void {
        if (self.broken or self.stopped) return error.FollowerGone;
        try self.batch.add(self.gpa, call);
        self.calls += 1;
        if (collective) try self.flush();
    }

    /// The lane backend's notes to rank 1 (cuda_lanes.Notes): each rides with the next call that has collectives.
    pub fn notes(self: *Leader) @import("cuda_lanes.zig").Notes {
        return .{ .ptr = self, .send = noteFn };
    }

    fn noteFn(p: *anyopaque, bytes: []const u8) void {
        const self = of(p);
        self.send(.{ .note = bytes }, false) catch |e| std.log.err("rank 0: a prompt-cache note not sent to rank 1 ({s})", .{@errorName(e)});
    }

    fn arm(self: *Leader, seconds: u64) void {
        if (self.watch) |w| w.arm(@max(seconds, w.base_s));
    }

    fn disarm(self: *Leader) void {
        if (self.watch) |w| w.disarm();
    }

    fn scratch(self: *Leader) Allocator {
        _ = self.arena.reset(.retain_capacity);
        return self.arena.allocator();
    }

    pub fn backend(self: *Leader) be.Backend {
        const v = self.inner.vtable;
        self.vtable = .{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .unspeculate = if (v.unspeculate != null) unspeculateFn else null,
            .probabilities = if (v.probabilities != null) probabilitiesFn else null,
            .tree = if (v.tree != null) treeFn else null,
            .alternatives = if (v.alternatives != null) alternativesFn else null,
            .prefill_step = if (v.prefill_step != null) prefillStepFn else null,
            .prefill_many = if (v.prefill_many != null) prefillManyFn else null,
            .release = releaseFn,
        };
        const i = self.inner; // f212's backend flags: the leader serves what its lanes do
        return .{ .ptr = self, .vtable = &self.vtable, .first_before_draft = i.first_before_draft, .penalties = i.penalties, .chunk_invariant = i.chunk_invariant, .open_fills = i.open_fills };
    }

    fn of(p: *anyopaque) *Leader {
        return @ptrCast(@alignCast(p));
    }

    fn specOf(s: *Stream) StreamSpec {
        return .{
            .key = keyOf(s),
            .max_new = s.max_new,
            .drafts = s.drafts,
            .sampling = s.sampling,
            .prompt = s.prompt(),
            .eos = s.eos,
            .chunks = s.chunks,
            .resume_key = if (s.reuse.saved) |saved| @intFromPtr(saved) else 0,
            .resume_at = s.reuse.at,
            .marks = s.reuse.marks,
            .media = if (media_mod.of(s)) |m| .{ .rows = @intCast(m.rowCount()), .delta = m.delta, .width = m.width } else null,
        };
    }

    /// f212's prefill_step: the first step sends the spec (rank 1 mirrors it), each step its key, then both step.
    fn prefillStepFn(p: *anyopaque, s: *Stream, max_rows: u32) anyerror!bool {
        const self = of(p);
        if (!self.open.contains(keyOf(s))) {
            try self.open.put(self.gpa, keyOf(s), {});
            try self.send(.{ .prefill_begin = specOf(s) }, false);
        }
        const keys = try self.scratch().alloc(u64, 1);
        keys[0] = keyOf(s);
        try self.send(.{ .prefill_step = keys }, true);
        self.arm(0);
        defer self.disarm();
        return self.inner.vtable.prefill_step.?(self.inner.ptr, s, max_rows);
    }

    fn prefillManyFn(p: *anyopaque, ss: []const *Stream) anyerror!bool {
        const self = of(p);
        // an image or video prompt prefills alone: its table and features go by broadcast inside its own prefill
        for (ss) |s| if (media_mod.of(s) != null) return false;
        const specs = try self.scratch().alloc(StreamSpec, ss.len);
        var seconds: usize = 0;
        for (specs, ss) |*sp, s| {
            try self.open.put(self.gpa, keyOf(s), {});
            sp.* = specOf(s);
            seconds += s.prompt_len;
        }
        try self.send(.{ .prefill_many = specs }, true);
        self.arm(seconds / 400);
        defer self.disarm();
        return self.inner.vtable.prefill_many.?(self.inner.ptr, ss);
    }

    fn prefillFn(p: *anyopaque, s: *Stream) anyerror!void {
        const self = of(p);
        try self.open.put(self.gpa, keyOf(s), {});
        try self.send(.{ .prefill = specOf(s) }, true);
        // a prompt's chunks: 300 s, or 2.5 ms a token past that (a 1M-token prompt)
        self.arm(s.prompt_len / 400);
        defer self.disarm();
        return self.inner.vtable.prefill(self.inner.ptr, s);
    }

    fn firstFn(p: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const self = of(p);
        try self.send(.{ .first = .{ .key = keyOf(s), .position = position } }, false);
        return self.inner.vtable.first(self.inner.ptr, s, position);
    }

    fn queueFn(p: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const self = of(p);
        try self.send(.{ .queue = .{ .key = keyOf(s), .feed = feed, .position = position } }, true);
        self.arm(0);
        defer self.disarm();
        return self.inner.vtable.queue(self.inner.ptr, s, feed, position);
    }

    fn readFn(p: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(p);
        return self.inner.vtable.read(self.inner.ptr, handle);
    }

    fn verifyFn(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(p);
        try self.send(.{ .verify = try windowCalls(self.scratch(), windows) }, true);
        self.arm(0);
        defer self.disarm();
        return self.inner.vtable.verify(self.inner.ptr, windows, out);
    }

    fn keepFn(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(p);
        try self.send(.{ .keep = .{ .windows = try windowCalls(self.scratch(), windows), .paths = paths } }, false);
        return self.inner.vtable.keep(self.inner.ptr, windows, paths);
    }

    fn draftFn(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(p);
        const a = self.scratch();
        const calls = try a.alloc(DraftCall, requests.len);
        for (calls, requests) |*c, r| {
            if (r.lanes != null) return error.TreesNotMirrored;
            c.* = .{ .key = keyOf(r.stream), .follow = r.follow, .first = r.first, .rows = r.rows, .start = r.start, .position = r.position, .depth = r.depth, .early = r.early, .ranks = r.ranks };
        }
        try self.send(.{ .draft = calls }, true);
        self.arm(0);
        defer self.disarm();
        return self.inner.vtable.draft(self.inner.ptr, requests);
    }

    fn unspeculateFn(p: *anyopaque, s: *Stream) anyerror!void {
        const self = of(p);
        try self.send(.{ .unspeculate = keyOf(s) }, true);
        self.arm(0);
        defer self.disarm();
        return self.inner.vtable.unspeculate.?(self.inner.ptr, s);
    }

    fn probabilitiesFn(p: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const self = of(p);
        return self.inner.vtable.probabilities.?(self.inner.ptr, s, out);
    }

    fn treeFn(p: *anyopaque, s: *Stream, gpa: Allocator) anyerror!?lanes.stream.Held {
        const self = of(p);
        return self.inner.vtable.tree.?(self.inner.ptr, s, gpa);
    }

    fn alternativesFn(p: *anyopaque, s: *Stream, out: []be.Alternative) anyerror!usize {
        const self = of(p);
        return self.inner.vtable.alternatives.?(self.inner.ptr, s, out);
    }

    fn releaseFn(p: *anyopaque, s: *Stream) void {
        const self = of(p);
        self.send(.{ .release = keyOf(s) }, false) catch |e| std.log.err("rank 0: release not sent to rank 1 ({s})", .{@errorName(e)});
        self.inner.vtable.release(self.inner.ptr, s);
        // the last open stream: rank 1 waits on the link until the next request
        _ = self.open.remove(keyOf(s));
        if (self.open.count() == 0 and !self.broken and !self.stopped) {
            self.flush() catch |e| std.log.err("rank 0: release not sent to rank 1 ({s})", .{@errorName(e)});
            self.channel.idle() catch |e| {
                self.broken = true;
                std.log.err("rank 0: idle not sent to rank 1 ({s})", .{@errorName(e)});
            };
        }
    }
};

// -- rank 1 -------------------------------------------------------------------------------------------------------

/// Rank 1's copy of a stream: what a backend reads of it, owned here.
const Mirrored = struct {
    stream: Stream,
    eos: []u32,
    chunks: []u32,
    marks: []u32,
    /// rank 0's media shape (its rows, rotary table and features reach this rank by broadcast)
    media: ?*media_mod.Media = null,
};

/// Rank 1's loop: rank 0's calls on `inner`, until a `stop` frame.
pub const Follower = struct {
    gpa: Allocator,
    inner: be.Backend,
    channel: Channel,
    streams: std.AutoHashMapUnmanaged(u64, *Mirrored) = .empty,
    arena: std.heap.ArenaAllocator,
    calls: u64 = 0,
    /// calls that failed here (rank 0 met the same failure on the same arguments)
    failed: u64 = 0,
    watch: ?*Watch = null,
    /// the lane backend's side of rank 0's notes and kept states (cuda_lanes: applyNote, savedFor)
    family: ?Family = null,

    pub const Family = struct {
        ptr: *anyopaque,
        note: *const fn (ptr: *anyopaque, bytes: []const u8) anyerror!void,
        saved: *const fn (ptr: *anyopaque, key: u64) ?*anyopaque,
    };

    pub fn init(gpa: Allocator, inner: be.Backend, channel: Channel) Follower {
        return .{ .gpa = gpa, .inner = inner, .channel = channel, .arena = .init(gpa) };
    }

    /// Releases the streams rank 0 left open.
    pub fn deinit(self: *Follower) void {
        var it = self.streams.iterator();
        while (it.next()) |kv| {
            self.inner.release(&kv.value_ptr.*.stream);
            self.free(kv.value_ptr.*);
        }
        self.streams.deinit(self.gpa);
        self.arena.deinit();
    }

    fn free(self: *Follower, m: *Mirrored) void {
        m.stream.deinit(self.gpa);
        self.gpa.free(m.eos);
        self.gpa.free(m.chunks);
        self.gpa.free(m.marks);
        if (m.media) |md| self.gpa.destroy(md);
        self.gpa.destroy(m);
    }

    /// Batches until rank 0 stops (returns) or the channel fails (its error: rank 0 is gone).
    pub fn run(self: *Follower) !void {
        while (try self.channel.recv()) |bytes| {
            var calls = try Calls.of(bytes);
            while (try calls.next()) |call| try self.apply(call);
        }
    }

    /// One call's bytes; a failure is logged and the loop goes on (rank 0 failed the same call too).
    pub fn apply(self: *Follower, bytes: []const u8) !void {
        _ = self.arena.reset(.retain_capacity);
        const call = try decode(self.arena.allocator(), bytes);
        self.calls += 1;
        if (self.watch) |w| w.arm(w.base_s + switch (call) {
            .prefill => |p| p.prompt.len / 400,
            .prefill_many => |ps| blk: {
                var rows: usize = 0;
                for (ps) |p| rows += p.prompt.len;
                break :blk rows / 400;
            },
            else => 0,
        });
        defer if (self.watch) |w| w.disarm();
        self.make(call) catch |e| {
            self.failed += 1;
            std.log.warn("rank 1: {s} failed ({s}), as on rank 0", .{ @tagName(call), @errorName(e) });
        };
    }

    fn stream(self: *Follower, key: u64) !*Stream {
        const m = self.streams.get(key) orelse return error.UnknownStream;
        return &m.stream;
    }

    fn windows(self: *Follower, a: Allocator, ws: []const WindowCall) ![]be.Window {
        const out = try a.alloc(be.Window, ws.len);
        for (out, ws) |*o, w| o.* = .{ .stream = try self.stream(w.key), .pending = w.pending, .held = w.held, .tokens = w.tokens, .parents = w.parents, .positions = w.positions, .early = w.early };
        return out;
    }

    /// Rank 1's own stream for rank 0's prefill; a stream prefilled again keeps its backend lane, as on rank 0.
    fn mirror(self: *Follower, p: StreamSpec) !*Stream {
        const m = try self.gpa.create(Mirrored);
        errdefer self.gpa.destroy(m);
        m.eos = try self.gpa.dupe(u32, p.eos);
        errdefer self.gpa.free(m.eos);
        m.chunks = try self.gpa.dupe(u32, p.chunks);
        errdefer self.gpa.free(m.chunks);
        m.marks = try self.gpa.dupe(u32, p.marks);
        errdefer self.gpa.free(m.marks);
        // rank 1's own copy of the state rank 0 resumes (none held: the backend agrees to start cold)
        const saved = if (p.resume_key != 0) (if (self.family) |f| f.saved(f.ptr, p.resume_key) else null) else null;
        m.media = null;
        errdefer if (m.media) |md| self.gpa.destroy(md);
        if (p.media) |pm| {
            const md = try self.gpa.create(media_mod.Media);
            md.* = .{ .rows = &.{}, .positions = &.{}, .delta = pm.delta, .features = &.{}, .width = pm.width, .follower_rows = pm.rows };
            m.media = md;
        }
        m.stream = try Stream.init(self.gpa, .{ .id = "rank0", .prompt = p.prompt, .max_new = p.max_new, .eos = m.eos, .sampling = p.sampling, .drafts = p.drafts, .chunks = m.chunks, .reuse = .{ .saved = saved, .at = p.resume_at, .marks = m.marks } });
        const gop = try self.streams.getOrPut(self.gpa, p.key);
        if (gop.found_existing) {
            const old = gop.value_ptr.*;
            self.inner.release(&old.stream);
            self.free(old);
        }
        gop.value_ptr.* = m;
        return &m.stream;
    }

    fn make(self: *Follower, call: Call) !void {
        const a = self.arena.allocator();
        const b = self.inner;
        switch (call) {
            .prefill => |p| try b.prefill(try self.mirror(p)),
            .prefill_many => |ps| {
                const f = b.vtable.prefill_many orelse return error.NoBursts;
                const ss = try a.alloc(*Stream, ps.len);
                for (ss, ps) |*x, p| x.* = try self.mirror(p);
                _ = try f(b.ptr, ss); // refused on rank 0 too (the same specs): rank 0 then prefills each alone
            },
            .prefill_begin => |p| _ = try self.mirror(p), // its fill begins with its first prefill_step
            .prefill_step => |keys| {
                const f = b.vtable.prefill_step orelse return error.NoFills;
                for (keys) |k| _ = try f(b.ptr, try self.stream(k), 0);
            },
            .first => |f| _ = try b.first(try self.stream(f.key), f.position),
            .queue => |q| _ = try b.queue(try self.stream(q.key), q.feed, q.position),
            .verify => |ws| {
                const wins = try self.windows(a, ws);
                const out = try a.alloc(be.Verified, ws.len);
                for (out, ws) |*o, w| o.* = .{ .sampled = try a.alloc(u32, w.rows()), .drafts = try a.alloc(u32, w.rows() - 1) };
                try b.verify(wins, out);
            },
            .keep => |k| try b.keep(try self.windows(a, k.windows), k.paths),
            .draft => |rs| {
                const reqs = try a.alloc(be.DraftRequest, rs.len);
                for (reqs, rs) |*q, r| q.* = .{ .stream = try self.stream(r.key), .follow = r.follow, .first = r.first, .rows = r.rows, .start = r.start, .position = r.position, .depth = r.depth, .early = r.early, .ranks = r.ranks };
                try b.draft(reqs);
            },
            .unspeculate => |key| {
                const f = b.vtable.unspeculate orelse return error.NoUnspeculate;
                try f(b.ptr, try self.stream(key));
            },
            .release => |key| {
                const kv = self.streams.fetchRemove(key) orelse return;
                b.release(&kv.value.stream);
                self.free(kv.value);
            },
            .note => |bytes| {
                const f = self.family orelse return error.NoFamilyForNotes;
                try f.note(f.ptr, bytes);
            },
        }
    }
};
