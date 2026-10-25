//! MiniMax-M3.1-Flash's lane backend: slots over pooled pages, radix-cache resume, copy drafts, no vision.
const std = @import("std");
const lanes = @import("lanes");
const iface = @import("iface.zig");
const engine = @import("engine.zig");
const follow = @import("follow.zig");
const pages_mod = @import("pages.zig");
const state = @import("state.zig");
const pool = @import("pool.zig");

const be = lanes.backend;

/// First tokens a handle names.
const ring = 1024;
pub const window_rows = iface.window_rows;
pub const max_rows = iface.window_rows;

/// Draft ring snapshots the device keeps (one per kept prompt mark with a ring); a snapshot past them keeps none.
pub const max_snaps: u32 = 8;

/// A stream's place in the backend: its slot, the pages and position of its lease, its prompt pass.
pub const Lane = struct {
    slot: u32,
    st: state.Stream,
    marks: []const u32 = &.{}, // where the pass keeps the prompt cache's prefix (rows, pages), from admit
    adopted: ?iface.Resume = null, // rows already in adopted pages: only the pass's first chunk says so
    pending_rows: ?usize = null, // the rows of the last verify, not yet committed
    first: ?u64 = null,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
    owner: *Lanes,
    stream: *lanes.Stream,
    taps: Taps = .{}, // where the dspark draft finds this stream's latest tapped rows
};

/// Tapped rows a stream's latest forward left: positions [start, start + rows) at tap rows [row, row + rows).
pub const Taps = struct { start: u32 = 0, rows: u32 = 0, row: u32 = 0 };

/// Grammar words of a round's windows on the device, where rank 0's draw reads them; the host supplies the buffer.
pub const Masks = struct {
    ptr: *anyopaque,
    put: *const fn (ptr: *anyopaque, row: usize, rows: usize, words: []const u32) anyerror!u64,
};

pub const Lanes = struct {
    gpa: std.mem.Allocator,
    x: follow.Leader,
    pages: ?*pages_mod.Pages = null, // made by `attach`, after open
    masks: ?Masks = null,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, *Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    shared: u64 = 0, // forwards that held several streams' windows
    resumed: u64 = 0, // prompt passes that started past row 0, from the prompt cache
    snap_free: [max_snaps]bool = @splat(true), // draft ring snapshot ids not in use (the device holds `max_snaps` buffers)
    stopped: bool = false, // the host is closing: no step may go to the other ranks any more
    tap_ids: []const u32 = &.{}, // the draft's layers the forward taps (set by the host when it loads a draft; else none)
    tap_hidden: u32 = 0, // and the hidden width of one tap

    pub fn deinit(self: *Lanes) void {
        var it = self.lanes.valueIterator();
        while (it.next()) |l| self.free(l.*);
        self.lanes.deinit(self.gpa);
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = &vt };
    }

    const vt: be.Backend.VTable = .{
        .prefill = prefillFn,
        .prefill_step = prefillStepFn,
        .first = firstFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .release = releaseFn,
        .reports_fill = true,
        .prepare_features = prepareFn,
        .features = featuresFn,
        .prefill_features = prefillFeaturesFn,
    };

    /// Round loop facts: copies up to the window and shared exact windows (Qwen3.5's facts for a model without a head).
    pub fn facts(self: *const Lanes) lanes.Model {
        var m = iface.laneFacts(window_rows, iface.shared_rows);
        // a draft's window next to a copy branch would be a tree, and trees are not built: copies and drafts alternate
        if (self.x.e.drafting()) m.context_copies = false;
        return m;
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn free(self: *Lanes, l: *Lane) void {
        if (l.marks.len > 0) self.gpa.free(l.marks);
        if (self.pages) |pg| pg.cache.end(&l.st, &pg.tables);
        self.gpa.destroy(l);
    }

    /// The lowest slot no stream holds, within the block table's rows.
    fn freeSlot(self: *const Lanes, slots: u32) ?u32 {
        var slot: u32 = 0;
        while (slot < slots) : (slot += 1) {
            var it = self.lanes.valueIterator();
            const taken = while (it.next()) |l| {
                if (l.*.slot == slot) break true;
            } else false;
            if (!taken) return slot;
        }
        return null;
    }

    fn cancelled(ptr: ?*anyopaque) bool {
        return @as(*lanes.Stream, @ptrCast(@alignCast(ptr.?))).isCancelled();
    }

    fn cancelOf(s: *lanes.Stream) ?iface.Cancel {
        return if (s.cancel_check == null) null else .{ .ptr = s, .check = cancelled };
    }

    /// The pass's marks for the leader: the prompt cache keeps the prefix at each (`keptAt`).
    fn marksOf(l: *Lane) ?iface.Marks {
        return if (l.marks.len == 0) null else .{ .at = l.marks, .ctx = l, .call = keptAt };
    }

    /// The pass stands at `at` (a page multiple): the store keeps the prefix, the stream swaps pages, all ranks hear.
    fn keptAt(ctx: *anyopaque, at: u32) void {
        const l: *Lane = @ptrCast(@alignCast(ctx));
        const self = l.owner;
        const pg = self.pages orelse return;
        var sink = pg.tables.sink(l.slot);
        pg.cache.keep(&l.st, l.stream.prompt(), at, sink.table()) catch |err| return std.log.warn("prompt prefix not kept at {d}: {s}", .{ at, @errorName(err) });
        const n = at / pool.page_tokens;
        self.x.sendPages(0, l.st.lease.ids.items[0..n]) catch |err| std.log.warn("the other ranks' pages at {d}: {s}", .{ at, @errorName(err) });
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    /// A request's admission and its lane: refusals come before any page or step is taken.
    fn begin(self: *Lanes, s: *lanes.Stream) !*Lane {
        if (s.images.len > 0) return error.VisionUnsupported;
        const pg = self.pages orelse return error.NoPool;
        const e = self.x.e;
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + window_rows >= e.capacity) return error.PromptTooLong;
        if (self.lanes.get(s)) |old| { // a second pass of the same stream starts over
            self.release(s, old);
        }
        const slot = self.freeSlot(@intCast(pg.tables.slots)) orelse return error.LaneMemory;
        const l = try self.gpa.create(Lane);
        errdefer self.gpa.destroy(l);
        l.* = .{ .slot = slot, .st = .{ .slot = slot }, .owner = self, .stream = s };
        if (self.x.e.drafting()) { // the slot's draft ring starts empty; a resumed prefix may restore a snapshot into it
            try self.x.bind(slot);
            try self.x.draftSnap(.reset, 0);
        }
        var sink = pg.tables.sink(slot);
        l.marks = try pg.cache.admit(self.gpa, &l.st, ids, s.history_len, s.max_new, pg.window, sink.table(), pg.max_pages);
        errdefer {
            self.gpa.free(l.marks);
            pg.cache.end(&l.st, &pg.tables);
        }
        // the resumed prefix is the lease's first pages and admit writes only the pages it took: write the whole row
        try pg.tables.write(slot, l.st.lease.ids.items, 0);
        try self.x.bind(slot);
        try self.x.begin(s.sampling, l.st.lease.ids.items);
        s.cached = l.st.pos;
        l.fill = l.st.pos;
        if (l.st.pos > 0) {
            l.adopted = .{ .at = l.st.pos };
            self.resumed += 1;
        }
        try self.lanes.put(self.gpa, s, l);
        return l;
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        while (!try prefillStepFn(ptr, s)) {}
    }

    /// One chunk of the stream's prompt pass; true once the first token is drawn. A chunk ends at a mark or its rows.
    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        return of(ptr).prefillStep(s, null);
    }

    /// `prefillStepFn`, handing `sink` the chunk's tapped rows once they are made (the dspark draft absorbs them).
    fn prefillStep(self: *Lanes, s: *lanes.Stream, sink: ?be.FeatureSink) !bool {
        if (s.isCancelled()) return error.Cancelled;
        const l = if (self.lanes.get(s)) |l| (if (l.fill != null) l else try self.begin(s)) else try self.begin(s);
        try self.x.bind(l.slot);
        const ids = s.prompt();
        const at = l.fill.?;
        const marks = marksOf(l);
        const to = engine.chunkEnd(at, ids.len, marks);
        const first = self.x.chunk(ids, at, l.adopted, cancelOf(s), marks) catch |err| {
            l.fill = null; // the pass is over; the host releases the stream
            return err;
        };
        l.adopted = null;
        if (self.x.e.drafting()) {
            l.taps = .{ .start = @intCast(at), .rows = @intCast(to - at), .row = 0 };
            if (sink) |k| try k.take(k.ptr, s, self.featuresOf(l, at, to - at), at);
            // a pass that ends keeps its last row's taps at the slot's window rows (Engine.prefillChunk copied them)
            if (first != null) l.taps = .{ .start = @intCast(ids.len - 1), .rows = 1, .row = iface.tapWindowRow(l.slot) };
        }
        if (first) |t| {
            l.first = self.take(t);
            l.fill = null;
            s.fill_at.store(@intCast(ids.len), .monotonic);
            return true;
        }
        l.fill = to;
        s.fill_at.store(@intCast(to), .monotonic);
        return false;
    }

    // -- the draft's ring snapshots, kept beside the prompt cache's marks --------------------------------------------

    /// The cache's hook: install on `pages.cache.ring` once the pool exists and a draft is loaded.
    pub fn ringHook(self: *Lanes) state.RingHook {
        return .{ .ctx = self, .save = ringSave, .restore = ringRestore, .drop = ringDrop };
    }

    fn live(self: *const Lanes) bool {
        return !self.stopped and self.x.link.lost == null;
    }

    /// The bound slot's ring as a new snapshot; null if the draft keeps no more or the step could not go.
    fn ringSave(ctx: *anyopaque, slot: u32, _: u32) ?u32 {
        const self: *Lanes = @ptrCast(@alignCast(ctx));
        if (!self.live()) return null;
        const id: u32 = for (self.snap_free, 0..) |free_, i| {
            if (free_) break @intCast(i);
        } else return null;
        self.x.bind(slot) catch return null;
        self.x.draftSnap(.save, id) catch return null;
        self.snap_free[id] = false;
        return id;
    }

    /// Snapshot `id` into the slot's (empty) ring; a failed step leaves the stream to draft from its resume point.
    fn ringRestore(ctx: *anyopaque, slot: u32, id: u32) void {
        const self: *Lanes = @ptrCast(@alignCast(ctx));
        if (!self.live()) return;
        self.x.bind(slot) catch return;
        self.x.draftSnap(.restore, id) catch |err| std.log.warn("draft ring snapshot {d} not restored: {s}", .{ id, @errorName(err) });
    }

    fn ringDrop(ctx: *anyopaque, id: u32) void {
        const self: *Lanes = @ptrCast(@alignCast(ctx));
        self.snap_free[id] = true;
        if (!self.live()) return;
        self.x.draftSnap(.drop, id) catch |err| std.log.warn("draft ring snapshot {d} not dropped: {s}", .{ id, @errorName(err) });
    }

    // -- the dspark draft's view of the target -----------------------------------------------------------------------

    /// The draft told its taps before any forward: they must be the layers the engine was opened to tap.
    fn prepareFn(ptr: *anyopaque, taps: []const u32) anyerror!void {
        const self = of(ptr);
        if (!self.x.e.drafting() or !std.mem.eql(u32, taps, self.tap_ids)) return error.TapsMismatch;
    }

    /// Rows [start, start + count) of the stream's latest tapped forward, in place on the device.
    fn featuresOf(self: *Lanes, l: *const Lane, start: u64, count: u64) be.Features {
        const row_bytes: u32 = @intCast(self.tap_ids.len * self.tap_hidden * 2);
        const row: u64 = l.taps.row + (start - l.taps.start);
        return .{ .buffer = self.x.e.p.ops.tapsBase(), .offset = row * row_bytes, .rows = @intCast(count), .row_bytes = row_bytes, .space = .device, .device = 0, .dtype = .bf16 };
    }

    fn featuresFn(ptr: *anyopaque, s: *lanes.Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
        const self = of(ptr);
        if (!std.mem.eql(u32, taps, self.tap_ids)) return error.TapsMismatch;
        const l = self.lanes.get(s) orelse return error.UnknownStream;
        if (count == 0 or start < l.taps.start or start + count > @as(u64, l.taps.start) + l.taps.rows) return error.TapsNotHeld;
        return self.featuresOf(l, start, count);
    }

    /// The whole prompt in chunks, each chunk's tapped rows going to `sink` before the next forward.
    fn prefillFeaturesFn(ptr: *anyopaque, s: *lanes.Stream, taps: []const u32, sink: be.FeatureSink) anyerror!void {
        const self = of(ptr);
        if (!std.mem.eql(u32, taps, self.tap_ids)) return error.TapsMismatch;
        while (!try self.prefillStep(s, sink)) {}
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const l = self.lanes.get(s) orelse return error.UnknownStream;
        return l.first orelse error.NoFirstToken;
    }

    fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// The stream's sequence bound, its last verify committed (`kept` rows, else all of them).
    fn bind(self: *Lanes, s: *const lanes.Stream, kept: ?usize) !*Lane {
        const l = self.lanes.get(s) orelse return error.UnknownStream;
        try self.x.bind(l.slot);
        if (l.pending_rows) |rows| try self.x.keep(kept orelse rows);
        l.pending_rows = null;
        return l;
    }

    /// Grammar words of rows [row, row + rows) of the round on the device; 0 for no grammar.
    fn putBits(self: *Lanes, row: usize, rows: usize, words: []const u32) !u64 {
        if (words.len == 0) return 0;
        const m = self.masks orelse return error.StructuresUnsupported;
        return m.put(m.ptr, row, rows, words);
    }

    /// A window's checks against the lane's position, and its ids: the pending token, then the host's drafts.
    fn window(self: *Lanes, w: be.Window, ids: []u32) !struct { l: *Lane, rows: usize } {
        if (w.parents) |parents| if (!lanes.accept.isChain(parents)) return error.TreesNotBuilt; // a chain's parents are a chain
        if (w.held > 0) return error.NoHeldDrafts; // no head holds drafts
        const l = try self.bind(w.stream, null);
        const rows = w.rows();
        if (rows > max_rows) return error.WindowTooWide;
        const pos = self.x.e.pos();
        for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
        ids[0] = w.pending;
        @memcpy(ids[1..][0..w.tokens.len], w.tokens);
        return .{ .l = l, .rows = rows };
    }

    /// Each window in its own forward on its stream's pages (the core sends one when no forward is shared).
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        if (windows.len > 1) return self.verifyShared(windows, out);
        for (windows, out) |w, *o| {
            var ids: [max_rows]u32 = undefined;
            const got = try self.window(w, &ids);
            var sampled: [max_rows]u32 = undefined;
            const bits = try self.putBits(0, got.rows, w.masks); // before the step goes out: a refusal after it strands the ranks
            try self.x.verify(ids[0..got.rows], sampled[0..got.rows], bits);
            @memcpy(o.sampled[0..got.rows], sampled[0..got.rows]);
            @memcpy(o.drafts[0..w.tokens.len], w.tokens);
            got.l.pending_rows = got.rows;
            got.l.taps = .{ .start = @intCast(w.positions[0] - 1), .rows = @intCast(got.rows), .row = iface.tapWindowRow(got.l.slot) };
        }
    }

    /// A round's windows in one forward (Engine.verifyShared), each stream's rows on its own pages.
    fn verifyShared(self: *Lanes, windows: []const be.Window, out: []be.Verified) !void {
        if (windows.len > iface.max_shared) return error.WindowTooWide;
        var parts: [iface.max_shared]iface.Part = undefined;
        var ids: [iface.max_shared][max_rows]u32 = undefined;
        var sampled: [iface.max_shared][max_rows]u32 = undefined;
        var lane: [iface.max_shared]*Lane = undefined;
        var row: usize = 0; // the windows' rows so far: each one's grammar rows start there
        for (windows, 0..) |w, k| {
            const got = try self.window(w, &ids[k]);
            if (row + got.rows > iface.shared_rows) return error.WindowTooWide;
            const bits = try self.putBits(row, got.rows, w.masks);
            parts[k] = .{ .slot = got.l.slot, .ids = ids[k][0..got.rows], .out = sampled[k][0..got.rows], .bits = bits };
            lane[k] = got.l;
            row += got.rows;
        }
        try self.x.verifyShared(parts[0..windows.len]);
        self.shared += 1;
        for (out[0..windows.len], parts[0..windows.len], lane[0..windows.len], windows) |*o, p, l, w| {
            @memcpy(o.sampled[0..p.ids.len], p.out);
            @memcpy(o.drafts[0..w.tokens.len], w.tokens);
            l.pending_rows = p.ids.len;
            l.taps = .{ .start = @intCast(w.positions[0] - 1), .rows = @intCast(p.ids.len), .row = iface.tapWindowRow(l.slot) };
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        for (windows, paths) |w, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
            _ = try self.bind(w.stream, path.len);
        }
    }

    /// The pack has no draft head: the core drafts by copies alone and never asks.
    fn draftFn(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {
        return error.NoDraftHead;
    }

    /// The stream left the rounds: its slot is freed on every rank and its pages go back (the store keeps its prefix).
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const l = self.lanes.get(s) orelse return;
        self.release(s, l);
        if (self.lanes.count() == 0) self.x.idle() catch |err| if (err != error.RankLost) std.log.err("the other ranks' idle step: {s}", .{@errorName(err)});
    }

    fn release(self: *Lanes, s: *const lanes.Stream, l: *Lane) void {
        _ = self.lanes.remove(s);
        self.x.release(l.slot) catch |err| if (err != error.RankLost) std.log.err("releasing slot {d}: {s}", .{ l.slot, @errorName(err) });
        self.free(l);
    }
};

test "the vtable is a head-less backend: no drafts held, windows exact, stepped prompts" {
    try std.testing.expect(Lanes.vt.prefill_step != null and Lanes.vt.first_masked == null and Lanes.vt.held == null);
    try std.testing.expect(Lanes.vt.reports_fill);
    const f = iface.laneFacts(window_rows, iface.shared_rows);
    try std.testing.expect(!f.mtp and f.hidden_rows and f.streams_exact);
}
