//! Recording fakes for the engine's tests: KV rows are the tokens, so a wrong page, slot or position changes the draw.
const std = @import("std");
const lanes = @import("lanes");
const kv = @import("core").kv_pool;
const iface = @import("iface.zig");
const engine = @import("engine.zig");
const ops = @import("engine_ops.zig");
const follow = @import("follow.zig");
const backend = @import("backend.zig");
const pages_mod = @import("pages.zig");
const pool = @import("pool.zig");
const dc = @import("dspark_config.zig");
const floor = @import("floor.zig");

const fake = lanes.fake;
const none: u32 = 0xffff_ffff;

/// A tapped row of the fake forward: the row it came from (a real tap holds the residuals of that row).
pub const TapRow = struct { slot: u32 = none, pos: u32 = 0, tok: u32 = 0 };

/// One stream's fake draft ring: the tokens of the positions [lo, lo + toks.len) it has absorbed.
pub const FakeRing = struct {
    lo: u32 = 0,
    toks: std.ArrayList(u32) = .empty,
    tokens: [8]i32 = @splat(-1),
    drafts: u32 = 0,
};

/// A fake draft: it predicts the fake target's next tokens from its ring; `wrong_every` > 0 spoils every n-th draft.
pub const FakeDraft = struct {
    rings: std.ArrayList(FakeRing) = .empty,
    snaps: std.AutoHashMapUnmanaged(u32, FakeRing) = .empty,
    wrong_every: u32 = 0,
    made: u32 = 0,
    absorbs: usize = 0,
    passes: usize = 0,
    gaps: usize = 0, // absorbs that did not continue the ring
    partial: usize = 0, // passes made over a ring that did not hold the whole context
};

pub const Run = struct { kind: ops.Rows, ids: []u32, segs: []iface.Seg, starts: []u32 };
const RowRef = struct { slot: u32, pos: usize };

/// One rank's device, forward and collectives, all recorded.
pub const Rank = struct {
    gpa: std.mem.Allocator,
    rank: usize,
    world: usize,
    table: []i32 = &.{},
    kv: []u32 = &.{},
    slots: u32 = 0,
    max_pages: usize = 0,
    runs: std.ArrayList(Run) = .empty,
    rows: [3]std.ArrayList(RowRef) = @splat(.empty),
    draws: usize = 0,
    syncs: usize = 0,
    inject: std.ArrayList(f64) = .empty, // rank 0: peers' gate votes, taken in order (0 when empty)
    results: std.ArrayList(f64) = .empty, // rank 0: the gate's answers, in order
    heard: []const f64 = &.{}, // another rank: rank 0's answers, replayed in order
    heard_at: usize = 0,
    peers: []const []const u8 = &.{}, // gather: the other ranks' bytes, by rank (empty: the same as ours)
    fail_run_at: ?usize = null, // the forward with this index fails (error.FakeKernel)
    draft_on: bool = false, // the engine runs a dspark draft: forwards tap, the draft's ring is `fd`
    taps: []TapRow = &.{},
    fd: FakeDraft = .{},
    mem_seq: []const u64 = &.{}, // MemAvailable at each memory gather (the last holds after); empty: no floor
    peer_mem: []const u64 = &.{}, // the other rank's, gather by gather (a collective sees both at the same step)
    mem_at: usize = 0,

    pub fn deinit(r: *Rank) void {
        for (r.runs.items) |x| {
            r.gpa.free(x.ids);
            r.gpa.free(x.segs);
            r.gpa.free(x.starts);
        }
        r.runs.deinit(r.gpa);
        for (&r.rows) |*l| l.deinit(r.gpa);
        r.inject.deinit(r.gpa);
        r.results.deinit(r.gpa);
        r.gpa.free(r.table);
        r.gpa.free(r.kv);
        r.gpa.free(r.taps);
        for (r.fd.rings.items) |*g| g.toks.deinit(r.gpa);
        r.fd.rings.deinit(r.gpa);
        var it = r.fd.snaps.valueIterator();
        while (it.next()) |g| g.toks.deinit(r.gpa);
        r.fd.snaps.deinit(r.gpa);
    }

    pub fn parts(r: *Rank, capacity: usize) engine.Parts {
        return .{
            .ops = .{ .ptr = r, .vtable = &.{ .attach = attachOps, .run = run, .logits = logits, .sync = sync, .deinit = noDeinit, .set_taps = setTaps, .taps_base = tapsBase, .tap_copy = tapCopy } },
            .draft = if (r.draft_on) .{ .ptr = r, .vtable = &.{ .attach = dAttach, .begin = dBegin, .absorb = dAbsorb, .pass = dPass, .result = dResult, .snap_save = dSave, .snap_restore = dRestore, .snap_drop = dDrop, .deinit = noDeinit } } else null,
            .device = .{ .ptr = r, .vtable = &.{ .attach = attach, .upload = upload, .deinit = noDeinit } },
            .comm = .{ .ptr = r, .vtable = &.{ .max_f64 = maxF64, .gather_bytes = gather, .check = check }, .rank = r.rank, .world = r.world },
            .drawer = .{ .ptr = r, .draw_fn = drawFn },
            .capacity = capacity,
            .page_bytes = 1 << 20,
            .floor = if (r.mem_seq.len > 0) .{} else null,
            .mem = if (r.mem_seq.len > 0) .{ .ctx = r, .read_fn = memRead } else null,
        };
    }

    fn memRead(p: ?*anyopaque, _: std.Io) ?u64 {
        const r = of(p.?);
        return r.mem_seq[@min(r.mem_at, r.mem_seq.len - 1)];
    }

    fn of(p: *anyopaque) *Rank {
        return @ptrCast(@alignCast(p));
    }

    fn noDeinit(_: *anyopaque) void {}

    fn attach(p: *anyopaque, slots: u32, count: usize, max_pages: usize, _: u64) anyerror!ops.PoolView {
        const r = of(p);
        r.gpa.free(r.table);
        r.gpa.free(r.kv);
        r.table = try r.gpa.alloc(i32, @as(usize, slots) * max_pages);
        @memset(r.table, -1);
        r.kv = try r.gpa.alloc(u32, count * pool.page_tokens);
        @memset(r.kv, none);
        r.slots = slots;
        r.max_pages = max_pages;
        return .{ .arena = 0, .page_bytes = 1 << 20, .pages = count, .table = 0, .slots = slots, .max_pages = max_pages };
    }

    fn attachOps(p: *anyopaque, _: ops.PoolView) anyerror!void {
        const r = of(p);
        if (!r.draft_on) return;
        r.gpa.free(r.taps);
        r.taps = try r.gpa.alloc(TapRow, iface.tapRows(r.slots));
        @memset(r.taps, .{});
    }

    fn setTaps(_: *anyopaque, _: []const u32) anyerror!void {}

    fn tapsBase(p: *anyopaque) u64 {
        return if (of(p).draft_on) 1 << 41 else 0;
    }

    fn tapCopy(p: *anyopaque, from: u32, to: u32) anyerror!void {
        const r = of(p);
        if (from >= r.taps.len or to >= r.taps.len) return error.FakeTap;
        r.taps[to] = r.taps[from];
    }

    // -- the fake draft -------------------------------------------------------------------------------------------

    fn dAttach(p: *anyopaque, slots: u32) anyerror!void {
        const r = of(p);
        while (r.fd.rings.items.len < slots) try r.fd.rings.append(r.gpa, .{});
        for (r.fd.rings.items) |*g| {
            g.toks.clearRetainingCapacity();
            g.lo = 0;
        }
    }

    fn dBegin(p: *anyopaque, slot: u32) void {
        const r = of(p);
        if (slot >= r.fd.rings.items.len) return;
        const g = &r.fd.rings.items[slot];
        g.toks.clearRetainingCapacity();
        g.lo = 0;
    }

    fn dAbsorb(p: *anyopaque, slot: u32, start: u32, rows: u32, base: u64, row0: u32) anyerror!void {
        const r = of(p);
        if (base != 1 << 41 or slot >= r.fd.rings.items.len or row0 + rows > r.taps.len) return error.FakeDraft;
        const g = &r.fd.rings.items[slot];
        if (g.toks.items.len == 0 or start != g.lo + g.toks.items.len) { // not the next rows: the ring starts over here
            if (g.toks.items.len != 0) r.fd.gaps += 1;
            g.toks.clearRetainingCapacity();
            g.lo = start;
        }
        for (0..rows) |k| {
            const t = r.taps[row0 + k];
            // the tapped row must be the stream's row at that position: a wrong row, slot or position is a plumbing bug
            if (t.slot != slot or t.pos != start + k) return error.FakeTapMismatch;
            try g.toks.append(r.gpa, t.tok);
        }
        r.fd.absorbs += 1;
    }

    fn dPass(p: *anyopaque, d: ops.DraftPass) anyerror!void {
        const r = of(p);
        const g = &r.fd.rings.items[d.slot];
        r.fd.passes += 1;
        g.drafts = 0;
        @memset(&g.tokens, -1);
        if (g.lo != 0 or g.toks.items.len != d.pos) { // a partial context drafts nothing here
            r.fd.partial += 1;
            return;
        }
        var hist: [1 << 14]u32 = undefined;
        @memcpy(hist[0..d.pos], g.toks.items);
        hist[d.pos] = d.anchor;
        var n: usize = d.pos + 1;
        for (0..7) |i| {
            var tok = fake.next(hist[0..n], null, 0);
            r.fd.made += 1;
            if (r.fd.wrong_every != 0 and r.fd.made % r.fd.wrong_every == 0) tok = (tok + 1) % fake.vocab;
            g.tokens[i] = @intCast(tok);
            hist[n] = tok;
            n += 1;
        }
        g.drafts = 7;
    }

    fn dResult(p: *anyopaque, slot: u32, out: *ops.DraftResult) anyerror!void {
        const g = of(p).fd.rings.items[slot];
        out.* = .{ .n = g.drafts, .tokens = g.tokens, .conf = @splat(0.5) };
    }

    fn dSave(p: *anyopaque, slot: u32, id: u32) anyerror!void {
        const r = of(p);
        const g = r.fd.rings.items[slot];
        var copy: FakeRing = .{ .lo = g.lo };
        try copy.toks.appendSlice(r.gpa, g.toks.items);
        try r.fd.snaps.put(r.gpa, id, copy);
    }

    fn dRestore(p: *anyopaque, slot: u32, id: u32) anyerror!void {
        const r = of(p);
        const snap = r.fd.snaps.get(id) orelse return error.FakeSnap;
        const g = &r.fd.rings.items[slot];
        g.toks.clearRetainingCapacity();
        try g.toks.appendSlice(r.gpa, snap.toks.items);
        g.lo = snap.lo;
    }

    fn dDrop(p: *anyopaque, id: u32) void {
        const r = of(p);
        if (r.fd.snaps.fetchRemove(id)) |old| {
            var g = old.value;
            g.toks.deinit(r.gpa);
        }
    }

    fn upload(p: *anyopaque, slot: u32, from: usize, entries: []const i32) anyerror!void {
        const r = of(p);
        if (slot >= r.slots or from + entries.len > r.max_pages) return error.FakeTable;
        @memcpy(r.table[@as(usize, slot) * r.max_pages + from ..][0..entries.len], entries);
    }

    fn pageOf(r: *const Rank, slot: u32, pos: usize) !usize {
        const id = r.table[@as(usize, slot) * r.max_pages + pos / pool.page_tokens];
        if (id < 0) return error.FakeNoPage;
        return @as(usize, @intCast(id)) * pool.page_tokens + pos % pool.page_tokens;
    }

    fn run(p: *anyopaque, f: ops.Forward) anyerror!void {
        const r = of(p);
        if (r.fail_run_at) |n| if (n == r.runs.items.len) return error.FakeKernel;
        try r.runs.append(r.gpa, .{ .kind = f.kind, .ids = try r.gpa.dupe(u32, f.ids), .segs = try r.gpa.dupe(iface.Seg, f.segs), .starts = try r.gpa.dupe(u32, f.starts) });
        const list = &r.rows[@backingInt(f.kind)];
        list.clearRetainingCapacity();
        for (f.segs, f.starts) |g, start| {
            for (0..g.rows) |k| {
                const at = try r.pageOf(g.slot, start + k);
                r.kv[at] = f.ids[g.row0 + k];
                if (r.draft_on) {
                    const row = if (f.kind == .prompt) g.row0 + k else iface.tapWindowRow(g.slot) + k;
                    r.taps[row] = .{ .slot = g.slot, .pos = @intCast(start + k), .tok = f.ids[g.row0 + k] };
                }
                if (f.kind != .prompt or k + 1 == g.rows) try list.append(r.gpa, .{ .slot = g.slot, .pos = start + k });
            }
        }
    }

    fn logits(_: *anyopaque, kind: ops.Rows) u64 {
        return (@as(u64, @backingInt(kind)) + 1) << 40;
    }

    fn sync(p: *anyopaque) anyerror!void {
        of(p).syncs += 1;
    }

    fn maxF64(p: *anyopaque, v: f64) anyerror!f64 {
        const r = of(p);
        if (r.rank != 0) {
            defer r.heard_at += 1;
            return r.heard[r.heard_at];
        }
        const other: f64 = if (r.inject.items.len > 0) r.inject.orderedRemove(0) else 0;
        const out = @max(v, other);
        try r.results.append(r.gpa, out);
        return out;
    }

    fn gather(p: *anyopaque, mine: []const u8, out: []u8) anyerror!void {
        const r = of(p);
        if (r.mem_seq.len > 0 and mine.len == @sizeOf(floor.Sample)) { // a memory gather: the peer's reading at this step
            defer r.mem_at += 1;
            const theirs: floor.Sample = .{ .avail = r.peer_mem[@min(r.mem_at, r.peer_mem.len - 1)] };
            for (0..r.world) |q| @memcpy(out[q * mine.len ..][0..mine.len], if (q == r.rank) mine else std.mem.asBytes(&theirs));
            return;
        }
        for (0..r.world) |q| {
            const dst = out[q * mine.len ..][0..mine.len];
            if (q == r.rank or q >= r.peers.len) @memcpy(dst, mine) else @memcpy(dst, r.peers[q]);
        }
    }

    fn check(_: *anyopaque) anyerror!void {}

    /// Rank 0 draws as the fake target does: from the tokens in the stream's pages, keyed by position.
    fn drawFn(p: *anyopaque, d: ops.Draw) anyerror!void {
        const r = of(p);
        r.draws += 1;
        if (r.rank != 0) return;
        const kind = (d.base >> 40) - 1;
        std.debug.assert(d.total == r.rows[@intCast(kind)].items.len);
        for (0..d.rows) |k| {
            const ref = r.rows[@intCast(kind)].items[d.row0 + k];
            var history: [1 << 14]u32 = undefined;
            for (0..ref.pos + 1) |q| history[q] = r.kv[try r.pageOf(ref.slot, q)];
            d.out[k] = fake.next(history[0 .. ref.pos + 1], d.rule, d.first + k);
        }
    }
};

/// What a rank 0 sent, in order, for a rank 1 to replay.
pub const Tape = struct {
    gpa: std.mem.Allocator,
    steps: std.ArrayList(struct { h: iface.Header, ids: []u32 }) = .empty,
    cursor: usize = 0,
    fail_at: ?usize = null, // the send with this index fails (error.WireBroken), as a lost peer's broadcast does

    pub fn deinit(t: *Tape) void {
        for (t.steps.items) |s| t.gpa.free(s.ids);
        t.steps.deinit(t.gpa);
    }

    pub fn wire(t: *Tape) follow.Wire {
        return .{ .ptr = t, .vtable = &.{ .send = send, .receive = receive } };
    }

    fn send(p: *anyopaque, h: *const iface.Header, ids: []const u32) anyerror!void {
        const t: *Tape = @ptrCast(@alignCast(p));
        if (t.fail_at) |n| if (n == t.steps.items.len) return error.WireBroken;
        try t.steps.append(t.gpa, .{ .h = h.*, .ids = try t.gpa.dupe(u32, ids) });
    }

    fn receive(p: *anyopaque, buf: []u32) anyerror!iface.Header {
        const t: *Tape = @ptrCast(@alignCast(p));
        if (t.cursor >= t.steps.items.len) return error.TapeEnd;
        const s = t.steps.items[t.cursor];
        if (s.ids.len > buf.len) return error.TooManyIds;
        @memcpy(buf[0..s.ids.len], s.ids);
        t.cursor += 1;
        return s.h;
    }

    pub fn count(t: *const Tape, kind: iface.Kind) usize {
        var n: usize = 0;
        for (t.steps.items) |s| n += @intFromBool(s.h.kind == @backingInt(kind));
        return n;
    }
};

/// The liveness lines as counters.
pub const Hooks = struct {
    on: bool = true,
    sleeps: usize = 0,
    wakes: usize = 0,
    ended: usize = 0,
    failed: [4][]const u8 = undefined,
    failures: usize = 0,

    pub fn hooks(h: *Hooks) follow.Hooks {
        return .{ .ctx = h, .live = live, .sleep = sleep, .wake = wake, .failed = failed_, .ended = ended_ };
    }

    fn of(p: *anyopaque) *Hooks {
        return @ptrCast(@alignCast(p));
    }
    fn live(p: *anyopaque) bool {
        return of(p).on;
    }
    fn sleep(p: *anyopaque) void {
        of(p).sleeps += 1;
    }
    fn wake(p: *anyopaque) void {
        of(p).wakes += 1;
    }
    fn failed_(p: *anyopaque, name: []const u8) void {
        const h = of(p);
        if (h.failures < h.failed.len) h.failed[h.failures] = name;
        h.failures += 1;
    }
    fn ended_(p: *anyopaque) void {
        of(p).ended += 1;
    }
};

/// Rank 0 with its lanes backend over the page state, and a rank 1 to replay the tape on (heap: it points at itself).
pub const World = struct {
    gpa: std.mem.Allocator,
    r0: Rank,
    r1: Rank,
    e0: *engine.Engine,
    e1: *engine.Engine,
    tape: Tape,
    hooks0: Hooks = .{},
    hooks1: Hooks = .{},
    link0: follow.Link,
    link1: follow.Link,
    pg: *pages_mod.Pages,
    lanes: backend.Lanes,
    bits: u64 = 0,

    pub fn create(gpa: std.mem.Allocator, context: usize, slots: u32, count: usize) !*World {
        return createWith(gpa, context, slots, count, false);
    }

    /// `draft`: both ranks tap and hold a fake draft ring, and the lanes backend knows the draft's taps.
    pub fn createWith(gpa: std.mem.Allocator, context: usize, slots: u32, count: usize, draft: bool) !*World {
        return createMem(gpa, context, slots, count, draft, .{ &.{}, &.{} });
    }

    /// `mem[r]`: rank r's MemAvailable at each memory gather (both empty: no floor); the default floor gates them.
    pub fn createMem(gpa: std.mem.Allocator, context: usize, slots: u32, count: usize, draft: bool, mem: [2][]const u64) !*World {
        const w = try gpa.create(World);
        errdefer gpa.destroy(w);
        const cap = std.mem.alignForward(usize, context + iface.window_rows, pool.page_tokens);
        w.* = .{ .gpa = gpa, .r0 = .{ .gpa = gpa, .rank = 0, .world = 2, .draft_on = draft, .mem_seq = mem[0], .peer_mem = mem[1] }, .r1 = .{ .gpa = gpa, .rank = 1, .world = 2, .draft_on = draft, .mem_seq = mem[1], .peer_mem = mem[0] }, .e0 = undefined, .e1 = undefined, .tape = .{ .gpa = gpa }, .link0 = undefined, .link1 = undefined, .pg = undefined, .lanes = undefined };
        errdefer { // a start that fails (the floor at attach) leaves nothing behind
            w.tape.deinit();
            w.r0.deinit();
            w.r1.deinit();
        }
        w.e0 = try engine.Engine.make(gpa, std.testing.io, w.r0.parts(cap));
        errdefer w.e0.deinit();
        w.e1 = try engine.Engine.make(gpa, std.testing.io, w.r1.parts(cap));
        errdefer w.e1.deinit();
        w.link0 = try follow.Link.init(gpa, w.tape.wire(), w.hooks0.hooks(), cap);
        errdefer w.link0.deinit();
        w.link1 = try follow.Link.init(gpa, w.tape.wire(), w.hooks1.hooks(), cap);
        errdefer w.link1.deinit();
        const max_pages = iface.maxPages(cap);
        const layout: kv.Layout = .{ .fixed_bytes = 4 * max_pages, .token_bytes = 1024, .page_tokens = pool.page_tokens, .snap_bytes = pool.snap_bytes, .all_paged = true };
        w.pg = try pages_mod.Pages.create(gpa, layout, slots, count, max_pages, iface.window_rows);
        errdefer w.pg.destroy();
        w.pg.tables.upload = w.e0.uploadHook();
        w.lanes = .{ .gpa = gpa, .x = .{ .e = w.e0, .link = &w.link0 }, .pages = w.pg };
        if (draft) {
            w.pg.cache.ring = w.lanes.ringHook();
            w.lanes.tap_ids = dc.flash.tapIds();
            w.lanes.tap_hidden = dc.flash.hidden;
        }
        w.link0.asleep = true;
        try w.lanes.x.attach(slots, count);
        return w;
    }

    pub fn destroy(w: *World) void {
        w.lanes.deinit();
        w.pg.destroy();
        w.link0.deinit();
        w.link1.deinit();
        w.e0.deinit();
        w.e1.deinit();
        w.tape.deinit();
        w.r0.deinit();
        w.r1.deinit();
        w.gpa.destroy(w);
    }

    /// Ends the tape and runs rank 1 over it; its gate answers are rank 0's.
    pub fn replay(w: *World) !void {
        try w.lanes.x.end();
        w.r1.heard = w.r0.results.items;
        try follow.follow(w.e1, &w.link1);
    }
};
