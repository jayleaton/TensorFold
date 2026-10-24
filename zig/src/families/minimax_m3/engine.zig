//! One TP rank's MiniMax-M3.1-Flash engine: slots over the pooled KV pages, chunked prompts, windows and draws.
const std = @import("std");
const cuda = @import("cuda");
const iface = @import("iface.zig");
const pool = @import("pool.zig");
const ops_ = @import("engine_ops.zig");
const floor_ = @import("floor.zig");
const rows_plan = @import("rows_plan.zig");

pub const Sampling = iface.Sampling;
pub const Part = iface.Part;
pub const Resume = iface.Resume;
pub const Marks = iface.Marks;
pub const Cancel = iface.Cancel;

/// A draft ring step: save the bound slot's ring as snapshot `id`, restore it, drop `id`, or empty the slot's ring.
pub const SnapOp = enum(u32) { save = 1, restore = 2, drop = 3, reset = 4 };
pub const prefill_rows = iface.prefill_rows;
pub const window_rows = iface.window_rows;
pub const shared_rows = iface.shared_rows;
pub const max_shared = iface.max_shared;

/// The request's rule as the engine runs it: null (or temperature 0) decodes greedily.
pub fn check(s: ?Sampling) ?Sampling {
    const x = s orelse return null;
    return if (x.temperature > 0) x else null;
}

/// A stream's sequence on the engine: how it draws, the committed rows, and the rows its last window left uncommitted.
pub const Seq = struct { sampling: ?Sampling = null, pos: usize = 0, pending: usize = 0 };

/// Everything an engine runs on; engine_dev.zig fills it from the device, the host tests from fakes.
pub const Parts = struct {
    ops: ops_.Ops,
    device: ops_.Device,
    comm: ops_.Comm,
    drawer: ops_.Drawer,
    capacity: usize,
    page_bytes: u64,
    checkpoint_digest: [32]u8 = @splat(0),
    stream: ?cuda.Stream = null, // the engine's stream (the host opens the RoCE gather on it)
    group: ?*cuda.tp_group.Group = null, // the group the forward holds: the host's RoCE small gather goes into it
    close: ?Closer = null,
    draft: ?ops_.Draft = null, // the dspark draft (--drafter): its rings, block pass and results; the forward taps for it
    floor: ?floor_.Floor = null, // the ranks' memory floor, with `mem` on every rank or on none (floor.zig)
    mem: ?floor_.Mem = null,
    rows: ?rows_plan.Plan = null, // prompt chunk rows under the forward's score scratch (null: fixed chunk rows)
};

/// Frees what an engine's parts own (device buffers, the communicator, the forward's buffers) at `deinit`.
pub const Closer = struct { ptr: *anyopaque, call: *const fn (ptr: *anyopaque) void };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    p: Parts,
    rank: usize,
    world: usize,
    capacity: usize,
    max_pages: usize,
    seqs: std.ArrayList(Seq) = .empty,
    slot: u32 = 0,
    view: ?ops_.PoolView = null,
    last_rows: [3]usize = @splat(0), // rows of the last pass of each kind: the gathered logits' rank stride
    rows: rows_plan.Plan, // where prompt chunks end: rank 0 cuts them, the other ranks run the rows they are sent
    checkpoint_digest: [32]u8,
    peer_low: std.atomic.Value(u64) = .init(floor_.unknown), // the other ranks' lowest MemAvailable at the last gate

    /// The engine over `parts` with the one sequence a block table starts with (slot 0); `attach` makes the pool.
    pub fn make(gpa: std.mem.Allocator, io: std.Io, parts: Parts) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .io = io, .p = parts, .rank = parts.comm.rank, .world = parts.comm.world, .capacity = parts.capacity, .max_pages = iface.maxPages(parts.capacity), .checkpoint_digest = parts.checkpoint_digest, .rows = parts.rows orelse rows_plan.Plan.fixed(1, iface.maxPages(parts.capacity)) };
        try e.seqs.append(gpa, .{});
        return e;
    }

    /// The host opened the RoCE small gather on its group: the forward and the failure check use it from now on.
    pub fn setGroup(e: *Engine, g: *const cuda.tp_group.Group) void {
        if (e.p.group) |own| own.fast = g.fast;
    }

    /// Rank 0's `Tables.upload`: its block-table pushes reach the device on the engine's stream.
    pub fn uploadHook(e: *Engine) pool.Upload {
        return .{ .ctx = e, .call = uploadFn };
    }

    fn uploadFn(ctx: *anyopaque, slot: u32, from: usize, entries: []const i32) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        return e.upload(slot, from, entries);
    }

    /// The production constructor (iface.conformsEngine's `init`): weights, agreed fingerprint, forward, device.
    pub const init = @import("engine_dev.zig").init;

    pub fn deinit(e: *Engine) void {
        e.p.ops.sync() catch {};
        if (e.p.close) |c| c.call(c.ptr);
        e.seqs.deinit(e.gpa);
        e.gpa.destroy(e);
    }

    /// The pool of `slots` streams and `pages` pages, made now on this rank (rank 0 tells the others first).
    pub fn attach(e: *Engine, slots: u32, pages: usize) !void {
        const view = try e.p.device.vtable.attach(e.p.device.ptr, slots, pages, e.max_pages, e.p.page_bytes);
        try e.p.ops.attach(view);
        if (e.p.draft) |d| try d.attach(slots);
        e.view = view;
        while (e.seqs.items.len < slots) try e.seqs.append(e.gpa, .{});
        const f = e.p.floor orelse return;
        const low = try e.memLow() orelse return;
        switch (f.boot(low.low)) { // the pool is made on every rank: what it left is the floor serving starts from
            .ok => {},
            .under_target => if (e.rank == 0) std.log.warn("minimax memory: rank {d} has {d:.2} GiB MemAvailable with the pool made, under the {d:.2} GiB floor target (TF_MM3_FLOOR_GIB): prompts beside others will wait", .{ low.rank, floor_.gibs(low.low), floor_.gibs(f.target) }),
            .under_hard => {
                if (e.rank == 0) std.log.warn("minimax memory: rank {d} has {d:.2} GiB MemAvailable with the pool made, under the {d:.2} GiB hard floor and {d:.2} GiB chunk transient: every rank stops", .{ low.rank, floor_.gibs(low.low), floor_.gibs(f.hard), floor_.gibs(f.transient) });
                return error.MemoryFloor;
            },
        }
    }

    /// Every rank's MemAvailable through one small gather (a collective: every rank calls it at the same step).
    fn memLow(e: *Engine) !?floor_.Low {
        const m = e.p.mem orelse return null;
        var all: [8]floor_.Sample = undefined;
        if (e.world > all.len) return error.BadWorld;
        const mine: floor_.Sample = .{ .avail = m.read(e.io) orelse floor_.unknown };
        try e.p.comm.gatherBytes(std.mem.asBytes(&mine), std.mem.sliceAsBytes(all[0..e.world]));
        const low = floor_.lowest(all[0..e.world], e.rank);
        e.peer_low.store(low.others, .release);
        return low;
    }

    /// Rank 0's admission: the floor's target on this rank now and on the others at their last gate (none: true).
    pub fn floorAdmits(e: *const Engine) bool {
        const f = e.p.floor orelse return true;
        const m = e.p.mem orelse return true;
        const mine = m.read(e.io) orelse floor_.unknown;
        return f.admits(@min(mine, e.peer_low.load(.acquire)));
    }

    /// `entries` of `slot`'s table row from entry `from` to the device (also rank 0's `Tables.upload`).
    pub fn upload(e: *Engine, slot: u32, from: usize, entries: []const i32) !void {
        return e.p.device.vtable.upload(e.p.device.ptr, slot, from, entries);
    }

    /// Page ids `ids` at entry `from` of `slot`'s table row, as rank 0 decided them.
    pub fn setPages(e: *Engine, slot: u32, from: usize, ids: []const u32) !void {
        if (from + ids.len > e.max_pages) return error.TableFull;
        var buf: [256]i32 = undefined;
        var at: usize = 0;
        while (at < ids.len) {
            const n = @min(buf.len, ids.len - at);
            for (buf[0..n], ids[at..][0..n]) |*d, id| d.* = @intCast(id);
            try e.upload(slot, from + at, buf[0..n]);
            at += n;
        }
    }

    /// Slot `slot`'s table row back to -1.
    pub fn clearRow(e: *Engine, slot: u32) !void {
        const row = try e.gpa.alloc(i32, e.max_pages);
        defer e.gpa.free(row);
        @memset(row, -1);
        try e.upload(slot, 0, row);
    }

    /// Binds slot `slot` for the steps that follow.
    pub fn bind(e: *Engine, slot: u32) !void {
        if (slot >= e.seqs.items.len) return error.BadSlot;
        e.slot = slot;
    }

    /// Frees slot `slot`'s sequence (its pages are rank 0's to return) and clears its table row.
    pub fn release(e: *Engine, slot: u32) void {
        if (slot >= e.seqs.items.len) return;
        e.seqs.items[slot] = .{};
        if (e.p.draft) |d| if (e.view != null) d.begin(slot);
        if (e.view != null) e.clearRow(slot) catch |err| std.log.err("clearing slot {d}'s block table: {s}", .{ slot, @errorName(err) });
    }

    /// Bytes a stream holds beyond its pages: its block-table row.
    pub fn seqBytes(e: *const Engine) u64 {
        return pool.Dims.tableBytes(e.max_pages);
    }

    fn cur(e: *Engine) *Seq {
        return &e.seqs.items[e.slot];
    }

    /// The committed length of the bound sequence: the pending token sits here, row r of a window draws pos + 1 + r.
    pub fn pos(e: *const Engine) usize {
        return e.seqs.items[e.slot].pos;
    }

    /// A follower's begin (no image rows exist here to clear).
    pub fn beginFollowing(e: *Engine, sampling: ?Sampling) !void {
        return e.begin(sampling);
    }

    /// A fresh sequence on the bound slot with `sampling` (null or temperature 0: greedy).
    pub fn begin(e: *Engine, sampling: ?Sampling) !void {
        e.cur().* = .{ .sampling = check(sampling) };
    }

    /// The whole prompt in chunks (a mark cuts one), then the first token; `from` resumes adopted pages.
    pub fn prefill(e: *Engine, ids: []const u32, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !u32 {
        var start = try e.prefillStart(ids.len, from);
        while (true) {
            const end = e.chunkEnd(start, ids.len, marks);
            if (try e.prefillChunk(ids[start..end], start, ids.len, if (start == (if (from) |r| r.at else 0)) from else null, cancel, marks)) |first| return first;
            start = end;
        }
    }

    /// Where a prompt pass's chunk starting at `start` of a `len`-id prompt ends: the plan's rows, or the next mark.
    pub fn chunkEnd(e: *const Engine, start: usize, len: usize, marks: ?Marks) usize {
        return e.rows.chunkEnd(start, len, marks);
    }

    /// Where a prompt pass of `len` ids starts: 0, or the rows already in the adopted pages.
    fn prefillStart(e: *Engine, len: usize, from: ?Resume) !usize {
        if (len == 0 or len + window_rows >= e.capacity) return error.PromptTooLong;
        const r = from orelse return 0;
        if (r.at == 0 or r.at >= len or r.at % pool.page_tokens != 0) return error.BadResume;
        return r.at;
    }

    /// One chunk `part` = ids[start..] of a `len`-id prompt; the first token once it ends the prompt, else null.
    pub fn prefillChunk(e: *Engine, part: []const u32, start: usize, len: usize, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !?u32 {
        const begun = try e.prefillStart(len, from);
        if (from != null) {
            if (begun != start) return error.BadChunk;
            e.cur().pos = start; // the rows before it are in the adopted pages
        }
        const rows = part.len;
        const s = e.cur();
        if (start >= len or start % pool.page_tokens != 0 or rows == 0 or rows > prefill_rows or start + rows > len) return error.BadChunk;
        if (start + rows < len and rows % pool.page_tokens != 0) return error.BadChunk; // the next chunk starts on a page
        if (s.pos != start) return error.BadChunk;
        if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
        try e.gate(cancel);
        const segs = [1]iface.Seg{.{ .slot = e.slot, .row0 = 0, .rows = @intCast(rows) }};
        try e.p.ops.run(.{ .kind = .prompt, .ids = part, .segs = &segs, .starts = &.{@intCast(start)}, .head = start + rows == len });
        s.pos = start + rows;
        if (start + rows < len) return null;
        // the prompt's last row is what the first draft starts from: keep its tap where no other forward writes
        if (e.p.draft != null) try e.p.ops.tapCopy(@intCast(rows - 1), iface.tapWindowRow(e.slot));
        var t: [1]u32 = undefined;
        e.last_rows[@backingInt(ops_.Rows.prompt)] = 1; // only the last row of the chunk goes through the head
        try e.drawPass(.prompt, 0, 1, len, 0, &t);
        return t[0];
    }

    /// The chunk gate on every rank: rank 0's cancel and the lowest rank's memory decide for all, before the forward.
    pub fn gate(e: *Engine, cancel: ?Cancel) !void {
        const mine: f64 = if (cancel) |c| @floatFromInt(@intFromBool(c.check(c.ptr))) else 0;
        if (try e.p.comm.maxF64(mine) > 0) return error.Cancelled;
        const f = e.p.floor orelse return;
        const low = try e.memLow() orelse return;
        if (!f.stops(low.low)) return;
        // every rank saw the same samples: each stops this chunk before its forward, none waits on another
        if (e.rank == 0) std.log.warn("minimax memory: rank {d} has {d:.2} GiB MemAvailable, under the {d:.2} GiB hard floor plus a {d:.2} GiB chunk: the prompt stops before its next forward", .{ low.rank, floor_.gibs(low.low), floor_.gibs(f.hard), floor_.gibs(f.transient) });
        return error.MemoryFloor;
    }

    /// A verify window: `ids` (pending token, then drafts) at the committed position; `out[r]` drawn at pos + 1 + r.
    pub fn verify(e: *Engine, ids: []const u32, out: []u32, bits: u64) !void {
        if (ids.len == 0 or ids.len > window_rows or out.len < ids.len) return error.WindowTooWide;
        const s = e.cur();
        if (s.pos + ids.len >= e.capacity) return error.ContextFull;
        const segs = [1]iface.Seg{.{ .slot = e.slot, .row0 = 0, .rows = @intCast(ids.len) }};
        try e.p.ops.run(.{ .kind = .window, .ids = ids, .segs = &segs, .starts = &.{@intCast(s.pos)} });
        s.pending = ids.len;
        e.last_rows[@backingInt(ops_.Rows.window)] = ids.len;
        try e.drawPass(.window, 0, ids.len, s.pos + 1, bits, out);
    }

    /// Several streams' windows in one forward: row ops are shared, attention reads each stream's own pages.
    pub fn verifyShared(e: *Engine, parts: []const Part) !void {
        if (parts.len == 0 or parts.len > max_shared) return error.WindowTooWide;
        var segs: [max_shared]iface.Seg = undefined;
        var starts: [max_shared]u32 = undefined;
        var ids: [shared_rows]u32 = undefined;
        var rows: usize = 0;
        for (parts, 0..) |p, i| {
            if (p.ids.len == 0 or p.ids.len > window_rows or rows + p.ids.len > shared_rows or p.out.len < p.ids.len) return error.WindowTooWide;
            for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
            if (p.slot >= e.seqs.items.len) return error.BadSlot;
            const s = e.seqs.items[p.slot];
            if (s.pos + p.ids.len >= e.capacity) return error.ContextFull;
            segs[i] = .{ .slot = p.slot, .row0 = @intCast(rows), .rows = @intCast(p.ids.len) };
            starts[i] = @intCast(s.pos);
            @memcpy(ids[rows..][0..p.ids.len], p.ids);
            rows += p.ids.len;
        }
        try e.p.ops.run(.{ .kind = .shared, .ids = ids[0..rows], .segs = segs[0..parts.len], .starts = starts[0..parts.len] });
        e.last_rows[@backingInt(ops_.Rows.shared)] = rows;
        for (parts, segs[0..parts.len]) |p, g| {
            try e.bind(p.slot);
            e.cur().pending = p.ids.len;
            try e.drawPass(.shared, g.row0, g.rows, e.cur().pos + 1, p.bits, p.out);
        }
    }

    /// Commits the first `n` rows of the last window (the rest were rejected drafts; their KV rows are overwritten).
    pub fn keep(e: *Engine, n: usize) !void {
        const s = e.cur();
        if (n > s.pending) return error.BadKeep;
        s.pos += n;
        s.pending = 0;
    }

    /// Rows [row0, row0 + rows) of the last pass of `kind` drawn at positions first, first + 1, ... (rank 0 draws).
    fn drawPass(e: *Engine, kind: ops_.Rows, row0: usize, rows: usize, first: u64, bits: u64, out: []u32) !void {
        if (rows == 0) return;
        const base = e.p.ops.logits(kind);
        try e.p.drawer.draw(.{ .base = base, .total = e.last_rows[@backingInt(kind)], .row0 = row0, .rows = rows, .first = first, .rule = e.cur().sampling, .bits = bits, .out = out });
        return e.checkGathers(); // the draw read rows the forward's gathers made
    }

    /// The contract's draw: `rows` gathered rows at `logits` (a whole pass) drawn at positions first, first + 1, ...
    pub fn draw(e: *Engine, logits: u64, rows: usize, first: u64, bits: u64, out: []u32) !void {
        if (rows == 0) return;
        try e.p.drawer.draw(.{ .base = logits, .total = rows, .row0 = 0, .rows = rows, .first = first, .rule = e.cur().sampling, .bits = bits, .out = out });
        return e.checkGathers();
    }

    /// A failed RoCE gather's error, after a synchronize whose result read gathered rows.
    pub fn checkGathers(e: *const Engine) !void {
        return e.p.comm.check();
    }

    pub fn sync(e: *Engine) !void {
        return e.p.ops.sync();
    }

    /// Whether this engine runs the dspark draft (its forwards tap).
    pub fn drafting(e: *const Engine) bool {
        return e.p.draft != null;
    }

    /// The bound stream's positions start ... start + rows - 1 are tap rows [row, row + rows): into its draft ring.
    pub fn absorb(e: *Engine, start: usize, rows: usize, row: usize) !void {
        const d = e.p.draft orelse return error.NoDraft;
        if (rows == 0 or start + rows > e.capacity or row + rows > iface.tapRows(@intCast(e.seqs.items.len))) return error.BadAbsorb;
        try d.absorb(e.slot, @intCast(start), @intCast(rows), e.p.ops.tapsBase(), @intCast(row));
    }

    /// One block pass for the bound stream: `anchor` is its pending token at position `at` (the committed length).
    pub fn draftPass(e: *Engine, anchor: u32, at: usize) !void {
        const d = e.p.draft orelse return error.NoDraft;
        if (at >= e.capacity) return error.ContextFull;
        try d.pass(.{ .slot = e.slot, .anchor = anchor, .pos = @intCast(at) });
    }

    /// The bound stream's last pass, waited for.
    pub fn draftResult(e: *Engine, out: *ops_.DraftResult) !void {
        const d = e.p.draft orelse return error.NoDraft;
        try d.result(e.slot, out);
        return e.checkGathers();
    }

    /// The bound stream's ring saved as snapshot `id`, restored from it, or `id` dropped.
    pub fn draftSnap(e: *Engine, what: SnapOp, id: u32) !void {
        const d = e.p.draft orelse return;
        switch (what) {
            .save => try d.snapSave(e.slot, id),
            .restore => try d.snapRestore(e.slot, id),
            .drop => d.snapDrop(id),
            .reset => d.begin(e.slot),
        }
    }
};

comptime {
    iface.conformsEngine(Engine);
}

test "a chunk ends at the pass's chunk rows, at a mark, or at the prompt's end" {
    const p = rows_plan.Plan.fixed(1, iface.maxPages(65536 + window_rows));
    try std.testing.expectEqual(@as(usize, 2048), p.chunkEnd(0, 5000, null));
    try std.testing.expectEqual(@as(usize, 100), p.chunkEnd(0, 100, null));
    const Hook = struct {
        fn call(_: *anyopaque, _: u32) void {}
    };
    var x: u8 = 0;
    const m: Marks = .{ .at = &.{ 512, 1024, 4096 }, .ctx = &x, .call = Hook.call };
    try std.testing.expectEqual(@as(usize, 512), p.chunkEnd(0, 5000, m));
    try std.testing.expectEqual(@as(usize, 1024), p.chunkEnd(512, 5000, m));
    try std.testing.expectEqual(@as(usize, 3072), p.chunkEnd(1024, 5000, m));
    try std.testing.expectEqual(@as(usize, 4096), p.chunkEnd(3072, 5000, m));
    try std.testing.expectEqual(@as(usize, 5000), p.chunkEnd(4096, 5000, m));
}

test "temperature 0 is greedy" {
    try std.testing.expectEqual(@as(?Sampling, null), check(.{ .seed = 1, .temperature = 0 }));
    try std.testing.expect(check(.{ .seed = 1, .temperature = 0.5 }) != null);
}
