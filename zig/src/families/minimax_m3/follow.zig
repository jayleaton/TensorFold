//! Rank 0's steps to the other ranks over the NCCL broadcast control link: a 64-byte header, then ids (no TCP mirror).
const std = @import("std");
const iface = @import("iface.zig");
const engine = @import("engine.zig");
const Engine = engine.Engine;

pub const Kind = iface.Kind;
pub const Header = iface.Header;

/// The link's two directions: rank 0 sends a step, the others receive it (NCCL broadcast in engine_dev.zig).
pub const Wire = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// The header, then its `ids` (header.n of them) to every rank; the ids fit the link's capacity.
        send: *const fn (ptr: *anyopaque, h: *const Header, ids: []const u32) anyerror!void,
        /// The next step: its header, its ids in `buf[0..h.n]`; error.TooManyIds if they do not fit `buf`.
        receive: *const fn (ptr: *anyopaque, buf: []u32) anyerror!Header,
    };
};

/// What a rank does outside the link: the liveness lines (cuda/tp_watch.zig in production, recorders in tests).
pub const Hooks = struct {
    ctx: *anyopaque,
    /// Whether liveness lines exist (more than one rank).
    live: *const fn (ctx: *anyopaque) bool,
    /// Follower: wait for rank 0's wake without spinning in NCCL.
    sleep: *const fn (ctx: *anyopaque) void,
    /// Rank 0: the other ranks' sleep returns.
    wake: *const fn (ctx: *anyopaque) void,
    /// This rank stops with `name`: every other rank hears it and stops too.
    failed: *const fn (ctx: *anyopaque, name: []const u8) void,
    /// This rank ends on purpose: a line closing from here is expected.
    ended: *const fn (ctx: *anyopaque) void,
};

pub const Link = struct {
    wire: Wire,
    hooks: Hooks,
    ids: []u32,
    gpa: std.mem.Allocator,
    asleep: bool = false, // rank 0: the other ranks sleep on their lines until its next step
    lost: ?anyerror = null, // rank 0: the first error the wire gave; every later step is error.RankLost

    /// `asked_ids` is at least the context, so a whole prompt may cross as one step; a table row always fits.
    pub fn init(gpa: std.mem.Allocator, wire: Wire, hooks: Hooks, asked_ids: usize) !Link {
        const cap = capacityFor(asked_ids);
        return .{ .wire = wire, .hooks = hooks, .ids = try gpa.alloc(u32, cap), .gpa = gpa };
    }

    pub fn deinit(l: *Link) void {
        l.gpa.free(l.ids);
    }

    /// Ids a step can carry: a chunk, a table row, a shared window's parts and rows, and what the caller asked for.
    pub fn capacityFor(asked: usize) usize {
        return @max(@max(asked, iface.prefill_rows), 2 * iface.max_shared + iface.shared_rows);
    }
};

/// A shared step's ids: for each part its slot and row count, then every part's ids; returns the words used.
pub fn encodeShared(buf: []u32, parts: []const iface.Part) !usize {
    var n: usize = 2 * parts.len;
    for (parts, 0..) |p, i| {
        if (n + p.ids.len > buf.len) return error.WindowTooWide;
        buf[2 * i] = p.slot;
        buf[2 * i + 1] = @intCast(p.ids.len);
        @memcpy(buf[n..][0..p.ids.len], p.ids);
        n += p.ids.len;
    }
    return n;
}

/// The parts of `k` windows in `words` (encodeShared's form), their drawn tokens going to `outs`.
pub fn decodeShared(words: []const u32, k: usize, parts: []iface.Part, outs: []u32) !void {
    if (k == 0 or k > iface.max_shared or k > parts.len or words.len < 2 * k) return error.BadStep;
    var at: usize = 2 * k;
    var row: usize = 0;
    for (parts[0..k], 0..) |*p, i| {
        const n = words[2 * i + 1];
        if (n == 0 or at + n > words.len or row + n > outs.len) return error.BadStep;
        p.* = .{ .slot = words[2 * i], .ids = words[at..][0..n], .out = outs[row..][0..n] };
        at += n;
        row += n;
    }
}

pub const SnapOp = engine.SnapOp;

/// Rank 0's engine: every step goes to the other ranks first, then runs here.
pub const Leader = struct {
    e: *Engine,
    link: *Link,

    fn send(x: Leader, h: Header, ids: []const u32) !void {
        const l = x.link;
        if (l.lost != null) return error.RankLost;
        var hh = h;
        hh.slot = x.e.slot;
        hh.n = @intCast(ids.len);
        if (ids.len > l.ids.len) return error.TooManyIds;
        if (l.asleep) { // the other ranks wait on their lines, not in NCCL: wake them for this step
            l.hooks.wake(l.hooks.ctx);
            l.asleep = false;
        }
        l.wire.vtable.send(l.wire.ptr, &hh, ids) catch |err| return x.lose(err);
    }

    /// The wire broke: the other ranks hear it, and every step from now on is refused without touching NCCL.
    fn lose(x: Leader, err: anyerror) error{RankLost} {
        x.link.lost = err;
        x.link.hooks.failed(x.link.hooks.ctx, @errorName(err));
        return error.RankLost;
    }

    /// No stream is left: the other ranks sleep on their lines until the next step instead of spinning in NCCL.
    pub fn idle(x: Leader) !void {
        if (x.link.lost != null or x.link.asleep or !x.link.hooks.live(x.link.hooks.ctx)) return;
        try x.send(.{ .kind = @backingInt(Kind.idle) }, &.{});
        x.link.asleep = true;
    }

    /// The pool of `slots` streams and `pages` pages on every rank (the idle kind with flag 1).
    pub fn attach(x: Leader, slots: u32, pages: usize) !void {
        try x.send(.{ .kind = @backingInt(Kind.idle), .flags = 1, .depth = slots, .a = pages }, &.{});
        try x.e.attach(slots, pages);
    }

    pub fn bind(x: Leader, slot: u32) !void {
        try x.e.bind(slot);
    }

    /// Frees a stream's sequence on every rank.
    pub fn release(x: Leader, slot: u32) !void {
        try x.e.bind(slot);
        try x.send(.{ .kind = @backingInt(Kind.release) }, &.{});
        x.e.release(slot);
    }

    /// A fresh sequence on the bound slot; `row` is its block-table row (the lease's page ids) as rank 0 admitted it.
    pub fn begin(x: Leader, s: ?engine.Sampling, row: []const u32) !void {
        const c = engine.check(s);
        var h: Header = .{ .kind = @backingInt(Kind.begin) };
        if (c) |v| {
            h.flags = 1;
            h.seed = v.seed;
            h.temperature = v.temperature;
            h.top_p = v.top_p;
            h.min_p = v.min_p;
            h.top_k = v.top_k;
        }
        try x.send(h, row);
        try x.e.begin(c);
    }

    /// Table entries from `from` on the other ranks (the keep kind with flag 1): rank 0 already wrote its own.
    pub fn sendPages(x: Leader, from: usize, ids: []const u32) !void {
        try x.send(.{ .kind = @backingInt(Kind.keep), .flags = 1, .depth = @intCast(from) }, ids);
    }

    /// One chunk `ids[start..chunkEnd]` of a prompt pass on every rank; the first token once the chunk ends the prompt.
    pub fn chunk(x: Leader, ids: []const u32, start: usize, from: ?engine.Resume, cancel: ?engine.Cancel, marks: ?engine.Marks) !?u32 {
        if (start >= ids.len) return error.BadChunk;
        const part = ids[start..engine.chunkEnd(start, ids.len, marks)];
        // a mark keeps the prefix before the step goes out: its page swap must reach the other ranks first
        if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
        try x.send(.{ .kind = @backingInt(Kind.chunk), .a = if (from) |r| r.at else 0, .depth = @intCast(start), .top_k = @intCast(ids.len) }, part);
        return x.e.prefillChunk(part, start, ids.len, from, cancel, null);
    }

    /// The whole prompt as chunk steps (the other ranks never see marks, only chunks), then the first token.
    pub fn prefill(x: Leader, ids: []const u32, from: ?engine.Resume, cancel: ?engine.Cancel, marks: ?engine.Marks) !u32 {
        var start: usize = if (from) |r| r.at else 0;
        var adopted = from;
        while (true) {
            if (try x.chunk(ids, start, adopted, cancel, marks)) |first| return first;
            start = engine.chunkEnd(start, ids.len, marks);
            adopted = null;
        }
    }

    pub fn verify(x: Leader, ids: []const u32, out: []u32, bits: u64) !void {
        try x.send(.{ .kind = @backingInt(Kind.verify) }, ids);
        try x.e.verify(ids, out, bits);
    }

    pub fn verifyShared(x: Leader, parts: []const iface.Part) !void {
        if (parts.len == 0 or parts.len > iface.max_shared) return error.WindowTooWide;
        var buf: [2 * iface.max_shared + iface.shared_rows]u32 = undefined;
        const n = try encodeShared(&buf, parts);
        try x.send(.{ .kind = @backingInt(Kind.shared), .depth = @intCast(parts.len) }, buf[0..n]);
        try x.e.verifyShared(parts);
    }

    pub fn keep(x: Leader, n: usize) !void {
        try x.send(.{ .kind = @backingInt(Kind.keep), .a = n }, &.{});
        try x.e.keep(n);
    }

    /// The bound stream's positions start ... into its draft ring from tap rows [row, row + rows), on every rank.
    pub fn absorb(x: Leader, start: usize, rows: usize, row: usize) !void {
        try x.send(.{ .kind = @backingInt(Kind.absorb), .depth = @intCast(start), .a = rows | (row << 32) }, &.{});
        try x.e.absorb(start, rows, row);
    }

    /// One draft block pass for the bound stream on every rank: `anchor` is its pending token at position `pos`.
    pub fn draftPass(x: Leader, anchor: u32, pos: usize) !void {
        try x.send(.{ .kind = @backingInt(Kind.draft), .depth = @intCast(pos), .a = anchor }, &.{});
        try x.e.draftPass(anchor, pos);
    }

    /// The bound stream's draft ring saved as snapshot `id`, restored from it, or `id` dropped, on every rank.
    pub fn draftSnap(x: Leader, what: SnapOp, id: u32) !void {
        if (!x.e.drafting()) return;
        try x.send(.{ .kind = @backingInt(Kind.snap), .flags = @backingInt(what), .depth = id }, &.{});
        try x.e.draftSnap(what, id);
    }

    /// The other ranks leave their loops; nothing is sent when the link already broke.
    pub fn end(x: Leader) !void {
        x.link.hooks.ended(x.link.hooks.ctx);
        if (x.link.lost != null) return;
        try x.send(.{ .kind = @backingInt(Kind.end) }, &.{});
    }
};

fn kindOf(h: Header) !Kind {
    return std.enums.fromInt(Kind, h.kind) orelse error.BadStep;
}

/// A rank other than 0 runs rank 0's steps until `end`; other errors are told to the others and returned.
pub fn follow(e: *Engine, link: *Link) !void {
    return run(e, link) catch |err| {
        link.hooks.failed(link.hooks.ctx, @errorName(err));
        return err;
    };
}

fn run(e: *Engine, link: *Link) !void {
    var sampled: [iface.shared_rows]u32 = undefined;
    link.hooks.sleep(link.hooks.ctx); // rank 0 starts with no stream: its first step wakes this rank
    while (true) {
        const h = try link.wire.vtable.receive(link.wire.ptr, link.ids);
        const ids = link.ids[0..h.n];
        const kind = try kindOf(h);
        switch (kind) {
            .end, .idle => {},
            else => try e.bind(h.slot),
        }
        switch (kind) {
            .begin => {
                try e.setPages(h.slot, 0, ids);
                try e.beginFollowing(if (h.flags & 1 != 0) .{ .seed = h.seed, .temperature = h.temperature, .top_p = h.top_p, .min_p = h.min_p, .top_k = h.top_k } else null);
            },
            .prefill, .chunk => {
                const from: ?engine.Resume = if (h.a != 0) .{ .at = @intCast(h.a) } else null;
                const start: usize = if (kind == .prefill) (if (from) |r| r.at else 0) else h.depth;
                const len: usize = if (kind == .prefill) ids.len else h.top_k;
                if (kind == .prefill) {
                    _ = e.prefill(ids, from, null, null) catch |err| switch (err) {
                        error.Cancelled, error.MemoryFloor => {}, // rank 0's gate stopped the same chunk
                        else => return err,
                    };
                } else {
                    _ = e.prefillChunk(ids, start, len, from, null, null) catch |err| switch (err) {
                        error.Cancelled, error.MemoryFloor => {}, // rank 0's gate stopped the same chunk
                        else => return err,
                    };
                }
            },
            .verify => try e.verify(ids, sampled[0..ids.len], 0),
            .shared => {
                var parts: [iface.max_shared]iface.Part = undefined;
                try decodeShared(ids, h.depth, &parts, &sampled);
                try e.verifyShared(parts[0..h.depth]);
            },
            .keep => if (h.flags & 1 != 0) try e.setPages(h.slot, h.depth, ids) else try e.keep(@intCast(h.a)),
            .release => e.release(h.slot),
            .absorb => try e.absorb(h.depth, @intCast(h.a & 0xffff_ffff), @intCast(h.a >> 32)),
            .draft => try e.draftPass(@intCast(h.a), h.depth),
            .snap => try e.draftSnap(std.enums.fromInt(SnapOp, h.flags) orelse return error.BadStep, h.depth),
            .idle => if (h.flags & 1 != 0) try e.attach(h.depth, @intCast(h.a)) else link.hooks.sleep(link.hooks.ctx),
            .end => {
                link.hooks.ended(link.hooks.ctx); // rank 0 ends on purpose: its line closing from here is expected
                return;
            },
        }
    }
}

test "the follower's header is 64 bytes, round trips, and its kinds are the contract's" {
    const h: Header = .{ .kind = @backingInt(Kind.chunk), .n = 3, .depth = 2048, .a = 4096, .seed = 1 << 62, .temperature = 0.6, .top_p = 0.95, .top_k = 7000, .slot = 3 };
    var back: Header = undefined;
    @memcpy(std.mem.asBytes(&back), std.mem.asBytes(&h));
    try std.testing.expectEqual(h, back);
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(Header));
    try std.testing.expectEqual(Kind.chunk, try kindOf(back));
    try std.testing.expectError(error.BadStep, kindOf(.{ .kind = 0 }));
    try std.testing.expectError(error.BadStep, kindOf(.{ .kind = 13 }));
    try std.testing.expectEqual(Kind.snap, try kindOf(.{ .kind = 12 }));
}

test "a shared step's words round trip and a malformed one is refused" {
    var a = [_]u32{ 1, 2, 3 };
    var b = [_]u32{ 9, 8 };
    var oa: [3]u32 = undefined;
    var ob: [2]u32 = undefined;
    const parts = [_]iface.Part{ .{ .slot = 4, .ids = &a, .out = &oa }, .{ .slot = 1, .ids = &b, .out = &ob } };
    var buf: [2 * iface.max_shared + iface.shared_rows]u32 = undefined;
    const n = try encodeShared(&buf, &parts);
    try std.testing.expectEqual(@as(usize, 4 + 5), n);
    var back: [iface.max_shared]iface.Part = undefined;
    var outs: [iface.shared_rows]u32 = undefined;
    try decodeShared(buf[0..n], 2, &back, &outs);
    try std.testing.expectEqual(@as(u32, 4), back[0].slot);
    try std.testing.expectEqualSlices(u32, &b, back[1].ids);
    try std.testing.expectEqual(@as(usize, 2), back[1].out.len);
    try std.testing.expectError(error.BadStep, decodeShared(buf[0 .. n - 1], 2, &back, &outs));
    try std.testing.expectError(error.BadStep, decodeShared(buf[0..n], 0, &back, &outs));
    try std.testing.expectError(error.WindowTooWide, encodeShared(buf[0..5], &parts));
}

test "the link carries a chunk, a table row and every shared window" {
    try std.testing.expect(Link.capacityFor(0) >= iface.prefill_rows);
    try std.testing.expect(Link.capacityFor(100_000) == 100_000);
}
