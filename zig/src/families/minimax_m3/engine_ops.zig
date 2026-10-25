//! The seam between the engine loop (engine, follow, backend) and the forward and prompt passes (forward, prefill).
const std = @import("std");
const cuda = @import("cuda");
const iface = @import("iface.zig");
const config = @import("config.zig");
const shard = @import("shard.zig");
const weights = @import("weights.zig");
const ranks = @import("ranks.zig");

pub const Sampling = iface.Sampling;

/// Which forward buffer holds the logits of a pass; a prompt pass keeps only its last row, at row 0.
pub const Rows = enum { prompt, window, shared };

/// One forward: `ids` are every segment's rows in order, `starts[i]` segment i's first position; no `head`: no logits.
pub const Forward = struct { kind: Rows, ids: []const u32, segs: []const iface.Seg, starts: []const u32, head: bool = true };

/// The device pool a forward reads and writes: the page arena and the int32 block table (iface decision M3).
pub const PoolView = struct { arena: u64, page_bytes: u64, pages: usize, table: u64, slots: u32, max_pages: usize };

/// What the forward and prompt passes provide; all calls come from the engine thread, in step order on every rank.
pub const Ops = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// The pool the engine allocated (again, when `attach` is called again); no forward is in flight.
        attach: *const fn (ptr: *anyopaque, view: PoolView) anyerror!void,
        /// Enqueues one forward on the engine's stream; the KV rows of its ids are written at their positions.
        run: *const fn (ptr: *anyopaque, f: Forward) anyerror!void,
        /// Device address of the fp32 logits of the last pass of `kind`, gathered on every rank: [rank][row][slot].
        logits: *const fn (ptr: *anyopaque, kind: Rows) u64,
        /// Waits for every enqueued forward.
        sync: *const fn (ptr: *anyopaque) anyerror!void,
        deinit: *const fn (ptr: *anyopaque) void,
        /// The draft's layers (ascending) the forward copies into its taps buffer from the next `attach` on.
        set_taps: ?*const fn (ptr: *anyopaque, ids: []const u32) anyerror!void = null,
        /// Device address of row 0 of the taps buffer (0: none).
        taps_base: ?*const fn (ptr: *anyopaque) u64 = null,
        /// Copies tap row `from` to tap row `to` (a prompt's last row to its slot's window region), enqueued.
        tap_copy: ?*const fn (ptr: *anyopaque, from: u32, to: u32) anyerror!void = null,
    };

    pub fn setTaps(o: Ops, ids: []const u32) !void {
        const f = o.vtable.set_taps orelse return error.TapsUnsupported;
        return f(o.ptr, ids);
    }
    pub fn tapsBase(o: Ops) u64 {
        const f = o.vtable.taps_base orelse return 0;
        return f(o.ptr);
    }
    pub fn tapCopy(o: Ops, from: u32, to: u32) !void {
        const f = o.vtable.tap_copy orelse return error.TapsUnsupported;
        return f(o.ptr, from, to);
    }

    pub fn attach(o: Ops, view: PoolView) !void {
        return o.vtable.attach(o.ptr, view);
    }
    pub fn run(o: Ops, f: Forward) !void {
        return o.vtable.run(o.ptr, f);
    }
    pub fn logits(o: Ops, kind: Rows) u64 {
        return o.vtable.logits(o.ptr, kind);
    }
    pub fn sync(o: Ops) !void {
        return o.vtable.sync(o.ptr);
    }
};

/// The ranks' collectives (NCCL in engine_dev.zig): the engine never touches the library in its own logic.
pub const Comm = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    rank: usize,
    world: usize,

    pub const VTable = struct {
        /// The ranks' maximum of `v`, on every rank.
        max_f64: *const fn (ptr: *anyopaque, v: f64) anyerror!f64,
        /// Every rank's `mine` (equal lengths) in `out`, in rank order.
        gather_bytes: *const fn (ptr: *anyopaque, mine: []const u8, out: []u8) anyerror!void,
        /// A failed small gather's error, after a synchronize that read one.
        check: *const fn (ptr: *anyopaque) anyerror!void,
    };

    pub fn maxF64(c: Comm, v: f64) !f64 {
        return c.vtable.max_f64(c.ptr, v);
    }
    pub fn gatherBytes(c: Comm, mine: []const u8, out: []u8) !void {
        return c.vtable.gather_bytes(c.ptr, mine, out);
    }
    pub fn check(c: Comm) !void {
        return c.vtable.check(c.ptr);
    }
};

/// A draw of rows [row0, row0 + rows) of a `total`-row pass gathered at `base`; `bits`: host address of grammar words.
pub const Draw = struct { base: u64, total: usize, row0: usize, rows: usize, first: u64, rule: ?Sampling, bits: u64, out: []u32 };

/// The token draw: rank 0 alone draws (another rank's `out` stays untouched).
pub const Drawer = struct {
    ptr: *anyopaque,
    draw_fn: *const fn (ptr: *anyopaque, d: Draw) anyerror!void,

    pub fn draw(r: Drawer, d: Draw) !void {
        return r.draw_fn(r.ptr, d);
    }
};

/// The device block table and page arena, made at `attach` (engine_dev.zig); the host fakes it with an array.
pub const Device = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// A fresh arena of `pages` pages and a table of `slots` rows of -1, replacing any earlier ones.
        attach: *const fn (ptr: *anyopaque, slots: u32, pages: usize, max_pages: usize, page_bytes: u64) anyerror!PoolView,
        /// `entries` written at entry `from` of `slot`'s table row.
        upload: *const fn (ptr: *anyopaque, slot: u32, from: usize, entries: []const i32) anyerror!void,
        deinit: *const fn (ptr: *anyopaque) void,
    };
};

/// One pass of the draft block: the anchor (the stream's pending token) sits at position `pos`; the ring holds < pos.
pub const DraftPass = struct { slot: u32, anchor: u32, pos: u32 };

/// What a pass left: the drafted tokens and the confidence of each (`n` of them; token -1: none was found).
pub const DraftResult = struct { n: u32 = 0, tokens: [8]i32 = @splat(-1), conf: [8]f32 = @splat(0) };

/// The dspark draft on one rank: per-stream K/V rings over the target's tapped rows, the block pass, its results.
pub const Draft = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Rings for `slots` streams, all empty (again, when `attach` is called again).
        attach: *const fn (ptr: *anyopaque, slots: u32) anyerror!void,
        /// The slot's stream starts over: its ring is empty.
        begin: *const fn (ptr: *anyopaque, slot: u32) void,
        /// Tap rows [row0, row0 + rows) of the taps buffer at `base` are the slot's positions start ... into its ring.
        absorb: *const fn (ptr: *anyopaque, slot: u32, start: u32, rows: u32, base: u64, row0: u32) anyerror!void,
        /// Enqueues the block pass for the slot; its drafts stay on the device.
        pass: *const fn (ptr: *anyopaque, p: DraftPass) anyerror!void,
        /// The slot's last pass, waited for (rank 0 reads it; the others may too).
        result: *const fn (ptr: *anyopaque, slot: u32, out: *DraftResult) anyerror!void,
        /// The slot's ring now (copied to snapshot `id`), restored from `id`, or `id` dropped (ranks agree on ids).
        snap_save: *const fn (ptr: *anyopaque, slot: u32, id: u32) anyerror!void,
        snap_restore: *const fn (ptr: *anyopaque, slot: u32, id: u32) anyerror!void,
        snap_drop: *const fn (ptr: *anyopaque, id: u32) void,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn attach(d: Draft, slots: u32) !void {
        return d.vtable.attach(d.ptr, slots);
    }
    pub fn begin(d: Draft, slot: u32) void {
        d.vtable.begin(d.ptr, slot);
    }
    pub fn absorb(d: Draft, slot: u32, start: u32, rows: u32, base: u64, row0: u32) !void {
        return d.vtable.absorb(d.ptr, slot, start, rows, base, row0);
    }
    pub fn pass(d: Draft, p: DraftPass) !void {
        return d.vtable.pass(d.ptr, p);
    }
    pub fn result(d: Draft, slot: u32, out: *DraftResult) !void {
        return d.vtable.result(d.ptr, slot, out);
    }
    pub fn snapSave(d: Draft, slot: u32, id: u32) !void {
        return d.vtable.snap_save(d.ptr, slot, id);
    }
    pub fn snapRestore(d: Draft, slot: u32, id: u32) !void {
        return d.vtable.snap_restore(d.ptr, slot, id);
    }
    pub fn snapDrop(d: Draft, id: u32) void {
        d.vtable.snap_drop(d.ptr, id);
    }
};

/// What `openOps` is given: the loaded weights of this rank and the handles a forward needs.
pub const OpenArgs = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    cfg: *const config.Config,
    share: shard.Share,
    weights: *const weights.Weights,
    settings: ranks.Settings,
    place: ranks.Place,
    capacity: usize,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    /// The ranks' group over `lib` and `comm`; the host sets its RoCE small gather (`fast`) after `openOps`.
    group: *const cuda.tp_group.Group,
};

/// Whether forward.zig and prefill.zig are in this tree. MERGE: becomes true when `openOps` below calls them.
pub fn available() bool {
    return true;
}

/// MERGE POINT with branch mm3-t8a: the body becomes `return forward.open(args);` (forward.zig builds the buffers).
pub fn openOps(args: OpenArgs) !Ops {
    return @import("forward_ops.zig").Impl.open(args);
}

test "the pass kinds are the three buffers a forward owns" {
    try std.testing.expectEqual(@as(usize, 2), @backingInt(Rows.shared));
    try std.testing.expect(available());
}
