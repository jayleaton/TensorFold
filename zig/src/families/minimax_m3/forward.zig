//! MiniMax-M3.1-Flash's forward on CUDA (plan T8a): decode, verify and shared windows, prompt chunks; UNEXERCISED.
const std = @import("std");
const cuda = @import("cuda");
const plan = @import("forward_plan.zig");
const exec = @import("forward_exec.zig");
const iface = @import("iface.zig");
const weights = @import("weights.zig");
const rows_cfg = @import("rows.zig");
const Config = @import("config.zig").Config;
const rk = cuda.minimax_m3_rows;

pub const Geometry = plan.Geometry;
pub const Modules = exec.Modules;
pub const Buffers = exec.Buffers;
pub const Link = exec.Link;
pub const Pages = exec.Pages;
pub const Head = plan.Head;
pub const HeadMode = plan.HeadMode;
pub const Error = error{ BadCall, BadCommit };

/// One forward call: the ids of its rows and where they sit.
pub const Call = struct {
    ids: []const i32,
    slot: u32 = 0, // one stream: its slot, the position of its first row
    pos0: u32 = 0,
    segs: []const iface.Seg = &.{}, // several streams' windows, rows in order (empty: one stream)
    positions: []const i32 = &.{}, // shared: the position of every row
    prompt: bool = false, // a prompt chunk: prompt kernels, one row of logits at most
    head: Head = .all,
    mode: HeadMode = .gather,
};

/// Where a call left the head: `rows` rows of logits ([rank][row][slot] fp32 once gathered) or (logit, id) pairs.
pub const Out = struct { rows: u32, slot: u32, world: u32, picks: bool };

fn maxOf(xs: []const i32) u32 {
    var m: i32 = 0;
    for (xs) |x| m = @max(m, x);
    return @intCast(m);
}

pub const Forward = struct {
    dev: exec.Device,
    tab: []i32, // host copy of a shared window's per-row table offsets

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, st: cuda.Stream, m: *Modules, w: *const weights.Weights, g: *const Geometry, b: *Buffers, pages: Pages, link: ?Link) !Forward {
        const c = g.cfg;
        var o = try rows_cfg.opts(.{}, .{ .rot = c.rotary_dim, .theta = c.rope_theta, .eps = c.eps }, .folded, .folded, true);
        o.flags.kv8 = pages.cache.kv8; // the writer and the readers (Paged.kv8, from the same Cache) agree on the pool's layout
        return .{
            .dev = .{ .d = d, .st = st, .m = m, .w = w, .g = g, .b = b, .pages = pages, .link = link, .opts = o },
            .tab = try gpa.alloc(i32, iface.shared_rows),
        };
    }

    pub fn deinit(f: *Forward, gpa: std.mem.Allocator) void {
        gpa.free(f.tab);
        f.* = undefined;
    }

    /// The call's Shape: rows, kind and the largest position the scoring grid must cover.
    pub fn shape(c: Call) Error!plan.Shape {
        const n: u32 = @intCast(c.ids.len);
        const shared = c.segs.len > 0;
        if (n == 0 or (shared and (c.prompt or c.positions.len != n))) return error.BadCall;
        var rows: u32 = 0;
        for (c.segs) |s| rows += s.rows;
        if (shared and rows != n) return error.BadCall;
        const last = if (shared) maxOf(c.positions) else c.pos0 + n - 1;
        return .{ .rows = n, .prompt = c.prompt, .shared = shared, .max_pos = last, .head = c.head, .mode = c.mode };
    }

    /// The ids (and a shared window's positions and table offsets) go to the device; the ops are queued on the stream.
    pub fn run(f: *Forward, c: Call) !Out {
        const sh = try shape(c);
        const dev = &f.dev;
        const b = dev.b;
        dev.pages.slot = c.slot;
        dev.pages.pos0 = c.pos0;
        if (dev.taps.base != 0) { // the rows' streams, for the tap ops to find where each stream's rows go
            dev.taps.prompt = c.prompt;
            dev.taps.n = @intCast(@max(c.segs.len, 1));
            if (c.segs.len > 0) @memcpy(dev.taps.segs[0..c.segs.len], c.segs) else dev.taps.segs[0] = .{ .slot = c.slot, .row0 = 0, .rows = sh.rows };
        }
        try cuda.copy_source.enqueue(dev.d, b.ptr(.ids), std.mem.sliceAsBytes(c.ids), dev.st);
        if (sh.shared) {
            if (c.segs.len > iface.max_shared) return error.BadCall;
            var segs: [iface.max_shared]rk.geom.Seg = undefined;
            for (c.segs, segs[0..c.segs.len]) |s, *o| o.* = .{ .slot = s.slot, .rows = s.rows };
            try dev.pages.mirror.setRows(dev.st, f.tab[0..sh.rows], segs[0..c.segs.len]);
            try cuda.copy_source.enqueue(dev.d, b.ptr(.pos), std.mem.sliceAsBytes(c.positions), dev.st);
        }
        try plan.plan(dev.g, sh, dev);
        return out(dev.g, sh);
    }

    /// Where `sh`'s head left its rows.
    pub fn out(g: *const Geometry, sh: plan.Shape) Out {
        const n: u32 = if (sh.head == .all) sh.rows else 1;
        return .{ .rows = if (sh.head == .none) 0 else n, .slot = g.slot, .world = g.world, .picks = sh.mode == .picks };
    }

    /// The device address of the head's result for `o` (a call's `Out` carries the layout, not the address).
    pub fn address(f: *const Forward, o: Out) u64 {
        const id: plan.Buf = if (o.picks) (if (o.world > 1) .pairs_all else .pairs) else (if (o.world > 1) .hgather else .hlog);
        return f.dev.b.ptr(id);
    }

    /// The next position after keeping `keep` of a window's rows (later rows are overwritten, no state rolls back).
    pub fn commit(pos: usize, rows: usize, keep: usize, prompt: bool) Error!usize {
        if (keep == 0 or keep > rows or (prompt and keep != rows)) return error.BadCommit;
        return pos + keep;
    }
};

test {
    _ = @import("forward_test.zig");
}
