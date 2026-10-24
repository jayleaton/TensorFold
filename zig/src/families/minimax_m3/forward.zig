//! MiniMax-M3.1-Flash's forward on CUDA (plan T8a): decode, verify and shared windows, prompt chunks; UNEXERCISED.
const std = @import("std");
const cuda = @import("cuda");
const plan = @import("forward_plan.zig");
const exec = @import("forward_exec.zig");
const iface = @import("iface.zig");
const weights = @import("weights.zig");
const rows_cfg = @import("rows.zig");
const Config = @import("config.zig").Config;
const graphs = @import("graphs.zig");
const rows_plan = @import("rows_plan.zig");
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
    pos: []i32, // host copy of a graphed window's positions
    graphs: ?*graphs.Graphs = null, // windows replay captured graphs (null: every call eager)
    graphed: plan.Shape = .{ .rows = 1 }, // the rows-form shape the graph body plans

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, st: cuda.Stream, m: *Modules, w: *const weights.Weights, g: *const Geometry, b: *Buffers, pages: Pages, link: ?Link) !Forward {
        const c = g.cfg;
        var o = try rows_cfg.opts(.{}, .{ .rot = c.rotary_dim, .theta = c.rope_theta, .eps = c.eps }, .folded, .folded, true);
        o.flags.kv8 = pages.cache.kv8; // the writer and the readers (Paged.kv8, from the same Cache) agree on the pool's layout
        return .{
            .dev = .{ .d = d, .st = st, .m = m, .w = w, .g = g, .b = b, .pages = pages, .link = link, .opts = o },
            .tab = try gpa.alloc(i32, iface.shared_rows),
            .pos = try gpa.alloc(i32, iface.shared_rows),
        };
    }

    pub fn deinit(f: *Forward, gpa: std.mem.Allocator) void {
        if (f.graphs) |g| g.destroy();
        gpa.free(f.pos);
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
        if (f.graphs) |g| if (!sh.prompt and sh.rows <= g.settings.rows_max) return f.runRows(g, c, sh);
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

    /// Graphs from now on under `s` (inactive settings: none); a group of ranks votes on every miss.
    pub fn useGraphs(f: *Forward, gpa: std.mem.Allocator, s: graphs.Settings) !void {
        if (f.graphs) |g| g.destroy();
        f.graphs = null;
        if (!s.active(f.dev.g.world)) return;
        const g = try graphs.Graphs.create(gpa, f.dev.d, s);
        errdefer g.destroy();
        if (f.dev.link) |l| try g.join(f.dev.d, l.group, f.dev.st, f.dev.g.world);
        f.graphs = g;
    }

    /// A window in rows form: per-row positions and table offsets staged, its graph replayed (eager on first use).
    fn runRows(f: *Forward, g: *graphs.Graphs, c: Call, sh: plan.Shape) !Out {
        const dev = &f.dev;
        const n = sh.rows;
        const one = [1]iface.Seg{.{ .slot = c.slot, .row0 = 0, .rows = n }};
        const segs: []const iface.Seg = if (sh.shared) c.segs else &one;
        if (segs.len > iface.max_shared) return error.BadCall;
        const pos = f.pos[0..n];
        if (sh.shared) {
            @memcpy(pos, c.positions);
        } else for (pos, 0..) |*p, k| {
            p.* = @intCast(c.pos0 + k);
        }
        const taps = dev.taps.base != 0 and dev.g.n_taps > 0;
        const key = try g.keyOf(n, sh.max_pos, @backingInt(sh.head), @backingInt(sh.mode), segs, taps);
        var rs = sh;
        rs.shared = true;
        rs.max_pos = g.scorePos(key, dev.pages.mirror.max_pages * rows_plan.block);
        try plan.check(dev.g, rs);
        dev.pages.slot = c.slot;
        dev.pages.pos0 = c.pos0;
        if (dev.taps.base != 0) {
            dev.taps.prompt = false;
            dev.taps.n = @intCast(segs.len);
            @memcpy(dev.taps.segs[0..segs.len], segs);
        }
        try cuda.copy_source.enqueue(dev.d, dev.b.ptr(.ids), std.mem.sliceAsBytes(c.ids), dev.st);
        var gs: [iface.max_shared]rk.geom.Seg = undefined;
        for (segs, gs[0..segs.len]) |s, *o| o.* = .{ .slot = s.slot, .rows = s.rows };
        try dev.pages.mirror.setRows(dev.st, f.tab[0..n], gs[0..segs.len]);
        try cuda.copy_source.enqueue(dev.d, dev.b.ptr(.pos), std.mem.sliceAsBytes(pos), dev.st);
        f.graphed = rs;
        const body: graphs.gc.Body = .{ .ctx = f, .run = graphBody };
        if (try g.first(key)) try graphBody(f, dev.st) else _ = try g.cache.run(key, dev.st, f.fingerprint(), body);
        return out(dev.g, sh);
    }

    fn graphBody(ctx: *anyopaque, st: cuda.Stream) anyerror!void {
        const f: *Forward = @ptrCast(@alignCast(ctx));
        std.debug.assert(st.handle == f.dev.st.handle);
        try plan.plan(f.dev.g, f.graphed, &f.dev);
    }

    /// Every address a graph bakes in: a change (a grown kernel scratch, a new pool or taps buffer) drops all graphs.
    pub fn fingerprint(f: *const Forward) u64 {
        const dev = &f.dev;
        var words: [exec.Buffers.slots + 16]u64 = undefined;
        var n: usize = 0;
        for (dev.b.mem[0..dev.b.n]) |m| {
            words[n] = m.ptr;
            n += 1;
        }
        const de = &dev.m.dense;
        for ([_]?cuda.DeviceBuffer{ de.xh, de.z, de.counters }) |m| {
            words[n] = if (m) |x| x.ptr ^ (@as(u64, x.len) << 1) else 0;
            n += 1;
        }
        const fast: u64 = if (dev.link) |l| @intFromPtr(l.group.fast) else 0;
        const rest = [_]u64{ dev.pages.cache.pool, dev.pages.mirror.tables.ptr, dev.pages.mirror.row_tab.ptr, dev.taps.base, dev.g.n_taps, dev.m.dspark.part.ptr, fast, @intFromPtr(dev.w) };
        @memcpy(words[n..][0..rest.len], &rest);
        n += rest.len;
        return graphs.gc.fingerprintOf(words[0..n]);
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
    _ = graphs;
    _ = rows_plan;
}
