//! DeepSeek-V4.1's lane backend (lanes.backend.Backend) over a `Target` forward and a DSpark `Pass`, with the DSpark
//! depth as the round's row policy (lanes.trim, policy.zig). Drafts come back to the host every pass (the depth reads
//! their confidences), so windows carry them as host tokens; trees are first-position siblings (tree.zig).
const std = @import("std");
const Allocator = std.mem.Allocator;
const lanes = @import("lanes");
const be = lanes.backend;
const Stream = lanes.Stream;
const iface = @import("iface.zig");
const dspark = @import("dspark.zig");
const tree = @import("tree.zig");
const depth = @import("depth.zig");
const calib = @import("calib.zig");
const costs_mod = @import("costs.zig");
const Policy = @import("policy.zig").Policy;

pub const max_block = 16;

pub const Options = struct {
    shape: dspark.Shape,
    depth: depth.Settings = .{},
    siblings: u32 = 0, // first-position siblings (TF_DSV41_TREE - 1; 0: chains only)
    sib_rows: u32 = 4, // a sibling branch's rows at most (TF_DSV41_TREE_ROWS)
    slots: u32 = 4, // streams at once
};

/// A stream's slot and what the backend holds for it between rounds.
pub const St = struct {
    slot: u32,
    len: u64 = 0, // rows the target holds (provisional after a window until keep)
    base: u64 = 0, // the last window's start
    first: u32 = 0, // the prompt's choice (for `first`)
    held: [max_block]u32 = undefined, // the last pass's drafts (the main chain)
    conf: [max_block]f64 = undefined,
    n: u32 = 0, // drafts held (0: none this round)
    start: u64 = 0, // the pending token's position when they were drafted
    sib: [tree.max_siblings]tree.Sibling = undefined,
    sib_tokens: [tree.max_siblings][max_block]u32 = undefined, // each sibling's root then its continuation
    sib_len: [tree.max_siblings]u32 = @splat(0),
    n_sib: u32 = 0,
    shown: std.ArrayList(u32) = .empty, // the drafts this round's window verifies (node indices, window order)
    shown_set: bool = false,

    /// Node i of the held tree: the main chain's drafts, then each sibling's rows.
    pub fn nodes(st: *const St) u32 {
        var n = st.n;
        for (st.sib_len[0..st.n_sib]) |l| n += l;
        return n;
    }
};

pub const Lanes = struct {
    gpa: Allocator,
    target: iface.Target,
    pass: iface.Pass,
    opts: Options,
    streams: std.AutoHashMapUnmanaged(*Stream, *St) = .empty,
    free: std.ArrayList(u32) = .empty,
    drawn: std.ArrayList(u32) = .empty, // handle -> token
    policy: Policy,
    scratch: std.heap.ArenaAllocator,
    // a pass's buffers, sized for opts.slots
    drafts: []u32,
    conf: []f32,
    cand: []i32,
    base: []f32,

    pub fn init(gpa: Allocator, target: iface.Target, pass: iface.Pass, costs: costs_mod.Costs, opts: Options) !*Lanes {
        const x = try gpa.create(Lanes);
        errdefer gpa.destroy(x);
        const b = opts.shape.block;
        if (b > max_block or opts.siblings > tree.max_siblings) return error.BadOptions;
        const c = opts.shape.candidates;
        x.* = .{
            .gpa = gpa,
            .target = target,
            .pass = pass,
            .opts = opts,
            .policy = try Policy.init(gpa, costs, opts),
            .scratch = std.heap.ArenaAllocator.init(gpa),
            .drafts = try gpa.alloc(u32, opts.slots * b),
            .conf = try gpa.alloc(f32, opts.slots * b),
            .cand = try gpa.alloc(i32, opts.slots * b * c),
            .base = try gpa.alloc(f32, opts.slots * b * c),
        };
        x.policy.owner = x;
        var s = opts.slots;
        while (s > 0) : (s -= 1) try x.free.append(gpa, s - 1);
        return x;
    }

    pub fn deinit(x: *Lanes) void {
        var it = x.streams.valueIterator();
        while (it.next()) |st| x.drop(st.*);
        x.streams.deinit(x.gpa);
        x.free.deinit(x.gpa);
        x.drawn.deinit(x.gpa);
        x.policy.deinit();
        x.scratch.deinit();
        x.gpa.free(x.drafts);
        x.gpa.free(x.conf);
        x.gpa.free(x.cand);
        x.gpa.free(x.base);
        x.gpa.destroy(x);
    }

    fn drop(x: *Lanes, st: *St) void {
        st.shown.deinit(x.gpa);
        x.gpa.destroy(st);
    }

    pub fn backend(x: *Lanes) be.Backend {
        return .{ .ptr = x, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .probabilities = probabilities, .tree = heldTree, .release = release } };
    }

    /// Put the DSpark depth in charge of the engine's rows.
    pub fn attach(x: *Lanes, e: *lanes.Engine) void {
        e.trim = x.policy.rows();
    }

    /// What the engine's Config reads of this model: 16 rows a slot, `batch_rows` a shared forward, the table's costs.
    pub fn model(x: *const Lanes, arena: Allocator, batch_rows: u32) !lanes.Model {
        const v = x.policy.depth.costs.verify;
        const window = try arena.alloc(lanes.config.Cost, @min(v.len, costs_mod.slot_rows));
        for (window, 0..) |*w, i| w.* = .{ .width = @intCast(i + 1), .ms = v[i] };
        const shared = try arena.alloc(lanes.config.Cost, @min(v.len, batch_rows));
        for (shared, 0..) |*w, i| w.* = .{ .width = @intCast(i + 1), .ms = v[i] };
        return .{ .exact_width = costs_mod.slot_rows, .mtp = true, .speculate = true, .speculate_early = false, .drafts = x.opts.shape.block, .window_costs = window, .mtp_step_ms = 0.0, .hidden_rows = true, .batch_rows = batch_rows, .max_streams = x.opts.slots, .shared_costs = shared, .draft_probabilities = true, .draft_streams = true };
    }

    pub fn state(x: *Lanes, s: *Stream) ?*St {
        return x.streams.get(s);
    }

    fn self(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn get(x: *Lanes, s: *Stream) !*St {
        return x.streams.get(s) orelse error.UnknownStream;
    }

    fn value(x: *Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| x.drawn.items[h],
            .value => |v| v,
        };
    }

    fn ingest(x: *Lanes, st: *St, start: u64, rows: []const u32) !void {
        const span = iface.ingestSpan(rows.len, x.opts.shape.window);
        if (span.count == 0) return;
        try x.pass.ingest(st.slot, start + span.skip, x.target.taps(st.slot), rows[span.skip..]);
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        if (x.streams.get(s) != null) return error.StreamLive;
        const slot = x.free.pop() orelse return error.NoSlot;
        errdefer x.free.append(x.gpa, slot) catch {};
        const st = try x.gpa.create(St);
        errdefer x.gpa.destroy(st);
        st.* = .{ .slot = slot };
        x.pass.reset(slot);
        x.policy.depth.reset(slot);
        const ids = s.prompt();
        st.first = try x.target.prefill(slot, ids, s.sampling, ids.len);
        st.len = ids.len;
        _ = x.scratch.reset(.retain_capacity);
        const rows = try x.scratch.allocator().alloc(u32, ids.len);
        for (rows, 0..) |*r, i| r.* = @intCast(i);
        try x.ingest(st, 0, rows);
        try x.streams.put(x.gpa, s, st);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(ptr);
        const st = try x.get(s);
        if (position != st.len) return error.PositionMismatch;
        try x.drawn.append(x.gpa, st.first);
        return x.drawn.items.len - 1;
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        const st = try x.get(s);
        if (position != st.len + 1) return error.PositionMismatch;
        const token = [_]u32{x.value(feed)};
        var choice: [1]u32 = undefined;
        var out = [_][]u32{&choice};
        try x.target.window(&.{.{ .slot = st.slot, .start = st.len, .tokens = &token, .parents = null, .draws = &.{position}, .sampling = s.sampling }}, &out);
        st.base = st.len;
        st.len += 1;
        try x.ingest(st, st.base, &.{0});
        try x.drawn.append(x.gpa, choice[0]);
        return x.drawn.items.len - 1;
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).drawn.items[handle];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        _ = x.scratch.reset(.retain_capacity);
        const a = x.scratch.allocator();
        const segs = try a.alloc(iface.Segment, windows.len);
        const choices = try a.alloc([]u32, windows.len);
        for (windows, segs, choices, out) |w, *g, *c, o| {
            const st = try x.get(w.stream);
            if (w.positions[0] != st.len + 1) return error.PositionMismatch;
            const tokens = try a.alloc(u32, w.rows());
            tokens[0] = w.pending;
            if (w.held > st.n) return error.HeldMismatch;
            @memcpy(tokens[1 .. 1 + w.held], st.held[0..w.held]);
            @memcpy(tokens[1 + w.held ..], w.tokens);
            const parents = w.parents; // each row's parent row (row 0: -1)
            g.* = .{ .slot = st.slot, .start = st.len, .tokens = tokens, .parents = parents, .draws = w.positions, .sampling = w.stream.sampling };
            c.* = o.sampled;
            @memcpy(o.drafts, tokens[1..]);
        }
        try x.target.window(segs, choices);
        for (windows) |w| {
            const st = try x.get(w.stream);
            st.base = st.len;
            st.len += w.rows();
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        for (windows, paths) |w, path| {
            const st = try x.get(w.stream);
            try x.target.keep(st.slot, path);
            st.len = st.base + path.len;
        }
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        _ = x.scratch.reset(.retain_capacity);
        const a = x.scratch.allocator();
        const asks = try a.alloc(iface.Ask, requests.len);
        const whose = try a.alloc(*St, requests.len);
        const depths = try a.alloc(u32, requests.len);
        var n: usize = 0;
        for (requests) |r| {
            const st = try x.get(r.stream);
            var at = st.len;
            if (r.rows) |path| {
                if (r.start != st.base) return error.PositionMismatch;
                try x.ingest(st, r.start, path); // a shared round drafts before its keep: the path, not st.len
                at = r.start + path.len;
            }
            if (r.position != at + 1) return error.PositionMismatch;
            const anchor = if (r.first) |f| x.value(f) else r.follow[r.follow.len - 1];
            st.n = 0;
            st.n_sib = 0;
            st.start = at;
            if (r.depth == 0) continue;
            if (n == x.opts.slots) return error.TooManySlots;
            asks[n] = .{ .slot = st.slot, .anchor = anchor, .start = at, .params = dspark.Params.of(r.stream.sampling, at) };
            whose[n] = st;
            depths[n] = r.depth;
            n += 1;
        }
        if (n == 0) return;
        const b = x.opts.shape.block;
        const c = x.opts.shape.candidates;
        const markov = if (x.opts.siblings > 0) x.pass.markov() else null;
        const props = try a.alloc(iface.Proposal, n);
        for (props, 0..) |*p, i| p.* = .{
            .drafts = x.drafts[i * b ..][0..b],
            .conf = x.conf[i * b ..][0..b],
            .cand = if (markov != null) x.cand[i * b * c ..][0 .. b * c] else null,
            .base = if (markov != null) x.base[i * b * c ..][0 .. b * c] else null,
        };
        try x.pass.propose(asks[0..n], props);
        for (props, whose[0..n], depths[0..n], asks[0..n]) |p, st, d, ask| {
            st.n = @min(d, b);
            @memcpy(st.held[0..st.n], p.drafts[0..st.n]);
            for (st.conf[0..st.n], p.conf[0..st.n]) |*q, f| q.* = f;
            if (markov) |m| try x.siblings(a, st, m, p, ask);
        }
    }

    /// The siblings after the pending token: the draft distribution's next first tokens and their Markov chains.
    fn siblings(x: *Lanes, a: Allocator, st: *St, m: dspark.Markov, p: iface.Proposal, ask: iface.Ask) !void {
        const c = x.opts.shape.candidates;
        const z0 = try a.alloc(f32, c);
        dspark.markovZ(m, p.cand.?[0..c], p.base.?[0..c], ask.anchor, z0);
        const smp = ask.params.sampling();
        const got = try tree.ranked(a, p.cand.?[0..c], z0, ask.start + 1, smp, st.held[0..1], st.sib[0..x.opts.siblings]);
        const rows = @min(x.opts.sib_rows, st.n);
        for (st.sib[0..got], 0..) |sib, j| {
            st.sib_tokens[j][0] = sib.token;
            const more = try tree.continuation(a, m, p.cand.?, p.base.?, c, sib.token, rows, ask.params, st.sib_tokens[j][1..]);
            st.sib_len[j] = @intCast(1 + more);
        }
        st.n_sib = @intCast(got);
    }

    /// The held drafts as host tokens: the main chain, then each sibling's root and its chain (gpa-owned).
    fn heldTree(ptr: *anyopaque, s: *Stream, gpa: Allocator) anyerror!?lanes.stream.Held {
        const x = self(ptr);
        const st = try x.get(s);
        if (st.n == 0) return null;
        const total = st.nodes();
        const tokens = try gpa.alloc(u32, total);
        errdefer gpa.free(tokens);
        @memcpy(tokens[0..st.n], st.held[0..st.n]);
        if (st.n_sib == 0) return .{ .count = total, .tokens = tokens };
        const parents = try gpa.alloc(i32, total);
        for (parents[0..st.n], 0..) |*q, i| q.* = @as(i32, @intCast(i)) - 1;
        var at: usize = st.n;
        for (st.sib_tokens[0..st.n_sib], st.sib_len[0..st.n_sib]) |toks, len| {
            for (0..len) |i| {
                tokens[at + i] = toks[i];
                parents[at + i] = if (i == 0) -1 else @intCast(at + i - 1);
            }
            at += len;
        }
        return .{ .count = total, .tokens = tokens, .parents = parents };
    }

    /// Each held draft's chance of landing from the raw confidences (the engine's own allocation, without the policy).
    fn probabilities(ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const x = self(ptr);
        const st = try x.get(s);
        var run: f64 = 1.0;
        for (out, 0..) |*o, i| {
            run *= if (i < st.n) st.conf[i] else 0.0;
            o.* = run;
        }
        return st.n > 0;
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        const kv = x.streams.fetchRemove(s) orelse return;
        const st = kv.value;
        x.target.release(st.slot);
        x.pass.reset(st.slot);
        x.policy.depth.reset(st.slot);
        x.free.append(x.gpa, st.slot) catch {};
        x.drop(st);
    }
};
