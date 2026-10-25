//! engine_ops.Ops over forward.zig (T8a): the pass kinds become `Call`s, the head's gathered fp32 rows are the logits.
const std = @import("std");
const cuda = @import("cuda");
const iface = @import("iface.zig");
const ops = @import("engine_ops.zig");
const fw = @import("forward.zig");
const plan = @import("forward_plan.zig");
const pool = @import("pool.zig");
const shard = @import("shard.zig");
const rk = cuda.minimax_m3_rows;

pub const Impl = struct {
    gpa: std.mem.Allocator,
    g: plan.Geometry,
    m: fw.Modules,
    b: fw.Buffers,
    f: fw.Forward,
    dims: pool.Dims,
    mirror: ?rk.Mirror = null,
    addr: [3]u64 = @splat(0), // where the last pass of each kind left its head
    positions: [iface.shared_rows]i32 = undefined,
    link: ?fw.Link = null,
    args: ops.OpenArgs = undefined,
    made: bool = false,
    dims_slots: u32 = 0,
    taps: ?cuda.DeviceBuffer = null, // the dspark draft's tapped rows (iface.tapRows(slots) of them), made at `attach`

    pub fn open(a: ops.OpenArgs) !ops.Ops {
        const self = try a.gpa.create(Impl);
        errdefer a.gpa.destroy(self);
        const layers: u32 = if (a.settings.layers > 0) a.settings.layers else a.cfg.layers;
        self.* = .{ .gpa = a.gpa, .g = try plan.Geometry.of(a.cfg, a.share, layers, iface.max_rows), .m = try fw.Modules.load(a.d), .b = undefined, .f = undefined, .dims = (try pool.Dims.of(a.cfg, a.share)).withKv8(a.settings.kv8) };
        errdefer self.m.deinit();
        self.b = try fw.Buffers.init(a.d, &self.g, iface.maxPages(a.capacity));
        errdefer self.b.deinit();
        const link: ?fw.Link = if (a.place.world > 1) .{ .group = a.group, .lib = a.lib, .comm = a.comm } else null;
        // the pages arrive at `attach`: the forward is made there, over the engine's arena and block table
        self.f = undefined;
        self.link = link;
        self.args = a;
        return .{ .ptr = self, .vtable = &.{ .attach = attach, .run = run, .logits = logits, .sync = sync, .deinit = deinit, .set_taps = setTaps, .taps_base = tapsBase, .tap_copy = tapCopy } };
    }

    fn of(p: *anyopaque) *Impl {
        return @ptrCast(@alignCast(p));
    }

    fn attach(p: *anyopaque, view: ops.PoolView) anyerror!void {
        const s = of(p);
        const a = s.args;
        if (s.made) {
            s.f.deinit(s.gpa);
            if (s.mirror) |*m| m.row_tab.free();
        }
        s.dims_slots = view.slots;
        const tables: cuda.DeviceBuffer = .{ .d = a.d, .ptr = view.table, .len = view.slots * view.max_pages * 4, .borrowed = true };
        var row_tab = try cuda.DeviceBuffer.alloc(a.d, iface.shared_rows * 4);
        errdefer row_tab.free();
        s.mirror = .{ .tables = tables, .row_tab = row_tab, .slots = view.slots, .max_pages = view.max_pages, .max_rows = iface.shared_rows };
        const pages: fw.Pages = .{ .cache = .{ .pool = view.arena, .page_bytes = view.page_bytes, .layer_off = 0, .kvh = s.g.kvh, .kv8 = s.dims.kv8 }, .layer_bytes = s.dims.layerToken() * pool.page_tokens, .mirror = &s.mirror.? };
        s.f = try fw.Forward.init(s.gpa, a.d, a.stream, &s.m, a.weights, &s.g, &s.b, pages, s.link);
        s.made = true;
        if (s.g.n_taps > 0) { // one row of the buffer = every tap's hidden state side by side, bf16
            if (s.taps) |*t| t.free();
            s.taps = null;
            const row_bytes = @as(usize, s.g.n_taps) * s.g.hidden * 2;
            s.taps = try cuda.DeviceBuffer.alloc(a.d, iface.tapRows(view.slots) * row_bytes);
            s.f.dev.taps.base = s.taps.?.ptr;
        }
    }

    /// The draft's layers: from the next `attach` on the forward copies those layers' outputs into the taps buffer.
    fn setTaps(p: *anyopaque, ids: []const u32) anyerror!void {
        const s = of(p);
        if (ids.len == 0 or ids.len > plan.max_taps) return error.BadTaps;
        for (ids, 0..) |id, i| {
            if (id >= s.g.layers or (i > 0 and id <= ids[i - 1])) return error.BadTaps;
            s.g.tap_ids[i] = id;
        }
        s.g.n_taps = @intCast(ids.len);
    }

    fn tapsBase(p: *anyopaque) u64 {
        const s = of(p);
        return if (s.taps) |t| t.ptr else 0;
    }

    fn tapCopy(p: *anyopaque, from: u32, to: u32) anyerror!void {
        const s = of(p);
        const t = s.taps orelse return error.BadTaps;
        const row_bytes = @as(u64, s.g.n_taps) * s.g.hidden * 2;
        if (from >= iface.tapRows(s.dims_slots) or to >= iface.tapRows(s.dims_slots)) return error.BadTaps;
        try s.args.d.check(s.args.d.api.cuMemcpyDtoDAsync_v2(t.ptr + to * row_bytes, t.ptr + from * row_bytes, row_bytes, s.args.stream.handle), "cuMemcpyDtoDAsync");
    }

    fn run(p: *anyopaque, f: ops.Forward) anyerror!void {
        const s = of(p);
        if (!s.made) return error.NoPool;
        const ids: []const i32 = @ptrCast(f.ids);
        const head: plan.Head = if (!f.head) .none else if (f.kind == .prompt) .last else .all;
        var call: fw.Call = .{ .ids = ids, .slot = f.segs[0].slot, .pos0 = f.starts[0], .prompt = f.kind == .prompt, .head = head };
        if (f.kind == .shared) {
            for (f.segs, f.starts) |g, start| for (0..g.rows) |k| {
                s.positions[g.row0 + k] = @intCast(start + k);
            };
            call.segs = f.segs;
            call.positions = s.positions[0..f.ids.len];
        }
        const out = try s.f.run(call);
        s.addr[@backingInt(f.kind)] = if (head == .none) 0 else s.f.address(out);
    }

    fn logits(p: *anyopaque, kind: ops.Rows) u64 {
        return of(p).addr[@backingInt(kind)];
    }

    fn sync(p: *anyopaque) anyerror!void {
        const s = of(p);
        try s.args.stream.synchronize();
    }

    fn deinit(p: *anyopaque) void {
        const s = of(p);
        if (s.made) {
            s.f.deinit(s.gpa);
            if (s.mirror) |*m| m.row_tab.free();
        }
        if (s.taps) |*t| t.free();
        s.b.deinit();
        s.m.deinit();
        s.gpa.destroy(s);
    }
};
