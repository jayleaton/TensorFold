//! The dspark draft's checkpoint plan for one rank, from the safetensors header alone; reuses load.zig and shard.zig.
const std = @import("std");
// Nothing is converted at load, and norms are not folded to 1 + w: the Gemma offset is an unverified switch.
const core = @import("core");
const st = core.safetensors;
const layout = core.exl3_layout;
const load = @import("load.zig");
const dc = @import("dspark_config.zig");
const dn = @import("dspark_names.zig");
const names = @import("names.zig");
const shard = @import("shard.zig");

pub const Catalog = load.Catalog;
pub const Ref = load.Ref;
pub const Slot = load.Slot;
pub const Item = load.Item;

/// One draft linear layer on this rank: float8 codes [n, k] and one ue8m0 scale byte per 32 inputs [n, k / 32].
pub const Mat = struct {
    proj: dn.Lin,
    layer: u16,
    n: u32, // rank-local output rows
    k: u32, // rank-local input width
    codes: u32, // slot ids
    scales: u32,
};

/// A bf16 tensor on this rank (as stored).
pub const Tensor = struct {
    role: dn.Plain,
    layer: u16,
    slot: u32,
    rows: u32,
    cols: u32, // 1 for vectors
};

pub const Options = struct {
    /// Cut the Markov tables by the head's vocab units; false: both ranks hold them whole and run the chain alike.
    markov_cut: bool = false,
};

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    share: shard.Share,
    opts: Options,
    slots: []Slot,
    items: []Item, // in (shard, offset) order
    mats: []Mat,
    plain: []Tensor,
    ignored: usize, // tensors of other namespaces (the target's) left unread

    pub fn deinit(p: *Plan) void {
        p.arena.deinit();
        p.* = undefined;
    }

    pub fn slotBytes(p: *const Plan) u64 {
        var n: u64 = 0;
        for (p.slots) |s| n += s.bytes;
        return n;
    }

    pub fn mat(p: *const Plan, proj: dn.Lin, layer: u32) ?*const Mat {
        for (p.mats) |*m| if (m.proj == proj and m.layer == layer) return m;
        return null;
    }

    pub fn tensor(p: *const Plan, role: dn.Plain, layer: u32) ?*const Tensor {
        for (p.plain) |*t| if (t.role == role and t.layer == layer) return t;
        return null;
    }

    /// `cat` holds the draft's shards; `share` is this rank's cut of the TARGET. Refuses a missing or odd tensor.
    pub fn build(gpa: std.mem.Allocator, cat: *Catalog, c: *const dc.Config, share: shard.Share, opts: Options) !Plan {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        var b: Builder = .{ .cat = cat, .c = c, .share = share, .a = arena.allocator() };
        var used = cat.map.valueIterator();
        while (used.next()) |r| r.used = false;
        try b.all(opts);
        std.mem.sort(Item, b.items.items, {}, struct {
            fn less(_: void, x: Item, y: Item) bool {
                return if (x.shard != y.shard) x.shard < y.shard else x.first() < y.first();
            }
        }.less);
        return .{
            .arena = arena,
            .share = share,
            .opts = opts,
            .slots = b.slots.items,
            .items = b.items.items,
            .mats = b.mats.items,
            .plain = b.plain.items,
            .ignored = try b.leftovers(),
        };
    }
};

/// The scale byte types a checkpoint may spell its ue8m0 exponents with.
fn isScale(r: *const Ref, shape: []const usize) bool {
    return r.is(.u8, shape) or r.is(.f8_e8m0, shape);
}

const Builder = struct {
    cat: *Catalog,
    c: *const dc.Config,
    share: shard.Share,
    a: std.mem.Allocator,
    slots: std.ArrayList(Slot) = .empty,
    items: std.ArrayList(Item) = .empty,
    mats: std.ArrayList(Mat) = .empty,
    plain: std.ArrayList(Tensor) = .empty,

    fn slot(b: *Builder, bytes: u64) !u32 {
        try b.slots.append(b.a, .{ .bytes = bytes });
        return @intCast(b.slots.items.len - 1);
    }

    fn find(b: *Builder, name: []const u8) !*Ref {
        const r = b.cat.get(name) orelse {
            std.log.warn("dspark checkpoint has no tensor {s}", .{name});
            return error.MissingTensor;
        };
        r.used = true;
        return r;
    }

    fn all(b: *Builder, opts: Options) !void {
        const c = b.c;
        const s = b.share;
        if (c.layers == 0) return error.BadLayers;
        try b.plainTensor(.fc, 0, &.{ c.hidden, c.fcIn() }, null);
        try b.plainTensor(.hidden_norm, 0, &.{c.hidden}, null);
        try b.plainTensor(.final_norm, 0, &.{c.hidden}, null);
        const vocab_rows: ?shard.Span = if (opts.markov_cut) s.vocab else null;
        try b.plainTensor(.markov_w1, 0, &.{ c.vocab, c.markov_rank }, vocab_rows);
        try b.plainTensor(.markov_w2, 0, &.{ c.vocab, c.markov_rank }, vocab_rows);
        if (c.confidence) {
            try b.plainTensor(.conf_w, 0, &.{ 1, c.confIn() }, null);
            try b.plainTensor(.conf_b, 0, &.{1}, null);
        }
        for (0..c.layers) |l| try b.layer(@intCast(l));
    }

    fn layer(b: *Builder, l: u16) !void {
        const c = b.c;
        const s = b.share;
        const hd: usize = c.head_dim;
        const q: shard.Span = .{ .lo = s.q_heads.lo * hd, .hi = s.q_heads.hi * hd };
        const kv: shard.Span = .{ .lo = s.kv_heads.lo * hd, .hi = s.kv_heads.hi * hd };
        try b.plainTensor(.input_ln, l, &.{c.hidden}, null);
        try b.plainTensor(.post_ln, l, &.{c.hidden}, null);
        try b.plainTensor(.q_norm, l, &.{c.head_dim}, null);
        try b.plainTensor(.k_norm, l, &.{c.head_dim}, null);
        try b.linear(.q, l, c.qWidth(), c.hidden, q);
        try b.linear(.k, l, c.kvWidth(), c.hidden, kv);
        try b.linear(.v, l, c.kvWidth(), c.hidden, kv);
        try b.linear(.o, l, c.hidden, c.qWidth(), q);
        try b.linear(.gate, l, c.ffn, c.hidden, s.dense);
        try b.linear(.up, l, c.ffn, c.hidden, s.dense);
        try b.linear(.down, l, c.hidden, c.ffn, s.dense);
    }

    fn bad(why: anyerror, name: []const u8, comptime fmt: []const u8, args: anytype) anyerror {
        std.log.warn("{s}: " ++ fmt, .{name} ++ args);
        return why;
    }

    fn plainTensor(b: *Builder, role: dn.Plain, l: u16, shape: []const usize, rows: ?shard.Span) !void {
        var buf: dn.Buf = undefined;
        const name = dn.plain(&buf, role, l);
        const r = try b.find(name);
        if (!r.is(.bf16, shape)) return bad(error.MisshapedTensor, name, "{t} {any} is not bf16 {any}", .{ r.dtype, r.shape[0..r.rank], shape });
        const cols: usize = if (shape.len > 1) shape[1] else 1;
        const row_bytes = cols * 2;
        var offset: u64 = 0;
        var len: u64 = r.len;
        var n_rows: usize = shape[0];
        if (rows) |sp| {
            offset = sp.lo * row_bytes;
            len = sp.len() * row_bytes;
            n_rows = sp.len();
        }
        const id = try b.slot(len);
        try b.items.append(b.a, .{ .shard = r.shard, .base = r.begin, .span = .{ .offset = offset, .len = len }, .slot = id, .dst = 0 });
        try b.plain.append(b.a, .{ .role = role, .layer = l, .slot = id, .rows = @intCast(n_rows), .cols = @intCast(cols) });
    }

    /// One mxfp8 matrix [n, k]: this rank's cut `sp` (output rows for a col cut, input width for a row cut).
    fn linear(b: *Builder, proj: dn.Lin, l: u16, n: usize, k: usize, sp: shard.Span) !void {
        const blk: usize = b.c.quant.block_k;
        var nb: [2]dn.Buf = undefined;
        const wn = dn.lin(&nb[0], proj, l, .weight);
        const sn = dn.lin(&nb[1], proj, l, .weight_scale_inv);
        const w = try b.find(wn);
        const sc = try b.find(sn);
        if (!w.is(.f8_e4m3, &.{ n, k })) return bad(error.MisshapedTensor, wn, "{t} {any} is not float8 e4m3 [{d}, {d}]", .{ w.dtype, w.shape[0..w.rank], n, k });
        if (k % blk != 0 or !isScale(sc, &.{ n, k / blk })) return bad(error.MisshapedTensor, sn, "{t} {any} is not one ue8m0 byte per {d} inputs [{d}, {d}]", .{ sc.dtype, sc.shape[0..sc.rank], blk, n, k / blk });
        const whole = if (proj.cut() == .col) n else k;
        if (sp.lo >= sp.hi or sp.hi > whole) return bad(error.SplitNotOnBlocks, wn, "cut [{d}, {d}) of {d}", .{ sp.lo, sp.hi, whole });
        if (proj.cut() == .row and (sp.lo % blk != 0 or sp.hi % blk != 0)) return bad(error.SplitNotOnBlocks, wn, "input cut [{d}, {d}) is not whole {d}-blocks", .{ sp.lo, sp.hi, blk });
        var code: layout.Span = undefined;
        var scale: layout.Span = undefined;
        var rn: usize = n;
        var rk: usize = k;
        switch (proj.cut()) {
            .col => { // rows lo..hi of [n, k] are one run
                code = .{ .offset = sp.lo * k, .len = sp.len() * k };
                scale = .{ .offset = sp.lo * (k / blk), .len = sp.len() * (k / blk) };
                rn = sp.len();
            },
            .row => { // columns lo..hi of every row: n runs, a row apart
                code = .{ .offset = sp.lo, .len = sp.len(), .stride = k, .count = n };
                scale = .{ .offset = sp.lo / blk, .len = sp.len() / blk, .stride = k / blk, .count = n };
                rk = sp.len();
            },
        }
        const cs = try b.slot(code.bytes());
        const ss = try b.slot(scale.bytes());
        try b.items.append(b.a, .{ .shard = w.shard, .base = w.begin, .span = code, .slot = cs, .dst = 0 });
        try b.items.append(b.a, .{ .shard = sc.shard, .base = sc.begin, .span = scale, .slot = ss, .dst = 0 });
        try b.mats.append(b.a, .{ .proj = proj, .layer = l, .n = @intCast(rn), .k = @intCast(rk), .codes = cs, .scales = ss });
    }

    /// A tensor in the draft's namespace no planned name took is a pack this plan does not understand.
    fn leftovers(b: *Builder) !usize {
        var ignored: usize = 0;
        var it = b.cat.map.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.used) continue;
            if (std.mem.startsWith(u8, kv.key_ptr.*, dn.root)) {
                std.log.warn("unexpected dspark tensor {s}", .{kv.key_ptr.*});
                return error.UnexpectedTensor;
            }
            ignored += 1;
        }
        return ignored;
    }
};

/// Adds one shard's JSON header (`data_len` data bytes after it) to `cat` as shard `index`, without a file.
pub fn addHeader(cat: *Catalog, index: u16, json: []const u8, data_len: usize) !void {
    const a = cat.arena.allocator();
    const header = try st.parseHeader(a, json, data_len);
    const data = 8 + json.len;
    for (header.keys(), header.values()) |name, e| try cat.add(name, .{
        .shard = index,
        .dtype = e.dtype,
        .rank = e.rank,
        .shape = e.shape,
        .begin = data + e.begin,
        .len = e.end - e.begin,
    });
}

/// The folder's checkpoint catalog: dspark/'s index.json and shard headers, read with direct I/O like the target's.
pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Catalog {
    return Catalog.open(gpa, io, dir);
}
