//! MiniMax-M3.1-Flash's checkpoint: header catalog, per-rank plan, direct-I/O stream; no shard is ever mapped.
const std = @import("std");
const builtin = @import("builtin");
const core = @import("core");
const st = core.safetensors;
const dio = core.direct_io;
const layout = core.exl3_layout;
const Config = @import("config.zig").Config;
const names = @import("names.zig");
const shard = @import("shard.zig");

comptime {
    if (builtin.cpu.arch.endian() != .little) @compileError("tensor bytes are little-endian and are copied as they are");
}

/// The mul1 codebook marker's value: exl3_decode.cuh hard-codes this multiplier.
pub const mul1_multiplier: u32 = 0x83DCD12D;

/// Half-bit widths (2 * bits) admitted for routed experts (the exl3_experts kernels): 3, 4, 5 and 6 bits.
pub const expert_halves = [_]u32{ 6, 8, 10, 12 };
/// Every other matrix (attention, dense and shared MLP, head) runs cuda.exl3.Dense: mul1 at 3 to 8 bits.
pub const dense_halves = [_]u32{ 6, 8, 10, 12, 14, 16 };

/// One stored tensor: where its bytes sit in which shard (absolute file offset) and what its header said.
pub const Ref = struct {
    shard: u16,
    dtype: st.DType,
    rank: u8,
    shape: [st.max_rank]usize,
    begin: u64,
    len: u64,
    used: bool = false,

    pub fn is(r: Ref, dtype: st.DType, shape: []const usize) bool {
        return r.dtype == dtype and r.rank == shape.len and std.mem.eql(usize, r.shape[0..r.rank], shape);
    }

    pub fn numel(r: Ref) usize {
        var n: usize = 1;
        for (r.shape[0..r.rank]) |d| n *= d;
        return n;
    }
};

/// Every tensor of a model folder by name, from the shard headers alone.
pub const Catalog = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    map: std.StringHashMapUnmanaged(Ref) = .empty,
    files: std.ArrayList(dio.File) = .empty,

    pub fn init(gpa: std.mem.Allocator) Catalog {
        return .{ .gpa = gpa, .arena = .init(gpa) };
    }

    pub fn deinit(c: *Catalog) void {
        for (c.files.items) |*f| f.close();
        c.files.deinit(c.gpa);
        c.map.deinit(c.gpa);
        c.arena.deinit();
        c.* = undefined;
    }

    /// Adds one tensor; a name seen twice (in one shard or two) is refused.
    pub fn add(c: *Catalog, name: []const u8, ref: Ref) !void {
        const got = try c.map.getOrPut(c.gpa, name);
        if (got.found_existing) {
            std.log.warn("tensor {s} is stored twice", .{name});
            return error.DuplicateTensor;
        }
        errdefer _ = c.map.remove(name);
        got.key_ptr.* = try c.arena.allocator().dupe(u8, name);
        got.value_ptr.* = ref;
    }

    /// The model folder's shards (the index's, else every model*.safetensors), their headers read with direct I/O.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Catalog {
        const paths = try core.checkpoint.shardFiles(gpa, io, dir);
        defer core.checkpoint.freeShardFiles(gpa, paths);
        var c = Catalog.init(gpa);
        errdefer c.deinit();
        const a = c.arena.allocator();
        var base_names: std.ArrayList([]const u8) = .empty;
        defer base_names.deinit(gpa);
        for (paths, 0..) |path, i| {
            if (i >= std.math.maxInt(u16)) return error.TooManyShards;
            var plain = try std.Io.Dir.cwd().openFile(io, path, .{});
            const size = plain.length(io) catch |e| {
                plain.close(io);
                return e;
            };
            plain.close(io);
            var f = try dio.File.open(path);
            try c.files.append(gpa, f);
            try base_names.append(gpa, std.fs.path.basename(path));
            const json = try readHeader(gpa, &f, size);
            defer gpa.free(json.mem);
            try uniqueKeys(gpa, json.text);
            const data = 8 + json.text.len;
            const header = try st.parseHeader(a, json.text, size - data);
            for (header.keys(), header.values()) |name, e| try c.add(name, .{
                .shard = @intCast(i),
                .dtype = e.dtype,
                .rank = e.rank,
                .shape = e.shape,
                .begin = data + e.begin,
                .len = e.end - e.begin,
            });
        }
        try c.crossCheckIndex(io, dir, base_names.items);
        return c;
    }

    /// With an index.json present, its weight map and the headers must name the same tensors in the same shards.
    fn crossCheckIndex(c: *Catalog, io: std.Io, dir: []const u8, shard_names: []const []const u8) !void {
        const path = try std.fs.path.join(c.gpa, &.{ dir, "model.safetensors.index.json" });
        defer c.gpa.free(path);
        const text = std.Io.Dir.cwd().readFileAlloc(io, path, c.gpa, .limited(st.index.max_bytes)) catch |e| switch (e) {
            error.FileNotFound => return,
            else => return e,
        };
        defer c.gpa.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, c.gpa, text, .{});
        defer parsed.deinit();
        const wm = try st.index.weightMap(parsed.value);
        if (wm.count() != c.map.count()) return error.IndexMismatch;
        var it = wm.iterator();
        while (it.next()) |kv| {
            const r = c.map.get(kv.key_ptr.*) orelse return error.IndexMismatch;
            if (!std.mem.eql(u8, shard_names[r.shard], kv.value_ptr.string)) return error.IndexMismatch;
        }
    }

    pub fn get(c: *Catalog, name: []const u8) ?*Ref {
        return c.map.getPtr(name);
    }
};

const Header = struct { mem: dio.Buffer, text: []const u8 };

/// The JSON header of a shard of `size` bytes, read through `f`'s direct I/O into an aligned buffer.
fn readHeader(gpa: std.mem.Allocator, f: *dio.File, size: u64) !Header {
    if (size < 8) return error.BadSafetensors;
    const first = try gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), dio.span(0, 8));
    defer gpa.free(first);
    const n = try st.headerSize(std.mem.readInt(u64, (try f.read(first, 0, 8))[0..8], .little), size);
    const mem = try gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), dio.span(8, n));
    errdefer gpa.free(mem);
    return .{ .mem = mem, .text = try f.read(mem, 8, n) };
}

/// A name twice in one header would let the parsed map keep either: count the top-level keys as written.
fn uniqueKeys(gpa: std.mem.Allocator, json: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var scanner = std.json.Scanner.initCompleteInput(a, json);
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var depth: usize = 0;
    while (true) {
        const tok = try scanner.nextAlloc(a, .alloc_if_needed);
        switch (tok) {
            .object_begin, .array_begin => depth += 1,
            .object_end, .array_end => depth -= 1,
            .string, .allocated_string => |s| if (depth == 1) {
                if ((try seen.getOrPut(a, s)).found_existing) return error.DuplicateTensor;
            },
            .end_of_document => return,
            else => {},
        }
    }
}

/// A device allocation the plan asks for; the sink decides what backs it.
pub const Slot = struct { bytes: u64 };

/// Bytes `span` of one stored tensor (absolute file offset `base`), appended at `dst` of `slot`.
pub const Item = struct {
    shard: u16,
    base: u64,
    span: layout.Span,
    slot: u32,
    dst: u64,
    fold: bool = false, // bf16 norm weights stored as fp32 1 + w (the destination holds 4-byte floats)

    pub fn first(i: Item) u64 {
        return i.base + i.span.offset;
    }
};

/// One EXL3 matrix on this rank: its width, rank-local shape and where its parts land.
pub const Matrix = struct {
    proj: names.Proj,
    layer: u16,
    expert: u16,
    halves: u8, // 2 * bits: the kernels' K2, from the trellis' last dim
    k: u32, // rank-local input width
    n: u32, // rank-local output width
    trellis: u32, // slots (for a routed expert: the layer's stacked slot) and this matrix' byte offsets in them
    suh: u32,
    svh: u32,
    trellis_off: u64,
    suh_off: u64,
    svh_off: u64,
    marker_shard: u16,
    marker_at: u64,
};

/// A plain tensor on this rank (fp16 index projections and fp32 router tensors as they are).
pub const PlainTensor = struct {
    role: names.Plain,
    layer: u16,
    slot: u32,
    dtype: st.DType, // as held in the slot: fp32 where the Gemma fold ran
    rows: u32,
    cols: u32, // 1 for vectors
    folded: bool,
};

/// One routed projection of a layer on this rank: the per-expert widths and pointer offsets, as the kernels take them.
pub const MoeProj = struct {
    halves: []i32, // [experts], 2 * bits of each expert's trellis
    trellis_off: []u64, // [experts], byte offset of each expert's trellis in `trellis_slot` (pointer = base + offset)
    trellis_slot: u32,
    suh_slot: u32, // fp16 [experts, k]
    svh_slot: u32, // fp16 [experts, n]
    k: u32,
    n: u32,
    histogram: [17]u32, // experts at each K2
};

pub const MoeLayer = struct { layer: u16, w1: MoeProj, w3: MoeProj, w2: MoeProj };

pub const Options = struct {
    /// Only these layers (a slice that fits one GPU); empty means all, and then no tensor may be left over.
    layers: []const u16 = &.{},
};

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    share: shard.Share,
    slots: []Slot,
    items: []Item, // in (shard, offset) order
    matrices: []Matrix,
    plain: []PlainTensor,
    moe: []MoeLayer,
    ignored: usize, // tensors of other namespaces (a vision tower) left unread

    pub fn deinit(p: *Plan) void {
        p.arena.deinit();
        p.* = undefined;
    }

    pub fn slotBytes(p: *const Plan) u64 {
        var n: u64 = 0;
        for (p.slots) |s| n += s.bytes;
        return n;
    }

    pub fn moeLayer(p: *const Plan, layer: u32) ?*const MoeLayer {
        for (p.moe) |*m| if (m.layer == layer) return m;
        return null;
    }

    pub fn matrix(p: *const Plan, proj: names.Proj, layer: u32, expert: u32) ?*const Matrix {
        for (p.matrices) |*m| if (m.proj == proj and m.layer == layer and m.expert == expert) return m;
        return null;
    }

    pub fn build(gpa: std.mem.Allocator, cat: *Catalog, c: *const Config, share: shard.Share, opts: Options) !Plan {
        var b: Builder = .{ .cat = cat, .c = c, .share = share, .a = undefined, .gemma = c.gemma_norm };
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        b.a = arena.allocator();
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
            .slots = b.slots.items,
            .items = b.items.items,
            .matrices = b.matrices.items,
            .plain = b.plain.items,
            .moe = b.moe.items,
            .ignored = try b.leftovers(opts),
        };
    }
};

const Builder = struct {
    cat: *Catalog,
    c: *const Config,
    share: shard.Share,
    a: std.mem.Allocator,
    gemma: bool,
    slots: std.ArrayList(Slot) = .empty,
    items: std.ArrayList(Item) = .empty,
    matrices: std.ArrayList(Matrix) = .empty,
    plain: std.ArrayList(PlainTensor) = .empty,
    moe: std.ArrayList(MoeLayer) = .empty,

    fn slot(b: *Builder, bytes: u64) !u32 {
        try b.slots.append(b.a, .{ .bytes = bytes });
        return @intCast(b.slots.items.len - 1);
    }

    fn find(b: *Builder, name: []const u8) !*Ref {
        const r = b.cat.get(name) orelse {
            std.log.warn("checkpoint has no tensor {s}", .{name});
            return error.MissingTensor;
        };
        r.used = true;
        return r;
    }

    fn all(b: *Builder, opts: Options) !void {
        const c = b.c;
        try b.plainTensor(.embed, 0, &.{ c.vocab, c.hidden }, &.{.bf16}, null);
        if (opts.layers.len == 0) {
            for (0..c.layers) |l| try b.layer(@intCast(l));
        } else for (opts.layers) |l| {
            if (l >= c.layers) return error.BadLayer;
            try b.layer(l);
        }
        try b.plainTensor(.final_norm, 0, &.{c.hidden}, &.{.bf16}, null);
        try b.denseGroup(.head, 0, c.hidden, c.vocab);
    }

    fn layer(b: *Builder, l: u16) !void {
        const c = b.c;
        const q = c.heads * c.head_dim;
        const kv = c.kv_heads * c.head_dim;
        const ix = c.index;
        try b.plainTensor(.input_ln, l, &.{c.hidden}, &.{.bf16}, null);
        try b.plainTensor(.post_ln, l, &.{c.hidden}, &.{.bf16}, null);
        try b.plainTensor(.q_norm, l, &.{c.head_dim}, &.{.bf16}, null);
        try b.plainTensor(.k_norm, l, &.{c.head_dim}, &.{.bf16}, null);
        try b.plainTensor(.index_q_proj, l, &.{ ix.heads * ix.dim, c.hidden }, &.{.f16}, b.share.indexRows(c));
        try b.plainTensor(.index_k_proj, l, &.{ ix.dim, c.hidden }, &.{.f16}, null);
        try b.plainTensor(.index_q_norm, l, &.{ix.dim}, &.{.bf16}, null);
        try b.plainTensor(.index_k_norm, l, &.{ix.dim}, &.{.bf16}, null);
        try b.denseGroup(.q, l, c.hidden, q);
        try b.denseGroup(.k, l, c.hidden, kv);
        try b.denseGroup(.v, l, c.hidden, kv);
        try b.denseGroup(.o, l, q, c.hidden);
        if (!c.isMoe(l)) {
            try b.denseGroup(.dense_gate, l, c.hidden, c.dense_inter);
            try b.denseGroup(.dense_up, l, c.hidden, c.dense_inter);
            return b.denseGroup(.dense_down, l, c.dense_inter, c.hidden);
        }
        try b.plainTensor(.router, l, &.{ c.experts, c.hidden }, &.{.f32}, null);
        try b.plainTensor(.router_bias, l, &.{c.experts}, &.{ .f32, .bf16 }, null);
        try b.denseGroup(.shared_gate, l, c.hidden, c.shared_inter);
        try b.denseGroup(.shared_up, l, c.hidden, c.shared_inter);
        try b.denseGroup(.shared_down, l, c.shared_inter, c.hidden);
        const w1 = try b.experts(.w1, l, c.hidden, c.moe_inter);
        const w3 = try b.experts(.w3, l, c.hidden, c.moe_inter);
        const w2 = try b.experts(.w2, l, c.moe_inter, c.hidden);
        try b.moe.append(b.a, .{ .layer = l, .w1 = w1, .w3 = w3, .w2 = w2 });
    }

    fn plainTensor(b: *Builder, role: names.Plain, l: u16, shape: []const usize, dtypes: []const st.DType, rows: ?shard.Span) !void {
        var buf: names.Buf = undefined;
        const name = names.plain(&buf, role, l);
        const r = try b.find(name);
        var ok = false;
        for (dtypes) |d| ok = ok or r.is(d, shape);
        if (!ok) {
            std.log.warn("{s}: {t} {any} is not the expected {any}", .{ name, r.dtype, r.shape[0..r.rank], shape });
            return error.MisshapedTensor;
        }
        const cols: usize = if (shape.len > 1) shape[1] else 1;
        const row_bytes = cols * r.dtype.size();
        var offset: u64 = 0;
        var len: u64 = r.len;
        var n_rows: usize = shape[0];
        if (rows) |s| {
            offset = s.lo * row_bytes;
            len = s.len() * row_bytes;
            n_rows = s.len();
        }
        const fold = b.gemma and role.isNorm();
        const held: st.DType = if (fold) .f32 else r.dtype;
        const slot_id = try b.slot(len / r.dtype.size() * held.size());
        try b.items.append(b.a, .{ .shard = r.shard, .base = r.begin, .span = .{ .offset = offset, .len = len }, .slot = slot_id, .dst = 0, .fold = fold });
        try b.plain.append(b.a, .{ .role = role, .layer = l, .slot = slot_id, .dtype = held, .rows = @intCast(n_rows), .cols = @intCast(cols), .folded = fold });
    }

    const Admitted = struct { group: layout.Group, halves: u8, trellis: *Ref, suh: *Ref, svh: *Ref, marker: *Ref };

    /// One group's four tensors: dtypes, shapes against the expected K x N, 128 alignment and an allowed width.
    fn admit(b: *Builder, prefix: []const u8, proj: names.Proj, k: usize, n: usize) !Admitted {
        var nb: [4]names.Buf = undefined;
        const trellis = try b.find(names.part(&nb[0], prefix, .trellis));
        const suh = try b.find(names.part(&nb[1], prefix, .suh));
        const svh = try b.find(names.part(&nb[2], prefix, .svh));
        const marker = try b.find(names.part(&nb[3], prefix, .mul1));
        const bad = struct {
            fn at(why: anyerror, p: []const u8, comptime fmt: []const u8, args: anytype) anyerror {
                std.log.warn("{s}: " ++ fmt, .{p} ++ args);
                return why;
            }
        }.at;
        if (trellis.dtype != .i16 or trellis.rank != 3) return bad(error.BadTrellis, prefix, "trellis is {t} rank {d}, not int16 rank 3", .{ trellis.dtype, trellis.rank });
        const words = trellis.shape[2];
        if (words % 8 != 0 or words < 16 or words > 128) return bad(error.BadTrellis, prefix, "trellis last dim {d} is not 16 * bits", .{words});
        const halves: u32 = @intCast(words / 8);
        if (std.mem.indexOfScalar(u32, if (proj.isExpert()) &expert_halves else &dense_halves, halves) == null) return bad(error.UnsupportedWidth, prefix, "{d} half-bits are not a width the kernels run", .{halves});
        if (trellis.shape[0] * 16 % 128 != 0 or trellis.shape[1] * 16 % 128 != 0) return bad(error.NotAligned, prefix, "K {d} x N {d} is not 128-aligned", .{ trellis.shape[0] * 16, trellis.shape[1] * 16 });
        if (trellis.shape[0] * 16 != k or trellis.shape[1] * 16 != n) return bad(error.MisshapedTensor, prefix, "trellis {any} is not K {d} x N {d}", .{ trellis.shape[0..3].*, k, n });
        if (!suh.is(.f16, &.{k}) or !svh.is(.f16, &.{n})) return bad(error.MisshapedTensor, prefix, "scales are not fp16 [{d}] and [{d}]", .{ k, n });
        if (marker.dtype != .i32 or marker.rank > 1 or marker.numel() != 1) return bad(error.MisshapedTensor, prefix, "mul1 marker is not one int32", .{});
        if (proj == .head and halves != 2 * b.c.quant.head_bits) return bad(error.UnsupportedWidth, prefix, "head is {d} half-bits, head_bits says {d}", .{ halves, b.c.quant.head_bits });
        return .{ .group = try layout.Group.fromShape(trellis.shape[0..3]), .halves = @intCast(halves), .trellis = trellis, .suh = suh, .svh = svh, .marker = marker };
    }

    fn denseGroup(b: *Builder, proj: names.Proj, l: u16, k: usize, n: usize) !void {
        var buf: names.Buf = undefined;
        const m = try b.admit(names.group(&buf, proj, l, 0), proj, k, n);
        _ = try b.dense(m, proj, l, try b.share.part(b.c, proj, m.group));
    }

    /// A matrix with slots of its own for the rank's part of the trellis and scales.
    fn dense(b: *Builder, m: Admitted, proj: names.Proj, l: u16, part: layout.Part) !void {
        const t = try b.slot(part.trellis.bytes());
        const su = try b.slot(part.suh.len);
        const sv = try b.slot(part.svh.len);
        try b.parts(m, part, .{ t, su, sv }, .{ 0, 0, 0 });
        try b.matrices.append(b.a, Builder.describe(m, proj, l, part, .{ t, su, sv }, .{ 0, 0, 0 }));
    }

    fn parts(b: *Builder, m: Admitted, part: layout.Part, slot_ids: [3]u32, dst: [3]u64) !void {
        const refs = [3]*Ref{ m.trellis, m.suh, m.svh };
        const spans = [3]layout.Span{ part.trellis, part.suh, part.svh };
        for (refs, spans, slot_ids, dst) |r, sp, s, d| try b.items.append(b.a, .{ .shard = r.shard, .base = r.begin, .span = sp, .slot = s, .dst = d });
    }

    fn describe(m: Admitted, proj: names.Proj, l: u32, part: layout.Part, s: [3]u32, off: [3]u64) Matrix {
        return .{
            .proj = proj,
            .layer = @intCast(l),
            .expert = 0,
            .halves = m.halves,
            .k = part.group.k(),
            .n = part.group.n(),
            .trellis = s[0],
            .suh = s[1],
            .svh = s[2],
            .trellis_off = off[0],
            .suh_off = off[1],
            .svh_off = off[2],
            .marker_shard = m.marker.shard,
            .marker_at = m.marker.begin,
        };
    }

    /// One routed projection of a layer: every expert's part stacked in three slots, widths and offsets per expert.
    fn experts(b: *Builder, proj: names.Proj, l: u16, k: usize, n: usize) !MoeProj {
        const e_n = b.c.experts;
        const found = try b.a.alloc(Admitted, e_n);
        const cuts = try b.a.alloc(layout.Part, e_n);
        const halves = try b.a.alloc(i32, e_n);
        var buf: names.Buf = undefined;
        for (found, cuts, halves, 0..) |*f, *cut, *h, e| {
            f.* = try b.admit(names.group(&buf, proj, l, @intCast(e)), proj, k, n);
            cut.* = try b.share.part(b.c, proj, f.group);
            h.* = f.halves;
        }
        var lay = try layout.Layout.of(b.a, cuts);
        defer lay.deinit(b.a);
        const suh_len = cuts[0].suh.len;
        const svh_len = cuts[0].svh.len;
        const t = try b.slot(lay.trellisBytes());
        const su = try b.slot(suh_len * e_n);
        const sv = try b.slot(svh_len * e_n);
        const offs = try b.a.alloc(u64, e_n);
        for (found, cuts, offs, 0..) |f, cut, *o, e| {
            o.* = lay.byteOffset(e);
            const dst = [3]u64{ o.*, e * suh_len, e * svh_len };
            try b.parts(f, cut, .{ t, su, sv }, dst);
            var mx = Builder.describe(f, proj, l, cut, .{ t, su, sv }, dst);
            mx.expert = @intCast(e);
            try b.matrices.append(b.a, mx);
        }
        return .{ .halves = halves, .trellis_off = offs, .trellis_slot = t, .suh_slot = su, .svh_slot = sv, .k = cuts[0].group.k(), .n = cuts[0].group.n(), .histogram = lay.histogram() };
    }

    /// Tensors no planned name took: a text-model name left over is a pack this plan does not understand.
    fn leftovers(b: *Builder, opts: Options) !usize {
        var ignored: usize = 0;
        var it = b.cat.map.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.used) continue;
            if (opts.layers.len == 0 and std.mem.startsWith(u8, kv.key_ptr.*, names.text_namespace)) {
                std.log.warn("unexpected tensor {s}", .{kv.key_ptr.*});
                return error.UnexpectedTensor;
            }
            ignored += 1;
        }
        return ignored;
    }
};

/// Where the streamed bytes go: a device allocation per slot, written at byte offsets.
pub const Sink = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, slot: u32, offset: u64, bytes: []const u8) anyerror!void,
};

pub const ShardEnd = @import("stream.zig").ShardEnd;
pub const Streamer = @import("stream.zig").Streamer;
pub const StreamOptions = @import("stream.zig").Options;
pub const StreamStats = @import("stream.zig").Stats;

/// bf16 norm weights as fp32 `1 + w` (Gemma RMSNorm), summed once in fp32; the caller's `out` holds the result.
pub fn foldGemma(out: []f32, bf16: []const u8) []f32 {
    const n = bf16.len / 2;
    for (out[0..n], 0..) |*o, i| {
        const w: f32 = @bitCast(@as(u32, std.mem.readInt(u16, bf16[2 * i ..][0..2], .little)) << 16);
        o.* = 1.0 + w;
    }
    return out[0..n];
}

test {
    _ = @import("stream.zig");
    _ = @import("stream_test.zig");
    _ = @import("config.zig");
    _ = names;
    _ = shard;
    _ = @import("load_test.zig");
    _ = @import("load_synth_test.zig");
    _ = @import("load_widths_test.zig");
}
