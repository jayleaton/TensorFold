//! The PLE layer and its n-gram table: shards opened and checked, a rank's heads' rows put on the GPU.
const std = @import("std");
const core = @import("core");
const lay = @import("cuda_layouts.zig");
const ngram = @import("cuda_ngram.zig");
const wt = @import("cuda_weights.zig");
const wl = @import("cuda_weights_loader.zig");

const st = core.safetensors;
const Tensor = st.Tensor;
const Loader = wl.Loader;
const Ple = wt.Ple;
const Weights = wt.Weights;
const NgramTable = wt.NgramTable;

/// The PLE layer: n-gram constants checked, table opened (host or GPU), k/v projections, fp32 norms, conv taps.
pub fn ple(L: *Loader, path: []const u8, name: []const u8, ple_index: u32, w: *Weights) !Ple {
    const c = L.c;
    const g = try ngram.NGram.init(c.ngramOptions(ple_index));
    var p: Ple = .{ .ngram = g };
    const base = try std.fmt.allocPrint(L.gpa, "{s}.ple_embedding.", .{name});
    defer L.gpa.free(base);
    var buf: [3][256]u8 = undefined;
    const mult = try L.expect(try std.fmt.bufPrint(&buf[0], "{s}layer_multipliers", .{base}), .i64, &.{g.n});
    const offs = try L.expect(try std.fmt.bufPrint(&buf[1], "{s}ngram_heads_offsets", .{base}), .i64, &.{g.heads});
    const sizes = try L.expect(try std.fmt.bufPrint(&buf[2], "{s}ngram_heads_vocab_sizes", .{base}), .i64, &.{g.heads});
    var consts: [3][ngram.max_heads]i64 = undefined;
    for ([_]Tensor{ mult, offs, sizes }, &consts) |t, *dst| for (0..t.numel()) |i| {
        dst[i] = std.mem.readInt(i64, t.bytes[8 * i ..][0..8], .little);
    };
    try g.check(consts[0][0..g.n], consts[1][0..g.heads], consts[2][0..g.heads]);
    if (w.table != null) return error.UnsupportedModel; // one PLE layer's table (this checkpoint has one)
    w.table = try L.openTable(base, g);
    const gpu = L.o.ngram_on_gpu orelse (L.o.world > 1);
    if (gpu) try L.tableToGpu(path, &w.table.?, g);
    var pb: [192]u8 = undefined;
    const sd = c.streams * c.hidden;
    p.key = try L.b16(try std.fmt.bufPrint(&pb, "{s}.key.b.weight", .{path}), try L.nameOf("{s}.key_proj", .{name}), sd, c.ple_dim);
    p.value = try L.b16(try std.fmt.bufPrint(&pb, "{s}.value.b.weight", .{path}), try L.nameOf("{s}.value_proj", .{name}), c.hidden, c.ple_dim);
    p.norm_key = try L.cscale(try std.fmt.bufPrint(&pb, "{s}.norm_key", .{path}), try L.nameOf("{s}.norm_key.weight", .{name}), sd);
    p.norm_query = try L.cscale(try std.fmt.bufPrint(&pb, "{s}.norm_query", .{path}), try L.nameOf("{s}.norm_query.weight", .{name}), sd);
    p.norm_conv = try L.cscale(try std.fmt.bufPrint(&pb, "{s}.norm_conv", .{path}), try L.nameOf("{s}.norm_conv.weight", .{name}), sd);
    const conv = try L.expect(try L.nameOf("{s}.conv1d.weight", .{name}), .bf16, &.{ sd, 1, c.ple_kernel });
    p.conv = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.conv", .{path}), "bfloat16", &.{ sd, c.ple_kernel }, conv.bytes);
    return p;
}

/// open_table: shards mapped again, FP8 with its table scale through a 256-entry bf16 LUT, or bf16 as stored.
pub fn openTable(L: *Loader, base: []const u8, g: ngram.NGram) !NgramTable {
    var t: NgramTable = .{ .gpa = L.gpa, .io = L.io };
    errdefer t.deinit();
    var opened: std.StringHashMapUnmanaged(usize) = .empty;
    defer opened.deinit(L.gpa);
    try t.starts.append(L.gpa, 0);
    var kind: ?st.DType = null;
    for (0..L.c.ngram_shards) |i| {
        var b: [256]u8 = undefined;
        var key = try std.fmt.bufPrint(&b, "{s}{s}ngram_embedding.shard_{d}.weight", .{ L.prefix, base, i });
        if (!L.has(key) and !L.hasExtra(key)) key = try std.fmt.bufPrint(&b, "{s}{s}ngram_embedding.shards.{d}.weight", .{ L.prefix, base, i });
        // the shard's file, mapped again for the table's lifetime: a checkpoint file or a ple-table file
        var file_i: ?usize = null;
        for (L.ck.files.items, 0..) |*f, fi| if (f.names.contains(key)) {
            file_i = fi;
        };
        var path: [:0]const u8 = undefined;
        var name_key: []const u8 = undefined;
        if (file_i) |fi| {
            const ck_file = &L.ck.files.items[fi];
            try L.ck.used.put(L.gpa, ck_file.names.getKey(key).?, {});
            path = L.paths[fi];
            name_key = ck_file.names.getKey(key).?;
        } else {
            const xi = L.extraIndex(key) orelse {
                std.log.err("checkpoint has no n-gram shard {s}", .{key});
                return error.MissingTensor;
            };
            path = L.extra_paths.items[xi];
            name_key = L.extra.items[xi].names.getKey(key).?;
        }
        const own = try opened.getOrPut(L.gpa, name_key);
        if (!own.found_existing) {
            own.value_ptr.* = t.files.items.len;
            try t.files.append(L.gpa, try st.File.open(L.gpa, L.io, path));
        }
        const tensor = t.files.items[own.value_ptr.*].get(key).?;
        if (kind) |k| if (k != tensor.dtype) return error.NgramShardsMixLayouts;
        kind = tensor.dtype;
        if (tensor.dtype != .f8_e4m3 and tensor.dtype != .bf16) {
            std.log.err("{s}: {t} n-gram rows; this engine reads FP8 (e4m3) or bf16 tables", .{ key, tensor.dtype });
            return error.UnsupportedQuantization;
        }
        if (tensor.rank != 2) return error.UnexpectedTensor;
        if (t.width == 0) t.width = @intCast(tensor.dim(1));
        if (tensor.dim(1) != t.width) return error.NgramShardsDiffer;
        try t.shards.append(L.gpa, tensor.bytes);
        try t.starts.append(L.gpa, t.starts.items[t.starts.items.len - 1] + tensor.dim(0));
    }
    t.rows = t.starts.items[t.starts.items.len - 1];
    t.fp8 = kind.? == .f8_e4m3;
    var b: [256]u8 = undefined;
    const sname = try std.fmt.bufPrint(&b, "{s}ngram_embedding.weight_scale", .{base});
    const sfull = try L.nameOf("{s}{s}", .{ L.prefix, sname });
    const in_ck = L.has(sfull);
    if (in_ck or L.extraIndex(sfull) != null) {
        const s = if (in_ck) try L.get(sname) else L.extra.items[L.extraIndex(sfull).?].get(sfull).?;
        if (s.numel() != 1) return error.UnexpectedTensor;
        t.scale = switch (s.dtype) {
            .bf16 => lay.bf16ToF32(std.mem.readInt(u16, s.bytes[0..2], .little)),
            .f32 => @bitCast(std.mem.readInt(u32, s.bytes[0..4], .little)),
            else => return error.UnexpectedTensor,
        };
    }
    if (t.fp8) t.lut = lay.ngramLut(t.scale) else if (t.scale != 1) return error.UnsupportedQuantization;
    if (t.width != g.dims) {
        std.log.err("the n-gram rows hold {d} values, expected {d}", .{ t.width, g.dims });
        return error.NgramWidth;
    }
    if (t.rows != g.rows) {
        std.log.err("n-gram tables hold {d} rows, expected {d}", .{ t.rows, g.rows });
        return error.NgramRows;
    }
    return t;
}

/// Rank r's heads [r * H / world, (r + 1) * H / world) of the table: their rows as stored and the LUT, on the GPU.
pub fn tableToGpu(L: *Loader, path: []const u8, t: *NgramTable, g: ngram.NGram) !void {
    const per = g.heads / L.o.world;
    const h0 = L.o.rank * per;
    const base: u64 = @intCast(g.offsets[h0]);
    const end: u64 = if (h0 + per == g.heads) @as(u64, @intCast(g.offsets[g.heads - 1] + g.sizes[g.heads - 1])) else @intCast(g.offsets[h0 + per]);
    const bytes: usize = if (t.fp8) t.width else 2 * t.width;
    var pb: [192]u8 = undefined;
    const rows = try L.out.begin(try std.fmt.bufPrint(&pb, "{s}.table.rows", .{path}), if (t.fp8) "uint8" else "bfloat16", &.{ end - base, t.width });
    L.out.ours();
    var id = base;
    while (id < end) {
        var s: usize = 0;
        while (t.starts.items[s + 1] <= id) s += 1;
        const stop = @min(end, t.starts.items[s + 1]);
        const from: usize = @intCast((id - t.starts.items[s]) * bytes);
        try L.out.put(t.shards.items[s][from..][0..@intCast((stop - id) * bytes)]);
        id = stop;
    }
    try L.out.end();
    const lut = try L.out.whole(try std.fmt.bufPrint(&pb, "{s}.table.lut", .{path}), "uint16", &.{256}, std.mem.sliceAsBytes(&t.lut));
    L.out.ours();
    t.gpu = .{ .rows = rows, .lut = lut, .base = base, .count = end - base, .head0 = h0, .heads = per };
}
