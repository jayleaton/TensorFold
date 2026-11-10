//! Triton kernels from captured cubins: one fn per qwen4_exp CUDA wrapper, same args and constexprs so find picks it.

const std = @import("std");
const launch = @import("cuda_triton_launch.zig");
pub const pow2 = launch.pow2;
pub const Recorder = launch.Recorder;
pub const Branch = launch.Branch;
pub const AttnGeometry = launch.AttnGeometry;
pub const AttnScratch = launch.AttnScratch;
pub const Cache = launch.Cache;
pub const Fp4 = launch.Fp4;
pub const b16SplitK = launch.b16SplitK;
pub const b16PartBytes = launch.b16PartBytes;
pub const fp4SplitK = launch.fp4SplitK;
pub const fp4PartBytes = launch.fp4PartBytes;
pub const Memo = launch.Memo;
pub const Tri = launch.Tri;

/// The checkpoint's dimensions a rank computes with (config.json's text_config; `rank(2)` halves what TP=2 splits).
pub const Dims = struct {
    hidden: usize = 2560,
    streams: usize = 4,
    low: usize = 320,
    heads: usize = 24,
    kv_heads: usize = 2,
    head_dim: usize = 256,
    index_heads: usize = 4,
    index_dim: usize = 128,
    rotary_half: usize = 32,
    experts: usize = 512,
    top_k: usize = 10,
    moe_width: usize = 640,
    vocab: usize = 248320,
    key_heads: usize = 16,
    value_heads: usize = 48,
    gdn_dim: usize = 128,
    ngram_heads: usize = 16,
    ngram_dim: usize = 160,
    ple_taps: usize = 4,
    ngram_size: usize = 3,
    eps: f32 = 1e-6,

    /// A rank's share at `world` ranks: heads, kv heads, DeltaNet heads, expert widths and the vocabulary split.
    pub fn rank(world: usize) Dims {
        const d: Dims = .{};
        return .{ .heads = d.heads / world, .kv_heads = d.kv_heads / world, .moe_width = d.moe_width / world, .vocab = d.vocab / world, .key_heads = d.key_heads / world, .value_heads = d.value_heads / world };
    }

    pub fn wide(d: Dims) usize {
        return d.streams * d.hidden;
    }
    /// Buffers.pa: [q | gate] per head, k, v, then the indexer's q heads and its k.
    pub fn attnWidth(d: Dims) usize {
        return d.heads * 2 * d.head_dim + 2 * d.kv_heads * d.head_dim + (d.index_heads + 1) * d.index_dim;
    }
    /// gdn.widths: the conv channels (q, k, v) and the projection row (+ z, b, a).
    pub fn convDim(d: Dims) usize {
        return 2 * d.key_heads * d.gdn_dim + d.value_heads * d.gdn_dim;
    }
    pub fn projWidth(d: Dims) usize {
        return d.convDim() + d.value_heads * d.gdn_dim + 2 * d.value_heads;
    }
    /// Rows of the n-gram conv tail: (ple_kernel - 1) * ngram_size.
    pub fn tailRows(d: Dims) usize {
        return (d.ple_taps - 1) * d.ngram_size;
    }
};

// -- tests: the Python wrappers' own launches (fixtures_cuda_triton.json) -------------------------------------

const testing = std.testing;
const fixture_text = @embedFile("fixtures_cuda_triton.json");

const Fixture = struct {
    parsed: std.json.Parsed(std.json.Value),

    fn load() !Fixture {
        return .{ .parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, fixture_text, .{}) };
    }
    fn deinit(f: *Fixture) void {
        f.parsed.deinit();
    }
    fn case(f: *const Fixture, id: []const u8) ![]const std.json.Value {
        const cases = f.parsed.value.object.get("cases").?.object;
        const c = cases.get(id) orelse {
            std.debug.print("no fixture case {s}\n", .{id});
            return error.MissingCase;
        };
        return c.array.items;
    }
};

fn jsonInt(v: std.json.Value) i64 {
    return v.integer;
}

/// The recorded launches equal the fixture's: kernel, grid, runtime args by name and every non-None constexpr.
fn expectLaunches(f: *const Fixture, id: []const u8, rec: *const Recorder) !void {
    const want = try f.case(id);
    errdefer std.debug.print("case {s}\n", .{id});
    try testing.expectEqual(want.len, rec.n);
    for (want, rec.launches[0..rec.n]) |w, got| {
        const o = w.object;
        try testing.expectEqualStrings(o.get("fn").?.string, got.name);
        const grid = o.get("grid").?.array.items;
        for (grid, got.grid) |g, x| try testing.expectEqual(jsonInt(g), @as(i64, x));
        const args = o.get("args").?.array.items;
        try testing.expectEqual(args.len, got.nargs);
        for (args) |arg| {
            const name = arg.object.get("name").?.string;
            const ty = arg.object.get("type").?.string;
            const mine = for (got.args[0..got.nargs]) |x| {
                if (std.mem.eql(u8, x.name, name)) break x;
            } else {
                std.debug.print("{s}: no argument {s}\n", .{ got.name, name });
                return error.MissingArgument;
            };
            errdefer std.debug.print("{s}: argument {s}\n", .{ got.name, name });
            switch (mine.value) {
                .ptr => |x| try testing.expectEqualStrings(ty, x.ty),
                .i32 => |x| {
                    try testing.expectEqualStrings(ty, "i32");
                    try testing.expectEqual(jsonInt(arg.object.get("int").?), @as(i64, x));
                },
                .f32 => |x| {
                    try testing.expectEqualStrings(ty, "fp32");
                    try testing.expectEqual(jsonInt(arg.object.get("f32").?), @as(i64, @as(u32, @bitCast(x))));
                },
                .u64 => return error.UnexpectedWord,
            }
        }
        const consts = o.get("consts").?.object;
        var live: usize = 0;
        var it = consts.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.* == .null) continue;
            live += 1;
            const name = kv.key_ptr.*;
            const mine = for (got.consts[0..got.nconsts]) |c| {
                if (std.mem.eql(u8, c.name, name)) break c;
            } else {
                std.debug.print("{s}: no constexpr {s}\n", .{ got.name, name });
                return error.MissingConst;
            };
            errdefer std.debug.print("{s}: constexpr {s}\n", .{ got.name, name });
            const v = kv.value_ptr.object;
            if (v.get("int")) |x| try testing.expectEqual(jsonInt(x), mine.int.?);
            if (v.get("f32")) |x| try testing.expectEqual(jsonInt(x), @as(i64, @as(u32, @bitCast(mine.f32.?))));
        }
        try testing.expectEqual(live, got.nconsts);
    }
}

/// Distinct 16-aligned stand-in addresses (types, not values, select a variant).
fn addr(i: u64) u64 {
    return 0x7f0000000000 + i * 0x100000;
}

fn check(f: *const Fixture, comptime fmt: []const u8, args: anytype, run: anytype) !void {
    var buf: [128]u8 = undefined;
    const id = try std.fmt.bufPrint(&buf, fmt, args);
    var rec: Recorder = .{};
    const t: Tri = .{ .set = null, .rec = &rec };
    try run.go(t);
    try expectLaunches(f, id, &rec);
}

/// flashnext_triton_fixtures.py ROWS: decode windows, the capture's prompt pieces and full chunks
const rows_cases = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 16, 18, 19, 29, 30, 116, 973, 974, 2048 };
const prompts = [_][2]usize{ .{ 0, 2048 }, .{ 4096, 19 }, .{ 139264, 2048 }, .{ 2048, 974 }, .{ 2048, 973 }, .{ 0, 30 }, .{ 0, 29 }, .{ 0, 18 }, .{ 2048, 206 }, .{ 2048, 205 } };

test "embedding, hyper-connection, norm and stream launches equal the Python wrappers'" {
    var f = try Fixture.load();
    defer f.deinit();
    const d: Dims = .{};
    const D = d.hidden;
    const S = d.streams;
    const W = d.wide();
    for (rows_cases) |r| {
        const R = struct {
            r: usize,
            d: Dims,
            copies: usize = 4,
            mode: usize = 0,
            y_f32: bool = false,
            ndn: usize = 0,
            group: usize = 0,
            fn go(c: @This(), t: Tri) !void {
                const dd = c.d;
                switch (c.mode) {
                    0 => try t.hcWriteback(addr(0), addr(0), addr(1), null, .none, c.r, dd.hidden, dd.streams),
                    1 => try t.hcWriteback(addr(0), addr(0), addr(1), addr(2), .{ .bf16 = addr(3) }, c.r, dd.hidden, dd.streams),
                    2 => try t.hcWriteback(addr(0), addr(0), addr(1), addr(2), .{ .moe = .{ .y = addr(3), .y_f32 = c.y_f32, .wts = addr(4), .slots = dd.top_k + 1 } }, c.r, dd.hidden, dd.streams),
                    3 => try t.hcWriteback(addr(0), addr(0), addr(1), addr(2), .{ .ranks = .{ .part = addr(3), .world = 2 } }, c.r, dd.hidden, dd.streams),
                    else => unreachable,
                }
            }
        };
        try check(&f, "hc_writeback/r{d}/m0", .{r}, R{ .r = r, .d = d });
        try check(&f, "hc_writeback/r{d}/m1", .{r}, R{ .r = r, .d = d, .mode = 1 });
        try check(&f, "hc_writeback/r{d}/m2/bf16", .{r}, R{ .r = r, .d = d, .mode = 2 });
        try check(&f, "hc_writeback/r{d}/m2/f32", .{r}, R{ .r = r, .d = d, .mode = 2, .y_f32 = true });
        try check(&f, "hc_writeback/r{d}/m3", .{r}, R{ .r = r, .d = d, .mode = 3 });
        inline for (.{ 4, 1 }) |copies| {
            try check(&f, "embed/r{d}/s{d}", .{ r, copies }, struct {
                r: usize,
                fn go(c: @This(), t: Tri) !void {
                    try t.embed(addr(0), addr(1), addr(2), c.r, 2560, copies);
                }
            }{ .r = r });
        }
        const Plain = struct {
            r: usize,
            d: Dims,
            which: enum { normed, act_inj, act, mix, norm_e, norm_h, add, ple_embed, ple_gate, ple_conv },
            fn go(c: @This(), t: Tri) !void {
                const dd = c.d;
                const wide = dd.wide();
                switch (c.which) {
                    .normed => try t.hcNormed(addr(0), addr(1), addr(2), addr(3), addr(4), c.r, dd.hidden, dd.streams, dd.eps),
                    .act_inj => try t.hcAct(addr(0), addr(1), addr(2), addr(3), c.r, dd.low + dd.streams, dd.streams, dd.low),
                    .act => try t.hcAct(addr(0), addr(1), addr(2), null, c.r, dd.low, dd.streams, dd.low),
                    .mix => try t.hcMix(addr(0), addr(1), addr(2), addr(3), c.r, dd.hidden, dd.streams),
                    .norm_e => try t.rmsnorm(addr(0), dd.hidden, addr(1), addr(2), addr(3), c.r, dd.hidden, null, dd.eps),
                    .norm_h => try t.rmsnorm(addr(0), wide, addr(1), addr(2), addr(3), c.r, wide, null, dd.eps),
                    .add => try t.addStreams(addr(0), addr(1), addr(2), c.r, dd.hidden, dd.streams),
                    .ple_embed => try t.pleEmbedBf16(addr(0), addr(1), addr(2), c.r, dd.ngram_heads, dd.ngram_dim, 1.0),
                    .ple_gate => try t.pleGate(addr(0), addr(1), addr(2), addr(3), addr(4), addr(5), addr(6), c.r, dd.hidden, dd.streams, dd.eps),
                    .ple_conv => try t.pleConv(addr(0), addr(1), addr(2), addr(3), addr(4), addr(5), addr(5), addr(6), c.r, dd.hidden, dd.streams, dd.ple_taps, dd.ngram_size, dd.eps),
                }
            }
        };
        try check(&f, "hc_normed/r{d}", .{r}, Plain{ .r = r, .d = d, .which = .normed });
        try check(&f, "hc_act/r{d}/n{d}", .{ r, d.low + S }, Plain{ .r = r, .d = d, .which = .act_inj });
        try check(&f, "hc_act/r{d}/n{d}", .{ r, d.low }, Plain{ .r = r, .d = d, .which = .act });
        try check(&f, "hc_mix/r{d}", .{r}, Plain{ .r = r, .d = d, .which = .mix });
        try check(&f, "rmsnorm/r{d}/d{d}", .{ r, D }, Plain{ .r = r, .d = d, .which = .norm_e });
        try check(&f, "rmsnorm/r{d}/d{d}", .{ r, W }, Plain{ .r = r, .d = d, .which = .norm_h });
        try check(&f, "add_streams/r{d}", .{r}, Plain{ .r = r, .d = d, .which = .add });
        try check(&f, "ple_embed_bf16/r{d}", .{r}, Plain{ .r = r, .d = d, .which = .ple_embed });
        try check(&f, "ple_embed_bf16/r{d}/tp2", .{r}, struct {
            r: usize,
            fn go(c: @This(), t: Tri) !void {
                try t.pleEmbedBf16(addr(0), addr(1), addr(2), c.r, 8, 160, 1.0);
            }
        }{ .r = r });
        try check(&f, "ple_gate/r{d}", .{r}, Plain{ .r = r, .d = d, .which = .ple_gate });
        try check(&f, "ple_conv/r{d}", .{r}, Plain{ .r = r, .d = d, .which = .ple_conv });
    }
}

test "MoE launches equal the Python wrappers': router, top-k, the shared expert's FP4 matmuls, TP=2 partials" {
    var f = try Fixture.load();
    defer f.deinit();
    for (rows_cases) |r| {
        try check(&f, "moe/r{d}", .{r}, struct {
            r: usize,
            fn go(c: @This(), t: Tri) !void {
                const d: Dims = .{};
                try t.router(addr(0), d.hidden, addr(1), addr(2), c.r, d.hidden, d.experts + 1);
                try t.topkRows(addr(2), addr(3), addr(4), c.r, d.experts, d.top_k);
            }
        }{ .r = r });
        inline for (.{ false, true }) |fp32| {
            try check(&f, "moe_partial/r{d}/{s}", .{ r, if (fp32) "f32" else "bf16" }, struct {
                r: usize,
                fn go(c: @This(), t: Tri) !void {
                    try t.moePartial(addr(0), fp32, addr(1), addr(2), c.r, 2560, 11);
                }
            }{ .r = r });
        }
        for ([_]usize{ 1, 2 }) |world| {
            const d = Dims.rank(world);
            const Gu = struct {
                r: usize,
                d: Dims,
                fn go(c: @This(), t: Tri) !void {
                    const n = 2 * c.d.moe_width;
                    try t.fp4mm(addr(0), c.d.hidden, .{ .weight = addr(1), .scale = addr(2), .scale2 = addr(3) }, addr(4), false, addr(5), c.r, n, c.d.hidden);
                }
            };
            try check(&f, "fp4/gu/r{d}/tp{d}", .{ r, world }, Gu{ .r = r, .d = d });
            inline for (.{ false, true }) |fp32| {
                try check(&f, "fp4/down/r{d}/tp{d}/{s}", .{ r, world, if (fp32) "f32" else "bf16" }, struct {
                    r: usize,
                    d: Dims,
                    fn go(c: @This(), t: Tri) !void {
                        const slot_stride = (c.d.top_k + 1) * c.d.moe_width;
                        try t.fp4mm(addr(0), slot_stride, .{ .weight = addr(1), .scale = addr(2), .scale2 = addr(3) }, addr(4), fp32, addr(5), c.r, c.d.hidden, c.d.moe_width);
                    }
                }{ .r = r, .d = d });
            }
        }
    }
}

test "bf16 matmul launches equal bf16.matmul's for every linear of the checkpoint, TP=1 and a TP=2 rank" {
    var f = try Fixture.load();
    defer f.deinit();
    const Mat = struct { name: []const u8, n: usize, k: usize, fp32: bool, replicated: bool };
    for ([_]usize{ 1, 2 }) |world| {
        const d = Dims.rank(world);
        const mats = [_]Mat{
            .{ .name = "hc_down", .n = d.low + d.streams, .k = d.wide(), .fp32 = true, .replicated = true },
            .{ .name = "mix_down", .n = d.low, .k = d.wide(), .fp32 = true, .replicated = true },
            .{ .name = "hc_up", .n = d.wide(), .k = d.low, .fp32 = false, .replicated = true },
            .{ .name = "gdn_proj", .n = d.projWidth(), .k = d.hidden, .fp32 = false, .replicated = false },
            .{ .name = "gdn_out", .n = d.hidden, .k = d.value_heads * d.gdn_dim, .fp32 = world > 1, .replicated = false },
            .{ .name = "attn_proj", .n = d.attnWidth(), .k = d.hidden, .fp32 = false, .replicated = false },
            .{ .name = "o_proj", .n = d.hidden, .k = d.heads * d.head_dim, .fp32 = world > 1, .replicated = false },
            .{ .name = "ple_key", .n = d.wide(), .k = d.hidden, .fp32 = false, .replicated = true },
            .{ .name = "ple_value", .n = d.hidden, .k = d.hidden, .fp32 = false, .replicated = true },
            .{ .name = "fc", .n = d.hidden, .k = d.hidden, .fp32 = false, .replicated = true },
            .{ .name = "head", .n = d.vocab, .k = d.hidden, .fp32 = false, .replicated = false },
        };
        for (rows_cases) |r| for (mats) |m| {
            if (world > 1 and m.replicated) continue;
            if (std.mem.eql(u8, m.name, "head") and r > 16) continue;
            try check(&f, "b16/{s}/r{d}/tp{d}", .{ m.name, r, world }, struct {
                r: usize,
                m: Mat,
                fn go(c: @This(), t: Tri) !void {
                    try t.b16mm(addr(0), c.m.k, addr(1), addr(2), c.m.fp32, addr(3), c.r, c.m.n, c.m.k);
                }
            }{ .r = r, .m = m });
        };
    }
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 18, 973, 2048 }) |n| {
        try check(&f, "b16/fc_h/r{d}", .{n * 4}, struct {
            r: usize,
            fn go(c: @This(), t: Tri) !void {
                try t.b16mm(addr(0), 2560, addr(1), addr(2), false, addr(3), c.r, 2560, 2560);
            }
        }{ .r = n * 4 });
    }
}

test "attention launches equal the Python wrappers': prep, gate, decode windows and prompt blocks" {
    var f = try Fixture.load();
    defer f.deinit();
    const sc: AttnScratch = .{ .po = addr(20), .pm = addr(21), .pl = addr(22), .ids = addr(23), .nk = addr(24), .sparse = addr(25), .scores = addr(26) };
    const cache: Cache = .{ .k = addr(10), .v = addr(11), .ks = addr(12), .vs = addr(12) };
    for ([_]usize{ 1, 2 }) |world| {
        const d = Dims.rank(world);
        for (rows_cases) |r| {
            try check(&f, "attn_prep/r{d}/tp{d}", .{ r, world }, struct {
                r: usize,
                d: Dims,
                fn go(c: @This(), t: Tri) !void {
                    try t.attnPrep(.{ .p = addr(0), .pos0 = addr(1), .qw = addr(2), .kw = addr(3), .iw = addr(4), .inv = addr(5), .q = addr(6), .cache = .{ .k = addr(10), .v = addr(11), .ks = addr(12), .vs = addr(13) }, .iq = addr(7), .ikc = addr(8) }, c.r, c.d);
                }
            }{ .r = r, .d = d });
            try check(&f, "attn_gate/r{d}/tp{d}", .{ r, world }, struct {
                r: usize,
                d: Dims,
                fn go(c: @This(), t: Tri) !void {
                    try t.attnGate(addr(0), addr(1), addr(2), addr(3), c.r, c.d);
                }
            }{ .r = r, .d = d });
        }
        const Decode = struct {
            rows: usize,
            pos: usize,
            ctx: ?usize,
            g: AttnGeometry,
            d: Dims,
            sc: AttnScratch,
            cache: Cache,
            fn go(c: @This(), t: Tri) !void {
                const keys = c.ctx orelse c.pos + c.rows;
                if (c.g.qsa) try t.qsaSelect(addr(30), addr(31), addr(32), addr(1), addr(33), addr(34), c.sc, c.rows, c.g, keys, c.d);
                try t.attention(addr(0), c.cache, addr(1), c.sc, addr(2), c.rows, c.g, keys, c.d);
            }
        };
        const Prefill = struct {
            start: usize,
            n: usize,
            g: AttnGeometry,
            d: Dims,
            sc: AttnScratch,
            cache: Cache,
            fn go(c: @This(), t: Tri) !void {
                if (c.g.qsa) try t.qsaPool(addr(31), addr(32), addr(1), addr(33), addr(34), c.n, c.g, c.d);
                var r0: usize = 0;
                while (r0 < c.n) : (r0 += 256) {
                    const m = @min(256, c.n - r0);
                    const ends = c.start + r0 + m;
                    if (c.g.qsa) try t.qsaRows(addr(30), addr(32), addr(1), c.sc, m, c.g, ends, c.d);
                    try t.attention(addr(0), c.cache, addr(1), c.sc, addr(2), m, c.g, ends, c.d);
                }
            }
        };
        for ([_]usize{ 1024, 262144, 262151, 1048576 }) |capacity| {
            const g = AttnGeometry.init(capacity, 2048, 4);
            for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 16 }) |rows| for ([_]usize{ 0, 100, 3000, 140000 }) |pos| for ([_]usize{ 0, 8192, 16384 }) |ctx| {
                if (pos + rows > capacity) continue;
                if (ctx != 0 and (ctx < pos + rows or ctx > capacity)) continue;
                try check(&f, "attention/tp{d}/c{d}/r{d}/p{d}/x{d}", .{ world, capacity, rows, pos, ctx }, Decode{ .rows = rows, .pos = pos, .ctx = if (ctx == 0) null else ctx, .g = g, .d = d, .sc = sc, .cache = cache });
            };
            for (prompts) |sn| {
                if (sn[0] + sn[1] > capacity) continue;
                try check(&f, "attention_prefill/tp{d}/c{d}/s{d}/n{d}", .{ world, capacity, sn[0], sn[1] }, Prefill{ .start = sn[0], .n = sn[1], .g = g, .d = d, .sc = sc, .cache = cache });
            }
        }
    }
}

test "commit launches equal forward.shift_windows': DeltaNet conv windows and the n-gram tail" {
    var f = try Fixture.load();
    defer f.deinit();
    const Shift = struct {
        keep: usize,
        old_l: usize,
        new_l: usize,
        new_row: usize,
        layers: usize,
        c: usize,
        taps: usize,
        fn go(s: @This(), t: Tri) !void {
            try t.shiftWindows(addr(0), addr(1), s.keep, s.old_l, s.new_l, s.new_row, s.layers, s.c, s.taps);
        }
    };
    for ([_]usize{ 1, 2 }) |world| {
        const d = Dims.rank(world);
        const conv = d.convDim();
        const proj = d.projWidth();
        for ([_][3]usize{ .{ 8, 1, 1 }, .{ 8, 7, 3 }, .{ 16, 16, 16 }, .{ 16, 5, 1 } }) |c| {
            try check(&f, "shift/conv/tp{d}/b{d}/r{d}/k{d}", .{ world, c[0], c[1], c[2] }, Shift{ .keep = c[2], .old_l = 3 * conv, .new_l = c[0] * proj, .new_row = proj, .layers = 36, .c = conv, .taps = 3 });
        }
        for ([_]usize{ 2048, 974, 30, 19, 18, 1 }) |n| {
            try check(&f, "shift/prefill/tp{d}/n{d}", .{ world, n }, Shift{ .keep = n, .old_l = 3 * conv, .new_l = 2048 * proj, .new_row = proj, .layers = 1, .c = conv, .taps = 3 });
        }
    }
    const d: Dims = .{};
    for ([_][3]usize{ .{ 8, 1, 1 }, .{ 8, 7, 4 }, .{ 2048, 2048, 2048 }, .{ 2048, 974, 974 }, .{ 2048, 30, 30 }, .{ 2048, 19, 19 }, .{ 2048, 18, 18 } }) |c| {
        try check(&f, "shift/ple/b{d}/r{d}/k{d}", .{ c[0], c[1], c[2] }, Shift{ .keep = c[2], .old_l = d.tailRows() * d.wide(), .new_l = c[0] * d.wide(), .new_row = d.wide(), .layers = 1, .c = d.wide(), .taps = d.tailRows() });
    }
}

test "split K and scratch sizes follow the Python shapes" {
    try testing.expectEqual(@as(usize, 32), b16SplitK(324, 10240));
    try testing.expectEqual(@as(usize, 1), b16SplitK(248320, 2560));
    try testing.expectEqual(@as(usize, 4), b16SplitK(2560, 6144));
    try testing.expectEqual(@as(usize, 2), b16SplitK(8240, 2560));
    try testing.expectEqual(@as(usize, 8), fp4SplitK(2560, 640));
    try testing.expectEqual(@as(usize, 4), fp4SplitK(2560, 320));
    try testing.expectEqual(@as(usize, 32 * 7 * 324 * 4), b16PartBytes(7, 324, 10240));
    const g = AttnGeometry.init(262144, 2048, 4);
    try testing.expectEqual(@as(usize, 5), g.nch);
    try testing.expectEqual(@as(usize, 65536), g.nb);
    try testing.expectEqual(@as(usize, 1), g.chunks(100));
    try testing.expectEqual(@as(usize, 5), g.chunks(null));
    try testing.expectEqual(@as(usize, 13952), (Dims{}).attnWidth());
    try testing.expectEqual(@as(usize, 7296), Dims.rank(2).attnWidth());
    try testing.expectEqual(@as(usize, 16480), (Dims{}).projWidth());
}
