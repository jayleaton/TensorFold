//! TF_DSV41_ROT_FOLD (3a): the knob, the decode windows on and off, and the epilogue's rotation against rot_in's.

const std = @import("std");
const testing = std.testing;
const block = @import("block.zig");
const block_attn = @import("block_attn.zig");
const calls = @import("block_calls.zig");
const check = @import("calls_check.zig");
const pk = @import("serve_knobs.zig");
const dense = @import("dsv41_kernels").dense;
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "TF_DSV41_ROT_FOLD: unset and 0 off, 1-3 bits 0-1, the reserved bit 2 and anything else refused" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expectEqual(@as(u3, 0), try pk.rotFold(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_ROT_FOLD", "0" }};
    try testing.expectEqual(@as(u3, 0), try pk.rotFold(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_ROT_FOLD", " 1 " }};
    try testing.expectEqual(@as(u3, 1), try pk.rotFold(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_ROT_FOLD", "2" }};
    try testing.expectEqual(@as(u3, 2), try pk.rotFold(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_ROT_FOLD", "3" }};
    try testing.expectEqual(@as(u3, 3), try pk.rotFold(&Fake.get));
    // 3c (bit 2) is reserved: a mask naming it is refused until it is built
    for ([_][]const u8{ "4", "5", "6", "7", "8", "-1", "on", "0x1", "1,2" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_ROT_FOLD", v }};
        try testing.expectError(error.BadKnob, pk.rotFold(&Fake.get));
    }
    try testing.expectEqual(@as(u3, 0), (block.Options{}).rot_fold);
}

// -- the emitted windows ---------------------------------------------------------------------------------------

fn argKey(a: std.mem.Allocator, out: *std.ArrayList(u8), x: calls.Arg) !void {
    switch (x) {
        .t, .opaque_table => |t| {
            try out.print(a, " {s}:{t}@{d}[", .{ switch (t.role) {
                .weight, .buf => |r| r,
                .empty => "-",
            }, t.dt, t.offset });
            for (t.shape, t.stride) |n, s| try out.print(a, "{d}/{d},", .{ n, s });
            try out.append(a, ']');
        },
        .list => |l| {
            try out.appendSlice(a, " (");
            for (l) |y| try argKey(a, out, y);
            try out.append(a, ')');
        },
        .i => |v| try out.print(a, " {d}", .{v}),
        .f => |v| try out.print(a, " {e}", .{v}),
        .b => |v| try out.print(a, " {any}", .{v}),
        .none => try out.appendSlice(a, " none"),
    }
}

pub fn oneKey(a: std.mem.Allocator, x: calls.Arg) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try argKey(a, &out, x);
    return out.items;
}

/// A call as compared here: name, grid, scope, streams and marks, and its first `upto` arguments.
pub fn key(a: std.mem.Allocator, c: calls.Call, upto: usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.print(a, "{s} {any} {t} side={any} fork={any} join={any} mark={d} wait={d} dside={any} djoin={any}", .{ c.name, c.grid, c.begin, c.side, c.fork, c.join, c.mark, c.wait, c.defer_side, c.defer_join });
    for (c.args[0..@min(upto, c.args.len)]) |x| {
        try out.print(a, " {s}=", .{x.name});
        try argKey(a, &out, x.arg);
    }
    return out.items;
}

pub fn firstRole(x: calls.Arg) ?[]const u8 {
    return switch (x) {
        .t => |t| switch (t.role) {
            .buf => |r| r,
            else => null,
        },
        .list => |l| if (l.len > 0) firstRole(l[0]) else null,
        else => null,
    };
}

/// wo_b's rot_in: the x3seg rotation whose input rows are L.oa.
fn woBRotIn(c: calls.Call) bool {
    if (!std.mem.eql(u8, c.name, "tf_dsv41_x3seg_v3.rot_in")) return false;
    const r = firstRole(c.args[0].arg) orelse return false;
    return std.mem.eql(u8, r, "L.oa");
}

/// A dense3 linear writing `role` (its first output).
pub fn writes(c: calls.Call, role: []const u8) bool {
    if (!std.mem.eql(u8, c.name, "tf_dsv41_dense3_v1.linear")) return false;
    const r = firstRole(c.args[3].arg) orelse return false;
    return std.mem.eql(u8, r, role);
}

fn isLinear(c: calls.Call) bool {
    return std.mem.eql(u8, c.name, "tf_dsv41_dense3_v1.linear");
}

/// The layer whose calls start at a `begin` (the window's first layer from the window's start).
fn layerOf(cs: []const calls.Call, i: usize, layers: []const u32) u32 {
    var l: usize = 0;
    for (cs[1 .. i + 1]) |c| l += @intFromBool(c.begin == .layer);
    return layers[l];
}

const Counts = struct { dropped: usize = 0, folded: usize = 0, wo_b: usize = 0 };

/// `on` is `off` less wo_b's rot_in, wo_a's linears carrying wo_b's rows, suh and stride; nothing else moves.
fn compareWindows(a: std.mem.Allocator, cfg: *const Config, layers: []const u32, n: i64, off: []const calls.Call, on: []const calls.Call) !Counts {
    var k: Counts = .{};
    var j: usize = 0;
    var rows: ?[]const u8 = null; // the rotated rows wo_a's epilogue wrote, for the wo_b linear that reads them
    for (off, 0..) |c, i| {
        if (woBRotIn(c)) {
            k.dropped += 1;
            continue;
        }
        try testing.expect(j < on.len);
        const d = on[j];
        j += 1;
        if (writes(c, "L.oa")) {
            k.folded += 1;
            try testing.expectEqual(@as(usize, 12), c.args.len);
            try testing.expectEqual(@as(usize, 15), d.args.len);
            try testing.expectEqualStrings(try key(a, c, 12), try key(a, d, 12));
            const L = layerOf(off, i, layers);
            const kb: i64 = @as(i64, cfg.o_groups / 2) * cfg.o_lora;
            var want: std.ArrayList(u8) = .empty;
            try want.print(a, " L.wo_b.xh:f16@0[{d}/1,]", .{n * kb});
            try testing.expectEqualStrings(want.items, try oneKey(a, d.args[12].arg));
            want = .empty;
            try want.print(a, " L{d}.attn.wo_b.suh:f16@0[{d}/1,]", .{ L, kb });
            try testing.expectEqualStrings(want.items, try oneKey(a, d.args[13].arg));
            try testing.expectEqual(kb, d.args[14].arg.i);
            rows = try oneKey(a, d.args[12].arg);
            continue;
        }
        try testing.expectEqualStrings(try key(a, c, c.args.len), try key(a, d, d.args.len));
        if (writes(c, "L.part")) {
            // wo_b's linear reads the very rows wo_a's epilogue wrote (role, dtype, extent)
            k.wo_b += 1;
            try testing.expectEqualStrings(rows.?, try oneKey(a, d.args[0].arg));
            rows = null;
        }
    }
    try testing.expectEqual(on.len, j);
    return k;
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    // a layer whose wo_a groups take two widths: two linear launches, each with its own segments' columns
    try w.dense.put(a, "L5.attn.wo_a.2", 12);
    return w;
}

test "TF_DSV41_ROT_FOLD: bit 0 drops each layer's wo_b rot_in and only adds wo_a's rotated-row arguments" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 8192;
    const pages = @divExact(limit, block.Pool.page);
    var served: block.Options = .{ .limit = limit, .r1 = true, .index_bound = true, .expert_topp = 0.85, .branches = true, .mhc_defer = true, .kv_norm_store = true };
    served.pool = .{ .comp_pages = @divExact(pages, 2) + 2, .ik_pages = pages + 1, .pts = pages, .split = true };
    var rows = served;
    rows.rows = true;
    for ([_]block.Options{ .{ .limit = limit }, served, rows }) |o| for ([_]i64{ 1, 5, 16, 17, 64 }) |n| {
        var on = o;
        on.rot_fold = 1;
        const off_cs = try block.emit(a, &cfg, &w, o, &backbone, n, 300, true);
        const on_cs = try block.emit(a, &cfg, &w, on, &backbone, n, 300, true);
        // flag off: H's calls (every layer's wo_b rot_in, every linear at its 12 arguments)
        var rot_ins: usize = 0;
        for (off_cs) |c| {
            rot_ins += @intFromBool(woBRotIn(c));
            if (isLinear(c)) try testing.expectEqual(@as(usize, 12), c.args.len);
        }
        try testing.expectEqual(backbone.len, rot_ins);
        const k = try compareWindows(a, &cfg, &backbone, n, off_cs, on_cs);
        try testing.expectEqual(backbone.len, k.dropped);
        try testing.expectEqual(backbone.len, k.wo_b);
        try testing.expectEqual(backbone.len + 1, k.folded); // L5's wo_a in two launches
        try testing.expectEqual(off_cs.len - backbone.len, on_cs.len);
    };
}

test "TF_DSV41_ROT_FOLD: woFold takes only whole 128-column groups; the knob's other bits do not fold wo_a" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var e: block.Emitter = .{ .a = a, .cfg = &cfg, .w = &w, .o = .{ .rot_fold = 1 }, .n = 4, .start = 0 };
    const x = try e.empty(.bf16);
    var po = [_]block.Emitter.Proj{
        .{ .w = "L0.attn.wo_a.0", .x = x, .y = x, .col = 0, .k = 4096, .n = 1024 },
        .{ .w = "L0.attn.wo_a.1", .x = x, .y = x, .col = 1024, .k = 4096, .n = 1024 },
    };
    try testing.expect(block_attn.woFold(&e, &po));
    po[1].col = 1088; // a group starting mid-block: its warps' columns straddle two rot_in blocks
    try testing.expect(!block_attn.woFold(&e, &po));
    po[1].col = 1024;
    po[1].n = 960;
    try testing.expect(!block_attn.woFold(&e, &po));
    po[1].n = 1024;
    for ([_]u3{ 0, 2, 4, 6 }) |m| {
        e.o.rot_fold = m;
        try testing.expect(!block_attn.woFold(&e, &po));
    }
}

// -- the epilogue's rotation against rot_in's, on the host --------------------------------------------------------

pub const had: f32 = 0.08838834764831845; // exl3_decode.cuh HAD_SCALE

/// __float2bfloat16_rn (cvt.rn.bf16.f32): nearest even, a NaN as the canonical 0x7fff.
pub fn bf16Bits(x: f32) u16 {
    if (std.math.isNan(x)) return 0x7fff;
    const u: u32 = @bitCast(x);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

pub fn widenBf16(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

pub fn widenF16(b: u16) f32 {
    return @as(f16, @bitCast(b));
}

pub fn f16Bits(x: f32) u16 {
    return @bitCast(@as(f16, @floatCast(x)));
}

/// exl3_decode.cuh fwht128: each lane's 4-point transform, then the xor-shuffle butterflies (lane bit m: o - v).
pub fn fwht128(v: *[32][4]f32) void {
    for (v) |*l| {
        const a = l[0] + l[1];
        const b = l[0] - l[1];
        const c = l[2] + l[3];
        const d = l[2] - l[3];
        l.* = .{ a + c, b + d, a - c, b - d };
    }
    var m: usize = 1;
    while (m < 32) : (m <<= 1) {
        const old = v.*;
        for (0..32) |lane| for (0..4) |j| {
            const o = old[lane ^ m][j];
            v[lane][j] = if (lane & m != 0) o - old[lane][j] else old[lane][j] + o;
        };
    }
}

/// x3seg.cu seg_rot_in_kernel (no norm): every 128-column block of `rows` bf16 rows x [rows, K] into fp16 xh.
fn rotIn(x: []const u16, suh: []const u16, xh: []u16, rows: usize, K: usize) void {
    for (0..rows) |row| for (0..K / 128) |blk| {
        var v: [32][4]f32 = undefined;
        for (0..32) |lane| for (0..4) |j| {
            const k = blk * 128 + 4 * lane + j;
            v[lane][j] = widenBf16(x[row * K + k]);
            v[lane][j] *= widenF16(suh[k]);
        };
        fwht128(&v);
        for (0..32) |lane| for (0..4) |j| {
            xh[row * K + blk * 128 + 4 * lane + j] = f16Bits(v[lane][j] * had);
        };
    };
}

const Trip = enum { bf16, none, f16 };

/// dense3_rot.cuh rot_out on one warp's 128 columns: y's floats rounded as store4 rounds them, widened as load4 does.
fn rotOut(y: []const f32, trip: Trip, suh: []const u16, xh: []u16) void {
    var v: [32][4]f32 = undefined;
    for (0..32) |lane| for (0..4) |j| {
        const f = y[4 * lane + j];
        v[lane][j] = switch (trip) {
            .bf16 => widenBf16(bf16Bits(f)),
            .f16 => widenF16(f16Bits(f)),
            .none => f,
        };
        v[lane][j] *= widenF16(suh[4 * lane + j]);
    };
    fwht128(&v);
    for (0..32) |lane| for (0..4) |j| {
        xh[4 * lane + j] = f16Bits(v[lane][j] * had);
    };
}

/// The fold over `segs`' launches: each segment's blocks at the columns its linTable y and rotOut entry name.
fn foldAll(a: std.mem.Allocator, segs: []const dense.Seg, y: []const f32, suh: []const u16, xh: []u16, rows: usize, ld: usize, trip: Trip) !void {
    const base: u64 = 0x100000;
    const lay = try dense.layout(segs, rows);
    const bufs = try a.alloc(dense.SegBufs, segs.len);
    defer a.free(bufs);
    for (bufs) |*b| b.* = .{ .T = 1, .svh = 1, .y = base, .ld = ld };
    for (lay.launches[0..lay.n_launches]) |l| {
        const tab = dense.linTable(lay, l, segs, bufs, .bf16, base, base, base);
        const cols = dense.launchCols(l, segs);
        const ro = try dense.rotOut(cols[0..l.count], base, base, ld);
        for (l.segs[0..l.count], 0..) |si, j| {
            const yc: usize = @intCast((tab.s[j].y - base) / 2);
            const xc: usize = @intCast((ro.xh[j] - base) / 2);
            const sc: usize = @intCast((ro.suh[j] - base) / 2);
            for (0..segs[si].n / 128) |nb| for (0..rows) |r| {
                const c0 = nb * 128;
                rotOut(y[r * ld + yc + c0 ..][0..128], trip, suh[sc + c0 ..][0..128], xh[r * @as(usize, @intCast(ro.ld)) + xc + c0 ..][0..128]);
            };
        }
    }
}

/// y's edge classes: signed zeros, infinities, NaNs, f32's largest, subnormals, bf16 ties, fp16 overflow.
pub const y_edges = [_]u32{ 0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00123, 0x7f7fffff, 0xff7fffff, 0x00000001, 0x807fffff, 0x3f808000, 0x3f818000, 0x3f80ffff, 0x477fe000, 0x47800000, 0xc7801000 };
const suh_edges = [_]u16{ 0x3c00, 0xbc00, 0x0000, 0x8000, 0x7bff, 0xfbff, 0x0001, 0x8001, 0x3555 };

fn randomY(r: std.Random, y: []f32) void {
    for (y) |*v| v.* = switch (r.uintLessThan(u8, 8)) {
        0 => @bitCast(y_edges[r.uintLessThan(usize, y_edges.len)]),
        1 => @bitCast(r.int(u32)), // any bit pattern
        2 => @floatCast(@as(f64, @floatFromInt(r.int(i16))) * 64.0), // past fp16 after the transform
        else => @floatCast(r.floatNorm(f64) * 0.25),
    };
}

pub fn randomSuh(r: std.Random, suh: []u16) void {
    for (suh) |*s| s.* = switch (r.uintLessThan(u8, 6)) {
        0 => suh_edges[r.uintLessThan(usize, suh_edges.len)],
        1, 2 => 0x3c00,
        3 => 0xbc00,
        else => f16Bits(@floatCast(r.floatNorm(f64))),
    };
}

/// wo_a-shaped groups at a small K: widths 10 / 12 / 10 / 4 (three launches), uneven column counts.
fn twinSegs() [4]dense.Seg {
    var segs: [4]dense.Seg = undefined;
    const ns = [_]usize{ 256, 128, 384, 256 };
    const k2s = [_]u32{ 10, 12, 10, 4 };
    var col: usize = 0;
    for (&segs, ns, k2s, 0..) |*s, n, k2, i| {
        const pl = dense.plan(512, n);
        s.* = .{ .k = 512, .n = n, .k2 = k2, .sk = pl[0], .wk = pl[1], .out = i, .col = col };
        col += n;
    }
    return segs;
}

test "rotation twin: wo_a's epilogue writes rot_in's exact fp16 rows (random and edge bf16 / fp16 values)" {
    const a = testing.allocator;
    const segs = twinSegs();
    const ld: usize = 1024;
    var prng = std.Random.DefaultPrng.init(0x3a_f01d);
    const r = prng.random();
    for ([_]usize{ 1, 3, 17 }) |rows| for (0..6) |_| {
        const y = try a.alloc(f32, rows * ld);
        defer a.free(y);
        const suh = try a.alloc(u16, ld);
        defer a.free(suh);
        randomY(r, y);
        randomSuh(r, suh);
        // H: the linear stores bf16 rows (L.oa), then rot_in reads them whole into L.wo_b.xh
        const oa = try a.alloc(u16, rows * ld);
        defer a.free(oa);
        for (oa, y) |*o, f| o.* = bf16Bits(f);
        const want = try a.alloc(u16, rows * ld);
        defer a.free(want);
        rotIn(oa, suh, want, rows, ld);
        // the fold: each launch's segments rotate their own blocks from the floats they store
        const got = try a.alloc(u16, rows * ld);
        defer a.free(got);
        @memset(got, 0xdead);
        try foldAll(a, &segs, y, suh, got, rows, ld, .bf16);
        try testing.expectEqualSlices(u16, want, got);
    };
}

test "rotation twin controls: no bf16 rounding, fp16 rounding or a mid-block group each change the bits" {
    const a = testing.allocator;
    const segs = twinSegs();
    const ld: usize = 1024;
    const rows: usize = 2;
    var prng = std.Random.DefaultPrng.init(0xc0_47e1);
    const r = prng.random();
    const y = try a.alloc(f32, rows * ld);
    defer a.free(y);
    const suh = try a.alloc(u16, ld);
    defer a.free(suh);
    for (y) |*v| v.* = @floatCast(r.floatNorm(f64)); // finite, rarely representable in bf16
    randomSuh(r, suh);
    const oa = try a.alloc(u16, rows * ld);
    defer a.free(oa);
    for (oa, y) |*o, f| o.* = bf16Bits(f);
    const want = try a.alloc(u16, rows * ld);
    defer a.free(want);
    rotIn(oa, suh, want, rows, ld);
    const got = try a.alloc(u16, rows * ld);
    defer a.free(got);
    for ([_]Trip{ .none, .f16 }) |trip| {
        try foldAll(a, &segs, y, suh, got, rows, ld, trip);
        try testing.expect(!std.mem.eql(u16, want, got));
    }
    // a group starting 64 columns into a block: its warp's 128 columns are not one rot_in block
    var block_got: [128]u16 = undefined;
    rotOut(y[64..192], .bf16, suh[64..192], &block_got);
    try testing.expect(!std.mem.eql(u16, want[64..192], &block_got));
    try testing.expectError(error.Unsupported, dense.rotOut(&.{ 0, 64 }, 0x1000, 0x2000, ld));
}
