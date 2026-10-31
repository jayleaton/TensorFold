//! TF_DSV41_KV_MIRROR: attention over the mirror (paged, the selection) against split KV's exchanged rows, every byte.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const Gpu = check.Gpu;
const attn = dsv41.ops.attn;
const rb: usize = 584;
const K: usize = 512;
const H: usize = 32;
const D: usize = 512;
const W: u32 = 2;
const ring: usize = 256;
const rope_rows: usize = 4096;

/// kv/mirror.zig's Geometry.pageOf (the runner has no kv module): owner-major runs a slot, then the discard page.
fn pageOf(slots: u32, mp: u32, s: u32, p: u32) u32 {
    if (p >= mp) return slots * mp;
    return s * mp + (p % W) * (mp / W) + p / W;
}

/// rows a window, rows-a-page shift, slots, pages a slot, mirrored pages a slot, row mode, R1's CTAs a chunk
const Case = struct { rows: u32, psh: u5, slots: u32, pts: u32, mp: u32, stacked: bool, split: u32 };

const cases = [_]Case{
    .{ .rows = 1, .psh = 7, .slots = 3, .pts = 16, .mp = 8, .stacked = false, .split = 4 },
    .{ .rows = 2, .psh = 8, .slots = 3, .pts = 16, .mp = 8, .stacked = true, .split = 4 },
    .{ .rows = 6, .psh = 7, .slots = 3, .pts = 16, .mp = 8, .stacked = true, .split = 1 },
    .{ .rows = 6, .psh = 8, .slots = 3, .pts = 16, .mp = 8, .stacked = false, .split = 0 },
    .{ .rows = 16, .psh = 7, .slots = 4, .pts = 24, .mp = 16, .stacked = true, .split = 1 },
};

/// A served record: 448 e4m3 bytes (exponent <= 7: finite), 64 bf16 rope values, 7 UE8M0 scales and a pad byte.
fn kvRow(rnd: std.Random, out: []u8) void {
    for (out[0..448]) |*b| b.* = rnd.int(u8) & 0xBF;
    for (0..64) |i| std.mem.writeInt(u16, out[448 + 2 * i ..][0..2], bf16(rnd.float(f32) * 2 - 1), .little);
    for (out[576..583]) |*b| b.* = 120 + rnd.uintLessThan(u8, 8);
    out[583] = 0;
}

fn bf16(x: f32) u16 {
    return @truncate(@as(u32, @bitCast(x)) >> 16);
}

fn up(gpu: Gpu, bytes: []const u8) !cuda.DeviceBuffer {
    return cuda.DeviceBuffer.fromHost(gpu.d, bytes);
}

fn zeroed(gpu: Gpu, n: usize) !cuda.DeviceBuffer {
    const b = try cuda.DeviceBuffer.alloc(gpu.d, @max(n, 16));
    try b.fill8(0, null);
    return b;
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels) !void {
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads and memsets
    defer stream.deinit();
    const o = k.others(stream);
    var prng = std.Random.DefaultPrng.init(0x4d31);
    for (cases) |c| try runCase(gpu, o, stream, prng.random(), c);
}

fn runCase(gpu: Gpu, o: dsv41.ops.Ops, stream: cuda.Stream, rnd: std.Random, c: Case) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const q: usize = @as(usize, 1) << c.psh;
    const R: usize = c.rows;
    // every slot's logical rows, the ranks' split storage (shuffled local pages, then the discard page)
    const rows = try a.alloc(u8, @as(usize, c.slots) * c.pts * q * rb);
    for (0..rows.len / rb) |i| kvRow(rnd, rows[i * rb ..][0..rb]);
    const owned = @as(usize, c.slots) * (c.pts / W);
    var local: [W][]u8 = undefined;
    var ct: [W][]i32 = undefined;
    for (0..W) |r| {
        local[r] = try a.alloc(u8, (owned + 1) * q * rb);
        rnd.bytes(local[r]);
        ct[r] = try a.alloc(i32, @as(usize, c.slots) * c.pts);
        const order = try a.alloc(u32, owned);
        for (order, 0..) |*x, i| x.* = @intCast(i);
        rnd.shuffle(u32, order);
        var next: usize = 0;
        for (0..c.slots * c.pts) |at| {
            if ((at % c.pts) % W != r) {
                ct[r][at] = @intCast(owned);
                continue;
            }
            const lp = order[next];
            next += 1;
            ct[r][at] = @intCast(lp);
            @memcpy(local[r][lp * q * rb ..][0 .. q * rb], rows[at * q * rb ..][0 .. q * rb]);
        }
    }
    // the mirror the stores leave, its table (one spare row past the slots: the planted control's shift)
    const tensor_pages = c.slots * c.mp + 1;
    const mirror = try a.alloc(u8, tensor_pages * q * rb);
    @memset(mirror, 0);
    for (0..c.slots) |s| for (0..c.pts * q) |t| {
        const pg = pageOf(c.slots, c.mp, @intCast(s), @intCast(t >> c.psh));
        if (pg == c.slots * c.mp) continue;
        @memcpy(mirror[(pg * q + t % q) * rb ..][0..rb], rows[((s * c.pts) * q + t) * rb ..][0..rb]);
    };
    const mt = try a.alloc(i32, (@as(usize, c.slots) + 1) * c.pts);
    for (0..c.slots + 1) |s| for (0..c.pts) |p| {
        mt[s * c.pts + p] = @intCast(pageOf(c.slots, c.mp, @intCast(s % c.slots), @intCast(p)));
    };
    // the window: each row's slot, position, and a selection of distinct visible rows inside the mirror
    const one: u32 = 1;
    const slot = try a.alloc(i64, R);
    const rslot = try a.alloc(i32, R);
    const pos64 = try a.alloc(i64, R);
    const cnt = try a.alloc(i32, R);
    const sel = try a.alloc(i32, R * K);
    const pool = try a.alloc(i32, @as(usize, c.mp) << c.psh);
    const p0: i32 = 1500;
    for (0..R) |r| {
        slot[r] = if (c.stacked) rnd.uintLessThan(u32, c.slots) else one;
        rslot[r] = @intCast(slot[r]);
        pos64[r] = if (c.stacked) 200 + @as(i64, rnd.uintLessThan(u32, 1800)) else @as(i64, p0) + @as(i64, @intCast(r));
        const nvis = 1 + rnd.uintLessThan(usize, pool.len);
        const n = @min(nvis, K);
        for (pool[0..nvis], 0..) |*x, i| x.* = @intCast(i);
        for (0..n) |i| std.mem.swap(i32, &pool[i], &pool[i + rnd.uintLessThan(usize, nvis - i)]);
        @memcpy(sel[r * K ..][0..n], pool[0..n]);
        @memset(sel[r * K + n .. (r + 1) * K], -1);
        cnt[r] = @intCast(n);
    }
    const qh = try a.alloc(u16, R * H * D);
    for (qh) |*x| x.* = bf16(rnd.float(f32) - 0.5);
    const nring: usize = if (c.stacked) @as(usize, c.slots) * ring else ring; // row mode reads (p % ring) + slot * ring
    const swa = try a.alloc(u8, nring * rb);
    for (0..nring) |i| kvRow(rnd, swa[i * rb ..][0..rb]);
    const sv = try a.alloc(u8, nring * 576);
    const ssc = try a.alloc(u8, nring * 8);
    for (0..nring) |i| {
        @memcpy(sv[i * 576 ..][0..576], swa[i * rb ..][0..576]);
        @memcpy(ssc[i * 8 ..][0..8], swa[i * rb + 576 ..][0..8]);
    }
    const cs = try a.alloc(f32, rope_rows * 64);
    for (cs) |*x| x.* = rnd.float(f32) * 2 - 1;

    var d_local: [W]cuda.DeviceBuffer = undefined;
    var d_ct: [W]cuda.DeviceBuffer = undefined;
    for (0..W) |r| {
        d_local[r] = try up(gpu, local[r]);
        d_ct[r] = try up(gpu, std.mem.sliceAsBytes(ct[r]));
    }
    defer for (0..W) |r| {
        d_local[r].free();
        d_ct[r].free();
    };
    var d_mirror = try up(gpu, mirror);
    defer d_mirror.free();
    var d_mt = try up(gpu, std.mem.sliceAsBytes(mt));
    defer d_mt.free();
    var d_sel = try up(gpu, std.mem.sliceAsBytes(sel));
    defer d_sel.free();
    var d_cnt = try up(gpu, std.mem.sliceAsBytes(cnt));
    defer d_cnt.free();
    var d_rslot = try up(gpu, std.mem.sliceAsBytes(rslot));
    defer d_rslot.free();
    var d_slot = try up(gpu, std.mem.sliceAsBytes(slot));
    defer d_slot.free();
    var d_pos = if (c.stacked) try up(gpu, std.mem.sliceAsBytes(pos64)) else try up(gpu, std.mem.asBytes(&p0));
    defer d_pos.free();
    var d_q = try up(gpu, std.mem.sliceAsBytes(qh));
    defer d_q.free();
    var d_sv = try up(gpu, sv);
    defer d_sv.free();
    var d_ssc = try up(gpu, ssc);
    defer d_ssc.free();
    var d_cs = try up(gpu, std.mem.sliceAsBytes(cs));
    defer d_cs.free();
    var d_recv = try zeroed(gpu, W * R * K * rb);
    defer d_recv.free();
    var d_tok = try zeroed(gpu, R * K * 4);
    defer d_tok.free();
    var d_lo = try zeroed(gpu, R * 4);
    defer d_lo.free();
    var d_sink = try zeroed(gpu, H * 4);
    defer d_sink.free();
    var d_po = try zeroed(gpu, 5 * R * H * D * 4);
    defer d_po.free();
    var d_pm = try zeroed(gpu, 5 * R * H * 4);
    defer d_pm.free();
    var d_pl = try zeroed(gpu, 5 * R * H * 4);
    defer d_pl.free();
    var d_ticket = try zeroed(gpu, 1024 * 4);
    defer d_ticket.free();
    var outs: [3]cuda.DeviceBuffer = undefined;
    for (&outs) |*x| x.* = try zeroed(gpu, R * H * D * 2);
    defer for (&outs) |*x| x.free();

    // split KV: each rank's dense gather into its shard of the all-gather's receive buffer, the remapped tokens
    for (0..W) |r| {
        const tab = d_ct[r].ptr + (if (c.stacked) 0 else @as(u64, one) * c.pts * 4);
        try o.kvDense(.{ .sel = d_sel.ptr, .rows = @intCast(R), .k = K, .table = tab, .pts = if (c.stacked) c.pts else 0, .rslot = if (c.stacked) d_rslot.ptr else 0, .psh = c.psh, .base = d_local[r].ptr, .row_bytes = rb, .world = W, .send = d_recv.ptr + r * R * K * rb, .tok = d_tok.ptr });
    }
    const base: attn.Args = .{
        .q = d_q.ptr,
        .ts = K,
        .cnt = d_cnt.ptr,
        .sv = d_sv.ptr,
        .ssc = d_ssc.ptr,
        .svs = 576,
        .sss = 8,
        .lo = d_lo.ptr,
        .pos = d_pos.ptr,
        .sl = if (c.stacked) d_slot.ptr else 0,
        .sink = d_sink.ptr,
        .cs = d_cs.ptr,
        .csst = 64,
        .po = d_po.ptr,
        .pm = d_pm.ptr,
        .pl = d_pl.ptr,
        .ticket = d_ticket.ptr,
        .R = @intCast(R),
        .H = H,
        .ring = ring,
        .has_comp = 1,
    };
    var split = base;
    split.cv = d_recv.ptr;
    split.csc = d_recv.ptr + 576;
    split.cvs = rb;
    split.css = rb;
    split.tok = d_tok.ptr;
    split.out = outs[0].ptr;
    // the mirror: KV_SPLIT=0's read (the selection through a page table) of the mirror's rows
    var mir = base;
    mir.cv = d_mirror.ptr;
    mir.csc = d_mirror.ptr + 576;
    mir.cvs = rb;
    mir.css = rb;
    mir.tok = d_sel.ptr;
    mir.pt = d_mt.ptr + (if (c.stacked) 0 else @as(u64, one) * c.pts * 4);
    mir.pts = if (c.stacked) c.pts else 0;
    mir.psh = c.psh;
    mir.out = outs[1].ptr;
    // planted control: the next slot's mirror table (other rows), which must change the output
    var wrong = mir;
    wrong.pt = mir.pt + c.pts * 4;
    wrong.out = outs[2].ptr;
    for ([_]attn.Args{ split, mir, wrong }) |x| {
        try d_ticket.fill8(0, stream.handle);
        try o.attention(x, 4, c.stacked, false, c.split);
    }
    try stream.synchronize();
    const got_split = try check.download(gpu, outs[0]);
    defer gpu.gpa.free(got_split);
    const got_mir = try check.download(gpu, outs[1]);
    defer gpu.gpa.free(got_mir);
    const got_wrong = try check.download(gpu, outs[2]);
    defer gpu.gpa.free(got_wrong);
    try check.expect(std.mem.indexOfNone(u8, got_split, &.{0}) != null, "kv-mirror: an all-zero output", .{});
    for (std.mem.bytesAsSlice(u16, got_split)) |x| {
        try check.expect(x & 0x7f80 != 0x7f80, "kv-mirror: a NaN or Inf in the split output", .{});
    }
    try check.sameBytes("kv-mirror attention: mirror vs split KV's exchanged rows", got_mir, got_split);
    try check.expect(!std.mem.eql(u8, got_wrong, got_split), "kv-mirror: the planted wrong table gave the same bytes", .{});

    // a restore's fill: each rank's run gathered, landed at its owner's part of the slot's run, equals the stores'
    const fill_rows: u32 = (c.mp / W) << c.psh;
    var d_fill = try zeroed(gpu, tensor_pages * q * rb);
    defer d_fill.free();
    const phys = try a.alloc(u32, fill_rows);
    var d_phys = try zeroed(gpu, fill_rows * 4);
    defer d_phys.free();
    for (0..c.slots) |s| for (0..W) |r| {
        for (0..c.mp / W) |j| {
            const lp: u32 = @intCast(ct[r][s * c.pts + r + W * j]);
            for (0..q) |i| phys[j * q + i] = (lp << c.psh) | @as(u32, @intCast(i));
        }
        try d_phys.upload(0, std.mem.sliceAsBytes(phys));
        const at = (s * c.mp * q + r * fill_rows) * rb;
        try o.kvGather(d_local[r].ptr, rb, d_phys.ptr, fill_rows, d_fill.ptr + at);
        try stream.synchronize();
    };
    const got_fill = try check.download(gpu, d_fill);
    defer gpu.gpa.free(got_fill);
    const run_bytes = c.slots * c.mp * q * rb;
    try check.sameBytes("kv-mirror fill: gathered runs vs the stores' mirror", got_fill[0..run_bytes], mirror[0..run_bytes]);
    check.pass("BITEXACT kv-mirror: attention through the mirror == split KV's exchange, fill == stores: R {d}, psh {d}, {s}, split {d}", .{ c.rows, c.psh, if (c.stacked) "row mode" else "one slot", c.split });
}
