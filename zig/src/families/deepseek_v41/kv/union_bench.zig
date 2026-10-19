//! tf-kv-union-bench [positions] [rows] [k] [capacity] [iters]: kx_union host round trip, base vs PF_HOSTGAP.

const std = @import("std");
const cuda = @import("cuda");
const split = @import("split.zig");
const uf = @import("union_fast.zig");

const psh = 6;
const world = 2;
const row_bytes = 584;

fn nowUs() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1e6 + @as(f64, @floatFromInt(ts.nsec)) / 1e3;
}

/// A prompt's last segment: row i sees the comp rows before its position (ratio 4), k distinct of them, -1 past them.
fn selection(r: std.Random, sel: []i32, rows: u32, k: u32, positions: u64) void {
    const start = positions - rows;
    for (0..rows) |i| {
        const seen: u32 = @intCast((start + i + 1) / 4);
        const row = sel[i * k ..][0..k];
        for (row, 0..) |*t, j| t.* = if (j < seen) @intCast(if (seen <= k) j else r.uintLessThan(u32, seen)) else -1;
    }
}

/// One round trip's time and its union's rows an owner.
const Trip = struct { us: f64, m: u32 };

const Bench = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    dsel: cuda.DeviceBuffer,
    dtok: cuda.DeviceBuffer,
    dphys: cuda.DeviceBuffer,
    pin_sel: cuda.HostBuffer,
    pin_up: cuda.HostBuffer,
    table: []u32,
    planner: uf.Planner = .{},
    n: usize,
    k: u32,

    fn input(b: *Bench, sel: []const i32, rank: u32) split.UnionIn {
        return .{ .sel = sel, .k = b.k, .table = b.table, .psh = psh, .world = world, .rank = rank, .row_bytes = row_bytes, .cap = 128 << 20 };
    }

    /// kv_state.unionOf's host side as at the base: drain, read back, plan with allocations, copy into staging, upload.
    fn before(b: *Bench, rank: u32) !Trip {
        const t0 = nowUs();
        try b.stream.synchronize();
        const sel = b.pin_sel.slice(i32)[0..b.n];
        try b.dsel.download(0, std.mem.sliceAsBytes(sel));
        const table = try b.gpa.alloc(u32, b.table.len);
        defer b.gpa.free(table);
        @memcpy(table, b.table);
        var blocks: std.ArrayList(split.Block) = .empty;
        defer {
            for (blocks.items) |*x| x.deinit(b.gpa);
            blocks.deinit(b.gpa);
        }
        var in = b.input(sel, rank);
        in.table = table;
        try split.planUnion(b.gpa, in, &blocks);
        const x = blocks.items[0];
        const up = b.pin_up.slice(u32);
        @memcpy(up[0..x.phys.len], x.phys);
        @memcpy(up[x.phys.len..][0..x.tokens.len], std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(x.tokens)));
        try b.upload(up, x.m);
        return .{ .us = nowUs() - t0, .m = x.m };
    }

    /// The same under TF_DSV41_PF_HOSTGAP: the planner writes straight into the staging.
    fn after(b: *Bench, rank: u32) !Trip {
        const t0 = nowUs();
        try b.stream.synchronize();
        const sel = b.pin_sel.slice(i32)[0..b.n];
        try b.dsel.download(0, std.mem.sliceAsBytes(sel));
        const in = b.input(sel, rank);
        const m = (try b.planner.measure(b.gpa, in)) orelse return error.SeveralBlocks;
        const up = b.pin_up.slice(u32);
        b.planner.write(in, up[0..m], std.mem.bytesAsSlice(i32, std.mem.sliceAsBytes(up[m..][0..b.n])));
        try b.upload(up, m);
        return .{ .us = nowUs() - t0, .m = m };
    }

    fn upload(b: *Bench, up: []u32, m: u32) !void {
        try b.dphys.uploadAsync(0, std.mem.sliceAsBytes(up[0..m]), b.stream.handle);
        try b.dtok.uploadAsync(0, std.mem.sliceAsBytes(up[m..][0..b.n]), b.stream.handle);
        try b.stream.synchronize();
    }

    /// What the device holds after a trip: phys then tokens.
    fn landed(b: *Bench, m: u32, out: []u8) !void {
        try b.dphys.download(0, out[0 .. 4 * m]);
        try b.dtok.download(0, out[4 * m ..][0 .. 4 * b.n]);
    }
};

fn arg(args: []const []const u8, i: usize, default: u64) !u64 {
    return if (args.len > i) try std.fmt.parseInt(u64, args[i], 10) else default;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const positions = try arg(args, 1, 37_982);
    const rows: u32 = @intCast(try arg(args, 2, 4096));
    const k: u32 = @intCast(try arg(args, 3, 512));
    const capacity = try arg(args, 4, 1 << 20);
    const iters = try arg(args, 5, 50);
    if (positions < rows or capacity < positions) return error.BadShape;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const n: usize = @as(usize, rows) * k;
    const pages: usize = @intCast(capacity / 256);
    const most = uf.ownerBound(pages, psh, world, n);
    var b: Bench = .{
        .gpa = gpa,
        .d = &driver,
        .stream = try cuda.Stream.init(&driver, true),
        .dsel = try cuda.DeviceBuffer.alloc(&driver, 4 * n),
        .dtok = try cuda.DeviceBuffer.alloc(&driver, 4 * n),
        .dphys = try cuda.DeviceBuffer.alloc(&driver, 4 * most),
        .pin_sel = try cuda.HostBuffer.alloc(&driver, 4 * n),
        .pin_up = try cuda.HostBuffer.alloc(&driver, 4 * (most + n)),
        .table = try gpa.alloc(u32, pages),
        .n = n,
        .k = k,
    };
    defer {
        b.planner.deinit(gpa);
        gpa.free(b.table);
        b.pin_up.free();
        b.pin_sel.free();
        b.dphys.free();
        b.dtok.free();
        b.dsel.free();
        b.stream.deinit();
    }
    var prng = std.Random.DefaultPrng.init(0x1100);
    const r = prng.random();
    for (b.table, 0..) |*t, i| t.* = @intCast(i / world);
    const sel = try gpa.alloc(i32, n);
    defer gpa.free(sel);
    const want = try gpa.alloc(u8, 4 * (most + n));
    defer gpa.free(want);
    const got = try gpa.alloc(u8, 4 * (most + n));
    defer gpa.free(got);
    var t_before: f64 = 0;
    var t_after: f64 = 0;
    var t_churn: f64 = 0;
    var bad: u64 = 0;
    for (0..iters) |it| {
        selection(r, sel, rows, k, positions);
        try b.dsel.upload(0, std.mem.sliceAsBytes(sel));
        const rank: u32 = @intCast(it % world);
        const x = try b.before(rank);
        try b.landed(x.m, want);
        const y = try b.after(rank);
        try b.landed(y.m, got);
        bad += @intFromBool(x.m != y.m or !std.mem.eql(u8, want[0 .. 4 * (x.m + n)], got[0 .. 4 * (y.m + n)]));
        // the base's growth of its upload staging (cuMemFreeHost + cuMemHostAlloc) as the union grows a row
        const c0 = nowUs();
        var h = try cuda.HostBuffer.alloc(&driver, 4 * (x.m + 1 + n));
        h.free();
        if (it > 0) {
            t_before += x.us;
            t_after += y.us;
            t_churn += nowUs() - c0;
        }
    }
    const per = @as(f64, @floatFromInt(@max(iters, 2) - 1));
    std.debug.print("kx_union host round trip: positions {d}, rows {d}, k {d}, capacity {d}, {d} trips (first dropped)\n", .{ positions, rows, k, capacity, iters });
    std.debug.print("  before: {d:.0} us a call; staging regrowth when m grows: +{d:.0} us\n", .{ t_before / per, t_churn / per });
    std.debug.print("  TF_DSV41_PF_HOSTGAP=1: {d:.0} us a call; bytes landed equal: {s} ({d} differ)\n", .{ t_after / per, if (bad == 0) "yes" else "NO", bad });
    return if (bad == 0) 0 else 1;
}
