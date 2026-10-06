//! tf-kv-split-test KVSPLIT_FATBIN: split KV's exchange on real GPUs, one process a rank (tp's settings: TF_TP_RANK, TF_TP_WORLD,
//! TF_TP_DEVICE, TF_TP_PORT, ...). Each rank's device pool stores only the comp rows it owns; the kernel agent's kvsplit kernels pack
//! the selected rows and tp's all-gather trades them. Every received row equals the replicated pool's and every token the reference
//! rule: dense windows (row mode, 3 slots, R 1 / 6 / 16, K 512, ratio 2 and 1), the same exchange replayed from a CUDA graph with new
//! selections, and capped prefill unions. Exit 0: all equal on this rank.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const sessions = @import("sessions");
const split = @import("split.zig");
const DevicePool = @import("device.zig").DevicePool;

/// The kvsplit kernels (zig/kernels/cuda/deepseek_v41/kvsplit.cu) behind split.Kernels.
const DeviceKernels = struct {
    d: *const cuda.Driver,
    dense_fn: cuda.Function,
    gather_fn: cuda.Function,

    fn grid(words: u64) u32 {
        return @intCast(@max(1, @min((words + 255) / 256, 4096)));
    }

    fn kernels(k: *DeviceKernels) split.Kernels {
        return .{ .ptr = k, .vtable = &.{ .dense = dense, .gather = gather } };
    }

    fn dense(ptr: *anyopaque, a: split.DenseArgs, stream: tp.collective.Stream) anyerror!void {
        const k: *DeviceKernels = @ptrCast(@alignCast(ptr));
        const words = @as(u64, a.rows) * a.k * (a.row_bytes / 8);
        if (words == 0) return;
        var args: cuda.Args = .{};
        args.add(a);
        try cuda.launch.launch(k.dense_fn, .{ .grid = .{ .x = grid(words) }, .block = .{ .x = 256 } }, .{ .d = k.d, .handle = stream }, &args);
    }

    fn gather(ptr: *anyopaque, base: u64, rb: u32, phys: u64, n: u32, send: u64, stream: tp.collective.Stream) anyerror!void {
        const k: *DeviceKernels = @ptrCast(@alignCast(ptr));
        const words = @as(u64, n) * (rb / 8);
        if (words == 0) return;
        var args: cuda.Args = .{};
        args.add(base);
        args.add(rb);
        args.add(phys);
        args.add(n);
        args.add(send);
        try cuda.launch.launch(k.gather_fn, .{ .grid = .{ .x = grid(words) }, .block = .{ .x = 256 } }, .{ .d = k.d, .handle = stream }, &args);
    }
};

/// The replicated pool's row t of slot s in family f, byte i (8-byte words, so a whole row is filled fast).
fn word(f: u64, s: u64, t: u64, i: u64) u64 {
    return (f << 56) ^ (s << 48) ^ (t << 20) ^ (i *% 0x9E3779B97F4A7C15);
}

const page = 256;
const fams = [_]sessions.Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = 584, .split = true },
    .{ .name = "comp.20", .ratio = 1, .row_bytes = 584, .split = true },
};

const Rank = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    comm: tp.Collective,
    stream: cuda.Stream,
    pool: *sessions.Pool,
    dp: *DevicePool,
    x: split.Exchange,
    lens: [3]u64,
    bad: u64 = 0,
    checks: u64 = 0,
    shown: u32 = 0,

    fn rowsOf(r: *const Rank, fi: usize, s: usize) u32 {
        return @intCast(r.lens[s] / fams[fi].ratio);
    }

    /// Selections for R rows (row r on slot r % 3), K entries, some -1; the same on every rank (same seed).
    fn select(r: *const Rank, rnd: std.Random, fi: usize, R: u32, K: u32, sel: []i32, rslot: []i32) void {
        for (rslot[0..R], 0..) |*s, i| s.* = @intCast(i % 3);
        for (sel[0 .. R * K], 0..) |*t, i| {
            const n = r.rowsOf(fi, @intCast(rslot[i / K]));
            t.* = if (rnd.uintLessThan(u32, 16) == 0) -1 else @intCast(rnd.uintLessThan(u32, n));
        }
    }

    /// Received rows and tokens against the reference: tok = owner*n + i, row = the replicated pool's.
    fn check(r: *Rank, fi: usize, sel: []const i32, rslot: []const i32, K: u32, tok: []const i32, recv: []const u8, psh: u32) void {
        const n: i64 = @intCast(sel.len);
        for (sel, tok, 0..) |t, o, i| {
            r.checks += 1;
            if (t < 0) {
                r.bad += @intFromBool(o != -1);
                continue;
            }
            const owner: i64 = split.ownerOf(@intCast(t), psh, 2);
            if (o != owner * n + @as(i64, @intCast(i))) {
                r.bad += 1;
                continue;
            }
            const row = std.mem.bytesAsSlice(u64, recv[@as(usize, @intCast(o)) * 584 ..][0..584]);
            const before = r.bad;
            for (row, 0..) |w, j| r.bad += @intFromBool(w != word(fi, @intCast(rslot[i / K]), @intCast(t), j));
            if (r.bad != before and r.shown < 6) {
                r.shown += 1;
                std.debug.print("MISMATCH fam {d} slot {d} entry {d} t {d} owner {d}: got {x} {x}, want {x} {x}\n", .{ fi, rslot[i / K], i, t, owner, row[0], row[1], word(fi, @intCast(rslot[i / K]), @intCast(t), 0), word(fi, @intCast(rslot[i / K]), @intCast(t), 1) });
            }
        }
    }

    fn dense(r: *Rank, rnd: std.Random, fi: u32, R: u32, graphs: bool) !void {
        const K = 512;
        const n = R * K;
        const g = r.gpa;
        const sel = try g.alloc(i32, n);
        defer g.free(sel);
        const rslot = try g.alloc(i32, R);
        defer g.free(rslot);
        const tok = try g.alloc(i32, n);
        defer g.free(tok);
        const recv_h = try g.alloc(u8, 2 * n * 584);
        defer g.free(recv_h);
        var dsel = try cuda.DeviceBuffer.alloc(r.d, n * 4);
        defer dsel.free();
        var dslot = try cuda.DeviceBuffer.alloc(r.d, R * 4);
        defer dslot.free();
        var dtok = try cuda.DeviceBuffer.alloc(r.d, n * 4);
        defer dtok.free();
        var send = try cuda.DeviceBuffer.alloc(r.d, n * 584);
        defer send.free();
        var recv = try cuda.DeviceBuffer.alloc(r.d, 2 * n * 584);
        defer recv.free();
        const v = r.dp.view(fi, 0);
        const a: split.DenseArgs = .{ .sel = dsel.ptr, .rows = R, .k = K, .table = v.pt, .pts = @intCast(v.pts), .rslot = dslot.ptr, .psh = @intCast(v.psh), .base = v.cv, .row_bytes = 584, .world = 2, .send = send.ptr, .tok = dtok.ptr };
        var exec: ?cuda.graph.Exec = null;
        defer if (exec) |*e| e.deinit();
        if (graphs) {
            try cuda.graph.beginCapture(r.stream, .global);
            try r.x.dense(a, recv.ptr, r.stream.handle);
            var gr = try cuda.graph.endCapture(r.stream);
            defer gr.deinit();
            exec = try gr.instantiate();
        }
        for (0..if (graphs) 3 else 1) |_| {
            r.select(rnd, fi, R, K, sel, rslot);
            try dsel.upload(0, std.mem.sliceAsBytes(sel));
            try dslot.upload(0, std.mem.sliceAsBytes(rslot));
            if (exec) |e| try e.launchOn(r.stream) else try r.x.dense(a, recv.ptr, r.stream.handle);
            try r.stream.synchronize();
            try dtok.download(0, std.mem.sliceAsBytes(tok));
            try recv.download(0, recv_h);
            r.check(fi, sel, rslot, K, tok, recv_h, @intCast(v.psh));
        }
    }

    fn unions(r: *Rank, rnd: std.Random, fi: u32, rank: u32) !u32 {
        const K = 512;
        const S = 256; // a prefill segment's rows
        const g = r.gpa;
        const sel = try g.alloc(i32, S * K);
        defer g.free(sel);
        for (sel) |*t| t.* = if (rnd.uintLessThan(u32, 16) == 0) -1 else @intCast(rnd.uintLessThan(u32, r.rowsOf(fi, 2)));
        const v = r.dp.view(fi, 2);
        const table = try g.alloc(u32, r.dp.pts);
        defer g.free(table);
        for (table, 0..) |*t, k| t.* = r.pool.slots.items[2].localTableAt(@intCast(k));
        var blocks: std.ArrayList(split.Block) = .empty;
        defer {
            for (blocks.items) |*b| b.deinit(g);
            blocks.deinit(g);
        }
        try split.planUnion(g, .{ .sel = sel, .k = K, .table = table, .psh = @intCast(v.psh), .world = 2, .rank = rank, .row_bytes = 584, .cap = 24 << 20 }, &blocks);
        for (blocks.items) |blk| {
            var phys = try cuda.DeviceBuffer.fromHost(r.d, std.mem.sliceAsBytes(blk.phys));
            defer phys.free();
            var send = try cuda.DeviceBuffer.alloc(r.d, blk.m * 584);
            defer send.free();
            var recv = try cuda.DeviceBuffer.alloc(r.d, 2 * blk.m * 584);
            defer recv.free();
            try r.x.unionBlock(v.cv, 584, phys.ptr, blk.m, send.ptr, recv.ptr, r.stream.handle);
            try r.stream.synchronize();
            const h = try g.alloc(u8, 2 * blk.m * 584);
            defer g.free(h);
            try recv.download(0, h);
            for (sel[blk.a * K .. blk.b * K], blk.tokens) |t, o| {
                r.checks += 1;
                if (t < 0) continue;
                const row = std.mem.bytesAsSlice(u64, h[@as(usize, @intCast(o)) * 584 ..][0..584]);
                for (row, 0..) |w, j| r.bad += @intFromBool(w != word(fi, 2, @intCast(t), j));
            }
        }
        return @intCast(blocks.items.len);
    }
};

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: tf-kv-split-test KVSPLIT_FATBIN (tp's TF_TP_* settings)\n", .{});
        return 2;
    }
    const image = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], gpa, .limited(64 << 20));
    defer gpa.free(image);
    const cfg = try tp.Config.fromEnv();
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, @intCast(cfg.device));
    defer ctx.deinit();
    var s: tp.Session = undefined;
    try s.open(&driver, cfg, "");
    var open = true;
    defer if (open) s.close();
    var module = try cuda.Module.load(&driver, image);
    defer module.unload();
    var dk: DeviceKernels = .{ .d = &driver, .dense_fn = try module.function("_ZN13dsv41_kvsplit12dense_kernelENS_9DenseArgsE"), .gather_fn = try module.function("_ZN13dsv41_kvsplit13gather_kernelEPKhjPKjjPh") };
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var pool = try sessions.Pool.init(gpa, .{ .families = &fams, .page = page }, 8192 * page, 2, cfg.rank);
    defer pool.deinit();
    const lens = [3]u64{ 300_000 + 77, 41_000, 1_000_000 + 5 };
    for (lens) |n| try (try pool.newSlot(1 << 20)).ensure(n);
    var dp = try DevicePool.init(gpa, &driver, &ctx, &pool, stream, 3, (1 << 20) / page);
    defer dp.deinit();
    try dp.syncTables();
    // each rank writes the rows of the pages it owns (kv_store's writes; the other rank's land on its discard page)
    for (fams, 0..) |f, fi| {
        const pb = f.pageBytes(page);
        const buf = try gpa.alloc(u8, pb);
        defer gpa.free(buf);
        const per = page / f.ratio;
        for (pool.slots.items, 0..) |sl, si| for (sl.mapped(), 0..) |pg, k| {
            if (!pool.owns(pg)) continue;
            const w = std.mem.bytesAsSlice(u64, buf);
            for (0..per) |row| for (0..73) |j| {
                w[row * 73 + j] = word(fi, si, k * per + row, j);
            };
            try dp.store().write(@intCast(fi), &.{pool.localPage(pg)}, buf);
        };
    }
    try stream.synchronize();
    var r: Rank = .{ .gpa = gpa, .d = &driver, .comm = s.comm(), .stream = stream, .pool = &pool, .dp = &dp, .x = .{ .comm = s.comm(), .kernels = dk.kernels() }, .lens = lens };
    var prng = std.Random.DefaultPrng.init(20261006);
    const rnd = prng.random();
    for ([_]u32{ 0, 1 }) |fi| for ([_]u32{ 1, 6, 16 }) |R| {
        try r.dense(rnd, fi, R, false);
        try r.dense(rnd, fi, R, true);
    };
    var blocks: u32 = 0;
    for ([_]u32{ 0, 1 }) |fi| blocks += try r.unions(rnd, fi, cfg.rank);
    // the time of one dense exchange (pack + all-gather), graph-replayed, R = 16, K = 512
    const t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
    for (0..20) |_| try r.dense(rnd, 1, 16, false);
    const dt = @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).toNanoseconds() - t0)) / 20e6;
    std.debug.print("rank {d}: {d} entries checked over dense (R 1/6/16, eager + 3 graph replays, ratio 2 and 1) and {d} union blocks; mismatches {d}; {d:.2} ms a dense round trip incl. host checks\n{s}\n", .{ cfg.rank, r.checks, blocks, r.bad, dt, if (r.bad == 0) "PASS" else "FAIL" });
    // both ranks finish, then close the session (its fate channel) at once, before the slower teardown below
    try r.comm.barrier();
    s.close();
    open = false;
    return if (r.bad == 0) 0 else 1;
}
