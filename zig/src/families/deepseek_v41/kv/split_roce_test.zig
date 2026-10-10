//! Split KV's Exchange on RoCE's protocol at prod's settings (hybrid routes, 2 stripes): each post inside slot and MR.

const std = @import("std");
const testing = std.testing;
const sessions = @import("sessions");
const tp = @import("tp");
const split = @import("split.zig");
const roce = tp.roce;
const rv = tp.roce_verbs;
const hybrid = tp.hybrid;

const page = 256;
const pts = 16;
const rb = 584; // the comp rows' bytes on the model
const K = 512;
const fams = [_]sessions.Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = rb, .split = true },
    .{ .name = "comp.1", .ratio = 1, .row_bytes = rb, .split = true },
};
const slot_lens = [_]u64{ 15 * page + 7, 9 * page, 13 * page + 100 };

/// prod-zig.env as serve.sh passes it to the engine (ZIG_COMM_BACKEND -> TF_COMM_BACKEND); TF_TP_ROCE_VAR_KB unset.
fn prodEnv(name: []const u8) ?[]const u8 {
    const table = .{ .{ "TF_TP_WORLD", "2" }, .{ "TF_COMM_BACKEND", "roce" }, .{ "GLM53_TF_ROCE_MAX_KB", "1024" }, .{ "GLM53_TF_ROCE_FAST", "1" } };
    inline for (table) |kv| if (std.mem.eql(u8, name, kv[0])) return kv[1];
    return null;
}

/// The 10-10 21:13 test's settings: prod's plus TF_TP_ROCE_VAR_KB=10240.
fn varEnv(name: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_TP_ROCE_VAR_KB")) return "10240";
    return prodEnv(name);
}

/// What one rank's posts did: count, the largest, and ranges a post would have taken outside its slot or the MR.
const Seen = struct {
    posts: u64 = 0,
    max_padded: u32 = 0,
    outside: u64 = 0,
    mr_len: u64,
    slot: u64,

    fn seen(ctx: *anyopaque, l: rv.Layout, slot: u32, padded: u32, s: []const rv.Stripe) void {
        const v: *Seen = @ptrCast(@alignCast(ctx));
        v.posts += 1;
        v.max_padded = @max(v.max_padded, padded);
        // the expected layout from roce.zig's region size, not from the wire's own numbers
        if (l.len != v.mr_len or l.slot != v.slot or l.recv != l.send + 2 * v.slot or l.send + 4 * v.slot != v.mr_len) v.outside += 1;
        var total: u64 = 0;
        for (s) |x| {
            const lo_send = l.send + slot * v.slot;
            const lo_recv = l.recv + slot * v.slot;
            if (x.local < lo_send or x.local + x.len > lo_send + v.slot or x.local + x.len > v.mr_len) v.outside += 1;
            if (x.remote < lo_recv or x.remote + x.len > lo_recv + v.slot or x.remote + x.len > v.mr_len) v.outside += 1;
            if (x.flag < l.flags or x.flag + 4 > l.send) v.outside += 1;
            total += x.len;
        }
        if (total != padded) v.outside += 1;
    }
};

/// A Session's `roce` collective on host ranks: hybrid.zig's routes pick RoCE (host side) or NCCL (host collective).
const Routed = struct {
    nccl: tp.Collective,
    r: *roce.Roce,
    cfg: tp.Config,
    roce_ops: u64 = 0,
    nccl_ops: u64 = 0,
    big_roce: u64 = 0,

    fn self_(p: *anyopaque) *Routed {
        return @ptrCast(@alignCast(p));
    }
    fn rankOf(p: *anyopaque) u32 {
        return self_(p).nccl.rank();
    }
    fn worldOf(p: *anyopaque) u32 {
        return self_(p).nccl.world();
    }
    fn allReduce(p: *anyopaque, s: u64, r: u64, n: usize, t: tp.DType, op: tp.Op, st: tp.collective.Stream) tp.collective.Error!void {
        return self_(p).nccl.allReduce(s, r, n, t, op, st);
    }
    fn took(h: *Routed, n: usize) void {
        h.roce_ops += 1;
        if (n > 1 << 20) h.big_roce += 1;
    }
    fn allGather(p: *anyopaque, s: u64, r: u64, count: usize, t: tp.DType, st: tp.collective.Stream) tp.collective.Error!void {
        const h = self_(p);
        const n = count * t.size();
        if (!hybrid.smallRoute(h.cfg.mailbox_max_bytes, n, h.r.fits(n))) {
            h.nccl_ops += 1;
            return h.nccl.allGather(s, r, count, t, st);
        }
        h.took(n);
        const in: [*]const u8 = @ptrFromInt(s);
        const out: [*]u8 = @ptrFromInt(r);
        h.r.hostGather(in[0..n], out[0 .. 2 * n]) catch return error.BackendFailed;
    }
    fn varFits(p: *anyopaque, n: usize) bool {
        const h = self_(p);
        return hybrid.varRoute(h.cfg.mailbox_max_bytes, h.cfg.roce_var_bytes, n, h.r.fits(n));
    }
    fn varAgreed(p: *anyopaque) tp.collective.Error!bool {
        return varFits(p, 16);
    }
    fn allGatherV(p: *anyopaque, s: u64, r: u64, stride: usize, lens: u64, t: tp.DType, st: tp.collective.Stream) tp.collective.Error!void {
        const h = self_(p);
        const n = stride * t.size();
        if (!varFits(p, n)) return allGather(p, s, r, stride, t, st);
        h.took(n);
        const l: [*]const i32 = @ptrFromInt(lens);
        const in: [*]const u8 = @ptrFromInt(s);
        const out: [*]u8 = @ptrFromInt(r);
        h.r.hostGatherV(in[0..n], out[0 .. 2 * n], .{ @intCast(l[0]), @intCast(l[1]) }) catch return error.BackendFailed;
    }
    fn exchange(p: *anyopaque, s: u64, r: u64, n: usize, t: tp.DType, peer: u32, st: tp.collective.Stream) tp.collective.Error!void {
        return self_(p).nccl.exchange(s, r, n, t, peer, st);
    }
    fn send(p: *anyopaque, b: u64, n: usize, t: tp.DType, peer: u32, st: tp.collective.Stream) tp.collective.Error!void {
        return self_(p).nccl.send(b, n, t, peer, st);
    }
    fn recv(p: *anyopaque, b: u64, n: usize, t: tp.DType, peer: u32, st: tp.collective.Stream) tp.collective.Error!void {
        return self_(p).nccl.recv(b, n, t, peer, st);
    }
    fn barrier(p: *anyopaque) tp.collective.Error!void {
        return self_(p).nccl.barrier();
    }
    fn check(p: *anyopaque) tp.collective.Error!void {
        self_(p).r.check() catch |e| return e;
        return self_(p).nccl.check();
    }
    fn abort(p: *anyopaque) void {
        self_(p).r.abort();
        self_(p).nccl.abort();
    }

    const vtable: tp.collective.Collective.VTable = .{ .kind = .hybrid, .rank = rankOf, .world = worldOf, .all_reduce = allReduce, .all_gather = allGather, .exchange = exchange, .send = send, .recv = recv, .barrier = barrier, .check = check, .abort = abort, .all_gather_v = allGatherV, .var_fits = varFits, .var_agreed = varAgreed };

    fn iface(h: *Routed) tp.Collective {
        return .{ .ptr = h, .vtable = &vtable };
    }
};

fn value(f: usize, s: usize, t: usize, i: usize) u8 {
    return @truncate(f *% 101 +% s *% 61 +% t *% 7 +% i *% 3 +% 1);
}

const Ctx = struct {
    get: *const fn ([]const u8) ?[]const u8,
    port: u16,
    lfd: tp.sock.Fd,
    seen: [2]Seen = undefined,
    big_roce: [2]u64 = .{ 0, 0 },
    roce_ops: [2]u64 = .{ 0, 0 },
};

fn rank(comm: tp.Collective, me: u32, ctx: *Ctx) anyerror!void {
    const gpa = testing.allocator;
    const cfg = try tp.Config.parse(ctx.get);
    try testing.expectEqual(tp.config.Backend.roce, cfg.backend);
    var b = if (me == 0) try tp.Bootstrap.acceptOn(ctx.lfd, 2, 10 * std.time.ns_per_s) else try tp.Bootstrap.open(.{ .world = 2, .rank = 1, .link_port = ctx.port, .connect_timeout_ns = 10 * std.time.ns_per_s });
    defer b.close();
    const slot = std.mem.alignForward(usize, cfg.roceSlotBytes(), 4096);
    ctx.seen[me] = .{ .mr_len = roce.Roce.regionBytes(slot), .slot = slot };
    var tap: roce.PostTap = .{ .ctx = &ctx.seen[me], .seen = Seen.seen };
    const r = try roce.Roce.init(null, &b, "", .{ .wire = .shm, .shm_stripes = 2, .tap = &tap, .max_bytes = cfg.roceSlotBytes(), .timeout_ns = 20 * std.time.ns_per_s });
    defer r.deinit();
    var routed: Routed = .{ .nccl = comm, .r = r, .cfg = cfg };
    const c = routed.iface();

    var pool = try sessions.Pool.init(gpa, .{ .families = &fams, .page = page }, 64 * page, 2, me);
    defer pool.deinit();
    var host = try sessions.HostPool.init(gpa, &pool);
    defer host.deinit();
    var slots: [3]*sessions.Slot = undefined;
    for (&slots, slot_lens) |*s, n| {
        s.* = try pool.newSlot(pts * page);
        try s.*.ensure(n);
    }
    for (fams, 0..) |f, fi| for (slots, 0..) |sl, si| for (sl.mapped(), 0..) |pg, k| {
        if (!pool.owns(pg)) continue;
        const v = host.pageOf(@intCast(fi), pool.localPage(pg));
        const per = page / f.ratio;
        for (0..per) |row| for (0..rb) |i| {
            v[row * rb + i] = value(fi, si, k * per + row, i);
        };
    };
    var tables: [3 * pts]i32 = undefined;
    for (slots, 0..) |s, si| for (0..pts) |k| {
        tables[si * pts + k] = @intCast(s.localTableAt(@intCast(k)));
    };
    var hk: split.HostKernels = .{};
    const x = try split.Exchange.init(c, hk.kernels(), .on);
    try testing.expect(x.var_ok);
    var prng = std.Random.DefaultPrng.init(42); // the same selections on both ranks
    const rnd = prng.random();

    // windows of R rows (ROWS_CAP 64), both families: R 3 is 897,024 B a stride (RoCE), R 4 1,196,032 (past 1 MiB)
    for (fams, 0..) |f, fi| for ([_]u32{ 1, 2, 3, 4, 6, 16, 32, 64 }) |R| {
        const n = R * K;
        const sel = try gpa.alloc(i32, n);
        defer gpa.free(sel);
        const rslot = try gpa.alloc(i32, R);
        defer gpa.free(rslot);
        for (rslot, 0..) |*s, i| s.* = @intCast(i % 3);
        for (sel, 0..) |*t, i| {
            const rows: u32 = @intCast(slot_lens[@intCast(rslot[i / K])] / f.ratio);
            t.* = if (rnd.uintLessThan(u32, 9) == 0) -1 else @intCast(rnd.uintLessThan(u32, rows));
        }
        const send = try gpa.alloc(u8, n * rb);
        defer gpa.free(send);
        const recv = try gpa.alloc(u8, 2 * n * rb);
        defer gpa.free(recv);
        const tok = try gpa.alloc(i32, n);
        defer gpa.free(tok);
        var lens: [split.max_world]i32 = @splat(-7);
        const psh: u32 = std.math.log2_int(u32, page / f.ratio);
        const da: split.DenseArgs = .{ .sel = @intFromPtr(sel.ptr), .rows = R, .k = K, .table = @intFromPtr(&tables), .pts = pts, .rslot = @intFromPtr(rslot.ptr), .psh = psh, .base = @intFromPtr(host.tensors[fi].ptr), .row_bytes = rb, .world = 2, .send = @intFromPtr(send.ptr), .tok = @intFromPtr(tok.ptr) };
        for (0..2) |pass| {
            @memset(recv, 0xEE);
            if (pass == 0) try x.window(split.PackArgs.of(da, me, @intFromPtr(&lens)), @intFromPtr(recv.ptr), null) else try x.dense(da, @intFromPtr(recv.ptr), null);
            for (sel, tok, 0..) |t, o, i| {
                if (t < 0) continue;
                const s: usize = @intCast(rslot[i / K]);
                for (recv[@as(usize, @intCast(o)) * rb ..][0..rb], 0..) |byte, j| if (byte != value(fi, s, @intCast(t), j)) {
                    std.debug.print("fam {d} R {d} pass {d}: entry {d} wrong\n", .{ fi, R, pass, i });
                    return error.TestUnexpectedResult;
                };
            }
        }
    };

    // union blocks (allGather of M rows): the 1 MiB edge (1,795 rows = 1,048,280 B) and the 128 MiB cap's block
    const max_m: u32 = (128 << 20) / (3 * rb);
    const rows_buf = try gpa.alloc(u8, @as(usize, max_m) * rb);
    defer gpa.free(rows_buf);
    for (rows_buf, 0..) |*v, i| v.* = @truncate(i *% 13 +% me);
    const phys = try gpa.alloc(u32, max_m);
    defer gpa.free(phys);
    for (phys, 0..) |*p, i| p.* = @intCast(max_m - 1 - i);
    for ([_]u32{ 1, 1795, 1796, max_m }) |m| {
        const send = try gpa.alloc(u8, @as(usize, m) * rb);
        defer gpa.free(send);
        const recv = try gpa.alloc(u8, 2 * @as(usize, m) * rb);
        defer gpa.free(recv);
        try x.unionBlock(@intFromPtr(rows_buf.ptr), rb, @intFromPtr(phys.ptr), m, @intFromPtr(send.ptr), @intFromPtr(recv.ptr), null);
        for (0..2) |o| for (0..m) |i| {
            const want: u8 = @truncate(@as(usize, phys[i]) * rb *% 13 +% o);
            if (recv[(o * m + i) * rb] != want) return error.TestUnexpectedResult;
        };
    }
    try c.check();
    try b.agree("done", "split roce test end");
    ctx.big_roce[me] = routed.big_roce;
    ctx.roce_ops[me] = routed.roce_ops;
}

fn runAt(get: *const fn ([]const u8) ?[]const u8) !Ctx {
    const l = try tp.bootstrap.testListener();
    defer tp.sock.close(l.fd);
    var ctx: Ctx = .{ .get = get, .port = l.port, .lfd = l.fd };
    try tp.host.run(2, &ctx, rank);
    return ctx;
}

test "prod's settings: every split exchange's RoCE post fits the 1 MiB slot and its MR; past 1 MiB goes to NCCL" {
    const ctx = try runAt(prodEnv);
    for (ctx.seen, ctx.big_roce, ctx.roce_ops) |s, big, ops| {
        try testing.expectEqual(@as(u64, 1 << 20), s.slot);
        try testing.expect(ops > 0 and s.posts >= ops);
        try testing.expectEqual(@as(u64, 0), s.outside);
        try testing.expect(s.max_padded <= 1 << 20);
        try testing.expectEqual(@as(u64, 0), big);
    }
}

test "TF_TP_ROCE_VAR_KB=10240: packed strides up to 10 MiB post on RoCE (prod never does), still inside the MR" {
    const ctx = try runAt(varEnv);
    for (ctx.seen, ctx.big_roce) |s, big| {
        try testing.expectEqual(@as(u64, 10240 << 10), s.slot);
        try testing.expectEqual(@as(u64, 0), s.outside);
        try testing.expect(big > 0 and s.max_padded > 1 << 20);
    }
}

test "the RoCE post guard: the 10-10 failing shape fits; an oversize, misaligned or foreign-slot post is refused before a verb" {
    const var_slot: u64 = 10240 << 10;
    const l: rv.Layout = .{ .send = 8192, .recv = 8192 + 2 * var_slot, .flags = 4096, .slot = var_slot, .len = roce.Roce.regionBytes(var_slot) };
    var out: [rv.max_hcas]rv.Stripe = undefined;
    // fam 1, R 32: a 9,568,256 B stride; the most a rank packs is the whole stride
    try rv.stripes(l, 2, 1, 32 * K * rb, &out);
    try testing.expectEqual(@as(u64, l.send + var_slot), out[0].local);
    try testing.expectEqual(@as(u32, 32 * K * rb / 2), out[1].len);
    try testing.expectEqual(l.recv + var_slot + 32 * K * rb, out[1].remote + out[1].len);
    try testing.expectError(error.OutOfRegion, rv.stripes(l, 2, 1, @intCast(var_slot + 16), &out));
    // slot 0 past its end stays inside the MR but writes slot 1: only the slot bounds refuse it
    try testing.expectError(error.OutOfRegion, rv.stripes(l, 2, 0, @intCast(var_slot + 16), &out));
    try testing.expectError(error.OutOfRegion, rv.stripes(l, 2, 0, 24, &out));
    try testing.expectError(error.OutOfRegion, rv.stripes(l, 2, 2, 16, &out));
    try testing.expectError(error.OutOfRegion, rv.stripes(l, 3, 0, 16, &out));
    // a layout whose registered length is short of its slots: every post refused, not posted past the MR
    var short = l;
    short.len -= 16;
    try testing.expectError(error.OutOfRegion, rv.stripes(short, 2, 1, @intCast(var_slot), &out));
    // a peer whose slot or region differs is refused at connect
    const mine: rv.Record = .{ .n_hca = 1, .slot_bytes = var_slot, .region_len = l.len };
    var peer = mine;
    peer.slot_bytes = 1 << 20;
    try testing.expectError(error.BadRecord, rv.VerbsWire.pairing(mine, peer));
}
