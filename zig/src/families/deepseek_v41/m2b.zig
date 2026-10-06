//! `tf-dsv41-m1 m2b PACK REF OUT` (one process a rank: TF_TP_RANK, TF_TP_WORLD=2, TF_TP_DEVICE, TF_TP_PORT): M2b's
//! gate. Our forward at real TP=2 (the tp runtime's NCCL session, both ranks' exchanges on the wire) over the
//! reference's layers and the head, from the Python engine's slot state after its prompt, decoding greedily one token
//! a window; against tools/zig/dsv41_m2b_ref.py's run: every token, each step's logits digest on this rank, and the
//! first window's per-block streams. Greedy = Python's forward.greedy: each rank's max (lowest column on ties), the
//! (value, id) pairs all-gathered as float64, the higher value (the lower rank on ties).

const std = @import("std");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const tp = @import("tp");
const Config = @import("config.zig").Config;
const Pack = @import("pack.zig").Pack;
const plan = @import("plan.zig");
const named = @import("named.zig");
const load = @import("load.zig");
const buffers = @import("buffers.zig");
const forms = @import("forms.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const m2a = @import("m2a.zig");
const Value = std.json.Value;

fn sha(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

pub fn main(gpa: std.mem.Allocator, io: std.Io, pack_dir: []const u8, ref: []const u8, out_dir: []const u8) !u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const tcfg = try tp.Config.fromEnv();
    const rank = tcfg.rank;
    try cwd.createDirPath(io, out_dir);
    const meta = try std.json.parseFromSliceLeaky(Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ ref, "ref.json" }), a, .limited(1 << 28)), .{});
    var lo: u32 = 0;
    var hi: u32 = 0;
    {
        var it = std.mem.tokenizeScalar(u8, meta.object.get("layers").?.string, '-');
        lo = try std.fmt.parseInt(u32, it.next().?, 10);
        hi = try std.fmt.parseInt(u32, it.next().?, 10);
    }
    const layers = try a.alloc(u32, hi - lo + 1);
    for (layers, 0..) |*l, i| l.* = lo + @as(u32, @intCast(i));

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, @intCast(tcfg.device));
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var kernels = try dk.Kernels.load(&ctx);
    defer kernels.deinit();
    const session = try gpa.create(tp.Session);
    defer gpa.destroy(session);
    // TF_TP_KERNEL_IMAGE: tp_mailbox.fatbin (the mailbox / RoCE kernels): TF_COMM_BACKEND=roce then takes the small
    // exchanges over RoCE (the hybrid), NCCL the large ones; unset: NCCL alone
    const image: []const u8 = if (std.c.getenv("TF_TP_KERNEL_IMAGE")) |p|
        try cwd.readFileAllocOptions(io, std.mem.span(p), a, .limited(1 << 26), .@"16", null)
    else
        &.{};
    try session.open(&driver, tcfg, image);
    defer session.close();
    const comm = session.comm();

    const t0 = std.Io.Clock.awake.now(io);
    const cfg = try Config.read(gpa, io, pack_dir);
    var pack = try Pack.open(gpa, io, pack_dir);
    defer pack.deinit();
    var pl = try plan.build(gpa, &cfg, &pack, .{ .rank = rank, .world = 2, .blocks = layers, .top = true });
    defer pl.deinit();
    var w: load.Weights = .{ .gpa = gpa };
    defer w.deinit();
    for (pl.layers) |*l| {
        var b: named.Builder = .{ .gpa = gpa, .io = io, .pack = &pack };
        defer b.deinit();
        try b.layer(l, cfg.o_groups, rank, 2);
        try w.upload(&driver, &b);
    }
    {
        var b: named.Builder = .{ .gpa = gpa, .io = io, .pack = &pack };
        defer b.deinit();
        try b.top(&pl);
        try w.upload(&driver, &b);
    }
    var aot = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, try std.fs.path.join(a, &.{ ref, "aot" }));
    defer aot.deinit();
    const wd = try m2a.widths(a, &w, &pl);
    var runner: run.Runner = .{ .gpa = gpa, .d = &driver, .stream = stream, .kernels = &kernels, .triton = &aot, .weights = &w };
    defer runner.deinit();
    // M2B_TOPP (the reference's TF_DSV41_EXPERT_TOPP): unset = top-p off (no prune; the router's tail folds kit
    // rounding and grouping), else prod's prune at that mass
    const topp: ?f64 = if (std.c.getenv("M2B_TOPP")) |v| try std.fmt.parseFloat(f64, std.mem.span(v)) else null;
    var f = fwd.Forward.init(gpa, &cfg, &wd, .{ .expert_topp = topp }, &runner, comm);
    defer f.deinit();
    f.layers = layers;
    var bp = blk: {
        // the plan at both trace settings (the traced program adds one role)
        f.opts.trace = true;
        var p = try f.plan(&.{1});
        f.programs.clearRetainingCapacity();
        f.opts.trace = false;
        try p.add(try f.program(1));
        break :blk p;
    };
    try runner.bind(&bp);
    const src: forms.Source = .{ .d = &driver, .weights = &w, .plan = &pl };
    for (bp.sizes.keys()) |role| {
        if (buffers.scopeOf(role) != .persistent) continue;
        var ra = std.heap.ArenaAllocator.init(gpa);
        defer ra.deinit();
        const bytes = (try forms.build(ra.allocator(), &src, role)) orelse continue;
        try cuda.DeviceBuffer.upload(.{ .d = &driver, .ptr = runner.addressOf(role).?, .len = bytes.len }, 0, bytes);
    }
    const rope = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ ref, "rope.bin" }), a, .limited(1 << 28));
    const half = (rope.len - 16) / 2;
    for ([_][]const u8{ "s.rope.main", "s.rope.comp" }, 0..) |role, i| if (runner.addressOf(role)) |p|
        try cuda.DeviceBuffer.upload(.{ .d = &driver, .ptr = p, .len = half }, 0, rope[16 + i * half ..][0..half]);
    const ehb = try cwd.readFileAllocOptions(io, try std.fs.path.join(a, &.{ ref, "engram-host.bin" }), a, .limited(1 << 26), .@"4", null);
    var lbuf: [eh.max_layers]u32 = undefined;
    const host = try eh.Host.load(ehb, cfg.engram_heads, cfg.engram_pad, cfg.engram_vocab, &lbuf);
    f.engram = &host;
    // this rank's slot after the reference's prompt
    const sdir = try std.fmt.allocPrint(a, "{s}/rank{d}/state", .{ ref, rank });
    const st = try std.json.parseFromSliceLeaky(Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ sdir, "state.json" }), a, .limited(1 << 24)), .{});
    var sit = st.object.get("roles").?.object.iterator();
    while (sit.next()) |e| {
        const bytes = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ sdir, e.value_ptr.string }), a, .limited(1 << 32));
        if (bp.sizes.get(e.key_ptr.*) == null or bp.sizes.get(e.key_ptr.*).? < bytes.len) return error.StateRole;
        try cuda.DeviceBuffer.upload(.{ .d = &driver, .ptr = runner.addressOf(e.key_ptr.*).?, .len = bytes.len }, 0, bytes);
    }
    f.slot.pos = @intCast(st.object.get("pos").?.integer);
    const tail = st.object.get("tail").?.array.items;
    for (tail, 0..) |t, i| f.slot.tail[i] = @intCast(t.integer);
    f.slot.tail_len = tail.len;
    try driver.check(driver.api.cuCtxSynchronize(), "cuCtxSynchronize");
    try comm.barrier();
    const load_s = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
    std.debug.print("rank {d}: {d} layers, {d:.2} GB, load {d:.1} s, position {d}\n", .{ rank, layers.len, @as(f64, @floatFromInt(w.bytes)) / 1e9, load_s, f.slot.pos });

    // greedy decoding against the reference's tokens
    const want_tokens = meta.object.get("tokens").?.array.items;
    const mine = meta.object.get("ranks").?.array.items[rank].object;
    const want_logits = mine.get("logits_sha256").?.array.items;
    const V: usize = cfg.vocab / 2;
    const host_logits = try a.alloc(u8, 4 * V);
    var pair = try cuda.DeviceBuffer.alloc(&driver, 64);
    defer pair.free();
    var trace: std.AutoArrayHashMapUnmanaged(u32, [64]u8) = .empty;
    defer trace.deinit(gpa);
    var tok: u32 = @intCast(want_tokens[0].integer);
    var log: std.Io.Writer.Allocating = .init(gpa);
    defer log.deinit();
    var tokens_equal: usize = 0;
    var logits_equal: usize = 0;
    var layers_equal: usize = 0;
    var first_layer: ?u32 = null;
    const steps = want_tokens.len - 1;
    const tw0 = std.Io.Clock.awake.now(io);
    for (0..steps) |step| {
        f.opts.trace = step == 0;
        f.trace = if (step == 0) &trace else null;
        const ids = [_]u32{tok};
        try f.window(&ids);
        try stream.synchronize();
        if (step == 0) {
            var it = mine.get("layers_first_window").?.object.iterator();
            while (it.next()) |e| {
                const L = try std.fmt.parseInt(u32, e.key_ptr.*, 10);
                const got = trace.get(L) orelse return error.TraceMissing;
                if (std.mem.eql(u8, &got, e.value_ptr.string)) layers_equal += 1 else if (first_layer == null) {
                    first_layer = L;
                }
            }
        }
        try cuda.DeviceBuffer.download(.{ .d = &driver, .ptr = runner.addressOf("w.logits").?, .len = host_logits.len }, 0, host_logits);
        const ld = sha(host_logits);
        const lok = std.mem.eql(u8, &ld, want_logits[step].string);
        logits_equal += @intFromBool(lok);
        // this rank's max, the lowest column on ties (pick.argmax); -0.0 == +0.0
        const vals = std.mem.bytesAsSlice(f32, host_logits);
        var best: usize = 0;
        for (vals, 0..) |x, i| if (x > vals[best]) {
            best = i;
        };
        var p: [2]f64 = .{ vals[best], @floatFromInt(best + rank * V) };
        try cuda.DeviceBuffer.upload(pair, 0, std.mem.asBytes(&p));
        try comm.allGather(pair.ptr, pair.ptr + 16, 2, .f64, stream.handle);
        try stream.synchronize();
        var both: [4]f64 = undefined;
        try cuda.DeviceBuffer.download(pair, 16, std.mem.asBytes(&both));
        const r: usize = if (both[2] > both[0]) 1 else 0; // the first rank on ties
        tok = @intFromFloat(both[2 * r + 1]);
        const want: u32 = @intCast(want_tokens[step + 1].integer);
        const tok_ok = tok == want;
        tokens_equal += @intFromBool(tok_ok);
        try log.writer.print("{{\"step\": {d}, \"token\": {d}, \"want\": {d}, \"logits_equal\": {}}}\n", .{ step, tok, want, lok });
        try f.keep(0);
        if (!tok_ok) break; // the sequences have diverged: later steps compare nothing
    }
    const dec_s = @as(f64, @floatFromInt(tw0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
    f.trace = null;
    const nl = mine.get("layers_first_window").?.object.count();
    const ok = tokens_equal == steps and logits_equal == steps and layers_equal == nl;
    try log.writer.print("{{\"rank\": {d}, \"steps\": {d}, \"tokens_equal\": {d}, \"logits_equal\": {d}, \"layers_equal\": {d}, \"layers\": {d}, \"first_differing_layer\": {?d}, \"load_s\": {d:.1}, \"decode_s\": {d:.1}, \"pass\": {}}}\n", .{ rank, steps, tokens_equal, logits_equal, layers_equal, nl, first_layer, load_s, dec_s, ok });
    try cwd.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/m2b-rank{d}.jsonl", .{ out_dir, rank }), .data = log.written() });
    std.debug.print("{s} M2b rank {d}: tokens {d}/{d}, logits {d}/{d}, first window blocks {d}/{d} (first differing {?d}), decode {d:.1} s\n", .{ if (ok) "PASS" else "FAIL", rank, tokens_equal, steps, logits_equal, steps, layers_equal, nl, first_layer, dec_s });
    return if (ok) 0 else 1;
}
