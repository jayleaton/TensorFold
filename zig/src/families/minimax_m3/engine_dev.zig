//! The engine on the device: NCCL collectives, page arena, block table, control wire, `Engine.init` (never run).
const std = @import("std");
const cuda = @import("cuda");
const iface = @import("iface.zig");
const ops = @import("engine_ops.zig");
const engine = @import("engine.zig");
const follow = @import("follow.zig");
const draw = @import("draw.zig");
const ranks = @import("ranks.zig");
const config = @import("config.zig");
const shard = @import("shard.zig");
const pool = @import("pool.zig");
const load = @import("load.zig");
const weights_device = @import("weights_device.zig");
const dc = @import("dspark_config.zig");
const dspark_device = @import("dspark_device.zig");
const forward_ops = @import("forward_ops.zig");

/// GLM's NCCL defaults, set when the environment leaves them unset: connect at init, no cuMem host buffers.
pub const nccl_defaults = [_][2][:0]const u8{
    .{ "NCCL_RUNTIME_CONNECT", "0" }, // every connection is registered at init, while memory is free
    .{ "NCCL_CUMEM_HOST_ENABLE", "0" }, // cuMem host buffers wore out GB10's driver pool under load
    .{ "NCCL_CUMEM_ENABLE", "0" },
};

/// Sets each default that `get` finds unset (setenv with overwrite 0, as GLM does).
pub fn applyNcclDefaults(get: *const fn ([*:0]const u8) ?[*:0]const u8, set: *const fn ([*:0]const u8, [*:0]const u8) void) void {
    for (nccl_defaults) |kv| if (get(kv[0].ptr) == null) set(kv[0].ptr, kv[1].ptr);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

fn getenvC(name: [*:0]const u8) ?[*:0]const u8 {
    return std.c.getenv(name);
}

fn setenvC(name: [*:0]const u8, value: [*:0]const u8) void {
    _ = setenv(name, value, 0);
}

pub fn setNcclDefaults() void {
    applyNcclDefaults(getenvC, setenvC);
}

/// Every peer's send/recv connected now, before the weights fill memory (connected on first use, ibv_reg_mr can fail).
pub fn warmPeers(d: *const cuda.Driver, g: *const cuda.tp_group.Group) !void {
    const bytes = 4 << 20; // a prompt piece's exchange: every channel a large send/recv takes
    var buf = try cuda.DeviceBuffer.alloc(d, 2 * bytes * g.world);
    defer buf.free();
    var s = try cuda.Stream.init(d, false);
    defer s.deinit();
    try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
    const posted: anyerror!void = post: {
        for (0..g.world) |peer| if (peer != g.rank) {
            g.send(s, buf.ptr + peer * bytes, bytes, peer) catch |err| break :post err;
            g.recv(s, buf.ptr + (g.world + peer) * bytes, bytes, peer) catch |err| break :post err;
        };
        break :post {};
    };
    try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
    try posted;
    try s.synchronize();
}

/// The collectives over NCCL on the engine's stream.
pub const NcclComm = struct {
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    rank: usize,
    world: usize,
    one: cuda.DeviceBuffer, // one f64 every rank all-reduces to its maximum
    group: *const cuda.tp_group.Group,

    pub fn interface(c: *NcclComm) ops.Comm {
        return .{ .ptr = c, .vtable = &.{ .max_f64 = maxF64, .gather_bytes = gatherBytes, .check = checkFn }, .rank = c.rank, .world = c.world };
    }

    fn of(ptr: *anyopaque) *NcclComm {
        return @ptrCast(@alignCast(ptr));
    }

    fn maxF64(ptr: *anyopaque, v: f64) anyerror!f64 {
        const c = of(ptr);
        if (c.world == 1) return v;
        var x = v;
        try c.one.uploadAsync(0, std.mem.asBytes(&x), c.stream.handle);
        try c.lib.check(c.lib.api.ncclAllReduce(c.one.ptr, c.one.ptr, 1, .f64, .max, c.comm, c.stream.handle), "ncclAllReduce");
        try c.stream.synchronize();
        try c.one.download(0, std.mem.asBytes(&x));
        return x;
    }

    /// Every rank's `mine` in rank order, over one ncclAllGather of bytes (glm/cuda_ranks.zig's `gather`).
    fn gatherBytes(ptr: *anyopaque, mine: []const u8, out: []u8) anyerror!void {
        const c = of(ptr);
        if (out.len != mine.len * c.world) return error.BadGather;
        if (c.world == 1) return @memcpy(out, mine);
        var dev = try cuda.DeviceBuffer.alloc(c.d, out.len);
        defer dev.free();
        try dev.uploadAsync(c.rank * mine.len, mine, c.stream.handle);
        try c.lib.check(c.lib.api.ncclAllGather(dev.ptr + c.rank * mine.len, dev.ptr, mine.len, .u8, c.comm, c.stream.handle), "ncclAllGather");
        try c.stream.synchronize();
        try dev.download(0, out);
    }

    fn checkFn(ptr: *anyopaque) anyerror!void {
        try of(ptr).group.check();
    }
};

/// The fingerprint through the byte gather, for `ranks.agree`.
pub fn gatherFingerprint(c: ops.Comm, mine: ranks.Fingerprint, out: []ranks.Fingerprint) anyerror!void {
    try c.gatherBytes(std.mem.asBytes(&mine), std.mem.sliceAsBytes(out));
}

/// The ranks agree on what they run before any weight loads; any difference is error.RankMismatch on every rank.
pub fn agreeFingerprints(gpa: std.mem.Allocator, c: ops.Comm, mine: ranks.Fingerprint) !void {
    const all = try gpa.alloc(ranks.Fingerprint, c.world);
    defer gpa.free(all);
    try ranks.agree(c, gatherFingerprint, mine, all);
}

/// The page arena and the int32 block table on the device (iface decision M3).
pub const DevicePool = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    arena: ?cuda.DeviceBuffer = null,
    table: ?cuda.DeviceBuffer = null,
    max_pages: usize = 0,

    pub fn interface(p: *DevicePool) ops.Device {
        return .{ .ptr = p, .vtable = &.{ .attach = attach, .upload = upload, .deinit = deinitFn } };
    }

    fn of(ptr: *anyopaque) *DevicePool {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn release(p: *DevicePool) void {
        p.stream.synchronize() catch {};
        if (p.arena) |*a| a.free();
        if (p.table) |*t| t.free();
        p.arena = null;
        p.table = null;
    }

    fn attach(ptr: *anyopaque, slots: u32, pages: usize, max_pages: usize, page_bytes: u64) anyerror!ops.PoolView {
        const p = of(ptr);
        p.release();
        var arena = try cuda.DeviceBuffer.alloc(p.d, @max(1, pages * page_bytes));
        errdefer arena.free();
        var table = try cuda.DeviceBuffer.alloc(p.d, @max(1, slots * max_pages * iface.decisions.table_entry_bytes));
        errdefer table.free();
        try table.fill32(0xFFFF_FFFF, p.stream.handle); // every entry -1
        p.arena = arena;
        p.table = table;
        p.max_pages = max_pages;
        return .{ .arena = arena.ptr, .page_bytes = page_bytes, .pages = pages, .table = table.ptr, .slots = slots, .max_pages = max_pages };
    }

    fn upload(ptr: *anyopaque, slot: u32, from: usize, entries: []const i32) anyerror!void {
        const p = of(ptr);
        const t = p.table orelse return error.NoPool;
        const at = (@as(usize, slot) * p.max_pages + from) * iface.decisions.table_entry_bytes;
        try t.uploadAsync(at, std.mem.sliceAsBytes(entries), p.stream.handle);
    }

    fn deinitFn(ptr: *anyopaque) void {
        of(ptr).release();
    }
};

/// The control link over ncclBroadcast on the engine's stream: a 64-byte header, then its ids.
pub const NcclWire = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    buf: cuda.DeviceBuffer,

    pub fn init(d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, ids: usize) !NcclWire {
        return .{ .lib = lib, .comm = comm, .stream = stream, .buf = try cuda.DeviceBuffer.alloc(d, 64 + ids * 4) };
    }

    pub fn deinit(w: *NcclWire) void {
        w.buf.free();
    }

    pub fn wire(w: *NcclWire) follow.Wire {
        return .{ .ptr = w, .vtable = &.{ .send = send, .receive = receive } };
    }

    fn post(w: *NcclWire, offset: usize, bytes: usize) !void {
        try w.lib.check(w.lib.api.ncclBroadcast(w.buf.ptr + offset, w.buf.ptr + offset, bytes, .u8, 0, w.comm, w.stream.handle), "ncclBroadcast");
    }

    /// Rank 0: enqueued on the stream and not waited for; the stream orders the next step's upload after it.
    fn send(ptr: *anyopaque, h: *const iface.Header, ids: []const u32) anyerror!void {
        const w: *NcclWire = @ptrCast(@alignCast(ptr));
        if (64 + ids.len * 4 > w.buf.len) return error.TooManyIds;
        try w.buf.uploadAsync(0, std.mem.asBytes(h), w.stream.handle);
        try w.post(0, 64);
        if (ids.len == 0) return;
        try w.buf.uploadAsync(64, std.mem.sliceAsBytes(ids), w.stream.handle);
        try w.post(64, ids.len * 4);
    }

    fn receive(ptr: *anyopaque, out: []u32) anyerror!iface.Header {
        const w: *NcclWire = @ptrCast(@alignCast(ptr));
        try w.post(0, 64);
        try w.stream.synchronize();
        var h: iface.Header = undefined;
        try w.buf.download(0, std.mem.asBytes(&h));
        if (h.n > out.len or 64 + @as(usize, h.n) * 4 > w.buf.len) return error.TooManyIds;
        if (h.n > 0) {
            try w.post(64, @as(usize, h.n) * 4);
            try w.stream.synchronize();
            try w.buf.download(64, std.mem.sliceAsBytes(out[0..h.n]));
        }
        return h;
    }
};

/// The device-to-host copy of a pass's gathered logits for rank 0's draw: the stream drains first.
pub const Download = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,

    pub fn fetch(x: *Download) draw.Fetch {
        return .{ .ptr = x, .call = call };
    }

    fn call(ptr: *anyopaque, addr: u64, out: []u8) anyerror!void {
        const x: *Download = @ptrCast(@alignCast(ptr));
        try x.stream.synchronize();
        try x.d.check(x.d.api.cuMemcpyDtoH_v2(out.ptr, addr, out.len), "cuMemcpyDtoH");
    }
};

/// A round's grammar words in host memory, where rank 0's draw reads them (the address is the `bits` of a window).
pub const Bits = struct {
    gpa: std.mem.Allocator,
    words: usize, // u32 words of one row
    buf: []u32 = &.{},

    pub fn interface(b: *Bits) @import("backend.zig").Masks {
        return .{ .ptr = b, .put = put };
    }

    pub fn deinit(b: *Bits) void {
        b.gpa.free(b.buf);
    }

    fn put(ptr: *anyopaque, row: usize, rows: usize, words: []const u32) anyerror!u64 {
        const b: *Bits = @ptrCast(@alignCast(ptr));
        if (words.len != rows * b.words or row + rows > iface.shared_rows) return error.InvalidGrammarMasks;
        if (b.buf.len == 0) b.buf = try b.gpa.alloc(u32, iface.shared_rows * b.words);
        @memcpy(b.buf[row * b.words ..][0..words.len], words);
        return @intFromPtr(b.buf[row * b.words ..].ptr);
    }
};

pub fn hashFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![32]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    return ranks.hashBytes(bytes);
}

/// The liveness lines (cuda/tp_watch.zig) as the follower and the leader use them; `io` is the process's.
pub fn hooksWith(io: *const std.Io) follow.Hooks {
    return .{ .ctx = @constCast(io), .live = hookLive, .sleep = hookSleep, .wake = hookWake, .failed = hookFailed, .ended = hookEnded };
}

fn hookLive(_: *anyopaque) bool {
    return cuda.tp_watch.open();
}

fn hookSleep(ctx: *anyopaque) void {
    const io: *const std.Io = @ptrCast(@alignCast(ctx));
    cuda.tp_watch.sleep(io.*);
}

fn hookWake(_: *anyopaque) void {
    cuda.tp_watch.wake();
}

fn hookFailed(_: *anyopaque, name: []const u8) void {
    cuda.tp_watch.fail(name);
}

fn hookEnded(_: *anyopaque) void {
    cuda.tp_watch.ended();
}

/// What an engine owns on the device; `close` frees it in the order it was made.
const Owner = struct {
    gpa: std.mem.Allocator,
    stream: cuda.Stream,
    comm: NcclComm,
    pool: DevicePool,
    loaded: weights_device.Loaded,
    cfg: config.Config,
    group: cuda.tp_group.Group, // what the forward holds; the host copies its RoCE small gather in after open
    forward: ops.Ops,
    download: Download,
    drawer: *draw.HostDrawer,
    draft: ?*dspark_device.Device = null,

    fn close(ptr: *anyopaque) void {
        const o: *Owner = @ptrCast(@alignCast(ptr));
        o.stream.synchronize() catch {};
        o.drawer.deinit();
        if (o.draft) |dd| dd.interface().vtable.deinit(dd);
        o.forward.vtable.deinit(o.forward.ptr);
        o.pool.release();
        o.comm.one.free();
        o.loaded.deinit();
        o.stream.deinit();
        o.gpa.destroy(o);
    }
};

/// sha256 over the draft folder's config.json and its tensors' names, shards, shapes and byte ranges (no data read).
pub fn draftDigest(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![32]u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    var cat = try load.Catalog.open(gpa, io, dir);
    defer cat.deinit();
    const weights_sum = try weights_device.digest(gpa, &cat);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(&ranks.hashBytes(text));
    h.update(&weights_sum);
    return h.finalResult();
}

/// Opens rank `rank` of `world` over `comm`: the ranks agree on what they run, then the weights and forward load.
pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, checkpoint: []const u8, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, rank: usize, world: usize, o: iface.EngineOptions) !*engine.Engine {
    if (world > ranks.max_world) return error.TooManyRanks;
    if (rank >= world) return error.NoRank;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ checkpoint, "config.json" }), a, .limited(1 << 24));
    const quant_path = try std.fs.path.join(a, &.{ checkpoint, "quantization_config.json" });
    const quant: [32]u8 = hashFile(a, io, quant_path) catch ranks.hashBytes("");
    const c = try config.parse(gpa, text);
    const share = try shard.Share.of(&c, rank, world);
    const dims = (try pool.Dims.of(&c, share)).withKv8(o.settings.kv8);
    var cat = try load.Catalog.open(gpa, io, checkpoint);
    defer cat.deinit();
    const digest = try weights_device.digest(gpa, &cat);
    const own = try gpa.create(Owner);
    errdefer gpa.destroy(own);
    var stream = try cuda.Stream.init(d, false);
    errdefer stream.deinit();
    own.* = .{ .gpa = gpa, .stream = stream, .cfg = c, .comm = undefined, .pool = .{ .d = d, .stream = stream }, .loaded = undefined, .group = .{ .lib = lib.*, .comm = comm, .rank = rank, .world = world }, .forward = undefined, .download = .{ .d = d, .stream = stream }, .drawer = undefined };
    var one = try cuda.DeviceBuffer.alloc(d, 8);
    errdefer one.free();
    own.comm = .{ .d = d, .lib = lib, .comm = comm, .stream = stream, .rank = rank, .world = world, .one = one, .group = &own.group };
    var fp = ranks.Fingerprint.of(.{ .rank = rank, .world = world }, o.settings, o.capacity, ranks.hashBytes(text), quant, try hashFile(a, io, "/proc/self/exe"), digest);
    if (o.drafter) |dir| fp.draft = try draftDigest(gpa, io, dir); // the ranks agree on the draft too, or the first absorb would not find it
    try agreeFingerprints(gpa, own.comm.interface(), fp);
    if (!ops.available()) return error.ForwardNotBuilt; // the weights are not read for a build that cannot run them
    own.loaded = try weights_device.loadWeights(gpa, io, d, stream, checkpoint, &own.cfg, share, o.settings.layers);
    errdefer own.loaded.deinit();
    own.forward = try ops.openOps(.{ .gpa = gpa, .io = io, .d = d, .ctx = ctx, .stream = stream, .cfg = &own.cfg, .share = share, .weights = &own.loaded.weights, .settings = o.settings, .place = .{ .rank = rank, .world = world }, .capacity = o.capacity, .lib = lib, .comm = comm, .group = &own.group });
    errdefer own.forward.vtable.deinit(own.forward.ptr);
    if (o.drafter) |dir| { // the dspark draft: its layers are tapped by every forward from now on, its weights go to this rank
        const dtext = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 24));
        const dcfg = try dc.parse(gpa, dtext);
        if (dc.admit(&dcfg, &c, .{ .world = world, .window_rows = iface.window_rows })) |why| {
            std.log.err("{s}", .{why.text()});
            return error.DrafterRefused;
        }
        try own.forward.setTaps(dcfg.tapIds());
        const impl: *forward_ops.Impl = @ptrCast(@alignCast(own.forward.ptr)); // openOps builds exactly this
        own.draft = try dspark_device.Device.open(gpa, io, d, ctx, stream, dir, dcfg, share, &impl.m, &own.loaded.weights, &impl.g, impl.link);
    }
    errdefer if (own.draft) |dd| dd.interface().vtable.deinit(dd);
    const spans = try a.alloc(shard.Span, world);
    for (spans, 0..) |*sp, q| sp.* = (try shard.Share.of(&c, q, world)).vocab;
    own.drawer = try draw.HostDrawer.init(gpa, rank, spans, c.vocab, own.download.fetch());
    errdefer own.drawer.deinit();
    const e = try engine.Engine.make(gpa, io, .{
        .ops = own.forward,
        .device = own.pool.interface(),
        .comm = own.comm.interface(),
        .drawer = own.drawer.drawer(),
        .capacity = o.capacity,
        .page_bytes = dims.pageBytes(),
        .checkpoint_digest = digest,
        .stream = stream,
        .group = &own.group,
        .close = .{ .ptr = own, .call = Owner.close },
        .draft = if (own.draft) |dd| dd.interface() else null,
    });
    try stream.synchronize();
    return e;
}

test "NCCL defaults are set only where the environment leaves them unset" {
    const Env = struct {
        var set_names: [3]?[:0]const u8 = .{ null, null, null };
        var n: usize = 0;
        fn get(name: [*:0]const u8) ?[*:0]const u8 {
            if (std.mem.eql(u8, std.mem.span(name), "NCCL_CUMEM_ENABLE")) return "1"; // the operator chose
            return null;
        }
        fn set(name: [*:0]const u8, value: [*:0]const u8) void {
            std.testing.expectEqualStrings("0", std.mem.span(value)) catch unreachable;
            set_names[n] = std.mem.span(name);
            n += 1;
        }
    };
    applyNcclDefaults(Env.get, Env.set);
    try std.testing.expectEqual(@as(usize, 2), Env.n);
    try std.testing.expectEqualStrings("NCCL_RUNTIME_CONNECT", Env.set_names[0].?);
    try std.testing.expectEqualStrings("NCCL_CUMEM_HOST_ENABLE", Env.set_names[1].?);
}
