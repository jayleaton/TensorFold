//! A tensor-parallel group over NCCL (NCCL_COMM_ID; rows in rank order); two ranks add RoCE small gathers (roce.zig).
const std = @import("std");
const nccl = @import("nccl.zig");
const driver = @import("driver.zig");
const stream_ = @import("stream.zig");
const memory = @import("memory.zig");
const roce = @import("roce.zig");
const roce_mark = @import("roce_mark.zig");
const tp_watch = @import("tp_watch.zig");

const Stream = stream_.Stream;

pub const Group = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: usize,
    world: usize,
    aborted: std.atomic.Value(bool) = .init(false),
    fast: ?*roce.Gather = null, // two ranks' small all-gathers by RoCE writes (openRoce), when both ranks brought it up
    coherent: bool = false, // opt-in (MiniMax): TF_ROCE_MARK is read and written, a failed RoCE gather stops every rank

    /// Rank `rank` of `world` (NCCL opened at run time; NCCL_COMM_ID names rank 0's address, every rank the same).
    pub fn open(rank: usize, world: usize) !Group {
        if (std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
        var lib = try nccl.Library.open();
        errdefer lib.close();
        try lib.requireSupportedVersion();
        var id: nccl.UniqueId = undefined;
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    pub fn close(g: *Group) void {
        if (g.fast) |f| f.deinit();
        g.destroyComm();
        g.lib.close();
        g.* = undefined;
    }

    fn destroyComm(g: *Group) void {
        if (!g.aborted.swap(true, .acq_rel)) _ = g.lib.api.ncclCommDestroy(g.comm);
    }

    pub fn abort(g: *Group) void {
        if (!g.aborted.swap(true, .acq_rel)) _ = g.lib.api.ncclCommAbort(g.comm);
    }

    /// Two ranks: small gathers go by RoCE if both ask (TF_ROCE; coherent: no TF_ROCE_MARK) and bring it up; else NCCL.
    pub fn openRoce(g: *Group, gpa: std.mem.Allocator, io: std.Io, d: *const driver.Driver, s: Stream, max_bytes: usize) !void {
        if (g.world != 2 or g.fast != null) return;
        g.fast = try roce.Gather.init(gpa, io, d, g, s, wantsRoce(g.coherent, io, roce.wanted(), roce_mark.path()), max_bytes);
        if (g.coherent) if (g.fast) |f| {
            f.mark = .of(io);
        };
    }

    /// What this rank asks for: TF_ROCE alone, or with `coherent` also no marker from an earlier failed run.
    fn wantsRoce(coherent: bool, io: std.Io, want: bool, mark: ?[]const u8) bool {
        return if (coherent) roce_mark.wanted(io, want, mark) else want;
    }

    /// A failed RoCE gather's error (its rows hold nothing valid): check after a sync; coherent, every rank stops.
    pub fn check(g: *const Group) !void {
        if (g.fast) |f| f.check() catch |e| {
            if (g.coherent) tp_watch.fatal("RoCE all-gather failed");
            return e;
        };
    }

    /// `bytes` at `mine` from every rank into `all` in rank order (world * bytes there), on `s`.
    pub fn allGather(g: *const Group, s: Stream, mine: u64, all: u64, bytes: usize) !void {
        if (g.fast) |f| if (f.fits(bytes)) return f.gather(s.handle, mine, all, bytes);
        try g.lib.check(g.lib.api.ncclAllGather(mine, all, bytes, .u8, g.comm, s.handle), "ncclAllGather");
    }

    /// `bytes` at `ptr` to rank `peer`, on `s` (its matching recv takes the same count).
    pub fn send(g: *const Group, s: Stream, ptr: u64, bytes: usize, peer: usize) !void {
        try g.lib.check(g.lib.api.ncclSend(ptr, bytes, .u8, @intCast(peer), g.comm, s.handle), "ncclSend");
    }

    /// `bytes` from rank `peer` into `ptr`, on `s`.
    pub fn recv(g: *const Group, s: Stream, ptr: u64, bytes: usize, peer: usize) !void {
        try g.lib.check(g.lib.api.ncclRecv(ptr, bytes, .u8, @intCast(peer), g.comm, s.handle), "ncclRecv");
    }

    /// Rank 0 and rank `peer` connect both ways with one small exchange (`ptr`: 128 bytes, 64 out and 64 in).
    pub fn connect(g: *const Group, s: Stream, ptr: u64, peer: usize) !void {
        try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
        const posted: anyerror!void = post: {
            g.send(s, ptr, 64, peer) catch |err| break :post err;
            g.recv(s, ptr + 64, 64, peer) catch |err| break :post err;
            break :post {};
        };
        try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
        try posted;
        try s.synchronize();
    }

    /// Every rank's `mine` in rank order into `out` (world of them), waited for.
    pub fn gather(g: *const Group, comptime T: type, d: *const driver.Driver, s: Stream, mine: T, out: []T) !void {
        if (out.len != g.world) return error.BadWorld;
        const size = @sizeOf(T);
        var dev = try memory.DeviceBuffer.alloc(d, size * g.world);
        defer dev.free();
        try dev.uploadAsync(g.rank * size, std.mem.asBytes(&mine), s.handle);
        try g.allGather(s, dev.ptr + g.rank * size, dev.ptr, size);
        try s.synchronize();
        try g.check();
        try dev.download(0, std.mem.sliceAsBytes(out));
    }
};

test "a group that does not opt in ignores the RoCE marker; a coherent one serves on NCCL while it exists" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(dir);
    const at = try std.fs.path.join(std.testing.allocator, &.{ dir, "roce-failed" });
    defer std.testing.allocator.free(at);
    roce_mark.write(io, at, 1, "earlier run");
    try std.testing.expect(Group.wantsRoce(false, io, true, at)); // GLM and others: TF_ROCE alone, as before
    try std.testing.expect(!Group.wantsRoce(true, io, true, at));
    try std.testing.expect(!Group.wantsRoce(false, io, false, at));
    const g: Group = .{ .lib = undefined, .comm = null, .rank = 0, .world = 2 };
    try std.testing.expect(!g.coherent);
    try g.check(); // no RoCE gather: nothing to fail
}

test "an aborted two-rank communicator is not destroyed a second time" {
    const Fake = struct {
        var aborts: usize = 0;
        var destroys: usize = 0;
        fn abort(_: nccl.Comm) callconv(.c) nccl.Result {
            aborts += 1;
            return 0;
        }
        fn destroy(_: nccl.Comm) callconv(.c) nccl.Result {
            destroys += 1;
            return 0;
        }
    };
    var group: Group = .{ .lib = undefined, .comm = null, .rank = 0, .world = 2 };
    group.lib.api.ncclCommAbort = Fake.abort;
    group.lib.api.ncclCommDestroy = Fake.destroy;
    group.abort();
    group.abort();
    group.destroyComm();
    try std.testing.expectEqual(@as(usize, 1), Fake.aborts);
    try std.testing.expectEqual(@as(usize, 0), Fake.destroys);
}
