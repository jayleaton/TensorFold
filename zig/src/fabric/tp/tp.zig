//! Tensor parallelism across processes (two DGX Sparks, one GPU each): bootstrap, collectives, fail-fast; `Session.open` brings a rank up.
const std = @import("std");
const cuda = @import("cuda");

pub const collective = @import("collective.zig");
pub const reference = @import("reference.zig");
pub const config = @import("config.zig");
pub const sock = @import("sock.zig");
pub const bootstrap = @import("bootstrap.zig");
pub const fate = @import("fate.zig");
pub const host = @import("host.zig");
pub const nccl = @import("nccl.zig");
pub const mailbox = @import("mailbox.zig");
pub const hybrid = @import("hybrid.zig");

pub const Collective = collective.Collective;
pub const DType = collective.DType;
pub const Op = collective.Op;
pub const Config = config.Config;
pub const Bootstrap = bootstrap.Bootstrap;
pub const Fate = fate.Fate;

pub const Session = struct {
    cfg: Config,
    lib: cuda.nccl.Library,
    nccl: nccl.Nccl,
    mail: ?mailbox.Mailbox = null,
    route: hybrid.Hybrid = undefined,
    fate: ?Fate = null,

    /// Brings this rank up (in place: the session's address must not change afterwards). `mailbox_image` is
    /// mailbox.cu's fatbin, empty when the build has none (then the mailbox backend falls back to NCCL).
    pub fn open(s: *Session, d: *const cuda.Driver, cfg: Config, mailbox_image: []const u8) !void {
        var b = try Bootstrap.open(cfg);
        defer b.close();
        try b.agree(&cfg.digest(), "settings (TF_TP_WORLD, TF_COMM_BACKEND, TF_TP_MAILBOX_MAX_KB, TF_DSV41_FAILFAST)");
        var lib = try cuda.nccl.Library.openPath(cfg.nccl_lib);
        errdefer lib.close();
        const v = try lib.version();
        try b.agree(std.mem.asBytes(&v), "NCCL version");
        var id: cuda.nccl.UniqueId = undefined;
        if (cfg.rank == 0) try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        try b.broadcast(&id.internal, &id.internal);
        s.* = .{ .cfg = cfg, .lib = lib, .nccl = undefined };
        s.nccl = try nccl.Nccl.init(&s.lib, d, id, cfg.rank, cfg.world);
        errdefer s.nccl.deinit();
        if (cfg.backend == .mailbox) {
            if (try mailboxPossible(&b, mailbox_image)) {
                s.mail = try mailbox.Mailbox.init(d, &b, mailbox_image, cfg.mailbox_max_bytes, cfg.mailbox_timeout_ns);
                s.route = .{ .mail = &s.mail.?, .nccl = &s.nccl, .max_bytes = cfg.mailbox_max_bytes };
            } else if (cfg.rank == 0) {
                std.log.warn("[tensorfold] tp: TF_COMM_BACKEND=mailbox needs two ranks on one host and the kernel; every rank uses NCCL", .{});
            }
        }
        if (cfg.failfast) {
            s.fate = Fate.init(cfg, b.release());
            s.fate.?.onFail(.{ .ctx = s, .call = abortHook });
            try s.fate.?.start();
        }
        std.log.info("[tensorfold] tp: rank {d} of {d} up ({t}, NCCL {d}, fail-fast {s})", .{ cfg.rank, cfg.world, s.comm().kind(), v, if (cfg.failfast) "on" else "off" });
    }

    fn mailboxPossible(b: *Bootstrap, image: []const u8) !bool {
        if (b.world != 2) return false;
        var id_buf: [64]u8 = undefined;
        const id = sock.hostId(&id_buf);
        var mine: [65]u8 = @splat(0);
        mine[0] = @intFromBool(image.len > 0);
        @memcpy(mine[1..][0..id.len], id);
        var blobs: [bootstrap.max_ranks][]const u8 = undefined;
        var buf: [2 * 65]u8 = undefined;
        const all = try b.allGather(&mine, &buf, &blobs);
        return all[0][0] == 1 and std.mem.eql(u8, all[0], all[1]);
    }

    fn abortHook(ctx: *anyopaque, _: []const u8) void {
        const s: *Session = @ptrCast(@alignCast(ctx));
        s.comm().abort();
    }

    /// The communicator the model uses.
    pub fn comm(s: *Session) Collective {
        if (s.mail != null) return s.route.iface();
        return s.nccl.iface();
    }

    /// This rank failed inside a round: every rank goes down (exit 70). Without fail-fast it only aborts.
    pub fn fatal(s: *Session, where: []const u8, err: anyerror) void {
        if (s.fate) |*f| return f.fatal(where, err);
        s.comm().abort();
    }

    /// A normal stop: every rank calls it (rank 0 first marks the stop as expected with `fate.expectStop`).
    pub fn close(s: *Session) void {
        if (s.fate) |*f| f.close();
        if (s.mail) |*m| m.deinit();
        s.nccl.deinit();
        s.lib.close();
    }
};

test {
    _ = collective;
    _ = reference;
    _ = config;
    _ = sock;
    _ = bootstrap;
    _ = mailbox;
    _ = @import("host_test.zig");
    _ = @import("fate_test.zig");
}
