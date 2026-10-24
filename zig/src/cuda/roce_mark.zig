//! The RoCE failure marker (TF_ROCE_MARK): a failed RoCE gather writes it; while it exists, this rank asks for NCCL.
const std = @import("std");

pub const var_name = "TF_ROCE_MARK";

/// TF_ROCE_MARK's path on this host's persistent volume; unset or empty: no marker is read or written.
pub fn path() ?[]const u8 {
    const v = std.mem.span(std.c.getenv(var_name) orelse return null);
    return if (v.len == 0) null else v;
}

/// Whether the marker at `p` exists (an earlier run's RoCE path failed on this host).
pub fn present(io: std.Io, p: []const u8) bool {
    var file = std.Io.Dir.cwd().openFile(io, p, .{}) catch return false;
    file.close(io);
    return true;
}

/// Whether this rank asks for RoCE: `want` (TF_ROCE) and no marker; both ranks still agree in roce.Gather.init.
pub fn wanted(io: std.Io, want: bool, p: ?[]const u8) bool {
    if (!want) return false;
    const at = p orelse return true;
    if (!present(io, at)) return true;
    std.log.warn("RoCE all-gather: {s} marks an earlier RoCE failure on this host; serving on NCCL (delete it on every host to retry RoCE)", .{at});
    return false;
}

/// A gather's marker: TF_ROCE_MARK read at setup, written at most once.
pub const Mark = struct {
    path: ?[]const u8 = null,
    io: ?std.Io = null,
    done: bool = false,

    pub fn of(io: std.Io) Mark {
        return .{ .path = path(), .io = io };
    }

    pub fn once(m: *Mark, rank: u8, why: []const u8) void {
        if (m.done) return;
        m.done = true;
        if (m.path) |p| if (m.io) |io| write(io, p, rank, why);
    }
};

/// Writes the marker at `p` naming this rank and the failure; a write that fails is logged, never fatal.
pub fn write(io: std.Io, p: []const u8, rank: u8, why: []const u8) void {
    var buf: [320]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, "rank {d}: {s}\n", .{ rank, why }) catch buf[0..0];
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = text }) catch |e| {
        std.log.err("RoCE all-gather: cannot write the failure marker {s} ({s})", .{ p, @errorName(e) });
        return;
    };
    std.log.warn("RoCE all-gather: wrote {s}; the next start serves on NCCL until it is deleted", .{p});
}

test "the marker: absent asks for RoCE, a written one asks for NCCL and names the rank, TF_ROCE=0 always NCCL" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(dir);
    const at = try std.fs.path.join(std.testing.allocator, &.{ dir, "roce-failed" });
    defer std.testing.allocator.free(at);
    try std.testing.expect(wanted(io, true, at));
    try std.testing.expect(wanted(io, true, null));
    try std.testing.expect(!wanted(io, false, at));
    write(io, at, 1, "ProxyFailed");
    try std.testing.expect(present(io, at));
    try std.testing.expect(!wanted(io, true, at));
    var m: Mark = .{ .path = at, .io = io };
    m.once(0, "again");
    m.once(0, "and again");
    try std.testing.expect(m.done);
    var text_buf: [64]u8 = undefined;
    const text = try std.Io.Dir.cwd().readFile(io, at, &text_buf);
    try std.testing.expectEqualStrings("rank 0: again\n", text); // a mark writes once
}
