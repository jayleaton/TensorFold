//! TF_DSV41_RUNNER_TRACE: the row windows' GPU time, from a timing event pair around each window's launches.
const std = @import("std");
const cuda = @import("cuda");

pub const GpuPair = struct {
    gpa: std.mem.Allocator,
    start: cuda.Event,
    end: cuda.Event,
    pending: bool = false, // the pair holds a window not yet read
    ns: u64 = 0, // the windows' GPU time read so far
    lost: u64 = 0, // windows a later one began before they were done (not read)

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !*GpuPair {
        const g = try gpa.create(GpuPair);
        errdefer gpa.destroy(g);
        var start = try cuda.Event.init(d, true);
        errdefer start.deinit();
        g.* = .{ .gpa = gpa, .start = start, .end = try cuda.Event.init(d, true) };
        return g;
    }

    pub fn deinit(g: *GpuPair) void {
        g.start.deinit();
        g.end.deinit();
        g.gpa.destroy(g);
    }

    /// Before a window's launches: the last window read if it is done (never a wait), then the start mark.
    pub fn begin(g: *GpuPair, stream: cuda.Stream) !void {
        g.fold();
        if (g.pending) g.lost += 1;
        g.pending = false;
        try g.start.record(stream);
    }

    /// After a window's launches (outside any capture).
    pub fn finish(g: *GpuPair, stream: cuda.Stream) void {
        g.end.record(stream) catch return;
        g.pending = true;
    }

    fn fold(g: *GpuPair) void {
        if (!g.pending) return;
        if (!(g.end.done() catch false)) return;
        const ms = cuda.Event.elapsedMs(g.start, g.end) catch return;
        g.ns += @intFromFloat(@max(0.0, ms) * 1e6);
        g.pending = false;
    }

    /// The windows' GPU time so far, the last one included when it is done.
    pub fn total(g: *GpuPair) u64 {
        g.fold();
        return g.ns;
    }
};
