//! TF_DSV41_RUNNER_TRACE=1 (stderr) or =<path>: rounds around each runner fire (lanes round_trace), with their work.
const std = @import("std");
const lanes = @import("lanes");
const batch = @import("batch.zig");
const ln = @import("draft/lanes.zig");
const er = @import("engram_rows.zig");
const GpuPair = @import("batch_gpu_time.zig").GpuPair;

/// Per step: row windows and graphs, GPU and host time, speculative DSpark passes by outcome, Engram row lookups.
const names = [_][]const u8{ "windows", "rows", "padded", "builds", "captures", "eager", "replayed", "evicted", "dropped", "gpu_us", "lost", "prep_us", "stage_us", "launch_us", "keep_us", "spec_launched", "spec_hit", "spec_commit", "spec_slots", "spec_unused", "spec_pending", "spec_other", "spec_held", "spec_fed", "engram_hits", "engram_misses" };

/// What the counters are read from (rank 0's): the row batch, the DSpark lanes, Engram's rows.
const Src = struct { rows: ?*batch.Batch, x: *ln.Lanes, engram: ?*er.Rows };

/// Opens the trace when TF_DSV41_RUNNER_TRACE is set; TF_DSV41_RUNNER_TRACE_ROUNDS: rounds on each side (12).
pub fn open(gpa: std.mem.Allocator, rows: ?*batch.Batch, x: *ln.Lanes, engram: ?*er.Rows) !void {
    const v = std.c.getenv("TF_DSV41_RUNNER_TRACE") orelse return;
    const p = std.mem.span(v);
    if (p.len == 0 or std.mem.eql(u8, p, "0")) return;
    const fd: c_int = if (std.mem.eql(u8, p, "1")) 2 else std.c.open(v, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return std.log.warn("TF_DSV41_RUNNER_TRACE: cannot open {s}", .{p});
    var n: u32 = 12;
    if (std.c.getenv("TF_DSV41_RUNNER_TRACE_ROUNDS")) |r| n = std.fmt.parseInt(u32, std.mem.span(r), 10) catch return error.BadRunnerTraceRounds;
    if (rows) |b| b.gpu = try GpuPair.init(gpa, b.f.runner.d);
    const src = try gpa.create(Src); // lives as long as the process's trace
    src.* = .{ .rows = rows, .x = x, .engram = engram };
    lanes.round_trace.open(fd, n, .{ .ctx = src, .names = &names, .read = read });
    std.log.info("dsv41 runner trace: {d} rounds around each fire to {s}", .{ n, if (fd == 2) "stderr" else p });
}

fn read(ctx: *anyopaque, out: []u64) void {
    const src: *Src = @ptrCast(@alignCast(ctx));
    var all: [names.len]u64 = @splat(0);
    if (src.rows) |b| {
        const st = b.stats;
        var g: @TypeOf(b.g.?.cache.stats) = .{};
        if (b.g) |*gr| g = gr.cache.stats;
        var gpu_ns: u64 = 0;
        var lost: u64 = 0;
        if (b.gpu) |t| {
            gpu_ns = t.total();
            lost = t.lost;
        }
        const v = [_]u64{ st.windows, st.rows, st.padded, st.builds, st.captures, st.eager, g.replayed, g.evicted, g.dropped, gpu_ns / 1000, lost, st.prep_ns / 1000, st.stage_ns / 1000, st.launch_ns / 1000, st.keep_ns / 1000 };
        @memcpy(all[0..v.len], &v);
    }
    if (src.x.spec) |sp| {
        const m = sp.stats.miss;
        const other = m.get(.reset) + m.get(.inputs);
        const v = [_]u64{ sp.stats.launched, sp.stats.hit, m.get(.commit), m.get(.slots), m.get(.unused), m.get(.pending), other, sp.stats.held, sp.stats.fed };
        @memcpy(all[15..][0..v.len], &v);
    }
    if (src.engram) |r| {
        all[24] = r.stats.hits;
        all[25] = r.stats.misses;
    }
    @memcpy(out, all[0..out.len]);
}
