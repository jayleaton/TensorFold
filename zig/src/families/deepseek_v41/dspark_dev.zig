//! DSpark's pass with its candidates on the GPU and the pass in CUDA graphs (TF_DSV41_DRAFT_GRAPHS=1, every rank
//! alike; default off: dspark_gpu.zig / dspark_slots.zig's host step), as Python's drafter runs it on its CUDA path
//! (draft_graphs.py): the pass, its candidates and their gather on the device, one copy back.
//!
//! - **ds_out on the device** (`Dev.launch`, the glue step's body): glue.cu's `ds_cands` (each rank's best k columns a
//!   row by (value desc, column asc): dspark.candidates' exact selection), the ranks' all-gather and one async copy of
//!   the gathered [W][rows][2 k] (and the head hidden when the confidence head is loaded) into pinned memory. Nothing
//!   in the pass reads the device on the host, so the whole pass is one stream sequence: graphable, and a speculative
//!   pass (`Pass.begin`) launches all of it. The host keeps the chain (draft/dspark.zig), so draft ids are the host
//!   path's bit for bit: the same candidates (an exact selection), the same chain.
//! - **statics** from pinned staging (one region a pass of the round), no stream sync before them.
//! - **checks**: the first `check_passes` passes also run the host's candidates on this rank's logits and compare.
//!   After them every rank's verdict is agreed (one all-gather, at the same pass on every rank, before any graph is
//!   captured): all equal -> graphs on (when TF_DSV41_DRAFT_GRAPHS asks); any differ -> every rank back on the host
//!   step for the rest of the run (logged), so a bad kernel costs a few drafts, never a reply or a hang.
//! - **graphs**: graph_cache over the runner's stream, a key a pass size (slots), the addresses the pass bakes in
//!   fingerprinted; TF_DSV41_DRAFT_GRAPHS_MAX (32, Python's), TF_DSV41_GRAPH_FLOOR_GIB (graphs.zig).

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const calls = @import("calls.zig");
const run = @import("run.zig");
const graphs = @import("graphs.zig");
const rows_mod = @import("dspark_rows.zig");
const dspark = @import("draft/dspark.zig");
const ph = @import("phases.zig");

pub const Mode = enum { host, device, graphs };

/// TF_DSV41_DRAFT_GRAPHS: 1 = device candidates + graphed passes, `device` = device candidates eager (A / B), unset
/// or 0 = the host step.
pub fn modeFromEnv() !Mode {
    const v = std.mem.span(std.c.getenv("TF_DSV41_DRAFT_GRAPHS") orelse return .host);
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return .host;
    if (std.mem.eql(u8, v, "1")) return .graphs;
    if (std.mem.eql(u8, v, "device")) return .device;
    return error.BadDraftGraphs;
}

pub const check_passes = 8;

pub const Dev = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    comm: tp.collective.Collective,
    stream: cuda.Stream,
    /// what TF_DSV41_DRAFT_GRAPHS asked; `active` is what runs (host after a failed check)
    want: Mode,
    active: Mode,
    k: usize,
    world: usize,
    hidden: usize,
    /// rows of every pass of a round together (the regions' size)
    rows_max: usize,
    conf: bool,
    send: cuda.DeviceBuffer,
    gathered: cuda.DeviceBuffer,
    /// pinned: gathered fp32 [rows_max][W][2 k] region-major, then the head hidden bf16 [rows_max][D]
    host: cuda.HostBuffer,
    done: cuda.Event,
    agreement: graphs.Agreement,
    engine: graphs.gc.CudaEngine,
    cache: graphs.gc.Cache(u32),
    /// pinned statics staging: `regions` regions of `region_bytes` (a pass of the round each)
    stage: cuda.HostBuffer,
    region_bytes: usize,
    regions: usize,
    checked: u32 = 0,
    mismatches: u64 = 0,
    stats: struct { passes: u64 = 0, graphed: u64 = 0 } = .{},

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, comm: tp.collective.Collective, stream: cuda.Stream, want: Mode, k: usize, hidden: usize, rows_max: usize, conf: bool, region_bytes: usize, regions: usize) !*Dev {
        const x = try gpa.create(Dev);
        errdefer gpa.destroy(x);
        const W: usize = comm.world();
        const pk = rows_max * 2 * k * 4;
        var max: u32 = 32;
        if (std.c.getenv("TF_DSV41_DRAFT_GRAPHS_MAX")) |v| max = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        x.* = .{
            .gpa = gpa,
            .d = d,
            .comm = comm,
            .stream = stream,
            .want = want,
            .active = .device,
            .k = k,
            .world = W,
            .hidden = hidden,
            .rows_max = rows_max,
            .conf = conf,
            .send = try cuda.DeviceBuffer.alloc(d, pk),
            .gathered = try cuda.DeviceBuffer.alloc(d, pk * W),
            .host = try cuda.HostBuffer.alloc(d, pk * W + 2 * rows_max * hidden),
            .done = try cuda.Event.init(d, false),
            .agreement = try graphs.Agreement.init(d, comm, stream),
            .engine = .{ .d = d },
            .cache = undefined,
            .stage = try cuda.HostBuffer.alloc(d, region_bytes * regions),
            .region_bytes = region_bytes,
            .regions = regions,
        };
        x.cache = graphs.gc.Cache(u32).init(gpa, x.engine.engine(), .{ .on = true, .max = max, .min_keep = 2 });
        x.cache.agree = x.agreement.agree();
        x.cache.room = roomFn;
        if (std.c.getenv("TF_DSV41_GRAPH_FLOOR_GIB")) |v| floor_gib = std.fmt.parseFloat(f64, std.mem.span(v)) catch 5.0;
        std.log.scoped(.dsv41).info("draft graphs: {t} (candidates on the device, the first {d} passes checked against the host's)", .{ want, check_passes });
        return x;
    }

    pub fn deinit(x: *Dev) void {
        const cs = x.cache.stats;
        std.log.scoped(.dsv41).info("draft graphs: {t}, {d} passes, {d} graphed; {d} captured, {d} replayed, {d} eager; {d} checked, {d} mismatched", .{ x.active, x.stats.passes, x.stats.graphed, cs.captured, cs.replayed, cs.eager, x.checked, x.mismatches });
        x.cache.deinit();
        x.send.free();
        x.gathered.free();
        x.host.free();
        x.done.deinit();
        x.agreement.deinit();
        x.stage.free();
        x.gpa.destroy(x);
    }

    /// The device path is on (candidates on the GPU); false: the host step.
    pub fn on(x: *const Dev) bool {
        return x.active != .host;
    }

    /// Region `g`'s pinned staging for the statics: (int64 [n64], int32 [n32]); the previous round's copies out of it
    /// finished before (every round ends at `wait`).
    pub fn statics(x: *Dev, g: usize, n64: usize, n32: usize) !struct { []i64, []i32 } {
        if (g >= x.regions or 8 * n64 + 4 * n32 > x.region_bytes) return error.Statics;
        const b = x.stage.bytes[g * x.region_bytes ..][0..x.region_bytes];
        return .{ @alignCast(std.mem.bytesAsSlice(i64, b[0 .. 8 * n64])), @alignCast(std.mem.bytesAsSlice(i32, b[8 * n64 ..][0 .. 4 * n32])) };
    }

    /// ds_out on the device for the pass's `rows` rows at row offset `at` of the round (a stream sequence: no sync, no
    /// allocation; graphable): candidates, the ranks' gather, the copies back.
    pub fn launch(x: *Dev, r: *run.Runner, logits_t: calls.Tensor, hid_t: calls.Tensor, rows: usize, at: usize) !void {
        if (at + rows > x.rows_max) return error.Rows;
        const k = x.k;
        const V: usize = @intCast(logits_t.shape[1]);
        const pk = rows * 2 * k * 4;
        const ops = r.kernels.others(r.stream);
        const send = x.send.ptr + at * 2 * k * 4;
        const dst = x.gathered.ptr + at * 2 * k * 4 * x.world;
        try ops.dsCandidates(try r.tensorAddr(logits_t), V, rows, V, k, @intCast(x.comm.rank() * V), send);
        try x.comm.allGather(send, dst, rows * 2 * k, .f32, r.stream.handle);
        try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(x.host.bytes.ptr + at * 2 * k * 4 * x.world, dst, pk * x.world, r.stream.handle), "cuMemcpyDtoHAsync");
        if (x.conf) {
            const ho = x.rows_max * 2 * k * 4 * x.world + at * 2 * x.hidden;
            try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(x.host.bytes.ptr + ho, try r.tensorAddr(hid_t), 2 * rows * x.hidden, r.stream.handle), "cuMemcpyDtoHAsync");
        }
    }

    /// After the round's launches: one event the host waits on.
    pub fn mark(x: *Dev) !void {
        try x.done.record(x.stream);
    }

    pub fn wait(x: *Dev) !void {
        try x.done.synchronize();
    }

    /// Region `at`'s gathered candidates into rows of [W k] (`cand` / `cval`, Python's gather_cols order) and the head
    /// hidden as fp32 (`hid`, when the confidence head is loaded); every id checked (dspark_gpu.checkCandidates).
    pub fn unpack(x: *Dev, rows: usize, at: usize, vocab: usize, cand: []i32, cval: []f32, hid: ?[]f32) !void {
        const k = x.k;
        const all: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, x.host.bytes[at * 2 * k * 4 * x.world ..][0 .. rows * 2 * k * 4 * x.world]));
        rows_mod.ungather(all, x.world, rows, k, cand, cval);
        if (hid) |h| {
            const ho = x.rows_max * 2 * k * 4 * x.world + at * 2 * x.hidden;
            const h16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, x.host.bytes[ho..][0 .. 2 * rows * x.hidden]));
            for (h16, h[0 .. rows * x.hidden]) |b, *o| o.* = dspark.bf16(b);
        }
        for (cand[0 .. rows * x.world * k], 0..) |id, i| if (id < -1 or id >= @as(i64, @intCast(vocab))) {
            std.log.err("dspark: gathered candidate {d} at {d} is not a token id", .{ id, i });
            return error.BadCandidates;
        };
    }

    /// A check pass (the first `check_passes`): this rank's logits rows [rows, V] at `logits` on the host's rule,
    /// against this rank's part of region `at` (after `wait`). Any difference is logged and counted.
    pub fn check(x: *Dev, logits: u64, V: usize, rows: usize, at: usize) !void {
        if (x.checked >= check_passes) return;
        x.checked += 1;
        const k = x.k;
        const host = try x.gpa.alloc(f32, rows * V);
        defer x.gpa.free(host);
        try cuda.DeviceBuffer.download(.{ .d = x.d, .ptr = logits, .len = 4 * host.len }, 0, std.mem.sliceAsBytes(host));
        const ids = try x.gpa.alloc(i32, k);
        defer x.gpa.free(ids);
        const vals = try x.gpa.alloc(f32, k);
        defer x.gpa.free(vals);
        const rank = x.comm.rank();
        const all: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, x.host.bytes[at * 2 * k * 4 * x.world ..][0 .. rows * 2 * k * 4 * x.world]));
        for (0..rows) |i| {
            try dspark.candidates(x.gpa, host[i * V ..][0..V], @intCast(rank * V), ids, vals);
            const got = all[(rank * rows + i) * 2 * k ..][0 .. 2 * k];
            for (0..k) |j| {
                const gid: i32 = @bitCast(got[k + j]);
                if (gid != ids[j] or @as(u32, @bitCast(got[j])) != @as(u32, @bitCast(vals[j]))) {
                    if (x.mismatches < 4) std.log.err("draft graphs: device candidates differ from the host's (row {d} slot {d}: {d} {e} vs {d} {e})", .{ i, j, gid, got[j], ids[j], vals[j] });
                    x.mismatches += 1;
                    break;
                }
            }
        }
    }

    /// After a pass (every rank, in the same order): at the last check pass the ranks agree on the device path; all
    /// equal -> TF_DSV41_DRAFT_GRAPHS' mode, else the host step from here on, every rank.
    pub fn settle(x: *Dev) !void {
        x.stats.passes += 1;
        if (x.active != .device or x.stats.passes != check_passes) return;
        const ag = x.agreement.agree();
        const ok = try ag.all(ag.ctx, x.mismatches == 0);
        if (ok) {
            x.active = x.want;
            std.log.scoped(.dsv41).info("draft graphs: {d} passes' device candidates equal the host's on every rank: {t}", .{ check_passes, x.active });
        } else {
            x.active = .host;
            std.log.err("draft graphs: device candidates differ on some rank ({d} rows here); the host step from here on", .{x.mismatches});
        }
    }

    /// The pass's launches as the key's graph (once the checks passed and graphs are asked), else eagerly. The caller
    /// staged the statics; `body` issues the pass's calls on the runner's stream.
    pub fn issue(x: *Dev, key: u32, fp: u64, body: graphs.gc.Body) !void {
        if (x.active != .graphs) return body.run(body.ctx, x.stream);
        const out = try x.cache.run(key, x.stream, fp, body);
        if (out != .eager) x.stats.graphed += 1;
    }

    /// The device buffers a graphed pass bakes in besides the runner's roles.
    pub fn words(x: *const Dev, out: []u64) usize {
        const w = [_]u64{ x.send.ptr, x.gathered.ptr, @intFromPtr(x.host.bytes.ptr), @intFromPtr(x.stage.bytes.ptr) };
        @memcpy(out[0..w.len], &w);
        return w.len;
    }
};

var floor_gib: f64 = 5.0;
fn roomFn() bool {
    return graphs.roomAbove(floor_gib);
}

/// The fingerprint of a pass: the runner's owned buffers, its scratch, the pass's own and `extra`.
pub fn fingerprint(r: *const run.Runner, x: *const Dev, extra: []const u64) u64 {
    var words: [80]u64 = undefined;
    var m: usize = 0;
    for (r.owned.items) |b| {
        if (m + 2 > 60) break;
        words[m] = b.ptr;
        words[m + 1] = b.len;
        m += 2;
    }
    if (r.scratch) |s| {
        words[m] = s.ptr;
        m += 1;
    }
    m += x.words(words[m..]);
    for (extra) |e| if (m < words.len) {
        words[m] = e;
        m += 1;
    };
    return graphs.gc.fingerprintOf(words[0..m]);
}

test "draft graphs: the knob" {
    try std.testing.expectEqual(Mode.host, try modeFromEnv());
}
