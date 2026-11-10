//! Flash Next engine CUDA graphs: solo window and MTP step captures, shared-round graphs, capture faults.
const std = @import("std");
const cuda = @import("cuda");
const state = @import("cuda_state.zig");
const fwd = @import("cuda_forward.zig");
const mtp = @import("cuda_mtp.zig");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const Seq = engine.Seq;
const GraphKey = engine.GraphKey;
const max_multi_graphs = engine.max_multi_graphs;
const CaptureFault = engine.CaptureFault;
const bucket = engine.bucket;

// -- CUDA graphs ------------------------------------------------------------------------------------------

/// Graphs of sequence `seq` released: all, or those of a buffer version other than `keep`.
pub fn dropGraphs(e: *Engine, seq: usize, keep: ?u32) void {
    return e.dropGraphsOf(seq, keep, null);
}

pub fn dropGraphsOf(e: *Engine, seq: usize, keep: ?u32, media: ?u64) void {
    while (true) {
        var it = e.graphs.iterator();
        const found = while (it.next()) |kv| {
            if (kv.key_ptr.seq == seq and (keep == null or kv.key_ptr.version != keep.? or (media != null and kv.key_ptr.media != media.?))) break kv.key_ptr.*;
        } else null;
        const k = found orelse return;
        var g = e.graphs.fetchRemove(k).?.value;
        g.deinit();
    }
}

/// Image/video graphs are keyed by attachment (GraphKey.media); TF_FLASHNEXT_MEDIA_GRAPHS=0 runs them eagerly.
pub fn graphable(e: *Engine, rows: usize) bool {
    return e.graphs_on and e.f.dump == null and e.f.prof == null and rows <= e.buf.b.rows and (e.media_graphs or e.bound.media == null);
}

pub fn mediaKey(seq: *const Seq) u64 {
    return if (seq.media) |m| m.epoch else 0;
}

/// The round scratch's parity the staged window writes (a graph bakes in both parities' buffers).
pub fn parity(e: *Engine) ?u32 {
    return e.f.cur_par;
}

pub fn body(e: *Engine, comptime head: bool, rows: usize) !void {
    if (head) {
        _ = try mtp.compute(&e.f, e.bound, &e.mbuf.?, rows);
    } else _ = try e.f.compute(e.bound, &e.buf, rows, true);
}

/// Replays the staged window's graph, capturing it first after an eager run (Graphs.forward / mtp_forward).
pub fn graphed(e: *Engine, comptime head: bool, rows: usize, ctx: usize, par: u32) !void {
    const seq = e.bound;
    const key: GraphKey = .{ .seq = @intFromPtr(seq), .version = seq.st.version, .mtp = head, .rows = @intCast(rows), .parity = par, .ctx = ctx, .media = mediaKey(seq) };
    e.f.context = ctx;
    defer e.f.context = null;
    if (e.graphs.get(key)) |g| return g.launchOn(e.stream);
    e.dropGraphsOf(key.seq, key.version, key.media);
    try e.body(head, rows);
    // the eager run above is this call's result; a capture that fails from here on only costs the graph
    if (e.solo_fault.pause > 0) {
        e.solo_fault.pause -= 1;
        return;
    }
    try e.stream.synchronize();
    if (!e.capBegin(&e.solo_fault, "solo")) return;
    const ok = if (e.body(head, rows)) |_| true else |err| e.capNote(&e.solo_fault, "solo", "capture", err);
    var exec = e.capEnd(&e.solo_fault, "solo", ok) orelse return;
    errdefer exec.deinit();
    try e.graphs.put(e.gpa, key, exec);
    e.captures += 1;
    e.capLog("solo");
}

/// Starts a thread-local capture on the engine's stream; false (noted) when the driver refuses.
pub fn capBegin(e: *Engine, f: *CaptureFault, comptime class: []const u8) bool {
    cuda.graph.beginCapture(e.stream, .thread_local) catch |err| {
        _ = e.capNote(f, class, "begin capture", err);
        return false;
    };
    return true;
}

/// Ends the capture and, when `ok`, instantiates and uploads it; null on any failure with nothing left allocated.
pub fn capEnd(e: *Engine, f: *CaptureFault, comptime class: []const u8, ok: bool) ?cuda.graph.Exec {
    var g = cuda.graph.endCapture(e.stream) catch |err| {
        if (ok) _ = e.capNote(f, class, "end capture", err);
        return null;
    };
    defer g.deinit();
    if (!ok) return null;
    e.instantiated += 1;
    if (e.inject_every > 0 and e.instantiated % e.inject_every == 0) {
        std.log.warn("graph instantiate: injected failure (TF_FLASHNEXT_GRAPH_FAIL={d})", .{e.inject_every});
        _ = e.capNote(f, class, "instantiate", error.CudaFailed);
        return null;
    }
    if (e.graph_log) { // an earlier asynchronous error anywhere in the context would surface in the next graph call
        const d = e.ctx.d;
        const r = d.api.cuCtxSynchronize();
        if (r != 0) std.log.warn("graph log: cuCtxSynchronize before an instantiate returned {s} ({d})", .{ d.errorName(r), r });
    }
    var exec = g.instantiate() catch |err| {
        _ = e.capNote(f, class, "instantiate", err);
        return null;
    };
    exec.upload(e.stream) catch |err| {
        exec.deinit();
        _ = e.capNote(f, class, "upload", err);
        return null;
    };
    f.streak = 0;
    return exec;
}

/// A failed capture: warn, run eagerly, pause the class 64 rounds (doubling per failure up to 4096); returns false.
pub fn capNote(e: *Engine, f: *CaptureFault, comptime class: []const u8, what: []const u8, err: anyerror) bool {
    f.total += 1;
    f.streak +|= 1;
    f.pause = @as(u32, 64) << @intCast(@min(f.streak - 1, 6));
    const free: usize = if (e.ctx.memInfo()) |m| m.free else |_| 0;
    std.log.warn("{s} graph {s} failed ({s}); the round runs eagerly (same result), no {s} captures for {d} rounds (failure {d}, {d} in a row); graphs live: {d} solo, {d} shared; {d} instantiated; {d} MiB device memory free", .{ class, what, @errorName(err), class, f.pause, f.total, f.streak, e.graphs.count(), e.multi.count(), e.instantiated, free >> 20 });
    return false;
}

pub fn capLog(e: *Engine, comptime class: []const u8) void {
    if (!e.graph_log or e.captures % 50 != 0) return;
    const free: usize = if (e.ctx.memInfo()) |m| m.free else |_| 0;
    std.log.info("graphs after {d} captures ({s}): {d} solo, {d} shared live; {d} instantiated; {d} MiB device memory free", .{ e.captures, class, e.graphs.count(), e.multi.count(), e.instantiated, free >> 20 });
}

/// Graphs.warm: every decode window and MTP step size captured on the engine's sequence before serving; the count.
pub fn warmGraphs(e: *Engine) !usize {
    if (!e.graphs_on) return 0;
    const before = e.captures;
    const keep = e.bound;
    defer e.bound = keep;
    e.bound = e.own;
    const zeros: [state.max_rows]u32 = @splat(0);
    const rows = @min(e.depth + 1, e.buf.b.rows, state.max_rows);
    // each window twice: a forward writes the other parity each time
    try e.f.reset(e.own);
    for (1..rows + 1) |r| for (0..2) |_| try e.forward(zeros[0..r]);
    try e.f.reset(e.own);
    if (e.mbuf) |*mb| for (1..rows + 1) |n| try e.mtpForward(mb, zeros[0..n], e.buf.b.streams);
    try e.f.reset(e.own);
    try e.stream.synchronize();
    return e.captures - before;
}

/// Shared graphs released: those baking in sequence `seq`, or all (null).
pub fn dropMulti(e: *Engine, seq: ?usize) void {
    while (true) {
        var it = e.multi.iterator();
        const found = while (it.next()) |kv| {
            if (seq == null or std.mem.indexOfScalar(usize, kv.value_ptr.seqs, seq.?) != null) break kv.key_ptr.*;
        } else null;
        const k = found orelse return;
        var g = e.multi.fetchRemove(k).?.value;
        g.exec.deinit();
        e.gpa.free(g.seqs);
    }
}

/// A shared step's composition key (sequences, versions, rows, parities, `ctxs` buckets); null if DeltaNet apart.
pub fn multiKey(e: *Engine, comptime head: bool, segs: []const fwd.Seg, ctxs: []usize) ?u64 {
    var h = std.hash.Wyhash.init(if (head) 0x6d7470 else 0x6d61696e);
    for (segs, ctxs) |sg, *c| {
        const st = &sg.seq.st;
        var par: u32 = 0;
        if (!head and e.g.linear_layers > 0) par = e.f.cur_par;
        const n = sg.a1 - sg.a0;
        c.* = bucket((if (head) st.mtp_len else st.pos) + n, st.capacity);
        const words = [_]u64{ @intFromPtr(sg.seq), st.version, n, par, c.*, mediaKey(sg.seq) };
        h.update(std.mem.asBytes(&words));
    }
    return h.final();
}

/// A shared step through its graph (captured on the composition's second sight), else eagerly; staging outside.
pub fn runMulti(e: *Engine, comptime head: bool, segs: []const fwd.Seg, x: *const fwd.Bufs) !void {
    var ctxs: [64]usize = undefined;
    if (segs.len > ctxs.len) return error.TooManyStreams;
    const media = !e.media_graphs and for (segs) |sg| {
        if (sg.seq.media != null) break true;
    } else false;
    const key = if (e.graphs_on and e.f.dump == null and e.f.prof == null and segs.len <= e.multi_graphs and !media) e.multiKey(head, segs, ctxs[0..segs.len]) else null;
    const k = key orelse {
        if (head) _ = try mtp.computeSegs(&e.f, segs, x) else _ = try e.f.computeSegs(segs, x, true);
        return;
    };
    e.f.contexts = ctxs[0..segs.len];
    defer e.f.contexts = null;
    if (e.multi.get(k)) |g| return g.exec.launchOn(e.stream);
    if (e.seen.count() > 16384) e.seen.clearRetainingCapacity();
    const again = (try e.seen.getOrPut(e.gpa, k)).found_existing;
    if (head) _ = try mtp.computeSegs(&e.f, segs, x) else _ = try e.f.computeSegs(segs, x, true);
    if (!again) return;
    if (e.multi.count() >= max_multi_graphs) {
        e.dropMulti(null);
        e.seen.clearRetainingCapacity();
    }
    if (e.multi_fault.pause > 0) {
        e.multi_fault.pause -= 1;
        return;
    }
    try e.stream.synchronize();
    if (!e.capBegin(&e.multi_fault, "shared")) return;
    const r = if (head) mtp.computeSegs(&e.f, segs, x) else e.f.computeSegs(segs, x, true);
    const ok = if (r) |_| true else |err| e.capNote(&e.multi_fault, "shared", "capture", err);
    var exec = e.capEnd(&e.multi_fault, "shared", ok) orelse return;
    errdefer exec.deinit();
    const seqs = try e.gpa.alloc(usize, segs.len);
    errdefer e.gpa.free(seqs);
    for (segs, seqs) |sg, *q| q.* = @intFromPtr(sg.seq);
    try e.multi.put(e.gpa, k, .{ .exec = exec, .seqs = seqs });
    e.captures += 1;
    e.capLog("shared");
}

/// mtp_forward through a graph when it fits one (the stage stays outside).
pub fn mtpForward(e: *Engine, mb: *const fwd.Bufs, next: []const u32, streams: u64) !void {
    try mtp.stage(&e.f, e.bound, mb, next, streams);
    const n = next.len;
    if (e.graphable(n) and mb == &e.mbuf.?) {
        try e.graphed(true, n, bucket(e.bound.st.mtp_len + n, e.bound.st.capacity), 0);
    } else _ = try mtp.compute(&e.f, e.bound, mb, n);
}
