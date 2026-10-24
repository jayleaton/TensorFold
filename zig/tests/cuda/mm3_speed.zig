//! tf-cuda-test mm3-graph-eq / mm3-rows-eq / mm3-speed-bench: graph replay equals eager, adaptive rows equal fixed.
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("minimax_cuda");
const check = @import("check.zig");
const synth = @import("mm3_synth.zig");

const Seg = mm.iface.Seg;
const steps = 24;

const Args = struct { layers: u32 = 1, kv8: bool = false, first_moe: ?u32 = null, decode_only: bool = false };

fn parseArgs(rest: []const [:0]const u8) !Args {
    var a: Args = .{};
    for (rest, 0..) |s, i| {
        if (std.mem.eql(u8, s, "kv8")) {
            a.kv8 = true;
        } else if (std.mem.eql(u8, s, "decode")) {
            a.decode_only = true;
        } else if (std.mem.startsWith(u8, s, "moe=")) {
            a.first_moe = try std.fmt.parseInt(u32, s[4..], 10);
        } else if (i == 0) {
            a.layers = try std.fmt.parseInt(u32, s, 10);
        } else return error.BadArgument;
    }
    return a;
}

fn options(a: Args, slots: u32, capacity: u32, score_mib: u32) synth.Options {
    return .{ .layers = a.layers, .slots = slots, .capacity_tokens = capacity, .kv8 = a.kv8, .first_moe = a.first_moe, .score_mib = score_mib };
}

fn randomIds(gpa: std.mem.Allocator, n: usize, vocab: u32, seed: u64) ![]i32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const ids = try gpa.alloc(i32, n);
    for (ids) |*p| p.* = @intCast(prng.random().uintLessThan(u32, vocab));
    return ids;
}

fn hashOf(xs: []const f32) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(xs));
}

fn ms(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// What a script left: every call's logits hash, every drawn token and each slot's KV bytes.
const Rec = struct {
    hashes: std.ArrayList(u64) = .empty,
    tokens: std.ArrayList(u32) = .empty,
    kv: [3][]u8 = .{ &.{}, &.{}, &.{} },

    fn deinit(r: *Rec, gpa: std.mem.Allocator) void {
        r.hashes.deinit(gpa);
        r.tokens.deinit(gpa);
        for (r.kv) |k| gpa.free(k);
    }

    fn note(r: *Rec, gpa: std.mem.Allocator, rows: []const f32, tok: u32) !void {
        try r.hashes.append(gpa, hashOf(rows));
        try r.tokens.append(gpa, tok);
    }

    fn same(a: *const Rec, b: *const Rec, what: []const u8) bool {
        if (!std.mem.eql(u64, a.hashes.items, b.hashes.items)) {
            for (a.hashes.items, b.hashes.items, 0..) |x, y, i| if (x != y) {
                std.debug.print("FAIL {s}: call {d} logits differ ({d} calls)\n", .{ what, i, a.hashes.items.len });
                break;
            };
            return false;
        }
        if (!std.mem.eql(u32, a.tokens.items, b.tokens.items)) {
            std.debug.print("FAIL {s}: tokens differ\n", .{what});
            return false;
        }
        for (a.kv, b.kv, 0..) |x, y, s| {
            var id_buf: [64]u8 = undefined;
            const id = std.fmt.bufPrint(&id_buf, "{s} kv slot {d}", .{ what, s }) catch unreachable;
            check.sameBytes(id, y, x) catch return false;
        }
        return true;
    }
};

fn draft(step: usize, i: usize, vocab: u32) i32 {
    return @intCast((step * 7919 + i * 104729 + 13) % vocab);
}

/// Prompts on 3 slots, then decode rows, verify windows kept in part, and shared windows over all three slots.
fn script(gpa: std.mem.Allocator, m: *synth.Model, prompts: [3][]const i32, rec: *Rec) !void {
    const v: usize = m.geometry().own_vocab;
    const vocab = m.cfg.vocab;
    var pos: [3]u32 = undefined;
    var tok: [3]u32 = undefined;
    for (0..3) |s| {
        tok[s] = try m.prefill(@intCast(s), prompts[s], 2048);
        pos[s] = @intCast(prompts[s].len);
        try rec.note(gpa, m.lastLogits(), tok[s]);
    }
    for (0..steps) |step| {
        for (0..3) |s| {
            if (s == 2 and step % 6 == 3) { // a 5-row verify window that keeps 1 to 3 rows; the rest are overwritten
                var ids: [5]i32 = undefined;
                ids[0] = @intCast(tok[2]);
                for (1..5) |i| ids[i] = draft(step, i, vocab);
                const out = try m.decode(2, pos[2], &ids);
                defer gpa.free(out);
                const keep = 1 + step % 3;
                tok[2] = synth.argmax(out[(keep - 1) * v ..][0..v]);
                try rec.note(gpa, out, tok[2]);
                pos[2] += @intCast(keep);
                continue;
            }
            const out = try m.decode(@intCast(s), pos[s], &[_]i32{@intCast(tok[s])});
            defer gpa.free(out);
            tok[s] = synth.argmax(out);
            try rec.note(gpa, out, tok[s]);
            pos[s] += 1;
        }
        if (step % 5 == 4) { // slots 1, 0, 2 with 1, 2, 3 rows; each keeps its first row
            const segs = [_]Seg{ .{ .slot = 1, .row0 = 0, .rows = 1 }, .{ .slot = 0, .row0 = 1, .rows = 2 }, .{ .slot = 2, .row0 = 3, .rows = 3 } };
            var ids: [6]i32 = undefined;
            var at: [6]i32 = undefined;
            for (segs) |g| for (0..g.rows) |k| {
                ids[g.row0 + k] = if (k == 0) @intCast(tok[g.slot]) else draft(step, g.row0 + k, vocab);
                at[g.row0 + k] = @intCast(pos[g.slot] + k);
            };
            const out = try m.shared(&segs, &ids, &at);
            defer gpa.free(out);
            for (segs) |g| {
                tok[g.slot] = synth.argmax(out[g.row0 * v ..][0..v]);
                pos[g.slot] += 1;
            }
            try rec.note(gpa, out, tok[0] ^ tok[1] ^ tok[2]);
        }
    }
    for (0..3) |s| rec.kv[s] = try m.kvBytes(@intCast(s), pos[s]);
}

pub fn graphEq(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const a = try parseArgs(rest);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const m = try synth.Model.init(gpa, gpu.d, stream, options(a, 3, 4096, 64));
    defer m.deinit();
    // slot 0 crosses the 2048-token bucket edge while decoding; slots 1 and 2 sit in buckets 0 and 1
    const lens = [3]usize{ 2030, 1500, 2200 };
    var prompts: [3][]i32 = undefined;
    for (&prompts, lens, 0..) |*p, n, s| p.* = try randomIds(gpa, n, m.cfg.vocab, 0x6571 + s);
    defer for (prompts) |p| gpa.free(p);
    const ps: [3][]const i32 = .{ prompts[0], prompts[1], prompts[2] };

    var eager: Rec = .{};
    defer eager.deinit(gpa);
    try script(gpa, m, ps, &eager);
    try m.fwd().useGraphs(gpa, .{});
    var cold: Rec = .{};
    defer cold.deinit(gpa);
    try script(gpa, m, ps, &cold);
    const c1 = m.fwd().graphs.?.cache.stats;
    var warm: Rec = .{};
    defer warm.deinit(gpa);
    try script(gpa, m, ps, &warm);
    const c2 = m.fwd().graphs.?.cache.stats;
    const held = m.fwd().graphs.?.cache.count();
    try m.fwd().useGraphs(gpa, .{ .on = false });

    var failed = false;
    const calls = eager.hashes.items.len;
    if (eager.same(&cold, "mm3.graph-eq cold")) check.pass("mm3.graph-eq cold: {d} calls' logits, tokens and 3 slots' KV equal eager ({d} captured, {d} replayed, {d} eager)", .{ calls, c1.captured, c1.replayed, c1.eager }) else failed = true;
    if (eager.same(&warm, "mm3.graph-eq warm")) check.pass("mm3.graph-eq warm: equal eager again ({d} new captures, {d} replays)", .{ c2.captured - c1.captured, c2.replayed - c1.replayed }) else failed = true;
    if (c1.replayed == 0 or c2.replayed - c1.replayed <= c1.replayed) {
        std.debug.print("FAIL mm3.graph-eq: graphs did not replay as keyed ({d} replayed cold, {d} warm)\n", .{ c1.replayed, c2.replayed - c1.replayed });
        failed = true;
    }
    std.debug.print("RESULT mm3-graph-eq layers={d} kv8={} moe={?d}: {d} calls a pass, {d} graphs held, {s}\n", .{ a.layers, a.kv8, a.first_moe, calls, held, if (failed) "FAIL" else "PASS" });
    if (failed) return error.TestFailed;
}

/// One prefill and `steps` greedy decode rows from slot 0: the prompt row, then every row's logits hash and token.
fn greedy(gpa: std.mem.Allocator, m: *synth.Model, ids: []const i32, chunk_rows: u32, rec: *Rec) !void {
    var tok = try m.prefill(0, ids, chunk_rows);
    try rec.note(gpa, m.lastLogits(), tok);
    var pos: u32 = @intCast(ids.len);
    for (0..steps) |_| {
        const out = try m.decode(0, pos, &[_]i32{@intCast(tok)});
        defer gpa.free(out);
        tok = synth.argmax(out);
        try rec.note(gpa, out, tok);
        pos += 1;
    }
    rec.kv[0] = try m.kvBytes(0, pos);
}

pub fn rowsEq(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const a = try parseArgs(rest);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const n = 8192 + 333;
    var fixed: Rec = .{};
    defer fixed.deinit(gpa);
    var cuts: [2]Rec = .{ .{}, .{} };
    defer for (&cuts) |*c| c.deinit(gpa);
    var ids: []i32 = &.{};
    defer gpa.free(ids);
    var fixed_chunks: u32 = 0;
    {
        const m = try synth.Model.init(gpa, gpu.d, stream, options(a, 1, 12288, 64));
        defer m.deinit();
        ids = try randomIds(gpa, n, m.cfg.vocab, 0x726f7773);
        try greedy(gpa, m, ids, 2048, &fixed);
        fixed_chunks = m.chunks;
        for (&cuts, [_]u32{ 512, 1024 }) |*c, rows| try greedy(gpa, m, ids, rows, c);
    }
    // a 1 MiB bound: 2048-row chunks while they fit, then fewer whole pages as positions grow (the engine's plan)
    const m = try synth.Model.init(gpa, gpu.d, stream, options(a, 1, 12288, 1));
    defer m.deinit();
    var adapt: Rec = .{};
    defer adapt.deinit(gpa);
    try greedy(gpa, m, ids, 0, &adapt);
    var rows_seen: [8]usize = @splat(0);
    var at: usize = 0;
    var k: usize = 0;
    while (at < n and k < rows_seen.len) : (k += 1) {
        rows_seen[k] = m.plan.rowsAt(at, n);
        at += rows_seen[k];
    }
    var failed = false;
    if (fixed.same(&cuts[0], "mm3.rows-eq 512")) check.pass("mm3.rows-eq: 512-row chunks equal 2048-row chunks: prompt row, {d} greedy tokens and KV", .{steps}) else failed = true;
    if (fixed.same(&cuts[1], "mm3.rows-eq 1024")) check.pass("mm3.rows-eq: 1024-row chunks equal 2048-row chunks", .{}) else failed = true;
    if (fixed.same(&adapt, "mm3.rows-eq adaptive")) check.pass("mm3.rows-eq: the bounded plan's chunks {any} under a {d}-byte scratch equal fixed rows", .{ rows_seen[0..k], m.plan.scratch }) else failed = true;
    std.debug.print("RESULT mm3-rows-eq layers={d} kv8={} moe={?d}: {d} tokens, fixed {d} chunks, adaptive {d} chunks, {s}\n", .{ a.layers, a.kv8, a.first_moe, n, fixed_chunks, m.chunks, if (failed) "FAIL" else "PASS" });
    if (failed) return error.TestFailed;
}

/// Milliseconds a forward of `call` takes to the stream's end, median of `reps` after `warm` untimed runs.
fn timeCall(io: std.Io, m: *synth.Model, call: mm.forward.Call, warm: usize, reps: usize, out: []f64) !f64 {
    for (0..warm) |_| {
        _ = try m.fwd().run(call);
        try m.settle();
    }
    for (out[0..reps]) |*t| {
        const t0 = check.now(io);
        _ = try m.fwd().run(call);
        try m.settle();
        t.* = ms(check.now(io) - t0);
    }
    return check.median(out[0..reps]);
}

pub fn bench(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const a = try parseArgs(rest);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const ctxs = [_]u32{ 2048, 8192, 32768 };
    const m = try synth.Model.init(gpa, gpu.d, stream, options(a, 4, 32768 + 256, 64));
    defer m.deinit();
    const ids = try randomIds(gpa, 32768 + 64, m.cfg.vocab, 0x62656e63);
    defer gpa.free(ids);
    var times: [64]f64 = undefined;
    const reps = 40;

    // prefill: whole prompts of each length at each chunk size, and at the plan's rows (median of 3)
    if (!a.decode_only) for (ctxs) |n| for ([_]u32{ 512, 1024, 2048, 0 }) |rows| {
        var t: [3]f64 = undefined;
        for (&t) |*x| {
            const t0 = check.now(io);
            _ = try m.prefill(0, ids[0..n], rows);
            x.* = ms(check.now(io) - t0);
        }
        const med = check.median(&t);
        std.debug.print("RESULT mm3-bench prefill layers={d} moe={?d} tokens={d} rows={s}{d}: {d:.1} ms, {d:.0} tok/s (median of 3)\n", .{ a.layers, a.first_moe, n, if (rows == 0) "plan:" else "", if (rows == 0) m.plan.rowsAt(0, n) else rows, med, @as(f64, @floatFromInt(n)) / (med / 1e3) });
    };

    // decode windows at each context: eager, then graphs (warm-up runs the first-use eager step and the capture);
    // `decode` times them over the zeroed pool instead, as sparse attention reads its selected blocks either way
    if (!a.decode_only) for (1..4) |s| {
        _ = try m.prefill(@intCast(s), ids[0..32768], 2048);
    };
    for (ctxs) |n| {
        if (!a.decode_only) _ = try m.prefill(0, ids[0..n], 2048);
        const pos0: u32 = n;
        var positions: [64]i32 = undefined;
        const segs = [_]Seg{ .{ .slot = 0, .row0 = 0, .rows = 16 }, .{ .slot = 1, .row0 = 16, .rows = 16 }, .{ .slot = 2, .row0 = 32, .rows = 16 }, .{ .slot = 3, .row0 = 48, .rows = 16 } };
        for (segs) |g| for (0..16) |k| {
            positions[g.row0 + k] = @intCast((if (g.slot == 0) pos0 else 32768 - 64) + k);
        };
        const shapes = [_]struct { name: []const u8, call: mm.forward.Call }{
            .{ .name = "rows=1", .call = .{ .ids = ids[0..1], .slot = 0, .pos0 = pos0 } },
            .{ .name = "rows=4", .call = .{ .ids = ids[0..4], .slot = 0, .pos0 = pos0 } },
            .{ .name = "rows=16", .call = .{ .ids = ids[0..16], .slot = 0, .pos0 = pos0 } },
            .{ .name = "shared=4x16", .call = .{ .ids = ids[0..64], .segs = &segs, .positions = &positions } },
        };
        for (shapes) |sh| {
            try m.fwd().useGraphs(gpa, .{ .on = false });
            const e = try timeCall(io, m, sh.call, 3, reps, &times);
            try m.fwd().useGraphs(gpa, .{});
            const g = try timeCall(io, m, sh.call, 3, reps, &times);
            const st = m.fwd().graphs.?.cache.stats;
            std.debug.print("RESULT mm3-bench decode layers={d} moe={?d} prefilled={} ctx={d} {s}: eager {d:.3} ms, graph {d:.3} ms ({d:.2}x, {d} replays), median of {d}\n", .{ a.layers, a.first_moe, !a.decode_only, n, sh.name, e, g, e / g, st.replayed, reps });
        }
    }
    try m.fwd().useGraphs(gpa, .{ .on = false });
}
