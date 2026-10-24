//! tf-cuda-test mm3-synth-smoke [LAYERS] [kv8] [moe=N]: the synthetic MiniMax-M3.1-Flash end to end on one GPU.
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("minimax_cuda");
const check = @import("check.zig");
const synth = @import("mm3_synth.zig");

const prompt_len = 3000;
const chunk_rows = 2048;
const steps = 32;
const window = 4; // the verify window re-runs the last `window` decode rows

/// One greedy run on a slot: the prompt's token, then every decode row's logits and the token it picked.
const Run = struct {
    tokens: [steps + 1]u32, // tokens[0] from the prompt, tokens[i + 1] from decode row i
    first: []const f32, // the prompt's last row
    rows: []f32, // decode row i at [i * vocab ..]
    prefill_ns: u64,
    step_ns: [steps]u64,

    fn deinit(r: *Run, gpa: std.mem.Allocator) void {
        gpa.free(r.first);
        gpa.free(r.rows);
    }

    fn row(r: *const Run, i: usize, v: usize) []const f32 {
        return r.rows[i * v ..][0..v];
    }
};

fn sequence(gpa: std.mem.Allocator, io: std.Io, m: *synth.Model, slot: u32, prompt: []const i32) !Run {
    const v: usize = m.geometry().own_vocab;
    var r: Run = .{ .tokens = undefined, .first = &.{}, .rows = try gpa.alloc(f32, steps * v), .prefill_ns = 0, .step_ns = undefined };
    errdefer r.deinit(gpa);
    const t0 = check.now(io);
    r.tokens[0] = try m.prefill(slot, prompt, chunk_rows);
    r.prefill_ns = @intCast(check.now(io) - t0);
    r.first = try gpa.dupe(f32, m.lastLogits());
    for (0..steps) |i| {
        const id: i32 = @intCast(r.tokens[i]);
        const s0 = check.now(io);
        const out = try m.decode(slot, @intCast(prompt.len + i), &[_]i32{id});
        defer gpa.free(out);
        r.tokens[i + 1] = synth.argmax(out);
        r.step_ns[i] = @intCast(check.now(io) - s0);
        @memcpy(r.rows[i * v ..][0..v], out);
    }
    return r;
}

fn allFinite(xs: []const f32) ?usize {
    for (xs, 0..) |x, i| if (!std.math.isFinite(x)) return i;
    return null;
}

fn bytesOf(xs: []const f32) []const u8 {
    return std.mem.sliceAsBytes(xs);
}

const Args = struct { layers: u32 = 1, kv8: bool = false, first_moe: ?u32 = null };

fn parseArgs(rest: []const [:0]const u8) !Args {
    var a: Args = .{};
    for (rest, 0..) |s, i| {
        if (std.mem.eql(u8, s, "kv8")) {
            a.kv8 = true;
        } else if (std.mem.startsWith(u8, s, "moe=")) {
            a.first_moe = try std.fmt.parseInt(u32, s[4..], 10);
        } else if (i == 0) {
            a.layers = try std.fmt.parseInt(u32, s, 10);
        } else return error.BadArgument;
    }
    return a;
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const args = try parseArgs(rest);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const b0 = check.now(io);
    const m = try synth.Model.init(gpa, gpu.d, stream, .{ .layers = args.layers, .slots = 2, .capacity_tokens = 4096, .kv8 = args.kv8, .first_moe = args.first_moe });
    defer m.deinit();
    const g = m.geometry();
    var moe: u32 = 0;
    for (0..g.layers) |l| moe += @intFromBool(g.isMoe(@intCast(l)));
    std.debug.print("INFO mm3-synth: {d} layers ({d} routed), kv8 {}, {d:.2} GiB weights, built in {d:.1} s\n", .{ g.layers, moe, args.kv8, @as(f64, @floatFromInt(m.weight_bytes)) / (1 << 30), ms(@intCast(check.now(io) - b0)) / 1e3 });

    var prng = std.Random.DefaultPrng.init(0x7072_6f6d);
    var prompt: [prompt_len]i32 = undefined;
    for (&prompt) |*p| p.* = @intCast(prng.random().uintLessThan(u32, m.cfg.vocab));
    const v: usize = g.own_vocab;
    var failed = false;

    var a = try sequence(gpa, io, m, 0, &prompt);
    defer a.deinit(gpa);
    const bad = allFinite(a.first) orelse allFinite(a.rows);
    if (bad) |i| {
        std.debug.print("FAIL mm3.synth.finite: logit {d} is {e}\n", .{ i, if (allFinite(a.first) != null) a.first[i] else a.rows[i] });
        failed = true;
    } else check.pass("mm3.synth.finite prompt row and {d} decode rows", .{steps});
    var sq: f64 = 0;
    var top: f32 = 0;
    for (a.rows) |x| {
        sq += @as(f64, x) * x;
        top = @max(top, @abs(x));
    }
    std.debug.print("INFO decode logits: rms {d:.3}, largest magnitude {d:.3}\n", .{ @sqrt(sq / @as(f64, @floatFromInt(a.rows.len))), top });

    var b = try sequence(gpa, io, m, 1, &prompt);
    defer b.deinit(gpa);
    const total = prompt_len + steps;
    const kv_a = try m.kvBytes(0, total);
    defer gpa.free(kv_a);
    if (slotsAgree(&a, &b) and kvAgree(m, kv_a, total)) check.pass("mm3.synth.slots: slot 1 (other pages) equals slot 0: tokens, logits and {d} KV bytes", .{kv_a.len}) else failed = true;

    if (try verify(gpa, m, &a, v, kv_a, total)) check.pass("mm3.synth.verify: a {d}-row window equals its decode rows, KV unchanged", .{window}) else failed = true;
    if (try sharedWindow(gpa, m, &a, v)) check.pass("mm3.synth.shared: two slots' windows in one forward equal their decode rows", .{}) else failed = true;

    var sorted: [steps]f64 = undefined;
    var sum: f64 = 0;
    for (a.step_ns, &sorted) |t, *s| {
        s.* = ms(t);
        sum += s.*;
    }
    const pms = ms(a.prefill_ns);
    std.debug.print("RESULT mm3-synth layers={d} routed={d} kv8={}: prefill {d} tokens in {d:.1} ms ({d:.0} tok/s), decode {d:.3} ms/step mean {d:.3} median ({d} steps, logits copy and host argmax included)\n", .{ g.layers, moe, args.kv8, prompt_len, pms, prompt_len / (pms / 1e3), sum / steps, check.median(&sorted), steps });
    std.debug.print("INFO tokens {any}\n", .{a.tokens});
    if (failed) return error.TestFailed;
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn slotsAgree(a: *const Run, b: *const Run) bool {
    if (!std.mem.eql(u32, &a.tokens, &b.tokens)) {
        std.debug.print("FAIL mm3.synth.slots tokens: {any} vs {any}\n", .{ a.tokens, b.tokens });
        return false;
    }
    check.sameBytes("mm3.synth.slots prompt row", bytesOf(b.first), bytesOf(a.first)) catch return false;
    check.sameBytes("mm3.synth.slots decode rows", bytesOf(b.rows), bytesOf(a.rows)) catch return false;
    return true;
}

fn kvAgree(m: *synth.Model, kv_a: []const u8, total: usize) bool {
    const kv_b = m.kvBytes(1, total) catch return false;
    defer m.gpa.free(kv_b);
    check.sameBytes("mm3.synth.slots kv", kv_b, kv_a) catch return false;
    return true;
}

/// Decode rows steps - window .. steps - 1 again as one window on slot 0: the same logits and the same KV bytes.
fn verify(gpa: std.mem.Allocator, m: *synth.Model, a: *const Run, v: usize, kv_a: []const u8, total: usize) !bool {
    const first = steps - window;
    var ids: [window]i32 = undefined;
    for (&ids, a.tokens[first..steps]) |*d, t| d.* = @intCast(t);
    const got = try m.decode(0, prompt_len + first, &ids);
    defer gpa.free(got);
    var ok = true;
    for (0..window) |r| {
        var id_buf: [64]u8 = undefined;
        const id = std.fmt.bufPrint(&id_buf, "mm3.synth.verify row {d}", .{r}) catch unreachable;
        check.sameBytes(id, bytesOf(got[r * v ..][0..v]), bytesOf(a.row(first + r, v))) catch {
            ok = false;
        };
    }
    const kv = try m.kvBytes(0, total);
    defer gpa.free(kv);
    check.sameBytes("mm3.synth.verify kv", kv, kv_a) catch {
        ok = false;
    };
    return ok;
}

/// Slot 0 re-runs decode rows 28, 29 and slot 1 rows 30, 31 in one shared forward.
fn sharedWindow(gpa: std.mem.Allocator, m: *synth.Model, a: *const Run, v: usize) !bool {
    const first = steps - window;
    const segs = [_]mm.iface.Seg{ .{ .slot = 0, .row0 = 0, .rows = 2 }, .{ .slot = 1, .row0 = 2, .rows = 2 } };
    var ids: [window]i32 = undefined;
    var pos: [window]i32 = undefined;
    for (0..window) |r| {
        ids[r] = @intCast(a.tokens[first + r]);
        pos[r] = @intCast(prompt_len + first + r);
    }
    const got = try m.shared(&segs, &ids, &pos);
    defer gpa.free(got);
    var ok = true;
    for (0..window) |r| {
        var id_buf: [64]u8 = undefined;
        const id = std.fmt.bufPrint(&id_buf, "mm3.synth.shared row {d}", .{r}) catch unreachable;
        check.sameBytes(id, bytesOf(got[r * v ..][0..v]), bytesOf(a.row(first + r, v))) catch {
            ok = false;
        };
    }
    return ok;
}
