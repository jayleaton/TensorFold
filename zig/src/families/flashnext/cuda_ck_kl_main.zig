//! kl: one precision's teacher-forced prompt rows on disk, then another's against them (KL, top-1, NLL, margins).
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");

const prompt = flash.prompt;
const vocab = prompt.vocabulary;

/// Prompt rows a pass takes at once: past ck.prompt_tokens, so both precisions run their prompt kernels.
const rows_at_once = 256;

fn score(raw: u16) f64 {
    return @as(f32, @bitCast(@as(u32, raw) << 16));
}

const Row = struct { best: usize, second: f64, lse: f64 };

fn stats(r: []const u16) Row {
    var best: usize = 0;
    var top = -std.math.inf(f64);
    for (r, 0..) |v, i| if (score(v) > top) {
        top = score(v);
        best = i;
    };
    var second = -std.math.inf(f64);
    var sum: f64 = 0;
    for (r, 0..) |v, i| {
        sum += @exp(score(v) - top);
        if (i != best) second = @max(second, score(v));
    }
    return .{ .best = best, .second = second, .lse = top + @log(sum) };
}

const Tally = struct {
    rows: usize = 0,
    agree: usize = 0,
    kl: f64 = 0,
    kl_max: f64 = 0,
    nll_ref: f64 = 0,
    nll_run: f64 = 0,
    scored: usize = 0,
    flip_margin: f64 = 0, // the largest reference top-2 margin among rows whose top-1 differs

    fn row(t: *Tally, ref: []const u16, run: []const u16, next: ?u32) void {
        const a = stats(ref);
        const b = stats(run);
        var kl: f64 = 0;
        for (ref, run) |ra, rb| {
            const la = score(ra) - a.lse;
            kl += @exp(la) * (la - (score(rb) - b.lse));
        }
        t.rows += 1;
        t.kl += kl;
        t.kl_max = @max(t.kl_max, kl);
        if (a.best == b.best) t.agree += 1 else t.flip_margin = @max(t.flip_margin, score(ref[a.best]) - a.second);
        if (next) |n| {
            t.nll_ref += a.lse - score(ref[n]);
            t.nll_run += b.lse - score(run[n]);
            t.scored += 1;
        }
    }

    fn report(t: *const Tally, what: []const u8) void {
        if (t.rows == 0) return;
        const n: f64 = @floatFromInt(t.rows);
        const s: f64 = @floatFromInt(@max(t.scored, 1));
        std.debug.print("{s}: {d} rows, top-1 {d}/{d} ({d:.2}%), KL(ref||run) mean {e:.4} max {e:.3}, ppl ref {d:.4} run {d:.4} ({d:.2}%), largest flipped margin {d:.4}\n", .{ what, t.rows, t.agree, t.rows, 100 * @as(f64, @floatFromInt(t.agree)) / n, t.kl / n, t.kl_max, @exp(t.nll_ref / s), @exp(t.nll_run / s), 100 * (@exp((t.nll_run - t.nll_ref) / s) - 1), t.flip_margin });
    }
};

fn readIds(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, "[], \t\r\n");
    while (it.next()) |part| try ids.append(gpa, try std.fmt.parseInt(u32, part, 10));
    return ids.toOwnedSlice(gpa);
}

pub fn main(init: std.process.Init) !u8 {
    try flash.env_validate.startup();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 6 or !(std.mem.eql(u8, args[3], "dump") or std.mem.eql(u8, args[3], "against"))) {
        std.debug.print("usage: flashnext-kl <model-dir> <checkpoint|full> <dump|against> <file> <ids-file>...\n", .{});
        return 2;
    }
    const checkpoint = std.mem.eql(u8, args[2], "checkpoint");
    const dump = std.mem.eql(u8, args[3], "dump");
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var glue = try flash.Glue.load(&driver);
    defer glue.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const own: flash.Tri = .{ .own = &glue, .s = stream };
    var e = try prompt.Engine.initRank(gpa, io, &driver, &stream, args[1], 1, 0, checkpoint);
    defer e.deinit();
    var out = try cuda.DeviceBuffer.alloc(&driver, rows_at_once * vocab * 2);
    defer out.free();
    const host = try gpa.alloc(u16, rows_at_once * vocab);
    defer gpa.free(host);
    const ref = try gpa.alloc(u16, vocab);
    defer gpa.free(ref);
    var file = if (dump) try std.Io.Dir.cwd().createFile(io, args[4], .{}) else try std.Io.Dir.cwd().openFile(io, args[4], .{});
    defer file.close(io);
    var offset: u64 = 0;
    var all: Tally = .{};
    for (args[5..]) |path| {
        const ids = try readIds(gpa, io, path);
        defer gpa.free(ids);
        var seq = try e.newSeq(ids.len + 1);
        defer seq.deinit(gpa);
        var t: Tally = .{};
        var at: usize = 0;
        while (at < ids.len) {
            const m = @min(rows_at_once, ids.len - at);
            try e.forward(flash.Tri, own, &seq, ids[at..][0..m], .{ .logits = .all, .logits_out = out.ptr });
            try stream.synchronize();
            try out.download(0, std.mem.sliceAsBytes(host[0 .. m * vocab]));
            for (0..m) |r| {
                const row = host[r * vocab ..][0..vocab];
                if (dump) {
                    try file.writePositionalAll(io, std.mem.sliceAsBytes(row), offset);
                } else {
                    if (try file.readPositionalAll(io, std.mem.sliceAsBytes(ref), offset) != vocab * 2) return error.EndOfStream;
                    const next: ?u32 = if (at + r + 1 < ids.len) ids[at + r + 1] else null;
                    t.row(ref, row, next);
                    all.row(ref, row, next);
                }
                offset += vocab * 2;
            }
            at += m;
        }
        if (!dump) t.report(std.fs.path.basename(path));
    }
    if (!dump) all.report("all");
    return 0;
}
