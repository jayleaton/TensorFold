//! The served engine without HTTP (tf-dsv41-m1):
//!
//!   follow PACK ASSETS          ranks > 0: load this rank (model.zig), replay rank 0's forward operations until it
//!                               stops (the server's close or generate's end)
//!   generate PACK ASSETS TRACE  rank 0: every line of TRACE (TF_DSV41_TRACE_TOKENS' file from tensorfold-dsv41) replayed
//!                               through the same engine (serve_engine.Served: lanes over the GPU target) as one
//!                               greedy request of the recorded prompt and length; PASS when every reply's ids equal
//!                               the recorded ones (M4's "HTTP tokens == CLI")
//!
//! TF_DSV41_DRAFTS=0: no DSpark (serial windows through the same lanes), as the server's --no-drafts.

const std = @import("std");
const api = @import("engine_api");
const model = @import("model.zig");
const se = @import("serve_engine.zig");

fn drafts() bool {
    const v = std.c.getenv("TF_DSV41_DRAFTS") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

pub fn follow(gpa: std.mem.Allocator, io: std.Io, pack: []const u8, assets: []const u8) !u8 {
    const m = try model.Model.open(gpa, io, pack, assets, .{ .drafts = drafts(), .own_prefill = model.ownPrefillEnv(), .layers = try model.layersEnv(gpa) });
    defer m.close();
    if (m.rank() == 0) return error.RankZeroServes;
    std.debug.print("rank {d}: loaded, following rank 0\n", .{m.rank()});
    try m.f.follow();
    std.debug.print("rank {d}: rank 0 stopped\n", .{m.rank()});
    return 0;
}

const Box = struct {
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?api.Reason = null,
    drafted: u64 = 0,
    accepted: u64 = 0,
    gpa: std.mem.Allocator,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(b.gpa, t) catch {},
            .finished => |f| {
                b.drafted = f.stats.drafted;
                b.accepted = f.stats.accepted;
                b.done = f.reason;
            },
            else => {},
        }
    }

    fn wait(b: *Box) api.Reason {
        while (true) {
            b.mutex.lockUncancelable(b.io);
            const d = b.done;
            b.mutex.unlock(b.io);
            if (d) |r| return r;
            std.Io.sleep(b.io, .fromMilliseconds(2), .awake) catch {};
        }
    }
};

pub fn generate(gpa: std.mem.Allocator, io: std.Io, pack: []const u8, assets: []const u8, trace: []const u8) !u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, trace, a, .limited(1 << 30));
    const s = try se.Served.open(gpa, io, pack, assets, drafts(), null);
    defer se.Served.close(s);
    const eng = s.engine();
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var n: usize = 0;
    var equal: usize = 0;
    var id: api.Id = 1;
    var drafted: u64 = 0;
    var accepted: u64 = 0;
    while (lines.next()) |line| : (id += 1) {
        const Line = struct { prompt: []const u32, tokens: []const u32, drafted: ?u64 = null };
        const rec = try std.json.parseFromSliceLeaky(Line, a, line, .{ .ignore_unknown_fields = true });
        var box: Box = .{ .io = io, .gpa = gpa };
        defer box.tokens.deinit(gpa);
        const req: api.Request = .{ .prompt = rec.prompt, .max_tokens = @intCast(rec.tokens.len), .drafts = drafts() };
        try eng.submit(id, &req, .{ .ctx = &box, .event = Box.event });
        const why = box.wait();
        const same = std.mem.eql(u32, box.tokens.items, rec.tokens);
        // the first token index where the replay leaves the recorded reply (-1: none), and both draft counts
        const first: i64 = if (same) -1 else @intCast(std.mem.indexOfDiff(u32, box.tokens.items, rec.tokens) orelse @min(box.tokens.items.len, rec.tokens.len));
        n += 1;
        equal += @intFromBool(same);
        drafted += box.drafted;
        accepted += box.accepted;
        std.debug.print("{{\"line\": {d}, \"prompt\": {d}, \"tokens\": {d}, \"recorded\": {d}, \"finished\": \"{t}\", \"equal\": {}, \"first_differing\": {d}, \"drafted\": {d}, \"recorded_drafted\": {?d}, \"accepted\": {d}}}\n", .{ n, rec.prompt.len, box.tokens.items.len, rec.tokens.len, why, same, first, box.drafted, rec.drafted, box.accepted });
    }
    const ok = n > 0 and equal == n;
    std.debug.print("{s} M4 CLI == HTTP: {d}/{d} replies equal\n", .{ if (ok) "PASS" else "FAIL", equal, n });
    // DSpark's acceptance on these replies (drafts on): accepted / drafted, and tokens a verify round
    const rate: f64 = if (drafted > 0) @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(drafted)) else 0;
    std.debug.print("DSpark: {d} drafted, {d} accepted ({d:.3})\n", .{ drafted, accepted, rate });
    return if (ok) 0 else 1;
}
