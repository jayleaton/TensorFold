//! Shared grammar cuts and masks through the actual family lane backend, without a CUDA driver.
const std = @import("std");
const lanes = @import("lanes");
const ln = @import("draft/lanes.zig");
const iface = @import("draft/iface.zig");
const mask_rows = @import("grammar_rows.zig");
const grammar_gpu = @import("grammar_gpu.zig");
const a = std.testing.allocator;
const all_ones = [2]u32{ std.math.maxInt(u32), std.math.maxInt(u32) };

const Target = struct {
    prompt: [2]u32 = @splat(0),
    masked_rows: usize = 0,
    seen_drafts: usize = 0,
    masked: bool = true, // false: a request without a grammar (no prompt or window masks)
    staged: u32 = 0, // rows Masks.stage keeps, as grammar_gpu.segments hands them over
    stages: usize = 0, // forwards whose masks Masks.stage would copy, upload and send to the followers

    fn self(p: *anyopaque) *Target {
        return @ptrCast(@alignCast(p));
    }
    fn promptMask(p: *anyopaque, _: u32, mask: []const u32) anyerror!void {
        if (mask.len == 0 and !self(p).masked) {
            self(p).prompt = all_ones;
            return;
        }
        if (mask.len != 2) return error.MissingPromptMask;
        @memcpy(&self(p).prompt, mask);
    }
    fn choose(bits: []const u32) !u32 {
        var logits: [64]f32 = @splat(1);
        logits[0] = 100;
        mask_rows.maskRows(&logits, 64, &.{0}, bits, 2, 0);
        const id = @import("greedy.zig").argmax(&logits);
        if (!std.math.isFinite(logits[id])) return error.GrammarDeadEnd;
        return @intCast(id);
    }
    fn prefill(p: *anyopaque, _: u32, _: []const u32, _: ?lanes.Sampling, _: u64) anyerror!u32 {
        return choose(&self(p).prompt);
    }
    fn window(p: *anyopaque, segs: []const iface.Segment, out: [][]u32) anyerror!void {
        const t = self(p);
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        var kept: [64]mask_rows.Row = undefined;
        if (mask_rows.constraining(try grammar_gpu.segmentRows(arena.allocator(), segs, 2), 64, t.staged, &kept)) |list| {
            t.staged = @intCast(list.len);
            t.stages += 1;
        }
        for (segs, out) |s, choices| {
            if (s.masks.len != s.tokens.len * 2 and (t.masked or s.masks.len != 0)) return error.MissingWindowMasks;
            self(p).seen_drafts += s.tokens.len - 1;
            for (choices, 0..) |*choice, i| {
                choice.* = try choose(if (s.masks.len == 0) &all_ones else s.masks[i * 2 ..][0..2]);
                self(p).masked_rows += 1;
            }
        }
    }
    fn keep(_: *anyopaque, _: u32, _: []const u32) anyerror!void {}
    fn taps(_: *anyopaque, _: u32) iface.Taps {
        return .{ .address = 0, .rows = 1, .stride = 1 };
    }
    fn release(_: *anyopaque, _: u32) void {}
    fn target(t: *Target) iface.Target {
        return .{ .ptr = t, .vtable = &.{ .prompt_mask = promptMask, .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release } };
    }
};

/// One request through the family lane backend over `target`; its emitted tokens (the caller frees them).
fn serve(target: *Target, drafts: bool, grammar: ?*lanes.grammar.Constraint) ![]u32 {
    var pass: @import("draft/drive.zig").Oracle = .{ .serial = &.{ 5, 6, 7, 31 }, .prompt_len = 2, .vocab = 64 };
    var prices = try @import("draft/costs.zig").defaults(a, 64);
    defer prices.deinit(a);
    const x = try ln.Lanes.init(a, target.target(), pass.pass(), prices, .{
        .shape = .{ .hidden = 1, .block = 3, .candidates = 1, .window = 1 },
        .slots = 1,
        .depth = .{ .mode = .static, .cap = 3, .skip = false },
    });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var cfg = try lanes.Config.init(a, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    var stream = try lanes.Stream.init(a, .{ .id = "grammar", .prompt = &.{ 1, 2 }, .max_new = 8, .drafts = drafts, .sampling = if (drafts) .{ .seed = 11, .temperature = 0.8, .top_k = 8, .top_p = 0.9 } else null, .eos = &.{31}, .grammar = grammar });
    defer stream.deinit(a);
    try e.addStream(&stream);
    while (e.activeCount() > 0) try e.step();
    try std.testing.expectEqualStrings("", stream.problem);
    return a.dupe(u32, stream.emitted());
}

test "grammar family: first token and drafted prefix masks reach the target" {
    for ([_]bool{ false, true }) |drafts| {
        var target: Target = .{};
        var sequence: lanes.grammar.Sequence = .{ .want = &.{ 5, 6, 7 }, .stop = 31, .gpa = a };
        var grammar = lanes.grammar.Constraint.init(sequence.rules(), 2, null);
        defer grammar.rules.vtable.free(grammar.rules.ptr);
        const got = try serve(&target, drafts, &grammar);
        defer a.free(got);
        try std.testing.expectEqualSlices(u32, &.{ 5, 6, 7, 31 }, got);
        try std.testing.expect(target.masked_rows > 0);
        try std.testing.expect(lanes.grammar.allows(&target.prompt, 5));
        // a constraining grammar (required, named, structured) still stages its masks for every rank
        try std.testing.expect(target.stages > 0);
    }
}

test "grammar family: a request with no active mask stages and sends nothing" {
    for ([_]bool{ false, true }) |drafts| {
        // no grammar (plain or unconstrained auto tools): no window or prompt masks, nothing staged or sent a step
        var plain: Target = .{ .masked = false };
        a.free(try serve(&plain, drafts, null));
        try std.testing.expect(plain.masked_rows > 0);
        try std.testing.expectEqual(@as(usize, 0), plain.stages);
        // a bound grammar whose start token (here a think close, 30) never comes: no row is ever constrained
        var sequence: lanes.grammar.Sequence = .{ .want = &.{ 5, 6, 7 }, .stop = 31, .gpa = a };
        var grammar = lanes.grammar.Constraint.init(sequence.rules(), 2, 30);
        defer grammar.rules.vtable.free(grammar.rules.ptr);
        var idle: Target = .{ .masked = false };
        a.free(try serve(&idle, drafts, &grammar));
        try std.testing.expectEqual(@as(usize, 0), idle.stages);
    }
}

test "grammar family: rows allowing every token are not staged; a cleared stage still reaches the followers" {
    const ones = [2]u32{ std.math.maxInt(u32), std.math.maxInt(u32) };
    const some = [2]u32{ std.math.maxInt(u32), 0x7fff_ffff }; // token 63 disallowed
    var out: [4]mask_rows.Row = undefined;
    try std.testing.expect(mask_rows.free(&ones, 64));
    try std.testing.expect(!mask_rows.free(&some, 64));
    // the padding columns past the vocabulary do not count: only tokens below it reach a logit
    try std.testing.expect(mask_rows.free(&some, 63));
    try std.testing.expect(mask_rows.free(&.{ std.math.maxInt(u32), 0xff }, 40));
    try std.testing.expect(!mask_rows.free(&.{ std.math.maxInt(u32), 0x7f }, 40));
    try std.testing.expect(!mask_rows.free(&.{ 0xffff_fffe, std.math.maxInt(u32) }, 64));
    const free_rows = [_]mask_rows.Row{ .{ .index = 0, .bits = &ones }, .{ .index = null, .bits = &ones } };
    try std.testing.expect(mask_rows.constraining(&.{}, 64, 0, &out) == null);
    try std.testing.expect(mask_rows.constraining(&free_rows, 64, 0, &out) == null);
    const mixed = [_]mask_rows.Row{ .{ .index = 0, .bits = &ones }, .{ .index = 1, .bits = &some }, .{ .index = 2, .bits = &ones }, .{ .index = null, .bits = &some } };
    const kept = mask_rows.constraining(&mixed, 64, 0, &out).?;
    try std.testing.expectEqual(@as(usize, 2), kept.len);
    try std.testing.expectEqual(@as(?u32, 1), kept[0].index);
    try std.testing.expectEqual(@as(?u32, null), kept[1].index);
    // masks staged before: an empty stage clears them on every rank (sent), never skipped
    try std.testing.expectEqual(@as(usize, 0), mask_rows.constraining(&free_rows, 64, 2, &out).?.len);
    try std.testing.expectEqual(@as(usize, 0), mask_rows.constraining(&.{}, 64, 1, &out).?.len);
}

test "grammar family: TF_DSV41_GRAMMAR=0 refuses structured output; unset, empty or 1 serve it" {
    const env = @import("sys").env;
    defer _ = env.unsetenv("TF_DSV41_GRAMMAR");
    _ = env.unsetenv("TF_DSV41_GRAMMAR");
    try std.testing.expect(try grammar_gpu.enabled());
    const Case = struct { v: [:0]const u8, on: anyerror!bool };
    for ([_]Case{ .{ .v = "0", .on = false }, .{ .v = "", .on = true }, .{ .v = "1", .on = true }, .{ .v = "yes", .on = error.BadGrammarKnob } }) |c| {
        _ = env.setenv("TF_DSV41_GRAMMAR", c.v, 1);
        if (c.on) |on| try std.testing.expectEqual(on, try grammar_gpu.enabled()) else |e| try std.testing.expectError(e, grammar_gpu.enabled());
    }
}

test "grammar family: rejected draft prefix rolls back and rejects empty or malformed masks" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var sequence: lanes.grammar.Sequence = .{ .want = &.{ 5, 6, 7 }, .stop = 31, .gpa = a };
    var grammar = lanes.grammar.Constraint.init(sequence.rules(), 2, null);
    defer grammar.rules.vtable.free(grammar.rules.ptr);
    try std.testing.expect(try grammar.follow(5));
    const cut = try grammar.window(arena.allocator(), &.{ 5, 6, 42, 7 }, &.{ -1, 0, 1, 2 }, false, 2);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, cut.rows);
    try std.testing.expectEqual(@as(usize, 1), sequence.at);
    try std.testing.expect(lanes.grammar.allows(cut.masks[0..2], 6));
    try std.testing.expect(lanes.grammar.allows(cut.masks[2..4], 7));
    try mask_rows.validate(&.{.{ .index = 0, .bits = cut.masks[0..2] }}, 2, 64);
    try std.testing.expectError(error.GrammarDeadEnd, mask_rows.validate(&.{.{ .index = 0, .bits = &.{ 0, 0 } }}, 2, 64));
    try std.testing.expectError(error.GrammarWords, mask_rows.validate(&.{.{ .index = 0, .bits = &.{1} }}, 2, 64));
}

test "grammar family: masks use global tokens across uneven rank spans and retain allowed NaN" {
    const words = 40;
    var bits: [words]u32 = @splat(0);
    bits[12] = 1;
    for (0..4) |rank| {
        const part = try @import("partition.zig").share(1280, rank, 4);
        const logits = try a.alloc(f32, part.len * 2);
        defer a.free(logits);
        @memset(logits, std.math.nan(f32));
        mask_rows.maskRows(logits, part.len, &.{1}, &bits, words, part.lo);
        for (logits[0..part.len]) |v| try std.testing.expect(std.math.isNan(v));
        for (logits[part.len..], 0..) |v, col| {
            if (part.lo + col == 384) try std.testing.expect(std.math.isNan(v)) else try std.testing.expectEqual(-std.math.inf(f32), v);
        }
    }
}

test "grammar family: thinking prefix stays free until the close token, then draft masks start" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var sequence: lanes.grammar.Sequence = .{ .want = &.{ 5, 6 }, .stop = 31, .gpa = a };
    var grammar = lanes.grammar.Constraint.init(sequence.rules(), 2, 30);
    defer grammar.rules.vtable.free(grammar.rules.ptr);
    try std.testing.expect(try grammar.next(arena.allocator()) == null);
    const window = try grammar.window(arena.allocator(), &.{ 9, 10, 30, 5, 42 }, &.{ -1, 0, 1, 2, 3 }, false, 4);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, window.rows);
    try std.testing.expect(std.mem.allEqual(u32, window.masks[0..4], std.math.maxInt(u32)));
    try std.testing.expect(lanes.grammar.allows(window.masks[4..6], 5));
    try std.testing.expect(lanes.grammar.allows(window.masks[6..8], 6));
    try std.testing.expectEqual(@as(usize, 0), sequence.at);
    try std.testing.expect(!grammar.active);
    try std.testing.expect(try grammar.follow(30));
    const next = (try grammar.next(arena.allocator())).?;
    try std.testing.expect(lanes.grammar.allows(next, 5));
}
