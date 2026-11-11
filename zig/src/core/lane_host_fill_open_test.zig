//! lane_host_fill's Share parsing and its open-chunk rule (Backend.open_fills).
const std = @import("std");
const fill = @import("lane_host_fill.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Share = fill.Share;

test "a share of 0.5 owes the decoders a step's time; 1 owes nothing; rounds pay it down" {
    var s: Share = .{ .p = 0.5 };
    s.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 1000), s.debt_ns);
    s.stepped(1000, false);
    try std.testing.expectEqual(@as(i64, 1000), s.debt_ns);
    s.rounded(400);
    s.rounded(800);
    try std.testing.expectEqual(@as(i64, 0), s.debt_ns);
    var one: Share = .{};
    one.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 0), one.debt_ns);
    var q: Share = .{ .p = 0.25 };
    q.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 3000), q.debt_ns);
}

test "a share below the floor, NaN and inf fall back to 1; the floor and 1 stand" {
    for ([_][]const u8{ "1e-12", "0.0099", "0", "-0.5", "nan", "inf", "-inf", "2", "x", "" }) |text| {
        try std.testing.expectEqual(@as(f64, 1), Share.fromText(text).p);
    }
    try std.testing.expectEqual(@as(f64, 0.01), Share.fromText("0.01").p);
    try std.testing.expectEqual(@as(f64, 1), Share.fromText("1").p);
}

test "prefill service: TENSORFOLD_PREFILL_BUDGET is whole rows; unset, 0 or anything else is none, main's order" {
    for ([_][]const u8{ "", "0", "x", "-512", "+512", "5e2", "4294967296" }) |text| {
        try std.testing.expectEqual(@as(u64, 0), Share.budget(text));
    }
    try std.testing.expectEqual(@as(u64, 512), Share.budget("512"));
    try std.testing.expectEqual(@as(u64, 1), Share.budget(" 1\t"));
    try std.testing.expectEqual(@as(u64, 0), (Share{}).rows);
}

test "a step's debt saturates instead of overflowing" {
    var s: Share = .{ .p = Share.min_share };
    s.stepped(std.math.maxInt(i64), true);
    try std.testing.expect(s.debt_ns > 0);
    s.stepped(std.math.maxInt(i64), true);
    try std.testing.expectEqual(@as(i64, std.math.maxInt(i64)), s.debt_ns);
    s.rounded(1000);
    try std.testing.expect(s.debt_ns < std.math.maxInt(i64));
}

test "prefill service: policy text and rows; unset or unknown is the default order, TOKENS clamped" {
    for ([_][]const u8{ "", "stock", "FCFS", "lifo", "batched", "x" }) |text| try std.testing.expectEqual(Share.Policy.stock, Share.policyOf(text));
    try std.testing.expectEqual(Share.Policy.fcfs, Share.policyOf("fcfs"));
    try std.testing.expectEqual(Share.Policy.fcfs, Share.policyOf(" fcfs\t"));
    try std.testing.expectEqual(Share.Policy.stock, (Share{}).policy);
    try std.testing.expectEqual(@as(u32, 2048), (Share{}).tokens);
    for ([_][]const u8{ "", "0", "x", "-1", "4294967296" }) |text| try std.testing.expectEqual(@as(u32, 2048), Share.tokensOf(text));
    for ([_]u32{ 2048, 4096, 8192, 16384 }) |n| { // the default and the measured options pass through unchanged
        var buf: [8]u8 = undefined;
        try std.testing.expectEqual(n, Share.tokensOf(try std.fmt.bufPrint(&buf, "{d}", .{n})));
    }
    try std.testing.expectEqual(@as(u32, 64), Share.tokensOf("1"));
    try std.testing.expectEqual(@as(u32, 64), Share.tokensOf("63"));
    try std.testing.expectEqual(@as(u32, 1000), Share.tokensOf("1000"));
    try std.testing.expectEqual(@as(u32, 8193), Share.tokensOf("8193"));
    try std.testing.expectEqual(@as(u32, 16384), Share.tokensOf("16384"));
    try std.testing.expectEqual(@as(u32, 16384), Share.tokensOf("16385"));
    try std.testing.expectEqual(@as(u32, 16384), Share.tokensOf("65536"));
}

const OpenFills = struct {
    fn event(_: *anyopaque, _: @import("engine_api.zig").Id, _: *const @import("engine_api.zig").Event) void {}
};

/// `n` short prompts admitted at once to a stepped fake that holds `open` chunks (0: any), filled to the end.
fn fillShort(n: usize, open: u32) !struct { busy: usize, live: usize, steps: usize } {
    const lanes = @import("lanes");
    const api = @import("engine_api.zig");
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    // seven calls a prompt, as flashnext_mia's 8-layer slices of a short prompt
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 7, .open_fills = open };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4 });
    defer {
        @import("lane_host_end.zig").closeAll(&host);
        host.stop();
    }
    const prompt: [46]u32 = @splat(3);
    var requests: [4]api.Request = undefined;
    for (0..n) |i| {
        requests[i] = .{ .prompt = &prompt, .max_tokens = 64, .drafts = false };
        try host.engine().submit(@intCast(10 + i), &requests[i], .{ .ctx = &host, .event = OpenFills.event });
        try std.testing.expect(host.admitOne());
    }
    var visits: usize = 0;
    while (host.filling.items.len > 0 and visits < 64) : (visits += 1) fill.run(&host);
    return .{ .busy = target.busy, .live = core.live.items.len, .steps = target.prefill_count };
}

test "prefill service: one open chunk at a time; aged prompts wait for it to end (FILL=1 PromptBuffersBusy)" {
    // the lane trace's c4 wave: the first prompt's seven calls aged the others, and an aged one began mid-chunk
    const one = try fillShort(4, 1);
    try std.testing.expectEqual(@as(usize, 0), one.busy);
    try std.testing.expectEqual(@as(usize, 4), one.live);
    try std.testing.expectEqual(@as(usize, 4 * 7), one.steps);
    // no limit: the picks may switch prompts mid-chunk, as such a backend allows
    const any = try fillShort(4, 0);
    try std.testing.expectEqual(@as(usize, 0), any.busy);
    try std.testing.expectEqual(@as(usize, 4), any.live);
}
