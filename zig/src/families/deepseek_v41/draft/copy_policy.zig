//! Opt-in suffix-copy suppression beside a loaded DSpark head; target verification is unchanged.
const std = @import("std");
const lanes = @import("lanes");
const lookup_mod = @import("lookup.zig");

pub const Settings = struct {
    no_copies: bool = false,

    pub fn fromEnv(get: *const fn ([*:0]const u8) ?[]const u8) !Settings {
        const raw = get("TF_DSV41_NO_COPIES") orelse return .{};
        const value = std.mem.trim(u8, raw, " \t\r\n");
        for ([_][]const u8{ "1", "true", "on" }) |v| {
            if (std.ascii.eqlIgnoreCase(value, v)) return .{ .no_copies = true };
        }
        for ([_][]const u8{ "", "0", "false", "off" }) |v| {
            if (std.ascii.eqlIgnoreCase(value, v)) return .{};
        }
        return error.BadNoCopies;
    }

    pub fn suppressed(s: Settings, head_loaded: bool) bool {
        return s.no_copies and head_loaded;
    }

    /// Close the core's existing copy gate without shrinking verified head or shared-forward capacity.
    pub fn apply(s: Settings, head_loaded: bool, model: *lanes.Model) void {
        if (s.suppressed(head_loaded)) model.copy_drafts = false;
    }

    /// Avoid retaining and indexing history for a proposer the core will never ask.
    pub fn lookup(s: Settings, head_loaded: bool, settings: lookup_mod.Settings) lookup_mod.Settings {
        var out = settings;
        if (s.suppressed(head_loaded)) out.lookup = false;
        return out;
    }
};

test "copy suppression is explicit, defaults off, and rejects invalid switches" {
    const Env = struct {
        var raw: ?[]const u8 = null;
        fn get(_: [*:0]const u8) ?[]const u8 {
            return raw;
        }
    };
    try std.testing.expect(!(try Settings.fromEnv(&Env.get)).no_copies);
    for ([_][]const u8{ "1", " TRUE ", "on" }) |v| {
        Env.raw = v;
        try std.testing.expect((try Settings.fromEnv(&Env.get)).no_copies);
    }
    for ([_][]const u8{ "", "0", "false", "OFF" }) |v| {
        Env.raw = v;
        try std.testing.expect(!(try Settings.fromEnv(&Env.get)).no_copies);
    }
    Env.raw = "maybe";
    try std.testing.expectError(error.BadNoCopies, Settings.fromEnv(&Env.get));
}

test "copy suppression preserves head widths and leaves copies available without a loaded head" {
    for ([_]bool{ false, true }) |on| for ([_]bool{ false, true }) |loaded| {
        const s: Settings = .{ .no_copies = on };
        var model: lanes.Model = .{ .exact_width = 16, .drafts = 5, .mtp = true, .speculate = true };
        s.apply(loaded, &model);
        var cfg = try lanes.Config.init(std.testing.allocator, model, 16, 15);
        defer cfg.deinit(std.testing.allocator);
        try std.testing.expectEqual(@as(u32, if (on and loaded) 0 else 15), cfg.max_copy);
        try std.testing.expectEqual(@as(u32, 5), cfg.most_drafts);
        try std.testing.expectEqual(@as(u32, 16), cfg.family_width);
        try std.testing.expectEqual(!(on and loaded), s.lookup(loaded, .{}).lookup);
        try std.testing.expect(!s.lookup(loaded, .{ .lookup = false }).lookup);
    };
}

test "copy suppression closes the real proposal gate for both proposers and avoids priced history storage" {
    const a = std.testing.allocator;
    const span = [_]u32{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 };
    const prompt = span ++ span;
    var verify: [16]f64 = @splat(20);
    const costs: @import("costs.zig").Costs = .{ .verify = &verify, .draft = 5 };
    for ([_]bool{ false, true }) |on| for ([_]bool{ false, true }) |loaded| {
        const settings: Settings = .{ .no_copies = on };
        var model: lanes.Model = .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 5 };
        settings.apply(loaded, &model);
        var cfg = try lanes.Config.init(a, model, 16, 15);
        defer cfg.deinit(a);
        const planner = lookup_mod.Planner.init(settings.lookup(loaded, .{ .plan = true }), &costs, loaded);
        var priced = try lookup_mod.Proposer.init(&planner, a, &prompt, &.{});
        defer priced.deinit();
        try std.testing.expectEqual(@as(usize, if (on and loaded) 0 else prompt.len), priced.req.index.tokens.items.len);
        var suffix = try lanes.SuffixLookup.init(a, .{ .min_match = 4 });
        defer suffix.deinit();
        var target: lanes.fake.Fake = .{ .gpa = a };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var engine = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        engine.alone = true;
        for ([_]lanes.proposer.Proposer{ priced.proposer(), suffix.proposer() }) |proposer| {
            var stream = try lanes.Stream.init(a, .{ .id = "copy", .prompt = &prompt, .max_new = 32, .proposer = proposer });
            defer stream.deinit(a);
            const copied = try lanes.windows.copyProposal(&engine, &stream);
            try std.testing.expectEqual(!(on and loaded), copied.tokens.len > 0);
        }
    };
}
