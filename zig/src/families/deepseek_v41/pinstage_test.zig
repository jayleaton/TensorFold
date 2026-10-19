//! TF_DSV41_PF_PINSTAGE's host rules (`zig build test-dsv41-knobs`): the knob is 1 alone, and the rows ready() stages.
const std = @import("std");
const testing = std.testing;
const pk = @import("prod_knobs.zig");

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "pin stage: the knob, and the staged rows (off: the forward's, as before)" {
    Fake.pairs = &.{};
    try testing.expect(!pk.pfPinStage(&Fake.get));
    for ([_][]const u8{ "0", "2", "on", "" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_PF_PINSTAGE", v }};
        try testing.expect(!pk.pfPinStage(&Fake.get));
    }
    Fake.pairs = &.{.{ "TF_DSV41_PF_PINSTAGE", " 1" }};
    try testing.expect(pk.pfPinStage(&Fake.get));
    Fake.pairs = &.{};
    try testing.expectEqual(@as(usize, 2048), pk.stageRows(2048, 4096, false));
    try testing.expectEqual(@as(usize, 4096), pk.stageRows(2048, 4096, true));
    try testing.expectEqual(@as(usize, 8192), pk.stageRows(2048, 8192, true));
    // no workspace (TF_DSV41_PF_4K unset): the forward's rows either way
    try testing.expectEqual(@as(usize, 2048), pk.stageRows(2048, 0, true));
}
