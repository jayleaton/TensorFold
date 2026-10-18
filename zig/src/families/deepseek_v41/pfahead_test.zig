//! TF_DSV41_PF_AHEAD's host rules (`zig build test-dsv41-knobs`): the knob, and which prompt calls read their own rows.
const std = @import("std");
const testing = std.testing;
const pk = @import("prod_knobs.zig");
const fp = @import("forward_prefill.zig");

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "pf ahead: the knob is 1 alone, and a call reads its own rows unless the last read-ahead covered them" {
    Fake.pairs = &.{};
    try testing.expect(!pk.pfAhead(&Fake.get));
    for ([_][]const u8{ "0", "2", "on", "" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_PF_AHEAD", v }};
        try testing.expect(!pk.pfAhead(&Fake.get));
    }
    Fake.pairs = &.{.{ "TF_DSV41_PF_AHEAD", "1" }};
    try testing.expect(pk.pfAhead(&Fake.get));
    Fake.pairs = &.{};
    const none = std.math.maxInt(u64);
    // a prompt's first piece (nothing read ahead) and a resumed prompt read their own rows
    try testing.expect(fp.ownReads(0, none, 0, 4096));
    try testing.expect(fp.ownReads(8192, none, 0, 4096));
    // the next piece that the last call read ahead in full: nothing more to issue
    try testing.expect(!fp.ownReads(4096, 4096, 4096, 4096));
    try testing.expect(!fp.ownReads(28672, 28672, 4096, 4080));
    // a longer piece than the guess, a piece at another position, an empty call
    try testing.expect(fp.ownReads(4096, 4096, 2048, 4096));
    try testing.expect(fp.ownReads(4112, 4096, 4096, 4096));
    try testing.expect(!fp.ownReads(4096, none, 0, 0));
}
