//! MiniMax-M3.1 calls the official template rendered (tools/zig/minimax_tool_calls.py), read back whole and streamed.
const std = @import("std");
const json = @import("json");
const check = @import("tool_minimax_check.zig");

const fixture = @embedFile("fixtures/minimax_tool_calls.json");

test "minimax goldens: every rendered call parses back to its arguments, and streams the same at every split" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = (try json.parse(a, fixture)).ok.array;
    var exact: usize = 0;
    var lossy: usize = 0;
    var skipped: usize = 0;
    for (cases) |c| {
        const name = c.get("name").?.string;
        errdefer std.debug.print("golden case {s}\n", .{name});
        // a skipped case still streams as it parses; only its arguments are not the ones rendered
        const parsed = try check.check(a, c.get("reply").?.string, c.get("tools").?.array);
        if (c.get("skip").? != .null) {
            skipped += 1;
            continue;
        }
        if (c.get("lossy").? != .null) lossy += 1 else exact += 1;
        const want = c.get("calls").?.array;
        const got = parsed.calls orelse &.{};
        try std.testing.expectEqual(want.len, got.len);
        for (want, got) |w, g| {
            try std.testing.expectEqualStrings(w.get("name").?.string, g.get("function").?.get("name").?.string);
            // exact JSON text, so a string never passes for the number or boolean it spells
            const expected = try json.stringify(a, w.get("arguments").?, .{ .ascii = false, .compact = true });
            try std.testing.expectEqualStrings(expected, g.get("function").?.get("arguments").?.string);
        }
    }
    try std.testing.expect(exact >= 17 and lossy >= 5 and skipped >= 2);
}
