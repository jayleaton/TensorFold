//! TF_DSV41_EXPERT_MIN_K: the decode prune's floor on kept routed experts (block.Options.expert_min_k; lossy below 3).
const std = @import("std");

pub const Get = *const fn ([]const u8) ?[]const u8;

/// The prune keeps at least this many routed experts a row (unset: 3, prod's rule); 1-6, anything else refused.
pub fn minK(get: Get) !i64 {
    const raw = std.mem.trim(u8, get("TF_DSV41_EXPERT_MIN_K") orelse return 3, " ");
    const k = std.fmt.parseInt(i64, raw, 10) catch return bad(raw);
    if (k < 1 or k > 6) return bad(raw);
    return k;
}

fn bad(raw: []const u8) error{BadKnob} {
    std.log.warn("TF_DSV41_EXPERT_MIN_K={s}: 1 to 6 (the routed top-k)", .{raw});
    return error.BadKnob;
}

test "TF_DSV41_EXPERT_MIN_K: unset is prod's 3; 1-6 taken; others refused" {
    const T = struct {
        var v: ?[]const u8 = null;
        fn get(name: []const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_DSV41_EXPERT_MIN_K")) v else null;
        }
    };
    try std.testing.expectEqual(@as(i64, 3), try minK(T.get));
    T.v = "2";
    try std.testing.expectEqual(@as(i64, 2), try minK(T.get));
    T.v = " 6 ";
    try std.testing.expectEqual(@as(i64, 6), try minK(T.get));
    for ([_][]const u8{ "0", "7", "two", "" }) |x| {
        T.v = x;
        try std.testing.expectError(error.BadKnob, minK(T.get));
    }
}
