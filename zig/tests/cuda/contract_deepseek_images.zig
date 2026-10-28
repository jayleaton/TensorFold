//! Every registered DeepSeek image is explicit; unknown names continue to the refusing common resolver.
const std = @import("std");
pub const names = @import("contract_deepseek_names").names;
fn Blob(comptime name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile("contract_fatbin_" ++ name).*;
    };
}
pub fn image(name: []const u8) ?[]const u8 {
    inline for (names) |n| if (std.mem.eql(u8, name, n)) return &Blob(n).bytes;
    return null;
}
