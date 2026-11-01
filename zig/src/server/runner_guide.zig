//! TF_DSV41_RUNNER_GUIDE: RUNNER-GUIDE.md (T5) in a chat's system message, so the model asks for exact results early.
const std = @import("std");
const json = @import("json");
const Value = json.Value;

/// RUNNER-GUIDE.md without its final newline (76 tokens on DeepSeek V4.1).
pub const text = "Exact computation is available. Write the full expression inline as `<expr> =`, or a Python fenced cell with `print(<expr>)` followed by a `Result:` header. The exact value will follow. Request the whole computation early; wait for the result instead of working digits, loops, carries or long division by hand. Unsupported requests leave your own continuation.";

/// `messages` with the guide after the first system message's text, or in a new system message before the rest.
pub fn with(a: std.mem.Allocator, messages: Value) std.mem.Allocator.Error!Value {
    if (messages != .array) return messages;
    const out = try a.alloc(Value, messages.array.len + 1);
    if (messages.array.len > 0) if (messages.array[0].strField("role")) |role| if (std.mem.eql(u8, role, "system")) {
        if (messages.array[0].get("content")) |content| if (content == .string) {
            const m = try json.newObject(a);
            var it = messages.array[0].object.iterator();
            while (it.next()) |e| try m.put(a, e.key_ptr.*, e.value_ptr.*);
            try m.put(a, "content", .{ .string = try std.mem.concat(a, u8, &.{ content.string, "\n\n", text }) });
            out[0] = .{ .object = m };
            @memcpy(out[1..messages.array.len], messages.array[1..]);
            return .{ .array = out[0..messages.array.len] };
        };
    };
    const m = try json.newObject(a);
    try m.put(a, "role", .{ .string = "system" });
    try m.put(a, "content", .{ .string = text });
    out[0] = .{ .object = m };
    @memcpy(out[1..], messages.array);
    return .{ .array = out };
}

test "runner guide: T5's RUNNER-GUIDE.md bytes; joins a system message or leads the chat" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text ++ "\n", &digest, .{});
    try std.testing.expectEqualStrings("b99224b5ebf6f7e9323bfc011200073211e1fc9381968e067f1a336893a8a7bb", &std.fmt.bytesToHex(digest, .lower));
    const user = try json.newObject(a);
    try user.put(a, "role", .{ .string = "user" });
    try user.put(a, "content", .{ .string = "What is 47*83?" });
    var one = [_]Value{.{ .object = user }};
    const led = try with(a, .{ .array = &one });
    try std.testing.expectEqual(@as(usize, 2), led.array.len);
    try std.testing.expectEqualStrings("system", led.array[0].strField("role").?);
    try std.testing.expectEqualStrings(text, led.array[0].get("content").?.string);
    try std.testing.expectEqualStrings("user", led.array[1].strField("role").?);
    const sys = try json.newObject(a);
    try sys.put(a, "role", .{ .string = "system" });
    try sys.put(a, "content", .{ .string = "Be brief." });
    var two = [_]Value{ .{ .object = sys }, .{ .object = user } };
    const joined = try with(a, .{ .array = &two });
    try std.testing.expectEqual(@as(usize, 2), joined.array.len);
    try std.testing.expectEqualStrings("Be brief.\n\n" ++ text, joined.array[0].get("content").?.string);
    try std.testing.expectEqualStrings("Be brief.", two[0].get("content").?.string); // the request is not changed
}
