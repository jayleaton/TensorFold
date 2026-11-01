//! TF_DSV41_RUNNER_GUIDE: RUNNER-GUIDE.md (T5) in a chat's system message, so the model asks for exact results early.
const std = @import("std");
const json = @import("json");
const Value = json.Value;

/// RUNNER-GUIDE.md without its final newline (76 tokens on DeepSeek V4.1).
pub const text = "Exact computation is available. Write the full expression inline as `<expr> =`, or a Python fenced cell with `print(<expr>)` followed by a `Result:` header. The exact value will follow. Request the whole computation early; wait for the result instead of working digits, loops, carries or long division by hand. Unsupported requests leave your own continuation.";

/// The most a request's own "runner_guide" may hold.
pub const max_request = 512;

/// The server's default guide, read once at boot from TF_DSV41_RUNNER_GUIDE: "1" is `text`, any other value a file.
pub const Set = struct {
    default: []const u8 = "", // "": chats get no guide unless they ask

    pub fn load(a: std.mem.Allocator, io: std.Io, knob: []const u8) !Set {
        if (knob.len == 0) return .{};
        if (std.mem.eql(u8, knob, "1")) return .{ .default = text };
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, knob, a, .limited(64 << 10));
        return .{ .default = trim(bytes) };
    }

    pub fn describe(s: Set) void {
        @import("log.zig").line("runner guide: default {d} bytes", .{s.default.len});
    }
};

/// Trailing whitespace off, so a file's final newline and a request's text read alike.
fn trim(t: []const u8) []const u8 {
    return std.mem.trimEnd(u8, t, " \t\r\n");
}

pub const Applied = struct {
    messages: Value,
    guide: ?[]const u8, // the text injected ("" none); null when the runner is off
};

/// `messages` with `asked` (a request's guide) or the default; "" adds nothing, nor does a null `set` (runner off).
pub fn apply(a: std.mem.Allocator, set: ?*const Set, asked: ?Value, messages: Value) error{ BadGuide, OutOfMemory }!Applied {
    const s = set orelse return .{ .messages = messages, .guide = null };
    const chosen = if (asked) |v| switch (v) {
        .null => s.default,
        .string => |t| if (t.len <= max_request) trim(t) else return error.BadGuide,
        else => return error.BadGuide,
    } else s.default;
    return .{ .messages = if (chosen.len > 0) try with(a, messages, chosen) else messages, .guide = chosen };
}

/// `messages` with `guide` after the first system message's text, or in a new system message before the rest.
pub fn with(a: std.mem.Allocator, messages: Value, guide: []const u8) std.mem.Allocator.Error!Value {
    if (messages != .array) return messages;
    const out = try a.alloc(Value, messages.array.len + 1);
    if (messages.array.len > 0) if (messages.array[0].strField("role")) |role| if (std.mem.eql(u8, role, "system")) {
        if (messages.array[0].get("content")) |content| if (content == .string) {
            const m = try json.newObject(a);
            var it = messages.array[0].object.iterator();
            while (it.next()) |e| try m.put(a, e.key_ptr.*, e.value_ptr.*);
            try m.put(a, "content", .{ .string = try std.mem.concat(a, u8, &.{ content.string, "\n\n", guide }) });
            out[0] = .{ .object = m };
            @memcpy(out[1..messages.array.len], messages.array[1..]);
            return .{ .array = out[0..messages.array.len] };
        };
    };
    const m = try json.newObject(a);
    try m.put(a, "role", .{ .string = "system" });
    try m.put(a, "content", .{ .string = guide });
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
    const led = try with(a, .{ .array = &one }, text);
    try std.testing.expectEqual(@as(usize, 2), led.array.len);
    try std.testing.expectEqualStrings("system", led.array[0].strField("role").?);
    try std.testing.expectEqualStrings(text, led.array[0].get("content").?.string);
    try std.testing.expectEqualStrings("user", led.array[1].strField("role").?);
    const sys = try json.newObject(a);
    try sys.put(a, "role", .{ .string = "system" });
    try sys.put(a, "content", .{ .string = "Be brief." });
    var two = [_]Value{ .{ .object = sys }, .{ .object = user } };
    const joined = try with(a, .{ .array = &two }, text);
    try std.testing.expectEqual(@as(usize, 2), joined.array.len);
    try std.testing.expectEqualStrings("Be brief.\n\n" ++ text, joined.array[0].get("content").?.string);
    try std.testing.expectEqualStrings("Be brief.", two[0].get("content").?.string); // the request is not changed
}

test "runner guide: boot default, a request's override, an empty one and the runner off" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "RUNNER-GUIDE.md", .data = text ++ "\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "terse.md", .data = "Ask for exact values.\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "empty.md", .data = "\n" });
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const builtin = try Set.load(a, io, "1");
    try std.testing.expectEqualStrings(text, builtin.default);
    try std.testing.expectEqualStrings(text, (try Set.load(a, io, try std.fs.path.join(a, &.{ root, "RUNNER-GUIDE.md" }))).default);
    const terse = try Set.load(a, io, try std.fs.path.join(a, &.{ root, "terse.md" }));
    const none = try Set.load(a, io, "");
    try std.testing.expectEqualStrings("", (try Set.load(a, io, try std.fs.path.join(a, &.{ root, "empty.md" }))).default);
    try std.testing.expectError(error.FileNotFound, Set.load(a, io, "/nonexistent/guide.md"));

    const user = try json.newObject(a);
    try user.put(a, "role", .{ .string = "user" });
    try user.put(a, "content", .{ .string = "What is 47*83?" });
    var one = [_]Value{.{ .object = user }};
    const chat: Value = .{ .array = &one };
    const plain = try json.stringify(a, chat, .{});
    const Case = struct { set: ?*const Set, asked: ?Value, want: []const u8, guide: ?[]const u8 };
    const with_text = try json.stringify(a, try with(a, chat, text), .{});
    const with_terse = try json.stringify(a, try with(a, chat, "Ask for exact values."), .{});
    for ([_]Case{
        .{ .set = &builtin, .asked = null, .want = with_text, .guide = text }, // the boot default
        .{ .set = &builtin, .asked = .{ .string = "Ask for exact values.\n" }, .want = with_terse, .guide = "Ask for exact values." }, // override
        .{ .set = &terse, .asked = null, .want = with_terse, .guide = "Ask for exact values." }, // the same text from a boot file
        .{ .set = &none, .asked = .{ .string = "Ask for exact values." }, .want = with_terse, .guide = "Ask for exact values." },
        .{ .set = &builtin, .asked = .{ .string = "" }, .want = plain, .guide = "" }, // empty: no guide
        .{ .set = &none, .asked = null, .want = plain, .guide = "" },
        .{ .set = null, .asked = .{ .string = "Ask for exact values." }, .want = plain, .guide = null }, // runner off: ignored
    }) |c| {
        const got = try apply(a, c.set, c.asked, chat);
        try std.testing.expectEqualStrings(c.want, try json.stringify(a, got.messages, .{}));
        if (c.guide) |g| try std.testing.expectEqualStrings(g, got.guide.?) else try std.testing.expect(got.guide == null);
    }
    const long = try a.alloc(u8, max_request + 1);
    @memset(long, 'x');
    try std.testing.expectError(error.BadGuide, apply(a, &builtin, .{ .string = long }, chat));
    _ = try apply(a, &builtin, .{ .string = long[0..max_request] }, chat);
    try std.testing.expectError(error.BadGuide, apply(a, &builtin, .{ .bool = true }, chat));
}
