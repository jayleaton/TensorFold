//! Message memo equivalence across normalized chat boundaries, rendering suffixes and request masks.
const std = @import("std");
const tok = @import("tokenizer.zig");
const a = std.testing.allocator;

fn fixture(first: bool, normalize: bool) !tok.Tokenizer {
    const source = try std.fmt.allocPrint(a,
        \\{{"added_tokens":[{{"content":"<U>","normalized":true}},{{"content":"<A>","normalized":true}},
        \\{{"content":"<é>","normalized":true}},{{"content":"<L>","normalized":true,"lstrip":true}},
        \\{{"content":"<R>","normalized":true,"rstrip":true}},{{"content":"<E>","special":true,"normalized":false}}],
        \\"normalizer":{s},"pre_tokenizer":{s},
        \\"model":{{"type":"BPE","unk_token":"?","vocab":{{"?":0,"a":1,"b":2," ":3,"é":4,
        \\"漢":5,"🙂":6,"�":7,"▁":8,"ab":9}},"merges":["a b"]}}}}
    , .{
        if (normalize) "{\"type\":\"NFC\"}" else "null",
        if (first) "{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"prepend_scheme\":\"first\",\"split\":true}" else "null",
    });
    defer a.free(source);
    return tok.parse(a, source);
}

fn body() ![]u8 {
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(a);
    for (0..160) |_| try text.appendSlice(a, "ab é漢🙂 ");
    return text.toOwnedSlice(a);
}

fn equalEncode(fresh: *const tok.Tokenizer, memo: *const tok.Tokenizer, text: []const u8) !void {
    const expected = try fresh.encode(a, text);
    defer a.free(expected);
    const actual = try memo.encode(a, text);
    defer a.free(actual);
    try std.testing.expectEqualSlices(u32, expected, actual);
}

test "message memo reuses normalized marker gaps across history prompt and next turn" {
    const long = try body();
    defer a.free(long);
    const history = try std.mem.concat(a, u8, &.{ "<E><U>", long });
    defer a.free(history);
    const prompt = try std.mem.concat(a, u8, &.{ history, "<A>" });
    defer a.free(prompt);
    const next = try std.mem.concat(a, u8, &.{ prompt, "ab<E><U>é<A>" });
    defer a.free(next);
    var fresh = try fixture(false, false);
    defer fresh.deinit();
    for ([_]bool{ false, true }) |enabled| {
        var cached = try fixture(false, false);
        defer cached.deinit();
        cached.message_memo = enabled;
        cached.memo = try tok.Memo.create(a, 1 << 20);
        for ([_][]const u8{ history, prompt, next }) |text| try equalEncode(&fresh, &cached, text);
        try std.testing.expectEqual(@as(u64, if (enabled) 2 else 0), cached.memo.?.hits);
    }
}

test "message memo preserves multibyte lossy first and stripping boundary token ids" {
    const long = try body();
    defer a.free(long);
    const cases = [_][]const []const u8{
        &.{long},
        &.{ "<U>", long },
        &.{ "<U><A>", long, "<E>" },
        &.{ long, "<U>", long },
        &.{ "ab　<L>", long, "<A>" },
        &.{ "<R>　 ", long, "<A>" },
        &.{ "<e\u{301}>", long, "<A>" },
        &.{ "<é>", long, "<A>" },
        &.{ "<U>\xff", long, "\xc3<A>" },
        &.{ "<U>", long, "<A" },
        &.{ "<U>", long, "<A>" },
        &.{ "<U>", long, "<" },
        &.{ "<U>", long[0..1023], "<A>" },
        &.{ "<U>", long[0..1024], "<A>" },
        &.{ "<U>", long[0..1025], "<A>" },
    };
    for ([_]bool{ false, true }) |first| {
        var fresh = try fixture(first, !first);
        defer fresh.deinit();
        var cached = try fixture(first, !first);
        defer cached.deinit();
        cached.message_memo = true;
        cached.memo = try tok.Memo.create(a, 1 << 20);
        for (0..2) |round| for (0..cases.len) |k| {
            const i = if (round == 0) k else cases.len - 1 - k;
            const text = try std.mem.concat(a, u8, cases[i]);
            defer a.free(text);
            try equalEncode(&fresh, &cached, text);
        };
        try std.testing.expect(cached.memo.?.hits > 0);
    }
}

test "message memo preserves masks and plain encoding across warmed chat markers" {
    const long = try body();
    defer a.free(long);
    const text = try std.mem.concat(a, u8, &.{ "<U>", long, "<A>", long, "<E>" });
    defer a.free(text);
    var fresh = try fixture(false, false);
    defer fresh.deinit();
    var cached = try fixture(false, false);
    defer cached.deinit();
    cached.message_memo = true;
    cached.memo = try tok.Memo.create(a, 1 << 20);
    try equalEncode(&fresh, &cached, text);
    const masks = [_][]const tok.Span{
        &.{.{ 3, 3 + long.len }},
        &.{.{ 1, 2 }},
        &.{.{ 3 + long.len + 1, 3 + long.len + 2 }},
        &.{.{ 0, text.len }},
    };
    const before_masked = cached.memo.?.hits;
    for (masks) |spans| {
        const expected = try fresh.encodeMasked(a, text, spans);
        defer a.free(expected);
        const actual = try cached.encodeMasked(a, text, spans);
        defer a.free(actual);
        try std.testing.expectEqualSlices(u32, expected, actual);
    }
    try std.testing.expect(cached.memo.?.hits > before_masked);
    const hits = cached.memo.?.hits;
    const expected = try fresh.encodePlain(a, text);
    defer a.free(expected);
    const actual = try cached.encodePlain(a, text);
    defer a.free(actual);
    try std.testing.expectEqualSlices(u32, expected, actual);
    try std.testing.expectEqual(hits, cached.memo.?.hits);
}

test "message memo hits still refuse model vocabulary ids that alias an added token" {
    const long = try a.alloc(u8, 2048);
    defer a.free(long);
    @memset(long, 'a');
    const source = try std.fmt.allocPrint(a,
        \\{{"added_tokens":[{{"content":"<U>","normalized":true}}],
        \\"model":{{"type":"WordLevel","unk_token":"?","vocab":{{"?":0,"<U>":1,"{s}":1}}}}}}
    , .{long});
    defer a.free(source);
    var cached = try tok.parse(a, source);
    defer cached.deinit();
    cached.message_memo = true;
    cached.memo = try tok.Memo.create(a, 1 << 20);
    const ids = try cached.encode(a, long);
    defer a.free(ids);
    try std.testing.expectEqualSlices(u32, &.{1}, ids);
    const hits = cached.memo.?.hits;
    try std.testing.expectError(error.UnsafeAddedTokenMask, cached.encodeMasked(a, long, &.{.{ 0, long.len }}));
    try std.testing.expectEqual(hits + 1, cached.memo.?.hits);
}
