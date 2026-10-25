//! MiniMax-M3.1-Flash tokenizer and chat template on TensorFold's engines, byte for byte against Python goldens.
const std = @import("std");
const tokenizer = @import("tokenizer");
const template = @import("template");
const options = @import("mm3_options");

const template_source = @embedFile("fixtures/chat_template.jinja");
const goldens_json = @embedFile("fixtures/template_goldens.json");
const cases_json = @embedFile("fixtures/tokenizer_cases.json");

fn jsonOf(v: std.json.Value, key: []const u8) std.json.Value {
    return if (v == .object) (v.object.get(key) orelse .null) else .null;
}

test "chat template: every golden case renders byte for byte as Python Jinja2 renders it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, goldens_json, .{});
    var compiled = try template.compile(gpa, template_source, null);
    defer compiled.deinit();
    var errors: usize = 0;
    for (parsed.array.items) |case| {
        const name = jsonOf(case, "name").string;
        const context = jsonOf(case, "context");
        const gen = jsonOf(case, "add_generation_prompt").bool;
        var diag = template.Diag{};
        defer if (diag.msg.len > 0) gpa.free(diag.msg);
        const got = template.render(gpa, &compiled, jsonOf(case, "messages"), jsonOf(case, "tools"), context, gen, &diag);
        if (jsonOf(case, "error") == .string) {
            try std.testing.expectError(error.TemplateFailed, got);
            try std.testing.expect(diag.raised);
            try std.testing.expectEqualStrings(jsonOf(case, "error").string, diag.msg);
            errors += 1;
            continue;
        }
        const text = got catch |err| {
            std.debug.print("case {s}: {s}: {s}\n", .{ name, @errorName(err), diag.msg });
            return err;
        };
        defer gpa.free(text);
        std.testing.expectEqualStrings(jsonOf(case, "output").string, text) catch |err| {
            std.debug.print("case {s} differs\n", .{name});
            return err;
        };
    }
    try std.testing.expect(parsed.array.items.len >= 50);
    try std.testing.expectEqual(@as(usize, 2), errors);
}

test "chat template: the thinking modes only change the instruction line and the generation prefix" {
    const gpa = std.testing.allocator;
    const msgs = "[{\"role\":\"user\",\"content\":\"hi\"}]";
    const want = [_]struct { mode: []const u8, tail: []const u8 }{
        .{ .mode = "\"enabled\"", .tail = "]~b]ai\n<mm:think>" },
        .{ .mode = "\"disabled\"", .tail = "]~b]ai\n</mm:think>" },
        .{ .mode = "\"adaptive\"", .tail = "]~b]ai\n" },
        .{ .mode = "null", .tail = "]~b]ai\n" },
    };
    for (want) |w| {
        const src = try std.fmt.allocPrint(gpa, "{{\"messages\":{s},\"thinking_mode\":{s}}}", .{ msgs, w.mode });
        defer gpa.free(src);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, src, .{});
        defer parsed.deinit();
        const out = try template.renderChat(gpa, template_source, parsed.value.object.get("messages").?, .null, parsed.value, true, null);
        defer gpa.free(out);
        try std.testing.expect(std.mem.endsWith(u8, out, w.tail));
    }
}

fn loadReal(a: std.mem.Allocator) !tokenizer.Tokenizer {
    return tokenizer.loadTokenizer(std.testing.io, a, options.tokenizer_path) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("skipped: {s} not found (fetch the checkpoint's tokenizer.json and pass -Dmm3-tokenizer=PATH)\n", .{options.tokenizer_path});
            return error.SkipZigTest;
        },
        else => return err,
    };
}

test "tokenizer: the real tokenizer.json encodes and decodes as the Python tokenizers library does" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tok = try loadReal(gpa);
    defer tok.deinit();
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, cases_json, .{});
    const specials = jsonOf(parsed, "specials").object;
    var it = specials.iterator();
    while (it.next()) |e| try std.testing.expectEqual(@as(?u32, @intCast(e.value_ptr.integer)), tok.specialTokenId(e.key_ptr.*));
    for (jsonOf(parsed, "cases").array.items) |case| {
        const text = jsonOf(case, "text").string;
        const want = jsonOf(case, "ids").array.items;
        const ids = try tok.encode(gpa, text);
        defer gpa.free(ids);
        errdefer std.debug.print("text: {s}\n", .{text[0..@min(text.len, 80)]});
        try std.testing.expectEqual(want.len, ids.len);
        for (want, ids) |w, g| try std.testing.expectEqual(@as(u32, @intCast(w.integer)), g);
        const back = try tok.decode(gpa, ids, false);
        defer gpa.free(back);
        try std.testing.expectEqualStrings(jsonOf(case, "decoded").string, back);
        const skipped = try tok.decode(gpa, ids, true);
        defer gpa.free(skipped);
        try std.testing.expectEqualStrings(jsonOf(case, "decoded_skip").string, skipped);
    }
}
