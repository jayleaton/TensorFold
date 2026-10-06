//! Goldens from the Python side, read where the generators wrote them: TF_DSV41_MODEL (the release tokenizer.json's
//! directory) and TF_DSV41_GOLDEN (zig/tests/server/dsv41/gen_*.py's output). Skipped when either is unset.
const std = @import("std");
const testing = std.testing;
const tokenizer = @import("tokenizer.zig");

pub fn env(name: []const u8) ?[]const u8 {
    return testing.environ.getPosix(name);
}

pub fn readGolden(a: std.mem.Allocator, name: []const u8) !?[]u8 {
    const dir = env("TF_DSV41_GOLDEN") orelse return null;
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    return try std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(1 << 30));
}

const Reader = struct {
    b: []const u8,
    at: usize = 0,
    fn int(r: *Reader) u32 {
        const v = std.mem.readInt(u32, r.b[r.at..][0..4], .little);
        r.at += 4;
        return v;
    }
    fn bytes(r: *Reader, n: usize) []const u8 {
        defer r.at += n;
        return r.b[r.at..][0..n];
    }
    fn ids(r: *Reader, a: std.mem.Allocator, n: usize) ![]u32 {
        const out = try a.alloc(u32, n);
        for (out) |*x| x.* = r.int();
        return out;
    }
};

pub fn loadTokenizer(a: std.mem.Allocator) !?*tokenizer.Tokenizer {
    const dir = env("TF_DSV41_MODEL") orelse return null;
    return try tokenizer.Tokenizer.load(a, testing.io, dir);
}

test "the release tokenizer encodes the golden corpus to tokenizers' ids exactly" {
    const a = testing.allocator;
    const tok = try loadTokenizer(a) orelse return error.SkipZigTest;
    defer tok.deinit();
    const data = try readGolden(a, "tokenizer.golden") orelse return error.SkipZigTest;
    defer a.free(data);
    var r: Reader = .{ .b = data };
    const enc = try tok.acquire();
    defer tok.release(enc);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    var docs: usize = 0;
    var bad: usize = 0;
    while (r.at < data.len) {
        const text = r.bytes(r.int());
        const want = try r.ids(a, r.int());
        defer a.free(want);
        out.clearRetainingCapacity();
        try enc.encode(a, text, &out);
        docs += 1;
        if (!std.mem.eql(u32, want, out.items)) {
            bad += 1;
            if (bad <= 5) {
                var first: usize = 0;
                while (first < @min(want.len, out.items.len) and want[first] == out.items[first]) first += 1;
                std.debug.print("doc {d}: ids differ at {d} (want {d} got {d}); text {f}\n", .{ docs - 1, first, want.len, out.items.len, std.json.fmt(text[0..@min(text.len, 160)], .{}) });
                std.debug.print("  want {any}\n  got  {any}\n", .{ want[first..@min(want.len, first + 8)], out.items[first..@min(out.items.len, first + 8)] });
            }
        }
    }
    std.debug.print("tokenizer golden: {d}/{d} documents equal\n", .{ docs - bad, docs });
    try testing.expectEqual(@as(usize, 0), bad);
}

test "the release tokenizer decodes ids to tokenizers' text exactly, whole and a token at a time" {
    const a = testing.allocator;
    const tok = try loadTokenizer(a) orelse return error.SkipZigTest;
    defer tok.deinit();
    const data = try readGolden(a, "decode.golden") orelse return error.SkipZigTest;
    defer a.free(data);
    var r: Reader = .{ .b = data };
    var cases: usize = 0;
    var bad: usize = 0;
    var streamed: std.ArrayList(u8) = .empty;
    defer streamed.deinit(a);
    while (r.at < data.len) {
        const ids = try r.ids(a, r.int());
        defer a.free(ids);
        const want = r.bytes(r.int());
        const got = try tok.decodeAlloc(a, ids);
        defer a.free(got);
        var d: tokenizer.Detokenizer = .{ .tok = tok };
        streamed.clearRetainingCapacity();
        for (ids) |id| try d.push(a, id, &streamed);
        try d.flush(a, &streamed);
        cases += 1;
        if (!std.mem.eql(u8, want, got) or !std.mem.eql(u8, want, streamed.items)) {
            bad += 1;
            if (bad <= 5) std.debug.print("decode case {d}: want {f}\n got {f}\n", .{ cases - 1, std.json.fmt(want[0..@min(want.len, 120)], .{}), std.json.fmt(got[0..@min(got.len, 120)], .{}) });
        }
    }
    std.debug.print("decode golden: {d}/{d} cases equal\n", .{ cases - bad, cases });
    try testing.expectEqual(@as(usize, 0), bad);
}

const json = @import("json");
const template = @import("template.zig");

fn imageText(ctx: ?*anyopaque, _: json.Value) []const u8 {
    const s: *const []const u8 = @ptrCast(@alignCast(ctx.?));
    return s.*;
}

/// Each line of a gen_template.py golden rendered here: equal text, or a refusal where Python refused.
fn checkTemplates(lines: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var it = std.mem.splitScalar(u8, lines, '\n');
    var cases: usize = 0;
    var bad: usize = 0;
    var jinja: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();
        const c = (try json.parseText(a, line)).ok;
        cases += 1;
        var image_text: []const u8 = "";
        var opts: template.Options = .{
            .thinking = c.get("thinking").?.bool,
            .budget = template.budgetOf(c.get("budget").?) orelse 0,
            .drop_thinking = c.get("drop_thinking").?.bool,
            .add_generation_prompt = c.get("generation").?.bool,
        };
        if (c.get("tools")) |t| if (t == .array) {
            opts.tools = t.array;
        };
        if (c.get("response_format")) |rf| if (rf != .null) {
            opts.response_format = rf;
        };
        if (c.get("image_text")) |t| if (t == .string) {
            image_text = t.string;
            opts.image = .{ .ctx = @ptrCast(&image_text), .text = imageText };
        };
        var problem: template.Problem = .{};
        const got = template.render(a, c.get("messages").?, opts, &problem);
        const want = c.get("expected").?;
        const ok = if (want == .string) (if (got) |g| std.mem.eql(u8, g, want.string) else |_| false) else (if (got) |_| false else |e| e == error.Refused);
        if (c.get("jinja")) |j| if (j == .string) {
            jinja += 1;
        };
        if (!ok) {
            bad += 1;
            if (bad <= 3) {
                std.debug.print("template case {d} differs ({s})\n", .{ cases - 1, if (want == .string) "want text" else c.get("error").?.string });
                if (got) |g| {
                    if (want == .string) {
                        var k: usize = 0;
                        while (k < @min(g.len, want.string.len) and g[k] == want.string[k]) k += 1;
                        std.debug.print("  at byte {d}:\n  want {f}\n  got  {f}\n", .{ k, std.json.fmt(want.string[k..@min(want.string.len, k + 160)], .{}), std.json.fmt(g[k..@min(g.len, k + 160)], .{}) });
                    }
                } else |_| std.debug.print("  refused: {s}\n", .{problem.message});
            }
        }
    }
    std.debug.print("template golden: {d}/{d} equal to encoding.encode ({d} of them also the release chat_template.jinja's text)\n", .{ cases - bad, cases, jinja });
    try testing.expectEqual(@as(usize, 0), bad);
}

test "the chat encoding renders the committed conversations as the Python engine and the release template do" {
    try checkTemplates(@import("dsv41_serve_fixtures").template);
}

test "the chat encoding renders the full generated set (TF_DSV41_GOLDEN/template.jsonl)" {
    const data = try readGolden(testing.allocator, "template.jsonl") orelse return error.SkipZigTest;
    defer testing.allocator.free(data);
    try checkTemplates(data);
}
