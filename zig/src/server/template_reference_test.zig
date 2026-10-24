//! MiniMax-M3.1 serving goldens: each fixture request through the server's message and template path, byte for byte.
const std = @import("std");
const json = @import("json");
const template = @import("template");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const prompt = @import("prompt.zig");
const HfText = @import("hf_text.zig").HfText;
const fixtures = @import("mm3_serve_fixtures");

test {
    _ = messages_mod; // the serving rules' own tests run in this step too
}

/// A chat template behind hf_text's render (its globals, thinking_mode word and context), without a tokenizer.
const Text = struct {
    hf: HfText,

    fn init(gpa: std.mem.Allocator, source: []const u8) !Text {
        var t: Text = .{ .hf = .{ .gpa = gpa, .arena = .init(gpa), .tok = undefined, .source = source } };
        t.hf.compiled = try template.compile(gpa, source, null);
        return t;
    }

    fn deinit(t: *Text) void {
        t.hf.compiled.?.deinit();
        t.hf.arena.deinit();
    }
};

/// `prompt.renderIds`' text for a fixture request: `forTemplate` with MiniMax's late-system role, then the template.
fn renderText(cx: *errors.Cx, t: *Text, c: json.Value, problem: *[]const u8) !?[]const u8 {
    const tools: []const json.Value = if (c.get("tools").? == .array) c.get("tools").?.array else &.{};
    const effort: ?[]const u8 = if (c.get("effort").? == .string) c.get("effort").?.string else null;
    const options = prompt.renderOptions(tools, c.get("thinking").?.bool, effort, c.get("generation").?.bool);
    const messages = messages_mod.forTemplate(cx, c.get("messages"), "system", false, .{}) catch |e| switch (e) {
        error.OutOfMemory => return e,
        error.Refused => {
            problem.* = cx.message;
            return null;
        },
    };
    const r = t.hf.text().render(cx.a, messages, options, problem) catch |e| switch (e) {
        error.OutOfMemory => return e,
        error.Template => return null,
    };
    return r.text;
}

test "MiniMax-M3.1 serving: every reference request renders as the template renders it after the serving rules" {
    const gpa = std.testing.allocator;
    var m31 = try Text.init(gpa, fixtures.m31);
    defer m31.deinit();
    var mia = try Text.init(gpa, fixtures.mia);
    defer mia.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cases: usize = 0;
    var bad: usize = 0;
    var refused: usize = 0;
    var skipped: usize = 0;
    var ruled: usize = 0;
    var it = std.mem.splitScalar(u8, fixtures.reference, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();
        const c = (try json.parseText(a, line)).ok;
        if (c.get("skip") != null) { // a request the server refuses on purpose (the line says why)
            skipped += 1;
            continue;
        }
        cases += 1;
        const t = if (std.mem.eql(u8, c.get("template").?.string, "mia")) &mia else &m31;
        var cx: errors.Cx = .{ .a = a };
        var problem: []const u8 = "";
        const got = try renderText(&cx, t, c, &problem);
        const want = c.get("expected").?;
        const ok = if (want == .string) (got != null and std.mem.eql(u8, got.?, want.string)) else (got == null and std.mem.indexOf(u8, problem, c.get("error").?.string) != null);
        if (want != .string) refused += 1;
        if (c.get("rules").?.array.len > 0) ruled += 1;
        if (ok) continue;
        bad += 1;
        if (bad > 4) continue;
        std.debug.print("reference case {s} ({s}) differs\n", .{ c.get("name").?.string, c.get("template").?.string });
        if (got == null) {
            std.debug.print("  refused: {s}\n", .{problem});
        } else if (want == .string) {
            const g = got.?;
            const k = std.mem.indexOfDiff(u8, g, want.string) orelse g.len;
            const from = k -| 40;
            std.debug.print("  at byte {d}:\n  want {f}\n  got  {f}\n", .{ k, std.json.fmt(want.string[from..@min(want.string.len, k + 120)], .{}), std.json.fmt(g[from..@min(g.len, k + 120)], .{}) });
        } else std.debug.print("  want refusal {s}, got text\n", .{c.get("error").?.string});
    }
    std.debug.print("minimax serve reference: {d}/{d} equal to Python Jinja2 ({d} refused as it refuses, {d} through a serving rule), {d} skipped\n", .{ cases - bad, cases, refused, ruled, skipped });
    try std.testing.expect(cases >= 60);
    try std.testing.expectEqual(@as(usize, 0), bad);
}
