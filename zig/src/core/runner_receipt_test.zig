//! The receipt's per-cell records: end to end on a lane host (held results), and the log's drafted/forced/cut rules.
const std = @import("std");
const rapi = @import("runner_api.zig");
const scan = @import("runner_scan.zig");
const records = @import("runner_receipt.zig");
const rt = @import("runner_test.zig");

const gpa = std.testing.allocator;

/// The first record of a receipt, parsed.
const Rec = struct { cell: u32, form: []const u8, route: []const u8, status: []const u8, lang: []const u8, span: []const u8, value: []const u8, field: []const u8, at: u64, token: u32, ids: u32, landed: u32, drafted: u32, forced: u32, path: []const u8 };

const Receipt = struct { cells: []const Rec };

fn first(receipt: []const u8) !std.json.Parsed(Receipt) {
    return std.json.parseFromSlice(Receipt, gpa, receipt, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}

test "runner receipt: a held fence result records its span, route and the bytes that landed after the header" {
    var back: rt.Back = .{};
    defer back.deinit();
    var m = [_]rt.Model{.{ .lead = rt.mul_cell }};
    var sums: [1][]u8 = undefined;
    const texts = try rt.replies(&m, .{ .summaries = &sums }, &back);
    defer rt.freeAll(texts);
    defer gpa.free(sums[0]);
    const p = try first(sums[0]);
    defer p.deinit();
    try std.testing.expectEqual(1, p.value.cells.len);
    const r = p.value.cells[0];
    try std.testing.expectEqualStrings("fence", r.form);
    try std.testing.expectEqualStrings("calc", r.route);
    try std.testing.expectEqualStrings("ok", r.status);
    try std.testing.expectEqualStrings("python", r.lang);
    try std.testing.expectEqualStrings("print(123456789012345678901234567890 * 987654321098765432109876543210)\n", r.span);
    // one token a character: the write starts at the header's newline (the result's), its bytes are the reply's there
    try std.testing.expectEqualStrings("content", r.field);
    try std.testing.expectEqual(rt.mul_cell.len - 1, r.at);
    try std.testing.expectEqual(rt.mul_cell.len - 1, r.token);
    try std.testing.expectEqualStrings(texts[0][r.at..][0..r.value.len], r.value);
    try std.testing.expect(std.mem.startsWith(u8, r.value, "\n" ++ rt.product));
    try std.testing.expectEqual(r.ids, r.landed);
    try std.testing.expectEqual(0, r.drafted);
    try std.testing.expectEqual(r.landed, r.forced);
    try std.testing.expectEqualStrings("held", r.path);
}

test "runner receipt: an inline value records the parsed expression and the value after `=`" {
    var back: rt.Back = .{};
    defer back.deinit();
    const lead = "Exact: 1.5+2.25 =";
    var m = [_]rt.Model{.{ .lead = lead }};
    var sums: [1][]u8 = undefined;
    const texts = try rt.replies(&m, .{ .summaries = &sums, .inline_math = true }, &back);
    defer rt.freeAll(texts);
    defer gpa.free(sums[0]);
    const p = try first(sums[0]);
    defer p.deinit();
    const r = p.value.cells[0];
    try std.testing.expectEqualStrings("inline", r.form);
    try std.testing.expectEqualStrings("1.5+2.25", std.mem.trim(u8, r.span, " "));
    try std.testing.expectEqualStrings(" 3.75", r.value);
    try std.testing.expectEqual(lead.len, r.at);
    try std.testing.expectEqualStrings(" 3.75", texts[0][r.at..][0..r.value.len]);
}

/// The fields records.write reads off a runner.Session and its cells.
const FakeCell = struct {
    form: rapi.Form = .repl,
    stated: ?scan.Stated = null,
    told: enum { none, owned, abstained } = .owned,
    owned_tokens: u32 = 0,
    translation: rapi.Translation = .{ .program = .{ .kind = .graph, .bytes = "" } },
    result: ?rapi.Result = .{ .status = .ok },
    lang: []const u8 = "python",
    code: []const u8 = "2**8",
};

const FakeSession = struct {
    r: struct { gpa: std.mem.Allocator = gpa } = .{},
    cells: struct { items: []const *FakeCell },
    held_ms: u64 = 0,
    resume_ms: u64 = 0,
    rec: records.Log = .{},
};

test "runner receipt: a sync write is drafted within its trigger's commit and forced after a round; a cut lands less" {
    var cell: FakeCell = .{};
    var s: FakeSession = .{ .cells = .{ .items = &.{&cell} } };
    defer s.rec.deinit(gpa);
    try s.rec.token(gpa, 1, "x", true);
    try s.rec.owned(gpa, 0, &.{ 7, 8, 9 });
    s.rec.synced();
    try s.rec.token(gpa, 7, "25", true); // the trigger's commit: the model's verify had the result's token
    s.rec.rounds += 1;
    try s.rec.token(gpa, 8, "6", true); // forced in the next round
    try s.rec.token(gpa, 5, "?", true); // not the result's: the write is cut there
    try s.rec.token(gpa, 9, "!", true);
    const p = try first(records.write(&s));
    defer p.deinit();
    const r = p.value.cells[0];
    try std.testing.expectEqualStrings("256", r.value);
    try std.testing.expectEqual(3, r.ids);
    try std.testing.expectEqual(2, r.landed);
    try std.testing.expectEqual(1, r.drafted);
    try std.testing.expectEqual(1, r.forced);
    try std.testing.expectEqualStrings("sync", r.path);
    try std.testing.expectEqualStrings("reasoning", r.field);
    try std.testing.expectEqual(1, r.at);
    try std.testing.expectEqual(1, r.token);
    try std.testing.expectEqualStrings("graph", r.route);
}

test "runner receipt: escapes, a long span as sha256, bytes cut mid-character as hex, and no records when off" {
    const long: [600]u8 = @splat('x');
    var cell: FakeCell = .{ .code = &long };
    var s: FakeSession = .{ .cells = .{ .items = &.{&cell} } };
    defer s.rec.deinit(gpa);
    try s.rec.token(gpa, 0, "", false); // the think end: content from here
    try s.rec.owned(gpa, 0, &.{ 1, 2 });
    try s.rec.token(gpa, 1, "\"a\\\n", false);
    try s.rec.token(gpa, 2, "\xcf", false);
    const got = records.write(&s);
    try std.testing.expect(std.mem.indexOf(u8, got, "\"span_len\":600,") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, "\"value_hex\":\"22615c0acf\",") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, "\"field\":\"content\",\"at\":0,") != null);
    cell.code = "say \"hi\"\n";
    s.rec.records.items[0].value.items.len = 4;
    const esc = records.write(&s);
    try std.testing.expect(std.mem.indexOf(u8, esc, "\"span\":\"say \\\"hi\\\"\\u000a\",\"value\":\"\\\"a\\\\\\u000a\",") != null);
    var off: FakeSession = .{ .cells = .{ .items = &.{&cell} }, .rec = .{ .on = false } };
    defer off.rec.deinit(gpa);
    try off.rec.owned(gpa, 0, &.{1});
    try off.rec.token(gpa, 1, "1", false);
    try std.testing.expect(std.mem.endsWith(u8, records.write(&off), "\"resume_ms\":0}"));
}
