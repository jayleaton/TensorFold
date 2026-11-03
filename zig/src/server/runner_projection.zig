//! Runner receipt anchors projected from the engine's raw field bytes into the reply's final reasoning and content.
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// One field's map: final byte p is raw byte `lead + p`, for p in [0, kept].
pub const Map = struct { raw: usize, lead: usize, kept: usize };

/// The engine's field bytes, as `Session.feed` keeps them: every token's bytes, none for the think-close token.
pub fn rawFields(a: Allocator, decode: anytype, ctx: anytype, tokens: []const u32, think_end: ?u32) ![2][]const u8 {
    const end = think_end orelse return .{ "", try decode(ctx, a, tokens) };
    const at = std.mem.indexOfScalar(u32, tokens, end) orelse return .{ try decode(ctx, a, tokens), "" };
    return .{ try decode(ctx, a, tokens[0..at]), try decode(ctx, a, tokens[at + 1 ..]) };
}

/// Where the final field sits in the raw one: after a head of only the think opener and whitespace; else null.
pub fn mapOf(raw: []const u8, final: []const u8, open: []const u8) ?Map {
    const lead = if (final.len == 0) raw.len else std.mem.indexOf(u8, raw, final) orelse return null;
    var head = raw[0..lead];
    if (open.len > 0 and std.mem.startsWith(u8, head, open)) head = head[open.len..];
    if (reply_text.pyStrip(head).len > 0) return null; // the reply dropped or rewrote more than a trimmed head
    return .{ .raw = raw.len, .lead = lead, .kept = final.len };
}

fn project(m: ?Map, raw: ?i64) ?usize {
    const map = m orelse return null;
    const r = std.math.cast(usize, raw orelse return null) orelse return null;
    if (r < map.lead or r - map.lead > map.kept) return null; // in a dropped head or tail: no final byte
    return r - map.lead;
}

fn put(a: Allocator, o: *json.Object, key: []const u8, n: ?usize) !void {
    try o.put(a, key, if (n) |v| try json.intValue(a, v) else .null);
}

fn mapValue(a: Allocator, m: ?Map) !Value {
    const map = m orelse return .null;
    const o = try json.newObject(a);
    try put(a, o, "raw_bytes", map.raw);
    try put(a, o, "dropped_lead", map.lead);
    try put(a, o, "final_bytes", map.kept);
    return .{ .object = o };
}

fn fieldMap(maps: [2]?Map, v: Value, key: []const u8) ?Map {
    const name = v.strField(key) orelse return null;
    if (std.mem.eql(u8, name, "reasoning")) return maps[0];
    if (std.mem.eql(u8, name, "content")) return maps[1];
    return null;
}

/// With abstain records only: `field_projection`, cells' `final_at`, abstain anchors in final bytes (raw: `raw_*`).
pub fn apply(a: Allocator, runner: Value, maps: [2]?Map) !void {
    const abstains = runner.get("abstains") orelse return;
    const fields = try json.newObject(a);
    try fields.put(a, "reasoning", try mapValue(a, maps[0]));
    try fields.put(a, "content", try mapValue(a, maps[1]));
    try runner.object.put(a, "field_projection", .{ .object = fields });
    if (runner.get("cells")) |cells| for (cells.items() orelse &.{}) |c| if (c == .object)
        try put(a, c.object, "final_at", project(fieldMap(maps, c, "field"), if (c.get("at")) |v| v.int64() else null));
    for (abstains.items() orelse &.{}) |x| if (x == .object) {
        const src = fieldMap(maps, x, "source_field");
        for ([_][]const u8{ "source_at", "source_end", "source_close_at", "resume_at" }) |key| {
            const raw = x.object.get(key) orelse continue;
            const m = if (key[0] == 'r') fieldMap(maps, x, "resume_field") else src;
            try x.object.put(a, try std.mem.concat(a, u8, &.{ "raw_", key }), raw);
            try put(a, x.object, key, project(m, raw.int64()));
        }
    };
}
