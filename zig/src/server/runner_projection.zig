//! Runner receipt anchors projected from the engine's raw field bytes into the reply's fields as the client gets them.
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const model_text = @import("model_text.zig");
const chat = @import("chat.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// One field's map: final byte p is raw byte `lead + p`, for p in [0, kept].
pub const Map = struct { raw: usize, lead: usize, kept: usize };

/// The runner's field bytes as runner_text.zig keeps them (none for the think close); null without token bytes.
pub fn rawFields(a: Allocator, text: model_text.Text, tokens: []const u32, think_end: ?u32) Allocator.Error!?[2][]const u8 {
    if (text.vtable.token_bytes == null) return null;
    var fields: [2]std.ArrayList(u8) = .{ .empty, .empty };
    var field: usize = if (think_end == null) 1 else 0;
    for (tokens) |t| {
        if (field == 0 and t == think_end.?) {
            field = 1;
            continue;
        }
        try fields[field].appendSlice(a, (try text.tokenBytes(a, t)) orelse "");
    }
    return .{ fields[0].items, fields[1].items };
}

/// The final field as one raw run after a blank head, or after any head when it occurs once; else null (U+FFFD, cuts).
pub fn mapOf(raw: []const u8, final: []const u8, open: []const u8) ?Map {
    const lead = if (final.len == 0) raw.len else std.mem.indexOf(u8, raw, final) orelse return null;
    var head = raw[0..lead];
    if (open.len > 0 and std.mem.startsWith(u8, head, open)) head = head[open.len..];
    const once = final.len > 0 and std.mem.indexOfPos(u8, raw, lead + 1, final) == null;
    if (reply_text.pyStrip(head).len > 0 and !once) return null;
    return .{ .raw = raw.len, .lead = lead, .kept = final.len };
}

/// The reasoning and content a stream sent, read off each chunk as the client gets it.
pub const Sent = struct {
    fields: [2]std.ArrayList(u8) = .{ .empty, .empty },
    streamed: bool = false,

    pub fn seen(s: *Sent, a: Allocator, payload: Value) Allocator.Error!Value {
        s.streamed = true;
        const choices = (payload.get("choices") orelse return payload).items() orelse return payload;
        if (choices.len == 0) return payload;
        const c = choices[0];
        if (c.get("delta")) |d| {
            try s.fields[0].appendSlice(a, json.strOr(d.get("reasoning_content")));
            try s.fields[1].appendSlice(a, json.strOr(d.get("content")));
        } else try s.fields[1].appendSlice(a, json.strOr(c.get("text")));
        return payload;
    }
};

/// The runtime with abstain anchors mapped onto the fields the client got: as sent, else after tool parsing.
pub fn runtimeOf(a: Allocator, reply: *const chat.Reply, sent: *const Sent, is_chat: bool) Allocator.Error!Value {
    const raw = reply.runner_raw orelse return reply.runtime;
    const runner = reply.runtime.get("runner") orelse return reply.runtime;
    const got: [2][]const u8 = if (sent.streamed) .{ sent.fields[0].items, sent.fields[1].items } else .{ if (is_chat) reply.reasoning orelse "" else "", reply.content };
    try apply(a, runner, .{ mapOf(raw[0], got[0], "<think>"), mapOf(raw[1], got[1], "") });
    return reply.runtime;
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

fn fieldOf(v: Value, key: []const u8) ?usize {
    const name = v.strField(key) orelse return null;
    if (std.mem.eql(u8, name, "reasoning")) return 0;
    if (std.mem.eql(u8, name, "content")) return 1;
    return null;
}

const anchors = [_][]const u8{ "source_at", "source_end", "source_close_at", "resume_at" };

/// A raw anchor past the bytes rebuilt here (tokens the runner fed that the reply dropped): that field has no map.
fn bound(maps: *[2]?Map, i: ?usize, raw: ?Value) void {
    const f = i orelse return;
    const m = maps[f] orelse return;
    const r = (raw orelse return).int64() orelse return;
    if (r < 0 or r > m.raw) maps[f] = null;
}

/// With abstain records only: `field_projection`, cells' `final_at`, abstain anchors in final bytes (raw: `raw_*`).
pub fn apply(a: Allocator, runner: Value, given: [2]?Map) !void {
    const abstains = runner.get("abstains") orelse return;
    const cells = if (runner.get("cells")) |c| c.items() orelse &.{} else &.{};
    var maps = given;
    for (cells) |c| bound(&maps, fieldOf(c, "field"), c.get("at"));
    for (abstains.items() orelse &.{}) |x| for (anchors) |key|
        bound(&maps, fieldOf(x, if (key[0] == 'r') "resume_field" else "source_field"), x.get(key));
    const fields = try json.newObject(a);
    try fields.put(a, "reasoning", try mapValue(a, maps[0]));
    try fields.put(a, "content", try mapValue(a, maps[1]));
    try runner.object.put(a, "field_projection", .{ .object = fields });
    for (cells) |c| if (c == .object) {
        const m = if (fieldOf(c, "field")) |f| maps[f] else null;
        try put(a, c.object, "final_at", project(m, if (c.get("at")) |v| v.int64() else null));
    };
    for (abstains.items() orelse &.{}) |x| if (x == .object) for (anchors) |key| {
        const raw = x.object.get(key) orelse continue;
        const f = fieldOf(x, if (key[0] == 'r') "resume_field" else "source_field");
        try x.object.put(a, try std.mem.concat(a, u8, &.{ "raw_", key }), raw);
        try put(a, x.object, key, project(if (f) |i| maps[i] else null, raw.int64()));
    };
}
