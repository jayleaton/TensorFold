//! Where a prompt leaves the stored entries: the session miss line's numbers (prefix.Index's paths and tails).

const std = @import("std");
const prefix = @import("prefix.zig");
const Digest = prefix.Digest;
const none = prefix.none;

/// A prompt's first token no stored path holds (`at`) and the longest entry sharing tokens [0, at) (`stored`).
pub const Divergence = struct {
    at: u64 = 0,
    stored: u64 = 0,

    fn take(d: *Divergence, at: u64, stored: u64) void {
        if (stored == 0) return;
        if (at > d.at or d.stored == 0 or (at == d.at and stored > d.stored)) d.* = .{ .at = at, .stored = stored };
    }
};

fn common(a: []const i32, b: []const i32) u64 {
    const n = @min(a.len, b.len);
    for (0..n) |i| if (a[i] != b[i]) return i;
    return n;
}

/// Prompt against index x's tag paths (a page's tokens compared while RAM holds them); digests: its chain.
pub fn diverge(x: anytype, tag: u32, prompt: []const i32, digests: []const Digest) !Divergence {
    const full: u32 = @intCast(prompt.len / x.page);
    const r = x.by_digest.get(prefix.root(tag)) orelse return .{};
    // the deepest node on the prompt's path, as Index.find
    var lo: u32 = 0;
    var hi: u32 = full;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (x.by_digest.contains(digests[mid - 1])) lo = mid else hi = mid - 1;
    }
    const n: u32 = if (lo == 0) r else x.by_digest.get(digests[lo - 1]).?;
    const at: u64 = @as(u64, lo) * x.page;
    const rest = prompt[@intCast(at)..];
    var best: Divergence = .{ .at = at };
    var e = x.nodes.items[n].first_entry;
    while (e != none) : (e = x.entries.items[e].next_at) {
        const en = &x.entries.items[e];
        best.take(at + common(en.tail, rest), en.pos);
    }
    var c = x.nodes.items[n].first_child;
    while (c != none) : (c = x.nodes.items[c].next_sib) {
        const ch = &x.nodes.items[c];
        if (ch.sub_all == 0) continue;
        best.take(at + if (ch.tokens) |t| common(t, rest) else 0, try longest(x, c));
    }
    return best;
}

/// The longest entry's pos in node n's subtree.
fn longest(x: anytype, n: u32) !u64 {
    var stack: std.ArrayList(u32) = .empty;
    defer stack.deinit(x.gpa);
    try stack.append(x.gpa, n);
    var m: u64 = 0;
    while (stack.pop()) |k| {
        const nd = &x.nodes.items[k];
        var e = nd.first_entry;
        while (e != none) : (e = x.entries.items[e].next_at) m = @max(m, x.entries.items[e].pos);
        var c = nd.first_child;
        while (c != none) : (c = x.nodes.items[c].next_sib) if (x.nodes.items[c].sub_all > 0) try stack.append(x.gpa, c);
    }
    return m;
}

const T = prefix.Index(u64);

fn add(x: *T, ids: []const i32) !void {
    var d: [64]Digest = undefined;
    prefix.chain(7, ids, x.page, d[0 .. ids.len / x.page]);
    _ = try x.insert(7, ids, d[0 .. ids.len / x.page], true, 0);
}

fn divergeOf(x: *T, prompt: []const i32) !Divergence {
    var d: [64]Digest = undefined;
    prefix.chain(7, prompt, x.page, d[0 .. prompt.len / x.page]);
    return diverge(x, 7, prompt, d[0 .. prompt.len / x.page]);
}

test "divergence: the first token past every stored path, inside a page and at a tail, with the longest sharer" {
    const gpa = std.testing.allocator;
    var x = T.init(gpa, 16);
    defer x.deinit();
    var chat: [300]i32 = undefined;
    for (&chat, 0..) |*t, i| t.* = @intCast(i + 1);
    try std.testing.expectEqual(Divergence{}, try divergeOf(&x, &chat));
    try add(&x, chat[0..40]);
    try add(&x, chat[0..250]);
    try add(&x, chat[0..120]);
    var p = chat;
    p[100] = -1; // inside a page past the 40-token entry: the 250- and 120-token entries both share 100 tokens
    try std.testing.expectEqual(Divergence{ .at = 100, .stored = 250 }, try divergeOf(&x, &p));
    p = chat;
    p[37] = -1; // inside the 40-token entry's tail
    try std.testing.expectEqual(Divergence{ .at = 37, .stored = 250 }, try divergeOf(&x, &p));
    p = chat;
    p[0] = -1;
    try std.testing.expectEqual(Divergence{ .at = 0, .stored = 250 }, try divergeOf(&x, &p));
    // a longer prompt holding every entry: nothing past the longest
    try std.testing.expectEqual(Divergence{ .at = 250, .stored = 250 }, try divergeOf(&x, &chat));
}
