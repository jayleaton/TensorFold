//! A unit's results by exact input: a hit returns the words a pure computation produced from the same key before.
const std = @import("std");

pub const slots_per_unit = 256;

const Entry = struct {
    used: bool = false,
    hash: u64 = 0,
    key: std.ArrayList(u64) = .empty,
    words: std.ArrayList(u32) = .empty,
};

/// Direct-mapped by key hash; a colliding put replaces the older entry, so a lookup only ever misses, never lies.
pub const Memo = struct {
    entries: []Entry = &.{},
    hits: u64 = 0,
    misses: u64 = 0,

    pub fn deinit(m: *Memo, gpa: std.mem.Allocator) void {
        for (m.entries) |*e| {
            e.key.deinit(gpa);
            e.words.deinit(gpa);
        }
        gpa.free(m.entries);
        m.* = .{};
    }

    pub fn hash(key: []const u64) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(key));
    }

    pub fn find(m: *Memo, key: []const u64, h: u64) ?[]const u32 {
        if (m.entries.len == 0) return null;
        const e = &m.entries[h % m.entries.len];
        if (!e.used or e.hash != h or !std.mem.eql(u64, e.key.items, key)) {
            m.misses += 1;
            return null;
        }
        m.hits += 1;
        return e.words.items;
    }

    pub fn put(m: *Memo, gpa: std.mem.Allocator, key: []const u64, h: u64, words: []const u32) !void {
        if (m.entries.len == 0) {
            m.entries = try gpa.alloc(Entry, slots_per_unit);
            for (m.entries) |*e| e.* = .{};
        }
        const e = &m.entries[h % m.entries.len];
        e.used = false;
        e.key.clearRetainingCapacity();
        try e.key.appendSlice(gpa, key);
        e.words.clearRetainingCapacity();
        try e.words.appendSlice(gpa, words);
        e.hash = h;
        e.used = true;
    }
};

test "a memo returns only what was stored under the same key" {
    const gpa = std.testing.allocator;
    var m: Memo = .{};
    defer m.deinit(gpa);
    const k1 = [_]u64{ 1, 2 };
    const k2 = [_]u64{ 1, 3 };
    try std.testing.expect(m.find(&k1, Memo.hash(&k1)) == null);
    try m.put(gpa, &k1, Memo.hash(&k1), &.{ 7, 8 });
    try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, m.find(&k1, Memo.hash(&k1)).?);
    try std.testing.expect(m.find(&k2, Memo.hash(&k1)) == null);
    try std.testing.expect(m.find(&k2, Memo.hash(&k2)) == null);
}
