//! TF_DSV41_AOT_CACHE=1: an AOT set's launch lookups kept by exact match key; a repeated launch skips the search.

const std = @import("std");
const aot = @import("aot.zig");
const Arg = aot.Arg;
const Const = aot.Const;
const Spec = aot.Spec;

/// The longest launch key the cache holds (a longer one goes to the search every time).
pub const key_max = 4096;

/// What `matches` reads of a launch, exactly: names in order, each argument's type and class, every constexpr's bits.
pub fn matchKey(buf: []u8, function: []const u8, args: []const Arg, consts: []const Const) ?[]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    keyInto(&w, function, args, consts) catch return null;
    return w.buffered();
}

fn keyInto(w: *std.Io.Writer, function: []const u8, args: []const Arg, consts: []const Const) !void {
    try field(w, function);
    for (args) |a| {
        try field(w, a.name);
        switch (a.value) {
            .ptr => |x| {
                try w.writeByte(if (x.addr % 16 == 0) 'P' else 'p');
                try field(w, x.ty);
            },
            .i32 => |x| try w.writeByte(if (x == 1) '1' else if (@mod(x, 16) == 0) 'D' else 'i'),
            .f32 => try w.writeByte('f'),
            .u64 => |x| try w.writeByte(if (x % 16 == 0) 'U' else 'u'),
        }
    }
    try w.writeByte('|');
    for (consts) |c| {
        try field(w, c.name);
        try w.writeByte(@as(u8, @intFromBool(c.int != null)) | @as(u8, @intFromBool(c.f32 != null)) << 1);
        if (c.int) |x| try w.writeInt(i64, x, .little);
        if (c.f32) |x| try w.writeInt(u32, @bitCast(x), .little);
    }
}

fn field(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeInt(u16, std.math.cast(u16, s.len) orelse return error.WriteFailed, .little);
    try w.writeAll(s);
}

/// `function`'s variants' specs in the set's order (`set`: an aot.Set), built in `a`.
pub fn collect(a: std.mem.Allocator, set: anytype, function: []const u8) ![]const Spec {
    var mine: std.ArrayList(Spec) = .empty;
    for (set.variants) |v| if (std.mem.eql(u8, v.spec.@"fn", function)) try mine.append(a, v.spec);
    return mine.items;
}

/// Launch key (`matchKey`) -> the search's variant and its parameters' argument slots; each function's spec list.
pub const Cache = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    picks: std.StringHashMapUnmanaged(Hit) = .empty,
    specs: std.StringHashMapUnmanaged([]const Spec) = .empty,
    hits: u64 = 0,

    pub const Hit = struct { at: u32, slots: []const u16 };

    pub fn on() bool {
        const v = std.c.getenv("TF_DSV41_AOT_CACHE") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
    }

    pub fn create(gpa: std.mem.Allocator, io: std.Io) !*Cache {
        const c = try gpa.create(Cache);
        c.* = .{ .gpa = gpa, .io = io };
        return c;
    }

    pub fn destroy(c: *Cache) void {
        var it = c.picks.iterator();
        while (it.next()) |e| {
            c.gpa.free(e.key_ptr.*);
            c.gpa.free(e.value_ptr.slots);
        }
        c.picks.deinit(c.gpa);
        var is = c.specs.iterator();
        while (is.next()) |e| {
            c.gpa.free(e.key_ptr.*);
            c.gpa.free(e.value_ptr.*);
        }
        c.specs.deinit(c.gpa);
        c.gpa.destroy(c);
    }

    pub fn get(c: *Cache, key: []const u8) ?Hit {
        c.mutex.lockUncancelable(c.io);
        defer c.mutex.unlock(c.io);
        const h = c.picks.get(key) orelse return null;
        c.hits += 1;
        return h;
    }

    /// Keeps the search's pick of `key` (variant `at`); a failed allocation leaves the key to the search.
    pub fn put(c: *Cache, key: []const u8, at: u32, spec: Spec, args: []const Arg) void {
        const slots = c.gpa.alloc(u16, spec.params.len) catch return;
        for (spec.params, slots) |p, *s| s.* = for (args, 0..) |x, i| {
            if (std.mem.eql(u8, x.name, p.name)) break std.math.cast(u16, i) orelse return c.gpa.free(slots);
        } else return c.gpa.free(slots);
        const k = c.gpa.dupe(u8, key) catch return c.gpa.free(slots);
        c.mutex.lockUncancelable(c.io);
        defer c.mutex.unlock(c.io);
        const got = c.picks.getOrPut(c.gpa, k) catch {
            c.gpa.free(k);
            return c.gpa.free(slots);
        };
        if (got.found_existing) {
            c.gpa.free(k);
            return c.gpa.free(slots);
        }
        got.value_ptr.* = .{ .at = at, .slots = slots };
    }

    /// `collect`'s list of `function` (`set`: an aot.Set), built once and held by the cache.
    pub fn specsOf(c: *Cache, set: anytype, function: []const u8) ![]const Spec {
        c.mutex.lockUncancelable(c.io);
        defer c.mutex.unlock(c.io);
        if (c.specs.get(function)) |s| return s;
        var list: std.ArrayList(Spec) = .empty;
        errdefer list.deinit(c.gpa);
        for (set.variants) |v| if (std.mem.eql(u8, v.spec.@"fn", function)) try list.append(c.gpa, v.spec);
        const k = try c.gpa.dupe(u8, function);
        errdefer c.gpa.free(k);
        const s = try list.toOwnedSlice(c.gpa);
        errdefer c.gpa.free(s);
        try c.specs.put(c.gpa, k, s);
        return s;
    }
};
