//! Optional K4 scratch placement; role names, views and launches stay unchanged.
const std = @import("std");
const buffers = @import("buffers.zig");
const calls = @import("block_calls.zig");

pub const members = [_][]const u8{ "L.k4~q", "L.k4~z", "s.k4~ex.xg", "L.k4~o", "s.k4~ex.z" };
const groups = [_]usize{ 0, 0, 0, 1, 1 };

pub fn enabled(tbo: bool) bool {
    const raw = std.c.getenv("TF_DSV41_PF_SCRATCH_ALIAS");
    return active(if (raw) |v| std.mem.span(v) else null, tbo);
}

pub fn active(raw: ?[]const u8, tbo: bool) bool {
    return !tbo and std.mem.eql(u8, raw orelse return false, "1");
}

fn member(role: []const u8) ?usize {
    for (members, 0..) |name, i| if (std.mem.eql(u8, role, name)) return i;
    return null;
}

pub const Layout = struct {
    on: bool,
    bytes: u64 = 0,
    at: u64 = 0,
    size: [2]u64 = @splat(0),
    base: [2]?u64 = @splat(null),

    pub fn init(p: *const buffers.Plan, on: bool) Layout {
        var l: Layout = .{ .on = on };
        for (p.sizes.keys(), p.sizes.values()) |role, bytes| {
            if (std.mem.indexOf(u8, role, "k4~") == null) continue;
            const n = std.mem.alignForward(u64, bytes, 256);
            if (on) if (member(role)) |i| {
                l.size[groups[i]] = @max(l.size[groups[i]], n);
                continue;
            };
            l.bytes += n;
        }
        for (l.size) |n| l.bytes += n;
        return l;
    }

    /// Called in plan order, exactly once per K4 role; off retains every original aligned offset.
    pub fn next(l: *Layout, role: []const u8, bytes: u64) u64 {
        if (l.on) if (member(role)) |i| {
            const g = groups[i];
            if (l.base[g]) |at| return at;
            const at = l.at;
            l.base[g] = at;
            l.at += l.size[g];
            return at;
        };
        const at = l.at;
        l.at += std.mem.alignForward(u64, bytes, 256);
        return at;
    }
};

fn touches(x: calls.Arg, i: usize) bool {
    return switch (x) {
        .t, .opaque_table => |t| t.role == .buf and std.mem.eql(u8, t.role.buf, members[i]),
        .list => |xs| for (xs) |y| {
            if (touches(y, i)) break true;
        } else false,
        else => false,
    };
}

fn disjoint(first: [members.len]?usize, last: [members.len]usize) !void {
    for (first, 0..) |a, i| {
        const ai = a orelse continue;
        for (first[i + 1 ..], i + 1..) |b, j| {
            const bj = b orelse continue;
            if (groups[i] == groups[j] and !(last[i] < bj or last[j] < ai)) return error.ScratchLifetimeOverlap;
        }
    }
}

/// Runner.issue sends these members only to main; FIFO device order, not host completion, ends each lifetime.
pub fn validate(cs: []const calls.Call) !void {
    var first: [members.len]?usize = @splat(null);
    var last: [members.len]usize = @splat(0);
    for (cs, 0..) |c, seq| {
        // Seg.run sets begin only at the window start and the top of its layer loop.
        if (c.begin != .none) {
            try disjoint(first, last);
            first = @splat(null);
            last = @splat(0);
        }
        for (members, 0..) |_, i| {
            var used = false;
            for (c.args) |x| used = used or touches(x.arg, i);
            if (!used) continue;
            if (c.side or c.defer_side or std.mem.startsWith(u8, c.name, "tf_dsv41_l2p"))
                return error.ScratchAsyncLifetime;
            if (first[i] == null) first[i] = seq;
            last[i] = seq;
        }
    }
    try disjoint(first, last);
}
