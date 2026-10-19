//! TF_DSV41_PF_HOSTGAP: split.planUnion's one-block bytes from reused scratch, scanning only the rows named.

const std = @import("std");
const split = @import("split.zig");

/// TF_DSV41_PF_HOSTGAP=1: a prefill union planned by `Planner` into grow-only pinned staging (default off).
pub fn hostgapOn() bool {
    const v = std.mem.span(std.c.getenv("TF_DSV41_PF_HOSTGAP") orelse return false);
    return std.mem.eql(u8, std.mem.trim(u8, v, " \t"), "1");
}

/// Rows an owner's list can hold at most: its pages of the table, whole, and never past the entries.
pub fn ownerBound(pages: usize, psh: u32, world: u32, entries: usize) usize {
    return @min(entries, ((pages + world - 1) / world) << @intCast(psh));
}

pub const Planner = struct {
    /// a row's mark, then its index + 1 in its owner's list, then its token + 1; zero past `hi` between calls
    seen: std.ArrayList(u32) = .empty,
    /// rows [0, hi) hold marks (one past the highest row named)
    hi: usize = 0,
    m: u32 = 0,

    pub fn deinit(p: *Planner, gpa: std.mem.Allocator) void {
        p.seen.deinit(gpa);
        p.* = .{};
    }

    /// The union as one block: m rows an owner, or null past the cap (planUnion's blocks then); planUnion's errors.
    pub fn measure(p: *Planner, gpa: std.mem.Allocator, in: split.UnionIn) !?u32 {
        if (in.world < 1 or in.world > split.max_world or in.rank >= in.world) return error.BadWorld;
        const total = in.table.len << @intCast(in.psh);
        if (p.seen.items.len < total) {
            const old = p.seen.items.len;
            try p.seen.resize(gpa, total);
            @memset(p.seen.items[old..], 0);
        }
        const seen = p.seen.items[0..total];
        const rows = in.sel.len / in.k;
        var hi: usize = 0;
        for (in.sel[0 .. rows * in.k]) |t| if (t >= 0) {
            const u: usize = @intCast(t);
            if (u >= total) {
                @memset(seen[0..hi], 0);
                return error.RowOutsideTable;
            }
            seen[u] = 1;
            hi = @max(hi, u + 1);
        };
        p.hi = hi;
        const per = @as(usize, 1) << @intCast(in.psh);
        var sizes: [split.max_world]u32 = @splat(0);
        var o: u32 = 0;
        var a: usize = 0;
        while (a < hi) : (a += per) {
            for (seen[a..@min(a + per, hi)]) |*x| if (x.* != 0) {
                sizes[o] += 1;
                x.* = sizes[o];
            };
            o = if (o + 1 == in.world) 0 else o + 1;
        }
        var m: u32 = 1;
        for (sizes[0..in.world]) |sz| m = @max(m, sz);
        if (@as(u64, in.world + 1) * m * in.row_bytes > in.cap and rows > 1) {
            @memset(seen[0..hi], 0);
            return null;
        }
        p.m = m;
        return m;
    }

    /// The measured block's phys [m] (this rank's local rows, padding 0) and tokens [rows x k]; clears the marks.
    pub fn write(p: *Planner, in: split.UnionIn, phys: []u32, tokens: []i32) void {
        const seen = p.seen.items[0..p.hi];
        const per = @as(usize, 1) << @intCast(in.psh);
        const mask: u32 = @intCast(per - 1);
        @memset(phys, 0);
        var o: u32 = 0;
        var a: usize = 0;
        var pg: usize = 0;
        while (a < seen.len) : ({
            a += per;
            pg += 1;
        }) {
            const base = o * p.m;
            const local = in.table[pg] << @intCast(in.psh);
            for (seen[a..@min(a + per, seen.len)], a..) |*x, t| if (x.* != 0) {
                if (o == in.rank) phys[x.* - 1] = local | (@as(u32, @intCast(t)) & mask);
                x.* += base;
            };
            o = if (o + 1 == in.world) 0 else o + 1;
        }
        for (in.sel[0..tokens.len], tokens) |t, *out| out.* = if (t < 0) -1 else @intCast(seen[@intCast(t)] - 1);
        @memset(seen, 0);
        p.hi = 0;
    }
};
