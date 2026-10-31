//! TF_DSV41_KV_MIRROR: each slot's first positions of every split comp family kept whole on each rank (layout).

const std = @import("std");
const sessions = @import("sessions");

pub const env = "TF_DSV41_KV_MIRROR";

pub const Error = error{BadMirror};

/// The knob's positions, rounded up to whole pages of every residue (page x world); unset, blank or 0: off.
pub fn parse(text: ?[]const u8, page: u32, world: u32) Error!u64 {
    const v = std.mem.trim(u8, text orelse return 0, " \t");
    if (v.len == 0) return 0;
    const n = std.fmt.parseInt(u64, v, 10) catch return error.BadMirror;
    if (n > 1 << 40 or world == 0) return error.BadMirror;
    const unit = @as(u64, page) * world;
    return (n + unit - 1) / unit * unit;
}

/// The knob from the environment (positions, 0: off).
pub fn fromEnv(page: u32, world: u32) Error!u64 {
    const raw: []const u8 = if (std.c.getenv(env)) |v| std.mem.span(v) else return 0;
    return parse(raw, page, world) catch |e| {
        if (!@import("builtin").is_test) std.log.err("[tensorfold] {s}={s}: positions each slot mirrors (0: off)", .{ env, raw });
        return e;
    };
}

/// A family's mirror: each slot's run of `pages` pages (owner-major: an all-gather's order), then one discard page.
pub const Geometry = struct {
    slots: u32,
    /// logical pages a slot mirrors (a multiple of world)
    pages: u32,
    world: u32,

    pub fn of(positions: u64, page: u32, slots: u32, world: u32) Geometry {
        return .{ .slots = slots, .pages = @intCast(positions / page), .world = world };
    }

    /// Pages a family's mirror tensor holds: every slot's run, then the discard page (rows past the mirror).
    pub fn tensorPages(g: Geometry) u32 {
        return g.slots * g.pages + 1;
    }

    pub fn discard(g: Geometry) u32 {
        return g.slots * g.pages;
    }

    /// Slot s's mirror page of logical page p: owner p mod W's part of the run at turn p / W; past it the discard page.
    pub fn pageOf(g: Geometry, s: u32, p: u32) u32 {
        if (p >= g.pages) return g.discard();
        return s * g.pages + (p % g.world) * (g.pages / g.world) + p / g.world;
    }

    /// The device table int32 [slots, pts] ("s.kv.mt"): the PT `_kv_store` writes and the attention reads through.
    pub fn table(g: Geometry, pts: u32, out: []i32) void {
        for (0..g.slots) |s| for (0..pts) |p| {
            out[s * pts + p] = @intCast(g.pageOf(@intCast(s), @intCast(p)));
        };
    }

    /// Rows a rank sends in a fill: its pages of the run, whole; owner o's land at o x this in the slot's run.
    pub fn fillRows(g: Geometry, psh: u32) u32 {
        return (g.pages / g.world) << @intCast(psh);
    }

    /// A fill's row list: logical pages rank, rank + W, ... of the run through the slot's split table `local`.
    pub fn fillPhys(g: Geometry, rank: u32, psh: u32, local: []const u32, out: []u32) void {
        const q = @as(u32, 1) << @intCast(psh);
        for (0..g.pages / g.world) |j| {
            const lp = local[rank + g.world * @as(u32, @intCast(j))];
            for (0..q) |i| out[j * q + i] = (lp << @intCast(psh)) | @as(u32, @intCast(i));
        }
    }

    /// A rank's device bytes: each split comp family's mirror tensor and the table (pts pages a slot).
    pub fn bytes(g: Geometry, families: []const sessions.Family, page: u32, pts: u32) u64 {
        var b: u64 = 4 * @as(u64, g.slots) * pts;
        for (families) |f| if (f.split) {
            b += @as(u64, g.tensorPages()) * f.pageBytes(page);
        };
        return b;
    }
};

test {
    _ = @import("mirror_test.zig");
}
