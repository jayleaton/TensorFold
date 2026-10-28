//! Reclaim deleted hash buckets before tombstones can exhaust a bounded cache's free buckets.
const std = @import("std");

pub const Churn = struct {
    dead: u32 = 0,

    /// Called after each successful deletion; rehash needs no allocation and invalidates entry pointers.
    pub fn dropped(c: *Churn, comptime Key: type, map: anytype) void {
        c.dead += 1;
        if (@as(u64, c.dead) * 10 < map.capacity()) return;
        map.rehash(std.hash_map.AutoContext(Key){});
        c.dead = 0;
    }
};
