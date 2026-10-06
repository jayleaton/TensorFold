//! Session memory, family-neutral: the paged KV pool's book, the bytes behind its pages, prefix entries, the RAM tier with its residency budget, and the streamed NVMe tier.

pub const freeset = @import("freeset.zig");
pub const pool = @import("pool.zig");
pub const Pool = pool.Pool;
pub const Slot = pool.Slot;
pub const Family = pool.Family;
pub const Layout = pool.Layout;
pub const pagestore = @import("pagestore.zig");
pub const PageStore = pagestore.PageStore;
pub const HostPool = pagestore.HostPool;
pub const prefix = @import("prefix.zig");
pub const lanes = @import("lanes.zig");

test {
    _ = freeset;
    _ = pool;
    _ = pagestore;
    _ = prefix;
    _ = lanes;
}
