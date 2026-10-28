//! Owned image inputs until a request finishes; admission is atomic even when allocation fails.
const std = @import("std");
const held = @import("held.zig");
const prep = @import("prep.zig");
const Allocator = std.mem.Allocator;
const Churn = @import("../map_churn.zig").Churn;
pub const Loc = struct { img: *const held.HeldImage, idx: u32, refs: u32 };

pub const Store = struct {
    gpa: Allocator,
    where: std.AutoHashMapUnmanaged(u32, Loc) = .empty,
    holds: std.AutoHashMapUnmanaged(u64, *const held.Held) = .empty,
    where_churn: Churn = .{},
    holds_churn: Churn = .{},

    pub fn deinit(s: *Store) void {
        var it = s.holds.valueIterator();
        while (it.next()) |h| s.free(h.*);
        s.holds.deinit(s.gpa);
        s.where.deinit(s.gpa);
    }

    pub fn hold(s: *Store, id: u64, src: *const held.Held) !void {
        if (s.holds.contains(id)) return error.DuplicateRequest;
        const h = try s.copy(src);
        errdefer s.free(h);
        var pending: std.AutoHashMapUnmanaged(u32, Loc) = .empty;
        defer pending.deinit(s.gpa);
        for (h.slice(), 0..) |*im, k| {
            if (seen(h, k)) continue;
            for (im.vids[0..im.n_vids], 0..) |v, i| {
                if (v < (1 << 24) or v >= (1 << 31)) return error.InvalidImageId;
                if (pending.contains(v)) return error.ImageIdCollision;
                if (s.where.get(v)) |loc| {
                    if (loc.idx != i or !std.mem.eql(u8, &loc.img.digest, &im.digest))
                        return error.ImageIdCollision;
                    if (loc.refs == std.math.maxInt(u32)) return error.TooManyHolders;
                }
                try pending.put(s.gpa, v, .{ .img = im, .idx = @intCast(i), .refs = 1 });
            }
        }
        try s.holds.ensureUnusedCapacity(s.gpa, 1);
        try s.where.ensureUnusedCapacity(s.gpa, pending.count());
        s.holds.putAssumeCapacity(id, h);
        var it = pending.iterator();
        while (it.next()) |e| {
            const loc = s.where.getOrPutAssumeCapacity(e.key_ptr.*);
            if (loc.found_existing) loc.value_ptr.refs += 1 else loc.value_ptr.* = e.value_ptr.*;
        }
    }

    pub fn release(s: *Store, id: u64) void {
        const h = (s.holds.fetchRemove(id) orelse return).value;
        s.holds_churn.dropped(u64, &s.holds);
        defer s.free(h);
        for (h.slice(), 0..) |*im, k| {
            if (seen(h, k)) continue;
            for (im.vids[0..im.n_vids]) |v| {
                const loc = s.where.getPtr(v) orelse unreachable;
                loc.refs -= 1;
                if (loc.refs == 0) {
                    _ = s.where.remove(v);
                    s.where_churn.dropped(u32, &s.where);
                } else if (loc.img == im) {
                    loc.img = s.other(im.digest) orelse unreachable;
                }
            }
        }
    }

    fn other(s: *Store, digest: [32]u8) ?*const held.HeldImage {
        var it = s.holds.valueIterator();
        while (it.next()) |h| for (h.*.slice()) |*im| {
            if (std.mem.eql(u8, &im.digest, &digest)) return im;
        };
        return null;
    }

    fn seen(h: *const held.Held, k: usize) bool {
        for (h.slice()[0..k]) |*im| if (std.mem.eql(u8, &im.digest, &h.slice()[k].digest)) return true;
        return false;
    }

    fn copy(s: *Store, src: *const held.Held) !*const held.Held {
        const imgs = try s.gpa.alloc(held.HeldImage, src.n);
        var done: usize = 0;
        errdefer {
            for (imgs[0..done]) |*im| s.freeImage(im);
            s.gpa.free(imgs);
        }
        for (src.slice(), imgs) |*im, *out| {
            if (im.vit_h == 0 or im.vit_w == 0 or im.vit_h > 4096 or im.vit_w > 4096)
                return error.VisionGrid;
            const lh = (im.vit_h + 2) / 3;
            const lw = (im.vit_w + 2) / 3;
            if (im.llm_h != lh or im.llm_w != lw or im.n_vids != prep.spanLen(lh, lw))
                return error.VisionGrid;
            out.* = im.*;
            const np = @as(usize, im.vit_h) * im.vit_w * 3 * 14 * 14;
            const patches = try s.gpa.dupe(u16, im.patches[0..np]);
            out.vids = (s.gpa.dupe(u32, im.vids[0..im.n_vids]) catch |e| {
                s.gpa.free(patches);
                return e;
            }).ptr;
            out.patches = patches.ptr;
            done += 1;
        }
        const h = try s.gpa.create(held.Held);
        h.* = .{ .images = imgs.ptr, .n = imgs.len };
        return h;
    }

    fn freeImage(s: *Store, im: *const held.HeldImage) void {
        s.gpa.free(im.patches[0 .. @as(usize, im.vit_h) * im.vit_w * 3 * 14 * 14]);
        s.gpa.free(im.vids[0..im.n_vids]);
    }

    fn free(s: *Store, h: *const held.Held) void {
        for (h.slice()) |*im| s.freeImage(im);
        s.gpa.free(h.images[0..h.n]);
        s.gpa.destroy(h);
    }
};
