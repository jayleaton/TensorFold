//! Flash Next lanes' kept prompt states: keepers, late fills, the prompt cache's hooks and rank 1's notes.
const std = @import("std");
const envs = @import("cuda_env.zig");
const cl = @import("cuda_lanes.zig");
const lanes_core = @import("cuda_lanes_core.zig");
const fill_part = @import("cuda_lanes_fill.zig");

const Note = cl.Note;
const CacheHooks = cl.CacheHooks;
const passNow = fill_part.passNow;
const passSince = fill_part.passSince;

pub fn Keep(comptime E: type) type {
    return struct {
        const Self = lanes_core.Lanes(E);
        const Kept = Self.Kept;
        const Lane = Self.Lane;
        const SnapPtr = Self.SnapPtr;
        const of = Self.of;

        /// A tentative Kept of the pass at `at`: a keeper sequence takes the prompt's first `at` rows.
        pub fn keep(self: *Self, l: *Lane, snap: SnapPtr, at: u32) !*Kept {
            return self.keepAt(l, snap, at, false);
        }

        pub fn keepAt(self: *Self, l: *Lane, snap: SnapPtr, at: u32, late: bool) !*Kept {
            const e = self.e;
            if (late) {
                const k = try self.gpa.create(Kept);
                errdefer self.gpa.destroy(k);
                k.* = .{ .snap = snap, .keeper = null, .src = l.seq, .at = at, .bytes = self.footprint(at) };
                k.key = @intFromPtr(k);
                try self.late.append(self.gpa, k);
                self.kept_bytes += k.bytes;
                return k;
            }
            const timed = comptime @hasField(E, "io") and @hasField(E, "stream");
            const times = timed and envs.get("TF_FLASHNEXT_PASS_TIMES") != null;
            var t_last: std.Io.Timestamp = if (times) passNow(e) else undefined;
            const keeper = try e.newSeq(at + 1);
            errdefer e.freeSeq(keeper);
            const t_new = if (times) passSince(e, &t_last) else 0;
            try e.copyPrefix(keeper, l.seq, at, snap.mtp_len);
            if (times) std.log.info("keep at {d}: new sequence {d:.1} ms, copy {d:.1} ms", .{ at, t_new, passSince(e, &t_last) });
            // the keeper now holds `at` rows: say so, or a later copyPrefix from it refuses (PrefixPastSource)
            if (@hasField(E, "f")) {
                try e.f.setPos(&keeper.st, at);
                try e.f.setMtpLen(&keeper.st, snap.mtp_len);
            }
            const k = try self.gpa.create(Kept);
            k.* = .{ .snap = snap, .keeper = keeper, .at = at, .bytes = self.footprint(at) };
            k.key = @intFromPtr(k);
            self.kept_bytes += k.bytes;
            return k;
        }

        /// Fills late Kepts' keepers at the start of collective calls; rank 1 follows rank 0's order and adopt note.
        pub fn fillLate(self: *Self) void {
            if (self.late.items.len == 0) return;
            const e = self.e;
            for (self.late.items) |k| {
                const src = k.src orelse continue;
                k.src = null;
                const keeper = e.newSeq(k.at + 1) catch |err| {
                    k.broken = true;
                    std.log.warn("prompt cache: no state kept at {d} tokens ({s})", .{ k.at, @errorName(err) });
                    continue;
                };
                e.copyPrefix(keeper, src, k.at, k.snap.mtp_len) catch |err| {
                    e.freeSeq(keeper);
                    k.broken = true;
                    std.log.warn("prompt cache: no state kept at {d} tokens ({s})", .{ k.at, @errorName(err) });
                    continue;
                };
                if (@hasField(E, "f")) {
                    e.f.setPos(&keeper.st, k.at) catch {};
                    e.f.setMtpLen(&keeper.st, k.snap.mtp_len) catch {};
                }
                k.keeper = keeper;
            }
            self.late.clearRetainingCapacity();
        }

        pub fn freeKept(self: *Self, k: *Kept) void {
            self.kept_bytes -= k.bytes;
            if (std.mem.indexOfScalar(*Kept, self.late.items, k)) |i| _ = self.late.orderedRemove(i);
            if (k.keeper) |kp| self.e.freeSeq(kp);
            self.e.freeSnapshot(k.snap);
            self.gpa.destroy(k);
        }

        /// Pass end: rank 0 frees what the store did not adopt and tells rank 1, which keeps them all until that note.
        pub fn endPass(self: *Self) void {
            if (self.follower) return;
            if (self.notes) |n| if (self.tentative.items.len > 0) {
                var buf: [1 + 4 + 64 * 12]u8 = undefined;
                var len: usize = 5;
                buf[0] = @backingInt(Note.pass);
                var count: u32 = 0;
                for (self.tentative.items) |t| if (t.adopted and len + 12 <= buf.len) {
                    std.mem.writeInt(u32, buf[len..][0..4], t.at, .little);
                    std.mem.writeInt(u64, buf[len + 4 ..][0..8], t.kept.key, .little);
                    len += 12;
                    count += 1;
                };
                std.mem.writeInt(u32, buf[1..5], count, .little);
                n.send(n.ptr, buf[0..len]);
            };
            self.dropTentative();
        }

        /// The last pass's states the store did not adopt.
        pub fn dropTentative(self: *Self) void {
            for (self.tentative.items) |t| if (!t.adopted) self.freeKept(t.kept);
            self.tentative.clearRetainingCapacity();
        }

        pub fn keptOf(self: *Self, saved: ?*anyopaque) ?*Kept {
            const p = saved orelse return null;
            for (self.kept.items) |k| if (@as(*anyopaque, @ptrCast(k)) == p) return if (k.keeper == null) null else k;
            return null;
        }

        /// Both ranks' answer to a yes/no both must share (one rank: its own).
        pub fn agreeOn(self: *Self, ok: bool) !bool {
            const g = self.gate orelse return ok;
            return g.agree(g.ptr, ok) catch |e| {
                g.fatal(g.ptr, "the ranks' agreement", e);
                return e;
            };
        }

        // -- the prompt cache's family (core/prompt_cache.zig Snapshots; native/cuda.zig makes the Store) ------

        pub fn cacheHooks(self: *Self, cache_budget: u64) CacheHooks {
            return .{ .ptr = self, .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .budget = cache_budget };
        }

        pub fn bytesFn(ptr: *anyopaque, at: u32) u64 {
            return of(ptr).footprint(at);
        }

        /// The store keeps the pass's state at `at`: the tentative Kept made there.
        pub fn saveFn(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
            _ = owner;
            const self = of(ptr);
            for (self.tentative.items) |*t| if (t.at == at and !t.adopted) {
                t.adopted = true;
                try self.kept.append(self.gpa, t.kept);
                return t.kept;
            };
            return error.NoStateKept;
        }

        /// The backend restores inside its pass (LaneHost's lookup, not begin).
        pub fn restoreFn(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
            return error.RestoredInThePass;
        }

        pub fn dropFn(ptr: *anyopaque, saved: *anyopaque) void {
            const self = of(ptr);
            const k = self.keptOf(saved) orelse return;
            // Store.keep may evict a Kept this pass made before rank 1 adopted it: unadopt it so endPass frees it
            for (self.tentative.items) |*t| if (t.kept == k and t.adopted) {
                t.adopted = false;
                if (std.mem.indexOfScalar(*Kept, self.kept.items, k)) |i| _ = self.kept.swapRemove(i);
                return;
            };
            if (self.notes) |n| {
                var buf: [9]u8 = undefined;
                buf[0] = @backingInt(Note.drop);
                std.mem.writeInt(u64, buf[1..9], k.key, .little);
                n.send(n.ptr, &buf);
            }
            self.forgetKept(k);
        }

        pub fn forgetKept(self: *Self, k: *Kept) void {
            if (std.mem.indexOfScalar(*Kept, self.kept.items, k)) |i| _ = self.kept.swapRemove(i);
            self.freeKept(k);
        }

        // -- rank 1: rank 0's prompt cache decisions ----------------------------------------------------------

        /// A note of rank 0's store (cuda_mirror carries it): the last pass's adopted states, or a dropped one.
        pub fn applyNote(self: *Self, bytes: []const u8) !void {
            if (bytes.len == 0) return error.BadNote;
            switch (@as(Note, @fromBackingInt(bytes[0]))) {
                .pass => {
                    if (bytes.len < 5) return error.BadNote;
                    const n = std.mem.readInt(u32, bytes[1..5], .little);
                    if (bytes.len != 5 + 12 * @as(usize, n)) return error.BadNote;
                    for (0..n) |i| {
                        const at = std.mem.readInt(u32, bytes[5 + 12 * i ..][0..4], .little);
                        const key = std.mem.readInt(u64, bytes[9 + 12 * i ..][0..8], .little);
                        for (self.tentative.items) |*t| if (t.at == at and !t.adopted) {
                            t.adopted = true;
                            t.kept.key = key;
                            try self.kept.append(self.gpa, t.kept);
                            break;
                        };
                    }
                    self.dropTentative();
                },
                .drop => {
                    if (bytes.len != 9) return error.BadNote;
                    const key = std.mem.readInt(u64, bytes[1..9], .little);
                    for (self.kept.items) |k| if (k.key == key) return self.forgetKept(k);
                },
                _ => return error.BadNote,
            }
        }

        /// Rank 1's own state for rank 0's key (null: it holds none; the resume then runs from 0 on both).
        pub fn savedFor(self: *Self, key: u64) ?*anyopaque {
            for (self.kept.items) |k| if (k.key == key) return k;
            return null;
        }
    };
}
