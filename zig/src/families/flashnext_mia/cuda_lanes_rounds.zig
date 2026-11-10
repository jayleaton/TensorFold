//! Flash Next lanes' rounds: admission and lane lifetimes, shared verify windows, commits, drafts and release.
const std = @import("std");
const lanes = @import("lanes");
const media_mod = @import("cuda_media.zig");
const cl = @import("cuda_lanes.zig");
const lanes_core = @import("cuda_lanes_core.zig");

const be = lanes.backend;
const max_rows = cl.max_rows;
const ring = cl.ring;
const max_streams = cl.max_streams;

pub fn Rounds(comptime E: type) type {
    return struct {
        const Self = lanes_core.Lanes(E);
        const Lane = Self.Lane;
        const of = Self.of;

        /// An admitted prompt's window and reservation (admit), for its lane (makeLane).
        pub const Admission = struct { limit: usize, need: usize, before: usize };

        /// Local prefill admission (no collective); ranks agree before makeLane so none waits alone in a collective.
        pub fn admit(self: *Self, s: *lanes.Stream, ahead: usize) !Admission {
            const ids = s.prompt();
            const limit = ids.len + s.max_new + self.window;
            if (ids.len == 0 or limit > self.context) return error.PromptTooLong;
            // an image or video prompt also holds its rotary table and features while it fills (attachMedia)
            const media_bytes: usize = if (media_mod.of(s)) |m| std.mem.alignForward(usize, ids.len * 12, 256) + std.mem.alignForward(usize, m.featureBytes(), 256) else 0;
            const need = self.footprint(limit) + media_bytes;
            const before = if (self.lanes.get(s)) |old| old.reserved else 0;
            if (!self.overcommit and self.reserved - before + ahead + need + self.kept_bytes > self.budget()) return error.NoRoom;
            return .{ .limit = limit, .need = need, .before = before };
        }

        /// The agreed prefill's steps: the last round's commits, the stream's lane and a new sequence.
        pub fn makeLane(self: *Self, s: *lanes.Stream, a: Admission) !*Lane {
            const e = self.e;
            const limit = a.limit;
            const need = a.need;
            const before = a.before;
            try self.settle();
            const l = if (self.lanes.get(s)) |old| blk: {
                e.freeSeq(old.seq);
                break :blk old;
            } else blk: {
                const l = try self.gpa.create(Lane);
                errdefer self.gpa.destroy(l);
                try self.lanes.put(self.gpa, s, l);
                break :blk l;
            };
            self.reserved -= before;
            l.* = .{ .seq = e.newSeq(limit) catch |err| {
                _ = self.lanes.remove(s);
                self.gpa.destroy(l);
                return err;
            } };
            l.reserved = need;
            self.reserved += need;
            return l;
        }

        /// A stream's caches at its whole window (the engine's seqBytes; 0 for an engine that does not say).
        pub fn footprint(self: *const Self, rows: usize) usize {
            return if (@hasDecl(E, "seqBytes")) self.e.seqBytes(rows) else 0;
        }

        /// The bytes the sequences' caches may take on this rank (the engine's budget; unbounded without one).
        pub fn budget(self: *const Self) usize {
            return if (@hasField(E, "budget")) self.e.budget.limit else std.math.maxInt(usize);
        }

        pub fn dropLane(self: *Self, s: *const lanes.Stream, l: *Lane) void {
            for (self.late.items) |k| if (k.src == l.seq) {
                k.src = null;
                k.broken = true;
            };
            _ = self.lanes.remove(s);
            self.stopFill(l);
            self.reserved -= l.reserved;
            self.forget(l);
            self.e.freeSeq(l.seq);
            self.gpa.destroy(l);
        }

        pub fn forget(self: *Self, l: *Lane) void {
            for (self.opened.items, 0..) |o, i| if (o == l) {
                _ = self.opened.swapRemove(i);
                return;
            };
        }

        pub fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
            const self = of(ptr);
            if (position != s.prompt_len) return error.PositionMismatch;
            if (self.lanes.get(s)) |l| if (l.first) |h| {
                l.first = null;
                return h;
            };
            if (self.next == 0) return error.NoFirstToken;
            return self.next - 1;
        }

        pub fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
            _ = .{ ptr, s, feed, position };
            return error.NotPipelined;
        }

        pub fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
            const self = of(ptr);
            if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
            return self.drawn[handle % ring];
        }

        /// Every stream's window in one forward, each row drawn at its position; extra held drafts cost rows, not bits.
        pub fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const self = of(ptr);
            self.fillLate();
            var ids: [max_streams][max_rows]u32 = undefined;
            var wins: [max_streams]E.Window = undefined;
            var ls: [max_streams]*Lane = undefined;
            self.prepareVerify(windows, &ids, &wins, &ls) catch |err| {
                try self.agree(err);
                unreachable;
            };
            try self.agree(null);
            const e = self.e;
            const n = windows.len;
            e.verifyMany(wins[0..n]) catch |err| return self.lost("a round's forward", err);
            self.round += 1;
            for (windows, out, ls[0..n], 0..) |w, o, l, i| {
                const rows = w.rows();
                e.sampleWindow(i, w.stream.sampling, o.sampled[0..rows]) catch |err| return self.lost("a window's draws", err);
                @memcpy(o.drafts[0 .. rows - 1], ids[i][1..rows]);
                l.held_n = 0;
                l.pending_rows = rows;
                l.index = i;
                l.round = self.round;
                self.opened.append(self.gpa, l) catch |err| return self.lost("a window's bookkeeping", err);
            }
        }

        /// The windows' checks and rows, the last round committed.
        pub fn prepareVerify(self: *Self, windows: []const be.Window, ids: *[max_streams][max_rows]u32, wins: *[max_streams]E.Window, ls: *[max_streams]*Lane) !void {
            if (windows.len == 0 or windows.len > self.streams) return error.TooManyStreams;
            try self.opened.ensureTotalCapacity(self.gpa, windows.len);
            try self.settle();
            for (windows, 0..) |w, i| {
                if (w.parents != null) return error.TreesNotBuilt;
                const rows = w.rows();
                if (rows > self.window) return error.WindowTooWide;
                const l = try self.lane(w.stream);
                for (windows[0..i]) |v| if (v.stream == w.stream) return error.StreamTwiceInRound;
                for (w.positions, 0..) |p, r| if (p != self.e.pos() + 1 + r) return error.PositionMismatch;
                ids[i][0] = w.pending;
                for (ids[i][1..][0..w.held], 0..) |*d, j| d.* = if (j < l.held_n) l.held[j] else if (l.held_n > 0) l.held[l.held_n - 1] else w.pending;
                @memcpy(ids[i][1 + w.held ..][0..w.tokens.len], w.tokens);
                wins[i] = .{ .seq = l.seq, .tokens = ids[i][0..rows] };
                ls[i] = l;
            }
        }

        /// Each window keeps its path's rows (a window its draft already committed is left as it is).
        pub fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            const self = of(ptr);
            if (windows.len != paths.len) return error.PathsWindowsMismatch;
            for (windows, paths) |w, path| {
                if (path.len == 0) return error.EmptyPath;
                for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
                const l = self.lanes.get(w.stream) orelse return error.UnknownStream;
                const rows = l.pending_rows orelse continue;
                if (path.len > rows) return error.PathTooLong;
                l.pending_rows = null;
                // a commit runs no collective, but a failed one leaves this rank's caches apart from the other's
                self.e.commitWindow(l.index, path.len) catch |err| return self.lost("a window's commit", err);
            }
        }

        /// Heads absorb each request's kept rows, then chain up to `depth` drafts in shared steps; `tree` gets counts.
        pub fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
            const self = of(ptr);
            self.fillLate();
            self.prepareDraft(requests) catch |err| {
                try self.agree(err);
                unreachable;
            };
            try self.agree(null);
            const e = self.e;
            var reqs: [max_streams]E.DraftReq = undefined;
            var firsts: [max_streams][1]u32 = undefined;
            var ls: [max_streams]*Lane = undefined;
            const conf = self.confidenceNow();
            for (requests, 0..) |r, i| {
                const l = self.lanes.get(r.stream).?;
                ls[i] = l;
                // a shared round drafts before it keeps: the kept rows are this request's, commit them first
                if (r.rows) |rows| if (l.pending_rows) |_| {
                    l.pending_rows = null;
                    e.commitWindow(l.index, rows.len) catch |err| return self.lost("a window's commit", err);
                };
                l.held_n = 0;
                const depth = @min(r.depth, self.window - 1);
                if (r.rows == null) firsts[i] = .{self.value(r.first.?)};
                reqs[i] = .{
                    .seq = l.seq,
                    .source = if (r.rows != null) .{ .window = l.index } else .prefill_last,
                    .next = if (r.rows != null) r.follow else &firsts[i],
                    .count = depth,
                    .sampling = r.stream.sampling,
                    .confidence = conf,
                    .out = &l.held,
                };
            }
            e.draftMany(reqs[0..requests.len]) catch |err| return self.lost("the head's drafts", err);
            for (reqs[0..requests.len], ls[0..requests.len]) |q, l| l.held_n = q.drafted;
        }

        /// Draft stop rule now: running product while few streams draft, per-draft rule above (same on both ranks).
        pub fn confidenceNow(self: *const Self) f64 {
            if (self.confidence >= 0) return self.confidence;
            var live: usize = 0;
            var it = self.lanes.valueIterator();
            while (it.next()) |l| live += @intFromBool(l.*.fill == null);
            return if (live <= self.product_streams) self.confidence else self.wide_confidence;
        }

        /// The drafts' checks: each request's window is the last round's, its draft lands after its kept rows.
        pub fn prepareDraft(self: *Self, requests: []const be.DraftRequest) !void {
            if (!self.drafting) return error.NoDraftHead;
            if (requests.len > self.streams) return error.TooManyStreams;
            for (requests) |r| {
                if (r.lanes != null or r.early) return error.TreesNotBuilt;
                const l = try self.lane(r.stream);
                var after = self.e.pos();
                if (r.rows) |rows| {
                    for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                    if (r.follow.len != rows.len or rows.len == 0) return error.FollowMismatch;
                    if (l.round != self.round) return error.NoWindowThisRound;
                    if (l.pending_rows) |pending| {
                        if (rows.len > pending) return error.PathTooLong;
                        after += rows.len;
                    }
                } else if (r.first == null) return error.NoFirstToken;
                if (r.position != after + 1) return error.PositionMismatch;
            }
        }

        /// The drafts held for the stream's next window: the chain's length (it may stop short of the depth asked).
        pub fn heldFn(ptr: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
            _ = gpa;
            const self = of(ptr);
            const l = self.lanes.get(s) orelse return null;
            return .{ .count = @intCast(l.held_n) };
        }

        pub fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
            const self = of(ptr);
            const kv = self.lanes.fetchRemove(s) orelse return;
            // a late Kept of this lane takes its sequence as keeper: release runs no collective, so none is made here
            var kept_seq = false;
            var i: usize = 0;
            while (i < self.late.items.len) {
                const k = self.late.items[i];
                if (k.src != kv.value.seq) {
                    i += 1;
                    continue;
                }
                _ = self.late.orderedRemove(i);
                k.src = null;
                if (kept_seq) { // a second late Kept of one lane (one pass keeps at most one near its end)
                    k.broken = true;
                    continue;
                }
                if (@hasField(E, "f")) {
                    self.e.f.setPos(&kv.value.seq.st, k.at) catch {
                        k.broken = true;
                        continue;
                    };
                    self.e.f.setMtpLen(&kv.value.seq.st, k.snap.mtp_len) catch {
                        k.broken = true;
                        continue;
                    };
                }
                k.keeper = kv.value.seq;
                kept_seq = true;
            }
            if (kv.value.fill != null) {
                self.stopFill(kv.value); // a cancelled fill (the host discards it between steps, on both ranks)
                self.dropTentative();
            }
            self.reserved -= kv.value.reserved;
            self.forget(kv.value);
            if (!kept_seq) self.e.freeSeq(kv.value.seq);
            self.gpa.destroy(kv.value);
        }
    };
}
