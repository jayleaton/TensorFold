//! Flash Next lanes' prompt fills: sliced fills between rounds, prefills, bursts and the marked prompt pass.
const std = @import("std");
const lanes = @import("lanes");
const envs = @import("cuda_env.zig");
const media_mod = @import("cuda_media.zig");
const cl = @import("cuda_lanes.zig");
const lanes_core = @import("cuda_lanes_core.zig");

const max_rows = cl.max_rows;
const late_keep_rows = cl.late_keep_rows;
const FillState = cl.FillState;

pub fn Fill(comptime E: type) type {
    return struct {
        const Self = lanes_core.Lanes(E);
        const Kept = Self.Kept;
        const Lane = Self.Lane;
        const Filling = Self.Filling;
        const SnapPtr = Self.SnapPtr;
        const Admission = Self.Admission;
        const of = Self.of;

        // -- prompts filling between rounds (the lanes core's prefill_begin / prefill_step) --------------------

        /// f212's prefill_step: the first call begins the fill, each steps it `fill_layers` layers; true once in.
        pub fn fillStepFn(ptr: *anyopaque, s: *lanes.Stream, cap: u32) anyerror!bool {
            _ = cap;
            const self = of(ptr);
            if (self.lanes.get(s) == null) try prefillBeginFn(ptr, s);
            var state: [1]FillState = .{.filling};
            try prefillStepFn(ptr, &.{s}, &state);
            return state[0] == .done;
        }

        /// The prefill's admission and agreements, its resume made, the first segment's fill started.
        pub fn prefillBeginFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            const self = of(ptr);
            self.fillLate();
            const e = self.e;
            const a = self.admit(s, 0) catch |err| {
                try self.agree(err);
                unreachable;
            };
            try self.agree(null);
            const l = self.makeLane(s, a) catch |err| return self.lost("a prompt's sequence", err);
            const requested = s.reuse.at > 0;
            var from: ?*Kept = if (requested) self.keptOf(s.reuse.saved) else null;
            if (requested) {
                if (!(try self.agreeOn(from != null))) from = null;
            }
            var f: Filling = .{ .fl = undefined };
            if (from) |k| {
                e.copyPrefix(l.seq, k.keeper.?, k.snap.pos, k.snap.mtp_len) catch |err| {
                    self.dropLane(s, l);
                    return self.lost("a prompt's resume", err);
                };
                f.cur = k.snap;
                f.at = k.snap.pos;
            }
            s.cached = @intCast(f.at);
            s.reuse_failed = requested and from == null;
            self.dropTentative();
            self.startSegment(s, l, &f) catch |err| {
                self.dropLane(s, l);
                // a media attachment both ranks refused together (memory, bad rows): this request fails, not the server
                if (err == error.MediaRefused) return err;
                return self.lost("a prompt's fill", err);
            };
            l.fill = f;
        }

        /// The next segment's fill: up to the next mark (its state kept there), else to the prompt's end.
        pub fn startSegment(self: *Self, s: *lanes.Stream, l: *Lane, f: *Filling) !void {
            const ids = s.prompt();
            const next_mark: ?u32 = for (s.reuse.marks) |m| {
                if (m > f.at and m < ids.len) break m;
            } else null;
            if (next_mark) |m| {
                f.fl = try self.e.fillBegin(l.seq, ids[0..m], s.sampling, .{ .keep_at = m, .resume_from = f.cur });
                f.final = false;
            } else {
                f.fl = try self.e.fillBegin(l.seq, ids, s.sampling, .{ .resume_from = f.cur, .media = media_mod.of(s) });
                f.final = true;
            }
        }

        /// `fill_layers` more layers of every filling prompt; segments keep their marks, a done prompt draws `first`.
        pub fn prefillStepFn(ptr: *anyopaque, streams: []const *lanes.Stream, states: []FillState) anyerror!void {
            const self = of(ptr);
            self.fillLate();
            try self.agree(null); // nothing to prepare: the ranks only meet before the slices' collectives
            for (streams, states) |s, *st| {
                const l = self.lanes.get(s) orelse return error.UnknownStream;
                const f = &(l.fill orelse return error.NotFilling);
                const done = self.e.fillStep(f.fl, self.fill_layers) catch |err| return self.lost("a prompt's fill", err);
                if (!done) continue;
                if (!f.final) {
                    const m: u32 = @intCast(f.fl.prompt.len);
                    const snap = f.fl.kept orelse return self.lost("a prompt's fill", error.NoStateKept);
                    f.fl.kept = null; // the pass's now
                    self.e.fillFree(f.fl);
                    if (f.owned) self.e.freeSnapshot(f.cur.?);
                    f.cur = snap;
                    f.owned = true;
                    if (self.keep(l, snap, m)) |k| {
                        f.owned = false;
                        self.tentative.append(self.gpa, .{ .at = m, .kept = k }) catch |err| return self.lost("a prompt's fill", err);
                    } else |err| std.log.warn("prompt cache: no state kept at {d} tokens ({s})", .{ m, @errorName(err) });
                    if (s.reuse.hook) |h| h.at(h.ptr, s, m);
                    f.at = m;
                    self.startSegment(s, l, f) catch |err| return self.lost("a prompt's fill", err);
                    continue;
                }
                _ = self.take(f.fl.first);
                self.e.fillFree(f.fl);
                if (f.owned) self.e.freeSnapshot(f.cur.?);
                l.fill = null;
                self.endPass();
                st.* = .done;
            }
        }

        pub fn stopFill(self: *Self, l: *Lane) void {
            if (comptime !@hasDecl(E, "fillBegin")) return; // an engine without fills never starts one
            const f = l.fill orelse return;
            self.e.fillFree(f.fl);
            if (f.owned) self.e.freeSnapshot(f.cur.?);
            l.fill = null;
        }

        // -- the vtable -----------------------------------------------------------------------------------------

        /// A new sequence for the stream, its prompt, and the first draw at the prompt's length with its sampling.
        pub fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            const self = of(ptr);
            self.fillLate();
            const e = self.e;
            const a = self.admit(s, 0) catch |err| {
                try self.agree(err);
                unreachable;
            };
            try self.agree(null);
            const l = self.makeLane(s, a) catch |err| return self.lost("a prompt's sequence", err);
            // a resume both ranks can make (each holds the state), else both prefill from the start
            const requested = s.reuse.at > 0;
            var from: ?*Kept = if (requested) self.keptOf(s.reuse.saved) else null;
            if (requested) {
                if (!(try self.agreeOn(from != null))) from = null;
            }
            e.bind(l.seq);
            const first = self.pass(s, l, from) catch |err| {
                self.dropLane(s, l);
                if (err == error.MediaRefused) return err;
                return self.lost("a prompt's prefill", err);
            };
            _ = self.take(first);
        }

        /// A prompt burst: admit all first (a refusal: false, prefill singly), then share prefillMany passes.
        pub fn prefillManyFn(ptr: *anyopaque, ss: []const *lanes.Stream) anyerror!bool {
            const self = of(ptr);
            if (comptime !@hasDecl(E, "prefillMany")) return false;
            const e = self.e;
            if (ss.len < 2 or ss.len > max_rows) return false;
            // an image or video prompt prefills alone (attachMedia, the MODE 2 prompt rows)
            for (ss) |s| if (s.reuse.at > 0 or s.reuse.marks.len > 0 or media_mod.of(s) != null) return false;
            var ls: [max_rows]*Lane = undefined;
            var adm: [max_rows]Admission = undefined;
            var admitted: usize = 0;
            var ahead: usize = 0; // the burst's earlier streams' reservations, not made yet
            var failure: ?anyerror = null;
            for (ss) |s| {
                adm[admitted] = self.admit(s, ahead) catch |err| {
                    failure = err;
                    break;
                };
                ahead += adm[admitted].need;
                admitted += 1;
            }
            self.agree(failure) catch return false;
            var made: usize = 0;
            for (ss, adm[0..ss.len]) |s, a| {
                ls[made] = self.makeLane(s, a) catch |err| return self.burstLost(ss, ls[0..made], err);
                made += 1;
            }
            for (ss) |s| {
                s.cached = 0;
                s.reuse_failed = false;
            }
            self.dropTentative();
            var i: usize = 0;
            while (i < ss.len) {
                var j = i;
                var rows: usize = 0;
                while (j < ss.len and rows + ss[j].prompt().len <= e.prefill_rows) : (j += 1) rows += ss[j].prompt().len;
                if (j - i >= 2) {
                    var items: [max_rows]E.ManyPrompt = undefined;
                    for (items[0 .. j - i], ss[i..j], ls[i..j]) |*it, s, l| it.* = .{ .seq = l.seq, .prompt = s.prompt(), .sampling = s.sampling };
                    if (e.manyFit(items[0 .. j - i])) {
                        var firsts: [max_rows]u32 = undefined;
                        e.prefillMany(items[0 .. j - i], &firsts) catch |err| return self.burstLost(ss, ls[0..made], err);
                        for (ls[i..j], firsts[0 .. j - i]) |l, f| l.first = self.take(f);
                        i = j;
                        continue;
                    }
                }
                e.bind(ls[i].seq);
                const r = e.prefillWith(ss[i].prompt(), ss[i].sampling, .{}) catch |err| return self.burstLost(ss, ls[0..made], err);
                ls[i].first = self.take(r.first);
                i += 1;
            }
            self.endPass();
            return true;
        }

        pub fn burstLost(self: *Self, ss: []const *lanes.Stream, ls: []*Lane, err: anyerror) anyerror {
            for (ss[0..ls.len], ls) |s, l| self.dropLane(s, l);
            return self.lost("a burst's prefill", err);
        }

        /// The prompt pass from a kept state or 0, cut at each mark to keep a tentative state, then to the end.
        pub fn pass(self: *Self, s: *lanes.Stream, l: *Lane, from: ?*Kept) !u32 {
            const e = self.e;
            const ids = s.prompt();
            var cur: ?SnapPtr = null; // the state the next segment resumes
            var owned = false; // `cur` is this pass's own (no Kept holds it)
            defer if (owned) e.freeSnapshot(cur.?);
            var at: usize = 0;
            if (from) |k| {
                try e.copyPrefix(l.seq, k.keeper.?, k.snap.pos, k.snap.mtp_len);
                cur = k.snap;
                at = k.snap.pos;
            }
            s.cached = @intCast(at);
            s.reuse_failed = s.reuse.at > 0 and from == null;
            self.dropTentative();
            // TF_FLASHNEXT_PASS_TIMES=1: each part's wall time (the stream drained at each edge), in the log
            const timed = comptime @hasField(E, "io") and @hasField(E, "stream");
            const times = timed and envs.get("TF_FLASHNEXT_PASS_TIMES") != null;
            var t_last: std.Io.Timestamp = if (times) passNow(e) else undefined;
            var parts: [3]f64 = @splat(0); // to the marks, keeping, the rest
            for (s.reuse.marks) |m| {
                if (m <= at or m >= ids.len) continue;
                const r = try e.prefillWith(ids[0..m], s.sampling, .{ .keep_at = m, .resume_from = cur });
                if (times) parts[0] += passSince(e, &t_last);
                if (owned) e.freeSnapshot(cur.?);
                cur = r.kept.?;
                owned = true;
                // late only the pass's last mark: the source's rows and key ring stay until the next call
                const last = for (s.reuse.marks) |n| {
                    if (n > m and n < ids.len) break false;
                } else true;
                const late = self.late_keep and last and ids.len - m <= late_keep_rows;
                if (self.keepAt(l, cur.?, m, late)) |k| {
                    owned = false; // the Kept holds it now
                    try self.tentative.append(self.gpa, .{ .at = m, .kept = k });
                } else |err| std.log.warn("prompt cache: no state kept at {d} tokens ({s})", .{ m, @errorName(err) });
                if (s.reuse.hook) |h| h.at(h.ptr, s, m);
                at = m;
                if (times) parts[1] += passSince(e, &t_last);
            }
            const r = try e.prefillWith(ids, s.sampling, .{ .resume_from = cur, .media = media_mod.of(s) });
            self.endPass();
            if (times) {
                parts[2] = passSince(e, &t_last);
                std.log.info("pass times: {d} tokens from {d}, marks {any}: to the marks {d:.1} ms, keeping {d:.1} ms, the rest {d:.1} ms", .{ ids.len, s.cached, s.reuse.marks, parts[0], parts[1], parts[2] });
            }
            return r.first;
        }
    };
}

pub fn passNow(e: anytype) std.Io.Timestamp {
    e.stream.synchronize() catch {};
    return std.Io.Timestamp.now(e.io, .awake);
}

/// ms since `last` (the stream drained first), and `last` moved to now.
pub fn passSince(e: anytype, last: *std.Io.Timestamp) f64 {
    const now = passNow(e);
    const d = last.durationTo(now);
    last.* = now;
    return @as(f64, @floatFromInt(d.nanoseconds)) / 1e6;
}
