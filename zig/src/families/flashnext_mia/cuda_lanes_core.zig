//! Flash Next lanes' core: the generic Lanes type, its fields and vtable, and its shared round helpers.
const std = @import("std");
const envs = @import("cuda_env.zig");
const lanes = @import("lanes");
const cl = @import("cuda_lanes.zig");
const fill_part = @import("cuda_lanes_fill.zig");
const keep_part = @import("cuda_lanes_keep.zig");
const rounds_part = @import("cuda_lanes_rounds.zig");
const costs_part = @import("cuda_lanes_costs.zig");

const be = lanes.backend;
const max_rows = cl.max_rows;
const default_confidence = cl.default_confidence;
const draft_prior = cl.draft_prior;
const ring = cl.ring;
const shared_points = cl.shared_points;
const Gate = cl.Gate;
const Notes = cl.Notes;
const max_drafts = cl.max_drafts;

pub fn Lanes(comptime E: type) type {
    return struct {
        const Self = @This();
        /// the engine's sequence handle, as newSeq returns it
        const SeqPtr = @typeInfo(@typeInfo(@TypeOf(E.newSeq)).@"fn".return_type.?).error_union.payload;

        /// prefillWith's result and its kept state (cuda_engine's Prefilled / Snapshot)
        const Prefilled = @typeInfo(@typeInfo(@TypeOf(E.prefillWith)).@"fn".return_type.?).error_union.payload;
        pub const SnapPtr = @typeInfo(@FieldType(Prefilled, "kept")).optional.child;
        /// the engine's prompt fill (fillBegin / fillStep / fillFree), when it has one
        const FillPtr = if (@hasDecl(E, "fillBegin")) @typeInfo(@typeInfo(@TypeOf(E.fillBegin)).@"fn".return_type.?).error_union.payload else *anyopaque;

        /// A kept prompt state: engine snapshot at `at` plus a keeper sequence; `key` is rank 0's address of it.
        pub const Kept = struct {
            snap: SnapPtr,
            keeper: ?SeqPtr,
            at: u32,
            bytes: u64,
            key: u64 = 0,
            src: ?SeqPtr = null,
            broken: bool = false,
        };

        const Tentative = struct { at: u32, kept: *Kept, adopted: bool = false };

        pub const Lane = struct {
            seq: SeqPtr,
            /// the last verify's uncommitted rows of this stream (kept only on drops; a shared round drafts first)
            pending_rows: ?usize = null,
            /// this stream's window in the round `round` (the engine's sampleWindow / commitWindow index)
            index: usize = 0,
            round: u64 = 0,
            held: [max_drafts]u32 = undefined,
            held_n: usize = 0,
            /// the caches of this stream's whole window, reserved at admission (the guaranteed class)
            reserved: usize = 0,
            /// the prompt filling between rounds (sliced fills), null once it is in
            fill: ?Filling = null,
            /// the first token's handle of a burst's prompt (firstFn gives it once; else the last one taken)
            first: ?u64 = null,
        };

        /// A prompt filling in slices: the engine fill of the current segment, its resume state and the pass position.
        pub const Filling = struct {
            fl: FillPtr,
            cur: ?SnapPtr = null,
            owned: bool = false,
            at: usize = 0,
            final: bool = false,
        };

        gpa: std.mem.Allocator,
        e: *E,
        drafting: bool,
        context: usize,
        /// a window's rows at most: the engine's depth + 1 (its buffers' rows and the kernel set's windows)
        window: usize,
        /// streams a shared forward holds (--parallel; the engine's Options.streams)
        streams: usize = 1,
        confidence: f64 = default_confidence,
        /// running product (confidence < 0) while at most this many streams draft, per-draft `wide_confidence` above
        product_streams: usize = std.math.maxInt(usize),
        wide_confidence: f64 = 0.5,
        gate: ?Gate = null,
        /// admit only if the window's caches fit the budget; `overcommit` (TF_FLASHNEXT_OVERCOMMIT=1) uses free now
        overcommit: bool = false,
        reserved: usize = 0,
        /// kept prompt states: every live one (the store's), the last pass's not yet adopted ones, their bytes
        kept: std.ArrayList(*Kept) = .empty,
        tentative: std.ArrayList(Tentative) = .empty,
        kept_bytes: u64 = 0,
        /// the prompt cache's budget on this rank (cuda_native.cacheBudget; snap_ready hands f212's store this much)
        cache_budget: u64 = 0,
        /// keep a pass's last state late, after the first token (TF_FLASHNEXT_KEEP_LATE=0 off; the copy is the same)
        late_keep: bool = true,
        late: std.ArrayList(*Kept) = .empty,
        /// rank 1 at two ranks: the last pass's kept states wait for rank 0's note of which the store took
        follower: bool = false,
        /// rank 0 at two ranks: where the store's decisions go to rank 1 (cuda_mirror's Leader.note)
        notes: ?Notes = null,
        /// prompts fill in slices between rounds (prefill_begin / prefill_step), `fill_layers` layers a step
        sliced: bool = false,
        fill_layers: usize = 8,
        lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, *Lane) = .empty,
        /// the last verify's lanes whose windows wait for their commits: committed before the next forward
        opened: std.ArrayList(*Lane) = .empty,
        round: u64 = 0,
        drawn: [ring]u32 = undefined,
        next: u64 = 0,
        costs: [max_rows]lanes.config.Cost = undefined,
        cost_count: usize = 0,
        shared: [shared_points]lanes.config.Cost = undefined,
        shared_count: usize = 0,
        mtp_ms: f64 = 0,

        pub const fillStepFn = fill_part.Fill(E).fillStepFn;
        pub const prefillBeginFn = fill_part.Fill(E).prefillBeginFn;
        pub const startSegment = fill_part.Fill(E).startSegment;
        pub const prefillStepFn = fill_part.Fill(E).prefillStepFn;
        pub const stopFill = fill_part.Fill(E).stopFill;
        pub const prefillFn = fill_part.Fill(E).prefillFn;
        pub const prefillManyFn = fill_part.Fill(E).prefillManyFn;
        pub const burstLost = fill_part.Fill(E).burstLost;
        pub const pass = fill_part.Fill(E).pass;
        pub const keep = keep_part.Keep(E).keep;
        pub const keepAt = keep_part.Keep(E).keepAt;
        pub const fillLate = keep_part.Keep(E).fillLate;
        pub const freeKept = keep_part.Keep(E).freeKept;
        pub const endPass = keep_part.Keep(E).endPass;
        pub const dropTentative = keep_part.Keep(E).dropTentative;
        pub const keptOf = keep_part.Keep(E).keptOf;
        pub const agreeOn = keep_part.Keep(E).agreeOn;
        pub const cacheHooks = keep_part.Keep(E).cacheHooks;
        pub const bytesFn = keep_part.Keep(E).bytesFn;
        pub const saveFn = keep_part.Keep(E).saveFn;
        pub const restoreFn = keep_part.Keep(E).restoreFn;
        pub const dropFn = keep_part.Keep(E).dropFn;
        pub const forgetKept = keep_part.Keep(E).forgetKept;
        pub const applyNote = keep_part.Keep(E).applyNote;
        pub const savedFor = keep_part.Keep(E).savedFor;
        pub const Admission = rounds_part.Rounds(E).Admission;
        pub const admit = rounds_part.Rounds(E).admit;
        pub const makeLane = rounds_part.Rounds(E).makeLane;
        pub const footprint = rounds_part.Rounds(E).footprint;
        pub const budget = rounds_part.Rounds(E).budget;
        pub const dropLane = rounds_part.Rounds(E).dropLane;
        pub const forget = rounds_part.Rounds(E).forget;
        pub const firstFn = rounds_part.Rounds(E).firstFn;
        pub const queueFn = rounds_part.Rounds(E).queueFn;
        pub const readFn = rounds_part.Rounds(E).readFn;
        pub const verifyFn = rounds_part.Rounds(E).verifyFn;
        pub const prepareVerify = rounds_part.Rounds(E).prepareVerify;
        pub const keepFn = rounds_part.Rounds(E).keepFn;
        pub const draftFn = rounds_part.Rounds(E).draftFn;
        pub const confidenceNow = rounds_part.Rounds(E).confidenceNow;
        pub const prepareDraft = rounds_part.Rounds(E).prepareDraft;
        pub const heldFn = rounds_part.Rounds(E).heldFn;
        pub const releaseFn = rounds_part.Rounds(E).releaseFn;
        pub const measure = costs_part.Measure(E).measure;

        pub fn init(gpa: std.mem.Allocator, e: *E, drafting: bool, context: usize, window: usize) Self {
            const late = if (envs.get("TF_FLASHNEXT_KEEP_LATE")) |v| !std.mem.eql(u8, v, "0") else true;
            return .{ .gpa = gpa, .e = e, .drafting = drafting, .context = context, .window = @max(1, @min(window, max_rows)), .late_keep = late };
        }

        pub fn deinit(self: *Self) void {
            var it = self.lanes.valueIterator();
            while (it.next()) |l| {
                self.stopFill(l.*);
                self.e.freeSeq(l.*.seq);
                self.gpa.destroy(l.*);
            }
            self.lanes.deinit(self.gpa);
            self.opened.deinit(self.gpa);
            self.dropTentative();
            self.tentative.deinit(self.gpa);
            for (self.kept.items) |k| self.freeKept(k);
            self.kept.deinit(self.gpa);
            self.late.deinit(self.gpa);
        }

        pub fn backend(self: *Self) be.Backend {
            if (@hasDecl(E, "fillBegin") and self.sliced) return .{ .ptr = self, .vtable = &.{
                .prefill = prefillFn,
                .first = firstFn,
                .queue = queueFn,
                .read = readFn,
                .verify = verifyFn,
                .keep = keepFn,
                .draft = draftFn,
                .tree = heldFn,
                .release = releaseFn,
                .prefill_step = fillStepFn,
            } };
            return .{ .ptr = self, .vtable = &.{
                .prefill = prefillFn,
                .first = firstFn,
                .queue = queueFn,
                .read = readFn,
                .verify = verifyFn,
                .keep = keepFn,
                .draft = draftFn,
                .tree = heldFn,
                .release = releaseFn,
                .prefill_many = if (@hasDecl(E, "prefillMany")) prefillManyFn else null,
            } };
        }

        /// Round loop setup facts: windows up to `window` rows (16 max), `streams` per forward, this GPU's costs.
        pub fn facts(self: *const Self) lanes.Model {
            const many = self.streams > 1;
            return .{
                .exact_width = if (self.drafting) @intCast(self.window) else 1,
                .gpu_tokens = false,
                .mtp = self.drafting,
                .speculate = self.drafting,
                .speculate_early = false,
                .draft_prior = &draft_prior,
                .drafts = @intCast(self.window - 1),
                .window_costs = self.costs[0..self.cost_count],
                .mtp_step_ms = self.mtp_ms,
                .streams_exact = true,
                .hidden_rows = many,
                .batch_rows = @intCast(self.streams * self.window),
                .max_streams = @intCast(self.streams),
                .shared_costs = self.shared[0..self.shared_count],
                .draft_streams = many,
                // the running-product rule (confidence < 0) stops each chain itself: the round loop asks the most
                .draft_stops = self.drafting and self.confidence < 0,
                .draft_stops_most = @intCast(@min(self.product_streams, std.math.maxInt(u32))),
                // each sequence keeps its prompt's last streams until its first drafts: joiners draft together
                .join_batch = self.drafting,
            };
        }

        /// The lane, its sequence bound and its stream's sampling set.
        pub fn lane(self: *Self, s: *const lanes.Stream) !*Lane {
            const l = self.lanes.get(s) orelse return error.UnknownStream;
            self.e.bind(l.seq);
            if (@hasDecl(E, "setSampling")) self.e.setSampling(s.sampling);
            return l;
        }

        /// Commits every still-uncommitted window of the last verify whole before the next forward overwrites its rows.
        pub fn settle(self: *Self) !void {
            defer self.opened.clearRetainingCapacity();
            for (self.opened.items) |o| {
                const rows = o.pending_rows orelse continue;
                o.pending_rows = null;
                try self.e.commitWindow(o.index, rows);
            }
        }

        /// Every rank's prepare done: return this rank's error, or error.OtherRankFailed, before any collective.
        pub fn agree(self: *Self, failed: ?anyerror) !void {
            if (self.gate) |g| {
                const all = g.agree(g.ptr, failed == null) catch |e| {
                    g.fatal(g.ptr, "the ranks' agreement", e);
                    return e;
                };
                if (!all) return failed orelse error.OtherRankFailed;
            }
            if (failed) |e| return e;
        }

        /// An engine call failed after the agreement: with two ranks this one leaves (they may be apart now).
        pub fn lost(self: *Self, what: []const u8, err: anyerror) anyerror {
            if (self.gate) |g| g.fatal(g.ptr, what, err);
            return err;
        }

        pub fn take(self: *Self, token: u32) u64 {
            const h = self.next;
            self.drawn[h % ring] = token;
            self.next += 1;
            return h;
        }

        pub fn value(self: *const Self, feed: be.Feed) u32 {
            return switch (feed) {
                .handle => |h| self.drawn[h % ring],
                .value => |v| v,
            };
        }

        pub fn of(ptr: *anyopaque) *Self {
            return @ptrCast(@alignCast(ptr));
        }
    };
}
