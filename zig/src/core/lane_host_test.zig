//! Host memory reporting and isolated request refusal through LaneHost.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Memory = api.Memory;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Engine = api.Engine;

test "a lane host reports its backend's memory counts, and none without them" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try std.testing.expect(host.engine().memory(false) == null);
    const Counts = struct {
        resets: u32 = 0,
        fn read(ctx: ?*anyopaque, reset_peak: bool) ?Memory {
            const c: *@This() = @ptrCast(@alignCast(ctx.?));
            if (reset_peak) c.resets += 1;
            return .{ .active = 5, .peak = if (reset_peak) 5 else 9 };
        }
    };
    var counts: Counts = .{};
    host.memory = .{ .ctx = &counts, .read = Counts.read };
    try std.testing.expectEqual(@as(u64, 9), host.engine().memory(false).?.peak);
    try std.testing.expectEqual(@as(u64, 5), host.engine().memory(true).?.peak);
    try std.testing.expectEqual(@as(u32, 1), counts.resets);
}

test "a request the backend refuses fails alone, in the backend's words" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .refuse_sampled = true };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Words = struct {
        fn text(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return if (err == error.SamplingRefused) "send temperature 0" else null;
        }
    };
    host.explain = .{ .text = Words.text };
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        message: []const u8 = "",
        tokens: usize = 0,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| {
                    b.done = f.reason;
                    b.message = f.message;
                },
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var plain: Box = .{};
    var sampled: Box = .{};
    const greedy: Request = .{ .prompt = &prompt, .max_tokens = 64 };
    const keyed: Request = .{ .prompt = &prompt, .max_tokens = 64, .sampling = .{ .seed = 3, .temperature = 0.7, .top_k = 5 } };
    const e = host.engine();
    try e.submit(1, &greedy, .{ .ctx = &plain, .event = Box.event });
    try e.submit(2, &keyed, .{ .ctx = &sampled, .event = Box.event });
    try std.testing.expectEqual(Reason.failed, sampled.wait());
    try std.testing.expectEqualStrings("send temperature 0", sampled.message);
    try std.testing.expectEqual(Reason.length, plain.wait());
    try std.testing.expectEqual(@as(usize, 64), plain.tokens);
}

test "a lane host serves the core's own tokens, in order, and cancels between rounds" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var box: Box = .{};
    defer box.tokens.deinit(gpa);
    const request: Request = .{ .prompt = &prompt, .max_tokens = 24 };
    const e = host.engine();
    try e.submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(Reason.length, box.wait());
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &prompt);
    for (box.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 24), box.tokens.items.len);
    var gone: Box = .{};
    defer gone.tokens.deinit(gpa);
    const long: Request = .{ .prompt = &prompt, .max_tokens = 100000 };
    try e.submit(2, &long, .{ .ctx = &gone, .event = Box.event });
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, gone.wait());

    const CancelPrefill = struct {
        engine: Engine,
        id: Id,
        at: usize,

        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk == c.at) c.engine.cancel(c.id);
        }
    };
    const chunked_prompt = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    var chunked: Box = .{};
    defer chunked.tokens.deinit(gpa);
    var prefill_cancel = CancelPrefill{ .engine = e, .id = 3, .at = 2 };
    target.prefill_chunks = 10;
    target.prefill_count = 0;
    target.prefill_hook = CancelPrefill.call;
    target.prefill_hook_ctx = &prefill_cancel;
    const chunked_request: Request = .{ .prompt = &chunked_prompt, .max_tokens = 1 };
    try e.submit(3, &chunked_request, .{ .ctx = &chunked, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, chunked.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count()); // its lane released

    // the lone driver's prompt pass (gpu_round.run starts with Backend.opening), cancelled the same way
    const Lone = struct {
        be: lanes.backend.Backend,

        fn run(ctx: *anyopaque, s: *lanes.Stream, _: api.LoneHooks) anyerror!bool {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            _ = try l.be.opening(gpa, s);
            return error.NotCancelled;
        }
    };
    var lone: Box = .{};
    defer lone.tokens.deinit(gpa);
    var lone_driver = Lone{ .be = target.backend() };
    host.lone = .{ .ctx = &lone_driver, .run = Lone.run };
    prefill_cancel.id = 4;
    target.prefill_count = 0;
    try e.submit(4, &chunked_request, .{ .ctx = &lone, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, lone.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

/// A learner that takes `total` steps over a lesson, then reports it learned in that many.
pub const FakeLearner = struct {
    sink: ?api.LearnSink = null,
    steps: u32 = 0,
    total: u32 = 3,
    examples: usize = 0,

    fn emit(l: *FakeLearner, event: api.LearnEvent) void {
        l.sink.?.event(l.sink.?.ctx, &event);
    }
    pub fn begin(ctx: *anyopaque, request: *const api.LearnRequest, sink: api.LearnSink) anyerror!void {
        const l: *FakeLearner = @ptrCast(@alignCast(ctx));
        l.* = .{ .sink = sink, .total = l.total, .examples = request.train.len };
    }
    pub fn step(ctx: *anyopaque) api.Learner.Step {
        const l: *FakeLearner = @ptrCast(@alignCast(ctx));
        l.steps += 1;
        if (l.steps < l.total) return .{ .done = false, .changed = false };
        l.emit(.{ .learned = .{ .recalled = true, .steps = l.steps, .loss = 0.5 } });
        l.emit(.{ .done = .{} });
        return .{ .done = true, .changed = true };
    }
    pub fn abort(ctx: *anyopaque) void {
        const l: *FakeLearner = @ptrCast(@alignCast(ctx));
        l.emit(.{ .done = .{ .message = "aborted" } });
    }
};

/// Learn events as tags, read on the test's thread once `done` arrives.
pub const LearnBox = struct {
    mutex: std.Io.Mutex = .init,
    tags: std.ArrayList(std.meta.Tag(api.LearnEvent)) = .empty,
    steps: u32 = 0,
    done: bool = false,

    pub fn sink(b: *LearnBox) api.LearnSink {
        return .{ .ctx = b, .event = event };
    }
    fn event(ctx: *anyopaque, e: *const api.LearnEvent) void {
        const b: *LearnBox = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        b.tags.append(std.testing.allocator, e.*) catch {};
        if (e.* == .learned) b.steps = e.learned.steps;
        if (e.* == .done) b.done = true;
    }
    pub fn wait(b: *LearnBox) void {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

test "a learn request steps while the engine idles, its events in order; a host without a learner refuses" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const example: api.Example = .{ .ids = &.{ 1, 2, 3 }, .start = 2 };
    const learn: api.LearnRequest = .{ .train = &.{example} };
    var refused: LearnBox = .{};
    defer refused.tags.deinit(gpa);
    var bare = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try std.testing.expectError(error.Unsupported, bare.engine().learn(&learn, refused.sink()));

    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    var fake: FakeLearner = .{};
    host.learner = .{ .ctx = &fake, .begin = FakeLearner.begin, .step = FakeLearner.step, .abort = FakeLearner.abort };
    try host.start();
    defer host.stop();
    var box: LearnBox = .{};
    defer box.tags.deinit(gpa);
    try host.engine().learn(&learn, box.sink());
    box.wait();
    try std.testing.expectEqualSlices(std.meta.Tag(api.LearnEvent), &.{ .learned, .done }, box.tags.items);
    try std.testing.expectEqual(@as(u32, 3), box.steps);
    try std.testing.expectEqual(@as(usize, 1), fake.examples);

    const Replies = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            r.mutex.lockUncancelable(std.testing.io);
            defer r.mutex.unlock(std.testing.io);
            if (e.* == .finished) r.done = e.finished.reason;
        }
    };
    var replies: Replies = .{};
    const prompt = [_]u32{ 3, 1, 4, 1, 5 };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 8 };
    var again: LearnBox = .{};
    defer again.tags.deinit(gpa);
    try host.engine().learn(&learn, again.sink());
    try host.engine().submit(9, &request, .{ .ctx = &replies, .event = Replies.event });
    again.wait();
    while (true) {
        replies.mutex.lockUncancelable(std.testing.io);
        const d = replies.done;
        replies.mutex.unlock(std.testing.io);
        if (d) |r| break try std.testing.expectEqual(Reason.length, r);
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
}

test "a failed piece fails its request alone: its final of the same round does not run, the host keeps serving" {
    // a job dropped for a failed piece must leave the round's finals: `drop` frees it, runFinals must not read it
    const gpa = std.testing.allocator;
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true, .join_tail = true }, 16, 15);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .fail_pieces = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.joinBackend(), clock.clock());
    defer core.deinit();
    // jobs on the page allocator: a freed job is unmapped, so reading one faults (the testing allocator hides it)
    const pa = std.heap.page_allocator;
    var host = LaneHost.init(pa, std.testing.io, &core, .{ .lanes = 4 });
    // every admitted prompt's rows in one piece and its final in the same round (as for short prompts)
    const Plan = struct {
        const Seq = struct { key: usize, n: u64, done: bool = false };
        seqs: std.ArrayList(Seq) = .empty,
        fn self(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn beginAdmit(_: *anyopaque) void {}
        fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            self(ctx).seqs.append(gpa, .{ .key = ask.key, .n = ask.prompt.len }) catch return .refuse;
            return .admit;
        }
        fn admitted(_: *anyopaque) anyerror!void {}
        fn arm(_: *anyopaque, _: usize) void {}
        fn began(_: *anyopaque, _: usize, _: bool) void {}
        fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
            pieces.clearRetainingCapacity();
            finals.clearRetainingCapacity();
            for (self(ctx).seqs.items) |*x| if (!x.done) {
                // the host's lists: its allocator (it frees them)
                try pieces.append(pa, .{ .key = x.key, .start = 0, .end = x.n - 1, .save = false });
                try finals.append(pa, x.key);
                x.done = true;
            };
        }
        fn left(ctx: *anyopaque, key: usize) void {
            const p = self(ctx);
            for (p.seqs.items, 0..) |x, i| if (x.key == key) {
                _ = p.seqs.orderedRemove(i);
                return;
            };
        }
        fn after(_: *anyopaque, _: f64, _: f64) void {}
    };
    var pl: Plan = .{};
    defer pl.seqs.deinit(gpa);
    host.rounds = .{ .ctx = &pl, .vtable = &.{ .begin_admit = Plan.beginAdmit, .admit = Plan.admit, .admitted = Plan.admitted, .arm = Plan.arm, .began = Plan.began, .plan = Plan.plan, .left = Plan.left, .after = Plan.after } };
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: usize = 0,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const p2 = [_]u32{ 2, 7, 1, 8, 2, 8 };
    var b1: Box = .{};
    var b2: Box = .{};
    var b3: Box = .{};
    const r1: Request = .{ .prompt = &p1, .max_tokens = 12 };
    const r2: Request = .{ .prompt = &p2, .max_tokens = 12 };
    const e = host.engine();
    // both admitted before the host's first round: the first piece fails, the second's request is served
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try host.start();
    defer host.stop();
    try std.testing.expectEqual(Reason.failed, b1.wait());
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(usize, 12), b2.tokens);
    // the host serves the next request
    try e.submit(3, &r1, .{ .ctx = &b3, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b3.wait());
    try std.testing.expectEqual(@as(usize, 12), b3.tokens);
}

test "halt (a stop's drain deadline): the rounds end at a round boundary, every request queued or running fails with the reason" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.pieceBackend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    // one prompt prefilling at a time (the next waits queued), 3 rows a round
    const Plan = struct {
        const Seq = struct { key: usize, n: u64, done: u64 = 0, decoding: bool = false };
        seqs: std.ArrayList(Seq) = .empty,
        rounds: u32 = 0,
        fn self(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn beginAdmit(_: *anyopaque) void {}
        fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            const p = self(ctx);
            for (p.seqs.items) |x| if (!x.decoding) return .wait;
            p.seqs.append(gpa, .{ .key = ask.key, .n = ask.prompt.len }) catch return .refuse;
            return .admit;
        }
        fn admitted(_: *anyopaque) anyerror!void {}
        fn arm(_: *anyopaque, _: usize) void {}
        fn began(_: *anyopaque, _: usize, _: bool) void {}
        fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
            const p = self(ctx);
            pieces.clearRetainingCapacity();
            finals.clearRetainingCapacity();
            for (p.seqs.items) |*x| if (!x.decoding) {
                if (x.done < x.n - 1) {
                    const end = @min(x.done + 3, x.n - 1);
                    try pieces.append(gpa, .{ .key = x.key, .start = x.done, .end = end, .save = false });
                    x.done = end;
                }
                if (x.done == x.n - 1) {
                    try finals.append(gpa, x.key);
                    x.decoding = true;
                }
            };
            p.rounds += 1;
        }
        fn left(ctx: *anyopaque, key: usize) void {
            const p = self(ctx);
            for (p.seqs.items, 0..) |x, i| if (x.key == key) {
                _ = p.seqs.orderedRemove(i);
                return;
            };
        }
        fn after(_: *anyopaque, _: f64, _: f64) void {}
    };
    var pl: Plan = .{};
    defer pl.seqs.deinit(gpa);
    host.rounds = .{ .ctx = &pl, .vtable = &.{ .begin_admit = Plan.beginAdmit, .admit = Plan.admit, .admitted = Plan.admitted, .arm = Plan.arm, .began = Plan.began, .plan = Plan.plan, .left = Plan.left, .after = Plan.after } };
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        message: []const u8 = "",
        /// the planner's rounds when the finish arrived (on the engine thread, between rounds)
        rounds_at_end: u32 = 0,
        plan: *Plan,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| {
                    b.done = f.reason;
                    b.message = f.message;
                    b.rounds_at_end = b.plan.rounds;
                },
                else => {},
            }
        }
        fn count(b: *@This()) usize {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return b.tokens.items.len;
        }
    };
    var long_prompt: [600]u32 = undefined;
    for (&long_prompt, 0..) |*t, i| t.* = @intCast(1 + i % 13);
    const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var boxes: [3]Box = .{ .{ .plan = &pl }, .{ .plan = &pl }, .{ .plan = &pl } };
    defer for (&boxes) |*b| b.tokens.deinit(gpa);
    // 1: decoding (no end of its own), 2: prefilling a long prompt in pieces, 3: queued behind 2
    const r1: Request = .{ .prompt = &p1, .max_tokens = 1 << 30 };
    const r2: Request = .{ .prompt = &long_prompt, .max_tokens = 1 << 30 };
    const r3: Request = .{ .prompt = &p1, .max_tokens = 8 };
    const e = host.engine();
    try host.start();
    try e.submit(1, &r1, .{ .ctx = &boxes[0], .event = Box.event });
    while (boxes[0].count() < 4) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    try e.submit(2, &r2, .{ .ctx = &boxes[1], .event = Box.event });
    try e.submit(3, &r3, .{ .ctx = &boxes[2], .event = Box.event });
    while (boxes[0].count() < 20) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    host.halt("server restarting: retry shortly");
    const rounds = pl.rounds;
    for (&boxes) |*b| {
        try std.testing.expectEqual(@as(?Reason, .failed), b.done);
        try std.testing.expectEqualStrings("server restarting: retry shortly", b.message);
        // no round was planned after the finishes: the loop stopped at the boundary they were sent at
        try std.testing.expectEqual(rounds, b.rounds_at_end);
    }
    // the decoding stream's tokens are whole rounds' (the fake's own sequence), the prefilling one had none yet
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &p1);
    for (boxes[0].tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 0), boxes[1].tokens.items.len);
    try std.testing.expectEqual(@as(usize, 0), boxes[2].tokens.items.len);
    try std.testing.expectEqual(@as(usize, 0), pl.seqs.items.len);
    // nothing runs after it; the stop that follows (a close) finds the thread gone
    std.Io.sleep(std.testing.io, .fromMilliseconds(20), .awake) catch {};
    try std.testing.expectEqual(rounds, pl.rounds);
    host.stop();
}
