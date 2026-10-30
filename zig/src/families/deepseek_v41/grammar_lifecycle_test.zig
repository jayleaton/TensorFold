//! Same-owner grammar lifetime through production staging, wire frames, follower hooks and mask application.
const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const lanes = @import("lanes");
const mock = @import("load_mock.zig");
const gpu = @import("grammar_gpu.zig");
const rows = @import("grammar_rows.zig");
const Forward = @import("forward.zig").Forward;
const Runner = @import("run.zig").Runner;
const SlotSet = @import("slots.zig").SlotSet;
const t = std.testing;
const a = t.allocator;
const vocab = 256;
const width = 128;
const words = vocab / 32;
var launches: usize = 0;

fn value(comptime T: type, args: [*]?*anyopaque, i: usize) T {
    return @as(*align(1) const T, @ptrCast(args[i].?)).*;
}

fn deviceWords(ptr: u64, n: usize) []align(1) const u32 {
    return std.mem.bytesAsSlice(u32, mock.memory[ptr / 256][ptr % 256 ..][0 .. n * 4]);
}

// Execute only the mask kernel's host equivalent, using the actual launch arguments and uploaded device bytes.
fn launch(_: cuda.abi.Function, _: c_uint, count: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
    const args = params.?;
    const dst = value(u64, args, 0);
    const stride = value(u64, args, 1);
    const indices = deviceWords(value(u64, args, 2), count);
    const bits = deviceWords(value(u64, args, 3), count * value(u32, args, 4));
    const lo = value(u32, args, 5) * 32;
    var logits: [2 * width]f32 = undefined;
    @memcpy(std.mem.sliceAsBytes(&logits), mock.memory[dst / 256][0 .. 2 * width * 4]);
    for (indices, 0..) |row, r| {
        for (0..value(u32, args, 6)) |k| {
            rows.maskWord(logits[@as(usize, row) * stride ..].ptr, bits[r * words + lo / 32 + k], @intCast(k * 32), value(u32, args, 7));
        }
    }
    @memcpy(mock.memory[dst / 256][0 .. 2 * width * 4], std.mem.sliceAsBytes(&logits));
    launches += 1;
    return 0;
}

fn loadPtx(out: *cuda.abi.Module, _: ?*const anyopaque, _: c_uint, _: ?[*]c_int, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
    mock.modules += 1;
    out.* = @ptrFromInt(mock.modules);
    return 0;
}

const Wire = struct {
    lead: tp.planlink.PlanLink,
    peer: tp.planlink.PlanLink,
    frames: std.ArrayList([]i64) = .empty,

    fn init() !Wire {
        const listener = try tp.bootstrap.testListener();
        defer tp.sock.close(listener.fd);
        const deadline = tp.sock.nowNs() + 5 * std.time.ns_per_s;
        const client = try tp.sock.connect("127.0.0.1", listener.port, deadline);
        errdefer tp.sock.close(client);
        const server = try tp.sock.accept(listener.fd, deadline);
        var root: [tp.planlink.max_ranks]tp.sock.Fd = @splat(-1);
        var peer = root;
        root[1] = server;
        peer[0] = client;
        return .{ .lead = .init(0, 2, root), .peer = .init(1, 2, peer) };
    }

    fn replay(w: *Wire, f: *Forward) !void {
        while (w.peer.received < w.lead.sent) {
            var buf: [64]i64 = undefined;
            const frame = try w.peer.recv(&buf);
            try w.frames.append(a, try a.dupe(i64, frame));
            try t.expectEqual(Forward.op_grammar, frame[0]);
            const hook = f.grammar.?;
            try hook.follow(hook.ptr, frame);
        }
    }

    fn deinit(w: *Wire) void {
        w.lead.close();
        w.peer.close();
        for (w.frames.items) |frame| a.free(frame);
        w.frames.deinit(a);
    }
};

const Rank = struct {
    runner: Runner,
    f: Forward,
    masks: *gpu.Masks,
    logits: cuda.DeviceBuffer,
    slots: *SlotSet,

    fn init(r: *Rank, d: *cuda.Driver, cfg: *@import("config.zig").Config, widths: *@import("block.zig").Widths, comm: tp.Collective) !void {
        r.runner = .{ .gpa = a, .d = d, .stream = .{ .d = d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
        r.f = Forward.init(a, cfg, widths, .{}, &r.runner, comm);
        r.masks = try gpu.Masks.init(a, &r.f, 2);
        r.logits = try cuda.DeviceBuffer.alloc(d, 2 * width * 4);
        try r.runner.external("w.logits", r.logits.ptr);
        const states = try a.alloc(@import("forward.zig").Slot, 2);
        @memset(states, .{});
        const tx = try a.alloc(@import("cuda_transaction.zig").Transaction, 2);
        @memset(tx, .{});
        r.slots = try a.create(SlotSet);
        r.slots.* = .{ .gpa = a, .f = &r.f, .n = 2, .states = states, .transactions = tx };
    }

    fn select(r: *Rank, slot: u32) !void {
        // SlotSet.view needs only Kv.select here; no model KV pages or arithmetic are exercised by this fixture.
        var k: @import("cuda_kv.zig").Kv = undefined;
        var pool_slots: [2]*@import("kv").sessions.Slot = undefined;
        var pool_states: [2]@import("kv").sessions.Slot = undefined;
        for (&pool_slots, &pool_states) |*p, *state| p.* = state;
        k.slots = &pool_slots;
        r.f.kv = &k;
        defer r.f.kv = null;
        try r.slots.view(slot);
    }

    fn check(r: *Rank, masked: bool, row: u32) !void {
        var logits: [2 * width]f32 = @splat(1);
        for (0..2) |i| {
            logits[i * width] = 100;
            logits[i * width + 5] = 10;
        }
        @memcpy(mock.memory[r.logits.ptr / 256][0 .. 2 * width * 4], std.mem.sliceAsBytes(&logits));
        const before = launches;
        const hook = r.f.grammar.?;
        try hook.apply(hook.ptr, 0, 2);
        @memcpy(std.mem.sliceAsBytes(&logits), mock.memory[r.logits.ptr / 256][0 .. 2 * width * 4]);
        for (0..2) |i| {
            const active = masked and i == row;
            const local = logits[i * width ..][0..width];
            const choice = @import("greedy.zig").argmax(local);
            try t.expectEqual(@as(usize, if (active) 5 else 0), choice);
            try t.expectEqual(@as(usize, r.masks.lo) + @as(usize, if (active) 5 else 0), r.masks.lo + choice);
            for (local, 0..) |v, col| {
                const expected: f32 = if (active and col != 5) -std.math.inf(f32) else if (col == 0) 100 else if (col == 5) 10 else 1;
                try t.expectEqual(expected, v);
            }
        }
        try t.expectEqual(before + @as(usize, @intFromBool(masked)), launches);
    }

    fn deinit(r: *Rank) void {
        r.masks.deinit();
        r.logits.free();
        r.slots.deinit();
        r.f.arena.deinit();
        r.runner.deinit();
    }
};

fn checkBoth(ranks: *[2]Rank, masked: bool, row: u32) !void {
    for (ranks, 0..) |*r, rank| {
        try t.expectEqual(@as(u32, @intCast(rank * width)), r.masks.lo);
        try r.check(masked, row);
    }
}

fn runLifecycle(cancelled: bool, next_slot: u32) !void {
    var d = mock.init();
    d.api.cuLaunchKernel = launch;
    d.api.cuModuleLoadDataEx = loadPtx;
    launches = 0;
    var wire = try Wire.init();
    defer wire.deinit();
    var group = tp.host.Group.init(2);
    var comms = [_]tp.host.Rank{ group.rank(0), group.rank(1) };
    var cfg: @import("config.zig").Config = .{ .vocab = vocab };
    var widths: @import("block.zig").Widths = .{};
    var ranks: [2]Rank = undefined;
    try ranks[0].init(&d, &cfg, &widths, comms[0].iface());
    defer ranks[0].deinit();
    try ranks[1].init(&d, &cfg, &widths, comms[1].iface());
    defer ranks[1].deinit();
    ranks[0].f.link = &wire.lead;
    ranks[1].f.link = &wire.peer;
    for (&ranks) |*r| try r.select(0);
    var bits: [words]u32 = @splat(0);
    bits[0] = 1 << 5;
    bits[4] = 1 << 5;
    try gpu.stage(&ranks[0].f, &.{.{ .index = null, .bits = &bits }});
    try wire.replay(&ranks[1].f);
    try checkBoth(&ranks, true, 1);

    // Cancellation releases a replacement stage before it is applied; completion releases the already applied stage.
    if (cancelled) {
        try gpu.stage(&ranks[0].f, &.{.{ .index = 0, .bits = &bits }});
        try wire.replay(&ranks[1].f);
    }
    ranks[0].f.quiet = true;
    for (&ranks) |*r| {
        try r.f.release();
        try r.select(next_slot);
    }
    const before_clear = wire.frames.items.len;
    ranks[0].f.quiet = false;
    try gpu.segments(&ranks[0].f, &.{.{ .slot = next_slot, .start = 0, .tokens = &.{1}, .parents = null, .draws = &.{0}, .sampling = null }});
    try wire.replay(&ranks[1].f);
    try checkBoth(&ranks, false, 0);
    try t.expectEqual(before_clear + 1, wire.frames.items.len);
    try t.expectEqualSlices(i64, &.{ Forward.op_grammar, words, 0 }, wire.frames.items[before_clear]);

    var seq: lanes.grammar.Sequence = .{ .want = &.{5}, .stop = 31, .gpa = a };
    var grammar = lanes.grammar.Constraint.init(seq.rules(), words, 30);
    defer grammar.rules.vtable.free(grammar.rules.ptr);
    try t.expect((try grammar.next(a)) == null);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const inactive = try grammar.window(arena.allocator(), &.{ 1, 2 }, &.{ -1, 0 }, false, 2);
    const before_idle = wire.frames.items.len;
    try gpu.segments(&ranks[0].f, &.{.{ .slot = next_slot, .start = 1, .tokens = &.{ 1, 2 }, .parents = null, .draws = &.{ 0, 1 }, .sampling = null, .masks = inactive.masks }});
    try wire.replay(&ranks[1].f);
    try checkBoth(&ranks, false, 0);
    try t.expectEqual(before_idle, wire.frames.items.len);
    for (&ranks) |*r| try r.select(0);
    try gpu.segments(&ranks[0].f, &.{.{ .slot = 0, .start = 0, .tokens = &.{1}, .parents = null, .draws = &.{0}, .sampling = null, .masks = &bits }});
    try wire.replay(&ranks[1].f);
    try checkBoth(&ranks, true, 0);
    try t.expectEqual(before_idle + 1, wire.frames.items.len);
}

test "grammar lifecycle: completion clears production leader and follower masks before plain and inactive requests" {
    for ([_]u32{ 0, 1 }) |slot| try runLifecycle(false, slot);
}

test "grammar lifecycle: cancellation clears production leader and follower masks before slot reuse" {
    for ([_]u32{ 0, 1 }) |slot| try runLifecycle(true, slot);
}
