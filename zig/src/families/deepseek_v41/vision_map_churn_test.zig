//! Churn the actual image owners while a retained request and bounded span resources stay live.
const std = @import("std");
const t = std.testing;
const api = @import("engine_api");
const held = @import("vision/held.zig");
const Store = @import("vision/store.zig").Store;
const Native = @import("vision_native.zig").Native;
const Wrapper = @import("vision_engine.zig").Wrapper(Store);
const Churn = @import("map_churn.zig").Churn;

fn freeBuckets(map: anytype) !void {
    const capacity = map.capacity();
    try t.expect(capacity > 0);
    var free: usize = 0;
    for (map.metadata.?[0..capacity]) |bucket| free += @intFromBool(bucket.isFree());
    try t.expect(free * 10 >= capacity);
}

const patches: [3 * 3 * 588]u16 = @splat(0x3f80);

test "vision churn: actual Store retains image inputs while both indexes reclaim deleted buckets" {
    var store: Store = .{ .gpa = t.allocator };
    defer store.deinit();
    var ids = [_]u32{ (1 << 24) + 1, (1 << 24) + 2, (1 << 24) + 3, (1 << 24) + 4 };
    var image: held.HeldImage = .{ .patches = &patches, .vit_h = 3, .vit_w = 3, .llm_h = 1, .llm_w = 1, .digest = @splat(0), .vids = &ids, .n_vids = 4 };
    const src: held.Held = .{ .images = @ptrCast(&image), .n = 1 };
    try store.hold(1, &src);
    const retained = store.holds.get(1).?;
    const retained_id = ids[0];
    for (0..2048) |i| {
        std.mem.writeInt(u64, image.digest[0..8], i + 1, .little);
        for (&ids, 0..) |*id, j| id.* = @intCast((1 << 24) + 5 + i * 4 + j);
        try store.hold(i + 2, &src);
        store.release(i + 2);
        try t.expectEqual(@as(u32, 1), store.holds.count());
        try t.expectEqual(@as(u32, 4), store.where.count());
        try t.expect(store.holds.get(1).? == retained);
        const loc = store.where.get(retained_id).?;
        try t.expect(loc.img == &retained.slice()[0]);
        try t.expectEqual(@as(u32, 1), loc.refs);
        try t.expectEqual(retained_id, loc.img.vids[0]);
        try t.expectEqualSlices(u16, &patches, loc.img.patches[0..patches.len]);
        try freeBuckets(store.where);
        try freeBuckets(store.holds);
        try t.expect(store.where.capacity() <= 32 and store.holds.capacity() <= 16);
    }
    store.release(1);
    try t.expectEqual(@as(u32, 0), store.where.count());
    try t.expectEqual(@as(u32, 0), store.holds.count());
}

const Inner = struct {
    retained: ?*const api.Request = null,
    sink: ?api.Sink = null,

    fn engine(f: *Inner) api.Engine {
        return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const f: *Inner = @ptrCast(@alignCast(ctx));
        if (id == 1) {
            f.retained = request;
            f.sink = sink;
        } else {
            sink.event(sink.ctx, id, &.{ .tokens = request.prompt[0..1] });
            sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
        }
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Result = struct {
    token: ?u32 = null,
    finished: bool = false,
    transients: usize = 0,

    fn event(ctx: *anyopaque, id: api.Id, ev: *const api.Event) void {
        const r: *Result = @ptrCast(@alignCast(ctx));
        if (id != 1) {
            if (ev.* == .finished) r.transients += 1;
            return;
        }
        switch (ev.*) {
            .tokens => |tokens| r.token = tokens[0],
            .finished => r.finished = true,
            else => {},
        }
    }
};

test "vision churn: actual Wrapper keeps a retained request and its result through transient completions" {
    var native = try Native.init(t.allocator, t.io, .{}, .{});
    defer native.deinit();
    var store: Store = .{ .gpa = t.allocator };
    defer store.deinit();
    var inner: Inner = .{};
    var wrapper: Wrapper = .{ .gpa = t.allocator, .io = t.io, .inner = inner.engine(), .native = &native, .rows = &store };
    defer wrapper.deinit();
    var pixels: [3 * 6 * 588]f32 = @splat(1);
    const image: api.Image = .{ .pixels = &pixels, .gh = 3, .gw = 6, .tokens = 5, .at = 0 };
    const request: api.Request = .{ .prompt = &.{ 129264, 129264, 129264, 129264, 129264 }, .images = &.{image}, .max_tokens = 1 };
    var result: Result = .{};
    const sink: api.Sink = .{ .ctx = &result, .event = Result.event };
    const engine = wrapper.engine();
    try engine.submit(1, &request, sink);
    const retained = inner.retained.?;
    const first = retained.prompt[0];
    var retained_ids: [5]u32 = undefined;
    @memcpy(&retained_ids, retained.prompt);
    for (0..1024) |i| {
        pixels[0] = @floatFromInt(i + 2);
        try engine.submit(i + 2, &request, sink);
        try t.expectEqual(@as(u32, 1), wrapper.active.count());
        try t.expectEqual(@as(u32, 1), store.holds.count());
        try t.expectEqual(@as(u32, 5), store.where.count());
        try t.expectEqualSlices(u32, &retained_ids, retained.prompt);
        try t.expectEqual(@as(u16, 0x3f80), store.where.get(first).?.img.patches[0]);
        try freeBuckets(wrapper.active);
        try freeBuckets(store.holds);
        try freeBuckets(store.where);
        try t.expect(wrapper.active.capacity() <= 16);
    }
    try t.expectEqual(@as(usize, 1024), result.transients);
    inner.sink.?.event(inner.sink.?.ctx, 1, &.{ .tokens = retained.prompt[0..1] });
    try t.expectEqual(@as(?u32, first), result.token);
    inner.sink.?.event(inner.sink.?.ctx, 1, &.{ .finished = .{ .reason = .stop } });
    try t.expect(result.finished);
    try t.expectEqual(@as(u32, 0), wrapper.active.count());
    try t.expectEqual(@as(u32, 0), store.where.count());
}

test "vision churn: production span helper preserves a retained resource and bounded resource indices" {
    var map: std.AutoHashMapUnmanaged([32]u8, usize) = .empty;
    defer map.deinit(t.allocator);
    var churn: Churn = .{};
    const retained: [32]u8 = @splat(0);
    var resources = [_]bool{ true, false };
    try map.put(t.allocator, retained, 0);
    const capacity = map.capacity();
    for (0..4096) |i| {
        var digest: [32]u8 = @splat(0);
        std.mem.writeInt(u64, digest[0..8], i + 1, .little);
        try t.expect(!resources[1]);
        resources[1] = true;
        try map.put(t.allocator, digest, 1);
        const resource = map.fetchRemove(digest).?.value;
        churn.dropped([32]u8, &map);
        try t.expectEqual(@as(usize, 1), resource);
        resources[resource] = false;
        try t.expect(resources[map.get(retained).?]);
        try t.expect(!resources[1]);
        try t.expectEqual(@as(u32, 1), map.count());
        try t.expectEqual(capacity, map.capacity());
        try freeBuckets(map);
    }
}
