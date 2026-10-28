//! Exercise the production wrapper with actual host image ownership and a scripted lane engine.
const std = @import("std");
const t = std.testing;
const api = @import("engine_api");
const Native = @import("vision_native.zig").Native;
const Store = @import("vision/store.zig").Store;
const Wrapper = @import("vision_engine.zig").Wrapper(Store);

const Fake = struct {
    sink: ?api.Sink = null,
    fail: bool = false,
    synchronous: bool = false,
    cancelled: bool = false,
    events: u32 = 0,
    fn engine(f: *Fake) api.Engine {
        return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.fail) return error.Busy;
        if (req.images.len != 0 or req.prompt[0] < (1 << 24)) return error.InvalidRequest;
        f.sink = sink;
        if (f.synchronous) sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(ctx: *anyopaque, _: api.Id) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.cancelled = true;
    }
    fn status(_: *anyopaque, _: *api.Status, _: []u32) void {}
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn event(ctx: *anyopaque, _: api.Id, _: *const api.Event) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.events += 1;
    }
};

fn lifecycle(a: std.mem.Allocator) !void {
    var n = try Native.init(a, t.io, .{}, .{});
    defer n.deinit();
    var store: Store = .{ .gpa = a };
    defer store.deinit();
    var fake: Fake = .{};
    var wrapper: Wrapper = .{ .gpa = a, .io = t.io, .inner = fake.engine(), .native = &n, .rows = &store };
    defer wrapper.deinit();
    const image: api.Image = .{ .pixels = &pixels, .gh = 3, .gw = 6, .tokens = 5, .at = 0 };
    const req: api.Request = .{ .prompt = &.{ 129264, 129264, 129264, 129264, 129264 }, .images = &.{image}, .max_tokens = 1 };
    const engine = wrapper.engine();
    const sink: api.Sink = .{ .ctx = &fake, .event = Fake.event };
    const small: api.Image = .{ .pixels = pixels[0 .. 3 * 3 * 588], .gh = 3, .gw = 3, .tokens = 4, .at = 0 };
    const rejected: api.Request = .{ .prompt = &.{ 129264, 129264, 129264, 129264 }, .images = &.{small}, .max_tokens = 1 };
    try t.expectError(error.InvalidRequest, engine.submit(9, &rejected, sink));
    try t.expect(fake.sink == null);
    try t.expectEqual(@as(u32, 0), store.holds.count());
    try t.expectEqual(@as(u32, 0), store.where.count());
    try t.expectEqual(@as(u32, 0), wrapper.active.count());
    try engine.submit(1, &req, sink);
    try t.expectEqual(@as(u32, 1), store.holds.count());
    engine.cancel(1);
    try t.expect(fake.cancelled);
    try t.expectEqual(@as(u32, 1), store.holds.count());
    fake.sink.?.event(fake.sink.?.ctx, 1, &.{ .finished = .{ .reason = .cancelled } });
    try t.expectEqual(@as(u32, 0), store.holds.count());
    try t.expectEqual(@as(u32, 0), wrapper.active.count());
    try t.expectEqual(@as(u32, 1), fake.events);
    fake.fail = true;
    try t.expectError(error.Busy, engine.submit(2, &req, sink));
    try t.expectEqual(@as(u32, 0), store.holds.count());
    try t.expectEqual(@as(u32, 0), wrapper.active.count());
    fake.fail = false;
    fake.synchronous = true;
    try engine.submit(3, &req, sink);
    try t.expectEqual(@as(u32, 0), wrapper.active.count());
    try t.expectEqual(@as(u32, 0), store.holds.count());
}
const pixels: [3 * 6 * 588]f32 = @splat(1.0);

test "image wrapper cancellation, submit rejection and synchronous completion release actual inputs" {
    try lifecycle(t.allocator);
}
