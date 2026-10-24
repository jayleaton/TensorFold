//! Host tests of the memory floor: chunk gates on every rank, boot after the pool, rank 0's admission, the backend.
const std = @import("std");
const lanes = @import("lanes");
const fakes = @import("engine_fake.zig");
const floor = @import("floor.zig");

const gpa = std.testing.allocator;
const World = fakes.World;
const G = floor.GiB;

fn prompt(n: usize, salt: u32) ![]u32 {
    const p = try gpa.alloc(u32, n);
    for (p, 0..) |*t, i| t.* = @intCast((i * 7 + salt) % 90);
    return p;
}

fn stream(ids: []const u32) !lanes.Stream {
    return lanes.Stream.init(gpa, .{ .id = "f", .prompt = ids, .max_new = 8 });
}

test "rank 1 under the floor: both ranks stop the same chunk before its forward, and rank 0 holds prompts beside others" {
    // gathers: the pool's attach, then the prompt's first chunk gate
    const w = try World.createMem(gpa, 3000, 2, 60, false, .{ &.{9 * G}, &.{ 9 * G, 3 * G } });
    defer w.destroy();
    try std.testing.expect(w.e0.floorAdmits());
    const ids = try prompt(2500, 3);
    defer gpa.free(ids);
    var s = try stream(ids);
    defer s.deinit(gpa);
    const b = w.lanes.backend();
    try std.testing.expectError(error.MemoryFloor, b.prefill(&s)); // alone: the floor's words, not a requeue
    try std.testing.expectEqual(@as(usize, 0), w.r0.runs.items.len);
    try std.testing.expectEqual(3 * G, w.e0.peer_low.load(.acquire));
    try std.testing.expect(!w.e0.floorAdmits()); // rank 1's last reading keeps the next prompt waiting
    b.release(&s);
    try std.testing.expect(w.pg.pages.idle());
    try w.replay(); // rank 1 hears the same samples, skips the same forward and goes on to the release
    try std.testing.expectEqual(@as(usize, 0), w.r1.runs.items.len);
    try std.testing.expectEqual(@as(usize, 0), w.hooks1.failures);
}

test "above the floor the gates pass on both ranks and the same forwards run" {
    const w = try World.createMem(gpa, 3000, 2, 60, false, .{ &.{9 * G}, &.{8 * G} });
    defer w.destroy();
    const ids = try prompt(2500, 5);
    defer gpa.free(ids);
    var s = try stream(ids);
    defer s.deinit(gpa);
    const b = w.lanes.backend();
    try b.prefill(&s);
    b.release(&s);
    try std.testing.expect(w.e0.floorAdmits());
    try w.replay();
    try std.testing.expectEqual(@as(usize, 2), w.r0.runs.items.len);
    try std.testing.expectEqual(w.r0.runs.items.len, w.r1.runs.items.len);
    try std.testing.expectEqual(@as(usize, 0), w.hooks1.failures);
}

test "a pool that leaves a rank under the hard floor stops the start on every rank" {
    try std.testing.expectError(error.MemoryFloor, World.createMem(gpa, 3000, 2, 60, false, .{ &.{9 * G}, &.{3 * G} }));
    // the follower decides from the same samples: a second attach under the floor stops it, and it tells the others
    const w = try World.createMem(gpa, 3000, 2, 60, false, .{ &.{9 * G}, &.{ 9 * G, 4 * G } });
    defer w.destroy();
    try std.testing.expectError(error.MemoryFloor, w.lanes.x.attach(2, 60));
    try std.testing.expectError(error.MemoryFloor, w.replay());
    try std.testing.expectEqual(@as(usize, 1), w.hooks1.failures);
    try std.testing.expectEqualStrings("MemoryFloor", w.hooks1.failed[0]);
}

test "under the floor beside a running stream a prompt waits (LaneMemory); alone it is refused" {
    // gathers: attach, A's chunk, B's chunk (rank 1 low), C's chunk alone (still low)
    const w = try World.createMem(gpa, 3000, 3, 90, false, .{ &.{9 * G}, &.{ 9 * G, 9 * G, 4 * G } });
    defer w.destroy();
    const b = w.lanes.backend();
    const ida = try prompt(300, 1);
    defer gpa.free(ida);
    var a = try stream(ida);
    defer a.deinit(gpa);
    try b.prefill(&a);
    const idb = try prompt(400, 2);
    defer gpa.free(idb);
    var sb = try stream(idb);
    defer sb.deinit(gpa);
    try std.testing.expectError(error.LaneMemory, b.prefill(&sb));
    b.release(&sb);
    b.release(&a);
    var c = try stream(idb);
    defer c.deinit(gpa);
    try std.testing.expectError(error.MemoryFloor, b.prefill(&c));
    b.release(&c);
    try w.replay();
    try std.testing.expectEqual(@as(usize, 0), w.hooks1.failures);
    try std.testing.expectEqual(@as(usize, 1), w.r1.runs.items.len); // A's prompt alone ran
}
