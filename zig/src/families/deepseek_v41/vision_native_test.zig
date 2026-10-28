//! Shared API preparation, real decoding, request validation and image ownership without a device.
const std = @import("std");
const t = std.testing;
const api = @import("engine_api");
const native = @import("vision_native.zig");
const prep = @import("vision/prep.zig");
const held = @import("vision/held.zig");
const Store = @import("vision/store.zig").Store;
const assets = @import("vision_assets.zig");
const Pack = @import("pack.zig").Pack;

fn fixture() held.HeldImage {
    return .{ .patches = &patches, .vit_h = 3, .vit_w = 3, .llm_h = 1, .llm_w = 1, .digest = @splat(3), .vids = &ids, .n_vids = 4 };
}
const patches: [3 * 3 * 588]u16 = @splat(0x3f80);
const ids = [_]u32{ (1 << 24) + 1, (1 << 24) + 2, (1 << 24) + 3, (1 << 24) + 4 };

fn admitAndRelease(a: std.mem.Allocator) !void {
    var store: Store = .{ .gpa = a };
    defer store.deinit();
    var im = fixture();
    const h: held.Held = .{ .images = @ptrCast(&im), .n = 1 };
    store.hold(1, &h) catch |err| {
        try t.expectEqual(@as(u32, 0), store.where.count());
        try t.expectEqual(@as(u32, 0), store.holds.count());
        return err;
    };
    try t.expect(store.where.get(ids[0]).?.img.patches != im.patches);
    try store.hold(2, &h);
    try t.expectEqual(@as(u32, 2), store.where.get(ids[0]).?.refs);
    store.release(1);
    try t.expectEqual(@as(u32, 1), store.where.get(ids[0]).?.refs);
    try t.expectEqual(@as(u16, 0x3f80), store.where.get(ids[0]).?.img.patches[0]);
    store.release(2);
    store.release(2);
    try t.expectEqual(@as(u32, 0), store.where.count());
}

test "image ownership: live requests survive another holder releasing, with all allocation failures" {
    try t.checkAllAllocationFailures(t.allocator, admitAndRelease, .{});
}

test "image ownership: duplicate request and colliding ids leave the first request intact" {
    var store: Store = .{ .gpa = t.allocator };
    defer store.deinit();
    var im = fixture();
    const h: held.Held = .{ .images = @ptrCast(&im), .n = 1 };
    try store.hold(1, &h);
    try t.expectError(error.DuplicateRequest, store.hold(1, &h));
    im.digest = @splat(8);
    try t.expectError(error.ImageIdCollision, store.hold(2, &h));
    try t.expectEqual(@as(u32, 1), store.holds.count());
    try t.expectEqual(@as(u32, 1), store.where.get(ids[0]).?.refs);
    im.vit_h = 0;
    try t.expectError(error.VisionGrid, store.hold(3, &h));
}

test "native offer decodes actual PNG, caps geometry and preserves BF16 patches through the shared API" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var n = try native.Native.init(t.allocator, t.io, .{}, .{});
    defer n.deinit();
    const offer = n.offer();
    const raw = @embedFile("vision/fixtures/golden/rgb8.png");
    const p = try offer.prepare(offer.ctx, a, raw, 128);
    try t.expect(p.tokens <= 128);
    try t.expectEqual(prep.spanLen((p.gh + 2) / 3, (p.gw + 2) / 3), p.tokens);
    try t.expectEqual(@as(usize, p.gh) * p.gw * 588, p.pixels.len);
    const prompt = try a.alloc(u32, p.tokens + 2);
    @memset(prompt, prep.IMAGE_TOKEN);
    prompt[0] = 1;
    prompt[prompt.len - 1] = 2;
    const images = [_]api.Image{.{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .at = 1, .tokens = p.tokens }};
    const req: api.Request = .{ .prompt = prompt, .images = &images, .max_tokens = 1 };
    const owned = try n.request(a, &req);
    try t.expectEqual(@as(usize, 0), owned.request.images.len);
    try t.expectEqual(@as(u32, 1), owned.request.prompt[0]);
    try t.expectEqual(@as(u32, 2), owned.request.prompt[prompt.len - 1]);
    try t.expect(owned.request.prompt[1] >= (1 << 24));
    try t.expectEqualSlices(u32, owned.request.prompt, owned.request.cacheKey());
    for (p.pixels, 0..) |f, i| try t.expectEqual(prep.bf16(f), owned.images.slice()[0].patches[i]);
    const again = try n.request(a, &req);
    try t.expectEqualSlices(u32, owned.request.prompt, again.request.prompt);
    var bad_images = images;
    bad_images[0].tokens -= 1;
    var bad = req;
    bad.images = &bad_images;
    try t.expectError(error.VisionGrid, n.request(a, &bad));
    prompt[1] = 17;
    try t.expectError(error.ImagePlaceholder, n.request(a, &req));
    try t.expectError(error.Unsupported, offer.prepare(offer.ctx, a, "not an image", 128));
    try t.expectError(error.NoGrid, offer.prepare(offer.ctx, a, raw, 4));
    try t.expectError(error.UnsupportedVisionGeometry, offer.prepare(offer.ctx, a, raw, 5));
}

test "missing vision assets and unsupported geometry refuse before touching CUDA" {
    var p = Pack.init(t.allocator);
    defer p.deinit();
    try t.expectError(error.MissingVisionWeight, assets.validate(&p, .{}));
    try t.expectError(error.UnsupportedVisionGeometry, assets.validate(&p, .{ .heads = 0 }));
    try t.expectError(error.UnsupportedVisionGeometry, native.Native.init(t.allocator, t.io, .{ .patch = 16 }, .{}));
    try t.expectError(error.BadVisionLimits, native.Native.init(t.allocator, t.io, .{}, .{ .image_tokens = 4 }));
    try t.expectError(error.BadSettings, prep.plan(0, 1, &.{}));
    try t.expectError(error.BadSettings, prep.plan(1, 1, &.{ .downsample = 0 }));
}

test "native admission refuses nine patches before allocation and accepts eighteen patches unchanged" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var n = try native.Native.init(t.allocator, t.io, .{}, .{ .image_tokens = 5 });
    defer n.deinit();
    const pixels: [3 * 6 * 588]f32 = @splat(1);
    const small: api.Image = .{ .pixels = pixels[0 .. 3 * 3 * 588], .gh = 3, .gw = 3, .tokens = 4, .at = 0 };
    const rejected: api.Request = .{ .prompt = &.{ 129264, 129264, 129264, 129264 }, .images = &.{small}, .max_tokens = 1 };
    var no_memory: [0]u8 = .{};
    var exhausted = std.heap.FixedBufferAllocator.init(&no_memory);
    try t.expectError(error.UnsupportedVisionGeometry, n.request(exhausted.allocator(), &rejected));
    const valid: api.Image = .{ .pixels = &pixels, .gh = 3, .gw = 6, .tokens = 5, .at = 0 };
    const accepted: api.Request = .{ .prompt = &.{ 129264, 129264, 129264, 129264, 129264 }, .images = &.{valid}, .max_tokens = 1 };
    const owned = try n.request(arena.allocator(), &accepted);
    const image = owned.images.slice()[0];
    try t.expectEqual(@as(u32, 3), image.vit_h);
    try t.expectEqual(@as(u32, 6), image.vit_w);
    try t.expectEqual(@as(u32, 1), image.llm_h);
    try t.expectEqual(@as(u32, 2), image.llm_w);
    try t.expectEqual(@as(u64, 5), image.n_vids);
    for (image.patches[0..pixels.len]) |value| try t.expectEqual(@as(u16, 0x3f80), value);
}
