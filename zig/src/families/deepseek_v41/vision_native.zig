//! The shared native vision contract: encoded bytes to patches, typed request images to owned virtual prompt ids.
const std = @import("std");
const api = @import("engine_api");
const prep = @import("vision/prep.zig");
const decode = @import("vision/decode.zig");
const held = @import("vision/held.zig");
const vids = @import("vision/vids.zig");
const Allocator = std.mem.Allocator;

pub const Native = struct {
    gpa: Allocator,
    io: std.Io,
    settings: prep.Settings,
    limits: api.VisionOpen,
    registry: vids.Registry,

    pub fn init(gpa: Allocator, io: std.Io, settings: prep.Settings, limits: api.VisionOpen) !Native {
        if (settings.patch != 14 or settings.downsample != 3) return error.UnsupportedVisionGeometry;
        if (limits.image_tokens < 5 or limits.image_tokens > (1 << 16) or limits.max_images == 0)
            return error.BadVisionLimits;
        return .{ .gpa = gpa, .io = io, .settings = settings, .limits = limits, .registry = .{ .gpa = gpa } };
    }

    pub fn deinit(n: *Native) void {
        n.registry.deinit();
    }

    pub fn offer(n: *Native) api.Vision {
        return .{ .ctx = n, .prepare = prepare, .image_token = n.settings.image_token, .image_tokens = n.limits.image_tokens, .max_images = n.limits.max_images };
    }

    fn prepare(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) !api.PreparedImage {
        const n: *Native = @ptrCast(@alignCast(ctx));
        if (bytes.len > n.settings.max_bytes) return error.TooLarge;
        var settings = n.settings;
        settings.cap_tokens = @min(@min(settings.budget(), n.limits.image_tokens), max_tokens);
        if (settings.cap_tokens < 5) return error.NoGrid;
        var which: []const u8 = "";
        var rgb = try decode.rgb(a, bytes, settings.max_pixels, &which);
        defer rgb.deinit(a);
        var p = try prep.preprocess(a, rgb, &settings);
        defer p.deinit(a);
        if (@as(u64, p.vit_h) * p.vit_w < 16) return error.UnsupportedVisionGeometry;
        const pixels = try a.alloc(f32, p.patches.len);
        errdefer a.free(pixels);
        for (p.patches, pixels) |b, *f| f.* = @bitCast(@as(u32, b) << 16);
        return .{ .pixels = pixels, .gh = p.vit_h, .gw = p.vit_w, .tokens = p.tokens(), .hash = std.mem.readInt(u64, p.digest[0..8], .little) };
    }

    /// Allocate in a request arena and retain until its finished event, including cancellation.
    pub fn request(n: *Native, a: Allocator, src: *const api.Request) !Owned {
        if (src.images.len > n.limits.max_images) return error.TooManyImages;
        for (src.images) |im| {
            if (im.gh == 0 or im.gw == 0 or im.gh > 4096 or im.gw > 4096) return error.VisionGrid;
            if (@as(u64, im.gh) * im.gw < 16) return error.UnsupportedVisionGeometry;
        }
        var req = src.*;
        const prompt = try a.dupe(u32, src.prompt);
        errdefer a.free(prompt);
        for (prompt) |id| if (id >= vids.VBASE) return error.InvalidImageId;
        const images = try a.alloc(held.HeldImage, src.images.len);
        var done: usize = 0;
        errdefer {
            for (images[0..done]) |im| {
                a.free(im.patches[0 .. @as(usize, im.vit_h) * im.vit_w * 588]);
                a.free(im.vids[0..im.n_vids]);
            }
            a.free(images);
        }
        var end: usize = 0;
        var total: u64 = 0;
        for (src.images, images) |im, *out| {
            const lh = (im.gh + 2) / 3;
            const lw = (im.gw + 2) / 3;
            if (im.tokens != prep.spanLen(lh, lw)) return error.VisionGrid;
            total += im.tokens;
            if (total > n.limits.image_tokens) return error.TooManyImageTokens;
            const stop = std.math.add(usize, im.at, im.tokens) catch return error.VisionGrid;
            if (im.at < end or stop > prompt.len) return error.VisionGrid;
            end = stop;
            for (prompt[im.at..stop]) |id| if (id != n.settings.image_token) return error.ImagePlaceholder;
            const np = @as(usize, im.gh) * im.gw * 588;
            if (im.pixels.len != np) return error.VisionGrid;
            const patches = try a.alloc(u16, np);
            errdefer a.free(patches);
            for (im.pixels, patches) |f, *b| {
                if (!std.math.isFinite(f)) return error.InvalidImagePixel;
                b.* = prep.bf16(f);
            }
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            h.update(prep.VERSION);
            var grid: [8]u8 = undefined;
            std.mem.writeInt(u32, grid[0..4], im.gh, .little);
            std.mem.writeInt(u32, grid[4..8], im.gw, .little);
            h.update(&grid);
            h.update(std.mem.sliceAsBytes(patches));
            var digest: [32]u8 = undefined;
            h.final(&digest);
            const registered = try n.registry.vids(n.io, &digest, im.tokens);
            defer n.gpa.free(registered);
            const ids = try a.dupe(u32, registered);
            @memcpy(prompt[im.at..stop], ids);
            out.* = .{ .patches = patches.ptr, .vit_h = im.gh, .vit_w = im.gw, .llm_h = lh, .llm_w = lw, .digest = digest, .vids = ids.ptr, .n_vids = ids.len };
            done += 1;
        }
        req.prompt = prompt;
        req.images = &.{};
        // Cache identity follows the ids the forward actually consumed, including registry collision salts.
        req.cache_key = prompt;
        return .{ .request = req, .images = .{ .images = images.ptr, .n = images.len } };
    }
};

pub const Owned = struct { request: api.Request, images: held.Held };
