//! Validate every required tower tensor before opening libraries or allocating device buffers.
const std = @import("std");
const Pack = @import("pack.zig").Pack;

pub const Shape = struct {
    layers: u32 = 32,
    heads: u32 = 16,
    dim: u32 = 1024,
    inter: u32 = 2816,
    patch: u32 = 14,
    ratio: u32 = 3,
    theta: f32 = 10000,
    out: u32 = 5120,

    pub fn validate(s: Shape) !void {
        if (s.layers == 0 or s.layers > 128 or s.heads == 0 or s.heads > 32 or s.dim != s.heads * 64 or
            s.dim % 128 != 0 or s.inter == 0 or s.inter > 65536 or s.patch != 14 or s.ratio != 3 or
            s.out == 0 or s.out > 65536 or !std.math.isFinite(s.theta) or s.theta <= 0)
            return error.UnsupportedVisionGeometry;
    }
};

pub fn validate(pack: *const Pack, s: Shape) !void {
    try s.validate();
    try tensor(pack, "vision.patch_embed.proj.weight", &.{ s.dim, 3 * s.patch * s.patch }, true);
    try tensor(pack, "vision.patch_embed.proj.bias", &.{s.dim}, false);
    var name: [128]u8 = undefined;
    for (0..s.layers) |i| {
        inline for (.{ "norm1", "norm2" }) |norm|
            try tensor(pack, try std.fmt.bufPrint(&name, "vision.blocks.{d}." ++ norm ++ ".weight", .{i}), &.{s.dim}, true);
        inline for (.{ .{ "attn.wqkv", 3 * s.dim, s.dim }, .{ "attn.wo", s.dim, s.dim }, .{ "mlp.w1", 2 * s.inter, s.dim }, .{ "mlp.w2", s.dim, s.inter } }) |p| {
            try tensor(pack, try std.fmt.bufPrint(&name, "vision.blocks.{d}." ++ p[0] ++ ".weight", .{i}), &.{ p[1], p[2] }, true);
            try tensor(pack, try std.fmt.bufPrint(&name, "vision.blocks.{d}." ++ p[0] ++ ".bias", .{i}), &.{p[1]}, false);
        }
    }
    try tensor(pack, "vision.norm.weight", &.{s.dim}, true);
    try tensor(pack, "aligner.w1.weight", &.{ s.out, s.ratio * s.ratio * s.dim }, true);
    try tensor(pack, "aligner.w1.bias", &.{s.out}, false);
    try tensor(pack, "aligner.w2.weight", &.{ s.out, s.out }, true);
    try tensor(pack, "aligner.w2.bias", &.{s.out}, false);
    inline for (.{ "image_start", "image_end", "image_newline" }) |name_|
        try tensor(pack, name_, &.{s.out}, true);
}

fn tensor(pack: *const Pack, name: []const u8, shape: []const usize, required: bool) !void {
    const info = pack.get(name) orelse {
        if (required) return error.MissingVisionWeight;
        return;
    };
    if (info.dtype != .bf16) return error.VisionDtype;
    if (!std.mem.eql(usize, info.dims(), shape)) return error.VisionWeightShape;
    var bytes: u64 = 2;
    for (shape) |d| bytes = std.math.mul(u64, bytes, d) catch return error.VisionWeightShape;
    if (bytes != info.nbytes) return error.VisionWeightShape;
}
