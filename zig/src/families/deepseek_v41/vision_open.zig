//! Image boot order: set planning flags on every rank, then open rank zero's tower, then admit requests.
const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("forward.zig");
const weights = @import("cuda_weights.zig");
const pack_mod = @import("pack.zig");
const tower_mod = @import("vision_tower.zig");
const rows_mod = @import("vision_rows.zig");
const buffers = @import("buffers.zig");
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.dsv41);

/// Before Forward.plan on every rank; failure requires discarding the partially loaded model.
pub fn plan(gpa: Allocator, io: std.Io, f: *fwd.Forward, d: *const cuda.Driver, w: *weights.Weights, bias_dir: []const u8) !void {
    if (f.planned_rows != 0) return error.ImagesAlreadyPlanned;
    if (bias_dir.len == 0) return error.MissingBiasVl;
    try loadBiasVl(gpa, io, f, d, bias_dir, w);
    f.images = true;
    f.image_bias = true;
}

/// After buffer planning, before serving; rank zero alone owns tower weights and encoded spans.
pub fn tower(gpa: Allocator, io: std.Io, f: *fwd.Forward, pack: *const pack_mod.Pack, d: *const cuda.Driver, rank: u32, shape: tower_mod.Shape) !void {
    if (!f.images or !f.image_bias) return error.ImagesUnplanned;
    if (rank != 0) return;
    if (f.vision != null) return error.VisionAlreadyOpen;
    if (shape.out != f.cfg.hidden) return error.VisionWidth;
    const t = try tower_mod.Tower.load(gpa, io, d, pack, shape);
    errdefer t.deinit();
    const r = try gpa.create(rows_mod.Rows);
    r.* = rows_mod.Rows.init(gpa, io, d, t);
    f.vision = r;
}

pub fn checkPlan(f: *fwd.Forward, p: *const buffers.Plan) !void {
    if (!f.images) return;
    var arena = std.heap.ArenaAllocator.init(f.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    if (try @import("vision_plan.zig").missing(a, f.cfg, f.widths, f.opts, try f.backbone(a), f.opts.prefill_rows, f.image_bias, p) != null) return error.ImagesUnplanned;
}

/// Only after a successful GPU drain; failure leaves these owners attached to Forward for retry.
pub fn closeAfterDrain(f: *fwd.Forward) void {
    const r: *rows_mod.Rows = @ptrCast(@alignCast(f.vision orelse return));
    const t = r.tower;
    const gpa = r.gpa;
    r.deinit();
    gpa.destroy(r);
    t.deinit();
    f.vision = null;
}

fn loadBiasVl(gpa: Allocator, io: std.Io, f: *fwd.Forward, d: *const cuda.Driver, dir: []const u8, w: *weights.Weights) !void {
    var src = try pack_mod.Pack.open(gpa, io, dir);
    defer src.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var n: usize = 0;
    for (try f.backbone(a)) |L| {
        const E = f.cfg.expertsOf(L).count;
        const name = try std.fmt.allocPrint(a, "layers.{d}.ffn.gate.bias_vl", .{L});
        const info = src.get(name) orelse {
            log.err("{s}: no {s} (bias_vl_fetch.py writes all 43)", .{ dir, name });
            return error.MissingBiasVl;
        };
        if (info.rank != 1 or info.shape[0] != E) return error.BiasVlShape;
        const element: u64 = switch (info.dtype) {
            .f32 => 4,
            .bf16 => 2,
            else => return error.BiasVlDtype,
        };
        if (info.nbytes != E * element) return error.BiasVlShape;
        const raw = try a.alloc(u8, @intCast(info.nbytes));
        try src.read(io, info, .{ .offset = 0, .len = info.nbytes }, raw);
        const vals = try a.alloc(f32, E);
        switch (info.dtype) {
            .f32 => @memcpy(std.mem.sliceAsBytes(vals), raw[0 .. 4 * E]),
            .bf16 => for (vals, 0..) |*x, i| {
                x.* = @bitCast(@as(u32, std.mem.readInt(u16, raw[2 * i ..][0..2], .little)) << 16);
            },
            else => return error.BiasVlDtype,
        }
        const bytes = std.mem.sliceAsBytes(vals);
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &sha, .{});
        const key = try std.fmt.allocPrint(w.gpa, "L{d}.moe.bias_vl", .{L});
        errdefer w.gpa.free(key);
        if (w.map.contains(key)) return error.DuplicateWeight;
        try w.bufs.ensureUnusedCapacity(w.gpa, 1);
        try w.map.ensureUnusedCapacity(w.gpa, 1);
        const buf = try cuda.DeviceBuffer.fromHost(d, bytes);
        var shape: [4]usize = @splat(1);
        shape[0] = E;
        w.bufs.appendAssumeCapacity(buf);
        w.map.putAssumeCapacity(key, .{ .ptr = buf.ptr, .len = bytes.len, .kind = .f32, .sha256 = sha, .shape = shape, .rank = 1 });
        w.bytes += bytes.len;
        n += 1;
    }
    log.info("images: gate.bias_vl of {d} layers from {s} (TF_DSV41_BIAS_VL)", .{ n, dir });
}

pub const Config = struct {
    settings: @import("vision/prep.zig").Settings,
    shape: tower_mod.Shape,
    limits: @import("engine_api").VisionOpen,
};

/// Parse checkpoint JSON in a model-lifetime arena; URL fetching stays entirely in the shared server.
pub fn readConfig(a: Allocator, text: []const u8, requested: @import("engine_api").VisionOpen) !Config {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    if (root != .object) return error.BadVisionConfig;
    const vc = root.object.get("vision_config") orelse return error.NoVisionTower;
    if (vc != .object) return error.BadVisionConfig;
    var problem: []const u8 = "";
    var settings = try @import("vision/prep.zig").Settings.read(a, text, null, &problem);
    settings.bias_vl = true;
    var shape: tower_mod.Shape = .{};
    inline for (.{ .{ "layers", "num_hidden_layers" }, .{ "heads", "num_attention_heads" }, .{ "dim", "hidden_size" }, .{ "inter", "intermediate_size" }, .{ "patch", "patch_size" }, .{ "ratio", "downsample_ratio" } }) |field| {
        const value = vc.object.get(field[1]) orelse return error.BadVisionConfig;
        if (value != .integer) return error.BadVisionConfig;
        @field(shape, field[0]) = std.math.cast(u32, value.integer) orelse return error.BadVisionConfig;
    }
    shape.out = outputWidth(root) orelse return error.BadVisionConfig;
    if (vc.object.get("rope_theta")) |theta| shape.theta = switch (theta) {
        .integer => @floatFromInt(theta.integer),
        .float => @floatCast(theta.float),
        else => return error.BadVisionConfig,
    };
    try shape.validate();
    if (settings.max_tokens < 5 or settings.max_tokens > (1 << 16) or
        settings.min_pixels > settings.max_pixels or settings.image_token >= (1 << 24)) return error.BadVisionConfig;
    if (settings.max_wh_ratio) |ratio| if (!std.math.isFinite(ratio) or ratio <= 0) return error.BadVisionConfig;
    var limits = requested;
    if (!limits.image_tokens_set) limits.image_tokens = settings.max_tokens;
    if (limits.image_tokens < 5 or limits.image_tokens > (1 << 16) or limits.max_images == 0)
        return error.BadVisionLimits;
    settings.max_images = limits.max_images;
    settings.cap_tokens = @min(settings.max_tokens, limits.image_tokens);
    settings.fetch = false;
    return .{ .settings = settings, .shape = shape, .limits = limits };
}

/// Match native config precedence: the first valid text_config, llm_config or root hidden width.
fn outputWidth(root: std.json.Value) ?u32 {
    for ([_]?std.json.Value{ root.object.get("text_config"), root.object.get("llm_config"), root }) |part| {
        const obj = part orelse continue;
        if (obj != .object) continue;
        const value = obj.object.get("hidden_size") orelse continue;
        if (value != .integer) continue;
        const width = std.math.cast(u32, value.integer) orelse continue;
        if (width > 0) return width;
    }
    return null;
}
