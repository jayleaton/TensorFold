//! MiniMax-M3's checkpoint shape from config.json: GQA with partial rotary, sparse block attention, MoE and vision.
const std = @import("std");

pub const max_layers = 128;
pub const max_experts = 512;

/// The class names this reader takes; another MiniMax class (M2's full-attention one) is a different family.
pub const architectures = [_][]const u8{ "MiniMaxM3SparseForConditionalGeneration", "MiniMaxM3SparseForCausalLM" };

/// Where a parse failed: the JSON key it was reading, for the refusal text.
pub const Problem = struct { key: []const u8 = "" };

pub const Error = error{ BadMiniMaxM3Config, UnsupportedMiniMaxM3, OutOfMemory };

pub const Layer = struct { moe: bool, sparse: bool, index_value_off: bool };

/// MiniMax Sparse Attention: an index branch scores fixed-size blocks and the main branch attends the chosen ones.
pub const Sparse = struct {
    index_dim: u32 = 128,
    index_heads: u32 = 4,
    /// Index-key heads cached per token; not in config.json, the reference implementation keeps one.
    index_key_heads: u32 = 1,
    topk_blocks: u32 = 16,
    block: u32 = 128,
    init_blocks: u32 = 0,
    local_blocks: u32 = 1,

    /// Tokens one query attends at most in a sparse layer: the chosen blocks plus the always-kept ones.
    pub fn attended(s: Sparse) u64 {
        return @as(u64, s.topk_blocks + s.init_blocks + s.local_blocks) * s.block;
    }
};

pub const Vision = struct {
    hidden: u32 = 1280,
    heads: u32 = 16,
    layers: u32 = 32,
    intermediate: u32 = 5120,
    patch: u32 = 14,
    image_size: u32 = 2016,
    projection: u32 = 6144,
    rope_theta: f32 = 10000,
    spatial_merge: u32 = 2,
    temporal_patch: u32 = 2,
    max_frames: u32 = 4,
};

pub const Config = struct {
    hidden: u32 = 6144,
    layers: u32 = 60,
    heads: u32 = 64,
    kv_heads: u32 = 4,
    head_dim: u32 = 128,
    vocab: u32 = 200064,
    context: u32 = 1048576,
    eps: f32 = 1e-6,
    /// Norm weights are stored as offsets from one: y = x * (1 + w).
    gemma_norm: bool = true,
    rope_theta: f32 = 5e6,
    rotary_dim: u32 = 64,
    tie_embeddings: bool = false,
    dense_intermediate: u32 = 12288,
    moe_intermediate: u32 = 3072,
    shared_intermediate: u32 = 3072,
    experts: u32 = 128,
    topk: u32 = 4,
    shared_experts: u32 = 1,
    routed_scale: f32 = 2.0,
    /// swigluoai: gate * sigmoid(alpha * gate) * (up + 1), both clamped to the limit.
    swiglu_alpha: f32 = 1.702,
    swiglu_limit: f32 = 7.0,
    mtp_modules: u32 = 0,
    nextn_layers: u32 = 0,
    sparse: Sparse = .{},
    vision: ?Vision = null,
    image_token: ?u32 = null,
    video_token: ?u32 = null,
    layer: [max_layers]Layer = undefined,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, p: *Problem) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes, p);
    }

    pub fn denseLayers(c: *const Config) u32 {
        var n: u32 = 0;
        for (c.layer[0..c.layers]) |l| n += @intFromBool(!l.moe);
        return n;
    }

    pub fn sparseLayers(c: *const Config) u32 {
        var n: u32 = 0;
        for (c.layer[0..c.layers]) |l| n += @intFromBool(l.sparse);
        return n;
    }

    /// Query heads over KV heads: how many query heads read one cached head.
    pub fn group(c: *const Config) u32 {
        return c.heads / c.kv_heads;
    }

    /// Cached bytes one token adds on one rank: KV heads split across ranks, index keys replicated on each.
    pub fn cacheBytes(c: *const Config, element_bytes: u32, tp: u32) struct { kv: u64, index: u64 } {
        const kv_heads: u64 = if (tp > 1 and c.kv_heads % tp == 0) c.kv_heads / tp else c.kv_heads;
        const kv = 2 * @as(u64, c.layers) * kv_heads * c.head_dim * element_bytes;
        const index = @as(u64, c.sparseLayers()) * c.sparse.index_key_heads * c.sparse.index_dim * element_bytes;
        return .{ .kv = kv, .index = index };
    }
};

fn field(o: std.json.ObjectMap, key: []const u8, p: *Problem) Error!std.json.Value {
    p.key = key;
    return o.get(key) orelse error.BadMiniMaxM3Config;
}

fn check(ok: bool, key: []const u8, p: *Problem) Error!void {
    if (ok) return;
    p.key = key;
    return error.BadMiniMaxM3Config;
}

fn number(o: std.json.ObjectMap, key: []const u8, p: *Problem) Error!f64 {
    return switch (try field(o, key, p)) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadMiniMaxM3Config,
    };
}

fn int(o: std.json.ObjectMap, key: []const u8, p: *Problem) Error!u32 {
    const v = try field(o, key, p);
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.BadMiniMaxM3Config;
    return @intCast(v.integer);
}

fn positive(o: std.json.ObjectMap, key: []const u8, p: *Problem) Error!u32 {
    const n = try int(o, key, p);
    return if (n == 0) error.BadMiniMaxM3Config else n;
}

fn flag(o: std.json.ObjectMap, key: []const u8, p: *Problem) Error!bool {
    const v = try field(o, key, p);
    return if (v == .bool) v.bool else error.BadMiniMaxM3Config;
}

/// A string key that must hold one value this reader implements; another value is a different model.
fn expect(o: std.json.ObjectMap, key: []const u8, want: []const u8, p: *Problem) Error!void {
    const v = try field(o, key, p);
    if (v != .string) return error.BadMiniMaxM3Config;
    if (!std.mem.eql(u8, v.string, want)) return error.UnsupportedMiniMaxM3;
}

/// A per-layer 0/1 list of exactly `n` entries.
fn bits(o: std.json.ObjectMap, key: []const u8, n: u32, p: *Problem) Error![]const std.json.Value {
    const v = try field(o, key, p);
    if (v != .array or v.array.items.len != n) return error.BadMiniMaxM3Config;
    for (v.array.items) |b| if (b != .integer or (b.integer != 0 and b.integer != 1)) return error.BadMiniMaxM3Config;
    return v.array.items;
}

fn object(o: std.json.ObjectMap, key: []const u8, p: *Problem) Error!std.json.ObjectMap {
    const v = try field(o, key, p);
    return if (v == .object) v.object else error.BadMiniMaxM3Config;
}

fn architecture(o: std.json.ObjectMap, p: *Problem) Error!void {
    const v = try field(o, "architectures", p);
    if (v != .array or v.array.items.len == 0 or v.array.items[0] != .string) return error.BadMiniMaxM3Config;
    for (architectures) |a| if (std.mem.eql(u8, v.array.items[0].string, a)) return;
    return error.UnsupportedMiniMaxM3;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, p: *Problem) Error!Config {
    var doc = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch |e| {
        p.key = "(config.json is not JSON)";
        return if (e == error.OutOfMemory) error.OutOfMemory else error.BadMiniMaxM3Config;
    };
    defer doc.deinit();
    if (doc.value != .object) return error.BadMiniMaxM3Config;
    const top = doc.value.object;
    try architecture(top, p);
    const t = if (top.get("text_config")) |v| (if (v == .object) v.object else return error.BadMiniMaxM3Config) else top;
    var c = Config{};
    try text(&c, t, p);
    try sparse(&c, t, p);
    if (top.get("vision_config") != null) {
        c.vision = try vision(try object(top, "vision_config", p), p);
        c.image_token = try int(top, "image_token_index", p);
        c.video_token = try int(top, "video_token_index", p);
        try check(c.image_token.? < c.vocab, "image_token_index", p);
        try check(c.video_token.? < c.vocab, "video_token_index", p);
        try check(c.vision.?.projection == c.hidden, "projection_dim", p);
    }
    p.key = "";
    return c;
}

fn text(c: *Config, t: std.json.ObjectMap, p: *Problem) Error!void {
    c.hidden = try positive(t, "hidden_size", p);
    c.layers = try positive(t, "num_hidden_layers", p);
    c.heads = try positive(t, "num_attention_heads", p);
    c.kv_heads = try positive(t, "num_key_value_heads", p);
    c.head_dim = try positive(t, "head_dim", p);
    c.vocab = try positive(t, "vocab_size", p);
    c.context = try positive(t, "max_position_embeddings", p);
    c.eps = @floatCast(try number(t, "rms_norm_eps", p));
    c.gemma_norm = try flag(t, "use_gemma_norm", p);
    c.rope_theta = @floatCast(try number(t, "rope_theta", p));
    c.rotary_dim = try positive(t, "rotary_dim", p);
    c.tie_embeddings = try flag(t, "tie_word_embeddings", p);
    c.dense_intermediate = try positive(t, "dense_intermediate_size", p);
    c.moe_intermediate = try positive(t, "intermediate_size", p);
    c.shared_intermediate = try positive(t, "shared_intermediate_size", p);
    c.experts = try positive(t, "num_local_experts", p);
    c.topk = try positive(t, "num_experts_per_tok", p);
    c.shared_experts = try int(t, "n_shared_experts", p);
    c.routed_scale = @floatCast(try number(t, "routed_scaling_factor", p));
    c.swiglu_alpha = @floatCast(try number(t, "swiglu_alpha", p));
    c.swiglu_limit = @floatCast(try number(t, "swiglu_limit", p));
    if (t.get("num_mtp_modules") != null) c.mtp_modules = try int(t, "num_mtp_modules", p);
    if (t.get("num_nextn_predict_layers") != null) c.nextn_layers = try int(t, "num_nextn_predict_layers", p);
    try check(c.layers <= max_layers, "num_hidden_layers", p);
    try check(c.heads % c.kv_heads == 0, "num_key_value_heads", p);
    try check(c.head_dim % 32 == 0 and c.head_dim <= 256, "head_dim", p);
    try check(c.hidden % 64 == 0, "hidden_size", p);
    try check(c.eps > 0, "rms_norm_eps", p);
    p.key = "rotary_dim";
    if (c.rotary_dim % 2 != 0 or c.rotary_dim > c.head_dim) return error.BadMiniMaxM3Config;
    if (t.get("partial_rotary_factor") != null) {
        const factor = try number(t, "partial_rotary_factor", p);
        if (@abs(factor * @as(f64, @floatFromInt(c.head_dim)) - @as(f64, @floatFromInt(c.rotary_dim))) > 0.5) return error.BadMiniMaxM3Config;
    }
    // operators this family's plan implements; another value means a different model, refused rather than guessed
    if (try flag(t, "attention_output_gate", p)) return error.UnsupportedMiniMaxM3;
    if (!try flag(t, "use_qk_norm", p)) return error.UnsupportedMiniMaxM3;
    try expect(t, "qk_norm_type", "per_head", p);
    try expect(t, "hidden_act", "swigluoai", p);
    try expect(t, "scoring_func", "sigmoid", p);
    if (!try flag(t, "use_routing_bias", p)) return error.UnsupportedMiniMaxM3;
    p.key = "num_local_experts";
    if (c.experts > max_experts or c.experts % 4 != 0 or c.topk > c.experts or c.topk > 16) return error.UnsupportedMiniMaxM3;
    p.key = "intermediate_size";
    if (c.moe_intermediate % 16 != 0 or c.shared_intermediate % 16 != 0 or c.dense_intermediate % 16 != 0)
        return error.UnsupportedMiniMaxM3;
    p.key = "swiglu_limit";
    if (!(c.swiglu_limit > 0) or !(c.swiglu_alpha > 0) or !(c.routed_scale > 0)) return error.BadMiniMaxM3Config;
    const moe = try bits(t, "moe_layer_freq", c.layers, p);
    for (moe, 0..) |b, i| c.layer[i] = .{ .moe = b.integer == 1, .sparse = false, .index_value_off = false };
}

fn sparse(c: *Config, t: std.json.ObjectMap, p: *Problem) Error!void {
    const s = try object(t, "sparse_attention_config", p);
    if (!try flag(s, "use_sparse_attention", p)) return error.UnsupportedMiniMaxM3;
    c.sparse.index_dim = try positive(s, "sparse_index_dim", p);
    c.sparse.index_heads = try positive(s, "sparse_num_index_heads", p);
    c.sparse.topk_blocks = try positive(s, "sparse_topk_blocks", p);
    c.sparse.block = try positive(s, "sparse_block_size", p);
    c.sparse.init_blocks = try int(s, "sparse_init_block", p);
    c.sparse.local_blocks = try int(s, "sparse_local_block", p);
    try expect(s, "sparse_score_type", "max", p);
    p.key = "sparse_block_size";
    if (!std.math.isPowerOfTwo(c.sparse.block) or c.sparse.index_dim % 32 != 0 or c.sparse.index_dim > 256)
        return error.UnsupportedMiniMaxM3;
    const on = try bits(s, "sparse_attention_freq", c.layers, p);
    const off = try bits(s, "sparse_disable_index_value", c.layers, p);
    for (c.layer[0..c.layers], on, off) |*l, a, v| {
        l.sparse = a.integer == 1;
        l.index_value_off = v.integer == 1;
    }
}

fn vision(v: std.json.ObjectMap, p: *Problem) Error!Vision {
    var out = Vision{};
    out.hidden = try positive(v, "hidden_size", p);
    out.heads = try positive(v, "num_attention_heads", p);
    out.layers = try positive(v, "num_hidden_layers", p);
    out.intermediate = try positive(v, "intermediate_size", p);
    out.patch = try positive(v, "patch_size", p);
    out.image_size = try positive(v, "image_size", p);
    out.projection = try positive(v, "projection_dim", p);
    out.rope_theta = @floatCast(try number(v, "rope_theta", p));
    try expect(v, "position_embedding_type", "rope", p);
    try expect(v, "rope_mode", "3d", p);
    const merge = try object(v, "img_token_compression_config", p);
    try expect(merge, "image_token_compression_method", "patch_merge", p);
    out.spatial_merge = try positive(merge, "spatial_merge_size", p);
    out.temporal_patch = try positive(merge, "temporal_patch_size", p);
    if (v.get("vision_segment_max_frames") != null) out.max_frames = try positive(v, "vision_segment_max_frames", p);
    p.key = "vision_config";
    if (out.hidden % out.heads != 0 or out.image_size % out.patch != 0) return error.BadMiniMaxM3Config;
    return out;
}

test {
    _ = @import("config_test.zig");
}
