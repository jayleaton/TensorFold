//! MiniMax-M3.1-Flash's shape: config.json's text_config for the model, its top-level quantization_config for EXL3.
const std = @import("std");

pub const max_layers = 64;
pub const model_type = "minimax_m3_vl";

/// The EXL3 version this family was written against; a newer minor or patch is admitted, an older one is not.
pub const min_version = [3]u32{ 1, 6, 1 };

pub const Error = error{
    BadConfig, // a key is missing or has the wrong type
    NotMiniMaxM3, // model_type is not minimax_m3_vl
    NoQuantization, // no top-level quantization_config
    NotExl3, // quant_method is not exl3
    UnsupportedCodebook, // codebook is not mul1
    UnsupportedVersion, // EXL3 version older than 1.6.1, newer than 1.x, or unreadable
    UnsupportedOutScales, // out_scales is not "always": svh would not carry the output scales
    Unsupported, // a text_config feature this family does not implement
    BadShape, // widths the kernels cannot cut: not 128-aligned, heads not whole groups
    VisionUnsupported, // an image or video request: the pack ships no vision tower
};

/// The block-sparse attention index (sparse_attention_config): one shared key head, per-kv-head index queries.
pub const Index = struct {
    dim: u32,
    heads: u32,
    topk_blocks: u32,
    block: u32,
    init_block: u32, // a count of forced leading blocks (0: none), not an index
    local_block: u32, // a switch: nonzero forces the query's own block
};

pub const Quant = struct {
    version: [3]u32,
    bits: f32, // the pack's average; each matrix's own width comes from its trellis
    head_bits: u32,
};

pub const Config = struct {
    hidden: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    vocab: u32,
    max_positions: u32,
    eps: f32,
    rope_theta: f64,
    dense_inter: u32,
    moe_inter: u32,
    shared_inter: u32,
    experts: u32,
    topk: u32,
    routed_scale: f32,
    swiglu_alpha: f32,
    swiglu_limit: f32,
    gemma_norm: bool, // use_gemma_norm: every RMS norm multiplies by 1 + w
    moe: std.StaticBitSet(max_layers), // layer i holds routed experts (moe_layer_freq)
    index: Index,
    quant: Quant,
    has_vision_config: bool, // config.json describes a tower the pack does not ship

    pub fn isMoe(c: *const Config, layer: u32) bool {
        return layer < c.layers and c.moe.isSet(layer);
    }

    pub fn denseLayers(c: *const Config) u32 {
        return c.layers - @as(u32, @intCast(c.moe.count()));
    }

    /// Query heads sharing one kv head (and so one index head).
    pub fn group(c: *const Config) u32 {
        return c.heads / c.kv_heads;
    }

    /// The real MiniMax-M3.1-Flash widths and layer counts, for tools that want the shipped model only.
    pub fn isFlash(c: *const Config) bool {
        const want = flash;
        return c.hidden == want.hidden and c.layers == want.layers and c.heads == want.heads and c.kv_heads == want.kv_heads and
            c.vocab == want.vocab and c.experts == want.experts and c.topk == want.topk and c.moe_inter == want.moe_inter and
            c.dense_inter == want.dense_inter and c.moe.eql(want.moe);
    }
};

/// The shipped model's text_config, spelled out (the fixture test proves parse() gives exactly this).
pub const flash: Config = .{
    .hidden = 6144,
    .layers = 60,
    .heads = 64,
    .kv_heads = 4,
    .head_dim = 128,
    .rotary_dim = 64,
    .vocab = 200064,
    .max_positions = 1048576,
    .eps = 1e-6,
    .rope_theta = 1e7,
    .dense_inter = 12288,
    .moe_inter = 3072,
    .shared_inter = 3072,
    .experts = 128,
    .topk = 4,
    .routed_scale = 2.0,
    .swiglu_alpha = 1.702,
    .swiglu_limit = 7.0,
    .gemma_norm = true,
    .moe = flashMoe(),
    .index = .{ .dim = 128, .heads = 4, .topk_blocks = 16, .block = 128, .init_block = 0, .local_block = 1 },
    .quant = .{ .version = .{ 1, 6, 1 }, .bits = 3.55, .head_bits = 6 },
    .has_vision_config = true,
};

fn flashMoe() std.StaticBitSet(max_layers) {
    var s = std.StaticBitSet(max_layers).empty;
    var l: usize = 3;
    while (l < 60) : (l += 1) s.set(l);
    return s;
}

const Obj = std.json.ObjectMap;

fn int(o: Obj, key: []const u8) Error!u32 {
    const v = o.get(key) orelse return error.BadConfig;
    if (v != .integer) return error.BadConfig;
    return std.math.cast(u32, v.integer) orelse error.BadConfig;
}

fn float(o: Obj, key: []const u8) Error!f64 {
    const v = o.get(key) orelse return error.BadConfig;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadConfig,
    };
}

fn flag(o: Obj, key: []const u8, default: ?bool) Error!bool {
    const v = o.get(key) orelse return default orelse error.BadConfig;
    return if (v == .bool) v.bool else error.BadConfig;
}

fn text(o: Obj, key: []const u8) Error![]const u8 {
    const v = o.get(key) orelse return error.BadConfig;
    return if (v == .string) v.string else error.BadConfig;
}

fn object(o: Obj, key: []const u8) Error!Obj {
    const v = o.get(key) orelse return error.BadConfig;
    return if (v == .object) v.object else error.BadConfig;
}

/// A per-layer 0/1 list of exactly `layers` entries.
fn perLayer(o: Obj, key: []const u8, layers: u32) Error!std.StaticBitSet(max_layers) {
    const v = o.get(key) orelse return error.BadConfig;
    if (v != .array or v.array.items.len != layers) return error.BadConfig;
    var s = std.StaticBitSet(max_layers).empty;
    for (v.array.items, 0..) |e, i| {
        if (e != .integer or (e.integer != 0 and e.integer != 1)) return error.BadConfig;
        if (e.integer == 1) s.set(i);
    }
    return s;
}

fn semver(text_: []const u8) ?[3]u32 {
    var out: [3]u32 = undefined;
    var it = std.mem.splitScalar(u8, text_, '.');
    for (&out) |*p| p.* = std.fmt.parseInt(u32, it.next() orelse return null, 10) catch return null;
    return if (it.next() == null) out else null;
}

/// quantization_config: exl3, codebook mul1, out_scales, version 1.6.1 or later 1.x; mean bits `bits` or `avg_bits`.
fn quant(q: Obj) Error!Quant {
    if (!std.mem.eql(u8, try text(q, "quant_method"), "exl3")) return error.NotExl3;
    if (!std.mem.eql(u8, try text(q, "codebook"), "mul1")) return error.UnsupportedCodebook;
    if (!std.mem.eql(u8, try text(q, "out_scales"), "always")) return error.UnsupportedOutScales;
    const v = semver(try text(q, "version")) orelse return error.UnsupportedVersion;
    if (v[0] != min_version[0] or std.mem.order(u32, &v, &min_version) == .lt) return error.UnsupportedVersion;
    return .{ .version = v, .bits = @floatCast(float(q, "bits") catch try float(q, "avg_bits")), .head_bits = try int(q, "head_bits") };
}

/// Parses config.json (text_config and the top-level quantization_config); refuses what the kernels cannot run.
pub fn parse(gpa: std.mem.Allocator, json: []const u8) (Error || std.mem.Allocator.Error || error{SyntaxError})!Config {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, json, .{}) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.SyntaxError,
    };
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadConfig;
    const root = parsed.value.object;
    if (!std.mem.eql(u8, try text(root, "model_type"), model_type)) return error.NotMiniMaxM3;
    const t = try object(root, "text_config");
    const qv = root.get("quantization_config") orelse return error.NoQuantization;
    if (qv != .object) return error.NoQuantization;
    const q = try quant(qv.object);

    const layers = try int(t, "num_hidden_layers");
    if (layers == 0 or layers > max_layers) return error.BadConfig;
    const sp = try object(t, "sparse_attention_config");
    if (!try flag(sp, "use_sparse_attention", null)) return error.Unsupported;
    if (!std.mem.eql(u8, try text(sp, "sparse_score_type"), "max")) return error.Unsupported;
    // every layer selects blocks and none has an index value or output projection: the tensors say the same
    const all = std.StaticBitSet(max_layers).full;
    var every = all;
    every.setRangeValue(.{ .start = layers, .end = max_layers }, false);
    if (!(try perLayer(sp, "sparse_attention_freq", layers)).eql(every)) return error.Unsupported;
    if (!(try perLayer(sp, "sparse_disable_index_value", layers)).eql(every)) return error.Unsupported;

    const c: Config = .{
        .hidden = try int(t, "hidden_size"),
        .layers = layers,
        .heads = try int(t, "num_attention_heads"),
        .kv_heads = try int(t, "num_key_value_heads"),
        .head_dim = try int(t, "head_dim"),
        .rotary_dim = try int(t, "rotary_dim"),
        .vocab = try int(t, "vocab_size"),
        .max_positions = try int(t, "max_position_embeddings"),
        .eps = @floatCast(try float(t, "rms_norm_eps")),
        .rope_theta = try float(t, "rope_theta"),
        .dense_inter = try int(t, "dense_intermediate_size"),
        .moe_inter = try int(t, "intermediate_size"),
        .shared_inter = try int(t, "shared_intermediate_size"),
        .experts = try int(t, "num_local_experts"),
        .topk = try int(t, "num_experts_per_tok"),
        .routed_scale = @floatCast(try float(t, "routed_scaling_factor")),
        .swiglu_alpha = @floatCast(try float(t, "swiglu_alpha")),
        .swiglu_limit = @floatCast(try float(t, "swiglu_limit")),
        .gemma_norm = try flag(t, "use_gemma_norm", null),
        .moe = try perLayer(t, "moe_layer_freq", layers),
        .index = .{
            .dim = try int(sp, "sparse_index_dim"),
            .heads = try int(sp, "sparse_num_index_heads"),
            .topk_blocks = try int(sp, "sparse_topk_blocks"),
            .block = try int(sp, "sparse_block_size"),
            .init_block = try int(sp, "sparse_init_block"),
            .local_block = try int(sp, "sparse_local_block"),
        },
        .quant = q,
        .has_vision_config = root.contains("vision_config"),
    };
    try check(t, c);
    return c;
}

/// Features and widths: what the kernels and the 128-wide Hadamard blocks need.
fn check(t: Obj, c: Config) Error!void {
    if (try flag(t, "attention_output_gate", false) or try flag(t, "tie_word_embeddings", false)) return error.Unsupported;
    if (!try flag(t, "use_qk_norm", null) or !std.mem.eql(u8, try text(t, "qk_norm_type"), "per_head")) return error.Unsupported;
    if (!std.mem.eql(u8, try text(t, "hidden_act"), "swigluoai")) return error.Unsupported;
    if (!std.mem.eql(u8, try text(t, "scoring_func"), "sigmoid") or !try flag(t, "use_routing_bias", null)) return error.Unsupported;
    if (try int(t, "n_shared_experts") != 1) return error.Unsupported;
    if (t.get("rope_scaling") != null or t.get("rope_parameters") != null) return error.Unsupported; // plain RoPE only
    if (c.head_dim != 128 or c.index.dim != c.head_dim or c.index.block != 128) return error.BadShape;
    if (c.rotary_dim == 0 or c.rotary_dim > c.head_dim or c.rotary_dim % 2 != 0) return error.BadShape;
    if (c.heads == 0 or c.kv_heads == 0 or c.heads % c.kv_heads != 0) return error.BadShape;
    if (c.index.heads != c.kv_heads or c.index.topk_blocks == 0) return error.BadShape; // index head g selects for kv head g
    if (c.experts == 0 or c.topk == 0 or c.topk > c.experts or c.experts > 1024) return error.BadShape;
    const wide = [_]u32{ c.hidden, c.heads * c.head_dim, c.kv_heads * c.head_dim, c.dense_inter, c.moe_inter, c.shared_inter, c.vocab };
    for (wide) |w| if (w == 0 or w % 128 != 0) return error.BadShape;
}

/// generation_config.json: the defaults a request without sampling settings takes.
pub const Generation = struct { bos: u32, eos: u32, temperature: f32, top_p: f32 };

pub fn parseGeneration(gpa: std.mem.Allocator, json: []const u8) !Generation {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadConfig;
    const o = parsed.value.object;
    return .{ .bos = try int(o, "bos_token_id"), .eos = try int(o, "eos_token_id"), .temperature = @floatCast(try float(o, "temperature")), .top_p = @floatCast(try float(o, "top_p")) };
}

/// The pack has no vision tower: a request carrying images or video is refused with this error.
pub fn admitRequest(images: usize, videos: usize) error{VisionUnsupported}!void {
    if (images != 0 or videos != 0) return error.VisionUnsupported;
}

pub const fixture = @embedFile("fixtures/config.json");

test "the shipped config.json parses to the spelled-out shape" {
    const c = try parse(std.testing.allocator, fixture);
    try std.testing.expectEqualDeep(flash.index, c.index);
    try std.testing.expectEqualDeep(flash.quant, c.quant);
    try std.testing.expect(c.moe.eql(flash.moe));
    try std.testing.expect(c.isFlash());
    inline for (.{ "hidden", "layers", "heads", "kv_heads", "head_dim", "rotary_dim", "vocab", "max_positions", "eps", "rope_theta", "dense_inter", "moe_inter", "shared_inter", "experts", "topk", "routed_scale", "swiglu_alpha", "swiglu_limit", "gemma_norm", "has_vision_config" }) |f|
        try std.testing.expectEqual(@field(flash, f), @field(c, f));
    try std.testing.expectEqual(@as(u32, 3), c.denseLayers());
    try std.testing.expect(!c.isMoe(2) and c.isMoe(3) and c.isMoe(59) and !c.isMoe(60));
    try std.testing.expectEqual(@as(u32, 16), c.group());
}

/// The fixture with one `from` text replaced by `to`; the result is the caller's.
fn patched(gpa: std.mem.Allocator, from: []const u8, to: []const u8) ![]u8 {
    try std.testing.expect(std.mem.indexOf(u8, fixture, from) != null);
    return std.mem.replaceOwned(u8, gpa, fixture, from, to);
}

fn refuses(want: anyerror, from: []const u8, to: []const u8) !void {
    const gpa = std.testing.allocator;
    const json = try patched(gpa, from, to);
    defer gpa.free(json);
    try std.testing.expectError(want, parse(gpa, json));
}

test "only exl3 mul1 with output scales and version 1.6.1 or newer is admitted" {
    try refuses(error.NotExl3, "\"quant_method\": \"exl3\"", "\"quant_method\": \"gptq\"");
    try refuses(error.UnsupportedCodebook, "\"codebook\": \"mul1\"", "\"codebook\": \"mcg\"");
    try refuses(error.UnsupportedCodebook, "\"codebook\": \"mul1\"", "\"codebook\": \"3inst\"");
    try refuses(error.UnsupportedOutScales, "\"out_scales\": \"always\"", "\"out_scales\": \"auto\"");
    try refuses(error.UnsupportedVersion, "\"version\": \"1.6.1\"", "\"version\": \"1.6.0\"");
    try refuses(error.UnsupportedVersion, "\"version\": \"1.6.1\"", "\"version\": \"2.0.0\"");
    try refuses(error.UnsupportedVersion, "\"version\": \"1.6.1\"", "\"version\": \"latest\"");
    const gpa = std.testing.allocator;
    const newer = try patched(gpa, "\"version\": \"1.6.1\"", "\"version\": \"1.7.0\"");
    defer gpa.free(newer);
    try std.testing.expectEqual([3]u32{ 1, 7, 0 }, (try parse(gpa, newer)).quant.version);
}

test "the 3.06 bpw pack spells its mean bits avg_bits and its head 8 bits" {
    const gpa = std.testing.allocator;
    const json = try patched(gpa, "\"bits\": 3.55,", "\"avg_bits\": 3.06,");
    defer gpa.free(json);
    const q = (try parse(gpa, json)).quant;
    try std.testing.expectApproxEqAbs(@as(f32, 3.06), q.bits, 1e-6);
}

test "a quantization_config inside text_config only is not the top level's" {
    const gpa = std.testing.allocator;
    const json =
        \\{"model_type": "minimax_m3_vl", "text_config": {"quantization_config": {"quant_method": "exl3"}}}
    ;
    try std.testing.expectError(error.NoQuantization, parse(gpa, json));
    try refuses(error.NotMiniMaxM3, "\"model_type\": \"minimax_m3_vl\"", "\"model_type\": \"glm5_next\"");
}

test "unsupported text_config features and unaligned widths are refused" {
    try refuses(error.Unsupported, "\"attention_output_gate\": false", "\"attention_output_gate\": true");
    try refuses(error.Unsupported, "\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true");
    try refuses(error.Unsupported, "\"hidden_act\": \"swigluoai\"", "\"hidden_act\": \"silu\"");
    try refuses(error.Unsupported, "\"scoring_func\": \"sigmoid\"", "\"scoring_func\": \"softmax\"");
    try refuses(error.Unsupported, "\"sparse_score_type\": \"max\"", "\"sparse_score_type\": \"lse\"");
    try refuses(error.Unsupported, "\"qk_norm_type\": \"per_head\"", "\"qk_norm_type\": \"per_layer\"");
    try refuses(error.BadShape, "\"hidden_size\": 6144", "\"hidden_size\": 6100");
    try refuses(error.BadShape, "\"sparse_num_index_heads\": 4", "\"sparse_num_index_heads\": 2");
    try refuses(error.BadShape, "\"sparse_block_size\": 128", "\"sparse_block_size\": 64");
    try refuses(error.BadShape, "\"num_key_value_heads\": 4", "\"num_key_value_heads\": 5");
    try refuses(error.BadConfig, "\"num_hidden_layers\": 60", "\"num_hidden_layers\": 59");
    try refuses(error.BadConfig, "\"num_hidden_layers\": 60", "\"num_hidden_layers\": 65");
    try refuses(error.BadConfig, "\"head_dim\": 128", "\"head_dim\": \"128\"");
}

test "a layer without block selection is refused" {
    const gpa = std.testing.allocator;
    // the key has no digit of its own: the first 1 after it is layer 0's entry
    const at = std.mem.indexOf(u8, fixture, "\"sparse_attention_freq\": [").?;
    const json = try gpa.dupe(u8, fixture);
    defer gpa.free(json);
    json[std.mem.indexOfScalarPos(u8, json, at, '1').?] = '0';
    try std.testing.expectError(error.Unsupported, parse(gpa, json));
}

test "generation defaults and the vision refusal" {
    const g = try parseGeneration(std.testing.allocator, @embedFile("fixtures/generation_config.json"));
    try std.testing.expectEqual(Generation{ .bos = 200019, .eos = 200020, .temperature = 1.0, .top_p = 0.95 }, g);
    try admitRequest(0, 0);
    try std.testing.expectError(error.VisionUnsupported, admitRequest(1, 0));
    try std.testing.expectError(error.VisionUnsupported, admitRequest(0, 2));
}
