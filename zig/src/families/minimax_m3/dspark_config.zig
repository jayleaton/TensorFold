//! The base model's dspark draft: its config, what it must share with the target, its memory and its refusals.
const std = @import("std");
// Shapes are VERIFIED (the pack's config and header); what the tensors mean is not: see dspark_ref.zig.
const target = @import("config.zig");
const shard = @import("shard.zig");

pub const max_taps = 8;
pub const architecture = "DSparkMiniMaxDraftModel";

pub const Error = error{
    BadConfig, // a key is missing, has the wrong type, or two copies of one setting disagree
    NotDspark, // not a DSparkMiniMaxDraftModel of model_type minimax_m3_vl
    Unsupported, // a draft feature this family does not implement
    BadShape, // a width or head count the kernels or the cuts cannot take
};

/// How the linear layers are stored: mxfp8 = float8 e4m3 codes, one power-of-two (ue8m0) scale a 1 x 32 block.
pub const Quant = struct { block_k: u32, activation_dynamic: bool };

pub const Config = struct {
    layers: u32,
    hidden: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    vocab: u32,
    eps: f32,
    rope_theta: f64,
    ffn: u32, // dspark_ffn_hidden_size: the dense swigluoai MLP's width
    window: u32, // sliding_window: every layer is sliding_attention
    block: u32, // dspark_block_size: rows of one pass
    mask_token: u32, // dspark_noise_token_id: the rows after the anchor
    markov_rank: u32,
    swiglu_alpha: f32,
    swiglu_limit: f32,
    gemma_norm: bool, // use_gemma_norm (a switch in the reference decides what the draft really does)
    confidence: bool,
    confidence_markov: bool, // the confidence head also reads the Markov embedding of the previous token
    taps: [max_taps]u32,
    n_taps: u32,
    quant: Quant,

    pub fn tapIds(c: *const Config) []const u32 {
        return c.taps[0..c.n_taps];
    }

    pub fn qWidth(c: *const Config) u32 {
        return c.heads * c.head_dim;
    }

    pub fn kvWidth(c: *const Config) u32 {
        return c.kv_heads * c.head_dim;
    }

    /// fc's input width: the taps' hidden states side by side.
    pub fn fcIn(c: *const Config) u32 {
        return c.n_taps * c.hidden;
    }

    /// The confidence projection's input: the hidden row, then (with Markov) the previous token's Markov embedding.
    pub fn confIn(c: *const Config) u32 {
        return c.hidden + if (c.confidence_markov) c.markov_rank else 0;
    }
};

/// The shipped draft's config.json, spelled out (the fixture test proves parse() gives exactly this).
pub const flash: Config = .{
    .layers = 5,
    .hidden = 6144,
    .heads = 64,
    .kv_heads = 4,
    .head_dim = 128,
    .rotary_dim = 64,
    .vocab = 200064,
    .eps = 1e-6,
    .rope_theta = 1e7,
    .ffn = 12288,
    .window = 4096,
    .block = 7,
    .mask_token = 200058,
    .markov_rank = 256,
    .swiglu_alpha = 1.702,
    .swiglu_limit = 7.0,
    .gemma_norm = true,
    .confidence = true,
    .confidence_markov = true,
    .taps = .{ 3, 17, 31, 45, 59, 0, 0, 0 },
    .n_taps = 5,
    .quant = .{ .block_k = 32, .activation_dynamic = true },
};

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

fn flag(o: Obj, key: []const u8) Error!bool {
    const v = o.get(key) orelse return error.BadConfig;
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

fn list(o: Obj, key: []const u8) Error![]const std.json.Value {
    const v = o.get(key) orelse return error.BadConfig;
    return if (v == .array) v.array.items else error.BadConfig;
}

/// A list of whole numbers into `out`; returns how many.
fn ids(o: Obj, key: []const u8, out: []u32) Error!u32 {
    const items = try list(o, key);
    if (items.len == 0 or items.len > out.len) return error.BadConfig;
    for (items, 0..) |e, i| {
        if (e != .integer) return error.BadConfig;
        out[i] = std.math.cast(u32, e.integer) orelse return error.BadConfig;
    }
    return @intCast(items.len);
}

/// Parses the draft's config.json; refuses what the loader, the kernels or the cuts cannot take.
pub fn parse(gpa: std.mem.Allocator, json: []const u8) (Error || std.mem.Allocator.Error || error{SyntaxError})!Config {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, json, .{}) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.SyntaxError,
    };
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadConfig;
    const root = parsed.value.object;
    const arch = try list(root, "architectures");
    if (arch.len != 1 or arch[0] != .string or !std.mem.eql(u8, arch[0].string, architecture)) return error.NotDspark;
    if (!std.mem.eql(u8, try text(root, "model_type"), target.model_type)) return error.NotDspark;
    const t = try object(root, "text_config");
    const df = try object(t, "dflash_config");
    const ds = try object(t, "dspark_config");

    var c: Config = .{
        .layers = try int(t, "num_hidden_layers"),
        .hidden = try int(t, "hidden_size"),
        .heads = try int(t, "num_attention_heads"),
        .kv_heads = try int(t, "num_key_value_heads"),
        .head_dim = try int(t, "head_dim"),
        .rotary_dim = try int(t, "rotary_dim"),
        .vocab = try int(t, "vocab_size"),
        .eps = @floatCast(try float(t, "rms_norm_eps")),
        .rope_theta = try float(t, "rope_theta"),
        .ffn = try int(t, "dspark_ffn_hidden_size"),
        .window = try int(t, "sliding_window"),
        .block = try int(t, "dspark_block_size"),
        .mask_token = try int(t, "dspark_noise_token_id"),
        .markov_rank = try int(t, "dspark_markov_rank"),
        .swiglu_alpha = @floatCast(try float(t, "swiglu_alpha")),
        .swiglu_limit = @floatCast(try float(t, "swiglu_limit")),
        .gemma_norm = try flag(t, "use_gemma_norm"),
        .confidence = try flag(t, "enable_confidence_head"),
        .confidence_markov = try flag(t, "confidence_head_with_markov"),
        .taps = @splat(0),
        .n_taps = 0,
        .quant = undefined,
    };
    // the same setting is spelled in dspark_*, dspark_config and dflash_config: all copies must agree
    var again: [max_taps]u32 = undefined;
    c.n_taps = try ids(t, "dspark_target_layer_ids", &c.taps);
    if (try ids(df, "target_layer_ids", &again) != c.n_taps or !std.mem.eql(u32, again[0..c.n_taps], c.taps[0..c.n_taps])) return error.BadConfig;
    if (try int(df, "block_size") != c.block or try int(df, "mask_token_id") != c.mask_token or try int(df, "num_hidden_layers") != c.layers) return error.BadConfig;
    if (try int(df, "sliding_window") != c.window or try int(ds, "mask_token_id") != c.mask_token or try int(ds, "markov_rank") != c.markov_rank) return error.BadConfig;
    if (try int(ds, "ffn_hidden_size") != c.ffn) return error.BadConfig;
    // the pack's features: dense FFN, plain Markov head, bf16 aux GEMMs, sliding layers, no hybrid fusion, MoE or MTP
    if (!try flag(ds, "use_dense_ffn") or !try flag(t, "dspark_use_dense_ffn")) return error.Unsupported;
    if (!std.mem.eql(u8, try text(ds, "markov_head_type"), "vanilla") or !std.mem.eql(u8, try text(t, "dspark_markov_head_type"), "vanilla")) return error.Unsupported;
    if (!std.mem.eql(u8, try text(ds, "aux_gemm_precision"), "bf16")) return error.Unsupported;
    if (try flag(ds, "hybrid_context_fusion") or try flag(t, "dspark_hybrid_context_fusion")) return error.Unsupported;
    if (try flag(t, "attention_output_gate") or try flag(t, "tie_word_embeddings") or try flag(t, "use_mtp")) return error.Unsupported;
    if (!try flag(t, "use_qk_norm") or !std.mem.eql(u8, try text(t, "qk_norm_type"), "per_head")) return error.Unsupported;
    if (!std.mem.eql(u8, try text(t, "hidden_act"), "swigluoai")) return error.Unsupported;
    if (t.get("rope_scaling") != null or t.get("rope_parameters") != null) return error.Unsupported;
    if (t.get("sparse_attention_config") orelse std.json.Value{ .null = {} } != .null) return error.Unsupported;
    for (try list(df, "layer_types")) |e| if (e != .string or !std.mem.eql(u8, e.string, "sliding_attention")) return error.Unsupported;
    if ((try list(df, "layer_types")).len != c.layers) return error.BadConfig;
    for (try list(t, "moe_layer_freq")) |e| if (e != .integer or e.integer != 0) return error.Unsupported;
    // the quantization block of the pack: the draft's linear layers are mxfp8, 1 x 32 blocks
    const q = try object(root, "quantization_config");
    if (!std.mem.eql(u8, try text(q, "quant_method"), "mxfp8")) return error.Unsupported;
    const wb = try list(q, "weight_block_size");
    if (wb.len != 2 or wb[0] != .integer or wb[1] != .integer or wb[0].integer != 1 or wb[1].integer != 32) return error.Unsupported;
    c.quant = .{ .block_k = 32, .activation_dynamic = std.mem.eql(u8, try text(q, "activation_scheme"), "dynamic") };
    try check(c);
    return c;
}

/// Widths and counts: what the cuts, the 128-wide units and the scale blocks need.
fn check(c: Config) Error!void {
    if (c.layers == 0 or c.layers > 16 or c.block < 2 or c.block > 16 or c.window < 128 or c.window % 128 != 0) return error.BadShape;
    if (c.head_dim != 128 or c.rotary_dim == 0 or c.rotary_dim > c.head_dim or c.rotary_dim % 2 != 0) return error.BadShape;
    if (c.heads == 0 or c.kv_heads == 0 or c.heads % c.kv_heads != 0) return error.BadShape;
    if (c.mask_token >= c.vocab or c.markov_rank == 0 or c.markov_rank > 4096) return error.BadShape;
    if (c.confidence_markov and !c.confidence) return error.BadConfig;
    const wide = [_]u32{ c.hidden, c.heads * c.head_dim, c.kv_heads * c.head_dim, c.ffn, c.vocab };
    for (wide) |w| if (w == 0 or w % 128 != 0) return error.BadShape;
    // scale blocks run along the input axis: every K, and every rank's K cut, is whole blocks (cuts are 128s)
    if (c.hidden % c.quant.block_k != 0 or c.ffn % c.quant.block_k != 0 or c.qWidth() % c.quant.block_k != 0) return error.BadShape;
}

pub const fixture = @embedFile("fixtures/dspark_config.json");

test "the shipped dspark/config.json parses to the spelled-out shape" {
    const c = try parse(std.testing.allocator, fixture);
    inline for (@typeInfo(Config).@"struct".field_names) |name| {
        if (comptime std.mem.eql(u8, name, "quant")) continue;
        try std.testing.expectEqualDeep(@field(flash, name), @field(c, name));
    }
    try std.testing.expectEqualDeep(flash.quant, c.quant);
    try std.testing.expectEqualSlices(u32, &.{ 3, 17, 31, 45, 59 }, c.tapIds());
    try std.testing.expectEqual(@as(u32, 30720), c.fcIn());
    try std.testing.expectEqual(@as(u32, 6400), c.confIn()); // confidence_head.proj.weight is [1, 6400]
    try std.testing.expectEqual(@as(u32, 8192), c.qWidth());
    try std.testing.expectEqual(@as(u32, 512), c.kvWidth());
}

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

test "a different class, hybrid fusion, another Markov head, MoE or a non-mxfp8 pack is refused" {
    try refuses(error.NotDspark, "\"DSparkMiniMaxDraftModel\"", "\"MiniMaxM3ForCausalLM\"");
    try refuses(error.Unsupported, "\"hybrid_context_fusion\": false", "\"hybrid_context_fusion\": true");
    try refuses(error.Unsupported, "\"markov_head_type\": \"vanilla\"", "\"markov_head_type\": \"gated\"");
    try refuses(error.Unsupported, "\"aux_gemm_precision\": \"bf16\"", "\"aux_gemm_precision\": \"fp8\"");
    try refuses(error.Unsupported, "\"quant_method\": \"mxfp8\"", "\"quant_method\": \"exl3\"");
    try refuses(error.Unsupported, "\"use_dense_ffn\": true", "\"use_dense_ffn\": false");
    try refuses(error.Unsupported, "\"hidden_act\": \"swigluoai\"", "\"hidden_act\": \"silu\"");
    try refuses(error.Unsupported, "\"use_mtp\": false", "\"use_mtp\": true");
    try refuses(error.Unsupported, "\"sparse_attention_config\": null", "\"sparse_attention_config\": {}");
}

test "settings spelled twice must agree, and widths must be whole" {
    try refuses(error.BadConfig, "\"block_size\": 7", "\"block_size\": 6");
    try refuses(error.BadConfig, "\"sliding_window\": 4096\n    }", "\"sliding_window\": 2048\n    }");
    try refuses(error.BadConfig, "\"num_hidden_layers\": 5,\n      \"num_target_layers\"", "\"num_hidden_layers\": 4,\n      \"num_target_layers\"");
    try refuses(error.BadConfig, "\"dspark_noise_token_id\": 200058,\n    \"dspark_use_dense_ffn\"", "\"dspark_noise_token_id\": 200064,\n    \"dspark_use_dense_ffn\""); // the copies disagree
    try refuses(error.BadShape, "\"head_dim\": 128,\n    \"vocab_size\"", "\"head_dim\": 96,\n    \"vocab_size\"");
}

/// What a draft may and may not share with the target it reads: the same residual width, heads, rope and vocabulary.
pub const Refusal = enum {
    hidden_differs,
    vocab_differs,
    attention_differs,
    rope_differs,
    norm_eps_differs,
    tap_out_of_range,
    block_too_wide,
    needs_two_ranks_or_one,
    mask_token_is_special_hole,

    pub fn text(r: Refusal) []const u8 {
        return switch (r) {
            .hidden_differs => "--drafter: the draft's hidden width is not the model's: the draft reads the model's own hidden states and cannot be attached to another model",
            .vocab_differs => "--drafter: the draft's vocabulary is not the model's: it drafts through the model's own embedding and head",
            .attention_differs => "--drafter: the draft's head counts or head width differ from the model's, so the shared sharding cannot cut it",
            .rope_differs => "--drafter: the draft's rotary width or rope theta differ from the model's",
            .norm_eps_differs => "--drafter: the draft's norm epsilon differs from the model's",
            .tap_out_of_range => "--drafter: the draft taps a layer the model does not have",
            .block_too_wide => "--drafter: a draft block plus its pending token does not fit one verify window of this engine",
            .needs_two_ranks_or_one => "--drafter: this build runs the draft on one or two ranks only",
            .mask_token_is_special_hole => "--drafter: the draft's mask token is not a token of the model's vocabulary",
        };
    }
};

/// Where the draft would run: its ranks and the verify window the engine's lanes take.
pub const Place = struct { world: usize, window_rows: u32 };

/// null when the draft can sit beside `t`; else the first thing that cannot be.
pub fn admit(c: *const Config, t: *const target.Config, p: Place) ?Refusal {
    if (c.hidden != t.hidden) return .hidden_differs;
    if (c.vocab != t.vocab) return .vocab_differs;
    if (c.heads != t.heads or c.kv_heads != t.kv_heads or c.head_dim != t.head_dim) return .attention_differs;
    if (c.rotary_dim != t.rotary_dim or c.rope_theta != t.rope_theta) return .rope_differs;
    if (c.eps != t.eps) return .norm_eps_differs;
    for (c.tapIds()) |id| if (id >= t.layers) return .tap_out_of_range;
    if (c.mask_token >= t.vocab) return .mask_token_is_special_hole;
    if (c.block + 1 > p.window_rows) return .block_too_wide;
    if (p.world == 0 or p.world > 2) return .needs_two_ranks_or_one;
    return null;
}

test "the shipped draft sits beside the shipped model, and each mismatch has its own words" {
    const c = try parse(std.testing.allocator, fixture);
    const here: Place = .{ .world = 2, .window_rows = 16 };
    try std.testing.expect(admit(&c, &target.flash, here) == null);
    var t = target.flash;
    t.hidden = 4096;
    try std.testing.expectEqual(Refusal.hidden_differs, admit(&c, &t, here).?);
    t = target.flash;
    t.vocab = 128000;
    try std.testing.expectEqual(Refusal.vocab_differs, admit(&c, &t, here).?);
    t = target.flash;
    t.kv_heads = 8;
    try std.testing.expectEqual(Refusal.attention_differs, admit(&c, &t, here).?);
    t = target.flash;
    t.rope_theta = 1e6;
    try std.testing.expectEqual(Refusal.rope_differs, admit(&c, &t, here).?);
    t = target.flash;
    t.layers = 59; // tap 59 is the last layer of a 60-layer model and past a 59-layer one
    try std.testing.expectEqual(Refusal.tap_out_of_range, admit(&c, &t, here).?);
    try std.testing.expectEqual(Refusal.block_too_wide, admit(&c, &target.flash, .{ .world = 2, .window_rows = 7 }).?);
    try std.testing.expect(admit(&c, &target.flash, .{ .world = 2, .window_rows = 8 }) == null); // pending + 7 drafts
    try std.testing.expectEqual(Refusal.needs_two_ranks_or_one, admit(&c, &target.flash, .{ .world = 3, .window_rows = 16 }).?);
    inline for (std.enums.values(Refusal)) |r| try std.testing.expect(std.mem.startsWith(u8, r.text(), "--drafter: "));
}

// ---- memory the draft adds to a rank ------------------------------------------------------------------------------

/// Elements of one draft layer's linear weights for a rank holding `q` query rows, `kv` kv rows and `ffn` MLP rows.
fn layerElems(c: *const Config, q: u64, kv: u64, ffn: u64) u64 {
    const h: u64 = c.hidden;
    return q * h + 2 * kv * h + h * q + 3 * h * ffn;
}

pub const Weights = struct {
    codes: u64, // float8 weights, one byte each
    scales: u64, // one ue8m0 byte per 32 input elements
    fc: u64, // bf16
    markov: u64, // bf16 w1 and w2
    small: u64, // norms, the confidence projection
    pub fn total(w: Weights) u64 {
        return w.codes + w.scales + w.fc + w.markov + w.small;
    }
};

/// Bytes on one rank: layers cut as the target's are; fc and Markov tables replicated (markov_cut: cut by vocab units).
pub fn rankWeights(c: *const Config, s: shard.Share, markov_cut: bool) Weights {
    const q = @as(u64, s.q_heads.len()) * c.head_dim;
    const kv = @as(u64, s.kv_heads.len()) * c.head_dim;
    const per_layer = layerElems(c, q, kv, s.dense.len());
    const vocab_rows: u64 = if (markov_cut) s.vocab.len() else c.vocab;
    const norms = 2 * @as(u64, c.hidden) * 2 + 2 * @as(u64, c.head_dim) * 2;
    return .{
        .codes = c.layers * per_layer,
        .scales = c.layers * per_layer / c.quant.block_k,
        .fc = @as(u64, c.hidden) * c.fcIn() * 2,
        .markov = 2 * vocab_rows * c.markov_rank * 2,
        .small = c.layers * norms + 2 * @as(u64, c.hidden) * 2 + if (c.confidence) @as(u64, c.confIn()) * 2 + 2 else 0,
    };
}

/// The whole pack's bytes (one rank holding everything), as the safetensors header counts them.
pub fn wholeWeights(c: *const Config) Weights {
    return rankWeights(c, shard.Share.of(&target.flash, 0, 1) catch unreachable, false);
}

/// Rows a stream's context ring holds: the window and the rows one verify window can add before the oldest are due out.
pub fn ringRows(c: *const Config, window_rows: u32) u64 {
    return @as(u64, c.window) + window_rows;
}

/// Bytes a token adds to the context ring of one rank (per-layer K and V of the rank's kv heads), bf16.
pub fn ringTokenBytes(c: *const Config, s: shard.Share) u64 {
    return @as(u64, c.layers) * 2 * s.kv_heads.len() * c.head_dim * 2;
}

/// One stream's K/V ring on one rank; the 61,440-byte-a-token feature cache is never kept (see featureTokenBytes).
pub fn ringBytes(c: *const Config, s: shard.Share, window_rows: u32) u64 {
    return ringRows(c, window_rows) * ringTokenBytes(c, s);
}

/// bf16 tapped hidden states a token needs to be absorbed: 5 x 6144 x 2 = 61,440 bytes. Transient, per ingest chunk.
pub fn featureTokenBytes(c: *const Config) u64 {
    return @as(u64, c.fcIn()) * 2;
}

/// Scratch not growing with streams: one absorb chunk's tapped rows, fc out and hidden_norm out (an estimate).
pub fn absorbScratch(c: *const Config, absorb_rows: u32) u64 {
    return @as(u64, absorb_rows) * (featureTokenBytes(c) + 2 * @as(u64, c.hidden) * 2);
}

/// Scratch one stream's pass adds: the block's activations and the head slice's logits and candidates.
pub fn laneScratch(c: *const Config, s: shard.Share) u64 {
    const h: u64 = c.hidden;
    const rows: u64 = c.block;
    const kvw = @as(u64, s.kv_heads.len()) * c.head_dim;
    const block = rows * (4 * h * 2 + @as(u64, s.q_heads.len()) * c.head_dim * 2 * 2 + 2 * kvw * 2 + 3 * @as(u64, s.dense.len()) * 4 + 2 * h * 4);
    const head = rows * (@as(u64, s.vocab.len()) * 4 + 64 * 8 * 2); // logits slice fp32, 64 candidates (id, value) a rank
    return block + head;
}

pub fn scratchBytes(c: *const Config, s: shard.Share, absorb_rows: u32, lanes: u32) u64 {
    return absorbScratch(c, absorb_rows) + @as(u64, lanes) * laneScratch(c, s);
}

/// CUDA module, graphs and NCCL buffers the draft's kernels add: a planning figure, to be replaced by a measurement.
pub const runtime_estimate: u64 = 512 << 20;

/// What the draft costs a rank before KV: weights and runtime once, per_lane (ring and pass scratch) for every stream.
pub const Cost = struct { weights: u64, runtime: u64, per_lane: u64 };

pub fn cost(c: *const Config, s: shard.Share, window_rows: u32, absorb_rows: u32, markov_cut: bool) Cost {
    return .{
        .weights = rankWeights(c, s, markov_cut).total(),
        .runtime = runtime_estimate + absorbScratch(c, absorb_rows),
        .per_lane = ringBytes(c, s, window_rows) + laneScratch(c, s),
    };
}

test "weight bytes: the whole pack equals the safetensors total, a rank holds 1.34 GiB" {
    const c = flash;
    const whole = wholeWeights(&c);
    // model.safetensors.index.json total_size 2,301,852,674 = codes + scales + bf16 tensors
    try std.testing.expectEqual(@as(u64, 2_301_852_674), whole.total());
    try std.testing.expectEqual(@as(u64, 1_667_235_840), whole.codes);
    try std.testing.expectEqual(@as(u64, 52_101_120), whole.scales);
    try std.testing.expectEqual(@as(u64, 377_487_360), whole.fc);
    try std.testing.expectEqual(@as(u64, 204_865_536), whole.markov);
    try std.testing.expectEqual(@as(u64, 162_818), whole.small);
    const r0 = try shard.Share.of(&target.flash, 0, 2);
    const r1 = try shard.Share.of(&target.flash, 1, 2);
    const w0 = rankWeights(&c, r0, false);
    try std.testing.expectEqual(w0.total(), rankWeights(&c, r1, false).total());
    try std.testing.expectEqual(@as(u64, 1_442_184_194), w0.total());
    // the layers halve; fc, the Markov tables and the small tensors are replicated
    try std.testing.expectEqual(whole.codes / 2, w0.codes);
    try std.testing.expectEqual(whole.fc, w0.fc);
    try std.testing.expectEqual(whole.markov, w0.markov);
    // cutting the Markov tables by the head's units saves what rank 1's units are not
    const cut0 = rankWeights(&c, r0, true);
    try std.testing.expect(cut0.markov < w0.markov and cut0.markov > w0.markov / 2);
}

test "the context ring is bounded by the window: 5 KiB a token a rank, 21 MB a stream, never the 61,440-byte feature row" {
    const c = flash;
    const r0 = try shard.Share.of(&target.flash, 0, 2);
    try std.testing.expectEqual(@as(u64, 61_440), featureTokenBytes(&c));
    try std.testing.expectEqual(@as(u64, 5_120), ringTokenBytes(&c, r0));
    try std.testing.expectEqual(@as(u64, 4112), ringRows(&c, 16));
    try std.testing.expectEqual(@as(u64, 21_053_440), ringBytes(&c, r0, 16));
    // the target's KV for the same rows is 76,800 B a token: the ring is 1/15 of it and does not grow past the window
    const full = try shard.Share.of(&target.flash, 0, 1);
    try std.testing.expectEqual(@as(u64, 10_240), ringTokenBytes(&c, full));
}

test "scratch splits into a fixed absorb chunk and a part per stream; one stream costs about 2.0 GiB a rank" {
    const c = flash;
    const r0 = try shard.Share.of(&target.flash, 0, 2);
    const one = cost(&c, r0, 16, 2048, false);
    try std.testing.expect(one.per_lane > ringBytes(&c, r0, 16));
    // an absorb chunk of 2048 rows: 61,440 + 2 x 12,288 bytes a row; lanes add only their own part
    try std.testing.expectEqual(@as(u64, 2048 * (61_440 + 24_576)), absorbScratch(&c, 2048));
    try std.testing.expectEqual(scratchBytes(&c, r0, 2048, 1) + 3 * laneScratch(&c, r0), scratchBytes(&c, r0, 2048, 4));
    // weights 1.34 GiB + 0.5 GiB module/graph estimate + 0.16 GiB absorb chunk + the stream: about 2.0 GiB a rank
    const all = one.weights + one.runtime + one.per_lane;
    try std.testing.expect(all > 1_900_000_000 * 1 and all < 2_300_000_000 * 1);
}

// ---- what a verify window costs on this MoE (an expectation, not a measurement) ----------------------------------

/// Expected distinct experts a window of `rows` independent rows touches in one layer.
pub fn distinctExperts(experts: u32, topk: u32, rows: u32) f64 {
    const e: f64 = @floatFromInt(experts);
    const keep = 1.0 - @as(f64, @floatFromInt(topk)) / e;
    return e * (1.0 - std.math.pow(f64, keep, @floatFromInt(rows)));
}

/// Window bytes against a one-row step: (other + experts x distinct(rows) / distinct(1)) / (other + experts).
pub fn windowCostRatio(other_bytes: f64, expert_bytes_one_row: f64, experts: u32, topk: u32, rows: u32) f64 {
    const grow = distinctExperts(experts, topk, rows) / distinctExperts(experts, topk, 1);
    return (other_bytes + expert_bytes_one_row * grow) / (other_bytes + expert_bytes_one_row);
}

test "window cost: one row costs one step; 8 rows read about 7x the experts and about 3.4x the bytes (independent routing)" {
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), distinctExperts(128, 4, 1), 1e-12);
    try std.testing.expect(distinctExperts(128, 4, 8) > 28 and distinctExperts(128, 4, 8) < 29);
    // engine-port-plan section 5: about 14.5 GB a token, 5.7 GB of it routed experts
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), windowCostRatio(8.8, 5.7, 128, 4, 1), 1e-12);
    const r8 = windowCostRatio(8.8, 5.7, 128, 4, 8);
    try std.testing.expect(r8 > 3.3 and r8 < 3.5);
    var last: f64 = 0;
    for (1..17) |rows| {
        const r = windowCostRatio(8.8, 5.7, 128, 4, @intCast(rows));
        try std.testing.expect(r > last);
        last = r;
    }
}
