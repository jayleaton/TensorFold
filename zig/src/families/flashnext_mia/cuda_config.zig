//! Flash Next's dimensions for the CUDA engine: config.json read as Python's Config.read (weight_types.py) does.
const std = @import("std");
const ngram = @import("cuda_ngram.zig");
const quant = @import("cuda_config_quant.zig");
const parse_part = @import("cuda_config_parse.zig");

/// config.json model_type values this family serves (Python qwen4_exp MODEL_TYPES).
pub const model_types = [_][]const u8{ "qwen4_exp", "qwen3_8_flash_next" };
pub const max_layers = 64;
pub const max_ple = 8;
pub const max_eos = 8;
pub const max_mtp_layers = 4;

pub const LayerType = enum { linear, attention };
pub const Quant = enum { mlx, modelopt, gptq };

/// Why a check refused the checkpoint, kept for the caller's log line (tests read it).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const RopeType = enum { default, yarn };

/// YaRN as HF / vLLM name it (rope_parameters with rope_type "yarn"); the caller's override for 1M contexts.
pub const Yarn = struct {
    factor: f64,
    original_max_position_embeddings: u64,
    beta_fast: f64 = 32,
    beta_slow: f64 = 1,
    /// null: HF's default 0.1 * ln(factor) + 1
    attention_factor: ?f64 = null,
    truncate: bool = true,
};

pub const Rope = struct {
    kind: RopeType = .default,
    theta: f64,
    /// partial_rotary_factor: rotary_dim = int(head_dim * partial)
    partial: f64,
    rotary_dim: u32,
    mrope_section: [3]u32,
    mrope_interleaved: bool,
    /// text_config.max_position_embeddings (the native window)
    max_position: u64,
    /// YaRN's fields (kind == .yarn), from the checkpoint or the caller
    yarn: ?Yarn = null,
};

pub const Options = struct {
    /// replaces the checkpoint's rope scaling with YaRN (work/PLAN.md "1M context")
    yarn: ?Yarn = null,
};

/// text_config.mtp and its siblings: the multi-token-prediction head.
pub const Mtp = struct {
    layers: u32 = 0,
    layer_types: [max_mtp_layers]LayerType = @splat(.attention),
    hybrid: bool = false,
    rope_theta: ?f64 = null,
    hidden_from_layer: ?i64 = null,
    dedicated_embeddings: bool = false,
};

pub const Algo = quant.Algo;
pub const Spec = quant.Spec;
pub const Group = quant.Group;
pub const QuantizedLayer = quant.QuantizedLayer;
pub const ModelOpt = quant.ModelOpt;
pub const glob = quant.glob;
pub const Rule = quant.Rule;
pub const Gptq = quant.Gptq;
pub const regexMatch = quant.regexMatch;

/// A rank's shares of the split dimensions (Python weights.load's `replace(full, heads // world, ...)`).
pub const Shares = struct { heads: u32, kv_heads: u32, nk: u32, nv: u32, moe_width: u32, shared_width: u32 };

pub const Config = struct {
    arena: std.heap.ArenaAllocator,
    model_type: []const u8,
    hidden: u32,
    layers: u32,
    layer_types: [max_layers]LayerType = @splat(.linear),
    vocab: u32,
    eps: f64,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rope: Rope,
    nk: u32,
    nv: u32,
    dk: u32,
    dv: u32,
    conv_kernel: u32,
    experts: u32,
    top_k: u32,
    moe_width: u32,
    shared_width: u32,
    norm_topk: bool,
    /// hc_count, hc_lowrank: the hyper-connection streams and their low rank
    streams: u32,
    low: u32,
    index_heads: u32,
    index_dim: u32,
    index_kv_heads: u32,
    index_budget: u32,
    index_ratio: u32,
    /// zero-indexed decoder layers with the n-gram embedding (ple_layer_ids are one-indexed)
    ple_layers: [max_ple]u32 = @splat(0),
    ple_count: u32 = 0,
    ple_dim: u32,
    ple_kernel: u32,
    ngram_size: u32,
    heads_per_ngram: u32,
    ngram_base: u64,
    ngram_divisor: u64,
    ngram_shards: u32,
    seed: u64,
    /// the n-gram resets' EOS (text_config.eos_token_id, its first entry)
    ple_eos: i64,
    /// end-of-reply ids: config.json's (the root's, else text_config's), then generation_config.json's not yet listed
    eos: [max_eos]i64 = @splat(0),
    eos_count: u32 = 0,
    bos: ?i64 = null,
    tie_embeddings: bool,
    /// MLX's affine group (32 when the block names none, as Python reads it) and bits
    group_size: u32,
    bits: u32,
    quant: Quant,
    /// NVFP4's block (config_groups.group_0.weights.group_size, else 16)
    nvfp4_group: u32,
    modelopt: ?ModelOpt = null,
    /// a GPTQ export's settings (quant == .gptq: the INT4-AutoRound checkpoint)
    gptq: ?Gptq = null,
    mtp: Mtp = .{},

    pub fn deinit(self: *Config) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// `dir`/config.json and, when present, `dir`/generation_config.json.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(text);
        const gpath = try std.fs.path.join(gpa, &.{ dir, "generation_config.json" });
        defer gpa.free(gpath);
        const gen: ?[]u8 = std.Io.Dir.cwd().readFileAlloc(io, gpath, gpa, .limited(1 << 20)) catch |e| switch (e) {
            error.FileNotFound => null,
            else => return e,
        };
        defer if (gen) |g| gpa.free(g);
        var why: Why = .{};
        return parse(gpa, text, gen, o, &why) catch |e| {
            if (why.len > 0) std.log.err("{s}: {s}", .{ path, why.text() });
            return e;
        };
    }

    pub fn layerType(self: Config, i: usize) LayerType {
        return self.layer_types[i];
    }

    pub fn count(self: Config, kind: LayerType) u32 {
        var n: u32 = 0;
        for (self.layer_types[0..self.layers]) |k| n += @intFromBool(k == kind);
        return n;
    }

    /// 2 * nk * dk + nv * dv: the DeltaNet conv's channels (q, k, v)
    pub fn convDim(self: Config) u32 {
        return 2 * self.nk * self.dk + self.nv * self.dv;
    }

    /// index_budget / index_ratio: the indexer's top blocks
    pub fn topBlocks(self: Config) u32 {
        return self.index_budget / self.index_ratio;
    }

    pub fn pleLayers(self: *const Config) []const u32 {
        return self.ple_layers[0..self.ple_count];
    }

    /// The position of decoder layer `i` among the PLE layers (Python `cfg.ple_layers.index(i)`), or null.
    pub fn pleIndex(self: Config, i: u32) ?u32 {
        for (self.ple_layers[0..self.ple_count], 0..) |l, j| if (l == i) return @intCast(j);
        return null;
    }

    pub fn stops(self: *const Config) []const i64 {
        return self.eos[0..self.eos_count];
    }

    pub fn isEos(self: Config, token: i64) bool {
        for (self.eos[0..self.eos_count]) |e| if (e == token) return true;
        return false;
    }

    /// The n-gram row ids' options for PLE layer `ple_index` (Python Config.ngram).
    pub fn ngramOptions(self: Config, ple_index: u32) ngram.Options {
        return .{ .vocab = self.vocab, .ngram_size = self.ngram_size, .heads_per_ngram = self.heads_per_ngram, .vocab_base = self.ngram_base, .divisor = self.ngram_divisor, .seed = self.seed, .eos = self.ple_eos, .embed_dim = self.ple_dim, .ple_index = ple_index };
    }

    /// The n-gram heads: (ngram_size - 1) * heads_per_ngram.
    pub fn ngramHeads(self: Config) u32 {
        return (self.ngram_size - 1) * self.heads_per_ngram;
    }

    /// Rank shares of `world`: split dims divided; refused unless they divide with NVFP4 groups, 32-col blocks whole.
    pub fn shares(self: Config, world: u32) !Shares {
        if (world == 0) return error.BadWorld;
        for ([_]u32{ self.heads, self.kv_heads, self.nk, self.nv, self.moe_width, self.shared_width }) |d| if (d % world != 0) return error.WorldDoesNotDivide;
        const w = self.moe_width / world;
        const s = self.shared_width / world;
        if (w % 32 != 0 or w % self.nvfp4_group != 0 or s % 64 != 0) return error.WorldDoesNotDivide;
        if (self.ngramHeads() % world != 0) return error.WorldDoesNotDivide;
        return .{ .heads = self.heads / world, .kv_heads = self.kv_heads / world, .nk = self.nk / world, .nv = self.nv / world, .moe_width = w, .shared_width = s };
    }

    pub fn checkGptq(self: Config, why: *Why) !void {
        return checkGptqImpl(self, why);
    }

    /// The INT4-AutoRound format (GPTQ experts and head, block-FP8 dense linears).
    pub fn int4ar(self: Config) bool {
        return self.quant == .gptq;
    }

    /// Served: the ModelOpt NVFP4 export (NVFP4 experts, FP8_PB_WO MTP, FP8 n-gram, bf16 rest); else refused with why.
    pub fn check(self: Config, why: *Why) !void {
        var known = false;
        for (model_types) |m| known = known or std.mem.eql(u8, m, self.model_type);
        if (!known) {
            why.set("model_type {s} is not Flash Next's (qwen4_exp, qwen3_8_flash_next)", .{self.model_type});
            return error.UnsupportedModel;
        }
        if (self.quant == .gptq) return self.checkGptq(why);
        if (self.quant != .modelopt) {
            why.set("the CUDA Flash Next engine reads the ModelOpt NVFP4 export; config.json's quantization is {t}", .{self.quant});
            return error.UnsupportedQuantization;
        }
        const mo = self.modelopt.?;
        if (!std.mem.eql(u8, mo.algo, "MIXED_PRECISION") and !std.mem.eql(u8, mo.algo, "NVFP4")) {
            why.set("ModelOpt quant_algo {s}; the engine reads MIXED_PRECISION (NVFP4 experts) exports", .{mo.algo});
            return error.UnsupportedQuantization;
        }
        if (self.nvfp4_group != 16) {
            why.set("NVFP4 group size {d}; the expert kernels read blocks of 16", .{self.nvfp4_group});
            return error.UnsupportedQuantization;
        }
        for (mo.layers) |l| {
            const experts = std.mem.endsWith(u8, l.name, ".mlp.experts");
            const ok = if (std.mem.startsWith(u8, l.name, "mtp.") and experts)
                l.algo == .fp8_pb_wo and (l.group orelse 128) == 128
            else if (experts)
                l.algo == .nvfp4 and (l.group orelse 16) == 16
            else if (std.mem.endsWith(u8, l.name, ".ngram_embedding"))
                l.algo == .fp8
            else
                false;
            if (!ok) {
                why.set("ModelOpt quantizes {s} as {s}; the engine reads NVFP4 routed experts, FP8_PB_WO MTP experts and an FP8 n-gram table, the rest bf16", .{ l.name, l.algo_name });
                return error.UnsupportedQuantization;
            }
        }
        if (self.layers == 0 or self.layers > max_layers) {
            why.set("{d} layers; the engine holds up to {d}", .{ self.layers, max_layers });
            return error.UnsupportedModel;
        }
        if (self.ngram_size < 2 or self.ngram_size > ngram.max_n or self.ngramHeads() > ngram.max_heads or self.ple_dim % self.ngramHeads() != 0) {
            why.set("n-gram size {d} with {d} heads a gram does not tile {d} dims", .{ self.ngram_size, self.heads_per_ngram, self.ple_dim });
            return error.UnsupportedModel;
        }
    }
};

pub const int4ar_group = quant.int4ar_group;
const checkGptqImpl = quant.checkGptqImpl;

pub const parse = parse_part.parse;

// ---- tests: the nvidia/Qwen3.8-Flash-Next-NVFP4 snapshot's config.json and generation_config.json ----------------

const fixture = @embedFile("fixtures_cuda_config.json");
const fixture_generation = @embedFile("fixtures_cuda_generation_config.json");

fn testConfig(o: Options) !Config {
    var why: Why = .{};
    return parse(std.testing.allocator, fixture, fixture_generation, o, &why);
}

test "the NVFP4 checkpoint's config.json reads as Python's Config.read" {
    var c = try testConfig(.{});
    defer c.deinit();
    try std.testing.expectEqualStrings("qwen4_exp", c.model_type);
    try std.testing.expectEqual(@as(u32, 2560), c.hidden);
    try std.testing.expectEqual(@as(u32, 48), c.layers);
    try std.testing.expectEqual(@as(u32, 248320), c.vocab);
    try std.testing.expectEqual(@as(f64, 1e-6), c.eps);
    try std.testing.expectEqual(@as(u32, 24), c.heads);
    try std.testing.expectEqual(@as(u32, 2), c.kv_heads);
    try std.testing.expectEqual(@as(u32, 256), c.head_dim);
    try std.testing.expectEqual(@as(f64, 1e7), c.rope.theta);
    try std.testing.expectEqual(@as(u32, 64), c.rope.rotary_dim);
    try std.testing.expectEqual([3]u32{ 11, 11, 10 }, c.rope.mrope_section);
    try std.testing.expect(c.rope.mrope_interleaved);
    try std.testing.expectEqual(RopeType.default, c.rope.kind);
    try std.testing.expectEqual(@as(u64, 262144), c.rope.max_position);
    try std.testing.expectEqual(@as(u32, 16), c.nk);
    try std.testing.expectEqual(@as(u32, 48), c.nv);
    try std.testing.expectEqual(@as(u32, 128), c.dk);
    try std.testing.expectEqual(@as(u32, 128), c.dv);
    try std.testing.expectEqual(@as(u32, 4), c.conv_kernel);
    try std.testing.expectEqual(@as(u32, 10240), c.convDim());
    try std.testing.expectEqual(@as(u32, 512), c.experts);
    try std.testing.expectEqual(@as(u32, 10), c.top_k);
    try std.testing.expectEqual(@as(u32, 640), c.moe_width);
    try std.testing.expectEqual(@as(u32, 640), c.shared_width);
    try std.testing.expectEqual(@as(u32, 4), c.streams);
    try std.testing.expectEqual(@as(u32, 320), c.low);
    try std.testing.expectEqual(@as(u32, 4), c.index_heads);
    try std.testing.expectEqual(@as(u32, 128), c.index_dim);
    try std.testing.expectEqual(@as(u32, 2048), c.index_budget);
    try std.testing.expectEqual(@as(u32, 4), c.index_ratio);
    try std.testing.expectEqual(@as(u32, 512), c.topBlocks());
    try std.testing.expectEqualSlices(u32, &.{1}, c.pleLayers());
    try std.testing.expectEqual(@as(?u32, 0), c.pleIndex(1));
    try std.testing.expectEqual(@as(?u32, null), c.pleIndex(2));
    try std.testing.expectEqual(@as(u32, 2560), c.ple_dim);
    try std.testing.expectEqual(@as(u32, 4), c.ple_kernel);
    try std.testing.expectEqual(@as(u32, 3), c.ngram_size);
    try std.testing.expectEqual(@as(u32, 8), c.heads_per_ngram);
    try std.testing.expectEqual(@as(u32, 16), c.ngramHeads());
    try std.testing.expectEqual(@as(u64, 20_000_000), c.ngram_base);
    try std.testing.expectEqual(@as(u64, 128), c.ngram_divisor);
    try std.testing.expectEqual(@as(u32, 128), c.ngram_shards);
    try std.testing.expectEqual(@as(u64, 1234), c.seed);
    try std.testing.expectEqual(@as(i64, 248044), c.ple_eos);
    // config.json names <|endoftext|> only (text_config); generation_config.json adds <|im_end|>
    try std.testing.expectEqualSlices(i64, &.{ 248044, 248046 }, c.stops());
    try std.testing.expect(c.isEos(248046));
    try std.testing.expectEqual(@as(u32, 32), c.group_size);
    try std.testing.expectEqual(@as(u32, 4), c.bits);
    try std.testing.expectEqual(Quant.modelopt, c.quant);
    try std.testing.expectEqual(@as(u32, 16), c.nvfp4_group);
    try std.testing.expectEqual(@as(u32, 36), c.count(.linear));
    try std.testing.expectEqual(@as(u32, 12), c.count(.attention));
    for (0..48) |i| try std.testing.expectEqual(if (i % 4 == 3) LayerType.attention else LayerType.linear, c.layerType(i));
    try std.testing.expectEqual(@as(u32, 1), c.mtp.layers);
    try std.testing.expect(c.mtp.hybrid);
    try std.testing.expectEqual(LayerType.attention, c.mtp.layer_types[0]);
    try std.testing.expectEqual(@as(?f64, 1e7), c.mtp.rope_theta);
    try std.testing.expectEqual(@as(?i64, null), c.mtp.hidden_from_layer);
    try std.testing.expect(!c.mtp.dedicated_embeddings);
    try std.testing.expect(!c.tie_embeddings);
    var why: Why = .{};
    try c.check(&why);
}

test "ModelOpt's quantization_config: groups, quantized layers and ignore globs" {
    var c = try testConfig(.{});
    defer c.deinit();
    const mo = c.modelopt.?;
    try std.testing.expectEqualStrings("MIXED_PRECISION", mo.algo);
    try std.testing.expectEqualStrings("modelopt", mo.producer);
    try std.testing.expectEqual(@as(usize, 3), mo.groups.len);
    try std.testing.expectEqual(@as(usize, 50), mo.layers.len);
    const l3 = mo.layer("model.language_model.layers.3.mlp.experts").?;
    try std.testing.expectEqual(Algo.nvfp4, l3.algo);
    try std.testing.expectEqual(@as(?u32, 16), l3.group);
    const m = mo.layer("mtp.layers.0.mlp.experts").?;
    try std.testing.expectEqual(Algo.fp8_pb_wo, m.algo);
    try std.testing.expectEqual(@as(?u32, 128), m.group);
    try std.testing.expectEqual(Algo.fp8, mo.layer("model.language_model.layers.1.ple.ple_embedding.ngram_embedding").?.algo);
    try std.testing.expectEqual(@as(usize, 292), mo.ignore.len);
    try std.testing.expect(mo.ignored("lm_head"));
    try std.testing.expect(mo.ignored("model.language_model.hyper_connection_mixer.input_mix_weight_up"));
    try std.testing.expect(mo.ignored("model.visual.blocks.3.attn.qkv"));
    try std.testing.expect(mo.ignored("model.language_model.layers.9.mlp.shared_expert_gate"));
    try std.testing.expect(!mo.ignored("model.language_model.layers.9.mlp.experts"));
    var g1: ?Group = null;
    for (mo.groups) |g| if (std.mem.eql(u8, g.name, "group_1")) {
        g1 = g;
    };
    try std.testing.expectEqual(@as(u32, 8), g1.?.weights.?.bits);
    try std.testing.expectEqual(@as(?u32, 128), g1.?.weights.?.group);
    try std.testing.expect(g1.?.inputs.?.dynamic);
    try std.testing.expectEqualStrings("mtp.layers.0.mlp.experts", g1.?.targets[0]);
}

test "a YaRN override and the alias model_type" {
    var c = try testConfig(.{ .yarn = .{ .factor = 4, .original_max_position_embeddings = 262144 } });
    defer c.deinit();
    try std.testing.expectEqual(RopeType.yarn, c.rope.kind);
    try std.testing.expectEqual(@as(f64, 4), c.rope.yarn.?.factor);
    try std.testing.expectEqual(@as(u64, 262144), c.rope.yarn.?.original_max_position_embeddings);
    try std.testing.expectEqual(@as(f64, 32), c.rope.yarn.?.beta_fast);
    try std.testing.expectEqual(@as(?f64, null), c.rope.yarn.?.attention_factor);

    const alias =
        \\{"model_type": "qwen3_8_flash_next", "eos_token_id": [248046, 248044],
        \\ "text_config": {"hidden_size": 64, "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
        \\  "vocab_size": 1000, "rms_norm_eps": 1e-6, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        \\  "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
        \\  "linear_conv_kernel_dim": 4, "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
        \\  "shared_expert_intermediate_size": 128, "eos_token_id": 7,
        \\  "rope_parameters": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 4096,
        \\   "rope_theta": 1000000, "partial_rotary_factor": 0.5}},
        \\ "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        \\  "quantized_layers": {"model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16}}}}
    ;
    var why: Why = .{};
    var b = try parse(std.testing.allocator, alias, "{\"eos_token_id\": 9}", .{}, &why);
    defer b.deinit();
    try std.testing.expectEqualStrings("qwen3_8_flash_next", b.model_type);
    try std.testing.expectEqualSlices(i64, &.{ 248046, 248044, 9 }, b.stops());
    try std.testing.expectEqual(@as(i64, 7), b.ple_eos);
    try std.testing.expectEqual(RopeType.yarn, b.rope.kind);
    try std.testing.expectEqual(@as(f64, 2), b.rope.yarn.?.factor);
    try std.testing.expectEqual(@as(u32, 8), b.rope.rotary_dim);
    try std.testing.expectEqual(@as(u32, 64), b.ple_dim);
    try std.testing.expectEqual(@as(u32, 0), b.ple_count);
    try b.check(&why);
    const s = try b.shares(2);
    try std.testing.expectEqual(@as(u32, 2), s.heads);
    try std.testing.expectEqual(@as(u32, 32), s.moe_width);
    try std.testing.expectEqual(@as(u32, 64), s.shared_width);
}

test "check refuses another quantization, and shares refuse a world that does not divide" {
    var c = try testConfig(.{});
    defer c.deinit();
    const s = try c.shares(2);
    try std.testing.expectEqual(Shares{ .heads = 12, .kv_heads = 1, .nk = 8, .nv = 24, .moe_width = 320, .shared_width = 320 }, s);
    try std.testing.expectError(error.WorldDoesNotDivide, c.shares(3));
    c.nvfp4_group = 32;
    var why: Why = .{};
    try std.testing.expectError(error.UnsupportedQuantization, c.check(&why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "group size 32") != null);
    c.nvfp4_group = 16;
    c.quant = .mlx;
    try std.testing.expectError(error.UnsupportedQuantization, c.check(&why));
}

test "glob" {
    try std.testing.expect(glob("a*", "abc"));
    try std.testing.expect(glob("*.mlp.experts", "x.layers.1.mlp.experts"));
    try std.testing.expect(!glob("*.mlp.experts", "x.layers.1.mlp.experts.0"));
    try std.testing.expect(glob("a?c", "abc"));
    try std.testing.expect(glob("*", ""));
    try std.testing.expect(!glob("abc", "ab"));
    try std.testing.expect(glob("m*l*", "model.language"));
}

test "the INT4-AutoRound checkpoint's config.json: GPTQ int4 g128 experts and head, top-5 routing" {
    var why: Why = .{};
    var c = try parse(std.testing.allocator, @embedFile("fixtures_cuda_config_int4ar.json"), "{\"eos_token_id\": [248046, 248044]}", .{}, &why);
    defer c.deinit();
    try std.testing.expectEqual(Quant.gptq, c.quant);
    try std.testing.expect(c.int4ar());
    try std.testing.expectEqual(@as(u32, 5), c.top_k);
    try std.testing.expectEqual(@as(u32, 512), c.experts);
    try std.testing.expectEqual(@as(u32, 640), c.moe_width);
    try std.testing.expectEqual(@as(u32, 1280), c.shared_width);
    const g = c.gptq.?;
    try std.testing.expectEqual(@as(u32, 4), g.bits);
    try std.testing.expectEqual(@as(u32, 128), g.group);
    try std.testing.expect(g.sym and !g.desc_act and g.lm_head);
    try std.testing.expectEqual(@as(usize, 11), g.dynamic.len);
    try std.testing.expect(g.quantized("model.language_model.layers.7.mlp.experts.300.down_proj"));
    try std.testing.expect(g.quantized("lm_head"));
    for ([_][]const u8{ "model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.3.self_attn.o_proj", "model.language_model.layers.3.mlp.shared_expert.up_proj", "model.language_model.layers.3.mlp.shared_expert_gate", "model.language_model.layers.3.mlp.gate", "model.language_model.layers.1.ple.key_proj", "model.language_model.embed_tokens", "model.language_model.layers.48.mlp.experts.0.gate_proj", "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down", "model.visual.blocks.0.attn.qkv", "mtp.fc_hidden" }) |m| try std.testing.expect(!g.quantized(m));
    try c.check(&why);
    try std.testing.expectEqualSlices(i64, &.{ 248044, 248046 }, c.stops());
    const s2 = try c.shares(2);
    try std.testing.expectEqual(@as(u32, 320), s2.moe_width);
    try std.testing.expectEqual(@as(u32, 640), s2.shared_width);
}

test "regexMatch: re.match of the GPTQ rules' subset" {
    try std.testing.expect(regexMatch(".*lm_head$", "lm_head"));
    try std.testing.expect(!regexMatch(".*lm_head$", "lm_head.x"));
    try std.testing.expect(regexMatch(".*\\.gate$", "a.mlp.gate"));
    try std.testing.expect(!regexMatch(".*\\.gate$", "a.mlp.shared_expert_gate"));
    try std.testing.expect(regexMatch(".*layers\\.48\\..*", "model.layers.48.mlp"));
    try std.testing.expect(!regexMatch(".*layers\\.48\\..*", "model.layers.4.mlp"));
    try std.testing.expect(regexMatch("abc", "abcdef"));
    try std.testing.expect(!regexMatch("abd", "abcdef"));
    try std.testing.expect(regexMatch("a+b", "aaab"));
    try std.testing.expect(!regexMatch("a+b", "b"));
}

test "GPTQ checks: zero layers, a head that is not 4-bit, 8-bit experts" {
    var why: Why = .{};
    const base = @embedFile("fixtures_cuda_config_int4ar.json");
    var c = try parse(std.testing.allocator, base, null, .{}, &why);
    defer c.deinit();
    var g = c.gptq.?;
    // the head unquantized
    g.lm_head = false;
    var rules: [16]Rule = undefined;
    var n: usize = 0;
    for (c.gptq.?.dynamic) |r| if (r.skip) {
        rules[n] = r;
        n += 1;
    };
    g.dynamic = rules[0..n];
    var c2 = c;
    c2.gptq = g;
    try std.testing.expectError(error.UnsupportedQuantization, c2.check(&why));
    // an 8-bit head rule
    var rules8: [16]Rule = undefined;
    for (c.gptq.?.dynamic, 0..) |r, i| rules8[i] = if (!r.skip) .{ .skip = false, .pattern = r.pattern, .bits = 8 } else r;
    g = c.gptq.?;
    g.dynamic = rules8[0..c.gptq.?.dynamic.len];
    c2.gptq = g;
    try std.testing.expectError(error.UnsupportedQuantization, c2.check(&why));
    // 8-bit base: experts refused
    g = c.gptq.?;
    g.bits = 8;
    c2.gptq = g;
    try std.testing.expectError(error.UnsupportedQuantization, c2.check(&why));
    // zero layers: refused, no underflow
    c2 = c;
    c2.layers = 0;
    try std.testing.expectError(error.UnsupportedModel, c2.check(&why));
}
