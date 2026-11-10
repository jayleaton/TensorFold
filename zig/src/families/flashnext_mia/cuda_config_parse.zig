//! config.json and generation_config.json parsed into a Config, as Python's Config.read reads them.
const std = @import("std");
const config = @import("cuda_config.zig");
const quant = @import("cuda_config_quant.zig");
const Config = config.Config;
const Why = config.Why;
const Options = config.Options;
const Rope = config.Rope;
const Yarn = config.Yarn;
const max_ple = config.max_ple;
const max_eos = config.max_eos;
const max_layers = config.max_layers;
const max_mtp_layers = config.max_mtp_layers;
const Algo = quant.Algo;
const Spec = quant.Spec;
const Group = quant.Group;
const QuantizedLayer = quant.QuantizedLayer;
const ModelOpt = quant.ModelOpt;
const Gptq = quant.Gptq;
const Rule = quant.Rule;

const Value = std.json.Value;
const Obj = std.json.ObjectMap;

fn field(o: Obj, key: []const u8) ?Value {
    const v = o.get(key) orelse return null;
    return if (v == .null) null else v;
}

fn number(v: Value) !f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch error.BadConfig,
        else => error.BadConfig,
    };
}

fn integer(v: Value) !i64 {
    return switch (v) {
        .integer => |i| i,
        .float => |f| if (@trunc(f) == f) @intFromFloat(f) else error.BadConfig,
        else => error.BadConfig,
    };
}

/// A required count (> 0 where `positive`).
fn need(o: Obj, key: []const u8, why: *Why) !u32 {
    const v = field(o, key) orelse {
        why.set("config.json has no {s}", .{key});
        return error.BadConfig;
    };
    const i = integer(v) catch {
        why.set("config.json's {s} is not an integer", .{key});
        return error.BadConfig;
    };
    if (i < 0 or i > std.math.maxInt(u32)) {
        why.set("config.json's {s} is {d}", .{ key, i });
        return error.BadConfig;
    }
    return @intCast(i);
}

fn opt(o: Obj, key: []const u8, default: u32, why: *Why) !u32 {
    if (field(o, key) == null) return default;
    return need(o, key, why);
}

fn optWide(o: Obj, key: []const u8, default: u64) !u64 {
    const v = field(o, key) orelse return default;
    const i = try integer(v);
    if (i < 0) return error.BadConfig;
    return @intCast(i);
}

fn optFloat(o: Obj, key: []const u8, default: f64) !f64 {
    return number(field(o, key) orelse return default);
}

fn optBool(o: Obj, key: []const u8, default: bool) bool {
    const v = field(o, key) orelse return default;
    return if (v == .bool) v.bool else default;
}

fn object(o: Obj, key: []const u8) ?Obj {
    const v = field(o, key) orelse return null;
    return if (v == .object) v.object else null;
}

fn str(a: std.mem.Allocator, o: Obj, key: []const u8) ![]const u8 {
    const v = field(o, key) orelse return "";
    return if (v == .string) try a.dupe(u8, v.string) else "";
}

/// An id or a list of ids (eos_token_id's two spellings).
fn ids(v: ?Value, out: *std.ArrayList(i64), a: std.mem.Allocator) !void {
    const x = v orelse return;
    switch (x) {
        .array => |arr| for (arr.items) |e| try out.append(a, try integer(e)),
        else => try out.append(a, try integer(x)),
    }
}

fn algoOf(name: []const u8) Algo {
    if (std.ascii.eqlIgnoreCase(name, "NVFP4")) return .nvfp4;
    if (std.ascii.eqlIgnoreCase(name, "FP8")) return .fp8;
    if (std.ascii.eqlIgnoreCase(name, "FP8_PB_WO")) return .fp8_pb_wo;
    return .other;
}

fn spec(a: std.mem.Allocator, o: ?Obj) !?Spec {
    const s = o orelse return null;
    return .{
        .bits = @intCast(try optWide(s, "num_bits", 0)),
        .kind = try str(a, s, "type"),
        .group = if (field(s, "group_size")) |g| @as(u32, @intCast(try integer(g))) else null,
        .dynamic = optBool(s, "dynamic", false),
    };
}

fn modelOpt(a: std.mem.Allocator, q: Obj) !ModelOpt {
    var groups: std.ArrayList(Group) = .empty;
    if (object(q, "config_groups")) |cg| {
        var it = cg.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.* != .object) continue;
            const g = kv.value_ptr.object;
            var targets: std.ArrayList([]const u8) = .empty;
            if (field(g, "targets")) |t| if (t == .array) for (t.array.items) |x| if (x == .string) try targets.append(a, try a.dupe(u8, x.string));
            try groups.append(a, .{ .name = try a.dupe(u8, kv.key_ptr.*), .targets = targets.items, .weights = try spec(a, object(g, "weights")), .inputs = try spec(a, object(g, "input_activations")) });
        }
    }
    var layers: std.ArrayList(QuantizedLayer) = .empty;
    if (object(q, "quantized_layers")) |ql| {
        var it = ql.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.* != .object) return error.BadConfig;
            const l = kv.value_ptr.object;
            const name = try str(a, l, "quant_algo");
            try layers.append(a, .{ .name = try a.dupe(u8, kv.key_ptr.*), .algo = algoOf(name), .algo_name = name, .group = if (field(l, "group_size")) |g| @as(u32, @intCast(try integer(g))) else null });
        }
    }
    var ignore: std.ArrayList([]const u8) = .empty;
    if (field(q, "ignore")) |v| if (v == .array) for (v.array.items) |x| if (x == .string) try ignore.append(a, try a.dupe(u8, x.string));
    var producer: []const u8 = "";
    var version: []const u8 = "";
    if (object(q, "producer")) |p| {
        producer = try str(a, p, "name");
        version = try str(a, p, "version");
    }
    return .{ .algo = try str(a, q, "quant_algo"), .producer = producer, .version = version, .groups = groups.items, .layers = layers.items, .ignore = ignore.items };
}

fn gptqOf(a: std.mem.Allocator, q: Obj, why: *Why) !Gptq {
    var rules: std.ArrayList(Rule) = .empty;
    if (object(q, "dynamic")) |d| {
        var it = d.iterator();
        while (it.next()) |kv| {
            const key = kv.key_ptr.*;
            if (key.len < 2 or key[1] != ':' or (key[0] != '-' and key[0] != '+')) {
                why.set("GPTQ dynamic rule {s}: expected \"-:regex\" or \"+:regex\"", .{key});
                return error.BadConfig;
            }
            var bits: ?u32 = null;
            if (kv.value_ptr.* == .object) if (field(kv.value_ptr.object, "bits")) |b| {
                bits = @intCast(try integer(b));
            };
            try rules.append(a, .{ .skip = key[0] == '-', .pattern = try a.dupe(u8, key[2..]), .bits = bits });
        }
    }
    return .{
        .bits = @intCast(try optWide(q, "bits", 4)),
        .group = @intCast(try optWide(q, "group_size", 128)),
        .sym = optBool(q, "sym", true),
        .desc_act = optBool(q, "desc_act", false),
        .lm_head = optBool(q, "lm_head", false),
        .format = try str(a, q, "checkpoint_format"),
        .dynamic = rules.items,
    };
}

fn yarnOf(r: Obj, why: *Why) !Yarn {
    const factor = optFloat(r, "factor", 0) catch 0;
    const original = optWide(r, "original_max_position_embeddings", 0) catch 0;
    if (!(factor > 0) or original == 0) {
        why.set("rope_type yarn needs factor and original_max_position_embeddings", .{});
        return error.BadConfig;
    }
    return .{
        .factor = factor,
        .original_max_position_embeddings = original,
        .beta_fast = try optFloat(r, "beta_fast", 32),
        .beta_slow = try optFloat(r, "beta_slow", 1),
        .attention_factor = if (field(r, "attention_factor")) |v| try number(v) else null,
        .truncate = optBool(r, "truncate", true),
    };
}

/// config.json's text (and generation_config.json's, when the model has one) into a Config.
pub fn parse(gpa: std.mem.Allocator, text: []const u8, generation: ?[]const u8, o: Options, why: *Why) !Config {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const root_v = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch {
        why.set("config.json is not JSON", .{});
        return error.BadConfig;
    };
    if (root_v != .object) return error.BadConfig;
    const raw = root_v.object;
    const t = object(raw, "text_config") orelse raw;
    const rope_o = object(t, "rope_parameters");
    const model_type = try str(a, raw, "model_type");

    const hidden = try need(t, "hidden_size", why);
    const heads = try need(t, "num_attention_heads", why);
    const head_dim = if (field(t, "head_dim") != null) try need(t, "head_dim", why) else hidden / @max(heads, 1);
    const partial = if (rope_o) |r| (if (field(r, "partial_rotary_factor")) |v| try number(v) else try optFloat(t, "partial_rotary_factor", 0.25)) else try optFloat(t, "partial_rotary_factor", 0.25);
    var rope: Rope = .{
        .theta = if (rope_o) |r| try optFloat(r, "rope_theta", 10_000_000) else 10_000_000,
        .partial = partial,
        .rotary_dim = @intFromFloat(@trunc(@as(f64, @floatFromInt(head_dim)) * partial)),
        .mrope_section = .{ 11, 11, 10 },
        .mrope_interleaved = if (rope_o) |r| optBool(r, "mrope_interleaved", true) else true,
        .max_position = try optWide(t, "max_position_embeddings", 262144),
    };
    if (rope_o) |r| {
        if (field(r, "mrope_section")) |v| {
            if (v != .array or v.array.items.len != 3) {
                why.set("rope_parameters.mrope_section must hold three sizes", .{});
                return error.BadConfig;
            }
            for (v.array.items, 0..) |x, i| rope.mrope_section[i] = @intCast(try integer(x));
        }
        if (field(r, "rope_type")) |v| if (v == .string) {
            if (std.mem.eql(u8, v.string, "yarn")) {
                rope.kind = .yarn;
                rope.yarn = try yarnOf(r, why);
            } else if (!std.mem.eql(u8, v.string, "default")) {
                why.set("rope_type {s}; the engine reads default rope (or YaRN)", .{v.string});
                return error.UnsupportedRope;
            }
        };
    }
    if (o.yarn) |y| {
        if (!(y.factor > 0) or y.original_max_position_embeddings == 0) return error.BadConfig;
        rope.kind = .yarn;
        rope.yarn = y;
    }

    var c: Config = .{
        .arena = undefined,
        .model_type = model_type,
        .hidden = hidden,
        .layers = try need(t, "num_hidden_layers", why),
        .vocab = try need(t, "vocab_size", why),
        .eps = optFloat(t, "rms_norm_eps", -1) catch -1,
        .heads = heads,
        .kv_heads = try need(t, "num_key_value_heads", why),
        .head_dim = head_dim,
        .rope = rope,
        .nk = try need(t, "linear_num_key_heads", why),
        .nv = try need(t, "linear_num_value_heads", why),
        .dk = try need(t, "linear_key_head_dim", why),
        .dv = try need(t, "linear_value_head_dim", why),
        .conv_kernel = try need(t, "linear_conv_kernel_dim", why),
        .experts = try need(t, "num_experts", why),
        .top_k = try need(t, "num_experts_per_tok", why),
        .moe_width = try need(t, "moe_intermediate_size", why),
        .shared_width = try need(t, "shared_expert_intermediate_size", why),
        .norm_topk = optBool(t, "norm_topk_prob", true),
        .streams = try opt(t, "hc_count", 4, why),
        .low = try opt(t, "hc_lowrank", 320, why),
        .index_heads = try opt(t, "indexer_n_heads", 4, why),
        .index_dim = try opt(t, "indexer_head_dim", 128, why),
        .index_kv_heads = try opt(t, "indexer_kv_heads", 1, why),
        .index_budget = try opt(t, "indexer_budget", 2048, why),
        .index_ratio = try opt(t, "indexer_compress_ratio", 4, why),
        .ple_dim = if (field(t, "ple_embed_dim") != null) try need(t, "ple_embed_dim", why) else hidden,
        .ple_kernel = try opt(t, "ple_conv_kernel_size", 4, why),
        .ngram_size = try opt(t, "ngram_size", 3, why),
        .heads_per_ngram = try opt(t, "heads_per_ngram", 8, why),
        .ngram_base = try optWide(t, "ngram_vocab_size_base", 20_000_000),
        .ngram_divisor = try optWide(t, "make_ngram_vocab_size_divisible_by", 128),
        .ngram_shards = try opt(t, "split_ngram_parts", 128, why),
        .seed = try optWide(t, "seed", 1234),
        .ple_eos = 0,
        .tie_embeddings = optBool(raw, "tie_word_embeddings", optBool(t, "tie_word_embeddings", false)),
        .group_size = 32,
        .bits = 4,
        .quant = .mlx,
        .nvfp4_group = 16,
    };
    if (c.eps <= 0) {
        why.set("config.json has no rms_norm_eps", .{});
        return error.BadConfig;
    }
    if (c.index_ratio == 0 or c.ngram_divisor == 0) return error.BadConfig;
    if (field(t, "bos_token_id")) |v| c.bos = integer(v) catch null;

    // layer types: linear_attention -> linear, every other name -> attention (Python's reading)
    const lt = field(t, "layer_types") orelse {
        why.set("config.json has no layer_types", .{});
        return error.BadConfig;
    };
    if (lt != .array or lt.array.items.len != c.layers or c.layers > max_layers) {
        why.set("layer_types lists {d} layers for num_hidden_layers {d} (at most {d})", .{ if (lt == .array) lt.array.items.len else 0, c.layers, max_layers });
        return error.BadConfig;
    }
    for (lt.array.items, 0..) |v, i| c.layer_types[i] = if (v == .string and std.mem.eql(u8, v.string, "linear_attention")) .linear else .attention;

    // ple_layer_ids are one-indexed: a sorted set of zero-indexed layers
    if (field(t, "ple_layer_ids")) |v| {
        if (v != .array) return error.BadConfig;
        for (v.array.items) |x| {
            const id = try integer(x) - 1;
            if (id < 0 or id >= c.layers) {
                why.set("ple_layer_ids names layer {d} of {d}", .{ id + 1, c.layers });
                return error.BadConfig;
            }
            const l: u32 = @intCast(id);
            if (c.pleIndex(l) != null) continue;
            if (c.ple_count == max_ple) return error.BadConfig;
            c.ple_layers[c.ple_count] = l;
            c.ple_count += 1;
        }
        std.mem.sort(u32, c.ple_layers[0..c.ple_count], {}, std.sort.asc(u32));
    }

    // end-of-reply ids: config.json's own (root, else text_config), then generation_config.json's not yet listed
    var stop: std.ArrayList(i64) = .empty;
    const teos = field(t, "eos_token_id");
    if (teos) |v| c.ple_eos = switch (v) {
        .array => |arr| if (arr.items.len > 0) try integer(arr.items[0]) else 0,
        else => try integer(v),
    };
    ids(field(raw, "eos_token_id") orelse teos, &stop, a) catch {
        why.set("config.json's eos_token_id is not an id or a list of ids", .{});
        return error.BadConfig;
    };
    if (generation) |g| {
        const gv = std.json.parseFromSliceLeaky(Value, a, g, .{}) catch {
            why.set("generation_config.json is not JSON", .{});
            return error.BadConfig;
        };
        if (gv == .object) {
            var more: std.ArrayList(i64) = .empty;
            ids(field(gv.object, "eos_token_id"), &more, a) catch {
                why.set("generation_config.json's eos_token_id is not an id or a list of ids", .{});
                return error.BadConfig;
            };
            for (more.items) |id| if (std.mem.indexOfScalar(i64, stop.items, id) == null) try stop.append(a, id);
        }
    }
    // a list may repeat an id; keep its first place
    for (stop.items) |id| {
        if (std.mem.indexOfScalar(i64, c.eos[0..c.eos_count], id) != null) continue;
        if (c.eos_count == max_eos) return error.BadConfig;
        c.eos[c.eos_count] = id;
        c.eos_count += 1;
    }
    if (c.eos_count == 0) {
        why.set("no eos_token_id in config.json or generation_config.json", .{});
        return error.BadConfig;
    }

    // quantization: "quantization" (MLX) or "quantization_config" (ModelOpt), as Python reads either
    const q = object(raw, "quantization") orelse object(raw, "quantization_config");
    if (q) |qq| {
        const method = try str(a, qq, "quant_method");
        if (method.len == 0 or std.ascii.eqlIgnoreCase(method, "mlx")) {
            c.quant = .mlx;
        } else if (std.ascii.eqlIgnoreCase(method, "modelopt")) {
            c.quant = .modelopt;
            c.modelopt = try modelOpt(a, qq);
        } else if (std.ascii.eqlIgnoreCase(method, "gptq")) {
            c.quant = .gptq;
            c.gptq = try gptqOf(a, qq, why);
        } else {
            why.set("quant_method {s}; Flash Next reads MLX or ModelOpt checkpoints", .{method});
            return error.UnsupportedQuantization;
        }
        c.group_size = try opt(qq, "group_size", 32, why);
        c.bits = try opt(qq, "bits", 4, why);
        if (object(qq, "config_groups")) |cg| if (object(cg, "group_0")) |g0| if (object(g0, "weights")) |w| {
            c.nvfp4_group = try opt(w, "group_size", 16, why);
        };
    }

    // the MTP head
    c.mtp.layers = try opt(t, "mtp_num_hidden_layers", 0, why);
    c.mtp.dedicated_embeddings = optBool(t, "mtp_use_dedicated_embeddings", false);
    if (object(t, "mtp")) |m| {
        c.mtp.layers = try opt(m, "num_hidden_layers", c.mtp.layers, why);
        c.mtp.hybrid = optBool(m, "hybrid", false);
        if (field(m, "rope_theta")) |v| c.mtp.rope_theta = try number(v);
        if (field(m, "mtp_use_hidden_state_from_layer")) |v| c.mtp.hidden_from_layer = try integer(v);
        if (field(m, "layer_types")) |v| if (v == .array) for (v.array.items, 0..) |x, i| {
            if (i >= max_mtp_layers) break;
            c.mtp.layer_types[i] = if (x == .string and std.mem.eql(u8, x.string, "linear_attention")) .linear else .attention;
        };
    }
    if (c.mtp.layers > max_mtp_layers) return error.BadConfig;
    c.arena = arena;
    return c;
}
