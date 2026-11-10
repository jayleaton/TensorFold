//! quantization_config rules: ModelOpt layers and ignore globs, GPTQ dynamic rules, the INT4-AutoRound check.
const std = @import("std");
const ngram = @import("cuda_ngram.zig");
const config = @import("cuda_config.zig");
const Config = config.Config;
const Why = config.Why;
const max_layers = config.max_layers;

/// ModelOpt's per-module algorithms in quantized_layers.
pub const Algo = enum { nvfp4, fp8, fp8_pb_wo, other };

pub const Spec = struct { bits: u32, kind: []const u8, group: ?u32, dynamic: bool };
pub const Group = struct { name: []const u8, targets: []const []const u8, weights: ?Spec, inputs: ?Spec };
pub const QuantizedLayer = struct { name: []const u8, algo: Algo, algo_name: []const u8, group: ?u32 };

/// quantization_config of a ModelOpt export (quant_method "modelopt").
pub const ModelOpt = struct {
    algo: []const u8,
    producer: []const u8 = "",
    version: []const u8 = "",
    groups: []const Group,
    layers: []const QuantizedLayer,
    ignore: []const []const u8,

    /// The quantized_layers entry of `module` (e.g. "model.language_model.layers.3.mlp.experts").
    pub fn layer(self: ModelOpt, module: []const u8) ?QuantizedLayer {
        for (self.layers) |l| if (std.mem.eql(u8, l.name, module)) return l;
        return null;
    }

    /// Whether an ignore glob matches `module` (fnmatch '*' and '?', as ModelOpt writes them).
    pub fn ignored(self: ModelOpt, module: []const u8) bool {
        for (self.ignore) |g| if (glob(g, module)) return true;
        return false;
    }
};

pub fn glob(pattern: []const u8, name: []const u8) bool {
    var p: usize = 0;
    var n: usize = 0;
    var star: ?usize = null;
    var mark: usize = 0;
    while (n < name.len) {
        if (p < pattern.len and (pattern[p] == '?' or pattern[p] == name[n])) {
            p += 1;
            n += 1;
        } else if (p < pattern.len and pattern[p] == '*') {
            star = p;
            mark = n;
            p += 1;
        } else if (star) |s| {
            p = s + 1;
            mark += 1;
            n = mark;
        } else return false;
    }
    while (p < pattern.len and pattern[p] == '*') p += 1;
    return p == pattern.len;
}

/// One `dynamic` GPTQ/AutoRound rule: "-:regex" leaves matches unquantized, "+:regex" overrides them (re.match).
pub const Rule = struct { skip: bool, pattern: []const u8, bits: ?u32 = null };

/// quantization_config of a GPTQ export (quant_method "gptq": AutoRound's and GPTQModel's).
pub const Gptq = struct {
    bits: u32,
    group: u32,
    sym: bool,
    desc_act: bool,
    lm_head: bool,
    format: []const u8 = "",
    dynamic: []const Rule = &.{},

    /// The bits `module` is quantized at (its last matching "+" rule's, else the base), null when unquantized.
    pub fn bitsOf(self: Gptq, module: []const u8) ?u32 {
        if (!self.quantized(module)) return null;
        var bits = self.bits;
        for (self.dynamic) |r| if (!r.skip and regexMatch(r.pattern, module)) if (r.bits) |b| {
            bits = b;
        };
        return bits;
    }

    /// Whether `module` is quantized: no "-" rule matches it (the lm_head only when `lm_head`).
    pub fn quantized(self: Gptq, module: []const u8) bool {
        for (self.dynamic) |r| if (r.skip and regexMatch(r.pattern, module)) return false;
        if (std.mem.eql(u8, module, "lm_head") and !self.lm_head) {
            for (self.dynamic) |r| if (!r.skip and regexMatch(r.pattern, module)) return true;
            return false;
        }
        return true;
    }
};

/// Python's re.match for GPTQ rules' subset: literals, '.', '\x', '*' and '+', final '$'; prefix match without '$'.
pub fn regexMatch(pattern: []const u8, text: []const u8) bool {
    return reAt(pattern, 0, text, 0);
}

fn reAtom(pattern: []const u8, p: usize) struct { len: usize, lit: ?u8 } {
    if (pattern[p] == '\\' and p + 1 < pattern.len) return .{ .len = 2, .lit = pattern[p + 1] };
    if (pattern[p] == '.') return .{ .len = 1, .lit = null };
    return .{ .len = 1, .lit = pattern[p] };
}

fn reAt(pattern: []const u8, p: usize, text: []const u8, t: usize) bool {
    if (p == pattern.len) return true;
    if (pattern[p] == '$' and p + 1 == pattern.len) return t == text.len;
    const atom = reAtom(pattern, p);
    const next = p + atom.len;
    const one = struct {
        fn f(lit: ?u8, c: u8) bool {
            return if (lit) |l| l == c else true;
        }
    }.f;
    if (next < pattern.len and (pattern[next] == '*' or pattern[next] == '+')) {
        const least: usize = if (pattern[next] == '+') 1 else 0;
        var n: usize = 0;
        while (t + n < text.len and one(atom.lit, text[t + n])) n += 1;
        var k: usize = n + 1;
        while (k > least) {
            k -= 1;
            if (reAt(pattern, next + 1, text, t + k)) return true;
        }
        return false;
    }
    if (t < text.len and one(atom.lit, text[t])) return reAt(pattern, next, text, t + 1);
    return false;
}

pub const int4ar_group = 128;

/// The INT4-AutoRound checkpoint (GPTQ int4 experts and lm_head, block-FP8 dense, bf16 MTP): what we serve of GPTQ.
pub fn checkGptqImpl(self: Config, why: *Why) !void {
    const g = self.gptq.?;
    if (self.layers == 0 or self.layers > max_layers) {
        why.set("{d} layers; the engine holds up to {d}", .{ self.layers, max_layers });
        return error.UnsupportedModel;
    }
    if (g.bits != 4 or g.group != int4ar_group or !g.sym or g.desc_act) {
        why.set("GPTQ {d}-bit, group {d}, sym {}, desc_act {}; the engine reads 4-bit symmetric groups of 128 without act order", .{ g.bits, g.group, g.sym, g.desc_act });
        return error.UnsupportedQuantization;
    }
    var buf: [128]u8 = undefined;
    const expert = std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.experts.0.gate_proj", .{self.layers - 1}) catch unreachable;
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
        var eb: [160]u8 = undefined;
        const name = std.fmt.bufPrint(&eb, "model.language_model.layers.0.mlp.experts.0.{s}", .{proj}) catch unreachable;
        if (g.bitsOf(name) != 4 or g.bitsOf(expert) != 4) {
            why.set("GPTQ quantizes the routed experts at {?d} bits; the engine reads 4-bit routed experts", .{g.bitsOf(name)});
            return error.UnsupportedQuantization;
        }
    }
    if (g.bitsOf("lm_head") != 4) {
        why.set("GPTQ's lm_head is {?d}-bit (null: unquantized); the engine reads the INT4-AutoRound 4-bit head", .{g.bitsOf("lm_head")});
        return error.UnsupportedQuantization;
    }
    for ([_][]const u8{ "model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.3.self_attn.q_proj", "model.language_model.layers.0.mlp.shared_expert.gate_proj", "model.language_model.layers.0.mlp.gate", "model.language_model.embed_tokens" }) |m| if (g.quantized(m)) {
        why.set("GPTQ quantizes {s}; the engine reads only the routed experts and lm_head as GPTQ int4", .{m});
        return error.UnsupportedQuantization;
    };
    if (self.layers == 0 or self.layers > max_layers) {
        why.set("{d} layers; the engine holds up to {d}", .{ self.layers, max_layers });
        return error.UnsupportedModel;
    }
    if (self.ngram_size < 2 or self.ngram_size > ngram.max_n or self.ngramHeads() > ngram.max_heads or self.ple_dim % self.ngramHeads() != 0) {
        why.set("n-gram size {d} with {d} heads a gram does not tile {d} dims", .{ self.ngram_size, self.heads_per_ngram, self.ple_dim });
        return error.UnsupportedModel;
    }
    if (self.moe_width % int4ar_group != 0 or self.hidden % int4ar_group != 0) {
        why.set("expert width {d} / hidden {d} are not whole groups of {d}", .{ self.moe_width, self.hidden, int4ar_group });
        return error.UnsupportedModel;
    }
}
