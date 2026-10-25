//! The draft's tensor names (VERIFIED from the pack's header): `language_model.model.dspark.` and its layers.
const std = @import("std");
// It has no embedding or head tensor: it must use the target's (UNVERIFIED).
const names = @import("names.zig");

pub const root = "language_model.model.dspark.";
pub const Buf = names.Buf;
pub const Cut = names.Cut;

/// The seven linear layers of a draft layer: float8 codes `weight` [n, k] and a ue8m0 `weight_scale_inv` [n, k / 32].
pub const Lin = enum {
    q,
    k,
    v,
    o,
    gate,
    up,
    down,

    fn tail(p: Lin) []const u8 {
        return switch (p) {
            .q => "self_attn.q_proj",
            .k => "self_attn.k_proj",
            .v => "self_attn.v_proj",
            .o => "self_attn.o_proj",
            .gate => "mlp.gate_proj",
            .up => "mlp.up_proj",
            .down => "mlp.down_proj",
        };
    }

    /// The axis a tensor-parallel rank cuts, as the target's matrices are: o and down cut input rows (partial sums).
    pub fn cut(p: Lin) Cut {
        return switch (p) {
            .o, .down => .row,
            else => .col,
        };
    }
};

pub const Part = enum {
    weight,
    weight_scale_inv,

    fn tail(p: Part) []const u8 {
        return @tagName(p);
    }
};

/// Tensors stored as bf16: the quantization_config ignores fc, the Markov tables and the confidence projection.
pub const Plain = enum {
    fc, // [hidden, taps x hidden]
    hidden_norm, // [hidden]
    final_norm, // [hidden]
    markov_w1, // [vocab, rank]
    markov_w2, // [vocab, rank]
    conf_w, // [1, hidden + rank]
    conf_b, // [1]
    input_ln, // layer: [hidden]
    post_ln,
    q_norm, // layer: [head_dim]
    k_norm,

    pub fn isLayer(p: Plain) bool {
        return switch (p) {
            .input_ln, .post_ln, .q_norm, .k_norm => true,
            else => false,
        };
    }

    /// RMS norm weights (what a Gemma-style offset could apply to).
    pub fn isNorm(p: Plain) bool {
        return p.isLayer() or p == .hidden_norm or p == .final_norm;
    }

    fn tail(p: Plain) []const u8 {
        return switch (p) {
            .fc => "fc.weight",
            .hidden_norm => "hidden_norm.weight",
            .final_norm => "final_norm.weight",
            .markov_w1 => "markov_head.markov_w1.weight",
            .markov_w2 => "markov_head.markov_w2.weight",
            .conf_w => "confidence_head.proj.weight",
            .conf_b => "confidence_head.proj.bias",
            .input_ln => "input_layernorm.weight",
            .post_ln => "post_attention_layernorm.weight",
            .q_norm => "self_attn.q_norm.weight",
            .k_norm => "self_attn.k_norm.weight",
        };
    }
};

const layer_mid = "layers.{d}.decoder_layer.";

pub fn lin(buf: *Buf, p: Lin, layer: u32, part: Part) []const u8 {
    return std.fmt.bufPrint(buf, root ++ layer_mid ++ "{s}.{s}", .{ layer, p.tail(), part.tail() }) catch unreachable;
}

pub fn plain(buf: *Buf, p: Plain, layer: u32) []const u8 {
    const r = if (p.isLayer()) std.fmt.bufPrint(buf, root ++ layer_mid ++ "{s}", .{ layer, p.tail() }) else std.fmt.bufPrint(buf, root ++ "{s}", .{p.tail()});
    return r catch unreachable;
}

/// A tensor name taken apart.
pub const Name = struct {
    layer: u32 = 0,
    what: union(enum) { plain: Plain, lin: struct { proj: Lin, part: Part } },
};

fn number(s: *[]const u8) ?u32 {
    const end = std.mem.indexOfScalar(u8, s.*, '.') orelse return null;
    if (end == 0 or end > 3 or (end > 1 and s.*[0] == '0')) return null; // one spelling per index
    const n = std.fmt.parseInt(u32, s.*[0..end], 10) catch return null;
    s.* = s.*[end + 1 ..];
    return n;
}

/// The role of a checkpoint tensor name, or null for a name this family does not know.
pub fn parse(name: []const u8) ?Name {
    if (!std.mem.startsWith(u8, name, root)) return null;
    var rest = name[root.len..];
    for (std.enums.values(Plain)) |p| {
        if (!p.isLayer() and std.mem.eql(u8, rest, p.tail())) return .{ .what = .{ .plain = p } };
    }
    const lead = "layers.";
    if (!std.mem.startsWith(u8, rest, lead)) return null;
    rest = rest[lead.len..];
    var out: Name = .{ .what = undefined };
    out.layer = number(&rest) orelse return null;
    const mid = "decoder_layer.";
    if (!std.mem.startsWith(u8, rest, mid)) return null;
    rest = rest[mid.len..];
    for (std.enums.values(Plain)) |p| {
        if (p.isLayer() and std.mem.eql(u8, rest, p.tail())) {
            out.what = .{ .plain = p };
            return out;
        }
    }
    for (std.enums.values(Lin)) |p| {
        const t = p.tail();
        if (rest.len > t.len + 1 and std.mem.startsWith(u8, rest, t) and rest[t.len] == '.') {
            const part = std.meta.stringToEnum(Part, rest[t.len + 1 ..]) orelse return null;
            out.what = .{ .lin = .{ .proj = p, .part = part } };
            return out;
        }
    }
    return null;
}

test "names round-trip through parse for every role" {
    var b1: Buf = undefined;
    for (std.enums.values(Plain)) |p| {
        const n = parse(plain(&b1, p, 4)).?;
        try std.testing.expectEqual(p, n.what.plain);
        try std.testing.expectEqual(@as(u32, if (p.isLayer()) 4 else 0), n.layer);
    }
    for (std.enums.values(Lin)) |p| for (std.enums.values(Part)) |w| {
        const n = parse(lin(&b1, p, 3, w)).?;
        try std.testing.expectEqual(p, n.what.lin.proj);
        try std.testing.expectEqual(w, n.what.lin.part);
        try std.testing.expectEqual(@as(u32, 3), n.layer);
    };
}

test "the checkpoint's spellings" {
    var b: Buf = undefined;
    try std.testing.expectEqualStrings("language_model.model.dspark.layers.2.decoder_layer.mlp.down_proj.weight_scale_inv", lin(&b, .down, 2, .weight_scale_inv));
    try std.testing.expectEqualStrings("language_model.model.dspark.layers.0.decoder_layer.self_attn.q_norm.weight", plain(&b, .q_norm, 0));
    try std.testing.expectEqualStrings("language_model.model.dspark.markov_head.markov_w2.weight", plain(&b, .markov_w2, 9));
    try std.testing.expectEqualStrings("language_model.model.dspark.confidence_head.proj.bias", plain(&b, .conf_b, 9));
    try std.testing.expectEqualStrings("language_model.model.dspark.fc.weight", plain(&b, .fc, 9));
}

test "unknown, misspelled and non-canonical names are not roles" {
    for ([_][]const u8{
        "language_model.model.layers.3.self_attn.q_proj.weight", // the target's namespace
        "language_model.model.dspark.layers.03.decoder_layer.mlp.up_proj.weight",
        "language_model.model.dspark.layers.0.decoder_layer.mlp.up_proj.bias",
        "language_model.model.dspark.layers.0.decoder_layer.mlp.up_proj.weight_scale",
        "language_model.model.dspark.layers.0.decoder_layer.self_attn.index_q_proj.weight",
        "language_model.model.dspark.layers.0.input_layernorm.weight", // no decoder_layer
        "language_model.model.dspark.embed_tokens.weight", // the draft carries no embedding
        "language_model.model.dspark.lm_head.weight", // and no head
        "language_model.model.dspark.markov_head.embed.weight", // DeepSeek's spelling, not this pack's
        "language_model.model.dspark.layers.0.decoder_layer.",
    }) |bad| try std.testing.expect(parse(bad) == null);
}

test "o and down cut input rows, everything else output columns" {
    for (std.enums.values(Lin)) |p| {
        try std.testing.expectEqual(if (p == .o or p == .down) Cut.row else Cut.col, p.cut());
    }
}
