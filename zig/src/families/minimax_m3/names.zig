//! MiniMax-M3.1-Flash's tensor names: the text model is `language_model.model.*`, the head `language_model.lm_head.*`.
const std = @import("std");

pub const lm = "language_model.model.";
pub const head_prefix = "language_model.lm_head";
pub const text_namespace = "language_model.";

/// Every EXL3 matrix kind: each is a group of suh, svh, mul1 and trellis tensors.
pub const Proj = enum {
    q,
    k,
    v,
    o,
    dense_gate,
    dense_up,
    dense_down,
    shared_gate,
    shared_up,
    shared_down,
    w1, // routed expert gate (the SiLU side is unverified)
    w3, // routed expert up
    w2, // routed expert down
    head,

    pub fn isExpert(p: Proj) bool {
        return p == .w1 or p == .w3 or p == .w2;
    }

    /// Which axis a tensor-parallel rank cuts: output columns (no reduction) or input rows (partial sums).
    pub fn cut(p: Proj) Cut {
        return switch (p) {
            .o, .dense_down, .shared_down, .w2 => .row,
            else => .col,
        };
    }

    fn tail(p: Proj) []const u8 {
        return switch (p) {
            .q => "self_attn.q_proj",
            .k => "self_attn.k_proj",
            .v => "self_attn.v_proj",
            .o => "self_attn.o_proj",
            .dense_gate => "mlp.gate_proj",
            .dense_up => "mlp.up_proj",
            .dense_down => "mlp.down_proj",
            .shared_gate => "block_sparse_moe.shared_experts.gate_proj",
            .shared_up => "block_sparse_moe.shared_experts.up_proj",
            .shared_down => "block_sparse_moe.shared_experts.down_proj",
            .w1, .w3, .w2 => "",
            .head => "",
        };
    }
};

pub const Cut = enum { col, row };

pub const Part = enum { suh, svh, mul1, trellis };

/// Tensors stored as they are (or with the Gemma fold): norms, the embedding, the fp16 index projections, the router.
pub const Plain = enum {
    embed,
    final_norm,
    input_ln,
    post_ln,
    q_norm,
    k_norm,
    index_q_proj,
    index_k_proj,
    index_q_norm,
    index_k_norm,
    router, // fp32 [experts, hidden]
    router_bias, // e_score_correction_bias [experts]

    pub fn isGlobal(p: Plain) bool {
        return p == .embed or p == .final_norm;
    }

    /// RMS norm weights: stored as w, applied as 1 + w when use_gemma_norm is set.
    pub fn isNorm(p: Plain) bool {
        return switch (p) {
            .final_norm, .input_ln, .post_ln, .q_norm, .k_norm, .index_q_norm, .index_k_norm => true,
            else => false,
        };
    }

    fn tail(p: Plain) []const u8 {
        return switch (p) {
            .embed => "embed_tokens.weight",
            .final_norm => "norm.weight",
            .input_ln => "input_layernorm.weight",
            .post_ln => "post_attention_layernorm.weight",
            .q_norm => "self_attn.q_norm.weight",
            .k_norm => "self_attn.k_norm.weight",
            .index_q_proj => "self_attn.index_q_proj.weight",
            .index_k_proj => "self_attn.index_k_proj.weight",
            .index_q_norm => "self_attn.index_q_norm.weight",
            .index_k_norm => "self_attn.index_k_norm.weight",
            .router => "block_sparse_moe.gate.weight",
            .router_bias => "block_sparse_moe.e_score_correction_bias",
        };
    }
};

pub const max_name = 160;
pub const Buf = [max_name]u8;

/// A group's prefix: `layer` and `expert` are ignored where the matrix has none.
pub fn group(buf: *Buf, p: Proj, layer: u32, expert: u32) []const u8 {
    const r = switch (p) {
        .head => std.fmt.bufPrint(buf, "{s}", .{head_prefix}),
        .w1, .w3, .w2 => std.fmt.bufPrint(buf, lm ++ "layers.{d}.block_sparse_moe.experts.{d}.{s}", .{ layer, expert, @tagName(p) }),
        else => std.fmt.bufPrint(buf, lm ++ "layers.{d}.{s}", .{ layer, p.tail() }),
    };
    return r catch unreachable; // the longest name is under 100 bytes
}

pub fn part(buf: *Buf, prefix: []const u8, which: Part) []const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ prefix, @tagName(which) }) catch unreachable;
}

pub fn plain(buf: *Buf, p: Plain, layer: u32) []const u8 {
    const r = if (p.isGlobal()) std.fmt.bufPrint(buf, lm ++ "{s}", .{p.tail()}) else std.fmt.bufPrint(buf, lm ++ "layers.{d}.{s}", .{ layer, p.tail() });
    return r catch unreachable;
}

/// A tensor name taken apart.
pub const Name = struct {
    layer: u32 = 0,
    expert: u32 = 0,
    what: union(enum) { plain: Plain, part: struct { proj: Proj, part: Part } },
};

fn partOf(s: []const u8) ?Part {
    return std.meta.stringToEnum(Part, s);
}

fn number(s: *[]const u8) ?u32 {
    const end = std.mem.indexOfScalar(u8, s.*, '.') orelse return null;
    if (end == 0 or end > 5 or (end > 1 and s.*[0] == '0')) return null; // no sign, no leading zero: one spelling per index
    const n = std.fmt.parseInt(u32, s.*[0..end], 10) catch return null;
    s.* = s.*[end + 1 ..];
    return n;
}

/// The role of a checkpoint tensor name, or null for a name this family does not know.
pub fn parse(name: []const u8) ?Name {
    if (std.mem.startsWith(u8, name, head_prefix ++ ".")) {
        const p = partOf(name[head_prefix.len + 1 ..]) orelse return null;
        return .{ .what = .{ .part = .{ .proj = .head, .part = p } } };
    }
    if (!std.mem.startsWith(u8, name, lm)) return null;
    var rest = name[lm.len..];
    for ([_]Plain{ .embed, .final_norm }) |g| if (std.mem.eql(u8, rest, g.tail())) return .{ .what = .{ .plain = g } };
    if (!std.mem.startsWith(u8, rest, "layers.")) return null;
    rest = rest["layers.".len..];
    var out: Name = .{ .what = undefined };
    out.layer = number(&rest) orelse return null;
    for (std.enums.values(Plain)) |p| {
        if (!p.isGlobal() and std.mem.eql(u8, rest, p.tail())) return .{ .layer = out.layer, .what = .{ .plain = p } };
    }
    const experts = "block_sparse_moe.experts.";
    if (std.mem.startsWith(u8, rest, experts)) {
        rest = rest[experts.len..];
        out.expert = number(&rest) orelse return null;
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        const proj = std.meta.stringToEnum(Proj, rest[0..dot]) orelse return null;
        if (!proj.isExpert()) return null;
        out.what = .{ .part = .{ .proj = proj, .part = partOf(rest[dot + 1 ..]) orelse return null } };
        return out;
    }
    for (std.enums.values(Proj)) |p| {
        if (p.isExpert() or p == .head) continue;
        const t = p.tail();
        if (rest.len > t.len + 1 and std.mem.startsWith(u8, rest, t) and rest[t.len] == '.') {
            out.what = .{ .part = .{ .proj = p, .part = partOf(rest[t.len + 1 ..]) orelse return null } };
            return out;
        }
    }
    return null;
}

test "names round-trip through parse for every role" {
    var buf: Buf = undefined;
    var buf2: Buf = undefined;
    for (std.enums.values(Plain)) |p| {
        const n = parse(plain(&buf, p, 17)).?;
        try std.testing.expectEqual(p, n.what.plain);
        try std.testing.expectEqual(@as(u32, if (p.isGlobal()) 0 else 17), n.layer);
    }
    for (std.enums.values(Proj)) |p| {
        for (std.enums.values(Part)) |w| {
            const prefix = group(&buf, p, 59, 127);
            const n = parse(part(&buf2, prefix, w)).?;
            try std.testing.expectEqual(p, n.what.part.proj);
            try std.testing.expectEqual(w, n.what.part.part);
            try std.testing.expectEqual(@as(u32, if (p == .head) 0 else 59), n.layer);
            try std.testing.expectEqual(@as(u32, if (p.isExpert()) 127 else 0), n.expert);
        }
    }
}

test "the checkpoint's spellings" {
    var buf: Buf = undefined;
    try std.testing.expectEqualStrings("language_model.model.layers.3.block_sparse_moe.experts.5.w2", group(&buf, .w2, 3, 5));
    try std.testing.expectEqualStrings("language_model.lm_head", group(&buf, .head, 9, 9));
    try std.testing.expectEqualStrings("language_model.model.layers.0.mlp.down_proj", group(&buf, .dense_down, 0, 0));
    try std.testing.expectEqualStrings("language_model.model.layers.4.block_sparse_moe.shared_experts.up_proj", group(&buf, .shared_up, 4, 0));
    try std.testing.expectEqualStrings("language_model.model.norm.weight", plain(&buf, .final_norm, 5));
    try std.testing.expectEqualStrings("language_model.model.layers.4.block_sparse_moe.e_score_correction_bias", plain(&buf, .router_bias, 4));
}

test "unknown, misspelled and non-canonical names are not roles" {
    for ([_][]const u8{
        "language_model.model.layers.3.self_attn.q_proj.bias", // EXL3 groups here carry no bias
        "language_model.model.layers.03.self_attn.q_proj.suh",
        "language_model.model.layers.3.block_sparse_moe.experts.x.w1.suh",
        "language_model.model.layers.3.block_sparse_moe.experts.1.w4.suh",
        "language_model.model.layers.3.block_sparse_moe.experts.1.w1",
        "language_model.model.layers.3.self_attn.index_v_proj.weight",
        "language_model.model.layers.3.self_attn.q_proj",
        "language_model.lm_head.weight",
        "language_model.mtp.0.eh_proj.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
        "language_model.model.layers.3.",
    }) |bad| try std.testing.expect(parse(bad) == null);
}

test "o, down and w2 cut input rows, everything else output columns" {
    for (std.enums.values(Proj)) |p| {
        const rows = p == .o or p == .dense_down or p == .shared_down or p == .w2;
        try std.testing.expectEqual(if (rows) Cut.row else Cut.col, p.cut());
    }
}
