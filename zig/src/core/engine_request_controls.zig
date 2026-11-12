//! Tool-call and grammar constraints carried by one native request.
const grammar = @import("lanes").grammar;

/// tool_choice "required" or a named function: outside a think block the answer opens a call to an offered tool.
pub const CallGate = @import("lanes").call_gate.Spec;

/// response_format and the guided_* fields: every token keeps the reply inside this grammar.
pub const Structure = struct {
    kind: enum { json, json_schema, regex, choice, grammar, structural_tag },
    /// The name a refusal gives it: a body field, tool_choice, or a route's own (text.format, output_config.format).
    field: []const u8 = "response_format",
    /// The schema (JSON text), regex, choices (a JSON array), EBNF or structural tag; empty for any JSON object.
    text: []const u8 = "",
    /// With thinking on, the grammar starts after this token.
    after: ?u32 = null,
    /// The server's compiler, which has compiled `text` once already (the engine's compile is then a cache hit).
    compiler: ?*grammar.Compiler = null,
    /// That compile, which the engine reads instead of compiling again; the server frees it after `finished`.
    compiled: ?grammar.Compiled = null,
    /// grammar_budget: mask tokens after which the reply could no longer complete within its token limits.
    budget: bool = false,

    pub fn free(st: Structure) void {
        if (st.compiled) |g| g.free();
    }
};
