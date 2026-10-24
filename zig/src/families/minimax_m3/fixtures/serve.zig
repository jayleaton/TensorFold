//! MiniMax-M3.1 serving fixtures: both chat templates and the requests tools/zig/minimax_reference.py renders.

/// The official MiniMax-M3.1-Flash chat_template.jinja (always adaptive).
pub const m31 = @embedFile("chat_template.m31-54e4e69d.jinja");
/// The thinking_mode-switch variant the family's template goldens check.
pub const mia = @embedFile("chat_template.jinja");
/// One request per line: OpenAI messages, tools, thinking, effort, and the prompt each template renders (or a skip).
pub const reference = @embedFile("serve_reference.jsonl");
