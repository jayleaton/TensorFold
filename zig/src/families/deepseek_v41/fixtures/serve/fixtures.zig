//! The serving goldens committed with the code (recipe/server_tests/gen_*.py wrote them from the Python engine).
pub const template = @embedFile("template.jsonl");
/// DeepSeek's own encoding.py renders (tools/zig/dsv41_reference.py; the release's encoding/ at 314113f)
pub const reference = @embedFile("reference.jsonl");
pub const dsml = @embedFile("dsml.jsonl");
pub const sampling = @embedFile("sampling.jsonl");
pub const mini_tokenizer = @embedFile("mini_tokenizer.json");
