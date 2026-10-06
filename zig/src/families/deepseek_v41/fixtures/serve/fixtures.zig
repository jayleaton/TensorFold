//! The serving goldens committed with the code (zig/tests/server/dsv41/gen_*.py wrote them from the Python engine).
pub const template = @embedFile("template.jsonl");
pub const dsml = @embedFile("dsml.jsonl");
pub const sampling = @embedFile("sampling.jsonl");
