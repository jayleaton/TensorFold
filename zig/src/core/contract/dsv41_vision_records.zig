//! Exact launch envelopes from vision_kernels.zig; compiler usage remains pending until measured from fatbins.
const registry = @import("registry.zig");
const R = registry.CudaRecord;
fn record(comptime suffix: []const u8, threads: u32) R {
    return .{ .launch = .{ .name = "dsv41_vision_" ++ suffix, .image = "dsv41_vision", .symbol = "dsv41_vision_" ++ suffix, .threads = threads, .min_sm = 80 }, .usage = registry.none };
}
pub const records = [_]R{
    record("rms", 512),
    record("rope", 1024),
    record("attn", 128),
    record("silu_mul", 256),
    record("gelu", 256),
    record("add", 256),
    record("unfold", 256),
    record("span", 256),
};
