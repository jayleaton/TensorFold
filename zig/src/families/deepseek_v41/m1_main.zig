//! `tf-dsv41-m1`: DeepSeek-V4.1's M1 gate on one GPU, our loader's weights and the captured block replayed against the Python engine's bits.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const dsv41_kernels = @import("dsv41_kernels");

comptime {
    _ = cuda;
    _ = core;
    _ = dsv41_kernels;
}

pub fn main() !u8 {
    std.debug.print("usage: tf-dsv41-m1 weights|replay PACK CAPTURE\n", .{});
    return 2;
}
