//! Shared values and helpers of the dspark contract writers.
const std = @import("std");
const core = @import("core");

const c = core.contract.spec;

pub const hidden = 6144;
pub const eps32: f32 = 1e-6;
pub const eps: f64 = eps32;
pub const rot = 64;
pub const theta = 1e7;
pub const nq = 32; // a rank's q heads
pub const kvh = 2;

pub fn zeroDepth(_: c.Shape) usize {
    return 0;
}

pub fn bf(rng: std.Random, sd: f32) u16 {
    return c.toBf16(rng.floatNorm(f32) * sd);
}

pub fn isNanBf(h: u16) bool {
    return std.math.isNan(c.fromBf16(h));
}
