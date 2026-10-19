//! TF_DSV41_RG_V2 (ours, router_gemv2.cu): prefill's gemv_kernel<16, 8, 2> launch as gemv2_kernel, the same bits.
const std = @import("std");
const cuda = @import("cuda");

/// gemv2_kernel<RT 16, NW 8, EW 8>: gemv_kernel<16, 8, 2>'s row tiles and counters, 64 experts a CTA
pub const symbol: [:0]const u8 = "_ZN8dsv41_rg12gemv2_kernelILi16ELi8ELi8EEEvNS_4ArgsE";
pub const RT = 16;
pub const NW = 8;
pub const EW = 8;
pub const threads = NW * 32;
/// the fewest rows v2 takes: 6 CTAs a row tile at 384 experts (1 an SM), at least two full waves on 48 SMs
pub const min_rows = 256;

pub const Error = error{BadKnob};

/// TF_DSV41_RG_V2: unset, empty or 0 off, 1 on; any other value refused.
pub fn enabled(v: ?[]const u8) Error!bool {
    const s = std.mem.trim(u8, v orelse return false, " ");
    if (s.len == 0 or std.mem.eql(u8, s, "0")) return false;
    if (std.mem.eql(u8, s, "1")) return true;
    return error.BadKnob;
}

/// Whether a route launch (rt, nw, ew) runs as v2: prod's prefill config at min_rows or more, E a multiple of EW.
pub fn takes(R: c_int, E: c_int, rt: u32, nw: u32, ew: u32) bool {
    return rt == RT and nw == 8 and ew == 2 and R >= min_rows and E >= EW and @rem(E, EW) == 0;
}

/// launch2<16, 8, 8>'s grid: (experts / 64, row tiles of 16), gemv_kernel<16, 8, 2>'s grid.y (the same counters).
pub fn grid(E: usize, R: usize) [3]usize {
    return .{ (E + NW * EW - 1) / (NW * EW), (R + RT - 1) / RT, 1 };
}

pub const Rg2 = struct {
    module: cuda.Module,
    f: cuda.Function,

    pub fn resolve(module: cuda.Module) !Rg2 {
        return .{ .module = module, .f = try module.function(symbol) };
    }
};

test "TF_DSV41_RG_V2: off unless 1, junk refused" {
    try std.testing.expect(!try enabled(null));
    try std.testing.expect(!try enabled(""));
    try std.testing.expect(!try enabled("0"));
    try std.testing.expect(!try enabled(" 0 "));
    try std.testing.expect(try enabled("1"));
    try std.testing.expect(try enabled(" 1"));
    for ([_][]const u8{ "2", "on", "true", "8", "1,2" }) |v| try std.testing.expectError(error.BadKnob, enabled(v));
}

test "RG_V2 takes prod's prefill launch only (decode's narrow and the other wide configs keep theirs)" {
    try std.testing.expect(takes(2048, 384, 16, 8, 2));
    try std.testing.expect(takes(4100, 384, 16, 8, 2));
    try std.testing.expect(takes(256, 384, 16, 8, 2));
    try std.testing.expect(!takes(255, 384, 16, 8, 2));
    try std.testing.expect(!takes(37, 384, 16, 8, 2));
    for ([_]u32{ 1, 2, 4, 8, 16 }) |rt| for ([_]u32{ 1, 2, 4, 8 }) |nw| try std.testing.expect(!takes(2048, 384, rt, nw, 0));
    try std.testing.expect(!takes(2048, 384, 16, 8, 1));
    try std.testing.expect(!takes(2048, 384, 16, 4, 2));
    try std.testing.expect(!takes(2048, 384, 8, 8, 2));
    try std.testing.expect(!takes(2048, 380, 16, 8, 2));
    try std.testing.expect(takes(2048, 512, 16, 8, 2));
}

test "RG_V2 grid: gemv_kernel<16, 8, 2>'s row tiles, a quarter of its expert blocks" {
    try std.testing.expectEqual([3]usize{ 6, 128, 1 }, grid(384, 2048));
    try std.testing.expectEqual([3]usize{ 6, 257, 1 }, grid(384, 4100));
    try std.testing.expectEqual([3]usize{ 8, 16, 1 }, grid(512, 256));
    // the counters: one a row tile, as dsv41_rg_route_cuda's check (R + rt - 1) / rt for the call's rt 16
    for ([_]usize{ 256, 300, 2048, 4100 }) |r| try std.testing.expectEqual((r + 15) / 16, grid(384, r)[1]);
}
