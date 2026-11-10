//! The int4 check's device buffers: uploads, zeroed scratch and downloads, freed together.
const std = @import("std");
const cuda = @import("cuda");

const Allocator = std.mem.Allocator;

pub const Dev = struct {
    d: *const cuda.Driver,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    gpa: Allocator,

    pub fn put(dv: *Dev, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(dv.d, bytes);
        try dv.bufs.append(dv.gpa, b);
        return b.ptr;
    }

    pub fn zeros(dv: *Dev, n: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(dv.d, @max(n, 16));
        try b.fill8(0, null);
        // cuMemsetD8 runs on the legacy stream, which the checks' stream does not wait for: finish the fill first
        try dv.d.check(dv.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
        try dv.bufs.append(dv.gpa, b);
        return b.ptr;
    }

    pub fn get(dv: *Dev, ptr: u64, out: []u8) !void {
        const b: cuda.DeviceBuffer = .{ .d = dv.d, .ptr = ptr, .len = out.len };
        try b.download(0, out);
    }

    pub fn deinit(dv: *Dev) void {
        for (dv.bufs.items) |*b| b.free();
        dv.bufs.deinit(dv.gpa);
    }
};
