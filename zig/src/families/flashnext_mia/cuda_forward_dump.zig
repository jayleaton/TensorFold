//! Flash Next per-layer tensor dumps (capture.py Dumps): digests checked or tensors written.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("cuda_triton.zig");
const W = @import("cuda_weights.zig");
const Allocator = std.mem.Allocator;
const fwd = @import("cuda_forward.zig");
const Forward = fwd.Forward;

/// Per-layer tensors named as capture.py Dumps: each one's sha256, checked against digests.json, or dumped to `dir`.
pub const Dump = struct {
    gpa: Allocator,
    io: std.Io,
    want: ?*const std.json.ObjectMap = null,
    dir: ?[]const u8 = null,
    prefix: []const u8 = "",
    mtp: bool = false,
    seen: std.StringHashMapUnmanaged(usize) = .empty,
    names: std.ArrayList([]u8) = .empty,
    equal: usize = 0,
    differ: usize = 0,
    missing: usize = 0,
    /// mismatches printed at most
    show: usize = 12,
    prefix_buf: [16]u8 = undefined,

    pub fn deinit(d: *Dump) void {
        d.seen.deinit(d.gpa);
        for (d.names.items) |n| d.gpa.free(n);
        d.names.deinit(d.gpa);
    }

    pub fn setChunk(d: *Dump, chunk: ?usize) void {
        d.prefix = if (chunk) |c| std.fmt.bufPrint(&d.prefix_buf, "c{d:0>2}_", .{c}) catch "" else "";
    }

    fn put(d: *Dump, f: *Forward, name: []const u8, ptr: u64, bytes: usize) !void {
        var nb: [160]u8 = undefined;
        const base = try std.fmt.bufPrint(&nb, "{s}{s}{s}", .{ d.prefix, if (d.mtp) "mtp_" else "", name });
        const key = try d.gpa.dupe(u8, base);
        try d.names.append(d.gpa, key);
        const gop = try d.seen.getOrPut(d.gpa, key);
        var full_buf: [176]u8 = undefined;
        var full: []const u8 = key;
        if (gop.found_existing) {
            gop.value_ptr.* += 1;
            full = try std.fmt.bufPrint(&full_buf, "{s}#{d}", .{ key, gop.value_ptr.* });
        } else gop.value_ptr.* = 1;
        const host = try d.gpa.alloc(u8, bytes);
        defer d.gpa.free(host);
        try f.s.synchronize();
        const buf: cuda.DeviceBuffer = .{ .d = f.d, .ptr = ptr, .len = bytes };
        try buf.download(0, host);
        var sum: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(host, &sum, .{});
        const hex = std.fmt.bytesToHex(sum, .lower);
        if (d.dir) |dir| {
            var pb: [512]u8 = undefined;
            var fb: [176]u8 = undefined;
            const file = try std.fmt.bufPrint(&fb, "{s}", .{full});
            for (file) |*ch| if (ch.* == '#') {
                ch.* = '_';
            };
            const path = try std.fmt.bufPrint(&pb, "{s}/{s}.bin", .{ dir, file });
            try std.Io.Dir.cwd().writeFile(d.io, .{ .sub_path = path, .data = host });
        }
        const want = d.want orelse return;
        const entry = want.get(full) orelse {
            d.missing += 1;
            if (d.missing <= d.show) std.debug.print("  dump {s}: not in the oracle's digests\n", .{full});
            return;
        };
        const ws = entry.object.get("sha256").?.string;
        if (std.mem.eql(u8, ws, &hex)) {
            d.equal += 1;
        } else {
            d.differ += 1;
            if (d.differ <= d.show) std.debug.print("  dump {s}: DIFFER ({d} bytes)\n", .{ full, bytes });
        }
    }
};

pub fn put(f: *Forward, name: []const u8, ptr: u64, bytes: usize) !void {
    if (f.dump) |d| try d.put(f, name, ptr, bytes);
}

pub fn putLayer(f: *Forward, layer: *const W.Layer, what: []const u8, ptr: u64, bytes: usize) !void {
    if (f.dump == null) return;
    var buf: [64]u8 = undefined;
    const name = if (layer.index < 0)
        try std.fmt.bufPrint(&buf, "L{d}_{s}", .{ layer.index, what })
    else
        try std.fmt.bufPrint(&buf, "L{d:0>2}_{s}", .{ @as(u32, @intCast(layer.index)), what });
    try f.put(name, ptr, bytes);
}

pub fn putBranch(f: *Forward, layer: *const W.Layer, what: []const u8, br: tri.Branch, R: usize) !void {
    if (f.dump == null) return;
    const D = f.g.hidden;
    var buf: [32]u8 = undefined;
    switch (br) {
        .bf16 => |p| try f.putLayer(layer, try std.fmt.bufPrint(&buf, "{s}_m1", .{what}), p, R * D * 2),
        .moe => |m| try f.putLayer(layer, try std.fmt.bufPrint(&buf, "{s}_m2", .{what}), m.y, R * m.slots * D * @as(usize, if (m.y_f32) 4 else 2)),
        .ranks => |r| try f.putLayer(layer, try std.fmt.bufPrint(&buf, "{s}_m3", .{what}), r.part, r.world * R * D * 4),
        .slices => |s| try f.putLayer(layer, try std.fmt.bufPrint(&buf, "{s}_m4", .{what}), s.part, s.sk * R * D * 4),
        .none => {},
    }
}
