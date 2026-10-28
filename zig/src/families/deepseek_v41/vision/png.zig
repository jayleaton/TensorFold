//! PNG as Pillow 12.3 opens it (``PngImagePlugin`` + ``ZipDecode``): the mode table ``_MODES`` (16-bit samples keep
const std = @import("std");
const pixels = @import("pixels.zig");
const Allocator = std.mem.Allocator;
const Image = pixels.Image;

pub const Error = error{ BadPng, UnsupportedPng, TooManyPixels } || Allocator.Error;

pub const signature = "\x89PNG\r\n\x1a\n";

const Header = struct {
    w: u32,
    h: u32,
    depth: u8,
    color: u8,
    interlace: bool,
    mode: pixels.Mode,

    fn bpp(hd: Header) usize { // bytes a pixel for the filters (at least 1)
        return @max(1, hd.bitsPerPixel() / 8);
    }
    fn bitsPerPixel(hd: Header) usize {
        const n: usize = switch (hd.color) {
            0, 3 => 1,
            2 => 3,
            4 => 2,
            6 => 4,
            else => unreachable,
        };
        return n * hd.depth;
    }
    fn rowBytes(hd: Header, w: usize) usize {
        return (w * hd.bitsPerPixel() + 7) / 8;
    }
};

fn modeOf(depth: u8, color: u8) ?pixels.Mode {
    return switch (color) {
        0 => switch (depth) {
            1 => .bilevel,
            2, 4, 8 => .l,
            16 => .i16,
            else => null,
        },
        2 => if (depth == 8 or depth == 16) .rgb else null,
        3 => if (depth <= 8 and std.math.isPowerOfTwo(depth)) .p else null,
        4 => if (depth == 8) .la else if (depth == 16) .rgba else null,
        6 => if (depth == 8 or depth == 16) .rgba else null,
        else => null,
    };
}

fn be32(b: []const u8) u32 {
    return std.mem.readInt(u32, b[0..4], .big);
}

/// Width and height from the header alone (the pixel cap is checked before decoding).
pub fn size(raw: []const u8) Error![2]u32 {
    if (raw.len < 33 or !std.mem.eql(u8, raw[0..8], signature) or !std.mem.eql(u8, raw[12..16], "IHDR")) return error.BadPng;
    return .{ be32(raw[16..20]), be32(raw[20..24]) };
}

pub fn decode(gpa: Allocator, raw: []const u8, max_pixels: u64) Error!Image {
    if (raw.len < 8 or !std.mem.eql(u8, raw[0..8], signature)) return error.BadPng;
    var at: usize = 8;
    var hd: ?Header = null;
    var palette = pixels.defaultPalette();
    var palette_len: usize = 0;
    var transparent = false;
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(gpa);
    var seen_idat = false;
    while (at + 12 <= raw.len) {
        const len = be32(raw[at..]);
        if (len > raw.len - at - 12) return error.BadPng;
        const cid = raw[at + 4 .. at + 8];
        const data = raw[at + 8 .. at + 8 + len];
        const crc = be32(raw[at + 8 + len ..]);
        at += 12 + len;
        if (std.mem.eql(u8, cid, "IDAT")) {
            if (hd == null) return error.BadPng;
            if (seen_idat and idat.items.len == 0) return error.BadPng;
            seen_idat = true;
            try idat.appendSlice(gpa, data);
            continue;
        }
        if (seen_idat) { // Pillow reads the image's consecutive IDAT chunks, then stops
            if (idat.items.len > 0) break;
            continue;
        }
        var c = std.hash.Crc32.init();
        c.update(cid);
        c.update(data);
        if (c.final() != crc) return error.BadPng;
        if (std.mem.eql(u8, cid, "IHDR")) {
            if (len < 13) return error.BadPng;
            if (data[11] != 0) return error.BadPng;
            const mode = modeOf(data[8], data[9]) orelse return error.UnsupportedPng;
            hd = .{ .w = be32(data[0..]), .h = be32(data[4..]), .depth = data[8], .color = data[9], .interlace = data[12] != 0, .mode = mode };
            if (hd.?.w == 0 or hd.?.h == 0) return error.BadPng;
            if (@as(u64, hd.?.w) * hd.?.h > max_pixels) return error.TooManyPixels;
        } else if (std.mem.eql(u8, cid, "PLTE")) {
            if (hd == null or hd.?.mode != .p) continue;
            const n = @min(len / 3, 256);
            for (0..n) |i| @memcpy(palette[i * 4 ..][0..3], data[i * 3 ..][0..3]);
            palette_len = n;
        } else if (std.mem.eql(u8, cid, "tRNS")) {
            if (hd == null or hd.?.mode != .p) continue;
            if (data.len > palette_len or data.len > 256) return error.BadPng; // ``_MODES``: one alpha a palette entry
            var simple_zero = false; // ``_simple_palette``: one 0, the rest 255
            var zeros: usize = 0;
            var others = false;
            for (data) |v| {
                if (v == 0) zeros += 1 else if (v != 255) others = true;
            }
            simple_zero = zeros == 1 and !others;
            if (simple_zero) {
                const i = std.mem.indexOfScalar(u8, data, 0).?;
                palette[i * 4 + 3] = 0;
                transparent = true;
            } else if (zeros > 0 or others or data.len > 0) {
                for (data[0..@min(data.len, 256)], 0..) |v, i| palette[i * 4 + 3] = v;
                transparent = true;
            }
        } else if (std.mem.eql(u8, cid, "IEND")) break;
    }
    const h = hd orelse return error.BadPng;
    if (idat.items.len < 2) return error.BadPng;
    const rows = try inflate(gpa, h, idat.items);
    defer gpa.free(rows);
    var im: Image = .{ .w = h.w, .h = h.h, .mode = h.mode, .data = try gpa.alloc(u8, @as(usize, h.w) * h.h * pixels.channels(h.mode)), .palette = palette, .transparent = transparent };
    errdefer im.deinit(gpa);
    try unfilterAll(gpa, h, rows, &im);
    return im;
}

const passes = [7][4]u8{ .{ 0, 0, 8, 8 }, .{ 4, 0, 8, 8 }, .{ 0, 4, 4, 8 }, .{ 2, 0, 4, 4 }, .{ 0, 2, 2, 4 }, .{ 1, 0, 2, 2 }, .{ 0, 1, 1, 2 } };

fn passDims(h: Header, p: [4]u8) [2]usize {
    if (!h.interlace) return .{ h.w, h.h };
    const w = if (h.w > p[0]) (@as(usize, h.w) - p[0] + p[2] - 1) / p[2] else 0;
    const ht = if (h.h > p[1]) (@as(usize, h.h) - p[1] + p[3] - 1) / p[3] else 0;
    return .{ w, ht };
}

/// The filtered scanlines: IDAT's zlib stream (header skipped, the checksum not required, as ZipDecode stops once
fn inflate(gpa: Allocator, h: Header, z: []const u8) Error![]u8 {
    var total: usize = 0;
    const n_pass: usize = if (h.interlace) 7 else 1;
    for (passes[0..n_pass]) |p| {
        const d = passDims(h, if (h.interlace) p else .{ 0, 0, 1, 1 });
        if (d[0] > 0 and d[1] > 0) total += d[1] * (1 + h.rowBytes(d[0]));
    }
    if ((z[0] & 0x0f) != 8 or (z[1] & 0x20) != 0 or ((@as(u16, z[0]) << 8) | z[1]) % 31 != 0) return error.BadPng;
    const out = try gpa.alloc(u8, total);
    errdefer gpa.free(out);
    var in: std.Io.Reader = .fixed(z[2..]);
    var window: [std.compress.flate.max_window_len]u8 = undefined;
    var dec: std.compress.flate.Decompress = .init(&in, .raw, &window);
    dec.reader.readSliceAll(out) catch return error.BadPng;
    return out;
}

fn paeth(a: u8, b: u8, c: u8) u8 {
    const p = @as(i16, a) + b - c;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - c);
    if (pa <= pb and pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

/// Undo one row's filter in place against the previous (unfiltered) row.
fn unfilter(kind: u8, row: []u8, prev: []const u8, bpp: usize) Error!void {
    switch (kind) {
        0 => {},
        1 => for (bpp..row.len) |i| {
            row[i] +%= row[i - bpp];
        },
        2 => for (row, prev) |*x, b| {
            x.* +%= b;
        },
        3 => for (row, 0..) |*x, i| {
            const a: u16 = if (i >= bpp) row[i - bpp] else 0;
            x.* +%= @intCast((a + prev[i]) >> 1);
        },
        4 => for (row, 0..) |*x, i| {
            const a = if (i >= bpp) row[i - bpp] else 0;
            const c = if (i >= bpp) prev[i - bpp] else 0;
            x.* +%= paeth(a, prev[i], c);
        },
        else => return error.BadPng,
    }
}

fn unfilterAll(gpa: Allocator, h: Header, rows: []u8, im: *Image) Error!void {
    const bpp = h.bpp();
    const zero = try gpa.alloc(u8, h.rowBytes(h.w));
    defer gpa.free(zero);
    @memset(zero, 0);
    var at: usize = 0;
    const n_pass: usize = if (h.interlace) 7 else 1;
    for (0..n_pass) |pi| {
        const p = if (h.interlace) passes[pi] else [4]u8{ 0, 0, 1, 1 };
        const d = passDims(h, p);
        if (d[0] == 0 or d[1] == 0) continue;
        const rb = h.rowBytes(d[0]);
        var prev: []const u8 = zero[0..rb];
        for (0..d[1]) |y| {
            const kind = rows[at];
            const row = rows[at + 1 .. at + 1 + rb];
            try unfilter(kind, row, prev, bpp);
            store(h, im, row, p[1] + y * p[3], p[0], p[2], d[0]);
            prev = row;
            at += 1 + rb;
        }
    }
}

/// One unfiltered row into the image at row ``y``, pixels ``x0 + i dx``, in Pillow's mode.
fn store(h: Header, im: *Image, row: []const u8, y: usize, x0: usize, dx: usize, n: usize) void {
    const ch = pixels.channels(h.mode);
    const line = im.data[y * @as(usize, h.w) * ch ..][0 .. @as(usize, h.w) * ch];
    for (0..n) |i| {
        const x = x0 + i * dx;
        const o = line[x * ch ..][0..ch];
        switch (h.depth) {
            1, 2, 4 => {
                const bits = i * h.depth;
                const v = (row[bits / 8] >> @intCast(8 - h.depth - bits % 8)) & ((@as(u8, 1) << @intCast(h.depth)) - 1);
                o[0] = if (h.mode == .p) v else switch (h.depth) {
                    1 => if (v != 0) 255 else 0,
                    2 => v * 0x55,
                    else => v * 0x11,
                };
            },
            8 => if (h.color == 4 and h.mode == .la) @memcpy(o, row[i * 2 ..][0..2]) else @memcpy(o, row[i * ch ..][0..ch]),
            16 => switch (h.color) {
                0 => std.mem.writeInt(u16, o[0..2], std.mem.readInt(u16, row[i * 2 ..][0..2], .big), .native),
                2 => for (0..3) |c| {
                    o[c] = row[i * 6 + c * 2];
                },
                4 => { // "LA;16B" -> RGBA: L's high byte thrice, A's high byte
                    @memset(o[0..3], row[i * 4]);
                    o[3] = row[i * 4 + 2];
                },
                6 => for (0..4) |c| {
                    o[c] = row[i * 8 + c * 2];
                },
                else => unreachable,
            },
            else => unreachable,
        }
    }
}

fn appendChunk(gpa: Allocator, out: *std.ArrayList(u8), id: []const u8, data: []const u8) !void {
    var len: [4]u8 = undefined;
    std.mem.writeInt(u32, &len, @intCast(data.len), .big);
    try out.appendSlice(gpa, &len);
    try out.appendSlice(gpa, id);
    try out.appendSlice(gpa, data);
    var c = std.hash.Crc32.init();
    c.update(id);
    c.update(data);
    var crc: [4]u8 = undefined;
    std.mem.writeInt(u32, &crc, c.final(), .big);
    try out.appendSlice(gpa, &crc);
}

/// A 1x1 8-bit indexed PNG with the given palette and tRNS chunks.
fn testPng(gpa: Allocator, palette: []const u8, trns: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, signature);
    var ihdr: [13]u8 = @splat(0);
    std.mem.writeInt(u32, ihdr[0..4], 1, .big);
    std.mem.writeInt(u32, ihdr[4..8], 1, .big);
    ihdr[8] = 8;
    ihdr[9] = 3;
    try appendChunk(gpa, &out, "IHDR", &ihdr);
    try appendChunk(gpa, &out, "PLTE", palette);
    try appendChunk(gpa, &out, "tRNS", trns);
    try appendChunk(gpa, &out, "IDAT", "\x78\x9c\x63\x60\x00\x00\x00\x02\x00\x01");
    try appendChunk(gpa, &out, "IEND", "");
    return out.toOwnedSlice(gpa);
}

test "indexed transparency is bounded by the palette entry count and 256 entries" {
    const a = std.testing.allocator;
    const one = [_]u8{ 0, 0, 0 };
    const full: [768]u8 = @splat(0);
    var over: [257]u8 = @splat(0xff);
    over[256] = 0;
    for ([_][]const u8{ one[0..], full[0..] }) |palette| {
        const raw = try testPng(a, palette, &over);
        defer a.free(raw);
        try std.testing.expectError(error.BadPng, decode(a, raw, 64 * 1024 * 1024));
    }
    // the largest valid tRNS: one zero on the last palette entry
    var ok: [256]u8 = @splat(0xff);
    ok[255] = 0;
    const raw = try testPng(a, &full, &ok);
    defer a.free(raw);
    var im = try decode(a, raw, 64 * 1024 * 1024);
    defer im.deinit(a);
    try std.testing.expect(im.transparent);
    try std.testing.expectEqual(@as(u8, 0), im.palette[255 * 4 + 3]);
}

test "an interlaced pass dimensions its rows without wrapping the header's 32 bits" {
    var h: Header = .{ .w = std.math.maxInt(u32), .h = 1, .depth = 8, .color = 0, .interlace = true, .mode = .l };
    const d = passDims(h, .{ 0, 0, 8, 8 });
    try std.testing.expectEqual(@as(usize, 536_870_912), d[0]);
    try std.testing.expectEqual(@as(usize, 1), d[1]);
    h.w = 1;
    try std.testing.expectEqual(@as(usize, 0), passDims(h, .{ 4, 0, 8, 8 })[0]);
}
