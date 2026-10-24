//! tf-cuda-test mm3-layer-parity: one real-pack MiniMax-M3.1-Flash layer (world 1, planLayer's ops), partials as .npy.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const mm = @import("minimax_cuda");
const check = @import("check.zig");

const fw = mm.forward;
const plan = mm.forward_plan;
const Buf = plan.Buf;
const rk = cuda.minimax_m3_rows;

pub const usage = "  mm3-layer-parity PACK LAYER IN.npy OUT_DIR [IDS.npy|-] [HEAD_IN.npy]  one layer of a pack on given rows\n";

/// A read .npy whose dtype and row width must be `descr` and `cols`.
fn readRows(a: std.mem.Allocator, io: std.Io, path: []const u8, descr: []const u8, cols: usize) !core.npy.Array {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
    const x = try core.npy.parse(bytes);
    const width: usize = if (x.rank == 1) 1 else x.shape[1];
    if (!std.mem.eql(u8, x.descr, descr) or x.rank > 2 or width != cols or x.shape[0] == 0) {
        std.debug.print("FAIL {s}: {s} {any}, expected {s} [rows, {d}]\n", .{ path, x.descr, x.shape[0..x.rank], descr, cols });
        return error.BadInput;
    }
    return x;
}

/// Downloads rows of the step's buffers after each op that finishes a partial, and writes them as .npy files.
const Dump = struct {
    a: std.mem.Allocator,
    io: std.Io,
    f: *fw.Forward,
    out: []const u8,
    rows: u32,

    fn bytesOf(d: *const Dump, buf: Buf, rows: u32) !struct { []u8, usize } {
        const row = d.f.dev.g.rowBytes(buf);
        const host = try d.a.alloc(u8, rows * row);
        try d.f.dev.st.synchronize();
        try d.f.dev.b.mem[@backingInt(buf)].download(0, host);
        return .{ host, row };
    }

    /// `rows` rows of `buf` as `<name>.npy`; `descr` is the element type (bf16 rows go out as their `<u2` bits).
    fn save(d: *const Dump, name: []const u8, buf: Buf, rows: u32, descr: []const u8) !void {
        const host, const row = try d.bytesOf(buf, rows);
        const item: usize = std.fmt.parseInt(usize, descr[2..], 10) catch unreachable;
        var dict_buf: [160]u8 = undefined;
        const dict = try std.fmt.bufPrint(&dict_buf, "{{'descr': '{s}', 'fortran_order': False, 'shape': ({d}, {d}), }}", .{ descr, rows, row / item });
        const pad = 64 - (10 + dict.len + 1) % 64;
        const head = 10 + dict.len + pad + 1;
        const file = try d.a.alloc(u8, head + host.len);
        @memcpy(file[0..8], "\x93NUMPY\x01\x00");
        std.mem.writeInt(u16, file[8..10], @intCast(dict.len + pad + 1), .little);
        @memcpy(file[10..][0..dict.len], dict);
        @memset(file[10 + dict.len ..][0..pad], ' ');
        file[head - 1] = '\n';
        @memcpy(file[head..], host);
        const path = try std.fmt.allocPrint(d.a, "{s}/{s}.npy", .{ d.out, name });
        try std.Io.Dir.cwd().writeFile(d.io, .{ .sub_path = path, .data = file });
        std.debug.print("  wrote {s} ({d} x {d} {s})\n", .{ path, rows, row / item, descr });
    }

    /// The plan's sink: launch the op, then keep what it finished.
    pub fn emit(d: *Dump, op: plan.Op) !void {
        try d.f.dev.emit(op);
        const n = d.rows;
        switch (op) {
            .embed => try d.save("emb", .emb, n, "<u2"),
            .linear => |o| switch (o.w) {
                .o => try d.save("attn", .part, n, "<f4"),
                .dense_down => try d.save("mlp", .part, n, "<f4"),
                .shared_down => try d.save("shared", .shr, n, "<f4"),
                else => {},
            },
            .norm_add => |o| switch (o.norm) {
                .input => try d.save("in_norm", .a, n, "<u2"),
                .post => {
                    try d.save("mid", .res, n, "<f4");
                    try d.save("post_norm", .a, n, "<u2");
                },
                .final => {},
            },
            .router => try d.save("router_logits", .logits_r, n, "<f4"),
            .route => {
                try d.save("pick", .pick, n, "<i4");
                try d.save("wts", .wts, n, "<f4");
            },
            .experts => |o| if (o.row0 + o.rows == n) try d.save("routed", .routed, n, "<f4"),
            else => {},
        }
    }
};

fn upload(f: *fw.Forward, buf: Buf, bytes: []const u8) !void {
    try f.dev.st.synchronize();
    try f.dev.b.mem[@backingInt(buf)].upload(0, bytes);
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 4) {
        std.debug.print("{s}", .{usage});
        return error.MissingArgument;
    }
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = gpu.io;
    const dir = rest[0];
    const layer = try std.fmt.parseInt(u16, rest[1], 10);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 24));
    const c = try mm.config.parse(a, text);
    if (layer >= c.layers) return error.BadLayer;
    const x = try readRows(a, io, rest[2], "<f4", c.hidden);
    const n: u32 = @intCast(x.shape[0]);
    if (n > mm.iface.prefill_rows) return error.TooManyRows;
    const share = try mm.shard.Share.of(&c, 0, 1);

    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const t0 = check.now(io);
    var loaded = try mm.weights_device.loadLayers(gpu.gpa, io, gpu.d, stream, dir, &c, share, &.{layer});
    defer loaded.deinit();
    std.debug.print("  layer {d}: {d} MiB of weights in {d:.1} s\n", .{ layer, loaded.bytes >> 20, @as(f64, @floatFromInt(check.now(io) - t0)) / 1e9 });

    const g = try plan.Geometry.of(&c, share, layer + 1, n);
    var m = try fw.Modules.load(gpu.d);
    defer m.deinit();
    const max_pages = mm.iface.maxPages(n);
    var b = try fw.Buffers.init(gpu.d, &g, max_pages);
    defer b.deinit();
    const dims = try mm.pool.Dims.of(&c, share);
    var pool = try cuda.DeviceBuffer.alloc(gpu.d, max_pages * dims.pageBytes());
    defer pool.free();
    try pool.fill8(0, null);
    var mirror = try rk.Mirror.init(gpu.d, 1, max_pages, mm.iface.shared_rows);
    defer mirror.deinit();
    const table = try a.alloc(i32, max_pages);
    for (table, 0..) |*e, i| e.* = @intCast(i);
    try mirror.put(0, 0, table);
    const pages: fw.Pages = .{ .cache = .{ .pool = pool.ptr, .page_bytes = dims.pageBytes(), .layer_off = 0, .kvh = g.kvh }, .layer_bytes = dims.layerToken() * mm.pool.page_tokens, .mirror = &mirror };
    var f = try fw.Forward.init(gpu.gpa, gpu.d, stream, &m, &loaded.weights, &g, &b, pages, null);
    defer f.deinit(gpu.gpa);
    var dump: Dump = .{ .a = a, .io = io, .f = &f, .out = rest[3], .rows = n };

    // one prompt chunk at positions 0 .. n - 1 of slot 0: the rows enter as the residual (res = 0, then res += rows)
    const sh: plan.Shape = .{ .rows = n, .prompt = true, .max_pos = n - 1, .head = .none };
    try plan.check(&g, sh);
    try dump.emit(.{ .stage = .{ .rows = n, .shared = false } });
    if (rest.len > 4 and !std.mem.eql(u8, rest[4], "-")) {
        const ids = try readRows(a, io, rest[4], "<i4", 1);
        if (ids.shape[0] != n) return error.BadInput;
        try upload(&f, .ids, ids.data);
        try dump.emit(.{ .embed = .{ .rows = n } });
    }
    try dump.emit(.{ .zero = .{ .buf = .res, .rows = n } });
    try upload(&f, .part, x.data);
    try dump.emit(.{ .norm_add = .{ .x = .{ .buf = .part }, .x_f32 = true, .norm = .input, .layer = layer, .y = .{ .buf = .a }, .rows = n } });
    const t1 = check.now(io);
    try plan.planLayer(&g, sh, layer, &dump);
    try dump.save("out", .res, n, "<f4");
    const status = try f.dev.status();
    std.debug.print("  layer {d}: {d} rows in {d:.1} ms (dumps included), status {d}\n", .{ layer, n, @as(f64, @floatFromInt(check.now(io) - t1)) / 1e6, status });
    try check.expect(status == 0, "mm3-layer-parity: a launch found no page (status {d})", .{status});

    if (rest.len > 5) try headRows(&dump, rest[5], c.hidden);
    check.pass("mm3-layer-parity layer {d} rows {d}", .{ layer, n });
}

/// The final norm and the head on given rows (at most the head's 128): their fp32 logits as head.npy.
fn headRows(d: *Dump, path: []const u8, hidden: u32) !void {
    const x = try readRows(d.a, d.io, path, "<f4", hidden);
    const rows: u32 = @intCast(x.shape[0]);
    if (rows > plan.head_rows or rows > d.rows) return error.TooManyRows;
    try d.f.dev.emit(.{ .zero = .{ .buf = .res, .rows = rows } });
    try upload(d.f, .part, x.data);
    try d.f.dev.emit(.{ .norm_add = .{ .x = .{ .buf = .part }, .x_f32 = true, .norm = .final, .layer = 0, .y = .{ .buf = .a }, .rows = rows } });
    try d.f.dev.emit(.{ .linear = .{ .w = .head, .layer = 0, .x = .{ .buf = .a }, .y = .{ .buf = .hlog }, .rows = rows, .prompt = false } });
    try d.save("final_norm", .a, rows, "<u2");
    try d.save("head", .hlog, rows, "<f4");
}
