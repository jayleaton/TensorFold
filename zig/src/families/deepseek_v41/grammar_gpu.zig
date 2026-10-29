//! Apply shared lanes' masks to each rank's fp32 logits, with no second matcher or scheduler.
const std = @import("std");
const cuda = @import("cuda");
const Forward = @import("forward.zig").Forward;
const rows = @import("grammar_rows.zig");
const iface = @import("draft/iface.zig");
const Allocator = std.mem.Allocator;
pub const Row = rows.Row;
pub const enabled = rows.enabled;

pub const Masks = struct {
    gpa: Allocator,
    f: *Forward,
    words: u32,
    max_rows: u32,
    lo: u32,
    width: u32,
    module: cuda.Module,
    kernel: cuda.Function,
    host: cuda.HostBuffer,
    dev: cuda.DeviceBuffer,
    count: u32 = 0,
    message: std.ArrayList(i64) = .empty,
    kept: []Row,

    pub fn init(gpa: Allocator, f: *Forward, max_rows: u32) !*Masks {
        if (f.grammar != null or max_rows == 0) return error.GrammarSetup;
        const part = try @import("partition.zig").share(f.cfg.vocab, f.comm.rank(), f.comm.world());
        const words = (f.cfg.vocab + 31) / 32;
        const bytes = 4 * @as(usize, max_rows) * (words + 1);
        var host = try cuda.HostBuffer.alloc(f.runner.d, bytes);
        errdefer host.free();
        var dev = try cuda.DeviceBuffer.alloc(f.runner.d, bytes);
        errdefer dev.free();
        var log: [4096]u8 = @splat(0);
        var module = try cuda.Module.loadPtx(f.runner.d, @embedFile("grammar_mask_ptx"), &log);
        errdefer module.unload();
        const kernel = try module.function(rows.symbol);
        const kept = try gpa.alloc(Row, max_rows);
        errdefer gpa.free(kept);
        const x = try gpa.create(Masks);
        x.* = .{ .gpa = gpa, .f = f, .words = words, .max_rows = max_rows, .lo = part.lo, .width = part.len, .module = module, .kernel = kernel, .host = host, .dev = dev, .kept = kept };
        f.grammar = .{ .ptr = x, .follow = follow, .apply = applyFn };
        return x;
    }

    pub fn deinit(x: *Masks) void {
        x.f.runner.stream.synchronize() catch {};
        x.f.grammar = null;
        x.message.deinit(x.gpa);
        x.gpa.free(x.kept);
        x.module.unload();
        x.dev.free();
        x.host.free();
        x.gpa.destroy(x);
    }

    fn bits(x: *Masks) []u32 {
        return @alignCast(std.mem.bytesAsSlice(u32, x.host.bytes[0 .. 4 * @as(usize, x.max_rows) * x.words]));
    }

    fn indices(x: *Masks) []u32 {
        const off = 4 * @as(usize, x.max_rows) * x.words;
        return @alignCast(std.mem.bytesAsSlice(u32, x.host.bytes[off..]));
    }

    pub fn stage(x: *Masks, all: []const Row) !void {
        try rows.validate(all, x.words, x.max_rows);
        // Rows allowing every token are no-ops; with none now or staged there is nothing to wait on, apply or send.
        const list = rows.constraining(all, x.f.cfg.vocab, x.count, x.kept) orelse return;
        // The previous async upload must finish before its pinned source is reused.
        if (x.count != 0) try x.f.runner.stream.synchronize();
        x.count = @intCast(list.len);
        for (list, 0..) |r, i| {
            x.indices()[i] = r.index orelse std.math.maxInt(u32);
            @memcpy(x.bits()[i * x.words ..][0..x.words], r.bits);
        }
        if (x.f.leads()) {
            try x.message.resize(x.gpa, 3 + list.len * (x.words + 1));
            const msg = x.message.items;
            msg[0] = Forward.op_grammar;
            msg[1] = x.words;
            msg[2] = x.count;
            for (list, 0..) |_, i| {
                const at = 3 + i * (x.words + 1);
                msg[at] = x.indices()[i];
                for (x.bits()[i * x.words ..][0..x.words], msg[at + 1 ..][0..x.words]) |b, *v| v.* = b;
            }
            try x.f.link.?.send(msg);
        }
    }

    fn follow(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const x: *Masks = @ptrCast(@alignCast(ptr));
        if (msg.len < 3 or msg[1] != x.words or msg[2] < 0 or msg[2] > x.max_rows) return error.BadPlan;
        const count: usize = @intCast(msg[2]);
        if (msg.len != 3 + count * (x.words + 1)) return error.BadPlan;
        var list: std.ArrayList(Row) = .empty;
        defer list.deinit(x.gpa);
        var arena = std.heap.ArenaAllocator.init(x.gpa);
        defer arena.deinit();
        for (0..count) |i| {
            const at = 3 + i * (x.words + 1);
            const index = std.math.cast(u32, msg[at]) orelse return error.BadPlan;
            const b = try arena.allocator().alloc(u32, x.words);
            for (b, msg[at + 1 ..][0..x.words]) |*v, w| v.* = std.math.cast(u32, w) orelse return error.BadPlan;
            try list.append(x.gpa, .{ .index = if (index == std.math.maxInt(u32)) null else index, .bits = b });
        }
        try x.stage(list.items);
    }

    fn applyFn(ptr: *anyopaque, row0: u32, n: u32) anyerror!void {
        const x: *Masks = @ptrCast(@alignCast(ptr));
        if (x.count == 0) return;
        const idx = x.indices();
        for (idx[0..x.count]) |*i| if (i.* == std.math.maxInt(u32)) {
            if (n == 0) return error.GrammarRows;
            i.* = row0 + n - 1;
        };
        const r = x.f.runner;
        const off = 4 * @as(usize, x.max_rows) * x.words;
        try x.dev.uploadAsync(0, x.host.bytes[0 .. 4 * @as(usize, x.count) * x.words], r.stream.handle);
        try x.dev.uploadAsync(off, x.host.bytes[off..][0 .. 4 * @as(usize, x.count)], r.stream.handle);
        var args: cuda.launch.Args = .{};
        args.add(r.addressOf("w.logits") orelse return error.Unbound);
        args.add(@as(u64, x.width));
        args.add(x.dev.ptr + off);
        args.add(x.dev.ptr);
        args.add(x.words);
        args.add(x.lo / 32);
        args.add(x.width / 32);
        args.add(x.width);
        try cuda.launch.launch(x.kernel, .{ .grid = .{ .x = (x.width / 32 + rows.threads - 1) / rows.threads, .y = x.count }, .block = .{ .x = rows.threads } }, r.stream, &args);
        // Keep masks across the forward's greedy and separate keyed segments; they are idempotent.
    }
};

pub fn stage(f: *Forward, list: []const Row) !void {
    const hook = f.grammar orelse {
        if (list.len != 0) return error.StructuresUnsupported;
        return;
    };
    const x: *Masks = @ptrCast(@alignCast(hook.ptr));
    return x.stage(list);
}

pub fn segments(f: *Forward, segs: []const iface.Segment) !void {
    var arena = std.heap.ArenaAllocator.init(f.gpa);
    defer arena.deinit();
    return stage(f, try segmentRows(arena.allocator(), segs, (f.cfg.vocab + 31) / 32));
}

/// The windows' masked rows, each at its row of the forward's "w.logits" (the segments in window order).
pub fn segmentRows(a: Allocator, segs: []const iface.Segment, words: u32) ![]Row {
    var list: std.ArrayList(Row) = .empty;
    var at: u32 = 0;
    for (segs) |s| {
        if (s.masks.len != 0) {
            if (s.masks.len != s.tokens.len * words) return error.GrammarWords;
            for (0..s.tokens.len) |i| try list.append(a, .{ .index = at + @as(u32, @intCast(i)), .bits = s.masks[i * words ..][0..words] });
        }
        at += @intCast(s.tokens.len);
    }
    return list.items;
}
