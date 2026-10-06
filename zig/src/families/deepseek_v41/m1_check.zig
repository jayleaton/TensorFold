//! `tf-dsv41-m1 check`: block.zig's calls for every captured decode window against the capture's launches, on the host:
//! each call's kernel, grid, scalars, tensor shapes / strides / dtypes / offsets, weights by name, and its roles' storage
//! (a role is one captured storage for its scope, two roles in one launch are two storages).

const std = @import("std");
const calls = @import("calls.zig");
const block = @import("block.zig");
const Config = @import("config.zig").Config;
const Value = std.json.Value;

/// Dense K2s from the capture's weights.json (each "<prefix>.trellis" int16 [K/16, N/16, 8 K2]), the experts' ranges
/// from a "layer gu_lo gu_hi d_lo d_hi" listing.
pub fn widthsFromCapture(a: std.mem.Allocator, weights_json: []const u8, experts: []const u8) !block.Widths {
    var w: block.Widths = .{};
    const root = try std.json.parseFromSliceLeaky(Value, a, weights_json, .{});
    var it = root.object.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        if (!std.mem.endsWith(u8, name, ".trellis")) continue;
        const shape = e.value_ptr.object.get("shape").?.array.items;
        if (shape.len != 3) continue;
        try w.dense.put(a, name[0 .. name.len - ".trellis".len], @intCast(@divExact(shape[2].integer, 8)));
    }
    try parseExperts(a, &w, experts);
    return w;
}

pub fn parseExperts(a: std.mem.Allocator, w: *block.Widths, text: []const u8) !void {
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line[0] == '#') continue;
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        var v: [5]u32 = undefined;
        for (&v) |*x| x.* = try std.fmt.parseInt(u32, f.next() orelse return error.BadListing, 10);
        try w.experts.put(a, v[0], .{ .{ v[1], v[2] }, .{ v[3], v[4] } });
    }
}

/// (role, captured buffer id) of every buffer-backed tensor of a call (its arguments and the capture's, the same shape).
pub fn bindings(a: std.mem.Allocator, want: calls.Arg, cap: Value, out: *std.ArrayList(Binding)) !void {
    switch (want) {
        .list => |items| for (items, cap.object.get("items").?.array.items) |x, y| try bindings(a, x, y, out),
        .t, .opaque_table => |t| switch (t.role) {
            .buf => |r| if (cap.object.get("buf")) |id| {
                var numel: i64 = 1;
                for (t.shape) |d| numel *= d;
                if (numel > 0) try out.append(a, .{ .role = r, .id = id.integer });
            },
            else => {},
        },
        else => {},
    }
}

pub const Binding = struct { role: []const u8, id: i64 };

/// The window's dataflow by content: a role's bytes when a call reads it are its bytes after its last call, unless an
/// op the capture does not record (torch glue) wrote it between (listed: the forward runs those). A role read for the
/// first time from the storage another role last used, holding what it left there, is wired wrong (reported).
pub const Flow = struct {
    a: std.mem.Allocator,
    /// role -> (captured storage, its bytes' digest) after the role's last call
    last: std.StringHashMapUnmanaged([2][]const u8) = .empty,
    glue: std.StringArrayHashMapUnmanaged(u32) = .empty,
    /// window / layer roles first touched by a call that leaves them unchanged: inputs the forward's glue fills
    inputs: std.StringArrayHashMapUnmanaged(u32) = .empty,
    miswired: u32 = 0,

    pub fn begin(f: *Flow, b: calls.Begin) void {
        if (b == .none) return;
        var drop: std.ArrayList([]const u8) = .empty;
        var it = f.last.keyIterator();
        while (it.next()) |k| {
            if (std.mem.startsWith(u8, k.*, "s.")) continue;
            if (b == .layer and std.mem.startsWith(u8, k.*, "w.")) continue;
            drop.append(f.a, k.*) catch {};
        }
        for (drop.items) |k| _ = f.last.remove(k);
    }

    pub fn call(f: *Flow, c: *const calls.Call, op: std.json.ObjectMap, log: *std.Io.Writer, where: []const u8) !void {
        var bs: std.ArrayList(Binding) = .empty;
        for (c.args, op.get("args").?.array.items) |x, y| try bindings(f.a, x.arg, y, &bs);
        const bufs = op.get("buffers").?.array.items;
        for (bs.items) |b| {
            const o = for (bufs) |v| {
                if (v.object.get("id").?.integer == b.id) break v.object;
            } else continue;
            const before = o.get("before").?;
            if (before != .string) continue;
            const key = o.get("key").?.string;
            if (f.last.get(b.role)) |h| {
                if (!std.mem.eql(u8, h[1], before.string)) {
                    const g = try f.glue.getOrPut(f.a, b.role);
                    if (!g.found_existing) g.value_ptr.* = 0;
                    g.value_ptr.* += 1;
                }
                continue;
            }
            // a first touch the call leaves as it was is a read: something outside the calls filled it
            if (!std.mem.startsWith(u8, b.role, "s.") and std.mem.eql(u8, before.string, o.get("after").?.string)) {
                const g = try f.inputs.getOrPut(f.a, b.role);
                if (!g.found_existing) g.value_ptr.* = 0;
                g.value_ptr.* += 1;
            }
            var it = f.last.iterator();
            while (it.next()) |e| if (std.mem.eql(u8, e.value_ptr[0], key) and std.mem.eql(u8, e.value_ptr[1], before.string)) {
                f.miswired += 1;
                try log.print("  {s} {s}: {s} first read holds what {s} last held\n", .{ where, c.name, b.role, e.key_ptr.* });
                break;
            };
        }
        for (bs.items) |b| {
            const o = for (bufs) |v| {
                if (v.object.get("id").?.integer == b.id) break v.object;
            } else continue;
            try f.last.put(f.a, b.role, .{ try f.a.dupe(u8, o.get("key").?.string), try f.a.dupe(u8, o.get("after").?.string) });
        }
    }
};

pub const Window = struct { set: []const u8, phase: []const u8, layers: []const u32, n: i64, start: i64, ops: []std.json.ObjectMap };

/// The capture's decode windows in order: each set's windows (meta.json "windows") from the prompt's end.
pub fn windows(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![]Window {
    const cwd = std.Io.Dir.cwd();
    const meta = try std.json.parseFromSliceLeaky(Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "meta.json" }), a, .limited(1 << 24)), .{});
    const prompt = meta.object.get("prompt").?.integer;
    const sizes = meta.object.get("windows").?.array.items;
    const text = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "ops.jsonl" }), a, .limited(1 << 34));
    var out: std.ArrayList(Window) = .empty;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var cur: ?*Window = null;
    var ops: std.ArrayList(std.json.ObjectMap) = .empty;
    while (lines.next()) |line| {
        const op = (try std.json.parseFromSliceLeaky(Value, a, line, .{})).object;
        const set = op.get("set").?.string;
        const phase = op.get("phase").?.string;
        if (phase.len < 2 or phase[0] != 'w') continue;
        if (cur == null or !std.mem.eql(u8, cur.?.set, set) or !std.mem.eql(u8, cur.?.phase, phase)) {
            if (cur) |c| c.ops = try ops.toOwnedSlice(a);
            const n = try std.fmt.parseInt(i64, phase[1..], 10);
            var start = prompt;
            for (sizes) |s| {
                if (s.integer == n) break;
                start += s.integer;
            }
            const ls = meta.object.get("sets").?.object.get(set).?.object.get("layers").?.array.items;
            const layers = try a.alloc(u32, ls.len);
            for (ls, layers) |x, *y| y.* = @intCast(x.integer);
            try out.append(a, .{ .set = set, .phase = phase, .layers = layers, .n = n, .start = start, .ops = &.{} });
            cur = &out.items[out.items.len - 1];
        }
        try ops.append(a, op);
    }
    if (cur) |c| c.ops = try ops.toOwnedSlice(a);
    return out.items;
}

/// Checks every window; one line a window plus each mismatch (up to `show` a window) on `log`. Returns mismatches.
pub fn checkAll(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, experts: []const u8, log: *std.Io.Writer, show: usize) !usize {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const w = try widthsFromCapture(a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "weights.json" }), a, .limited(1 << 28)), experts);
    const cfg: Config = .{};
    const ws = try windows(a, io, dir);
    var bad: usize = 0;
    var roles: calls.Roles = .{ .a = a };
    var flow: Flow = .{ .a = a };
    var last_set: []const u8 = "";
    for (ws) |win| {
        if (!std.mem.eql(u8, last_set, win.set)) { // each set ran in its own engine
            roles = .{ .a = a };
            flow.last = .empty;
        }
        last_set = win.set;
        const cs = try block.emit(a, &cfg, &w, .{}, win.layers, win.n, win.start, true);
        var equal: usize = 0;
        var shown: usize = 0;
        const m = @min(cs.len, win.ops.len);
        for (cs[0..m], win.ops[0..m], 0..) |*c, op, i| {
            roles.begin(c.begin);
            flow.begin(c.begin);
            try flow.call(c, op, log, try std.fmt.allocPrint(a, "{s} {s} #{d}", .{ win.set, win.phase, i }));
            if (try calls.check(a, c, op, &roles)) |why| {
                bad += 1;
                if (shown < show) try log.print("  {s} {s} #{d} {s}: {s}\n", .{ win.set, win.phase, i, c.name, why });
                shown += 1;
            } else equal += 1;
        }
        if (cs.len != win.ops.len) bad += 1;
        try log.print("{s} {s} (layers {any}, n {d}, from {d}): {d}/{d} calls equal, {d} emitted\n", .{ win.set, win.phase, win.layers, win.n, win.start, equal, win.ops.len, cs.len });
        try log.flush();
    }
    try log.print("dataflow: {d} first reads hold another role's bytes; roles written between calls by uncaptured ops (reads):", .{flow.miswired});
    for (flow.glue.keys(), flow.glue.values()) |k, v| try log.print(" {s}={d}", .{ k, v });
    try log.print("\nread-only first touches (inputs glue fills; in-place updates not listed):", .{});
    for (flow.inputs.keys(), flow.inputs.values()) |k, v| try log.print(" {s}={d}", .{ k, v });
    try log.print("\n", .{});
    return bad + flow.miswired;
}
