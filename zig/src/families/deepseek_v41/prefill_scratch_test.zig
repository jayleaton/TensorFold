//! Production K4 placement and launch lifetimes; no device allocation or model arithmetic is simulated.
const std = @import("std");
const t = std.testing;
const scratch = @import("prefill_scratch.zig");
const buffers = @import("buffers.zig");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const ws = @import("forward_prefill_k4.zig");
const check = @import("calls_check.zig");

const cfg: @import("config.zig").Config = .{};
const GmMode = @import("knob_types.zig").GmMode;
const bare_members = [_][]const u8{ "L.q", "L.z", "s.ex.xg", "L.o", "s.ex.z" };

fn layerCount(part: bp.Part) !usize {
    return if (part == .encoder) @as(usize, try @import("ced.zig").decoderStart(&cfg)) + 1 else cfg.layers;
}

fn program(a: std.mem.Allocator, o: block.Options, rows: i64, start: i64, part: bp.Part) ![]calls.Call {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    const layers = try a.alloc(u32, try layerCount(part));
    for (layers, 0..) |*l, i| l.* = @intCast(i);
    // The listing supplies expert ranges; legacy and split launches also need the present-width bitsets.
    for (layers) |layer| try w.gm.put(a, layer, .{ (1 << 4) | (1 << 6) | (1 << 8), (1 << 4) | (1 << 10) });
    const big = rows > o.prefill_rows;
    const plan_options = if (big) ws.k4Options(o) else o;
    const cs = if (part == .encoder)
        try bp.emitReplay(a, &cfg, &w, plan_options, layers, rows, start, part)
    else
        try bp.emitPrefill(a, &cfg, &w, plan_options, layers, rows, start, false);
    return if (big) bp.ownCalls(a, cs, ws.K4.tag) else cs;
}

fn productionValidate(cs: []const calls.Call, big: bool, alias: bool) !void {
    var state: @import("forward_prefill.zig").State = .{};
    state.k4ws.ready = true;
    state.k4ws.alias = alias;
    var f: @import("forward.zig").Forward = undefined;
    f.prefill_state = &state;
    try ws.acquireSegment(&f, big, cs);
}

fn hasRole(arg: calls.Arg, role: []const u8) bool {
    return switch (arg) {
        .t, .opaque_table => |tensor| tensor.role == .buf and std.mem.eql(u8, role, tensor.role.buf),
        .list => |xs| for (xs) |x| {
            if (hasRole(x, role)) break true;
        } else false,
        else => false,
    };
}

fn inspectProgram(cs: []const calls.Call, part: bp.Part, big: bool) !void {
    try scratch.validate(cs);
    try productionValidate(cs, big, true);
    const members = if (big) scratch.members else bare_members;
    const count = try layerCount(part);
    var boundaries: usize = 0;
    var seen: [scratch.members.len]bool = @splat(false);
    for (cs) |c| {
        if (c.begin != .none) {
            if (boundaries > 0) for (seen) |v| try t.expect(v);
            try t.expectEqual(if (boundaries == 0) calls.Begin.window else calls.Begin.layer, c.begin);
            boundaries += 1;
            seen = @splat(false);
        }
        for (members, 0..) |role, i| for (c.args) |x| {
            seen[i] = seen[i] or hasRole(x.arg, role);
        };
        if (!big) for (scratch.members) |role| for (c.args) |x| try t.expect(!hasRole(x.arg, role));
    }
    try t.expectEqual(count, boundaries);
    // An encoder's last layer emits only its site/compressor/stash; full prefill completes every layer.
    for (seen) |v| try t.expectEqual(part != .encoder, v);
}

fn options(rank: u32, gm: GmMode, profile: usize) block.Options {
    return .{
        .rank = rank,
        .gm_v2 = gm,
        .limit = 1 << 20,
        .rope_rows = (1 << 20) + 2048,
        .index_budget = 64 << 20,
        .taps = true,
        .branches = profile > 0,
        .pf_overlap = if (profile > 0) 4 else 0,
        .pf_overlap_site = profile > 1,
        .r1 = profile == 3,
        .mhc_defer = profile == 3,
    };
}

test "K4 scratch: both ranks encoder and full passes preserve lifetimes for every MoE mode and resumed start" {
    for ([_]u32{ 0, 1 }) |rank| for ([_]bp.Part{ .encoder, .whole }) |part| {
        for ([_]GmMode{ .off, .one, .split }) |gm| {
            for ([_]i64{ 2049, 3072, 4095, 4096 }) |rows| {
                for ([_]i64{ 0, 2047, 2048, 2049, 8192, 131072 }) |start| for (0..4) |profile| {
                    var arena = std.heap.ArenaAllocator.init(t.allocator);
                    defer arena.deinit();
                    const cs = try program(arena.allocator(), options(rank, gm, profile), rows, start, part);
                    try inspectProgram(cs, part, true);
                    var async = false;
                    for (cs) |c| async = async or c.side or c.defer_side;
                    if (profile > 0) try t.expect(async);
                };
            }
        }
    };
}

test "K4 scratch: consecutive full ragged and legacy chunks retain distinct workspace lifetimes" {
    for ([_]u32{ 0, 1 }) |rank| for ([_]bp.Part{ .encoder, .whole }) |part| {
        for ([_]GmMode{ .off, .one, .split }) |gm| {
            var arena = std.heap.ArenaAllocator.init(t.allocator);
            defer arena.deinit();
            const a = arena.allocator();
            var sequence: std.ArrayList(calls.Call) = .empty;
            var state: @import("forward_prefill.zig").State = .{};
            state.k4ws.ready = true;
            state.k4ws.alias = true;
            var f: @import("forward.zig").Forward = undefined;
            f.prefill_state = &state;
            var start: i64 = 2047;
            for ([_]i64{ 4096, 3073, 4096, 2049, 2048 }) |rows| {
                const cs = try program(a, options(rank, gm, 3), rows, start, part);
                try inspectProgram(cs, part, rows > 2048);
                try ws.acquireSegment(&f, rows > 2048, cs);
                try sequence.appendSlice(a, cs);
                start += rows;
            }
            try scratch.validate(sequence.items);
            try productionValidate(sequence.items, true, true);
        }
    };
}

fn expectRejected(expected: anyerror, cs: []const calls.Call) !void {
    try t.expectError(expected, scratch.validate(cs));
    try t.expectError(expected, productionValidate(cs, true, true));
    try productionValidate(cs, true, false);
    try productionValidate(cs, false, true);
}

test "K4 scratch: unjoined side-stream use and interleaved live members are rejected before launches" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cs = try program(a, .{ .gm_v2 = .one, .limit = 1 << 20, .rope_rows = (1 << 20) + 2048 }, 4096, 0, .encoder);
    try scratch.validate(cs);
    // Moving the actual query producer to a side stream leaves its device work pending after host return.
    var changed = false;
    for (cs) |*c| {
        if (!std.mem.eql(u8, c.name, "_rope")) continue;
        c.side = true;
        try expectRejected(error.ScratchAsyncLifetime, cs);
        c.side = false;
        c.defer_side = true;
        try expectRejected(error.ScratchAsyncLifetime, cs);
        c.defer_side = false;
        const original_name = c.name;
        c.name = "tf_dsv41_l2p.planted";
        try expectRejected(error.ScratchAsyncLifetime, cs);
        c.name = original_name;
        changed = true;
        break;
    }
    try t.expect(changed);
    // One real tensor added to the query's consumer makes its alias live in the same launch.
    const arg: calls.Arg = .{ .t = .{ .role = .{ .buf = "L.k4~z" }, .dt = .bf16, .shape = &.{1}, .stride = &.{1} } };
    changed = false;
    for (cs) |*c| {
        if (!std.mem.eql(u8, c.name, "_rope")) continue;
        const args = try a.alloc(calls.Named, c.args.len + 1);
        @memcpy(args[0..c.args.len], c.args);
        args[c.args.len] = .{ .arg = arg };
        c.args = args;
        try expectRejected(error.ScratchLifetimeOverlap, cs);
        changed = true;
        break;
    }
    try t.expect(changed);
}

test "K4 scratch: flag off offsets are unchanged and flag on shares only the proven regions" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var p: buffers.Plan = .{ .a = a };
    try p.add(try program(a, .{ .gm_v2 = .one, .limit = 1 << 20, .rope_rows = (1 << 20) + 2048 }, 4096, 0, .encoder));
    var off = scratch.Layout.init(&p, false);
    var on = scratch.Layout.init(&p, true);
    var legacy: u64 = 0;
    var found: [scratch.members.len]bool = @splat(false);
    var offsets: [scratch.members.len]u64 = undefined;
    for (p.sizes.keys(), p.sizes.values()) |role, bytes| {
        if (std.mem.indexOf(u8, role, ws.K4.tag) == null) continue;
        try t.expectEqual(legacy, off.next(role, bytes));
        legacy += std.mem.alignForward(u64, bytes, 256);
        const at = on.next(role, bytes);
        try t.expectEqual(@as(u64, 0), at % 256);
        try t.expect(at + bytes <= on.bytes);
        for (scratch.members, 0..) |name, i| if (std.mem.eql(u8, role, name)) {
            found[i] = true;
            offsets[i] = at;
        };
    }
    try t.expectEqual(legacy, off.bytes);
    try t.expectEqual(ws.planBytes(&p), off.bytes);
    try t.expectEqual(off.bytes, off.at);
    try t.expectEqual(on.bytes, on.at);
    for (found) |v| try t.expect(v);
    try t.expectEqual(offsets[0], offsets[1]);
    try t.expectEqual(offsets[0], offsets[2]);
    try t.expectEqual(offsets[3], offsets[4]);
    try t.expect(offsets[0] != offsets[3]);
    try t.expectEqual(@as(u64, 214 << 20), off.bytes - on.bytes);
}

test "K4 scratch: explicit opt in only, and TBO leaves its in flight namespaces unchanged" {
    try t.expect(!scratch.active(null, false));
    try t.expect(!scratch.active("0", false));
    try t.expect(!scratch.active("true", false));
    try t.expect(scratch.active("1", false));
    try t.expect(!scratch.active("1", true));
}
