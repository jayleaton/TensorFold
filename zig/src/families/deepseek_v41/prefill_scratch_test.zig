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

fn program(a: std.mem.Allocator, o: block.Options, rows: i64, start: i64) ![]calls.Call {
    const cfg: @import("config.zig").Config = .{};
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    return bp.ownCalls(a, try bp.emitReplay(a, &cfg, &w, ws.k4Options(o), &enc, rows, start, .encoder), ws.K4.tag);
}

test "K4 scratch: real asynchronous overlap and deferred programs keep aliased members on the compute stream" {
    for ([_]i64{ 2049, 3072, 4095, 4096 }) |rows| for ([_]i64{ 0, 8192, 131072 }) |start| {
        for (0..4) |profile| {
            var arena = std.heap.ArenaAllocator.init(t.allocator);
            defer arena.deinit();
            const a = arena.allocator();
            const o: block.Options = .{
                .gm_v2 = if (profile == 0) .off else .one,
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
            const cs = try program(a, o, rows, start);
            // No host synchronization separates launches: validate requires device FIFO on one stream.
            try scratch.validate(cs);
            var p: buffers.Plan = .{ .a = a };
            try p.add(cs);
            for (scratch.members) |role| try t.expect(p.sizes.contains(role));
            var async = false;
            var boundaries: usize = 0;
            var complete: usize = 0;
            var seen: [scratch.members.len]bool = @splat(false);
            for (cs) |c| {
                if (c.begin != .none) {
                    if (boundaries > 0) {
                        for (seen) |v| try t.expect(v);
                        complete += 1;
                    }
                    boundaries += 1;
                    seen = @splat(false);
                }
                for (scratch.members, 0..) |role, i| for (c.args) |x| {
                    if (x.arg != .t or x.arg.t.role != .buf) continue;
                    seen[i] = seen[i] or std.mem.eql(u8, role, x.arg.t.role.buf);
                };
                async = async or c.side or c.defer_side;
            }
            // Twenty whole layers, then the encoder endpoint compressor: no subphase resets.
            try t.expectEqual(@as(usize, 21), boundaries);
            try t.expectEqual(@as(usize, 20), complete);
            for (seen) |v| try t.expect(!v);
            if (profile > 0) try t.expect(async);
        }
    };
}

test "K4 scratch: unjoined side-stream use and interleaved live members are rejected before launches" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cs = try program(a, .{ .gm_v2 = .one, .limit = 1 << 20, .rope_rows = (1 << 20) + 2048 }, 4096, 0);
    try scratch.validate(cs);
    // Moving the actual query producer to a side stream leaves its device work pending after host return.
    var changed = false;
    for (cs) |*c| {
        if (!std.mem.eql(u8, c.name, "_rope")) continue;
        c.side = true;
        try t.expectError(error.ScratchAsyncLifetime, scratch.validate(cs));
        c.side = false;
        const original_name = c.name;
        c.name = "tf_dsv41_l2p.planted";
        try t.expectError(error.ScratchAsyncLifetime, scratch.validate(cs));
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
        try t.expectError(error.ScratchLifetimeOverlap, scratch.validate(cs));
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
    try p.add(try program(a, .{ .gm_v2 = .one, .limit = 1 << 20, .rope_rows = (1 << 20) + 2048 }, 4096, 0));
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
