//! The mirror's call frames and batches in tests: every call round-trips bit for bit and a cut one is refused.
const std = @import("std");
const lanes = @import("lanes");
const be = lanes.backend;
const Sampling = lanes.Sampling;
const codec = @import("cuda_mirror.zig");
const Call = codec.Call;
const StreamSpec = codec.StreamSpec;
const WindowCall = codec.WindowCall;
const DraftCall = codec.DraftCall;
const encode = codec.encode;
const decode = codec.decode;
const Batch = codec.Batch;
const Calls = codec.Calls;

pub fn roundTrip(arena: *std.heap.ArenaAllocator, call: Call) !Call {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    try encode(&list, gpa, call);
    // a frame cut anywhere is refused, never read past its end
    for (0..list.items.len) |cut| {
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        if (decode(scratch.allocator(), list.items[0..cut])) |_| return error.TestCutFrameDecoded else |_| {}
    }
    return decode(arena.allocator(), list.items);
}

test "every call survives its frame bit for bit, and a cut frame is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const ar = &arena;
    const smp: Sampling = .{ .seed = 0x7fff_0123_4567_89ab, .temperature = 0.6, .top_k = 20, .top_p = 0.95, .min_p = 0.05 };
    {
        const c = try roundTrip(ar, .{ .prefill = .{ .key = 0xdead_beef_0000_1111, .max_new = 4096, .drafts = true, .sampling = smp, .prompt = &.{ 1, 2, 3, 151643 }, .eos = &.{ 151645, 151643 }, .chunks = &.{2048} } });
        const p = c.prefill;
        try std.testing.expectEqual(@as(u64, 0xdead_beef_0000_1111), p.key);
        try std.testing.expectEqual(@as(u32, 4096), p.max_new);
        try std.testing.expect(p.drafts);
        try std.testing.expectEqual(smp, p.sampling.?);
        try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 151643 }, p.prompt);
        try std.testing.expectEqualSlices(u32, &.{ 151645, 151643 }, p.eos);
        try std.testing.expectEqualSlices(u32, &.{2048}, p.chunks);
        try std.testing.expectEqual(@as(u64, 0), p.resume_key);
        const r = try roundTrip(ar, .{ .prefill = .{ .key = 2, .max_new = 1, .drafts = true, .sampling = null, .prompt = &.{ 1, 2, 3 }, .eos = &.{}, .chunks = &.{}, .resume_key = 0xabcdef, .resume_at = 30000, .marks = &.{ 1500, 31000 } } });
        try std.testing.expectEqual(@as(u64, 0xabcdef), r.prefill.resume_key);
        try std.testing.expectEqual(@as(u32, 30000), r.prefill.resume_at);
        try std.testing.expectEqualSlices(u32, &.{ 1500, 31000 }, r.prefill.marks);
        const n = try roundTrip(ar, .{ .note = &.{ 1, 0, 0, 0, 0 } });
        try std.testing.expectEqualSlices(u8, &.{ 1, 0, 0, 0, 0 }, n.note);
    }
    {
        const c = try roundTrip(ar, .{ .prefill = .{ .key = 1, .max_new = 0, .drafts = false, .sampling = null, .prompt = &.{}, .eos = &.{}, .chunks = &.{} } });
        try std.testing.expect(c.prefill.sampling == null and !c.prefill.drafts and c.prefill.prompt.len == 0);
    }
    {
        const specs = [_]StreamSpec{
            .{ .key = 5, .max_new = 8, .drafts = true, .sampling = null, .prompt = &.{ 1, 2, 3 }, .eos = &.{9}, .chunks = &.{} },
            .{ .key = 6, .max_new = 9, .drafts = false, .sampling = null, .prompt = &.{ 4, 5 }, .eos = &.{}, .chunks = &.{} },
        };
        const c = try roundTrip(ar, .{ .prefill_many = &specs });
        try std.testing.expectEqual(@as(usize, 2), c.prefill_many.len);
        try std.testing.expectEqual(@as(u64, 6), c.prefill_many[1].key);
        try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, c.prefill_many[0].prompt);
        try std.testing.expectEqualSlices(u32, &.{ 4, 5 }, c.prefill_many[1].prompt);
    }
    {
        const c = try roundTrip(ar, .{ .first = .{ .key = 9, .position = 1 << 40 } });
        try std.testing.expectEqual(@as(u64, 9), c.first.key);
        try std.testing.expectEqual(@as(u64, 1 << 40), c.first.position);
    }
    {
        const c = try roundTrip(ar, .{ .queue = .{ .key = 3, .feed = .{ .handle = 77 }, .position = 12 } });
        try std.testing.expectEqual(be.Feed{ .handle = 77 }, c.queue.feed);
        const d = try roundTrip(ar, .{ .queue = .{ .key = 3, .feed = .{ .value = 151644 }, .position = 12 } });
        try std.testing.expectEqual(be.Feed{ .value = 151644 }, d.queue.feed);
    }
    const ws = [_]WindowCall{
        .{ .key = 5, .pending = 42, .held = 6, .tokens = &.{}, .parents = null, .positions = &.{ 100, 101, 102, 103, 104, 105, 106 }, .early = false },
        .{ .key = 6, .pending = 7, .held = 0, .tokens = &.{ 8, 9 }, .parents = &.{ -1, 0, 0 }, .positions = &.{ 10, 11, 11 }, .early = true },
    };
    {
        const c = try roundTrip(ar, .{ .verify = &ws });
        try std.testing.expectEqual(@as(usize, 2), c.verify.len);
        for (c.verify, ws) |g, w| {
            try std.testing.expectEqual(w.key, g.key);
            try std.testing.expectEqual(w.pending, g.pending);
            try std.testing.expectEqual(w.held, g.held);
            try std.testing.expectEqual(w.early, g.early);
            try std.testing.expectEqual(w.rows(), g.rows());
            try std.testing.expectEqualSlices(u32, w.tokens, g.tokens);
            try std.testing.expectEqualSlices(u64, w.positions, g.positions);
            try std.testing.expectEqual(w.parents == null, g.parents == null);
            if (w.parents) |p| try std.testing.expectEqualSlices(i32, p, g.parents.?);
        }
    }
    {
        const c = try roundTrip(ar, .{ .keep = .{ .windows = ws[0..1], .paths = &.{&.{ 0, 1, 2 }} } });
        try std.testing.expectEqual(@as(usize, 1), c.keep.windows.len);
        try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, c.keep.paths[0]);
    }
    {
        const rs = [_]DraftCall{
            .{ .key = 5, .follow = &.{}, .first = .{ .handle = 3 }, .rows = null, .start = 99, .position = 100, .depth = 15, .early = false, .ranks = false },
            .{ .key = 5, .follow = &.{ 4, 5, 6 }, .first = null, .rows = &.{ 0, 1, 2 }, .start = 100, .position = 104, .depth = 0, .early = true, .ranks = true },
        };
        const c = try roundTrip(ar, .{ .draft = &rs });
        try std.testing.expectEqual(@as(usize, 2), c.draft.len);
        try std.testing.expectEqual(be.Feed{ .handle = 3 }, c.draft[0].first.?);
        try std.testing.expect(c.draft[0].rows == null and c.draft[0].follow.len == 0);
        try std.testing.expectEqual(@as(u32, 15), c.draft[0].depth);
        try std.testing.expect(c.draft[1].first == null);
        try std.testing.expectEqualSlices(u32, &.{ 4, 5, 6 }, c.draft[1].follow);
        try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, c.draft[1].rows.?);
        try std.testing.expectEqual(@as(u64, 104), c.draft[1].position);
        try std.testing.expect(c.draft[1].early and c.draft[1].ranks);
    }
    {
        const c = try roundTrip(ar, .{ .release = 0xffff_ffff_ffff_fff0 });
        try std.testing.expectEqual(@as(u64, 0xffff_ffff_ffff_fff0), c.release);
        const d = try roundTrip(ar, .{ .unspeculate = 2 });
        try std.testing.expectEqual(@as(u64, 2), d.unspeculate);
    }
}

test "the frame layout is the documented one" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    try encode(&list, gpa, .{ .first = .{ .key = 0x0102030405060708, .position = 5 } });
    try std.testing.expectEqualSlices(u8, &.{ 2, 8, 7, 6, 5, 4, 3, 2, 1, 5, 0, 0, 0, 0, 0, 0, 0 }, list.items);
    list.clearRetainingCapacity();
    const w = [_]WindowCall{.{ .key = 1, .pending = 2, .held = 0, .tokens = &.{3}, .parents = null, .positions = &.{ 4, 5 }, .early = false }};
    try encode(&list, gpa, .{ .verify = &w });
    try std.testing.expectEqualSlices(u8, &(.{ 4, 1, 0, 0, 0 } ++ .{ 1, 0, 0, 0, 0, 0, 0, 0 } ++ .{ 2, 0, 0, 0 } ++ .{ 0, 0, 0, 0 } ++
        .{ 1, 0, 0, 0, 3, 0, 0, 0 } ++ .{0} ++ .{ 2, 0, 0, 0 } ++ .{ 4, 0, 0, 0, 0, 0, 0, 0 } ++ .{ 5, 0, 0, 0, 0, 0, 0, 0 } ++ .{0}), list.items);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    try std.testing.expectError(error.UnknownOp, decode(arena.allocator(), &.{99}));
    try std.testing.expectError(error.TrailingBytes, decode(arena.allocator(), &.{ 8, 1, 0, 0, 0, 0, 0, 0, 0, 0 }));
    try std.testing.expectError(error.BadFrame, decode(arena.allocator(), &.{ 3, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }));
}

test "a batch holds its calls in order and refuses a cut one" {
    const gpa = std.testing.allocator;
    var b: Batch = .{};
    defer b.deinit(gpa);
    try b.add(gpa, .{ .first = .{ .key = 1, .position = 2 } });
    try b.add(gpa, .{ .release = 7 });
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var calls = try Calls.of(b.bytes());
    const c1 = try decode(arena.allocator(), (try calls.next()).?);
    const c2 = try decode(arena.allocator(), (try calls.next()).?);
    try std.testing.expect((try calls.next()) == null);
    try std.testing.expectEqual(@as(u64, 2), c1.first.position);
    try std.testing.expectEqual(@as(u64, 7), c2.release);
    var cut = try Calls.of(b.bytes()[0 .. b.bytes().len - 1]);
    _ = try cut.next();
    try std.testing.expectError(error.ShortBatch, cut.next());
    b.clear();
    try std.testing.expectEqual(@as(usize, 0), b.bytes().len);
}
