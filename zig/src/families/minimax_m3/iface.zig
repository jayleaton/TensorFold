//! The contract the MiniMax-M3.1-Flash engine, forward, prefill, backend, follower, weights and kernels share.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const pool = @import("pool.zig");
const ranks = @import("ranks.zig");
const sparse = cuda.minimax_m3_sparse;

pub const Sampling = lanes.Sampling;
pub const Paged = sparse.Paged;
pub const Rows = sparse.Rows;
pub const Scratch = sparse.Scratch;

// M12: a prompt chunk, whole pages (GLM's fw.prefill_rows).
pub const prefill_rows: u32 = 2048;
// Rows of one verify window; UNVERIFIED on a GPU (widths 1, 2, 3, 16, 17, 64, 128 are its gate).
pub const window_rows: u32 = 16;
// Rows of one forward shared by several streams' windows (GLM: 32), and the streams in it.
pub const shared_rows: u32 = 64;
pub const max_shared: u32 = 16;
// Widest forward any caller makes; the router, expert and sparse scratch are sized by it.
pub const max_rows: u32 = prefill_rows;

comptime {
    if (prefill_rows % pool.page_tokens != 0) @compileError("a prompt chunk must be whole pages: marks and resumes sit on pages");
    if (window_rows > shared_rows or shared_rows > prefill_rows) @compileError("window <= shared <= chunk");
    if (max_shared * window_rows < shared_rows) @compileError("a shared forward must be fillable by max_shared windows");
}

/// The 13 mismatches of the GLM-reuse study, resolved; the prose is docs/interface-decisions.md.
pub const decisions = struct {
    // M1: every stream, the engine's own too, is a slot of the paged pool; no full-window sequence lives outside it.
    pub const all_paged = true;
    pub const free_streams: u32 = 0;
    // M2: the backend owns a radix prompt cache over the pool's pages; the host builds no `pc.Store`.
    pub const owns_prompt_cache = true;
    // M3: one int32 device table of slots x max_pages entries; a window's row reads at slot * max_pages.
    pub const table_entry_bytes: usize = 4;
    // M5: the T0 Switches stay at their compile-time defaults; none is runtime-selectable, so none is fingerprinted.
    pub const switches_runtime = false;
    // M7: Layer.shared_input stays false (correct, slower) until a device-side check of the gate/up scales exists.
    pub const shared_input = false;
    // M11: TF_ROCE and TF_ROCE_MAX_KB are in Settings and in the fingerprint (ranks.zig).
    pub const roce_in_fingerprint = true;
    // M13: the head is cut in 128-row units; logits cross ranks in LogitsGather slots, greedy picks in mergePicks.
    pub const head_unit: usize = 128;
};

// One stream's window of a shared verify: its slot, its ids and its drawn tokens (GLM's Part).
pub const Part = struct { slot: u32, ids: []const u32, out: []u32, bits: u64 = 0 };
// Where a prompt pass resumes: `at` rows already in the stream's adopted pages (a multiple of the page); no state id.
pub const Resume = struct { at: usize };
// Rank 0's marks in a prompt pass: `call` may keep the prefix at a mark (a page multiple).
pub const Marks = struct { at: []const u32, ctx: *anyopaque, call: *const fn (ctx: *anyopaque, at: u32) void };
pub const Cancel = struct { ptr: ?*anyopaque, check: *const fn (?*anyopaque) bool };
// One stream's rows in a shared forward: rows [row0, row0 + rows) of the batch belong to `slot`.
pub const Seg = struct { slot: u32, row0: u32, rows: u32 };

// What `Engine.init` takes besides the device and the checkpoint.
pub const EngineOptions = struct {
    capacity: usize, // tokens a sequence holds (context + window_rows, whole pages)
    settings: ranks.Settings,
    slots: u32 = 1, // streams the block table has rows for; `attach` raises it to the planned streams
    drafter: ?[]const u8 = null, // the dspark draft's folder (the same on every rank): its taps, weights and rings load with the engine
};

// M3: entry offset of `slot`'s row in the device table (`Paged.row_tab` holds one per batch row).
pub fn tableOffset(slot: u32, max_pages: usize) i32 {
    return @intCast(@as(usize, slot) * max_pages);
}

// Fills `out[row0 .. row0 + rows]` of every segment with that segment's table offset.
pub fn rowTabs(out: []i32, segs: []const Seg, max_pages: usize) void {
    for (segs) |s| @memset(out[s.row0..][0..s.rows], tableOffset(s.slot, max_pages));
}

// Pages a rank's table row holds: whole pages of the capacity.
pub fn maxPages(capacity: usize) usize {
    return (capacity + pool.page_tokens - 1) / pool.page_tokens;
}

// Bytes of the whole device table for `slots` streams (the layout's fixed part is one row of it, per stream).
pub fn tableBytes(slots: usize, max_pages: usize) usize {
    return slots * max_pages * decisions.table_entry_bytes;
}

pub const Kind = enum(u32) { begin = 1, prefill, chunk, verify, shared, keep, release, end, idle, absorb, draft, snap };

// Tapped rows of the dspark draft: a chunk at rows [0, prefill_rows), a stream's last window at its slot's rows.
pub fn tapWindowRow(slot: u32) u32 {
    return prefill_rows + slot * window_rows;
}

// Rows the taps buffer holds for `slots` streams.
pub fn tapRows(slots: u32) usize {
    return @as(usize, prefill_rows) + @as(usize, slots) * window_rows;
}

// One step as it crosses: 64 bytes, then `n` u32 ids.
pub const Header = extern struct {
    kind: u32,
    n: u32 = 0, // ids that follow
    depth: u32 = 0, // chunk: where it starts; shared: its parts
    flags: u32 = 0, // begin: sampled; verify, shared: whole rows
    a: u64 = 0, // keep: rows; prefill, chunk: the rows resumed from the adopted pages
    seed: u64 = 0, // begin: the sampling seed
    temperature: f64 = 0,
    top_p: f64 = 0,
    min_p: f64 = 0,
    top_k: u32 = 0, // begin: top_k; chunk: the prompt's length
    slot: u32 = 0, // the stream's slot the step runs on

    comptime {
        std.debug.assert(@sizeOf(Header) == 64);
    }
};

pub fn laneFacts(window: u32, verify: u32) lanes.Model {
    return .{ .exact_width = window, .hidden_rows = true, .batch_rows = verify, .max_streams = max_shared, .streams_exact = true, .copy_match = 5, .context_copies = true, .lone_cost_tile = 16 };
}

// One method the backend, prefill and follower call; `conformsEngine` compares each against it.
const Method = struct { name: []const u8, args: []const type, ret: type };

fn methods(comptime E: type) []const Method {
    const gpa = std.mem.Allocator;
    return &[_]Method{
        .{ .name = "init", .args = &.{ gpa, std.Io, *const cuda.Driver, *const cuda.Context, []const u8, *const cuda.nccl.Library, cuda.nccl.Comm, usize, usize, EngineOptions }, .ret = *E },
        .{ .name = "deinit", .args = &.{*E}, .ret = void },
        .{ .name = "bind", .args = &.{ *E, u32 }, .ret = void },
        .{ .name = "release", .args = &.{ *E, u32 }, .ret = void },
        .{ .name = "seqBytes", .args = &.{*const E}, .ret = u64 },
        .{ .name = "pos", .args = &.{*const E}, .ret = usize },
        .{ .name = "begin", .args = &.{ *E, ?Sampling }, .ret = void },
        .{ .name = "beginFollowing", .args = &.{ *E, ?Sampling }, .ret = void },
        .{ .name = "prefill", .args = &.{ *E, []const u32, ?Resume, ?Cancel, ?Marks }, .ret = u32 },
        .{ .name = "prefillChunk", .args = &.{ *E, []const u32, usize, usize, ?Resume, ?Cancel, ?Marks }, .ret = ?u32 },
        .{ .name = "gate", .args = &.{ *E, ?Cancel }, .ret = void },
        .{ .name = "verify", .args = &.{ *E, []const u32, []u32, u64 }, .ret = void },
        .{ .name = "verifyShared", .args = &.{ *E, []const Part }, .ret = void },
        .{ .name = "keep", .args = &.{ *E, usize }, .ret = void },
        .{ .name = "draw", .args = &.{ *E, u64, usize, u64, u64, []u32 }, .ret = void },
        .{ .name = "checkGathers", .args = &.{*const E}, .ret = void },
    };
}

fn payload(comptime R: type) type {
    return if (@typeInfo(R) == .error_union) @typeInfo(R).error_union.payload else R;
}

// Compile error naming the first method of `E` whose parameters or result differ from the contract.
pub fn conformsEngine(comptime E: type) void {
    inline for (comptime methods(E)) |m| {
        if (!@hasDecl(E, m.name)) @compileError("Engine lacks " ++ m.name);
        const info = @typeInfo(@TypeOf(@field(E, m.name))).@"fn";
        if (info.param_types.len != m.args.len) @compileError("Engine." ++ m.name ++ " takes a different number of parameters");
        inline for (info.param_types, m.args) |p, want| {
            if (p.? != want) @compileError("Engine." ++ m.name ++ " has a parameter of type " ++ @typeName(p.?) ++ ", the contract has " ++ @typeName(want));
        }
        if (payload(info.return_type.?) != m.ret) @compileError("Engine." ++ m.name ++ " returns " ++ @typeName(payload(info.return_type.?)) ++ ", the contract has " ++ @typeName(m.ret));
    }
}

// What `native/cuda.zig` reads from this family's namespace (the `@hasDecl` surface); a compile error names the gap.
pub fn conformsNative(comptime F: type) void {
    inline for (.{ "model_type", "formats", "default_context", "max_segments", "Options", "Loaded", "open", "explain", "explains_open", "weightShare", "capacity", "attach", "fits", "window_rows", "owns_prompt_cache", "cache_limit" }) |name| {
        if (!@hasDecl(F, name)) @compileError("the family namespace lacks " ++ name);
    }
    if (!@hasField(F.Options, "context") or !@hasField(F.Options, "grant") or !@hasField(F.Options, "lanes")) @compileError("Options needs context, grant, lanes");
    inline for (.{ "backend", "facts", "rows", "stream_bytes", "free_streams", "ctx", "deinit", "layout" }) |name| {
        if (!@hasField(F.Loaded, name)) @compileError("Loaded lacks " ++ name);
    }
}

const Fake = struct {
    pub fn init(_: std.mem.Allocator, _: std.Io, _: *const cuda.Driver, _: *const cuda.Context, _: []const u8, _: *const cuda.nccl.Library, _: cuda.nccl.Comm, _: usize, _: usize, _: EngineOptions) !*Fake {
        return error.Unused;
    }
    pub fn deinit(_: *Fake) void {}
    pub fn bind(_: *Fake, _: u32) !void {}
    pub fn release(_: *Fake, _: u32) void {}
    pub fn seqBytes(_: *const Fake) u64 {
        return 0;
    }
    pub fn pos(_: *const Fake) usize {
        return 0;
    }
    pub fn begin(_: *Fake, _: ?Sampling) !void {}
    pub fn beginFollowing(_: *Fake, _: ?Sampling) !void {}
    pub fn prefill(_: *Fake, _: []const u32, _: ?Resume, _: ?Cancel, _: ?Marks) !u32 {
        return 0;
    }
    pub fn prefillChunk(_: *Fake, _: []const u32, _: usize, _: usize, _: ?Resume, _: ?Cancel, _: ?Marks) !?u32 {
        return null;
    }
    pub fn gate(_: *Fake, _: ?Cancel) !void {}
    pub fn verify(_: *Fake, _: []const u32, _: []u32, _: u64) !void {}
    pub fn verifyShared(_: *Fake, _: []const Part) !void {}
    pub fn keep(_: *Fake, _: usize) !void {}
    pub fn draw(_: *Fake, _: u64, _: usize, _: u64, _: u64, _: []u32) !void {}
    pub fn checkGathers(_: *const Fake) !void {}
};

test "an engine with every method of the contract conforms (a wrong one is a compile error, see the decls above)" {
    conformsEngine(Fake);
}

test "the follower's header is the 64 bytes GLM's is, and its kinds start at 1" {
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(Header));
    try std.testing.expectEqual(@as(u32, 1), @backingInt(Kind.begin));
    try std.testing.expectEqual(@as(u32, 9), @backingInt(Kind.idle));
    try std.testing.expectEqual(@as(u32, 12), @backingInt(Kind.snap));
    try std.testing.expectEqual(@as(u32, 2048 + 3 * 16), tapWindowRow(3));
    try std.testing.expectEqual(@as(usize, 2048 + 16), tapRows(1));
}

test "M3: a shared window's rows read their own slot's table row" {
    var out = [_]i32{ -1, -1, -1, -1, -1, -1 };
    rowTabs(&out, &.{ .{ .slot = 2, .row0 = 0, .rows = 3 }, .{ .slot = 0, .row0 = 3, .rows = 2 } }, 512);
    try std.testing.expectEqualSlices(i32, &.{ 1024, 1024, 1024, 0, 0, -1 }, &out);
    try std.testing.expectEqual(@as(usize, 513), maxPages(65536 + window_rows));
    try std.testing.expectEqual(@as(usize, 3 * 512 * 4), tableBytes(3, 512));
}

test "M4: the layout's page is the geometry's page and its fixed part is one table row, per stream" {
    const cfg = &@import("config.zig").flash;
    const dims = try pool.Dims.of(cfg, try @import("shard.zig").Share.of(cfg, 0, 2));
    const l = dims.layout(512);
    try std.testing.expectEqual(dims.pageBytes(), l.pageBytes());
    try std.testing.expectEqual(@as(u64, tableBytes(1, 512)), l.fixed_bytes);
    try std.testing.expectEqual(@as(u64, 76_800), dims.tokenBytes());
}

test "M12/M1: widths nest and the lane facts keep windows exact over several streams" {
    const f = laneFacts(window_rows, shared_rows);
    try std.testing.expectEqual(window_rows, f.exact_width);
    try std.testing.expectEqual(shared_rows, f.batch_rows);
    try std.testing.expect(f.hidden_rows and f.streams_exact and !f.mtp);
    try std.testing.expect(decisions.all_paged and decisions.free_streams == 0);
}
