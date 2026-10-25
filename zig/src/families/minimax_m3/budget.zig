//! A rank's memory budget for MiniMax-M3.1-Flash: weights, runtime, workspaces and host reserve first, the KV pool
//! the rest. Gives the largest context that fits, or the words that refuse an explicit CONTEXT that cannot.
const std = @import("std");
const kv = @import("core").kv_pool;
const pool = @import("pool.zig");
const dspark = @import("dspark_config.zig");

pub const GiB: u64 = 1 << 30;
const gib_f: f64 = @floatFromInt(GiB);

/// What a rank sets aside besides weights and KV, in bytes. Two conventions fill it in (below).
pub const Terms = struct {
    /// Memory the budget leaves free for the OS and the host: not spent on weights, KV or workspaces.
    reserve: u64,
    /// CUDA context, NCCL and graph memory outside the weight estimate.
    runtime: u64,
    /// Prompt workspaces other than the block-score chunk: rotation buffers, scratch, the expert plan.
    workspace: u64,
    /// The prompt cache's host-side bounded state and a restore's staging set.
    session_ram: u64 = 0,
    staging: u64 = 0,
    /// Rows of a prefill chunk and the index heads a rank scores: they size the block-score chunk (fp32 per block).
    prefill_rows: u32 = 8192,
    index_heads: u32 = 2,
    /// The dspark draft's weights, runtime and per-lane ring and scratch (dspark_config.cost); zero without a draft.
    draft: dspark.Cost = .{ .weights = 0, .runtime = 0, .per_lane = 0 },
};

/// `t` with the draft's cost on it: its weights and runtime are spent before any KV, each lane adds a ring.
pub fn withDraft(t: Terms, cost: dspark.Cost) Terms {
    var u = t;
    u.draft = cost;
    return u;
}

/// What the draft takes of a budget serving `lanes` streams.
pub fn draftBytes(t: Terms, lanes: u32) u64 {
    if (t.draft.weights == 0 and t.draft.runtime == 0 and t.draft.per_lane == 0) return 0;
    return t.draft.weights + t.draft.runtime + @as(u64, lanes) * t.draft.per_lane;
}

/// Unified memory's reserve default and floor (native/cuda_memory.zig: `floor(true)`), as the server applies them.
pub const unified_reserve_least_gib: f64 = 2;
pub const unified_reserve_floor: u64 = 4 * GiB;
pub const unified_reserve_share: u64 = 10;

pub const Error = error{BadReserve};

/// TENSORFOLD_MEMORY_RESERVE_GIB (2 GiB up to the memory's size), else a tenth of `total` and at least 4 GiB.
pub fn reserveBytes(text: ?[]const u8, total: u64) Error!u64 {
    const t = std.mem.trim(u8, text orelse "", " ");
    if (t.len == 0) return @max(unified_reserve_floor, total / unified_reserve_share);
    const gb = std.fmt.parseFloat(f64, t) catch return error.BadReserve;
    if (!(gb >= unified_reserve_least_gib)) return error.BadReserve;
    const bytes: u64 = @intFromFloat(@round(gb * gib_f));
    return if (bytes <= total) bytes else error.BadReserve;
}

/// TensorFold's admission: the reserve above, 4.5 GiB of CUDA context, NCCL and graphs (the GLM recipe measured
/// 3.5-4.8 GiB outside its estimate; plan section 5 takes 4-5), 1.5 GiB of prompt workspaces.
pub fn tensorfold(total: u64, reserve_text: ?[]const u8) Error!Terms {
    return .{ .reserve = try reserveBytes(reserve_text, total), .runtime = 4 * GiB + GiB / 2, .workspace = GiB + GiB / 2 };
}

/// The host's convention: its granted room is net of its reserve and limit, so only what it cannot see is priced here.
pub fn hosted() Terms {
    return .{ .reserve = 0, .runtime = 4 * GiB + GiB / 2, .workspace = GiB + GiB / 2 };
}

/// The DeepSeek sibling's priced floor (families/deepseek_v41/kv/budget.zig): a hard floor of 4 GiB MemAvailable kept
/// after boot, runtime 1.0, workspace 4.0 (graphs 1.2, prefill 1.3, select .25, drafting .35, slack .9), session RAM
/// .25 and the 128 MiB staging set. The workspace is its fitted total, so no block-score chunk is added on top.
pub fn dsv41() Terms {
    return .{ .reserve = 4 * GiB, .runtime = GiB, .workspace = 4 * GiB, .session_ram = GiB / 4, .staging = 128 << 20, .prefill_rows = 0 };
}

pub const Input = struct {
    /// MemAvailable before the weights load (or the budget the operator names).
    available: u64,
    /// The rank's weights: load.Plan.slotBytes().
    weights: u64,
    terms: Terms,
    lanes: u32 = 1,
    lanes_fixed: bool = false,
    /// Rows past the reply a stream reserves for its last verify window.
    window: u64 = pool.page_tokens,
    /// Prompt snapshot slots (null: kv_pool's default).
    snaps: ?usize = null,
    /// The model's positions (config.max_positions).
    max_positions: u64 = 1 << 20,
};

pub const Capacity = struct {
    /// Bytes past weights and every term, workspace for `context` included.
    room: u64,
    /// The block-score chunk and other workspaces at `context`, bytes.
    workspace: u64,
    /// The longest prompt + reply one stream may hold (pages x 128, less the verify window), at most max_positions.
    context: u64,
    lanes: u32,
    snaps: usize,
    pages: usize,
    pool_tokens: u64,
    pool_bytes: u64,
};

pub const Reason = enum { beyond_model, no_room, too_many_lanes, does_not_fit };

pub const Refusal = struct {
    reason: Reason,
    requested: u64,
    /// The largest context that would fit, and the pool it brings (0: none).
    largest: u64,
    largest_pool_tokens: u64,
    /// What the request's pages cost, and what the budget leaves for KV at that context.
    need_bytes: u64,
    room_bytes: u64,
    weights: u64,
    available: u64,
    max_positions: u64,
    token_bytes: u64,
    lanes: u32,
    /// Bytes of the budget the dspark draft takes (0: no draft), named in the words so the operator can drop it.
    draft: u64 = 0,

    pub fn format(r: Refusal, w: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (r.reason) {
            .beyond_model => try w.print("CONTEXT={d} is past the model's {d} positions: use at most {d}", .{ r.requested, r.max_positions, r.max_positions }),
            .no_room => {
                try w.print("the budget of {d:.2} GiB leaves no KV room beside the weights ({d:.2} GiB), the runtime and the reserve: free memory or lower the reserve", .{ g(r.available), g(r.weights) });
                try draftClause(r, w);
            },
            .too_many_lanes => try w.print("--parallel {d} needs a page and a table row for each extra stream, and the budget leaves {d:.2} GiB for KV: serve fewer streams", .{ r.lanes, g(r.room_bytes) }),
            .does_not_fit => try w.print(
                "CONTEXT={d} needs {d:.2} GiB of KV a rank ({d} bytes a token) but the budget of {d:.2} GiB leaves {d:.2} GiB after the weights ({d:.2} GiB), runtime, workspaces and reserve; the largest context that fits is {d} tokens (pool {d} tokens): set CONTEXT to at most {d}",
                .{ r.requested, g(r.need_bytes), r.token_bytes, g(r.available), g(r.room_bytes), g(r.weights), r.largest, r.largest_pool_tokens, r.largest },
            ),
        }
        if (r.reason == .does_not_fit) try draftClause(r, w);
    }

    fn draftClause(r: Refusal, w: *std.Io.Writer) std.Io.Writer.Error!void {
        if (r.draft != 0) try w.print("; the dspark draft takes {d:.2} GiB of that budget: serve without --drafter to use it for KV", .{g(r.draft)});
    }
};

fn g(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / gib_f;
}

pub const Verdict = union(enum) { ok: Capacity, refused: Refusal };

/// The block-score chunk (fp32 per index head, row and 128-key block) a prefill chunk of the terms' rows needs at `context`.
pub fn scoreBytes(t: Terms, context: u64) u64 {
    const rows = @min(@as(u64, t.prefill_rows), context);
    return rows * t.index_heads * @as(u64, @intCast(pages(context))) * 4;
}

fn pages(tokens: u64) usize {
    return @intCast((tokens + pool.page_tokens - 1) / pool.page_tokens);
}

pub fn workspaceBytes(t: Terms, context: u64) u64 {
    return t.workspace + scoreBytes(t, context);
}

/// What the budget leaves for the pool at `context`; null when the fixed terms alone pass it.
fn roomFor(in: Input, context: u64) ?u64 {
    const t = in.terms;
    const spent = std.math.add(u64, in.weights, t.reserve) catch return null;
    const more = t.runtime + t.session_ram + t.staging + workspaceBytes(t, context) + draftBytes(t, in.lanes);
    const all = std.math.add(u64, spent, more) catch return null;
    return if (all > in.available) null else in.available - all;
}

/// The pool a stream of `p` pages gets, or why not.
fn grant(d: pool.Dims, in: Input, p: usize) error{ NoRoom, TooMany }!struct { cap: Capacity, plan: kv.Plan } {
    const context = contextOf(in, p);
    const room = roomFor(in, context) orelse return error.NoRoom;
    const l = d.layout(p);
    const snaps = in.snaps orelse kv.most_snaps;
    const plan = kv.plan(l, .{ .room = room, .engine = 0, .lanes = in.lanes, .fixed = in.lanes_fixed, .cache = @as(u64, @min(snaps, kv.most_snaps)) * l.snap_bytes, .snaps = in.snaps }) catch |e| return switch (e) {
        error.NoStream => error.NoRoom,
        error.TooMany => error.TooMany,
    };
    // the stream's own p pages, and a page for each other stream that was granted
    if (plan.pages < p + (plan.lanes - 1)) return error.NoRoom;
    return .{ .cap = .{
        .room = room,
        .workspace = workspaceBytes(in.terms, context),
        .context = context,
        .lanes = plan.lanes,
        .snaps = plan.snaps,
        .pages = plan.pages,
        .pool_tokens = @as(u64, plan.pages) * pool.page_tokens,
        .pool_bytes = @as(u64, plan.pages) * l.pageBytes(),
    }, .plan = plan };
}

/// The context a stream of `p` pages holds: its pages less the verify window, at most max_positions.
fn contextOf(in: Input, p: usize) u64 {
    return @min(in.max_positions, (@as(u64, p) * pool.page_tokens) -| in.window);
}

/// The most pages a stream may have (the model's positions plus the window).
fn mostPages(in: Input) usize {
    return pages(in.max_positions + in.window);
}

/// Binary search for the most pages that fit (monotone: more pages only shrink the room and add need).
fn largestPages(d: pool.Dims, in: Input) ?usize {
    var lo: usize = 0; // fits
    var hi: usize = mostPages(in) + 1; // does not fit
    if (grant(d, in, 1)) |_| {} else |_| return null;
    lo = 1;
    while (hi - lo > 1) {
        const mid = lo + (hi - lo) / 2;
        if (grant(d, in, mid)) |_| lo = mid else |_| hi = mid;
    }
    return lo;
}

/// The largest context (explicit `context` null) or the verdict on `context`: its capacity, or a refusal that names
/// the largest context fitting the same budget.
pub fn check(d: pool.Dims, in: Input, context: ?u64) Verdict {
    const best = largestPages(d, in);
    const largest: u64 = if (best) |p| contextOf(in, p) else 0;
    var refuse: Refusal = .{
        .reason = .no_room,
        .requested = context orelse 0,
        .largest = largest,
        .largest_pool_tokens = 0,
        .need_bytes = 0,
        .room_bytes = roomFor(in, 0) orelse 0,
        .weights = in.weights,
        .available = in.available,
        .max_positions = in.max_positions,
        .token_bytes = d.tokenBytes(),
        .lanes = in.lanes,
        .draft = draftBytes(in.terms, in.lanes),
    };
    if (best) |p| refuse.largest_pool_tokens = (grant(d, in, p) catch unreachable).cap.pool_tokens;
    const want = context orelse largest;
    if (context) |c| if (c > in.max_positions) {
        refuse.reason = .beyond_model;
        return .{ .refused = refuse };
    };
    if (best == null) {
        refuse.reason = if (in.lanes > 1 and in.lanes_fixed) .too_many_lanes else .no_room;
        return .{ .refused = refuse };
    }
    const p = pages(want + in.window);
    refuse.need_bytes = @as(u64, p) * d.pageBytes();
    refuse.room_bytes = roomFor(in, want) orelse 0;
    if (grant(d, in, p)) |ok| {
        var cap = ok.cap;
        cap.context = want;
        return .{ .ok = cap };
    } else |e| {
        refuse.reason = if (e == error.TooMany) .too_many_lanes else .does_not_fit;
        return .{ .refused = refuse };
    }
}

/// The text of a refusal, for the server's startup message.
pub fn refusalText(a: std.mem.Allocator, r: Refusal) ![]u8 {
    return std.fmt.allocPrint(a, "{f}", .{r});
}
