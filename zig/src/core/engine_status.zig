//! Engine readiness and status values, read without entering a device context.

/// Thread-safe backend readiness; no GPU context or blocking device work is allowed on HTTP threads.
pub const HealthSource = struct {
    ctx: *anyopaque,
    read: *const fn (*anyopaque) bool,
};

pub const Status = struct {
    healthy: bool = true,
    /// Requests in prefill or decode, and those waiting for a lane.
    running: u32 = 0,
    waiting: u32 = 0,
    decode_tokens_per_second: f64 = 0,
    prefill_tokens_per_second: f64 = 0,
    /// Requests that gave a lane up to a later one; null when the engine does not count them.
    preemptions: ?u64 = null,
    warming: bool = false,
    /// Live streams written to the caller's buffer: tokens each holds.
    streams: usize = 0,
    /// Generated tokens held by live streams, excluding their prompts.
    generation_tokens: u64 = 0,
    /// Filling prompts' total tokens and those still to prefill (null: no prompt is going in).
    fill_prompt_tokens: ?u32 = null,
    fill_tokens_left: ?u32 = null,
    /// A prompt pass is running whose backend reports no `fill_at`: the totals stay null rather than guessed.
    fill_unknown: bool = false,
    /// Changes whenever the filling set does: a reader pairs its drain samples with one fill.
    fill_run: u64 = 0,
    /// The oldest queued request's wait in seconds (0: none queued).
    oldest_queued_s: f64 = 0,
    /// The round pace against its own baseline (core/pace.zig); null: the engine keeps none.
    pace: ?PaceStatus = null,
    /// The backend's graph caches, the first `graph_caches` of them (0: it keeps none).
    graphs: [4]GraphStatus = @splat(.{}),
    graph_caches: u8 = 0,
};

pub const PaceStatus = struct {
    /// Seconds since the last round ended (null: none yet) and its length.
    last_round_age_s: ?f64 = null,
    last_round_s: f64 = 0,
    /// Decode round seconds, quick average and baseline (0: not enough rounds yet).
    decode_s: f64 = 0,
    decode_base_s: f64 = 0,
    /// Prompt rows a second, quick average and baseline (0: not enough pieces yet).
    prefill_rows_s: f64 = 0,
    prefill_base_rows_s: f64 = 0,
    /// How long the pace has been past its limits (0: it is not).
    slow_s: f64 = 0,
};

/// One graph cache's counts for /health (a backend's keyed CUDA graphs).
pub const GraphStatus = struct {
    name: []const u8 = "",
    held: u32 = 0,
    captured: u64 = 0,
    replayed: u64 = 0,
    eager: u64 = 0,
    floor_hits: u64 = 0,
    /// A new key last met no room on some rank (it ran eager); false again once a capture had room.
    under_floor: bool = false,
};

/// A backend's graph caches for LaneHost; read from HTTP threads, so it must not touch the GPU.
pub const GraphsSource = struct {
    ctx: ?*anyopaque = null,
    /// Fills `out` and returns how many it filled.
    read: *const fn (ctx: ?*anyopaque, out: []GraphStatus) usize,
};
