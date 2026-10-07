//! Prod's engine knobs (dsv41-tensorfold-spark config/prod.env) that the Zig engine reads under the Python engine's
//! names, where the emitters' prod path (M1_KNOBS=prod) does not already fix them (docs DSV41-ZIG-PORT.md, "Prod knob
//! parity"). Every rank reads the same environment, so the ranks agree; each knob defaults to today's behaviour.
//!
//! - `TF_DSV41_PF_DENSE` = off | fused, `_CFG` = id[,group], `_TABLE` = a JSON tuning table (pfdense.py): the dense EXL3
//!   prefill GEMM with the trellis decoded inside it (`tf_dsv41_pfdense_v1.gemm`) instead of the W_q unpack + Triton
//!   `_gemm` in column blocks. The same bits by construction (pfdense.py's exactness notes: the same mma chain over the
//!   same k16 slices from +0, the same epilogue); the configuration only moves where an output is computed.
//! - `TF_DSV41_PREFILL_CHUNK` / `TF_DSV41_PREFILL_ROWS`: a prompt segment's rows (the smaller of the two; the boot
//!   buffer plan's prefill program is sized for it). Prefill is segmentation-invariant (memory.py, G19).
//! - `TF_DSV41_PREFETCH_AHEAD`: a prompt segment's Engram reads issued before the previous segment runs (prefetch.py).
//! - `TF_DSV41_PREFILL`: full (Zig's prefill) | replay (Python's CED replay, not ported: warned at boot).
const std = @import("std");
const dk = @import("dsv41_kernels");

const pfd = dk.ops.pfd;

pub const Error = error{ BadKnob, BadTable };

/// Reads one setting; null when unset or empty (the environment in `env`, a fixture in tests).
pub const Get = *const fn ([]const u8) ?[]const u8;

/// pfdense.SMS: the heuristic prices waves on GB10's 48 SMs on every GPU (the pick never changes a bit, but the
/// launches then match Python's on the pods' captures too).
pub const pfd_sms: usize = 48;

/// The fused dense prefill GEMM's choices (pfdense.pick): an override, the JSON table, else the heuristic.
pub const PfDense = struct {
    entries: []const Entry = &.{},
    /// TF_DSV41_PF_DENSE_CFG: (id, group) for every shape
    override: ?[2]u32 = null,

    pub const Entry = struct { k: u32, n: u32, cfg: u32, group: u32 = 8, small: ?u32 = null, small_rows: u32 = 0 };

    /// pfdense.why_not's shape and width rules (the codebook is mul1 for every dense group of our packs).
    pub fn eligible(k: i64, n: i64, k2: u32, lanes: bool) bool {
        const widths: []const u32 = if (lanes) &.{ 8, 10, 12 } else &.{ 8, 10, 12, 16 };
        if (std.mem.indexOfScalar(u32, widths, k2) == null) return false;
        return k > 0 and n > 0 and @rem(k, 128) == 0 and @rem(n, 128) == 0;
    }

    /// (cfg, group) for an [m, k] @ [k, n] call: the override, the table (its `small` at m <= small_rows), else
    /// the heuristic; a choice whose stage does not divide K falls back to the heuristic.
    pub fn pick(p: *const PfDense, k: usize, n: usize, k2: u32, m: usize) [2]u32 {
        var got: ?[2]u32 = p.override;
        if (got == null) for (p.entries) |e| {
            if (e.k != k or e.n != n) continue;
            const c = if (e.small != null and m <= e.small_rows) e.small.? else e.cfg;
            got = .{ c, e.group };
            break;
        };
        if (got) |g| if (g[0] < pfd.cfgs.len and k % (16 * pfd.cfgs[g[0]][3]) == 0) return g;
        const h = pfd.heuristic(k, n, k2, m, pfd_sms);
        return .{ @intCast(h[0]), @intCast(h[1]) };
    }

    /// pfdense._table_file: `{"K,N": {"cfg": id, "group": g, "small": id, "small_rows": r}}`.
    pub fn parseTable(a: std.mem.Allocator, json: []const u8) ![]const Entry {
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.BadTable;
        if (parsed != .object) return error.BadTable;
        const out = try a.alloc(Entry, parsed.object.count());
        var it = parsed.object.iterator();
        var i: usize = 0;
        while (it.next()) |kv| : (i += 1) {
            var ks = std.mem.tokenizeScalar(u8, kv.key_ptr.*, ',');
            const k = std.fmt.parseInt(u32, std.mem.trim(u8, ks.next() orelse return error.BadTable, " "), 10) catch return error.BadTable;
            const n = std.fmt.parseInt(u32, std.mem.trim(u8, ks.next() orelse return error.BadTable, " "), 10) catch return error.BadTable;
            if (kv.value_ptr.* != .object) return error.BadTable;
            const o = kv.value_ptr.object;
            out[i] = .{ .k = k, .n = n, .cfg = try field(o, "cfg") orelse return error.BadTable };
            if (try field(o, "group")) |g| out[i].group = g;
            out[i].small = try field(o, "small");
            out[i].small_rows = (try field(o, "small_rows")) orelse 0;
        }
        return out;
    }

    fn field(o: std.json.ObjectMap, name: []const u8) !?u32 {
        const v = o.get(name) orelse return null;
        return switch (v) {
            .integer => |x| std.math.cast(u32, x) orelse error.BadTable,
            else => error.BadTable,
        };
    }

    fn parseOverride(raw: []const u8) ![2]u32 {
        var it = std.mem.tokenizeScalar(u8, raw, ',');
        const id = std.fmt.parseInt(u32, std.mem.trim(u8, it.next() orelse return error.BadKnob, " "), 10) catch return error.BadKnob;
        const g: u32 = if (it.next()) |x| std.fmt.parseInt(u32, std.mem.trim(u8, x, " "), 10) catch return error.BadKnob else 8;
        if (it.next() != null or id >= pfd.cfgs.len or g < 1 or g > 1024) return error.BadKnob;
        return .{ id, g };
    }
};

/// TF_DSV41_PF_DENSE (off | fused; 0 / no / false = off): the fused path's settings, null when off. The table file
/// is read through `io` (its bytes and the table owned by `a`).
pub fn pfDense(a: std.mem.Allocator, io: std.Io, get: Get) !?*const PfDense {
    const raw = get("TF_DSV41_PF_DENSE") orelse return null;
    const v = std.mem.trim(u8, raw, " ");
    const off = [_][]const u8{ "off", "0", "no", "false" };
    for (off) |o| if (std.ascii.eqlIgnoreCase(v, o)) return null;
    if (!std.ascii.eqlIgnoreCase(v, "fused")) {
        std.log.warn("TF_DSV41_PF_DENSE={s}: expected off or fused", .{raw});
        return error.BadKnob;
    }
    const p = try a.create(PfDense);
    p.* = .{};
    if (get("TF_DSV41_PF_DENSE_CFG")) |c| p.override = PfDense.parseOverride(c) catch {
        std.log.warn("TF_DSV41_PF_DENSE_CFG={s}: expected id[,group], id 0-{d}, group 1-1024", .{ c, pfd.cfgs.len - 1 });
        return error.BadKnob;
    };
    if (get("TF_DSV41_PF_DENSE_TABLE")) |path| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24)) catch |e| {
            std.log.warn("TF_DSV41_PF_DENSE_TABLE={s}: {t}", .{ path, e });
            return e;
        };
        p.entries = PfDense.parseTable(a, bytes) catch |e| {
            std.log.warn("TF_DSV41_PF_DENSE_TABLE={s}: not a pfdense tuning table ({t})", .{ path, e });
            return e;
        };
    }
    return p;
}

/// A prompt segment's rows: min(TF_DSV41_PREFILL_CHUNK, TF_DSV41_PREFILL_ROWS), on the 16-row grid, 32-2,048; null
/// when neither is set (the forward's default, 2,048).
pub fn prefillRows(get: Get) !?u32 {
    var rows: ?u32 = null;
    for ([_][]const u8{ "TF_DSV41_PREFILL_CHUNK", "TF_DSV41_PREFILL_ROWS" }) |name| if (get(name)) |raw| {
        const v = std.fmt.parseInt(u32, std.mem.trim(u8, raw, " "), 10) catch return badRows(name, raw);
        if (v < 32 or v > 2048 or v % 16 != 0) return badRows(name, raw);
        rows = if (rows) |r| @min(r, v) else v;
    };
    return rows;
}

fn badRows(name: []const u8, raw: []const u8) error{BadKnob} {
    std.log.warn("{s}={s}: expected a multiple of 16 in 32..2048", .{ name, raw });
    return error.BadKnob;
}

/// TF_DSV41_PREFETCH_AHEAD (0 by default): the next prompt segment's Engram reads before this one runs.
pub fn prefetchAhead(get: Get) bool {
    const v = get("TF_DSV41_PREFETCH_AHEAD") orelse return false;
    return !std.mem.eql(u8, std.mem.trim(u8, v, " "), "0");
}

/// prod-perf1's decode R1 under Python's names (ATTN_SPLIT=4 and the seven switches at 1), or TF_DSV41_R1 (ours,
/// which wins): Zig runs R1 whole or not at all, so a partial set is logged and runs without it (R1 is the same bits).
pub fn r1(get: Get) bool {
    if (get("TF_DSV41_R1")) |v| return std.mem.eql(u8, std.mem.trim(u8, v, " "), "1");
    const names = [_][]const u8{ "TF_DSV41_ATTN_SPLIT", "TF_DSV41_ATTN_ROPE", "TF_DSV41_MHC_TAIL", "TF_DSV41_MHC_DEFER", "TF_DSV41_MOE_BF16", "TF_DSV41_PRUNE_KIT", "TF_DSV41_RG_PRUNE", "TF_DSV41_RMS_FOLD" };
    var set: usize = 0;
    var prod: usize = 0;
    for (names, 0..) |n, i| if (get(n)) |raw| {
        set += 1;
        if (std.mem.eql(u8, std.mem.trim(u8, raw, " "), if (i == 0) "4" else "1")) prod += 1;
    };
    if (prod == names.len) return true;
    if (set > 0) std.log.warn("decode R1: {d} of its {d} Python knobs at prod's values; Zig runs R1 whole (TF_DSV41_R1=1, or all of ATTN_SPLIT=4 ATTN_ROPE MHC_TAIL MHC_DEFER MOE_BF16 PRUNE_KIT RG_PRUNE RMS_FOLD = 1) or not at all: off", .{ prod, names.len });
    return false;
}

/// TF_DSV41_INDEX_BUDGET_MIB (Python's default 256; prod 64): the indexer's scores and sort keys a launch, in bytes;
/// prefill's row blocks follow it (csa2 backend; the same bits at any budget), so its scratch shrinks with it.
pub fn indexBudget(get: Get) !?i64 {
    const raw = get("TF_DSV41_INDEX_BUDGET_MIB") orelse return null;
    const v = std.fmt.parseInt(i64, std.mem.trim(u8, raw, " "), 10) catch return badBudget(raw);
    if (v < 1 or v > 4096) return badBudget(raw);
    return v << 20;
}

fn badBudget(raw: []const u8) error{BadKnob} {
    std.log.warn("TF_DSV41_INDEX_BUDGET_MIB={s}: expected 1..4096", .{raw});
    return error.BadKnob;
}

pub const PrefillMode = enum { full, replay };

/// TF_DSV41_PREFILL (Python's default: full). Zig runs full prefill whatever it says; `replay` is logged at boot.
pub fn prefillMode(get: Get) !PrefillMode {
    const raw = get("TF_DSV41_PREFILL") orelse return .full;
    return std.meta.stringToEnum(PrefillMode, std.mem.trim(u8, raw, " ")) orelse {
        std.log.warn("TF_DSV41_PREFILL={s}: expected full or replay", .{raw});
        return error.BadKnob;
    };
}

/// The knobs onto a forward before its buffer plan, on every rank (model.zig serving, m2b.zig's gates): the fused
/// dense prefill GEMM, the segment rows, the indexer's budget, the Engram read-ahead of the next segment;
/// `TF_DSV41_PREFILL=replay` is logged. R1 is read where the forward is made (`r1`).
pub fn apply(f: *@import("forward.zig").Forward, a: std.mem.Allocator, io: std.Io, rank: u32) !void {
    f.opts.pfd = try pfDense(a, io, &env);
    if (try prefillRows(&env)) |r| f.opts.prefill_rows = r;
    if (try indexBudget(&env)) |b| f.opts.index_budget = b;
    f.prefill_state.ahead = prefetchAhead(&env);
    if (rank != 0) return;
    const log = std.log.scoped(.dsv41);
    if (try prefillMode(&env) == .replay)
        log.warn("TF_DSV41_PREFILL=replay: the Zig engine runs full prefill (CED replay is not ported); prompts over 128 tokens leave other state than Python prod's", .{});
    log.info("prod knobs: pf_dense {s}, prefill rows {d}, index budget {d} MiB, R1 {s}, prefetch ahead {s}", .{ if (f.opts.pfd != null) "fused" else "off", f.opts.prefill_rows, f.opts.index_budget >> 20, if (f.opts.r1) "on" else "off", if (f.prefill_state.ahead) "on" else "off" });
}

pub fn env(name: []const u8) ?[]const u8 {
    var buf: [64]u8 = undefined;
    if (name.len >= buf.len) return null;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const v = std.mem.span(std.c.getenv(buf[0..name.len :0]) orelse return null);
    return if (v.len == 0) null else v;
}
