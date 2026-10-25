//! Which rank this process is and whether every rank agrees on what it runs (plan T7; the contract is spelled out in the
//! recipe's docs/rank-contract.md). TensorFold v1.1.2's CUDA build has no --tp/--rank/--master flags: a family reads its
//! own environment, as GLM does (TF_GLM_RANK, TF_GLM_WORLD, NCCL_COMM_ID). This family reads TF_MM3_RANK, TF_MM3_WORLD and
//! NCCL_COMM_ID. Everything here is host code: the fingerprint is carried by a gather the engine supplies (`agree`).
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const rank_var = "TF_MM3_RANK";
pub const world_var = "TF_MM3_WORLD";
pub const layers_var = "TF_MM3_LAYERS";
pub const roce_var = "TF_ROCE";
pub const roce_cap_var = "TF_ROCE_MAX_KB";
/// NCCL's own rendezvous variable: rank 0's IPv4 address and port, the same text on every rank.
pub const comm_id_var = "NCCL_COMM_ID";

/// The most ranks the family cuts for: it has 4 kv heads, and a rank without one is refused by `shard.Share.of`.
pub const max_world = 4;

pub const Error = error{
    NoRank, // TF_MM3_RANK or TF_MM3_WORLD missing, not a number, or the rank is not inside the world
    TooManyRanks,
    NoCommId, // a world of 2 or more without NCCL_COMM_ID
    BadCommId, // not IPv4:port, or a port whose successor (the liveness line's) is not a port
    BadSetting,
};

/// An environment lookup: `environment` for the process, a fake in tests.
pub const Get = *const fn ([:0]const u8) ?[]const u8;

pub fn environment(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

pub const Place = struct { rank: usize, world: usize };

/// TF_MM3_RANK of TF_MM3_WORLD; both are required (a lone GPU says TF_MM3_WORLD=1 TF_MM3_RANK=0).
pub fn place(get: Get) Error!Place {
    const r = get(rank_var) orelse return error.NoRank;
    const w = get(world_var) orelse return error.NoRank;
    const p: Place = .{
        .rank = std.fmt.parseInt(usize, r, 10) catch return error.NoRank,
        .world = std.fmt.parseInt(usize, w, 10) catch return error.NoRank,
    };
    if (p.world < 1 or p.rank >= p.world) return error.NoRank;
    if (p.world > max_world) return error.TooManyRanks;
    return p;
}

/// Rank 0's address and port as NCCL_COMM_ID names them. The ranks' liveness lines (cuda/tp_watch.zig) listen on the port
/// after it, so the largest usable port is 65534.
pub const Rendezvous = struct { ip: [4]u8, port: u16 };

pub fn rendezvous(text: []const u8) Error!Rendezvous {
    const colon = std.mem.lastIndexOfScalar(u8, text, ':') orelse return error.BadCommId;
    const port = std.fmt.parseInt(u16, text[colon + 1 ..], 10) catch return error.BadCommId;
    if (port == 0 or port == 65535) return error.BadCommId;
    var ip: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, text[0..colon], '.');
    for (&ip) |*b| b.* = std.fmt.parseInt(u8, it.next() orelse return error.BadCommId, 10) catch return error.BadCommId;
    if (it.next() != null) return error.BadCommId;
    return .{ .ip = ip, .port = port };
}

/// What `open` checks before it touches CUDA: the place, and for 2 ranks or more a usable NCCL_COMM_ID.
pub fn admit(get: Get) Error!Place {
    const p = try place(get);
    if (p.world > 1) _ = try rendezvous(get(comm_id_var) orelse return error.NoCommId);
    return p;
}

/// The refusal text for `err`, naming the variables to set (the server prints it; mirrors glm/cuda_native.zig).
pub fn refusal(err: Error) []const u8 {
    return switch (err) {
        error.NoRank => "MiniMax-M3.1-Flash runs at TP over hosts: set TF_MM3_RANK and TF_MM3_WORLD on every host (WORLD 1: one GPU, with TF_MM3_LAYERS for a thin model)",
        error.TooManyRanks => "MiniMax-M3.1-Flash has 4 kv heads and runs on at most 4 ranks: set TF_MM3_WORLD to 4 or fewer (the recipe uses 2)",
        error.NoCommId => "MiniMax-M3.1-Flash's ranks meet over NCCL: with TF_MM3_WORLD 2 or more, set NCCL_COMM_ID to rank 0's address:port on every host",
        error.BadCommId => "NCCL_COMM_ID is rank 0's IPv4 address and a port from 1 to 65534, as 192.0.2.1:29741 (the liveness line takes the next port)",
        error.BadSetting => "a TF_MM3_* setting is not one of its choices",
    };
}

/// Forward knobs read from the environment. Only the thin-model switch exists in v1; anything that changes the
/// arithmetic must be here and in the fingerprint so both ranks are made to agree.
pub const Settings = struct {
    layers: u32 = 0, // TF_MM3_LAYERS: the first this many layers only, for one-GPU checks (0: all)
    roce_kb: u32 = 4096, // TF_ROCE_MAX_KB, the small-gather RoCE path's cap in KiB; 0 when TF_ROCE=0 keeps every gather on NCCL
    kv8: bool = false, // --kv-cache fp8 (native.Options.kv8): K and V pages as e4m3 rows with fp32 scales; set by `open`, not read here

    pub fn read(get: Get, problem: *[]const u8) Error!Settings {
        var s: Settings = .{};
        if (get(layers_var)) |v| {
            s.layers = std.fmt.parseInt(u32, v, 10) catch 0;
            if (s.layers == 0) {
                problem.* = "TF_MM3_LAYERS: the first how many layers, 1 or more";
                return error.BadSetting;
            }
        }
        if (get(roce_cap_var)) |v| {
            s.roce_kb = std.fmt.parseInt(u32, v, 10) catch 0;
            if (s.roce_kb == 0 or s.roce_kb > std.math.maxInt(u32) / 1024) {
                problem.* = "TF_ROCE_MAX_KB: an integer from 1 through 4194303 KiB, or unset for 4096 KiB";
                return error.BadSetting;
            }
        }
        if (get(roce_var)) |v| if (std.mem.eql(u8, v, "0")) {
            s.roce_kb = 0;
        };
        return s;
    }
};

/// What every rank must agree on, as bytes the communicator carries.
pub const Fingerprint = extern struct {
    layout: u32 = 1, // this struct's version
    world: u32,
    rank: u32,
    layers: u32, // TF_MM3_LAYERS (0: all)
    reduce: u32 = 1, // tp.zig's reduction_version: the fp32 rank-order sum
    roce: u32, // Settings.roce_kb: the ranks open the RoCE small-gather path together or not at all
    kv8: u32 = 0, // Settings.kv8: the ranks' pages are one layout or the other, never mixed
    reserved: u32 = 0, // keeps `capacity` 8-aligned without padding
    capacity: u64, // tokens a sequence holds
    config: [32]u8, // sha256 of config.json
    quant: [32]u8, // sha256 of quantization_config.json: codebook, version, every matrix's bits
    program: [32]u8, // sha256 of this executable
    weights: [32]u8, // digest of the checkpoint's metadata and shard bytes (the engine's checkpoint digest)
    draft: [32]u8 = @splat(0), // the dspark draft: sha256 of its config.json and its shard metadata; zeros without one

    pub const Field = enum { layout, rank, world, layers, reduce, roce, kv8, capacity, config, quant, program, weights, draft };

    pub fn of(p: Place, s: Settings, capacity: usize, config: [32]u8, quant: [32]u8, program: [32]u8, weights: [32]u8) Fingerprint {
        return .{
            .world = @intCast(p.world),
            .rank = @intCast(p.rank),
            .layers = s.layers,
            .roce = s.roce_kb,
            .kv8 = @intFromBool(s.kv8),
            .capacity = capacity,
            .config = config,
            .quant = quant,
            .program = program,
            .weights = weights,
        };
    }
};

pub const Mismatch = struct { field: Fingerprint.Field, rank: usize };

/// The first field where rank r's fingerprint differs from rank 0's (or r holds the wrong rank number), null when all agree.
pub fn mismatch(all: []const Fingerprint) ?Mismatch {
    for (all, 0..) |f, r| {
        if (f.layout != all[0].layout) return .{ .field = .layout, .rank = r };
        if (f.rank != r) return .{ .field = .rank, .rank = r };
        if (f.world != all.len) return .{ .field = .world, .rank = r };
        inline for (.{ "world", "layers", "reduce", "roce", "kv8", "capacity", "config", "quant", "program", "weights", "draft" }) |name| {
            if (!std.mem.eql(u8, std.mem.asBytes(&@field(f, name)), std.mem.asBytes(&@field(all[0], name))))
                return .{ .field = @field(Fingerprint.Field, name), .rank = r };
        }
    }
    return null;
}

pub fn explain(f: Fingerprint.Field) []const u8 {
    return switch (f) {
        .layout => "a different build of this check",
        .rank => "a rank number given twice or out of order",
        .world => "the ranks were started with different world sizes",
        .layers => "TF_MM3_LAYERS",
        .reduce => "a different reduction order",
        .roce => "TF_ROCE or TF_ROCE_MAX_KB",
        .kv8 => "--kv-cache (one rank serves fp8 pages, another bf16)",
        .capacity => "the sequence capacity (the context)",
        .config => "config.json",
        .quant => "quantization_config.json (codebook, version or bit widths)",
        .program => "the executable",
        .weights => "checkpoint metadata or shard bytes",
        .draft => "--drafter (the dspark draft's folder, config or shard bytes)",
    };
}

pub fn hashBytes(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    Sha256.hash(bytes, &out, .{});
    return out;
}

/// `gather` puts every rank's `mine` in `out` in rank order (an ncclAllGather of bytes, as glm/cuda_ranks.zig does it).
/// Any disagreement is error.RankMismatch on every rank, printing what differs.
pub fn agree(ctx: anytype, comptime gather: fn (@TypeOf(ctx), Fingerprint, []Fingerprint) anyerror!void, mine: Fingerprint, out: []Fingerprint) !void {
    try gather(ctx, mine, out);
    if (mismatch(out)) |m| {
        std.debug.print("REFUSE rank settings: rank {d}'s {s} differs from rank 0's ({s})\n", .{ m.rank, @tagName(m.field), explain(m.field) });
        return error.RankMismatch;
    }
}

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: [:0]const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "the place: both variables, a rank inside its world, at most four ranks" {
    Fake.pairs = &.{ .{ rank_var, "1" }, .{ world_var, "2" } };
    try std.testing.expectEqual(Place{ .rank = 1, .world = 2 }, try place(&Fake.get));
    Fake.pairs = &.{.{ rank_var, "1" }};
    try std.testing.expectError(error.NoRank, place(&Fake.get));
    Fake.pairs = &.{ .{ rank_var, "2" }, .{ world_var, "2" } };
    try std.testing.expectError(error.NoRank, place(&Fake.get));
    Fake.pairs = &.{ .{ rank_var, "x" }, .{ world_var, "2" } };
    try std.testing.expectError(error.NoRank, place(&Fake.get));
    Fake.pairs = &.{ .{ rank_var, "0" }, .{ world_var, "0" } };
    try std.testing.expectError(error.NoRank, place(&Fake.get));
    Fake.pairs = &.{ .{ rank_var, "0" }, .{ world_var, "5" } };
    try std.testing.expectError(error.TooManyRanks, place(&Fake.get));
    Fake.pairs = &.{ .{ rank_var, "0" }, .{ world_var, "1" } }; // one GPU needs no rendezvous
    try std.testing.expectEqual(Place{ .rank = 0, .world = 1 }, try admit(&Fake.get));
}

test "two ranks need NCCL_COMM_ID as IPv4:port with room for the liveness port after it" {
    Fake.pairs = &.{ .{ rank_var, "0" }, .{ world_var, "2" } };
    try std.testing.expectError(error.NoCommId, admit(&Fake.get));
    Fake.pairs = &.{ .{ rank_var, "0" }, .{ world_var, "2" }, .{ comm_id_var, "192.0.2.1:29741" } };
    try std.testing.expectEqual(Place{ .rank = 0, .world = 2 }, try admit(&Fake.get));
    try std.testing.expectEqual(Rendezvous{ .ip = .{ 192, 0, 2, 1 }, .port = 29741 }, try rendezvous("192.0.2.1:29741"));
    const bad = [_][]const u8{ "192.0.2.1", "host:29741", "192.0.2:29741", "192.0.2.1.5:29741", "192.0.2.1:0", "192.0.2.1:65535", "192.0.2.256:1", "192.0.2.1:x" };
    for (bad) |text| try std.testing.expectError(error.BadCommId, rendezvous(text));
    Fake.pairs = &.{ .{ rank_var, "1" }, .{ world_var, "2" }, .{ comm_id_var, "192.0.2.1:65535" } };
    try std.testing.expectError(error.BadCommId, admit(&Fake.get));
}

test "settings: TF_MM3_LAYERS is a count of 1 or more" {
    var problem: []const u8 = "";
    Fake.pairs = &.{};
    try std.testing.expectEqual(@as(u32, 0), (try Settings.read(&Fake.get, &problem)).layers);
    Fake.pairs = &.{.{ layers_var, "4" }};
    try std.testing.expectEqual(@as(u32, 4), (try Settings.read(&Fake.get, &problem)).layers);
    for ([_][]const u8{ "0", "abc", "-1" }) |v| {
        Fake.pairs = &.{.{ layers_var, v }};
        try std.testing.expectError(error.BadSetting, Settings.read(&Fake.get, &problem));
    }
    try std.testing.expectEqualStrings("TF_MM3_LAYERS: the first how many layers, 1 or more", problem);
}

fn fp(rank: usize) Fingerprint {
    const h: [32]u8 = @splat(1);
    return Fingerprint.of(.{ .rank = rank, .world = 2 }, .{}, 131072, hashBytes("config"), hashBytes("quant"), h, hashBytes("weights"));
}

test "settings: TF_ROCE=0 turns the RoCE path off and TF_ROCE_MAX_KB must be a usable cap, both agreed by the ranks" {
    var problem: []const u8 = "";
    Fake.pairs = &.{};
    try std.testing.expectEqual(@as(u32, 4096), (try Settings.read(&Fake.get, &problem)).roce_kb);
    Fake.pairs = &.{.{ roce_cap_var, "1024" }};
    try std.testing.expectEqual(@as(u32, 1024), (try Settings.read(&Fake.get, &problem)).roce_kb);
    Fake.pairs = &.{ .{ roce_var, "0" }, .{ roce_cap_var, "1024" } };
    try std.testing.expectEqual(@as(u32, 0), (try Settings.read(&Fake.get, &problem)).roce_kb);
    Fake.pairs = &.{.{ roce_cap_var, "0" }};
    try std.testing.expectError(error.BadSetting, Settings.read(&Fake.get, &problem));
    var all = [_]Fingerprint{ fp(0), Fingerprint.of(.{ .rank = 1, .world = 2 }, .{ .roce_kb = 0 }, 131072, hashBytes("config"), hashBytes("quant"), @splat(1), hashBytes("weights")) };
    try std.testing.expectEqual(Mismatch{ .field = .roce, .rank = 1 }, mismatch(&all).?);
}

test "fingerprints: agreeing ranks pass; config, quantization, weights and the rest name the first difference" {
    var all = [_]Fingerprint{ fp(0), fp(1) };
    try std.testing.expect(mismatch(&all) == null);

    all[1].config = hashBytes("another config");
    try std.testing.expectEqual(Mismatch{ .field = .config, .rank = 1 }, mismatch(&all).?);
    all[1] = fp(1);
    all[1].quant = hashBytes("quantization with 5-bit shared experts");
    try std.testing.expectEqual(Mismatch{ .field = .quant, .rank = 1 }, mismatch(&all).?);
    all[1] = fp(1);
    all[1].weights = hashBytes("a different shard set");
    try std.testing.expectEqual(Mismatch{ .field = .weights, .rank = 1 }, mismatch(&all).?);
    all[1] = fp(1);
    all[1].program = @splat(2);
    try std.testing.expectEqual(Mismatch{ .field = .program, .rank = 1 }, mismatch(&all).?);
    all[1] = fp(1);
    all[1].capacity = 65536;
    try std.testing.expectEqual(Mismatch{ .field = .capacity, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(.{ .rank = 1, .world = 2 }, .{ .layers = 4 }, 131072, hashBytes("config"), hashBytes("quant"), @splat(1), hashBytes("weights"));
    try std.testing.expectEqual(Mismatch{ .field = .layers, .rank = 1 }, mismatch(&all).?);
    all[1] = fp(1);
    all[1].reduce = 2;
    try std.testing.expectEqual(Mismatch{ .field = .reduce, .rank = 1 }, mismatch(&all).?);
    all[1] = fp(0); // the same rank number twice
    try std.testing.expectEqual(Mismatch{ .field = .rank, .rank = 1 }, mismatch(&all).?);
    all[1] = fp(1);
    all[1].draft = hashBytes("a draft the other rank does not load");
    try std.testing.expectEqual(Mismatch{ .field = .draft, .rank = 1 }, mismatch(&all).?);
    all[1] = fp(1);
    all[1].world = 3;
    try std.testing.expectEqual(Mismatch{ .field = .world, .rank = 1 }, mismatch(&all).?);
    // rank 0 is the reference: a rank 0 that differs makes rank 1 the one named
    all[0].weights = hashBytes("x");
    all[1] = fp(1);
    try std.testing.expectEqual(Mismatch{ .field = .weights, .rank = 1 }, mismatch(&all).?);
}

const Wire = struct {
    others: []const Fingerprint,
    fn gather(w: Wire, mine: Fingerprint, out: []Fingerprint) anyerror!void {
        for (out, 0..) |*o, r| o.* = if (r == mine.rank) mine else w.others[r];
    }
};

test "agree refuses a mismatch on the rank that sees it and passes agreeing ranks" {
    var out: [2]Fingerprint = undefined;
    const good = [_]Fingerprint{ fp(0), fp(1) };
    try agree(Wire{ .others = &good }, Wire.gather, fp(0), &out);
    var odd = good;
    odd[1].weights = hashBytes("other");
    try std.testing.expectError(error.RankMismatch, agree(Wire{ .others = &odd }, Wire.gather, fp(0), &out));
    try std.testing.expectError(error.RankMismatch, agree(Wire{ .others = &odd }, Wire.gather, odd[1], &out));
}

test "the fingerprint has no padding bytes to compare by accident" {
    try std.testing.expectEqual(@as(usize, 4 * 8 + 8 + 160), @sizeOf(Fingerprint));
}
