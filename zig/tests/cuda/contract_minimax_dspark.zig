//! MiniMax-M3.1-Flash dspark draft kernels under the contract on CUDA: the set, the runner and the bench.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const common = @import("contract_minimax_dspark_common.zig");
const rows = @import("contract_minimax_dspark_rows.zig");
const heads = @import("contract_minimax_dspark_heads.zig");
const attn_p = @import("contract_minimax_dspark_attn.zig");
const search = @import("contract_minimax_dspark_search.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const ds = cuda.minimax_m3_dspark;

const hidden = common.hidden;
const nq = common.nq;
const kvh = common.kvh;
const Tap = rows.Tap;
const Add = rows.Add;
const AddForm = rows.AddForm;
const Norm = rows.Norm;
const NormForm = rows.NormForm;
const normSpec = rows.normSpec;
const Block = heads.Block;
const Ring = heads.Ring;
const Attn = attn_p.Attn;
const attn_cfgs = attn_p.attn_cfgs;
const attnSpec = attn_p.attnSpec;
const ring_rows = attn_p.ring_rows;
const base_pos = attn_p.base_pos;
const TopC = search.TopC;
const Chain = search.Chain;
const topc_spec = search.topc_spec;
const chain_spec = search.chain_spec;
const slot_n = search.slot_n;
const pad = search.pad;
const cand = search.cand;
const rank = search.rank;

/// The writers of this file as harness kernels, owned by `Set` until `deinit`.
pub const Set = struct {
    taps: [1]Tap,
    adds: [3]Add,
    norms: [3]Norm,
    blocks: [2]Block,
    rings: [1]Ring,
    attns: [attn_cfgs.len]Attn,
    topcs: [1]TopC,
    chains: [1]Chain,

    pub fn init(self: *Set, gpu: @import("check.zig").Gpu, s: cuda.Stream, k: *const ds.Dspark) void {
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const a = gpu.gpa;
        self.taps[0] = .{ .rig = rig, .k = k, .arena = .init(a) };
        for (std.enums.values(AddForm), &self.adds) |f, *x| x.* = .{ .rig = rig, .form = f, .k = k, .arena = .init(a) };
        for (std.enums.values(NormForm), &self.norms) |f, *x| x.* = .{ .rig = rig, .form = f, .k = k, .arena = .init(a) };
        for ([_]bool{ true, false }, &self.blocks) |f, *x| x.* = .{ .rig = rig, .gemma = f, .k = k, .arena = .init(a) };
        self.rings[0] = .{ .rig = rig, .k = k, .arena = .init(a) };
        for (attn_cfgs, &self.attns) |f, *x| x.* = .{ .rig = rig, .f = f, .k = k, .arena = .init(a) };
        self.topcs[0] = .{ .rig = rig, .k = k, .arena = .init(a) };
        self.chains[0] = .{ .rig = rig, .k = k, .arena = .init(a) };
    }

    pub fn deinit(self: *Set) void {
        inline for (.{ "taps", "adds", "norms", "blocks", "rings", "attns", "topcs", "chains" }) |n| for (&@field(self, n)) |*x| x.deinit();
    }

    pub fn append(self: *Set, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        inline for (.{ "taps", "adds", "norms", "blocks", "rings", "attns", "topcs", "chains" }) |n| for (&@field(self, n)) |*x| try list.append(a, x.kernel());
    }
};

/// A loaded module and its writers, registered into a kernel list; the writers point into it, so it lives on the heap.
pub const Holder = struct {
    gpa: std.mem.Allocator,
    k: *ds.Dspark,
    set: *Set,

    pub fn open(gpu: @import("check.zig").Gpu, s: cuda.Stream, list: *std.ArrayList(g.Kernel)) !Holder {
        const a = gpu.gpa;
        const k = try a.create(ds.Dspark);
        errdefer a.destroy(k);
        k.* = try ds.Dspark.load(gpu.d);
        errdefer k.unload();
        const set = try a.create(Set);
        errdefer a.destroy(set);
        set.init(gpu, s, k);
        errdefer set.deinit();
        try set.append(a, list);
        return .{ .gpa = a, .k = k, .set = set };
    }

    pub fn close(h: *Holder) void {
        h.set.deinit();
        h.k.unload();
        h.gpa.destroy(h.set);
        h.gpa.destroy(h.k);
    }
};

/// tf-cuda-test mm3-dspark-bench: milliseconds a launch of the shipped shapes takes (no contract).
pub fn bench(gpu: @import("check.zig").Gpu) !void {
    const check = @import("check.zig");
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var k = try ds.Dspark.load(gpu.d);
    defer k.unload();
    var set: Set = undefined;
    set.init(gpu, s, &k);
    defer set.deinit();
    const runs = 50;
    const Item = struct { name: []const u8, shape: c.Shape };
    inline for (.{
        .{ "attns", 0, Item{ .name = "tf_mm3_ds_attn 7 rows, 4096 context keys", .shape = .{ .rows = 7, .n = nq, .k = 4096 } } },
        .{ "attns", 0, Item{ .name = "tf_mm3_ds_attn 7 rows, 1000 context keys", .shape = .{ .rows = 7, .n = nq, .k = 1000 } } },
        .{ "topcs", 0, Item{ .name = "tf_mm3_ds_topc 7 rows, n 100096, c 64", .shape = .{ .rows = 7, .n = slot_n, .k = 64 } } },
        .{ "chains", 0, Item{ .name = "tf_mm3_ds_chain 7 rows, 2 x 64 candidates", .shape = .{ .rows = 7, .n = cand, .k = 2 } } },
        .{ "norms", 0, Item{ .name = "tf_mm3_ds_norm_bf16 7 rows", .shape = .{ .rows = 7, .n = hidden, .k = hidden } } },
        .{ "norms", 0, Item{ .name = "tf_mm3_ds_norm_bf16 2048 rows of 6144", .shape = .{ .rows = 2048, .n = hidden, .k = hidden } } },
        .{ "blocks", 0, Item{ .name = "tf_mm3_ds_block_heads 7 rows", .shape = .{ .rows = 7, .n = nq, .k = 3000 } } },
        .{ "rings", 0, Item{ .name = "tf_mm3_ds_ring_put 16 rows", .shape = .{ .rows = 16, .n = kvh, .k = 5 } } },
        .{ "taps", 0, Item{ .name = "tf_mm3_ds_tap 64 rows", .shape = .{ .rows = 64, .n = hidden, .k = 5 } } },
    }) |it| {
        const w = &@field(set, it[0])[it[1]];
        if (!try w.prepare(it[2].shape, .random, 7)) return error.RefusedShape;
        try w.run(); // warm
        const t0 = check.now(gpu.io);
        for (0..runs) |_| try w.run();
        const ms = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6 / runs;
        std.debug.print("BENCH {s}: {d:.4} ms a launch (incl. host sync)\n", .{ it[2].name, ms });
    }
}

/// tf-cuda-test mm3-dspark-contract: these kernels alone through gates 1 to 4.
pub fn run(gpu: @import("check.zig").Gpu) !void {
    const check = @import("check.zig");
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var k = try ds.Dspark.load(gpu.d);
    defer k.unload();
    var set: Set = undefined;
    set.init(gpu, s, &k);
    defer set.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    try set.append(a, &kernels);
    var t: g.Tally = .{};
    try g.runAll(a, kernels.items, 0x6d34, &t);
    std.debug.print("SUMMARY mm3-dspark-contract {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings\n", .{ t.kernels, t.passed, t.failed, t.known });
    try check.expect(t.failed == 0, "mm3-dspark-contract: {d} gate failures", .{t.failed});
}

test "the served shapes fit the launches" {
    for (normSpec(.bf16_gemma).served) |s| _ = try ds.rowLaunch(s.rows);
    for (topc_spec.served) |s| _ = try ds.topcLaunch(s.rows, s.n, s.n + pad, s.k);
    for (attnSpec(attn_cfgs[0]).served) |s| {
        const a: ds.Attn = .{ .nq = nq, .kvh = kvh, .ring_rows = ring_rows, .window = 1 << 20, .scale = 1 };
        _ = try ds.attnSplit(a, @intCast(s.rows), @intCast(s.k + base_pos), base_pos);
    }
    for (chain_spec.served) |s| try ds.checkChain(.{ .hidden = hidden, .rank = rank, .cand = @intCast(s.n), .world = @intCast(s.k), .rows = @intCast(s.rows) });
    try std.testing.expectEqual(@as(usize, 100_096), slot_n);
}
