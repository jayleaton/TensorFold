//! `tensorfold` for Flash Next (qwen4_exp) on CUDA: token runs and exactness checks against the Python capture.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const flashnext = @import("flashnext_mia");

const Engine = flashnext.Engine;
const decode = flashnext.decode;
const fwd = flashnext.forward;
const Allocator = std.mem.Allocator;

const checks_part = @import("flashnext_cuda_checks.zig");
const gates_part = @import("flashnext_cuda_gates.zig");
const bench_part = @import("flashnext_cuda_bench.zig");
const kernel_checks_part = @import("flashnext_cuda_kernel_checks.zig");

const agree = checks_part.agree;
const teacher = checks_part.teacher;
const prefillCheck = checks_part.prefillCheck;
const checkWeights = checks_part.checkWeights;
const gate = gates_part.gate;
const gateMedia = gates_part.gateMedia;
const gateMany = gates_part.gateMany;
const reuse = gates_part.reuse;
const pool = bench_part.pool;
const bench = bench_part.bench;
const benchMany = bench_part.benchMany;
const mmCheck = kernel_checks_part.mmCheck;
const expertsCheck = kernel_checks_part.expertsCheck;
const qsaCheck = kernel_checks_part.qsaCheck;
const fp4Check = kernel_checks_part.fp4Check;
const glueCheck = kernel_checks_part.glueCheck;

const usage =
    \\usage (Flash Next, qwen4_exp):
    \\  tensorfold run MODEL --tokens ID,ID,... [--tokens-file PATH] [--max-tokens N] [--no-drafts] [--report PATH]
    \\           [--temperature T --top-k K --top-p P --min-p M --seed S] [--ignore-eos] [--context N] [--kernels DIR]
    \\  tensorfold teacher MODEL TEACHER.json [--dump CAPTURE/teacher]   (48 teacher-forced steps: tokens, logits,
    \\           MTP draws; with --dump every step's per-layer digests)
    \\  tensorfold prefill MODEL PROMPTS.json NAME [--dump CAPTURE/prefill/NAME] [--teacher TEACHER.json]
    \\  tensorfold check-weights MODEL WEIGHTS.json   (every device buffer's sha256 against the oracle's)
    \\  tensorfold gate-many MODEL CAPTURE_DIR --streams N [--max-tokens N]   (N streams in shared rounds mixing the
    \\           captured prompts and rules: all drafted, all serial, then mixed; each stream = its solo run)
    \\           [--arrive K --fill-layers L]   (odd streams arrive at rounds K, 2K, ... and fill their prompts L
    \\           layers a round between the others' rounds) [--burst]   (the streams present at the start prefill
    \\           together, one prompt pass over each group that fits)
    \\  tensorfold gate-media MODEL DIR [--only CASE,...] [--streams N] [--burst]   (vision_ref.py's image and video cases: prompt
    \\    digests, serial and drafted greedy tokens against Python's, then all cases in shared rounds == solo; --burst adds text prompts around them: bursts must leave the media prompts to prefill alone)
    \\  tensorfold reuse MODEL CAPTURE_DIR [--only NAME,...]   (kept prompt states: a resumed next turn and a copied
    \\           prefix in another sequence decode as a fresh prefill does)
    \\  tensorfold bench-many MODEL --streams MAX --kind code,prose,chat,structured,dash-prose,dash-code,dash-structured [--rows N,N,...] [--reps K] [--max-tokens N]
    \\           [--ignore-eos] [--no-drafts] [--depth D] [--served]   (N streams of a kind's prompts in shared rounds, for each N in --rows:
    \\           aggregate tok/s; code greedy, prose and chat sampled; TF_FLASHNEXT_PROFILE=1 times rounds by part)
    \\  tensorfold pool MODEL --streams N   (N sequences grown in turn until the memory budget refuses: the tokens
    \\           the rank's caches hold at N streams)
    \\  tensorfold bench MODEL --tokens-file PATH[,PATH...] [--reps N] [--max-tokens N] [--no-drafts]   (each
    \\           prompt N times in one load: prefill seconds and tok/s, the reply's first token and sha)
    \\  tensorfold mm-check MODEL [--rows N,N,...]   (no weights: the prompt matmuls' in-program K
    \\           slices against Python's split kernels + _reduce, bytes compared, then their speed)
    \\  tensorfold qsa-check MODEL   (no weights: the prompt indexer's _scores_rows and tiled select against Python's)
    \\  tensorfold fp4-check MODEL [--rows N,N,...]   (no weights: the shared expert's single-slice NVFP4 matmul on
    \\           fn_ops' K-serial kernel against Triton's _fp4mm, bytes compared, then both timed)
    \\  tensorfold experts-check MODEL [--rows N,N,...]   (no weights: the prompt expert kernel on larger items
    \\           against Python's on 16-pair items, random experts and skewed routing, bytes compared, timed)
    \\  tensorfold fp8-check MODEL DIR   (no weights: the block-FP8 lane matmul (cuda_fp8.zig, qmmf FP8G) against
    \\           Python's Fp8BlockLinear on tools/zig/flashnext_mia/check_fp8block.py's cases in DIR, bytes compared, timed)
    \\  tensorfold int4-check MODEL   (no weights: the GPTQ int4 expert and head kernels (cuda_int4.zig) against a host
    \\           reference from the GPTQ bytes, row/tile invariance, plan == dense, then their speed)
    \\  tensorfold agree MODEL CAPTURE_DIR --report OUT.json [--against REF.json] [--max-tokens N] [--only NAME,...]
    \\           (each captured prompt's greedy serial reply written to OUT.json; with REF.json, another run's replies
    \\           teacher-forced: the top-1 agreement of this engine's greedy draws with them, and where the free
    \\           replies first differ; a lossy mode's quality against the exact path)
    \\  every command: [--tp 2 --rank R --master ADDR --master-port PORT] (both ranks run the same command)
    \\           [--kv-dtype bf16|fp8]   (the attention caches' format; fp8: e4m3 rows, a power-of-two scale a head row)
    \\  tensorfold gate MODEL CAPTURE_DIR [--max-tokens N] [--only NAME,...] [--report PATH]   (every captured
    \\           prompt serial and drafted, greedy and sampled, against results.json; drafted == serial)
    \\
;

/// Whether `dir`'s config.json names a Flash Next model type (cuda_config.model_types).
pub fn wants(gpa: Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(gpa, &.{ dir, "config.json" }) catch return false;
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24)) catch return false;
    defer gpa.free(text);
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return false;
    defer parsed.deinit();
    const mt = parsed.value.object.get("model_type") orelse return false;
    if (mt != .string) return false;
    for (flashnext.config.model_types) |m| if (std.mem.eql(u8, m, mt.string)) return true;
    return false;
}

pub const Options = struct {
    model: []const u8,
    tokens: []u32 = &.{},
    max_tokens: usize = 256,
    sampling: lanes.Sampling = .{ .seed = 0, .temperature = 0 },
    drafts: bool = true,
    report: ?[]const u8 = null,
    kernels: ?[]const u8 = null,
    device: u32 = 0,
    dump: ?[]const u8 = null,
    teacher: ?[]const u8 = null,
    only: ?[]const u8 = null,
    context: usize = 262144,
    stop_eos: bool = true,
    graphs: bool = true,
    tp: u32 = 1,
    streams: usize = 1,
    kv_gib: ?f64 = null,
    reserve_gib: f64 = 12,
    repeat: usize = 1,
    against_solo: bool = false,
    arrive: usize = 0,
    fill_layers: usize = 0,
    burst: bool = false,
    rank: u32 = 0,
    master: []const u8 = "",
    master_port: u16 = 29551,
    reps: usize = 1,
    rows: []const u8 = "",
    kind: []const u8 = "code",
    depth: usize = 6,
    /// bench-many: the served confidence rule (confidenceSetting) instead of Python's 0.70
    served: bool = false,
    /// the attention caches' format (--kv-dtype bf16|fp8)
    kv: flashnext.state.KvDtype = .bf16,
};

pub fn parseIds(gpa: Allocator, text: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n[]");
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseInt(u32, t, 10));
    return out.toOwnedSlice(gpa);
}

pub fn main(init: std.process.Init, args: []const []const u8) !u8 {
    const gpa = init.gpa;
    if (args.len < 3) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var o: Options = .{ .model = args[2] };
    defer gpa.free(o.tokens);
    var positional: std.ArrayList([]const u8) = .empty;
    defer positional.deinit(gpa);
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const v = if (i + 1 < args.len) args[i + 1] else "";
        if (std.mem.eql(u8, a, "--tokens")) {
            gpa.free(o.tokens);
            o.tokens = try parseIds(gpa, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--tokens-file") and std.mem.eql(u8, args[1], "bench")) {
            try positional.append(gpa, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--tokens-file")) {
            const text = try std.Io.Dir.cwd().readFileAlloc(init.io, v, gpa, .limited(1 << 26));
            defer gpa.free(text);
            gpa.free(o.tokens);
            o.tokens = try parseIds(gpa, text);
            i += 1;
        } else if (std.mem.eql(u8, a, "--max-tokens")) {
            o.max_tokens = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--temperature")) {
            o.sampling.temperature = try std.fmt.parseFloat(f64, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--top-k")) {
            o.sampling.top_k = try std.fmt.parseInt(u32, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--top-p")) {
            o.sampling.top_p = try std.fmt.parseFloat(f64, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--min-p")) {
            o.sampling.min_p = try std.fmt.parseFloat(f64, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--seed")) {
            o.sampling.seed = try std.fmt.parseInt(u64, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--report")) {
            o.report = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--kernels")) {
            o.kernels = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--device")) {
            o.device = try std.fmt.parseInt(u32, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--dump")) {
            o.dump = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--teacher") or std.mem.eql(u8, a, "--against")) {
            o.teacher = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--only")) {
            o.only = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--context")) {
            o.context = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--kv-gib")) {
            o.kv_gib = try std.fmt.parseFloat(f64, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--repeat")) {
            o.repeat = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--arrive")) {
            o.arrive = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--fill-layers")) {
            o.fill_layers = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--burst")) {
            o.burst = true;
        } else if (std.mem.eql(u8, a, "--against-solo")) {
            o.against_solo = true;
        } else if (std.mem.eql(u8, a, "--streams")) {
            o.streams = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--tp")) {
            o.tp = try std.fmt.parseInt(u32, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--rank")) {
            o.rank = try std.fmt.parseInt(u32, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--master")) {
            o.master = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--master-port")) {
            o.master_port = try std.fmt.parseInt(u16, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--reps")) {
            o.reps = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--depth")) {
            o.depth = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--kv-dtype")) {
            o.kv = flashnext.state.KvDtype.parse(v) orelse {
                std.debug.print("--kv-dtype {s}: bf16 or fp8\n", .{v});
                return 2;
            };
            i += 1;
        } else if (std.mem.eql(u8, a, "--served")) {
            o.served = true;
        } else if (std.mem.eql(u8, a, "--kind")) {
            o.kind = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--rows")) {
            o.rows = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--no-drafts")) {
            o.drafts = false;
        } else if (std.mem.eql(u8, a, "--ignore-eos")) {
            o.stop_eos = false;
        } else if (std.mem.eql(u8, a, "--eager")) {
            o.graphs = false;
        } else if (std.mem.startsWith(u8, a, "--")) {
            std.debug.print("unknown option {s}\n{s}", .{ a, usage });
            return 2;
        } else try positional.append(gpa, a);
    }
    const kernels = o.kernels orelse init.environ_map.get("TENSORFOLD_CUDA_KERNELS") orelse {
        std.debug.print("the Triton kernel set: --kernels DIR or TENSORFOLD_CUDA_KERNELS (aot.json + cubins/)\n", .{});
        return 2;
    };
    const cmd = args[1];
    const rest = positional.items;
    if (init.environ_map.get("TENSORFOLD_MEMORY_RESERVE_GIB")) |v| o.reserve_gib = try std.fmt.parseFloat(f64, v);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, @intCast(o.device));
    defer ctx.deinit();
    if (std.mem.eql(u8, cmd, "mm-check")) return mmCheck(gpa, init.io, &ctx, kernels, o);
    if (std.mem.eql(u8, cmd, "experts-check")) return expertsCheck(gpa, &ctx, o);
    if (std.mem.eql(u8, cmd, "fp8-check") and rest.len == 1) {
        const ok = try flashnext.fp8.check(gpa, init.io, &ctx, rest[0]);
        std.debug.print("{s} fp8-check: the block-FP8 lane matmul {s} Python's Fp8BlockLinear\n", .{ if (ok) "PASS" else "FAIL", if (ok) "byte-equal to" else "DIFFERS from" });
        return if (ok) 0 else 1;
    }
    if (std.mem.eql(u8, cmd, "int4-check")) {
        var k = try flashnext.kernels.Kernels.load(&ctx);
        defer k.deinit();
        var stream = try cuda.Stream.init(ctx.d, true);
        defer stream.deinit();
        const ok = try flashnext.int4_check.check(gpa, &ctx, .{ .k = &k, .s = stream });
        std.debug.print("{s} int4-check: the GPTQ int4 kernels against the host order, row and tile invariance, plan == dense\n", .{if (ok) "PASS" else "FAIL"});
        return if (ok) 0 else 1;
    }
    if (std.mem.eql(u8, cmd, "qsa-check")) return qsaCheck(gpa, init.io, &ctx, kernels);
    if (std.mem.eql(u8, cmd, "fp4-check")) return fp4Check(gpa, init.io, &ctx, kernels, o);
    if (std.mem.eql(u8, cmd, "glue-check")) return glueCheck(gpa, init.io, &ctx, kernels);
    const mtp = !std.mem.eql(u8, cmd, "check-weights-nomtp");
    // two ranks: both run the same command (every engine call in the same order), rank 0 listening on --master
    var server: ?flashnext.link.Server = null;
    defer if (server) |sv| sv.close();
    var lk: ?flashnext.link.Link = null;
    defer if (lk) |l| l.close();
    var comm: ?flashnext.comm.Comm = null;
    defer if (comm) |*c| c.deinit();
    if (o.tp == 2) {
        const address = try std.Io.net.IpAddress.parse(o.master, o.master_port);
        if (o.rank == 0) {
            server = try flashnext.link.Server.open(address);
            std.debug.print("rank 0: waiting for rank 1 on {s}:{d}\n", .{ o.master, o.master_port });
            lk = try server.?.accept(600_000);
        } else {
            lk = try flashnext.link.Link.join(init.io, address, 600_000);
        }
        var sb: [256]u8 = undefined;
        try lk.?.agree(gpa, o.rank, try std.fmt.bufPrint(&sb, "{s} {s} context {d} mtp {} kv {s}", .{ cmd, std.fs.path.basename(o.model), o.context, mtp, @tagName(o.kv) }));
        comm = try flashnext.comm.Comm.init(gpa, lk.?, o.rank, 2);
        std.debug.print("rank {d} of 2: NCCL joined\n", .{o.rank});
    } else if (o.tp != 1) return error.BadTp;
    std.debug.print("loading {s} (kernels {s})\n", .{ o.model, kernels });
    const yarn = try yarnFor(gpa, init.io, o.model, init.environ_map.get(flashnext.rope.env_factor), init.environ_map.get(flashnext.rope.env_ramp));
    if (yarn) |y| std.debug.print("YaRN: factor {d} over {d} positions\n", .{ y.factor, y.original });
    const e = try Engine.init(gpa, init.io, &ctx, o.model, kernels, .{ .context = o.context, .mtp = mtp, .graphs = o.graphs, .rank = o.rank, .world = o.tp, .comm = if (comm) |*c| c else null, .yarn = yarn, .streams = o.streams, .depth = o.depth, .reserve_gib = o.reserve_gib, .kv = o.kv, .kv_budget = if (o.kv_gib) |g| @as(usize, @intFromFloat(g * (1 << 30))) else null });
    if (o.graphs and (std.mem.eql(u8, cmd, "gate") or std.mem.eql(u8, cmd, "run"))) {
        const t0 = now(init.io);
        const n = try e.warmGraphs();
        std.debug.print("{d} decode graphs captured in {d:.1}s\n", .{ n, since(init.io, t0) });
    }
    std.debug.print("sequence memory {d:.2} GiB, {d:.2} GiB a {d}-row sequence\n", .{ @as(f64, @floatFromInt(e.budget.limit)) / (1 << 30), @as(f64, @floatFromInt(e.seqBytes(e.max_len))) / (1 << 30), e.max_len });
    defer e.deinit();
    std.debug.print("loaded in {d:.1}s: {d} weight buffers, {d:.2} GiB; cache rows {d}\n", .{ e.load_seconds, e.w.named.items.len, @as(f64, @floatFromInt(e.w.bytes)) / (1 << 30), e.max_len });
    if (std.mem.eql(u8, cmd, "run")) return run(gpa, init.io, e, o);
    if (std.mem.eql(u8, cmd, "bench")) return bench(gpa, init.io, e, o, positional.items);
    if (std.mem.eql(u8, cmd, "teacher") and rest.len == 1) return teacher(gpa, init.io, e, rest[0], o.dump, o.drafts);
    if (std.mem.eql(u8, cmd, "prefill") and rest.len == 2) return prefillCheck(gpa, init.io, e, rest[0], rest[1], o.dump, o.teacher);
    if (std.mem.eql(u8, cmd, "check-weights") and rest.len == 1) return checkWeights(gpa, init.io, e, rest[0]);
    if (std.mem.eql(u8, cmd, "gate") and rest.len == 1) return gate(gpa, init.io, e, rest[0], o);
    // --served: the server's hybrid stop rule in generateMany too (TF_FLASHNEXT_PRODUCT_STREAMS)
    if (o.served) e.product_streams = flashnext.engine.productStreams();
    if (std.mem.eql(u8, cmd, "gate-many") and rest.len == 1) return gateMany(gpa, init.io, e, rest[0], o);
    if (std.mem.eql(u8, cmd, "gate-media") and rest.len == 1) return gateMedia(gpa, init.io, e, rest[0], o);
    if (std.mem.eql(u8, cmd, "bench-many")) return benchMany(gpa, init.io, e, o);
    if (std.mem.eql(u8, cmd, "reuse") and rest.len == 1) return reuse(gpa, init.io, e, rest[0], o);
    if (std.mem.eql(u8, cmd, "pool")) return pool(gpa, e, o);
    if (std.mem.eql(u8, cmd, "agree") and rest.len == 1) return agree(gpa, init.io, e, rest[0], o);
    std.debug.print("{s}", .{usage});
    return 2;
}

/// YaRN from the config's rope parameters or TF_FLASHNEXT_YARN (cuda_rope.fromConfig, the patched Python's rule).
fn yarnFor(gpa: Allocator, io: std.Io, dir: []const u8, factor_env: ?[]const u8, ramp_env: ?[]const u8) !?flashnext.rope.Yarn {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const doc = try readJson(gpa, io, path);
    defer doc.deinit();
    const top = doc.value.object;
    const text_cfg = if (top.get("text_config")) |t| (if (t == .object) t.object else top) else top;
    const native: u64 = if (text_cfg.get("max_position_embeddings")) |v| (if (v == .integer and v.integer > 0) @intCast(v.integer) else 262144) else 262144;
    const empty: std.json.ObjectMap = .empty;
    const params = if (text_cfg.get("rope_parameters") orelse text_cfg.get("rope_scaling")) |r| (if (r == .object) r.object else empty) else empty;
    return flashnext.rope.fromConfig(params, native, factor_env, ramp_env);
}

pub fn readJson(gpa: Allocator, io: std.Io, path: []const u8) !std.json.Parsed(std.json.Value) {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(text);
    return std.json.parseFromSlice(std.json.Value, gpa, text, .{});
}

pub fn shaHex(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

pub fn deviceSha(gpa: Allocator, e: *Engine, ptr: u64, len: usize) ![64]u8 {
    const host = try gpa.alloc(u8, len);
    defer gpa.free(host);
    try e.stream.synchronize();
    const buf: cuda.DeviceBuffer = .{ .d = e.ctx.d, .ptr = ptr, .len = len };
    try buf.download(0, host);
    return shaHex(host);
}

pub fn now(io: std.Io) std.Io.Timestamp {
    return std.Io.Clock.awake.now(io);
}

pub fn since(io: std.Io, t: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(now(io).toNanoseconds() - t.toNanoseconds())) / 1e9;
}

pub const Generated = struct {
    tokens: []u32,
    prefill_s: f64,
    decode_s: f64,
    rounds: usize,
    drafted: usize,
    accepted: usize,
};

/// Python Engine.generate on one GPU: `_serial` (no MTP head) or `_decode` (MTP-drafted rounds), capped by the window.
pub fn generate(gpa: Allocator, io: std.Io, e: *Engine, prompt: []const u32, max_tokens: usize, sampling: ?lanes.Sampling, drafts: bool, stop_eos: bool) !Generated {
    return generateMedia(gpa, io, e, prompt, max_tokens, sampling, drafts, stop_eos, null);
}

/// `generate` for a prompt with images or video frames (`media`: its rows, rotary positions and features).
pub fn generateMedia(gpa: Allocator, io: std.Io, e: *Engine, prompt: []const u32, max_tokens: usize, sampling: ?lanes.Sampling, drafts: bool, stop_eos: bool, media: ?*const flashnext.media.Media) !Generated {
    const room = @as(i64, @intCast(e.max_len)) - @as(i64, @intCast(prompt.len)) - @as(i64, @intCast(e.depth)) - 1;
    if (room < 1) return error.PromptPastWindow;
    const count = @max(1, @min(max_tokens, @as(usize, @intCast(room))));
    e.absorbing = drafts;
    const t0 = now(io);
    const first = (try e.prefillWith(prompt, sampling, .{ .media = media })).first;
    try e.stream.synchronize();
    const prefill_s = since(io, t0);
    if ((stop_eos and e.isEos(first)) or count <= 1) {
        const one = try gpa.alloc(u32, 1);
        one[0] = first;
        return .{ .tokens = one, .prefill_s = prefill_s, .decode_s = 0, .rounds = 0, .drafted = 0, .accepted = 0 };
    }
    const t1 = now(io);
    const opts: decode.Options = .{ .count = count, .depth = e.depth, .confidence = flashnext.engine.confidenceSetting(), .stop_eos = stop_eos };
    var r = if (drafts and e.depth > 0) try decode.drafted(Engine, e, gpa, first, opts) else try decode.serial(Engine, e, gpa, first, opts);
    errdefer r.deinit(gpa);
    try e.stream.synchronize();
    const decode_s = since(io, t1);
    return .{ .tokens = try r.tokens.toOwnedSlice(gpa), .prefill_s = prefill_s, .decode_s = decode_s, .rounds = r.rounds, .drafted = r.drafted, .accepted = r.accepted };
}

pub fn sha12(gpa: Allocator, tokens: []const u32) ![12]u8 {
    const text = try core.ids_json.write(gpa, tokens);
    defer gpa.free(text);
    const hex = shaHex(text);
    return hex[0..12].*;
}

fn run(gpa: Allocator, io: std.Io, e: *Engine, o: Options) !u8 {
    if (o.tokens.len == 0) {
        std.debug.print("run: --tokens or --tokens-file\n", .{});
        return 2;
    }
    const s: ?lanes.Sampling = if (o.sampling.temperature > 0) o.sampling else null;
    const g = try generate(gpa, io, e, o.tokens, o.max_tokens, s, o.drafts, o.stop_eos);
    defer gpa.free(g.tokens);
    const sha = try sha12(gpa, g.tokens);
    const steps = @max(1, g.tokens.len - 1);
    const ms = g.decode_s * 1e3 / @as(f64, @floatFromInt(steps));
    std.debug.print("tokens {d} sha {s} prefill {d:.4}s decode {d:.4}s {d:.3} ms/token rounds {d} drafted {d} accepted {d}\n", .{ g.tokens.len, sha, g.prefill_s, g.decode_s, ms, g.rounds, g.drafted, g.accepted });
    if (o.report) |path| {
        const report = .{ .engine = "zig-cuda-flashnext", .prompt_tokens = o.tokens, .tokens = g.tokens, .sha = sha, .prefill_seconds = g.prefill_s, .decode_seconds = g.decode_s, .ms_per_token = ms, .rounds = g.rounds, .drafted = g.drafted, .accepted = g.accepted, .drafts = o.drafts, .sampling = s, .load_seconds = e.load_seconds, .max_len = e.max_len };
        const json = try std.json.Stringify.valueAlloc(gpa, report, .{});
        defer gpa.free(json);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
    }
    return 0;
}

pub fn idsOf(gpa: Allocator, v: std.json.Value) ![]u32 {
    const items = v.array.items;
    const ids = try gpa.alloc(u32, items.len);
    for (items, ids) |x, *y| y.* = @intCast(x.integer);
    return ids;
}

pub fn listed(only: ?[]const u8, name: []const u8) bool {
    const l = only orelse return true;
    var it = std.mem.tokenizeAny(u8, l, ",");
    while (it.next()) |x| if (std.mem.eql(u8, x, name)) return true;
    return false;
}
