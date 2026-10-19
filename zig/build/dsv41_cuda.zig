//! DeepSeek-V4.1's CUDA build: its fatbins, `dsv41_kernels`, the three binaries and its emitter host tests.
const std = @import("std");
const cuda_build = @import("cuda.zig");
const Kernel = @import("cuda_kernels.zig").Kernel;

/// DeepSeek-V4.1's kernels (zig/kernels/cuda/deepseek_v41), each with its extension's `extra_cuda_cflags`.
const dsv41_kernels = [_]Kernel{
    .{ .name = "dsv41_exl3_experts", .src = "deepseek_v41/exl3_experts", .flags = &.{ "-O3", "-lineinfo" } }, // tensorfold_exl3_experts_v1
    .{ .name = "dsv41_x3ld", .src = "deepseek_v41/x3ld", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_x3ld_v1
    .{ .name = "dsv41_x3pf", .src = "deepseek_v41/x3pf", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_x3pf_v1
    .{ .name = "dsv41_x3gm", .src = "deepseek_v41/x3gm", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_x3gm_v1
    .{ .name = "dsv41_x3gm3", .src = "deepseek_v41/x3gm3", .flags = &.{ "-O3", "-lineinfo" } }, // ours: x3gm v3 (x3gm.cu's device code)
    .{ .name = "dsv41_x3ld_epi", .src = "deepseek_v41/x3ld_epi", .flags = &.{ "-O3", "-lineinfo" } }, // ours: x3ld + the expert epilogues (TF_DSV41_X3LD_EPI)
    .{ .name = "dsv41_x3gm_plan", .src = "deepseek_v41/x3gm_plan", .flags = &.{"-O3"} }, // ours: x3gm.plan's torch ops
    .{ .name = "dsv41_topk_keys", .src = "deepseek_v41/topk_keys", .flags = &.{"-O3"} }, // ours: pick.top's torch.topk
    .{ .name = "dsv41_kvsplit", .src = "deepseek_v41/kvsplit", .flags = &.{"-O3"} }, // ours: split KV's exchange copies
    .{ .name = "dsv41_glue", .src = "deepseek_v41/glue", .flags = &.{ "-O3", "--fmad=false" } }, // ours: the window's torch glue
    .{ .name = "dsv41_pfglue", .src = "deepseek_v41/prefill_glue", .flags = &.{ "-O3", "--fmad=false" } }, // ours: the prefill segment's torch glue
    .{ .name = "dsv41_pointwise", .src = "deepseek_v41/pointwise", .flags = &.{ "-O3", "--fmad=false" } }, // ours: prefill MoE's torch pointwise ops
    .{ .name = "dsv41_linear", .src = "deepseek_v41/linear", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tensorfold_exl3_linear_v4
    .{ .name = "dsv41_x3seg", .src = "deepseek_v41/x3seg", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tf_dsv41_x3seg_v3
    .{ .name = "dsv41_dense3", .src = "deepseek_v41/dense3", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tf_dsv41_dense3_v1
    .{ .name = "dsv41_attn", .src = "deepseek_v41/attn_cuda", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // tf_dsv41_attn_cuda_v1
    .{ .name = "dsv41_topk", .src = "deepseek_v41/topk_cuda", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // tf_dsv41_attn_cuda_v1
    .{ .name = "dsv41_topk_b", .src = "deepseek_v41/topk_b", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // ours: topk_cuda.cu row-bounded (TF_DSV41_INDEX_BOUND)
    .{ .name = "dsv41_mhc", .src = "deepseek_v41/mhc_cuda", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // tf_dsv41_mhc_cuda_v1
    .{ .name = "dsv41_mhc_pf", .src = "deepseek_v41/mhc_pf", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // tf_dsv41_mhc_pf_v1
    .{ .name = "dsv41_router_gemv", .src = "deepseek_v41/router_gemv", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_router_gemv_v2
    .{ .name = "dsv41_kv_glue", .src = "deepseek_v41/kv_glue", .flags = &.{ "-O3", "-lineinfo" } }, // KV RMS uses explicit rounded FP64; store preserves Triton FMA
    .{ .name = "dsv41_router_glue", .src = "deepseek_v41/router_glue", .flags = &.{ "-O3", "-lineinfo" } }, // histogram group + upstream rotation; same FMA policy as experts
    .{ .name = "dsv41_pfdense_k8", .src = "deepseek_v41/pfdense_k8", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tf_dsv41_pfdense_v1
    .{ .name = "dsv41_pfdense_k10", .src = "deepseek_v41/pfdense_k10", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
    .{ .name = "dsv41_pfdense_k12", .src = "deepseek_v41/pfdense_k12", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
    .{ .name = "dsv41_pfdense_k16", .src = "deepseek_v41/pfdense_k16", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
    .{ .name = "dsv41_engram_gate", .src = "deepseek_v41/engram_gate", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_engram_gate_v1
    .{ .name = "dsv41_l2pace", .src = "deepseek_v41/l2pace", .flags = &.{"-O3"} }, // tf_dsv41_l2pace_v1
    .{ .name = "dsv41_l2pf", .src = "deepseek_v41/l2pf", .flags = &.{"-O3"} }, // tf_dsv41_l2pf_v1 (glm5_next/spark/l2pf.cu)
};

/// Optional kernels (-Dnvcc, or -Dfatbins when the dir holds one); else an empty image and the feature refuses at boot.
const optional = [_]Kernel{
    .{ .name = "dsv41_sampling", .src = "deepseek_v41/sampling", .flags = &.{ "-O3", "--fmad=false" } }, // sampling.cu
    .{ .name = "dsv41_vision", .src = "deepseek_v41_vision/vision", .flags = &.{"-O3"} }, // vision.cu: fmad on, as torch
    .{ .name = "dsv41_lin2", .src = "deepseek_v41/lin2", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // ours: lin2.cu
    .{ .name = "dsv41_router_gemv2", .src = "deepseek_v41/router_gemv2", .flags = &.{ "-O3", "-lineinfo" } }, // ours: router_gemv.cu's flags
    .{ .name = "dsv41_mhc_pf_fold", .src = "deepseek_v41/mhc_pf_fold", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // ours: mhc_pf.cu's flags
    .{ .name = "dsv41_pfrot", .src = "deepseek_v41/pfrot", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // ours: pfdense's flags
};

/// Where the fatbins come from: cuda.zig's -Dnvcc (with its version text), -Dfatbins and -Dsm, and its fatbins step.
pub const Source = struct { nvcc: ?[]const u8, version: ?std.Build.LazyPath, prebuilt: ?[]const u8, sms: []const u8, step: *std.Build.Step };

pub const Images = struct { main: [dsv41_kernels.len]?std.Build.LazyPath = @splat(null), optional: [optional.len]?std.Build.LazyPath = @splat(null), any: bool = false };

/// Every DeepSeek-V4.1 fatbin from `src`, installed by its fatbins step.
pub fn images(b: *std.Build, src: Source) Images {
    var out: Images = .{ .any = src.nvcc != null or src.prebuilt != null };
    for (dsv41_kernels, &out.main) |k, *image| {
        if (src.prebuilt) |dir| {
            image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
        } else if (src.nvcc) |tool| image.* = cuda_build.fatbin(b, tool, src.version.?, k, src.sms);
        if (image.*) |file| src.step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
    }
    for (optional, &out.optional) |k, *image| {
        if (src.prebuilt) |dir| {
            const file = b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) });
            if (std.Io.Dir.cwd().access(b.graph.io, file, .{})) |_| {
                image.* = b.graph.cwdRelativePath(file);
            } else |_| {}
        } else if (src.nvcc) |tool| image.* = cuda_build.fatbin(b, tool, src.version.?, k, src.sms);
        if (image.*) |file| src.step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
    }
    return out;
}

/// The DeepSeek-V4.1 kernel module (zig/src/families/deepseek_v41/kernels.zig) over `cuda`; null `with`: host-only.
pub fn kernelsModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, with: ?*const Images) *std.Build.Module {
    const options = b.addOptions();
    const embed = if (with) |w| w.any and for (w.main) |i| {
        if (i == null) break false;
    } else true else false;
    options.addOption(bool, "with_kernels", embed);
    options.addOption(usize, "dsv41_images", dsv41_kernels.len); // also keeps this options file apart from the runtime's
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kernels.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addOptions("dsv41_kernel_options", options);
    if (embed) for (dsv41_kernels, with.?.main) |k, image| m.addAnonymousImport(b.fmt("dsv41_fatbin_{s}", .{k.name}), .{ .root_source_file = image.? });
    for (optional, 0..) |k, i| {
        const image = if (embed) with.?.optional[i] else null;
        m.addAnonymousImport(b.fmt("dsv41_fatbin_{s}", .{k.name}), .{ .root_source_file = image orelse b.addWriteFiles().add("empty.fatbin", "") });
    }
    return m;
}

fn tpModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    return @import("tp.zig").module(b, target, optimize, cuda);
}

fn slimCore(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    // the checkpoint reader alone (the family's only use of core): the serving module brings the tokenizer itself
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors_root.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

/// Linux: tf-dsv41-test, tf-dsv41-m1 (every rank; rank 1 `follow`s), tf-dsv41-samp and tensorfold-dsv41 (rank 0).
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, src: Source, build_options: *std.Build.Step.Options) void {
    const imgs = images(b, src);
    const dsv41 = kernelsModule(b, target, optimize, cuda, &imgs);
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/dsv41_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("cuda", cuda);
    runner.addImport("dsv41_kernels", dsv41);
    const runner_exe = b.addExecutable(.{ .name = "tf-dsv41-test", .root_module = runner });
    b.installArtifact(runner_exe);
    b.step("compile-dsv41-tests", "Compile tf-dsv41-test without executing a GPU").dependOn(&runner_exe.step);
    const m1 = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/m1_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m1.addImport("cuda", cuda);
    m1.addImport("core", slimCore(b, target, optimize));
    m1.addImport("dsv41_kernels", dsv41);
    const m1_tp = tpModule(b, target, optimize, cuda);
    m1.addImport("tp", m1_tp); // M2's forward: the TP collectives
    m1.addImport("kv", @import("kv.zig").module(b, target, optimize, cuda, m1_tp)); // M5: the paged pool, split KV, sessions
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m1.addImport("lanes", lanes); // M3: the GPU target's lanes types
    // M4: the served engine (serve_engine.zig) for follow / generate, over the server's engine seam
    m1.addImport("engine_api", b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} }));
    const m1_serve = @import("dsv41_sampling.zig").serve(b, target, optimize, lanes).serve;
    m1.addImport("dsv41_serve", m1_serve); // the keyed sampler's Picker
    // TF_DSV41_CALIB=measure's calibration text (calib_gpu.zig, via serve_engine.zig): the serving module's tokenizer
    m1.addImport("tokenizer", m1_serve.import_table.get("tokenizer").?);
    @import("dsv41_grammar.zig").addTo(b, m1, target, optimize); // structured output (grammar_gpu.zig)
    @import("dsv41_sampling.zig").gpu(b, target, optimize, cuda, dsv41); // tf-dsv41-samp: the keyed sampler's device steps
    const m1_exe = b.addExecutable(.{ .name = "tf-dsv41-m1", .root_module = m1 });
    b.installArtifact(m1_exe);
    // tensorfold-dsv41: the native server over DeepSeek-V4.1's CUDA engine (rank 0; the others: tf-dsv41-m1 follow)
    const engine = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/serve_engine.zig"), .target = target, .optimize = optimize, .link_libc = true });
    engine.addImport("cuda", cuda);
    engine.addImport("core", core); // the same core module as the checkpoint commands (cli.zig)
    engine.addImport("dsv41_kernels", dsv41);
    const engine_tp = tpModule(b, target, optimize, cuda);
    engine.addImport("tp", engine_tp);
    engine.addImport("kv", @import("kv.zig").module(b, target, optimize, cuda, engine_tp));
    @import("dsv41_grammar.zig").addTo(b, engine, target, optimize); // structured output (grammar_gpu.zig)
    const server = b.addExecutable(.{ .name = "tensorfold-dsv41", .root_module = @import("dsv41_server.zig").withEngine(b, target, .ReleaseSafe, engine, core, build_options) });
    b.installArtifact(server);
    const all = b.step("dsv41", "tf-dsv41-test, tf-dsv41-m1 and tensorfold-dsv41 (DeepSeek-V4.1 on CUDA) into zig-out/bin");
    for ([_]*std.Build.Step.Compile{ runner_exe, m1_exe, server }) |exe| all.dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// The family's host tests on the real emitters: `step` name, root file, description, and whether it needs `lanes`.
const emitter_tests = [_]struct { []const u8, []const u8, []const u8, bool }{
    .{ "test-dsv41-rows", "rows_test.zig", "DeepSeek-V4.1 row mode (several slots) on the real emitter, host only", false },
    .{ "test-dsv41-wide", "wide_test.zig", "DeepSeek-V4.1 decode windows past 16 rows on the real emitter, host only", false },
    .{ "test-dsv41-knobs", "prod_knobs_test.zig", "DeepSeek-V4.1 prod knobs (fused dense prefill, rows, read-ahead) on the real prefill emitter, host only", false },
    .{ "test-dsv41-mdraft", "mdraft_test.zig", "DeepSeek-V4.1 DSpark drafting over several slots on the real emitter, host only", true },
    .{ "test-dsv41-longpf", "longpf_test.zig", "DeepSeek-V4.1 long prompts (stream top-k, split KV's row blocks) on the real prefill emitter, host only", false },
    .{ "test-dsv41-vision-plan", "vision_plan_test.zig", "DeepSeek-V4.1 image prompts' buffer roles (the boot check, the plan's options), host only", false },
    .{ "test-dsv41-gm2pf", "gm2pf_test.zig", "DeepSeek-V4.1 x3gm v2 (TF_DSV41_GM_V2) on the real prefill emitter, host only", false },
};

/// The kernels module's own tests, the decode graphs' and every emitter test above, under `step` (test-cuda-host).
pub fn hostTests(b: *std.Build, cuda: *std.Build.Module, step: *std.Build.Step) void {
    const host = b.graph.host;
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = kernelsModule(b, host, .debug, cuda, null) })).step);
    // the DSV4.1 decode graphs' buckets, statics and floor (R4; the cache itself is in the runtime's tests)
    const graphs = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/graphs.zig"), .target = host, .optimize = .debug, .link_libc = true });
    graphs.addImport("cuda", cuda);
    graphs.addImport("tp", tpModule(b, host, .debug, cuda));
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = graphs })).step);
    for (emitter_tests) |t| {
        const m = b.createModule(.{ .root_source_file = b.path(b.fmt("zig/src/families/deepseek_v41/{s}", .{t[1]})), .target = host, .optimize = .debug, .link_libc = true });
        m.addImport("cuda", cuda);
        m.addImport("dsv41_kernels", kernelsModule(b, host, .debug, cuda, null));
        m.addImport("tp", tpModule(b, host, .debug, cuda));
        // the device chain's staging reads draft/dspark.zig's Params (the lanes core's Sampling)
        if (t[3]) m.addImport("lanes", b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .debug, .link_libc = true }));
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
        b.step(t[0], t[2]).dependOn(&run.step);
        step.dependOn(&run.step);
    }
}
