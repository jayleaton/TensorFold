//! Tensor parallelism's build (zig/src/fabric/tp): host tests on any OS, and on Linux the `tf-tp-test` GPU runner with
//! the mailbox and roce_gather kernels, built by `-Dtp-nvcc=` or embedded from `-Dtp-fatbin=` / `-Dtp-roce-fatbin=`.

const std = @import("std");

/// The CUDA runtime without the engine's kernels (tp only needs the driver, memory, streams and NCCL).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    return cuda;
}

/// The `tp` module over `cuda` (an engine passes its own runtime module); verbs (zig/src/fabric/verbs.zig) for RoCE.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/tp/tp.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("verbs", b.createModule(.{ .root_source_file = b.path("zig/src/fabric/verbs.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    return m;
}

/// A .cu as a fatbin: sm_120 (RTX PRO 6000), sm_121 (GB10), and compute_90 PTX for anything else.
fn tpFatbin(b: *std.Build, nvcc: []const u8, src: []const u8, name: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin", "-O3", "-std=c++17", "-gencode=arch=compute_120,code=sm_120", "-gencode=arch=compute_121,code=sm_121", "-gencode=arch=compute_90,code=compute_90", "-o" });
    const out = run.addOutputFileArg(name);
    run.addFileArg(b.path(src));
    return out;
}

/// `pub const mailbox` and `pub const roce_gather: []const u8`, each a fatbin's bytes or empty.
fn kernelModule(b: *std.Build, image: ?std.Build.LazyPath, roce: ?std.Build.LazyPath) *std.Build.Module {
    const wf = b.addWriteFiles();
    var text: std.ArrayList(u8) = .empty;
    for ([_]?std.Build.LazyPath{ image, roce }, [_][]const u8{ "mailbox", "roce_gather" }) |file, name| {
        if (file) |f| {
            _ = wf.addCopyFile(f, b.fmt("tp_{s}.fatbin", .{name}));
            // the driver takes 8-byte aligned images (Module.load refuses others); @embedFile alone promises nothing
            text.print(b.allocator, "const {s}_bytes align(16) = @embedFile(\"tp_{s}.fatbin\").*;\npub const {s}: []const u8 = &{s}_bytes;\n", .{ name, name, name, name }) catch @panic("OOM");
        } else text.print(b.allocator, "pub const {s}: []const u8 = \"\";\n", .{name}) catch @panic("OOM");
    }
    return b.createModule(.{ .root_source_file = wf.add("tp_kernel.zig", text.items) });
}

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const tests = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, runtime(b, host, .debug)) }));
    b.step("test-tp", "Tensor parallelism's host tests: bootstrap, fate channel, in-process ranks (no GPU)").dependOn(&tests.step);
    test_step.dependOn(&tests.step);
    if (target.result.os.tag != .linux) return;
    const nvcc = b.option([]const u8, "tp-nvcc", "nvcc that builds the TP mailbox kernel");
    const prebuilt = b.option([]const u8, "tp-fatbin", "absolute path of a prebuilt tp_mailbox.fatbin to embed");
    const roce_prebuilt = b.option([]const u8, "tp-roce-fatbin", "absolute path of a prebuilt roce_gather.fatbin (TF_TP_ROCE_ONESHOT)");
    const image: ?std.Build.LazyPath = if (prebuilt) |p| b.graph.cwdRelativePath(p) else if (nvcc) |tool| tpFatbin(b, tool, "zig/src/fabric/tp/mailbox.cu", "tp_mailbox.fatbin") else null;
    const roce: ?std.Build.LazyPath = if (roce_prebuilt) |p| b.graph.cwdRelativePath(p) else if (nvcc) |tool| tpFatbin(b, tool, "zig/kernels/cuda/roce_gather.cu", "roce_gather.fatbin") else null;
    const cuda = runtime(b, target, optimize);
    const runner = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/tp/gpu_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("cuda", cuda);
    runner.addImport("tp", module(b, target, optimize, cuda));
    runner.addImport("tp_kernel", kernelModule(b, image, roce));
    const exe = b.addExecutable(.{ .name = "tf-tp-test", .root_module = runner });
    b.installArtifact(exe);
    b.step("tf-tp-test", "TP=2 on two GPUs: bootstrap, every collective bit-exact, graph capture, fail-fast, latency").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
