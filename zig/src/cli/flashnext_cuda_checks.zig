//! Flash Next CUDA CLI checks against the Python oracle: agreement, teacher steps, prefill, weights.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flashnext = @import("flashnext_mia");
const cli = @import("flashnext_cuda.zig");

const Engine = flashnext.Engine;
const fwd = flashnext.forward;
const Allocator = std.mem.Allocator;
const Options = cli.Options;
const readJson = cli.readJson;
const deviceSha = cli.deviceSha;
const now = cli.now;
const since = cli.since;
const generate = cli.generate;
const sha12 = cli.sha12;
const idsOf = cli.idsOf;
const listed = cli.listed;

/// agree: each prompt's greedy serial reply; with --against, top-1 agreement teacher-forced on the reference's replies.
pub fn agree(gpa: Allocator, io: std.Io, e: *Engine, capture: []const u8, o: Options) !u8 {
    var pb: [512]u8 = undefined;
    const prompts = try readJson(gpa, io, try std.fmt.bufPrint(&pb, "{s}/prompts.json", .{capture}));
    defer prompts.deinit();
    var ref: ?std.json.Parsed(std.json.Value) = null;
    defer if (ref) |*r| r.deinit();
    if (o.teacher) |path| ref = try readJson(gpa, io, path);
    var report: std.ArrayList(u8) = .empty;
    defer report.deinit(gpa);
    try report.appendSlice(gpa, "{");
    var agree_n: usize = 0;
    var total_n: usize = 0;
    var it = prompts.value.object.iterator();
    var first_key = true;
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        if (!listed(o.only, name)) continue;
        const ids = try idsOf(gpa, kv.value_ptr.*);
        defer gpa.free(ids);
        const g = try generate(gpa, io, e, ids, o.max_tokens, null, false, false);
        defer gpa.free(g.tokens);
        const text = try core.ids_json.write(gpa, g.tokens);
        defer gpa.free(text);
        try report.print(gpa, "{s}\"{s}\":{s}", .{ if (first_key) "" else ",", name, text });
        first_key = false;
        const r = ref orelse {
            std.debug.print("agree {s}: {d} tokens sha {s}\n", .{ name, g.tokens.len, &(try sha12(gpa, g.tokens)) });
            continue;
        };
        const want_v = r.value.object.get(name) orelse continue;
        const want = try idsOf(gpa, want_v);
        defer gpa.free(want);
        if (want.len == 0) {
            std.debug.print("agree {s}: the reference reply is empty, skipped\n", .{name});
            continue;
        }
        var differ_at: usize = @min(want.len, g.tokens.len);
        for (want[0..differ_at], g.tokens[0..differ_at], 0..) |a, b, i| if (a != b) {
            differ_at = i;
            break;
        };
        // teacher-forced: the prompt's draw against want[0], then want[i] fed and the draw against want[i + 1]
        e.absorbing = false;
        var hits: usize = @intFromBool(try e.prefill(ids, null) == want[0]);
        var one: [1]u32 = undefined;
        for (0..want.len - 1) |i| {
            try e.forward(&.{want[i]});
            try e.sample(1, e.pos() + 1, &one);
            try e.commit(1, 1);
            hits += @intFromBool(one[0] == want[i + 1]);
        }
        agree_n += hits;
        total_n += want.len;
        std.debug.print("agree {s}: top-1 {d} of {d} ({d:.2}%), free replies equal for the first {d} of {d} tokens\n", .{ name, hits, want.len, 100.0 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(want.len)), differ_at, want.len });
    }
    try report.appendSlice(gpa, "}");
    if (o.report) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = report.items });
    if (total_n > 0) std.debug.print("agree: top-1 {d} of {d} teacher-forced steps ({d:.2}%)\n", .{ agree_n, total_n, 100.0 * @as(f64, @floatFromInt(agree_n)) / @as(f64, @floatFromInt(total_n)) });
    return 0;
}

/// A step's dump against `dir`/digests.json (the capture's per-layer digests), or none without a dir.
const StepDump = struct {
    parsed: ?std.json.Parsed(std.json.Value) = null,
    dump: fwd.Dump,

    fn open(gpa: Allocator, io: std.Io, dir: ?[]const u8) !StepDump {
        var s: StepDump = .{ .dump = .{ .gpa = gpa, .io = io } };
        const d = dir orelse return s;
        const path = try std.fs.path.join(gpa, &.{ d, "digests.json" });
        defer gpa.free(path);
        s.parsed = readJson(gpa, io, path) catch |err| {
            std.debug.print("  no digests at {s} ({s})\n", .{ path, @errorName(err) });
            return s;
        };
        return s;
    }

    fn arm(s: *StepDump, e: *Engine) void {
        if (s.parsed) |*p| s.dump.want = &p.value.object;
        e.f.dump = &s.dump;
    }

    fn close(s: *StepDump, e: *Engine, label: []const u8) usize {
        e.f.dump = null;
        if (s.parsed != null) std.debug.print("  {s}: dumps {d} equal, {d} differ, {d} not in the oracle\n", .{ label, s.dump.equal, s.dump.differ, s.dump.missing });
        const bad = s.dump.differ;
        s.dump.deinit();
        if (s.parsed) |*p| p.deinit();
        return bad;
    }
};

/// The capture's teacher steps: step i forwards tokens[i] at pos i greedily, the MTP head absorbs, then commit(1, 1).
pub fn teacher(gpa: Allocator, io: std.Io, e: *Engine, path: []const u8, dump_root: ?[]const u8, with_mtp: bool) !u8 {
    const t = try readJson(gpa, io, path);
    defer t.deinit();
    const ob = t.value.object;
    const tokens = ob.get("tokens").?.array.items;
    const sampled = ob.get("sampled").?.array.items;
    const logits = ob.get("logits_sha256").?.array.items;
    const mtp_next = ob.get("mtp_next").?.array.items;
    const mtp_sha = ob.get("mtp_logits_sha256").?.array.items;
    e.setSampling(null);
    try e.reset();
    var bad: usize = 0;
    var bad_mtp: usize = 0;
    var bad_dumps: usize = 0;
    const V = e.g.head_n;
    for (tokens, 0..) |tok, step| {
        var dir_buf: [512]u8 = undefined;
        const dir: ?[]const u8 = if (dump_root) |r| try std.fmt.bufPrint(&dir_buf, "{s}/step{d:0>3}", .{ r, step }) else null;
        var sd = try StepDump.open(gpa, io, dir);
        sd.arm(e);
        try e.forward(&.{@intCast(tok.integer)});
        var got: [1]u32 = undefined;
        try e.sample(1, e.pos() + 1, &got);
        const lsha = try deviceSha(gpa, e, e.buf.b.logits, V * 2);
        // the oracle's full tensors (dump steps): how close (two ranks have no bit-exact oracle)
        if (dir) |dd| {
            try closeness(gpa, io, e, dd, "final_mixed", e.buf.b.mixed, 0, e.g.hidden);
            try closeness(gpa, io, e, dd, "logits", e.buf.b.logits, @intCast(e.w.vocab_offset), V);
        }
        const ok = got[0] == sampled[step].integer and std.mem.eql(u8, &lsha, logits[step].string);
        var mtp_ok = true;
        var mtp_tok: u32 = 0;
        var msha_ok = true;
        if (with_mtp and step + 1 < tokens.len and e.w.mtp != null) {
            const lg = try e.teacherMtp(@intCast(tokens[step + 1].integer));
            const n = e.draws.?.columns;
            const msha = try deviceSha(gpa, e, lg, n * 2);
            mtp_tok = try e.drawDraft(e.pos() + 2);
            msha_ok = std.mem.eql(u8, &msha, mtp_sha[step].string);
            mtp_ok = mtp_tok == mtp_next[step].integer and msha_ok;
        }
        try e.commit(1, 1);
        var label: [32]u8 = undefined;
        bad_dumps += sd.close(e, try std.fmt.bufPrint(&label, "step {d}", .{step}));
        if (!ok) bad += 1;
        if (!mtp_ok) bad_mtp += 1;
        if (!ok or !mtp_ok) std.debug.print("step {d}: token {d} (oracle {d}) logits {s}; MTP draw {d} (oracle {d}) logits {s}\n", .{ step, got[0], sampled[step].integer, if (std.mem.eql(u8, &lsha, logits[step].string)) "equal" else "DIFFER", mtp_tok, if (step < mtp_next.len) mtp_next[step].integer else -1, if (msha_ok) "equal" else "DIFFER" });
    }
    const pass = bad == 0 and bad_mtp == 0 and bad_dumps == 0 and with_mtp;
    std.debug.print("{s} teacher: {d} of {d} steps' tokens and logits bit-equal, {d} of {d} MTP draws and logits, {d} dump mismatches\n", .{ if (pass) "PASS" else "FAIL", tokens.len - bad, tokens.len, mtp_next.len - bad_mtp, mtp_next.len, bad_dumps });
    return if (pass) 0 else 1;
}

fn bf16f(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// `n` bf16 values at `ptr` against `dir`/`name`.bin from `offset`: max |diff|, max |ref|, bit-equal, argmax agrees.
fn closeness(gpa: Allocator, io: std.Io, e: *Engine, dir: []const u8, name: []const u8, ptr: u64, offset: usize, n: usize) !void {
    var pb: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&pb, "{s}/{s}.bin", .{ dir, name });
    const ref = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26)) catch return;
    defer gpa.free(ref);
    if (ref.len < (offset + n) * 2) return;
    const host = try gpa.alloc(u16, n);
    defer gpa.free(host);
    try e.stream.synchronize();
    const buf: cuda.DeviceBuffer = .{ .d = e.ctx.d, .ptr = ptr, .len = n * 2 };
    try buf.download(0, std.mem.sliceAsBytes(host));
    var diff: f32 = 0;
    var amax: f32 = 0;
    var same: usize = 0;
    var best_a: usize = 0;
    var best_b: usize = 0;
    for (host, 0..) |h, j| {
        const r = std.mem.readInt(u16, ref[(offset + j) * 2 ..][0..2], .little);
        const a = bf16f(h);
        const b = bf16f(r);
        diff = @max(diff, @abs(a - b));
        amax = @max(amax, @abs(b));
        same += @intFromBool(h == r);
        if (a > bf16f(host[best_a])) best_a = j;
        if (b > bf16f(std.mem.readInt(u16, ref[(offset + best_b) * 2 ..][0..2], .little))) best_b = j;
    }
    std.debug.print("  {s} vs the oracle: max |diff| {d:.5} (|ref| max {d:.3}), {d} of {d} bit-equal, argmax {d} vs {d}\n", .{ name, diff, amax, same, n, best_a + offset, best_b + offset });
}

/// One captured prompt's prefill (chunks of 2048 into the MTP head) against teacher.json's "prefill" and the digests.
pub fn prefillCheck(gpa: Allocator, io: std.Io, e: *Engine, prompts_path: []const u8, name: []const u8, dump: ?[]const u8, teacher_path: ?[]const u8) !u8 {
    const p = try readJson(gpa, io, prompts_path);
    defer p.deinit();
    const ids = try idsOf(gpa, p.value.object.get(name) orelse return error.NoSuchPrompt);
    defer gpa.free(ids);
    var sd = try StepDump.open(gpa, io, dump);
    sd.arm(e);
    e.absorbing = true;
    const t0 = now(io);
    const first = try e.prefill(ids, null);
    try e.stream.synchronize();
    const secs = since(io, t0);
    const ls = try deviceSha(gpa, e, e.lastStreams(), e.g.wide() * 2);
    const bad_dumps = sd.close(e, name);
    std.debug.print("RESULT prefill {s}: {d} tokens in {d:.3}s, first token {d}, pos {d}, mtp_len {d}, last streams {s}\n", .{ name, ids.len, secs, first, e.bound.st.pos, e.bound.st.mtp_len, ls[0..12] });
    var pass = bad_dumps == 0;
    if (teacher_path) |tp| {
        const t = try readJson(gpa, io, tp);
        defer t.deinit();
        const want = t.value.object.get("prefill").?.object.get(name) orelse return error.NoSuchPrompt;
        const w = want.object;
        const same = first == w.get("first").?.integer and e.bound.st.pos == w.get("pos").?.integer and e.bound.st.mtp_len == w.get("mtp_len").?.integer and std.mem.eql(u8, &ls, w.get("last_streams").?.string);
        pass = pass and same;
        std.debug.print("{s} prefill {s} against the oracle: first {d} (oracle {d}), last streams {s}\n", .{ if (same) "PASS" else "FAIL", name, first, w.get("first").?.integer, if (std.mem.eql(u8, &ls, w.get("last_streams").?.string)) "equal" else "DIFFER" });
    }
    return if (pass) 0 else 1;
}

/// Every device buffer's sha256 against the oracle's weights.json (W4's digests), by Python's dotted name.
pub fn checkWeights(gpa: Allocator, io: std.Io, e: *Engine, path: []const u8) !u8 {
    const want = try readJson(gpa, io, path);
    defer want.deinit();
    const ds = try flashnext.weights.digests(gpa, &e.w);
    defer flashnext.weights.freeDigests(gpa, ds);
    var same: usize = 0;
    var bad: usize = 0;
    var ours: usize = 0;
    for (ds) |d| {
        if (!d.oracle) {
            ours += 1;
            continue;
        }
        const expect = want.value.object.get(d.name) orelse {
            std.debug.print("MISSING {s}: the oracle has no tensor of this name\n", .{d.name});
            bad += 1;
            continue;
        };
        if (std.mem.eql(u8, &d.sha256, expect.string)) same += 1 else {
            std.debug.print("DIFFER {s}: {d} bytes {s}\n", .{ d.name, d.len, d.shape });
            bad += 1;
        }
    }
    var names: usize = 0;
    var it = want.value.object.iterator();
    while (it.next()) |kv| names += @intFromBool(!std.mem.endsWith(u8, kv.key_ptr.*, ".shape"));
    const pass = bad == 0 and same == names;
    std.debug.print("{s} weights: {d} equal, {d} wrong, {d} ours alone, {d} in the oracle\n", .{ if (pass) "PASS" else "FAIL", same, bad, ours, names });
    return if (pass) 0 else 1;
}
