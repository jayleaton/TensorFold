//! Flash Next CUDA CLI gates: captured prompts, media cases, shared rounds and prompt reuse.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const flashnext = @import("flashnext_mia");
const cli = @import("flashnext_cuda.zig");

const Engine = flashnext.Engine;
const decode = flashnext.decode;
const Allocator = std.mem.Allocator;
const Options = cli.Options;
const readJson = cli.readJson;
const shaHex = cli.shaHex;
const now = cli.now;
const since = cli.since;
const generate = cli.generate;
const generateMedia = cli.generateMedia;
const sha12 = cli.sha12;
const idsOf = cli.idsOf;
const listed = cli.listed;

/// The run gate: every captured prompt, serial and drafted, greedy and sampled, token for token against results.json.
pub fn gate(gpa: Allocator, io: std.Io, e: *Engine, capture: []const u8, o: Options) !u8 {
    var pb: [512]u8 = undefined;
    const prompts = try readJson(gpa, io, try std.fmt.bufPrint(&pb, "{s}/prompts.json", .{capture}));
    defer prompts.deinit();
    const results = try readJson(gpa, io, try std.fmt.bufPrint(&pb, "{s}/results.json", .{capture}));
    defer results.deinit();
    const res = results.value.object.get("results") orelse {
        std.debug.print("{s}/results.json has no results yet (a partial capture)\n", .{capture});
        return 2;
    };
    const samplings = [_]struct { name: []const u8, s: ?lanes.Sampling }{ .{ .name = "greedy", .s = null }, .{ .name = "sampled", .s = .{ .seed = 1234, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.0 } } };
    var bad: usize = 0;
    var total: usize = 0;
    var report: std.ArrayList(u8) = .empty;
    defer report.deinit(gpa);
    try report.appendSlice(gpa, "{");
    for (samplings) |smp| {
        var it = prompts.value.object.iterator();
        while (it.next()) |kv| {
            const name = kv.key_ptr.*;
            if (!listed(o.only, name)) continue;
            const ids = try idsOf(gpa, kv.value_ptr.*);
            defer gpa.free(ids);
            var serial_tokens: ?[]u32 = null;
            defer if (serial_tokens) |s| gpa.free(s);
            for ([_]bool{ false, true }) |drafts| {
                if (drafts and !o.drafts) continue; // gate --no-drafts: the serial runs alone
                const label = if (drafts) "drafted" else "serial";
                var kb: [128]u8 = undefined;
                const key = try std.fmt.bufPrint(&kb, "{s}-{s}/{s}", .{ label, smp.name, name });
                const g = try generate(gpa, io, e, ids, o.max_tokens, smp.s, drafts, true);
                var keep_tokens = false;
                defer if (!keep_tokens) gpa.free(g.tokens);
                const want = res.object.get(key);
                var equal = false;
                var oracle_n: usize = 0;
                if (want) |w| {
                    const wt = w.object.get("tokens").?.array.items;
                    oracle_n = wt.len;
                    // a shorter --max-tokens compares the oracle's first tokens
                    equal = wt.len == g.tokens.len or (g.tokens.len == o.max_tokens and wt.len > g.tokens.len);
                    if (equal) for (wt[0..g.tokens.len], g.tokens) |a, b| {
                        if (a.integer != b) {
                            equal = false;
                            break;
                        }
                    };
                }
                var db: [64]u8 = undefined;
                var first_diff: usize = 0;
                if (!equal and want != null) {
                    const wt = want.?.object.get("tokens").?.array.items;
                    while (first_diff < @min(wt.len, g.tokens.len) and wt[first_diff].integer == g.tokens[first_diff]) first_diff += 1;
                }
                var same_serial = true;
                if (drafts) {
                    if (serial_tokens) |s| same_serial = std.mem.eql(u32, s, g.tokens);
                } else {
                    serial_tokens = g.tokens;
                    keep_tokens = true;
                }
                total += 1;
                // no oracle at two ranks or for an empty results.json (INT4-AutoRound): drafted == serial is the gate
                const no_oracle = e.world > 1 or res.object.count() == 0;
                const ok = (equal or no_oracle) and same_serial;
                const verdict = if (no_oracle) (if (!same_serial) "DIFFER" else if (equal) "TP1-SAME" else if (e.world > 1) "TP1-OTHER" else "SELF") else if (ok) "EQUAL" else "DIFFER";
                if (!ok) bad += 1;
                const sha = try sha12(gpa, g.tokens);
                const steps = @max(1, g.tokens.len - 1);
                std.debug.print("{s} {s}: {d} tokens sha {s} (oracle {d}{s}{s}) prefill {d:.3}s decode {d:.2} ms/token rounds {d} accepted {d}{s}\n", .{ verdict, key, g.tokens.len, sha, oracle_n, if (want == null) ", missing" else "", if (!equal and want != null) try std.fmt.bufPrint(&db, ", first difference at {d}", .{first_diff}) else "", g.prefill_s, g.decode_s * 1e3 / @as(f64, @floatFromInt(steps)), g.rounds, g.accepted, if (drafts and !same_serial) "; drafted != serial" else "" });
                if (report.items.len > 1) try report.appendSlice(gpa, ",");
                const entry = .{ .tokens = g.tokens, .sha = sha, .equal = equal, .drafted_equals_serial = same_serial, .prefill_s = g.prefill_s, .decode_s = g.decode_s, .rounds = g.rounds, .drafted = g.drafted, .accepted = g.accepted };
                const json = try std.json.Stringify.valueAlloc(gpa, entry, .{});
                defer gpa.free(json);
                try report.print(gpa, "\"{s}\":{s}", .{ key, json });
            }
        }
    }
    try report.appendSlice(gpa, "}");
    if (o.report) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = report.items });
    std.debug.print("{s} gate: {d} of {d} runs equal the oracle's tokens (drafted == serial)\n", .{ if (bad == 0) "PASS" else "FAIL", total - bad, total });
    return if (bad == 0) 0 else 1;
}

/// One image or video case of zig/tests/cuda/flashnext_mia/vision_ref.py: the prompt, its media and Python's results.
const MediaCase = struct {
    name: []const u8,
    tokens: []u32,
    media: flashnext.media.Media,
    blobs: [2][]align(16) u8,
    rows: []u32,
    expected: ?std.json.Value,
    parsed: std.json.Parsed(std.json.Value),

    fn deinit(c: *MediaCase, gpa: Allocator) void {
        gpa.free(c.tokens);
        gpa.free(c.rows);
        for (c.blobs) |b| gpa.free(b);
        c.parsed.deinit();
        gpa.free(c.name);
    }
};

fn loadMediaCase(gpa: Allocator, io: std.Io, dir: []const u8, name: []const u8, lead: bool) !MediaCase {
    var pb: [512]u8 = undefined;
    const parsed = try readJson(gpa, io, try std.fmt.bufPrint(&pb, "{s}/{s}/bundle.json", .{ dir, name }));
    errdefer parsed.deinit();
    const b = parsed.value.object;
    const tokens = try idsOf(gpa, b.get("tokens").?);
    errdefer gpa.free(tokens);
    const rows = try idsOf(gpa, b.get("rows").?);
    errdefer gpa.free(rows);
    const pos = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fmt.bufPrint(&pb, "{s}/{s}/positions.bin", .{ dir, name }), gpa, .limited(1 << 32), .@"16", null);
    errdefer gpa.free(pos);
    // a following rank reads no features (they reach it by rank 0's broadcast): its box needs no features.bin
    const feats = if (lead) try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fmt.bufPrint(&pb, "{s}/{s}/features.bin", .{ dir, name }), gpa, .limited(1 << 34), .@"16", null) else try gpa.alignedAlloc(u8, .@"16", 0);
    errdefer gpa.free(feats);
    if (lead) {
        const want = b.get("features_sha256").?.string;
        const got = shaHex(feats);
        if (!std.mem.eql(u8, want, &got)) return error.FeaturesShaMismatch;
    }
    const pos_i32: []const i32 = std.mem.bytesAsSlice(i32, pos);
    return .{ .name = try gpa.dupe(u8, name), .tokens = tokens, .media = .{ .rows = rows, .positions = pos_i32, .delta = b.get("delta").?.integer, .features = feats, .width = @intCast(b.get("width").?.integer) }, .blobs = .{ pos, feats }, .rows = rows, .expected = b.get("expected"), .parsed = parsed };
}

/// Two ranks: whether both hold the same `v` (an all-gather of it); one rank: true.
fn sameOnRanks(e: *Engine, v: u64) !bool {
    const cm = e.f.comm orelse return true;
    const vote = e.f.sc.vote;
    const buf: cuda.DeviceBuffer = .{ .d = e.f.d, .ptr = vote, .len = 8 };
    try e.stream.synchronize();
    try buf.upload(0, std.mem.asBytes(&v));
    try cm.allGather(vote, vote + 64, 1, .u64, e.stream.handle);
    try e.stream.synchronize();
    var both: [2]u64 = undefined;
    const got: cuda.DeviceBuffer = .{ .d = e.f.d, .ptr = vote + 64, .len = 16 };
    try got.download(0, std.mem.sliceAsBytes(&both));
    return both[0] == both[1];
}

fn tokensHash(tokens: []const u32) u64 {
    return std.hash.Wyhash.hash(0x6d65646961, std.mem.sliceAsBytes(tokens));
}

fn tokensEqual(want: std.json.Value, got: []const u32) bool {
    const w = want.array.items;
    if (w.len != got.len) return false;
    for (w, got) |a, b| if (a.integer != b) return false;
    return true;
}

/// gate-media DIR: every vision_ref.py case's digests and tokens against Python's, then all cases in shared rounds.
pub fn gateMedia(gpa: Allocator, io: std.Io, e: *Engine, dir: []const u8, o: Options) !u8 {
    var names: std.ArrayList([]const u8) = .empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    {
        var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
        defer d.close(io);
        var it = d.iterate();
        while (try it.next(io)) |entry| {
            if ((entry.kind != .directory and entry.kind != .sym_link) or !listed(o.only, entry.name)) continue;
            var pb: [512]u8 = undefined;
            std.Io.Dir.cwd().access(io, try std.fmt.bufPrint(&pb, "{s}/{s}/bundle.json", .{ dir, entry.name }), .{}) catch continue;
            try names.append(gpa, try gpa.dupe(u8, entry.name));
        }
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    const cases = try gpa.alloc(MediaCase, names.items.len);
    var loaded: usize = 0;
    defer {
        for (cases[0..loaded]) |*c| c.deinit(gpa);
        gpa.free(cases);
    }
    for (names.items, cases) |n, *c| {
        // a following rank holds no features: they must reach it by rank 0's broadcast, as in serving
        c.* = try loadMediaCase(gpa, io, dir, n, e.rank == 0);
        loaded += 1;
        // as in serving: rank 1 has only the rows' count; the rows, table and features come from rank 0
        if (e.rank != 0) {
            // (read before the assignment: the struct literal writes into c.media as it goes)
            const nrows: u32 = @intCast(c.media.rows.len);
            const delta = c.media.delta;
            const width = c.media.width;
            c.media = .{ .rows = &.{}, .positions = &.{}, .delta = delta, .features = &.{}, .width = width, .follower_rows = nrows };
        }
    }
    var digests: std.ArrayList([2][16]u8) = .empty;
    defer digests.deinit(gpa);
    const keep_digest = e.digest;
    defer {
        e.digest = keep_digest;
        e.digests = null;
    }
    var bad: usize = 0;
    var total: usize = 0;
    const solo = try gpa.alloc([]u32, cases.len);
    var solo_n: usize = 0;
    defer {
        for (solo[0..solo_n]) |t| gpa.free(t);
        gpa.free(solo);
    }
    for (cases) |*c| {
        const exp = c.expected;
        const count = if (exp) |x| x.object.get("serial").?.array.items.len else o.max_tokens;
        var serial_tokens: ?[]u32 = null;
        defer if (serial_tokens) |t| gpa.free(t);
        for ([_]bool{ false, true }) |drafts| {
            if (drafts and !o.drafts) continue;
            digests.clearRetainingCapacity();
            e.digest = !drafts and e.world == 1;
            e.digests = if (e.digest) &digests else null;
            const g = try generateMedia(gpa, io, e, c.tokens, count, null, drafts, false, &c.media);
            e.digest = false;
            e.digests = null;
            var keep = false;
            defer if (!keep) gpa.free(g.tokens);
            const label = if (drafts) "drafted" else "serial";
            var equal = true;
            var digest_ok = true;
            if (exp) |x| {
                equal = tokensEqual(x.object.get(label).?, g.tokens);
                if (!drafts and e.world == 1) {
                    const want = x.object.get("digests").?.array.items;
                    digest_ok = want.len == digests.items.len;
                    if (digest_ok) for (want, digests.items) |w, got| {
                        const ws = w.object.get("streams").?.string;
                        const wl = if (w.object.get("logits")) |l| l.string else "----------------";
                        if (!std.mem.eql(u8, ws, &got[0]) or !std.mem.eql(u8, wl, &got[1])) digest_ok = false;
                    };
                }
            }
            // drafted == serial on every case, with or without an oracle
            const same_serial = if (drafts) (if (serial_tokens) |t| std.mem.eql(u32, t, g.tokens) else true) else true;
            // two ranks: the peer's tokens must be these (rank identity)
            const ranks_same = try sameOnRanks(e, tokensHash(g.tokens));
            if (!ranks_same) std.debug.print("ranks differ for {s}\n", .{c.name});
            const ok = (equal or e.world > 1) and digest_ok and same_serial and ranks_same;
            total += 1;
            if (!ok) bad += 1;
            const sha = try sha12(gpa, g.tokens);
            if (drafts and !same_serial) std.debug.print("drafted != serial for {s}\n", .{c.name});
            std.debug.print("{s} media {s}/{s}: {d} prompt tokens, {d} visual rows, {d} tokens sha {s}{s}{s} prefill {d:.3}s decode {d:.2} ms/token rounds {d} accepted {d}\n", .{ if (!ok) "DIFFER" else if (e.world > 1) (if (equal) "TP1-SAME" else "TP1-OTHER") else "EQUAL", c.name, label, c.tokens.len, c.rows.len, g.tokens.len, sha, if (exp == null) " (no oracle)" else "", if (!digest_ok) " (prompt digests differ)" else "", g.prefill_s, g.decode_s * 1e3 / @as(f64, @floatFromInt(@max(1, g.tokens.len - 1))), g.rounds, g.accepted });
            if (drafts) {
                solo[solo_n] = g.tokens;
                solo_n += 1;
                keep = true;
            } else if (o.drafts) {
                serial_tokens = g.tokens;
                keep = true;
            }
        }
    }
    // shared rounds: every case together (each equals its solo drafted run), then arriving one a round in layer slices
    if (o.drafts and solo_n == cases.len and cases.len > 1 and cases.len <= 64) {
        // --burst: text prompts around the media prompts, so media prompts prefill alone; every stream == its solo run
        const n_text: usize = if (o.burst) 4 else 0;
        const text_solo = try gpa.alloc([]u32, n_text);
        var text_n: usize = 0;
        defer {
            for (text_solo[0..text_n]) |t| gpa.free(t);
            gpa.free(text_solo);
        }
        const text_len = [_]usize{ 200, 150, 100, 60 };
        for (0..n_text) |k| {
            const pl = @min(cases[0].tokens.len, text_len[k]);
            const g = try generateMedia(gpa, io, e, cases[0].tokens[0..pl], 48, null, true, false, null);
            text_solo[k] = g.tokens;
            text_n += 1;
        }
        for ([_]usize{ 0, 8 }) |fill| {
            const total_reqs = cases.len + n_text;
            const reqs = try gpa.alloc(Engine.Request, total_reqs);
            defer {
                for (reqs) |*r| r.tokens.deinit(gpa);
                gpa.free(reqs);
            }
            const want = try gpa.alloc([]const u32, total_reqs);
            defer gpa.free(want);
            const names_of = try gpa.alloc([]const u8, total_reqs);
            defer gpa.free(names_of);
            var at: usize = 0;
            var ti: usize = 0;
            for (cases, 0..) |*c, ci| {
                // two texts before the first media prompt, two after it, then the media prompts as they come
                while (ti < n_text and ((ci == 0 and ti < 2) or (ci == 1 and ti >= 2))) : (ti += 1) {
                    reqs[at] = .{ .prompt = cases[0].tokens[0..@min(cases[0].tokens.len, text_len[ti])], .max_tokens = text_solo[ti].len, .stop_eos = false, .arrive = if (fill > 0) at else 0, .confidence = flashnext.engine.confidenceSetting() };
                    want[at] = text_solo[ti];
                    names_of[at] = "text";
                    at += 1;
                }
                reqs[at] = .{ .prompt = c.tokens, .max_tokens = solo[ci].len, .stop_eos = false, .media = &c.media, .arrive = if (fill > 0) at else 0, .confidence = flashnext.engine.confidenceSetting() };
                want[at] = solo[ci];
                names_of[at] = c.name;
                at += 1;
            }
            while (ti < n_text) : (ti += 1) {
                reqs[at] = .{ .prompt = cases[0].tokens[0..@min(cases[0].tokens.len, text_len[ti])], .max_tokens = text_solo[ti].len, .stop_eos = false, .arrive = if (fill > 0) at else 0, .confidence = flashnext.engine.confidenceSetting() };
                want[at] = text_solo[ti];
                names_of[at] = "text";
                at += 1;
            }
            try e.generateManyWith(gpa, reqs, .{ .fill_layers = fill, .burst = o.burst });
            for (reqs, want, names_of) |r, w, nm| {
                const same = std.mem.eql(u32, r.tokens.items, w) and try sameOnRanks(e, tokensHash(r.tokens.items));
                total += 1;
                if (!same) bad += 1;
                std.debug.print("{s} media {s}/together{s}{s}: {d} tokens{s}\n", .{ if (same) "EQUAL" else "DIFFER", nm, if (fill > 0) "-arriving" else "", if (o.burst) "-burst" else "", r.tokens.items.len, if (same) " == solo" else " != solo" });
            }
        }
    }
    std.debug.print("{s} gate-media: {d} of {d} runs equal\n", .{ if (bad == 0) "PASS" else "FAIL", total - bad, total });
    return if (bad == 0) 0 else 1;
}

/// The shared-round gate: N streams decoded together, drafted, serial and mixed, each against its solo run.
pub fn gateMany(gpa: Allocator, io: std.Io, e: *Engine, capture: []const u8, o: Options) !u8 {
    var pb: [512]u8 = undefined;
    const prompts = try readJson(gpa, io, try std.fmt.bufPrint(&pb, "{s}/prompts.json", .{capture}));
    defer prompts.deinit();
    const results = try readJson(gpa, io, try std.fmt.bufPrint(&pb, "{s}/results.json", .{capture}));
    defer results.deinit();
    const res = results.value.object.get("results") orelse return error.PartialCapture;
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(gpa);
    var it = prompts.value.object.iterator();
    while (it.next()) |kv| if (listed(o.only, kv.key_ptr.*)) try names.append(gpa, kv.key_ptr.*);
    const sampled: lanes.Sampling = .{ .seed = 1234, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.0 };
    const N = o.streams;
    const ids = try gpa.alloc([]u32, names.items.len);
    defer {
        for (ids) |x| gpa.free(x);
        gpa.free(ids);
    }
    for (names.items, ids) |nm, *x| {
        // --repeat K: the prompt's ids K times (long contexts for the memory checks; no oracle: --against-solo)
        const one = try idsOf(gpa, prompts.value.object.get(nm).?);
        defer gpa.free(one);
        x.* = try gpa.alloc(u32, one.len * o.repeat);
        for (0..o.repeat) |k| @memcpy(x.*[k * one.len ..][0..one.len], one);
    }
    var bad: usize = 0;
    var total: usize = 0;
    for ([_][]const u8{ "drafted", "serial", "mixed" }) |pass| {
        const reqs = try gpa.alloc(Engine.Request, N);
        defer {
            for (reqs) |*r| r.tokens.deinit(gpa);
            gpa.free(reqs);
        }
        for (reqs, 0..) |*r, i| {
            const greedy = (i + i / names.items.len) % 2 == 0;
            const drafts = if (std.mem.eql(u8, pass, "drafted")) true else if (std.mem.eql(u8, pass, "serial")) false else i % 2 == 1;
            // --arrive K --fill-layers L: odd streams arrive at rounds K, 2K, ... and fill L layers between rounds
            const arrive = if (o.fill_layers > 0 and i % 2 == 1) o.arrive * (1 + i / 2) else 0;
            // --served: the server's stop rule (the running product) instead of Python's 0.70
            const conf = if (o.served) flashnext.engine.confidenceSetting() else flashnext.engine.default_confidence;
            r.* = .{ .prompt = ids[i % names.items.len], .max_tokens = o.max_tokens, .sampling = if (greedy) null else sampled, .drafts = drafts, .arrive = arrive, .confidence = conf };
        }
        const t0 = now(io);
        try e.generateManyWith(gpa, reqs, .{ .fill_layers = o.fill_layers, .burst = o.burst });
        const secs = since(io, t0);
        const pass_times = e.times;
        var produced: usize = 0;
        for (reqs, 0..) |r, i| {
            produced += r.tokens.items.len;
            const name = names.items[i % names.items.len];
            var kb: [128]u8 = undefined;
            const key = try std.fmt.bufPrint(&kb, "{s}-{s}/{s}", .{ if (r.drafts) "drafted" else "serial", if (r.sampling == null) "greedy" else "sampled", name });
            var equal = false;
            var first_diff: usize = 0;
            if (o.against_solo) {
                // the same request alone in this engine
                var solo = [1]Engine.Request{.{ .prompt = r.prompt, .max_tokens = r.max_tokens, .sampling = r.sampling, .drafts = r.drafts, .confidence = r.confidence }};
                defer solo[0].tokens.deinit(gpa);
                try e.generateMany(gpa, &solo);
                const st = solo[0].tokens.items;
                while (first_diff < @min(st.len, r.tokens.items.len) and st[first_diff] == r.tokens.items[first_diff]) first_diff += 1;
                equal = st.len == r.tokens.items.len and first_diff == st.len;
            } else if (res.object.get(key)) |w| {
                const wt = w.object.get("tokens").?.array.items;
                const n = @min(wt.len, r.tokens.items.len);
                while (first_diff < n and wt[first_diff].integer == r.tokens.items[first_diff]) first_diff += 1;
                equal = first_diff == n and (wt.len == r.tokens.items.len or r.tokens.items.len == o.max_tokens);
            }
            total += 1;
            if (!equal and (e.world == 1 or o.against_solo)) bad += 1;
            const sha = try sha12(gpa, r.tokens.items);
            std.debug.print("{s} {s} stream {d} {s}: {d} tokens sha {s}{s} rounds {d} accepted {d}\n", .{ if (equal) "EQUAL" else if (e.world > 1) "TP2" else "DIFFER", pass, i, key, r.tokens.items.len, sha, if (equal) "" else try std.fmt.bufPrint(&pb, " (first difference at {d})", .{first_diff}), r.rounds, r.accepted });
        }
        const tm = pass_times;
        const dec = tm.verify + tm.sample + tm.draft;
        std.debug.print("RESULT {s} x{d}: {d} tokens in {d:.2}s ({d:.1} tokens/s together); prefill {d:.2}s, then {d:.1} tokens/s decoding over {d} rounds: verify {d:.1} ms (enqueued in {d:.1}), draws+commits {d:.1} ms, drafting {d:.1} ms a round\n", .{ pass, N, produced, secs, @as(f64, @floatFromInt(produced)) / secs, tm.prefill, @as(f64, @floatFromInt(produced - N)) / dec, tm.rounds, tm.verify * 1e3 / @as(f64, @floatFromInt(@max(1, tm.rounds))), tm.verify_host * 1e3 / @as(f64, @floatFromInt(@max(1, tm.rounds))), tm.sample * 1e3 / @as(f64, @floatFromInt(@max(1, tm.rounds))), tm.draft * 1e3 / @as(f64, @floatFromInt(@max(1, tm.rounds))) });
    }
    std.debug.print("{s} gate-many x{d}: {d} of {d} streams equal their solo runs\n", .{ if (bad == 0) "PASS" else "FAIL", N, total - bad, total });
    return if (bad == 0) 0 else 1;
}

/// Decode `count` tokens (drafted) on the bound sequence after `first`.
fn decodeBound(gpa: Allocator, e: *Engine, first: u32, count: usize) ![]u32 {
    var r = try decode.drafted(Engine, e, gpa, first, .{ .count = count, .depth = e.depth, .confidence = flashnext.engine.confidenceSetting(), .stop_eos = false });
    return r.tokens.toOwnedSlice(gpa);
}

/// Prompt reuse: P2 = P ++ 16 reply tokens, fresh vs resumed from kept state vs forked (copyPrefix); all must agree.
pub fn reuse(gpa: Allocator, io: std.Io, e: *Engine, capture: []const u8, o: Options) !u8 {
    var pb: [512]u8 = undefined;
    const prompts = try readJson(gpa, io, try std.fmt.bufPrint(&pb, "{s}/prompts.json", .{capture}));
    defer prompts.deinit();
    const sampled: lanes.Sampling = .{ .seed = 1234, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.0 };
    var bad: usize = 0;
    var total: usize = 0;
    e.absorbing = true;
    var it = prompts.value.object.iterator();
    while (it.next()) |kv| {
        if (!listed(o.only, kv.key_ptr.*)) continue;
        const p = try idsOf(gpa, kv.value_ptr.*);
        defer gpa.free(p);
        for ([_]?lanes.Sampling{ null, sampled }) |smp| {
            // the reply P would get, so P2 is a realistic next turn
            const own = e.bound;
            const f0 = try e.prefill(p, smp);
            const reply = try decodeBound(gpa, e, f0, 17);
            defer gpa.free(reply);
            const p2 = try gpa.alloc(u32, p.len + 16);
            defer gpa.free(p2);
            @memcpy(p2[0..p.len], p);
            @memcpy(p2[p.len..], reply[1..17]);
            // fresh
            const fresh = try decodeBound(gpa, e, try e.prefill(p2, smp), 48);
            defer gpa.free(fresh);
            // resumed: keep at len - 1, a reply that overwrites the rows past it, then the next turn
            const kept = try e.prefillWith(p, smp, .{ .keep_at = p.len - 1 });
            const snap = kept.kept.?;
            defer e.freeSnapshot(snap);
            const other = try decodeBound(gpa, e, kept.first, 32);
            gpa.free(other);
            const resumed = try decodeBound(gpa, e, (try e.prefillWith(p2, smp, .{ .resume_from = snap })).first, 48);
            defer gpa.free(resumed);
            // forked into another sequence by its prefix
            const b = try e.newSeq(e.max_len);
            defer e.freeSeq(b);
            try e.copyPrefix(b, own, snap.pos, snap.mtp_len);
            e.bind(b);
            const forked = try decodeBound(gpa, e, (try e.prefillWith(p2, smp, .{ .resume_from = snap })).first, 48);
            defer gpa.free(forked);
            e.bind(own);
            const ok_r = std.mem.eql(u32, fresh, resumed);
            const ok_f = std.mem.eql(u32, fresh, forked);
            total += 2;
            bad += @as(usize, @intFromBool(!ok_r)) + @intFromBool(!ok_f);
            std.debug.print("{s} reuse {s} {s}: {d}-token next turn resumed at {d}: resumed {s}, forked {s} (fresh sha {s})\n", .{ if (ok_r and ok_f) "EQUAL" else "DIFFER", kv.key_ptr.*, if (smp == null) "greedy" else "sampled", p2.len, snap.pos, if (ok_r) "equal" else "DIFFERS", if (ok_f) "equal" else "DIFFERS", try sha12(gpa, fresh) });
        }
    }
    std.debug.print("{s} reuse: {d} of {d} resumed runs equal their fresh prefills\n", .{ if (bad == 0) "PASS" else "FAIL", total - bad, total });
    return if (bad == 0) 0 else 1;
}
