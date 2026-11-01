//! Form trace: real census reasoning lines own `[exact: …]`, ambiguous ones abstain; scanner and lane-host replies.
const std = @import("std");
const trace = @import("runner_trace.zig");
const scan = @import("runner_scan.zig");
const rapi = @import("runner_api.zig");
const rt = @import("runner_test.zig");

const gpa = std.testing.allocator;

/// The Python source a line states (after `lines` defined its names), or null.
fn stated(lines: []const []const u8) !?[]u8 {
    var t = trace.Tracer.init(gpa);
    defer t.deinit();
    var last: ?[]u8 = null;
    for (lines) |l| {
        if (last) |x| gpa.free(x);
        last = null;
        if (try t.line(l)) |f| last = try gpa.dupe(u8, f.source);
    }
    return last;
}

fn expectStated(lines: []const []const u8, want: []const u8) !void {
    const got = try stated(lines) orelse {
        std.debug.print("abstained: {s}\n", .{lines[lines.len - 1]});
        return error.TestExpectedStatement;
    };
    defer gpa.free(got);
    try std.testing.expectEqualStrings(want, got);
}

fn expectNone(lines: []const []const u8) !void {
    if (try stated(lines)) |got| {
        defer gpa.free(got);
        std.debug.print("stated {s} from: {s}\n", .{ got, lines[lines.len - 1] });
        return error.TestUnexpectedStatement;
    }
}

// Verbatim T6 census reasoning lines (t6-census/raw).
const bn019 = "We need answer division. Need compute quotient remainder for huge integer. Need be accurate. We can use big integer arithmetic manually. Need divide A=314159265358979323846264338327950288419716939937510 by B=271828182845904523536. Need quotient and remainder.";
const bn001 = "We need answer exact multiplication. Need compute. Need be careful. Need multiply A=12345678901234567890 (20 digits?) B=9876543210987654321 (19 digits). Need exact product.";
const bn008 = "We need answer compute 7^123 mod 1000000007. Need calculate accurately. Need provide maybe concise. Let's compute carefully. Mod p=1,000,000,007 prime. Need exponentiation.";
const bn015 = "Let A = 0xFEDCBA9876543210, B=0x0123456789ABCDEF. N = A*2^64 + B.";
const bn017 = "We need answer exact product 41*42*...*60. Need compute. Need be careful. Could express as 60!/40! maybe exact integer. User asks exact product. Need provide number. Let's compute accurately.";
const bn020 = "We need answer. Need compute exact. Need be careful. Expression (10^80 + 123456789)^2 - 10^160. Expand: 2*123456789*10^80 + 123456789^2. Need exact decimal. Need compute 123456789^2. Let's calculate. 123,456,789^2. Known? 123456789^2 = 15241578750190521? Let's verify. 123456789*123456789.";
const rc009 = "We need answer. Need compute exact value 2^100 - 3^50. Need run code? As AI no execution environment? We can compute manually with big ints, verify. Need exact. Let's calculate carefully. Need maybe use Python mentally? Let's derive.";
const ar008 = "We need answer compute. Need be careful. 2^37 - 3^19. Let's calculate.";
const bn002 = "Need divide 10^30 -1 by 97. Let's calculate exact quotient. N = 999,999,999,999,999,999,999,999,999,999 (30 nines). Divide by 97. Need quotient. We can do long division or use formula. Need exact.";
const bn006 = "We need answer arithmetic. Need be careful. Need subtract A=876543210987654321098765432109 from B=1000000000000000000000000000000 (1 followed? Let's count). Need compute B - A.";
const bn010 = [_][]const u8{
    "Let's denote A=12,345,678,901,234,567,890 (20 digits? Let's count). B=98,765,432,109,876,543,210 (20 digits? Actually 98 quintillion = 20 digits). Need gcd.",
    "So gcd(A,B)=gcd(A, 900,000,000,090). Let C=900,000,000,090.",
    "Need gcd(A,C). A=12,345,678,901,234,567,890. C ~9e11. Need compute A mod C. Could use factors.",
};

test "runner trace: census statements extract exactly what the line states" {
    try expectStated(&.{bn019}, "divmod(314159265358979323846264338327950288419716939937510, 271828182845904523536)");
    try expectStated(&.{bn001}, "(12345678901234567890) * (9876543210987654321)");
    try expectStated(&.{bn008}, "7**123%1000000007");
    try expectStated(&.{bn015}, "(0xFEDCBA9876543210)*2**64+(0x0123456789ABCDEF)");
    try expectStated(&.{bn017}, "41*42*43*44*45*46*47*48*49*50*51*52*53*54*55*56*57*58*59*60");
    try expectStated(&.{rc009}, "2**100-3**50");
    try expectStated(&.{ar008}, "2**37-3**19");
    try expectStated(&.{bn002}, "divmod(10**30-1, 97)");
    try expectStated(&bn010, "(12345678901234567890)%(900000000090)");
}

test "runner trace: a statement too wide for the frontend yields to the line's next one (big-number-020)" {
    try expectStated(&.{bn020}, "123456789**2");
}

test "runner trace: the display line and the owned text" {
    var t = trace.Tracer.init(gpa);
    defer t.deinit();
    const f = (try t.line(bn019)).?;
    try std.testing.expect(f.divmod);
    try std.testing.expectEqualStrings("314159265358979323846264338327950288419716939937510 ÷ 271828182845904523536", f.display);
    const line = (try trace.owned(gpa, f.display, true, "(1155727349790921717910215407028, 23663814570294126502)\n")).?;
    defer gpa.free(line);
    try std.testing.expectEqualStrings("[exact: 314159265358979323846264338327950288419716939937510 ÷ 271828182845904523536 = 1155727349790921717910215407028 remainder 23663814570294126502]\n", line);
    const g = (try t.line(bn008)).?;
    try std.testing.expectEqualStrings("7^123 mod 1000000007", g.display);
    var fresh = trace.Tracer.init(gpa); // bn019 above defined A and B; bn015's other A poisons it
    defer fresh.deinit();
    const h = (try fresh.line(bn015)).?;
    try std.testing.expectEqualStrings("0xFEDCBA9876543210*2^64 + 0x0123456789ABCDEF", h.display);
    for ([_][]const u8{ "3.75\n", "1\n2\n", "", "(1, 2)\n" }) |out| try std.testing.expectEqual(null, try trace.owned(gpa, "x", false, out));
    for ([_][]const u8{ "(1, 2, 3)\n", "7\n", "(1,2)\n" }) |out| try std.testing.expectEqual(null, try trace.owned(gpa, "x", true, out));
}

test "runner trace: doubts, lists, hypotheticals and two readings abstain" {
    try expectNone(&.{bn006}); // "(1 followed? Let's count)" sits in B's sentence
    try expectNone(&.{ "Let D = 271828182845904523536.", "Let's compute D*155, D*156 etc." });
    try expectNone(&.{ "Let M = 1234567890.", "If P = M*(10^10+1) then fine.", "Need compute P*2." });
    try expectNone(&.{ "Let A = 5. Let M = 7.", "Need compute A + 3 mod M." });
    try expectNone(&.{"Need compute 2^3^2."});
    try expectNone(&.{ "Let M = 7.", "Need compute (5 mod M)*2." });
    try expectNone(&.{"Need compute A ~2.718e20 * 3."});
    try expectNone(&.{"Need compute 3^400 mod 10^18? Not sure."});
    try expectNone(&.{"Need compute 2^64*3, right?"}); // the doubt sits after the phrase, in its sentence
    try expectNone(&.{ "Let A = 12345678901 (≈1.2e10).", "Need compute A*1000003." }); // an approximate definition binds nothing
    try expectNone(&.{"Need compute 1.5*2."});
    try expectNone(&.{"Need compute 012*3."});
    try expectNone(&.{"Need compute [314,159,265]*2."});
    try expectNone(&.{"Need compute positions 12-20."});
    try expectNone(&.{"Need compute 12,34 + 5."});
    try expectNone(&.{"Need compute 2^N."});
    try expectNone(&.{"Need compute 41*43*...*60."});
}

test "runner trace: a name after an operator is no definition; a redefinition poisons the name" {
    try expectStated(&.{ "Let B = 5.", "A * B = 37,276,043,023,296,000 * 273,589,847,231,500,800.", "Need compute B*1000003." }, "(5)*1000003");
    try expectNone(&.{ "Let A = 5.", "Let A = 6.", "Need compute A*1000003." });
    try expectStated(&.{ "Let A = 12,345.", "A = 12345.", "Need compute A*1000003." }, "(12345)*1000003");
}

test "runner trace: negative controls: prose, code and small inline sums state nothing" {
    try expectNone(&.{"We need answer. Need be careful. Let's think about the algorithm."});
    try expectNone(&.{"for i in range(10): total = total * 3"});
    try expectStated(&.{"Need compute 2+3."}, "2+3"); // a small sum after a goal head is still a statement
}

test "runner trace: hand-trace cues" {
    for ([_][]const u8{ "Step i=0: rem=314.", "We can perform long division or use modular arithmetic.", "314,159 * 271,828.", "Let's do base 1000.", "Then carry 3 into the next group", "Row 0: [0,1,2]", "So 41*42=1722." }) |l|
        try std.testing.expect(trace.cue(l));
    for ([_][]const u8{ "So that is all.", "Need quotient and remainder.", "We need compute. Let's denote B." }) |l|
        try std.testing.expect(!trace.cue(l));
}

fn feedAll(s: *scan.Scanner, tokens: []const []const u8, out: *std.ArrayList(scan.Events)) !void {
    for (tokens) |t| try out.append(gpa, try s.feed(gpa, t, true));
}

test "runner trace: the scanner closes a trace cell at its line and triggers at the cue line's end" {
    var s: scan.Scanner = .{ .cfg = .{ .trace = true } };
    defer s.deinit(gpa);
    var ev: std.ArrayList(scan.Events) = .empty;
    defer ev.deinit(gpa);
    try feedAll(&s, &.{ bn019, "\n", "\n", "We need compute.", " Nothing yet.\n", "We can perform long division or use modular arithmetic.", "\n", "Step i=0: rem=314.\n" }, &ev);
    const c = ev.items[1].closed.?;
    try std.testing.expectEqual(scan.Stated{ .display = c.trace.?.display, .divmod = true }, c.trace.?);
    try std.testing.expectEqualStrings("divmod(314159265358979323846264338327950288419716939937510, 271828182845904523536)", c.code);
    try std.testing.expect(c.form == .repl);
    for (ev.items[0..6]) |e| try std.testing.expectEqual(@as(?scan.Trigger, null), e.trigger);
    try std.testing.expectEqual(scan.Trigger{ .index = 0, .trace = true }, ev.items[6].trigger.?);
    try std.testing.expectEqual(@as(?scan.Trigger, null), ev.items[7].trigger); // owned once
}

test "runner trace: off by default, out of window, spoiled tokens wait for the next cue" {
    var off: scan.Scanner = .{ .cfg = .{} };
    defer off.deinit(gpa);
    var ev: std.ArrayList(scan.Events) = .empty;
    defer ev.deinit(gpa);
    try feedAll(&off, &.{ bn019 ++ "\n", "We can perform long division.\n" }, &ev);
    for (ev.items) |e| try std.testing.expect(e.closed == null and e.trigger == null);
    ev.clearRetainingCapacity();
    var far: scan.Scanner = .{ .cfg = .{ .trace = true, .trace_window = 2 } };
    defer far.deinit(gpa);
    try feedAll(&far, &.{ bn019 ++ "\n", "a", "b", "c", "We can perform long division.\n" }, &ev);
    try std.testing.expect(ev.items[0].closed != null and ev.items[4].trigger == null);
    ev.clearRetainingCapacity();
    var spoil: scan.Scanner = .{ .cfg = .{ .trace = true } };
    defer spoil.deinit(gpa);
    try feedAll(&spoil, &.{ bn019 ++ "\n", "We can perform long division.\nStep", " i=0: rem=314.\n" }, &ev);
    try std.testing.expect(ev.items[1].trigger == null);
    try std.testing.expectEqual(scan.Trigger{ .index = 0, .trace = true }, ev.items[2].trigger.?);
}

test "runner trace: a fence's code lines state nothing, and a trace cell is no notebook prior" {
    var s: scan.Scanner = .{ .cfg = .{ .trace = true } };
    defer s.deinit(gpa);
    var ev: std.ArrayList(scan.Events) = .empty;
    defer ev.deinit(gpa);
    try feedAll(&s, &.{ "```python\n", "Need compute 2^100 - 3^50.\n", "Step 1\n", "```\n" }, &ev);
    for (ev.items[0..3]) |e| try std.testing.expect(e.closed == null and e.trigger == null);
}

/// The circuit's printed values for the trace programs (the fake frontend passes the Python source through).
const TraceBack = struct {
    delay: u32 = 0,
    fault: bool = false,
    jobs: std.ArrayList(struct { prog: []u8, polls: u32 }) = .empty,

    const answers = [_][2][]const u8{
        .{ "(987654) * (137)", "135308598\n" },
        .{ "7**123%1000000007", "937329259\n" },
    };

    fn backend(b: *TraceBack) rapi.Backend {
        return .{ .ctx = b, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
    }
    fn start(ctx: *anyopaque, _: u64, p: rapi.Program, _: u64) error{ OutOfMemory, Busy }!rapi.JobId {
        const b: *TraceBack = @ptrCast(@alignCast(ctx));
        try b.jobs.append(gpa, .{ .prog = try gpa.dupe(u8, p.bytes), .polls = 0 });
        return b.jobs.items.len - 1;
    }
    fn poll(ctx: *anyopaque, job: rapi.JobId) rapi.Poll {
        const b: *TraceBack = @ptrCast(@alignCast(ctx));
        const j = &b.jobs.items[job];
        j.polls += 1;
        if (j.polls <= b.delay) return .running;
        if (b.fault) return .{ .done = .{ .status = .fault, .out = "" } };
        for (answers) |x| if (std.mem.eql(u8, x[0], j.prog)) return .{ .done = .{ .status = .ok, .out = x[1], .steps = 100 } };
        return .{ .done = .{ .status = .fault, .out = "" } };
    }
    fn cancel(_: *anyopaque, _: rapi.JobId) void {}
    fn release(_: *anyopaque, _: u64) void {}
    fn deinit(b: *TraceBack) void {
        for (b.jobs.items) |j| gpa.free(j.prog);
        b.jobs.deinit(gpa);
    }
};

/// One reply; `guess` is the model's own line after `lead` (a digit line is itself a cue).
fn traced(lead: []const u8, guess: []const u8, on: bool, delay: u32, fault: bool) ![]u8 {
    var back: TraceBack = .{ .delay = delay, .fault = fault };
    defer back.deinit();
    var unused: rt.Back = .{};
    defer unused.deinit();
    var m = [_]rt.Model{.{ .lead = lead, .guess = guess }};
    const texts = try rt.replies(&m, .{ .trace = on, .backend = back.backend() }, &unused);
    defer gpa.free(texts);
    return texts[0];
}

const mul_lead = "Need multiply A=987654 B=137. Need exact product.\nLet's do long multiplication.\n";

test "runner trace: the reply owns one [exact: …] line after the cue, then the model goes on" {
    const got = try traced(mul_lead, "ok", true, 0, false);
    defer gpa.free(got);
    try std.testing.expectEqualStrings(mul_lead ++ "[exact: 987654 * 137 = 135308598]\nDone.", got);
    const pow = "We need answer compute 7^123 mod 1000000007. Let's compute carefully.\nWe can do it by hand.\n";
    const same = try traced(pow, "ok", true, 0, false);
    defer gpa.free(same);
    try std.testing.expectEqualStrings(pow ++ "[exact: 7^123 mod 1000000007 = 937329259]\nDone.", same);
}

test "runner trace: late runs give the same reply; faults, the form off and no cue keep the model's text" {
    const early = try traced(mul_lead, "ok", true, 0, false);
    defer gpa.free(early);
    const late = try traced(mul_lead, "ok", true, 6, false);
    defer gpa.free(late);
    try std.testing.expectEqualStrings(early, late);
    const fault = try traced(mul_lead, "ok", true, 0, true);
    defer gpa.free(fault);
    try std.testing.expectEqualStrings(mul_lead ++ "ok\nDone.", fault);
    const off = try traced(mul_lead, "ok", false, 0, false);
    defer gpa.free(off);
    try std.testing.expectEqualStrings(mul_lead ++ "ok\nDone.", off);
    const quiet = "Need multiply A=987654 B=137. Need exact product.\nThat is the task.\n";
    const nocue = try traced(quiet, "ok", true, 0, false);
    defer gpa.free(nocue);
    try std.testing.expectEqualStrings(quiet ++ "ok\nDone.", nocue);
    const digits = try traced(quiet, "135308598", true, 0, false); // the model's own digit line is a hand-work cue
    defer gpa.free(digits);
    try std.testing.expectEqualStrings(quiet ++ "135308598\n[exact: 987654 * 137 = 135308598]\n", digits);
}

fn env(name: [:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

fn readAll(path: []const u8) ![]u8 {
    var threaded: std.Io.Threaded = .init_single_threaded;
    return std.Io.Dir.cwd().readFileAlloc(threaded.io(), path, gpa, .limited(1 << 30));
}

test "runner trace: census replay equals the offline study (TF_TRACE_CENSUS, TF_TRACE_EXPECT)" {
    const path = env("TF_TRACE_CENSUS") orelse {
        std.debug.print("skip trace census replay: TF_TRACE_CENSUS unset\n", .{});
        return;
    };
    const data = try readAll(path);
    defer gpa.free(data);
    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(gpa);
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |row| {
        if (row.len == 0) continue;
        var parsed = try std.json.parseFromSlice(struct { id: []const u8, text: []const u8 }, gpa, row, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        var s: scan.Scanner = .{ .cfg = .{ .trace = true, .fence = false, .repl = false } };
        defer s.deinit(gpa);
        var codes: std.ArrayList([]u8) = .empty;
        defer {
            for (codes.items) |c| gpa.free(c);
            codes.deinit(gpa);
        }
        var parts = std.mem.splitScalar(u8, parsed.value.text, '\n');
        while (parts.next()) |part| {
            const ev = try s.feed(gpa, part, true);
            if (ev.closed) |c| try codes.append(gpa, try gpa.dupe(u8, c.code));
            const nl = try s.feed(gpa, "\n", true);
            if (nl.closed) |c| try codes.append(gpa, try gpa.dupe(u8, c.code));
            if (nl.trigger) |t| if (t.trace) try got.print(gpa, "{s}\t{s}\n", .{ parsed.value.id, codes.items[t.index] });
        }
    }
    std.debug.print("trace census replay: {d} owned lines\n{s}", .{ std.mem.count(u8, got.items, "\n"), got.items });
    const expect = env("TF_TRACE_EXPECT") orelse return;
    const want = try readAll(expect);
    defer gpa.free(want);
    try std.testing.expectEqualStrings(want, got.items);
}
