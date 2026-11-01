//! TF_DSV41_RUNNER_BACKEND=graph: one integer operation runs on T1's graph bank, the rest on the frontend and circuit.
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const Family = @import("runner_graph_packet.zig").Family;
const Allocator = std.mem.Allocator;

/// `A op B` over integer literals with a result within 62 digits: its packet (by source shape only), else null.
pub fn packet(a: Allocator, cell: api.Cell) Allocator.Error!?[]u8 {
    if (cell.form == .fence or !std.mem.eql(u8, cell.lang, "python")) return null;
    const src = std.mem.trim(u8, cell.code, " \t\n");
    var i: usize = 0;
    const lhs = number(src, &i) orelse return null;
    skip(src, &i);
    const op = operator(src, &i, cell.form) orelse return null;
    skip(src, &i);
    const rhs = number(src, &i) orelse return null;
    if (i != src.len) return null;
    const dl = lhs.len - @intFromBool(lhs[0] == '-');
    const dr = rhs.len - @intFromBool(rhs[0] == '-');
    const fits = switch (op) {
        .add, .sub => @max(dl, dr) + 1 <= 62,
        .mul => dl + dr <= 62,
        .div, .mod => !std.mem.eql(u8, rhs, "0") and !std.mem.eql(u8, rhs, "-0"),
        .pow => blk: {
            const k = std.fmt.parseInt(u16, rhs, 10) catch break :blk false;
            break :blk k <= 999 and dl * @as(usize, k) <= 62;
        },
    };
    if (!fits) return null;
    return graph.Builder.binary(a, op, lhs, rhs) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => null,
    };
}

fn skip(src: []const u8, i: *usize) void {
    while (i.* < src.len and (src[i.*] == ' ' or src[i.*] == '\t')) i.* += 1;
}

/// An optionally negative decimal literal without leading zeros.
fn number(src: []const u8, i: *usize) ?[]const u8 {
    const start = i.*;
    if (i.* < src.len and src[i.*] == '-') i.* += 1;
    const digits = i.*;
    while (i.* < src.len and std.ascii.isDigit(src[i.*])) i.* += 1;
    const n = i.* - digits;
    if (n == 0 or n > 62 or (n > 1 and src[digits] == '0')) return null;
    return src[start..i.*];
}

fn operator(src: []const u8, i: *usize, form: api.Form) ?Family {
    const ops = [_]struct { []const u8, Family, bool }{
        .{ "**", .pow, true },
        .{ "//", .div, true },
        .{ "×", .mul, true },
        .{ "−", .sub, true },
        .{ "^", .pow, true },
        .{ "+", .add, true },
        .{ "-", .sub, true },
        .{ "*", .mul, true },
        .{ "%", .mod, false },
    };
    // inline `/` and `%` are exact division and percent; in Python `^` is XOR
    for (ops) |o| if (std.mem.startsWith(u8, src[i.*..], o[0])) {
        if (form == .inline_expr and !o[2]) return null;
        if (form == .repl and std.mem.eql(u8, o[0], "^")) return null;
        if (form == .repl and (std.mem.eql(u8, o[0], "×") or std.mem.eql(u8, o[0], "−"))) return null;
        i.* += o[0].len;
        return o[1];
    };
    return null;
}

/// A frontend that tries the graph route first, then `inner`.
pub const Routed = struct {
    inner: api.Frontend,

    pub fn frontend(r: *Routed) api.Frontend {
        return .{ .ctx = r, .translate = translate };
    }

    fn translate(ctx: ?*anyopaque, a: Allocator, cell: api.Cell) Allocator.Error!api.Translation {
        const r: *Routed = @ptrCast(@alignCast(ctx.?));
        if (try packet(a, cell)) |bytes| return .{ .program = .{ .kind = .graph, .bytes = bytes } };
        return r.inner.translate(r.inner.ctx, a, cell);
    }
};

/// Graph programs to the graph backend, calc and ELF to the circuit; graph job ids carry the top bit.
pub const Dispatch = struct {
    circuit: api.Backend,
    graph: api.Backend,

    const tag: api.JobId = 1 << 63;

    pub fn backend(d: *Dispatch) api.Backend {
        return .{ .ctx = d, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
    }
    fn start(ctx: *anyopaque, stream: u64, p: api.Program, budget: u64) error{ OutOfMemory, Busy }!api.JobId {
        const d: *Dispatch = @ptrCast(@alignCast(ctx));
        if (p.kind == .graph) return tag | try d.graph.start(stream, p, budget);
        return d.circuit.start(stream, p, budget);
    }
    fn poll(ctx: *anyopaque, job: api.JobId) api.Poll {
        const d: *Dispatch = @ptrCast(@alignCast(ctx));
        return if (job & tag != 0) d.graph.poll(job & ~tag) else d.circuit.poll(job);
    }
    fn cancel(ctx: *anyopaque, job: api.JobId) void {
        const d: *Dispatch = @ptrCast(@alignCast(ctx));
        if (job & tag != 0) d.graph.cancel(job & ~tag) else d.circuit.cancel(job);
    }
    fn release(ctx: *anyopaque, stream: u64) void {
        const d: *Dispatch = @ptrCast(@alignCast(ctx));
        d.graph.release(stream);
        d.circuit.release(stream);
    }
};

test "runner graph route: one integer operation within 62 digits becomes a named plan" {
    const a = std.testing.allocator;
    for ([_]struct { api.Form, []const u8, []const u8 }{
        .{ .inline_expr, "47 * 83", "{\"version\":1,\"values\":[\"47\",\"83\"],\"plan\":\"mul-4\",\"ops\":[],\"result\":0}" },
        .{ .inline_expr, "2^37", "{\"version\":1,\"values\":[\"2\"],\"plan\":\"pow-37-62\",\"ops\":[],\"result\":0}" },
        .{ .inline_expr, "9876 − 4321", "{\"version\":1,\"values\":[\"9876\",\"4321\"],\"plan\":\"sub-8\",\"ops\":[],\"result\":0}" },
        .{ .repl, "314159 * 271828", "{\"version\":1,\"values\":[\"314159\",\"271828\"],\"plan\":\"mul-16\",\"ops\":[],\"result\":0}" },
        .{ .repl, "-7 // 2", "{\"version\":1,\"values\":[\"-7\",\"2\"],\"plan\":\"div-1\",\"ops\":[],\"result\":0}" },
        .{ .repl, "1071 % 462", "{\"version\":1,\"values\":[\"1071\",\"462\"],\"plan\":\"mod-4\",\"ops\":[],\"result\":0}" },
    }) |c| {
        const got = (try packet(a, .{ .form = c[0], .lang = "python", .code = c[1] })) orelse return error.TestExpectedRoute;
        defer a.free(got);
        try std.testing.expectEqualStrings(c[2], got);
    }
}

test "runner graph route: everything else stays with the frontend and the circuit" {
    const a = std.testing.allocator;
    for ([_]struct { api.Form, []const u8 }{
        .{ .inline_expr, "1/2" },
        .{ .inline_expr, "50 % 3" },
        .{ .inline_expr, "1.5 * 2" },
        .{ .inline_expr, "2 * 3 * 4" },
        .{ .inline_expr, "(2 + 3)" },
        .{ .inline_expr, "007 + 1" },
        .{ .inline_expr, "2**100" },
        .{ .inline_expr, "2 ** -1" },
        .{ .repl, "2 ^ 10" },
        .{ .repl, "7 // 0" },
        .{ .repl, "1/2" },
        .{ .repl, "x * 2" },
        .{ .repl, "12345678901234567890123456789012 * 123456789012345678901234567890123" },
        .{ .fence, "print(2 * 3)" },
    }) |c| {
        if (try packet(a, .{ .form = c[0], .lang = "python", .code = c[1] })) |got| {
            defer a.free(got);
            std.debug.print("routed: {s}\n", .{c[1]});
            return error.TestUnexpectedRoute;
        }
    }
}

test "runner graph route: graph programs reach the graph backend, the rest the circuit, by job id" {
    const Fake = struct {
        name: u8,
        started: u32 = 0,
        released: u32 = 0,
        fn backend(f: *@This()) api.Backend {
            return .{ .ctx = f, .vtable = &.{ .start = st, .poll = pl, .cancel = cn, .release = rl } };
        }
        fn st(ctx: *anyopaque, _: u64, _: api.Program, _: u64) error{ OutOfMemory, Busy }!api.JobId {
            const f: *@This() = @ptrCast(@alignCast(ctx));
            f.started += 1;
            return 7;
        }
        fn pl(ctx: *anyopaque, job: api.JobId) api.Poll {
            const f: *@This() = @ptrCast(@alignCast(ctx));
            return .{ .done = .{ .status = .ok, .out = if (f.name == 'g') "graph" else "circuit", .steps = job } };
        }
        fn cn(_: *anyopaque, _: api.JobId) void {}
        fn rl(ctx: *anyopaque, _: u64) void {
            const f: *@This() = @ptrCast(@alignCast(ctx));
            f.released += 1;
        }
    };
    var c: Fake = .{ .name = 'c' };
    var g: Fake = .{ .name = 'g' };
    var d: Dispatch = .{ .circuit = c.backend(), .graph = g.backend() };
    const b = d.backend();
    const gj = try b.start(1, .{ .kind = .graph, .bytes = "{}" }, 10);
    const cj = try b.start(1, .{ .kind = .calc, .bytes = "2*3" }, 10);
    try std.testing.expectEqualStrings("graph", b.poll(gj).done.out);
    try std.testing.expectEqual(@as(u64, 7), b.poll(gj).done.steps);
    try std.testing.expectEqualStrings("circuit", b.poll(cj).done.out);
    b.release(1);
    try std.testing.expect(c.started == 1 and g.started == 1 and c.released == 1 and g.released == 1);
}

test "runner graph route: routed cells give Python's values on the boot-loaded bank (TF_RUNNER_GRAPH_BANK)" {
    const a = std.testing.allocator;
    const path = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest;
    var bank = try graph.Bank.load(a, std.mem.span(path));
    defer bank.deinit();
    const b = try graph.backend(a, &bank, 1);
    defer graph.destroy(b);
    for ([_]struct { api.Form, []const u8, []const u8 }{
        .{ .inline_expr, "47 * 83", "3901" }, .{ .inline_expr, "2^37", "137438953472" },
        .{ .inline_expr, "9876 − 4321", "5555" },
        .{ .repl, "-7 // 2", "-4" },          .{ .repl, "1071 % 462", "147" },
        .{ .repl, "-7 % 3", "2" },            .{ .repl, "314159 * 271828", "85397212652" },
        .{ .inline_expr, "3 - 10", "-7" },    .{ .inline_expr, "99999999999999999999999999999999 * 99999999999999999999999999999", "9999999999999999999999999999899900000000000000000000000000001" },
    }) |c| {
        const bytes = (try packet(a, .{ .form = c[0], .lang = "python", .code = c[1] })) orelse {
            std.debug.print("not routed: {s}\n", .{c[1]});
            continue;
        };
        defer a.free(bytes);
        const job = try b.start(1, .{ .kind = .graph, .bytes = bytes }, 4_000_000);
        const r = while (true) switch (b.poll(job)) {
            .running => std.Thread.yield() catch {},
            .done => |r| break r,
        };
        try std.testing.expectEqual(api.Status.ok, r.status);
        try std.testing.expectEqualStrings(c[2], std.mem.trimEnd(u8, r.out, "\n"));
    }
}
