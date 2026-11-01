//! TF_DSV41_RUNNER_BACKEND=graph: pure-integer calc programs run as one packet of T1's bank plans.
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const packet = @import("runner_graph_packet.zig");
const Family = packet.Family;
const Allocator = std.mem.Allocator;

/// One node of the program, children first; `digits` bounds its value's decimal digits from syntax alone.
const Node = struct {
    kind: enum { literal, neg, binary, pow },
    family: Family = .add,
    l: u32 = 0,
    r: u32 = 0,
    text: []const u8 = "",
    k: u16 = 0,
    digits: u32,
    plan: []const u8 = "",
};

const Invalid = error{ Unsupported, OutOfMemory };

/// Calc firmware input over integer literals, + - * // % ** and parentheses, every value within 62 digits.
const Parser = struct {
    a: Allocator,
    src: []const u8,
    pos: usize = 0,
    nodes: std.ArrayList(Node) = .empty,
    depth: u32 = 0,

    fn peek(p: *Parser, token: []const u8) bool {
        return std.mem.startsWith(u8, p.src[p.pos..], token);
    }
    fn take(p: *Parser, token: []const u8) bool {
        if (!p.peek(token)) return false;
        p.pos += token.len;
        return true;
    }
    fn add(p: *Parser, n: Node) Invalid!u32 {
        if (n.digits > 62 or p.nodes.items.len >= packet.max_ops) return error.Unsupported;
        try p.nodes.append(p.a, n);
        return @intCast(p.nodes.items.len - 1);
    }
    fn at(p: *Parser, i: u32) Node {
        return p.nodes.items[i];
    }

    fn expr(p: *Parser) Invalid!u32 {
        p.depth += 1;
        defer p.depth -= 1;
        if (p.depth > 32) return error.Unsupported;
        var l = try p.term();
        while (true) {
            const f: Family = if (p.take("+")) .add else if (p.take("-")) .sub else return l;
            l = try p.binary(f, l, try p.term());
        }
    }

    fn term(p: *Parser) Invalid!u32 {
        var l = try p.unary();
        while (true) {
            if (p.peek("**")) return error.Unsupported;
            const f: Family = if (p.take("//")) .div else if (p.peek("/")) return error.Unsupported else if (p.take("*")) .mul else if (p.take("%")) .mod else return l;
            l = try p.binary(f, l, try p.unary());
        }
    }

    fn unary(p: *Parser) Invalid!u32 {
        if (p.take("+")) return p.unary();
        if (!p.take("-")) return p.power();
        const c = try p.unary();
        const n = p.at(c);
        if (n.kind == .literal and n.text[0] != '-')
            return p.add(.{ .kind = .literal, .text = try std.mem.concat(p.a, u8, &.{ "-", n.text }), .digits = n.digits });
        // -x is 0 - x
        return p.add(.{ .kind = .neg, .r = c, .digits = n.digits, .plan = try plan(p.a, .sub, 1, n.digits) });
    }

    fn power(p: *Parser) Invalid!u32 {
        const base = try p.atom();
        if (!p.take("**")) return base;
        // the exponent is control data: a plain literal, so it selects a prepared pow-<k>-62 schedule
        const start = p.pos;
        while (p.pos < p.src.len and std.ascii.isDigit(p.src[p.pos])) p.pos += 1;
        const k = std.fmt.parseInt(u16, p.src[start..p.pos], 10) catch return error.Unsupported;
        if (k > 999 or (p.pos - start > 1 and p.src[start] == '0') or p.peek("**") or p.peek(".")) return error.Unsupported;
        const b = p.at(base);
        const digits: u32 = if (k == 0) 1 else if (b.kind == .literal) literalPowDigits(b.text, k) else b.digits * k;
        return p.add(.{ .kind = .pow, .l = base, .k = k, .digits = digits, .plan = try std.fmt.allocPrint(p.a, "pow-{d}-62", .{k}) });
    }

    fn atom(p: *Parser) Invalid!u32 {
        if (p.take("(")) {
            const e = try p.expr();
            if (!p.take(")")) return error.Unsupported;
            return e;
        }
        const start = p.pos;
        while (p.pos < p.src.len and std.ascii.isDigit(p.src[p.pos])) p.pos += 1;
        const n = p.pos - start;
        if (n == 0 or (n > 1 and p.src[start] == '0') or p.peek(".")) return error.Unsupported;
        return p.add(.{ .kind = .literal, .text = p.src[start..p.pos], .digits = @intCast(n) });
    }

    fn binary(p: *Parser, f: Family, l: u32, r: u32) Invalid!u32 {
        const dl = p.at(l).digits;
        const dr = p.at(r).digits;
        if (f == .div or f == .mod) {
            // only a nonzero literal divisor: Python raises on zero, a computed divisor could be zero
            const d = p.at(r);
            if (d.kind != .literal or std.mem.eql(u8, std.mem.trimStart(u8, d.text, "-"), "0")) return error.Unsupported;
        }
        const digits = switch (f) {
            .add, .sub => @max(dl, dr) + 1,
            .mul => dl + dr,
            .div => dl,
            .mod => dr,
            .pow => unreachable,
        };
        return p.add(.{ .kind = .binary, .family = f, .l = l, .r = r, .digits = digits, .plan = try plan(p.a, f, dl, dr) });
    }
};

/// Builder.binary's bucket: add/sub cover the wider operand plus a carry, mul the digit sum, div/mod the wider operand.
fn plan(a: Allocator, f: Family, dl: u32, dr: u32) Invalid![]const u8 {
    const needed = switch (f) {
        .add, .sub => @max(dl, dr) + 1,
        .mul => dl + dr,
        .div, .mod => @max(dl, dr),
        .pow => unreachable,
    };
    if (needed > 62) return error.Unsupported;
    const width = for ([_]u8{ 1, 2, 4, 8, 16, 32, 62 }) |n| {
        if (n >= needed) break n;
    } else unreachable;
    return std.fmt.allocPrint(a, "{t}-{d}", .{ f, width });
}

/// Digits of |b|**k bounded from the literal's own digits: floor(k * log10|b|) + 1, rounded up for safety.
fn literalPowDigits(text: []const u8, k: u16) u32 {
    const digits = std.mem.trimStart(u8, text, "-");
    if (std.mem.eql(u8, digits, "0") or std.mem.eql(u8, digits, "1")) return 1;
    const lead = @min(digits.len, 15);
    const m = std.fmt.parseFloat(f64, digits[0..lead]) catch return 1 << 20;
    // past 15 digits the leading part rounds up, so the bound stays above the true value
    const log = std.math.log10(if (digits.len > lead) m + 1 else m) + @as(f64, @floatFromInt(digits.len - lead));
    const bound = @floor(@as(f64, @floatFromInt(k)) * log + 1e-9) + 1;
    return if (bound > 1 << 20) 1 << 20 else @intFromFloat(bound);
}

/// A prepared constant back to the decimal text that binds it (plan data such as 0 and 1, never an answer).
fn constantText(a: Allocator, v: *const packet.Value) ![]u8 {
    var buf: [63]u8 = undefined;
    var top: usize = 0;
    var digits: [62]u8 = undefined;
    for (&digits, 0..) |*d, i| {
        d.* = 0;
        for (0..4) |bit| if (v[i / 31][2 + i % 31 * 4 + bit] != 0) {
            d.* |= @as(u8, 1) << @intCast(bit);
        };
        if (d.* > 9) return error.GraphLiteral;
        if (d.* != 0) top = i;
    }
    var n: usize = 0;
    if (v[0][126] != 0) {
        buf[0] = '-';
        n = 1;
    }
    var i = top + 1;
    while (i > 0) : (i -= 1) {
        buf[n] = '0' + digits[i - 1];
        n += 1;
    }
    const text = try a.dupe(u8, buf[0..n]);
    const back = try packet.literal(text);
    if (!std.mem.eql(u8, std.mem.asBytes(&back), std.mem.asBytes(v))) return error.GraphLiteral;
    return text;
}

/// The plan names, children first, for calc input `src`; null when any part is not a graph route.
fn shape(arena: Allocator, src: []const u8) Allocator.Error!?[]const Node {
    if (src.len == 0 or src.len > 4096) return null;
    var p: Parser = .{ .a = arena, .src = src };
    _ = p.expr() catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
    if (p.pos != src.len) return null;
    return p.nodes.items;
}

/// Calc input `src` as one graph packet inlining each operation's named plan from `bank`, else null.
pub fn lower(a: Allocator, bank: *const graph.Bank, src: []const u8) Allocator.Error!?[]u8 {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const nodes = try shape(arena, src) orelse return null;
    // pass 1: every bound value first (literals, then each plan's constants), as the packet requires
    var values: std.ArrayList([]const u8) = .empty;
    const value_at = try arena.alloc(u32, nodes.len);
    const const_at = try arena.alloc(u32, nodes.len);
    const plans = try arena.alloc(?*const @import("runner_graph_bank.zig").PreparedPlan, nodes.len);
    var zero: ?u32 = null;
    for (nodes, 0..) |n, i| {
        plans[i] = null;
        if (n.kind == .literal) {
            value_at[i] = @intCast(values.items.len);
            try values.append(arena, n.text);
            continue;
        }
        const pp = bank.plan(n.plan) orelse return null;
        if (pp.inputs != @as(u32, if (n.kind == .pow) 1 else 2)) return null;
        plans[i] = pp;
        const_at[i] = @intCast(values.items.len);
        for (pp.constants) |*c| try values.append(arena, constantText(arena, c) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null);
        if (n.kind == .neg and zero == null) {
            zero = @intCast(values.items.len);
            try values.append(arena, "0");
        }
    }
    if (values.items.len > packet.max_values) return null;
    // pass 2: each plan's ops with its inputs, constants and outputs renumbered into the packet
    var ops: std.ArrayList(packet.Op) = .empty;
    var count: u32 = @intCast(values.items.len);
    for (nodes, 0..) |n, i| {
        const pp = plans[i] orelse continue;
        const ins: [2]u32 = switch (n.kind) {
            .literal => unreachable,
            .neg => .{ zero.?, value_at[n.r] },
            .binary => .{ value_at[n.l], value_at[n.r] },
            .pow => .{ value_at[n.l], 0 },
        };
        const m: Map = .{ .ins = ins[0..pp.inputs], .consts = const_at[i], .n_consts = @intCast(pp.constants.len), .base = count };
        for (pp.ops) |op| {
            if (ops.items.len >= packet.max_ops) return null;
            var o = op;
            o.a = m.ref(op.a);
            if (op.b) |b| o.b = m.ref(b);
            if (op.args) |args| {
                const re = try arena.alloc(u32, args.len);
                for (args, re) |x, *y| y.* = m.ref(x);
                o.args = re;
            }
            if (op.copies.len != 0) {
                const re = try arena.dupe(packet.Copy, op.copies);
                for (re) |*c| c.ref = m.ref(c.ref);
                o.copies = re;
            }
            if (op.flags) |f| o.flags = m.ref(f);
            try ops.append(arena, o);
            count += op.outputs;
        }
        value_at[i] = m.ref(pp.result);
    }
    const result = value_at[nodes.len - 1];
    _ = bank.validate(values.items.len, ops.items, result) catch return null;
    return graph.Builder.encode(a, .{ .values = values.items, .ops = ops.items, .result = result }) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => null,
    };
}

/// A plan's own references (inputs, then constants, then its outputs) to packet references.
const Map = struct {
    ins: []const u32,
    consts: u32,
    n_consts: u32,
    base: u32,

    fn ref(m: Map, r: u32) u32 {
        const n_in: u32 = @intCast(m.ins.len);
        if (r < n_in) return m.ins[r];
        if (r < n_in + m.n_consts) return m.consts + (r - n_in);
        return m.base + (r - n_in - m.n_consts);
    }
};

/// A frontend that runs `inner`, then moves a pure-integer calc program onto the graph bank; calc and ELF otherwise.
pub const Routed = struct {
    inner: api.Frontend,
    bank: *const graph.Bank,

    pub fn frontend(r: *Routed) api.Frontend {
        return .{ .ctx = r, .translate = translate };
    }

    fn translate(ctx: ?*anyopaque, a: Allocator, cell: api.Cell) Allocator.Error!api.Translation {
        const r: *Routed = @ptrCast(@alignCast(ctx.?));
        const t = try r.inner.translate(r.inner.ctx, a, cell);
        if (t != .program or t.program.kind != .calc) return t;
        const bytes = try lower(a, r.bank, t.program.bytes) orelse return t;
        return .{ .program = .{ .kind = .graph, .bytes = bytes } };
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

fn planNames(a: Allocator, src: []const u8) !?[]u8 {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const nodes = try shape(arena_state.allocator(), src) orelse return null;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    for (nodes) |n| if (n.kind != .literal) {
        if (out.items.len != 0) try out.append(a, ' ');
        try out.appendSlice(a, n.plan);
    };
    return try out.toOwnedSlice(a);
}

test "runner graph route: pure-integer calc programs become bank plans sized from their syntax" {
    const a = std.testing.allocator;
    for ([_][2][]const u8{
        .{ "2**100-3**50", "pow-100-62 pow-50-62 sub-32" },
        .{ "47*83", "mul-4" },
        .{ "-7//2", "div-1" },
        .{ "1071%462", "mod-4" },
        .{ "-(2+3)", "add-2 sub-4" },
        .{ "(-3)**5", "pow-5-62" },
        .{ "10**61", "pow-61-62" },
        .{ "(2+3)*4-1", "add-2 mul-4 sub-4" },
        .{ "42", "" },
    }) |c| {
        const got = (try planNames(a, c[0])) orelse {
            std.debug.print("not routed: {s}\n", .{c[0]});
            return error.TestExpectedRoute;
        };
        defer a.free(got);
        try std.testing.expectEqualStrings(c[1], got);
    }
}

test "runner graph route: decimals, true division, computed exponents or divisors and wide values stay on calc" {
    const a = std.testing.allocator;
    for ([_][]const u8{
        "1/2",
        "1.5*2",
        "2**-1",
        "2**3**2",
        "2**(3)",
        "7//0",
        "7%(1+1)",
        "10**62",
        "1000000000000000000000000000000000000000*100000000000000000000000000000",
        "007+1",
        "x*2",
        "",
        "(1+2",
        "2**1000",
        "3^2",
    }) |src| {
        if (try planNames(a, src)) |got| {
            defer a.free(got);
            std.debug.print("routed: {s}\n", .{src});
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
