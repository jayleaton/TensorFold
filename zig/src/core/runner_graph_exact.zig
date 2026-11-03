//! Exact inline arithmetic lowered to existing prepared kernels; operand answers never run on the host.
const std = @import("std");
const packet = @import("runner_graph_packet.zig");
const Bank = @import("runner_graph_bank.zig").Bank;
const Allocator = std.mem.Allocator;
const Failure = error{ OutOfMemory, Unsupported };
const Kind = enum { literal, add, sub, mul, div, neg, pow };
const Node = struct {
    kind: Kind,
    a: u32 = 0,
    b: u32 = 0,
    text: []const u8 = "",
    scale: u8 = 0,
    decimal: bool = false,
    exponent: u16 = 0,
};
const Bound = struct { ref: u32, digits: usize };
const Number = struct {
    n: Bound,
    d: Bound = .{ .ref = 1, .digits = 1 },
    scale: ?u8 = 0,
    decimal: bool = false,
};

/// INLINE only: the caller owns form/language admission and invokes this after the integer binder refuses.
pub fn expression(a: Allocator, bank: *const Bank, source: []const u8) Allocator.Error!?[]u8 {
    if (source.len == 0 or source.len > 4096 or std.mem.indexOfAny(u8, source, "\x00\r\n") != null)
        return null;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    return build(a, arena, bank, source) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => null,
    };
}

fn build(a: Allocator, arena: Allocator, bank: *const Bank, source: []const u8) ![]u8 {
    const normalized = try normalize(arena, source);
    var p: Parser = .{ .a = arena, .source = normalized };
    const root = try p.sum();
    p.space();
    if (p.at != normalized.len) return error.Unsupported;
    var l: Lower = .{ .a = arena, .bank = bank, .nodes = p.nodes.items };
    try l.initialize();
    const value = try l.emit(root);
    const print = try l.finish(value);
    const count = try bank.validate(l.values.items.len, l.ops.items, print.ref);
    try bank.validatePrints(&.{print}, count);
    return packet.Builder.program(a, l.values.items, l.ops.items, &.{print});
}

fn normalize(a: Allocator, source: []const u8) Failure![]const u8 {
    var result: std.ArrayList(u8) = .empty;
    var at: usize = 0;
    while (at < source.len) {
        var found = false;
        for ([_][]const u8{ "×", "÷", "−" }, [_]u8{ '*', '/', '-' }) |symbol, ascii| {
            if (!std.mem.startsWith(u8, source[at..], symbol)) continue;
            try result.append(a, ascii);
            at += symbol.len;
            found = true;
            break;
        }
        if (found) continue;
        if (source[at] >= 128) return error.Unsupported;
        try result.append(a, source[at]);
        at += 1;
    }
    return result.items;
}

const Parser = struct {
    a: Allocator,
    source: []const u8,
    at: usize = 0,
    depth: usize = 0,
    nodes: std.ArrayList(Node) = .empty,
    heights: [32]u8 = @splat(0),

    fn space(p: *Parser) void {
        while (p.at < p.source.len and (p.source[p.at] == ' ' or p.source[p.at] == '\t')) p.at += 1;
    }

    fn take(p: *Parser, text: []const u8) bool {
        p.space();
        if (!std.mem.startsWith(u8, p.source[p.at..], text)) return false;
        p.at += text.len;
        return true;
    }

    fn push(p: *Parser, node: Node) Failure!u32 {
        if (p.nodes.items.len >= 32) return error.Unsupported;
        const height: u8 = switch (node.kind) {
            .literal => 1,
            .neg, .pow => p.heights[node.a] + 1,
            else => @max(p.heights[node.a], p.heights[node.b]) + 1,
        };
        if (height > 24) return error.Unsupported;
        p.heights[p.nodes.items.len] = height;
        try p.nodes.append(p.a, node);
        return @intCast(p.nodes.items.len - 1);
    }

    fn sum(p: *Parser) Failure!u32 {
        var lhs = try p.product();
        while (true) {
            const op: Kind = if (p.take("+")) .add else if (p.take("-")) .sub else break;
            lhs = try p.push(.{ .kind = op, .a = lhs, .b = try p.product() });
        }
        return lhs;
    }

    fn product(p: *Parser) Failure!u32 {
        var lhs = try p.unary();
        while (true) {
            const op: Kind = if (p.take("*")) .mul else if (p.take("/")) .div else break;
            lhs = try p.push(.{ .kind = op, .a = lhs, .b = try p.unary() });
        }
        return lhs;
    }

    fn unary(p: *Parser) Failure!u32 {
        if (p.depth >= 24) return error.Unsupported;
        p.depth += 1;
        defer p.depth -= 1;
        if (p.take("+")) return p.unary();
        if (p.take("-")) return p.push(.{ .kind = .neg, .a = try p.unary() });
        const lhs = try p.atom();
        if (!p.take("**") and !p.take("^")) return lhs;
        p.space();
        const start = p.at;
        while (p.at < p.source.len and std.ascii.isDigit(p.source[p.at])) p.at += 1;
        if (start == p.at or p.at - start > 3) return error.Unsupported;
        const exponent = std.fmt.parseInt(u16, p.source[start..p.at], 10) catch return error.Unsupported;
        if (exponent > 999) return error.Unsupported;
        return p.push(.{ .kind = .pow, .a = lhs, .exponent = exponent });
    }

    fn atom(p: *Parser) Failure!u32 {
        if (p.take("(")) {
            const root = try p.sum();
            if (!p.take(")")) return error.Unsupported;
            return root;
        }
        p.space();
        const start = p.at;
        while (p.at < p.source.len and std.ascii.isDigit(p.source[p.at])) p.at += 1;
        if (p.at == start) return error.Unsupported;
        const whole_end = p.at;
        var decimal = false;
        var scale: usize = 0;
        if (p.at < p.source.len and p.source[p.at] == '.') {
            decimal = true;
            p.at += 1;
            const fraction_start = p.at;
            while (p.at < p.source.len and std.ascii.isDigit(p.source[p.at])) p.at += 1;
            scale = p.at - fraction_start;
            if (scale == 0 or scale > 18) return error.Unsupported;
        }
        if (whole_end - start + scale > 62) return error.Unsupported;
        const text = try p.a.alloc(u8, whole_end - start + scale);
        @memcpy(text[0 .. whole_end - start], p.source[start..whole_end]);
        if (decimal) @memcpy(text[whole_end - start ..], p.source[whole_end + 1 .. p.at]);
        var first: usize = 0;
        while (first + 1 < text.len and text[first] == '0') first += 1;
        return p.push(.{ .kind = .literal, .text = text[first..], .scale = @intCast(scale), .decimal = decimal });
    }
};

const Lower = struct {
    a: Allocator,
    bank: *const Bank,
    nodes: []const Node,
    values: std.ArrayList([]const u8) = .empty,
    ops: std.ArrayList(packet.Op) = .empty,
    literals: [32]u32 = @splat(0),
    scales: [19]u32 = undefined,
    powers: [19]u32 = undefined,
    produced: usize = 0,

    fn constant(l: *Lower, text: []const u8) Failure!u32 {
        if (l.produced != 0) return error.Unsupported;
        const ref: u32 = @intCast(l.values.items.len);
        try l.values.append(l.a, text);
        return ref;
    }

    fn initialize(l: *Lower) Failure!void {
        try l.values.appendSlice(l.a, &.{ "0", "1", "10", "100" });
        for (0..19) |s| {
            l.scales[s] = try l.constant(try std.fmt.allocPrint(l.a, "{d}", .{s}));
            const power = try l.a.alloc(u8, s + 1);
            @memset(power, '0');
            power[0] = '1';
            l.powers[s] = try l.constant(power);
        }
        for (l.nodes, 0..) |node, i| if (node.kind == .literal) {
            l.literals[i] = try l.constant(node.text);
        };
    }

    fn width(digits: usize) Failure!u16 {
        for ([_]u16{ 1, 2, 4, 8, 16, 32, 62 }) |w| if (digits <= w) return w;
        return error.Unsupported;
    }

    fn operation(l: *Lower, name: []const u8, args: []const u32, digits: usize, outputs: u8) Failure!Bound {
        const w = try width(digits);
        if (l.ops.items.len >= 1024 or l.values.items.len + l.produced + outputs > packet.max_values)
            return error.Unsupported;
        if (std.mem.eql(u8, name, "fraction") or std.mem.eql(u8, name, "divmod")) {
            if (args.len != 2 or outputs != 2) return error.Unsupported;
        } else {
            const t = l.bank.find(name, w) orelse return error.Unsupported;
            if (t.inputs != args.len or t.outputs != outputs) return error.Unsupported;
        }
        const ref: u32 = @intCast(l.values.items.len + l.produced);
        try l.ops.append(l.a, .{
            .op = name,
            .width = w,
            .args = try l.a.dupe(u32, args),
            .outputs = outputs,
            .checked = true,
        });
        l.produced += outputs;
        return .{ .ref = ref, .digits = digits };
    }

    fn binary(l: *Lower, name: []const u8, x: Bound, y: Bound, digits: usize) Failure!Bound {
        return l.operation(name, &.{ x.ref, y.ref }, @max(digits, @max(x.digits, y.digits)), 1);
    }

    fn multiply(l: *Lower, x: Bound, y: Bound) Failure!Bound {
        if (x.ref == 1) return y;
        if (y.ref == 1) return x;
        return l.binary("decimal_mul", x, y, x.digits + y.digits);
    }

    fn scaled(l: *Lower, x: Bound, places: u8) Failure!Bound {
        var result = x;
        for (0..places) |_| result = try l.operation("scale10", &.{result.ref}, result.digits + 1, 1);
        return result;
    }

    fn ratio(l: *Lower, x: Number) Number {
        if (x.scale) |s| return .{
            .n = x.n,
            .d = .{ .ref = if (s == 0) 1 else l.powers[s], .digits = @as(usize, s) + 1 },
            .scale = null,
            .decimal = x.decimal,
        };
        return x;
    }

    fn combine(l: *Lower, kind: Kind, x: Number, y: Number) Failure!Number {
        const decimal = x.decimal or y.decimal;
        if (kind != .div and x.scale != null and y.scale != null) {
            const sx = x.scale.?;
            const sy = y.scale.?;
            if (kind == .mul) {
                if (@as(u16, sx) + sy > 18) return error.Unsupported;
                return .{ .n = try l.multiply(x.n, y.n), .scale = sx + sy, .decimal = decimal };
            }
            const s = @max(sx, sy);
            const nx = try l.scaled(x.n, s - sx);
            const ny = try l.scaled(y.n, s - sy);
            return .{
                .n = try l.binary(@tagName(kind), nx, ny, @max(nx.digits, ny.digits) + 1),
                .scale = s,
                .decimal = decimal,
            };
        }
        const rx = l.ratio(x);
        const ry = l.ratio(y);
        if (kind == .div) {
            const nonzero = try l.binary("compare", ry.n, .{ .ref = 0, .digits = 1 }, ry.n.digits);
            _ = try l.operation("assert_true", &.{nonzero.ref}, 1, 1);
        }
        if (kind == .mul or kind == .div) return .{
            .n = try l.multiply(rx.n, if (kind == .div) ry.d else ry.n),
            .d = try l.multiply(rx.d, if (kind == .div) ry.n else ry.d),
            .scale = null,
            .decimal = decimal,
        };
        const nx = try l.multiply(rx.n, ry.d);
        const ny = try l.multiply(ry.n, rx.d);
        return .{
            .n = try l.binary(@tagName(kind), nx, ny, @max(nx.digits, ny.digits) + 1),
            .d = try l.multiply(rx.d, ry.d),
            .scale = null,
            .decimal = decimal,
        };
    }

    fn emit(l: *Lower, index: u32) Failure!Number {
        const node = l.nodes[index];
        if (node.kind == .literal) return .{
            .n = .{ .ref = l.literals[index], .digits = node.text.len },
            .scale = node.scale,
            .decimal = node.decimal,
        };
        var lhs = try l.emit(node.a);
        if (node.kind == .neg) {
            lhs.n = try l.binary("sub", .{ .ref = 0, .digits = 1 }, lhs.n, lhs.n.digits);
            return lhs;
        }
        if (node.kind == .pow) {
            var n = node.exponent;
            var result: Number = .{ .n = .{ .ref = 1, .digits = 1 }, .decimal = lhs.decimal };
            while (n != 0) {
                if (n & 1 != 0) result = try l.combine(.mul, result, lhs);
                n >>= 1;
                if (n != 0) lhs = try l.combine(.mul, lhs, lhs);
            }
            return result;
        }
        return l.combine(node.kind, lhs, try l.emit(node.b));
    }

    fn finish(l: *Lower, value: Number) Failure!packet.Print {
        if (!value.decimal) {
            const r = l.ratio(value);
            const reduced = try l.operation("fraction", &.{ r.n.ref, r.d.ref }, @max(r.n.digits, r.d.digits), 2);
            return .{ .ref = reduced.ref, .second = reduced.ref + 1, .style = .fraction };
        }
        if (value.scale) |s| return l.decimalPrint(value.n, s);
        const reduced = try l.operation("fraction", &.{ value.n.ref, value.d.ref }, @max(value.n.digits, value.d.digits), 2);
        if (value.n.digits + 18 > 62) return error.Unsupported;
        const numerator = try l.scaled(.{ .ref = reduced.ref, .digits = value.n.digits }, 18);
        const quotient = try l.operation("divmod", &.{ numerator.ref, reduced.ref + 1 }, @max(numerator.digits, value.d.digits), 2);
        const unequal = try l.binary("compare", .{ .ref = quotient.ref + 1, .digits = quotient.digits }, .{ .ref = 0, .digits = 1 }, quotient.digits);
        const equal = try l.binary("sub", .{ .ref = 1, .digits = 1 }, .{ .ref = unequal.ref, .digits = 1 }, 1);
        _ = try l.operation("assert_true", &.{equal.ref}, 1, 1);
        return l.decimalPrint(.{ .ref = quotient.ref, .digits = numerator.digits }, 18);
    }

    fn decimalPrint(l: *Lower, coefficient: Bound, scale: u8) Failure!packet.Print {
        var c = coefficient;
        var precision: Bound = .{ .ref = l.scales[scale], .digits = 2 };
        for (0..scale) |_| {
            const qr = try l.operation("divmod", &.{ c.ref, 2 }, @max(c.digits, 2), 2);
            const flag = try l.binary("compare", .{ .ref = qr.ref + 1, .digits = qr.digits }, .{ .ref = 0, .digits = 1 }, qr.digits);
            c = try l.operation("select", &.{ c.ref, qr.ref, flag.ref }, @max(c.digits, 2), 1);
            const delta = try l.operation("select", &.{ 0, 1, flag.ref }, 1, 1);
            precision = try l.binary("sub", precision, .{ .ref = delta.ref, .digits = 1 }, 2);
        }
        c = try l.binary("add", c, .{ .ref = 0, .digits = 1 }, c.digits);
        const control = try l.binary("add", .{ .ref = 3, .digits = 3 }, precision, 3);
        return .{ .ref = c.ref, .second = control.ref, .style = .prepared };
    }
};
