//! Bounded literal binding packets and pure frontend serialization; operations contain no executable graph IR.
const std = @import("std");
const Allocator = std.mem.Allocator;
pub const max_packet = 1 << 20;
pub const max_ops = 4096;
pub const max_values = 8192;
pub const max_arity = 16;
pub const max_digits: usize = 128;
pub const max_groups = (max_digits + 30) / 31;
pub const Value = [max_groups][128]f32;
pub const bit_scale: f32 = 128;
pub const reference_scale: f32 = 16777216;

pub const Copy = struct {
    ref: u32,
    from: u8 = 0,
    to: u8 = 0,
    digits: u8 = 62,
};

pub const Op = struct {
    op: []const u8,
    width: u8 = 62,
    a: u32 = 0,
    b: ?u32 = null,
    args: ?[]const u32 = null,
    outputs: u8 = 1,
    copies: []const Copy = &.{},
    flags: ?u32 = null,
};

pub const Packet = struct {
    version: u32 = 1,
    values: []const []const u8,
    plan: ?[]const u8 = null,
    ops: []const Op = &.{},
    result: u32 = 0,
};

pub const Plan = struct {
    name: []const u8,
    inputs: u32,
    literals: []const []const u8 = &.{},
    ops: []const Op,
    result: u32,
    algorithm: ?[]const u8 = null,
    width: u8 = 62,
};

pub const Family = enum { add, sub, mul, div, mod, pow };
pub const Algorithm = enum { powmod, gcd, isqrt, range_product, factorial, digit_sum, divmod, square_offset };
pub const Shape = enum { sum, squares, factorial, fib, list_sum };

pub const Builder = struct {
    pub fn encode(a: Allocator, packet: Packet) ![]u8 {
        if (packet.version != 1 or packet.values.len > max_values or packet.ops.len > max_ops)
            return error.GraphPacketSize;
        const bytes = try std.json.Stringify.valueAlloc(a, packet, .{});
        errdefer a.free(bytes);
        if (bytes.len > max_packet) return error.GraphPacketSize;
        return bytes;
    }

    pub fn named(a: Allocator, name: []const u8, values: []const []const u8) ![]u8 {
        return encode(a, .{ .values = values, .plan = name });
    }

    /// Selects a preloaded plan from operand metadata without evaluating its answer.
    pub fn binary(a: Allocator, family: Family, lhs: []const u8, rhs: []const u8) ![]u8 {
        _ = try literal(lhs);
        _ = try literal(rhs);
        const wa = lhs.len - @intFromBool(lhs[0] == '-' or lhs[0] == '+');
        const wb = rhs.len - @intFromBool(rhs[0] == '-' or rhs[0] == '+');
        if (family == .pow) {
            const exponent = std.fmt.parseInt(i16, rhs, 10) catch return error.GraphExponent;
            if (exponent < 0 or exponent > 999) return error.GraphExponent;
            const needed = wa * @max(@as(usize, 1), @as(usize, @intCast(exponent)));
            const width: u8 = if (needed > 62) 128 else 62;
            const name = try std.fmt.allocPrint(a, "pow-{d}-{d}", .{ exponent, width });
            defer a.free(name);
            return named(a, name, &.{lhs});
        }
        const needed = switch (family) {
            .add, .sub => @max(wa, wb) + 1,
            .mul => wa + wb,
            .div, .mod => @max(wa, wb),
            .pow => unreachable,
        };
        const width = for ([_]u8{ 1, 2, 4, 8, 16, 32, 62, 128 }) |n| {
            if (n >= @min(max_digits, needed)) break n;
        } else unreachable;
        const name = try std.fmt.allocPrint(a, "{s}-{d}", .{ @tagName(family), width });
        defer a.free(name);
        return named(a, name, &.{ lhs, rhs });
    }

    /// Selects an offline integer algorithm using operand widths without evaluating the result.
    pub fn integer(a: Allocator, family: Algorithm, operands: []const []const u8) ![]u8 {
        const arity: usize = switch (family) {
            .powmod => 3,
            .gcd, .range_product, .divmod, .square_offset => 2,
            .isqrt, .factorial, .digit_sum => 1,
        };
        if (operands.len != arity) return error.GraphSlotArity;
        var needed: usize = 1;
        for (operands) |operand| {
            _ = try literal(operand);
            const digits = operand.len - @intFromBool(operand[0] == '-' or operand[0] == '+');
            needed = @max(needed, digits);
        }
        const width: u8 = if (family == .factorial or family == .range_product) 128 else blk: {
            for ([_]u8{ 1, 2, 4, 8, 16, 32, 62, 128 }) |n| if (needed <= n) break :blk n;
            unreachable;
        };
        const name = try std.fmt.allocPrint(a, "{s}-{d}", .{ @tagName(family), width });
        defer a.free(name);
        return named(a, name, operands);
    }

    /// Binds a recognized bounded-loop shape to its offline control schedule.
    pub fn loop(a: Allocator, shape: Shape, bound: u16, values: []const []const u8) ![]u8 {
        return loopWidth(a, shape, bound, values, 62);
    }

    pub fn loopWidth(a: Allocator, shape: Shape, bound: u16, values: []const []const u8, width: u8) ![]u8 {
        if (width != 62 and width != 128) return error.GraphLoop;
        if (bound > 128 or values.len != (if (shape == .list_sum) @as(usize, bound) else 0))
            return error.GraphLoop;
        for (values) |value| _ = try literal(value);
        const name = try std.fmt.allocPrint(a, "loop-{s}-{d}-{d}", .{ @tagName(shape), bound, width });
        defer a.free(name);
        return named(a, name, values);
    }
};

pub fn widthValid(n: u8) bool {
    return switch (n) {
        1, 2, 4, 8, 16, 32, 62, 128 => true,
        else => false,
    };
}

pub fn literal(text: []const u8) !Value {
    if (text.len == 0) return error.GraphLiteral;
    const signed = text[0] == '-' or text[0] == '+';
    const digits = text[@intFromBool(signed)..];
    if (digits.len == 0 or digits.len > max_digits) return error.GraphLiteral;
    var value: Value = @splat(@splat(0));
    for (&value) |*row| {
        row[0] = reference_scale;
        row[1] = reference_scale;
    }
    value[0][126] = if (text[0] == '-') bit_scale else 0;
    for (digits, 0..) |ch, i| {
        if (ch < '0' or ch > '9') return error.GraphLiteral;
        const digit = digits.len - i - 1;
        for (0..4) |bit| value[digit / 31][2 + digit % 31 * 4 + bit] =
            if ((ch - '0') >> @as(u3, @intCast(bit)) & 1 != 0) bit_scale else 0;
    }
    return value;
}

pub fn copyBits(out: *Value, source: *const Value, c: Copy) void {
    for (0..c.digits) |i| {
        const src = @as(usize, c.from) + i;
        const dst = @as(usize, c.to) + i;
        @memcpy(out[dst / 31][2 + dst % 31 * 4 ..][0..4], source[src / 31][2 + src % 31 * 4 ..][0..4]);
    }
}

test "frontend binding selects offline plans and rejects invalid controls" {
    const t = std.testing;
    const sum = try Builder.binary(t.allocator, .add, "-999", "1");
    defer t.allocator.free(sum);
    try t.expect(std.mem.indexOf(u8, sum, "add-4") != null);
    const power = try Builder.binary(t.allocator, .pow, "2", "100");
    defer t.allocator.free(power);
    try t.expect(std.mem.indexOf(u8, power, "pow-100-128") != null);
    try t.expectError(error.GraphExponent, Builder.binary(t.allocator, .pow, "2", "-1"));
    try t.expectError(error.GraphLiteral, Builder.binary(t.allocator, .mul, "12x", "3"));
    const squares = try Builder.loop(t.allocator, .squares, 10, &.{});
    defer t.allocator.free(squares);
    try t.expect(std.mem.indexOf(u8, squares, "loop-squares-10-62") != null);
    try t.expectError(error.GraphLoop, Builder.loop(t.allocator, .squares, 129, &.{}));
    try t.expectError(error.GraphLoop, Builder.loop(t.allocator, .list_sum, 2, &.{"1"}));
}

test "128 digit bindings span five groups and retain explicit wide plans" {
    const t = std.testing;
    const text: [max_digits]u8 = @splat('9');
    const too_wide: [max_digits + 1]u8 = @splat('9');
    const value = try literal(&text);
    try t.expectEqual(bit_scale, value[4][14]);
    try t.expectEqual(@as(f32, 0), value[4][15]);
    try t.expectError(error.GraphLiteral, literal(&too_wide));
    var copied: Value = @splat(@splat(0));
    copyBits(&copied, &value, .{ .ref = 0, .from = 124, .to = 30, .digits = 4 });
    try t.expectEqual(bit_scale, copied[0][122]);
    try t.expectEqual(bit_scale, copied[1][10]);
    const sum = try Builder.binary(t.allocator, .add, &text, "1");
    defer t.allocator.free(sum);
    try t.expect(std.mem.indexOf(u8, sum, "add-128") != null);
    const power = try Builder.binary(t.allocator, .pow, &text, "1");
    defer t.allocator.free(power);
    try t.expect(std.mem.indexOf(u8, power, "pow-1-128") != null);
    const loop = try Builder.loopWidth(t.allocator, .factorial, 80, &.{}, 128);
    defer t.allocator.free(loop);
    try t.expect(std.mem.indexOf(u8, loop, "loop-factorial-80-128") != null);
    try t.expectError(error.GraphLoop, Builder.loopWidth(t.allocator, .fib, 10, &.{}, 32));
}

test "integer algorithms bind arity and maximum operand width" {
    const t = std.testing;
    const narrow: [40]u8 = @splat('9');
    const wide_text: [63]u8 = @splat('9');
    const too_wide: [129]u8 = @splat('9');
    const modular = try Builder.integer(t.allocator, .powmod, &.{ &narrow, "3", &narrow });
    defer t.allocator.free(modular);
    try t.expect(std.mem.indexOf(u8, modular, "powmod-62") != null);
    const wide = try Builder.integer(t.allocator, .gcd, &.{ &wide_text, "3" });
    defer t.allocator.free(wide);
    try t.expect(std.mem.indexOf(u8, wide, "gcd-128") != null);
    const power = try Builder.binary(t.allocator, .pow, "2", "10");
    defer t.allocator.free(power);
    try t.expect(std.mem.indexOf(u8, power, "pow-10-62") != null);
    try t.expectError(error.GraphSlotArity, Builder.integer(t.allocator, .powmod, &.{ "2", "3" }));
    try t.expectError(error.GraphLiteral, Builder.integer(t.allocator, .isqrt, &.{"x"}));
    try t.expectError(error.GraphLiteral, Builder.integer(t.allocator, .factorial, &.{&too_wide}));
}
