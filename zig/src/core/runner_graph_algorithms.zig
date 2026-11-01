//! Integer algorithms orchestrate prepared FFNs and transfer raw rows without decoding operands.
const std = @import("std");
const api = @import("runner_api.zig");
const packet = @import("runner_graph_packet.zig");
const End = @FieldType(api.RunEnd, "end");
const Cancel = ?*const std.atomic.Value(bool);

/// Emits one result line; divmod emits the Python tuple (quotient, remainder) through prepared emitters.
pub fn run(m: anytype, kind: []const u8, width: u8, budget: u64, out: *std.ArrayList(u8), cancel: Cancel) !api.RunEnd {
    if (!packet.widthValid(width) or m.values.len < 32) return error.GraphAlgorithmScratch;
    const arity: usize = if (std.mem.eql(u8, kind, "powmod")) 3 else if (std.mem.eql(u8, kind, "gcd") or std.mem.eql(u8, kind, "range_product") or
        std.mem.eql(u8, kind, "divmod") or std.mem.eql(u8, kind, "square_offset")) 2 else if (std.mem.eql(u8, kind, "isqrt") or std.mem.eql(u8, kind, "factorial") or
        std.mem.eql(u8, kind, "digit_sum")) 1 else return error.GraphAlgorithm;
    var inputs: [3]packet.Value = undefined;
    @memcpy(inputs[0..arity], m.values[0..arity]);
    const zero = try packet.literal("0");
    for (m.values[0..32]) |*value| value.* = zero;
    m.values[1] = try packet.literal("1");
    @memcpy(m.values[2 .. 2 + arity], inputs[0..arity]);
    const input_digits = @max(1, std.mem.trimStart(u8, m.parsed.value.values[0], "+-0").len);
    var c: Context(@TypeOf(m)) = .{ .m = m, .width = width, .budget = budget, .out = out, .cancel = cancel, .input_digits = input_digits };
    c.execute(kind) catch |err| switch (err) {
        error.GraphAlgorithmStopped => return .{ .end = c.end, .steps = m.steps },
        error.GraphAlgorithmLimit => return .{ .end = .budget, .steps = m.steps },
        error.GraphAlgorithmArithmetic => {
            m.values[0][0][127] = packet.bit_scale;
            c.emit(0) catch |emit_err| switch (emit_err) {
                error.GraphAlgorithmStopped => return .{ .end = c.end, .steps = m.steps },
                else => return emit_err,
            };
        },
        else => return err,
    };
    return .{ .end = .halted, .steps = m.steps };
}

fn Context(comptime M: type) type {
    return struct {
        m: M,
        width: u8,
        budget: u64,
        out: *std.ArrayList(u8),
        cancel: Cancel,
        input_digits: usize,
        end: End = .halted,

        const Self = @This();

        fn call(c: *Self, name: []const u8, refs: []const u32, dest: usize) !void {
            const width = if (std.mem.eql(u8, name, "meta")) @max(c.width, 8) else c.width;
            var needed: usize = 1;
            for (refs) |ref| needed = @max(needed, occupiedDigits(&c.m.values[ref]));
            if (std.mem.eql(u8, name, "mul")) {
                needed = occupiedDigits(&c.m.values[refs[0]]) + occupiedDigits(&c.m.values[refs[1]]);
            } else if (std.mem.eql(u8, name, "add") or std.mem.eql(u8, name, "sub")) {
                needed += 1;
            } else if (std.mem.eql(u8, name, "meta")) needed = @max(needed, 8);
            const selected = bucket(@min(needed, width));
            const t = (if (c.m.adaptive_widths) c.m.bank.find(name, selected) else null) orelse
                c.m.bank.find(name, width) orelse return error.GraphTemplate;
            c.end = try c.m.invoke(t, refs, dest, c.budget, c.out, c.cancel, false);
            if (c.end != .halted) return error.GraphAlgorithmStopped;
            for (dest..dest + t.outputs) |i| try clean(&c.m.values[i]);
        }

        fn emit(c: *Self, ref: u32) !void {
            return c.emitNamed("emit", ref);
        }

        fn emitNamed(c: *Self, name: []const u8, ref: u32) !void {
            const needed = if (c.m.adaptive_widths) occupiedDigits(&c.m.values[ref]) else c.width;
            const width: u8 = if (needed > 62) 128 else 62;
            const t = c.m.bank.find(name, width) orelse return error.GraphEmitMissing;
            c.end = try c.m.invoke(t, &.{ref}, 0, c.budget, c.out, c.cancel, true);
            if (c.end != .halted) return error.GraphAlgorithmStopped;
        }

        fn move(c: *Self, dest: usize, source: usize) void {
            c.m.values[dest] = c.m.values[source];
        }

        fn meta(c: *Self, a: u32, b: u32) !void {
            try c.call("meta", &.{ a, b }, 20);
        }

        fn flag(c: *Self, digit: usize) !bool {
            return predicate(&c.m.values[22], digit);
        }

        fn nonzero(c: *Self, ref: u32) !bool {
            try c.meta(ref, 0);
            return c.flag(0);
        }

        /// Slots 24..30 are private to long division; both result slots must be outside that interval.
        fn unsigned(c: *Self, a: u32, b: u32, q: usize, r: usize) !void {
            c.move(24, a);
            c.move(25, b);
            c.move(26, 0);
            c.move(27, 0);
            var digit: usize = c.width;
            while (digit > 0) {
                digit -= 1;
                c.move(28, 0);
                packet.copyBits(&c.m.values[28], &c.m.values[24], .{
                    .ref = 24,
                    .from = @intCast(digit),
                    .digits = 1,
                });
                try c.call("divstep", &.{ 27, 25, 28 }, 29);
                c.move(27, 29);
                packet.copyBits(&c.m.values[26], &c.m.values[30], .{
                    .ref = 30,
                    .to = @intCast(digit),
                    .digits = 1,
                });
            }
            c.move(q, 26);
            c.move(r, 27);
        }

        fn execute(c: *Self, kind: []const u8) !void {
            if (std.mem.eql(u8, kind, "divmod")) {
                try c.meta(2, 3);
                try c.unsigned(20, 21, 5, 6);
                try c.call("divfinal", &.{ 5, 6, 2, 3 }, 10);
                try c.emitNamed("emit_first", 10);
                return c.emitNamed("emit_last", 11);
            }
            const result: u32 = if (std.mem.eql(u8, kind, "gcd")) try c.gcd() else if (std.mem.eql(u8, kind, "isqrt")) try c.isqrt() else if (std.mem.eql(u8, kind, "digit_sum")) try c.digitSum() else if (std.mem.eql(u8, kind, "powmod")) try c.powmod() else if (std.mem.eql(u8, kind, "square_offset")) try c.squareOffset() else try c.product(std.mem.eql(u8, kind, "factorial"));
            try c.emit(result);
        }

        fn gcd(c: *Self) !u32 {
            try c.meta(2, 3);
            c.move(5, 20);
            c.move(6, 21);
            var iterations: usize = 0;
            while (try c.nonzero(6)) {
                if (iterations >= @as(usize, c.width) * 8) return error.GraphAlgorithmLimit;
                iterations += 1;
                try c.unsigned(5, 6, 7, 8);
                c.move(5, 6);
                c.move(6, 8);
            }
            return 5;
        }

        fn isqrt(c: *Self) !u32 {
            try c.meta(2, 0);
            if (try c.flag(2)) return error.GraphAlgorithmArithmetic;
            if (!try c.flag(0)) return 0;
            c.move(5, 20);
            const nines: [packet.max_digits]u8 = @splat('9');
            c.m.values[6] = try packet.literal(nines[0 .. (c.input_digits + 1) / 2]);
            for (0..@as(usize, c.width) * 8) |_| {
                try c.unsigned(5, 6, 7, 8);
                try c.call("half", &.{6}, 12);
                c.move(9, 12);
                c.move(10, 13);
                try c.call("half", &.{7}, 12);
                try c.call("add", &.{ 9, 12 }, 11);
                if (try predicate(&c.m.values[10], 0)) {
                    if (try predicate(&c.m.values[13], 0)) {
                        try c.call("add", &.{ 11, 1 }, 11);
                    }
                }
                try c.meta(11, 6);
                if (!try c.flag(4)) return 6;
                c.move(6, 11);
            }
            return error.GraphAlgorithmLimit;
        }

        fn digitSum(c: *Self) !u32 {
            try c.call("digit_sum", &.{2}, 6);
            return 6;
        }

        fn squareOffset(c: *Self) !u32 {
            try c.meta(2, 3);
            if (!try c.flag(1)) return 0;
            try c.call("add", &.{ 2, 3 }, 5);
            try c.call("add", &.{ 2, 5 }, 6);
            try c.call("mul", &.{ 6, 3 }, 7);
            return 7;
        }

        /// Range products use [start, stop); factorial includes n and never constructs n+1.
        fn product(c: *Self, factorial: bool) !u32 {
            try c.meta(2, if (factorial) 0 else 3);
            if (try c.flag(2)) return error.GraphAlgorithmArithmetic;
            if (try c.flag(3)) return error.GraphAlgorithmArithmetic;
            c.move(5, if (factorial) 1 else 20);
            c.move(6, if (factorial) 20 else 21);
            c.move(7, 1);
            var iterations: usize = 0;
            while (true) {
                try c.meta(5, 6);
                const less = try c.flag(4);
                const equal = try c.flag(5);
                if (!less and !(factorial and equal)) return 7;
                if (iterations >= 10000) return error.GraphAlgorithmLimit;
                iterations += 1;
                try c.call("mul", &.{ 7, 5 }, 7);
                if (factorial and equal) return 7;
                try c.call("add", &.{ 5, 1 }, 5);
            }
        }

        /// One prepared Horner step consumes a decimal multiplier digit; modulus stays in slot 7.
        fn modmul(c: *Self, a: u32, b: u32, dest: usize) !void {
            c.move(17, a);
            c.move(18, b);
            c.move(16, 0);
            var digit = occupiedDigits(&c.m.values[18]);
            while (digit > 0) {
                digit -= 1;
                c.move(19, 0);
                packet.copyBits(&c.m.values[19], &c.m.values[18], .{
                    .ref = 18,
                    .from = @intCast(digit),
                    .digits = 1,
                });
                try c.call("mulmodstep", &.{ 16, 17, 19, 7 }, 16);
            }
            c.move(dest, 16);
        }

        fn powmod(c: *Self) !u32 {
            try c.meta(2, 4);
            c.move(5, 20);
            c.move(7, 21);
            c.move(9, 22);
            if (!try c.flag(1)) return error.GraphAlgorithmArithmetic;
            try c.meta(3, 0);
            if (try c.flag(2)) return error.GraphAlgorithmArithmetic;
            c.move(6, 20);
            try c.unsigned(5, 7, 10, 11);
            try c.call("divfinal", &.{ 10, 11, 2, 7 }, 10);
            c.move(5, 11);
            try c.unsigned(1, 7, 10, 11);
            c.move(8, 11);
            var iterations: usize = 0;
            while (try c.nonzero(6)) {
                if (iterations >= @as(usize, c.width) * 4) return error.GraphAlgorithmLimit;
                iterations += 1;
                try c.call("half", &.{6}, 12);
                c.move(6, 12);
                if (try predicate(&c.m.values[13], 0)) try c.modmul(8, 5, 8);
                if (try c.nonzero(6)) try c.modmul(5, 5, 5);
            }
            if (try predicate(&c.m.values[9], 3)) {
                if (try c.nonzero(8)) try c.call("sub", &.{ 8, 7 }, 8);
            }
            return 8;
        }
    };
}

fn occupiedDigits(value: *const packet.Value) usize {
    var digits: usize = packet.max_digits;
    while (digits > 1) : (digits -= 1) {
        const at = digits - 1;
        for (value[at / 31][2 + at % 31 * 4 ..][0..4]) |bit| if (bit != 0) return digits;
    }
    return digits;
}

fn bucket(needed: usize) u8 {
    for ([_]u8{ 1, 2, 4, 8, 16, 32, 62, 128 }) |width| if (needed <= width) return width;
    unreachable;
}

fn clean(value: *const packet.Value) !void {
    const bit = value[0][127];
    if (bit == packet.bit_scale) return error.GraphAlgorithmArithmetic;
    if (bit != 0) return error.GraphPredicate;
}

fn predicate(value: *const packet.Value, digit: usize) !bool {
    try clean(value);
    const bits = value[digit / 31][2 + digit % 31 * 4 ..][0..4];
    for (bits[1..]) |bit| if (bit != 0) return error.GraphPredicate;
    if (bits[0] == 0) return false;
    if (bits[0] == packet.bit_scale) return true;
    return error.GraphPredicate;
}
