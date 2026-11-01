const std = @import("std");
const Invalid = error{Unsupported};
const limit: i512 = std.math.pow(i512, 10, 62);
const Value = struct { n: i512, d: i512 = 1, scale: usize = 0, width: usize = 0 };

// Checks firmware value limits; the returned program remains the source expression, not this validation value.
pub fn valid(source: []const u8) bool {
    if (source.len == 0 or source.len > 255) return false;
    var p: Parser = .{ .src = source };
    _ = p.expr() catch return false;
    return p.pos == source.len;
}

fn magnitude(n: i512) i512 {
    return if (n < 0) -n else n;
}

fn gcd(a: i512, b: i512) i512 {
    var x = magnitude(a);
    var y = magnitude(b);
    while (y != 0) {
        const r = @mod(x, y);
        x = y;
        y = r;
    }
    return x;
}

fn checked(n: i512, d: i512) Invalid!Value {
    if (d == 0) return error.Unsupported;
    const g = gcd(n, d);
    const sign: i512 = if (d < 0) -1 else 1;
    const v: Value = .{ .n = @divTrunc(n, g) * sign, .d = @divTrunc(d, g) * sign };
    if (magnitude(v.n) >= limit or v.d >= limit) return error.Unsupported;
    return v;
}

fn integer(v: Value) Invalid!i512 {
    if (v.d != 1 or v.scale != 0) return error.Unsupported;
    return v.n;
}

fn digits(n: i512) usize {
    var v = magnitude(n);
    var count: usize = 1;
    while (v >= 10) : (count += 1) v = @divTrunc(v, 10);
    return count;
}

fn rational(op: u8, a: Value, b: Value) Invalid!Value {
    if (op == '*') {
        if (digits(a.n) + digits(b.n) > 62 or
            (a.d != 1 and b.d != 1 and digits(a.d) + digits(b.d) > 62)) return error.Unsupported;
        return checked(a.n * b.n, a.d * b.d);
    }
    if (op == '/') {
        if (b.n == 0 or (b.d != 1 and digits(a.n) + digits(b.d) > 62) or
            (a.d != 1 and digits(a.d) + digits(b.n) > 62)) return error.Unsupported;
        return checked(a.n * b.d, a.d * b.n);
    }
    if ((b.d != 1 and digits(a.n) + digits(b.d) > 62) or
        (a.d != 1 and digits(b.n) + digits(a.d) > 62) or
        (a.d != 1 and b.d != 1 and digits(a.d) + digits(b.d) > 62)) return error.Unsupported;
    const x = a.n * b.d;
    const y = if (op == '+') b.n * a.d else -b.n * a.d;
    if ((x < 0) == (y < 0) and @max(digits(x), digits(y)) + 1 > 62) return error.Unsupported;
    return checked(x + y, a.d * b.d);
}

fn width(v: Value) usize {
    return if (v.width == 0) digits(v.n) else v.width;
}

fn decimal(op: u8, a: Value, b: Value) Invalid!Value {
    if (op != '+' and op != '-' and op != '*') return error.Unsupported;
    if (op == '*') {
        const w = width(a) + width(b);
        if (w > 62) return error.Unsupported;
        var v = try checked(a.n * b.n, 1);
        v.scale = a.scale + b.scale;
        v.width = w;
        return v;
    }
    // Firmware shift() leaves shifted zero untrimmed; reject it before later magnitude comparisons.
    if ((a.n == 0 and a.scale < b.scale) or (b.n == 0 and b.scale < a.scale)) return error.Unsupported;
    const scale = @max(a.scale, b.scale);
    const wa = width(a) + scale - a.scale;
    const wb = width(b) + scale - b.scale;
    if (wa > 62 or wb > 62) return error.Unsupported;
    const x = a.n * std.math.pow(i512, 10, scale - a.scale);
    const y = b.n * std.math.pow(i512, 10, scale - b.scale) * @as(i512, if (op == '+') 1 else -1);
    const w = @max(wa, wb) + @intFromBool((x < 0) == (y < 0));
    if (w > 62) return error.Unsupported;
    var v = try checked(x + y, 1);
    v.scale = scale;
    v.width = w;
    return v;
}

const Parser = struct {
    src: []const u8,
    pos: usize = 0,

    fn take(self: *Parser, s: []const u8) bool {
        if (!std.mem.startsWith(u8, self.src[self.pos..], s)) return false;
        self.pos += s.len;
        return true;
    }

    fn expr(self: *Parser) Invalid!Value {
        var a = try self.term();
        while (self.pos < self.src.len) {
            const op = self.src[self.pos];
            if (op != '+' and op != '-') break;
            self.pos += 1;
            const b = try self.term();
            if ((a.scale != 0 or b.scale != 0) and (a.d != 1 or b.d != 1)) return error.Unsupported;
            a = if (a.scale != 0 or b.scale != 0) try decimal(op, a, b) else try rational(op, a, b);
        }
        return a;
    }

    fn term(self: *Parser) Invalid!Value {
        var a = try self.factor();
        while (self.pos < self.src.len) {
            const op = self.src[self.pos];
            if (op != '*' and op != '/' and op != '%') break;
            self.pos += 1;
            const floor = op == '/' and self.take("/");
            const b = try self.factor();
            if (floor or op == '%') {
                const x = try integer(a);
                const y = try integer(b);
                if (y == 0) return error.Unsupported;
                if ((x < 0) != (y < 0) and @rem(x, y) != 0 and digits(@divTrunc(x, y)) == 62)
                    return error.Unsupported;
                a = try checked(if (floor) @divFloor(x, y) else x - @divFloor(x, y) * y, 1);
            } else if (a.scale != 0 or b.scale != 0) {
                if (op != '*' or a.d != 1 or b.d != 1) return error.Unsupported;
                a = try decimal(op, a, b);
            } else if (op == '/' or a.d != 1 or b.d != 1) {
                a = try rational(op, a, b);
            } else a = try rational('*', a, b);
        }
        return a;
    }

    fn factor(self: *Parser) Invalid!Value {
        if (self.take("+")) return self.factor();
        if (self.take("-")) {
            var v = try self.factor();
            v.n = -v.n;
            return v;
        }
        var a = try self.atom();
        if (self.take("**")) {
            var exp = try integer(try self.factor());
            if (exp < 0 or exp > 999) return error.Unsupported;
            var base = Value{ .n = try integer(a) };
            a = .{ .n = 1 };
            while (exp > 0) {
                if (@mod(exp, 2) == 1) a = try rational('*', a, base);
                exp = @divTrunc(exp, 2);
                if (exp > 0) base = try rational('*', base, base);
            }
        }
        return a;
    }

    fn atom(self: *Parser) Invalid!Value {
        if (self.take("gcd(")) {
            const a = try integer(try self.expr());
            if (!self.take(",")) return error.Unsupported;
            const b = try integer(try self.expr());
            if (!self.take(")")) return error.Unsupported;
            return .{ .n = gcd(a, b) };
        }
        if (self.take("(")) {
            const v = try self.expr();
            if (!self.take(")")) return error.Unsupported;
            return v;
        }
        var n: i512 = 0;
        var scale: usize = 0;
        var dot = false;
        var count: usize = 0;
        const start = self.pos;
        while (self.pos < self.src.len) {
            const c = self.src[self.pos];
            if (c == '.' and !dot) {
                dot = true;
            } else if (std.ascii.isDigit(c)) {
                count += 1;
                if (count > 62) return error.Unsupported;
                n = n * 10 + c - '0';
                if (dot) scale += 1;
            } else break;
            self.pos += 1;
        }
        if (start == self.pos) return error.Unsupported;
        return .{ .n = n, .scale = scale };
    }
};

test "value preflight refuses overflow and arithmetic errors" {
    for ([_][]const u8{ "2**100", "-2**2", "(-2)**2", "7//-3", "1/3+2/3", "1.25*2.4" }) |s|
        try std.testing.expect(valid(s));
    for ([_][]const u8{ "10**100", "2**1000", "2**-1", "1/0", "1//0", "1%0", "gcd(1/2,3)", "(1/2)**3" }) |s|
        try std.testing.expect(!valid(s));
}
