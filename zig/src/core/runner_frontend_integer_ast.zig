const std = @import("std");

pub const Invalid = error{Unsupported};
pub const limit: i512 = std.math.pow(i512, 10, 62);
pub const Op = enum(u8) { literal, neg, add, sub, mul, floor, mod, pow, powmod, digits, divmod, gcd, isqrt, factorial, range_product, digit_sum };
pub const Node = struct {
    op: Op,
    l: usize = 0,
    r: usize = 0,
    k: usize = 0,
    value: i512 = 0,
    text: []const u8 = "",
    base: u8 = 10,
};
pub const Parser = struct {
    source: []const u8,
    pos: usize = 0,
    nodes: [64]Node = undefined,
    count: usize = 0,
    proved: [64]bool = @splat(false),
    names: [32][]const u8 = undefined,
    roots: [32]usize = undefined,
    binding_count: usize = 0,
    require_math_import: bool = false,
    math_module: bool = false,
    math_imports: u8 = 0,

    // Names and literal slices borrow source storage; bind after parsing the RHS to preserve prior roots.
    pub fn bind(self: *Parser, name: []const u8, root: usize) Invalid!void {
        if (name.len == 0 or root >= self.count) return error.Unsupported;
        for (self.names[0..self.binding_count], self.roots[0..self.binding_count]) |bound_name, *bound_root| {
            if (std.mem.eql(u8, name, bound_name)) {
                bound_root.* = root;
                return;
            }
        }
        if (self.binding_count == self.names.len) return error.Unsupported;
        self.names[self.binding_count] = name;
        self.roots[self.binding_count] = root;
        self.binding_count += 1;
    }

    pub fn parse(self: *Parser) Invalid!usize {
        self.space();
        if (self.pos == self.source.len or self.source[self.pos] == '+') return error.Unsupported;
        const root = try self.expr(0);
        self.space();
        if (self.pos != self.source.len) return error.Unsupported;
        _ = try self.prove(root, root);
        return root;
    }

    fn space(self: *Parser) void {
        while (self.pos < self.source.len and (self.source[self.pos] == ' ' or self.source[self.pos] == '\t'))
            self.pos += 1;
    }

    fn take(self: *Parser, s: []const u8) bool {
        self.space();
        if (!std.mem.startsWith(u8, self.source[self.pos..], s)) return false;
        self.pos += s.len;
        return true;
    }

    fn need(self: *Parser, s: []const u8) Invalid!void {
        if (!self.take(s)) return error.Unsupported;
    }

    fn add(self: *Parser, node: Node) Invalid!usize {
        if (self.count == self.nodes.len) return error.Unsupported;
        self.nodes[self.count] = node;
        self.count += 1;
        return self.count - 1;
    }

    fn expr(self: *Parser, depth: usize) Invalid!usize {
        if (depth > 12) return error.Unsupported;
        var left = try self.term(depth);
        while (true) {
            const op: Op = if (self.take("+")) .add else if (self.take("-")) .sub else break;
            left = try self.add(.{ .op = op, .l = left, .r = try self.term(depth) });
        }
        return left;
    }

    fn term(self: *Parser, depth: usize) Invalid!usize {
        var left = try self.factor(depth);
        while (true) {
            const op: Op = if (self.take("//")) .floor else if (self.take("%")) .mod else if (self.take("*")) .mul else break;
            const right = try self.factor(depth);
            const n = self.nodes[left];
            if (op == .mod and n.op == .pow) {
                left = try self.add(.{ .op = .powmod, .l = n.l, .r = n.r, .k = right });
            } else if (op == .mod and n.op == .neg and self.nodes[n.l].op == .pow) {
                const p = self.nodes[n.l];
                const reduced = try self.add(.{ .op = .powmod, .l = p.l, .r = p.r, .k = right });
                const negative = try self.add(.{ .op = .neg, .l = reduced });
                left = try self.add(.{ .op = .mod, .l = negative, .r = right });
            } else {
                left = try self.add(.{ .op = op, .l = left, .r = right });
            }
        }
        return left;
    }

    fn factor(self: *Parser, depth: usize) Invalid!usize {
        if (depth > 12) return error.Unsupported;
        if (self.take("+")) return self.factor(depth + 1);
        if (self.take("-")) return self.add(.{ .op = .neg, .l = try self.factor(depth + 1) });
        const left = try self.atom(depth);
        if (self.take("**")) return self.add(.{ .op = .pow, .l = left, .r = try self.factor(depth + 1) });
        return left;
    }

    fn atom(self: *Parser, depth: usize) Invalid!usize {
        if (self.take("(")) {
            const n = try self.expr(depth + 1);
            try self.need(")");
            return n;
        }
        if (try self.algorithm(depth)) |n| return n;
        if (self.word("sum")) return self.digitSum(depth);
        if (self.word("divmod")) {
            try self.need("(");
            const a = try self.expr(depth + 1);
            try self.need(",");
            const b = try self.expr(depth + 1);
            try self.need(")");
            return self.add(.{ .op = .divmod, .l = a, .r = b });
        }
        if (self.word("pow")) {
            try self.need("(");
            const a = try self.expr(depth + 1);
            try self.need(",");
            const b = try self.expr(depth + 1);
            if (!self.take(",")) return error.Unsupported;
            const m = try self.expr(depth + 1);
            try self.need(")");
            return self.add(.{ .op = .powmod, .l = a, .r = b, .k = m });
        }
        if (self.word("len")) {
            try self.need("(");
            try self.need("str");
            try self.need("(");
            const n = try self.expr(depth + 1);
            try self.need(")");
            try self.need(")");
            return self.add(.{ .op = .digits, .l = n });
        }
        if (self.word("int")) return self.convert();
        self.space();
        if (self.pos < self.source.len and
            (std.ascii.isAlphabetic(self.source[self.pos]) or self.source[self.pos] == '_'))
        {
            const name = try self.identifier();
            for (self.names[0..self.binding_count], self.roots[0..self.binding_count]) |bound_name, root| {
                if (std.mem.eql(u8, name, bound_name)) return root;
            }
            return error.Unsupported;
        }
        return self.literal();
    }

    fn algorithm(self: *Parser, depth: usize) Invalid!?usize {
        const qualified = self.word("math");
        if (qualified) try self.need(".");
        const op: Op = if (self.word("gcd")) .gcd else if (self.word("isqrt")) .isqrt else if (self.word("factorial")) .factorial else if (self.word("prod")) .range_product else {
            if (qualified) return error.Unsupported;
            return null;
        };
        if (self.require_math_import) {
            const mask: u8 = switch (op) {
                .gcd => 1,
                .isqrt => 2,
                .factorial => 4,
                .range_product => 8,
                else => unreachable,
            };
            if (if (qualified) !self.math_module else self.math_imports & mask == 0) return error.Unsupported;
        }
        try self.need("(");
        if (op == .range_product) {
            try self.need("range");
            try self.need("(");
        }
        var a = try self.expr(depth + 1);
        var b: usize = 0;
        var k: usize = 0;
        if (op == .gcd) {
            try self.need(",");
            b = try self.expr(depth + 1);
        }
        if (op == .range_product) {
            if (self.take(",")) {
                b = try self.expr(depth + 1);
            } else {
                b = a;
                a = try self.add(.{ .op = .literal, .value = 0, .text = "0" });
            }
            k = if (self.take(",")) try self.expr(depth + 1) else try self.add(.{
                .op = .literal,
                .value = 1,
                .text = "1",
            });
            try self.need(")");
        }
        try self.need(")");
        return try self.add(.{ .op = op, .l = a, .r = b, .k = k });
    }

    fn word(self: *Parser, token: []const u8) bool {
        self.space();
        const rest = self.source[self.pos..];
        if (!std.mem.startsWith(u8, rest, token)) return false;
        if (rest.len > token.len and (std.ascii.isAlphanumeric(rest[token.len]) or rest[token.len] == '_'))
            return false;
        self.pos += token.len;
        return true;
    }

    fn identifier(self: *Parser) Invalid![]const u8 {
        self.space();
        const start = self.pos;
        while (self.pos < self.source.len and
            (std.ascii.isAlphanumeric(self.source[self.pos]) or self.source[self.pos] == '_')) self.pos += 1;
        if (start == self.pos or std.ascii.isDigit(self.source[start])) return error.Unsupported;
        const name = self.source[start..self.pos];
        if (std.mem.eql(u8, name, "int") or std.mem.eql(u8, name, "__debug__")) return error.Unsupported;
        const reserved = [_][]const u8{ "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield" };
        for (reserved) |token| if (std.mem.eql(u8, name, token)) return error.Unsupported;
        return name;
    }

    fn digitSum(self: *Parser, depth: usize) Invalid!usize {
        try self.need("(");
        const mapped = self.word("map");
        if (mapped) {
            try self.need("(");
            try self.need("int");
            try self.need(",");
        } else {
            try self.need("int");
            try self.need("(");
            const name = try self.identifier();
            try self.need(")");
            if (!self.word("for")) return error.Unsupported;
            const binding = try self.identifier();
            if (!std.mem.eql(u8, name, binding) or !self.word("in")) return error.Unsupported;
        }
        try self.need("str");
        try self.need("(");
        const value = try self.expr(depth + 1);
        try self.need(")");
        if (mapped) try self.need(")");
        try self.need(")");
        return self.add(.{ .op = .digit_sum, .l = value });
    }

    fn convert(self: *Parser) Invalid!usize {
        try self.need("(");
        self.space();
        if (self.pos == self.source.len) return error.Unsupported;
        const quote = self.source[self.pos];
        if (quote != '\'' and quote != '"') return error.Unsupported;
        self.pos += 1;
        const begin = self.pos;
        while (self.pos < self.source.len and self.source[self.pos] != quote) : (self.pos += 1) {}
        if (self.pos == self.source.len) return error.Unsupported;
        const text = std.mem.trim(u8, self.source[begin..self.pos], " \t");
        self.pos += 1;
        var base: u8 = 10;
        if (self.take(",")) {
            if (self.take("16")) base = 16 else if (self.take("2")) base = 2 else return error.Unsupported;
        }
        try self.need(")");
        var digits = text;
        var sign: i512 = 1;
        if (digits.len != 0 and (digits[0] == '-' or digits[0] == '+')) {
            if (digits[0] == '-') sign = -1;
            digits = digits[1..];
        }
        if (digits.len >= 2 and digits[0] == '0' and
            ((base == 16 and (digits[1] == 'x' or digits[1] == 'X')) or
                (base == 2 and (digits[1] == 'b' or digits[1] == 'B')))) digits = digits[2..];
        return self.add(.{ .op = .literal, .value = sign * try radix(digits, base), .text = digits, .base = base });
    }

    fn literal(self: *Parser) Invalid!usize {
        self.space();
        const begin = self.pos;
        var base: u8 = 10;
        if (self.pos + 2 <= self.source.len and self.source[self.pos] == '0') {
            const c = self.source[self.pos + 1];
            if (c == 'x' or c == 'X') base = 16 else if (c == 'b' or c == 'B') base = 2;
            if (base != 10) self.pos += 2;
        }
        const start = self.pos;
        while (self.pos < self.source.len and digit(self.source[self.pos]) < base) : (self.pos += 1) {}
        if (base == 10 and self.pos - begin > 1 and self.source[begin] == '0') return error.Unsupported;
        const text = self.source[start..self.pos];
        return self.add(.{ .op = .literal, .value = try radix(text, base), .text = text, .base = base });
    }

    fn prove(self: *Parser, index: usize, root: usize) Invalid!i512 {
        const n = self.nodes[index];
        if (self.proved[index]) return n.value;
        if (n.op == .literal) return n.value;
        const a = try self.prove(n.l, root);
        const b = switch (n.op) {
            .neg, .digits, .isqrt, .factorial, .digit_sum => 0,
            else => try self.prove(n.r, root),
        };
        const result = switch (n.op) {
            .literal => unreachable,
            .neg => -a,
            .add => a + b,
            .sub => a - b,
            .mul => a * b,
            .floor, .mod, .divmod => blk: {
                if (b == 0 or (n.op == .divmod and index != root)) return error.Unsupported;
                if (n.op == .mod) break :blk @mod(a, b);
                const q = @divFloor(a, b);
                if (n.op == .divmod) _ = try bound(@mod(a, b));
                break :blk q;
            },
            .pow => try power(a, b),
            .powmod => blk: {
                const m = try self.prove(n.k, root);
                if (m == 0 or b < 0) return error.Unsupported;
                var exponent = b;
                var base = @mod(a, m);
                var result = @mod(@as(i512, 1), m);
                while (exponent != 0) {
                    if (@mod(exponent, 2) != 0) result = @mod(result * base, m);
                    exponent = @divTrunc(exponent, 2);
                    if (exponent != 0) base = @mod(base * base, m);
                }
                break :blk result;
            },
            .gcd => blk: {
                var x = if (a < 0) -a else a;
                var y = if (b < 0) -b else b;
                while (y != 0) {
                    const r = @mod(x, y);
                    x = y;
                    y = r;
                }
                break :blk x;
            },
            .isqrt => blk: {
                if (a < 0) return error.Unsupported;
                var x = a;
                var y = @divTrunc(x + 1, 2);
                while (y < x) {
                    x = y;
                    y = @divTrunc(x + @divTrunc(a, x), 2);
                }
                break :blk x;
            },
            .factorial => blk: {
                if (a < 0 or a > 48) return error.Unsupported;
                var result: i512 = 1;
                var x: i512 = 2;
                while (x <= a) : (x += 1) result = try bound(result * x);
                break :blk result;
            },
            .range_product => blk: {
                const step = try self.prove(n.k, root);
                if (step == 0) return error.Unsupported;
                var x = a;
                var count: usize = 0;
                var result: i512 = 1;
                while (if (step > 0) x < b else x > b) : (x += step) {
                    if (count == 10000) return error.Unsupported;
                    count += 1;
                    result = try bound(result * try bound(x));
                }
                break :blk result;
            },
            .digit_sum => blk: {
                if (a < 0) return error.Unsupported;
                var x = a;
                var result: i512 = 0;
                while (x != 0) : (x = @divTrunc(x, 10)) result += @mod(x, 10);
                break :blk result;
            },
            .digits => blk: {
                var magnitude = if (a < 0) -a else a;
                var count: i512 = if (a < 0) 2 else 1;
                while (magnitude >= 10) : (magnitude = @divTrunc(magnitude, 10)) count += 1;
                break :blk count;
            },
        };
        self.nodes[index].value = try bound(result);
        self.proved[index] = true;
        return result;
    }
};

fn digit(c: u8) u8 {
    return if (c >= '0' and c <= '9') c - '0' else if (c >= 'a' and c <= 'f') c - 'a' + 10 else if (c >= 'A' and c <= 'F') c - 'A' + 10 else 255;
}

fn radix(text: []const u8, base: u8) Invalid!i512 {
    if (text.len == 0) return error.Unsupported;
    var n: i512 = 0;
    for (text) |c| {
        if (digit(c) >= base) return error.Unsupported;
        n = try bound(n * base + digit(c));
    }
    return n;
}

fn bound(n: i512) Invalid!i512 {
    if (n <= -limit or n >= limit) return error.Unsupported;
    return n;
}

fn power(a: i512, b: i512) Invalid!i512 {
    if (b < 0) return error.Unsupported;
    var exponent = b;
    var base = a;
    var result: i512 = 1;
    while (exponent != 0) {
        if (@mod(exponent, 2) != 0) result = try bound(result * base);
        exponent = @divTrunc(exponent, 2);
        if (exponent != 0) base = try bound(base * base);
    }
    return result;
}
