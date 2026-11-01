const std = @import("std");

pub const Invalid = error{Unsupported};
pub const limit: i512 = std.math.pow(i512, 10, 62);
pub const Op = enum(u8) { literal, neg, add, sub, mul, floor, mod, pow, powmod, digits, divmod };
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
        if (self.take("divmod")) {
            try self.need("(");
            const a = try self.expr(depth + 1);
            try self.need(",");
            const b = try self.expr(depth + 1);
            try self.need(")");
            return self.add(.{ .op = .divmod, .l = a, .r = b });
        }
        if (self.take("pow")) {
            try self.need("(");
            const a = try self.expr(depth + 1);
            try self.need(",");
            const b = try self.expr(depth + 1);
            if (!self.take(",")) return error.Unsupported;
            const m = try self.expr(depth + 1);
            try self.need(")");
            return self.add(.{ .op = .powmod, .l = a, .r = b, .k = m });
        }
        if (self.take("len")) {
            try self.need("(");
            try self.need("str");
            try self.need("(");
            const n = try self.expr(depth + 1);
            try self.need(")");
            try self.need(")");
            return self.add(.{ .op = .digits, .l = n });
        }
        if (self.take("int")) return self.convert();
        return self.literal();
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
            .neg, .digits => 0,
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
