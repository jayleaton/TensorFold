const std = @import("std");
const ast = @import("runner_frontend_integer_ast.zig");
const compiler = @import("runner_frontend_program_compile.zig");

pub fn ambiguousPercent(a: std.mem.Allocator, source: []const u8) std.mem.Allocator.Error!bool {
    if (std.mem.indexOfScalar(u8, source, '%') == null) return false;
    var parser = ast.Parser{ .source = source };
    _ = parser.parse() catch return false;
    const percentage = try @import("runner_frontend_inline.zig").lower(a, source, true) orelse return false;
    defer a.free(percentage);
    return true;
}

// Proof values decide admission; emitted nodes retain the requested arithmetic and original literal values.
pub fn lower(a: std.mem.Allocator, source: []const u8) std.mem.Allocator.Error!?[]u8 {
    var code = std.mem.trim(u8, source, " \t\r\n");
    if (std.mem.startsWith(u8, code, ">>> ")) code = std.mem.trim(u8, code[4..], " \t\r\n");
    if (std.mem.startsWith(u8, code, "print")) {
        const rest = std.mem.trimStart(u8, code[5..], " \t");
        if (rest.len < 3 or rest[0] != '(' or rest[rest.len - 1] != ')') return null;
        code = rest[1 .. rest.len - 1];
    }
    if (code.len == 0 or code.len > 4096) return null;
    var parser = ast.Parser{ .source = code };
    const root = parser.parse() catch return null;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, @embedFile("runner_frontend_integer_runtime.c"));
    try out.appendSlice(a, "\nstatic const struct tf_node tf_nodes[] = {\n");
    var emitter = Emitter{ .a = a, .parser = &parser, .out = &out };
    _ = try emitter.node(root);
    try out.print(a, "}};\nint main(void) {{ return tf_run(tf_nodes, {d}); }}\n", .{emitter.count});
    return try out.toOwnedSlice(a);
}

pub fn compile(a: std.mem.Allocator, io: std.Io, code: []const u8, config: compiler.Config) std.mem.Allocator.Error!?[]u8 {
    const source = try lower(a, code) orelse return null;
    defer a.free(source);
    var runtime_config = config;
    runtime_config.optimize = true;
    return compiler.compile(a, io, source, runtime_config);
}

const Emitter = struct {
    a: std.mem.Allocator,
    parser: *const ast.Parser,
    out: *std.ArrayList(u8),
    count: usize = 0,
    mapped: [64]?usize = @splat(null),

    fn node(self: *Emitter, index: usize) std.mem.Allocator.Error!usize {
        if (self.mapped[index]) |mapped| return mapped;
        const n = self.parser.nodes[index];
        var l: usize = 0;
        var r: usize = 0;
        var k: usize = 0;
        if (n.op != .literal) l = try self.node(n.l);
        if (n.op != .literal and n.op != .neg and n.op != .digits) r = try self.node(n.r);
        if (n.op == .powmod) k = try self.node(n.k);
        if (n.op == .literal) {
            try self.out.print(self.a, "{{0,0,0,{d},\"{s}{s}\"}},\n", .{
                n.base, if (n.value < 0) "-" else "", n.text,
            });
        } else {
            try self.out.print(self.a, "{{{d},{d},{d},{d},0}},\n", .{ @backingInt(n.op), l, r, k });
        }
        const result = self.count;
        self.count += 1;
        self.mapped[index] = result;
        return result;
    }
};
