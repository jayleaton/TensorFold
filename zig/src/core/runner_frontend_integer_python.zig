const std = @import("std");
const ast = @import("runner_frontend_integer_ast.zig");
const integer = @import("runner_frontend_integer.zig");
const compiler = @import("runner_frontend_program_compile.zig");
const cost = @import("runner_frontend_integer_cost.zig");
const Allocator = std.mem.Allocator;
const Failure = Allocator.Error || ast.Invalid;

// Immutable bindings retain expression nodes; proof values never replace the requested runtime arithmetic.
pub fn lower(a: Allocator, code: []const u8) Allocator.Error!?[]u8 {
    const result = try lowerWithCost(a, code, 0) orelse return null;
    return result.source;
}

pub fn lowerWithCost(a: Allocator, code: []const u8, budget: usize) Allocator.Error!?integer.Lowered {
    return checked(a, code, budget) catch |err| switch (err) {
        error.Unsupported => null,
        error.OutOfMemory => error.OutOfMemory,
    };
}

fn checked(a: Allocator, code: []const u8, budget: usize) Failure!integer.Lowered {
    if (code.len == 0 or code.len > 4096 or std.mem.indexOfScalar(u8, code, '\r') != null)
        return error.Unsupported;
    var parser: ast.Parser = .{ .source = "", .require_math_import = true };
    var lines = std.mem.splitScalar(u8, code, '\n');
    var output: ?usize = null;
    var started = false;
    while (lines.next()) |raw| {
        const comment = std.mem.indexOfScalar(u8, raw, '#') orelse raw.len;
        var line = std.mem.trimEnd(u8, raw[0..comment], " \t");
        if (line.len == 0) continue;
        if (output != null or line[0] == ' ' or line[0] == '\t') return error.Unsupported;
        if (std.mem.startsWith(u8, line, ">>> ")) line = line[4..];
        if (std.mem.eql(u8, line, "import math")) {
            if (started) return error.Unsupported;
            parser.math_module = true;
            continue;
        }
        if (std.mem.startsWith(u8, line, "from math import ")) {
            if (started) return error.Unsupported;
            var names = std.mem.splitScalar(u8, line[17..], ',');
            while (names.next()) |part| {
                const name = std.mem.trim(u8, part, " ");
                parser.math_imports |= if (std.mem.eql(u8, name, "gcd")) @as(u8, 1) else if (std.mem.eql(u8, name, "isqrt"))
                    @as(u8, 2)
                else if (std.mem.eql(u8, name, "factorial"))
                    @as(u8, 4)
                else if (std.mem.eql(u8, name, "prod"))
                    @as(u8, 8)
                else
                    return error.Unsupported;
            }
            continue;
        }
        started = true;
        var target: ?[]const u8 = null;
        var printed = false;
        if (std.mem.indexOfScalar(u8, line, '=')) |equal| {
            const name = std.mem.trim(u8, line[0..equal], " ");
            if (!validBinding(name)) return error.Unsupported;
            target = name;
            line = std.mem.trim(u8, line[equal + 1 ..], " ");
        } else if (std.mem.startsWith(u8, line, "print")) {
            printed = true;
            line = std.mem.trimStart(u8, line[5..], " ");
            if (line.len < 3 or line[0] != '(' or line[line.len - 1] != ')') return error.Unsupported;
            line = line[1 .. line.len - 1];
        }
        if (target == null and !printed and
            (parser.binding_count != 0 or parser.math_module or parser.math_imports != 0)) return error.Unsupported;
        parser.source = line;
        parser.pos = 0;
        const root = try parser.parse();
        if (target) |name| {
            if (parser.nodes[root].op == .divmod) return error.Unsupported;
            try parser.bind(name, root);
        } else output = root;
    }
    const root = output orelse return error.Unsupported;
    var steps: ?usize = null;
    if (budget != 0) steps = cost.estimate(&parser, root, budget) orelse return error.Unsupported;
    return .{ .source = try integer.lowerParser(a, &parser, root), .estimated_steps = steps };
}

pub fn compile(a: Allocator, io: std.Io, code: []const u8, config: compiler.Config) Allocator.Error!?[]u8 {
    const result = try lowerWithCost(a, code, config.step_budget) orelse return null;
    defer a.free(result.source);
    var trusted = config;
    trusted.estimated_steps = result.estimated_steps;
    if (trusted.optimize != .O0) trusted.optimize = .O2;
    trusted.opaque_inputs = true;
    return compiler.compile(a, io, result.source, trusted);
}

fn validBinding(name: []const u8) bool {
    if (name.len == 0 or !(std.ascii.isAlphabetic(name[0]) or name[0] == '_')) return false;
    for (name) |c| if (!(std.ascii.isAlphanumeric(c) or c == '_')) return false;
    for ([_][]const u8{
        "print",  "pow",    "divmod",    "int",       "str",          "len",     "sum",    "map",    "range", "all",   "abs",   "math",
        "gcd",    "isqrt",  "factorial", "prod",      "True",         "False",   "None",   "and",    "or",    "not",   "if",    "else",
        "for",    "while",  "in",        "is",        "def",          "return",  "import", "from",   "as",    "class", "del",   "yield",
        "lambda", "global", "nonlocal",  "try",       "except",       "finally", "with",   "assert", "raise", "pass",  "break", "continue",
        "async",  "await",  "elif",      "__debug__", "__builtins__",
    }) |reserved| if (std.mem.eql(u8, name, reserved)) return false;
    return true;
}

test "wide assignment bindings preserve requests and validate unused expressions" {
    const a = std.testing.allocator;
    for ([_][]const u8{
        "a = 314159\nb = 271828\nprint(a * b)",
        "x = 10**40\ny = pow(x,2,97)\nprint(y)",
        "a = 7\na = a+2\nprint(a*a)",
        "import math\nn = 12345678901234567890\nprint(math.isqrt(n))",
        "from math import gcd\na = 48\nb = 18\nprint(gcd(a,b))",
    }) |code| {
        const source = (try lower(a, code)).?;
        defer a.free(source);
        try std.testing.expect(std.mem.indexOf(u8, source, "tf_run(tf_nodes,") != null);
    }
    for ([_][]const u8{
        "x = 1//0\nprint(3)",                    "x = 10**62\nprint(3)", "x = x+1\nprint(x)",
        "print(math.gcd(48,18))",                "print(gcd(48,18))",    "from math import *\nprint(3)",
        "import math as m\nprint(m.gcd(48,18))", "pow = 2\nprint(pow)",  "a = divmod(7,3)\nprint(a)",
        "x = 2\nprint(x)\nprint(x)",             "import os\nprint(3)",  "x = 1.25\nprint(x)",
        "__builtins__ = 0\nprint(3)",            "x = 3\nx*2",
    }) |code| try std.testing.expectEqual(@as(?[]u8, null), try lower(a, code));
}
