const std = @import("std");
const api = @import("runner_api.zig");
const calc = @import("runner_frontend_calc.zig");
const program = @import("runner_frontend_program.zig");

pub const Frontend = api.Frontend;
pub const Cell = api.Cell;
pub const Translation = api.Translation;
pub const Context = struct {
    io: std.Io,
    compiler: program.Config = .{ .clang = "/usr/bin/clang" },
    exact_python: bool = false,

    pub fn frontend(self: *Context) Frontend {
        return .{ .ctx = self, .translate = translate };
    }
};

pub fn frontend() Frontend {
    return .{ .translate = translate };
}

// The integrator sets exact_python only from TF_DSV41_RUNNER_EXACT, which defaults off.
fn language(lang: []const u8) enum { python, exact, c, unsupported } {
    if (std.mem.eql(u8, lang, "python") or std.mem.eql(u8, lang, "python3") or std.mem.eql(u8, lang, "py"))
        return .python;
    if (std.mem.eql(u8, lang, "calc") or std.mem.eql(u8, lang, "exact")) return .exact;
    if (std.mem.eql(u8, lang, "c")) return .c;
    return .unsupported;
}

fn translate(raw: ?*anyopaque, a: std.mem.Allocator, cell: Cell) std.mem.Allocator.Error!Translation {
    const context: ?*Context = if (raw) |ptr| @ptrCast(@alignCast(ptr)) else null;
    const lang = language(cell.lang);
    if (lang == .unsupported) return .{ .abstain = .language };
    if (lang == .exact and (context == null or !context.?.exact_python)) return .{ .abstain = .unsupported };
    if (cell.prior.len != 0) return .{ .abstain = .unsupported };
    if (cell.code.len > 4096) return .{ .abstain = .too_long };
    var normalized: ?[]u8 = null;
    defer if (normalized) |bytes| a.free(bytes);
    var input = cell.code;
    if (cell.form == .inline_expr and lang == .python) {
        normalized = try normalizeInline(a, input);
        input = normalized.?;
    }
    if (lang == .python or lang == .exact) {
        // Python true division and floats differ from firmware exact arithmetic in both value and print format.
        const exact_only = std.mem.indexOfScalar(u8, input, '.') != null or
            containsDivision(input) or std.mem.indexOf(u8, input, "gcd") != null;
        const allow_calc = !exact_only or lang == .exact or (context != null and context.?.exact_python);
        if (allow_calc) {
            if (try calc.lower(a, input)) |bytes| return .{ .program = .{ .kind = .calc, .bytes = bytes } };
        }
    }
    if (lang == .exact or cell.form == .inline_expr) return .{ .abstain = .unsupported };
    var source: []const u8 = cell.code;
    var owned: ?[]u8 = null;
    defer if (owned) |bytes| a.free(bytes);
    if (lang == .python) {
        const body = try program.lowerPython(a, cell.code) orelse return .{ .abstain = .unsupported };
        defer a.free(body);
        owned = try program.wrapBody(a, body);
        source = owned.?;
    } else if (!program.compiler.hasMain(source)) {
        owned = try program.wrapBody(a, source);
        source = owned.?;
    }
    const bytes = if (context) |ctx|
        try program.compile(a, ctx.io, source, ctx.compiler)
    else blk: {
        var threaded = std.Io.Threaded.init(a, .{});
        defer threaded.deinit();
        break :blk try program.compile(a, threaded.io(), source, .{ .clang = "/usr/bin/clang" });
    };
    if (bytes) |elf| return .{ .program = .{ .kind = .elf, .bytes = elf } };
    return .{ .abstain = .unsupported };
}

fn containsDivision(code: []const u8) bool {
    var i: usize = 0;
    while (i < code.len) : (i += 1) {
        if (code[i] == '/') {
            if (i + 1 == code.len or code[i + 1] != '/') return true;
            i += 1;
        }
    }
    return false;
}

// Inline math has typographic operators; executable Python keeps its original syntax.
fn normalizeInline(a: std.mem.Allocator, source: []const u8) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var pos: usize = 0;
    while (pos < source.len) {
        if (std.mem.startsWith(u8, source[pos..], "×")) {
            try out.append(a, '*');
            pos += "×".len;
        } else if (std.mem.startsWith(u8, source[pos..], "−")) {
            try out.append(a, '-');
            pos += "−".len;
        } else {
            try out.append(a, source[pos]);
            pos += 1;
        }
    }
    return out.toOwnedSlice(a);
}
