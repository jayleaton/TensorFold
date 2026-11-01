const std = @import("std");
const api = @import("runner_api.zig");
const calc = @import("runner_frontend_calc.zig");
const program = @import("runner_frontend_program.zig");
const inline_math = @import("runner_frontend_inline.zig");

pub const Frontend = api.Frontend;
pub const Cell = api.Cell;
pub const Translation = api.Translation;
pub const Context = struct {
    io: std.Io,
    compiler: program.Config = .{ .clang = "/usr/bin/clang" },
    exact_tags: bool = false,
    exact_inline: bool = true,

    pub fn frontend(self: *Context) Frontend {
        return .{ .ctx = self, .translate = translate };
    }
};

pub fn frontend() Frontend {
    return .{ .translate = translate };
}

// T3 binds EXACT by form: inline defaults on; explicit calc/exact code tags default off.
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
    if (lang == .exact and (context == null or !context.?.exact_tags)) return .{ .abstain = .unsupported };
    if (cell.code.len > 4096) return .{ .abstain = .too_long };
    if (cell.form == .inline_expr) {
        if (lang != .python) return .{ .abstain = .unsupported };
        if (try inline_math.lower(a, cell.code, if (context) |ctx| ctx.exact_inline else true)) |bytes|
            return .{ .program = .{ .kind = .calc, .bytes = bytes } };
        return .{ .abstain = .unsupported };
    }
    if (cell.prior.len != 0) return .{ .abstain = .unsupported };
    if (lang == .python or lang == .exact) {
        const exact_only = std.mem.indexOfScalar(u8, cell.code, '.') != null or
            containsDivision(cell.code) or std.mem.indexOf(u8, cell.code, "gcd") != null;
        if (!exact_only or lang == .exact) {
            if (try calc.lower(a, cell.code)) |bytes| return .{ .program = .{ .kind = .calc, .bytes = bytes } };
        }
    }
    if (lang == .exact) return .{ .abstain = .unsupported };
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
