//! Replays captured Python extension calls through the Zig launches: `ext`.`func` with the pybind call's own
//! arguments (tensors bound to device buffers holding the captured bytes) re-issued on a stream, so the M1 block
//! replay compares every .cu kernel's outputs with the Python engine's on the same inputs. Each binding mirrors the
//! extension's C++ entry (argument order, the values it derives from tensor shapes / strides / dtypes, its checks).

const std = @import("std");
const cuda = @import("cuda");
const kernels = @import("kernels.zig");
const exl3 = @import("kernels_exl3.zig");
const dense = @import("kernels_dense.zig");

const Kernels = kernels.Kernels;

pub const DType = enum { f16, bf16, f32, f64, i8, u8, i16, i32, i64, bool };
pub const Tensor = struct { ptr: u64, dtype: DType, shape: []const i64, stride: []const i64 }; // element strides, ptr already at storage_offset
pub const Arg = union(enum) { tensor: Tensor, int: i64, float: f64, boolean: bool, none, list: []const Arg };
pub const Kw = struct { name: []const u8, value: Arg };

/// Device bytes a call may use beyond its arguments (cleared per call by the caller).
pub const Scratch = struct {
    ptr: u64,
    len: usize,
    used: usize = 0,

    pub fn take(s: *Scratch, n: usize) !u64 {
        const at = std.mem.alignForward(usize, s.used, 256);
        if (at + n > s.len) return error.ScratchTooSmall;
        s.used = at + n;
        return s.ptr + at;
    }
};

pub const Error = error{ MissingArgument, WrongArgument, Unsupported };

/// The call's arguments by pybind position, or by keyword when passed so (`names` lists the binding's parameters).
const Call = struct {
    args: []const Arg,
    kwargs: []const Kw,
    names: []const []const u8,

    fn get(c: Call, i: usize) Error!Arg {
        if (i < c.args.len) return c.args[i];
        for (c.kwargs) |kw| if (std.mem.eql(u8, kw.name, c.names[i])) return kw.value;
        return error.MissingArgument;
    }

    fn opt(c: Call, i: usize) Error!?Arg {
        const a = c.get(i) catch |e| switch (e) {
            error.MissingArgument => return null,
            else => return e,
        };
        return if (a == .none) null else a;
    }

    fn tensor(c: Call, i: usize) Error!Tensor {
        return switch (try c.get(i)) {
            .tensor => |t| t,
            else => error.WrongArgument,
        };
    }

    /// An optional tensor's address (None or an empty tensor: 0, as the bindings pass nullptr).
    fn ptrOrNull(c: Call, i: usize) Error!u64 {
        const a = (try c.opt(i)) orelse return 0;
        return switch (a) {
            .tensor => |t| if (numel(t) == 0) 0 else t.ptr,
            else => error.WrongArgument,
        };
    }

    fn ptr(c: Call, i: usize) Error!u64 {
        return (try c.tensor(i)).ptr;
    }

    fn int(c: Call, i: usize) Error!i64 {
        return switch (try c.get(i)) {
            .int => |v| v,
            .boolean => |b| @intFromBool(b),
            else => error.WrongArgument,
        };
    }

    fn usz(c: Call, i: usize) Error!usize {
        const v = try c.int(i);
        if (v < 0) return error.WrongArgument;
        return @intCast(v);
    }

    fn u(c: Call, i: usize) Error!u32 {
        return @intCast(try c.usz(i));
    }

    fn float(c: Call, i: usize) Error!f64 {
        return switch (try c.get(i)) {
            .float => |v| v,
            .int => |v| @floatFromInt(v),
            else => error.WrongArgument,
        };
    }

    fn boolean(c: Call, i: usize, default: bool) Error!bool {
        const a = (try c.opt(i)) orelse return default;
        return switch (a) {
            .boolean => |b| b,
            .int => |v| v != 0,
            else => error.WrongArgument,
        };
    }

    fn list(c: Call, i: usize) Error![]const Arg {
        return switch (try c.get(i)) {
            .list => |l| l,
            else => error.WrongArgument,
        };
    }
};

fn numel(t: Tensor) i64 {
    var n: i64 = 1;
    for (t.shape) |d| n *= d;
    return n;
}

fn dim(t: Tensor, i: usize) Error!usize {
    if (i >= t.shape.len) return error.WrongArgument;
    return @intCast(t.shape[i]);
}

fn ioType(t: DType) Error!dense.DType {
    return switch (t) {
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        else => error.WrongArgument,
    };
}

fn input(t: DType) Error!exl3.Input {
    return switch (t) {
        .bf16 => .bf16,
        .f16 => .f16,
        else => error.WrongArgument,
    };
}

fn elem(t: DType) usize {
    return switch (t) {
        .i8, .u8, .bool => 1,
        .f16, .bf16, .i16 => 2,
        .f32, .i32 => 4,
        .f64, .i64 => 8,
    };
}

fn is(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// One captured extension call re-issued on `s`; false when there is no binding for it (reported as skipped).
pub fn call(k: *const Kernels, s: cuda.Stream, ext: []const u8, func: []const u8, args: []const Arg, kwargs: []const Kw, scratch: *Scratch) !bool {
    _ = scratch;
    if (is(ext, "tensorfold_exl3_linear_v4")) return linearExt(k.linears(s), func, args, kwargs);
    if (is(ext, "tf_dsv41_x3seg_v3")) return segExt(k.linears(s), func, args, kwargs, false);
    if (is(ext, "tf_dsv41_dense3_v1")) return segExt(k.linears(s), func, args, kwargs, true);
    if (is(ext, "tensorfold_exl3_experts_v1")) return expertsExt(k.experts(s), func, args, kwargs);
    if (is(ext, "tf_dsv41_x3ld_v1") or is(ext, "tf_dsv41_x3pf_v1")) return groupedExt(k.experts(s), ext, func, args, kwargs);
    if (is(ext, "tf_dsv41_x3gm_v1")) return x3gmExt(k.experts(s), func, args, kwargs);
    return false;
}

// tensorfold_exl3_linear_v4 (linear.cpp)
fn linearExt(o: dense.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    if (is(func, "rot_in")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "suh", "xh" } };
        const x = try c.tensor(0);
        try o.rotIn(x.ptr, try ioType(x.dtype), try c.ptr(1), try c.ptr(2), try dim(x, 0), try dim(x, 1));
        return true;
    }
    if (is(func, "linear") or is(func, "linear_skip")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "xh", "T", "stride_k", "stride_nb", "svh", "bias", "y", "Z", "counters", "K2", "cb", "SK", "WK", "skip" } };
        if (try c.int(10) != 2) return false; // mul1 only is built (every DeepSeek-V4.1 matrix)
        const xh = try c.tensor(0);
        const y = try c.tensor(6);
        const skip: u64 = if (is(func, "linear_skip")) try c.ptrOrNull(13) else 0;
        try o.linear(xh.ptr, try c.ptr(1), try c.int(2), try c.int(3), try c.ptr(4), try c.ptrOrNull(5), y.ptr, try ioType(y.dtype), try c.ptrOrNull(7), try c.ptr(8), try dim(xh, 0), try dim(xh, 1), try dim(y, 1), try c.u(9), try c.usz(11), try c.usz(12), skip);
        return true;
    }
    if (is(func, "unpack")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "T", "W", "stride_k", "stride_nb", "K2", "cb" } };
        if (try c.int(5) != 2) return false;
        const w = try c.tensor(1);
        try o.unpack(try c.ptr(0), w.ptr, try dim(w, 0), try dim(w, 1), try c.int(2), try c.int(3), try c.u(4));
        return true;
    }
    return false;
}

// tf_dsv41_x3seg_v3 / tf_dsv41_dense3_v1 (x3seg.cpp, dense3.cpp): the tables built as dsv41_*_linear_cuda builds them
fn segExt(o: dense.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw, lanes: bool) !bool {
    if (is(func, "rot_in") and !lanes) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "xs", "suh", "xh", "xh_off", "pdl" } };
        const xs = try c.list(0);
        const suh = try c.list(1);
        const offs = try c.list(3);
        const xh = try c.ptr(2);
        if (xs.len < 1 or xs.len > dense.max_seg or suh.len != xs.len or offs.len != xs.len) return error.WrongArgument;
        var tab = std.mem.zeroes(dense.RotTable);
        tab.n = @intCast(xs.len);
        var M: usize = 0;
        for (xs, suh, offs, 0..) |xa, sa, oa, i| {
            if (xa != .tensor or sa != .tensor or oa != .int) return error.WrongArgument;
            const x = xa.tensor;
            if (i == 0) M = try dim(x, 0);
            tab.s[i] = .{ .x = x.ptr, .suh = sa.tensor.ptr, .xh = xh + 2 * @as(u64, @intCast(oa.int)), .ld = x.stride[0], .K = @intCast(try dim(x, 1)), .x_dtype = @intFromEnum(try ioType(x.dtype)) };
        }
        try o.segRotIn(tab, M, try c.boolean(4, false));
        return true;
    }
    if (is(func, "linear")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "xh", "T", "svh", "ys", "Z", "counters", "meta", "programs", "K2", "WK", "minb", "pdl" } };
        const T = try c.list(1);
        const svh = try c.list(2);
        const ys = try c.list(3);
        const meta = try c.list(6);
        const n = T.len;
        if (n < 1 or n > dense.max_seg or svh.len != n or ys.len != n or meta.len != 11 * n) return error.WrongArgument;
        const xh = try c.ptr(0);
        const z = try c.ptrOrNull(4);
        const counters = try c.ptr(5);
        if (ys[0] != .tensor) return error.WrongArgument;
        const y0 = ys[0].tensor;
        const M = try dim(y0, 0);
        const y_dtype = try ioType(y0.dtype);
        var tab = std.mem.zeroes(dense.LinTable);
        tab.n = @intCast(n);
        for (0..n) |i| {
            var m: [11]i64 = undefined;
            for (&m, meta[11 * i ..][0..11]) |*v, a| v.* = switch (a) {
                .int => |x| x,
                else => return error.WrongArgument,
            };
            if (T[i] != .tensor or svh[i] != .tensor or ys[i] != .tensor) return error.WrongArgument;
            const y = ys[i].tensor;
            tab.s[i] = .{
                .xh = xh + 2 * @as(u64, @intCast(m[6])),
                .T = T[i].tensor.ptr,
                .stride_k = m[4],
                .stride_nb = m[5],
                .svh = svh[i].tensor.ptr,
                .y = y.ptr + @as(u64, @intCast(m[9])) * elem(y.dtype),
                .Z = if (m[2] > 1) z + 4 * @as(u64, @intCast(m[7])) else 0,
                .counters = counters + 4 * @as(u64, @intCast(m[8])),
                .K = @intCast(m[0]),
                .N = @intCast(m[1]),
                .SK = @intCast(m[2]),
                .wk = @intCast(m[3]),
                .p0 = @intCast(m[10]),
                .ld = @intCast(y.stride[0]),
            };
        }
        const programs = try c.usz(7);
        const k2 = try c.u(8);
        const wk = try c.usz(9);
        const minb = try c.u(10);
        const pdl = try c.boolean(11, false);
        if (lanes) try o.lanesLinearRaw(tab, y_dtype, M, programs, k2, wk, minb, pdl) else try o.segLinearRaw(tab, y_dtype, M, programs, k2, wk, minb, pdl);
        return true;
    }
    if (is(func, "relayout") and lanes) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "src", "dst", "K2", "K", "lanes" } };
        const src = try c.tensor(0);
        try o.relayout(src.ptr, try c.ptr(1), @intCast(numel(src)), try c.u(2), try c.usz(3), try c.boolean(4, true));
        return true;
    }
    if (is(func, "unpack") and lanes) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "T", "W", "stride_k", "stride_nb", "K2" } };
        const w = try c.tensor(1);
        try o.lanesUnpack(try c.ptr(0), w.ptr, try dim(w, 0), try dim(w, 1), try c.int(2), try c.int(3), try c.u(4));
        return true;
    }
    return false;
}

const grouped_names = [_][]const u8{ "X0", "X1", "TP0", "TP1", "B0", "B1", "uids", "ucount", "members", "Z", "mats", "K", "N", "P", "SK", "slots", "cb" };

fn groupedArgs(c: Call, lo_i: usize) !exl3.Grouped {
    const uids = try c.tensor(6);
    const members = try c.tensor(8);
    return .{
        .x0 = try c.ptr(0),
        .x1 = try c.ptr(1),
        .tp0 = try c.ptr(2),
        .tp1 = try c.ptr(3),
        .k2_0 = try c.ptr(4),
        .k2_1 = try c.ptr(5),
        .uids = uids.ptr,
        .ucount = try c.ptr(7),
        .members = members.ptr,
        .z = try c.ptr(9),
        .mats = try c.usz(10),
        .K = try c.usz(11),
        .N = try c.usz(12),
        .P = try c.usz(13),
        .SK = try c.usz(14),
        .slots = try c.usz(15),
        .maxm = try dim(members, 1),
        .nexp = try dim(uids, 0),
        .lo = try c.u(lo_i),
        .hi = try c.u(lo_i + 1),
    };
}

// tensorfold_exl3_experts_v1 (experts.cpp)
fn expertsExt(o: exl3.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    if (is(func, "grouped")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &(grouped_names ++ [_][]const u8{ "nt", "warps", "pf", "lo", "hi" }) };
        if (try c.int(16) != 2) return false; // cb 2 (mul1) instances only
        try o.grouped(try groupedArgs(c, 20), try c.u(17), try c.u(18), try c.u(19));
        return true;
    }
    if (is(func, "group")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "pick", "uids", "ucount", "members", "R", "slots", "E" } };
        const members = try c.tensor(3);
        try o.group(try c.ptr(0), try c.ptr(1), try c.ptr(2), members.ptr, try c.usz(4), try c.usz(5), try c.usz(6), try dim(members, 1));
        return true;
    }
    if (is(func, "rot_in")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "x_stride", "pick", "suh0", "suh1", "out0", "out1", "rows", "K", "slots", "E" } };
        const x = try c.tensor(0);
        try o.rotIn(try input(x.dtype), x.ptr, try c.usz(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.ptr(6), try c.usz(7), try c.usz(8), try c.usz(9), try c.usz(10));
        return true;
    }
    if (is(func, "gateup_epilogue")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "Z", "pick", "svh_g", "svh_u", "suh_d", "xd", "rows", "P", "N", "SK", "slots", "E", "limit", "act_mode" } };
        try o.gateupEpilogue(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.usz(6), try c.usz(7), try c.usz(8), try c.usz(9), try c.usz(10), try c.usz(11), @floatCast(try c.float(12)), @intCast(try c.int(13)));
        return true;
    }
    if (is(func, "down_epilogue")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "Z", "pick", "svh_d", "y", "rows", "P", "D", "SK", "slots", "E" } };
        try o.downEpilogue(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.ptr(3), try c.usz(4), try c.usz(5), try c.usz(6), try c.usz(7), try c.usz(8), try c.usz(9));
        return true;
    }
    if (is(func, "combine")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "y", "wts", "out", "rows", "D", "slots" } };
        try o.combine(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.usz(3), try c.usz(4), try c.usz(5));
        return true;
    }
    if (is(func, "down_combine")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "Z", "pick", "svh_d", "y", "wts", "out", "rows", "P", "D", "SK", "slots", "E" } };
        try o.downCombine(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.usz(6), try c.usz(7), try c.usz(8), try c.usz(9), try c.usz(10), try c.usz(11));
        return true;
    }
    if (is(func, "dequant")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "T", "out", "k2", "cb" } };
        if (try c.int(3) != 2) return false;
        const t = try c.tensor(0);
        try o.dequant(t.ptr, try c.ptr(1), 16 * try dim(t, 0), 16 * try dim(t, 1), try c.u(2));
        return true;
    }
    return false;
}

// tf_dsv41_x3ld_v1.grouped (x3ld.cpp: ..., cb, nt, pd, probe, lo, hi, pdl) and tf_dsv41_x3pf_v1.grouped (..., cb, nt, mtl, lo, hi)
fn groupedExt(o: exl3.Ops, ext: []const u8, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    if (!is(func, "grouped")) return false;
    if (is(ext, "tf_dsv41_x3ld_v1")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &(grouped_names ++ [_][]const u8{ "nt", "pd", "probe", "lo", "hi", "pdl" }) };
        if (try c.int(16) != 2) return error.Unsupported; // the binding refuses it too
        try o.x3ld(try groupedArgs(c, 20), try c.u(17), try c.u(18), try c.u(19), try c.boolean(22, false));
        return true;
    }
    const c: Call = .{ .args = args, .kwargs = kwargs, .names = &(grouped_names ++ [_][]const u8{ "nt", "mtl", "lo", "hi" }) };
    if (try c.int(16) != 2) return error.Unsupported;
    try o.x3pf(try groupedArgs(c, 19), try c.u(17), try c.u(18));
    return true;
}

// tf_dsv41_x3gm_v1 (x3gm.cpp)
fn x3gmExt(o: exl3.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    if (is(func, "gateup")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "xg", "xu", "tg", "tu", "order", "pe", "poff", "pcnt", "npass", "ticket", "svh_g", "svh_u", "suh_d", "xd", "K", "N", "k2", "shx", "cfg", "use_ticket", "limit", "tables" } };
        try o.gmGateup(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.ptr(6), try c.ptr(7), try c.ptr(8), try c.ptr(9), try c.ptr(10), try c.ptr(11), try c.ptr(12), try c.ptr(13), try c.usz(14), try c.usz(15), try c.u(16), try c.boolean(17, false), try c.usz(18), try c.boolean(19, true), @floatCast(try c.float(20)), try c.boolean(21, false));
        return true;
    }
    if (is(func, "down")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "xd", "td", "order", "pe", "poff", "pcnt", "npass", "ticket", "svh_d", "y", "K", "N", "k2", "cfg", "use_ticket", "tables" } };
        try o.gmDown(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.ptr(6), try c.ptr(7), try c.ptr(8), try c.ptr(9), try c.usz(10), try c.usz(11), try c.u(12), try c.usz(13), try c.boolean(14, true), try c.boolean(15, false));
        return true;
    }
    if (is(func, "rot")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "x_stride", "pick", "suh0", "suh1", "out0", "out1", "K", "slots", "pairs", "mats" } };
        const x = try c.tensor(0);
        try o.gmRot(try input(x.dtype), x.ptr, try c.usz(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.ptr(6), try c.usz(7), try c.usz(8), try c.usz(9), try c.usz(10));
        return true;
    }
    if (is(func, "dequant")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "t", "out", "k2" } };
        const out = try c.tensor(1);
        try o.gmDequant(try c.ptr(0), out.ptr, try dim(out, 0), try dim(out, 1), try c.u(2));
        return true;
    }
    return false;
}

test "arguments by position and keyword" {
    const shape = [_]i64{ 2, 3 };
    const t: Tensor = .{ .ptr = 64, .dtype = .bf16, .shape = &shape, .stride = &.{ 3, 1 } };
    const c: Call = .{ .args = &.{ .{ .tensor = t }, .{ .int = 7 } }, .kwargs = &.{.{ .name = "pdl", .value = .{ .boolean = true } }}, .names = &.{ "x", "n", "z", "pdl" } };
    try std.testing.expectEqual(@as(u64, 64), try c.ptr(0));
    try std.testing.expectEqual(@as(usize, 7), try c.usz(1));
    try std.testing.expectEqual(@as(u64, 0), try c.ptrOrNull(2));
    try std.testing.expect(try c.boolean(3, false));
    try std.testing.expectError(error.WrongArgument, c.tensor(1));
}
