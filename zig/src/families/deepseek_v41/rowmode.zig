//! Row mode's program (Python rowmode.py / csa2 RowWin, G8): a decode window's calls (block.emit at R rows, its end at
//! the context bucket's) rewritten so one launch serves rows of any slots. Only the CSA2 steps change; everything else
//! (mHC, projections, experts, exchanges, head) is per row already.
//! - Every row's position, read slot, write slot and ratio-2 previous row come from the row table (rowtab.zig), bound as
//!   the persistent role "s.rows.tab": POS is the table's `pos` column (int64 [R]) in `_rope`, `_kv_store`,
//!   `_pool_norm`, `_index_k`, `_scores` and the CUDA top-k / attention; SL is `wslot` where a kernel writes
//!   (`_kv_store`, `_pool_norm`, `_index_k`) and `rslot` where it reads (`_scores`, attention); ROWS on.
//! - The slots' state is stacked (slots.zig): each layer's SWA ring "s.L<i>.swa.v / .s" is [S x ring] rows (slot s's
//!   ring at s x ring), the ratio-2 carries "s.L<i>.carry" [S, 2D] (slot s's row s), the page tables "s.kv.pt" /
//!   "s.kv.ct" [S, pts] (PTS = pts where a page table is passed, as compress._row_paging).
//! - `_pool_norm` at ratio 2 pools [the S carries | the rows] (OFF = S, PREV the table's column); at ratio 1 OFF = 0,
//!   PREV still the column (Python passes win.prev).
//! - Split KV's dense exchange (glue "kx_dense") reads the stacked split table with the int32 rslot copy (the glue's
//!   own args); a union (rows > 16) is refused: row windows stay within the dense exchange.
//! A call with row-mode knobs this file does not know is refused by name (error.UnknownRowsCall), so a new kernel in
//! block.zig cannot silently run one slot's arithmetic on every row.

const std = @import("std");
const calls = @import("calls.zig");
const rowtab = @import("rowtab.zig");
const Arg = calls.Arg;
const Tensor = calls.Tensor;

pub const Error = error{ UnknownRowsCall, NoPool, UnionInRows, BadCall };

pub const table_role = "s.rows.tab";

pub const Spec = struct {
    /// stacked slots
    slots: u32,
    /// the row table's columns hold this many rows
    rmax: u32,
    /// pages a slot's table holds (the stacked tables' row stride)
    pts: i64,
};

pub const Rewriter = struct {
    a: std.mem.Allocator,
    s: Spec,
    /// rows of the program (the bucket)
    R: i64,

    fn col(w: *const Rewriter, c: rowtab.Col) Arg {
        return .{ .t = .{ .role = .{ .buf = table_role }, .dt = .i64, .shape = w.dup(&.{w.R}), .stride = w.dup(&.{1}), .offset = @intCast(rowtab.colOffset(c, w.s.rmax)) } };
    }

    fn dup(w: *const Rewriter, xs: []const i64) []const i64 {
        return w.a.dupe(i64, xs) catch @panic("rowmode: out of memory");
    }

    /// A stacked role's tensor as the row program sees it: S x the rows (rings), [S, cols] (carries, page tables).
    fn stacked(w: *const Rewriter, t: Tensor) !Tensor {
        const b = switch (t.role) {
            .buf => |b| b,
            else => return t,
        };
        const S: i64 = w.s.slots;
        var x = t;
        if (isRing(b) or isCarry(b)) {
            if (t.offset != 0 or t.shape.len != 2) return error.BadCall;
            if (isRing(b)) {
                x.shape = w.dup(&.{ S * t.shape[0], t.shape[1] });
                x.stride = t.stride;
            } else {
                x.shape = w.dup(&.{ S, t.shape[1] });
                x.stride = w.dup(&.{ t.shape[1], 1 });
            }
            return x;
        }
        if (isTable(b)) {
            if (t.offset != 0 or t.shape.len != 1 or t.shape[0] != w.s.pts) return error.BadCall;
            x.shape = w.dup(&.{ S, w.s.pts });
            x.stride = w.dup(&.{ w.s.pts, 1 });
            return x;
        }
        return t;
    }

    fn arg(w: *const Rewriter, x: Arg) !Arg {
        return switch (x) {
            .t => |t| .{ .t = try w.stacked(t) },
            else => x,
        };
    }

    /// One call in row mode (a fresh Call; the input is not changed).
    pub fn call(w: *const Rewriter, c: calls.Call) !calls.Call {
        var out = c;
        const args = try w.a.alloc(calls.Named, c.args.len);
        for (c.args, args) |x, *y| y.* = .{ .name = x.name, .arg = try w.arg(x.arg) };
        out.args = args;
        if (c.glue) {
            if (eq(c.name, "glue.kx_union")) return error.UnionInRows;
            return out;
        }
        if (c.triton) return w.triton(out);
        if (eq(c.name, "tf_dsv41_attn_cuda_v1.attn")) {
            if (args.len < 23) return error.BadCall;
            args[9].arg = w.col(.pos);
            args[10].arg = w.col(.rslot);
            const pt_on = isTensor(args[11].arg);
            args[21].arg = .{ .i = if (pt_on) w.s.pts else 0 };
            return out;
        }
        if (eq(c.name, "tf_dsv41_attn_cuda_v1.topk")) {
            if (args.len < 22) return error.BadCall;
            args[17].arg = w.col(.pos);
            return out;
        }
        return out;
    }

    fn triton(w: *const Rewriter, c: calls.Call) !calls.Call {
        const args = @constCast(c.args);
        const writes = eq(c.name, "_kv_store") or eq(c.name, "_pool_norm") or eq(c.name, "_index_k");
        const reads = eq(c.name, "_scores");
        if (eq(c.name, "_rope")) {
            const p = find(args, "POS") orelse return error.BadCall;
            args[p].arg = w.col(.pos);
            return c;
        }
        if (!writes and !reads) {
            // a call with row knobs that is not one of the above: refused rather than run with one slot's view
            for (args) |x| if (eq(x.name, "ROWS") or eq(x.name, "SL")) return error.UnknownRowsCall;
            return c;
        }
        for (args) |*x| {
            if (eq(x.name, "POS")) x.arg = w.col(.pos);
            if (eq(x.name, "SL")) x.arg = w.col(if (writes) .wslot else .rslot);
            if (eq(x.name, "ROWS")) x.arg = .{ .b = true };
        }
        if (find(args, "PTS")) |i| {
            const pt = find(args, "PT") orelse return error.BadCall;
            args[i].arg = .{ .i = if (isTensor(args[pt].arg)) w.s.pts else 0 };
        }
        if (eq(c.name, "_pool_norm")) {
            const split = if (find(args, "SPLIT")) |i| (args[i].arg == .b and args[i].arg.b) else false;
            const prev = find(args, "PREV") orelse return error.BadCall;
            args[prev].arg = w.col(.prev);
            const off = find(args, "OFF") orelse return error.BadCall;
            args[off].arg = .{ .i = if (split) @as(i64, w.s.slots) else 0 };
        }
        return c;
    }
};

/// The row-mode program of `cs` (block.emit's window of R rows); `cs` is not changed.
pub fn transform(a: std.mem.Allocator, cs: []const calls.Call, R: u32, s: Spec) ![]const calls.Call {
    if (s.pts <= 0) return error.NoPool;
    const w: Rewriter = .{ .a = a, .s = s, .R = R };
    const out = try a.alloc(calls.Call, cs.len);
    for (cs, out) |c, *o| o.* = try w.call(c);
    return out;
}

fn eq(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

fn find(args: []const calls.Named, name: []const u8) ?usize {
    for (args, 0..) |x, i| if (eq(x.name, name)) return i;
    return null;
}

fn isTensor(x: Arg) bool {
    return switch (x) {
        .t => |t| t.role != .empty,
        else => false,
    };
}

/// "s.L<i>.swa.v" / ".s": a layer's SWA ring
pub fn isRing(b: []const u8) bool {
    return std.mem.startsWith(u8, b, "s.L") and (std.mem.endsWith(u8, b, ".swa.v") or std.mem.endsWith(u8, b, ".swa.s"));
}

/// "s.L<i>.carry": a ratio-2 kv source's carry
pub fn isCarry(b: []const u8) bool {
    return std.mem.startsWith(u8, b, "s.L") and std.mem.endsWith(u8, b, ".carry");
}

/// "s.kv.pt" / "s.kv.ct": a slot's page table / split table
pub fn isTable(b: []const u8) bool {
    return eq(b, "s.kv.pt") or eq(b, "s.kv.ct");
}

test "rowmode: stacked roles by name" {
    try std.testing.expect(isRing("s.L7.swa.v") and isRing("s.L7.swa.s") and !isRing("s.L7.comp.v"));
    try std.testing.expect(isCarry("s.L2.carry") and !isCarry("w.L2.comp"));
    try std.testing.expect(isTable("s.kv.pt") and isTable("s.kv.ct") and !isTable("s.kv.comp.L2"));
}
