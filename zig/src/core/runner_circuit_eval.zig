//! Exact CPU evaluation of one circuit step: token features, SwiGLU units, one-hot DeltaNet heads, output code.
const std = @import("std");
const model = @import("runner_circuit_model.zig");
const State = @import("runner_circuit_state.zig").State;

const Scalar = model.Scalar;
const hd = model.hd;

/// The residual a token starts from: dim 0 is one, each feature dim a function of the token's family and payload.
pub fn embed(c: *const model.Circuit, token: u32, x: []Scalar) !void {
    const fam, const pay = c.split(token) orelse return error.CircuitUnknownToken;
    @memset(x, 0);
    x[0] = 1;
    for (c.features) |f| x[f.dim] = f.value(fam, pay, c.family_bits);
}

/// Feeds one token through every batch in stored node order and returns the token the output code names.
pub fn step(s: *State, token: u32) !u32 {
    const c = s.circuit;
    try embed(c, token, s.x);
    for (c.batches) |*b| switch (b.*) {
        .ffn => |*f| ffn(f, s.x, s.work),
        .heads => |*h| try heads(s, h),
    };
    s.steps += 1;
    return decode(c, s.x);
}

fn commit(outs: []const model.Out, values: []const Scalar, x: []Scalar) void {
    for (outs, values) |o, v| x[o.dim] = if (o.into) x[o.dim] + v else v;
}

fn ffn(f: *const model.Ffn, x: []Scalar, work: []Scalar) void {
    const t = f.terms;
    const up = work[0..t];
    const gate = work[t .. 2 * t];
    const y = work[2 * t ..][0..f.outs.len];
    @memset(work[0 .. 2 * t + f.outs.len], 0);
    f.up.apply(x, up);
    f.gate.apply(x, gate);
    for (f.coef, f.slot, up, gate) |c, s, u, g| y[s] += c * u * @max(g, 0);
    commit(f.outs, y, x);
}

fn heads(s: *State, h: *const model.Heads) !void {
    const width = h.proj.outputs;
    const p = s.work[0..width];
    const values = s.work[width..][0..h.outs.len];
    const rest = s.work[width + h.outs.len ..];
    const keys = rest[0..hd];
    const query = rest[hd .. 2 * hd];
    const v = rest[2 * hd .. 3 * hd];
    const z = rest[3 * hd .. 4 * hd];
    const o = rest[4 * hd .. 5 * hd];
    @memset(p, 0);
    h.proj.apply(s.x, p);
    for (p[0..h.channels], h.taps, 0..) |*e, t, i| {
        const past = s.channelHistory(h.first_channel + @as(u32, @intCast(i)));
        const now = e.*;
        e.* = @max(t[0] * now + t[1] * past[0] + t[2] * past[1] + t[3] * past[2], 0);
        past.* = .{ now, past[0], past[1] };
    }
    for (h.heads) |*head| {
        scatter(head.k, p, keys, false);
        scatter(head.q, p, query, false);
        const key = try oneHot(keys);
        for (head.value_heads) |*vh| {
            if (vh.a) |col| try decay(s, vh.table, p[col]);
            if (key) |r| if (try writeEnabled(vh.beta, p)) {
                scatter(vh.v, p, v, false);
                try s.writeRow(vh.table, r, v);
            };
            try read(s, vh.table, query, o);
            scatter(vh.z, p, z, true);
            for (vh.out_cols, vh.out_w, values[vh.out_first..][0..vh.out_cols.len]) |col, w, *y| y.* = o[col] * z[col] * w;
        }
    }
    commit(h.outs, values, s.x);
}

fn scatter(picks: []const model.Pick, p: []const Scalar, into: []Scalar, rectify: bool) void {
    @memset(into, 0);
    for (picks) |k| into[k.at] = if (rectify) @max(p[k.column], 0) else p[k.column];
}

/// The row a key selects, null when no entry is positive; more than one positive entry is not a one-hot key.
fn oneHot(keys: []const Scalar) !?u32 {
    var found: ?u32 = null;
    for (keys, 0..) |k, r| if (k > 0) {
        if (found != null) return error.CircuitKeyNotOneHot;
        found = @intCast(r);
    };
    return found;
}

/// beta: absent is 1; 1 writes, an integer <= 0 does not, anything else breaks the circuit's exact range.
fn writeEnabled(beta: ?u32, p: []const Scalar) !bool {
    const b = p[beta orelse return true];
    if (b == 1) return true;
    if (b <= 0 and b == @round(b)) return false;
    return error.CircuitWriteGate;
}

fn decay(s: *State, table: u32, selector: Scalar) !void {
    if (selector == model.keep_selector) return;
    if (!s.writable(table)) return error.CircuitTableReadOnly;
    if (selector == model.halve_selector) {
        var live = s.present[table];
        while (live != 0) : (live &= live - 1) {
            for (s.row(table, @ctz(live))) |*e| e.* *= 0.5;
        }
        return;
    }
    if (selector != model.clear_selector) return error.CircuitDecaySelector;
    @memset(s.tables[table], 0);
    s.present[table] = 0;
}

/// Sum of the rows a multi-hot query selects, normalized by the ballast: o / sqrt((o0^2 + o1^2) / hd).
fn read(s: *State, table: u32, query: []const Scalar, o: []Scalar) !void {
    @memset(o, 0);
    var any = false;
    for (query, 0..) |q, r| {
        if (q <= 0 or !s.rowPresent(table, @intCast(r))) continue;
        for (o, s.row(table, @intCast(r))) |*a, b| a.* += b;
        any = true;
    }
    if (!any) return;
    const ref = o[0] * o[0] + o[1] * o[1];
    if (ref == 0) {
        for (o) |a| if (a != 0) return error.CircuitReadBallast;
        return;
    }
    const norm = @sqrt(ref / @as(Scalar, hd));
    for (o) |*a| a.* /= norm;
}

/// The output code: family bits then 16 payload bits, each the sign of a code dim (+kappa is 1, -kappa is 0).
pub fn decode(c: *const model.Circuit, x: []const Scalar) !u32 {
    var code: u64 = 0;
    for (c.code_dims, 0..) |d, i| {
        if (x[d] == 0 or !std.math.isFinite(x[d])) return error.CircuitOutputCode;
        code |= @as(u64, @intFromBool(x[d] > 0)) << @intCast(i);
    }
    const mask = (@as(u64, 1) << @intCast(c.family_bits)) - 1;
    const fam = code & mask;
    const pay: u32 = @intCast(code >> @intCast(c.family_bits));
    if (fam >= c.families.len) return error.CircuitOutputFamily;
    const f = c.families[@intCast(fam)];
    if (pay >= f.size) return error.CircuitOutputPayload;
    return f.base + pay;
}
