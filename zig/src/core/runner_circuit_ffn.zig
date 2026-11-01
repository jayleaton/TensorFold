//! An FFN batch part on the CPU: every form by scatter, then every firing unit by the sorted walk.
const model = @import("runner_circuit_model.zig");
const State = @import("runner_circuit_state.zig").State;

const Scalar = model.Scalar;
const none = model.no_form;

pub const Plan = struct {
    f: *const model.Ffn,
    /// Where this part's forms and nodes live in the per-machine arrays.
    up_off: u32,
    gate_off: u32,
    node_off: u32,
};

pub fn build(f: *const model.Ffn, offs: *[3]u32) Plan {
    const p: Plan = .{ .f = f, .up_off = offs[0], .gate_off = offs[1], .node_off = offs[2] };
    offs[0] += f.up.forms.outputs;
    offs[1] += f.gate.forms.outputs;
    offs[2] += @intCast(f.outs.len);
    return p;
}

/// The part's outputs for the running inputs (nz dims ascending, values in s.x); silent units (gate <= 0) add zero.
pub fn run(s: *State, p: *const Plan, nz: []const u32) []const Scalar {
    const f = p.f;
    const up = s.ffn_up[p.up_off..][0..f.up.forms.outputs];
    const gate = s.ffn_gate[p.gate_off..][0..f.gate.forms.outputs];
    const y = s.ffn_y[p.node_off..][0..f.outs.len];
    @memset(y, 0);
    f.up.forms.apply(s.x, nz, up);
    f.gate.forms.apply(s.x, nz, gate);
    for (gate, f.starts[0 .. f.starts.len - 1], f.starts[1..]) |v, lo, hi| for (f.order[lo..hi]) |t| {
        const g = f.gate.bias[t] + v;
        if (!(g > 0)) break;
        y[f.slot[t]] += f.coef[t] * upValue(f, t, up) * g;
    };
    for (f.always) |t| y[f.slot[t]] += f.coef[t] * upValue(f, t, up) * f.gate.bias[t];
    return y;
}

fn upValue(f: *const model.Ffn, t: u32, up: []const Scalar) Scalar {
    const uf = f.up.form[t];
    return if (uf == none) f.up.bias[t] else f.up.bias[t] + up[uf];
}
