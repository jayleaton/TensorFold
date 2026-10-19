//! Prod's prefill launch-fold knobs (prod_knobs.zig's, past its 600 lines): TF_DSV41_PF_ROT2 / _WOROT and apply's.
const dk = @import("dsv41_kernels");
const block = @import("block.zig");
const pk = @import("prod_knobs.zig");

/// TF_DSV41_PF_ROT2 (ours; unset / 0 off, 1 on): two fused projections of one input, one rot2 launch for both rot_in.
pub fn pfRot2(get: pk.Get) !bool {
    return dk.pfrot.enabled("TF_DSV41_PF_ROT2", get("TF_DSV41_PF_ROT2"));
}

/// TF_DSV41_PF_WOROT (ours; unset / 0 off, 1 on): wo_a's fused GEMMs rotate their output for wo_b, which skips rot_in.
pub fn pfWorot(get: pk.Get) !bool {
    return dk.pfrot.enabled("TF_DSV41_PF_WOROT", get("TF_DSV41_PF_WOROT"));
}

/// prod_knobs.apply's: TF_DSV41_MHC_PF_FOLD, _PF_ROT2, _PF_WOROT (the kernels load their modules under the same knobs).
pub fn apply(o: *block.Options, get: pk.Get) !void {
    o.mhc_pf_fold = try pk.mhcPfFold(get);
    o.pf_rot2 = try pfRot2(get);
    o.pf_worot = try pfWorot(get);
}
