// MiniMax-M3.1-Flash's swigluoai, one definition for every fused epilogue and the pointwise kernel.
// Written from the engine-port plan's formula (section 2), no third-party code:
//   g = min(g, limit) (act_mode 3: clamp(g, -limit, limit)), u = clamp(u, -limit, limit),
//   out = g * sigmoid(alpha * g) * (u + up_bias), fp32 throughout.
// alpha 1.702, limit 7 and up_bias 1 are the plan's REFERENCE values; the gate/up assignment, the "+1" and the gate
// clamp are UNVERIFIED against upstream, so each is a parameter (up_bias 0 drops the +1, mode 3 is the symmetric
// clamp; w1-is-gate is the caller's choice of which projection it passes as g).
// A NaN in g or u stays a NaN: the clamps are compares, not fminf/fmaxf (which hand back the limit).
// limit <= 0 means no clamp, as for the silu paths.
#pragma once

namespace tf_mm3 {

constexpr int ACT_SWIGLUOAI = 2;      // gate clamped above only
constexpr int ACT_SWIGLUOAI_SYM = 3;  // gate clamped on both sides

__device__ __forceinline__ float hi_clamp(float x, float l) { return x > l ? l : x; }
__device__ __forceinline__ float both_clamp(float x, float l) { return x > l ? l : (x < -l ? -l : x); }

// Exactly rounded division on purpose (act_rows.cu does the same): the same bits on every GPU.
__device__ __forceinline__ float swigluoai(float g, float u, float alpha, float limit, float up_bias, bool sym) {
    if (limit > 0.f) {
        g = sym ? both_clamp(g, limit) : hi_clamp(g, limit);
        u = both_clamp(u, limit);
    }
    const float sg = __fdiv_rn(g, __fadd_rn(1.f, expf(-__fmul_rn(alpha, g))));  // g * sigmoid(alpha * g)
    return __fmul_rn(sg, __fadd_rn(u, up_bias));
}

}  // namespace tf_mm3
