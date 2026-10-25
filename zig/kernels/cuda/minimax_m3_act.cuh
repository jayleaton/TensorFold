// MiniMax-M3.1-Flash swigluoai for every epilogue, fp32; compare clamps keep NaN, limit <= 0 is no clamp.
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
