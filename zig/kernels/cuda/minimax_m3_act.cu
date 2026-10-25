// MiniMax-M3.1-Flash swigluoai, pointwise (dense layers 0-2, shared expert); ref core/minimax_ref.zig `swigluoai`.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "minimax_m3_act.cuh"

namespace {
constexpr int NT = 256;
}

extern "C" __global__ void __launch_bounds__(NT) tf_mm3_swigluoai_bf16(
        const __nv_bfloat16* __restrict__ gate, long long g_ld, const __nv_bfloat16* __restrict__ up, long long u_ld,
        __nv_bfloat16* __restrict__ out, long long o_ld, int n, float alpha, float limit, float up_bias, int mode) {
    const int64_t r = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = tf_mm3::swigluoai(__bfloat162float(gate[r * g_ld + i]), __bfloat162float(up[r * u_ld + i]), alpha, limit,
                                      up_bias, mode == tf_mm3::ACT_SWIGLUOAI_SYM);
    out[r * o_ld + i] = __float2bfloat16_rn(v);
}

// fp32 gate and up (a dense linear with an fp32 output), bf16 out: the next linear's input.
extern "C" __global__ void __launch_bounds__(NT) tf_mm3_swigluoai_f32(
        const float* __restrict__ gate, long long g_ld, const float* __restrict__ up, long long u_ld,
        __nv_bfloat16* __restrict__ out, long long o_ld, int n, float alpha, float limit, float up_bias, int mode) {
    const int64_t r = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = tf_mm3::swigluoai(gate[r * g_ld + i], up[r * u_ld + i], alpha, limit, up_bias,
                                      mode == tf_mm3::ACT_SWIGLUOAI_SYM);
    out[r * o_ld + i] = __float2bfloat16_rn(v);
}

// MiniMax router: top-k by sigmoid(logit) + bias (ties lower id, NaN as -inf), a warp a row; ref routerPick.
extern "C" __global__ void __launch_bounds__(256) tf_mm3_route_warp(const float* __restrict__ logits,
        const float* __restrict__ bias, int* __restrict__ pick, float* __restrict__ wts, int rows, int ne, int topk,
        int normalize, float scale) {
    constexpr int PER = 4;
    const int lane = threadIdx.x & 31;
    const int64_t r = static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    const float* lg = logits + r * ne;
    float x[PER], v[PER];
    bool taken[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int e = lane + 32 * j;
        x[j] = e < ne ? lg[e] : 0.f;
        float s = __fadd_rn(__fdiv_rn(1.f, __fadd_rn(1.f, expf(-x[j]))), bias != nullptr && e < ne ? bias[e] : 0.f);
        v[j] = (e < ne && s == s) ? s : -INFINITY;
        taken[j] = e >= ne;
    }
    float xk = 0.f;  // lane k's pick's logit
    int pk = 0;      // lane k's pick
    for (int k = 0; k < topk; ++k) {
        float bv = -INFINITY;
        int bi = ne;
        bool any = false;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int e = lane + 32 * j;
            if (!taken[j] && (!any || v[j] > bv)) bv = v[j], bi = e, any = true;  // e rises with j: the lower id keeps a tie
        }
        if (!any) bv = -INFINITY, bi = 0x7FFFFFFF;
        for (int m = 16; m > 0; m >>= 1) {
            const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, m);
            const int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, m);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        float xb = 0.f;
#pragma unroll
        for (int j = 0; j < PER; ++j)
            if (lane + 32 * j == bi) taken[j] = true, xb = x[j];
        xb = __shfl_sync(0xFFFFFFFFu, xb, bi & 31);
        if (lane == k) xk = xb, pk = bi;
    }
    const float wk = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-xk)));
    float sum = 0.f;
    for (int k = 0; k < topk; ++k) sum = __fadd_rn(sum, __shfl_sync(0xFFFFFFFFu, wk, k));
    if (lane < topk) {
        pick[r * topk + lane] = pk;
        wts[r * topk + lane] = __fmul_rn(normalize ? __fdiv_rn(wk, sum) : wk, scale);
    }
}
