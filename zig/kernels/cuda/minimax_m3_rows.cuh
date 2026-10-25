// MiniMax-M3.1-Flash row kernels: launch flags and helpers of minimax_m3_rows.cu, written from the engine-port plan.
#pragma once

#include "minimax_m3_sparse.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>

namespace tf_mm3 {

// Bits of the flags argument of tf_mm3_rows: each unverified semantic of core/minimax_ref.zig Switches is one.
constexpr int ROWS_NORM_QK = 1;      // Gemma-norm q and k per head (use_qk_norm, qk_norm_type per_head)
constexpr int ROWS_NORM_INDEX = 2;   // Gemma-norm the index query heads and the index key
constexpr int ROWS_INDEX_ROPE = 4;   // rotate the index query and key (Switches.index_rope)
constexpr int ROWS_INDEX_FP16 = 8;   // the index projections' outputs are fp16 (else bf16)
constexpr int ROWS_ROUND_NORM = 16;  // round the norm output to bf16 before the rotation (Switches.bf16_norm_out)
constexpr int ROWS_GPTJ = 32;        // rotate (2i, 2i + 1) pairs (Switches.rope_layout gptj_interleaved); else (i, i + rot/2)
constexpr int ROWS_KV8 = 64;         // the pool is the fp8 layout: k and v as e4m3 rows with an fp32 scale each (docs/fp8-kv.md)

// Bits kernels may set in the optional status word.
constexpr int ROWS_STATUS_NO_PAGE = 1;    // a row's block table entry was negative: its K/V/IK were not written
constexpr int ROWS_STATUS_PAST_TABLE = 2; // a row's position is past its block table row

constexpr int ROWS_WARPS = 8;
constexpr int ROWS_THREADS = ROWS_WARPS * 32;
constexpr int MAX_ROT_PAIRS = 64;

constexpr int ROUTER_WARPS = 4;    // experts a block
constexpr int ROUTER_ROWS = 4;     // rows a warp; a row's bits do not depend on how many share the warp
constexpr int ROUTER_CHUNK = 128;  // k is a multiple: 32 lanes x 4 values a step

__device__ __forceinline__ float rn_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float4 widen4(uint2 w) { return make_float4(lo(w.x), hi(w.x), lo(w.y), hi(w.y)); }

__device__ __forceinline__ float4 widen4_half(uint2 w) {
    return make_float4(__half2float(__ushort_as_half(w.x & 0xffffu)), __half2float(__ushort_as_half(w.x >> 16)),
                       __half2float(__ushort_as_half(w.y & 0xffffu)), __half2float(__ushort_as_half(w.y >> 16)));
}

__device__ __forceinline__ uint2 pack4(float a, float b, float c, float d) {
    const unsigned x = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(a)) |
                       ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(b)) << 16);
    const unsigned y = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(c)) |
                       ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(d)) << 16);
    return make_uint2(x, y);
}

// A 128-value row (4 a lane, whole warp) as e4m3 codes and the fp32 scale amax / 448; a NaN or inf row: NaN scale.
__device__ __forceinline__ void quant_row8(float4 x, unsigned char* codes, float* scale, int lane) {
    const bool bad = !(fabsf(x.x) <= FLT_MAX) || !(fabsf(x.y) <= FLT_MAX) || !(fabsf(x.z) <= FLT_MAX) || !(fabsf(x.w) <= FLT_MAX);
    const bool any_bad = __any_sync(0xffffffffu, bad);
    float amax = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float inv = 1.f, sc = 1.f;
    if (amax > 0.f) {
        inv = __fdiv_rn(448.f, amax);
        sc = __fdiv_rn(amax, 448.f);
        if (sc < 0x1p-126f) {
            sc = 0x1p-126f;
            inv = __fdiv_rn(1.f, sc);
        }
    }
    unsigned w = 0;
    if (!any_bad) {
        w = (unsigned)__nv_cvt_float_to_fp8(__fmul_rn(x.x, inv), __NV_SATFINITE, __NV_E4M3) |
            ((unsigned)__nv_cvt_float_to_fp8(__fmul_rn(x.y, inv), __NV_SATFINITE, __NV_E4M3) << 8) |
            ((unsigned)__nv_cvt_float_to_fp8(__fmul_rn(x.z, inv), __NV_SATFINITE, __NV_E4M3) << 16) |
            ((unsigned)__nv_cvt_float_to_fp8(__fmul_rn(x.w, inv), __NV_SATFINITE, __NV_E4M3) << 24);
    }
    reinterpret_cast<unsigned*>(codes)[lane] = w;
    if (lane == 0) *scale = any_bad ? __int_as_float(0x7fc00000) : sc;
}

} // namespace tf_mm3
