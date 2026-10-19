// rot_in's arithmetic spelled out (ours): linear.cu's rot_in_kernel as nvcc 13 compiles it, so other kernels match it.
#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace dsv41_rot {

constexpr float HAD = 0.08838834764831845f;               // decode.cuh's HAD_SCALE, 1 / sqrt(128)

// 4 bf16 (or fp16 when !bf16) of x at element i, widened exactly (linear.cu's load4)
__device__ __forceinline__ void load4x(const void* p, bool bf16, size_t i, float (&v)[4]) {
    const uint2 u = __ldcg(reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(p) + i));
    if (bf16) {
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

// lane's 4 values of a 128-block: v = x * s, fwht128, * HAD; the scale-and-first-butterfly as rot_in's SASS has it
__device__ __forceinline__ void rot128(const float (&x)[4], const float (&s)[4], int lane, float (&v)[4]) {
    const float p1 = __fmul_rn(x[1], s[1]), p3 = __fmul_rn(x[3], s[3]);
    const float a = __fmaf_rn(x[0], s[0], p1), b = __fmaf_rn(x[0], s[0], -p1);
    const float c = __fmaf_rn(x[2], s[2], p3), d = __fmaf_rn(x[2], s[2], -p3);
    v[0] = __fadd_rn(a, c);
    v[1] = __fadd_rn(b, d);
    v[2] = __fsub_rn(a, c);
    v[3] = __fsub_rn(b, d);
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? __fsub_rn(o, v[j]) : __fadd_rn(v[j], o);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __fmul_rn(v[j], HAD);
}

// the rotated row piece as rot_in stores it: fp16, round to nearest
__device__ __forceinline__ void store4h(half* p, size_t i, const float (&v)[4]) {
    const half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
    uint2 u;
    u.x = *reinterpret_cast<const uint32_t*>(&a);
    u.y = *reinterpret_cast<const uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p + i) = u;
}

}  // namespace dsv41_rot
