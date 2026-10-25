// MiniMax-M3.1-Flash selected-block GQA attention (float64 reference: core/minimax_ref.zig attendBlocks).
// One CTA per (row, kv head): the 16 query heads of the group share one K/V stream over the row's selected blocks, taken in
// the select kernel's ascending order, 32 keys at a time with an online softmax. The key order, the 32-key tile and the
// fp32 chains are fixed by the row's own selection alone, so a row's bits do not depend on the window width, on the other
// rows of a call, or on whether the call is a decode window or a prompt chunk (pos null: pos0 + r).
// Keys past the row's position are never read (an unwritten page may hold NaN) and a -1 entry ends the list.
// Scores: sequential fp32 dot over d = 0..127, then times `scale`; softmax weights expf(s - m) in fp32; P.V in key order.

// kv8 != 0 (launch parameter; 0: the unchanged bf16 pool): e4m3 K/V rows with an fp32 scale each (docs/fp8-kv.md).
#include <cuda_fp8.h>

#include "minimax_m3_sparse.cuh"

using namespace tf_mm3;

namespace {

constexpr int TILE = 32;     // keys a step
constexpr int KSTRIDE = 136; // K row stride in shared (bf16): 272 B, 16-byte aligned and not a bank multiple

/// max that propagates a NaN (the float64 reference's softmax does).
__device__ __forceinline__ float nmax(float a, float b) {
    if (a != a) return a;
    if (b != b) return b;
    return a > b ? a : b;
}

/// Two e4m3 codes (low 16 bits of w, or the high 16 with `second`) as floats; exact, NaN codes stay NaN.
__device__ __forceinline__ float2 codes2(uint32_t w, bool second) {
    const __nv_fp8x2_storage_t c = (__nv_fp8x2_storage_t)(second ? (w >> 16) : (w & 0xffffu));
    return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(c, __NV_E4M3)));
}

__device__ __forceinline__ uint32_t pack2(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&v);
}

} // namespace

extern "C" __global__ void __launch_bounds__(256) tf_mm3_attn(
    const bf16* __restrict__ q,        // [rows, kvh * 16, 128] post norm and rotary
    const int* __restrict__ sel,       // [rows, kvh, 16] ascending block ids, -1 padded
    const int* __restrict__ pos,
    int pos0,
    const int* __restrict__ row_tab,
    const int* __restrict__ tables,
    const char* __restrict__ pool,
    long long page_bytes, long long layer_off,
    int kvh,
    bf16* __restrict__ out,            // [rows, kvh * 16, 128]
    float scale,
    int kv8) {
    const int r = blockIdx.x, g = blockIdx.y, t = threadIdx.x;
    const int h = t >> 4, c = t & 15; // query head of the group, 8-dim slice of its output
    const int p = row_pos(pos, pos0, r);
    const int* tab = tables + row_table(row_tab, r);
    const int* ids = sel + ((long long)r * kvh + g) * TOPK;
    const long long head = ((long long)r * kvh + g) * GROUP + h;

    __shared__ float sq[GROUP * D];
    __shared__ __align__(16) bf16 sk[TILE * KSTRIDE];
    __shared__ __align__(16) bf16 sv[TILE * D];
    __shared__ float sp[GROUP * TILE];
    // fp8 tiles reuse sk and sv: K codes 32 rows of 33 words, then the 32 key scales and 32 value scales
    uint32_t* sk8 = reinterpret_cast<uint32_t*>(sk);
    float* sks = reinterpret_cast<float*>(sk) + TILE * 33;
    float* svs = sks + TILE;

    {
        const uint4 w = __ldg(reinterpret_cast<const uint4*>(q + head * D) + c);
        float* d = sq + h * D + c * 8;
        d[0] = lo(w.x); d[1] = hi(w.x); d[2] = lo(w.y); d[3] = hi(w.y);
        d[4] = lo(w.z); d[5] = hi(w.z); d[6] = lo(w.w); d[7] = hi(w.w);
    }
    float o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = 0.f;
    float m = -INFINITY, l = 0.f;

    for (int s = 0; s < TOPK; ++s) {
        const int b = ids[s];
        if (b < 0) break;
        const char* base = page_layer(pool, page_bytes, layer_off, __ldg(tab + b));
        const bf16* kb = reinterpret_cast<const bf16*>(base) + kv_elem(kvh, 0, g, 0);
        const bf16* vb = reinterpret_cast<const bf16*>(base) + kv_elem(kvh, 1, g, 0);
        const unsigned char* kb8 = reinterpret_cast<const unsigned char*>(base) + kv8_byte(kvh, 0, g, 0);
        const unsigned char* vb8 = reinterpret_cast<const unsigned char*>(base) + kv8_byte(kvh, 1, g, 0);
        const float* ksc = reinterpret_cast<const float*>(base + scale8_byte(kvh, 0, g, 0));
        const float* vsc = reinterpret_cast<const float*>(base + scale8_byte(kvh, 1, g, 0));
        for (int k0 = 0; k0 < BLOCK; k0 += TILE) {
            const int nvalid = min(TILE, p + 1 - (b * BLOCK + k0));
            if (nvalid <= 0) break;
            __syncthreads();
            if (kv8) {
                {
                    const int row = t >> 3, vec = t & 7;  // 256 threads: 32 rows x 8 sixteen-byte pieces, K then V
                    uint4 kw = make_uint4(0, 0, 0, 0), vw = kw;
                    if (row < nvalid) {
                        kw = __ldg(reinterpret_cast<const uint4*>(kb8 + (long long)(k0 + row) * D) + vec);
                        vw = __ldg(reinterpret_cast<const uint4*>(vb8 + (long long)(k0 + row) * D) + vec);
                    }
                    uint32_t* kd = sk8 + row * 33 + vec * 4;
                    kd[0] = kw.x; kd[1] = kw.y; kd[2] = kw.z; kd[3] = kw.w;
                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(sv) + row * D + vec * 16) = vw;
                }
                if (t < TILE) {
                    sks[t] = t < nvalid ? __ldg(ksc + k0 + t) : 0.f;
                    svs[t] = t < nvalid ? __ldg(vsc + k0 + t) : 0.f;
                }
            } else
            for (int i = t; i < TILE * 16; i += 256) {
                const int row = i >> 4, vec = i & 15;
                uint4 kw = make_uint4(0, 0, 0, 0), vw = kw;
                if (row < nvalid) {
                    kw = __ldg(reinterpret_cast<const uint4*>(kb + (long long)(k0 + row) * D) + vec);
                    vw = __ldg(reinterpret_cast<const uint4*>(vb + (long long)(k0 + row) * D) + vec);
                }
                *reinterpret_cast<uint4*>(sk + row * KSTRIDE + vec * 8) = kw;
                *reinterpret_cast<uint4*>(sv + row * D + vec * 8) = vw;
            }
            __syncthreads();
            for (int u = 0; u < 2; ++u) {
                const int j = c + 16 * u;
                if (j >= nvalid) continue;
                const float* qh = sq + h * D;
                if (kv8) {
                    const uint32_t* kr = sk8 + j * 33;
                    float a = 0.f;
#pragma unroll 8
                    for (int e = 0; e < D / 4; ++e) {
                        const uint32_t w = kr[e];
                        const float2 c01 = codes2(w, false), c23 = codes2(w, true);
                        a = __fadd_rn(a, __fmul_rn(qh[4 * e], c01.x));
                        a = __fadd_rn(a, __fmul_rn(qh[4 * e + 1], c01.y));
                        a = __fadd_rn(a, __fmul_rn(qh[4 * e + 2], c23.x));
                        a = __fadd_rn(a, __fmul_rn(qh[4 * e + 3], c23.y));
                    }
                    sp[h * TILE + j] = __fmul_rn(__fmul_rn(a, sks[j]), scale);
                } else {
                    const uint32_t* kr = reinterpret_cast<const uint32_t*>(sk + j * KSTRIDE);
                    float a = 0.f;
#pragma unroll 8
                    for (int e = 0; e < D / 2; ++e) {
                        const uint32_t w = kr[e];
                        a = __fadd_rn(a, __fmul_rn(qh[2 * e], lo(w)));
                        a = __fadd_rn(a, __fmul_rn(qh[2 * e + 1], hi(w)));
                    }
                    sp[h * TILE + j] = __fmul_rn(a, scale);
                }
            }
            __syncthreads();
            float mt = sp[h * TILE];
            for (int j = 1; j < nvalid; ++j) mt = nmax(mt, sp[h * TILE + j]);
            const float mn = nmax(m, mt);
            const float alpha = (m == -INFINITY) ? 0.f : expf(__fsub_rn(m, mn));
            l = __fmul_rn(l, alpha);
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = __fmul_rn(o[i], alpha);
            for (int j = 0; j < nvalid; ++j) {
                const float pj = expf(__fsub_rn(sp[h * TILE + j], mn));
                l = __fadd_rn(l, pj);
                if (kv8) {
                    const uint2 vw = *reinterpret_cast<const uint2*>(reinterpret_cast<const unsigned char*>(sv) + j * D + c * 8);
                    const float ps = __fmul_rn(pj, svs[j]);
                    const float2 v0 = codes2(vw.x, false), v1 = codes2(vw.x, true), v2 = codes2(vw.y, false), v3 = codes2(vw.y, true);
                    o[0] = __fadd_rn(o[0], __fmul_rn(ps, v0.x)); o[1] = __fadd_rn(o[1], __fmul_rn(ps, v0.y));
                    o[2] = __fadd_rn(o[2], __fmul_rn(ps, v1.x)); o[3] = __fadd_rn(o[3], __fmul_rn(ps, v1.y));
                    o[4] = __fadd_rn(o[4], __fmul_rn(ps, v2.x)); o[5] = __fadd_rn(o[5], __fmul_rn(ps, v2.y));
                    o[6] = __fadd_rn(o[6], __fmul_rn(ps, v3.x)); o[7] = __fadd_rn(o[7], __fmul_rn(ps, v3.y));
                } else {
                    const uint4 vw = *reinterpret_cast<const uint4*>(sv + j * D + c * 8);
                    o[0] = __fadd_rn(o[0], __fmul_rn(pj, lo(vw.x))); o[1] = __fadd_rn(o[1], __fmul_rn(pj, hi(vw.x)));
                    o[2] = __fadd_rn(o[2], __fmul_rn(pj, lo(vw.y))); o[3] = __fadd_rn(o[3], __fmul_rn(pj, hi(vw.y)));
                    o[4] = __fadd_rn(o[4], __fmul_rn(pj, lo(vw.z))); o[5] = __fadd_rn(o[5], __fmul_rn(pj, hi(vw.z)));
                    o[6] = __fadd_rn(o[6], __fmul_rn(pj, lo(vw.w))); o[7] = __fadd_rn(o[7], __fmul_rn(pj, hi(vw.w)));
                }
            }
            m = mn;
        }
    }
    uint4 w;
    w.x = pack2(__fdiv_rn(o[0], l), __fdiv_rn(o[1], l));
    w.y = pack2(__fdiv_rn(o[2], l), __fdiv_rn(o[3], l));
    w.z = pack2(__fdiv_rn(o[4], l), __fdiv_rn(o[5], l));
    w.w = pack2(__fdiv_rn(o[6], l), __fdiv_rn(o[7], l));
    reinterpret_cast<uint4*>(out + head * D)[c] = w;
}
