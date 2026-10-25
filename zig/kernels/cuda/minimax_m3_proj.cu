// MiniMax-M3.1-Flash fp16-weight index projection and the per-rank greedy pick; no third-party code.

// tf_mm3_proj_{bf16,f32}: out[r][e] = sum_k w[e][k] * x[r][k] for the index query (n0 outputs) and the index key (n1).

// Weights are fp16 (not quantized), x is the normed hidden state, bf16 or fp32; every product is exact in fp32.

// A warp owns one output and up to 4 rows. Lane l sums chunks l, l + 32, ... of 4 values in four fp32 fma chains, then

// (a0 + a1) + (a2 + a3), then a fixed xor tree 16 .. 1: the order never looks at the rows, the tile or the neighbours.

// The fp32 sum is rounded once to fp16 or bf16 (launch parameter). NaN and infinities propagate as float64 does.

// tf_mm3_pick: a CTA per row scans the rank's vocabulary slice of the padded logits and writes (logit, id).

// Order: larger logit wins, equal logits go to the lower id, a NaN loses to every number (all NaN: the lowest id).

// That is a total order, so any reduction tree gives the same pair.
#include "minimax_m3_rows.cuh"

using namespace tf_mm3;

namespace {

constexpr int PROJ_WARPS = 4;  // outputs a block
constexpr int PROJ_ROWS = 4;   // rows a warp
constexpr int PROJ_CHUNK = 128; // k is a multiple: 32 lanes x 4 values a step
constexpr int PICK_THREADS = 256;

__device__ __forceinline__ float tree32(float a) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, o));
    return a;
}

__device__ __forceinline__ float4 load_x(const bf16* p) { return widen4(__ldg(reinterpret_cast<const uint2*>(p))); }
__device__ __forceinline__ float4 load_x(const float* p) { return __ldg(reinterpret_cast<const float4*>(p)); }

template <typename X>
__device__ __forceinline__ void proj_body(const X* __restrict__ x, long long x_ld, const unsigned short* __restrict__ w0,
                                          int n0, unsigned short* __restrict__ o0, long long o0_ld,
                                          const unsigned short* __restrict__ w1, int n1, unsigned short* __restrict__ o1,
                                          long long o1_ld, int rows, int k, int out_bf16) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * PROJ_WARPS + warp;
    if (e >= n0 + n1) return;
    const bool second = e >= n0;
    const int j = second ? e - n0 : e;
    const unsigned short* we = (second ? w1 : w0) + (long long)j * k;
    unsigned short* out = second ? o1 : o0;
    const long long out_ld = second ? o1_ld : o0_ld;
    const long long r0 = (long long)blockIdx.y * PROJ_ROWS;
    float acc[PROJ_ROWS][4];
#pragma unroll
    for (int i = 0; i < PROJ_ROWS; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll 4
    for (int c = 0; c < k / PROJ_CHUNK; ++c) {
        const int at = 4 * (lane + 32 * c);
        const float4 wv = widen4_half(__ldg(reinterpret_cast<const uint2*>(we + at)));
#pragma unroll
        for (int i = 0; i < PROJ_ROWS; ++i) {
            if (r0 + i >= rows) continue;
            const float4 xv = load_x(x + (r0 + i) * x_ld + at);
            acc[i][0] = __fmaf_rn(xv.x, wv.x, acc[i][0]);
            acc[i][1] = __fmaf_rn(xv.y, wv.y, acc[i][1]);
            acc[i][2] = __fmaf_rn(xv.z, wv.z, acc[i][2]);
            acc[i][3] = __fmaf_rn(xv.w, wv.w, acc[i][3]);
        }
    }
#pragma unroll
    for (int i = 0; i < PROJ_ROWS; ++i) {
        const float sum = tree32(__fadd_rn(__fadd_rn(acc[i][0], acc[i][1]), __fadd_rn(acc[i][2], acc[i][3])));
        if (lane == 0 && r0 + i < rows)
            out[(r0 + i) * out_ld + j] = out_bf16 ? __bfloat16_as_ushort(__float2bfloat16_rn(sum))
                                                  : __half_as_ushort(__float2half_rn(sum));
    }
}

// True when candidate (bv, bi) beats (av, ai): a total order, NaN last, ties to the lower id.
__device__ __forceinline__ bool beats(float bv, int bi, float av, int ai) {
    const bool an = av != av, bn = bv != bv;
    if (an && bn) return bi < ai;
    if (bn) return false;
    if (an) return true;
    return bv > av || (bv == av && bi < ai);
}

} // namespace

// x [rows, k] at x_ld elements a row; w0 fp16 [n0, k] -> o0 [rows, n0] at o0_ld; w1 [n1, k] -> o1 (n1 may be 0).

// k a multiple of 128; x_ld a multiple of 4; outputs fp16 bits, or bf16 bits when out_bf16 is nonzero.
extern "C" __global__ void __launch_bounds__(PROJ_WARPS * 32) tf_mm3_proj_bf16(
    const bf16* __restrict__ x, long long x_ld, const unsigned short* __restrict__ w0, int n0, unsigned short* __restrict__ o0,
    long long o0_ld, const unsigned short* __restrict__ w1, int n1, unsigned short* __restrict__ o1, long long o1_ld, int rows,
    int k, int out_bf16) {
    proj_body<bf16>(x, x_ld, w0, n0, o0, o0_ld, w1, n1, o1, o1_ld, rows, k, out_bf16);
}

// The same with an fp32 input: a bf16 hidden cast to fp32 gives the same bits as the bf16 form.
extern "C" __global__ void __launch_bounds__(PROJ_WARPS * 32) tf_mm3_proj_f32(
    const float* __restrict__ x, long long x_ld, const unsigned short* __restrict__ w0, int n0, unsigned short* __restrict__ o0,
    long long o0_ld, const unsigned short* __restrict__ w1, int n1, unsigned short* __restrict__ o1, long long o1_ld, int rows,
    int k, int out_bf16) {
    proj_body<float>(x, x_ld, w0, n0, o0, o0_ld, w1, n1, o1, o1_ld, rows, k, out_bf16);
}

// logits fp32 [rows][ld] (the first n of a row are the rank's slice, the rest is padding); out [rows][2] words:

// the winning logit's bits, then id0 + its index.
extern "C" __global__ void __launch_bounds__(PICK_THREADS) tf_mm3_pick(const float* __restrict__ logits, long long ld,
                                                                       int n, int id0, unsigned* __restrict__ out) {
    __shared__ float sv[PICK_THREADS / 32];
    __shared__ int si[PICK_THREADS / 32];
    const float* row = logits + (long long)blockIdx.x * ld;
    float bv = __int_as_float(0x7fc00000); // NaN: loses to everything
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += PICK_THREADS) {
        const float v = __ldg(row + i);
        if (beats(v, i, bv, bi)) {
            bv = v;
            bi = i;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (beats(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        sv[threadIdx.x >> 5] = bv;
        si[threadIdx.x >> 5] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = sv[0];
        int i = si[0];
        for (int w = 1; w < PICK_THREADS / 32; ++w) {
            if (beats(sv[w], si[w], v, i)) {
                v = sv[w];
                i = si[w];
            }
        }
        out[2 * blockIdx.x] = __float_as_uint(v);
        out[2 * blockIdx.x + 1] = (unsigned)(id0 + i);
    }
}
