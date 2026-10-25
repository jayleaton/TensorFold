// MiniMax-M3.1-Flash index scoring and per-128-key block maxima (float64 reference: core/minimax_ref.zig blockScores).
//   s[j] = (iq[g] . IK[j]) * scale   for keys j <= pos of a row, fp32; bmax[r][g][b] = max over the block's valid keys.
// One CTA per (block, index head g, row): 4 warps, a warp scores its keys one at a time with the 32 lanes split over the
// 128 dims (4 each, fixed order) and a fixed xor tree, so a score's bits depend on nothing but its two vectors.
// NaN rule (stated, matches the reference): fmax drops a NaN key score; a block whose keys are all NaN scores -inf.
#include "minimax_m3_sparse.cuh"

using namespace tf_mm3;

extern "C" __global__ void __launch_bounds__(128) tf_mm3_index(
    const bf16* __restrict__ iq,       // [rows, kvh, 128] post norm and rotary
    const int* __restrict__ pos,       // [rows] or null (pos0 + r)
    int pos0,
    const int* __restrict__ row_tab,   // [rows] offsets into tables, or null
    const int* __restrict__ tables,    // page ids by block
    const char* __restrict__ pool,
    long long page_bytes, long long layer_off,
    int kvh,
    float* __restrict__ bmax,          // [rows, kvh, bm_stride]
    int bm_stride,
    float scale,
    int kv8) {                         // 1: the pool is the fp8 layout (only where IK sits differs)
    const int b = blockIdx.x, g = blockIdx.y, r = blockIdx.z;
    const int p = row_pos(pos, pos0, r);
    if (b > p / BLOCK) return;
    const int page = __ldg(tables + row_table(row_tab, r) + b);
    const char* base = page_layer(pool, page_bytes, layer_off, page);
    const bf16* ik = reinterpret_cast<const bf16*>(base + ik_byte(kvh, kv8));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint2 qw = __ldg(reinterpret_cast<const uint2*>(iq + ((long long)r * kvh + g) * D) + lane);
    const float q0 = lo(qw.x), q1 = hi(qw.x), q2 = lo(qw.y), q3 = hi(qw.y);
    const int valid = min(BLOCK, p + 1 - b * BLOCK);
    float best = -INFINITY;
    for (int j = warp; j < valid; j += 4) {
        const uint2 kw = __ldg(reinterpret_cast<const uint2*>(ik + (long long)j * D) + lane);
        float a = __fmul_rn(q0, lo(kw.x));
        a = __fadd_rn(a, __fmul_rn(q1, hi(kw.x)));
        a = __fadd_rn(a, __fmul_rn(q2, lo(kw.y)));
        a = __fadd_rn(a, __fmul_rn(q3, hi(kw.y)));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, o));
        best = fmaxf(best, __fmul_rn(a, scale));
    }
    __shared__ float part[4];
    if (lane == 0) part[warp] = best;
    __syncthreads();
    if (threadIdx.x == 0)
        bmax[((long long)r * kvh + g) * bm_stride + b] = fmaxf(fmaxf(part[0], part[1]), fmaxf(part[2], part[3]));
}
