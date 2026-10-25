// MiniMax-M3.1-Flash forced-block top-16 (float64 reference: core/minimax_ref.zig forced, topk, selectBlocks).
// A row with nb = pos / 128 + 1 blocks attends all of them when nb <= 16, else the 16 best by (score desc, then block id
// by the tie rule) with forced blocks scored +inf. The forcing rules and the tie rule are UNVERIFIED upstream semantics,
// so they are launch parameters mirroring minimax_ref.Switches: init_n/init_inclusive (sparse_init_block count or
// inclusive), local_n/local_trailing (sparse_local_block as "own block if > 0" or "trailing window of n"), higher_tie.
// A NaN block score ranks as -inf (the reference's rule). Output: ascending block ids, -1 padded, per (row, kv head).
#include "minimax_m3_sparse.cuh"

using namespace tf_mm3;

namespace {

__device__ __forceinline__ bool forced(int b, int own, int init_n, int init_inclusive, int local_n, int local_trailing) {
    const bool init = init_inclusive ? b <= init_n : b < init_n;
    const bool local = local_trailing ? (b <= own && own - b < local_n) : (local_n > 0 && b == own);
    return init || local;
}

/// True when (sa, ia) ranks strictly before (sb, ib): larger score, then the tie rule's block id.
__device__ __forceinline__ bool before(float sa, int ia, float sb, int ib, int higher) {
    return sa > sb || (sa == sb && (higher ? ia > ib : ia < ib));
}

} // namespace

extern "C" __global__ void __launch_bounds__(256) tf_mm3_select(
    const float* __restrict__ bmax,    // [rows, kvh, bm_stride]
    const int* __restrict__ pos,
    int pos0,
    int* __restrict__ sel,             // [rows, kvh, 16]
    int kvh, int bm_stride,
    int init_n, int init_inclusive, int local_n, int local_trailing, int higher_tie) {
    const int g = blockIdx.x, r = blockIdx.y;
    const int p = row_pos(pos, pos0, r);
    const int nb = p / BLOCK + 1, own = nb - 1;
    int* out = sel + ((long long)r * kvh + g) * TOPK;
    const int t = threadIdx.x;
    if (nb <= TOPK) {
        if (t < TOPK) out[t] = t < nb ? t : -1;
        return;
    }
    const float* sc = bmax + ((long long)r * kvh + g) * bm_stride;
    __shared__ float ws[8];
    __shared__ int wi[8];
    __shared__ int chosen[TOPK];
    __shared__ float prev_s;
    __shared__ int prev_i;
    for (int k = 0; k < TOPK; ++k) {
        float bs = -INFINITY;
        int bi = -1;
        for (int b = t; b < nb; b += 256) {
            float s = forced(b, own, init_n, init_inclusive, local_n, local_trailing) ? INFINITY : __ldg(sc + b);
            if (s != s) s = -INFINITY;
            if (k > 0 && !before(prev_s, prev_i, s, b, higher_tie)) continue;
            if (bi < 0 || before(s, b, bs, bi, higher_tie)) { bs = s; bi = b; }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float os = __shfl_xor_sync(0xffffffffu, bs, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (oi >= 0 && (bi < 0 || before(os, oi, bs, bi, higher_tie))) { bs = os; bi = oi; }
        }
        if ((t & 31) == 0) { ws[t >> 5] = bs; wi[t >> 5] = bi; }
        __syncthreads();
        if (t == 0) {
            float s = ws[0];
            int i = wi[0];
            for (int w = 1; w < 8; ++w)
                if (wi[w] >= 0 && (i < 0 || before(ws[w], wi[w], s, i, higher_tie))) { s = ws[w]; i = wi[w]; }
            chosen[k] = i;
            prev_s = s;
            prev_i = i;
        }
        __syncthreads();
    }
    if (t < TOPK) {
        int rank = 0;
        for (int u = 0; u < TOPK; ++u) rank += chosen[u] < chosen[t];
        out[rank] = chosen[t];
    }
}
