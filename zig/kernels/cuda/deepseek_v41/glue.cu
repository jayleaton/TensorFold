// The window's glue (tensorfold-dsquant forward.py / pick.py, prod path): the torch ops between the kernels of a
// window, each element computed as torch's CUDA kernels compute it (fp32 opmath; casts through __float2bfloat16 /
// __float2half, the intrinsics c10::BFloat16 / c10::Half use on CUDA). Built with --fmad=false: none has a
// multiply-add, and torch runs each as its own kernel, every intermediate rounded where it is stored.
//
//   embed_send   rows[i] = 0 <= ids[i] - lo < V ? bf16_rn(f32 embed[ids[i] - lo]) : +0  (index_select .to(f32), where, .to(bf16):
//                exact but for NaN, which the round trip makes canonical 0x7FFF)
//   embed_sum    out[i] = repeat x4 of bf16_rn(((f32 recv[0] + f32 recv[1]) + ...))    (allsum: clone + adds, .to)
//   cast_bf16    y = bf16_rn(x)                                                       (the MoE partial, .to(bf16))
//   kit_weights  y = f32(f16_rn(x))                                                   (pick.kit_weights)
//   kit_logits   x = f32(bf16_rn(x)) in place                                         (pick.kit_logits)
//   positions    pos[0] = start; pos64[i] = start + i; lo[i] = 0                       (Win.make)
//   positions_dev  the same from a device start (a graphed window: staged before each replay)
//   widen_f32    y = f32(x) of bf16 x                                                (q.float(): exact, the DSpark pass)
//   gather_cols  out[i][w c + j] = recv[w][i][j]                                      (permute(1, 0, 2).reshape)
//   carry        carry[j] = f32(src[row][j]), row = *accepted (device) or the host's   (the ratio-2 compressor carry)
//   ds_cands     DSpark's candidates (dspark.candidates / draft/dspark.zig): each row's best k fp32 logits by (value
//                desc, column asc), -0.0 == +0.0, as [vals k | ids k as int bits] (an exact selection: integers only)
//   pick_pack    pairs[r] = (f64 vals[r], f64 (cols[r] + id0))     (forward.greedy's pair: a rank's best, topk_keys k 1)
//   pick_merge   out[r] = the id of the best pair over the ranks [W][n][2] (higher value; on a tie or NaN the lower
//                rank: forward.greedy's merge, strict >); out[n] = accepted (rows 0.. whose pick is the next row's id:
//                a chain window's kept drafts), out[n + 1] = bonus = out[accepted], out[n + 2] = start + accepted + 1

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace dsv41_glue {

__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }
__device__ __forceinline__ uint16_t to_bf(float x) { return __bfloat16_as_ushort(__float2bfloat16(x)); }

// one thread a (row, column): rows [n, D] bf16 from the rank's embedding slice [V, D]
__global__ void __launch_bounds__(256) embed_send_kernel(const long long* __restrict__ ids, int n,
                                                         const uint16_t* __restrict__ embed, long long vocab_lo, int V,
                                                         int D, uint16_t* __restrict__ rows) {
    const long long total = (long long)n * D;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (long long)gridDim.x * blockDim.x) {
        const long long i = e / D, d = e - i * D;
        const long long local = ids[i] - vocab_lo;
        rows[e] = (local >= 0 && local < V) ? to_bf(bf(embed[local * D + d])) : (uint16_t)0;
    }
}

// recv [W, n, D] bf16 (the ranks' rows, rank order) -> out [n, 4 D] bf16
__global__ void __launch_bounds__(256) embed_sum_kernel(const uint16_t* __restrict__ recv, int W, int n, int D,
                                                        uint16_t* __restrict__ out) {
    const long long total = (long long)n * D, plane = total;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (long long)gridDim.x * blockDim.x) {
        float s = bf(recv[e]);
        for (int w = 1; w < W; ++w) s = __fadd_rn(s, bf(recv[w * plane + e]));
        const uint16_t b = to_bf(s);
        const long long i = e / D, d = e - i * D;
        uint16_t* o = out + i * 4 * D + d;
#pragma unroll
        for (int r = 0; r < 4; ++r) o[r * D] = b;
    }
}

__global__ void __launch_bounds__(256) cast_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y,
                                                        long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        y[i] = to_bf(x[i]);
}

__global__ void __launch_bounds__(256) kit_weights_kernel(const float* __restrict__ x, float* __restrict__ y,
                                                          long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        y[i] = __half2float(__float2half(x[i]));
}

__global__ void __launch_bounds__(256) kit_logits_kernel(float* __restrict__ x, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        x[i] = bf(to_bf(x[i]));
}

__global__ void __launch_bounds__(256) positions_kernel(int start, int n, int* __restrict__ pos,
                                                        long long* __restrict__ pos64, int* __restrict__ lo) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) pos[0] = start;
    if (i < n) {
        pos64[i] = (long long)start + i;
        lo[i] = 0;
    }
}

// a graphed window's positions: `start` from the window's staged statics, so one captured graph serves every window
__global__ void __launch_bounds__(256) positions_dev_kernel(const int* __restrict__ start, int n, int* __restrict__ pos,
                                                            long long* __restrict__ pos64, int* __restrict__ lo) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = *start;
    if (i == 0) pos[0] = s;
    if (i < n) {
        pos64[i] = (long long)s + i;
        lo[i] = 0;
    }
}

// recv [W, n, c] -> out [n, W c] (bf16 words, any 2-byte type)
__global__ void __launch_bounds__(256) gather_cols_kernel(const uint16_t* __restrict__ recv, int W, int n, int c,
                                                          uint16_t* __restrict__ out) {
    const long long total = (long long)W * n * c;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (long long)gridDim.x * blockDim.x) {
        const long long i = e / ((long long)W * c), r = e - i * W * c, w = r / c, j = r - w * c;
        out[e] = recv[(w * n + i) * c + j];
    }
}

// carry [cols] fp32 <- bf16 row `row` of src (row stride ld elements); the row from the device when `accepted` is set
__global__ void __launch_bounds__(256) carry_kernel(const uint16_t* __restrict__ src, long long ld,
                                                    const int* __restrict__ accepted, int row, int cols,
                                                    float* __restrict__ carry) {
    const int r = accepted ? *accepted : row;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < cols; j += gridDim.x * blockDim.x)
        carry[j] = bf(src[(long long)r * ld + j]);
}

// bf16 -> fp32 (exact: the bits shifted into the high half), n elements
__global__ void __launch_bounds__(256) widen_f32_kernel(const uint16_t* __restrict__ x, float* __restrict__ y,
                                                        long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        y[i] = bf(x[i]);
}

// DSpark's candidates: one CTA a row of fp32 logits [rows, V] (row stride ld). Keys
//   u = (mono(bits(x + 0.0)) << 32) | (0xFFFFFFFF - col)
// are unique in a row and order rows by (value desc, column asc) with -0.0 == +0.0, the host rule (dspark.candidates).
// A radix select of the k-th largest key, 8 bits a pass from the top, stops at the pass whose digit holds exactly the
// keys still needed (after 4 passes when the k-th value is unique); the winners are the keys whose selected top bits are
// >= the prefix, exactly c = min(k, V) of them; a bitonic sort (descending) gives the host's order. Out: packed
// [rows][2 k] fp32: values then id0 + column as int bits; slots past c: -inf and id -1 (the host's padding).
constexpr int DS_NT = 1024, DS_KMAX = 128;

__device__ __forceinline__ unsigned long long ds_key(float x, int col) {
    const unsigned int b = __float_as_uint(x + 0.0f);                      // -0.0 + 0.0 = +0.0
    const unsigned int mono = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return (static_cast<unsigned long long>(mono) << 32) | (0xFFFFFFFFull - static_cast<unsigned int>(col));
}

__global__ void __launch_bounds__(DS_NT) ds_cands_kernel(const float* __restrict__ lg, long long ld, int V, int k,
                                                         int id0, float* __restrict__ out) {
    __shared__ int hist[256];
    __shared__ unsigned long long s_prefix, s_mask;
    __shared__ int s_need, s_done, s_n;
    __shared__ unsigned long long key_sh[DS_KMAX];
    const int tid = threadIdx.x;
    const float* row = lg + (long long)blockIdx.x * ld;
    float* o = out + (long long)blockIdx.x * 2 * k;
    const int c = V < k ? V : k;
    if (tid == 0) { s_prefix = 0; s_mask = 0; s_need = c; s_done = 0; }
    __syncthreads();
    for (int shift = 56; shift >= 0 && c > 0; shift -= 8) {
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        const unsigned long long prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < V; i += DS_NT) {
            const unsigned long long u = ds_key(row[i], i);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int need = s_need, d = 255;
            for (; d > 0 && hist[d] < need; --d) need -= hist[d];
            s_need = need;
            s_prefix = prefix | ((unsigned long long)d << shift);
            s_mask = mask | (255ull << shift);
            s_done = hist[d] == need;                                       // the digit's keys: all winners
        }
        __syncthreads();
        if (s_done) break;
    }
    const unsigned long long prefix = s_prefix, mask = s_mask;
    if (tid == 0) s_n = 0;
    __syncthreads();
    for (int i = tid; i < V && c > 0; i += DS_NT) {
        const unsigned long long u = ds_key(row[i], i);
        if ((u & mask) >= prefix) {
            const int at = atomicAdd(&s_n, 1);
            if (at < DS_KMAX) key_sh[at] = u;
        }
    }
    __syncthreads();
    int m = 1;
    while (m < c) m <<= 1;
    for (int i = c + tid; i < m; i += DS_NT) key_sh[i] = 0;                // sorts last
    __syncthreads();
    for (int size = 2; size <= m; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < m; i += DS_NT) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0;
                    const unsigned long long a = key_sh[i], b = key_sh[j];
                    if (desc ? a < b : a > b) { key_sh[i] = b; key_sh[j] = a; }
                }
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < k; i += DS_NT) {
        if (i < c) {
            const int col = (int)(0xFFFFFFFFull - (key_sh[i] & 0xFFFFFFFFull));
            o[i] = row[col];
            o[k + i] = __int_as_float(id0 + col);
        } else {
            o[i] = -__int_as_float(0x7F800000);
            o[k + i] = __int_as_float(-1);
        }
    }
}

// one thread a row
__global__ void __launch_bounds__(256) pick_pack_kernel(const float* __restrict__ vals, const long long* __restrict__ cols,
                                                        int n, long long id0, double* __restrict__ pairs) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    pairs[2 * r] = static_cast<double>(vals[r]);
    pairs[2 * r + 1] = static_cast<double>(cols[r] + id0);
}

// one block: a thread a row picks, then thread 0 walks the chain (n <= 1024)
__global__ void __launch_bounds__(1024) pick_merge_kernel(const double* __restrict__ all, int W, int n,
                                                          const long long* __restrict__ ids, long long start,
                                                          long long* __restrict__ out) {
    const int r = threadIdx.x;
    if (r < n) {
        int bk = 0;
        for (int k = 1; k < W; k++)
            if (all[2 * (k * n + r)] > all[2 * (bk * n + r)]) bk = k;
        out[r] = static_cast<long long>(all[2 * (bk * n + r) + 1]);
    }
    __syncthreads();
    if (r == 0) {
        int acc = 0;
        if (ids != nullptr)
            while (acc + 1 < n && ids[acc + 1] == out[acc]) acc++;
        out[n] = acc;
        out[n + 1] = out[acc];
        out[n + 2] = start + acc + 1;
    }
}

}  // namespace dsv41_glue
