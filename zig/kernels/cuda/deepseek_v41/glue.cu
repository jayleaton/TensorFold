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
//   gather_cols  out[i][w c + j] = recv[w][i][j]                                      (permute(1, 0, 2).reshape)
//   carry        carry[j] = f32(src[row][j]), row = *accepted (device) or the host's   (the ratio-2 compressor carry)

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

}  // namespace dsv41_glue
