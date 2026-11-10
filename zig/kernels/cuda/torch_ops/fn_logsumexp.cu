// torch.logsumexp(x, -1, keepdim=True) on fp32 rows [R, N] with torch 2.13's bits: ATen's reduce config and order.
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kMaxThreads = 512;

__device__ __forceinline__ float tf_lse_term(const float* row, int64_t at, float m) {
    return expf(__fsub_rn(row[at], m));
}

// The lane tree of a (bw, bh) block: shared memory halvings down to 32 lanes, then warp shuffles from 16 down.
__device__ float tf_lse_lane_tree(float v, float* shared) {
    int dim_x = blockDim.x;
    const int me = threadIdx.x + threadIdx.y * blockDim.x;
    if (dim_x > 32) {
        shared[me] = v;
        for (int offset = dim_x / 2; offset >= 32; offset >>= 1) {
            __syncthreads();
            if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
                v = __fadd_rn(v, shared[me + offset]);
                shared[me] = v;
            }
        }
        dim_x = 32;
    }
    __syncthreads();
    for (int offset = dim_x >> 1; offset > 0; offset >>= 1)
        v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, offset, 32));
    return v;
}

// The warp tree over threadIdx.y at each lane.
__device__ float tf_lse_warp_tree(float v, float* shared) {
    const int me = threadIdx.x + threadIdx.y * blockDim.x;
    shared[me] = v;
    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
        __syncthreads();
        if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
            v = __fadd_rn(v, shared[threadIdx.x + (threadIdx.y + offset) * blockDim.x]);
            shared[me] = v;
        }
    }
    return v;
}

}  // namespace

// Row maxima with NaN winning, then |max| == inf set to 0 (torch's masked_fill before the subtraction).
extern "C" __global__ void tf_fn_lse_max_kernel(const float* x, float* maxes, uint32_t rows, uint32_t n) {
    __shared__ float best[256];
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    float m = -INFINITY;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[uint64_t(row) * n + i];
        m = (v != v || m != m) ? (m != m ? m : v) : fmaxf(m, v);
    }
    best[threadIdx.x] = m;
    __syncthreads();
    for (uint32_t half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            const float a = best[threadIdx.x], b = best[threadIdx.x + half];
            best[threadIdx.x] = (a != a) ? a : (b != b) ? b : fmaxf(a, b);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float top = best[0];
        maxes[row] = fabsf(top) == INFINITY ? 0.0f : top;
    }
}

// sum(exp(x - m)) in ATen's order; `partial[row * ctas + blockIdx.y]` per CTA.
extern "C" __global__ void tf_fn_lse_sum_kernel(const float* x, const float* maxes, float* partial, uint32_t rows,
                                                uint32_t n, uint32_t ctas, uint32_t split) {
    __shared__ float shared[kMaxThreads];
    const int64_t bw = blockDim.x, bh = blockDim.y;
    const int64_t out_idx = split ? int64_t(blockIdx.x) : int64_t(threadIdx.y) + int64_t(blockIdx.x) * bh;
    const int64_t first = split ? int64_t(threadIdx.x) + int64_t(threadIdx.y) * bw + int64_t(blockIdx.y) * bw * bh
                                : int64_t(threadIdx.x);
    const int64_t stride = split ? bw * bh * int64_t(ctas) : bw;
    const bool owns_tail = (!split || threadIdx.y == 0) && (ctas == 1 || blockIdx.y == 0);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (out_idx < rows) {
        const float* row = x + out_idx * int64_t(n);
        const float m = maxes[out_idx];
        int64_t end = n, base = 0;
        int shift = int((uint64_t(out_idx) * n) % 4);
        if (shift > 0) {
            if (int(threadIdx.x) >= shift && threadIdx.x < 4 && owns_tail)
                acc[0] = __fadd_rn(0.0f, tf_lse_term(row, int64_t(threadIdx.x) - shift, m));
            end += shift - 4;
            base = 4 - shift;
        }
        int64_t idx = first;
        while (idx * 4 + 3 < end) {
            for (int i = 0; i < 4; i++) acc[i] = __fadd_rn(acc[i], tf_lse_term(row, base + idx * 4 + i, m));
            idx += stride;
        }
        const int64_t tail = end - end % 4 + threadIdx.x;
        if (owns_tail && tail < end) acc[0] = __fadd_rn(acc[0], tf_lse_term(row, base + tail, m));
        for (int i = 1; i < 4; i++) acc[0] = __fadd_rn(acc[0], acc[i]);
    }
    float v = tf_lse_lane_tree(acc[0], shared);
    if (split) v = tf_lse_warp_tree(v, shared);
    if (threadIdx.x == 0 && (!split || threadIdx.y == 0) && out_idx < rows)
        partial[out_idx * int64_t(ctas) + blockIdx.y] = v;
}

// log(sum) + m; with ctas > 1 the partials first, by the global-reduce tree of a (bw, bh) block.
extern "C" __global__ void tf_fn_lse_finish_kernel(const float* partial, const float* maxes, float* out,
                                                   uint32_t rows, uint32_t ctas) {
    __shared__ float shared[kMaxThreads];
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    float s;
    if (ctas == 1) {
        s = partial[row];
    } else {
        const uint32_t threads = blockDim.x * blockDim.y;
        float v = 0.0f;
        for (uint32_t t = threadIdx.x + threadIdx.y * blockDim.x; t < ctas; t += threads)
            v = __fadd_rn(v, partial[uint64_t(row) * ctas + t]);
        v = tf_lse_warp_tree(v, shared);
        s = tf_lse_lane_tree(v, shared);
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) out[row] = __fadd_rn(logf(s), maxes[row]);
}

// ---- development launchers (ctypes parity checks only) ----

static int tf_lse_last_pow2(int v) {
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    const int p = v - (v >> 1);
    return p > 1 ? p : 1;
}

static int64_t tf_lse_div_up(int64_t a, int64_t b) { return (a + b - 1) / b; }

// out: bw, bh, grid_x, ctas, split. Returns nonzero for shapes outside this kernel's contract (N < 128, R < 1).
extern "C" int tf_fn_logsumexp_config(uint64_t rows, uint64_t n, int num_mp, int max_threads_mp, uint32_t* out) {
    if (rows < 1 || n < 128 || rows > 65535 || n > (uint64_t(1) << 31)) return 1;
    const int64_t dim0 = int64_t(n) / 4, dim1 = int64_t(rows);
    const int d0 = dim0 < kMaxThreads ? tf_lse_last_pow2(int(dim0)) : kMaxThreads;
    const int d1 = dim1 < kMaxThreads ? tf_lse_last_pow2(int(dim1)) : kMaxThreads;
    int bw = d0 < 32 ? d0 : 32;
    const int bh = d1 < kMaxThreads / bw ? d1 : kMaxThreads / bw;
    bw = d0 < kMaxThreads / bh ? d0 : kMaxThreads / bh;
    int64_t step = bw, rows_per_block = 1;
    const int64_t vpt = tf_lse_div_up(int64_t(n), step);
    const int64_t threshold = 16 * bh < 256 ? 16 * bh : 256;
    const bool split = vpt >= threshold;
    if (split) step *= bh;
    else rows_per_block = bh;
    const int64_t grid_x = tf_lse_div_up(int64_t(rows), rows_per_block);
    const int64_t target = int64_t(num_mp) * (max_threads_mp / (bw * bh));
    int64_t ctas = 1;
    const int64_t v = tf_lse_div_up(int64_t(n), step);
    if (split && v >= 256 && grid_x <= target) {
        const int64_t c1 = tf_lse_div_up(target, grid_x), c2 = tf_lse_div_up(v, 16), c3 = tf_lse_div_up(v, 256);
        ctas = (c1 < c2 ? c1 : c2);
        if (c3 > ctas) ctas = c3;
    }
    out[0] = uint32_t(bw);
    out[1] = uint32_t(bh);
    out[2] = uint32_t(grid_x);
    out[3] = uint32_t(ctas);
    out[4] = split ? 1u : 0u;
    return 0;
}

// scratch: rows * (1 + ctas) floats (maxes, then partials).
extern "C" cudaError_t tf_fn_logsumexp(const float* x, float* out, float* scratch, uint64_t rows, uint64_t n,
                                       cudaStream_t stream) {
    int device = 0;
    cudaError_t rc = cudaGetDevice(&device);
    if (rc != cudaSuccess) return rc;
    int mp = 0, mtpm = 0;
    if ((rc = cudaDeviceGetAttribute(&mp, cudaDevAttrMultiProcessorCount, device)) != cudaSuccess) return rc;
    if ((rc = cudaDeviceGetAttribute(&mtpm, cudaDevAttrMaxThreadsPerMultiProcessor, device)) != cudaSuccess) return rc;
    uint32_t c[5];
    if (!x || !out || !scratch || tf_fn_logsumexp_config(rows, n, mp, mtpm, c)) return cudaErrorInvalidValue;
    float* maxes = scratch;
    float* partial = scratch + rows;
    tf_fn_lse_max_kernel<<<uint32_t(rows), 256, 0, stream>>>(x, maxes, uint32_t(rows), uint32_t(n));
    tf_fn_lse_sum_kernel<<<dim3(c[2], c[3]), dim3(c[0], c[1]), 0, stream>>>(x, maxes, partial, uint32_t(rows),
                                                                          uint32_t(n), c[3], c[4]);
    tf_fn_lse_finish_kernel<<<uint32_t(rows), dim3(c[0], c[1]), 0, stream>>>(partial, maxes, out, uint32_t(rows), c[3]);
    return cudaGetLastError();
}
