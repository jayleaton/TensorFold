// DeepSeek-V4.1's keyed sampling, its device part (tensorfold-dsquant nucleus.py / slots.py, prod path). Built with
// --fmad=false: torch runs v / t, the subtraction and exp as separate float64 kernels, each result rounded.
//
//   stats  out[i] = (m, s): m = max_c f64(lg[i][c]), s = sum_c exp(f64(lg[i][c]) / t - m / t)   (nucleus.row_stats, one
//          temperature a window; each term as torch computes it, the sum's order is not torch's: nucleus.keep_count's
//          GUARD (1e-9) is what makes the choice independent of it)
//   pack   out[i] = [vals[i][0 .. k], f32 bits of int32 (cols[i][j] + id0)]                     (slots.cand_gather's
//          packed rows: topk_keys.cu's values and columns, the rank's first id added)

#include <cstdint>

namespace dsv41_sampling {

constexpr int NT = 1024;

__device__ __forceinline__ double warp_max(double v) {
    for (int o = 16; o > 0; o >>= 1) {
        const double u = __shfl_xor_sync(0xFFFFFFFFu, v, o);
        v = u > v ? u : v;
    }
    return v;
}

__device__ __forceinline__ double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    return v;
}

// One CTA a row: rows [rows, C] fp32 (row stride ld), out float64 [rows, 2].
__global__ void __launch_bounds__(NT) stats_kernel(const float* __restrict__ lg, long long ld, int C, double t,
                                                   double* __restrict__ out) {
    __shared__ double red[NT / 32];
    __shared__ double s_m;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* row = lg + (long long)blockIdx.x * ld;
    double m = -__longlong_as_double(0x7FF0000000000000LL);   // -inf
    for (int c = tid; c < C; c += NT) {
        const double v = (double)row[c];
        m = v > m ? v : m;
    }
    m = warp_max(m);
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (warp == 0) {
        double v = red[lane];
        v = warp_max(v);
        if (lane == 0) s_m = v;
    }
    __syncthreads();
    const double mx = s_m;
    const double mt = mx / t;                                  // (m / t)[:, None]
    double s = 0.0;
    for (int c = tid; c < C; c += NT) s += exp((double)row[c] / t - mt);
    s = warp_sum(s);
    __syncthreads();                                           // red is reused
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (warp == 0) {
        double v = red[lane];
        v = warp_sum(v);
        if (lane == 0) {
            out[2 * (long long)blockIdx.x] = mx;
            out[2 * (long long)blockIdx.x + 1] = v;
        }
    }
}

// One CTA a row: vals fp32 [rows, k], cols int64 [rows, k] -> out fp32 [rows, 2k].
__global__ void __launch_bounds__(256) pack_kernel(const float* __restrict__ vals, const long long* __restrict__ cols,
                                                   int k, int id0, float* __restrict__ out) {
    const long long r = blockIdx.x;
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        out[r * 2 * k + j] = vals[r * k + j];
        out[r * 2 * k + k + j] = __int_as_float((int)cols[r * k + j] + id0);
    }
}

}  // namespace dsv41_sampling
