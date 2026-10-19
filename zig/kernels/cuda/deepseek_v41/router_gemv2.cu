// Router GEMV v2 (ours, TF_DSV41_RG_V2): gemv_kernel's chains, butterfly and fused tail (router_gemv.cu, included).
#include "router_gemv.cu"

namespace dsv41_rg {

// 4 bf16 (two words, k ascending) -> fp32, exactly
__device__ __forceinline__ void widen4(uint32_t lo, uint32_t hi, float (&f)[4]) {
    f[0] = __uint_as_float(lo << 16); f[1] = __uint_as_float(lo & 0xffff0000u);
    f[2] = __uint_as_float(hi << 16); f[3] = __uint_as_float(hi & 0xffff0000u);
}

// gemv_kernel's chain a (row, expert) a lane (k = 8 (l + 32 j) + i, j then i ascending), RT x EW of them a warp
template <int RT, int NW, int EW>
__global__ void __launch_bounds__(NW * 32) gemv2_kernel(const Args a) {
    // x a step ahead in shared memory, widened (fp32 [row][half][lane]: conflict-free); W in registers
    constexpr int XH = RT / NW / 2;                 // x vectors a thread stages a half step (RT rows x 32 lanes)
    static_assert(RT % (2 * NW) == 0 && RT <= 32, "rows a tile: a multiple of the warps, lane r stores row r");
    __shared__ __align__(16) float4 xsh[2][RT][2][32];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r0 = blockIdx.y * RT;
    const int nr = min(RT, a.R - r0);
    const int e0 = (blockIdx.x * NW + warp) * EW;   // E % EW == 0 (kernels_rg2.zig): a warp's experts all live or none
    // a dead warp reads expert 0's rows and a row past the tile its last row's: those chains are never stored
    const uint16_t* wp = a.w + (size_t)(e0 < a.E ? e0 : 0) * D + (size_t)lane * 8;
    const uint16_t* xp[2 * XH];
#pragma unroll
    for (int q = 0; q < 2 * XH; ++q) xp[q] = a.x + (size_t)(r0 + min(warp + NW * q, nr - 1)) * a.xs + (size_t)lane * 8;

    auto wload = [&](int j, uint4 (&dst)[EW]) {
#pragma unroll
        for (int e = 0; e < EW; ++e) dst[e] = ldg_stream(wp + (size_t)e * D + (size_t)j * 256);
    };
    auto xload = [&](int j, int h, uint4 (&xr)[XH]) {
#pragma unroll
        for (int t = 0; t < XH; ++t) xr[t] = __ldg(reinterpret_cast<const uint4*>(xp[h * XH + t] + (size_t)j * 256));
    };
    auto xstore = [&](int buf, int h, const uint4 (&xr)[XH]) {
#pragma unroll
        for (int t = 0; t < XH; ++t) {
            const int q = h * XH + t;
            float f[8];
            widen8(xr[t], f);
            xsh[buf][warp + NW * q][0][lane] = make_float4(f[0], f[1], f[2], f[3]);
            xsh[buf][warp + NW * q][1][lane] = make_float4(f[4], f[5], f[6], f[7]);
        }
    };

    float acc[EW][RT];
#pragma unroll
    for (int e = 0; e < EW; ++e)
#pragma unroll
        for (int r = 0; r < RT; ++r) acc[e][r] = 0.0f;
    uint4 wr[EW];
    wload(0, wr);
    uint4 xr[XH];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        xload(0, h, xr);
        xstore(0, h, xr);
    }
    __syncthreads();

#pragma unroll 1
    for (int j = 0; j < NV; ++j) {
        const int b = j & 1;
        const bool more = j + 1 < NV;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (more) xload(j + 1, h, xr);          // half the next step's x, stored into the other buffer below
            float wf[EW][4];
#pragma unroll
            for (int e = 0; e < EW; ++e) widen4(h ? wr[e].z : wr[e].x, h ? wr[e].w : wr[e].y, wf[e]);
            if (h == 1 && more) wload(j + 1, wr);   // every word of this step's W is widened: refill for the next
#pragma unroll
            for (int r = 0; r < RT; ++r) {
                const float4 xv = xsh[b][r][h][lane];
                const float xf[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int e = 0; e < EW; ++e) acc[e][r] = __fmaf_rn(xf[i], wf[e][i], acc[e][r]);
            }
            if (more) xstore(b ^ 1, h, xr);         // the buffer step j - 1 read: every thread is past its barrier
        }
        __syncthreads();
    }

#pragma unroll
    for (int e = 0; e < EW; ++e)
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            float v = acc[e][r];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
            if (lane == r && r < nr && e0 < a.E) a.lg[(size_t)(r0 + r) * a.E + e0 + e] = v;
        }
    static_assert(sizeof(xsh) >= TAIL_INTS * sizeof(int), "the tail's picks fit the staged x");
    if (a.select) tail<NW, false>(a, r0, nr, tid, reinterpret_cast<int*>(&xsh[0][0][0][0]));
}

template <int RT, int NW, int EW>
cudaError_t launch2(const Args& a, cudaStream_t stream) {
    const dim3 grid((unsigned)((a.E + NW * EW - 1) / (NW * EW)), (unsigned)((a.R + RT - 1) / RT));
    gemv2_kernel<RT, NW, EW><<<grid, NW * 32, 0, stream>>>(a);
    return cudaGetLastError();
}

}  // namespace dsv41_rg

// The instance kernels_rg2.zig resolves: a row tile of 16 (gemv_kernel<16, 8, 2>'s counters), 8 warps of 8 experts.
template __global__ void dsv41_rg::gemv2_kernel<16, 8, 8>(const dsv41_rg::Args);
