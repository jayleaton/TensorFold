// TF_DSV41_ROT_FOLD (dense3.cu includes this): lanes_linear_kernel whose epilogue also writes the next rot_in's rows.

namespace dsv41_dense3 {

using namespace tf_exl3;

// The next rot_in's output: segment s's fp16 rows (stride ld) and suh, each at its first column (y's = next k)
struct RotOut {
    half* xh[MAXSEG];
    const half* suh[MAXSEG];
    int ld;
};

// One row's 128 columns of y (a warp, 4 a lane) as seg_rot_in_kernel would rotate them after reading y back.
__device__ __forceinline__ void rot_out(const float (&y)[4], int y_dtype, const half* suh, half* xh, int lane) {
    float v[4], sc[4];
    if (y_dtype == dsv41_x3seg::BF16) {                     // store4's rounding, then load4's widening
        const __nv_bfloat162 a = __floats2bfloat162_rn(y[0], y[1]), b = __floats2bfloat162_rn(y[2], y[3]);
        const float2 fa = __bfloat1622float2(a), fb = __bfloat1622float2(b);
        v[0] = fa.x; v[1] = fa.y; v[2] = fb.x; v[3] = fb.y;
    } else if (y_dtype == dsv41_x3seg::F16) {
        const half2 a = __floats2half2_rn(y[0], y[1]), b = __floats2half2_rn(y[2], y[3]);
        const float2 fa = __half22float2(a), fb = __half22float2(b);
        v[0] = fa.x; v[1] = fa.y; v[2] = fb.x; v[3] = fb.y;
    } else {
        v[0] = y[0]; v[1] = y[1]; v[2] = y[2]; v[3] = y[3];
    }
    // -- seg_rot_in_kernel's statements from here on (suh, the 128-point transform, the scale, the fp16 store) --
    dsv41_x3seg::load4(suh, dsv41_x3seg::F16, 4 * lane, sc);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= sc[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    dsv41_x3seg::store4(xh, dsv41_x3seg::F16, 4 * lane, v);
}

// lanes_linear_kernel's statements verbatim, plus the segment's RotOut entry and a rot_out after each final store
template <int K2, int CB, int WK, int MINB>
__global__ void __launch_bounds__(WK * 32, MINB) lanes_rot_kernel(const LinTable tab, int y_dtype, int M,
                                                                  const RotOut ro) {
    constexpr int G = group_steps(K2), GW = gwords<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int RH = min(M, 8);

    LinSeg s = tab.s[0];
    half* rxh = ro.xh[0];
    const half* rsuh = ro.suh[0];
#pragma unroll
    for (int i = 1; i < MAXSEG; ++i)
        if (i < tab.n && (int)blockIdx.x >= tab.s[i].p0) {
            s = tab.s[i];
            rxh = ro.xh[i];
            rsuh = ro.suh[i];
        }
    const half* __restrict__ xh = s.xh;
    const half* __restrict__ svh = s.svh;
    void* __restrict__ y = s.y;
    float* __restrict__ Z = s.Z;
    int* __restrict__ counters = s.counters;
    const int K = s.K, N = s.N, SK = s.SK, wk = s.wk, ld = s.ld;

    const int NB = N >> 7, q = (int)blockIdx.x - s.p0, nb = q % NB, split = q / NB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const bool on = warp < wk;                                // a WK = 4 segment in an 8-warp launch: warps 4-7 idle
    const int per_warp = (K >> 4) / SK / wk;
    const int kt0 = split * (per_warp * wk) + warp * per_warp;
    const int steps = on ? per_warp : 0, groups = steps / G;
    const uint32_t* base = s.T + nb * s.stride_nb + (size_t)kt0 * s.stride_k + 4 * lane;
    const size_t gstride = (size_t)G * s.stride_k;
    const int col0 = nb * 128;

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = min(16, M - m0);

        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        const half* x0 = xh + (size_t)r0 * K;
        const half* x1 = xh + (size_t)r1 * K;
        uint32_t cur[GW], nxt[GW];
        if (groups > 0) load_group(base, cur);
        if (pass == 0) pdl_wait();                            // weights above, xh / Z / counters below
        uint32_t a[4], an[4];
        if (steps > 0) load_a(x0, x1, kt0, t, a);
#pragma unroll 1
        for (int gi = 0; gi < groups; ++gi) {
            if (gi + 1 < groups) load_group(base + (size_t)(gi + 1) * gstride, nxt);
#pragma unroll
            for (int st = 0; st < G; ++st) {
                const int i = gi * G + st;
                if (i + 1 < steps) load_a(x0, x1, kt0 + i + 1, t, an);   // the next step's input, one step early
#pragma unroll
                for (int j = 0; j < 8; j += 2) {
                    uint32_t b[2][2][2];
                    pair_frags<K2, CB>(cur, st, j, lane, b);
                    mma16816(acc[j][0], a, b[0][0]);
                    mma16816(acc[j][1], a, b[0][1]);
                    mma16816(acc[j + 1][0], a, b[1][0]);
                    mma16816(acc[j + 1][1], a, b[1][1]);
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = an[c];
            }
#pragma unroll
            for (int c = 0; c < GW; ++c) cur[c] = nxt[c];
        }

        if (m0 + 16 >= M) pdl_trigger();                     // the last pass's weights are read

        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();
            if (on && g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
                    for (int w = 1; w < wk; ++w) {
                        const float4 qv = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += qv.x; v[1] += qv.y; v[2] += qv.z; v[3] += qv.w;
                    }
                    dsv41_x3seg::finish(v, lane, svh, nullptr, col0 + 4 * lane);
                    dsv41_x3seg::store4(y, y_dtype, (size_t)(m0 + rlo + r) * ld + col0 + 4 * lane, v);
                    rot_out(v, y_dtype, rsuh + col0, rxh + (size_t)(m0 + rlo + r) * ro.ld + col0, lane);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 sm = *reinterpret_cast<const float4*>(red + r * 128 + c);
                    for (int w = 1; w < wk; ++w) {
                        const float4 qv = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        sm.x += qv.x; sm.y += qv.y; sm.z += qv.z; sm.w += qv.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = sm;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 sm = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int qq = 1; qq < SK; ++qq) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)qq * M * N + at));
                        sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w;
                    }
                    float v[4] = {sm.x, sm.y, sm.z, sm.w};
                    dsv41_x3seg::finish(v, lane, svh, nullptr, col0 + 4 * lane);
                    dsv41_x3seg::store4(y, y_dtype, (size_t)(m0 + r) * ld + col0 + 4 * lane, v);
                    rot_out(v, y_dtype, rsuh + col0, rxh + (size_t)(m0 + r) * ro.ld + col0, lane);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;
            }
        }
        __syncthreads();
    }
}

}  // namespace dsv41_dense3
