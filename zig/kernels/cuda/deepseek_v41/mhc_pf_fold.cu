// mHC prefill fold (ours, TF_DSV41_MHC_PF_FOLD): mhc_pf.cu's site + mhc_cuda.cu's finish in one launch, same bits.
#include "mhc_pf.cu"
#include "mhc_cuda.cu"

namespace dsv41_mhc_pf {

// site_kernel's statements (mhc_pf.cu, copied; comments joined), byte-checked against it by proto/mhc_fold_bench.cu
template <bool POST, int COLL, bool FN32, bool TAP>
__device__ __forceinline__ void site_body(const Args& a, unsigned char* sm) {

    uint16_t* xs = reinterpret_cast<uint16_t*>(sm);                          // [4][BM][PB]
    uint16_t* cs = reinterpret_cast<uint16_t*>(sm + Lay<FN32>::XS);           // [BM][PB]
    unsigned char* fns = sm + Lay<FN32>::XS + Lay<FN32>::CS;                 // [4][24][PB or PF]
    const int b = blockIdx.x, r0 = blockIdx.y * BM, t = threadIdx.x, R = a.R;

    // fn's slice of block b into shared memory, in flight through phase A
    {
        constexpr int QR = FN32 ? CB / 4 : CB / 8;                           // 16-byte pieces a (stream, mix) row
        const char* f = static_cast<const char*>(a.fn);
        for (int v = t; v < 4 * NMIX * QR; v += NT) {
            const int jm = v / QR, q = v % QR, j = jm / NMIX, m = jm % NMIX;
            const long long src = static_cast<long long>(m) * 4 * D + j * D + b * CB;
            if (FN32) cp16(fns + (jm * PF + q * 4) * 4, f + (src + q * 4) * 4);
            else      cp16(fns + (jm * PB + q * 8) * 2, f + (src + q * 8) * 2);
        }
        cp_commit();
    }

    // ---- phase A: every load of the thread's items first, then the post / collapse / taps and the stores ----
    uint4 xv[IPT][4];
    float br[IPT][8];
#pragma unroll
    for (int n = 0; n < IPT; ++n) {
        const int i = t + n * NT, r = r0 + i / G8, col = b * CB + (i % G8) * 8;
        const uint16_t* xr = a.x + static_cast<long long>(r < R ? r : 0) * a.xs + col;
#pragma unroll
        for (int s = 0; s < 4; ++s) xv[n][s] = r < R ? ld_plain(xr + s * D) : make_uint4(0u, 0u, 0u, 0u);
    }
    if (POST) {
        const char* gp = static_cast<const char*>(a.g);
        const int gsz = a.gbf16 ? 2 : 4;
#pragma unroll
        for (int n = 0; n < IPT; ++n) {
            const int i = t + n * NT, r = r0 + i / G8, col = b * CB + (i % G8) * 8;
            const long long off = static_cast<long long>(r < R ? r : 0) * D + col;
            load8f(gp + off * gsz, a.gbf16, br[n]);
            for (int rk = 1; rk < a.world; ++rk) {
                float gv[8];
                load8f(gp + (rk * a.gr + off) * gsz, a.gbf16, gv);
#pragma unroll
                for (int k = 0; k < 8; ++k) br[n][k] = __fadd_rn(br[n][k], gv[k]);
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) br[n][k] = bfr(br[n][k]);
        }
    }
#pragma unroll
    for (int n = 0; n < IPT; ++n) {
        const int i = t + n * NT, rl = i / G8, q = i % G8, r = r0 + rl, col = b * CB + q * 8;
        const bool ok = r < R;
        float x[4][8];
#pragma unroll
        for (int s = 0; s < 4; ++s) widen8(xv[n][s], x[s]);
        if (POST && ok) {
            float y[4][8];
#pragma unroll
            for (int s = 0; s < 4; ++s) post1(a, r, s, br[n], x, y[s]);
            uint16_t* xo = a.xout + static_cast<long long>(r) * 4 * D + col;
#pragma unroll
            for (int s = 0; s < 4; ++s) *reinterpret_cast<uint4*>(xo + s * D) = pack8(y[s]);
#pragma unroll
            for (int s = 0; s < 4; ++s)
#pragma unroll
                for (int k = 0; k < 8; ++k) x[s][k] = y[s][k];
        }
        float cv[8];
        if (COLL == 2 && ok) {
            const float q0 = __ldg(a.pre + r * 4), q1 = __ldg(a.pre + r * 4 + 1);
            const float q2 = __ldg(a.pre + r * 4 + 2), q3 = __ldg(a.pre + r * 4 + 3);
#pragma unroll
            for (int k = 0; k < 8; ++k)
                cv[k] = bfr(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(q0, x[0][k]), __fmul_rn(q1, x[1][k])),
                                                __fmul_rn(q2, x[2][k])), __fmul_rn(q3, x[3][k])));
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) cv[k] = x[0][k];
        }
        const uint4 cp = pack8(cv);
        if (ok) {
            *reinterpret_cast<uint4*>(a.c + static_cast<long long>(r) * D + col) = cp;
            if (POST && TAP) {
                float tp[8];
#pragma unroll
                for (int k = 0; k < 8; ++k)
                    tp[k] = bfr(__fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(x[0][k], x[1][k]), x[2][k]), x[3][k]), 0.25f));
                *reinterpret_cast<uint4*>(a.tap + static_cast<long long>(r) * a.ts + col) = pack8(tp);
            }
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) *reinterpret_cast<uint4*>(xs + (s * BM + rl) * PB + q * 8) = pack8(x[s]);
        *reinterpret_cast<uint4*>(cs + rl * PB + q * 8) = cp;
    }
    cp_wait_all();
    __syncthreads();

    // ---- phase B: row rl's chains, stream j: h 0 mixes 0-12; h 1 mixes 13-23, kind 24 and (stream 0) kind 25 ----
    const int rl = t % BM, j = (t / BM) % 4, h = t / (4 * BM), r = r0 + rl;
    const bool coll = h == 1 && j == 0;
    const uint16_t* vrow = xs + (j * BM + rl) * PB;
    const uint16_t* crow = cs + rl * PB;
    float acc[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s) acc[s] = 0.0f;
    if (h == 0) {
#pragma unroll 1
        for (int k8 = 0; k8 < G8; ++k8) {
            float v[8];
            widen8(*reinterpret_cast<const uint4*>(vrow + k8 * 8), v);
#pragma unroll
            for (int s = 0; s < 13; ++s) {
                float w[8];
                fn8<FN32>(fns, j, s, k8, w);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[s] = __fmaf_rn(v[k], w[k], acc[s]);
            }
        }
    } else {
#pragma unroll 1
        for (int k8 = 0; k8 < G8; ++k8) {
            float v[8];
            widen8(*reinterpret_cast<const uint4*>(vrow + k8 * 8), v);
#pragma unroll
            for (int s = 0; s < 11; ++s) {
                float w[8];
                fn8<FN32>(fns, j, 13 + s, k8, w);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[s] = __fmaf_rn(v[k], w[k], acc[s]);
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[11] = __fmaf_rn(v[k], v[k], acc[11]);
            if (coll) {
                float cvv[8];
                widen8(*reinterpret_cast<const uint4*>(crow + k8 * 8), cvv);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[12] = __fmaf_rn(cvv[k], cvv[k], acc[12]);
            }
        }
    }
    if (r < R) {
        float* pr = a.part + ((static_cast<long long>(r) * 4 + j) * NB + b) * PW;
        if (h == 0) {
#pragma unroll
            for (int s = 0; s < 13; ++s) pr[s] = acc[s];
        } else {
#pragma unroll
            for (int s = 0; s < 12; ++s) pr[13 + s] = acc[s];             // mixes 13-23, [24] the square sum
            if (coll) pr[25] = acc[12];
        }
    }
}

}  // namespace dsv41_mhc_pf

namespace dsv41_mhc_fold {

constexpr int NB = dsv41_mhc_pf::NB, BM = dsv41_mhc_pf::BM, NT = dsv41_mhc_pf::NT, D = dsv41_mhc_pf::D;
constexpr int ROWV = dsv41_mhc::MAXR * dsv41_mhc::ROWB;           // the finish's row sums, then its 2 stages
static_assert(BM <= dsv41_mhc::MAXR && NT == dsv41_mhc::NT && D == dsv41_mhc::D && NB == dsv41_mhc::NB, "tiles");
static_assert(ROWV + 2 * dsv41_mhc::STAGE4 * 16 <= dsv41_mhc_pf::Lay<false>::BYTES, "the finish fits the site");

// the site's arguments, then _finish_k's: base / scale, the next site's coefficients, the norm weight and output
struct Args {
    dsv41_mhc_pf::Args s;
    const float* base; const float* scale;     // [24], [3]
    float* opre; float* opost; float* ocomb;   // [R, 4], [R, 4], [R, 16]
    const void* nw; int nwbf16;                // [D]
    uint16_t* out;                             // [R, D] bf16, the next sublayer's input
    int* cnt;                                  // [ceil(R / 16)] zeroed; the last arrival of a row tile resets it
    float eps, hc_eps, post_alpha; int iters;
};

// CTA = mhc_pf's (column block, 16 rows); the 40th CTA of a row tile to finish runs finish<true> for its rows from L2
template <bool POST, int COLL, bool FN32, bool TAP>
__global__ void __launch_bounds__(NT, 2) fold_kernel(const Args a) {
    extern __shared__ __align__(16) unsigned char sm[];
    __shared__ int last;
    const int t = threadIdx.x, rt = blockIdx.y;
    dsv41_mhc_pf::site_body<POST, COLL, FN32, TAP>(a.s, sm);
    __threadfence();                                         // the decode boundary's ticket (mhc_cuda.cu)
    __syncthreads();
    if (t == 0) last = atomicAdd(a.cnt + rt, 1) == NB - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (t == 0) atomicExch(a.cnt + rt, 0);                   // self-cleaning
    const int r0 = rt * BM;
    dsv41_mhc::Args f{};
    f.part = a.s.part + static_cast<long long>(r0) * 4 * NB * dsv41_mhc::PW;
    f.c = a.s.c + static_cast<long long>(r0) * D;
    f.base = a.base; f.scale = a.scale;
    f.opre = a.opre + r0 * 4; f.opost = a.opost + r0 * 4; f.ocomb = a.ocomb + r0 * 16;
    f.nw = a.nw; f.nwbf16 = a.nwbf16;
    f.out = a.out + static_cast<long long>(r0) * D;
    f.R = min(BM, a.s.R - r0);
    f.eps = a.eps; f.hc_eps = a.hc_eps; f.post_alpha = a.post_alpha; f.iters = a.iters;
    dsv41_mhc::finish<true>(f, reinterpret_cast<float*>(sm), reinterpret_cast<float4*>(sm + ROWV), t);
}

template <bool POST, int COLL, bool FN32, bool TAP>
cudaError_t launch_one(const Args& a, cudaStream_t stream) {
    constexpr int SMEM = dsv41_mhc_pf::Lay<FN32>::BYTES;
    static bool attr = false;
    if (!attr) {
        const cudaError_t e = cudaFuncSetAttribute(fold_kernel<POST, COLL, FN32, TAP>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        if (e != cudaSuccess) return e;
        attr = true;
    }
    fold_kernel<POST, COLL, FN32, TAP><<<dim3(NB, (a.s.R + BM - 1) / BM), NT, SMEM, stream>>>(a);
    return cudaGetLastError();
}

// mhc_pf's modes (0 boundary, 1 site from stream 0, 2 site with a carried pre-mix), bf16 fn, no taps: prefill's sites
inline cudaError_t dispatch(const Args& a, int mode, cudaStream_t stream) {
    if (a.s.R < 1 || a.s.world < 1 || a.s.world > dsv41_mhc_pf::MAXW) return cudaErrorInvalidValue;
    switch (mode) {
    case 0: return launch_one<true, 2, false, false>(a, stream);
    case 1: return launch_one<false, 1, false, false>(a, stream);
    case 2: return launch_one<false, 2, false, false>(a, stream);
    default: return cudaErrorInvalidValue;
    }
}

}  // namespace dsv41_mhc_fold

template __global__ void dsv41_mhc_fold::fold_kernel<true, 2, false, false>(const dsv41_mhc_fold::Args);
template __global__ void dsv41_mhc_fold::fold_kernel<false, 1, false, false>(const dsv41_mhc_fold::Args);
template __global__ void dsv41_mhc_fold::fold_kernel<false, 2, false, false>(const dsv41_mhc_fold::Args);
