// pfdense + the next projection's rotation (ours, TF_DSV41_PF_WOROT; and rot2: TF_DSV41_PF_ROT2), rot_in's bits.
#include "pfdense.cuh"
#include "rot_exact.cuh"

namespace dsv41_pfrot {

using dsv41_pfd::Args;
using dsv41_pfd::Cfg;
using dsv41_pfd::Lay;

// the rotated copy of this GEMM's output for the next projection: suh / xh at the output view's column 0
struct RotOut {
    const half* suh;          // [N] fp16, the next projection's suh slice
    half* xh;                 // [M, x_stride] fp16 at the same column
    long long x_stride;
};

// pfd_kernel's statements (pfdense.cuh, copied), then rot_in of the CTA's BM x 128 bf16 output block into r.xh
template <int K2, bool LANES, class C>
__global__ void __launch_bounds__(C::THREADS, (Lay<K2, LANES, C>::MINB)) pfdr_kernel(const Args a, const RotOut r) {
    using namespace dsv41_pfd;

    using L = Lay<K2, LANES, C>;
    constexpr int BM = C::BM, KS = C::KS, NST = C::NST, CPR = C::CPR, LDA = C::LDA, MT = C::MT, NT = C::NT;
    constexpr int TW = L::TW;
    extern __shared__ __align__(128) unsigned char smem[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int g = lane >> 2, t = lane & 3;
    const int M = a.M, K = a.K, NB = a.N >> 7, S = K / (16 * KS);

    // Triton _gemm's raster: GROUP row tiles a column sweep
    const int pid = blockIdx.x, nm = (M + BM - 1) / BM;
    const int per = a.group * NB, first = (pid / per) * a.group;
    const int rows = min(nm - first, a.group);
    const int pm = first + (pid % per) % rows, pn = (pid % per) / rows;
    const int m0 = pm * BM;
    const uint32_t* Tb = a.T + (size_t)pn * a.stride_nb;
    const half* xh = static_cast<const half*>(a.xh);
    unsigned char* bdec = smem + L::RING;

    // stage s's xh rows into ring slot s % NST (rows >= M: zero-filled, never stored)
    auto load_a = [&](int s) {
        half* A = reinterpret_cast<half*>(smem + (s % NST) * L::STAGE);
        const int k0 = s * KS * 16;
        for (int i = threadIdx.x; i < BM * CPR; i += C::THREADS) {
            const int r = i / CPR, c = i % CPR;
            const bool ok = m0 + r < M;
            cp_async16(A + r * LDA + swz<CPR>(r, c) * 8, xh + (size_t)(ok ? m0 + r : 0) * K + k0 + c * 8, ok);
        }
    };
    // stage s's words (KS k steps x 8 tiles, as stored) into ring slot `slot`
    auto load_w = [&](int s, int slot) {
        uint32_t* W = reinterpret_cast<uint32_t*>(smem + slot * L::STAGE + L::A_BYTES);
        for (int i = threadIdx.x; i < KS * 8 * K2; i += C::THREADS) {
            const int kk = i / (8 * K2), q = i % (8 * K2);
            cp_async16(W + kk * 8 * TW + q * 4, Tb + (size_t)(s * KS + kk) * a.stride_k + q * 4, true);
        }
    };
    auto words_of = [&](int slot) {
        return reinterpret_cast<const uint32_t*>(smem + slot * L::STAGE + L::A_BYTES);
    };
    auto bbuf = [&](int b) { return reinterpret_cast<uint4*>(bdec + b * L::BBUF); };

    float acc[MT][2 * NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 2 * NT; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;

    // prologue: W(0) in slot NST - 1 (free until iteration 0 refills it), then groups G_s = {A(s), W(s + 1)}
    load_w(0, NST - 1);
    cp_commit();
#pragma unroll
    for (int s = 0; s < NST - 1; ++s) {
        if (s < S) load_a(s);
        if (s + 1 < S) load_w(s + 1, s);
        cp_commit();
    }
    cp_wait<NST - 1>();
    bar_sync();
    decode_stage<K2, LANES, C>(words_of(NST - 1), bbuf(0), warp, lane);

#pragma unroll 1
    for (int s = 0; s < S; ++s) {
        cp_wait<NST - 2>();                                  // G_s landed: A(s), W(s + 1)
        bar_sync();                                          // ... for every thread; B(s) decoded; iteration s - 1 done
        {
            const int f = s + NST - 1;                       // refill the slot iteration s - 1 used
            if (f < S) load_a(f);
            if (f + 1 < S) load_w(f + 1, f % NST);
            cp_commit();
        }
        if (s + 1 < S) decode_stage<K2, LANES, C>(words_of(s % NST), bbuf((s + 1) & 1), warp, lane);
        const half* A = reinterpret_cast<const half*>(smem + (s % NST) * L::STAGE);
        const uint4* D = bbuf(s & 1);
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            uint32_t af[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const int r = (wm * MT + mt) * 16 + (lane & 15);
                ldsm4(af[mt], A + r * LDA + swz<CPR>(r, 2 * kk + (lane >> 4)) * 8);
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint4 bb = D[(kk * 8 + wn * NT + nt) * 32 + lane];
                const uint32_t b0[2] = {bb.x, bb.y}, b1[2] = {bb.z, bb.w};
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mma16816(acc[mt][2 * nt], af[mt], b0);
                    mma16816(acc[mt][2 * nt + 1], af[mt], b1);
                }
            }
        }
    }
    cp_wait<0>();
    bar_sync();                                              // the ring and the B buffers become the epilogue's planes

    // epilogue, round wm: this warp row's rows. Earlier rounds: wait them out (2 barriers each)
    __nv_bfloat16* Et = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Er = Et + C::EPR * 128;
    for (int r = 0; r < wm; ++r) {
        bar_sync();
        bar_sync();
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < 2 * NT; ++j) {
            const int col = 16 * (wn * NT + (j >> 1)) + 8 * (j & 1) + 2 * t;
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const float c0 = acc[mt][j][2 * hh], c1 = acc[mt][j][2 * hh + 1];
                const __nv_bfloat162 top = __floats2bfloat162_rn(c0, c1);
                const __nv_bfloat162 rest = __floats2bfloat162_rn(__fsub_rn(c0, __low2float(top)),
                                                                  __fsub_rn(c1, __high2float(top)));
                const int row = mt * 16 + g + 8 * hh;
                const int at = row * 128 + esw(row, col >> 3) * 8 + (col & 7);
                *reinterpret_cast<__nv_bfloat162*>(Et + at) = top;
                *reinterpret_cast<__nv_bfloat162*>(Er + at) = rest;
            }
        }
    bar_sync();
    {
        const uint32_t P = had_base(lane);
        const int part = wn % C::NSPLIT;
#pragma unroll
        for (int gi = 0; gi < C::GPW; ++gi) {
            const int grp = (wn / C::NSPLIT) * C::GPW + gi;
            float y[C::EN8][4];
#pragma unroll
            for (int e = 0; e < C::EN8; ++e)
#pragma unroll
                for (int c = 0; c < 4; ++c) y[e][c] = 0.f;
#pragma unroll
            for (int q = 0; q < 16; ++q) {                   // top's k16 steps 0-7, then rest's 0-7: one chain
                const int kb = q & 7;
                const int row = grp * 16 + (lane & 15);
                uint32_t af[4];
                ldsm4(af, (q < 8 ? Et : Er) + row * 128 + esw(row, 2 * kb + (lane >> 4)) * 8);
#pragma unroll
                for (int e = 0; e < C::EN8; ++e) {
                    uint32_t hb[2];
                    had_frag(P, kb, part * C::EN8 + e, hb);
                    mma16816_bf16(y[e], af, hb);
                }
            }
#pragma unroll
            for (int e = 0; e < C::EN8; ++e) {
                const int col = pn * 128 + 8 * (part * C::EN8 + e) + 2 * t;
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int row = m0 + wm * C::EPR + grp * 16 + g + 8 * hh;
                    if (row < M) store2(a, row, col, y[e][2 * hh], y[e][2 * hh + 1]);
                }
            }
        }
    }
    bar_sync();
    for (int r = wm + 1; r < C::WM; ++r) {
        bar_sync();
        bar_sync();
    }

    // every warp passed the last round's barrier after its stores: the block is complete and visible to the CTA
    for (int rr = warp; rr < BM && m0 + rr < M; rr += C::NW) {
        const long long row = m0 + rr;
        const int col = pn * 128 + 4 * lane;
        float x[4], s[4], v[4];
        dsv41_rot::load4x(a.out, true, (size_t)(row * a.o_stride + col), x);
        dsv41_rot::load4x(r.suh, false, (size_t)col, s);
        dsv41_rot::rot128(x, s, lane, v);
        dsv41_rot::store4h(r.xh, (size_t)(row * r.x_stride + col), v);
    }
}

// rot_in for two projections of one input: xh0 / xh1 = (x * suh0 / suh1) H / sqrt 128, x read once
struct Rot2Args {
    const void* x;            // [M, K] bf16 (x_bf16) or fp16, contiguous
    const half* suh0;
    const half* suh1;
    half* xh0;                // [M, K] fp16 each
    half* xh1;
    int x_bf16, M, K;
};

// rot_in_kernel's grid: (ceil(K / 512), M), 4 warps, a warp a (row, 128-block)
__global__ void __launch_bounds__(128) rot2_kernel(const Rot2Args a) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= a.K) return;
    const int k = blk * 128 + 4 * lane;
    const size_t at = (size_t)row * a.K + k;
    float x[4], s[4], v[4];
    dsv41_rot::load4x(a.x, a.x_bf16 != 0, at, x);
    dsv41_rot::load4x(a.suh0, false, k, s);
    dsv41_rot::rot128(x, s, lane, v);
    dsv41_rot::store4h(a.xh0, at, v);
    dsv41_rot::load4x(a.suh1, false, k, s);
    dsv41_rot::rot128(x, s, lane, v);
    dsv41_rot::store4h(a.xh1, at, v);
}

}  // namespace dsv41_pfrot

// the lanes widths (wo_a's form) at every configuration of pfdense_args.h
#define DSV41_PFR_INST(K2, ID, BM, WM, WN, KS, NST) \
    template __global__ void dsv41_pfrot::pfdr_kernel<K2, true, dsv41_pfd::Cfg<BM, WM, WN, KS, NST>>( \
        const dsv41_pfd::Args, const dsv41_pfrot::RotOut);
#define DSV41_PFR_K8(...) DSV41_PFR_INST(8, __VA_ARGS__)
#define DSV41_PFR_K10(...) DSV41_PFR_INST(10, __VA_ARGS__)
#define DSV41_PFR_K12(...) DSV41_PFR_INST(12, __VA_ARGS__)
DSV41_PFD_CFGS(DSV41_PFR_K8)
DSV41_PFD_CFGS(DSV41_PFR_K10)
DSV41_PFD_CFGS(DSV41_PFR_K12)
