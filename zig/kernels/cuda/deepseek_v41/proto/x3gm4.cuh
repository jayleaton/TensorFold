// x3gm v4 M1 (PORT.md 9.5, proto only): gm2_kernel<2, 2, GM_GU1>'s gate/up, warp-specialised; the same Xd bytes.

#pragma once

#define DSV41_X3GM_NO_INSTANCES
#include "../x3gm.cu"

namespace dsv41_x3gm {

// v4's pass table: passes over x3gm_plan's order (pe / poff / pcnt, npass of them), at most 128 members a pass
struct Args4 {
    const int* pe;
    const int* poff;
    const int* pcnt;
    const int* npass;
    int* ticket;              // [1] zeroed, or null (static stride)
};

template <int K2, int NS, int ND>
struct Cfg4 {
    static constexpr int DECODERS = 3, CONSUMERS = 16, WARPS = 1 + DECODERS + CONSUMERS, THREADS = WARPS * 32;
    static constexpr int BM = 128, TILES = 2 * NB, TW = 4 * K2, ER = 32, LDE = 132, MD = 2, META = BM + 4;
    static constexpr int ROW_B = 2 * BM * 32;                  // a stage's rows: [input][member][16 halves]
    static constexpr int WORD_B = TILES * TW * 4;              // a stage's words: [matrix][column tile][TW]
    static constexpr int STAGE = ROW_B + WORD_B, DEC_B = TILES * 32 * 16;
    static constexpr int EPI_B = 2 * ER * LDE * 4;
    static constexpr int DEC_OFF = NS * STAGE, EPI_OFF = DEC_OFF + ND * DEC_B, META_OFF = EPI_OFF + EPI_B;
    static constexpr int BAR_OFF = META_OFF + MD * META * 4;   // full / empty [NS], dfull / dempty [ND], meta [MD]
    static constexpr int SMEM = BAR_OFF + 8 * (2 * NS + 2 * ND + 2 * MD);
    static_assert(STAGE % 16 == 0 && BAR_OFF % 8 == 0, "16-byte stages, 8-byte barriers");
    static_assert(SMEM <= 101376, "one CTA an SM: the opt-in shared memory of a block");
};

__device__ __forceinline__ unsigned sh_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, int n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(sh_addr(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(sh_addr(b)) : "memory");
}
// one arrival on b once every cp.async this thread issued so far has landed (counted in b's init count)
__device__ __forceinline__ void mb_arrive_cp(uint64_t* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(sh_addr(b)) : "memory");
}
__device__ __forceinline__ void mb_wait(uint64_t* b, int parity) {
    uint32_t ok = 0;
    do {
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     "selp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(ok)
                     : "r"(sh_addr(b)), "r"(parity)
                     : "memory");
    } while (!ok);
}
__device__ __forceinline__ void consumers_sync() { asm volatile("bar.sync 1, 512;\n" ::: "memory"); }

// member row r's 16-byte chunk c in a stage: the chunk flips every 4 rows, so an ldmatrix phase is conflict-free
__device__ __forceinline__ int row_chunk(int r, int c) { return 2 * r + (c ^ ((r >> 2) & 1)); }

template <int K2, int NS, int ND>
__device__ __forceinline__ void gm4_loader(const Args& a, const Args4& b, unsigned char* smem) {
    using C = Cfg4<K2, NS, ND>;
    constexpr int TW = C::TW, WCH = C::TILES * K2;            // word chunks a stage (16 bytes each)
    const int lane = threadIdx.x & 31, K = a.K, KT = K >> 4, NTILES = a.N >> 4, NBLK = a.N >> 7;
    const int total = b.npass[0] * NBLK;
    const size_t se = (size_t)KT * NTILES * TW;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + C::BAR_OFF);
    uint64_t *full = bars, *empty = bars + NS, *mfull = bars + 2 * NS + 2 * ND, *mempty = mfull + C::MD;
    int j = 0;
    for (int mi = 0;; ++mi) {
        const int ms = mi % C::MD;
        if (mi >= C::MD) mb_wait(&mempty[ms], ((mi / C::MD) & 1) ^ 1);
        int item = 0;
        if (lane == 0) item = b.ticket ? atomicAdd(b.ticket, 1) : blockIdx.x + mi * gridDim.x;
        item = __shfl_sync(0xffffffffu, item, 0);
        int* meta = reinterpret_cast<int*>(smem + C::META_OFF) + ms * C::META;
        if (item >= total) {
            if (lane == 0) meta[0] = -1;
            mb_arrive(&mfull[ms]);
            cp_wait<0>();                                      // no copy outlives its thread
            return;
        }
        const int tt = item / NBLK, nb = item % NBLK;
        const int e = b.pe[tt], off = b.poff[tt], cnt = b.pcnt[tt];
        int rofs[C::BM / 32];                                  // member lane + 32 q's row offset (-1: absent)
#pragma unroll
        for (int q = 0; q < C::BM / 32; ++q) {
            const int i = lane + 32 * q, row = i < cnt ? a.order[off + i] : -1;
            meta[4 + i] = row;
            rofs[q] = row < 0 ? -1 : row * K;
        }
        if (lane == 0) {
            meta[0] = item;
            meta[1] = e;
            meta[2] = nb;
            meta[3] = cnt;
        }
        mb_arrive(&mfull[ms]);
        const uint32_t* te0 = a.tp0 ? reinterpret_cast<const uint32_t*>(a.tp0[e]) : a.t0 + (size_t)e * se;
        const uint32_t* te1 = a.tp1 ? reinterpret_cast<const uint32_t*>(a.tp1[e]) : a.t1 + (size_t)e * se;
        for (int kt = 0; kt < KT; ++kt, ++j) {
            const int s = j % NS;
            if (j >= NS) mb_wait(&empty[s], ((j / NS) & 1) ^ 1);
            unsigned char* st = smem + (size_t)s * C::STAGE;
            half* ra = reinterpret_cast<half*>(st);
#pragma unroll
            for (int q = 0; q < C::BM / 32; ++q) {
                if (rofs[q] < 0) continue;
                const int r = lane + 32 * q;
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const half* src = (m ? a.x1 : a.x0) + rofs[q] + kt * 16;
#pragma unroll
                    for (int c = 0; c < 2; ++c)
                        cp_async16(ra + (m * C::BM * 2 + row_chunk(r, c)) * 8, src + c * 8, true);
                }
            }
            uint32_t* rw = reinterpret_cast<uint32_t*>(st + C::ROW_B);
            const size_t w0 = ((size_t)kt * NTILES + nb * NB) * TW;  // the block's 8 tiles are contiguous a matrix
            for (int i = lane; i < WCH; i += 32) {
                const int m = i / (NB * K2), q = i % (NB * K2);
                cp_async16(rw + m * NB * TW + q * 4, (m ? te1 : te0) + w0 + q * 4, true);
            }
            mb_arrive_cp(&full[s]);
        }
    }
}

template <int K2, int NS, int ND>
__device__ __forceinline__ void gm4_decoder(const Args& a, unsigned char* smem, int dw) {
    using C = Cfg4<K2, NS, ND>;
    const int lane = threadIdx.x & 31, KT = a.K >> 4;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + C::BAR_OFF);
    uint64_t *full = bars, *empty = bars + NS, *dfull = bars + 2 * NS, *dempty = dfull + ND;
    uint64_t *mfull = dempty + ND, *mempty = mfull + C::MD;
    const LaneMap<K2> map(lane);
    int j = 0;
    for (int mi = 0;; ++mi) {
        const int ms = mi % C::MD;
        mb_wait(&mfull[ms], (mi / C::MD) & 1);
        const bool live = reinterpret_cast<const int*>(smem + C::META_OFF)[ms * C::META] >= 0;
        __syncwarp();
        if (lane == 0) mb_arrive(&mempty[ms]);
        if (!live) return;
        for (int kt = 0; kt < KT; ++kt, ++j) {
            const int s = j % NS, d = j % ND;
            mb_wait(&full[s], (j / NS) & 1);
            if (j >= ND) mb_wait(&dempty[d], ((j / ND) & 1) ^ 1);
            const uint32_t* rw = reinterpret_cast<const uint32_t*>(smem + (size_t)s * C::STAGE + C::ROW_B);
            uint4* dec = reinterpret_cast<uint4*>(smem + C::DEC_OFF + (size_t)d * C::DEC_B);
#pragma unroll
            for (int t = 0; t < (C::TILES + C::DECODERS - 1) / C::DECODERS; ++t) {
                const int tile = dw + C::DECODERS * t;         // matrix tile / 8, column tile tile % 8
                if (tile >= C::TILES) break;
                uint32_t af[4];
                decode_a<K2>(rw + tile * C::TW, map, af);
                dec[tile * 32 + lane] = make_uint4(af[0], af[1], af[2], af[3]);
            }
            __syncwarp();
            if (lane == 0) {
                mb_arrive(&dfull[d]);
                mb_arrive(&empty[s]);
            }
        }
    }
}

template <int K2, int NS, int ND>
__device__ __forceinline__ void gm4_consumer(const Args& a, unsigned char* smem, int cw) {
    using C = Cfg4<K2, NS, ND>;
    constexpr int ER = C::ER, LDE = C::LDE;
    const int lane = threadIdx.x & 31, KT = a.K >> 4, N = a.N;
    const int mat = cw >> 3, ch = (cw >> 2) & 1, mg = cw & 3;  // matrix, 64-column half, 32-member group
    const int g = lane >> 2, t = lane & 3;
    const int lr = (lane & 7) + ((lane >> 4) << 3), lc = (lane >> 3) & 1;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + C::BAR_OFF);
    uint64_t *full = bars, *empty = bars + NS, *dfull = bars + 2 * NS, *dempty = dfull + ND;
    uint64_t *mfull = dempty + ND, *mempty = mfull + C::MD;
    float* ep = reinterpret_cast<float*>(smem + C::EPI_OFF);  // [matrix][ER members][LDE]
    int j = 0;
    for (int mi = 0;; ++mi) {
        const int ms = mi % C::MD;
        mb_wait(&mfull[ms], (mi / C::MD) & 1);
        const int* meta = reinterpret_cast<const int*>(smem + C::META_OFF) + ms * C::META;
        if (meta[0] < 0) return;
        const int e = meta[1], nb = meta[2], cnt = meta[3];
        const int n8 = min(max((cnt - 32 * mg + 7) >> 3, 0), 4);  // this warp's live n8 groups (warp-uniform)
        float acc[4][4][4];
#pragma unroll
        for (int l = 0; l < 4; ++l)
#pragma unroll
            for (int n = 0; n < 4; ++n)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[l][n][c] = 0.f;
        for (int kt = 0; kt < KT; ++kt, ++j) {
            const int s = j % NS, d = j % ND;
            mb_wait(&full[s], (j / NS) & 1);
            mb_wait(&dfull[d], (j / ND) & 1);
            if (n8 > 0) {
                const half* ra = reinterpret_cast<const half*>(smem + (size_t)s * C::STAGE) + mat * C::BM * 16;
                uint32_t bf[2][4];
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    const int r = 32 * mg + 16 * np + lr;
                    if (2 * np < n8) ldsm4(bf[np], ra + row_chunk(r, lc) * 8);
                }
                const uint4* dec = reinterpret_cast<const uint4*>(smem + C::DEC_OFF + (size_t)d * C::DEC_B);
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const uint4 v = dec[(mat * NB + 4 * ch + l) * 32 + lane];
                    const uint32_t af[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        if (n < n8) {
                            const uint32_t b2[2] = {bf[n >> 1][2 * (n & 1)], bf[n >> 1][2 * (n & 1) + 1]};
                            mma16816(acc[l][n], af, b2);
                        }
                    }
                }
            }
            __syncwarp();
            if (lane == 0) {
                mb_arrive(&empty[s]);
                mb_arrive(&dempty[d]);
            }
        }

        // gm2_item's epilogue; round n = n8 group n of every member group, so acc[.][n] dies with it (no spills)
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            if (8 * n >= cnt) break;                           // block-uniform among consumers
            consumers_sync();                                  // the last round's readers are done with ep
            float* E = ep + ((size_t)mat * ER + 8 * mg + 2 * t) * LDE + ch * 64 + g;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                E[l * 16] = acc[l][n][0];
                E[LDE + l * 16] = acc[l][n][1];
                E[l * 16 + 8] = acc[l][n][2];
                E[LDE + l * 16 + 8] = acc[l][n][3];
            }
            consumers_sync();
#pragma unroll
            for (int h = 0; h < ER / C::CONSUMERS; ++h) {
                const int r = cw + C::CONSUMERS * h, m = 32 * (r >> 3) + 8 * n + (r & 7);
                if (m >= cnt) continue;
                const int row = meta[4 + m];
                const int c0 = 4 * lane;
                const int col = e * N + nb * 128 + c0;        // < E N: 32 bits (fewer live registers)
                float v[4];
                *reinterpret_cast<float4*>(v) = *reinterpret_cast<const float4*>(ep + (size_t)r * LDE + c0);
                tf_exl3::fwht128(v, lane);
                float u[4];
                *reinterpret_cast<float4*>(u) = *reinterpret_cast<const float4*>(ep + ((size_t)ER + r) * LDE + c0);
                tf_exl3::fwht128(u, lane);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {               // upstream gateup_epilogue, ACT_F32 (gm2_item's)
                    const float gg = fminf(v[jj] * HAD * __half2float(a.sv0[col + jj]), a.limit);
                    const float uu = fminf(fmaxf(u[jj] * HAD * __half2float(a.sv1[col + jj]), -a.limit), a.limit);
                    const float act = gg / (1.f + expf(-gg)) * uu;
                    v[jj] = act * __half2float(a.sd[col + jj]);
                }
                tf_exl3::fwht128(v, lane);
                half2* o = reinterpret_cast<half2*>(a.xd + (size_t)row * N + nb * 128 + c0);
                o[0] = __halves2half2(__float2half_rn(v[0] * HAD), __float2half_rn(v[1] * HAD));
                o[1] = __halves2half2(__float2half_rn(v[2] * HAD), __float2half_rn(v[3] * HAD));
            }
        }
        __syncwarp();
        if (lane == 0) mb_arrive(&mempty[ms]);
    }
}

// gate/up of b's passes; a as gm2_kernel's (x0 / x1, t0 / t1 or tp0 / tp1, order, scales, xd), its plan unused
template <int K2, int NS, int ND>
__global__ void __launch_bounds__(Cfg4<K2, NS, ND>::THREADS, 1) gm4_gu_kernel(const Args a, const Args4 b) {
    using C = Cfg4<K2, NS, ND>;
    extern __shared__ __align__(16) unsigned char smem[];
    const int warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) {
        uint64_t* bars = reinterpret_cast<uint64_t*>(smem + C::BAR_OFF);
        for (int i = 0; i < NS; ++i) {
            mb_init(&bars[i], 32);                             // full: the loader's lanes (cp.async arrivals)
            mb_init(&bars[NS + i], C::DECODERS + C::CONSUMERS);  // empty: a lane of each reader warp
        }
        for (int i = 0; i < ND; ++i) {
            mb_init(&bars[2 * NS + i], C::DECODERS);           // dfull
            mb_init(&bars[2 * NS + ND + i], C::CONSUMERS);     // dempty
        }
        for (int i = 0; i < C::MD; ++i) {
            mb_init(&bars[2 * NS + 2 * ND + i], 32);           // meta full: the loader's lanes
            mb_init(&bars[2 * NS + 2 * ND + C::MD + i], C::DECODERS + C::CONSUMERS);
        }
    }
    __syncthreads();
    if (warp == 0) gm4_loader<K2, NS, ND>(a, b, smem);
    else if (warp <= C::DECODERS) gm4_decoder<K2, NS, ND>(a, smem, warp - 1);
    else gm4_consumer<K2, NS, ND>(a, smem, warp - 1 - C::DECODERS);
}

// (K2, NS, ND): the deepest load ring beside a 2-deep decode ring; 6A: a 3-deep decode ring, one load stage less
#define GM4_CFG_4 4, 5, 2
#define GM4_CFG_6 6, 5, 2
#define GM4_CFG_6A 6, 4, 3
#define GM4_CFG_8 8, 4, 2
#define GM4_CFG_10 10, 4, 2

}  // namespace dsv41_x3gm
