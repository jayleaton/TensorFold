// x3gm v4 M1b (PORT.md 9.5, proto only): gm2_kernel<2, 2, GM_GU1>'s gate/up with a loader warp; the same Xd bytes.

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

template <int K2, int NS>
struct Cfg4 {
    static constexpr int WORKERS = 16, WARPS = 1 + WORKERS, THREADS = WARPS * 32;
    static constexpr int BM = 128, TILES = 2 * NB, TW = 4 * K2, ER = 32, LDE = 132, MD = 2, META = BM + 4;
    static constexpr int ROW_B = 2 * BM * 32;                  // a stage's rows: [input][member][16 halves]
    static constexpr int WORD_B = TILES * TW * 4;              // a stage's words: [matrix][column tile][TW]
    static constexpr int STAGE = ROW_B + WORD_B, EPI_B = 2 * ER * LDE * 4;
    static constexpr int EPI_OFF = NS * STAGE, META_OFF = EPI_OFF + EPI_B, BAR_OFF = META_OFF + MD * META * 4;
    static constexpr int SMEM = BAR_OFF + 8 * (2 * NS + 2 * MD);  // full / empty [NS], meta full / empty [MD]
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

template <int K2, int NS>
__device__ __forceinline__ void gm4_loader(const Args& a, const Args4& b, unsigned char* smem) {
    using C = Cfg4<K2, NS>;
    constexpr int TW = C::TW, WCH = C::TILES * K2, RQ = C::BM / 16;  // word chunks a stage; rows a lane
    const int lane = threadIdx.x & 31, K = a.K, KT = K >> 4, NTILES = a.N >> 4, NBLK = a.N >> 7;
    const int total = b.npass[0] * NBLK, c = lane & 1;       // lane pairs copy a row's 32 bytes (whole sectors)
    const size_t se = (size_t)KT * NTILES * TW;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + C::BAR_OFF);
    uint64_t *full = bars, *empty = bars + NS, *mfull = bars + 2 * NS, *mempty = mfull + C::MD;
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
#pragma unroll
        for (int q = 0; q < C::BM / 32; ++q) {
            const int i = lane + 32 * q;
            meta[4 + i] = i < cnt ? a.order[off + i] : -1;
        }
        int rofs[RQ];                                          // member (lane >> 1) + 16 q's row offset (-1: absent)
#pragma unroll
        for (int q = 0; q < RQ; ++q) {
            const int i = (lane >> 1) + 16 * q;
            rofs[q] = i < cnt ? a.order[off + i] * K + c * 8 : -1;
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
            for (int q = 0; q < RQ; ++q) {
                if (rofs[q] < 0) continue;
                const int r = (lane >> 1) + 16 * q;
#pragma unroll
                for (int m = 0; m < 2; ++m)
                    cp_async16(ra + (m * C::BM * 2 + row_chunk(r, c)) * 8, (m ? a.x1 : a.x0) + rofs[q] + kt * 16, true);
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

// worker cw: matrix cw / 8, column tile cw % 8, all 128 members; decodes its tile once a k tile (v2: once a 64)
template <int K2, int NS>
__device__ __forceinline__ void gm4_worker(const Args& a, unsigned char* smem, int cw) {
    using C = Cfg4<K2, NS>;
    constexpr int ER = C::ER, LDE = C::LDE;
    const int lane = threadIdx.x & 31, KT = a.K >> 4, N = a.N;
    const int mat = cw >> 3, ct = cw & 7;
    const int g = lane >> 2, t = lane & 3;
    const int lr = (lane & 7) + ((lane >> 4) << 3), lc = (lane >> 3) & 1;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + C::BAR_OFF);
    uint64_t *full = bars, *empty = bars + NS, *mfull = bars + 2 * NS, *mempty = mfull + C::MD;
    float* ep = reinterpret_cast<float*>(smem + C::EPI_OFF);  // [matrix][ER members][LDE]
    const LaneMap<K2> map(lane);
    int j = 0;
    for (int mi = 0;; ++mi) {
        const int ms = mi % C::MD;
        mb_wait(&mfull[ms], (mi / C::MD) & 1);
        const int* meta = reinterpret_cast<const int*>(smem + C::META_OFF) + ms * C::META;
        if (meta[0] < 0) return;
        const int e = meta[1], nb = meta[2], cnt = meta[3];
        const int n8 = (cnt + 7) >> 3;                         // the item's n8 groups, 1..16 (block-uniform)
        float acc[16][4];
#pragma unroll
        for (int n = 0; n < 16; ++n)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[n][q] = 0.f;
        for (int kt = 0; kt < KT; ++kt, ++j) {
            const int s = j % NS;
            mb_wait(&full[s], (j / NS) & 1);
            const unsigned char* st = smem + (size_t)s * C::STAGE;
            uint32_t af[4];
            const uint32_t* tile = reinterpret_cast<const uint32_t*>(st + C::ROW_B) + (mat * NB + ct) * C::TW;
            decode_a<K2>(tile, map, af);
            const half* ra = reinterpret_cast<const half*>(st) + mat * C::BM * 16;
#pragma unroll
            for (int np = 0; np < 8; ++np) {
                if (2 * np < n8) {                             // warp-uniform
                    uint32_t bf[4];
                    ldsm4(bf, ra + row_chunk(16 * np + lr, lc) * 8);
                    const uint32_t b01[2] = {bf[0], bf[1]}, b23[2] = {bf[2], bf[3]};
                    mma16816(acc[2 * np], af, b01);
                    if (2 * np + 1 < n8) mma16816(acc[2 * np + 1], af, b23);
                }
            }
            __syncwarp();
            if (lane == 0) mb_arrive(&empty[s]);
        }

        // gm2_item's epilogue; round rd = n8 groups 4 rd .. + 3 (members 32 rd ..), whose accumulators die with it
#pragma unroll
        for (int rd = 0; rd < 4; ++rd) {
            if (32 * rd >= cnt) break;                         // block-uniform among workers
            consumers_sync();                                  // the last round's readers are done with ep
#pragma unroll
            for (int nn = 0; nn < 4; ++nn) {
                float* E = ep + ((size_t)mat * ER + 8 * nn + 2 * t) * LDE + ct * 16 + g;
                E[0] = acc[4 * rd + nn][0];
                E[LDE] = acc[4 * rd + nn][1];
                E[8] = acc[4 * rd + nn][2];
                E[LDE + 8] = acc[4 * rd + nn][3];
            }
            consumers_sync();
#pragma unroll
            for (int h = 0; h < ER / C::WORKERS; ++h) {
                const int r = cw + C::WORKERS * h, m = 32 * rd + r;
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
        int mr = mi;                                           // an opaque copy: the slot is recomputed, not kept
        asm volatile("mov.u32 %0, %0;\n" : "+r"(mr));
        __syncwarp();
        if (lane == 0) mb_arrive(&mempty[mr % C::MD]);
    }
}

// gate/up of b's passes; a as gm2_kernel's (x0 / x1, t0 / t1 or tp0 / tp1, order, scales, xd), its plan unused
template <int K2, int NS>
__global__ void __launch_bounds__(Cfg4<K2, NS>::THREADS, 1) gm4_gu_kernel(const Args a, const Args4 b) {
    using C = Cfg4<K2, NS>;
    extern __shared__ __align__(16) unsigned char smem[];
    const int warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) {
        uint64_t* bars = reinterpret_cast<uint64_t*>(smem + C::BAR_OFF);
        for (int i = 0; i < NS; ++i) {
            mb_init(&bars[i], 32);                             // full: the loader's lanes (cp.async arrivals)
            mb_init(&bars[NS + i], C::WORKERS);                // empty: a lane of each worker
        }
        for (int i = 0; i < C::MD; ++i) {
            mb_init(&bars[2 * NS + i], 32);                    // meta full: the loader's lanes
            mb_init(&bars[2 * NS + C::MD + i], C::WORKERS);
        }
    }
    __syncthreads();
    if (warp == 0) gm4_loader<K2, NS>(a, b, smem);
    else gm4_worker<K2, NS>(a, smem, warp - 1);
}

// (K2, NS): the deepest load ring that fits; 6A: a shallower ring at the same width (ring-depth sensitivity)
#define GM4_CFG_4 4, 6
#define GM4_CFG_6 6, 6
#define GM4_CFG_6A 6, 4
#define GM4_CFG_8 8, 6
#define GM4_CFG_10 10, 6

}  // namespace dsv41_x3gm
