// MiniMax-M3.1-Flash dspark draft kernels: taps, residual add, Gemma norm, per-head norm + RoPE (block rows and the
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

typedef __nv_bfloat16 bf16;

constexpr int D = 128;           // head_dim
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int MAX_PAIRS = 64;    // rotary pairs (rot <= 128)
constexpr int KEYS = 128;        // keys a block-attention CTA handles
constexpr int KROW = 130;        // padded K tile row (bf16): 65 words, so lanes on consecutive keys hit distinct banks
constexpr int PART = 132;        // floats of one partial: m, l, 2 pad, 128 acc
constexpr int TOPC_THREADS = 256;
constexpr int TOPC_MAX = 128;
constexpr int CHAIN_THREADS = 256;
constexpr int CHAIN_MAX_CAND = 256; // world * cand
constexpr int CHAIN_MAX_ROWS = 16;

__device__ __forceinline__ float lo16(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi16(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ unsigned short bf_bits(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float4 widen4(uint2 w) { return make_float4(lo16(w.x), hi16(w.x), lo16(w.y), hi16(w.y)); }
__device__ __forceinline__ uint2 pack4(float a, float b, float c, float d) {
    return make_uint2((unsigned)bf_bits(a) | ((unsigned)bf_bits(b) << 16), (unsigned)bf_bits(c) | ((unsigned)bf_bits(d) << 16));
}

__device__ __forceinline__ float tree32(float a) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, o));
    return a;
}

// True when candidate (bv, bi) beats (av, ai): larger value wins, ties to the lower id, NaN last (a total order).
__device__ __forceinline__ bool beats(float bv, int bi, float av, int ai) {
    const bool an = av != av, bn = bv != bv;
    if (an && bn) return bi < ai;
    if (bn) return false;
    if (an) return true;
    return bv > av || (bv == av && bi < ai);
}

template <typename X> __device__ __forceinline__ float load1(const X* p);
template <> __device__ __forceinline__ float load1<float>(const float* p) { return __ldg(p); }
template <> __device__ __forceinline__ float load1<bf16>(const bf16* p) { return __bfloat162float(__ldg(p)); }

} // namespace

// ---- taps

// dst[r * dst_ld + slot * hidden + i] = bf16(src[r * hidden + i]); a CTA a row.
extern "C" __global__ void __launch_bounds__(THREADS) tf_mm3_ds_tap(const float* __restrict__ src, unsigned short* __restrict__ dst,
                                                                   long long dst_ld, int slot, int hidden, int rows) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* s = src + (long long)r * hidden;
    unsigned short* d = dst + (long long)r * dst_ld + (long long)slot * hidden;
    for (int i = threadIdx.x; i < hidden; i += THREADS) d[i] = bf_bits(__ldg(s + i));
}

// ---- residual add

template <typename B>
__device__ __forceinline__ void add_body(const float* a, const B* b, float* dst, long long n) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step) {
        const float x = a == nullptr ? 0.f : a[i];
        dst[i] = __fadd_rn(x, load1<B>(b + i));
    }
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_mm3_ds_add_bf16(const float* a, const bf16* __restrict__ b,
                                                                        float* dst, long long n) {
    add_body<bf16>(a, b, dst, n);
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_mm3_ds_add_f32(const float* a, const float* __restrict__ b,
                                                                       float* dst, long long n) {
    add_body<float>(a, b, dst, n);
}

// ---- RMS norm

// y = x * (1 / sqrt(mean(x^2) + eps)) * (w + 1 | w).
template <typename X>
__device__ __forceinline__ void norm_body(const X* __restrict__ x, const unsigned short* __restrict__ w, unsigned short* __restrict__ y,
                                          int rows, int hidden, float eps, int gemma) {
    __shared__ float part[WARPS];
    const int r = blockIdx.x;
    if (r >= rows) return;
    const X* xr = x + (long long)r * hidden;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hidden; i += THREADS) {
        const float v = load1<X>(xr + i);
        ss = __fadd_rn(ss, __fmul_rn(v, v));
    }
    ss = tree32(ss);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = part[0];
    for (int i = 1; i < WARPS; ++i) tot = __fadd_rn(tot, part[i]);
    const float inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(tot, (float)hidden), eps)));
    const float add = gemma ? 1.f : 0.f;
    unsigned short* yr = y + (long long)r * hidden;
    for (int i = threadIdx.x; i < hidden; i += THREADS) {
        const float wv = __uint_as_float((unsigned)__ldg(w + i) << 16);
        yr[i] = bf_bits(__fmul_rn(__fmul_rn(load1<X>(xr + i), inv), __fadd_rn(wv, add)));
    }
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_mm3_ds_norm_bf16(const bf16* __restrict__ x,
                                                                         const unsigned short* __restrict__ w,
                                                                         unsigned short* __restrict__ y, int rows, int hidden,
                                                                         float eps, int gemma) {
    norm_body<bf16>(x, w, y, rows, hidden, eps, gemma);
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_mm3_ds_norm_f32(const float* __restrict__ x,
                                                                        const unsigned short* __restrict__ w,
                                                                        unsigned short* __restrict__ y, int rows, int hidden,
                                                                        float eps, int gemma) {
    norm_body<float>(x, w, y, rows, hidden, eps, gemma);
}

// ---- per-head norm + RoPE

// Output dim d of the rotated head (neox_half): stage holds the normed head, cs the pair table.
__device__ __forceinline__ float rotated(const float* stage, const float (*cs)[MAX_PAIRS], int d, int rot) {
    const int half = rot >> 1;
    const int i = d < half ? d : d - half;
    const float a = stage[i], b = stage[i + half], c = cs[0][i], s = cs[1][i];
    return d < half ? __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)) : __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
}

// Row r of a block (outputs at row r) or of absorbed rows (K and V into ring slot (pos0 + r) % ring_rows
__device__ __forceinline__ void heads_body(const bf16* q, const bf16* k, const bf16* v, const unsigned short* qw,
                                           const unsigned short* kw, bf16* qn, bf16* kn, bf16* vn, int rows, int nq, int kvh,
                                           int rot, float eps, int gemma, double theta, int pos0, int ring_rows) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float cs[2][MAX_PAIRS];
    __shared__ float stage[WARPS][D];
    const int p = pos0 + r;
    const int half = rot >> 1;
    if (threadIdx.x < half) {
        double sn, cn;
        sincos((double)p * pow(theta, -2.0 * (double)threadIdx.x / (double)rot), &sn, &cn);
        cs[0][threadIdx.x] = (float)cn;
        cs[1][threadIdx.x] = (float)sn;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long long out_row = ring_rows > 0 ? (long long)(p % ring_rows) : (long long)r;
    const float add = gemma ? 1.f : 0.f;
    const int total = nq + 2 * kvh;
    for (int s = warp; s < total; s += WARPS) {
        int kind = 0, h = s; // 0 q, 1 k, 2 v
        if (s >= nq) { kind = 1; h = s - nq; }
        if (s >= nq + kvh) { kind = 2; h = s - nq - kvh; }
        const bf16* src = kind == 0 ? q + (long long)r * nq * D + h * D : kind == 1 ? k + (long long)r * kvh * D + h * D : v + (long long)r * kvh * D + h * D;
        bf16* dst = kind == 0 ? qn + (long long)out_row * nq * D + h * D : kind == 1 ? kn + out_row * kvh * D + h * D : vn + out_row * kvh * D + h * D;
        const uint2 raw = __ldg(reinterpret_cast<const uint2*>(src) + lane);
        if (kind == 2) {
            reinterpret_cast<uint2*>(dst)[lane] = raw;
            continue;
        }
        float4 x = widen4(raw);
        float ss = __fmul_rn(x.x, x.x);
        ss = __fadd_rn(ss, __fmul_rn(x.y, x.y));
        ss = __fadd_rn(ss, __fmul_rn(x.z, x.z));
        ss = __fadd_rn(ss, __fmul_rn(x.w, x.w));
        ss = tree32(ss);
        const float inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, 128.f), eps)));
        const float4 wv = widen4(__ldg(reinterpret_cast<const uint2*>(kind == 0 ? qw : kw) + lane));
        x.x = __fmul_rn(__fmul_rn(x.x, inv), __fadd_rn(wv.x, add));
        x.y = __fmul_rn(__fmul_rn(x.y, inv), __fadd_rn(wv.y, add));
        x.z = __fmul_rn(__fmul_rn(x.z, inv), __fadd_rn(wv.z, add));
        x.w = __fmul_rn(__fmul_rn(x.w, inv), __fadd_rn(wv.w, add));
        float* st = stage[warp];
        st[4 * lane] = x.x;
        st[4 * lane + 1] = x.y;
        st[4 * lane + 2] = x.z;
        st[4 * lane + 3] = x.w;
        __syncwarp();
        float o[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = 4 * lane + i;
            o[i] = d < rot ? rotated(st, cs, d, rot) : st[d];
        }
        __syncwarp();
        reinterpret_cast<uint2*>(dst)[lane] = pack4(o[0], o[1], o[2], o[3]);
    }
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_mm3_ds_block_heads(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, const unsigned short* __restrict__ qw,
    const unsigned short* __restrict__ kw, bf16* __restrict__ qn, bf16* __restrict__ kn, bf16* __restrict__ vn, int rows, int nq,
    int kvh, int rot, float eps, int gemma, double theta, int pos0) {
    heads_body(q, k, v, qw, kw, qn, kn, vn, rows, nq, kvh, rot, eps, gemma, theta, pos0, 0);
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_mm3_ds_ring_put(
    const bf16* __restrict__ k, const bf16* __restrict__ v, const unsigned short* __restrict__ kw, bf16* __restrict__ ring_k,
    bf16* __restrict__ ring_v, int rows, int kvh, int rot, float eps, int gemma, double theta, int pos0, int ring_rows) {
    heads_body(nullptr, k, v, nullptr, kw, nullptr, ring_k, ring_v, rows, 0, kvh, rot, eps, gemma, theta, pos0, ring_rows);
}

// ---- block attention

// Flash-decoding split. A CTA is (kv head g, key chunk c): chunk c < chunks - 1 is context keys c0 + 128 c ..., the
extern "C" __global__ void __launch_bounds__(THREADS) tf_mm3_ds_attn(
    const bf16* __restrict__ qn, const bf16* __restrict__ ring_k, const bf16* __restrict__ ring_v, const bf16* __restrict__ kn,
    const bf16* __restrict__ vn, float* __restrict__ part, int nq, int kvh, int ring_rows, int window, float scale, int rows, int p,
    int lo, int c0, int chunks) {
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* Ks = reinterpret_cast<unsigned short*>(smem);                       // [KEYS][KROW]
    unsigned short* Vs = Ks + KEYS * KROW;                                              // [KEYS][D]
    float* sq = reinterpret_cast<float*>(Vs + KEYS * D);                                // [WARPS][D]
    float* sp = sq + WARPS * D;                                                         // [WARPS][KEYS]
    const int g = blockIdx.x, c = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool block_chunk = c == chunks - 1;
    const int rep = nq / kvh;
    const int Q = rep * rows;
    int kp0 = 0, nk = rows;
    if (!block_chunk) {
        kp0 = c0 + KEYS * c;
        nk = min(KEYS, p - kp0);
    }
    // tiles: key j of the chunk is ring slot (kp0 + j) % ring_rows, or block row j
    for (int idx = threadIdx.x; idx < KEYS * (D / 8); idx += THREADS) {
        const int j = idx / (D / 8), part8 = idx % (D / 8);
        uint4 kv4 = make_uint4(0, 0, 0, 0), vv4 = make_uint4(0, 0, 0, 0);
        if (j < nk) {
            long long off;
            const bf16 *ks, *vs;
            if (block_chunk) {
                off = (long long)j * kvh * D + g * D + part8 * 8;
                ks = kn;
                vs = vn;
            } else {
                off = (long long)((kp0 + j) % ring_rows) * kvh * D + g * D + part8 * 8;
                ks = ring_k;
                vs = ring_v;
            }
            kv4 = __ldg(reinterpret_cast<const uint4*>(ks + off));
            vv4 = __ldg(reinterpret_cast<const uint4*>(vs + off));
        }
        unsigned* kd = reinterpret_cast<unsigned*>(Ks + j * KROW + part8 * 8);
        kd[0] = kv4.x;
        kd[1] = kv4.y;
        kd[2] = kv4.z;
        kd[3] = kv4.w;
        *reinterpret_cast<uint4*>(Vs + j * D + part8 * 8) = vv4;
    }
    __syncthreads();
    float* qs = sq + warp * D;
    float* ps = sp + warp * KEYS;
    for (int ql = warp; ql < Q; ql += WARPS) {
        const int row = ql / rep, head = g * rep + ql % rep;
        const float4 qv = widen4(__ldg(reinterpret_cast<const uint2*>(qn + (long long)row * nq * D + head * D) + lane));
        reinterpret_cast<float4*>(qs)[lane] = qv;
        __syncwarp();
        int lo_row = 0;
        if (!block_chunk) {
            const int w0 = p + row + 1 - window;
            lo_row = max(lo, w0 > 0 ? w0 : 0);
        }
        float s[KEYS / 32];
#pragma unroll
        for (int i = 0; i < KEYS / 32; ++i) {
            const int j = lane + 32 * i;
            const bool ok = j < nk && (block_chunk || kp0 + j >= lo_row);
            float acc = 0.f;
            if (ok) {
                const unsigned* kw = reinterpret_cast<const unsigned*>(Ks + j * KROW);
#pragma unroll 8
                for (int dp = 0; dp < D / 2; ++dp) {
                    const unsigned w = kw[dp];
                    acc = __fmaf_rn(qs[2 * dp], lo16(w), acc);
                    acc = __fmaf_rn(qs[2 * dp + 1], hi16(w), acc);
                }
            }
            s[i] = ok ? __fmul_rn(acc, scale) : -INFINITY;
        }
        float m = s[0];
#pragma unroll
        for (int i = 1; i < KEYS / 32; ++i) m = fmaxf(m, s[i]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float l = 0.f;
#pragma unroll
        for (int i = 0; i < KEYS / 32; ++i) {
            const float pj = s[i] == -INFINITY ? 0.f : expf(__fsub_rn(s[i], m));
            ps[lane + 32 * i] = pj;
            l = __fadd_rn(l, pj);
        }
        l = tree32(l);
        __syncwarp();
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int j = 0; j < nk; ++j) {
            const float pj = ps[j];
            const float4 vv = widen4(*reinterpret_cast<const uint2*>(Vs + j * D + 4 * lane));
            acc.x = __fmaf_rn(pj, vv.x, acc.x);
            acc.y = __fmaf_rn(pj, vv.y, acc.y);
            acc.z = __fmaf_rn(pj, vv.z, acc.z);
            acc.w = __fmaf_rn(pj, vv.w, acc.w);
        }
        float* out = part + (((long long)g * chunks + c) * Q + ql) * PART;
        if (lane == 0) {
            out[0] = m;
            out[1] = l;
        }
        reinterpret_cast<float4*>(out + 4)[lane] = acc;
        __syncwarp();
    }
}

// out[row][head][d] = sum_c w_c acc_c / sum_c w_c l_c with w_c = exp(m_c - max m)
extern "C" __global__ void __launch_bounds__(D) tf_mm3_ds_attn_merge(const float* __restrict__ part, bf16* __restrict__ out, int nq,
                                                                    int kvh, int rows, int chunks) {
    const int rep = nq / kvh;
    const int Q = rep * rows;
    const int g = blockIdx.x / Q, ql = blockIdx.x % Q;
    const int d = threadIdx.x;
    const float* base = part + ((long long)g * chunks * Q + ql) * PART;
    const long long stride = (long long)Q * PART;
    float M = base[0];
    for (int c = 1; c < chunks; ++c) M = fmaxf(M, base[c * stride]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < chunks; ++c) {
        const float mc = base[c * stride];
        const float w = mc == -INFINITY ? 0.f : expf(__fsub_rn(mc, M));
        den = __fadd_rn(den, __fmul_rn(w, base[c * stride + 1]));
        num = __fadd_rn(num, __fmul_rn(w, base[c * stride + 4 + d]));
    }
    const int row = ql / rep, head = g * rep + ql % rep;
    out[(long long)row * nq * D + head * D + d] = __float2bfloat16_rn(__fdiv_rn(num, den));
}

// ---- top-C candidates

namespace {

// An order key: bigger is better; -0 equals +0; every NaN is the smallest key (0).
__device__ __forceinline__ unsigned order_key(float v) {
    if (v != v) return 0u;
    unsigned u = __float_as_uint(v);
    if (u == 0x80000000u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

} // namespace

// logits [rows][ld] f32 (first n valid); out [rows][c] pairs {f32 logit, i32 id}: the c best by (key desc, id asc), id
extern "C" __global__ void __launch_bounds__(TOPC_THREADS) tf_mm3_ds_topc(const float* __restrict__ logits, long long ld, int n,
                                                                         int id0, int c, unsigned* __restrict__ out) {
    __shared__ unsigned hist[256];
    __shared__ unsigned s_prefix, s_rem;
    __shared__ unsigned gt_off[TOPC_THREADS + 1], eq_off[TOPC_THREADS + 1];
    __shared__ unsigned long long cand[TOPC_MAX]; // (key << 32) | (0xffffffff - index): sorts descending
    __shared__ unsigned s_need_eq;
    const float* row = logits + (long long)blockIdx.x * ld;
    unsigned* o = out + (long long)blockIdx.x * c * 2;
    const int t = threadIdx.x;
    const int want = min(c, n);
    if (want > 0) {
        if (t == 0) {
            s_prefix = 0;
            s_rem = (unsigned)want;
        }
        __syncthreads();
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            hist[t] = 0;
            __syncthreads();
            const unsigned prefix = s_prefix;
            const unsigned mask = pass == 0 ? 0u : (0xffffffffu << (32 - 8 * pass));
            for (int i = t; i < n; i += TOPC_THREADS) {
                const unsigned k = order_key(__ldg(row + i));
                if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255], 1u);
            }
            __syncthreads();
            if (t == 0) {
                unsigned rem = s_rem, acc = 0;
                int b = 255;
                for (; b > 0; --b) {
                    if (acc + hist[b] >= rem) break;
                    acc += hist[b];
                }
                s_prefix = prefix | ((unsigned)b << shift);
                s_rem = rem - acc;
            }
            __syncthreads();
        }
        const unsigned T = s_prefix;
        // slices
        const int per = (n + TOPC_THREADS - 1) / TOPC_THREADS;
        const int a = min(n, t * per), z = min(n, a + per);
        unsigned ng = 0, ne = 0;
        for (int i = a; i < z; ++i) {
            const unsigned k = order_key(__ldg(row + i));
            ng += k > T;
            ne += k == T;
        }
        gt_off[t] = ng;
        eq_off[t] = ne;
        __syncthreads();
        if (t == 0) {
            unsigned sg = 0, se = 0;
            for (int i = 0; i < TOPC_THREADS; ++i) {
                const unsigned x = gt_off[i], y = eq_off[i];
                gt_off[i] = sg;
                eq_off[i] = se;
                sg += x;
                se += y;
            }
            s_need_eq = (unsigned)want - sg; // sg < want by the choice of T
        }
        __syncthreads();
        unsigned pg = gt_off[t], pe = eq_off[t];
        for (int i = a; i < z; ++i) {
            const unsigned k = order_key(__ldg(row + i));
            if (k > T) {
                cand[pg++] = ((unsigned long long)k << 32) | (0xffffffffu - (unsigned)i);
            } else if (k == T) {
                if (pe < s_need_eq) cand[(want - s_need_eq) + pe] = ((unsigned long long)k << 32) | (0xffffffffu - (unsigned)i);
                ++pe;
            }
        }
    }
    __syncthreads();
    // bitonic sort of 128 slots, descending; empty slots are 0 (key 0 and index 0xffffffff would tie with a NaN: use
    if (t < TOPC_MAX && t >= want) cand[t] = 0ull;
    __syncthreads();
    for (int kk = 2; kk <= TOPC_MAX; kk <<= 1) {
        for (int j = kk >> 1; j > 0; j >>= 1) {
            if (t < TOPC_MAX) {
                const int ix = t ^ j;
                if (ix > t) {
                    const bool up = (t & kk) == 0; // descending overall
                    const unsigned long long x = cand[t], y = cand[ix];
                    if (up ? x < y : x > y) {
                        cand[t] = y;
                        cand[ix] = x;
                    }
                }
            }
            __syncthreads();
        }
    }
    if (t < c) {
        if (t < want) {
            const unsigned long long e = cand[t];
            const int i = (int)(0xffffffffu - (unsigned)(e & 0xffffffffu));
            o[2 * t] = __float_as_uint(__ldg(row + i));
            o[2 * t + 1] = (unsigned)(id0 + i);
        } else {
            o[2 * t] = __float_as_uint(-INFINITY);
            o[2 * t + 1] = 0xffffffffu;
        }
    }
}

// ---- Markov chain + confidence

// cands [world][rows][cand] {f32 logit, i32 id}; post bf16 [rows][hidden]; w1, w2 bf16 [vocab][rank]
extern "C" __global__ void __launch_bounds__(CHAIN_THREADS) tf_mm3_ds_chain(
    const unsigned* __restrict__ cands, const unsigned short* __restrict__ post, const unsigned short* __restrict__ w1,
    const unsigned short* __restrict__ w2, const unsigned short* __restrict__ conf_w, const unsigned short* __restrict__ conf_b,
    unsigned* __restrict__ out, int hidden, int rank, int cand, int world, int rows, unsigned anchor) {
    __shared__ float hd[CHAIN_MAX_ROWS];
    __shared__ float w1p[1024];
    __shared__ float zs[CHAIN_MAX_CAND];
    __shared__ int ids[CHAIN_MAX_CAND];
    __shared__ int s_tok;
    __shared__ float sv[WARPS];
    __shared__ int si[WARPS];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ncand = world * cand;
    // hidden part of every row's confidence: 4 fma chains a lane, (a0 + a1) + (a2 + a3), xor tree
    for (int r = warp; r < rows; r += WARPS) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int ch = 0; ch < hidden / 128; ++ch) {
            const int at = 4 * (lane + 32 * ch);
            const uint2 x = __ldg(reinterpret_cast<const uint2*>(post + (long long)r * hidden + at));
            const uint2 w = __ldg(reinterpret_cast<const uint2*>(conf_w + at));
            a0 = __fmaf_rn(lo16(x.x), lo16(w.x), a0);
            a1 = __fmaf_rn(hi16(x.x), hi16(w.x), a1);
            a2 = __fmaf_rn(lo16(x.y), lo16(w.y), a2);
            a3 = __fmaf_rn(hi16(x.y), hi16(w.y), a3);
        }
        const float sum = tree32(__fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3)));
        if (lane == 0) hd[r] = sum;
    }
    for (int i = threadIdx.x; i < ncand; i += CHAIN_THREADS) ids[i] = -1;
    __syncthreads();
    unsigned prev = anchor;
    const int per_lane = rank / 32; // rank dims a lane
    for (int r = 0; r < rows; ++r) {
        for (int i = threadIdx.x; i < rank; i += CHAIN_THREADS) w1p[i] = __uint_as_float((unsigned)__ldg(w1 + (long long)prev * rank + i) << 16);
        for (int i = threadIdx.x; i < ncand; i += CHAIN_THREADS) {
            const int wr = i / cand, k = i % cand;
            const unsigned* e = cands + (((long long)wr * rows + r) * cand + k) * 2;
            zs[i] = __uint_as_float(e[0]);
            ids[i] = (int)e[1];
        }
        __syncthreads();
        // z = base + dot(w2[id], w1[prev]); warps take candidates in turn, lane l owns dims l*per_lane ...
        for (int i = warp; i < ncand; i += WARPS) {
            const int id = ids[i];
            float z = zs[i];
            if (id >= 0) {
                float a = 0.f;
                const unsigned short* wr2 = w2 + (long long)id * rank + lane * per_lane;
                for (int d = 0; d < per_lane; ++d) a = __fmaf_rn(__uint_as_float((unsigned)__ldg(wr2 + d) << 16), w1p[lane * per_lane + d], a);
                a = tree32(a);
                z = __fadd_rn(z, a);
            }
            if (lane == 0) zs[i] = z;
        }
        __syncthreads();
        // best by the total order over the valid candidates
        float bv = __int_as_float(0x7fc00000);
        int bi = 0x7fffffff;
        for (int i = threadIdx.x; i < ncand; i += CHAIN_THREADS) {
            if (ids[i] >= 0 && beats(zs[i], ids[i], bv, bi)) {
                bv = zs[i];
                bi = ids[i];
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (beats(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            sv[warp] = bv;
            si[warp] = bi;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float v = sv[0];
            int b = si[0];
            for (int w = 1; w < WARPS; ++w) {
                if (beats(sv[w], si[w], v, b)) {
                    v = sv[w];
                    b = si[w];
                }
            }
            s_tok = b == 0x7fffffff ? -1 : b;
        }
        // confidence: the hidden part, then dot(w1[prev], conf_w[hidden ..]), then the bias
        if (warp == 1) {
            float a = 0.f;
            const unsigned short* cw = conf_w + hidden + lane * per_lane;
            for (int d = 0; d < per_lane; ++d) a = __fmaf_rn(w1p[lane * per_lane + d], __uint_as_float((unsigned)__ldg(cw + d) << 16), a);
            a = tree32(a);
            if (lane == 0) {
                const float s = __fadd_rn(__fadd_rn(hd[r], a), __uint_as_float((unsigned)__ldg(conf_b) << 16));
                out[2 * r + 1] = __float_as_uint(__fdiv_rn(1.f, __fadd_rn(1.f, expf(-s))));
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) out[2 * r] = (unsigned)s_tok;
        prev = (unsigned)s_tok;
        if (s_tok < 0) prev = anchor; // no candidate: the rest of the chain is empty anyway; keep indices valid
        __syncthreads();
    }
}
