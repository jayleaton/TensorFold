// TF_DSV41_ROT_FOLD bit 1 (attn_cuda.cu includes this inside namespace dsv41_attn): the merge writes wo_a's rows
#pragma once

constexpr float ROT_HAD = 0.08838834764831845f;          // exl3_decode.cuh HAD_SCALE, 1 / sqrt(128)

// wo_a's rot_in rows: group s reads o's row columns [s gk, (s + 1) gk); its fp16 rows (stride gk) and suh at column 0
struct RotIn {
    half* xh[8];
    const half* suh[8];
    int gk;
};

// x3seg seg_rot_in_kernel's v *= suh, fwht128, v *= HAD_SCALE as its SASS runs them (that unit contracts; this one not)
__device__ __forceinline__ void rot128(const float (&x)[4], const float (&s)[4], int lane, float (&v)[4]) {
    const float p1 = __fmul_rn(x[1], s[1]), p3 = __fmul_rn(x[3], s[3]);   // the scale fused into the first butterfly
    const float a = __fmaf_rn(x[0], s[0], p1), b = __fmaf_rn(x[0], s[0], -p1);
    const float c = __fmaf_rn(x[2], s[2], p3), d = __fmaf_rn(x[2], s[2], -p3);
    v[0] = __fadd_rn(a, c);
    v[1] = __fadd_rn(b, d);
    v[2] = __fsub_rn(a, c);
    v[3] = __fsub_rn(b, d);
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(FULL, v[j], m);
            v[j] = (lane & m) ? __fsub_rn(o, v[j]) : __fadd_rn(v[j], o);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __fmul_rn(v[j], ROT_HAD);
}

// merge_cols verbatim (attn_split.cuh), so split_kernel's own merge keeps its callers and its code
template <bool ROWS, int NC>
__device__ void merge_cols_r(const Args& a, uint8_t* sm, int smt, int r, int hb, int c0) {
    const int tid = threadIdx.x, H = a.H, R = a.R;
    float* ca = reinterpret_cast<float*>(sm + smt);              // [5][16] a, [5][16] b, [16] af, [16] den
    float* cb = ca + NCH * BMQ;
    float* fa = cb + NCH * BMQ;
    float* fd = fa + BMQ;
    bool cw[NCH];
#pragma unroll
    for (int c = 0; c < NCH; ++c) cw[c] = chunk_works(a, r, c);
    constexpr int UB = BMQ * (NC / 4) / NT;                      // float4 units a thread (8 at P = 1, 2 at P = 4)
    static_assert(UB * NT == BMQ * (NC / 4), "whole units");
    float4 pv[UB][NCH];
#pragma unroll
    for (int ub = 0; ub < UB; ++ub) {                            // the partials in flight first
        const int u = tid + ub * NT, hl = u / (NC / 4), d = c0 + (u % (NC / 4)) * 4, h = hb * BMQ + hl;
#pragma unroll
        for (int c = 0; c < NCH; ++c)
            pv[ub][c] = cw[c] ? __ldcg(reinterpret_cast<const float4*>(
                                    a.po + ((static_cast<long long>(c) * R + r) * H + h) * D + d))
                              : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    if (tid < BMQ) {                                             // a head's scalar chain (Triton's _merge, unrolled)
        const int h = hb * BMQ + tid;
        float pmv[NCH], plv[NCH];
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const long long base = (static_cast<long long>(c) * R + r) * H + h;
            pmv[c] = cw[c] ? __ldcg(a.pm + base) : -INFINITY;
            plv[c] = cw[c] ? __ldcg(a.pl + base) : 0.f;
        }
        const float sink = a.sink[h];
        float m = -INFINITY, l = 0.f;
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const float cm = pmv[c], cl = plv[c];
            const bool act = cl > 0.f;
            float aa, bb;
            if (c == 0) {
                const float nm = act ? fmaxf(cm, -INFINITY) : -INFINITY;
                bb = act ? texp(__fsub_rn(cm, nm)) : 0.f;
                aa = 0.f;
                l = __fmaf_rn(bb, cl, 0.f);
                m = nm;
            } else {
                const float nm = act ? fmaxf(m, cm) : m;
                aa = act ? (m == -INFINITY ? 0.f : texp(__fsub_rn(m, nm))) : 1.f;
                bb = act ? texp(__fsub_rn(cm, nm)) : 0.f;
                l = __fmaf_rn(l, aa, __fmul_rn(bb, cl));
                m = nm;
            }
            ca[c * BMQ + tid] = aa;
            cb[c * BMQ + tid] = bb;
        }
        const float top = fmaxf(m, sink);
        const float af = m == -INFINITY ? 0.f : texp(__fsub_rn(m, top));
        const float e = sink == -INFINITY ? 0.f : texp(__fsub_rn(sink, top));
        fa[tid] = af;
        fd[tid] = __fmaf_rn(l, af, e);
    }
    __syncthreads();
    const float* csr = a.cs + static_cast<long long>(row_pos<ROWS>(a, r)) * a.csst;
#pragma unroll
    for (int ub = 0; ub < UB; ++ub) {
        const int u = tid + ub * NT, hl = u / (NC / 4), d = c0 + (u % (NC / 4)) * 4, h = hb * BMQ + hl;
        float o[4];
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const float4 v = pv[ub][c];
            const float co[4] = {v.x, v.y, v.z, v.w};
            const float aa = ca[c * BMQ + hl], bb = cb[c * BMQ + hl];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                o[i] = c == 0 ? __fmaf_rn(bb, co[i], 0.f) : __fmaf_rn(o[i], aa, __fmul_rn(co[i], bb));
        }
        float x[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = div_full(__fmul_rn(o[i], fa[hl]), fd[hl]);
        uint16_t ob[4];
#pragma unroll
        for (int i = 0; i < 4; i += 2) {                         // pair (d + i, d + i + 1): GPT-J, inverse
            const int p = (d + i) >> 1;
            const float c = p >= 224 ? csr[p - 224] : 1.f, s = p >= 224 ? csr[32 + p - 224] : 0.f;
            const float ne = __fmaf_rn(x[i], c, __fmul_rn(x[i + 1], s));
            const float no = __fmaf_rn(x[i + 1], c, -__fmul_rn(x[i], s));
            const __nv_bfloat16 b0 = __float2bfloat16_rn(ne), b1 = __float2bfloat16_rn(no);
            ob[i] = *reinterpret_cast<const uint16_t*>(&b0);
            ob[i + 1] = *reinterpret_cast<const uint16_t*>(&b1);
        }
        uint2 w;
        w.x = ob[0] | (static_cast<uint32_t>(ob[1]) << 16);
        w.y = ob[2] | (static_cast<uint32_t>(ob[3]) << 16);
        *reinterpret_cast<uint2*>(a.out + (static_cast<long long>(r) * H + h) * D + d) = w;
    }
}

// merge_cols_r's units again: a warp holds one head's 128-column block (lane L its columns 4L..4L+3), as rot_in's
template <int NC>
__device__ __forceinline__ void rot_cols(const Args& a, const RotIn& ro, int r, int hb, int c0) {
    const int tid = threadIdx.x, lane = tid & 31, H = a.H;
    constexpr int UB = BMQ * (NC / 4) / NT;
#pragma unroll
    for (int ub = 0; ub < UB; ++ub) {
        const int u = tid + ub * NT, hl = u / (NC / 4), d = c0 + (u % (NC / 4)) * 4, h = hb * BMQ + hl;
        // this thread's own 4 bf16 of o as merge_cols_r stored them, widened as x3seg's load4 widens them
        const uint2 w = *reinterpret_cast<const uint2*>(a.out + (static_cast<long long>(r) * H + h) * D + d);
        const float2 o0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.x));
        const float2 o1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.y));
        const float x[4] = {o0.x, o0.y, o1.x, o1.y};
        const int col = h * D + d, s = col / ro.gk, j = col - s * ro.gk;   // wo_a's group and its k
        half* xh = ro.xh[0];
        const half* suh = ro.suh[0];
#pragma unroll
        for (int i = 1; i < 8; ++i)
            if (i == s) {
                xh = ro.xh[i];
                suh = ro.suh[i];
            }
        const uint2 su = *reinterpret_cast<const uint2*>(suh + j);
        const float2 s0 = __half22float2(*reinterpret_cast<const half2*>(&su.x));
        const float2 s1 = __half22float2(*reinterpret_cast<const half2*>(&su.y));
        const float sc[4] = {s0.x, s0.y, s1.x, s1.y};
        float v[4];
        rot128(x, sc, lane, v);
        const half2 p = __floats2half2_rn(v[0], v[1]), q = __floats2half2_rn(v[2], v[3]);
        uint2 st;
        st.x = *reinterpret_cast<const uint32_t*>(&p);
        st.y = *reinterpret_cast<const uint32_t*>(&q);
        *reinterpret_cast<uint2*>(xh + static_cast<long long>(r) * ro.gk + j) = st;   // x3seg's store4 (fp16)
    }
}

// split_kernel's statements verbatim (attn_split.cuh) on merge_cols_r, then rot_cols over the columns it stored
template <int KW, bool ROWS, bool HAS_HI, int P>
__global__ void __launch_bounds__(NT, 1) split_rot_kernel(Args a, const RotIn ro) {
    using G = Split<P>;
    extern __shared__ __align__(16) uint8_t sm[];
    const int k = P == 1 ? 0 : static_cast<int>(cl_rank());
    const int c = blockIdx.x / P, hb = blockIdx.y, r = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int H = a.H, R = a.R;
    if (!chunk_works(a, r, c)) return;                           // the whole cluster: no barrier is pending
    uint4 qv[32 * 32 / NT];                                      // q's A fragments: every load in flight at once
    {
        const uint16_t* qb = a.q + (static_cast<long long>(r) * H + hb * BMQ) * D;
#pragma unroll
        for (int u = 0; u < 32 * 32 / NT; ++u) {
            const int i = tid + u * NT, s = i >> 5, ln = i & 31, gg = ln >> 2, tt = ln & 3;
            const int k0 = klo<KW>(s, tt), k1 = khi<KW>(s, tt);
            qv[u].x = *reinterpret_cast<const uint32_t*>(qb + gg * D + k0);
            qv[u].y = *reinterpret_cast<const uint32_t*>(qb + (gg + 8) * D + k0);
            qv[u].z = *reinterpret_cast<const uint32_t*>(qb + gg * D + k1);
            qv[u].w = *reinterpret_cast<const uint32_t*>(qb + (gg + 8) * D + k1);
        }
        if (a.qrope) {                                           // the rope dims (k >= 448): their pair's rotation
            const float* csr = a.cs + static_cast<long long>(row_pos<ROWS>(a, r)) * a.csst;
#pragma unroll
            for (int u = 0; u < 32 * 32 / NT; ++u) {
                const int i = tid + u * NT, s = i >> 5, tt = (i & 31) & 3;
                const int k0 = klo<KW>(s, tt), k1 = khi<KW>(s, tt);
                if (k0 >= NOPE) {
                    qv[u].x = rope_word(qv[u].x, csr, (k0 - NOPE) >> 1);
                    qv[u].y = rope_word(qv[u].y, csr, (k0 - NOPE) >> 1);
                }
                if (k1 >= NOPE) {
                    qv[u].z = rope_word(qv[u].z, csr, (k1 - NOPE) >> 1);
                    qv[u].w = rope_word(qv[u].w, csr, (k1 - NOPE) >> 1);
                }
            }
        }
    }
    int* phys = reinterpret_cast<int*>(sm + G::SM_PHYS);
    const long long sl = ROWS ? a.sl[r] : 0;
    if (tid < CH) {                                              // attn_kernel's row map
        int row = -1;
        if (c < NCOMP) {
            const int idx = c * CH + tid;
            if (idx < a.cnt[r]) {
                const int tk = a.tok[static_cast<long long>(r) * a.ts + idx];
                if (tk >= 0) {
                    row = tk;
                    if (a.psh > 0) {
                        const int* ptr = a.pt + (ROWS ? sl * a.pts : 0);
                        row = (ptr[tk >> a.psh] << a.psh) + (tk & ((1 << a.psh) - 1));
                    }
                }
            }
        } else {
            const int hi = HAS_HI ? a.hi[r] : row_pos<ROWS>(a, r);
            const int lo = max(a.lo[r], hi - (WINDOW - 1));
            const int p = hi - (WINDOW - 1) + tid;
            if (p >= lo && p >= 0) row = (p % a.ring) + (ROWS ? static_cast<int>(sl) * a.ring : 0);
        }
        phys[tid] = row;
    }
    __syncthreads();
    {                                                            // the rows in flight before q's fragments
        const bool comp = c < NCOMP;
        const uint8_t* V = comp ? a.cv : a.sv;
        const uint8_t* S = comp ? a.csc : a.ssc;
        const long long vs = comp ? a.cvs : a.svs, ss = comp ? a.css : a.sss;
        const bool v16 = ((reinterpret_cast<uintptr_t>(V) | static_cast<uintptr_t>(vs)) & 15) == 0;
        const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(sm));
        const int x0 = col2byte(G::W * k), nsl = (col2byte(G::W * k + G::W) - x0) / 16;   // the slice's pieces a key
        for (int i = tid; i < CH * nsl; i += NT) {
            const int key = i / nsl, pc = i % nsl, row = phys[key];
            const long long rr = row >= 0 ? row : 0;
            stage_piece(base, G::SM_KP, key, G::SLB, pc, V + rr * vs + x0 + pc * 16, row >= 0, v16);
        }
        if (P > 1) {                                             // the S tiles' whole rows
            for (int i = tid; i < G::KS * (VB / 16); i += NT) {
                const int key = i / (VB / 16), pc = i % (VB / 16), row = phys[G::KS * k + key];
                const long long rr = row >= 0 ? row : 0;
                stage_piece(base, G::SM_KS, key, VB, pc, V + rr * vs + pc * 16, row >= 0, v16);
            }
        }
        for (int key = tid; key < CH; key += NT) {
            const int row = phys[key];
            cp8(base + G::SM_SC + key * SB, S + (row >= 0 ? row : 0) * ss, row >= 0);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
#pragma unroll
    for (int u = 0; u < 32 * 32 / NT; ++u) reinterpret_cast<uint4*>(sm + G::SM_Q)[tid + u * NT] = qv[u];
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if (P > 1) {                                                 // S's operands -> bf16 pairs, 4 k a step (deq2)
        uint32_t* kb = reinterpret_cast<uint32_t*>(sm + G::SM_KB);
        for (int i = tid; i < G::KS * (D / 4); i += NT) {
            const int key = i / (D / 4), k4 = (i % (D / 4)) * 4, w = k4 >> 1;
            uint32_t w0, w1;
            if (k4 < NOPE) {
                const uint32_t b = *reinterpret_cast<const uint32_t*>(sm + G::SM_KS + swz16(key, k4, VB));
                const float sc = pow2b(sm[G::SM_SC + (key + G::KS * k) * SB + (k4 >> 6)]);
                w0 = deq2(b & 0xffffu, sc, sc);
                w1 = deq2(b >> 16, sc, sc);
            } else {
                const uint2 b = *reinterpret_cast<const uint2*>(sm + G::SM_KS + swz16(key, NOPE + 2 * (k4 - NOPE), VB));
                w0 = b.x;
                w1 = b.y;
            }
            kb[key * (D / 2) + (w ^ bsw<KW>(key))] = w0;
            kb[key * (D / 2) + ((w + 1) ^ bsw<KW>(key))] = w1;
        }
        __syncthreads();
    }
    const float NINF = -INFINITY;
    float* tmax = reinterpret_cast<float*>(sm + G::SM_TMAX);
    float* wsum = reinterpret_cast<float*>(sm + G::SM_WSUM);
    float* rowv = reinterpret_cast<float*>(sm + G::SM_ROW);
    // S: chain pair w (+ 8): local tile w >> 2, block n8 = w & 3 (P = 1: attn_kernel's tiles ta and ta + 2)
    const int n8 = warp & 3;
    const bool sw = warp < 4 * G::TPC;                           // this warp has S chains
    float sv[G::NPW][4];
    bool val[G::NPW][2];
    int tile_of[G::NPW];
#pragma unroll
    for (int j = 0; j < G::NPW; ++j) {
        tile_of[j] = G::TPC * k + (warp >> 2) + 2 * j;
        for (int e = 0; e < 4; ++e) sv[j][e] = 0.f;
    }
    if (sw) {
        const uint4* qf = reinterpret_cast<const uint4*>(sm + G::SM_Q);
        int lk[G::NPW];
#pragma unroll
        for (int j = 0; j < G::NPW; ++j) lk[j] = 32 * (tile_of[j] - G::TPC * k) + 8 * n8 + g;
#pragma unroll 4
        for (int s = 0; s < 32; ++s) {
            const uint4 af = qf[s * 32 + lane];
            const int k0 = klo<KW>(s, t), k1 = khi<KW>(s, t);
#pragma unroll
            for (int j = 0; j < G::NPW; ++j) {
                if (P == 1) {
                    mma(sv[j], af, spair_k<P>(sm, lk[j], lk[j], k0), spair_k<P>(sm, lk[j], lk[j], k1));
                } else {
                    mma(sv[j], af, sbf_k<KW, P>(sm, lk[j], k0), sbf_k<KW, P>(sm, lk[j], k1));
                }
            }
        }
#pragma unroll
        for (int j = 0; j < G::NPW; ++j) {
            const int tile = tile_of[j], kc = 32 * tile + 8 * n8 + 2 * t;
            val[j][0] = phys[kc] >= 0;
            val[j][1] = phys[kc + 1] >= 0;
            for (int e = 0; e < 4; ++e)
                sv[j][e] = val[j][e & 1] ? __fmul_rn(sv[j][e], __int_as_float(0x3D3504F3)) : NINF;
            float m0 = fmaxf(sv[j][0], sv[j][1]), m1 = fmaxf(sv[j][2], sv[j][3]);
            m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 2));
            m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 1));
            m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 2));
            m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 1));
            if (t == 0) {
                tmax[(tile * 4 + n8) * 16 + g] = m0;
                tmax[(tile * 4 + n8) * 16 + g + 8] = m1;
            }
        }
    }
    if (P > 1) {                                                 // this rank's tile maxima pushed to the others
        __syncthreads();
        for (int i = tid; i < 64 * G::TPC * (P - 1); i += NT) {
            const int e = 64 * G::TPC * k + i % (64 * G::TPC), q = (k + 1 + i / (64 * G::TPC)) % P;
            st_peer(tmax + e, q, __float_as_uint(tmax[e]));
        }
        cl_arrive();
        const int x0 = col2byte(G::W * k);                       // meanwhile: PV's B pairs, the fp8 slice -> bf16,
        uint32_t* vb = reinterpret_cast<uint32_t*>(sm + G::SM_VB);   // 4 columns a step (deq2 per column)
        for (int i = tid; i < (CH / 2) * (G::W / 4); i += NT) {
            const int kp = i / (G::W / 4), x = (i % (G::W / 4)) * 4, n = G::W * k + x, k0 = 2 * kp, k1 = k0 + 1;
            uint32_t o[4];
            if (n < NOPE) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(sm + G::SM_KP + swz16(k0, n - x0, G::SLB));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(sm + G::SM_KP + swz16(k1, n - x0, G::SLB));
                const float s0 = pow2b(sm[G::SM_SC + k0 * SB + (n >> 6)]), s1 = pow2b(sm[G::SM_SC + k1 * SB + (n >> 6)]);
#pragma unroll
                for (int e = 0; e < 4; ++e) o[e] = deq2(((b0 >> (8 * e)) & 0xffu) | (((b1 >> (8 * e)) & 0xffu) << 8), s0, s1);
            } else {
                const uint2 a0 = *reinterpret_cast<const uint2*>(sm + G::SM_KP + swz16(k0, col2byte(n) - x0, G::SLB));
                const uint2 a1 = *reinterpret_cast<const uint2*>(sm + G::SM_KP + swz16(k1, col2byte(n) - x0, G::SLB));
                o[0] = (a0.x & 0xffffu) | (a1.x << 16);
                o[1] = (a0.x >> 16) | (a1.x & 0xffff0000u);
                o[2] = (a0.y & 0xffffu) | (a1.y << 16);
                o[3] = (a0.y >> 16) | (a1.y & 0xffff0000u);
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) vb[kp * G::W + ((x + e) ^ vsw<KW>(kp))] = o[e];
        }
        cl_wait();
    }
    __syncthreads();
    if (tid < BMQ) {                                             // the running max over the tiles, a row a thread
        float m = NINF;
        for (int tile = 0; tile < 4; ++tile) {
            const float* tm4 = tmax + tile * 64 + tid;
            const float tm = fmaxf(fmaxf(tm4[0], tm4[16]), fmaxf(tm4[32], tm4[48]));
            const bool act = tm != NINF;
            const float nm = act ? fmaxf(m, tm) : m;
            const float al = act ? (m == NINF ? 0.f : texp(__fsub_rn(m, nm))) : 1.f;
            rowv[tile * 16 + tid] = nm;
            rowv[64 + tile * 16 + tid] = al;
            rowv[128 + tile * 16 + tid] = act ? 1.f : 0.f;
            m = nm;
        }
        reinterpret_cast<float*>(sm + G::SM_MISC)[tid] = m;
    }
    __syncthreads();
    uint32_t* pf = reinterpret_cast<uint32_t*>(sm + G::SM_P);    // [tile][step][lane][4]
    if (sw) {
#pragma unroll
        for (int j = 0; j < G::NPW; ++j) {                       // p, the quad sums, bf16(p) -> fragments
            const int tile = tile_of[j];
            float p[4];
            for (int e = 0; e < 4; ++e) {
                const int row = g + (e >> 1) * 8;
                const bool act = rowv[128 + tile * 16 + row] != 0.f;
                p[e] = (val[j][e & 1] && act) ? texp(__fsub_rn(sv[j][e], rowv[tile * 16 + row])) : 0.f;
            }
            float a0 = __fadd_rn(p[0], p[1]), a1 = __fadd_rn(p[2], p[3]);
            a0 = __fadd_rn(a0, __shfl_xor_sync(FULL, a0, 2));
            a0 = __fadd_rn(a0, __shfl_xor_sync(FULL, a0, 1));
            a1 = __fadd_rn(a1, __shfl_xor_sync(FULL, a1, 2));
            a1 = __fadd_rn(a1, __shfl_xor_sync(FULL, a1, 1));
            if (t == 0) {
                wsum[(tile * 4 + n8) * 16 + g] = a0;
                wsum[(tile * 4 + n8) * 16 + g + 8] = a1;
            }
            int step, tq, hi;
            frag_of<KW>(8 * n8 + 2 * t, step, tq, hi);
            const __nv_bfloat162 lo2 = __floats2bfloat162_rn(p[0], p[1]), hi2 = __floats2bfloat162_rn(p[2], p[3]);
            uint32_t* f = pf + ((tile * 2 + step) * 32 + g * 4 + tq) * 4;
            f[2 * hi] = *reinterpret_cast<const uint32_t*>(&lo2);
            f[2 * hi + 1] = *reinterpret_cast<const uint32_t*>(&hi2);
        }
    }
    if (P > 1) {                                                 // this rank's row sums and fragments pushed
        __syncthreads();
        constexpr int NW = G::TPC * (64 + 256);                  // words of this rank's tiles
        for (int i = tid; i < NW * (P - 1); i += NT) {
            const int e = i % NW, q = (k + 1 + i / NW) % P;
            uint32_t* w = e < 64 * G::TPC ? reinterpret_cast<uint32_t*>(wsum) + 64 * G::TPC * k + e
                                          : pf + 256 * G::TPC * k + (e - 64 * G::TPC);
            st_peer(w, q, *w);
        }
        cl_arrive();                                             // the last cluster access: no barrier before exit
        cl_wait();
    }
    __syncthreads();
    if (tid < BMQ) {                                             // l: fma(l, alpha, (W0 + W2) + (W1 + W3))
        float l = 0.f;
        for (int tile = 0; tile < 4; ++tile) {
            const float* w4 = wsum + tile * 64 + tid;
            const float sum = __fadd_rn(__fadd_rn(w4[0], w4[32]), __fadd_rn(w4[16], w4[48]));
            l = __fmaf_rn(l, rowv[64 + tile * 16 + tid], sum);
        }
        reinterpret_cast<float*>(sm + G::SM_MISC)[16 + tid] = l;
    }
    const int x0 = col2byte(G::W * k);
    float acc[G::J][4];
#pragma unroll
    for (int j = 0; j < G::J; ++j)
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
    for (int tile = 0; tile < 4; ++tile) {                       // o = mma(P, kv, rn(o * alpha)), 2 k steps a tile
        const float al0 = rowv[64 + tile * 16 + g], al1 = rowv[64 + tile * 16 + g + 8];
#pragma unroll
        for (int j = 0; j < G::J; ++j) {
            acc[j][0] = __fmul_rn(acc[j][0], al0);
            acc[j][1] = __fmul_rn(acc[j][1], al0);
            acc[j][2] = __fmul_rn(acc[j][2], al1);
            acc[j][3] = __fmul_rn(acc[j][3], al1);
        }
        for (int step = 0; step < 2; ++step) {
            const uint4 af = reinterpret_cast<const uint4*>(pf)[(tile * 2 + step) * 32 + lane];
            const int k0 = 32 * tile + klo<KW>(step, t), k1 = 32 * tile + khi<KW>(step, t);
#pragma unroll
            for (int j = 0; j < G::J; ++j) {
                const int n = 64 * (G::J * k + j) + 8 * warp + g;
                if (P == 1) {
                    mma(acc[j], af, spair_n<P>(sm, k0, k0 + 1, n, x0), spair_n<P>(sm, k1, k1 + 1, n, x0));
                } else {
                    const uint32_t* vb = reinterpret_cast<const uint32_t*>(sm + G::SM_VB);
                    const int x = n - G::W * k;
                    mma(acc[j], af, vb[(k0 >> 1) * G::W + (x ^ vsw<KW>(k0 >> 1))],
                        vb[(k1 >> 1) * G::W + (x ^ vsw<KW>(k1 >> 1))]);
                }
            }
        }
    }
    const long long pbase = (static_cast<long long>(c) * R + r) * H + hb * BMQ;
#pragma unroll
    for (int j = 0; j < G::J; ++j) {
        const int n = 64 * (G::J * k + j) + 8 * warp + 2 * t;
        *reinterpret_cast<float2*>(a.po + (pbase + g) * D + n) = make_float2(acc[j][0], acc[j][1]);
        *reinterpret_cast<float2*>(a.po + (pbase + g + 8) * D + n) = make_float2(acc[j][2], acc[j][3]);
    }
    if (tid < BMQ) {                                             // every rank: the same (m, l), so each merge sees it
        a.pm[pbase + tid] = reinterpret_cast<float*>(sm + G::SM_MISC)[tid];
        a.pl[pbase + tid] = reinterpret_cast<float*>(sm + G::SM_MISC)[16 + tid];
    }
    int* flag = reinterpret_cast<int*>(sm + G::SM_MISC) + 32;
    int* tk = a.ticket + (r * (H / BMQ) + hb) * MAXP + k;
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        int need = 0;
        for (int cc = 0; cc < NCH; ++cc) need += chunk_works(a, r, cc) ? 1 : 0;
        *flag = atomicAdd(tk, 1) == need - 1;
    }
    __syncthreads();
    if (!*flag) return;
    __threadfence();
    if (tid == 0) atomicExch(tk, 0);                             // self-cleaning: the next launch / replay
    merge_cols_r<ROWS, G::W>(a, sm, G::SM_TMAX, r, hb, G::W * k);
    rot_cols<G::W>(a, ro, r, hb, G::W * k);                     // the fold: its own columns' rows
}
