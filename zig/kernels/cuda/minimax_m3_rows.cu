// MiniMax-M3.1-Flash row kernels from the engine-port plan (section 2, steps 3-6 and 11a); no third-party code.

// tf_mm3_rows: per-head Gemma RMSNorm of q, k and index heads, partial RoPE; q, iq to scratch, k, v, ik to pages.

// Reference: core/minimax_ref.zig gemmaNorm then rope. A CTA is a row, a head a warp: 4 values a lane, xor tree.

// Rotation angles are float64 (pos * theta^(-2i/rot), sincos) rounded once to fp32: fp32 angles are off by 1e-2 at 1e6.

// tf_mm3_router_*: fp32 logits[r][e] = sum_k w[e][k] * x[r][k]; a warp owns one expert and up to 4 rows.

// Lane l sums chunks l, l + 32, ... in four fp32 fma chains, then (a0 + a1) + (a2 + a3), then a fixed xor tree 16 .. 1.

// That order never looks at the row count, the tile or the neighbours: a row's logits are the same bits at any width.

// NaN rule: none stated, arithmetic propagates NaN and infinities as the float64 reference does.

// flags & ROWS_KV8 (off: the bf16 pool as before): k and v as quant_row8's e4m3 rows (docs/fp8-kv.md).
#include "minimax_m3_rows.cuh"

using namespace tf_mm3;

namespace {

__device__ __forceinline__ float tree32(float a) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, o));
    return a;
}

// Output dim d of the rotated head: stage holds the normed head, c and s the pair table.
__device__ __forceinline__ float rotated(const float* stage, const float (*cs)[MAX_ROT_PAIRS], int d, int rot, bool gptj) {
    const int half = rot >> 1;
    int i, ia, ib;
    if (gptj) {
        i = d >> 1;
        ia = 2 * i;
        ib = ia + 1;
    } else {
        i = d < half ? d : d - half;
        ia = i;
        ib = i + half;
    }
    const float a = stage[ia], b = stage[ib], c = cs[0][i], s = cs[1][i];
    const bool first = gptj ? (d & 1) == 0 : d < half;
    return first ? __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)) : __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
}

} // namespace

extern "C" __global__ void __launch_bounds__(ROWS_THREADS) tf_mm3_rows(
    const bf16* __restrict__ q_in, long long q_ld,         // [rows, nq * 128]
    const bf16* __restrict__ k_in, long long k_ld,         // [rows, kvh * 128]
    const bf16* __restrict__ v_in, long long v_ld,         // [rows, kvh * 128]
    const unsigned short* __restrict__ iq_in, long long iq_ld,  // [rows, niq * 128], fp16 or bf16 bits
    const unsigned short* __restrict__ ik_in, long long ik_ld,  // [rows, 128], fp16 or bf16 bits
    const float* __restrict__ q_w, const float* __restrict__ k_w, const float* __restrict__ iq_w,
    const float* __restrict__ ik_w,                        // [128] each, null when that norm is off
    bf16* __restrict__ q_out, long long q_out_ld, bf16* __restrict__ iq_out, long long iq_out_ld,
    const int* __restrict__ pos, int pos0,                 // [rows] or null (pos0 + r)
    const int* __restrict__ row_tab, const int* __restrict__ tables, int max_pages,
    char* pool, long long page_bytes, long long layer_off,
    int rows, int nq, int niq, int kvh, int has_ik, int flags, int rot, float eps, float w_add, float w_add_index, double theta,
    int* status) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int p = row_pos(pos, pos0, r);
    if (p < 0) return;
    __shared__ float cs[2][MAX_ROT_PAIRS];
    __shared__ float stage[ROWS_WARPS][D];
    const int half = rot >> 1;
    const int blk = p / BLOCK;
    const int page = blk < max_pages ? __ldg(tables + row_table(row_tab, r) + blk) : -1;
    if (threadIdx.x == 0 && status != nullptr && page < 0) atomicOr(status, blk < max_pages ? ROWS_STATUS_NO_PAGE : ROWS_STATUS_PAST_TABLE);
    if (threadIdx.x < half) {
        double sn, cn;
        sincos((double)p * pow(theta, -2.0 * (double)threadIdx.x / (double)rot), &sn, &cn);
        cs[0][threadIdx.x] = (float)cn;
        cs[1][threadIdx.x] = (float)sn;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int t = p % BLOCK;
    const bool gptj = (flags & ROWS_GPTJ) != 0;
    const bool kv8 = (flags & ROWS_KV8) != 0;
    const int total = nq + niq + 2 * kvh + (has_ik ? 1 : 0);
    char* layer = page >= 0 ? pool + (long long)page * page_bytes + layer_off : nullptr;
    for (int s = warp; s < total; s += ROWS_WARPS) {
        int kind = 0, h = s;  // 0 q, 1 index q, 2 k, 3 v, 4 index key
        if (s >= nq) { kind = 1; h = s - nq; }
        if (s >= nq + niq) { kind = 2; h = s - nq - niq; }
        if (s >= nq + niq + kvh) { kind = 3; h = s - nq - niq - kvh; }
        if (s >= nq + niq + 2 * kvh) { kind = 4; h = 0; }
        const unsigned short* src;
        const float* w;
        bool norm, rope, fp16 = false;
        switch (kind) {
        case 0: src = reinterpret_cast<const unsigned short*>(q_in + r * q_ld + h * D); w = q_w; norm = flags & ROWS_NORM_QK; rope = true; break;
        case 1: src = iq_in + r * iq_ld + h * D; w = iq_w; norm = flags & ROWS_NORM_INDEX; rope = flags & ROWS_INDEX_ROPE; fp16 = flags & ROWS_INDEX_FP16; break;
        case 2: src = reinterpret_cast<const unsigned short*>(k_in + r * k_ld + h * D); w = k_w; norm = flags & ROWS_NORM_QK; rope = true; break;
        case 3: src = reinterpret_cast<const unsigned short*>(v_in + r * v_ld + h * D); w = nullptr; norm = false; rope = false; break;
        default: src = ik_in + r * ik_ld; w = ik_w; norm = flags & ROWS_NORM_INDEX; rope = flags & ROWS_INDEX_ROPE; fp16 = flags & ROWS_INDEX_FP16; break;
        }
        const uint2 raw = __ldg(reinterpret_cast<const uint2*>(src) + lane);
        if (kind == 3) {  // v is stored as it came (fp8: as its bf16 value's e4m3 row)
            if (layer != nullptr) {
                if (kv8) quant_row8(widen4(raw), reinterpret_cast<unsigned char*>(layer + kv8_byte(kvh, 1, h, t)),
                                    reinterpret_cast<float*>(layer + scale8_byte(kvh, 1, h, t)), lane);
                else reinterpret_cast<uint2*>(layer + kv_elem(kvh, 1, h, t) * 2)[lane] = raw;
            }
            continue;
        }
        float4 x = fp16 ? widen4_half(raw) : widen4(raw);
        if (norm) {
            float ss = __fmul_rn(x.x, x.x);
            ss = __fadd_rn(ss, __fmul_rn(x.y, x.y));
            ss = __fadd_rn(ss, __fmul_rn(x.z, x.z));
            ss = __fadd_rn(ss, __fmul_rn(x.w, x.w));
            ss = tree32(ss);
            const float inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, 128.f), eps)));
            const float4 wv = __ldg(reinterpret_cast<const float4*>(w) + lane);
            const float wa = (kind == 1 || kind == 4) ? w_add_index : w_add;
            x.x = __fmul_rn(__fmul_rn(x.x, inv), __fadd_rn(wv.x, wa));
            x.y = __fmul_rn(__fmul_rn(x.y, inv), __fadd_rn(wv.y, wa));
            x.z = __fmul_rn(__fmul_rn(x.z, inv), __fadd_rn(wv.z, wa));
            x.w = __fmul_rn(__fmul_rn(x.w, inv), __fadd_rn(wv.w, wa));
            if (flags & ROWS_ROUND_NORM) x = make_float4(rn_bf16(x.x), rn_bf16(x.y), rn_bf16(x.z), rn_bf16(x.w));
        }
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
            o[i] = (rope && d < rot) ? rotated(st, cs, d, rot, gptj) : st[d];
        }
        __syncwarp();
        const uint2 packed = pack4(o[0], o[1], o[2], o[3]);
        if (kind == 0) {
            reinterpret_cast<uint2*>(q_out + r * q_out_ld + h * D)[lane] = packed;
        } else if (kind == 1) {
            reinterpret_cast<uint2*>(iq_out + r * iq_out_ld + h * D)[lane] = packed;
        } else if (layer != nullptr) {
            if (kind == 2 && kv8) {
                quant_row8(make_float4(rn_bf16(o[0]), rn_bf16(o[1]), rn_bf16(o[2]), rn_bf16(o[3])),
                           reinterpret_cast<unsigned char*>(layer + kv8_byte(kvh, 0, h, t)),
                           reinterpret_cast<float*>(layer + scale8_byte(kvh, 0, h, t)), lane);
            } else {
                const long long byte = kind == 2 ? kv_elem(kvh, 0, h, t) * 2 : ik_byte(kvh, kv8) + (long long)t * D * 2;
                reinterpret_cast<uint2*>(layer + byte)[lane] = packed;
            }
        }
    }
}

namespace {

__device__ __forceinline__ float4 load_x(const bf16* p) { return widen4(__ldg(reinterpret_cast<const uint2*>(p))); }
__device__ __forceinline__ float4 load_x(const float* p) { return __ldg(reinterpret_cast<const float4*>(p)); }

template <typename X>
__device__ __forceinline__ void router_body(const X* __restrict__ x, long long x_ld, const float* __restrict__ w,
                                            float* __restrict__ out, int rows, int ne, int k) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * ROUTER_WARPS + warp;
    if (e >= ne) return;
    const long long r0 = (long long)blockIdx.y * ROUTER_ROWS;
    float acc[ROUTER_ROWS][4];
#pragma unroll
    for (int i = 0; i < ROUTER_ROWS; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    const float* we = w + (long long)e * k;
#pragma unroll 4
    for (int j = 0; j < k / ROUTER_CHUNK; ++j) {
        const int c = 4 * (lane + 32 * j);
        const float4 wv = __ldg(reinterpret_cast<const float4*>(we + c));
#pragma unroll
        for (int i = 0; i < ROUTER_ROWS; ++i) {
            if (r0 + i >= rows) continue;
            const float4 xv = load_x(x + (r0 + i) * x_ld + c);
            acc[i][0] = __fmaf_rn(xv.x, wv.x, acc[i][0]);
            acc[i][1] = __fmaf_rn(xv.y, wv.y, acc[i][1]);
            acc[i][2] = __fmaf_rn(xv.z, wv.z, acc[i][2]);
            acc[i][3] = __fmaf_rn(xv.w, wv.w, acc[i][3]);
        }
    }
#pragma unroll
    for (int i = 0; i < ROUTER_ROWS; ++i) {
        const float sum = tree32(__fadd_rn(__fadd_rn(acc[i][0], acc[i][1]), __fadd_rn(acc[i][2], acc[i][3])));
        if (lane == 0 && r0 + i < rows) out[(r0 + i) * ne + e] = sum;
    }
}

} // namespace

// x bf16 [rows, k] at x_ld elements a row, w fp32 [ne, k], out fp32 [rows, ne]; k a multiple of 128, x_ld of 4.
extern "C" __global__ void __launch_bounds__(ROUTER_WARPS * 32) tf_mm3_router_bf16(
    const bf16* __restrict__ x, long long x_ld, const float* __restrict__ w, float* __restrict__ out, int rows, int ne, int k) {
    router_body<bf16>(x, x_ld, w, out, rows, ne, k);
}

// The same with an fp32 input: a bf16 hidden cast to fp32 gives the same bits as the bf16 form.
extern "C" __global__ void __launch_bounds__(ROUTER_WARPS * 32) tf_mm3_router_f32(
    const float* __restrict__ x, long long x_ld, const float* __restrict__ w, float* __restrict__ out, int rows, int ne, int k) {
    router_body<float>(x, x_ld, w, out, rows, ne, k);
}
