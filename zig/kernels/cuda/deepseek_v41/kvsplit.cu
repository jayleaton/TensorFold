// Split KV's exchange copies (zig/src/families/deepseek_v41/kv/split.zig: HostKernels is the reference): before the
// all-gather, the selected rows of a split family packed dense, and a row list gathered. Pure copies (bit-exact by
// construction), 8 bytes a thread (row bytes are multiples of 8: 584 FP8 KV rows, 384 / 132 + pad index rows), no
// host sync, graph-capturable.
//
//   dense:  for i < R K: t = max(sel[i], 0); s = rslot ? rslot[i / K] : 0; local = table[s pts + (t >> psh)]
//           send[i] = base row ((local << psh) | (t & (2^psh - 1)));  tok[i] = sel[i] < 0 ? -1 : ((t >> psh) % world) R K + i
//   gather: send[i] = base row phys[i]

#include <cstdint>

namespace dsv41_kvsplit {

// kv/split.zig DenseArgs (extern, by value): the same fields at the same offsets
struct DenseArgs {
    const int* sel;          // int32 [R, K] logical rows (-1: none)
    uint32_t rows, k;
    const int* table;        // int32 split page tables: [pages], or row mode's stacked [S, pts]
    uint32_t pts;
    const int* rslot;        // int32 [R] or null
    uint32_t psh;
    const uint8_t* base;     // the family's local rows
    uint32_t row_bytes, world;
    uint8_t* send;           // [R K, row bytes]
    int* tok;                // int32 [R, K]
};
static_assert(sizeof(DenseArgs) == 80, "kv/split.zig DenseArgs");

__global__ void __launch_bounds__(256) dense_kernel(const DenseArgs a) {
    const uint32_t wpr = a.row_bytes >> 3;                      // 8-byte words a row
    const long long n = (long long)a.rows * a.k;
    const long long words = n * wpr;
    for (long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; w < words; w += (long long)gridDim.x * blockDim.x) {
        const long long i = w / wpr;
        const uint32_t q = (uint32_t)(w - i * wpr);
        const int sv = a.sel[i];
        const uint32_t t = sv > 0 ? (uint32_t)sv : 0u;
        const long long s = a.rslot ? (long long)a.rslot[i / a.k] : 0;
        const uint32_t local = (uint32_t)a.table[s * a.pts + (t >> a.psh)];
        const unsigned long long row = ((unsigned long long)local << a.psh) | (t & ((1u << a.psh) - 1u));
        const uint64_t* src = reinterpret_cast<const uint64_t*>(a.base + row * a.row_bytes);
        reinterpret_cast<uint64_t*>(a.send + (unsigned long long)i * a.row_bytes)[q] = src[q];
        if (q == 0) a.tok[i] = sv < 0 ? -1 : (int)((long long)((t >> a.psh) % a.world) * n + i);
    }
}

__global__ void __launch_bounds__(256) gather_kernel(const uint8_t* __restrict__ base, uint32_t row_bytes,
                                                     const uint32_t* __restrict__ phys, uint32_t n,
                                                     uint8_t* __restrict__ send) {
    const uint32_t wpr = row_bytes >> 3;
    const long long words = (long long)n * wpr;
    for (long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; w < words; w += (long long)gridDim.x * blockDim.x) {
        const long long i = w / wpr;
        const uint32_t q = (uint32_t)(w - i * wpr);
        const uint64_t* src = reinterpret_cast<const uint64_t*>(base + (unsigned long long)phys[i] * row_bytes);
        reinterpret_cast<uint64_t*>(send + (unsigned long long)i * row_bytes)[q] = src[q];
    }
}

}  // namespace dsv41_kvsplit
