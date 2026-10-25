// MiniMax-M3.1-Flash block-sparse attention: constants and paged addressing shared by minimax_m3_{index,select,attn}.cu.
// Semantics: core/minimax_ref.zig (float64). Nothing here is copied from another project; written from the formulas in
// the engine-port plan (section 2, T5).
//
// Page = 128 tokens = one selection block. Inside one page and one layer (head-major, bf16):
//   K[kvh][128 tok][128] | V[kvh][128][128] | IK[128 tok][128]
// A pool address is  base + page * page_bytes + layer_off + element offset * 2; the kernels take page ids from a block
// table (tables[row_tab[r] + block]), never an address, so a pool may move or grow between launches.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace tf_mm3 {

constexpr int D = 128;     // head_dim and index_head_dim
constexpr int BLOCK = 128; // keys a selection block (= tokens a page)
constexpr int TOPK = 16;   // blocks a row attends (sparse_topk_blocks)
constexpr int GROUP = 16;  // query heads a kv head (64 / 4)

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float bf(uint32_t half_bits) { return __uint_as_float(half_bits << 16); }
__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

constexpr long long K_ELEMS = (long long)BLOCK * D; // one head's block of K (or V)

/// Element offset (bf16) of K (which 0) or V (which 1) head h, token t inside a page's layer region.
__device__ __forceinline__ long long kv_elem(int kvh, int which, int h, int t) {
    return ((long long)which * kvh + h) * K_ELEMS + (long long)t * D;
}

/// Element offset of index-key token t.
__device__ __forceinline__ long long ik_elem(int kvh, int t) { return 2LL * kvh * K_ELEMS + (long long)t * D; }

/// fp8 layout (kv8) of a layer: K8 [kvh][128][128] | V8 | IK bf16 [128][128] | K scales f32 [kvh][128] | V scales.

/// fp8 layout, bytes inside a layer region: the e4m3 row of K (which 0) or V (which 1), head h, token t.
__device__ __forceinline__ long long kv8_byte(int kvh, int which, int h, int t) {
    return ((long long)which * kvh + h) * K_ELEMS + (long long)t * D;
}

/// Byte offset inside a layer region of the index keys (K and V take 2 bytes a value, fp8's 1; IK is bf16 in both).
__device__ __forceinline__ long long ik_byte(int kvh, int kv8) { return 2LL * kvh * K_ELEMS * (kv8 ? 1 : 2); }

/// fp8 layout: byte offset inside a layer region of the fp32 scale of the row kv8_byte names (after the 32 KiB of IK).
__device__ __forceinline__ long long scale8_byte(int kvh, int which, int h, int t) {
    return 2LL * kvh * K_ELEMS + K_ELEMS * 2 + (((long long)which * kvh + h) * BLOCK + t) * 4;
}

__device__ __forceinline__ const char* page_layer(const char* pool, long long page_bytes, long long layer_off, int page) {
    return pool + (long long)page * page_bytes + layer_off;
}

/// A row's position: pos[r], or pos0 + r when pos is null (a prompt chunk of one stream).
__device__ __forceinline__ int row_pos(const int* pos, int pos0, int r) { return pos ? __ldg(pos + r) : pos0 + r; }

/// Offset of a row's block table in `tables` (row_tab null: one table for every row).
__device__ __forceinline__ int row_table(const int* row_tab, int r) { return row_tab ? __ldg(row_tab + r) : 0; }

} // namespace tf_mm3
