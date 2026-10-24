// Prompt-sized routed mul1 EXL3 experts on the tensor units: x3gm gate/up (two inputs), down, plan, rotation, combine.

#define DSV41_X3GM_NO_INSTANCES
#define X3GM_FLUSH_K 384  // the decode kernel's per-warp chain: every output an fp32 sum of 384-k chains
#include "exl3_block/x3gm.cu"
#include "exl3_block/x3gm_plan.cu"

// out[r] = the fma chain over slots in order of wts[r][q] * y[r * slots + q]: tf_exl3_down_combine's combine bits.
extern "C" __global__ void __launch_bounds__(256) tf_exl3_gm_combine(const float* __restrict__ y,
                                                                      const float* __restrict__ wts,
                                                                      float* __restrict__ out, int d, int slots) {
    const int r = blockIdx.x;
    const int c = blockIdx.y * 256 + threadIdx.x;
    if (c >= d) return;
    float acc = 0.f;
    for (int q = 0; q < slots; ++q) acc = fmaf(wts[r * slots + q], y[((size_t)r * slots + q) * d + c], acc);
    out[(size_t)r * d + c] = acc;
}

#define TF_GM_GU(K2, CFG) template __global__ void dsv41_x3gm::gm_kernel<2, 2, K2, CFG>(const dsv41_x3gm::Args);
#define TF_GM_DN(K2, CFG) template __global__ void dsv41_x3gm::gm_kernel<1, 1, K2, CFG>(const dsv41_x3gm::Args);
#define TF_GM_WIDTH(K2)                                                                                              \
    TF_GM_GU(K2, GM_GU0) TF_GM_GU(K2, GM_GU1) TF_GM_GU(K2, GM_GU2) TF_GM_GU(K2, GM_GU3) TF_GM_GU(K2, GM_GU4)          \
    TF_GM_GU(K2, GM_GU5) TF_GM_DN(K2, GM_DN0) TF_GM_DN(K2, GM_DN1) TF_GM_DN(K2, GM_DN2) TF_GM_DN(K2, GM_DN3)          \
    TF_GM_DN(K2, GM_DN4)
TF_GM_WIDTH(3)
TF_GM_WIDTH(4)
TF_GM_WIDTH(5)
TF_GM_WIDTH(6)
TF_GM_WIDTH(7)
TF_GM_WIDTH(8)
TF_GM_WIDTH(10)
// v2: every expert at its own width in one launch (ragged layers)
template __global__ void dsv41_x3gm::gm2_kernel<2, 2, GM_GU0>(const dsv41_x3gm::Args);
template __global__ void dsv41_x3gm::gm2_kernel<2, 2, GM_GU1>(const dsv41_x3gm::Args);
template __global__ void dsv41_x3gm::gm2_kernel<1, 1, GM_DN0>(const dsv41_x3gm::Args);
template __global__ void dsv41_x3gm::rot_kernel<__nv_bfloat16, 2>(const __nv_bfloat16*, int, const int*, const half*,
                                                                  const half*, half*, half*, int, int, int);
