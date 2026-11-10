// The MTP draft head's 4-bit matrices: the cluster K-slice forms of fn_qmm.cu (Ash Hart's qmm.cu), drafts only.
#include "fn_qmm.cu"

#define TF_QMMC(BM, F32) template __global__ void tf_fn_qmm::qmm_kernel<32, BM, 64, 1, 4, 4, F32, true, false>( \
    const __nv_bfloat16*, const float*, const uint32_t*, const __nv_bfloat16*, const __nv_bfloat16*, void*, float*, \
    int, int, int, int, int, int, int);
TF_QMMC(16, false)
TF_QMMC(32, false)
TF_QMMC(64, false)
TF_QMMC(16, true)
TF_QMMC(32, true)
TF_QMMC(64, true)
