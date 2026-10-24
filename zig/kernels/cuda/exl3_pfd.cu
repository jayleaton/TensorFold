// Prompt-sized dense mul1 EXL3 GEMMs: pfdense over stored tiles at K2 8, 10, 12, 16, pfdense_args.h's tiles 0, 3, 6.

#include "exl3_block/pfdense.cuh"

#define TF_PFD(K2, ID, BM, WM, WN, KS, NST) \
    template __global__ void dsv41_pfd::pfd_kernel<K2, false, dsv41_pfd::Cfg<BM, WM, WN, KS, NST>>(const dsv41_pfd::Args);
#define TF_PFD_WIDTH(K2) TF_PFD(K2, 0, 128, 2, 4, 2, 3) TF_PFD(K2, 3, 64, 2, 2, 2, 4) TF_PFD(K2, 6, 32, 1, 4, 2, 4)
TF_PFD_WIDTH(8)
TF_PFD_WIDTH(10)
TF_PFD_WIDTH(12)
TF_PFD_WIDTH(16)
