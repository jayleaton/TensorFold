#!/bin/bash
# Build and run x3gm4_bench on a DGX Spark in the CUDA devel image (arguments pass through; exit 1 on any byte diff).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
IMAGE="${X4_IMAGE:-nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04}"
ROOT="$(readlink -f ..)"
run() { docker run --rm --gpus all -v "$ROOT:$ROOT" -w "$PWD" -e CUDA_MODULE_LOADING=EAGER "$IMAGE" "$@"; }
# the release fatbin's flags for x3gm (zig/build/cuda_kernels.zig torch_flags, -O3 -lineinfo, plain sm_121): v2's SASS
FLAGS=(-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__
       -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -lineinfo
       -gencode=arch=compute_121,code=sm_121)
echo "+ nvcc ${FLAGS[*]} -Xptxas -v -o x3gm4_bench x3gm4_bench.cu"
run nvcc "${FLAGS[@]}" -Xptxas -v -o x3gm4_bench x3gm4_bench.cu 2>&1 | grep -E "error|warning|Compiling|spill|Used" || true
test -x x3gm4_bench
echo "+ timeout ${X4_TIMEOUT:-900} ./x3gm4_bench $*"
# a pipeline deadlock (an mbarrier phase never completing) ends in exit 124 instead of a hung GPU
run timeout "${X4_TIMEOUT:-900}" ./x3gm4_bench "$@"
