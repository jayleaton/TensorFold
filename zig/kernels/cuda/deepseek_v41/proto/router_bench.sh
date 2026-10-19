#!/bin/bash
# Build and run router_bench on a DGX Spark in the CUDA devel image (arguments pass through; exit 1 on any byte diff).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
IMAGE="${RG_IMAGE:-nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04}"
ROOT="$(readlink -f ..)"
run() { docker run --rm --gpus all -v "$ROOT:$ROOT" -w "$PWD" -e CUDA_MODULE_LOADING=EAGER "$IMAGE" "$@"; }
echo "+ nvcc -O3 -lineinfo -std=c++17 -gencode arch=compute_121a,code=sm_121a -o router_bench router_bench.cu"
run nvcc -O3 -lineinfo -std=c++17 -gencode arch=compute_121a,code=sm_121a -o router_bench router_bench.cu
echo "+ ./router_bench $*"
run ./router_bench "$@"
