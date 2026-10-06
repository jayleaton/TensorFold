# DeepSeek-V4.1-Flash on the Zig engine: the GPU kernels

What the Python engine's DeepSeek-V4.1 forward runs on the GPU (`tensorfold-dsquant`, branch `dsv41-quant-e2`: prod
G19 + E1/E2), and how each kernel comes over to the Zig engine. The plan for the rest of the port is
`docs/DSV41-ZIG-PORT.md`.

Paths below: `F` = `src/tensorfold/families/deepseek_v41/cuda`, `X` = `src/tensorfold/cuda/exl3`, `G` =
`src/tensorfold/families/glm5_next/spark` (all in the Python repo); `Z` = `zig/kernels/cuda/deepseek_v41` here.

## 1. How a kernel comes over

- **`.cu` kernels are copied, not rewritten.** `Z/sync.py` writes `Z/*.cu|cuh|h` from the Python sources: the same
  lines and namespaces (so the same mangled names and SASS). It cuts the torch host code (line ranges, or the
  `*_NO_TORCH` macros the Python compile tests already define) and appends the explicit instantiations the Python
  dispatchers launch. `python3 Z/sync.py` (no `--write`) fails if any copy differs from its source; each header names
  the source file and its git blob hash.
- **Build.** `zig/build/cuda.zig` lists them as `dsv41_kernels` with each extension's `extra_cuda_cflags`. They become
  fatbins (`-Dnvcc=` or `-Dfatbins=`), embedded by the `dsv41_kernels` module
  (`zig/src/families/deepseek_v41/kernels.zig`). On a host without CUDA, `Z/nvcc-docker.sh` runs the NGC PyTorch
  image's nvcc: `zig build -Dnvcc=$PWD/zig/kernels/cuda/deepseek_v41/nvcc-docker.sh -Dsm=121,120 fatbins install`.
- **Launches.** `kernels_exl3.zig` (routed experts), `kernels_dense.zig` (dense linears), `kernels_ops.zig`
  (attention, top-k, mHC, router, pfdense, Engram gate, L2). Each launch mirrors its C++ entry: instance choice, grid,
  block, shared memory, cluster, and the `TORCH_CHECK`s as error returns. The policies the Python wrappers compute
  are ported as pure functions with host tests: `linear.plan`, `fused_proj.layout`, x3gm's tile, fit and tuning
  tables, and the pfdense heuristic. Device pointers are `u64`. Weights tables (trellis address `int64 [E]`, K2
  `int32 [E]`, stacked suh / svh) come from the family's loader (`stage.zig`).
- **Streams.** Launches take device pointers only and never copy from the host. The caller feeds them with pinned
  memory and async copies on the launch stream. A pageable `cuMemcpyHtoD` runs on the legacy default stream and can
  return before its DMA lands, and a non-blocking stream does not wait for it. The M1 replay hit this race. The GPU
  tests use blocking streams for that reason.
- **Replay.** `kernels_replay.call(k, stream, ext, func, args, kwargs, scratch)` re-issues a captured pybind call
  through these launches. The M1 block gate uses it to compare every `.cu` call of a real forward with Python's.

### 1.1 Exactness evidence

| check | how | where it stands |
| --- | --- | --- |
| SASS equality | `Z/sass_check.sh`: each Python source compiled the way `torch.utils.cpp_extension` compiles it (torch headers, the extension's flags), against our fatbin. Every kernel of ours must exist in the Python object with identical SASS. | **sm_121, all equal**, see the table below. Run on GPU pods for sm_120 by `zig/tests/cuda/deepseek_v41/job.sh`. |
| bits on a GPU (PRO 6000, sm_120: **13/13 cases pass**, pod 3) | `zig/tests/cuda/deepseek_v41/oracle.py` writes fixtures from the Python engine's own extensions on seeded inputs. `tf-dsv41-test all <fixtures>` runs ours and compares every output bit for bit. | Routed experts so far: decode chain at 1, 3, 8, 16 and 64 rows; E1 (2, 12), (2, 10) and DSpark (8, 8) ranges; every x3ld setting with and without PDL; x3pf; x3gm uniform and ragged at every configuration that fits; dequant. Entry: `bash zig/tests/cuda/deepseek_v41/job.sh <out>`. |
| real forward | the port lead's M1 capture: every extension call of a real block, replayed through `kernels_replay` | **pod 5: 1,353 / 1,353 launches bit-exact, alone and chained** over 9 real blocks of our pack (sets A, B, C incl. the DSpark block 40) and a recorded 256-row prefill segment a set, **captured with the engine's code defaults**: upstream linear, x3seg, the upstream grouped experts chain, x3gm rot / gateup / down, linear unpack and experts combine. The prod-only kernels (x3ld, dense3, attn_cuda / topk, mhc_cuda, mhc_pf, router_gemv, pfdense, L2) are first replayed in pod 6 (prod.env's knobs); x3ld is covered meanwhile by the oracle checks |
| split KV copies | `tf-dsv41-test kvsplit` and the KV agent's 2-rank NCCL test | 3 / 3 bit-exact; 520,192 entries a rank, 0 mismatches |
| spills | `-Xptxas -v` per fatbin | equal to the Python build's. x3seg `<K2 6/10/12, 4, 3>` spills 32-68 B and dense3 at MINB 4 / 2 spills, in Python too. Prod's dense3 MINB 3 / 1 has 0 spills. |

SASS on sm_121 (NGC 26.07, nvcc 13.3.73; the e1e2 runtime image has the same nvcc). Kernels compared, all equal:

| fatbin | Python extension | kernels |
| --- | --- | --- |
| `dsv41_x3ld` | `tf_dsv41_x3ld_v1` | 15 |
| `dsv41_x3pf` | `tf_dsv41_x3pf_v1` | 6 |
| `dsv41_x3gm` | `tf_dsv41_x3gm_v1` | 130 (119 `gm_kernel`, 4 rot, 7 dequant) |
| `dsv41_exl3_experts` | `tensorfold_exl3_experts_v1` (cb 2) | 31 |
| `dsv41_linear` | `tensorfold_exl3_linear_v4` (cb 2) | 45 |
| `dsv41_x3seg` | `tf_dsv41_x3seg_v3` | 19 |
| `dsv41_dense3` | `tf_dsv41_dense3_v1` | 28 |
| `dsv41_attn` | `tf_dsv41_attn_cuda_v1` (attn_cuda.cu) | 8 |
| `dsv41_topk` | `tf_dsv41_attn_cuda_v1` (topk_cuda.cu) | 1 |
| `dsv41_mhc` | `tf_dsv41_mhc_cuda_v1` | 7 |
| `dsv41_mhc_pf` | `tf_dsv41_mhc_pf_v1` | 8 |
| `dsv41_router_gemv` | `tf_dsv41_router_gemv_v2` | 40 |
| `dsv41_pfdense_k8` / `k10` / `k12` / `k16` | `tf_dsv41_pfdense_v1` | 18 / 18 / 18 / 9 |
| `dsv41_engram_gate` | `tf_dsv41_engram_gate_v1` | 1 |
| `dsv41_l2pace` | `tf_dsv41_l2pace_v1` | 1 |
| `dsv41_l2pf` | `tf_dsv41_l2pf_v1` (`G/l2pf.cu`) | 4 |

On sm_120 (the PRO 6000) the first pod run showed x3ld 15/15, x3pf 6/6, upstream experts 31/31 and x3gm 130/130
SASS-equal to the Python engine's `.so` built on that GPU.

**Shared memory opt-in.** `Kernels.load` checks each instance's static + dynamic shared memory against the device's
per-block opt-in (101,376 B on sm_120 / sm_121). Instances above it are refused one by one: x3gm's gate/up cfg 3
with two rotated inputs (107-123 KB) is never picked by `x3gm.fits` anyway. Python sets the attribute at a launch,
so it never meets them.

Upstream EXL3 (`linear.cu`, `experts.cu`) is built for the mul1 codebook only. Every DeepSeek-V4.1 matrix is mul1;
the replay returns "no binding" for the others.

## 2. Inventory: what the forward runs

The prod deployment (`dsv41-tensorfold-spark/config/prod.env`) turns on about 15 kernel knobs beyond the defaults.
The tables mark each kernel **prod** (on in prod.env), **default** (on by the knob defaults, replaced in prod), or
**off**.

### 2.1 Decode / verify (1-128 rows a window, row-mode CUDA graphs)

| step (per layer unless noted) | kernel | source | kind | prod | Zig |
| --- | --- | --- | --- | --- | --- |
| mHC boundary / site, <= 16 rows | `boundary_kernel<POST,COLL,MIX,ALL4,FN32>` | `F/mhc_cuda.cu` | .cu | prod (`MHC_CUDA=1`) | ported |
| (same, default) | `_site_dec` + `_finish_k`; `_site` for 17+ rows | `F/mhc_dec.py`, `F/mhc.py` | Triton | default; prod for > 16 rows and the drafter's in-place boundaries | AOT (section 3) |
| attention projections `wq_a`, `wkv` (+ compressor), `wo_a`, `wq_b` (+ `ix_wq_b`), `wo_b` | `lanes_linear_kernel<K2,2,WK,MINB>` + x3seg's `seg_rot_in_kernel` | `F/dense3.cu`, `F/x3seg.cu` | .cu | prod (`DENSE_V3=1`); boot `to_lanes_kernel` | ported |
| (same, default) | `seg_linear_kernel<K2,2,WK,MINB>` | `F/x3seg.cu` | .cu | default | ported |
| indexer `wk`, Engram `wkv`, head, DSpark `main_proj`, `wq_b` without indexer | upstream `rot_in_kernel` + `linear_kernel<K2,2,WK>` | `X/linear.cu` | .cu | prod | ported |
| q / kv RMSNorm | `_rms2`, `_rms` | `F/rmsnorm.py` | Triton (fp64) | prod | AOT |
| RoPE | `_rope` | `F/rope_fused.py` | Triton | prod | AOT |
| indexer `weights_proj` | `_plain` (fp64) | `F/plain_fp64.py` | Triton | prod | AOT |
| SWA ring store, compressor | `_kv_store`, `_pool_norm`, `_index_k` | `F/csa2/compress.py` | Triton | prod | AOT |
| indexer scores | `_scores` | `F/csa2/index.py` | Triton | prod | AOT |
| index top-k (<= 64 rows) | `topk_kernel` (clusters <= 8) | `F/csa2/topk_cuda.cu` | .cu | prod (`ATTN_CUDA=1`) | ported |
| (same, default) | `_dtopk` | `F/csa2/dtopk.py` | Triton | default | AOT |
| sparse + SWA attention (<= 64 rows) | `attn_kernel<KW,ROWS,HAS_HI>` | `F/csa2/attn_cuda.cu` | .cu | prod | ported |
| (same, default; prod above 64 rows) | `_chunks` + `_merge` | `F/csa2/attn.py` | Triton | default | AOT |
| router | `narrow_kernel<RT,2>` (decode), `gemv_kernel<16,8,2>` (prefill) | `F/router_gemv.cu` | .cu | prod (`ROUTER=gemv`) | ported |
| (same, default) | `_fused` | `F/router.py` | Triton | default | AOT |
| expert top-p prune | `_prune` | `F/prune.py` | Triton | prod (`EXPERT_TOPP=0.85`) | AOT |
| expert grouping | `group_kernel` (1 CTA x 1,024) | `X/experts.cu` | .cu | prod (target layers; DSpark folds it into the router) | ported |
| expert input rotation | `rot_in_kernel<bf16>` | `X/experts.cu` | .cu | prod | ported |
| gate / up, down | **`ld_kernel<2,8,1,LO,HI,0>`** (LO, HI) = (2, 10), (2, 12) (E1), (8, 8) (DSpark) | `F/x3ld.cu` | .cu | prod (`EXPERT_LOADS=1`, cfg "8,1", no PDL) | ported, all 15 instances |
| SwiGLU + down-input rotation | `gateup_epilogue_kernel` (ACT_F32, clamp 10) | `X/experts.cu` | .cu | prod | ported |
| down epilogue + weighted combine | `down_combine_kernel` | `X/experts.cu` | .cu | prod | ported |
| (fallback) | `grouped_kernel<2,NT,4,PF,LO,HI>` | `X/experts_grouped.cuh` | .cu | when x3ld does not fit | ported (the x3ld reference) |
| Engram fuse (layers 1, 14) | `_fuse_dec` | `F/engram_dec.py` | Triton | prod | AOT |
| Engram gate copy (layers 1, 14) | `gate_copy_kernel` | `F/engram_gate.cu` | .cu | prod (`ENGRAM_GATE=1`) | ported (reads host-mapped control words) |
| L2 prefetch (side stream) | `segments_kernel<bulk>`, `paced_kernel` | `G/l2pf.cu`, `F/l2pace.cu` | .cu | prod (`L2PF=1`) | ported |
| all-gather of partials | `gather_fast_kernel` | `G/roce.cu` | .cu | prod (`COMM_BACKEND=roce`) | **not ported**: the port lead's comm layer (NCCL first) |
| head (once a window) | `linear_kernel<12,2,8>` | `X/linear.cu` | .cu | prod | ported |

### 2.2 Prefill (CED replay, 2,048-row segments, fast kernels)

| step | kernel | source | kind | prod | Zig |
| --- | --- | --- | --- | --- | --- |
| every dense EXL3 linear | `pfd_kernel<K2,LANES,Cfg>` (cfgs 0-7, K2 8 / 10 / 12 (+lanes) / 16) | `F/pfdense.cuh`, `pfdense_k*.cu` | .cu | prod (`PF_DENSE=fused`) | ported (+ `pfdense.heuristic`; the prod JSON table is the port lead's config) |
| (same, default) | `unpack_kernel<K2,2>` + `_gemm` | `X/linear.cu`, `X/prefill.py` | .cu + Triton | default | unpack ported; `_gemm` AOT |
| mHC site, >= 256 rows | `site_kernel<POST,COLL,FN32,TAP>` | `F/mhc_pf.cu` | .cu | prod (`MHC_PF=1`) | ported |
| attention | `_fused` (BMQ 32) | `F/csa2/attn_pf.py` | Triton | prod (`PREFILL_ATTN=fused`) | AOT |
| index selection, >= 4,096 visible keys | `_stream` + torch top-k merge | `F/csa2/stream_topk.py` | Triton + torch | prod (`INDEX_STREAM_MIN=4096`) | AOT + top-k `.cu` (section 3.2) |
| index selection, other rows | `_scores`, `_keys`, `_block_keys` + torch topk / sort | `F/csa2/index.py` | Triton + torch | prod | AOT + `.cu` |
| router | `gemv_kernel<16,8,2>` | `F/router_gemv.cu` | .cu | prod | ported |
| shared expert | 3 dense GEMMs + torch `clamp` / `silu` / `mul` | — | — | prod | pfdense + **`Z/pointwise.cu`** (ours) |
| routed rotation | `rot_kernel<bf16, XM>` | `F/x3gm.cu` | .cu | prod (`FAST_EXPERTS=gm`) | ported |
| routed gate / up | `gm_kernel<2,XM,K2,cfg 0 (XM 1) / 1 (XM 2)>` | `F/x3gm.cu` | .cu | prod | ported, all 119 instances (+ 4K's cfgs 2-5) |
| routed down | `gm_kernel<1,1,K2,cfg 0>` | `F/x3gm.cu` | .cu | prod | ported |
| x3gm passes | `x3gm.plan`: torch `argsort(stable)`, `bincount`, `cumsum`, `searchsorted` | `F/x3gm.py` | torch | prod | **`Z/x3gm_plan.cu`** (ours, section 3.2) |
| routed combine | `combine_kernel` | `X/experts.cu` | .cu | prod | ported |
| (off) | `pf_kernel<2,NT,MTL,LO,HI>` | `F/x3pf.cu` | .cu | off (`EXPERT_PREFILL=0`) | ported (6 instances) |

### 2.3 Drafter (DSpark, 3 blocks, 5 rows a slot)

The backbone kernels of 2.1 at DSpark's widths: x3seg / dense3 at K2 8, `linear_kernel<8,2,8|4>`, x3ld `(8, 8)`, the
router with the folded grouping (`narrow_kernel`, `group=1`), `_kv_store`, `_chunks` / `_merge` (in-place boundaries
stay on Triton `_site`), plus `_keys` (`index.sort_keys`) + torch `topk` / `gather` for the candidates and `_chain`
(`F/dspark.py`) for the Markov / confidence chain.

### 2.4 Present but not on the path

`x3tc` (Triton, `FAST_EXPERTS=tc`, 3-5x slower), `x3dn` (`DENSE=0`), `xmap` (`XMAP` off), `moe_fused` (`MOE_FUSED=0`),
`l2pf.experts_kernel`, router mode "one" (`_route`), `_budget`, `engram_pcache._apply`, `draft_head._q4`,
`engram._fuse` / `_dequant` (fallbacks), `dense3.to_strips_kernel` (bench), the probe instances. Not copied, except
x3pf and upstream `grouped_kernel` (cheap; the exactness references).

## 3. What is not a `.cu` copy: decision per item

The Python engine's Triton kernels have no `.cu` twin to copy. For M1 the port lead's plan is (a) for all of them.

- **(a) AOT.** Capture the cubins the Python engine's Triton builds at its constexprs, the Nemotron route
  (`zig/src/cuda/aot.zig`). This is bit-identical by construction, at no cost to write.
- **(b) a new `.cu`.** Proven bit-equal to the Triton twin by `oracle.py` fixtures. Worth it only where a rewrite
  also buys speed (`docs/DSV41-ZIG-PERF.md`, the perf study), or where AOT is awkward (dynamic shapes, launch
  overhead under graphs).

### 3.1 Triton-only kernels

| Triton kernel(s) | used in prod | decision | note |
| --- | --- | --- | --- |
| `rmsnorm._rms`, `_rms2`, `_rms_row` (fp64) | every block | (a) | fp64 reductions: AOT keeps the bits. A `.cu` would have to copy Triton's reduction tree; only worth it fused into the mHC finish (perf study). |
| `rope_fused._rope` | every block | (a) | |
| `plain_fp64._plain` (indexer `weights_proj`) | index layers | (a) | |
| `csa2/compress._pool_norm`, `_kv_store`, `_index_k` | every block | (a) | `_kv_store` writes the session rows; the port lead's KV layout (FP4 rows, split KV) decides later. |
| `csa2/index._scores`, `_keys`, `_block_keys` | index layers | (a) | candidate for (b) fused with the top-k (perf study: attention + indexer) |
| `csa2/stream_topk._stream` | prefill >= 4K keys | (a) | |
| `csa2/attn_pf._fused` | prefill | (a) | |
| `csa2/attn._chunks`, `_merge`, `dtopk._dtopk` | > 64 rows / default | (a) | prod's <= 64-row path is the ported `attn_kernel` / `topk_kernel`. Note: `attn_kernel` reads Triton's dot-operand k layout (`KW` 2 / 4, Triton 3.8: 4), a launch argument here. |
| `mhc._site`, `_finish_k`, `mhc_dec._site_dec` | > 16 rows, drafter, final | (a) | prod's <= 16-row boundary and >= 256-row prefill site are the ported `.cu` |
| `router._fused`, `_logits`, `_select` | default only | (a), if ever | prod is `router_gemv` |
| `prune._prune` | prod (`EXPERT_TOPP`) | (a) | |
| `engram_dec._fuse_dec` (+ `engram._dequant`) | layers 1, 14 | (a) | |
| `dspark._chain`, `keyed_uniform` | drafter | (a) | |
| `exl3/prefill._gemm` | default prefill | (a), if ever | prod is pfdense |

### 3.2 Torch ops the forward needs natively

Each item gets a small `.cu` here (ours, checked bit-equal to torch on fixtures), unless noted.

| op | where | plan |
| --- | --- | --- |
| `x3gm.plan`: stable argsort by expert, bincount, cumsum, searchsorted, clamps | prefill, per row block and width | **done**: `Z/x3gm_plan.cu`, `exl3.Ops.gmPlan`. One CTA: histogram, scan, stable scatter in index order, pass tables (E < 512). Integers: checked equal to torch on every x3gm fixture's plans and on prod-sized picks. |
| `pick.top` (topk on `pick.keys` + `gather`) | sampling, gate | **done**: `Z/topk_keys.cu`, `ops.Ops.topKeys`. Keys are unique, so any exact selection is torch's: radix select, then a bitonic sort. C <= 65,536 columns (a rank's 64,640), k <= 1,024. Checked against `pick.top` with heavy ties. |
| drafter candidates (`index.sort_keys` + `torch.topk` + `gather`), `vsample.merge` `torch.sort` | drafter | the same kernel once the key builder (`_keys`, Triton) is AOT; `vsample.merge` sorts unique keys too |
| index fallback `top_positions`: topk on int64 keys, sort, bit ops | prefill selection | as above |
| shared expert at prefill: `clamp_`, `silu`, `mul_`, `out += shared`, `torch.add(out=bf16)` | prefill | **done**: `Z/pointwise.cu` (`--fmad=false`; torch's `x / (1 + exp(-x))`, NaN-propagating clamps), `ops.Ops.swiglu / addInto / addBf16`, checked against torch |
| embedding: `index_select`, `where`, `clamp`, rank all-sum | every window | port lead (`forward.zig`); a gather `.cu` |
| casts: `q.to(fp32)` before `_rope`, partials `.to(bf16)`, `kit_weights` fp32 -> fp16 -> fp32 | decode | folded into the consumers where the Python path allows, else torch_ops pointwise |
| staging ring `index_copy_`, replay stash `cat` / `clone`, shadow-slot `copy_` | prefill, tree verify | device memcpy (`cuMemcpyAsync`); no kernel |

## 3.3 Perf notes that touch these launches (docs/DSV41-ZIG-PERF.md)

- **x3gm at 4,096-row blocks.** `exl3.gmTuned` takes Python's tuning table (`tune_default` = `x3gm.TUNE`). Its 4K
  entry (128-member cfg 2 / 3) is provisional in Python too. The perf study measured cfg 0 (64-member) about 30 %
  faster at 4,096 rows (15.96 vs 22.2-23.7 ms on 48 SMs). Every configuration gives the same bits, so the table is a
  speed choice: pass a measured table to `gmTuned`. Prod's 2,048-row blocks use cfg 0 / 1 + 0 either way.
- **Lazy module loading.** `Kernels.load` resolves every instance (`cuModuleGetFunction`) and reads its attributes
  at load. That loads each kernel before any persistent or spinning kernel (x3gm's ticket loop, the Engram gate,
  L2 pacing) can occupy the GPU. Lazy loading deadlocks a first launch queued behind such a kernel; keep the
  resolve-everything load, or set `CUDA_MODULE_LOADING=EAGER`.
- **attn_cuda merge.** The perf study's `proto/attn_fastmerge.cuh` issues every (m, l) and partial load before the
  chains: bit-identical, 33.6 -> 29.8 us at 1 row (48-SM emulation). It would land in the Python source
  (`F/csa2/attn_cuda.cu`), and `sync.py` regenerates this copy from there.

## 4. Replay bindings (`kernels_replay.call`)

| extension | functions |
| --- | --- |
| `tensorfold_exl3_linear_v4` | `rot_in`, `linear`, `linear_skip`, `unpack` (cb 2) |
| `tf_dsv41_x3seg_v3` | `rot_in`, `linear` |
| `tf_dsv41_dense3_v1` | `linear`, `relayout`, `unpack` |
| `tensorfold_exl3_experts_v1` | `grouped` (cb 2), `group`, `rot_in`, `gateup_epilogue`, `down_epilogue`, `combine`, `down_combine`, `dequant` |
| `tf_dsv41_x3ld_v1`, `tf_dsv41_x3pf_v1` | `grouped` |
| `tf_dsv41_x3gm_v1` | `gateup`, `down`, `rot`, `dequant` |
| `tf_dsv41_attn_cuda_v1` | `attn`, `topk` |
| `tf_dsv41_mhc_cuda_v1`, `tf_dsv41_mhc_pf_v1` | `run` |
| `tf_dsv41_router_gemv_v2` | `route` |
| `tf_dsv41_pfdense_v1` | `gemm`, `dequant` |
| `tf_dsv41_l2pace_v1`, `tf_dsv41_l2pf_v1` | `paced`, `segments` |
| not replayed | `tf_dsv41_engram_gate_v1.gate_copy` (host-mapped control words), `info` / `describe` queries |

## 5. Running the checks

- Copies current: `python3 zig/kernels/cuda/deepseek_v41/sync.py` (rewrite: `--write`).
- Spills and SASS against the Python build, no GPU: `OUT=<dir under ~/Documents> SM=121 bash
  zig/kernels/cuda/deepseek_v41/sass_check.sh [name ...]`. It runs at most 4 nvcc at a time.
- Host tests (plans, tables, struct layouts, replay argument handling): `zig build test`.
- GPU bits: `TF_DSV41_PY_SRC=<tensorfold-dsquant> TF_DSV41_TEST=<tf-dsv41-test> TF_DSV41_FATBINS=<fatbin dir> bash
  zig/tests/cuda/deepseek_v41/job.sh <out>` on a GPU with torch (the NGC image or the e1e2 runtime). The oracle uses
  seeded random weights and inputs, no checkpoint.
