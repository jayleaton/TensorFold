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

### 1.0 Sources (sync.py SOURCES)

Copies are read at git refs, never from a working tree.

| key | repo, ref | files |
| --- | --- | --- |
| base | `tensorfold-dsquant`, `dsv41-quant-e2` (0cf4ea1) | everything else |
| x3gm2 | `tensorfold-x3gm1`, `dsv41-x3gm1` (dsq-x3gm-1's x3gm v2, not merged) | `x3gm.cu` |
| r1 | `tensorfold-decode1`, `78b703d` (R1 + R1c, final; not merged) | `attn_cuda.cu`, `topk_cuda.cu`, `mhc_cuda.cu`, `exl3_experts.cu`, `router_gemv.cu`, `x3seg.cu`, `dense3.cu` (it includes x3seg.cu) |

When a branch lands in `dsv41-quant-e2`, point its files back at `base` (`FROM` in `sync.py`).

`dsv41-perf1` (bb0cff8) carries the same R1 sources plus x3gm v2. Its `roce.cu` wait fix belongs to the TP runtime,
not to these copies.

`sass_check.sh` exports each source at its ref (`git archive`) before compiling the Python side.

### 1.1 Exactness evidence

| check | how | where it stands |
| --- | --- | --- |
| SASS equality | `Z/sass_check.sh`: each Python source compiled the way `torch.utils.cpp_extension` compiles it (torch headers, the extension's flags), against our fatbin. Every kernel of ours must exist in the Python object with identical SASS. | **sm_121, all equal**, see the table below. Run on GPU pods for sm_120 by `zig/tests/cuda/deepseek_v41/job.sh`. |
| bits on a GPU (PRO 6000, sm_120: **13/13 cases pass**, pod 3) | `zig/tests/cuda/deepseek_v41/oracle.py` writes fixtures from the Python engine's own extensions on seeded inputs. `tf-dsv41-test all <fixtures>` runs ours and compares every output bit for bit. | Routed experts so far: decode chain at 1, 3, 8, 16 and 64 rows; E1 (2, 12), (2, 10) and DSpark (8, 8) ranges; every x3ld setting with and without PDL; x3pf; x3gm uniform and ragged at every configuration that fits; dequant. Entry: `bash zig/tests/cuda/deepseek_v41/job.sh <out>`. |
| real forward | the port lead's M1 capture: every extension call of a real block, replayed through `kernels_replay` | **pod 5: 1,353 / 1,353 launches bit-exact, alone and chained** over 9 real blocks of our pack (sets A, B, C incl. the DSpark block 40) and a recorded 256-row prefill segment a set, **captured with the engine's code defaults**: upstream linear, x3seg, the upstream grouped experts chain, x3gm rot / gateup / down, linear unpack and experts combine. The prod-only kernels (x3ld, dense3, attn_cuda / topk, mhc_cuda, mhc_pf, router_gemv, pfdense, L2) are first replayed in pod 6 (prod.env's knobs); x3ld is covered meanwhile by the oracle checks |
| recorded Python calls (`dsv41_capture.py`) | `job.sh` runs the twin's own GPU suites (attention + top-k, mHC decode / prefill, router GEMV incl. R1c prune, pfdense, dense3 / x3seg / linear) with a pytest plugin that records each allowed extension call: its arguments, and every storage they view before and after. `tf-dsv41-test` re-issues each call through `kernels_replay.call` and compares every storage bit for bit. A pre-R1 twin's attention and mHC calls also run as R1's split (P 1 / 4) and tail (+ defer, + coef_kernel) against the same outputs. | wired; first run in the next M1 pod |
| split KV copies | `tf-dsv41-test kvsplit` and the KV agent's 2-rank NCCL test | 3 / 3 bit-exact; 520,192 entries a rank, 0 mismatches |
| spills | `-Xptxas -v` per fatbin | equal to the Python build's. x3seg `<K2 6/10/12, 4, 3>` spills 32-68 B and dense3 at MINB 4 / 2 spills, in Python too. Prod's dense3 MINB 3 / 1 has 0 spills. |

**The M2c-2 Spark kit (2026-10-07, at 07935cc).**
- **Sources:** every copied source is byte-identical to the Spark prod engine's tree (decode1 `dsv41-perf1-night` 8474f31).
- **SASS:** `FATBINS=<zig-out/fatbin> SM=121 bash sass_check.sh` compares all 19 twin modules both ways, the fresh compile and the built fatbins: 441 kernels, all SASS-equal on sm_121, rc 0.
- **Python-only instances:** `linear`'s other codebooks (`3inst`, `mcg`; the pack is all `mul1`), and the CCCL `EmptyKernel`.
- **Spills:** our 5 modules without a Python twin have 0 spills on sm_121.
- **Kit fatbins:** the sm_121-only set is in `zig-dsv41-scratch/zig-out-sm121/fatbin`. Its SASS is identical module by module to the checked set, and `SHA256SUMS` covers the 24 `dsv41_*` files.
- **Shared memory:** GB10's opt-in limit is sm_120's (101,376 B), and the load refuses per device. Expect the same 7 x3gm cfg 3 / XM 2 refusals.

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
| routed gate / up | `gm_kernel<2,XM,K2,cfg 0 (XM 1) / 1 (XM 2)>` | `F/x3gm.cu` | .cu | v1 | ported, all 119 instances (+ 4K's cfgs 2-5) |
| routed down | `gm_kernel<1,1,K2,cfg 0>` | `F/x3gm.cu` | .cu | v1 | ported |
| routed gate / up, down (v2) | `gm2_kernel`: one launch a projection over every width, one plan (bm 64) a row block | `F/x3gm.cu` | .cu | prod (`GM_V2=1`) | ported: `exl3.Ops.gmPlan` + `gm2Gateup` / `gm2Down` (+ `gmRot`) |
| x3gm passes | `x3gm.plan`: torch `argsort(stable)`, `bincount`, `cumsum`, `searchsorted` | `F/x3gm.py` | torch | prod | **`Z/x3gm_plan.cu`** (ours, section 3.2) |
| routed combine | `combine_kernel` | `X/experts.cu` | .cu | prod | ported |
| (off) | `pf_kernel<2,NT,MTL,LO,HI>` | `F/x3pf.cu` | .cu | off (`EXPERT_PREFILL=0`) | ported (6 instances) |

**Prefill chunk sizes (2026-10-07).** Prod runs 2,048-row chunks (`PREFILL_CHUNK` / `ROWS` 2,048, adaptive 1,024 /
512, a short tail). `oracle.py` has x3gm cases at those sizes:
- `x3gm_pf2048_ragged`: 2,048 rows x 6 slots over prod's 384 experts, ragged widths 3-10, the shared expert folded (about 32 passes an expert at bm 64);
- `x3gm_pf2048_prod`: prod's geometry from pod 15's 2,048-row capture: D 5,120, I 1,152, gate / up widths 4 / 6 / 8 / 10, 12,288 slots over 16 experts (about 768 rows an expert);
- `x3gm_pf1024_k4`: 1,024 rows, uniform K2 4;
- `x3gm_pf512_shx`: 512 rows, the shared expert folded;
- `x3gm_pf_tail1337`: 1,337 rows, ragged.

Each runs the tuned v1 launches (v2's reference), the device plan, and v2 at both tickets, all bit for bit, in
`job.sh` (sm_120 on the pods; the same sources are SASS-equal on sm_121). Every other prefill launch is a copy
whose Zig wrapper checks the same limits as the Python extension's own checks: a chunk is launched as Python's
prefill launches it (the port lead's emitter). Pod 15's 2,048-row prod recapture replays the whole prefill
segment.

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
| the window's glue (block.zig's `glue.*` steps): `embed_rows`' masked send and `allsum` (+ repeat x4), the MoE partial `.to(bf16)`, `kit_weights`, `kit_logits`, `Win.make`'s positions, `gather_cols`, the ratio-2 compressor carry | every window | **done**: `Z/glue.cu` (`--fmad=false`; casts through `__float2bfloat16` / `__float2half`, the intrinsics torch's CUDA casts use), `ops.Ops.embedSend / embedSum / castBf16 / kitWeights / kitLogits / positions / gatherCols / carry`, checked against torch on edge values (oracle kind `glue`) |

**The prefill segment's glue (`Z/prefill_glue.cu`, 2026-10-07).** block_prefill.zig's glue steps, as torch runs them:

| glue step | Python (8474f31) | Zig |
| --- | --- | --- |
| `topk` | `index.top_positions` (int64 radix select, count <= 2,048) | `ops.Ops.topPositions` |
| `cand_keys` | `index.candidate_keys` | `candKeys` |
| `counts` | `backend.visible_counts` | `visibleCounts` |
| `gm_picks` | prefill_moe's kit picks | `gmPicks` |
| `gm_plan` | x3gm._run_ragged's width mask, then x3gm.plan | `widthMask` + `exl3.Ops.gmPlan` |
| `ix_w` | the indexer head weights fp64 -> bf16 -> fp32 | `f64Bf16F32` |
| `proj` | `blocks._projection` | `widenCat` |
| `stage_in` / `stage_out` | `blocks.stage_rows`, a tensor at a time | `ringCopy` |
| `proj_carry`, `zeros`, `gm_ticket`, `block_pos` | `torch.cat` / zeros / a scalar | `cuMemcpyDtoDAsync` / `cuMemsetD8Async` / `cuMemsetD32Async`: no kernel |
| `swiglu`, `moe_sum`, `kit_logits`, `positions`, `embed`, `exchange` | | pointwise.cu / glue.cu (above) |

- **GPU check:** oracle kind `pfglue` (edge keys: `KEY_NONE` pads, negative positions, rows shorter than `count`; fp64 double-rounding cases; NaN / inf), checked bit for bit by `tf-dsv41-test all`.
- **Build:** 0 spills, `--fmad=false`.

## 3.2b x3gm v2 and the R1 decode kernels

**x3gm v2** (`gm2_kernel`, dsq-x3gm-1, `ENGINE-X3GM2.md`) is the primary prefill x3gm. It runs one launch a
projection over every width and is bit-identical to `gm_kernel` (it skips predicated mma and computes addresses once
an item). It measured 23-38 % faster at 48 SMs.

- **Host contract** (`x3gm._run_v2`):
  - rot;
  - ONE plan over all pairs (`gmPlan`, bm 64);
  - `gm2Gateup` (cfg `gmTuned2`: 0 with one rotated input, else 1);
  - `gm2Down` (cfg 0);
  - `combine`.
- **k2e:** an int32 [E + 1] width table a projection. Uniform: one constant; ragged: per expert, each in K2S
  (`gm2WidthsOk`).
- **Ticket:** zeroed before each launch.
- **Args:** gains `k2e` after `tp1`, so `GmArgs` is now 160 B; gm_kernel launches leave it 0.
- **`gm2Shape`** ports `Smem2` / `Ring2`, the widest width's ring cut to keep 2 CTAs an SM. It is checked equal to the
  compiled constexprs for all five instances.
- **Spills:** the gate/up instances spill 24-108 B, in Python too, and outside the k loops.
- **GPU check:** every x3gm fixture runs gm2. With a Python twin that has `gateup2`, it compares against the twin's
  v2 outputs. Otherwise (the pods' e1e2 image) it compares against the fixture's own `gm_kernel` outputs, with the plan
  from `x3gm_plan.cu`. Both forms are the same bits by v2's contract.

**R1 decode** (`docs/DSV41-DECODE-R1.md` of the decode1 clone). Every knob off is today's launches, and all
variants give the same bits.

- **Attention:** `ops.attention(..., split)`. 0 is `attn_kernel`. 1 / 4 is `split_kernel<KW, ROWS, HAS_HI, P>`:
  - grid (5 P, H / 16, R); P 4 launches as a cluster of 4;
  - shared memory 98,816 / 100,864 B;
  - with split > 0 the ticket holds R H / 16 x `attn.MAXP` ints;
  - policy `attn.splitFor(R, 2)`: P 4 up to 2 rows, P 1 above.
- **mHC:**
  - `mhc.Args` gains `spin` (i64, -1 = today's finish) and `defer`.
  - `mhcBoundary` picks the TAIL instance when spin >= 0; the counter is then 3 self-resetting ints.
  - With defer = 1, `mhcCoef` (one CTA a row) runs the coefficients on a side stream forked after the boundary and
    joined before the next reader of opre / opost / ocomb.
  - Note: TAIL is a template parameter with a default, so every boundary symbol now carries `ELb0E` / `ELb1E`.
- **Experts:** `downCombine(..., out_bf16)` stores `bf16_rn(acc)`, the same bits as an fp32 out followed by
  `.to(bf16)` (`TF_DSV41_MOE_BF16`).
- **Prune:** `prune._prune`'s new KIT constexpr is a Triton variant. It is the port lead's AOT capture, not a copy.

**R1c** (final at `78b703d`). Every knob is off by default, and each fold gives the bits of its unfused launches.

- **q RoPE** (`TF_DSV41_ATTN_ROPE`):
  - `attn.Args.qrope` is set and split > 0; Args is now 240 B.
  - `split_kernel` rotates q's words at k >= 448 as it loads them, with `rope_fused._rope`'s ops.
  - The host passes un-rotated q and drops the rope launch.
  - `attention` refuses qrope with split 0, as the binding does.
- **Router prune** (`TF_DSV41_RG_PRUNE`):
  - `rg.Args.prune` is router_gemv's `Prune` (32 B at offset 132; Args is now 168 B).
  - `rg.pruneOf` takes the binding's 7 values (`prune.fold_args`).
  - The pass runs after the selection, with `_prune`'s ops, then the kit rounding. It has no spills, but it has not been
    checked on a GPU.
- **q_norm fold** (`TF_DSV41_RMS_FOLD`):
  - `dense.RotSeg` gains `nw`, `nw_dtype`, `inv_k` and `eps` (64 B).
  - `rotTable(..., eps)` fills them for each `RotIn` that has `nw`.
  - `seg_rot_in` computes `_rms_row`'s rn in Triton's float64 order and feeds bf16(fp32((x rn) w)) to the rotation.
  - The input must be bf16 rows.

SASS on sm_121 at `78b703d`, all equal to the Python build:

| file | kernels |
| --- | --- |
| attention | 24 |
| top-k | 1 |
| router_gemv | 40 (0 spills now) |
| x3seg | 19 |
| dense3 | 28 |
| mHC | 15 |
| experts | 32 |

SASS on sm_121, all equal to the Python build at the source's ref:

| file | kernels | of which new |
| --- | --- | --- |
| `x3gm.cu` | 135 | 5 gm2 |
| `attn_cuda.cu` | 24 | 16 split |
| `mhc_cuda.cu` | 15 | 7 tail + coef |
| `exl3_experts.cu` | 32 | down_combine fp32 / bf16 |

## 3.2c R4: CUDA graphs for decode windows

The decode path's graphs are captured from the same launches `run.zig` issues eagerly (Python graphs.py's
contract: replay == eager, bit for bit).
- `cuda/graph_cache.zig`: family-agnostic keyed capture / replay. It has an LRU cap (48), a fingerprint of the baked-in addresses, the memory floor's eviction, and graphs off after a capture fails.
  - Every decision a rank could take alone is AND-ed over the ranks (`Agree`), so TP=N and lanes stay in step.
  - The policy is host-tested with a fake engine.
- `deepseek_v41/graphs.zig`: Python's context buckets (2,048 to 32K, then 1.25x rounded up to 2,048), the key (rows, context bucket), `Statics`, `EngramStage`, and the agreement over the Collective (an i32 all-gather, any world).
- `forward.zig`: window() runs the graphed path when `TF_DSV41_GRAPHS=1`. It is off by default until a pod shows graphed == eager at TP=2.

**Inside a graph**
- Every launch of the window, the exchanges included. NCCL, the mailbox, RoCE and the hybrid all enqueue on the stream with device-side sequence words, so a replay pairs up like the eager run.
- The indexer scores the bucket's keys: the program is emitted at `bucketEnd + 1 - n`. Keys past a row's end score -inf and never reach its attention.
- `glue.cu`'s `positions_dev_kernel` reads the window's start from the statics. The glue GPU case checks it against the host form.

**Outside, on the host, each step**
- The inputs, by one copy each before the launch, with no synchronize:
  - the ids and the start (`Statics`);
  - each Engram layer's rows, hashed on the host (`EngramStage`). The graph keeps their all-gather and `gather_cols`.
- The choices: greedy, sampling and top-p read the logits on the host.
- `keep`'s accepted count and carry.
- The trace gate's downloads: a traced program is never graphed.

**Shapes**
- One slot, rows exact (Python's per-width graphs), up to the largest planned bucket.
- Row-mode padding to the row buckets (1-8, 12, 16, 24, 32, 48, 64) needs the stacked multi-slot state and comes with it.
- Graphs read state at fixed addresses: every buffer they touch is allocated once (bind; send and gathered at the planned rows). A change in any of them drops every graph.

## 3.2d x3gm v3 (`Z/x3gm3.cu`, ours): measured, bit-identical, small

**Where v2's time goes** (sm_120 SASS of its prod tiles, before writing v3):
- In the k loop of a stage at about 32 members an expert, about 70 % of instructions are the trellis decode:
  - per value one `IMAD`, one `IDP.4A`, half a `PRMT`, half an `HFMA2`, about 2 for window extraction;
  - 32 of 490 instructions are `HMMA`.
- The codebook arithmetic is already at its minimum; issue and tensor time are about equal at prod routing.
- v2's prod gate/up tile (GU1, two inputs) spills at 128 registers.
- Its down tile (DN0) runs 4-warp CTAs at 253 registers: 8 warps an SM.
- Every item starts with three dependent global reads (ticket, pass entry, member rows).

**What v3 changes** (`gm3_kernel`, over x3gm.cu's verbatim device code; sync.py's footer guard keeps x3gm.cu's
instances out of x3gm3.cu), all with the same mma chains and epilogue:
- **Claim-ahead:** the next item's ticket, pass entry and member rows are read during stages 0-4 of the current one, each load a stage before its use.
- **LEAN:** runs of windows that fit 32 bits (K2 3, 4, 7, 10) are read by one funnel shift, and byte-aligned windows by one permute.
- **1-column-tile warps:** 16 warps an SM for down, and 512-thread gate/up CTAs, without spills.
- The 11 variants are in `exl3.v3_variants`; `exl3.v3_default` is the measured choice.

**Pod (2026-10-07, 1x RTX PRO 6000, 188 SMs, $0.08)**
- **Bits:** every x3gm fixture with every v3 variant is equal to the Python engine's gm_kernel outputs (13 cases, 0 failed; prod's D 5,120 / I 1,152 geometry at 2,048 rows, the chunks, the 1,337-row tail).
- **Timing:** `x3gm3-bench` at prod shapes (384 experts, top-6, ragged 4 / 6 / 8 / 10, unique weights from DRAM), every output byte-equal to v2, ms (median of 9).

| rows, routing | v2 gate/up + down | v3 best gate/up | v3 best down (`v3[9]`) | v3 best total |
| --- | ---: | ---: | ---: | ---: |
| 512 uniform | 1.459 + 0.767 = 2.226 | 1.429 (`[3]`) | 0.746 (-2.8 %) | 2.175 (-2.3 %) |
| 1,024 uniform | 1.529 + 0.830 = 2.360 | 1.513 (`[3]`) | 0.800 (-3.7 %) | 2.313 (-2.0 %) |
| **2,048 uniform (prod chunk)** | 1.709 + 0.973 = **2.682** | 1.701 (`[1]`) | 0.922 (-5.2 %) | **2.623 (-2.2 %)** |
| 2,048 Zipf 0.8 | 2.001 + 1.120 = 3.121 | 1.973 (`[3]`) | 1.060 (-5.4 %) | 3.026 (-3.1 %) |

**What it means**
- **Down:** the 8-warp tile is the only real gain, -3 to -5 %. Its claim-ahead alone (`v3[6]`, v2's tile) gives -2.4 to -6.0 %, so most of the gain is per-item latency.
- **Gate/up:** none of the changes beats v2 by more than noise. The 512-thread tile (`v3[2]`) is 11-21 % slower: each warp reloads every member's B fragments for one column tile, so it issues twice the ldmatrix, in one CTA an SM. LEAN is within noise; ptxas already did most of it.
- **The perf study's "+17-22 % available" was v2's gain** (the predicated mma). After v2, the kernel is decode-issue and tensor balanced at prod routing. What is left in issue terms is a few percent, not a rewrite's worth.
- **On GB10** (DRAM-bound at prefill), the lever is bytes: the rotated X (2 × 126 MB a layer written and read) and the fp32 Y (252 MB written, read by combine). That needs a fused rotation or combine with its own exactness design, not issue work.

**Default:** `v3_default` = down `v3[9]` and gate/up `v3[1]` (`[4]` with one input): -2.2 % at prod's 2,048-row chunk, the same bits. Whether the Zig prefill takes v3 or stays on v2 is the port lead's call; both are exact. sm_121: the
included x3gm.cu is still 135 / 135 SASS-equal to Python (fresh compile and built fatbin); the v3 variants have the
same registers and spills as on sm_120.

**Flag:** `TF_DSV41_GM_V3=1` (`exl3.gmV3()`) puts prefill's x3gm on v3's defaults; off by default (v2).

## 3.2e The DRAM lever on GB10: estimated, not built (under 3 % of prefill)

**What was proposed:** fuse the input rotation into the gate/up operand load, and the combine into the down
epilogue, keeping v2's bits. The base: the Spark prod prefills 887 ms per 2,048-row chunk (32K replay), and a chunk
runs 20 x3gm layers. A layer per rank is 12,288 pairs, D 5,120, I 1,152, two rotated inputs (prod).

**Combine into the down epilogue: not bit-identical in any cheap form.**
- `combine_kernel` sums a row's 6 slots in slot order (`acc = fmaf(w, y, acc)` from +0).
- The 6 slots of a row come from 6 experts' passes, done by different CTAs at different times.
- An ordered in-epilogue sum needs slot-phased down launches, so every expert's weights are read 6 times.
- Atomics change the order; a bf16 Y changes the values. So Y's 252 MB write plus read stays.

**Rotation into the gate/up operand load: bytes against compute.**
- Bytes removed: the rotation's write (2 × 12,288 × 5,120 × 2 B = 252 MB) and gate/up's read of it (at least 252 MB; the 9 column-block items of a pass run together and share it in L2). Bytes added: the fused path's gathers of the bf16 rows, 21-126 MB. Net 377-482 MB a layer.
- At 273 GB/s that is 1.38-1.77 ms a layer, or 28-35 ms a chunk: **3.1-4.0 % if every byte were on the critical path**.
- The GB10 calibration (x3gm ≈ compute + 0.5-0.7 × DRAM time, section 2.1 of the perf doc) puts the rotation kernel fully on it (about 1.0 ms) and gate/up's read only partly (about 0.55 ms). The gathers cost 0.05-0.28 ms. Net about 1.3-1.5 ms a layer: **2.9-3.4 % before the added compute**.
- The added compute: a pass's rotated rows (655 KB an input) do not fit on chip, so each of a pass's 9 column-block items recomputes the 128-point Hadamard of every member row.
  - That is about 1.13 G values a layer at about 12 instructions a value, about 0.44 G warp-instructions, roughly 40 % on top of the trellis decode's 1.06 G.
  - After v2 the issue work is already as large as the tensor work (3.2d), so this lands on compute: +1.3-2.8 ms a layer on GB10.
- **Net: about -1 % to +1 % of prefill.** Under the 3 % bar: not built.

**What would move GB10's prefill instead.**
- Weight bytes: weights are about 2.5 GB a layer at about 3 bits, nearly the whole DRAM floor, and the ragged widths already set them.
- Larger chunks: R6, 4,096 rows, the weights streamed once per 4K, about -22 % x3gm a row, blocked on memory.
- FP8 on the tensor path: R5, quality-gated.

## 3.2f Decode glue fusion on GB10: estimated, not built (under 3 % of the window)

**Data.** The Zig M2b pods log only a window total ("decode 0.5 s"); they recorded no per-kernel trace. The
per-kernel numbers here are G14a's hardware-traced GB10 1-row window (DSV41-ZIG-PERF.md 1.2: 1,331 kernels, 21.4 ms in
prod's calibration) less R1's measured cuts. R1 is in prod since 2026-10-07: the attention split with the q RoPE folded,
mHC tail / defer, prune / kit / group in the router's tail, the bf16 MoE partial, q_norm in wq_b's input stage. It
cost -1.4 ms on the 1-row window, which leaves about 20.0 ms.

**What a small kernel costs.** Under graphs the launch is gone but each kernel keeps its ramp, its tail and the
in-graph gap, about 0.2 us. G14a's small classes average about 4 us a launch on a work floor near zero (0.05 ms
for about 330 launches). A fusion that keeps the op order saves about 3-4 us for each launch it removes. The gaps
alone are 0.27 ms a window over all 1,331 kernels, so a few hundred fewer launches save under 0.05 ms of gaps.

**Small kernels left after R1, and which fuse bit-identically.**

| class (G14a ms, launches) | bit-identical fusion | saving |
| --- | --- | ---: |
| copies (0.21) | written at their final offsets by the producer | ~0.15 |
| gate/up epilogue (40) | into x3ld's tail (same ops, same order) | ~0.14 |
| expert rot_in (0.15, 40) | into the router tail / x3ld's loader | ~0.14 |
| k RoPE + kv_store (~0.2) | elementwise pair, one launch | ~0.04-0.10 |
| rms2 0.27, `_plain` (fp64) 0.22, scores 0.13, Engram fuse 0.14 | Triton reductions: bit-identical only by copying Triton's reduction tree into CUDA, one project each | (ceiling ~0.5) |

**Estimate.**
- **Realistic** (the four exact-and-cheap rows): **-0.47 to -0.55 ms, 2.4-2.8 % of the 1-row window**. Under the 3 % bar: not built.
- **Ceiling:** every small kernel fused, the reductions included, is about -1.0 ms (~5 %). That needs four Triton reduction trees reproduced exactly.

**The bigger lever, already built.** The Zig decode path (`block.zig`) still follows the pre-R1 knob set
(`tools/zig/dsv41_m1/prod-knobs.env` has none of ATTN_SPLIT / ATTN_ROPE / MHC_TAIL / MHC_DEFER / MOE_BF16 / PRUNE_KIT /
RG_PRUNE / RMS_FOLD). R1's kernels are ported here and bit-checked through Zig (302 / 302 recorded twin calls, pods 11-14).
Wiring them in is worth R1's -1.4 ms (-6.5 %) a 1-row window, with bits unchanged (R1's knobs are bit-identical to
off): the port lead's `block.zig`, from a prod capture with prod-perf1.env's knobs.

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
| `tf_dsv41_x3seg_v3` | `rot_in` (R1c's `nw` / `eps` when captured), `linear` |
| `tf_dsv41_dense3_v1` | `linear`, `relayout`, `unpack` |
| `tensorfold_exl3_experts_v1` | `grouped` (cb 2), `group`, `rot_in`, `gateup_epilogue`, `down_epilogue`, `combine`, `down_combine`, `dequant` |
| `tf_dsv41_x3ld_v1`, `tf_dsv41_x3pf_v1` | `grouped` |
| `tf_dsv41_x3gm_v1` | `gateup`, `down`, `gateup2`, `down2` (v2), `rot`, `dequant` |
| `tf_dsv41_attn_cuda_v1` | `attn` (R1's `split` and R1c's `qrope` when captured, else 0), `topk` |
| `tf_dsv41_mhc_cuda_v1` | `run` (R1's `spin` / `defer` when captured, else today's finish), `coef` |
| `tf_dsv41_mhc_pf_v1` | `run` |
| `tf_dsv41_router_gemv_v2` | `route` (R1c's `prune` when captured, else off) |
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
