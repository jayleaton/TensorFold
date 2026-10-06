# DeepSeek-V4.1-Flash on the Zig engine: port plan

This is the plan for moving our DeepSeek-V4.1-Flash serving stack from TensorFold's Python engine to the Zig engine (branch `zig-flashnext`, `ZIG-PREVIEW.md`). The work happens on branch `dsv41-zig`.

The reference is the Python engine at `tensorfold-dsquant` branch `dsv41-quant-e2`. That is prod G19 (`7bd2d67`) plus our E1 and E2 changes, in `src/tensorfold/families/deepseek_v41/cuda/`. Below, `py/` stands for that directory.

The long-context features live on two sibling branches:
- `dsv41-kvsplit`: split KV.
- `dsv41-sessres`: session residency and streamed NVMe.

The quant repo's `docs/ENGINE-E1E2.md`, `LONG-CONTEXT.md` and `RESULTS.md` give the numbers to match.

## 0. Ownership and interfaces (source of truth)

Several agents work on this branch at once. Each one commits only its own paths (`git add <paths>`, never `-A`) and asks the port lead about any interface change. This section is updated whenever an interface changes.

| Owner | Paths | Interface it provides |
|---|---|---|
| Port lead (`dsq-zig-lead`) | `zig/src/families/deepseek_v41/*.zig` except `kernels*.zig`, `m1_*`, this document, `tools/zig/dsv41_*`, the root `build.zig` hook | `config.Config`, `pack.Pack`, `plan.Plan` (a rank's TP weight plan), `stage` (host images), `named` (the loader's tensors by name), `load` (device upload), the M1 gate `tf-dsv41-m1` |
| Kernels (`dsq-zig-kern-1`) | `zig/kernels/cuda/deepseek_v41/**`, `zig/src/families/deepseek_v41/kernels*.zig`, the DeepSeek parts of `zig/build/cuda.zig`, `zig/tests/cuda/deepseek_v41/**`, `docs/DSV41-ZIG-KERNELS.md` | `dsv41_kernels.Kernels` (`load`, `experts(stream)`, `linears(stream)`, ...); `dsv41_kernels.replay.call` (one captured extension call re-issued through its Ops) |
| TP (`dsq-zig-tp-1`) | `zig/src/fabric/tp/**` | NCCL bootstrap across two hosts (unique id over TCP), the communicator: `allGather(send, recv, stream)` in rank order, the plan link and fate channel |
| KV and sessions (`dsq-zig-kv-1`) | `zig/src/families/deepseek_v41/kv/**`, `zig/src/sessions/**` | paged pool (256-token pages, `comp.*` 584 B rows, `index_k`), per-slot rings and carries, split KV, the session store and NVMe tier |
| Server (`dsq-zig-srv-1`) | `zig/src/server/**`, `zig/src/core/tokenizer/**`, `zig/src/core/template/**`, sampling, tool parsing | `encoding` (DeepSeek's message encoder), DSML tool calls, reasoning effort, the CUDA engine opened by the server |
| Performance (`dsq-zig-perf-1`) | `zig/kernels/cuda/deepseek_v41/proto/**` | CUDA rewrites where they beat the port by a lot; each one lands behind the same launch interface and is proven bit-equal (or carries its own session tag) |
| Drafting (`dsq-zig-draft-1`) | `zig/src/families/deepseek_v41/draft/**`, `zig/src/core/lanes/trim.zig` and its hooks, the `test-dsv41-draft` step in `build.zig` | DSpark drafter, depth and trees; the `core/lanes` backend for DeepSeek (`draft.Lanes`) over `Target` and `Pass` (below) |

Rules for every owner:
- Interfaces first, so implementations can change behind them.
- Tests run the real implementation. A test that only checks its own fixture does not count.
- No nested per-element loops on hot paths, host or device.
- Where a CUDA rewrite gives a large speed-up, take the time to write it rather than port the old kernel as is.

**Shared types.**
- **Weights by name.** The loader names every tensor the way the M1 capture names the Python engine's tree: `L<layer>.attn.wq_a.trellis`, `L<layer>.attn.wo_a.<j>.suh`, `L<layer>.moe.w1.data` (ragged) or `.trellis` (stack), `L<layer>.moe.shared.0.w2.svh`, `L<layer>.engram.qk`, `embed`, `norm`, `head.trellis`.
- **Expert tables.** Per routed projection, an int64 `[E]` table of each expert's trellis device address and an int32 `[E]` K2 table. `suh`/`svh` are stacked `[E, K]`/`[E, N]`.
- **Drafting** (`draft/iface.zig`; the GPU forward implements both vtables, `draft/twin.zig` is the CPU twin the tests run):
  - `Target.prefill(slot, ids, sampling, draw) -> u32`: the prompt into the slot's state, the keyed choice after its last row.
  - `Target.window(segments, choices)`: one forward over every slot's `Segment{slot, start, tokens, parents, draws, sampling}`. Row 0 is the pending token at `start` (the slot's cache length); row r sits at `start` + its depth and attends the committed cache plus its ancestors (`parents[r]`, row 0: -1; null: a chain). `choices[i][r]` is row r's keyed choice at `draws[r]`. The rows stay in the state until `keep`; a chain window that is never kept keeps every row.
  - `Target.keep(slot, path)`: keep the listed rows of the last window (a tree path, row 0 first), drop the rest: pool truncation, SWA rings and ratio-2 carries roll back to the path.
  - `Target.taps(slot) -> Taps{address, rows, stride}`: the last window's (or prompt's) taps, bf16 `[rows, 3 x hidden]` (the streams' mean entering layers 37/38/39), row i = window row i.
  - `Pass.ingest(slot, start, taps, rows)`: taps rows `rows[i]` into the slot's context rings at positions `start + i` (the lanes backend sends only the kept path, last `window` rows). In a shared round the lane core drafts before `keep`, so ingest reads the path, never the pool.
  - `Pass.propose(asks, out)`: one pass over every `Ask{slot, anchor, start, params}` (anchor = the pending token at `start`, `params` = the chain kernel's keyed sampling); `out[i]` gets `block` drafts and confidences, and with trees the gathered candidates and base logits `[block, K]`. `Pass.reset(slot)` restarts a slot's context; `Pass.markov()` returns host copies of the Markov / confidence heads for trees (null: chains only).
  - Row cap: 64 rows a forward, 16 a slot (graph buckets 1-8, 12, 16, 24, 32, 48, 64). The depth prices rounds from the measured table (`draft.Costs`, read from Python's `calib-*.json`), never a clock.

## 1. How the Zig engine is built

| Part | Where | What it does |
|---|---|---|
| Build | `build.zig`, `zig/build/cuda.zig` | Linux: `.cu` files go to fatbins through `-Dnvcc=` with each Python extension's own nvcc flags, or prebuilt ones come in through `-Dfatbins=`. They are embedded in the binary (`zig/src/cuda/kernels.zig`). Without nvcc, the runtime builds host-only with empty images. `zig build test` runs the host unit tests on any OS. |
| CUDA runtime | `zig/src/cuda/` | `driver.zig` (dlopen of libcuda), `context`, `memory` (`DeviceBuffer`, `HostBuffer`), `stream` and `event`, `module` and `launch`, `graph.zig` (capture, or build node by node, then update in place), `cublaslt`, `nccl.zig` (dlopen of libnccl: `CommInitRank`/`InitAll`, `AllReduce`/`AllGather`/`Broadcast`), `triton.zig` and `aot.zig` (Triton cubins captured from the Python engine, launched as Triton 3.7's launcher does), `segments.zig` (staggered prompt segments). No torch anywhere. |
| Backend-neutral core | `zig/src/core/` | `safetensors.zig` and `checkpoint.zig` (header index, mmap), `tokenizer/` (HF `tokenizer.json`, parity with `tokenizers`), `template/` (Jinja2 as `apply_chat_template`), `copy_index.zig` (copies from the context), `engine_api.zig` (the engine a server drives), `lane_host.zig`. |
| Lane core | `zig/src/core/lanes/` | The host side of `lane_engine.py`: streams, windows, drafts and acceptance. A backend implements `backend.Backend` (`prefill`, `first`, `queue`, `read`, `verify`, `keep`, `draft`, `release`). The exactness rule is that drafted output is byte-identical to serial decode, and `--no-drafts` is the reference. A forward holds at most 32 rows. |
| Families | `zig/src/families/` | `nemotron/` is the model to copy. It has `cuda_weights` (loader into the Python layouts), `cuda_kernels` (`.cu` plus the captured Triton set), `cuda_forward` (kernel order as in Python), `cuda_state`, `cuda_engine`, `cuda_lanes` (lane backend), `cuda_mtp` (drafter), `cuda_sampler` and `cuda_dump` (per-block dumps for the oracle). `flashnext/` and `kimi_k3/` are Metal-only. |
| Server | `zig/src/server/` | OpenAI and Anthropic routes, SSE, tool parsing, grammar fields, metrics. Engines are opened through `zig/src/server/engines.zig`, which reads the backend module built in. Only `native/metal.zig` and `native/none.zig` exist today, so CUDA is not served yet. |
| CLI | `zig/src/cli/cuda_main.zig` | `tensorfold run MODEL --tokens ...`, `check-weights`, `teacher`, `prefill`, `rounds`, `lanes`. These are Nemotron's oracle checks. |
| Oracle runs | `zig/tests/cuda/box/run.sh`, `zig/tests/cuda/nemotron/capture.py`, `aot_pack.py` | Run the Python engine in `nvcr.io/nvidia/pytorch:26.07-py3` to capture Triton launches, weight digests, block dumps and tokens. Then build the Zig engine with the image's nvcc and compare. |

Rules from `ZIG-PREVIEW.md`:
- Output stays identical to `--no-drafts`.
- bf16 activations, fp32 accumulation, no precision trades.
- Files of 600 lines or fewer, with one-line comments.
- `zig build test` passes.

## 2. Where the port stands (this branch)

- **Toolchain.** Zig 0.17.0 runs from user space (`~/.local/opt/zig`). `zig build test` passes on the x86 dev box: 17/17 steps at the base commit (about 5 minutes cold), 22/22 with the family.
- **Cross-build for GB10.** `zig build -Dtarget=aarch64-linux-gnu` cross-compiles `tensorfold`, `tf-cuda-test` and `tf-dsv41-test` for GB10 in about 80 s without any CUDA toolkit.
  - Fatbins need nvcc. The kernel agent builds them in the local `tensorfold-dsv41` image (nvcc, `sm_120`/`sm_121`).
  - GPU tests need a GPU.
- **Family host side (`zig/src/families/deepseek_v41/`, step `zig build test-dsv41`):**

| File | Python twin | What it does |
|---|---|---|
| `config.zig` | `py/config.py` | Reads `config.json`'s `text_config`. Computes the CSA2 topology (`mode`: swa / full / reindex / reuse, `kvSource`, `indexSource`, candidates), Engram and DSpark. Refuses routing the kernels do not implement (`scoring_func` other than `sqrtsoftplus`, `topk_method` other than `noaux_tc`). |
| `exl3.zig` | `py/weights.py` `X3`, `X3Stack`, `X3Ragged`, `ragged()` | Group geometry (K2 = words / 8) and codebook markers. A rank's TP part as byte spans: rows are contiguous, columns are one run per K-tile row. The routed-expert `Layout` is a stack when there is one width and ragged (back-to-back offsets) when each expert has its own. |
| `pack.zig` | `py/weights.py` `Shards` | Indexes the pack from its headers with pread and no mmap (on GB10 the page cache is GPU memory). Follows the index's weight map, since real packs hold the same name in two shards. Reads spans and marker values. |
| `names.zig` | `py/weights.py` `Loader`, `py/drafter.py` | Every block's EXL3 groups, natives, splits and expected shapes, DSpark's heads, and the names the text engine knowingly leaves unread: vision, `image_*`, `gate.bias_vl`, and the bf16 `.weight` copies of EXL3 groups. |
| `stage.zig` | `py/weights.py` `Loader.x3`, `ragged()`, `torch.stack` | Reads a group's TP part into host images, and packs a routed projection's experts back to back with their int32 K2 table and byte offsets, the format the kernels take. Suh and svh are stacked `[E, K]`/`[E, N]`. The device loader only uploads these and turns offsets into addresses. |
| `plan.zig` | `py/weights.py` `Loader.load` | A rank's whole weight plan: parts, layouts, byte totals, gate/up width mismatches (x3gm cannot fuse those), and coverage, so a pack with an unplanned name is refused. |

- **Tests:**
  - the release `config.json`
  - a synthetic pack, named the way the real ones are, on both ranks
  - all 43 blocks of the q28 pack: routed plus shared trellis bytes per rank equal the Python loader's listing (`fixtures/q28-widths.txt`, generated by `tools/zig/dsv41_widths_fixture.py` from `q28-expert.json`)
  - real header dumps through `TF_DSV41_HEADERS`. Layers 2 and 20 of the v2 pack plan on both ranks; layer 2's down projection is ragged with K2 4/6/8 = 204/171/9.

### The M1 gate, as built

- **Capture** (`tools/zig/dsv41_m1_capture.py`, run in the e1e2 runtime on one sm_120 GPU).
  - The Python engine runs blocks 1–3 and 20–24 at rank 0's TP=2 shapes. The peer's half of every exchange is zero.
  - A 1,536-token prompt fills the KV state past the index top-512. Then decode windows of 1, 3 and 16 rows are recorded.
  - Every Triton launch and every extension call is logged with each buffer's bytes before and after. Output:
    - `ops.jsonl`
    - `blobs/` (content-addressed)
    - the weights by the loader's names, with sha256 (`weights.json`)
    - the launched cubins (`aot/aot.json`)
- **Replay** (`tf-dsv41-m1 replay PACK CAPTURE OUT`, files `m1_main.zig`, `m1_replay.zig`, `m1_triton.zig`, `named.zig`, `load.zig`).
  1. Our loader builds the captured blocks from the pack and checks every weight's digest against the capture.
  2. It re-issues every launch on its own buffers:
     - Triton cubins by hash, with global scratch where a kernel needs it.
     - Extension calls through `dsv41_kernels.replay.call`.
     - Weight arguments are bound to our loader's tensors by name, and pointer tables are relocated to our addresses.
  3. It compares every buffer:
     - **Alone:** each op starts from the captured inputs.
     - **Chained:** our own outputs feed the next op wherever nothing ran between them.
  - Exit 0 means every weight is equal and every op is equal alone. Ops without a binding are listed as skipped.
- **Pod job.** The quant repo's `runpod/session.sh zigm1 --gpu pro6000` (staged by `runpod/stage-zigm1.sh`) runs, in one pod:
  - the kernel agent's `zig/tests/cuda/deepseek_v41/job.sh`
  - the capture
  - the replay

  Pods are logged in the quant repo's `results/zig-pod-ledger.md`.

## 3. Python to Zig, component by component

Two things decide how each part is ported:

1. **Triton is the default in Python.** The Python engine's default hot path is mostly Triton. Its `.cu` files are opt-in twins, documented as bit-identical to the Triton kernels.
2. **Two ways to bring a kernel over:**
   - **AOT:** Triton-only kernels come over as captured cubins (`zig/src/cuda/aot.zig`, the Nemotron route), bit-identical by construction.
   - **CU:** where a `.cu` twin exists, it is copied verbatim and checked bit-equal to the Triton kernel by the kernel agent's fixture tool.

**Weights and model structure**

| Python (`py/`) | Zig (`zig/src/families/deepseek_v41/`) | State | Kernels |
|---|---|---|---|
| `config.py` | `config.zig` | done | — |
| `weights.py` (`Shards`, `Loader`, `X3*`) | `pack.zig`, `exl3.zig`, `names.zig`, `plan.zig` (host); `load.zig` (device upload: one trellis buffer per projection, suh/svh `[E,K]`/`[E,N]`, int64 address and int32 K2 tables per projection, the format the kernel agent's launches take) | host side and staging done; `load.zig` (uploads, address tables) is M1 | — |
| `fastboot.py` (prepared folder) | `prepared.zig`: one `data.bin` per rank, written from the planned device buffers, plus a manifest of offsets. No pickled tree: the plan is rebuilt from headers, which takes milliseconds. | M2 (boot time) | — |

**Forward pass**

| Python (`py/`) | Zig (`zig/src/families/deepseek_v41/`) | State | Kernels |
|---|---|---|---|
| `forward.py` (`_run`, `_layers`, `_finish`, `embed_rows`) | `forward.zig` (window), `block.zig` (one block's kernel order) | M1–M2 | the embedding row gather + `allsum` is a small `.cu` (new; torch `index_select`/`where`) |
| `mhc.py`, `mhc_dec.py` (Triton `_site`, `_finish_k`, `_site_dec`); `mhc_cuda.cu`, `mhc_pf.cu` | `mhc.zig` (boundary, site, `post_only`, final; coefficient scratch) | M1 | CU: `mhc_cuda.cu` (decode boundary), `mhc_pf.cu` (prefill site). Note: `exp` is the hardware approximation, so the bits match Triton only if the twin was proven equal. The kernel agent checks this first. |
| `blocks.attention`, `_out`, `_wo_b`, `fused_proj.py`, `rope_fused.py`, `rmsnorm.py` | `attention.zig` (projections, q/kv norms, RoPE, `wo_a`/`wo_b`) | M1 | CU: `x3seg.cu` (fused projection segments), upstream `exl3/linear.cu` + `decode.cuh`, `x3dn.cu`, `dense3.cu` (8-bit `ix_wk` fallback). AOT: `_rms`, `_rms2`, `_rms_row` (fp64 norms), `_rope`, `_plain` (`ix_wp`). The RoPE/YaRN tables are built on the host to Python's exact fp32 values; check them against a dump (risk R6). |
| `csa2/compress.py` (`_pool_norm`, `_kv_store`, `_index_k`), `csa2/fmt.py`, `stores.py` | `compress.zig` (ratio-2 carry, pooling, 584-byte records, index keys) | M1 | AOT: `_pool_norm`, `_kv_store`, `_index_k` |
| `csa2/index.py`, `dtopk.py`, `stream_topk.py`, `backend.select*`, `reindex`, `candidates` | `indexer.zig` (scores, top-512, candidate blocks at layer 20, reindex over 16,384 candidates) | M1 | AOT: `_scores`, `_keys`, `_block_keys`, `_dtopk`, `_stream`/`_offer`/`_compact`/`_kth`. CU: `topk_cuda.cu` (cluster radix select). The torch `topk`/`sort` fallback in `top_positions` is not used: the keys are unique int64, so any exact top-k gives the same set. |
| `csa2/attn.py` (`_chunks` + `_merge`), `attn_pf.py`; `attn_cuda.cu` | `attention.zig` (core: 512 selected rows plus the SWA window, sink, inverse RoPE) | M1 | CU: `attn_cuda.cu` (sm_90+, same chunk/merge order) or AOT `_chunks`/`_merge`. `attn_pf._fused` is fast-prefill only and has a different bit tag. |
| `engram*.py` (`engram_dec._fuse_dec`, `engram._dequant`), `engram_host.py` (hashing, shard rows), `engram_native.py`, `engram_gate.cu` | `engram.zig` (device fuse), `engram_host.zig` (n-gram hashing into 24 heads, 264-byte NVMe rows, shard reader; host code) | M1 (layers 1 and 14) | AOT: `_fuse_dec`, `_dequant`. `engram_gate.cu` is I/O orchestration: port it in M2 for overlap. |
| `router.py` (`_fused`, `_logits`, `_select`), `router_gemv.cu`, `moe_fused.cu`, `pick.kit_weights` | `moe.zig` (router, routed experts, shared expert, combine) | M1 | CU: `router_gemv.cu` or `moe_fused.cu` (router + group + rot_in + fp16 weights), once proven equal to `_fused`. Otherwise AOT `_fused`. Fast prefill uses `_logits`/`_select` (AOT). |
| `experts.py`, `exl3/experts.cu` + `experts_grouped.cuh`, `x3ld.cu`, `x3pf.cu`, `x3gm.cu`, `moe.py` | `moe.zig` | M1 (decode: grouped or `x3ld`); M2 (prefill: `x3gm`) | CU, all of it (mul1 codebook only, as in Python). `x3gm.plan` (argsort/bincount/searchsorted in torch) becomes a small `.cu` or host step; the kernel agent lists it. |
| `prefill_mm.py`, `exl3/prefill._gemm`, `pfdense.cuh`, `prefill_moe.py` | `prefill.zig` (2048-row segments, fast kernels) | M2 | AOT `_gemm` (unpack to fp16 + GEMM with Hadamard epilogue); CU `pfdense` (opt-in in Python, `PF_DENSE=fused` in prod) |
| `forward._finish`, `pick.py` (`kit_logits`, `top`), `slots.cand_gather`, `exact_sampling.choose` | `sampler.zig` (head, bf16 logits, per-rank top-k on int32 keys, candidate gather, keyed choice) | M2 | torch `topk` on keys: a `.cu` replacement in `zig/kernels/cuda/torch_ops` style (Nemotron already has `torch_topk.cu` and `torch_argmax.cu`; check that tie-breaking matches). |

**Drafting, rounds and memory**

| Python (`py/`) | Zig (`zig/src/families/deepseek_v41/`) | State | Kernels |
|---|---|---|---|
| `drafter.py`, `dspark.py` (`_chain`, keyed noise), `draft_head.py`, `spec.py`, `tree.py` | `drafter.zig` (DSpark: taps at layers 37–39, `main_proj`, context rings, 5-row blocks, Markov and confidence heads), wired as the lane core's draft hook | M3 | backbone kernels; AOT `_chain`, `keyed_uniform` |
| `lookup.py` (suffix index, l / d / s plan) | the lane core's own copy index (`core/copy_index.zig`) plus a DSpark proposer. Python's per-round planner policy becomes the proposer's choice. | M3 | — |
| `depth.py`, `calib.py` (E(k) − λ·verify[k]) | `core/draft_depth.zig` with DeepSeek's measured costs table (Nemotron uses `--costs`) | M3 | — |
| `pool.py`, `slots.py`, `state.py`, `rowmode.py`, `rowtab.py`, GLM `kvpool.py` | `state.zig` (paged pool: 256-token pages, `comp.*` 584 B rows, `index_k` fp8 132 B; per-slot SWA rings, ratio-2 carries, Engram lookback, DSpark taps), `pool.zig` (deterministic `take`, holders, `need_pages`) | M2 | `.cu` page copy (or `cuMemcpy`) |
| `rowgraphs.py`, `graphs.py` | `graphs.zig` (`cuda/graph.zig` capture keyed by row bucket × context bucket, per-row table `[6, rows]`, padding rows that write nothing, warm-up at buckets 1–8) | M2 (decode speed) | — |
| `batch.py`, `plan.py`, `rounds.py`, `planlink.py`, `failfast.py` | Mostly replaced by the lane core and `lane_host.zig` on rank 0. `tp.zig` holds the round plan (encode/decode, the same ordering of commits, finishes, admits, pieces and windows), the rank-1 follower loop, the TCP fate channel and the plan link. | M2 (follower), M4 (admission and fairness) | — |
| `memory.py` (priced floor, `Adapt`), `budget.py` (lossy, skip), `hostmem.py` | `memory.zig` (load check, floor against both ranks, prefill price, adaptive rows 2048/1024/512) | M4 | — |

**Server and long context**

| Python (`py/`) | Zig (`zig/src/families/deepseek_v41/`) | State | Kernels |
|---|---|---|---|
| `app.py`, GLM `server.py`, `encoding.py`, `dsml.py`, `structured.py` | Server: `zig/src/native/cuda.zig` (new: the CUDA engines a native server opens), with `engines.zig` picking it on Linux. `encoding.zig`: DeepSeek's `encode_messages`, native, since it is not Jinja. `dsml.zig`: the lenient DSML tool-call parser and streamer, plus `reasoning_effort` mapping and reasoning memory. | M4 | — |
| `sessions.py`, `sessdisk.py`, `residency.py` (sessres) | `sessions.zig` (prefix store on the 16-token grid, page sharing, LRU, spill), `sessdisk.zig` (NVMe format with 4 KiB-aligned segments and SHA-256 per segment, O_DIRECT, streamed through staging buffers, async writer and reader) | M5 | — |
| `kvsplit.py` (kvsplit) | `kvsplit.zig` (`comp.*` page k on rank k % 2, residue allocator, ctable plus discard page, the dense gather for decode and union exchange for prefill) | M5 | `.cu` gather/remap (from `blocks._attend_blocks`); NCCL all-gather |

**Python modules outside `py/` that the engine needs.** These live in the GLM family or the shared engine and must be ported as well: `glm5_next/spark/kvpool.py`, `server.py`, `roce.py` (one-shot RoCE all-gather and host mailbox), `tensorfold.cuda.comm` (`NCCL`, `fast_gather`), `tensorfold.engine.exact_sampling`.

## 4. TP=2 on two GB10s

**Processes.** One process per Spark, as today: `tensorfold serve /model --tp 2 --rank R --master HEAD_IP`, with rank 1 started first. The two nodes are linked by ConnectX-7 (192.168.100.10/11, `NCCL_SOCKET_IFNAME=enp1s0f1np1`).

**Bootstrap.** New in `tp.zig`:
1. Rank 0 calls `ncclGetUniqueId` and sends the 128-byte id over a TCP side channel on master port + 7 (Python's `planlink` TCP port).
2. Both ranks call `ncclCommInitRank(comm, 2, id, rank)` with `zig/src/cuda/nccl.zig`, which already exposes everything needed.
3. Both ranks all-gather their settings (context, chunk, token-map hash, pack fingerprint, knobs that change bytes) and refuse to start if they differ.
4. The same TCP socket stays open as the fate channel. Rank 1 reports its free memory every 0.5 s, and a failure on either rank ends both within about 1 s.

**Rank roles.**
- **Rank 0** owns the server, the lane core, admission, planning and sampling.
- **Rank 1** runs `follow()`: it receives each round's plan (an int stream over the plan link) and runs the same forward. It never decides anything.
- **Plan link.** It starts as a GPU all-gather over NCCL. The RoCE host mailbox (`roce.py`) comes later as a speed item.

**What each rank holds.** This is `py/weights.py`'s table, implemented in `names.zig` and `plan.zig`:

| Rank r of 2 holds | Tensors |
|---|---|
| Half the query heads | `wq_b` columns, the sinks, and `wo_a` groups 4r..4r+3 |
| Input-row half (partial sum) | `wo_b` |
| Intermediate half | routed and shared `w1`/`w3` columns; `w2` rows (partial sum) |
| Vocabulary half | the embedding rows and the head columns |
| Whole | `wq_a`, `wkv`, compressor, indexer (all 32 index heads), norms, `hc_*`, router, Engram `wkv`, DSpark heads |

- The single KV head means both ranks hold the whole KV cache. With split KV in M5, the `comp` pages are sharded instead.
- Every split falls on a 128-value boundary, so EXL3's Hadamard blocks stay whole.

**Exchanges.** There is no all-reduce. Every exchange is an all-gather of bf16 partials, and the receiver sums them in rank order inside the next mHC boundary kernel. That keeps both ranks' streams bit-identical.

Per 1-row window there are 81–83 exchanges:
- the embedding (once)
- Engram column concat (layers 1 and 14)
- two per block (after `wo_b`, after the MoE)
- the head's candidate gather (`[R, 2k]` fp32 per rank; greedy as max-then-lower-id)
- the DSpark passes

The all-gathers run on the compute stream so they can be captured in CUDA graphs, the way Python captures `fast_gather`. Decode alternates two persistent receive buffers.

**Exactness across ranks.** Both ranks run identical plans, page tables and kernels. M2 compares a hash of the stream state per layer between ranks on the first rounds.

## 5. Milestones

Effort is in engineering days for this agent team, assuming GPU windows are available when needed. GPU work is listed where it happens.

**M0. Toolchain and host side** (done, this branch).
- Zig in user space, `zig build test` green, aarch64 cross-build, the family's host side with 13 tests.

**M1. One block, bit-exact against its Python twin** (about 10–15 days).
- **Build:**
  - `load.zig`: device upload of a block from the plan, plus the expert tables.
  - The Triton AOT capture for DeepSeek's Triton-only kernels: a `tools/zig/dsv41_capture.py` in the PyTorch image against `dsv41-quant-e2`, packed with `zig/tests/cuda/nemotron/aot_pack.py`'s format.
  - `mhc.zig`, `attention.zig`, `compress.zig`, `indexer.zig`, `engram.zig`, `moe.zig`, `block.zig`.
  - A `tf-dsv41-test block LAYER FIXTURE_DIR` runner.
- **Blocks covered:**
  - layer 1 (SWA + Engram)
  - layer 2 (full: KV source, compressor gate, indexer)
  - layer 3 (reuse)
  - layer 20 (ratio 1, candidate source)
  - layer 24 (reindex)
  - layer 40 (DSpark block shape)
- **Inputs:** fixed rows at 1, 3 and 16 rows, with a prepared KV state (prompt rows written by the Python twin and loaded as the fixture's pool pages). Rank 0's slice is used, with the peer's partial taken from the fixture.
- **GPU:** one GPU of the same architecture as the Python run.
  - The bits only compare within one architecture, because the SASS differs between sm_120 and sm_121.
  - Either a GB10 window (the Sparks are owned by the Python-engine agent now) or a RunPod RTX PRO 6000 (sm_120) on the Python `tensorfold-dsv41:e1e2-…-sm120` image. RunPod needs approval: about 2 hours a run, roughly $4–5.

**M2. Whole model, TP=2, greedy token-identical on GB10** (about 15–20 days).
- **Build:**
  - `forward.zig` (window + prefill), `state.zig`/`pool.zig`, `sampler.zig`, `tp.zig` (bootstrap, follower, exchanges, fate channel).
  - `prefill.zig` with the exact-tag prefill first, then the fast kernels (`x3gm`, `_gemm`) under their own session tag as in Python.
  - `graphs.zig`, `prepared.zig`.
  - A `tensorfold run` path for DeepSeek: `zig/src/cli/dsv41_main.zig`, or a family switch in `cuda_main.zig`.
- **Pack:** Mia's 2.9 bpw pack first (one width per layer, so the stack layout only), then ours (q28/v2: ragged).
- **GPU:** both Sparks for several sessions of about 2 hours. This needs the Python-engine agent to release them, or a planned window like the E1/E2 one.

**M3. Lanes and drafter** (about 10–15 days).
- `lanes.zig` (the lane core's backend):
  - `verify` is a window of at most 32 rows, against Python's 16 per slot and 64 in total.
  - `keep` commits the accepted rows: pool truncation, SWA rings and ratio-2 carries roll back to the kept rows.
  - `draft` is the DSpark pass.
- `drafter.zig`; copy drafts from the lane core's copy index; DeepSeek's depth costs.
- Keyed sampling at T > 0 with the same Gumbel keys (seed, absolute position, id) as Python.
- **GPU:** both Sparks.

**M4. Server** (about 8–12 days).
- `zig/src/native/cuda.zig`, wiring into `engines.zig` for Linux builds.
- `encoding.zig`, `dsml.zig`, reasoning effort, `memory.zig` (the floor and adaptive prefill rows), admission and fairness for 4 slots, and `/metrics`.
- Grammar: the Zig server's structured-output fields and the engine's grammar enforcement. Python uses xgrammar; this needs a decision (R9).
- **GPU:** both Sparks, then a soak.

**M5. Long context** (about 15–20 days).
- `sessions.zig` and `sessdisk.zig` (streamed NVMe, residency budget, priced restores, as on `dsv41-sessres`).
- `kvsplit.zig` (as on `dsv41-kvsplit`), at the 1M context.
- FP4 KV rows (`dsv41-kvq`) after the gates pass.
- **GPU:** both Sparks; a context ladder up to 1M tokens.

**Total:** about 60–80 engineering days, or roughly 3 months calendar with GPU windows. The CUDA paths M1 and M2 carry most of the risk; M4 and M5 are mostly host code.

## 6. Exactness gates

| Milestone | Gate (equality unless stated) | Reference |
|---|---|---|
| M1 | Every op's output bits equal the Python twin's on the same GPU architecture, for each covered block, at 1, 3 and 16 rows. Then the block's output streams are equal. Also: a row's bits do not depend on how many rows share the window (one row alone against the same row in 2-, 3- and 16-row windows). Weight digests per tensor equal Python's `RankW` (a `check-weights`-style JSON). | `dsv41-quant-e2` in the PyTorch image, fixtures from the kernel agent's oracle |
| M2 | Greedy `--no-drafts` tokens are identical to the Python engine for 16 prompts × 256 tokens (code, prose, long prompt past one 2048-row chunk, tool call), with expert top-p off (`TF_DSV41_EXPERT_TOPP` unset) and then at the prod 0.85. Selected steps' bf16 logits are equal. Both ranks' per-layer stream hashes are equal. The prefill exact tag is equal to serial; the fast prefill tag is equal to Python's fast tag. Sanity only: teacher-forced top-1 0.9961 (Mia) / 0.9881 (q28). | Python prod configuration on the same pack |
| M3 | Drafted output is byte-identical to `--no-drafts` on the whole prompt set at T=0. At T > 0 it equals Python's keyed sampling for the same seed. Batched equals alone. Graph replay equals eager. | The Zig engine's own serial path, plus Python |
| M4 | HTTP replies have the same tokens as the CLI. `encoding.zig` output is token-identical to `encoding.encode` on the chat fixtures (tools, reasoning, images-as-text). `dsml.zig` produces the same parsed calls and stream fragments as `dsml.py` on recorded replies. | Python app on recorded fixtures |
| M5 | A resumed session's next tokens equal an uninterrupted run, from RAM and from NVMe. Split KV equals unsplit, bit-exact, at 1, 16 and 64 rows and across prefill. | The Zig engine's own unsplit, uninterrupted path |

Speed targets come from the Python engine's measurements:
- Single-stream decode (q28 pack): code about 83–85 tok/s, prose about 47 tok/s; serial about 28 tok/s; a 1-row window about 21.8 ms.
- Cold prefill at 32K: about 2,300 tok/s.
- Boot: about 40 s from a prepared folder.
- Pool: about 2.24M tokens at the 4 GiB floor (v2 pack).

The Zig engine is not shipped to prod until it matches or beats these.

## 7. Risks

1. **R1, Triton dependence.** Most of the default path is Triton. AOT cubins are tied to Triton's version, the constexprs and the specialization keys. The capture has to cover every shape the forward uses: row buckets 1–64, prefill 2048, context buckets.
   - Mitigation: capture from the exact Python image, and refuse to start on a missing variant (as `aot.zig` does).
   - Replace Triton kernels with proven `.cu` twins where they exist.
2. **R2, the 32-row cap.** The Zig engine holds at most 32 rows a forward. Python's row mode takes 64 (4 slots × 16). Up to M3 this limits concurrent drafting, not exactness. Raising it is upstream's open item 3.
3. **R3, no multi-host NCCL bootstrap in the Zig runtime.** `nccl.zig` has the calls but nothing exchanges the unique id across hosts, and no family has run NCCL yet. It is small to write, but it is new and sits on the critical path for M2. The RoCE fast all-gather is a later speed item; prod uses it today.
4. **R4, modules missing from the reference.** `kvpool`, `server`, `roce`, `comm` and `exact_sampling` live in GLM or shared code that the DeepSeek family imports. They need their own reading before M2 and M4.
5. **R5, GB10 memory headroom.** Weights are about 95 GiB per rank (v2), and the hard floor is 4 GiB.
   - The Zig engine must not hold more than Python does: no mmap of the pack (`pack.zig` already reads with pread), pinned staging sized like Python's, and scratch priced into the floor.
   - Boot-time load check as in `memory.load_check`.
6. **R6, host-computed tables.** RoPE/YaRN cos/sin tables, Engram hashing and the sampler's keyed uniforms must equal Python's bits.
   - Gate each with a dump.
   - Prefer computing them the way torch does: fp32 ops in the same order, or copy torch's kernels.
7. **R7, prod is not all-experts exact.** `TF_DSV41_EXPERT_TOPP=0.85` prunes experts lossily but stays rank- and batch-consistent. The `replay` prefill (CED) is approximate by design and has its own session tag.
   - The Zig twin has to implement the same pruning and tags to be token-identical to prod.
   - M2 gates both off first, then on.
8. **R8, GPU access.**
   - The Sparks belong to the Python-engine agent now.
   - M1 can run on a single same-architecture GPU (a RunPod RTX PRO 6000, after approval).
   - M2–M5 need both Sparks in planned windows.
9. **R9, structured output.** Python uses xgrammar 0.2.8 with DeepSeek's structural tag. The Zig server validates grammar fields but has no xgrammar. Choose either a native grammar matcher (the lane core already masks per row) or serving without grammar at first, behind `TF_DSV41_GRAMMAR=0`.
10. **R10, the vision tower.** It runs on rank 0 only in Python (`TF_DSV41_IMAGES=native`). It is out of scope for M1–M5; image requests are refused or served as placeholders.
11. **R11, upstream churn.** `zig-flashnext` moves fast. The family stays in its own directory, and touches of shared files are kept to one function each in `build.zig` and `zig/build/cuda.zig` (the latter is the kernel agent's).
