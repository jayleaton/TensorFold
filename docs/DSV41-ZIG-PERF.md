# DeepSeek-V4.1 on the Zig engine: where a CUDA rewrite pays (2026-10-06)

This study asks which DeepSeek-V4.1-Flash kernel paths are worth rewriting in CUDA for the Zig engine, instead of porting the Python engine's kernels as they are. The target is 2 DGX Sparks: GB10, sm_121, 48 SMs, 273 GB/s, TP=2 over CX7.

It has three parts:
- a time budget for decode and prefill;
- measurements of the current kernels on an RTX PRO 6000 (sm_120) under a GB10 emulation;
- two prototypes:
  - a bit-exact attention merge;
  - an x3gm ablation set that locates its time.

Code is in `zig/kernels/cuda/deepseek_v41/proto/` and builds standalone with `../nvcc-docker.sh`. The `.cu` copies it includes are pinned at kernel commit `2537e81`.

## 0. Bottom line

1. **Decode: the weight streamers are already at the DRAM floor.** In the prod 1-row window, the experts (x3ld) and the dense EXL3 projections (with L2PF) read about 3.5 GB a rank in 14.3 ms. At 240 GB/s that floor is about 14.5 ms. The rest of the window is latency-bound glue:
   - mHC boundaries;
   - exchanges (mostly the wait for the other rank);
   - the Engram host wait;
   - the attention core;
   - about 2 ms of small ops.

   No streamer rewrite (better GEMV, megakernel prefetch, fused verify) can buy much. A fused set of glue kernels can take **3-4.5 ms off every round**. That is about **+9-14% decode tok/s** at today's acceptance. The Zig host's own wins (Engram wait, graph submission) add **+4-7%** on top.
2. **Prefill: x3gm (routed experts, about 45% of a replay chunk) runs at about half its roofline. Neither bandwidth nor tensor rate is the limit.** On 48 SMs with unlimited DRAM it takes 10.3 ms a layer (2 bits, 2,048 rows):
   - Removing the mma saves 4.0 ms. That matches its full-rate time of 3.8 ms, so the mma does not overlap with anything.
   - Removing the trellis decode saves 0.8 ms; activation loads 0.5 ms; weight loads 0.1 ms.

   The kernel is bound by instruction issue: ring, sync, ldmatrix and bookkeeping. A rewrite that brings its compute under its DRAM time (7.1 ms at 2 bits, 10.6 ms at 3 bits) and overlaps the two is worth **-35-45% on x3gm**, about **+17-22% prefill tok/s**, with the same bits. Data movement does not change a chain.
3. **The big precision lever is FP8 on the tensor cores.** sm_120's `mma.sync ... kind::f8f6f4` e4m3 m16n8k32 runs at **2.0x** the fp16/bf16 rate (measured 1,943 vs 957 FLOP/SM/clk). fp16 accumulate is **not** faster than fp32 accumulate on this part. FP8 activations and decoded weights in prefill could give another +15-25% prefill. It changes quality, so it needs the user's sign-off and the quality gate.
4. **Not worth it:**
   - A persistent whole-window megakernel: in-graph gaps are 0.27 ms a window, and the streamers are at the floor.
   - Fused verify for 6-24 rows: the verify slope is the new experts' bytes at the floor.
   - fp16 accumulation.
5. **Prototype:** the attention core's merge, with every load issued before its chains:
   - **bit-identical** to `attn_cuda.cu`: 0 differing outputs at 1, 4 and 16 rows;
   - **33.6 to 29.8 us at 1 row (-11%)**, 36.5 to 31.2 us at 4 rows.

**Recommendation for the Zig engine's primary path:**
- Port the kernels as they are, to pass the M1/M2 exactness gates.
- Then make three things the primary path:
  - (a) the decode glue family of section 4 (R1);
  - (b) the Zig host's Engram and graph-submission fixes (R3, R4);
  - (c) x3gm v2 for prefill (R2).
- Keep FP8 prefill (R5) as a quality-gated option.
- Do not start a megakernel.

**Cost:** one RTX PRO 6000 hold pod for 59 min, **$2.07** (ledger: quant repo `results/zig-pod-ledger.md`). The Sparks were not touched.

## 1. The decode budget

### 1.1 Floor, from first principles (per rank, one 1-row window, Mia's pack, prod top-p 0.85)

| reads a token | GB | ms @ 273 GB/s | ms @ 240 GB/s (achievable) |
| --- | ---: | ---: | ---: |
| routed (~5.9 after pruning) + shared experts, 40 layers | 1.35 | 4.9 | 5.6 |
| dense EXL3 (attention 5-bit, head 6-bit, Engram wkv) | 2.14 | 7.8 | 8.9 |
| router bf16 (replicated) | 0.17 | 0.6 | 0.7 |
| mHC `fn`, KV rows, norms | ~0.1 | 0.4 | 0.4 |
| **total** | **~3.8** | **13.9** | **15.8** |

Fixed latency on top of the bytes:
- **Exchanges:** 83 a window at ~7.5 us of one-way transfer, about 0.6 ms.
- **Kernel boundaries:** 1,331 kernels a window graph at ~0.2 us each, about 0.27 ms.

Two other costs fall outside the window:
- **Graph submission** on the host: ~1.27 ms a round (REWRITE-PLAN 1.2).
- **The DSpark draft pass:** 3.52 ms, plus 0.65 ms a slot.

Our q28-v2 pack decodes within -2.6..+8.5% of Mia's (E1/E2 window), so the same budget holds.

### 1.2 Measured: the prod 1-row window, by class

Source: G14a's hardware-traced nsys of prod's kernels:
- MHC_CUDA, ATTN_CUDA, DENSE_V3, L2PF;
- `nsys-g14mn-off.sqlite`, graph 2: 26 one-row windows, 25.2 ms span under the trace;
- parsed by `proto/nsys_classes.py`.

Prod's calibration reads 21.4 ms for the same window: G15-G19 changes, plus best-of-3 calibration against the trace.

| class | ms / window (traced) | launches | floor | gap | rewrite? |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EXL3 (lanes_linear 5.51, linear 1.88, seg_rot_in 0.31) | 7.70 | 327 | shared with experts (below) | | no |
| routed + shared experts `ld_kernel` | 6.62 | 80 | shared | | no |
| *dense + experts together* | *14.32* | | *14.5 @ 240 GB/s* | *~0* | L2PF's side-stream reads (7.5 ms of `segments_kernel`) already overlap |
| exchanges `gather_kernel` | 2.65 | 83 | 0.6 transfer | 2.0 | mostly the rank-skew wait (DECODE-ROOFLINE 5); balance ranks, not kernels |
| mHC `boundary_kernel` (28 us each) | 2.30 | 81 | ~0.4 | 1.9 | **yes (R1a)** |
| Engram `gate_copy_kernel` (GPU waits on the host's NVMe reads) | 1.65 | 2 | 0 | 1.65 | host (R3) |
| attention core `attn_kernel` (30.6 us each, **10 CTAs**) | 1.24 | 40 | ~0.1 | 1.1 | **yes (R1b)** |
| router `narrow_kernel` (16 us, 3.9 MB) | 0.73 | 40 | 0.7 | ~0 | no |
| MoE small: prune, group, gate/up epilogue, down_combine | 0.64 | 160 | 0 | 0.64 | **yes (R1c)** |
| other small: rms2 0.27, `_plain` 0.22, rot_in 0.15, kv_store 0.15, Engram fuse 0.14, copies 0.21, topk + scores 0.13, rope 0.07 | ~1.36 | ~330 | ~0.05 | 1.3 | **yes (R1c)** |
| in-graph gaps | ~0.27 | | 0 | 0.27 | megakernel only |
| **total** | **~25.0** | **1,331** | **~15.8** | **~9.2** | |

**Verify rows.** The calibration is verify 1 / 2 / 4 / 6 / 8 / 16 / 24 / 32 / 48 / 64 rows = 21.4 / 25.8 / 34.5 / 42.6 / 48.7 / 69.0 / 96.9 / 109.1 / 143.1 / 185.8 ms.
- The marginal cost per row is 4.4, 4.35, 4.05, 3.05 and 2.5 ms up to 16 rows, then 1.5-3.5.
- That is the new distinct experts' bytes. G15b's expert map: U(1) ≈ 5.9, U(16) ≈ 36 a layer, at ~6.4 MB an expert a layer a rank, ≈ 2.1 ms a row at 240 GB/s for rows 8-16.
- So the verify curve is at the floor above its fixed part.
- A fused 6-24-row verify kernel cannot flatten it. Only fewer expert bytes can (a lower-bit pack, or slot-aware row pricing).

### 1.3 Measured on the PRO 6000 under the GB10 emulation (proto/glue_bench.cu, 48 SMs, cold L2)

| rows | `attn_kernel` us | mHC boundary (mode 0) us |
| ---: | ---: | ---: |
| 1 | 34.5 | 16.9 |
| 2 | 34.5 | 19.2 |
| 4 | 35.0 | 30.1 |
| 8 | 55.3 | 37.6 |
| 16 | 94.2 | 63.5 |

**Attention.**
- A CTA takes about 31 us whatever its key count. With 16, 32, 64 or 128 keys in its chunk: 30.7 / 30.9 / 30.7 / 30.7 us.
- 1 row and 4 rows (10 vs 40 CTAs) take the same time.
- So the kernel is a fixed latency chain a CTA. Phase timestamps of the 1-row launch (`proto/attn_phases.cuh`, us):

| phase | us |
| --- | ---: |
| row map + q fragments | 2.4 |
| key rows land (cp.async, 8-byte pieces) | 3.3 |
| S = qK^T (dequant from shared memory per element) | 6.2 |
| softmax | 0.9 |
| PV (dequant again) | 5.3 |
| partials + ticket | 1.5 |
| last CTA's merge | 9.5 |

**mHC.** 17 us under the emulation, against 28 us in the prod trace, where it also waits on the gathered partials. The finish (Sinkhorn + norm) runs on the last CTA after a ticket.

## 2. The prefill budget

Prod at 32K replay prefills 2,310 tok/s. That is 887 ms per 2,048-row chunk. A replay chunk runs the 20 encoder layers; the decoder replays the last 128 tokens once a prompt.

| part of a replay chunk (per rank) | ms | source | floor |
| --- | ---: | --- | --- |
| routed experts, x3gm, 20 layers | ~360-400 | GB10 gmbench: Mia 18.1 ms a layer, ours 15.4 (2-bit L3) to 29-30 (ragged 4-5-bit L22 / L39) | DRAM: 7.1 (2-bit) / 10.6 (3-bit) ms a layer, ~210 ms a chunk |
| dense GEMMs (pfdense, 57-78 TFLOP/s) | ~190 | G17: 375-386 ms for 40 layers | ~65 ms at ~115 TFLOP/s fp16, ~35 at FP8 |
| attention, indexer, mHC, router, exchanges, host | ~300-335 | by difference | ~60-80 (compute + CX7) |

**The third row is the biggest unknown.**
- The only prefill trace with today's kernels (G14a, a short prompt) shows the GPU busy for **36%** of the eager Python prefill span.
- For 2,048-row chunks the host share is unmeasured.
- The Zig engine launches natively, so part of this goes away with the port itself.
- A 32K prefill nsys on the Sparks would size the rest: it is the first item of the next Spark window.

### 2.1 x3gm on the PRO 6000 (proto/x3gm_bench.cu: 384 experts, top-6, D 5,120, I 1,152, uniform routing)

Full chip (188 SMs), 2,048 rows:
- K2 4 (2 bits): 2.87 ms;
- K2 6 (3 bits): 3.01 ms.

The time does not move with the weight bytes, so the kernel is compute-bound there.

GB10 emulation (48 SMs, unlimited DRAM, ms):

| variant (K2 4 / K2 6) | rot + gate/up + down | what it removes |
| --- | ---: | --- |
| x3gm as it is | **10.30 / 10.67** | |
| no trellis decode | 9.53 / 9.86 | -0.8 |
| no mma | 6.32 / 6.88 | **-4.0** (mma at full rate: 3.75 ms) |
| no weight loads | 10.22 / 10.23 | ~0 |
| no activation (member-row) loads | 9.83 | -0.5 |
| epilogue without scale loads / without Hadamard | 10.25 / 10.10 | ~0 |
| 4,096 rows, 64-member tiles (cfg 0), per 2,048 rows | 7.98 / 8.19 | **-22% a row** |
| 4,096 rows, G16's provisional 128-member tiles (cfg 2 / 3) | 11.1-11.9 a 2,048 rows | +8-15%: worse than cfg 0 |
| skewed routing (Zipf 0.8, 460 passes) | 12.09 | +17% |

The no-epilogue variant (ABL 5) is not used: it compiled to 64-77 registers, so it also dropped mma chains.

**Calibration to GB10.**
- 2-bit L3: 15.4 ms measured with plan and combine, against 10.3 ms emulated compute + 7.1 ms DRAM.
- Mia's 3-bit layers: 18.1 ms measured, against 10.7 + 10.6.
- On GB10, x3gm ≈ compute + ~0.5-0.7 × DRAM time: the two barely overlap.

### 2.2 Tensor-core rates (proto/mma_rate.cu, sm_120 at 2,430 MHz)

| mma.sync | FLOP / SM / clock | 48 SMs at 2.5 GHz |
| --- | ---: | ---: |
| f16 x f16 -> f32, m16n8k16 | 957 | 115 TFLOP/s |
| bf16 x bf16 -> f32 | 956 | 115 |
| f16 x f16 -> f16 | 973 | 117 (no gain) |
| e4m3 x e4m3 -> f32, m16n8k32 (`kind::f8f6f4` or plain) | 1,855-1,949 | 223-233 |

**GB10 cross-check.**
- A GB10 report measured 48.8 TFLOP/s for bf16 `mma.sync` and 4x that for NVFP4 ([dgpp PR #50](https://github.com/HawkBearPig/dgpp/pull/50)).
- [StorageReview](https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops) measured 99.8 (BF16) / 207.7 (FP8) TFLOP/s through a library matmul.
- Our own pfdense reaches 78 TFLOP/s on GB10, so the per-SM rate above holds to within GB10's sustained clock.
- `mma_rate` run once on a Spark would settle it: 10 s, no model.

## 3. The GB10 emulation (proto/gb10emu.cuh)

How it works:
- One 99 KB blocker CTA an SM parks every SM but 48 before the kernel under test is launched.
- The kernel then gets the GB10's SM count with its own occupancy (2 CTAs an SM for x3gm, as on GB10).
- Same code, same bits.

Two lessons:
- **Lazy module loading deadlocks behind the parked blockers.** `gb10::eager()` sets `CUDA_MODULE_LOADING=EAGER` before any CUDA call.
- **The DRAM hog did not work.** A hog on the other 140 SMs could only take a 48-SM stream read from 1,291 down to ~800 GB/s, never to 240. So bandwidth is modelled analytically and calibrated against GB10's measured gmbench, not emulated.

Caveats:
- GB10's per-SM tensor and issue rates are assumed equal to sm_120's (the same consumer-Blackwell SM).
- The PRO 6000's 128 MB L2 is larger than GB10's 24 MB.

## 4. Ranked rewrite candidates

Expected gains apply to the calibrated 21.4 ms window and today's acceptance:
- code ≈ 3.9 tokens a round, ~46 ms a round;
- prose ≈ 1.7 tokens a round, ~36 ms a round.

A saving of D ms a window is D ms a round: it is row-independent.

| # | candidate | expected | confidence | effort | exactness |
| ---: | --- | --- | --- | --- | --- |
| R1 | **Decode glue family** (a-d below), one CUDA kernel set for the latency-bound boundaries | **-3 to -4.5 ms a round: code +7-11%, prose +9-14%** | high: every term is a measured per-kernel latency, and the streamers are at the floor | 3-5 weeks | per-row chains kept: row-invariant, batched == alone; bits vs Python change only where a reduction is regrouped (own tag) |
| R2 | **x3gm v2**: issue-reduced, mma overlapped with decode / loads (warp-specialised producer, wider items, fewer stage syncs); compute under DRAM time | x3gm -35-45%: **prefill +17-22%** (32K replay ~2,310 -> ~2,750) | medium: the ablations say where the time is; the target is DRAM-bound | 2-3 weeks | **same bits** (only data movement changes, as x3gm's own configs prove) |
| R3 | Engram reads off the window (Zig host: issue at draft time, io_uring, GPU-side event before layers 1 / 14 only) | -1.0 to -1.6 ms a round: +3-5% | high | 3-5 days | exact (I/O only) |
| R4 | Graph submission hidden (Zig: split or pre-enqueued window graphs, or device launch) | -0.9 to -1.2 ms a round: +2-3% | high | 3-5 days | exact |
| R5 | FP8 (e4m3) tensor path for prefill: pfdense + x3gm (activations per-row scaled, decoded weights cast) | prefill +15-25% on top of R2 | medium | 2-3 weeks | **precision trade**: quality gate (top-1 >= 96%, MMLU within 1) |
| R6 | 4,096-row prefill chunks with cfg 0 tiles | x3gm -22% a row; weights streamed once per 4K | high on speed, blocked on memory | memory work | exact (segmentation-invariant) |
| R7 | Exchange skew (rank 1 ~11 us a layer late) | -1 to -1.5 ms a round | medium | find the cause first | exact |
| — | Whole-window megakernel | +0.3-1 ms beyond R1 | low | months | — |
| — | Fused verify 6-24 rows / flatter verify curve | ~0: the slope is expert bytes | high | — | — |
| — | fp16 accumulation | 0: same rate as fp32 accumulate | measured | — | — |

**R1 in detail:**

| part | today (traced, a window) | design | expected |
| --- | --- | --- | --- |
| a. mHC boundary + the next sublayer's input | 81 x 28 us = 2.3 ms | The finish's norm and the collapsed row move into the consumer's input loader (the x-group / router `rot_in`): their CTAs compute `bf16((c x rn) x w)` from c and rn. The finish keeps only the scalars; Sinkhorn is 20 iterations on 4 x 4. No 5,120-wide single-CTA pass. | -1.0 to -1.6 ms |
| b. attention core | 40 x 30.6 us = 1.24 ms | Split each 128-key chunk across 4 CTAs (S and PV dequant scale with keys: 11.5 us -> ~3); 16-byte key loads; a merge spread over the head tile's CTAs (thread-block cluster / DSMEM, or the consumer's `wo_a` loader). **Prototype done for the merge alone: -11%, bit-exact (section 5).** | -0.6 to -0.8 ms |
| c. MoE small ops + other small kernels | ~2.0 ms, ~490 launches | Prune / group / rot_in into the router's last CTA (b1 of REWRITE-PLAN); gate/up epilogue and down_combine into x3ld's tails; rms2 / rope / kv_store into the x-group's epilogue; copies written at final offsets. | -1.2 to -1.6 ms |
| d. fewer nodes | 1,331 -> ~700 a window | follows from a-c | -0.5 ms of host submission (R4's share) |

For the drafter (dsq-zig-draft-1):
- None of R1-R4 adds a row-count step.
- The attention split is by keys, a fixed partition per row, so per-row arithmetic does not depend on the window.
- The mHC CUDA boundary already covers R <= 16 (`MAXR`); above that, Python falls back to Triton. If the Zig glue kernel keeps that limit, verify[17..64] keeps its own curve. The depth table reads it either way.

## 5. Prototype: the latency-hiding attention merge (proto/attn_fastmerge.cuh)

**The change.** The last CTA's merge in `attn_cuda.cu`:
- loads each chunk's (m, l) inside its dependent chain;
- loads each float4 unit's 5 chunk partials one L2 round trip at a time, 8 units a thread.

The copy issues every (m, l) and all 8 x 5 partial loads first, then runs the same per-element chains in the same order. The kernel body is untouched.

| rows | attn_cuda | fast merge | outputs differing |
| ---: | ---: | ---: | ---: |
| 1 | 33.6 us | **29.8 us** | 0 of 16,384 |
| 4 | 36.5 | **31.2** | 0 of 65,536 |
| 16 | 92.3 | 90.1 | 0 of 262,144 |

That is -0.15 ms a 1-row window on its own. It is the first, exact step of R1b. Since it is bit-exact, it can go into the Python source (sync.py regenerates the copy) as well as into the Zig twin.

## 6. Files

All under `zig/kernels/cuda/deepseek_v41/proto/`:

| file | what |
| --- | --- |
| `gb10emu.cuh` | the 48-SM emulation (`gb10::Emu`: `init(mode)`, `begin()`, `end()`, `stream_gbps`), plus `gb10::eager()` |
| `x3gm_bench.cu` | x3gm at real shapes, production configs, `--abl 0..7`, `--rows`, `--zipf`, `--mode full\|sms\|gb10` |
| `x3gm_ablate.cuh` | gm_kernel with switches for decode / mma / weight loads / row loads / epilogue parts |
| `mma_rate.cu` | mma.sync throughput by type |
| `glue_bench.cu` | attention core + mHC boundary at 1-16 rows, cold L2; `ATTN_KEYS=1` (time vs keys), `ATTN_PHASES=1` (phase stamps); the fast-merge bit check |
| `attn_phases.cuh` | attn_cuda.cu with globaltimer stamps |
| `attn_fastmerge.cuh` | the merge prototype |
| `nsys_classes.py` | per-class times of window graph instances from an nsys SQLite export |

Build, for example:

```
../nvcc-docker.sh -O3 -lineinfo -std=c++17 -gencode arch=compute_120a,code=sm_120a -o x3gm_bench x3gm_bench.cu
```

`glue_bench` needs `--fmad=false` (attn / mHC's flags). Use `sm_121a` for a Spark.

## 7. Beating Python: round 3's phases, and the prompt tail join (2026-10-09, offline)

### 7.1 Where round 3's 4-stream code gap is

Source: round 3's phase reports, the code 4-stream cell (`zig-prof-20261008-133605`, phase deltas per cell). Zig's
`window` (92.9 ms) is longer than Python's (87.4 ms), but the two phases don't hold the same work:
- Zig's row window runs the previous round's deferred keep (ratio-2 carry, `op_rows_keep`), the row table and the
  Engram staging before its launch (3.8 ms outside `candidates`).
- Python times the same work as `commit` (2.1 ms a round), `prefetch` (0.6), `graph.stage` and `forward` (0.3), and
  its speculative pass launch as `spec` (5.2).

Per round, with everything counted:

| | window | draft pass | commit + prefetch + spec | per round |
| --- | ---: | ---: | ---: | ---: |
| Python | 87.4 | 6.3 | 7.9 | **~101.6** |
| Zig | 92.9 | 7.6 | 0.04 | **~100.5** |

So Zig's decode round is already ~1 % faster than Python's in this cell. The cell's gap is **prefill**: 2,153 ms in
30 calls against Python's 753 ms in 6 (per cell, the same 4 prompts). Two parts:
1. Python runs several slots' pieces in one forward (GLM 0560). Zig ran each prompt's pieces alone. dsq-zig-loop-1
   is porting this.
2. Zig ran each prompt's last token as a 1-row window of its own, then a 1-slot drafter pass, before the stream joined
   the rounds. Python's batcher runs a `finals` slot's last token as the pending row of the same round's shared
   window, drafted (batch.py `_pieces` then `_decode`; rounds.py `finish_prompt` then `_windows`). That is 7.2 below.

The 1-stream code cell shows the same: Zig's round (36.0 + 4.0 ms) is below Python's (35.8 + 3.9 + 0.7 commit + 0.9
spec), and Zig's prefill per prompt is ~60 ms longer (the extra window and pass).

### 7.2 TF_DSV41_TAIL_JOIN=1: a prompt's last token joins its first round

With prompts in pieces (`TF_DSV41_PREFILL_PIECES=1`), `finishStream` no longer runs the prompt's last token alone:
- `iface.Target.tail` (target.zig): the decoder replay and its taps to the drafter, the slot left at the last token's
  position with no window pending.
- `Engine.joined` (core/lanes): the stream's pending row is the prompt's last token at position `prompt_len - 1`;
  the head drafts after it now (anchor = the last token, as Python's `fwd.draft(slot, pending, start)`), and the
  stream's first round verifies it with the other streams' windows.
- `LaneHost`: with the knob on, the round's finals open before its decode step, so a finished prompt decodes in the
  same round (Python's `_plan`: `_pieces` then `_decode`).

Bits: the reply's first token is that round's choice at the prompt's last row, keyed at the same position as before
(`draws[0] = prompt_len`); rows are independent in a row window (batched == alone), so every reply is unchanged.
Only the drafts differ (drafted after the last prompt token instead of after the first reply token), which moves
acceptance, not tokens. Left on the old path: a request with a grammar (its first mask is staged on the prompt's last
row), an undrafted stream, a first token forced by the thinking budget, whole-prompt admission (`PREFILL_PIECES=0`).

Host gate: `engine_test` "a prompt's last token joins its first round's shared window" (greedy, keyed sampling, a
1-token prompt, undrafted, budget-cut; joined == not joined == whole prompts, token for token).

Expected: per prompt, one 1-row window and one 1-slot pass fewer (~40 ms on the Spark: a 1-row window ~36 ms, a pass
~4 ms), less the joined rows' share of the shared window (~4 ms a drafted row up to 16 rows). Code 1 stream ~+0.5 %;
the mix's 4 short streams and 4-stream code, where prefill is a larger share, +1.5-3 %; TTFT -30..-40 ms a request.
Spark step: one A/B cell pair with `TF_DSV41_PREFILL_PIECES=1` plus `TF_DSV41_TAIL_JOIN=1`, replies vs Python 4/4.

## 8. Several slots' prompt pieces in one run (TF_DSV41_PIECE_RUNS=1, 2026-10-09, offline)

### 8.1 What round 3's timeline shows

In the nsys timelines of the 4-stream code cell, every prefill launch group was counted by kind. The per-slot
attention launches (`_fused`, `_kv_store`, `_scores`) repeat for every slot in a forward. The expert matmul
(`gm2_kernel`) runs once per forward whatever the number of slots. A 20-layer half-pass is 20 attention launches and
40 expert launches:

| | half-passes (attention) | half-passes with their own expert launches |
| --- | ---: | ---: |
| Python | 14 | 12 |
| Zig | 14 | 14 |

Both engines ran the same pieces (the round planner's golden gate holds). Python ran two of them in another slot's
forward: its batcher puts a round's pieces in one encoder run (slots.prefill_runs, GLM 0560), so their rows share
the expert weight streams. Zig ran each piece alone: two more expert weight passes, ~91 ms of GPU time in the cell
(690 against 599 ms of `gm2_kernel`), ~0.8 % of 4-stream code. The gain grows with longer prompts arriving together.

### 8.2 The run

`block_prefill.emitMulti` emits one program over several slots' segments (Python's forward._run over several
contexts):
- the embedding, every mHC site, every dense projection, the experts and the exchanges run over all the run's rows
  (the segments' rows in order), exactly as a one-segment program of that many rows;
- each segment's CSA2 steps (SWA store through its slot's or the staging ring, the kv source's compressor, the
  indexer's selection, the fused core) run on its own slot, as blocks.attention's loop over its contexts. Glue
  `pf_seg` switches the slot's views (slots.zig `SlotSet.view`, every rank alike, no plan-link message) and writes the
  segment's positions and window starts. The core reads its rows of the run's q / kv / qi / weights / projections
  (`Emitter.row0`) and writes its rows of each wo_a group's plane (`OR` = the run's rows);
- rope over the run's rows reads every segment's positions (`w.mpos64`, glue `mpositions`); Engram's rows are each
  segment's after its own slot's tail;
- CED's encoder pass ends with the decoder's first layer's projection over every row, then each segment's compressor
  and stash (blocks.compress_rows, replay.encode).

`forward_prefill.promptMulti` runs it and commits each slot's segment (carries from its last row, Engram tail,
position, the stash's ids). `Forward.prefillMulti` sends it to the followers (op 52). The target takes a round's
pieces in one call (`iface.Target.pieces`, lanes `Backend.pieces`, `LaneHost` with `Config.piece_runs`) and runs them
as one run in replay mode. It falls back to a piece at a time for a segment of 16 rows or fewer (the short segment's
decode-kernel mix), an image, split KV, more rows than one 2,048-row prefill segment, and full mode (its drafter taps
each piece).

Host gate (`test-dsv41-knobs`, multi_test.zig):
- a one-segment run is the one-segment program, launch for launch and tensor for tensor (full mode and CED's encoder);
- a run of three segments at different positions and lengths (a staged 700-row segment at 4,000, a 100-row one at 0,
  a 33-row one at 129): its per-row launches are an all-rows program's, in order; each segment's CSA2 steps are its
  own program's, in order, at its rows of the run.

GPU gate (pod, `M2B_PIECES=1`, tools/zig/dsv41_pieces/job.sh): Python's batched reference (dsv41_m2b_ref.py
`--batched`: one encoder run over every prompt's piece), our ranks serving the prompts at once with the round planner
and the run (TF_DSV41_PIECE_RUNS=1) against it, and with the run off. Not run yet.

Expected: one expert weight pass saved per extra piece in a run, ~45 ms each on the Spark at bench prompt sizes; ~90 ms
in round 3's 4-stream code cell (~0.8 %), more on the 4-short-stream mix and with longer prompts arriving together.

## 9. Several slots' CED decoder replays in one run (TF_DSV41_REPLAY_RUNS=1, 2026-10-09, offline)

Python's rounds.py replays each finished prompt's tail alone (`finish_prompt` a slot at a time): layers 20-39 over up
to 127 rows, which at that row count streams most of 20 layers' expert weights. When several prompts finish in the same
round (a burst of requests, the 4-stream cells' first round), Zig now runs their replays as one decoder run on the
section 8 machinery (`block_prefill.emitMulti` with `.decoder`):
- every replayed row's per-row launches (mHC sites, projections, experts, exchanges) over all the tails' rows;
- each tail's rows entering the decoder loaded from its own slot's stash (glue `ced_load` at its rows of the run),
  every row's window from its tail's R0 (glue `pf_seg`'s decoder flag), its CSA2 steps on its slot;
- nothing committed: each slot stays at its prompt's last token; each slot's taps go to the drafter from its rows of
  "w.taps" (`GpuTarget.handTapsAt`), in the finals' order, as Python's `replayed` does.

The finals reach the target together before their `finish` / `tail` calls (lanes `Backend.finals`,
`iface.Target.replays`, `LaneHost.runFinals` with `Config.replay_runs`); `forward.replayMulti` sends the run to the
followers (op 53); a replayed slot's `finish` / `tail` then skips its own replay.

Bits: Python never batches a replay, so each batched tail must be its one-slot replay row for row. The dense prefill
GEMM is row-independent with the same bits in every configuration (pfdense.cuh), x3gm the same (its configs move data
only), and Python's mhc_pf is the bit-identical twin of its `_site` from 256 rows. The split mHC programs (32 rows or
fewer) and the short segment's decode kernels (16 or fewer) are other paths, so a tail of 32 rows or fewer (a prompt
of 33 tokens or fewer) replays alone. The GPU gate checks it: our replies with the knob on against Python's reference,
whose finals replay alone.

Host gate (multi_test.zig): a one-tail run is the one-slot decoder program tensor for tensor (33-127 rows, R0 0 and
past it); three tails (127 rows from 3,873, 64 and 33 from 0) are the all-rows launches plus each tail's own CSA2 steps
and stash load at its rows; a 32-row tail is refused.

GPU gate: the PIECES=1 stage (Sparks, run_host.sh) and M2B_PIECES (pod) add a third run, TF_DSV41_PIECE_RUNS=1 with
TF_DSV41_REPLAY_RUNS=1, against the batched reference ("replay runs: " in the log). Not run yet.

Expected: one expert weight pass of layers 20-39 saved per extra tail in a run, ~40-50 ms each on the Spark for 127-row
tails; four prompts finishing together: ~120-150 ms. Python cannot do this without batching its finals.

## 10. Long-prompt prefill: where a segment's time goes, and two levers (2026-10-09, offline)

Today's all-on run (`zig-allon`, 1 rep, TP=2): 32K prompt 2,499 tok/s (Python 2,436), 131K 2,310 (Python 2,266).
That is **400 ms per 1K prompt tokens at 32K, 433 at 131K**: 820 / 887 ms per 2,048-row CED encoder segment (layers
0-19, then layer 20's site and compressor).

### 10.1 Attribution per 1K prompt tokens (one rank; Zig all-on, 32K / 131K)

No trace covers a long Zig prefill yet (round 3's nsys holds only short pieces), so the long segment is rebuilt from
per-kernel measurements at 2,048 rows. The sum lands on the measured 820 ms within 10 ms.

| part | ms / 1K tokens | source | floor ms / 1K (2x GB10) | x floor |
| --- | ---: | --- | ---: | ---: |
| routed experts, x3gm v2 (rot, plan, gate/up, down, combine), 20 layers | **~186** | GB10 gmbench 9.2: v2 16.2 / 21.4 / 21.1 ms a layer at 2,048 rows (L3 / L22 / L39) | 146 (DRAM: trellis once + X / Y, 12.9-17.1 ms a layer) | 1.27 |
| dense EXL3 GEMMs (pfdense fused), 20 layers + layer 20's kv projection | **~93** | G17: 375-386 ms for 40 layers at 2,048 rows, one rank | 32 (115 TFLOP/s fp16 mma) | 2.9 |
| mHC sites + finish (mhc_pf), 41 boundaries | **~34** | mhc_pf.py: 231 MB a boundary, ~1.5-2 ms with the finish | 22 (DRAM) | 1.5 |
| attention, CSA2 compressor, indexer selection, fused core | **~20 / ~54** | context slope 32K -> 131K: +68 ms a segment for +58K mean keys | ~5 / ~15 | 4 |
| TP exchanges (41 bf16 all-gathers, 20 MiB out / 40 MiB in each, NCCL) | **~20** | ~1 ms each at CX7's ~22-24 GB/s, serialized on the compute stream (no prefill overlap before 10.3) | 17 serialized, 0 overlapped | — |
| router, shared-expert glue (swiglu, moe_sum), gm_picks | ~7 | short-piece nsys, scaled | ~2 | |
| Engram (layers 1 / 14: rows, fusion; wkv in dense) | ~2.5 | short-piece nsys; reads ahead (PREFETCH_AHEAD) | 0 (overlappable) | |
| rope, rms2, kv_store, staging copies, positions | ~5 | short-piece nsys | ~2 | |
| CED decoder replay + the last token's verify row, per prompt / segments | ~7 | 20 layers x 127 rows + 1 row, once a prompt | | |
| host and launch gaps (the GPU idle) | **~27** | the remainder; round 3's short pieces: 7 % main-stream idle | 0 | |
| **total** | **400 / 433** | measured | **~250 / ~260** (sum of floors) | 1.6 |

Read: x3gm is 47 % of the time and already near its DRAM floor *at 2,048 rows*. The floor itself is the lever: each
layer's trellis (~2.4 GB a rank at 2.8 bpw) streams once a block, and the 2,048-row block is the segment. The 512 and
2,048 rows points give T ≈ W + slope x rows with W = 8.6-13.1 ms a layer (the weights). The GB10 4,096-row figures
(36.1 / 47.1 / 45.3 ms) are exactly twice the 2,048-row ones because gmbench runs in the 1,024-row shared scratch:
x3gm.Buffers halves a 4,096-row block into two 2,048-row ones (G16-CHUNK4K.md 3), and the trellis streams twice. One
4,096-row block streams it once: **~24 / 30 / 30 ms a layer instead of 32 / 42 / 42, -27 to -29 % a row**.

### 10.2 Levers, ranked (expected prompt tok/s, 32K / 131K, from 2,499 / 2,310)

| # | lever | bits | expected | status |
| ---: | --- | --- | --- | --- |
| 1 | **4,096-row segments, x3gm one block** (TF_DSV41_PF_4K): the trellis streams once a 4K segment | same (segmentation-invariant: G16-CHUNK4K 2) | x3gm -50..-55 ms / 1K: **~2,900 / ~2,630 (+14-16 %)** | built (10.3) |
| 2 | **TP exchange overlap** (TF_DSV41_PF_OVERLAP): each exchange in row pieces on the side stream behind its producer's next piece | same (row-split GEMMs, row-wise glue; the gathered bytes at the same places) | -9..-12 ms / 1K net: **+3 %**; with 1: **~2,990 / ~2,700 (+17-20 %)** | built (10.4) |
| 3 | x3gm v2 128-member tiles at 4K (Zig-own gm2_kernel NG 16 instances, as x3gm3.cu adds v3's): one pass an expert at ~64 members instead of ~1.5 | same (tiles move data only) | x3gm -5..-10 % more: +3-5 % | not built: needs the Spark's timing (the PRO 6000 emulation, compute-bound, had them slower) |
| 4 | x3gm and dense GEMMs overlapped: two 2,048-row micro-batches of a 4K segment, one's x3gm (DRAM-bound) beside the other's attention and dense GEMMs (compute-bound), x3gm's grid capped to leave SMs | same | up to dense's 93 ms / 1K hidden; realistic +8-15 % | not built: SM partitioning has to be measured on the GB10 first |
| 5 | prefill graphs or host-gap removal (27 ms / 1K idle) | same | +4-5 % | needs a 32K nsys first (step P4-2 below) |
| 6 | attention at long context (indexer scores / top-k, fused core) | same | +1 % at 32K, +4 % at 131K | |
| 7 | mHC site at its DRAM floor, finish folded in | same (per-row chains kept) | +2-3 % | |
| — | FP8 e4m3 x3gm / pfdense | **changes bits** | +15-25 % | excluded (replies must equal Python prod) |
| — | redundant CED work | — | ~0: the encoder already skips layers 20-39; the per-segment stash is 5 small copies | — |

### 10.3 TF_DSV41_PF_4K: 4,096-row prefill segments (default off)

Python's G16 4K path (pf4k.py; built, tested on the CPU twin, never run on the Sparks), ported by its rules:
- `TF_DSV41_PF_4K=1` needs `TF_DSV41_PF_4K_MEMORY_OK=1`. Prefill segments and the round planner's rows become 4,096.
  An explicit `PREFILL_CHUNK` / `PREFILL_ROWS` other than 4,096 is refused, because prod's env sets both to 2,048 and
  would otherwise silently keep 2K segments under a 4K label. G19's adaptive rows still step 4,096 -> 2,048 -> 1,024 ->
  512 under memory pressure.
- The router runs in 2,048-row launches (router_gemv.route's CHUNK loop). Each half takes prefill's config. A ragged
  half of 16 rows or fewer takes config(E, rows) = (8, 1) with its own row tile, since route's use_narrow belongs to
  the call.
- x3gm takes the whole segment as one block (pf4k.gm_rows). The emitter's x3gm.Buffers refusal is lifted. The buffer
  plan sizes `s.ex.z` / `s.ex.xg` from the 4K program's views (503 MB of X / Y and 57 MB of Xd a rank, +239 MB over
  today's 264 MB z), as pf4k's arena does in Python. The tiles stay v2's G6 pair (x3gm.tuned2), so 4K requires
  `TF_DSV41_GM_V2`; v1's 4K tuning table is not ported.
- The staging ring becomes ceilPow2(4,096 + 127) = 8,192 rows. The boot check's prefill price uses 4,096 rows.
- `aot-needs` reads the knob: `tools/zig/dsv41_m4/fill-4k.env` is fill-prod.env with 4K and the overlap. Offline
  against the prod fill's signatures, the whole model emits under it (175,650 programs, 137 variants). 7 differ from
  prod's fill: the staging ring's six `_fused` and one `_kv_store` at `RING` 8,192 instead of 4,096 (a constexpr).
  Every other variant, the pieces' included, is prod's.
- Commit 7ba7e89b.

Bits: every kernel computes a row from that row alone, with a fixed order per row. Python's 4K tests (CPU twin:
logits and the whole slot state at 4,096 vs 2,048-row segments, full and replay; resumed and batched) hold for the
same kernels here.

Host gate (`pf4k_test.zig` in test-dsv41-knobs):
- the knob rules;
- a 4,096-row program is the 2,048-row program's calls, role for role, with each route as two CHUNK launches over
  rows [0, 2,048) / [2,048, 4,096), and one gateup2 + down2 over 24,576 pairs a MoE call;
- the ragged 2,056-row case;
- CED's encoder pass and a two-segment piece run at 4,096 rows emit;
- with the knob on, a segment of 2,048 rows or fewer is the 2K program launch for launch, except the staging ring's
  size.

### 10.4 TF_DSV41_PF_OVERLAP=k: the exchanges in row pieces (default off; needs TF_DSV41_BRANCHES=1, TP=2)

Before this, a prefill segment's 41 exchanges sat on the compute stream, each behind its whole producer. With k
pieces (2-8, each at least 256 rows), each layer's:
- **attention tail** runs wo_a's 4 groups + wo_b over rows [r0, r1);
- **MoE tail** runs combine over all rows (x3gm is one block), then the shared expert (w1, w3, swiglu, w2) and
  bf16(routed + shared) over rows [r0, r1).

Each piece's partial then goes out as glue `exchange_rows`. That step copies this rank's rows into the gathered
[2, n, D] at rank 0's / 1's slice and runs an NCCL send / recv of the peer's rows to the same places, so the gathered
buffer holds the whole all-gather's bytes. Pieces 1 .. k-1 are forked onto the branches' side stream (highest
priority) as soon as their producer is issued. The last runs on the main stream after a join, so the next mHC site
reads a whole buffer.

There is no other NCCL call between a fork and its join, so the communicator sees one operation at a time, in the
same order on both ranks. Bits: the same GEMM kernels over subsets of rows (pfdense is row-independent at any M and
config) and row-wise glue. No reduction is regrouped; the mixing site's rank-order sum is unchanged.

Host gate (`pfovl_test.zig`):
- the reader;
- at 1,024 / 2,048 / 4,096 rows and k = 2 / 4: every call outside the pieced producers and exchanges is the off
  program's, in order; k pieces an exchange, k-1 forked onto the side and the last joined;
- the pieces tile [0, n) on the 16-row grid, each right after the producer that wrote its rows;
- `branches.shared` (roles both streams touch inside a fork region) is exactly {L.part, L.moe16}, at disjoint rows;
- under 256 rows a piece, or without the side stream, it is the off program.

Commit 5e814225. Expected: at ~22-24 GB/s an exchange's 20 MiB takes ~0.9 ms. Pieces hide (k-1)/k of it behind wo_a + wo_b (~2 ms) and
the shared expert (~1 ms). Each piece costs a rot_in launch and a shorter GEMM wave: net -9..-12 ms / 1K, +3 %. At 4K
the same share (twice the bytes behind twice the work).

### 10.5 The Spark step (P4, 1 rep, against `zig-allon`)

| step | what | pass |
| --- | --- | --- |
| P4-0 (dev box) | build the kit with `TRITON_FILL=1 TRITON_FILL_ENV=tools/zig/dsv41_m4/fill-4k.env` (+ `TRITON_FILL_HAVE` the served aot dirs: 7 new variants, the staging ring's `_fused` x 6 / `_kv_store` at `RING` 8,192) | compile.json `"failed": []` |
| P4-1 | `scripts/v2-rollback.sh <allon1's overrides> TF_DSV41_PF_4K=1 TF_DSV41_PF_4K_MEMORY_OK=1 TF_DSV41_PREFILL_CHUNK=4096 TF_DSV41_PREFILL_ROWS=4096 TF_DSV41_PF_OVERLAP=4`, then `bench_http.py --long 32768,131072` and the 8 cells, `--compare` with `zig-allon/bench-allon1` | boot log `prefill rows 4096 (4K: one x3gm block a segment)` and `prefill overlap: ... 4 row pieces`; no `MissingTritonVariant`; **long replies == Python's** (sha 30710c1d388d4cd7 / c5c36138ceaea475) and T0 20/20; 32K >= 2,750, 131K >= 2,550 prompt tok/s; worker MemAvailable minimum above the floor |
| P4-1b (only on a failure) | the same with `TF_DSV41_PF_OVERLAP=0` (4K alone), then with 4K off and `TF_DSV41_PF_OVERLAP=4` alone (prod's 2K rows) | names the failing lever |
| P4-2 (optional) | one 32K prefill under the levers with `TF_DSV41_PROFILE=1` (or nsys) | replaces 10.1's model with a measured budget, and sizes lever 5 |

### 10.6 P4 on the Sparks (2026-10-09) and what changed

**4K without the drafter overlap (P4b): exact, prompt +7 %, decode -7..-14 %.** Code x4 lost 921 ms with the same
rounds, tokens and prefill. The time went to the window phase outside its launch: 0.52 ms a round in allon2b and
0.30 with the drafter overlap off (the one-pass base), but **11.67 ms in P4b**, while the launch itself got faster.
That is the row graphs losing their room: the cache captures only while MemAvailable >= TF_DSV41_GRAPH_FLOOR_GIB (5).
The crash run logged 8 keys held, 4 evicted and 1 eager, where every base run shows 0 / 0. The cause was 4K's
resident bytes. In the forward's buffer plan, 4,096-row segments grew the window / layer arena (which decode shares)
from 1,175 to 1,966 MiB and the prefill scratch from 356 to 649 MiB: **+1.1 GiB a rank, all the time**.

**Fix: the 4K workspace (forward_prefill.K4).**
- Under TF_DSV41_PF_4K the forward's options, buffer plan and arena are the 2K ones, byte for byte (prod_knobs.apply
  no longer touches them).
- A 4,096-row segment runs its program in its own namespace (block_prefill.ownCalls, "k4~"). Its roles are sized by
  its own plan at boot, allocated (zeroed) at a prompt's first 4K segment and freed at the prompt's CED finish. The
  slot's state and the constant tables are the forward's.
- Its send and gathered buffers are the workspace's while it runs (swapped), so the buffers decode graphs hold
  addresses of are never regrown.
- Host gate (pf4k_test.zig): every role the 4K program names outside its namespace is in the 2K plan, at no more than
  the 2K plan's size, and none of them is a scratch role. 4K adds no byte to the forward's plan; its workspace is
  > 1 GiB, held only while a long prompt prefills.

**The 4K + drafter overlap crash.** `cuEventRecordWithFlags: CUDA_ERROR_ILLEGAL_STATE` on both ranks was the drafter's
taps mark recorded with CU_EVENT_RECORD_EXTERNAL in an eager row window. That flag is valid only inside a capture.
4K produced the first eager window (no graph room). Fixed on dsv41-zig-draftovl 150f0c9f: an external node in a
capture, a plain record otherwise. Any eager window with TF_DSV41_DRAFT_OVERLAP=1 would have crashed the same way.

**Why TF_DSV41_PF_OVERLAP=4 lost 1.6-3.3 %.** The pieces cut the producers into 512-row GEMMs:
- wo_a's group GEMM at 512 x 1,024 is 32 tiles on 48 SMs, under one wave;
- the shared expert's w1 / w3 are 36 tiles;
- wo_b's is 3.3 waves against 13.3.

That is four times the launches (~32 GEMMs + 32 rot_in a layer instead of 8 + 8). The NCCL pieces still move the same
bytes at link speed, and their kernels hold SMs beside the shortened GEMMs. Net: ~+45 ms a segment of fragmentation
against ~20 ms hidden. k = 2 halves the fragmentation, but wo_a is still 1.3 waves. The structural fix is TBO (section
11): the other micro-batch's whole GEMMs run while one's exchange is in flight, nothing split. PF_OVERLAP stays off
(and is off under TBO).

**Prompt speed under 4K (+7 % against a modelled +14-16 %).** One 4,096-row x3gm block has never been timed on the
GB10; the model extrapolated from the 512 / 2,048-row points. proto/tbo_bench.cu times x3gm alone at 2,048 and
4,096 rows (its first table rows), which says how much of the model's per-row gain the GB10 gives. If one 4K block is
only ~15 % a row cheaper, ~2,700 / ~2,485 is 4K's ceiling. The next gains inside 4K are then the 128-member v2 tiles
(one pass an expert at ~64 members) and TBO.

## 11. Two-batch overlap: x3gm beside the dense GEMMs (TF_DSV41_PF_TBO, 2026-10-09, offline)

### 11.1 Why

Per 2,048-row micro-batch and layer (10.1), the two halves of a layer use different limits:
- **the MoE part** is x3gm v2, ~19 ms at ~1.27x its DRAM floor. Its SM issue is ~9.5 ms;
- **the attention part** is ~18.5 ms: the dense GEMMs (~9.5 ms, compute-bound), mHC sites (~3.5, DRAM), attention
  (~2), exchanges (~2) and small glue.

Run serially, the two parts add up to ~37.5 ms a layer for two micro-batches. If one micro-batch's MoE runs beside the
other's attention, the pair's layer is bounded by the SM work of both (~22-25 ms) and DRAM (~17 ms) instead. That is
**up to -25..-33 % a layer pair; expected +10-30 % prompt tok/s**, depending on how well the GB10 shares its SMs between
a persistent x3gm grid and pfdense's waves. proto/tbo_bench.cu measures that split (11.4).

### 11.2 The program (block_prefill.emitTbo)

A prompt's CED encoder segments go two at a time (forward_prefill.pairOf: two whole segments, or what is left halved
on the 16-row grid, each half at least 512 rows). The pair is one program on two streams:

```
main:  A.attn(L)        A.moe(L)          A.attn(L+1)        A.moe(L+1) ...
side:        [fork] B.attn(L)    B.moe(L)    [fork] B.attn(L+1) ...
```

- A's calls are exactly `emitReplay(.encoder)` of rows [start, start + na); B's are those of [start + na, start + na +
  nb), with B's scratch renamed (`tboRole`: "w." / "L." roles "w.b~" / "L.b~", scratch "s." roles "s.b~": x3gm's z / xg,
  the dense workspace, the router's counters, the staging ring). So each row runs a consecutive segment's arithmetic,
  which is segmentation-invariant (Python's 1K / 2K / 4K tests; the Zig 2K path relies on the same). **Bits: Python's.**
- Every B attention part is forked behind A's (the side stream waits for everything the main stream issued). So B
  reads the slot's state of layer L (SWA ring rows, the pool's compressed rows and index keys) only after A wrote it.
- A hands its ratio-2 carry of each kv source to the slot at the end of that layer's attention (glue `copy_rows`:
  the copy A's commit would make), before B's layer reads it.
- The only state both streams touch at once is a kv source's compressed rows. A's later layers read them up to A's
  positions while B's source layer appends past them. The boundary is even, so no ratio-2 pair straddles it.
- A's CED stash is left out: B's, of at least 128 rows, overwrites all of it.
- An Engram layer's A part joins B first when B ran an Engram step since the last join (the Engram glue's gathered
  buffer and pinned stages are one set). That happens once a pair, at layer 14.
- The program ends joined (glue `tbo_join`). The commit is the pair's: B's last rows' carries, then the Engram tail,
  the position and the stash ids after both.
- The glue runs each micro-batch's steps with its own host side: ids, window, Engram tail, MoE count and its own
  long-prompt buffers (forward_prefill.glueFn's swap).
- x3gm's persistent grid covers at most `TF_DSV41_PF_TBO_GM_SMS` SMs (default 32 of 48; 0: all). Its CTAs claim items
  by ticket, so the grid changes no bit. The remaining SMs run the other micro-batch's attention.
- The round planner's rows double (a piece is a pair), and the boot check prices a pair.

Knobs:
- `TF_DSV41_PF_TBO=1`: needs `TF_DSV41_BRANCHES=1` and replay. It turns `TF_DSV41_PF_OVERLAP` off, since B's work hides
  A's exchanges.
- `TF_DSV41_PF_TBO_GM_SMS=<8-48|0>`.
- TBO and `TF_DSV41_PF_4K` are exclusive for now: with both on, a prompt runs 4K segments in the 4K workspace (10.6).
  4K micro-batches (8,192-row pairs) would need B's namespace inside the workspace.

Refused, so the segments run one at a time: split KV, images, an odd boundary, a half under 512 rows.

### 11.3 Host gate (pftbo_test.zig in test-dsv41-knobs)

- the readers and the pair split;
- at (2,048 + 2,048 at 0), (1,504 + 1,505 at 6,144) and (512 + 512 at 40,000):
  - the main stream's calls == A's encoder program, call for call and role for role, except its stash;
  - the side's == B's renamed;
  - one `copy_rows` a ratio-2 kv source, each right before the fork of B's same layer;
  - one Engram join;
  - the closing join;
- `branches.shared` (roles both streams touch inside a fork region): constant tables and the kv sources' compressed rows
  only. It found the compressed rows on its first run, which is what the even boundary is for;
- refusals.

### 11.4 The Spark micro-benchmark (proto/tbo_bench.cu)

`zig/kernels/cuda/deepseek_v41/proto/tbo_bench.cu` (+ `tbo_bench.sh`, which builds and runs it in the Sparks' CUDA
devel image) runs one layer's x3gm v2 (rotation, gateup2 GU1 with two inputs, down2 DN0; 384 experts top-6, K2 6) and
the dense set of one layer (pfdense lanes at the Zig heuristic's configs: wq_a, wkv, wq_b, ix_wq_b, wo_a x 4, wo_b,
the shared expert's w1 / w3 / w2) at 2,048 and 4,096 rows. Each is median of 5 after a warm-up, with L2 flushed by a
64 MB memset. It times:
- serial (one stream);
- x3gm alone with its persistent grid capped at S x per_sm for S in 48 .. 24;
- the dense set alone;
- both at once (x3gm capped on the low-priority stream, the dense set on the high one);
- with `--green`, CUDA green contexts (x3gm S SMs, dense the rest; host-timed against a host-timed serial).

At every cap Y is compared on the device with the uncapped run: `x3gm bits: PASS (same at every cap)`. The output is a
table a row and `BEST R=<rows>: x3gm <S> SMs, <mode>, <ms> vs serial <ms> (-x %)`. Its x3gm-alone rows at 2,048 and
4,096 rows are also the first GB10 timing of one 4,096-row block (10.6).

Run on dgx-01, about a minute and no model (check the test-window lease first):
copy `zig/kernels/cuda/deepseek_v41/` there, then `proto/tbo_bench.sh --green`. Set
`TF_DSV41_PF_TBO_GM_SMS` to BEST's S.

### 11.5 Spark step (T1, after P4)

| step | what | pass |
| --- | --- | --- |
| T1-0 | `tbo_bench.sh` on dgx-01 (1 min, no model): the best x3gm SM count S | `x3gm bits: PASS`; a split under serial |
| T1-1 | allon2b's overrides + `TF_DSV41_PF_TBO=1 TF_DSV41_PF_TBO_GM_SMS=<S>` (4K off: pairs of 2 x 2,048), `bench_http.py --long 32768,131072` + the 8 cells | boot log `prefill tbo: ... at most S SMs`; long replies == Python's (sha 30710c1d388d4cd7 / c5c36138ceaea475), T0 20/20; prompt tok/s against P4-1; MemAvailable minimum (a pair holds two segments' activations) |
| T1-1b (on a failure) | `TF_DSV41_PF_TBO_GM_SMS=0` (no cap), then TBO off | names the part |

### 11.6 tbo_bench on dgx-01, and TBO with 4K (2026-10-09)

`tbo_bench.sh --green` (kit ff16c8e2): `x3gm bits: PASS` at every cap. Per 2,048 rows of one layer's x3gm + dense set:

| | ms / 2,048 rows | vs 2K serial |
| --- | ---: | ---: |
| 2K serial | 26.96 | |
| 2K, best split (green contexts, x3gm 24 SMs) | 23.75 | -11.9 % |
| 4K serial (one x3gm block) | 18.95 | -30 % |
| 4K, best split (priority streams, x3gm uncapped) | 17.90 | -34 % |

So the 4K block is the big term. The bench's -5.5 % at 4,096 rows is the overlap's gain on top of it: x3gm at 4,096
rows already fills the GB10. Ranking by the bench: **4K + TBO > 4K > TBO at 2K**. P4b's end-to-end +7 % is well under
the bench's x3gm + dense -30 %, which covers ~70 % of a segment and so predicts +20-25 %. Two candidates:
- P4b ran while 4K's resident plan bytes had pushed the node into memory pressure (10.6);
- something outside x3gm + dense grows at 4K.

The rerun of 3e5c3756 separates them.

**Changes (this commit):**
- **TBO's pairs run in the prefill workspace too** (forward_prefill.wsPair). Before, B's private roles (its arena and
  scratch, ~1.5 GiB at 2K) sat in the forward's resident plan: the P4b decode regression, for TBO. Now B's roles are
  "k4~b~". A 2K pair's A keeps the forward's 2K roles; a 4K pair's A is "k4~" too. The pair's send / gathered buffers
  are the workspace's while it runs. Host gate: a 2K or 4K pair names no role outside the workspace that the 2K plan
  does not hold, at no more than its size, and no "b~" role outside it.
- **TBO with TF_DSV41_PF_4K:** pairs of 4,096-row micro-batches (pairSplit at 4,096), each its 4K encoder program:
  two CHUNK route launches and one x3gm block a MoE call (host gate: the main / side streams' calls == A's / B's 4K
  programs).
- `TF_DSV41_PF_TBO_GM_SMS` defaults to 0 (all SMs: the 4,096-row best). The engine has priority streams and grid caps,
  not green contexts. Green contexts were the 2K best; at 4K the uncapped streams split wins, which is the case that
  matters.
- Memory: the workspace holds a 4K pair's two micro-batches (~2 x 2.6 GiB, from the 4K program's plan) while a long
  prompt prefills, priced at 8,192 rows (kv_state); a 2K pair holds ~1.5 GiB. G19's adaptive rows still step a piece
  down under pressure, and pairSplit halves a shorter rest.

**128-member x3gm tiles at 4K** (`--tiles`, new). At 4,096 rows an expert averages 64 members, so G6's 64-member passes
split about half the experts in two, and each such expert's trellis is fetched and decoded twice. x3gm.cu's NG 16
tiles take 128 members a pass. The bench times x3gm v2 alone at GU2 / GU3 / GU4 / DN2 / DN3 / DN4 pairs against G6's
GU1 / DN0, each on its own BM's plan, Y bit for bit against G6, and prints `TILES R=<rows>: best ...`. The PRO 6000's
emulation (section 2.1) had them slower; it was compute-bound with unlimited DRAM, and the GB10 is DRAM-bound at
4,096 rows. If a pair wins by more than a few %, the engine needs those gm2_kernel instances (Zig-own, in x3gm3.cu as
v3's are; the same bits by construction and by the bench's check) and a tile knob. Not built until the Spark says so.

### 11.7 Spark steps (T2, after the 3e5c3756 rerun)

| step | what | pass |
| --- | --- | --- |
| T2-0 | `proto/tbo_bench.sh --tiles --rows 4096` (dgx-01, ~1 min, no model) | `x3gm bits: PASS`; the TILES line decides the tile work |
| T2-1 | the all-on base + 4K (`TF_DSV41_PF_4K=1 TF_DSV41_PF_4K_MEMORY_OK=1 TF_DSV41_PREFILL_CHUNK=4096 TF_DSV41_PREFILL_ROWS=4096`) + `TF_DSV41_PF_TBO=1`, AOT fill-4k.env; `bench_http.py --long 32768,131072` + the 8 cells | boot log `prefill tbo: ... at most 0 SMs (0: all)` and `prefill 4K: workspace N MiB`; long replies == Python's (sha 30710c1d388d4cd7 / c5c36138ceaea475), T0 20/20; decode cells within noise of allon2b; `row graphs: ... 0 evicted, 0 eager`; worker MemAvailable minimum over 4 GiB; prompt tok/s against the 4K-alone rerun |
| T2-1b (on a failure) | TBO without 4K (2K pairs), then 4K without TBO | names the part |

### 10.7 The 131K decode loss after a 4K prompt (all-on run on kit ff16c8e2, 2026-10-09)

3e5c3756 + 150f0c9f were exact, and every short decode cell was within ±0.6 % of base. Prompt speed: 32K 2,668, 131K
2,462 tok/s (+6.2 %). 131K decode, the cell right after the 131K prompt, lost 12 %; 32K decode lost 1.6 %.

The long cell's phases (bench-next-allon against bench-k860-base) show the same rounds and tokens / round (2.667) and
+2 ms a round: the window outside its launch 1.72 -> 2.74 ms, the GPU candidates +1, the Engram wait per call +2.3.
That is graph churn again, smaller than P4b's.

The cause is in the graph cache (cuda/graph_cache.zig). A miss under the memory floor on any rank evicts the least
recent quarter and lowers the cap to what is left, and **nothing ever raised it again**. Holding the 2.5 GiB workspace
for a long prompt, or MemAvailable lagging the free on the GB10's unified memory, is a dip. After it, every new key
evicts an old one. 131K decode is the case that meets new keys (a new context bucket) right after the longest hold,
so it lost the most.

Fix: the next miss with room on every rank puts the cap back to `max`. The same bits; host tests show a dip followed
by 12 new keys gives only the evictions the cap forces, and the replays after it hit. Logs added:
- `graph cache: under the memory floor on this / another rank: cap A -> B (...)` when the floor bites;
- `graph cache: room again on every rank: cap B -> 48` when it recovers;
- `prefill 4K: workspace N MiB freed at the prompt's end (release k); MemAvailable X -> Y GiB`: the prefill-to-decode
  handoff, which tells whether the free reaches MemAvailable (suspect 2).

Spark check: the same all-on + 4K run. Pass:
- 131K decode within noise of base;
- the handoff line's MemAvailable after the free above TF_DSV41_GRAPH_FLOOR_GIB (5);
- at most a transient `cap ->` warning followed by `room again`;
- the server's stop line `row graphs: ... eager` against base's.

If the MemAvailable after the free stays low (the driver keeps the pages), the next step is one workspace allocated
at boot and held (priced in the boot check) instead of per prompt.

### 10.8 Where 4K's missing gain is (T2-0, dgx-01, 2026-10-09)

tbo_bench at 4,096 rows: x3gm 25.5 ms a layer, the dense set 12.0, serial 37.5. TBO at 4K buys 2.2 %, the best
128-member tiles (GU2 / DN0) 3.3 % of x3gm, about 1 % of a segment. GU2 / DN0 is not built: it needs a second plan
a MoE call (gate/up at BM 128, down at 64), a Zig-own gm2 instance and its resolution, for about 1 %.

The segment arithmetic is where the gap shows. At 4K a segment is 4,096 / 2,668 = 1,535 ms. 20 layers x 37.5 =
750 ms of it is x3gm + dense, which leaves **~785 ms outside them, ~392 ms a 2,048 rows**. At 2K the same split is
817 - 20 x 26.96 = 278 ms. The work outside x3gm + dense grew ~40 % per row with the segment.

Offline check: the emitted work does not grow. Every kernel's total grid and every glue step's tensor volume over one
4,096-row segment equals two consecutive 2,048-row segments' at 32K and 131K, except one ticket memset fewer a layer.
So the extra ~115 ms a 2,048 rows is runtime:
- a kernel's efficiency at 4K shapes (attention, mHC, the indexer's row blocks under the 64 MiB index budget, the
  exchange at 40 / 80 MiB);
- host stalls the GPU waits on (Engram's layer-1 fill, with only layer 0 queued ahead of it);
- the per-prompt workspace allocation.

Only a profile splits it.

**Found while reading the piece path: Zig had lost Python's Engram read-ahead across pieces.** With
TF_DSV41_PREFILL_PIECES=1 (the all-on base), each round's piece is its own `prefill` call. forward_prefill.prompt
reads ahead only to the next segment inside a call, so it never read ahead at all: every segment's Engram rows at
layers 1 and 14 were synchronous reads, even with TF_DSV41_PREFETCH_AHEAD=1. Layer 1 has only layer 0 queued ahead
of it on the GPU, so its read is exposed.

Python reads the next round's piece ahead (rounds._prefetch(ahead=prefetch.next_pieces): the rows after the piece,
as many as it held, up to the prompt's last token). Now Zig does the same:
- GpuTarget.piece sends those ids (Forward.prefillAhead, plan-link op 60, every rank);
- the call's last segment issues their reads behind itself (forward_prefill.ahead).

I/O only: the rows are the shard's whatever the path. This applies at 2K and at 4K. P5's profile shows it as
glue.engram_rows' time per 1K rows; compare with the 3e5c3756 run.

Tooling (this commit):
- the prefill profile records the encoder rows and CED tail rows it covered (`"rows"`, `"tail_rows"`);
- `gap.py prefill-ab --a <2K profile> --b <4K profile> --labels 2K,4K` prints, per 1K encoder rows: wall, GPU,
  host (wall - GPU), GPU by family, and the calls that moved most. A glue step's time includes a host stall the GPU
  waited on.

**Spark step P5** (two boots, 1 rep each, the same base as the all-on 4K run, kit with 4764d3b1 and this commit):

| step | what | pass / read |
| --- | --- | --- |
| P5-1 | base (4K off) + `TF_DSV41_PROFILE=/state/prof-2k.json`; `bench_http.py --long 131072` alone; stop the server cleanly (the profile is written at close) | `/state/prof-2k.json` has `"prefill"` with `"rows"` |
| P5-2 | the same + `TF_DSV41_PF_4K=1 TF_DSV41_PF_4K_MEMORY_OK=1 TF_DSV41_PREFILL_CHUNK=4096 TF_DSV41_PREFILL_ROWS=4096`, profile to `/state/prof-4k.json` | the handoff line `prefill 4K: workspace ... MemAvailable X -> Y GiB`; any `graph cache: ... cap` lines |
| P5-3 | dev box: `python3 tools/zig/dsv41_perf/gap.py prefill-ab --a prof-2k.json --b prof-4k.json --labels 2K,4K --top 25` (rank 0's, and rank 1's) | the family / calls whose ms per 1K rows grew: the next lever |

Profiling times every prefill call with an event pair (eager anyway). Read the prompt tok/s of these runs as
profiled, not as the speed result.

### 10.9 T2-1 (kit 1fc03a5b, 2026-10-09): 4K joins the all-on set; TBO + 4K refused at boot now

- **Arm 1, all-on + 4K: PASS.** Exact. Prompts 32K 2,661 (+9.2 % vs Python), 131K 2,495 (+10.1 %). 131K decode 64.8
  (base 65.2). The frees logged MemAvailable 5.10 -> 7.60 and 4.95 -> 7.46 GiB, with no cap drop and 0 evicted. The
  cache's cap recovery (10.7) holds.
- **Arm 1p, profile: no file.** The serving model never set the runner's profile; only tf-dsv41-m1's gates did. Now
  Model.open sets it from TF_DSV41_PROFILE and Model.close (the SIGTERM / SIGINT stop) writes it (`profile: <file>
  written (...)`).
- **Arm 2, 4K + TBO: rank 1 killed by the kernel's OOM** at its first long prompt (workspace 5,091 MiB). The boot
  check predicted 6.87 GiB worst without the workspace, so 1.8 GiB with it, under the 4.0 floor. Two fixes:
  - **Boot:** forward_prefill.workspaceBytes sizes the workspace from its own plan before the pool. kv_state.bootCheck
    subtracts it from the predicted worst MemAvailable and refuses under the hard floor, naming the pool cut or
    TBO's removal. At T2-1's numbers 4K alone passes (4.95 GiB measured at the free) and TBO + 4K is refused.
  - **Runtime:** k4Acquire checks MemAvailable against the workspace and the hard floor before allocating. On the
    GB10 an allocation past host memory is a kernel OOM kill, not an error return. A short check logs `prefill
    workspace: N MiB would leave X GiB ...` and returns OutOfMemory, so fail-fast takes both ranks down with that line.
    An allocation error is logged too.
- **TBO + 4K is not pursued further** (T2-0: at most 2.2 % at 4K).
- **The workspace's allocation is timed:** `prefill 4K: workspace N MiB allocated and zeroed in T ms (prompt k)`. On
  the GB10 it maps ~2.5 GiB of host pages once a prompt, outside the prefill profile's segments. At ~1 s it would be
  ~7 % of a 32K prompt's 14 s.

**The next prefill step:** P5's profile, the M2c-2 gate's (PREFILL=1 PROF_PREFILL=1) or the served one now. Read with
`gap.py prefill-ab` against a 2K run, together with the allocation line above and glue.engram_rows' share after the
read-ahead fix (10.8).

### 10.10 chain11 (kit fd323c37): the workspace guard downed 4K; now a fallback, never a failure

c96b9662's runtime guard refused the workspace on rank 1 (`2552 MiB would leave 3.87 GiB of 6.37 GiB`) and returned
OutOfMemory, so fail-fast took both ranks down, in the 4K P5 boot and in the all-on run.

**Fix:**
- No failure for room. Each prompt call decides on every rank alike whether it may use the workspace:
  forward_prefill.wsAgreed does one all-gathered flag (graphs.Agreement, the graph caches' collective) of
  `MemAvailable - workspace >= TF_DSV41_PF_WS_FLOOR_GIB` (default: the hard floor, 4.0).
- If any rank says no, the call runs 2,048-row segments, with the same bits (4K and 2K are segmentation-invariant).
  It logs `prefill workspace: ... does not fit on every rank (rank r: ...): this prompt piece runs 2,048-row segments
  (fallback k)` for the first 4 and every 64th.
- A held workspace (a later piece of the same prompt) needs no new decision. TBO pairs need the agreement too.
- Only a failed allocation after the agreement is still an error: it was agreed to fit.
- The workspace lines now print on every rank, tagged.

**The "1 GiB less headroom" is the ranks, not growth.** Kit fd323c37's boot check predicts 6.76-6.77 GiB worst on rank
1 and 7.45-8.08 on rank 0 (P5's 2K and 4K boots). Rank 1's 6.37 GiB before the workspace is its own baseline. t21's
4.95 / 5.10 GiB lows were rank 0's: the free line was rank 0's only. Rank 1 on t21 most likely dipped to ~3.9 GiB with
the workspace held, unchecked, without harm.

The changes between the kits (ca14bc81's counters, 0a080f59's ENGRAM_WARM default off, c96b9662) hold little
resident memory. The read-ahead's batches are ~1 KiB a record, tens of MiB a layer. So at the default floor rank 1
falls back on every prompt, and 4K would not run at all. Run 4K with `TF_DSV41_PF_WS_FLOOR_GIB=3.5` (t21's effective
operating point) until the workspace shrinks.

**Holding one workspace from boot is not viable:** 2.49 GiB at 1,790 B a token is ~1.49M pool tokens, more than the
whole 1.40M pool. It would also sit under the row graphs' 5 GiB floor on rank 1.

**The way to make 4K fit is a smaller workspace.** It holds the 4K program's whole window / layer arena (~2.0 GiB)
plus its scratch (0.65). The forward's 2K arena (1.2 GiB) is idle while a 4K segment runs, and 2K prefill pieces
already overwrite it between decode rounds. Laying the 4K roles that only prefill programs use over the forward
arena's prefill-only roles would cut the workspace by up to ~1 GiB. It needs the plan's prefill-only role set, the
forward and drafter state proven untouched (no cross-window "w." value under those ranges), and a device sync at the
4K segment's start for side-stream readers. Not built yet.

**2K profile at 131K** (P5, rank 0, ms per 1K prompt rows, 453 total, host 0.7):

| call | ms | share |
| --- | ---: | ---: |
| x3gm (gate/up 77.5, down 52.7, rot 13.9, combine 13.6) | 157.7 | 35 % |
| dense (pfdense 70.4, rot_in 15.6) | 86.0 | 19 % |
| `_stream` (long-context top-k, 219 calls, ~36 ms each) | 52.8 | 12 % |
| `_fused` (attention core) | 43.7 | 10 % |
| mHC (site 27.2, finish 8.8) | 36.0 | 8 % |
| exchanges | 28.7 | 6 % |
| router | 16.8 | 4 % |

So at 131K the long-context selection and attention (21 %) is the largest non-GEMM block.

**Spark step P5b:** the 4K boot of P5 with `TF_DSV41_PF_WS_FLOOR_GIB=3.5`.
- Pass: boot log `prefill 4K: rank 0/1: workspace ... allocated` on both ranks, no `does not fit`, exact replies.
- Then `gap.py prefill-ab --a prof-2k-r<r>.json --b prof-4k-r<r>.json --labels 2K,4K`.
- If a rank logs the fallback, the run still passes (2K bits); the profile then shows 2K.

## 12. The long-context stream top-k, row-blocked (TF_DSV41_STREAM_RB, 2026-10-09, offline)

### 12.1 What bounds `_stream`

At 131K, P5's 2K profile puts `_stream` (csa2/stream_topk.py: the indexer's scores and top-512 of layers 2 / 8 / 14
past 4,096 visible keys) at **52.8 ms per 1K rows, 12 % of prefill**: 219 calls of ~36 ms. A program is one (row, key
split of 16,384) pair. Per 64-key tile it runs a serial chain:
- the keys' L2 load (132 B a key: 128 e4m3 + an fp32 scale);
- the fp8 dequant;
- a [32 heads, 128] x [128, 64] `tl.dot`;
- the weighted ReLU head sum across 8 warps;
- the threshold count (a CTA reduction), the append scan, the store and a barrier.

A ~2,048-row call at ~38K mean visible keys is 7.8e7 (row, key) pairs: **~55 SM cycles a pair**. Against that:
- the tensor cores need ~9 cycles a pair (4,096 MAC at 478 MAC / SM / clock);
- the key bytes ~2 (L2-resident: 38K keys x 132 B = 5 MB a layer);
- the dequant and epilogue ~2-3.

So `_stream` is bound by latency and issue, about 4x above its own compute. sm_121 AOT build: 116 registers, 16 KB
shared, 2 CTAs (16 warps) an SM, so 2 rows in flight an SM, each re-loading and re-dequantising every key.

### 12.2 `_stream_rb2` / `_stream_rb4` (tools/zig/dsv41_triton/dsv41_zig_triton/stream_rb.py)

A Zig-own Triton twin of `_stream`'s positions mode with RB rows a program:
- each 64-key tile is loaded and dequantised once for the RB rows, under the union of their visibility masks;
- each row computes `score_tile`'s arithmetic on it: the same q, w and dot shape, and the same 8 warps, so the same
  head-sum tree;
- each row masks its invisible columns to -inf after the dot, as `score_tile` does (a dot output element depends on its
  own column only);
- each row has its own buffer, count and threshold, offering its keys tile by tile in the same order;
- tiles past a row's visible end offer nothing.

Keys are unique, so each row's split buffer is `_stream`'s **slot for slot**, and `merge` is unchanged.

| kernel | regs | stack | CTAs / SM | rows in flight / SM |
| --- | ---: | ---: | ---: | ---: |
| `_stream` | 116 | 0 | 2 | 2 |
| `_stream_rb2` | 128 | 16 B | 2 | 4 |
| `_stream_rb4` | 176 | 0 | 1 | 4 |

Expected: the key load and dequant per pair fall by RB, and twice the rows interleave their tile chains. That is
1.5-2x on `_stream` for RB2, **+5-7 % prompt tok/s at 131K**, less at 32K, where `_stream` is a smaller share.

**Exactness:**
- *The GPU check decides.* tools/zig/dsv41_triton is compiled by triton_fill.py beside the served tree, with
  `_stream`'s exact options rule. The kernels' bit identity rests on the same per-row expression compiling to the same
  dot and reduction layout.
- *The interpreter can't settle it:* the Triton interpreter mishandles bf16 here, and the reference `_stream` itself
  misses torch's true top-k in the rows where the two disagree. It passes the ragged and multi-split cases where the
  reference is right.

**S1 on 82ac5521 (dgx-01, kit 934aa26c): FAIL, 12 compares, 1 ulp.** rb2 ran x1.49-1.51 and rb4 x1.14, but a first
differing key moved by exactly 2^32, i.e. only the score half. Cause, from the TTGIR:
- `_stream`'s dot is `#mma warpsPerCTA = [2, 4]`; the twins' dots were `[1, 8]`.
- Triton's MMAv2 warps-per-tile pick (AccelerateMatmul) switches to the chained-dots split when a dot's slice reaches
  another dot in its region. The shared key tile linked the rows' dots.
- `[1, 8]` splits the 32 heads differently over the 8 warps. The head sum then took a shorter shuffle tree: 2/3 of
  `_stream`'s shuffles a dot, giving a different order and 1 ulp.

**Fix:** every row after the first runs in its own `if r < R` region (the ragged block's guard anyway), so no dot sees
another. Each dot is back to `[2, 4]`. The key tile is still loaded and dequantised once.

**Offline check:** `tools/zig/dsv41_triton/check_stream_rb.py --src <served src>` uses the fill's venv and ptxas, no
GPU. It compiles `_stream` and both twins for sm_121 with the fill options and asserts:
- every `#mma` layout is `_stream`'s;
- the dot count, the score path's float ops (mma, shuffles, fma / add / max) and the shuffle offsets are exactly RB
  times `_stream`'s.

It passes on the fix. On 82ac5521's kernel it fails 6 of its 8 checks (layout, ops, offsets). The PTX order differs
from `_stream`'s only by scheduling between rows' independent registers.

The separate regions may cost some of the 1.5x, since the rows' chains no longer interleave freely within a tile.
S1 measures it.

**Wiring:**
- `TF_DSV41_STREAM_RB=2|4` (default 0): the emitter swaps each positions-mode `_stream` for `_stream_rb<N>`. The grid
  is ceil(rows / N) x splits, with the same buffers and merge.
- Host test (streamrb_test.zig): every other call is unchanged; the tensors, constexprs and grids are as above.
- aot-needs lists the twin's variants with the new sigs.

### 12.3 Spark step S1 (~1 min, no model) and S2

| step | what | pass |
| --- | --- | --- |
| S1 | kit fill with `TRITON_FILL_ENV=tools/zig/dsv41_m4/fill-rb.env` (and a second with TF_DSV41_STREAM_RB=4); `TF_DSV41_TEST=<tf-dsv41-test> AOT=<kit aot> bash zig/tests/cuda/deepseek_v41/stream_rb_job.sh <out>` | `bits rb2: PASS` and `bits rb4: PASS` at every (start, rows); `stream-rb: PASS`; the speedups |
| S2 | all-on + `TF_DSV41_STREAM_RB=<the faster>`, `bench_http.py --long 32768,131072` | long replies == Python's (sha unchanged), T0 20/20; prompt tok/s against the same base |

### 12.4 S1 on 9dbdd6a9: exact, slower; the prefetch twin `_stream_pf` (TF_DSV41_STREAM_RB=1)

The recheck (old inexact variants removed): `stream-rb: PASS`, 12 of 12 compares bit-exact, but the twins are now
slower than `_stream`.

| shape | rb2 | rb4 |
| --- | ---: | ---: |
| 32K x 2,048 | x0.86 | x0.96 |
| 32K x 300 | x0.79 | x0.87 |
| 131K x 2,048 | x0.85 | x0.95 |
| 131K x 300 | x0.84 | x0.88 |

Separate regions keep the `[2, 4]` layout but serialize the rows' chains. The shared key tile then saves less than
the larger program costs.

What `_stream` does not do is overlap a tile's loads with anything. Each tile is page-table gather -> key-byte gather
-> dequant -> dot -> head sum -> offer, and its TTGIR has no async copy: the gathers are not loads Triton's pipeliner
takes, so `num_stages=3` changes nothing. It also reloads the row's q (32 x 128 bf16) and w every tile.

**`_stream_pf`** (stream_rb.py, TF_DSV41_STREAM_RB=1: one row a program):
- `_stream` MODE 0 statement for statement, except that tile t + 1's page-table and key loads (bytes and scales,
  `_raw_keys`) are issued before tile t's dequant, dot, head sum and offer, carried in registers (~8 a thread);
- q and w are loaded once.

Bits: the dot is score_tile's on the same operand values, one row a program. check_stream_rb.py: pf's MMA layout is
`[2, 4]`, and its dot, float ops and shuffle offsets are `_stream`'s exactly once. In its PTX, the loop's global loads
all precede the mma and none sits between the barrier and the mma; `_stream` waits on its loads before the mma.

Step S1b:

| step | what | pass |
| --- | --- | --- |
| S1b | kit fill with `TRITON_FILL_ENV=tools/zig/dsv41_m4/fill-pf.env`; `STREAM_RB_TWINS=1 TF_DSV41_TEST=<tf-dsv41-test> AOT=<kit aot> bash zig/tests/cuda/deepseek_v41/stream_rb_job.sh <out>` | `bits pf: PASS` at every (start, rows), `stream-rb: PASS`; `_stream_pf`'s x against `_stream` (worth S2 at >= x1.15) |
| S2b (if S1b gains) | all-on + `TF_DSV41_STREAM_RB=1`, `bench_http.py --long 32768,131072` | long replies == Python's; prompt tok/s (131K: `_stream` is 12.5 % of the 4K wall, so x1.3 is ~-3 %) |

## 13. P5b's reading, and the exchange hidden at 4K (TF_DSV41_PF_OVERLAP_SITE, 2026-10-09, offline)

### 13.1 What P5b says (dgx-01 / dgx-02, 131K, 154,644 encoder rows, 2K in 95 segments, 4K in 57)

Per 1K encoder rows (rank 1, the critical rank): 450.9 -> 421.5 ms (-6.5 %). x3gm gate/up + down -28.9; every other
family within ±2.2. The "+50 % non-GEMM a row at 4K" of T2-0 came from the broken runs.

At 4K, ranked, with what bounds each:

| family (4K) | ms / 1K | bound | exact lever |
| --- | ---: | --- | --- |
| x3gm gate/up2 + down2 (+ rot 13.8) | 100.9 | issue-bound: ~2x the trellis DRAM time, ~4x its mma time | 128-member tiles at 4K: -3.3 % of x3gm (T2-0); the rest needs a GB10 ablation of the kernel |
| pfdense GEMMs (+ rot_in 15.4) | 69.5 | 55-78 TFLOP/s per shape (G17), flat per row from 2K to 4K | none cheap: every config gives the same bits and the table already picks per shape; a new kernel design |
| `_stream` (indexer, long context) | 52.9 | per-tile latency | STREAM_RB (section 12, S1 recheck pending) |
| `_fused` attention | 43.6 | ~4x its mma floor | none exact: each row has its own list and window tiles (the online softmax's tile order fixes the bits), nothing to share |
| mHC sites + `_finish_k` | 35.3 | ~1.3x its DRAM floor (~27) | — |
| **exchanges** (40 a segment + embed) | **30.7** | **CX7 at ~14 GB/s (NCCL), serialized on the compute stream** | **hide it: nothing computed changes** |
| router, combine, small | ~35 | | |

The exchange is the one family that can go to ~0 without changing a kernel: the bytes stay the same, only their
timing moves. NCCL itself moves ~14 GB/s here and 11.7 GB/s at 4 MiB on the two-Spark gate (9.6 of
E1E2-SPARK-WINDOW), against ~24 GB/s for the two CX7 functions; a faster wire would need large-message RoCE numbers
first (our RoCE path stops at 256 KiB).

`glue.embed` +4.4 ms / 1K at 4K is rank 0's alone (rank 1: 1.22 -> 1.10). It is the segment's first exchange, so on
rank 0 it holds the wait for rank 1 to finish the previous segment: skew between ranks, not work.

### 13.2 TF_DSV41_PF_OVERLAP at 4K, and the site in pieces

PF_OVERLAP (10.4) lost at 2K because k = 4 made 512-row GEMMs. At 4K, k = 2 makes 2,048-row pieces: the 2K
program's own GEMM shapes, and pfdense costs the same per row at 2K and 4K (P5b: 71.3 vs 69.5 ms / 1K). Host-gated:
4K + PF_OVERLAP stays inside the 4K workspace (pf4k_test: its streams share only `L.k4~part` / `L.k4~moe16`).

PF_OVERLAP still leaves each exchange's last piece exposed: it runs on the main stream after a join, before the next
mixing site. **TF_DSV41_PF_OVERLAP_SITE=1** (default 0; needs PF_OVERLAP 2-8):
- every exchange piece p goes on the side stream (forked after its producer piece) and records **mark p + 1**;
- the next boundary site (mhc_pf.run + `_finish_k`) runs in the same row pieces. Piece p waits for mark p + 1 only
  (`Call.wait`), and the last joins. So the site's piece p runs while exchange pieces p + 1 .. k are in flight;
- a consumer that is not a boundary site (an Engram block's post, the head, the trace) joins first, as before. A program
  ending on an exchange (no consumer) gets its last piece back on the main stream (`unpend`).

Runner: the branches' side stream holds 8 mark events; `mark` records one after a side call, `wait` makes the main
stream wait for it. `branches.shared` models the marks: after a wait, roles only the side calls up to that mark touched
are no longer shared.

**Bits.**
- mhc_pf.run and `_finish_k` are row-wise: a row's streams (in place), its gathered rows, its coefficients, its
  partials. No row reads another, and the instance is chosen by mode / fn / tap, not rows.
- Each piece is >= 256 rows (pf_rows: mhc_pf as for the whole) and starts on the 16-row grid. So every view keeps
  16-byte alignment, and `_finish_k`'s AOT variants are the whole site's (the full gate's aot-needs: SITE=1 lists
  exactly SITE=0's variants).
- The gathered buffer's rank stride comes from the view (`stride0(g)`).
- The NCCL calls stay on one stream, in the same order on both ranks.

**Host tests.**
- pfovl_test: the reader.
- pfovl_test, the emitter at 2,048 rows and at 4,096 (under 4K), k = 2 / 4:
  - every call but the sites is PF_OVERLAP's program;
  - each boundary site after an exchange is k mhc_pf.run + `_finish_k` pairs tiling the rows in order, piece p
    waiting for mark p + 1 and the last joining;
  - every exchange piece is side + fork + mark p + 1;
  - the program's dangling last exchange is main + join;
  - the streams share only the partials and the gathered buffers (at disjoint rows);
  - without the pieces, the knob changes nothing.
- pf4k_test: 4K + PF_OVERLAP (+ SITE) inside the workspace.

**Expected at 4K, k = 2** (the 2K-row piece's exchange ~1.5 ms at 14 GB/s):
- piece 1's exchange runs behind the second producer piece: wo_a + wo_b ~2.3 ms, or the shared expert + moe_sum
  ~1.9 ms;
- piece 2's exchange runs behind the site's first piece: mhc_pf + `_finish_k` over 2,048 rows, ~1.8 ms.

Up to ~-27 of the 30.7 ms / 1K, less the NCCL kernels' SMs beside the GEMMs (2-4 of 48) and the site's DRAM beside
the wire: **about -20..-25 ms / 1K, +5-6 % prompt tok/s**, the same per row at 32K and 131K.

### 13.3 4K is not running in the all-on chain: the workspace misses by ~30 MB (TF_DSV41_PF_WS_SHARE)

P5b's logs (results/zig-p5b-20261009-153148/logs):
- `c18-allon`: **0 workspace allocations and >= 64 fallbacks on both ranks.** Every prompt piece ran 2,048-row
  segments: rank 1 had 5.96 GiB MemAvailable, and the 2,552 MiB workspace would leave 3.47 against the 3.5 floor.
- The P5b 4K run allocated once (prompt 1, 89 ms). The next prompt fell back: 5.81 GiB leaves 3.31.

So the all-on set gets none of 4K's -6.5 % today.

The workspace (prod-shape test config, 2,615 MiB) holds the 4K program's own copy of every window / layer role and of
its scratch. Three of them are the same size at 4K as in the forward's plan:
- `L.ix.stream`, 256 MiB (the index budget);
- `L.ix.keys`, 128 MiB;
- `s.pf.w`, 63 MiB.

A 4K segment never runs beside another prefill program (TBO's pairs aside), each role is its own allocation or arena
offset (buffers.zig: no aliasing), and the 2K segments use these roles the same way between the same programs.

**TF_DSV41_PF_WS_SHARE=1** (default 0; with PF_4K and no TBO):
- At boot (`forward_prefill.k4Trim`), every workspace role that the forward's plan holds at least as large moves out
  of the workspace into `K4.keep`. The 4K segment's calls name those roles bare (`block_prefill.ownCallsKeep`).
- The boot price (`workspaceBytes`) applies the same trim against the forward's prefill programs. Runtime trims
  against the whole plan (decode roles too), so it can only share more than the price assumed.
- Boot log: `prefill 4K: rank R: N roles shared with the forward (TF_DSV41_PF_WS_SHARE): workspace A -> B MiB`.

Expected: ~-447 MiB, so ~2.1 GiB instead of 2.55. Rank 1 then leaves ~3.5-3.9 GiB at the MemAvailable P5b saw, which
clears TF_DSV41_PF_WS_FLOOR_GIB=3.5 but not the default hard floor (4.0). Keep the 3.5 floor for 4K runs.

Bits: the same calls on the same roles' contents. Only the addresses of three scratch roles move.

Host test (pf4k_test `shared4k`):
- the trim keeps the index stream / keys and pf.w, but not x3gm's z or q (twice the 2K size);
- it saves >= 400 MiB;
- the 4K program as run names only workspace roles (at no more than the trimmed plan's size) or forward roles (at no
  less than the 2K plan's).

The next room is a liveness-packed workspace: x3gm's z (480 MiB, MoE only) beside attention's q / o (attention
only). It is not built, since it needs every role's write-before-read checked on the GPU.

### 13.4 Spark steps (1 rep each, against the current all-on set with 4K)

Run W1 first: X1-X3 measure against 4K, which needs W1 to run at all.

| step | what | pass |
| --- | --- | --- |
| W1 (first) | all-on + `TF_DSV41_PF_WS_SHARE=1` (`TF_DSV41_PF_WS_FLOOR_GIB=3.5`), `bench_http.py --long 32768,131072` | boot log `3 roles shared ... workspace ~2552 -> ~2105 MiB`; `allocated and zeroed` once a long prompt and **no `does not fit` line** (or far fewer than c18's >= 64); long replies == Python's, T0 20/20; 32K / 131K prompt tok/s vs c18 (expect P5b's -6.5 % a row: +6-7 %) |
| X0 | kit as built (the fill's fill-4k.env already lists PF_OVERLAP's pieces; SITE adds no variant) | compile.json `"failed": []` |
| X1 | all-on + `TF_DSV41_PF_OVERLAP=2`, `bench_http.py --long 32768,131072` | long replies == Python's (sha unchanged), T0 20/20; prompt tok/s vs all-on |
| X2 | X1 + `TF_DSV41_PF_OVERLAP_SITE=1` | the same; boot log `prefill overlap: ... 2 row pieces on the side stream, every piece; the next mixing site in the same pieces`; prompt tok/s vs X1 |
| X3 (if X2 gains) | X2 with `TF_DSV41_PF_OVERLAP=4`, and with `=3` | as X2; picks k |
| X4 (optional) | X2 with `TF_DSV41_PROFILE=1` at 131K, `gap.py prefill-ab` vs P5b's 4K profile | the exchange family on the main stream's critical path; wall per 1K rows |

On a failure: X1 alone names PF_OVERLAP at 4K, X2 the site pieces.

### 13.5 Is 32K behind 131K? (bench long cells against Python prod's best of 3)

| run | 32K prompt tok/s | vs Python | 131K | vs Python |
| --- | ---: | ---: | ---: | ---: |
| Python prod | 2,436.1 | | 2,265.7 | |
| c12-allon (2K segments) | 2,507.4 | +2.9 % | 2,318.4 | +2.3 % |
| c13-allon (2K segments) | 2,511.5 | +3.1 % | 2,317.5 | +2.3 % |
| t21-4k (4K running) | 2,661.3 | +9.2 % | 2,495.2 | +10.1 % |

At 2K, 32K is ahead. At 4K, it is 0.9 points behind: 4K saved 23.1 µs a prompt token at 32K and 30.6 at 131K. Two
fixed costs a prompt explain it:
- **The workspace's allocation and zero fill.** 89-92 ms a prompt, both ranks at once, with the GPU idle while the
  GB10 maps the pages: 0.6 % of 32K's 14.3 s TTFT, 0.15 % of 131K's 61.8 s.
- **The last piece.** PREFILL_PIECES cuts 4,096-row pieces (boot log `prompts in pieces of 4096 rows`). 37,974 tokens
  end on a ~1,110-row piece, and x3gm's trellis streams once a segment whatever its rows (T ≈ W + slope x rows, W
  ~118 ms a segment). That is ~90 ms over the 4K rate at 32K (0.6 %) against ~45 ms at 131K (0.07 %). Rebalancing
  the pieces does not help: the segment count, hence the W terms, stays the same.

Together ~1.1 points, the gap seen. WS_SHARE shrinks the first (fewer pages to map). Nothing else is 32K-specific:
per row, P5b's families scale the same, and attention / `_stream` grow with context, which makes 131K's per-row cost
larger for both engines.

### 13.6 Is the GPU idle between segments or pieces?

Little. The c13 nsys trace (code x4, all-on: prefill pieces between decode rounds) shows:
- inside the prefill spans the GPU is busy 98-99.6 % of the time;
- each decode window opens with 0.13-0.18 ms idle before `positions_kernel`;
- a prefill segment's start costs 0.15-0.5 ms (one 2.1 ms outlier at a piece boundary);
- a 2K segment synchronizes its stream before staging its ids (`n <= chunk`, Python's `ids_of`), a 4K segment does
  not.

A 131K prompt's 38 pieces add ~10-20 ms of idle in ~62 s (0.03 %). The one real idle is 13.5's workspace mapping
(~90 ms a prompt).

Pipelining the next segment:
- Engram's rows are already read ahead (op 60 across pieces, c96b9662).
- The embedding is the segment's first exchange: rank 1, the critical rank, spends 1.10 ms / 1K rows on `glue.embed`
  (0.26 %). Rank 0's 7.85 is its wait for rank 1, i.e. skew. Precomputing segment N + 1's embedding on the side
  stream would save at most that 0.26 %, so it is not built.
