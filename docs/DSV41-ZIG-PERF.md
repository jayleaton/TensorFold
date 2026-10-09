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

### 7.3 The drafter off the round's critical path: the chain on the device, the ingests overlapped (2026-10-09, offline)

Source: the Spark nsys of 2026-10-08 (`zig-nsys-20261008-134407`, code 4 streams, one rank), 117 decode rounds,
`tools/zig/dsv41_perf/nsys_draft.py`; the cell's phases (`bench-zign/phases`). Medians a round, in ms:

| | ms | what |
| --- | ---: | --- |
| round | 105.6 | pick to pick |
| window | 96.8 | of which 83 RoCE exchanges 3.26 (39 us each), GPU idle 0.50 |
| ingest | 1.29 | 4 ingests (a slot each, no collective): `main_proj` 0.19-0.22 each (the same weight streamed 4 times), then 3 blocks' wkv / kv_norm / store; 0.18 of it GPU idle |
| pass | 6.40 | the DSpark pass, its head 1.56, its exchanges 0.15 |
| after the pass | 0.65 | GPU idle: the host's wait, unpack, chain (`draft.chain` 0.39 a pass), the lanes |
| before the window | 0.66 | GPU idle: the deferred keep, the row table, the Engram staging (`engram.wait` 0.41 a window) |

What of the drafter can leave the critical path:
- **The chain** (0.39 ms a 4-slot pass, plus widening the head hidden, ~0.05): Python runs it on the device inside
  its drafting graph (drafter.pass_fn: `dspark.chain`, the `_chain` Triton kernel).
- **The ingests** (1.29 ms): they need only the window's taps (the boundaries entering layers 37-39), not its picks:
  spec.py ingests every row of the window (rows past the commit lie past every later pass's context). Python runs
  them after the pick too, one slot at a time. Their taps are final at layer 39's boundary, ~3 ms before the pick
  (layer 39's attention, MoE and exchanges, the final norm, the head and its pick).
- **Not the pass**: every row of it depends on the anchor (the bonus token) and the start (the accepted count).
- **Not the Engram wait** (before the window): it waits for the rows of the next window's ids, which are the drafts,
  so no drafter work is left to run then. The window's RoCE waits before layer 39 have no drafter work ready either.

`TF_DSV41_DRAFT_CHAIN=1` (default off; needs the device path, TF_DSV41_DRAFT_GRAPHS): ds_out also lays the gathered
candidates out as Python's cand / cval [rows, W k] (2 W strided device copies in gather_cols' order, inside the pass
and its graph) and keeps the head hidden on the device; after the round's passes, Python's own `_chain` (the AOT set's
cubin, `aot-needs` lists it for the fill; prod's set P capture has it) runs over every slot of the round at once, and
only the drafts and confidences come back. The drafts are then Python's bit for bit, where the host chain agreed with
them up to fp32 summation order. The first 8 rounds also chain on the host and count the differing drafts; past a
quarter (a wiring fault, not a tie) the host chain takes over. Expected: -0.45 ms a 4-slot round, -0.1 a 1-slot round.

`TF_DSV41_DRAFT_OVERLAP=1` (default off; needs the device-started round, TF_DSV41_GREEDY_GPU): a row window records an
event after its last call that writes taps (`Runner.taps_mark`, an external event record node in its graph, so
replays record it too; the row graphs' fingerprint holds the event). At the device start (`tryLaunch`, after the
pick is enqueued) the round's ingests go to a high-priority side stream that waits on that event, and the main stream
waits for them where it ran them before (before the passes). They are the serial ingests' programs, with their scratch
roles renamed (`dspark_emit.sideRoles`: "L.kv", "L.ds.mp", ... are block.zig's too) and their own position slot; the
rings and the taps are the same. Rounds whose forward recorded no mark (no taps call) keep the serial ingests.
Expected: up to -1.2 ms a 4-slot round (less what the ingests' small launches take from layer 39's streaming), up
to -0.35 a 1-slot round.

| cell | Python | Zig round 4 | expected with both |
| --- | ---: | ---: | ---: |
| code x1 | 90.4 | 89.8 | 90.5-90.9 (+0.8-1.2 %) |
| code x4 | 146.5 | 143.2 | 144.6-145.4 (+1.0-1.5 %) |
| mix: 4 short streams | 141.6 | 133.1 | 134.4-135.1 (+1.0-1.5 %) |
| prose x4 | 98.5 | 101.0 | 102.0-102.5 (+1.0-1.5 %) |
| 131K | 53.7 | 67.0 | 67.4-67.6 (+0.6-0.9 %) |

Host gates (`test-dsv41-mdraft`): the side ingest is the serial one on its own scratch; against prod row windows
(R1, pool, taps, BRANCHES, MHC_DEFER; 5-24 rows) the mark exists, nothing after it touches taps, and the side ingest
shares no role with the window but the taps and the rope table; `emitChain` is Python's captured launch (cap16:
names, dtypes, shapes, strides, weights, constexprs); the strided copies equal the host's ungather; the staged
inputs are `sampling_params`'. GPU steps (no pod: a pod's layer prefix has no DSpark target layers, so no taps and no
mark): see the port doc's "NEXT SPARK WINDOW: the drafter off the critical path".

Not done, worth it next: one `main_proj` over every slot's rows instead of one a slot (-0.6 ms a 4-slot round, the
weight streamed once; needs the batched rows' bits proven equal to a slot's alone on the GPU, then also overlapped).

## 8. The mHC family in decode (2026-10-09, offline)

### 8.1 Measured: round 4's code x4 cell (nsys, rank 0, the middle 60 % of the capture: 7.8 s, ~73 rounds)

| launch | rows | Zig n | Zig us | Python us | ms a round (Zig) |
| --- | --- | ---: | ---: | ---: | ---: |
| Triton `_site` (split, z 4) | 17-48 | 3,895 | 49.5 | 47.9 | 2.64 |
| Triton `_finish_k` (COEF) | 17-48 | ~4,360 | 12.7 | 11.9 | 0.76 |
| mhc_cuda `boundary_kernel` (tail) | 12-16 | 2,000 | 32.0 | 26.0 | 0.88 |
| mhc_cuda `coef_kernel` (deferred stream) | 12-16 | 2,000 | 9.4 | 8.4 | (overlapped) |
| Triton `_site` (z 1: DSpark's in-place boundaries, finals, Engram's post) | 5-40 | 736 | 9-72 | | 0.35 |
| Triton `_rms` (kv_norm beside the SWA store: not mHC) | | 4,400 | 5.3 | 5.5 | 0.32 |

A 4-stream round is ~107 ms; two in three of its windows have 17-48 rows (4 streams' drafts), past mhc_cuda's 16
(`MAXR`), so their 81 sites run Python's Triton pair at 62 us each, against ~32 us for a 16-row mhc_cuda boundary.
Python runs the same pair (it is prod's path), so this is a lead over Python, not parity.

**Zig's 16-row mhc_cuda boundary is slower than Python's, and not because of L2PF.** Zig 32.2 us against Python 25.9
when the boundary overlaps L2PF's `paced_kernel`. That is ~95 % of the 16-row boundaries in both engines, at the same
point inside the paced launch (it starts ~29 us after the paced launch, median, in both). The gap holds with nothing
beside the boundary too: the window's first site (no post, no gathered partials, no concurrent kernel) is 26.0 against
15.8 us, and the Engram block's site 19.0 against 15.1. Everything the scheduling could differ in is the same:
- stream priority 0 everywhere in both;
- the paced fork just before the exchange (G14 `at=after`), 2 CTAs x 32 threads, ~87 us, the same bytes;
- the boundary right after the exchange, coef_kernel deferred, no other kernel concurrent, no long-running kernel
  spanning any boundary;
- the launch configuration (113 registers, 20,656 + R x 128 B shared memory, 100 KB carveout executed) and the spin
  (8,000 cycles);
- window buffers 256-byte aligned.

The shift is uniform (deciles +3..+8 us), and coef_kernel is ~5 % slower too: the time is inside the kernels' memory
accesses, not in their placement. Two small placement items: Zig's boundary starts 0.8-1.5 us after the exchange
kernel ends, Python's 0.2 us (an extra dependency node after `tp_roce` in the window graph: the TP fabric's, ~0.1 ms
a window). Next step on a Spark: Nsight Compute of `boundary_kernel` in both engines (L2 / DRAM sectors, sector
misses by input), or the same with `TF_DSV41_L2PF=0` in both to confirm.

### 8.2 TF_DSV41_MHC_PFDEC=<rows> (block_wide.zig, default 0 = off): the Triton pair as two CUDA launches and the norm

A mixing site of a window of at least <rows> rows (1-64), decode windows, DSpark passes (their in-place boundaries
too) and short prefill segments alike:
1. mhc_pf's `site_kernel` (G16's prefill site, a verbatim copy: the same operations as `_site`, checked against Triton
   bit for bit in G13, any row count, all 4 streams a CTA, so in place is safe);
2. `_finish_k` without COEF (the normed input alone: the final's instance, NORM_FIRST computes it from the partials
   and the collapsed row only);
3. mhc_cuda's `coef_kernel` (R1: `_finish_k`'s sums, sigmoids and Sinkhorn in its instructions, a CTA a row; the
   Zig launcher now takes up to 64 rows, the kernel has no row limit), on the deferred stream with MHC_DEFER.

No new device code. Every output has the bits of the path it replaces: mhc_pf == `_site` (G13, and the oracle case
below), coef_kernel == `_finish_k`'s coefficients (R1's replay gate, row independent), the norm is `_finish_k`'s own
code. Finals and Engram's post-only keep their launches (mhc_pf always mixes).

Expected, from 8.1 (mhc_pf at 24 rows is 80 CTAs of ~49 KB loads each: ~15-20 us estimated, not measured):
- `PFDEC=17` (the Triton windows only): ~(62 - 21) us x 80 sites = ~3.3 ms a 17-48-row window, ~2.2 ms a 4-stream
  round (two in three windows) plus ~0.2 ms in the 4-slot DSpark pass: **code x4 / mix ~+2 %**; 1-stream cells
  unchanged.
- `PFDEC=1` also replaces mhc_cuda at <= 16 rows: 40 CTAs a site there against mhc_cuda's 160; whether that beats the
  ~22-32 us boundary is the bench's to say (+0..2 % at 1 stream if it does).

### 8.3 MHC_DEFER in the DSpark pass (fixed)

A DSpark pass of at most 16 rows (1-3 slots) runs block 0's site on mhc_cuda with R1, which leaves its coefficients
on the deferred stream. The pass's next boundary (Triton `_site`, in place) and its final read exactly those
coefficients, but the emitter joined the stream only at the end of the program: the reads raced `coef_kernel`
(Python's `mhc.boundary` / `mhc.final` call `await_coefs(prev)` first). `dspark_emit.zig` now joins before both. The
launches are unchanged (the M1 gates do not see streams); only the drafts can differ, toward Python's. Round 3's
1-stream code cell had Zig at 3.375 tokens a round against Python's 3.475; the Spark bench says whether this was part
of it.

### 8.4 Gates

- Host (`test-dsv41-wide`, mhc_pfdec_test.zig): at 1-64 rows, thresholds 1 and 17, MHC_DEFER on and off, every
  mixing site becomes the three calls on the tensors the old site read and wrote, every other call is unchanged,
  each deferred launch is joined before its reader (the off path's DSpark pass included), no new role is shared
  across the streams, row mode takes the programs.
- GPU (pod: `TF_DSV41_ORACLE_ONLY='mhcdec.*' bash zig/tests/cuda/deepseek_v41/job.sh <out>`): oracle cases
  `mhcdec_*` (1-64 rows, boundary with and without tap, in place, site1, site2) run Python's Triton decode path;
  `tf-dsv41-test` runs mhc_pf + coef_kernel on the same inputs and compares the streams, the collapsed row, the tap,
  the partials and the coefficients bit for bit, and the normed input from those partials on the host.
- Spark: an A/B of the code x4 and mix cells with `TF_DSV41_MHC_PFDEC=17` (then `=1`), replies vs Python 4/4.

## 8. The round's launches: where a round idles, and TF_DSV41_ROUND_GRAPH (2026-10-09, offline)

### 8.1 Measured: the GPU idles 2 % of a round

Source: the served 4-stream code cell under nsys (`zig-nsys-20261008-134407`, rank 0, Zig at 141.5 tok/s, Python at
146.7). Rounds are cut at `pick_pack_kernel` (once a round), over the capture's middle 60 % (72 rounds). Tool:
`tools/zig/dsv41_perf/roundgap.py <sqlite>`: the busy union of every kernel, copy and memset on every stream, the
idle between them, each gap named by its bounding launches.

| per round (code x4) | Zig | Python |
| --- | ---: | ---: |
| span | 106.5 ms | 104.6 ms |
| GPU busy | 104.4 | 102.0 |
| GPU idle | **2.13** (2.0 %) | **2.54** (2.4 %) |

Zig's GPU already idles less than Python's. A round is about 98 % GPU work, so even a perfect one-graph round cannot
gain more than ~2 % in this cell. What remains is GPU time: the window's expert stream (`ld_kernel`, 65.9 ms a
round), the exchanges (`tp_roce`, 4.0 ms, part of it spin while rank 1 catches up) and the glue. That is the R1 glue
family and the prefill work (sections 4 and 7), not the host.

Zig's idle, by where it falls in the round (medians):
- **pass end -> window: 1.36 ms.** In host order:
  - 0.64 ms: the pass's candidates and head hidden downloaded (225 KB), the host chain (`draft.chain`, 0.33 ms), the
    lanes' take, then the keep;
  - 0.23 ms: the keep's 12 `carry_kernel` launches and the row table's HtoD, paced by the host. The Engram gate's arm
    then synchronizes the stream (`Gate.arm` -> `syncRunner`), so the GPU has drained before the window is
    submitted;
  - **0.48 ms: the window graph's `cuGraphLaunch` itself.** The host is inside the call for 0.48 ms, and the window's
    first kernel starts as it returns.
- pick -> pass: 1.39 ms, but mostly busy. The device accept, the drafter's eager ingest kernels and `ds_stage_rows`
  are enqueued ahead by `after_pick`, so only ~0.2 ms of small gaps remain.
- inside graphs: ~0.7 ms of gaps under 10 us (kernel-to-kernel), not addressable by graph boundaries.

Only this cell was captured. In the 1-stream code cell, a ~39.6 ms round has the same per-round host costs, but the
chain is per slot (0.16 ms), so the transition is ~1.2 ms and the idle share of the round about 3-4 %.

### 8.2 Why not one graph for the whole round

The window's rows are not known when the pass is launched. Each slot's depth `k` is chosen after the chain from its
confidences (`win.allocate` -> `Policy.choose` -> `Depth.chooseJoint`, prod's depth policy). The window graph is
keyed by its row bucket. Its Engram rows are read on the host from the drafts (the gate's worker), and the
follower's window waits for rank 0's op 17. One launch across pass and window needs all of these on the device:
- the chain on the GPU, exactly;
- the depth policy on the GPU (f64 calibration tables);
- a conditional SWITCH node over row buckets, each body a whole window;
- device-side graph launch, or the Engram worker woken from the device;
- the follower's plan carried by the device.

That is several weeks of GPU-only work with no host gate, for at most ~2 % here. Two exact host changes take most of
the idle, plus a faster host chain:

### 8.3 TF_DSV41_ROUND_GRAPH (default off; round_graph.zig)

`1` = every part, or words: `split`, `gate`.
- **`split`**: the window's graph is captured and launched as a head (its first `TF_DSV41_ROUND_GRAPH_HEAD` layers,
  default 1) and a tail. The head's launch carries ~1/43 of the nodes, so the GPU starts within tens of microseconds
  and runs the head (0.8-2.2 ms a layer) while the host is inside the tail's 0.45 ms launch.
  - The boundary is the first call of layer `head` at which neither the branches' side stream nor the mHC deferred
    stream is open (`splitAt`). The head's end joins the L2 prefetchers, which adds an ordering, never a different
    launch.
  - Keys get a `part` (1 head, 2 tail). One cache at twice `TF_DSV41_GRAPHS_MAX` holds both. The row path
    (batch.zig) and the one-slot path (forward.zig) both split.
- **`gate`**: the Engram gate's arm waits for an event recorded after the last gated window, not for the whole
  stream. The guarantee is unchanged: every window armed before has read its rows. The keep's carries, the row table
  and the window's launch then overlap instead of draining first.
- Always on (no knob), **the host chain in vector lanes** (draft/dspark.zig `markovZ`, `Markov.hidDots`):
  - the Markov bias runs 8 candidates at a time, each lane its candidate's own sum in rank order (bf16 products are
    exact in f32);
  - the confidence head's hidden dot runs 8 rows at a time, each lane its row's own sequential sum;
  - same products, same additions, same order, no contraction: the same bits (dspark.zig's test against the
    one-at-a-time code, greedy and keyed chains).
  - Dev box, prod's shape (4 slots x 5 rows, 128 candidates, rank 256, hidden 5120, 129K vocab): 1.30 -> 0.69 ms a
    round. The Spark's old chain was 0.33 ms, so ~0.17 ms expected.

Bits: no launch, operand or order changes. Graph boundaries and host waits move, and the chain is the same arithmetic.
Replies, draws and drafts are the knob-off round's.

Host gates (2026-10-09):
- `test-dsv41`: the knob's words;
- `test-dsv41-rows`: `splitAt` on the served row window (row mode, 4 slots, R1 on / off, branches + MHC_DEFER
  on / off, 1 / 5 / 16 rows, heads of 1 / 2 / 7 layers). Head + tail is the window, no side or deferred fork is open
  at the boundary, the tail's first side launch follows its own fork, and the head holds exactly its layers;
- `test-dsv41-draft`: the chain lanes bit for bit.
- plus the full gate list, with the emitter checks at 0 mismatches.

Expected per round, both ranks: about -0.45 ms (split), -0.1..-0.2 ms (gate: the keep and table no longer drained
before the launch), -0.17 ms (chain at 4 streams, -0.05 at 1). The follower's start lag shrinks with its own launch.

| cell | round now | saved | expected tok/s |
| --- | ---: | ---: | ---: |
| code x1 | ~39.6 ms | ~0.6-0.7 ms | 89.8 -> ~91.3 (+1.6 %) |
| code x4 | ~106 ms | ~0.75-0.8 ms | 143.2 -> ~144.3 (+0.7 %) |
| mix (4 short) | ~106 ms + prefill | ~0.75 ms a round | 133.1 -> ~133.9 (+0.6 %) |
| prose x4 | ~106 ms | ~0.75 ms | 101.0 -> ~101.7 (+0.7 %) |
| 131K | longer rounds | ~0.6 ms | 67.0 -> ~67.3 (+0.4 %) |

This is not the lever that beats Python in the 4-stream cells: there the gap is prefill (section 7) and GPU time.

**NEXT SPARK WINDOW, step RG** (kit built from this branch; dsq-zig-loop-1):
1. **RG-1**: `zig-bench.sh zig` with prod-zig.env as is (knob off), then again with `TF_DSV41_ROUND_GRAPH=1` on both
   ranks (label `zig-rg`), `PARTS="replies speed"`.
   - PASS: replies 16/16 vs prod for both labels;
   - every T0 and T0.7 cell's reply sha equal between the two labels (keyed draws: the same keys);
   - tokens a round equal to the third decimal;
   - decode tok/s higher with the knob on (code x1 most).
2. **RG-2**: the nsys capture of the code x4 cell, plus one code x1, with the knob on, then
   `roundgap.py nsys-zig.sqlite`.
   - Expected: "pass end -> next graphed kernel" ~0.5 ms (was 1.36), "cuGraphLaunch host time before its first
     kernel" ~0.03 ms (was 0.48), GPU idle ~1.3 ms a round (was 2.13).
   - If the gain is lower, try `TF_DSV41_ROUND_GRAPH_HEAD=2`.
3. **Rollback**: `TF_DSV41_ROUND_GRAPH=0` (unset). On a FAIL, set the failing word off: `split` alone, or `gate` alone.

`TF_DSV41_DRAFT_BATCH_PROJ=1` (default off; `check` = on with the first 8 rounds compared; needs the device-started
round): the round's ingests run `main_proj` and `main_norm` once over every slot's rows (window rows [a0, a1), the
slots' rows back to back) instead of once a slot, then each slot's 3 blocks (wkv, kv_norm, store) read its rows of
"w.ds.mx" (`emitIngestProj` + `emitIngestBlocks`; the two together are `emitIngestAt`, call for call). With the
overlap it runs on the side stream, on its own scratch. Exactness: linear.cu's bits of a row depend on that row alone
for 1-128 rows. A warp's k range is fixed by (K, N, SK, WK), `dense.plan(K, N)` takes no row count, mma keeps rows
apart, and the warp and split sums are added in a fixed order; the launcher picks the same kernel and grid at any row
count (only M and the reduction buffer's size change). `rot_in` and `_rms` are one program a row. So the reduction
order does not depend on the row count, and batching is exact by construction. `check` proves it on the GPU: each
check round also projects every slot alone ("w.chk." roles) and compares the rows byte for byte before any block
reads them; any difference turns batching off, and that round runs the per-slot ingests, so no ring ever holds a
differing row. Expected: -0.6 ms a 4-slot round (`main_proj` 0.19-0.22 a slot, the weight streamed once instead of
4 times), +0.5 % code x4 / the mix / prose x4, nothing at 1 stream; with the overlap on, the saving is partly
already hidden (the ingests then sit beside the window's tail), so the two do not add fully.

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

## 10. Sampled rounds: the nucleus statistics on one SM a row (2026-10-09, offline)

**Correction (Spark S1 / S2, same day): the bench's T 0.7 rows are not nucleus rows, and the gap below was mostly a
measurement artifact; see "What the Spark runs showed" at the end of this section.**

Sampled decode (T 0.7, top-p 0.95) trailed Python in every run while T 0 was level
(k860-base: code x1 89.2 vs 92.2, code x4 142.8 vs 147.0, prose x1 49.0 vs 50.4 tok/s).

**Acceptance is not it.** Both engines choose every verify row by the same keyed rule (no residual sampling): the
keyed Gumbel-max of `uniform(seed, position, id)` over the row's top-k / top-p / min-p cut (Python
`nucleus.choose` / `choose_rows`, Zig `serve/sampling.zig` `Picker`), drafts kept while they equal the choices, the
bonus the first differing row's choice. The draw is keyed by position, so there is no consumption order to differ.
The bench's tokens a round at T 0.7: code x1 3.632 vs 3.623, code x4 3.722 vs 3.718, prose x1 1.607 vs 1.602, prose
x4 1.635 vs 1.634; replies equal.

**The round costs more.** From the acceptance logs of allon1, allon2b, r5-off and r5-on (choose -> last commit of
the forward, the same committed rows, T 0.7 minus T 0, medians weighted by count): code x1 +0.37 ms a forward,
code x4 +2.0 ms (every row count 11..24 within +0.7..+3.8), prose x1 +0.24. Python's sampled round costs what its
greedy round does (tok/s / tokens a round: x1 39.30 vs 39.37 ms, x4 101.2 vs 102.4).

Cause: `sampling.cu`'s `stats_kernel` (`nucleus.row_stats`: each row's max and sum of `exp(v / T - max / T)` in
float64 over the rank's 64,640 columns) ran one CTA a row: a segment of n rows used n of GB10's 48 SMs, each doing
~30 float64 operations a column (an IEEE double divide and exp) at FP64's 1/64 rate, about 0.4 ms a call whatever
n. Zig calls the sampler once a sampled segment (one call at x1, four at x4: +0.4 / +1.6 ms of the measured +0.37 /
+2.0); torch's `row_stats` spreads the same work over the whole GPU once a forward.

Fix (always on, tokens unchanged): `dsv41_sampling_stats_max / _sum / _fin`, grid (32 chunks of 2,048 columns,
rows): the chunk maxima in fp32 (the f64 of the fp32 max is the f64 max: the conversion is exact and monotone),
each chunk's sum of the same terms `exp(f64(v) / T - m / T)` with the row's exact max, the row's sum in chunk order.
Only the summation order changes, and it was not torch's before either: `nucleus.keep_count`'s GUARD (1e-9) is what
makes the token independent of the last bits of Z, and a guard hit takes the whole row (the same token). The max is
bit-equal. `TF_DSV41_SAMP_STATS=row` keeps the old kernel; a fatbin without the split symbols keeps it too (kits
built before it still load). `tf-dsv41-samp` checks both kernels against the host (max bit-equal, sum within 1e-12)
and prints each one's time a call at 1 / 5 / 16 rows.

Left for after the Spark numbers:
- One sampler call a forward: built, `TF_DSV41_SAMP_BATCH=1` (default off; see below).
- `TF_DSV41_SPEC_NUCLEUS=1`: the speculative pass for nucleus rounds. Zig's host holds the exact choices when
  `speculate` runs, so the knob needs no `device_z` and is exact; T 0 gets that overlap today (T 0.7's `draft.pass`
  +0.15 ms and `engram.wait` +0.1 ms a round are its absence). Python leaves it off; pod step `spec-samp` covers it.

### 8.5 TF_DSV41_ARENA: the device buffers from a few large chunks (default off)

The suspect behind 8.1's gap: Zig gives each buffer its own `cuMemAlloc` (`cuda/memory.zig`), so the small ones the
glue kernels read every launch (each layer's mHC `fn16` at 983 KB, norm weights, coefficients, router I/O, tables)
each sit on pages of their own among the weights, while PyTorch's caching allocator carves them from 2 MiB and
20 MiB+ segments. If GB10's unified memory pays for that spread (smaller pages, TLB misses on a cold launch), every
small kernel pays it, not only mHC.

`cuda/arena.zig` sits under `DeviceBuffer.alloc` / `free` when installed (`memory.setHook`):
- requests up to 4 MiB are bumped, 512-byte aligned, from shared 64 MiB chunks (a freed range is reused for the same
  rounded size); larger ones get a chunk of their own, rounded to the granularity;
- backend `vmm` (`TF_DSV41_ARENA=1`): one reserved virtual range (2x the device's memory), physical chunks of the
  device's recommended granularity (at least 2 MiB) mapped in order; `plain` (`TF_DSV41_ARENA=plain`, and the
  fallback without VMM): one `cuMemAlloc` a chunk;
- installed by `dev_arena.zig` right after the TP session opens (the fabric keeps its buffers) and before the
  weights load, in the server (`model.zig`) and the M2b gate, on every rank.

Only addresses change. Host gate: `dev_arena_test.zig` (every program kind emitted with and without the knob is the
same call list, argument for argument) and the arena's unit tests (packing, alignment, reuse, large chunks,
fallback, the hook).

Spark micro-check (seconds, one GPU, prod stopped or on rank 1 between benches): `tf-dsv41-test mhc-alloc [windows]`
times prod's 16-row boundary and its coef_kernel over 80 per-layer site buffers with a 512 MiB fill between launches,
with the buffers per `cuMemAlloc`, in the plain arena and in the VMM arena. If per-buffer is ~26 us and the arenas
~16 us, the theory holds and the knob goes into the next A/B; if all three match, the next suspects are the buffers'
relative addresses (L2 slice aliasing between x, xout, the gathered rows and the partials) and the instruction side
(the boundary's code cold after ~1 ms of other kernels in both engines, so less likely).

### 8.6 The probe's answers, and the writer hunt (round 4's nsys)

`mhc-alloc` (dgx-01): per-buffer 37.6, plain arena 36.6, VMM 38.6 us: allocation and page size are not it.
`mhc-probe` (dgx-01, f86954d3), boundary / coef us: overhead 1.7; cold-write 36.6 / 10.2; cold-read 22.3 / 8.2;
engine 19.5 / 8.2; warm 16.4 / 8.0; aligned2m, stagger, cross all 19.5 / 8.2. Dirty L2 lines cost the boundary
~14 us; slice aliasing and the cross-stream dependency cost nothing.

No Zig-only writer before a boundary in the traces:
- the kernels between two consecutive boundaries, by name and grid, over ~5,000 intervals a capture, are the same
  in both engines but for the exchange (Zig `tp_roce`, Python `gather_fast_kernel`) and Python's torch elementwise
  launches;
- memsets: 442 (Zig) / 492 (Python) in the whole capture, 64 B each near a window start; memcpys: none of Zig's within
  150 us before a boundary (Python's DtoD copies put 6.5 MB there);
- `tp_roce` is a port of `gather_fast_kernel` (G14): the same pinned host slots, the same release / acquire order.

Two exchange facts that do differ:
- **grid:** both engines size it one block per 16 KiB rounded to a power of two, capped at Python's
  GLM53_TF_ROCE_BLOCKS (8) and Zig's `max_blocks` (16): a 16-row window's 160 KiB exchange runs on 16 blocks in Zig
  and 8 in Python. The 16-row boundary is +6.3 us in Zig, the 12-row one (8 blocks in both) +4.4 us, so the grid is
  at most part of it. `TF_TP_ROCE_BLOCKS` / `GLM53_TF_ROCE_BLOCKS` now set the cap (power of two, 1-16, default 16
  = today; 8 = Python's).
- **after a short exchange the boundary is slower:** Zig 32.4 / 31.0 / 31.1 us after exchanges of < 15 / 15-30 /
  30-60 us, 26.2 / 25.1 after 60-200 / > 200 us; Python 24-26 whatever the exchange. Whatever slows Zig's boundary
  drains while an exchange waits: consistent with memory traffic the exchange leaves in flight.

Probe variants `sysmem` / `devmem` test the exchange's host traffic directly: 160 KiB written to pinned host memory and
320 KiB read back from it by kernels just before the boundary, against the same on device memory.

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

**One sampler call a forward (`TF_DSV41_SAMP_BATCH=1`, default off).** Per segment, Zig ran top-k, two all-gathers,
two blocking downloads and an op_sample (~0.2 ms each call, an estimate, x4 only); Python runs one `cand_gather` and
one statistics exchange a forward. With the knob, `Batch.targetWindow` hands a forward's nucleus segments (two or
more) to `GpuSampler.chooseSegments` (plan-link op 13: [op, n, (row0, rows, T, top_k, top_p, min_p) a segment]):
top-k over rows [first row0, last end), each segment's statistics at its own T, one gather of each, one download,
then each row picked as before; whole-row fallbacks after every pick. Other sampled segments (top_k rows, which may
start the speculative pass from the device choice) keep one call each.

**One sampler call a forward (`TF_DSV41_SAMP_BATCH=1`, default off).** Per segment, Zig runs top-k, an all-gather,
the device choice (`vs_choose`, the speculative round's start) and a blocking copy back, then the host picks, before
the next segment's call; Python runs one `cand_gather` a forward. With the knob, `Batch.targetWindow` hands a
forward's keyed segments (two or more; top_k rows and nucleus rows) to `GpuSampler.chooseSegments` (plan-link op 13:
[op, n, (row0, rows, T, top_k, top_p, min_p, seed, pos0) a segment], pos0 -1 = no device choice): one top-k at the
segments' largest count over rows [first row0, last end), the nucleus segments' statistics each at its own T, one
gather of each, one copy back; each top_k segment's `vs_choose` over the shared gather (its rows drawn at
pos0 - row offset + r, its own count) and the forward's `after_sample` with its rows, so the row window's
speculative round (`SlotPass.tryLaunch`, which waits for every armed slot's choice) starts after one call instead of
four; then each row picked, cut to its own count. Whole-row segments keep one call each.

Exactness: no reduction changes order (topk_keys a CTA a row, pack a row, the statistics a row from its own row and
T). A row whose count is below the shared k gets more candidates from each rank than its own call gathers; its merge
sorts every rank's candidates by (value, lower id) and keeps its count: the global top `count`, keys unique, each of
them within its own rank's top `count`, so the kept candidates and their order are the one-segment call's. Host test
(`test-dsv41-sampling`): nucleus segments (different T / top_p / min_p / seed, a row in no segment, two flat rows
taking the whole-row fallback) and mixed segments (the served top_k 20 / top_p 0.95, top_k 5 with min_p, top_k 300,
a nucleus segment, a segment whose first position is below its row offset), one and two ranks: batched == one call a
segment (tokens, whole-row count), every token == the keyed rule over its whole row, and each top_k segment's
`vs_choose` mirror over the shared gather == its host pick (a mutation dropping the row-offset shift fails it).

**What the Spark runs showed (S1 / S2 on 2e777c10).** T 0.7 hashes equal, tokens a round unchanged, and almost no
gain (code x1 89.2 -> 89.7, x4 142.8 -> 142.9). Two reasons:

- The bench's T 0.7 rows never take the nucleus path. `bench_http.py` sends `temperature`, `top_p` and `seed`, not
  `top_k`, and both servers fill it from their defaults: Python `cuda/server.py` `{"temperature": 1.0, "top_k": 20,
  "top_p": 0.95}`, Zig `server/main.zig` the same under the Spark wire (the serve log: "sampling: temperature 1.0,
  top_k 20, top_p 0.95"). Rows are top_k 20 rows: 21 candidates a rank, no statistics; the split kernel (and
  `SAMP_BATCH`'s nucleus-only first version, and `SPEC_NUCLEUS`) never ran. The split kernel stays right for clients
  that send top_k 0. Boot now logs the statistics path, and rank 0 logs the first keyed call of each path (whole
  rows / nucleus / top_k, per call or batched, device choice on or off).
- The 1-rep loop compares Zig's rep 0 with Python's best of 3. `zig-bench.sh` keeps each cell's best rep;
  `bench_http.py`'s one warm-up is code T 0, 1 stream, 32 tokens, so a cell's first rep pays the first sampled /
  4-stream use (graph captures, scratch) on both engines. Same rep on both sides:

  | cell | Python rep 0 | Zig rep 0 | Python best | Zig best (3-rep runs) |
  |---|---:|---:|---:|---:|
  | code T0.7 x1 | 89.1 / 88.9 | 89.1 / 89.4 / 90.0 | 92.2 / 92.2 | 92.4 / 92.5 / 92.0 |
  | prose T0.7 x1 | 47.4 / 47.6 | 49.3 / 49.3 | 50.4 / 50.4 | 50.8 / 50.5 |
  | code T0.7 x4 | 141.8 / 141.8 | 135.0 / 135.4 / 138.7 | 147.0 / 145.5 | 142.7 / 142.5 / 144.7 |
  | code T0 x4 | 140.4 / 142.5 | 141.4 / 140.3 / 141.5 | 145.4 / 146.5 | 143.0 / 142.5 / 143.8 |

  (Python: onepass and round 4 bench-prod; Zig: r5-off, r5-on, onepass base.) At x1 and in prose, T 0.7 is level
  rep for rep. The real gap is code x4: about -1.3 % at T 0 and -2 % at T 0.7 best for best, and Zig's first x4 T 0.7
  rep is 3-5 % colder than Python's. The sampled-only part at x4 (~0.7 % a round) is the four serialized sampler calls
  before the speculative round can start: what `SAMP_BATCH` now removes.

### 8.7 It is not the boundary: Zig's latency-bound kernels are all slower

Spark (ea98bb27): `sysmem` 20.5 = `devmem` 20.5 us (the exchange's host traffic is not it); `TF_TP_ROCE_BLOCKS=8`
exact, within +-0.6 % at x4, -2.4 % long (stays 16). No streamer writes with L2 hints (x3ld, dense3, L2PF: none), so
each layer's ~90 MB of weights flushes L2 in both engines, and the kernels writing before a boundary are the same.

Round 4's traces, every kernel kind's median Zig / Python (rows, grid alike), not only after an exchange:
`rot_in` (1 x 16) 4.8x, `rot_in` (12 x 24) 1.7x, `_rope` 1.4-1.6x, `seg_rot_in` 1.1-1.5x, `_kv_store` 1.1-1.5x,
gate/up epilogue 1.36x, `topk` 1.25x, the boundary 1.25x away from any exchange (19.3 vs 15.5 us) and 1.23x after one,
`coef_kernel` 1.11x, the 320-CTA `lanes_linear` 1.11x; the DRAM-bound streamers (`ld_kernel`) 1.0x. Small,
latency-bound kernels are slower whatever runs before them: an SM clock or memory-latency difference of the whole
GPU, not a writer. On GB10 the CPU and the GPU share one power budget. Zig's host sits in `cuEventSynchronize` 11.3 s
of the 13 s capture (the driver's `auto` schedule spins a core), with the RoCE proxy spinning another.

- `mhc-probe` `spin-1`, `spin-<cores/2>`, `spin-<cores>`: the engine pattern while host threads busy-spin.
- `TF_CUDA_SCHED=auto|spin|yield|blocking` (unset: the driver's `auto`, today): the primary context's schedule,
  set before the context starts (model.zig, m2b.zig). Host scheduling only.

## 11. Copy drafts at 4 streams: Python's priced lookup plan (TF_DSV41_LOOKUP_PLAN=1, 2026-10-09, offline)

The 2026-10-08 T0 x4 nsys pair (141.5 vs 145.55 tok/s, the same tokens a round): forwards of equal rows cost the
same in both engines (24-row bucket 85.5 / 85.4 ms, 36-grid 114.2 / 113.0 ms), but Zig ran 7 forwards of 32-43 rows
(141-181 ms) that Python never runs. The 10-09 acceptance logs (kits with the device chain) show what they are: three
DSpark windows plus a 12-16-row copy window on the fourth slot (code x4: 14-37 such rounds a run, 15.5 rows, 4.9
tokens kept, against ~3.7 for that slot's DSpark window; +27-67 ms a forward).

Cause: Zig's copies are the lanes core's (`lanes.SuffixLookup` through `windows.copyProposal`): a match of
`enter_match` (8) tokens copies up to 15 tokens, unpriced. Python (`deepseek_v41/cuda/lookup.py` `Planner`) prices
every lookup: the match length's band gives p (prior 0.45 / 0.7 / 0.8 / 0.9 at 0 / 8 / 16 / 32 tokens, worth 2
trials, decayed 0.85 a lookup round), `best_k` picks k >= 1 maximizing E(k) - rate x verify[k] against the
request's own rate (its last 8 DSpark / serial rounds' tokens a modelled ms), and a lookup round runs only with a
positive surplus (a 16+ band skipped 8 times is probed once).

Port: `draft/lookup.zig` (the plan on glm5_next's `SuffixIndex`: n 3, 64 candidates, extend 64, the eos cut, the
image virtual-id cut) as a priced lanes proposer: `Proposer.VTable.priced` makes `copyProposal` take its drafts as
they are, and `round` gives it every non-forced round's rows and kept drafts (Python's `observe`; a stream not asked
counts as serial). `LaneHost.proposers` makes one a request; the DSV4.1 serve engine installs it with
TF_DSV41_LOOKUP_PLAN=1 (default off), pricing with the measured cost table (`policy.depth.costs`). Python's own knobs
apply: TF_DSV41_LOOKUP (1), TF_DSV41_LOOKUP_MIN (4), TF_DSV41_MAX_ROWS (16). DSpark's skip stays Zig's.

Gate: `lookup_test.zig` replays `fixtures/lookup-golden.json` (tools/lookup_golden.py on prod 8474f31's lookup.py:
24 requests, 3,275 rounds, 465 lookup rounds; min match 3 / 4 / 6, max rows 8 / 12 / 16, the default and a measured
table, eos tokens in the drafts) round for round, kind and drafts, then the same rounds through the proposer from a
stream's context (a second ask in a round reads the decision back). Drafts never change a reply.

Spark A/B: code / prose x4 T0 with TF_DSV41_LOOKUP_PLAN=1 against the same kit without it; pass = equal T0 hashes,
no forward past 24 rows in the acceptance log when Python has none, and code x4 not slower.

Also here: the phases report carries "engram" counters (row reader hits / waited / misses / reads / bytes / wait_ms,
gate rounds / syncs / warm / failed; cumulative, a cell's are two snapshots' difference), for the cold rep 0
question (section 10 of the sampled work).

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

## 12. Engram reads ahead of a 4-slot window: Python's commit and draft-end warms (TF_DSV41_ENGRAM_WARM=1, 2026-10-09)

The cold rep (code T0.7 x4, no warm-up, kit e63bf019): rep 0 140.0 tok/s, rep 1 on the same prompts 148.9, rep 2 on
new salted prompts 142.9. Row-graph host time 73 / 0 / 13 ms; Engram reads found in flight ("waited") 7,896 / 112 /
5,811 for 490 / 53 / 473 ms of the gate worker's wait. The window phase grew 646 ms in rep 0 over rep 1 (the host's
engram.wait phase only 55): the GPU's gate kernel spins before layers 1 and 14 until the round job's rows land, and
layer 1 leaves the reads about one layer of cover.

Python (decode.py, 8474f31) starts them earlier, on both ranks: at every commit (rounds.py `_commit` ->
`prefetch_pending`: the next window's pending row, a gate warm before the DSpark pass) and at draft end
(`decode.draft`: a gate warm of the pending row and every drafted position, before depth and grammar cut them).
Zig's only early read was R3, the pending row at `op_ds_propose`, the one-slot pass's message; with TF_DSV41_SLOTS=4
and TF_DSV41_SLOT_DRAFTS=1 (the slot pass, ops 40-42) every read started at the window's arm.

Port: `iface.Target.warm` (forwarded by the timed and grammar wrappers), called by the lanes' `draft` at Python's two
points (each stream's pending row before the pass or its speculation, then each drafted slot's pending row and all
its block's drafts). `GpuTarget.warm` with TF_DSV41_ENGRAM_WARM=1 (default off): row mode -> `Batch.warmEngram`,
plan-link op 56 [op, n, (slot, start, rows, ids...)] so every rank warms its own columns, then a gate warm job a rank;
one slot -> `prefetchEngram` on this rank. A warm's lookback is the slot's committed tail plus, for a draft before
its window's keep, the pending window's rows its start implies are kept (`engram_host.lookbackAt`, R3's rule;
`engram_host.tailAfter` is the keep's own update). Reads only: a window's rows are its own round job's bytes whatever
was warmed, a wrong guess is one wasted read.

Gate: `engram_host` test "an Engram warm's lookback at a draft before its window's keep is the window's tail after it"
(500 random slot states, pending or kept), the emitter checks and the host tests.

Spark: the same 3-rep cold diagnostic with TF_DSV41_ENGRAM_WARM=1 (rep 0 cold, rep 1 same prompts, rep 2 salted):
pass = equal hashes, "waited" / wait_ms and rep 0's tok/s against the run above (gate_warm in the phases' "engram"
counts the warm jobs, which must be > 0).

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

## 13. The Engram read path: what a cold round waits for (2026-10-09)

Section 12's warms were exact but did not pay (kit 72e41538: rep 0 140.9 vs 140.0, waited 5,414 vs 7,896; the all-on
bench lost 2-6 % on four cells): the extra reads compete with the ones the GPU waits for. The read path first.

Configuration, knob by knob, Zig = Python (8474f31 engram_gate.py / engram_native.py / prefetch.py): a pool of
TF_DSV41_ENGRAM_QD = 64 `pread` threads (Docker's seccomp refuses io_uring in both), O_DIRECT shards (buffered + a
page drop where refused), 512 B sector extents merged across holes <= 4 KiB, an LRU of 65,536 records a layer
(TF_DSV41_ENGRAM_GATE_CACHE / the Zig `cap`, 17 MB), one gate worker issuing a round's two layers and waiting.

A cold round (zig-cold-20261009, code T0.7 x4, rank 0): ~249 extents of ~770 B, 3.9 ms of the gate worker's wait;
rep 1 (same prompts): 1 extent, 0.43 ms. Lookups found cached: rep 0 79 %, rep 2 (new salted prompts) 88 %, so the
cache already holds the shared rows; what remains is new content.

`tf-dsv41-m1 engram-probe DIR RANK WORLD` (no GPU): the shards' size, O_DIRECT reads at depth 1 / 8 / 32 / 64
(reads a second, p50 / p99 / max latency), and a round's batch (1 / 32 / 128 / 256 fresh rows a layer, both layers
issued then waited, the gate worker's order) through each reader: the threads, native AIO (TF_DSV41_ENGRAM_AIO=1:
one io_submit a batch, io_getevents on the waiting thread, no threads to wake), io_uring where allowed; then the
batches with every core but two spinning. On the dev box (in-memory files, ~1 us a read) every reader takes 1.3 ms
for 2 x 256 rows at idle clocks and 0.22 ms with the cores busy: the host side is CPU frequency / idle exits, not
the I/O call. GB10's numbers decide the fix.

TF_DSV41_ENGRAM_AIO=1 (default off): the AIO reader in serving (falls back to the threads, logged, when io_setup is
refused); TF_DSV41_ENGRAM_URING=0 forces a non-io_uring reader for A/Bs. Same extents into the same buffers:
engram_rows_test's golden reads (Python ShardRows' SHA-256s) pass in every mode, AIO included.

**GB10 numbers (results/zig-eprobe-20261009-130948, dgx-01).** Shards 2 x 192M rows, 47.21 GiB each, O_DIRECT. Device:
depth 64 serves 270k (image) - 390k (host) reads a second at p50 125-200 us; depth 1 3-4k a second at 200-300 us. A
2 x 256-row batch, p50: image threads 2.70 ms, image AIO 1.19 ms (in prod's containers io_uring is refused); host
threads 2.00, AIO 1.61, io_uring 1.66. AIO halves the batch in the image and is close to the device's throughput
(~512 reads at ~390k a second ~1.3 ms). The probe's "spinning cores" lines of that run were cache hits (each batch
size re-read its idle run's rows: one seed a size; 0.05 ms for 2 x 128 rows is under one device read), so no
wake-up conclusion follows from them; the probe now draws fresh rows each measurement and prints how many rows of each
line were read cold (`issued`).

TF_DSV41_ENGRAM_POLL=1 (with AIO, default off): a wait polls io_getevents with a zero timeout instead of sleeping in
the kernel, up to 20 ms, then sleeps: the CPU spins only while a caller waits for reads in flight. Exact (the same
completions); engram_rows_test's golden reads pass with it (mode aio_poll).

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
