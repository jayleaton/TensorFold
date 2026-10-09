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

## 10. Sampled rounds: the nucleus statistics on one SM a row (2026-10-09, offline)

Sampled decode (T 0.7, top-p 0.95, top_k 0: nucleus rows) trailed Python in every run while T 0 was level
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
- One sampler call a forward (Python's one `cand_gather` + one statistics exchange): Zig still runs top-k, two
  all-gathers and two blocking downloads a sampled segment (~0.2 ms each, an estimate, x4 only).
- `TF_DSV41_SPEC_NUCLEUS=1`: the speculative pass for nucleus rounds. Zig's host holds the exact choices when
  `speculate` runs, so the knob needs no `device_z` and is exact; T 0 gets that overlap today (T 0.7's `draft.pass`
  +0.15 ms and `engram.wait` +0.1 ms a round are its absence). Python leaves it off; pod step `spec-samp` covers it.
