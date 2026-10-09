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
