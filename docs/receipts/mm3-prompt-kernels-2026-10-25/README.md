# MiniMax-M3.1-Flash prompt kernels: contract and throughput receipt

Base: `f6505e0d34856b3e6685a5f057bae4a42d7bc9f2` (v1.1.2 with the MiniMax-M3 family patch). Branch: `mm3/prompt-kernels`.
Test date: 2026-10-24/25 (UTC). Platform: one NVIDIA GB10 (48 SMs, sm_121), driver 580.178.04, kernel 7.0.0-1019-nvidia,
nvcc 13.0.88, Zig 0.17.0. SM clock locked at 300-2100 MHz; 2080-2086 MHz under load in every timed run (the `CLOCK`
lines). Checkpoint: the EXL3 mul1 pack of MiniMax-M3.1-Flash (3.61 bpw; experts 3/3/3 bits, layers 3-17, 31 and 45-59
at 4/4/4, layer 18 at 3/3/4; attention q/o, dense MLP and shared expert 6 bits, k/v 8 bits). Checkpoint revision: not
captured.

## Scope

Prompt chunks now run the routed experts on x3gm (`exl3_gm`) in calls of up to 2,048 rows, and every dense projection
on pfdense (`exl3_pfd`); decode, verify windows and any layer x3gm cannot take keep the earlier kernels and their bits.
This receipt covers kernel correctness and single-rank kernel throughput. It is not a served or whole-model prefill
measurement: no full-model or TP2 load was run.

## Correctness (`tf-cuda-test mm3-prompt-contract`, `mm3-prompt-dense`)

Synthetic weights and inputs at TP2 rank 0's shapes (128 experts, top 4, model width 6144, expert width 1536,
swigluoai alpha 1.702, limit 7, up bias 1). Every output of the sampled rows is compared with an FP64 mirror of the
chain and its per-element worst-case bound; the old path runs on the same inputs.

| Kernel | Rows | Within FP64 bound | RMS error / old | Largest error / old | Bit identity |
| --- | --- | --- | --- | --- | --- |
| x3gm experts 3/3/3 | 1, 17, 300, 2048 | all | 0.999-1.001 | 0.94-1.025 | 25 tile pairs, v1 = v2, 1000 + 1048 and 128-row splits |
| x3gm experts 4/4/4 | same | all | 0.998-1.007 | 0.99-0.995 | same |
| x3gm experts 3/3/4 | same | all | 0.979-1.007 | 0.97-1.00 | same |
| x3gm experts, per-expert widths (v2) | same | all | 1.001-1.007 | 0.99-1.084 | 2 tile pairs, both splits |
| pfdense, 9 classes (q, k/v, o, dense, shared, 4- and 5-bit) | 1-2048 | all (max 0.0134 of it) | 0.999-1.001 | bf16 1.000, f32 up to 1.006 | a row's bytes independent of the call |

The FP64 bound is loose at these depths (expert fractions print as 0.0000), so the comparison with the old path and the
bit identities carry the evidence. A single 6144-long tensor-core chain (`experts-one-chain.log`) measured RMS error
1.25-1.30 times the decode kernel's at every width; the shipped image restarts the chain every 384 k into an fp32 total
(`X3GM_FLUSH_K`, defined only by `exl3_gm.cu`), which brought it to parity. Flush points are fixed k positions, so the
tile, v1/v2 choice and chunk split still do not change a bit.

## Throughput (`tf-cuda-test mm3-prompt-bench PACK 2048,8192,32768 5`)

Real pack tensors (rank 0's cut), median of 5 after 2 warm-ups, old and new in the same binary. Experts: old is the
decode grouped kernel in 256-row calls, new x3gm in 2,048-row calls; uniform and Zipf(0.8) routing. Dense: old is
128-row blocks of the decode linear, new pfdense. Kernel time only: no attention, routing, norms or exchanges.

| Class | 2K tok/s old -> new | 8K | 32K |
| --- | --- | --- | --- |
| Experts L20 3/3/3, uniform / Zipf | 26.4K -> 131K (4.98x) / 4.90x | 4.90x / 4.80x | 4.94x / 4.83x |
| Experts L3 4/4/4 | 4.91x / 4.82x | 5.21x / 4.73x | 5.15x / 4.93x |
| Experts L18 3/3/4 | 4.92x / 4.95x | 5.05x / 4.82x | 5.00x / 4.93x |
| Attention q, k, v, o | 127K -> 470K (3.70x) | 3.71x | 3.74x |
| Dense MLP (layers 0-2) | 61K -> 270K (4.45x) | 4.29x | 4.30x |
| Shared expert GEMMs | 313K -> 858K (2.74x) | 2.86x | 2.77x |

`bench-one-chain.log` is the same bench before the chain restart (4.3-5.2x); the restart's spill-free tiles are no
slower. Expert scratch per rank: 104 MB (256-row decode calls) and 428 MB (2,048-row x3gm calls).

## Gates

| Check | Result |
| --- | --- |
| `zig fmt --check build.zig zig` | 2 files, the same 2 as the base |
| `tools/zig/lean_check.py` | 636 problems on the base and the branch, none new by identity |
| `zig build test-cuda-host` | passed |
| `zig build test test-golden` | passed |
| `kernel-contract-limits` (sm_80, 86, 89, 120, 121) | 88 new launches recorded, 440 target checks pass; 885 inherited problems on base and branch alike |

## Not covered

Whole-model and TP2 prefill throughput, prompt/decode logit seams and teacher-forced token parity through the full
forward, and KV admission re-priced with the larger expert scratch on both ranks.
