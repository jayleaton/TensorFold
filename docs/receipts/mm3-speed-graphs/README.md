# MiniMax-M3.1-Flash: row CUDA graphs and adaptive prompt rows, single-GPU receipt

Speed items S3 (slot-independent row graphs, keyed round graphs, bounded cache) and S7 (adaptive prefill rows,
bounded block-score scratch) on the `minimax_m3` family. This receipt is one GPU, world 1, **synthetic weights**:
it establishes bit equality and a baseline before the two-rank run. It is not a checkpoint or served-model gate.

## Source and environment

- Base: `f6505e0d3` (v1.1.2 with MiaAI-Lab's `0001-minimax-m3-family.patch`, unmodified).
- Change: `88f68737b` (graphs, row plan) and `4321dbf53` (synthetic model and the three test commands).
- One NVIDIA DGX Spark, GB10/SM121, 128 GB unified memory, Ubuntu 24.04.5 LTS, kernel `7.0.0-1019-nvidia`,
  driver `580.178.04`, host `nvcc` 13.0 (V13.0.88), Zig 0.17.0, `zig build install -Dsm=121 -Doptimize=ReleaseSafe`.
- **SM clock cap 300-2100 MHz** (the node's `dgx-gpu-clock-cap`); sampled SM clock during every run below was
  2073-2086 MHz (median 2080/2086), hottest reading 89 °C, no thermal pause. Another job's ~0.3 GB process shared the
  GPU; no large job overlapped.
- Model: `mm3_synth.zig`, the shipped `config.json` widths at world 1 (hidden 6144, 64 q / 4 kv heads, 4 index heads,
  128 experts top-4 plus shared expert), seeded EXL3 mul1 codes: 3-bit experts, 4-bit attention and MLPs, 6-bit head.
  Layers 0-2 dense, later layers routed (or `moe=N`). Pages are a seeded permutation over the slots. bf16 KV unless
  `kv8`.

## Equivalence (bitwise)

| Command | Model | What is compared | Result |
| --- | --- | --- | --- |
| `mm3-graph-eq 1` | 1 dense layer | 79 calls (3 prompts, 1-row decode, 5-row verify keeping 1-3, 3-slot shared windows across the 2048 bucket edge): every logit row, tokens, 3 slots' KV bytes; eager vs graphs cold (4 captured, 68 replayed) and warm (76 replayed) | PASS |
| `mm3-graph-eq 2 moe=1` | dense + routed | same | PASS |
| `mm3-graph-eq 1 kv8` | 1 dense, fp8 KV | same | PASS |
| `mm3-rows-eq 1` | 1 dense layer | 8,525-token prompt: 512-, 1024-row and bounded-plan chunks (`2048, 2048, 1408, 1152, 1024, 845` under 1 MiB) vs 2048-row chunks: prompt row, 24 greedy tokens, KV bytes | PASS |
| `mm3-rows-eq 2 moe=1` | dense + routed | same | PASS |
| `mm3-rows-eq 2 kv8 moe=1` | dense + routed, fp8 KV | same | PASS |
| `mm3-synth-smoke` (1; `2 moe=1`; `1 kv8`; `2 kv8 moe=1`; `4`) | eager | finite logits; slot 1 (other pages) = slot 0; verify window and 2-slot shared window = decode rows | PASS |
| `test-minimax-engine` (host) | two-rank fakes | bounded plan's chunks: rank 1 runs rank 0's cuts; pages and first token equal fixed rows | PASS |

## Decode step: eager vs graph replay (ms, median of 40, stream-synchronized, no host argmax)

Full depth, 60 layers (3 dense + 57 routed), over a zeroed pool (`mm3-speed-bench 60 decode`):

| Window | ctx 2K eager / graph | ctx 8K eager / graph | ctx 32K eager / graph |
| --- | --- | --- | --- |
| 1 row | 90.68 / 85.85 (1.06x) | 90.80 / 85.67 (1.06x) | 93.40 / 87.53 (1.07x) |
| 4 rows | 156.02 / 150.16 (1.04x) | 158.74 / 151.83 (1.05x) | 160.90 / 155.85 (1.03x) |
| 16 rows | 348.59 / 344.40 (1.01x) | 360.18 / 354.91 (1.01x) | 367.17 / 362.03 (1.01x) |
| shared 4 x 16 | 835.88 / 848.90 (0.98x) | 881.65 / 842.50 (1.05x) | 866.16 / 856.04 (1.01x) |

8 layers (3 dense + 5 routed): 1 row 14.80 / 14.16 (1.04x) at 2K, 15.07 / 14.41 (1.05x) at 32K; 16 rows 1.01-1.02x;
shared 4 x 16 1.00-1.01x. 1 dense layer: 4.90 / 4.82 ms (1.02x). 2 layers (1 routed), prompts really prefilled:
6.49 / 6.32 (1.03x) at 2K, 6.48 / 6.34 (1.02x) at 32K.

## Prompt rows (whole prompt, ms, median of 3; `plan` = the bounded plan's rows under the 64 MiB default)

| Model | Tokens | 512 rows | 1024 rows | 2048 rows | plan |
| --- | --- | --- | --- | --- | --- |
| 1 dense layer | 2,048 | 115.1 | 116.5 | 121.2 | 119.3 (2048) |
| 1 dense layer | 8,192 | 528.6 | 532.8 | 532.5 | 534.3 (2048) |
| 1 dense layer | 32,768 | 2,382.7 | 2,409.0 | 2,410.9 | 2,404.8 (2048) |
| dense + routed | 2,048 | 312.4 | 316.3 | 317.3 | 318.8 (2048) |
| dense + routed | 8,192 | 1,427.8 | 1,427.0 | 1,419.7 | 1,411.5 (2048) |
| dense + routed | 32,768 | 6,439.3 | 6,380.4 | 6,339.7 | 6,337.1 (2048) |

## Reading the numbers

- Replay wins 4-7% on 1-4-row windows at depth and nothing measurable on 16-64-row windows. Unlike the Python
  campaigns this native forward is already GPU-bound per launch, so graphs remove launch gaps only. The served TP2 gain
  (exchanges between launches) is not measured here.
- Chunk rows barely change prompt time today because prompt experts run the decode grouped kernel in 256-row calls
  and dense prompts in 128-row blocks (interface decision R3). The adaptive plan's value now is memory: the block
  maxima drop from `2048 x kvh x max_pages x 4` bytes (134 MB a rank at TP2 and 1M context, 268 MB at world 1) to the
  64 MiB bound, without changing a bit. Its speed effect must be remeasured after real prompt kernels (S4).
- Synthetic expert codes are windows into one noise buffer, so cache behaviour differs from the 2-5-bit pack.

## Not yet covered (needs the TP2 run on the real pack)

- Graph capture with two ranks: `TF_MM3_GRAPHS_TP=1` is opt-in. Needs both ranks' replay = eager across RoCE small
  gathers and NCCL exchanges, the miss vote, and the fingerprint agreement (`speed` field) at boot.
- Keyed round graphs with dspark taps (layout in the key): no draft in the synthetic model.
- Served decode tok/s at C1/C2/C4, cold and warm, graphs on and off; capture memory per graph (`graph cache` log line)
  and the cache under long mixed load (held count, floor hits).
- Adaptive rows past 64K-1M context where the 64 MiB bound actually cuts chunks, on both ranks' observed minima;
  admission (`budget.zig` still prices an 8192-row score chunk, conservative) belongs with the robustness work.
- Cancellation, restore and page swaps between captured windows on the engine loop (here only the forward is driven).
