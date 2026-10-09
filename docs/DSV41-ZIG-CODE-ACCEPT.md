# High-acceptance draft row policy

`TF_DSV41_CODE_ACCEPT=1` enables an experimental expansion-only policy; unset/0 keeps the old policy. The DSpark
pass still generates its existing five-position chain. No extra pass, new tree, or sampling rule is introduced.

The old joint allocator maximizes `sum E(k) - historical_rate * C(real_rows)`. The target's row-mode forward instead
pads the real rows to `rowtab.BUCKETS`: 1..8, 12, 16, 24, 32, 48, 64. In particular, 17 real rows already run a
24-row graph. The new shared policy spends that padding on more of the proposals already available:

- Start with the old depths as floors; never take a row away from another slot.
- Eligible slots have at least four recent rounds and mean committed tokens/round >=3 over the last 16 rounds.
- Freeze cold/low-acceptance slots. Keep the whole round unchanged if any drafting slot is beyond position 4096,
  or the round contains undrafted windows. Static depth, joint mode 2/fairness, and lone trees keep their old rules.
- Keep each eligible slot in its existing linear context bucket (`TF_DSV41_GRAPH_BUCKET`, default 2048).
- Shared rounds only expand within their **existing** row bucket, bounded by both the real forward cap
  (`TF_DSV41_ROWS_CAP`, default 16) and the allocator limit (`TF_DSV41_GRAPH_ROWS_MAX`). If the old round needs
  several forwards, leave it unchanged. The observed window used cap64; cap16 does not allow a 24-row fill.
- Sort additional calibrated survival gains and scan every feasible total. Choose the largest current
  `sum E(k) / (C(bucket) + draft_ms + additional_slot_ms)`; require >=1% improvement over the old allocation.
  A lone chain uses exact-row costs and may increase its width when the same ratio test pays. Zero-depth lone
  windows keep their counterfactual calibration path.

The guards leave the observed prose cell (1.678 tokens/round) and long contexts on the old allocation. They are
acceptance guards, not a content classifier. Prose throughput and long-context performance still need an A/B bench.
Even with the same graph bucket, replacing duplicate padding with different tokens can increase routed-expert
traffic, pick/keep work, and host staging. Same shape is not proof of same cost.

## Offline model

Sources: `dsv41-exl3-quant/results/zig-window-20261009-032628/bench-zig` and its `logs/r0-bench0.log`, with round4
comparisons in `zig-round4-20261009-030020`. These served window logs give code x1 TPR 3.589, not the background's
3.375; they are different measurements. Request totals identify only the acceptance of **verified** drafts. They
cannot reveal acceptance of unverified suffixes or how often a round has spare padding.

Reproduce the table from the actual artifacts:

```bash
~/Documents/Projects/spark-work/heavy.sh python tools/zig/dsv41_perf/code_accept_model.py \
  --results ~/Documents/Projects/spark-work/dsv41-exl3-quant/results
```

Assumptions: fit constant conditional acceptance q to observed TPR and fractional mean verified depth; assume the
unverified suffix has the same q. The equivalent round cost is `streams * TPR / peak aggregate tok/s`, including
scheduling effects, **not direct GPU round timing**. For static-depth sensitivity, price each extra row at 3.68 ms
in x1 (cached verify1/16 secant) and an assumed 2.5 ms in x4. No additional draft pass is charged.

Direct phase cross-check from round4 `bench-r4-tj1`: code x1 snapshot01 sums window+draft.pass+host+spec+commit to
14,540.32 ms over 355 physical forward calls (40.959 ms/call). It includes boot/warmup work, so it is not a clean
cell delta. Code x4 snapshot02 minus snapshot01 sums those phases to 34,341.14 ms over 337 forward calls
(101.902 ms/call). Do not add nested `candidates` or `draft.wait` again. The changing occupancy and prompt forwards
make the slot-round counter a different denominator; the normalized cell proxies below retain that scheduling cost.

Each cell below is **tokens per slot per round / ms per shared round / aggregate tok/s**. Static policies are
comparisons, not the policy shipped here. All candidate values are conditional sensitivity estimates.

| Cell | Observed Zig | Static k3 | Static k4 | Static k5 |
|---|---|---|---|---|
| Code x1 | 3.589 / 39.96 / 89.8 | 3.288 / 38.03 / 86.5 | 3.863 / 41.71 / 92.6 | 4.363 / 45.39 / 96.1 |
| Code x4 | 3.765 / 105.17 / 143.2 | 3.308 / 97.35 / 135.9 | 3.892 / 107.35 / 145.0 | 4.403 / 117.35 / 150.1 |
| Code T0.7 x4 | 3.716 / 103.93 / 143.0 | 3.333 / 97.53 / 136.7 | 3.931 / 107.53 / 146.2 | 4.457 / 117.53 / 151.7 |
| Short-code mix x4 | 3.737 / 112.31 / 133.1 | 3.313 / 105.09 / 126.1 | 3.900 / 115.09 / 135.6 | 4.414 / 125.09 / 141.2 |
| Prose x4 | 1.678 / 66.45 / 101.0 | 2.060 / 82.84 / 99.5 | 2.161 / 92.84 / 93.1 | 2.217 / 102.84 / 86.2 |

At extra-row slopes 2..3.68 ms, static k5 x4 ranges are code 143.1..153.3, sampled code 143.8..155.2, and mix
134.7..144.1 tok/s. Static k5 alone does not reliably beat Python's 146.5 / 145.5 / 141.6 in these cells.

The bucket-fill model is stronger but conditional on four eligible streams filling k5 in an already-paid 24-row
bucket. It is a ceiling, not a forecast of the guarded policy's mean:

| Cell | Bucket-fill TPR / unchanged ms / ideal tok/s | With +10% round cost: ms / tok/s | With +15%: ms / tok/s |
|---|---|---|---|
| Code x4 | 4.403 / 105.17 / 167.5 | 115.69 / 152.3 | 120.94 / 145.6 |
| Code T0.7 x4 | 4.457 / 103.93 / 171.5 | 114.32 / 155.9 | 119.52 / 149.2 |
| Short-code mix x4 | 4.414 / 112.31 / 157.2 | 123.54 / 142.9 | 129.16 / 136.7 |
| Prose x4, held | 1.678 / 66.45 / 101.0 | held | held |
| Code x1, exact rows | 4.363 / 45.39 / 96.1 | not a padding win | not a padding win |

This model has a plausible path to beating Python in code x4 even with 10% extra cost. It does not establish that
the policy achieves these values. Cold starts, short final rounds, heterogeneous confidence, 12/16-row buckets,
and context guards lower available headroom. Keep the knob default off until the loop worker measures it.

## Exactness and gates

Only the verified proposal prefix changes. Target picks remain authoritative. Draft `dspark.Params.of` and target
draw positions remain keyed by absolute token position; no sampling counter or seed is advanced by this policy.
The engine test compares old/new policies with independent serial decoding for greedy and T0.7 keyed sampling,
including rejected drafts and unequal stream lengths, and asserts that expansion actually ran.

All local builds/tests use the shared two-slot wrapper and `-j4`:

```bash
H=~/Documents/Projects/spark-work/heavy.sh
Z=~/.local/opt/zig/zig
REF=~/Documents/Projects/spark-work/dsv41-zig-ref
"$H" "$Z" build -j4 test-dsv41-draft test-dsv41-mdraft test-dsv41-rows test-dsv41-wide test-dsv41-knobs test-kv
"$H" "$Z" build -j4 -Doptimize=Debug install
for cap in cap15 cap18 cap22 cap24-1 cap24-12 cap24-16; do
  "$H" ./zig-out/bin/tf-dsv41-m1 check "$REF/$cap" || exit
done
"$H" ./zig-out/bin/tf-dsv41-m1 dcheck "$REF/cap16"
```

`test-kv` includes sched. The check/dcheck comparisons use local Python launch captures and need no GPU. They
prove emitter parity; they do not prove real-model reply parity or speed.

Local validation on 2026-10-09: all six host targets passed (draft 38/38 tests), final install 29/29 build steps
succeeded, all six check captures and cap16 dcheck reported 0 mismatches on the rebuilt binary. The model script
matched the speed-request log block with maximum TPR discrepancy 0.000459 and exited 0. No GPU reply/speed gate ran.

For dsq-zig-loop-1: build the codeaccept commit into the same kit and run off/on pairs with all other knobs held
fixed, especially `TF_DSV41_ROWS_CAP=64`; compare greedy **and T0.7** replies token-for-token with Python prod.
Bench code x1/x4, sampled code x4, mix4, prose4, and long-context. Reject rollout on any reply mismatch, prose/long
regression, or an x4 code result that does not beat Python. No Spark credentials or pod are available to this worker.

Next-window instrumentation must include per round/slot: position, raw conf[5], calibrated conditional q[5], old/new
k, accepted path length, real row count, actual forward partitions and graph keys, C/host/draft/slot costs,
historical lambda, baseline/candidate E/C, and actual window/pass/host ms. Label EOS/partial rounds. Phase JSON
counters are cumulative: subtract successive snapshots. `TF_DSV41_DRAFT_TIMED=1` measures prices; it does not emit
acceptance traces. The new Depth counters `expanded_rounds` / `expanded_rows` show whether the policy acted.
