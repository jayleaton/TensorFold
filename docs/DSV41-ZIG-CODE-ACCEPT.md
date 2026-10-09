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
  Lone windows retain their old allocation and calibration, including zero-depth counterfactual records.
  The original `7953f4c6` also expanded exact-row lone windows; the measured-trace follow-up removes that path.

The guards leave the observed prose cell (1.678 tokens/round) and long contexts on the old allocation. They are
acceptance guards, not a content classifier. Prose throughput and long-context performance still need an A/B bench.
Even with the same graph bucket, replacing duplicate padding with different tokens can increase routed-expert
traffic, pick/keep work, and host staging. Same shape is not proof of same cost.

## Initial stationary model (superseded by measured-trace forecast below)

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

## Measured acceptance and pre-round-5 forecast

Prediction sealed through the worker task at **2026-10-09 04:16:51 UTC**, before reading any round-5 output.
Source exclusively `dsv41-exl3-quant/results/zig-acclog-20261009-040030`, baseline b434c247/r4 knobs/knob off.
Round 5's kit eff77843 contains the original 7953f4c6, not the shared-only follow-up. The point predictions for
its on arm are **code x1 89.2, code x4 147.1, prose x4 99.5 tok/s**. These predict warmed peak aggregate
decode speed, consistent with the lead's baseline table; they are not predictions of each cold repetition.

Join commits to the preceding choose by monotonic timestamp, not logger round id. Match one compatible drafted
commit (`rows == 1+k`) per slot. COPY/zero-depth commits may inherit the same logger id; quarantine those rather
than overwrite the drafted commit. Reconstruct trimmed rows from `others + sum(1+k)`, since choose.rows is
pre-trim. Filter the 384-token benchmark requests; warmups are excluded. Infer request resets from position
discontinuity and conservatively reset the last-16 history. COPY-only history is not recoverable completely.

Measured conditional code x4 acceptance by position is 0.860/0.907/0.877/0.839/0.828, versus mean logged
q 0.856/0.889/0.870/0.834/0.820, on 799/594/478/392/279 reached positions. This does not justify a global
increase in calibration. Fit position/decile bins with `q' = min(.995, q*(accepted+2)/(sum_q+2))` when >=10
samples reached the position; sparse bins retain q. Repetitions reuse deterministic prompts, so these are
not independent samples or statistical confidence intervals.

Replay the original floor depths and history guards. If an old prefix rejected, its expansion has exactly
zero gain. If all old drafts accepted, the unverified suffix remains censored: estimate its gain as the sum
of products of fitted conditional q beyond the old depth. This is a one-round counterfactual using old-policy
history; changed calibration and future round boundaries are not replayable from this trace.

Original policy: code x1 changes 41/215 drafted rounds, adds 52 rows, and predicts +12.29 tokens across 768
emitted tokens. The measured median adjacent-row cost is 4.162 ms, adding 216.41 ms over 8348.77 ms of stable
choose gaps: +1.60% gain versus +2.59% cost. Shared-only removes this negative expected tradeoff. Code x4
changes 123/216 rounds, adds 365 rows, predicts +78.11 tokens across 3072 emitted tokens, and prices two
single-window tail rows at 7.36 ms. Shared-only changes 121 rounds/363 rows, predicts +76.76 tokens, and
adds no row-bucket cost. Prose changes zero rounds. The two policies' actual raw per-drafted-slot TPR are
code1 3.563 -> 3.620/3.563, code4 3.760 -> 3.857/3.856, prose 1.680 unchanged.

The table gives **normalized emitted-token TPR / equivalent shared-round ms / predicted aggregate tok/s**.
For comparability, `T'=T*(1+gain/emitted)`, `C=streams*T/baseline_speed`, `C'=C*(1+extra_ms/stable_ms)`.
This is an occupancy-normalized cost model, not a direct GPU forward timing; raw keep is pre-emission and
therefore has a slightly different denominator. Stable choose gaps exclude discontinuous-position samples
but can still include host/logging/admission effects. Shared graph-shape cost is assumed fixed.

| Cell | Measured baseline | Round5 original 7953f4c6 | Shared-only follow-up |
|---|---|---|---|
| Code x1 | 3.563 / 39.55 / 90.09 | 3.620 / 40.57 / **89.22** | 3.563 / 39.55 / **90.09** |
| Code x4 | 3.760 / 104.82 / 143.47 | 3.855 / 104.86 / **147.07** | 3.854 / 104.82 / **147.05** |
| Prose x4 | 1.680 / 67.50 / 99.53 | 1.680 / 67.50 / **99.53** | 1.680 / 67.50 / **99.53** |

Using uncorrected logged q instead of the empirical bin fit predicts shared-only code4 **146.47**, versus
147.05 with the fit. Identifiable suffix-gain bounds for the original are 0..204 tokens, not an assurance
that the estimated 78.11 are realized. A +1% cost increase on changed shared rounds lowers shared-only code4
to **146.28**; +3% lowers it to **144.76**. Changed shared rounds account for 52.93% of stable elapsed time;
beating Python's 146.5 requires <0.72% additional cost on those rounds. Same graph shape may still increase
routed-expert and host work. This is a narrow predicted win and requires the loop's actual A/B.

An exploratory larger-bucket allocator adds no persuasive advantage. Stable all-drafted four-window gap
medians are bucket12 79.970 ms (9 samples), bucket16 86.112 ms (39), bucket24 114.528 ms (108). Crossing
12->16 has limited affordable opportunities; 16->24 has a much larger cost jump. The exploratory estimate
is approximately 147.12 tok/s versus shared-only 147.05, below fit/timing uncertainty. These conditional
medians do not establish causal bucket costs. No larger-bucket runtime policy is implemented.

No acceptance traces were provided for sampled code, the short-code mix, or long contexts. Their independent
throughput forecasts cannot be identified from these three cells. Long contexts and lone windows preserve
the old allocation; sampled replies and prose/long-context performance still require actual engine checks.

Reproduce both models using only the supplied baseline:

```bash
~/Documents/Projects/spark-work/heavy.sh python3 -B tools/zig/dsv41_perf/acclog_replay_test.py
~/Documents/Projects/spark-work/heavy.sh python3 -B tools/zig/dsv41_perf/acclog_replay.py \
  ~/Documents/Projects/spark-work/dsv41-exl3-quant/results/zig-acclog-20261009-040030
~/Documents/Projects/spark-work/heavy.sh python3 -B tools/zig/dsv41_perf/acclog_replay.py \
  ~/Documents/Projects/spark-work/dsv41-exl3-quant/results/zig-acclog-20261009-040030 --shared-only
```

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

The measured-trace shared-only follow-up passed the same six host targets (draft 38/38), including exercised
four-stream old/new/serial greedy and T0.7 token equality and unchanged lone-window draft/acceptance counters.
Four trace-join/censoring regressions passed; both original and shared-only forecasts were reconstructed from
the real baseline. Final install succeeded 29/29, and all six check captures plus cap16 dcheck again reported
0 mismatches on the rebuilt binary. Every build/test ran through heavy.sh with -j4. GPU gates remain external.

For dsq-zig-loop-1: build the codeaccept commit into the same kit and run off/on pairs with all other knobs held
fixed, especially `TF_DSV41_ROWS_CAP=64`; compare greedy **and T0.7** replies token-for-token with Python prod.
Bench code x1/x4, sampled code x4, mix4, prose4, and long-context. Reject rollout on any reply mismatch, prose/long
regression, or an x4 code result that does not beat Python. No Spark credentials or pod are available to this worker.

Next-window instrumentation must include per round/slot: position, raw conf[5], calibrated conditional q[5], old/new
k, accepted path length, real row count, actual forward partitions and graph keys, C/host/draft/slot costs,
historical lambda, baseline/candidate E/C, and actual window/pass/host ms. Label EOS/partial rounds. Phase JSON
counters are cumulative: subtract successive snapshots. `TF_DSV41_DRAFT_TIMED=1` measures prices; it does not emit
acceptance traces. The new Depth counters `expanded_rounds` / `expanded_rows` show whether the policy acted.
