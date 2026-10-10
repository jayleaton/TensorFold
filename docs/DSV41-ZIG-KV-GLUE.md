# KV producer glue (2026-10-09)

Base: `9ff34663`, branch `dsv41-zig-glue-route`. `TF_DSV41_KV_NORM_STORE=1` is a separate,
**default-off** experiment. It does not require the router fusion knob.

## Measured target

The read-only `tools/zig/dsv41_m1/kv_glue_timeline.py` reproduces these measurements from
`dsv41-exl3-quant/results/zig-nsys-20261008-134407/{nsys-pyn,nsys-zign}.sqlite`. Run it through
`spark-work/heavy.sh`, passing the Python database then the Zig database. It uses the same complete
middle-60% round windows as the router measurement: 72 Zig rounds, 71 Python rounds.

| Eligible producer | Pairs/round | Store ms/round | RMS ms/round | Pair ms/round | Store us/launch | RMS us/launch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Backbone SWA | 40 | 0.331280444 | 0.209877333 | 0.541157778 | 8.282011 | 5.246933 |
| DSpark ingest | 11.958333 | 0.026160000 | 0.042056889 | 0.068216889 | 2.187596 | 3.516952 |
| DSpark attention | 3.166667 | 0.005157333 | 0.009643111 | 0.014800444 | 1.628632 | 3.045193 |
| Total | **55.125** | **0.362597778** | **0.261577333** | **0.624175111** | | |

Python has 55 pairs/round and 0.570882014 summed ms/round. The four compressed stores per round
follow `_pool_norm`, not the KV RMS, and are excluded (0.033871556 store ms/round in Zig).
The original 0.396 ms store-family budget includes these excluded stores and omits RMS work.

Labels are source-grounded capture inferences, not recorded layer IDs or data pointers. Pairing
uses process/device/context/stream identity, matching row grids and kernel metadata, bounded
lookback, and stops at earlier stores or round markers. DSpark attention has q projection/cast/RoPE
between its KV norm and store; the last matching RMS is the KV producer. All stores were classified,
with no unmatched stores. Duration sums overlap across streams and are **not elapsed savings**.

## Fusion

One 128-thread CTA per row performs the KV-only RMS, rounds through fp32 to bf16, then performs
key RoPE and writes the FP8 SWA record. It replaces two launches with one for 512-wide bf16 KV
inputs, fp32 KV norm weights, and 1–64 rows. Backbone uses it only with R1. DSpark ingest/pass
use their existing norms without requiring R1. Larger rows and unsupported shapes retain the old
calls. Compressed pooling, query/main/final norms, prefill and all mHC kernels are unchanged.
The new `dsv41_kv_glue.fatbin` is required by this build; there are now 28 family modules.

The exactness contract comes from captured Triton cubins, not a substituted reduction:

- `_rms` at K512 uses four contiguous values `4*t+i` per thread. Sum order is
  `((a0+a1)+a2)+a3`, xor butterfly 16/8/4/2/1, then `(w0+w2)+(w1+w3)`.
  This differs from the existing folded `_rms_row` helper's layout. Squares, accumulation,
  variance and weighted output use explicitly rounded fp64 operations. The rsqrt approximation
  retains its audited refinement. Normalization rounds fp64 → fp32 → bf16 → fp32 before RoPE.
- RoPE retains `fma(even,c,-mul(odd,s))` and `fma(even,s,mul(odd,c))`. NoPE pairs also execute
  the original zero/one arithmetic, including zero times odd and explicit FMA instructions.
- Quantization retains seven 64-element maxima, the 1e-4 floor, exponent threshold,
  explicit FP8 nearest-even bit rounding and small-value add/subtract, then satfinite E4M3 conversion.
  RoPE dimensions round to bf16 and scale byte 7 is zero.
- Positions use the original signed int32 interpretation and signed power-of-two remainder.
  Row mode binds each row's int64 position and write slot. Negative slots skip cache writes;
  optional debug normalized output still includes those rows, like the unfused norm.

Production passes `norm_out=null`. The kernel's optional debug output enables intermediate-byte
comparisons without changing the production ABI. Compiled for sm_120/sm_121 with the production
macro flags and default FMA policy: 26 registers, 1,056 shared bytes, zero stack/spills. SASS was
inspected for the reduction tree, conversions, FMA operands and absence of runtime division.

## Gates and Spark handoff

All builds/tests use `heavy.sh`, with Zig `-j4`:

- Host rows/wide/mdraft/draft/knobs passed: 28/28 steps, 65 executed tests plus cached draft.
  Tests compare every unaffected call and fused argument, narrow/wide backbone emission,
  DSpark ingest and multislot passes, and row-table positions/write slots/stacked rings.
- `tf-dsv41-m1 check` cap15/18/22/24-1/24-12/24-16 and `dcheck` cap16 each reported
  **0 mismatches**. These capture-structure checks exercise the default-off path.
- The no-fatbin build passed 29/29 steps; the complete build with both fusion modules embedded
  passed 27/27 steps, including the server, M1 checker and actual Zig GPU test runner.
  GPU execution remains pending on Sparks.

Run the production byte oracle in the actual Spark Python environment:

```bash
tools/zig/dsv41_m1/kv_glue_gpu.sh --python-tree "$PYTHON_TREE" --reps 200
```

The wrapper acquires a heavy slot, compiles both GPU architectures and deletes its temporary
fatbin at exit. The oracle directly calls Python's `_rms` and `_kv_store` and compares every byte
of normalized rows, caches, guards and padding over 1,530 cases. It covers scalar/row-table/all-padding
modes, nonconsecutive positions, ring wrap, strided pointers, signed zeros, subnormals, large finite
values, NaNs/infinities in inputs/weights, reduction stress and bf16/FP8 ties. Repeated/debug/null-debug
launches must agree. Eager and captured-graph timings require two baseline nodes versus one fused node.

The normal Python `oracle.py` also writes `kv_norm_store_scalar` and `kv_norm_store_rows` fixtures.
`tf-dsv41-test all` (or `kvglue <fixture>`) checks both the actual Zig kernel launcher and its replay
argument binding against those bytes. Build/embed the new module first. Python fixture generation,
Zig binding execution, the 1,530 byte cases, enabled-knob reply equality and throughput are GPU gates
that have **not run on this worker**. The knob stays off until all pass and the Spark cells improve.
Test this knob alone and together with router grouping/rotation at unchanged prompts and acceptance.
The target removes about 55.125 launch nodes/round; elapsed savings and tok/s gain remain unmeasured.
