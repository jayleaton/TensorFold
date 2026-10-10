# Decode router glue (2026-10-09)

Base: `b434c247`, branch `dsv41-zig-glue-route`. The experiment is **off by default**.

## Measurement

Input: `dsv41-exl3-quant/results/zig-nsys-20261008-134407/{nsys-pyn,nsys-zign}.sqlite`.
`tools/zig/dsv41_m1/glue_route_timeline.py` reproduces the measurement with read-only SQLite connections.
Run it through `spark-work/heavy.sh`, passing the Python database then the Zig database.

Use the first and last once-per-round markers in the middle 60% of each capture:
Python `_chain`, Zig `ds_stage_rows_kernel`. These bracket 71 Python and 72 Zig complete rounds.
Zig bounds: 2,706,137,776–10,372,446,864 ns; Python: 2,630,870,934–10,055,315,424 ns.
Mean round wall duration: Zig 106.477 ms, Python 104.570 ms. Duration sums overlap across streams;
they are candidate budgets, not measured elapsed savings. The markers are different phases and these
are steady averages, not identically paired rounds.

| Rank | Opportunity | Zig launches/round | GPU ms/round | GPU us/launch | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | Wide expert grouping + input rotation | 29.667 pairs | 1.207 + 0.521 = 1.729 | group 40.696; rotation 17.571 | Implemented |
| 2 | KV-store producer fusion | 59.125 | 0.396 | 6.706 | SWA producer fusion implemented separately; see [KV glue](DSV41-ZIG-KV-GLUE.md) |
| 3 | Index query RoPE into its consumer | 11.167 | 0.052 | 4.644 | Small ceiling; needs the Triton score loader's bf16 rounding preserved |
| 4 | Carry producer integration | 12 | 0.030 | 2.479 | Small ceiling |
| 5 | Row statics/acceptance and gather staging | 1 stage + 1 accept + 2 gather | 0.016 | 4.991 / 2.796 / 4.096 | Small ceiling; dependencies prevent treating them as one operation |

The wide router itself costs 2.175 ms/round (29.667 launches, 73.331 us/launch), and narrow routing
costs 1.749 ms/round (13.5 launches, 129.555 us/launch). Narrow selection/prune/kit/group are already
folded under R1. The standalone group count equals the wide router count exactly. Python has 11.831
narrow launches/round; the Zig excess is a difference in issued windows, not a redundant standalone
narrow group. A cross-round scheduling change belongs to the round/draft owners.

The eligible expert rotations use grids (168,40,2), (224,40,2), (336,40,2) for backbone slots=7,
and (80,40,2) for DSpark slots=4. There are 2,136 of these, exactly matching standalone groups.
The measurement script also pairs groups with the next expert rotation on the same CUDA stream.
All expert rotations together cost 0.700 ms/round; only the 0.521 ms wide subset is a candidate here.

`_kv_store` already performs its key RoPE; there is no separate key-RoPE launch to remove.
The two Engram `gate_copy` launches total 0.430 ms/round but include readiness polling, so this
duration cannot be priced as removable copy work. Actual DtoD copies total only 0.0033 ms/round in Zig.
Prefill's paired ring copies are absent from this steady decode window and are not a decode speed lever.

## Implemented experiment

`TF_DSV41_ROUTER_GROUP_ROT=1` replaces an unfused group plus expert `rot_in` with
`tensorfold_exl3_experts_v1.group_rot_in` for bf16, 5,120-wide rows, at most 64 rows, 9 slots and 512
table entries. Existing narrow folded groups and unsupported shapes keep their original calls.
No mHC or RMS-norm code changes.

The new `dsv41_router_glue.fatbin` is a required module in this build (27 family modules).
CTA 0 computes grouping; every other CTA assigns one independent rotation tile per warp. Neither
output feeds the other. The following expert GEMV kernel supplies the visibility boundary, with
no counter, grid barrier or inter-CTA spin.

Grouping uses a shared integer histogram, ascending expert-id prefix scan, and each pick's number
of equal earlier picks. Members stay in row/slot order, are truncated to `maxm`, and are padded with
`-1`; invalid ids do not participate. Integer atomics affect counts only, never member placement.
The router GEMV, softplus, top-k tie rule, prune, weight sum and kit rounding remain untouched.

Rotation preserves the upstream `fwht128` tree and its FMA policy. **Do not compile with
`--fmad=false`**: upstream contracts the even input products into its first butterflies. The
standalone oracle records the audited SASS dataflow, including final fp16 rounding. Ordinary and
edge cases compare all bytes, including untouched invalid-pick rows and output sentinel guards.

The target removes about 29.667 launch nodes/round, accelerates the measured 1.207 ms integer grouping
budget, and shares the following wide rotation's launch. Actual savings must come from the GPU
oracle's eager/graph timings and Spark end-to-end cells. For planning, a 1 ms elapsed cut from a
106.477 ms round means about +0.95% tok/s at unchanged acceptance (roughly +1.4 tok/s from 143.2).
This is sensitivity, **not a measured or guaranteed gain**; the complete candidate budget is
1.729 ms/round, which still must execute the integer grouping and rotation work.

## Gates and runnable handoff

Host gate: `heavy.sh zig build test-dsv41-rows test-dsv41-wide test-dsv41-mdraft test-dsv41-draft
test-dsv41-knobs -j4 --summary all`. The fusion test compares every unaffected emitted call and all
combined arguments, checks one removed call per wide MoE, and checks row-mode transformation.
Run the built `tf-dsv41-m1 check` on cap15/18/22/24-1/24-12/24-16 and `dcheck` on cap16 through
`heavy.sh`; these capture-structure gates exercise the default-off path.

GPU: `tools/zig/dsv41_m1/glue_route_gpu.sh --reps 100` takes a heavy slot, compiles for sm_121/sm_120
and runs 96 bit-compare cases against the pinned upstream CUDA code. It covers rows 17/20/24/32/48/64,
shared/pruned/invalid/duplicate picks, truncated members, partial final tile packs, finite values,
signed zeros, subnormals, infinities and NaNs, repeated launches, sentinel bytes and eager/graph timing.
Performance shapes include backbone slots=7/E=385 and DSpark slots=4/E=17.
Set `NVCC` and `HEAVY` if their paths differ. The temporary executable is deleted at exit.

`tf-dsv41-test all` additionally checks the **actual Zig binding** against the new Python
`experts_group_rot_r24` fixture (production width, 24 rows, pruned picks) and every other eligible
expert-chain fixture. Build/embed the new module with the same torch flags and default FMA
policy before running it; a prebuilt kit must add `dsv41_router_glue.fatbin`.

No GPU is available in this worker's environment. Kernel bit comparisons, Python-identical replies
with the knob enabled, and the five Spark throughput cells remain pending. `dsq-zig-loop-1` should
compare off/on at fixed prompts/settings/acceptance, require byte-identical replies, and enable this
knob only after the bit and end-to-end gates pass. Keep it off if graph timing or any cell regresses.

Local evidence: all five requested host suites passed (62 freshly executed tests plus cached draft
tests); all six capture checks and draft cap16 reported 0 mismatches. The no-fatbin production build
passed, and the complete build with the new module embedded passed 27/27 steps (including the server,
M1 checker and CUDA test runner). Both new kernels compile for sm_121/sm_120 with zero spills (fused: 25 registers, 6,528 B
static shared). The standalone oracle's compiled SASS was inspected; its GPU execution is pending.
