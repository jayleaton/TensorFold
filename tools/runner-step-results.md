# Step-reduction investigation

Baseline: Qwenputer `bec1a0c9654ac8151b6f36ed0154a19c53141384`, original
T1 circuits from TensorFold `35d1ffd83`. Every step below excludes supplied GO
and includes generated HALT and refresh R. CPU work uses node01, nice 19,
ionice c3, at most 30 minutes per session, paused during shared QUIET windows.

## Measured baseline phases

| Workload | Boot | Parse | Arithmetic | Print/exit | Total |
|---|---:|---:|---:|---:|---:|
| 30-digit multiply | 487 | 212 | 1464 | 117 | 2280 |
| `2**100` | 487 | 102 | 2454 | 88 | 3131 |
| C squares loop, original O0 ELF | 5 | 0 | 159 | 435 | 599 |
| First 20 seed-0 calc cases combined | 9740 | 8182 | 47669 | 1776 | 67367 |

These are disjoint observable/caller-based policy counts, not source-line
profiling: boot ends at BSCAN/main; the first output starts an inclusive print
suffix; named helpers and their callers refine attribution. Inlining hides some
print preparation in parse/arithmetic. Unmapped sled instructions are retained,
not dropped. `runner-profile.md` defines the policy and its limitations.

Multiply has 837 arithmetic BD digit iterations, another 442 boot BD and 60
parse BD plus 60 ASCII SC scans. Power has 451 arithmetic BD, 442 boot BD,
4 parse BD and 4 SC. BMAC/BMSB BD tokens already compute digit and carry/borrow;
there is no independently measurable carry step to add. Across all phases,
multiply has 1278 BMAC/BMSB BD tokens, 627 with nonzero carry/borrow; power has 892 and 344.
These diagnostics overlap the phase totals. The C loop has no BD/SC/QS/QD work;
its arithmetic includes ten executed MULs and its print path uses scalar
multiply-high division by ten. All twenty cases across ten operation families retain their
original output/error behavior; thirty-six refresh tokens occur across the
whole requested/sample/loop profile.

## Options and quantified limits

| Change | Multiply steps / reduction | Power steps / reduction | Loop steps / reduction |
|---|---:|---:|---:|
| Skip measured boot using a valid resume snapshot | 1793 / 1.27x | 2644 / 1.18x | 594 / 1.01x |
| One step per existing digit/quotient subrecord, same control | 920 / 2.48x | 2288 / 1.37x | 599 / 1.00x |
| Also remove policy boot+parse, supply required state elsewhere | 771 / 2.96x | 2137 / 1.47x | 594 / 1.01x |
| Ideal BYTE+HALT print suffix, everything else unchanged | 2225 / 1.02x | 3076 / 1.02x | 169 / 3.54x |
| Compile the arithmetic schedule and emit only BYTE+HALT | 62 / 36.77x | 33 / 94.88x | 5 / 119.8x |

The first four rows are conditional accounting scenarios, not implemented
speedups. A resume snapshot needs correct counters, PC, bignum/RAM state and
entry semantics; it cannot simply be substituted for the initial GO snapshot.
The digit scenario models a native multi-digit MAC or parallel carry/digit
unit, but preserves all scalar control tokens. Even its optimistic bounds miss
5x for the two requested arithmetic workloads. Boot and printing alone cannot
solve their latency. Removing the interpreter's control schedule is essential.

Two concrete approaches are investigated. Direct AST RV32 firmware removes
runtime parsing and register allocation, uses base 10,000, unrolls convolution
columns and performs carry normalization once per column. Thirty-digit operands
need 64 limb products / sixteen columns, versus up to 900 decimal digit
products. Every scalar RV32 instruction and its value tokens still cost steps;
wide values and reciprocal division may erase the arithmetic-work saving.
`runner-fast-firmware.md` documents this candidate and its unsigned subset.

The selected compiled arithmetic graph uses parallel FFN digit products and
carry normalization, plus snapshot operand heads and a byte-selection cursor.
It is a general unsigned arithmetic frontend and bounded sum-of-squares example,
not host result evaluation. Exact arithmetic runs inside the graph. It removes
boot, parser, ISA dispatch, digit-record traces and RV32 print control. It needs
an explicit experimental entry point and frontend, not an ELF boot.
`runner-arithmetic.md` specifies the container, input slots and verification.

One-time compilation/loading and cached execution must be reported separately.
These graphs can be cached by operator schedule and operand widths; input bits
can then be retargeted without changing arithmetic weights. This prototype
builds per request. No CUDA or stock-model layer-packing latency is inferred
from step counts, and no training or baseline calc replacement is performed.

## Qualified prototypes

Both candidates were exported as new files. Original calc/multiply/loop assets
remain byte-identical to T1's original deliverables.

| Prototype | Multiply steps | Power steps | C loop steps |
|---|---:|---:|---:|
| Direct AST, base-10,000 RV32 / O2 C | 1261 (1.81x) | 1830 (1.71x) | 171 (3.50x) |
| Compiled FFNs with DeltaNet result latch | 62 (36.77x) | 33 (94.88x) | 5 (119.8x) |

The wider-limb RV32 phase counts (boot/parse/arithmetic/print) are
5/0/581/675 for multiply and 5/0/1469/356 for power. Fewer arithmetic steps
are offset by expensive scalar conversion/printing. The O2 loop has
5/0/122/44. This retains the existing loader and complete RV32 trace contract,
but misses the requested 5–10x reduction.

The selected FFN circuit latches the computed digits and overflow flag on GO.
Operand queries are then zero while printing. This cuts active arithmetic work
in the sparse CPU evaluator, although the evaluator still visits its FFN
batches. All arithmetic and output selection remain circuit operations.

Matched native measurements use unmodified T2 f32 loader/evaluator sources,
five runs per main circuit, node01 nice19, no GPU, and SHA-256 of every LEu32
token including GO. CPU time excludes QUIET pauses; recorded wall medians agree.
Run timing excludes JSON loading and State.init. Baselines measured about
6154 steps/s for multiply and 6092 for power, consistent with the supplied rate.

| Workload | Original CPU run | Compiled CPU run | Run reduction | New JSON load | Graph build CPU |
|---|---:|---:|---:|---:|---:|
| Multiply | 370.477 ms | 2.888 ms | 128.29x | 45.841 ms | 312.407 ms |
| Power | 513.959 ms | 12.541 ms | 40.98x | 384.090 ms | 2578.273 ms |
| Loop | 127.588 ms | 0.0314 ms | 4066x | 6.146 ms | 142.244 ms |

These are loaded-circuit execution gains, not cold end-to-end serving gains.
Rebuilding the power graph per request costs 2.58 seconds before serialization
and loading, so it loses to the already-loaded original calc circuit. Even
loading its prebuilt JSON per request takes about 384 ms. A reusable schedule
cache, preloading and explicit operand-slot loading are necessary for the warm
benefit at Result:. Retargeting a snapshot with identical weights is verified;
the production cache/frontend/entry-point integration is not implemented here.
The loop speedup compares compiled bounded arithmetic with a general ISA
interpreter circuit; it is not an arbitrary-ELF execution claim.

| Graph | Dims | FFN units | Heads | IR layers | Native batches | File bytes |
|---|---:|---:|---:|---:|---:|---:|
| Original P2 | 2679 | 42014 | 99 | 11 | 116 | 13989019 |
| Compiled multiply | 7575 | 25060 | 5 | 148 | 150 | 2458433 |
| Compiled power | 63091 | 212149 | 4 | 284 | 286 | 19033526 |
| Compiled loop | 595 | 1255 | 12 | 55 | 57 | 235815 |

Extra depth/dimensions are a material cost. In particular, 284 IR layers cannot
be treated as an 11-layer P2 step for a GPU latency estimate, or claimed to fit
unchanged stock model layers. No CUDA timing or model integration is qualified.
Carry lookahead, common-subexpression sharing and compiled templates are the
next useful changes. With an additional two-byte output family, byte emission
could theoretically drop to 32/17/3 steps respectively; that requires a new
output contract and is not part of these artifacts.

## Evidence and artifact identity

All three `.tfac` files pass serialized restoration, every next-byte/HALT token,
and upstream BF16/SwiGLU/DeltaNet exactness checks. Native free-run token hashes
also match. The 20-case seed-0 sample exercises five supported compiled
expressions and fifteen original-calc fallbacks, each matched natively in full.
Sample 9's upstream `error (too complex: split it)` outcome is explicit and
preserved. Seven zero/carry/power/62-digit boundary cases pass the IR checks;
unsupported syntax, corrupted payload and operand-bit mutation fail their
intended checks. Retargeting that operand with identical weights passes the
new reference result. Wide-limb and O2-loop circuits match their new ELF
reference machines' complete internal token traces natively.

Artifacts and receipts live under the shared track directory:
`t1-artifacts/steps-v1/` and `t1-evidence/steps-{report,native-report}.json`.
The report includes complete tiny output-token traces, SHA-256, graph sizes,
phase counters, all native repeats and preparation costs. Main asset identities:

| File | SHA-256 |
|---|---|
| multiply.tfac | f01778a97d1da4a847747e6f7206a9cd0086d3f863cf253afb5e07f22579f503 |
| power.tfac | 8d47354ba4d187502959a74237cc71226f9fe91863bda0391b44bbc356282dd3 |
| loop.tfac | e3bfc8b52b6f8580d6a3c48d8f8a075de0290bc31aa771f1b910d0c6b8f445c4 |

The separate `.bench.tfpc` files contain the same graph and operand/result
snapshot schema, with disposable unused ROM metadata for T2's existing loader.
Their entry is raw State.init + GO. They must not boot via calc/ELF Machine.boot.
The standard-format P2 firmware experiments are `wide-multiply.tfpc`,
`wide-power.tfpc` and `loop-o2.tfpc`; their matching ELFs and generated C are
included. Full MIT credit/license is retained. Baseline calc is not replaced.
