# Live graph bank qualification

The request path binds operands to boot-loaded weights. It never builds FFN graphs, invokes a compiler, runs Python, or computes arithmetic results on the host. The prepared bank has 36 templates and 1,680 named plans: five binary families across seven buckets, 1,000 power schedules, and five bounded loop families across129 bounds. Division/modulo share division-stage graphs; power and loop schedules share multiply/add graphs. Packed weights occupy10,520,761 bytes.

The CPU backend uses the unchanged frozen T2 model/state/evaluator from the previous T1 native proof. Its schema hash includes those exact source bytes and Zig version. T2 ABI changes require offline rebaking; request-time fallback compilation is forbidden. See [native contract](runner-graph-native.md) and [T1 integration addendum](runner-graph-interface.md).

## Actual native evidence

Receipts and deployable bank are delivered to the shared `t1-artifacts/graph-v1` folder and node01 `~/work/imr-t1/out/graph-v1`. Verification executes the actual serialized packs through T2, audits every returned token, and compares canonical BYTE tokens plus final HALT against the original reference outputs. Internal ALU HALTs remain in a separate raw token audit hash and count against the budget.

- 6,356/10,002 original goldens route to named graph plans; every routed output/token check passes. The remaining3,646 use the circuit route by syntax/grammar policy, independent of expected answers. All six binary families, simple binary expressions and the C-loop lowering are represented. No register-limit outcomes are silently discarded by outcome-based filtering.
- All20 routed samples pass:13 named graph requests and7 original-circuit fallbacks. The seven fallback runs match the original full ISA token hashes and step counts, including the old complexity-limit behavior.
- 236 additional native cases cover signed/floor division and modulo, zero, carries/borrows, overflow, `2**100`, and sum/squares/factorial/Fibonacci/list-sum plans at bounds0,1,10,128. The request process succeeds with an empty PATH. Expected values for these additional cases come from verification-only CPython, not the serving backend.
- Five request controls detect operand mutation, unknown plan, excessive literal width, unavailable template width and named-plan override. Three boot controls reject checksum corruption, stale schema and a correctly checksummed wrong-width template substitution at their intended gates.
- Nine real-backend lifecycle checks cover two concurrent workers, admission limits, once-only delivery, budget exhaustion, cancellation, release and stream reuse. These establish isolation/reuse; cancellation may race completion and does not claim deterministic interruption halfway through an ALU step.
- 246 separate upstream IR checks exercise the actual BF16/SwiGLU/DeltaNet invariants for arithmetic templates at small and large widths, independently of the native golden projection.

Initial fresh-process full-bank load:103.37ms wall/103.32ms process CPU; a second empty-PATH proof loaded in125.34ms. This is a cold process with potentially warm filesystem pages, not a disk-cache purge or GPU measurement. Every request report records zero compiled graphs; source entrypoint inspection independently excludes the circuit JSON compiler and subprocess entrypoints.

| Workload | Original generated steps | Shared-bank steps | Reduction |
|---|---:|---:|---:|
| Requested30-digit multiply | 2,280 | 63 | 36.2× |
| `2**100` | 3,131 | 42 | 74.5× |
| C-loop shape: sum of `i*i`, `i<10` | 599 | 25 | 24.0× |

Shared plans add internal ALU HALTs compared with the prior expression-fused prototypes (62/33/5). This cost buys operand-only requests and reusable precompiled weights. Initial single-run native CPU times, excluding bank load, were1.36ms/3.51ms/5.77ms respectively; these are evidence samples, not a repeated throughput benchmark. New bindings took microseconds and contained no graph compilation.

## Integration and limits

T3 adds `.graph` to Program.kind and selects backend id `graph`, loads Bank once, and keeps it alive past all workers. T2's calc/ELF boot switch must explicitly reject the added kind. T5 calls Builder.binary/loop or lowers supported bounded integer SSA to template calls; the qualified main corpus uses named plans. Compound legacy expressions, Python floating/true-division semantics, exact-inline rationals/decimal formatting and unsupported programs retain their existing frontend/circuit policy. This change does not broaden those semantics or replace the original calc.tfimr.

T4 can consume exactly the same prepared circuits, but needs bank upload and on-device plan dispatch, input/output row binding and intermediate tensor copies. Allocate from actual template dimensions, including46,024 dimensions for mul62. Preserve f32 tensor bits, fresh state, token bus, error flags, budgets and concurrent stream isolation. CPU qualification does not qualify GPU precision, memory placement or performance. No model service, Docker image or GPU kernel was changed.
