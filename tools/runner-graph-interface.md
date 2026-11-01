## T1 addendum: boot-loaded graph backend (2026-11-01)

This user-authorized addendum extends the earlier `circuit`-only contract with a
second backend id, `graph`. Earlier defaults stay unchanged. T1 owns the graph
bank, arithmetic templates, binding runtime and this addendum; T3 owns engine
selection and T5 owns frontend lowering. Acceptance receipts are in STATUS-T1.md.
The implementation follows T1 commit478c186ba; its final SHA is recorded there.

### Shared API and constructor

T3 adds `graph` to `runner_api.Program.kind`: `enum { calc, elf, graph }`.
For `.graph`, `Program.bytes` is the version-1 binding packet below. It is not
Python, ELF, graph IR or an executable filename. The Backend/Result/RunEnd/VTable
contracts, token budget, asynchronous poll and stream lifetime rules are unchanged.

```zig
const graph = @import("runner_graph.zig");
var bank = try graph.Bank.load(gpa, bank_path); // once, before admitting requests
const backend = try graph.backend(gpa, &bank, workers);
// shutdown: graph.destroy(backend); bank.deinit();
const bytes = try graph.Builder.named(gpa, "mul-62", &.{ lhs, rhs });
const program: api.Program = .{ .kind = .graph, .bytes = bytes };
```

`runner_graph.zig` exports `id = "graph"`, Bank, Machine, Builder, Packet, Op and
Plan. Its worker pool reuses T2's unchanged State/eval arithmetic. No second
engine scheduler, native bignum implementation, Python process or compiler runs
in a request. Bank lifetime exceeds all backend jobs. Every job owns fresh value
storage and T2 mutable state; prepared weights and plan constants are shared.
Start/poll/cancel/release stay nonblocking on the engine host thread; destroy
joins workers only at shutdown. Allocations use the caller's thread-safe allocator.

T3 accepts `TF_DSV41_RUNNER_BACKEND=circuit|graph` (default `circuit`) and adds
`TF_DSV41_RUNNER_GRAPH_BANK` for the bank manifest path. A selected graph backend
requires a successfully boot-loaded bank; never compile missing templates on a
request. A graph backend given `.calc` or `.elf` abstains. T3 may explicitly
route those kinds to the existing circuit backend when configured; it must not
reinterpret their bytes as graph packets. T5 emits `.graph` only for supported
bindings, otherwise keeps its existing calc/ELF/abstain policy. Enabling graph
routing must not change the default-off runner or Python/exact-inline semantics.

Adding the enum member also requires T2's `Machine.boot` switch to reject `.graph`
explicitly (for example `error.CircuitProgramKind`); graph jobs enter State.init
and bind slots instead of ISA boot. T1 does not edit T2/T3/T5-owned files. Its
native proof stages this one shared enum addition beside frozen T2 dependencies.

### T5 binding packets and reusable plans

The common fast path is named-plan lookup plus signed decimal literal binding:

```json
{"version":1,"plan":"mul-62","values":["1234567890","-9876543210"]}
```

The offline catalog includes add/sub/mul/div/mod digit buckets1,2,4,8,16,32,62;
powers `pow-<exponent>-62` for literal exponents0..999; and
`loop-<shape>-<bound>-62` for `sum`, `squares`, `factorial`, `fib`, `list_sum`,
bounds0..128. Power/loop schedules share ALU weights; the catalog does not contain
one expression/result graph per user or benchmark case. A power request binds
only its base; its exponent selects the prepared control schedule. Non-list loop
plans have zero request values; list_sum binds exactly bound values. Induction
indices, zero and one are offline plan data, never host-computed answers.

For add/sub, choose the smallest bucket covering max(operand digit widths)+1;
for mul, cover their sum; for div/mod cover their maximum. Clamp required
precision at62; the circuit computes overflow and prints `error\n`. Selection
uses syntax/shape bounds, never the arithmetic answer. Native binding refuses
nonzero high digit lanes outside the chosen width. Negative zero is normalized
inside the circuit. Division/modulo implement signed Python floor semantics.

T5's bounded assignment/tuple-update, literal/list-comprehension sum, factorial
and Fibonacci loop lowering can also describe a topological template-call plan:

```json
{"version":1,"values":["12","-3"],
 "ops":[{"op":"add","width":4,"a":0,"b":1}],"result":2}
```

This is weight-free frontend lowering, not FFN compilation. Named requests have
no request ops and result0; overriding a named plan is refused. Generic SSA is
optional for supported bounded bodies; legacy golden routing conservatively
keeps compound syntax on circuit unless separately qualified. Both paths use
only preloaded arithmetic weights. `Builder.encode` serializes Packet directly.
The normative field definitions are `runner_graph_packet.zig`; packet<=1MiB,
ops<=4096, total values<=8192, template arity<=16. Forward refs, unknown plans,
missing templates, invalid decimal literals, bad widths and slot counts refuse.

References index request values, offline constants, then call outputs. A call
has op/width/a/b, or args for larger arity, and outputs. Ordinary ALU outputs1;
divstep/divfinal output2. The special `bind` op copies declared tensor bit ranges
and optional sign/error flags. It performs no arithmetic, text conversion or
error combination. Division plans repeat divstep over fixed digit lanes and
assemble quotient lanes by these copies, then invoke divfinal. Powers use fixed
square/multiply calls. The final emit graph generates the result's text.

### Bank format, CPU reuse and T4 requirements

`runner_graph_pack.zig` is OFFLINE tooling: TFIMR001 graph-template bridge ->
TFGRPH01 prepared pack. `runner_graph_export.py` builds all reusable weights,
invokes the packer, and ships bank.json, `.tfgp` companions, provenance and the
full Qwenputer MIT credit/license. Source IR is unnecessary at boot or runtime.
Pack names resolve relative to bank.json, independent of the server's cwd.
Boot restores prepared batches/forms/state templates without JSON graph parsing
or IR compilation. Descriptor/plan JSON contains no arithmetic weights.

The 80-byte header contains magic8, LEu64 payload length, schema SHA25632 and
payload SHA25632. Runtime fields are encoded little-endian with exact float bits;
scalar slices may use run-length encoding. The schema fingerprint includes exact
Zig version and frozen T2 model/state/evaluator source. Changing the prepared
runtime requires an offline rebake, not a request-time compatibility compiler.
Packs are trusted deployment artifacts; checksum checks corruption. They carry
no ISA ROM/DROM/RAM image. T2's graph evaluation primitives are unchanged.

A Value is two128-column binary32 tensor rows: A_REF=16777216 at reference
columns0/1 when materialized; up to31 decimal digits per row in columns2..125,
four LSD-first bits per digit, set bit A_BIT=128. First-row126/127 are sign/error.
Readonly inputs `in<id>.<offset>` query on GO; writable `out<id>.<offset>` latch
results on GO. Offsets are0/31. Binding preserves destination reference ballast;
intermediate transfers copy digit/sign/error tensor lanes without host decoding.
ALU calls emit HALT after writing results; the final emitter outputs BYTEs and
HALT. Steps count every eval.step, including internal ALU HALTs. Raw trace and
visible BYTE+final-HALT projection are distinguished in the proof receipts.

T4 should upload the same prepared T2 sparse forms, coefficients, ordering and
snapshot rows once, retain per-job state, and replay the same validated plan.
Tensor bind/extract/copy operations should remain on-device between invocations;
no CPU readback of intermediate answers is required or permitted for arithmetic.
Reuse the existing one-hot DeltaNet and exact FFN kernels. Preserve read/write
order, sign/error bits, token budgets/cancel and independent concurrent jobs.
Allocate for each bank entry's actual dims/batches/tables, not fixed2679-dim P2
scratch: reusable 62-digit multiply has46024 dims and59 IR layers. CPU/GPU trace,
output, steps and snapshots must agree under the existing precision contract.
T1's CPU boot/arith proofs do not qualify CUDA performance or memory placement.

### T1 integration clarification: frontend selector and pack identity

`graph.Builder.binary(gpa, .mul, lhs, rhs)` performs the documented bucket
selection and returns a named binding packet. Its family enum also accepts
`.add`, `.sub`, `.div`, `.mod`, `.pow`. `graph.Builder.loop(gpa, .squares, 10,
&.{})` selects a bounded loop plan; `.list_sum` requires bound literal operands.
These helpers validate literals/control bounds and serialize packets only.
A power exponent is control data, not an operand evaluated on the host.
The generic SSA packet is optional; the qualified main route uses named plans.

Prepared payloads contain the operation name and width beside the Circuit;
boot verifies both against the manifest, in addition to schema and payload
SHA-256. A valid add-1 pack cannot be substituted for add-4. The emitter's
`output.cursor` is internal state and is excluded from input/output slot mapping.
