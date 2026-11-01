# Reusable signed decimal graph API

`runner_graph_ir.Builder(width, core, ir, np)` builds standalone arithmetic IR on the P2 token bus.
`Value(digits, sign, error)` contains little-endian decimal Exprs and boolean negative/error Exprs.
`Builder.input(index)` reads a width-sized Value; `finish(outputs, emit=False)` latches Values.
`signed_add(a,b)`, `signed_sub(a,b)`, `signed_mul(a,b)` and `select(flag,a,b)` return Values.
The unsigned primitives `add(a,b)` and `sub(a,b)` return `(digits, carry_or_borrow)` with
`max(len(a),len(b))` digits, including 63-digit intermediates. `mul(a,b,width)` returns
`(digits, overflow)`. `ge(z,t,lo,hi)`, `eq(z,t,lo,hi)`, `choose(flag,a,b)`, `any(flags)`
and `nz(digits)` are available to division hooks. `one`, `c` and `node(terms,name)` are public.

`build_template(root, op, width)` returns `(document, metadata)`. Supported buckets are
1, 2, 4, 8, 16, 32 and 62. Arithmetic operations add/sub/mul take inputs 0 and 1.
Select takes candidate Values 0 and 1, and condition Value 2 (nonzero selects input 0).
Emit takes input 0 and emits canonical signed decimal or `error\n` followed by HALT.
Division operations delegate to `runner_graph_div.build_divstep(builder, inputs)` and
`build_divfinal(builder, inputs)` when available. Divstep takes inputs 0/1/2 as remainder,
divisor and next digit; divfinal takes inputs 0/1/2/3 as quotient, remainder, dividend and
divisor. Hooks return a Value or list of Values.

Metadata `input_slots` and `output_slots` are lists of records with logical `index`, `head`,
`width`, `offset`, row 0, `cols` and `sign_col`/`error_col` (126/127 in the first group only).
Heads are named `in{index}.{offset}` and `out{index}.{offset}`. Each group holds at most
31 digits as four little-endian bits in columns 2..125. Snapshot rows begin with A_REF
ballast in columns 0/1 and all-zero digits; request values are bound by raw tensor writes.
Inputs are readonly, queried on GO only, with tiny 2^-90. Outputs latch on GO; non-emit
graphs predict HALT immediately, with the result already retained in output state.

Templates are compiled offline and loaded at boot. Request execution binds values and follows
static invocation plans; it never compiles FFNs or runs arithmetic in Python/native bignums.
Errors propagate through every operation, overflow is flagged, and negative zero normalizes.
Arithmetic uses BF16-exact small integer gates, parallel carry/borrow prefix scans, decimal
digit products at most 81, carry-save row compression with sums at most 27, and one final
carry-lookahead addition. Only these new graphs are stably
scheduled by dependency level; the existing ISA graph and T2 evaluator are unchanged.

## Task ledger

Standards inspected: AGENTS.md, CONTRIBUTING.md, docs/atc/SKILL.md, docs/recipes/SKILL.md,
.zig-version (0.17.0), build.zig, build.zig.zon, .github/workflows/no-gpu.yml and zig-fmt.yml.
No additional tools/AGENTS.md or tools/CLAUDE.md was found. Parent owns coordination and
verification; this helper edits only runner_graph_ir.py and this document, with no executions.
Required parent checks: `zig fmt --check build.zig zig` plus new tool Zig; lean_check.py
and lean_new.py against 427 baseline finding identities; `zig build test test-kernel-contract
test-golden fuzz-cortheon -j2 --summary all`; plain `zig build -j2`; focused native proofs.
All execution belongs on the parent's isolated verification host under its bounded quiet CPU policy.
The document uses the existing `tensorfold.runner.arithmetic` format and the existing
`runner_arithmetic.write(document, path)` HEADER/checksum container writer.

Source inspection covered carry/borrow propagation, product/compressor bounds, signed zero,
output group columns, division hook signatures, and the six-byte error cursor for width 1.
Runtime checks remain pending with the parent; no compile, interpreter, test or build ran here.
