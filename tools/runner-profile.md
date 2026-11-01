# Reference runner phase profile

`runner_profile.py` profiles the CPU reference at Qwenputer commit
`bec1a0c9654ac8151b6f36ed0154a19c53141384`. It imports the pinned reference using
`runner_trace.load_upstream`, including its revision, tracked-source and import-origin checks.
It builds no firmware or circuit, loads no weights, and does not measure inference throughput.
The output is a new JSON file; an existing destination is refused.

Run from the TensorFold checkout on the authorized compute node inside the parent's
session/QUIET guard. Each invocation must remain within its 30-minute job boundary:

```sh
timeout 30m nice -n 19 ionice -c 3 python3 tools/runner_profile.py \
  --qwenputer /tmp/t1-qwenputer --output /tmp/runner-profile.json \
  --loop-elf /tmp/runner-loop.elf --cases 20
```

The optional ELF must retain function symbols, use the reference ROM memory map, print
`285\n` (the sum of squares of integers 0 through 9), and exit zero. Use the already
built `tools/runner-loop.c` artifact; this profiler never invokes a compiler.
Omit `--loop-elf` when it is unavailable; JSON then reports zero loop cases rather than a pass.

By default, requested expressions are
`123456789012345678901234567890*987654321098765432109876543210` and `2**100`.
Repeated `--expr` arguments replace those defaults. A requested-expression-only run is:

```sh
timeout 30m nice -n 19 ionice -c 3 python3 tools/runner_profile.py \
  --qwenputer /tmp/t1-qwenputer --output /tmp/runner-profile-requested.json \
  --cases 0 \
  --expr '123456789012345678901234567890*987654321098765432109876543210' \
  --expr '2**100'
```

The normal run adds 20 deterministic seed-0 cases across the upstream ten operation families.
`ten_sample_case_ids` selects the first ten complete entries in `calc_cases`, one per family
when at least ten samples are requested. All cases retain their expression, oracle output,
actual output and exit status. Upstream register exhaustion is explicitly `machine_limit`
for samples and is rejected for requested expressions. Other mismatches or faults abort.

## What is counted

Every call to `Machine.record()` contributes all returned tokens to `generated_steps`.
`GO` is the supplied start token and is excluded; `trace_tokens_including_GO` adds it back.
`instructions` equals the observed change in `Machine.steps`, checked against its final value.
An `R` refresh record contributes one generated step and zero instructions. It has its own
`refresh_R` count and token family; it never increments the opcode histogram.
`records` includes both instruction records and refresh records. `HALT` is generated work.
A ten-million-generated-token safety limit bounds each trace.

ELF32 little-endian RISC-V `.symtab` entries locate functions without fixed PCs or external
disassembly. Only executable-section `STT_FUNC` entries are used. Sized entries leave gaps
unmapped; zero-sized entries extend only to the next symbol or their section end and carry
`size_inferred: true`. Symbol aliases and ranges are included in the receipt.

`top_functions` counts leaf work. `top_pcs` adds the resolved leaf name to each hot PC.
`top_callstacks` retains caller chains, following RV32 `JAL`/`JALR` link registers x1/x5 and
returns to saved continuation PCs. These are bounded top-30 tables; their omitted tails
mean they need not sum to totals. Complete opcode and token-family histograms do sum.
The reference has no debug-frame unwinder: inlined functions have no frame, tail calls add
no return frame, and unusual hand-written stack manipulation can defeat caller recovery.
`unmapped_tokens` makes missing leaf-symbol coverage visible.

## Disjoint phase policy

Each record, including a refresh, belongs to exactly one phase. All its tokens and its
instruction count use that phase. The profiler verifies that phase generated steps sum
exactly to the case total. Each phase includes its own opcode and token-family histograms.

- `boot`: startup and calculator register initialization until the first observed `BSCAN`;
  for a standalone loop, startup until the first instruction mapped to `main`.
- `arith`: named arithmetic functions and their descendants, including `apply`, integer,
  fraction, power and register-helper work reached from those callers. For the standalone
  loop, all post-boot work before printing is arithmetic.
- `print`: named print helpers, then the complete suffix beginning with the first actual
  `BPRD` or putchar `ECALL`, including newline and exit.
- `parse`: remaining calculator work after boot, including operand loading, dispatch,
  parser-owned allocation/cleanup and any inlined print preparation before first output.

Print takes precedence over boot and arithmetic. Otherwise the nearest arithmetic context
wins over its parser caller. Boot ends only at the observed boundary. Functions such as
`alloc` inherit their caller's phase rather than being universally called arithmetic.
Compiler clone suffixes are ignored for the small semantic function-name sets.

These are reproducible policy counts, not exact source-line phase boundaries. The firmware
inlines initialization and printing into its evaluator. Without source debug locations,
first `BSCAN` and first output give observable cut points; they cannot prove which preceding
scalar instructions prepare printing or perform parser cleanup. `boundaries` records the
actual PCs and generated-token positions. The measured print suffix is a conservative
coverage of printing; the parse residual can overstate removable parser work. No unsupported
source-address interval or numeric phase estimate is invented to fill that gap.

## Digit loops, carry and RV32 division

`subrecords_by_opcode` separates `BD`, `SC`, `QS` and `QD` tokens by their real parent opcode.
`BD` is a digit-file row iteration (`BMAC`, `BMSB`, or `BLD`); `SC` is an ASCII digit scan.
`QS` and `QD` are the start and per-nibble quotient work for scalar RV32 division/remainder.
The ordinary opcode histogram exposes scalar loop/control overhead around those records.
This separates bignum digit work from scalar division, even when both occur in `divmod`.

For `BMAC` and `BMSB`, each `BD` payload already encodes the output digit and carry/borrow.
`carry_diagnostics` counts these arithmetic `BD` tokens and those with nonzero outgoing
carry/borrow. They overlap the token and phase counts. They are never added as separate
steps. `BLD` copies digits and is excluded from arithmetic carry diagnostics.
`separate_carry_steps` is null because the trace cannot independently time carry propagation.

## Optimization bounds

Every case's `optimization` derives conditional token floors from its actual trace.
The one-hot carry / digit-parallel scenario replaces the `BD+SC+QS+QD` portion of each
record with a single step. For total generated steps `T`, subrecord tokens `S`, and records
containing such tokens `K`, this gives `T-S+K`. The more optimistic zero-cost scenario is
`T-S`. Both keep every other emitted token and all refreshes unchanged. Thus the latter is
a lower bound only under that fixed-control assumption; neither bounds all possible designs.
One-hot carry alone does not eliminate digit dependencies, and a feasible parallel circuit
can require extra stages, token changes, more width, or extra memory.

The AST-direct scenario removes measured policy boot and parse tokens, `O`, giving `T-O`.
A combined scenario uses `T-O-S_retained+K_retained`; retained subrecords belong only to
arithmetic and print, avoiding double subtraction of boot/parse digit initialization/loading.
This is an optimistic thought experiment: initialization, operand state, dispatch and parser
semantics must be supplied elsewhere, and inlined print setup may lie in the removed residual.
It quantifies the measured overhead budget, not proof that an AST circuit can remove it all.

Each scenario reports its conditional floor, `T/floor` speedup ceiling and whether that
ceiling could reach 5x or 10x. A null ceiling means a zero floor, not infinite demonstrated
speed. These are generated-step comparisons only. Confirm actual reductions with an exported
candidate circuit, exact token/output replay, and matched hardware timings before claiming an
optimization. No measured speedup or 5–10x result is asserted by this tool.

`calc_sums`, `sample_sums`, `loop_sums` and `all_sums` include complete per-phase, opcode,
refresh and token-family counts. Requested and random calculator cases appear once in
`calc_sums`; `all_sums` adds the optional loop once. Sample selection is a view, not extra work.

## Verification handoff

The parent runs the commands above on the authorized node and inspects the actual JSON:
oracle/output agreement, zero faults, instruction equality, refresh accounting, phase sums,
function coverage, all ten sample families and the loop's `285\n`. Profile twice into distinct
paths and compare JSON for deterministic results. No execution receipt is supplied with this
implementation-only handoff.

Repository gates remain Zig 0.17.0 `zig fmt --check build.zig zig`,
`python3 tools/zig/lean_check.py` with base/head identity comparison via `lean_new.py`,
`zig build test test-kernel-contract test-golden fuzz-cortheon`, plain `zig build`, and
`git diff --check`. The parent owns those node-side checks, session guards and cleanup.
