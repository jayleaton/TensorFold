# Optional 256-digit division bank

Append the division family to the qualified basic 256-digit bank:

```sh
python3 tools/runner_graph_extend.py --family division \
  --bank GRAPH_V5_NARROW_BANK --qwenputer QWENPUTER_CHECKOUT \
  --baseline ACCEPTED_BASELINE_TFIMR --packer RUNNER_GRAPH_PACK \
  --output GRAPH_V6_DIVISION_BANK
```

The exporter retains every old payload and manifest entry. Its resumable compiler/input identity includes the selected family. Existing 256-digit templates are allowed when their paths and plan names do not collide. Qwenputer's pinned MIT license and credit remain in the bank.

New packed templates are `meta-256`, `divstep-256`, `divfinal-256`, `emit_first-256`, and `emit_last-256`. New plans are `div-256`, `mod-256`, and `divmod-256`. They use the existing T2 evaluator and wire format, with nine logical row groups and the established packed width code `0` for public width `256`. T4 needs no new evaluator opcode.

`Builder.integer(.divmod, operands)` accepts up to 256 decimal digits per operand. Source binding supports `//`, `%`, and `divmod(a,b)`, including tuple printing and straight-line assignments. Signs follow Python floor division and modulo. Tuple arithmetic, tuple unpacking, excessive width, and overflowing intermediate operations retain refusal paths. The exact-integer domain requires a nonzero divisor. Direct named algorithms retain the existing `error\n` sentinel for invalid arithmetic; checked source-cell packets abstain. The sentinel is not a Python integer result.

The bank remains lazy: numerical weights are decoded on first family use, with no FFN compiler on a request. The 114-vector fixture covers signed 129/191/256-digit results, mutations and stale-answer controls, overflow and width refusals, named schedules, and full-width tuple emitters. Host qualification compares canonical BYTE/final-HALT tokens and complete raw traces in reused-state, fresh-state, and fixed-width modes against independent CPython references. Separate engine proofs use the routed frontend and `route.Dispatch`; ordinary graph workers alone do not execute sequences.

STATUS-T1.md identifies the qualified bank, source commit, proof patch, measured load cost, and replay counts. Historical STATED cells use REPL form with empty prior; source-derived partial inline probes are reported separately when original notebook context is unavailable.
