# Optional 256-digit modular exponentiation bank

Append this family to the qualified 256-digit division bank:

```sh
python3 tools/runner_graph_extend.py --family powmod \
  --bank GRAPH_V6_DIVISION_BANK --qwenputer QWENPUTER_CHECKOUT \
  --baseline ACCEPTED_BASELINE_TFIMR --packer RUNNER_GRAPH_PACK \
  --output GRAPH_V7_POWMOD_BANK
```

The extension retains every existing payload and encoded plan entry. It adds only `half-256`, `mulmodstep-256` and the `powmod-256` algorithm plan. Dependencies on the existing meta/divstep/divfinal/sub/emit/format kernels are checked before publication. The input bank must contain the qualified division family. Compiler input identities include the family; resume only within that same identity. Qwenputer's pinned MIT license and credit remain in the bank.

Use `Builder.integer(.powmod, &.{base, exponent, modulus})`, or source `pow(A,B,M)` / `(A**B)%M`. Operands may contain up to 256 decimal digits; exponents must be nonnegative and the modulus nonzero. Signs and modulo follow Python. The modular multiply step retains its extra carry digits before reduction; no full product is truncated to the public width. Oversized literals and overflowing earlier source expressions refuse. Named invalid arithmetic retains the legacy `error\n` sentinel, outside the exact Python result domain.

Empty-context Python REPL cells first try expression binding, then the existing bounded straight-line script binder. Thus `print(pow(3,400,5**18))` can use the same exact graph path as a fenced cell, including computed operands. The original cell-size and context restrictions remain; unknown syntax falls back to the normal frontend. This admission improvement also works for small operands and must be reported separately from the new width capacity.

The evaluator, graph packet layout and T4 GPU opcode set are unchanged. Weights remain lazy at boot, and requests only bind operands and existing schedules. Use `route.Routed` and `route.Dispatch` for engine integration; a bare graph backend does not execute frontend sequences. Retain T3's separate plan-width fix `1bd9f83fd` when integrating the 256-digit branch.

The fixture contains independent CPython references, high/low input mutations and stale-answer controls, 129/191/256-digit cases, explicit half/mulmodstep full-width carry probes, wide exponents, and width/overflow refusals. Source rows retain their real REPL/fence form. STATUS-T1.md records the qualified commit, bank checksum, raw-token parity, engine proof counts, historical coverage and lazy boot cost; support is not claimed until those checks finish.
