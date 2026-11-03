# Exact inline fractions and decimals

`runner_graph_bind.cell` now binds empty-context Python `inline_expr` arithmetic
containing true division or decimal literals to checked, precompiled graph-v7
kernels. Integer-only binding retains its existing first refusal opportunity.
The request path parses source and binds operands; it does not compile circuits,
execute host arithmetic, or embed answer bytes.

The retained inline replies use exact math semantics: `112 /29` prints `112/29`,
`8613/9` prints `957`, and `0.752 × 22.17` prints `16.67184`. Python REPL and
fenced cells do not enter this new fallback. Their existing float/Decimal/Fraction
routes remain responsible for their declared semantics.

Supported inline syntax is signed plain integer/decimal literals, parentheses,
`+ - * /`, literal nonnegative powers (`^` or `**`), and Unicode `× ÷ −`.
Decimal literals may omit digits before or after the point, provided at least
one digit remains: `.5`, `80.` and `134709035556.` retain exact decimal semantics.
Bare points and doubled points refuse; scientific notation remains unsupported.
The binder bounds source at 4096 bytes, AST at 32 nodes/depth24, intermediate
numerators/denominators at 62 decimal digits, decimal scale at 18, and exponents
at 999. It refuses unsupported syntax or widths before execution. The circuit
checks every divisor, including an inner division later multiplied by zero or
raised to power zero. Nonterminating decimal results decline with no answer;
integer-only divisions print a reduced fraction, omitting denominator one.

Decimal literals retain their source scale until arithmetic is complete. The
bank trims trailing decimal zeros through checked divmod/compare/select kernels
and uses its existing prepared formatter, including canonical integer and zero
printing. A decimal expression involving rational division must terminate within
18 places; no binary64 approximation or rounding is substituted.

## Integration and proof

Apply `runner-graph-exact-route.patch` to the route boundary, then
`runner-graph-exact-proof.patch` after the wide-powmod proof patch. The route
patch honors the typed frontend Context's `exact_inline=false` before taking
any inline shortcut; it identifies that frontend by its function pointer before
reading the Context. Other frontend types keep their existing contract. It adds
three literal `runner proof:` tests (59 to 62): retained TRACE divmod/mod coverage,
exact inline CPython goldens and changed operands through real Routed/Dispatch,
and domain/width/Python-form separation. Existing proof bodies are unchanged by
that initial patch.
The integration uses the existing `graph` backend and dispatches generic packets,
not the sequence-incompatible bare backend shortcut.

Generate independent references offline on an authorized Spark:

```sh
python3 -B tools/runner_graph_exact_goldens.py /scratch/exact-cases.jsonl
```

The generator uses CPython Fraction with source decimal lexemes and Decimal only
for exact finite rendering. The checked-in 442-case fixture includes 190 stale
answer controls and 13 domain/width/syntax refusals. Three evaluator modes must
match both canonical visible-token and complete raw trace SHA256s. The nine SYNC
residual expressions additionally match original native calc step/output
signatures; original model token boundaries were not retained, so this is a
semantic request replay, not an original token-stream replay.

Apply `runner-graph-point-proof.patch` after the initial exact proof patch. It
adds one named `runner proof:` test (62 to 63 executed tests), covering decimal
point forms and operand mutations through Routed/Dispatch on two stream IDs.
It deliberately reclassifies the old `.5*2` and `5.*2` syntax refusals as valid
expressions, replacing those negative probes with `.*2` and `5..*2`. The remaining
context, Python-form, division, width and scale controls retain their contracts.
The two retained live source forms are `80. /2` and `134709035556. /1000`; this
change extends literal binding to existing kernels rather than adding weights.

No bank rebake or GPU opcode change is needed. Continue to ship graph-v7-powmod
manifest SHA256 `26862f0a758af4e01a1d8c69d501ad49ee1657162b36d057b8a3c312fcf12512`.
Lazy boot remains unchanged structurally; qualification measured at most
235.139801 ms in the initial qualification; the decimal-point qualification
measured at most 232.294692 ms, prepared templates zero. Request compilation
remains absent. T4 evaluates
the same existing checked FFN kernels and byte formatting schedules.
