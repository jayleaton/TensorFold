# Exact inline fractions and decimals

`runner_graph_bind.cell` binds empty-context Python `inline_expr` mathematical
arithmetic to existing checked graph-v7 kernels after the integer binder declines.
It parses source and binds operands; it does not compile a circuit, calculate an
answer on the host, or embed answer bytes. Python REPL and fenced cells retain
their frontend contracts. This fallback does not implement binary64 semantics.

Default output is a canonical exact decimal: `1/2` prints `0.5`, `8613/9` prints
`957`, and `0.752 × 22.17` prints `16.67184`. A result without a finite decimal
representation within the output capacity refuses with no answer. For an exact
fraction, use the root-only mathematical formatting directive `fraction(expr)`:
`fraction(112/29)` prints `112/29`, `fraction(1/2)` prints `1/2`, and
`fraction(0/7)` prints `0`. It reduces numerator and denominator, makes the
denominator positive, and omits denominator one. It is not Python's
`Fraction(float)` constructor. Nested/mixed directives and extra arguments refuse.

This intentionally replaces the unreleased fallback's implicit fraction output
for integer-only true division. Bare `1/3`, `112/29`, and `7/12` now refuse;
explicit fraction requests preserve their exact mathematical values. Integer
operations accepted by the original integer binder retain their own output path.

## Grammar and bounded domain

Supported inline syntax is signed plain integer/decimal literals, parentheses,
`+ - * /`, nonnegative powers (`^` or `**`), and Unicode `× ÷ −`. Decimal literals
may omit digits before or after the point if at least one digit remains: `.5`,
`80.` and `134709035556.` are exact source rationals. A power exponent is either
an unsigned literal or a parenthesized arithmetic expression. Computed exponents
are evaluated and checked on the graph; they must be integral and in 0..999.
Scientific notation, names, Python float constructors and rounding calls decline.

The binder bounds source at 4096 bytes, AST at 32 nodes/depth24, source literal
coefficients at 62 digits and source decimal scale at 18. General checked
intermediates use existing buckets through 128 digits. Computed-power products
have a 62-digit bound and a fixed ten-round schedule; inactive high squarings bind
one before multiplication. Literal powers retain their existing bounded lowering.
Every intermediate must fit: a small final value does not excuse an overflowing
sum cross-product. Rational multiplication/division cross-cancels before products.
The circuit checks every divisor, including divisions later multiplied by zero or
raised to power zero, and divisions inside exponents.

Canonical decimal output uses at most 18 fractional places and a 128-digit
coefficient. Redundant decimal zeros and the point are removed; zero prints `0`.
For rational results, checked divmod selects the smallest exact scale from 0..18.
Failure to find one refuses; it does not approximate or round, and is not itself
proof of mathematical nontermination. Wide coefficients use the bank's existing
wide formatter; narrow prepared binary64-format callers retain their old kernel.
All arithmetic failure paths return no owned answer.

The retained rank-3 source `(3.14159/2.71828)*10^(51-21)` is nonterminating. Its
three census occurrences must therefore remain default-output refusals. The
explicit request `fraction((3.14159/2.71828)*10^(51-21))` is executable. Do not count
that different request as ownership of the original three cells or as realized
hand-work-token savings. The old <=2-digit census label counted integer AST
literals, not the decimal coefficients.

## Integration and proof

The existing `graph` backend uses Routed + Dispatch for generic packets and
sequences. A bare graph backend is not an equivalent integration test. Apply
`runner-graph-exact-route.patch` first: its typed frontend Context
`exact_inline=false` guard remains mandatory before any inline shortcut.
Apply the initial `runner-graph-exact-proof.patch`, then the point proof patch,
then `runner-graph-rational-proof.patch`. The latter intentionally replaces two
old successful-output tables with the new decimal-default oracle and appends one
named `runner proof:` test, raising the actual host proof count from 63 to 64.
Port into T3's current split proof modules while preserving all other tests and
the 600-line module rule. Refusal and successful cases use actual Dispatch;
unsupported grammar is separately checked at the graph frontend boundary.

Generate independent references offline on an authorized Spark:

```sh
python3 -B tools/runner_graph_exact_goldens.py /scratch/exact-cases.jsonl
python3 -B tools/runner_graph_rational_goldens.py /scratch/rational-cases.jsonl
```

CPython Fraction consumes original decimal lexemes, never rounded AST float
values. Decimal is used only for finite exact rendering. The first fixture has
362 cases; the new fixture has 451, including 174 stale-answer controls. Original
and mutated references are classified independently, retaining mutations that
cross into refusal even when the original already refuses. Arithmetic-domain,
output-capacity and grammar controls remain distinct. These are bounded vectors,
not a complete admission oracle for every literal-power intermediate.

Qualification requires fresh canonical visible-token and complete raw-token
hash equality in reused/adaptive, fresh/adaptive and reused/fixed modes; actual
Routed + Dispatch proofs on two streams with zero calc fallback starts; original
6356/10002 routable goldens and all 20 samples; and the project's full gates.
The historical SYNC recovery is semantic request replay: original model token
boundaries were not retained. No GPU or served-latency claim follows from CPU runs.

No bank rebake, new operation name or GPU evaluator opcode is needed. Continue
shipping graph-v7-powmod manifest SHA256
`26862f0a758af4e01a1d8c69d501ad49ee1657162b36d057b8a3c312fcf12512`.
The request path contains no compiler call; templates remain lazy at boot. T4
uses the same prepared checked kernels and output schedules.
