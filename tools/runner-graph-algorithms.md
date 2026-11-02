# Prepared integer algorithms

The graph bank extends `graph.Builder` with `integer(allocator, family, operands)`.
Families are `powmod`, `gcd`, `isqrt`, `range_product`, `factorial`, `digit_sum`,
and `divmod`; `pow(A,B,M)` in the frontend maps to `powmod`. Decimal operands are
bound to immutable plans named `<family>-<width>`. The buckets are 1, 2, 4, 8, 16,
32, 62 and 128 digits. Factorial and range products default to 128 output digits;
other families select a bucket from operand lengths. An explicitly named plan
can select a smaller checked width. Values exceeding that width are rejected.

All arithmetic, comparisons, parity, error flags and final decimal digits come
from the prepared FFN kernels. Native control copies raw tensor rows, schedules
kernels and follows validated compiled Boolean outputs. It never decodes a
bignum, calculates a result or compiles a graph. Algorithms use bounded native
control over the boot-loaded graph bank; they are not single fused graphs.

| Family | Semantics |
| --- | --- |
| `powmod(a,b,m)` | Python `pow(a,b,m)` for nonnegative b and nonzero signed m. |
| `gcd(a,b)` | Nonnegative greatest common divisor; gcd(0,0) is zero. |
| `isqrt(n)` | Floor square root of a nonnegative integer. |
| `range_product(a,b)` | Product of nonnegative integers in the half-open range [a,b); empty product is one. |
| `factorial(n)` | Factorial of a nonnegative integer. |
| `digit_sum(n)` | Sum of decimal digits of abs(n). |
| `divmod(a,b)` | Python signed floor quotient and remainder, emitted as `(q, r)\n`. |

Negative exponents and square roots, zero divisors/moduli and negative product
bounds produce `error\n`; frontends should abstain on these unsupported domains.
Negative modular exponents do not attempt an inverse. Every intermediate is
checked at its selected decimal width. Overflow produces `error\n`, never a
wrapped value. Thus 84! fits 128 digits, whereas 85! is rejected. Error output is
emitted by compiled weights; token budgets, faults and cancellation remain
separate backend outcomes. Products have a 10,000-iteration bound. GCD and
Newton iteration have a bound of eight times the decimal width; bit loops use
four times that width. Exhausting an iteration bound returns the backend budget status without result bytes.
The caller's step budget remains authoritative.

Modular multiplication uses one compiled decimal multiply-accumulate/reduce
step per multiplier digit. Each Horner step reduces 10r+a*d with five conditional
subtractions of 16m, 8m, 4m, 2m and m. Two widened carry digits avoid materializing
a 256-digit product when operands use the 128-digit bucket. Square root uses Newton iteration with
separate half values and parity carry, avoiding overflow in x + n//x. Its initial
upper bound is selected from validated operand length metadata. Division
uses the existing compiled digit step and signed floor correction kernels.
Tuple punctuation and both decimal values are emitted by prepared circuits.

Mutable State belongs to each request, with one lazily allocated State per used
template. Every invocation resets all persistent tables, presence bits,
convolution history, residuals and scratch, including early budget/cancel/fault
and unexpected-byte exits. Immutable weights remain shared; no State is shared
between streams or requests. The probe's `--fresh-states` mode is the comparison
control. Bank load and request binding perform no graph compilation.

A GPU evaluator must execute the same prepared kernels and preserve these raw
row transfers, checked predicate branches, reset boundaries and budgets. A
naive host control implementation needs predicate readbacks; its latency is not
qualified by the CPU receipts. Device-side control and copies can avoid those
round trips. CPU evidence does not qualify GPU or live engine routing.

The bank checksum and measured per-bucket steps, timings, State comparison and
proof counts are recorded in the task's immutable receipts and STATUS-T1.md.
Qwenputer arithmetic construction retains its MIT license and Divyansh Agrawal
credit in every exported bank's provenance and full license file.

The graph backend imports the four pinned `steps_v1` model/load/state/evaluator
files, byte-identical to its qualified graph-v1 sources. This follows T2's
per-kind evaluator recommendation. Its schema hashes the same source bytes, so
existing prepared graph weights remain valid. Calc/ELF continues to use T2's
current evaluator after integration. New recursive evaluator types must never
be passed to the frozen graph schema walker. Changes to pinned source bytes
require an offline rebake and full token qualification.

At 128 digits, output uses two bounded cursor counters. The first contributes
at most 64 bytes, and the second starts at byte 65. This keeps all generated
coefficients within FP32 range and the scaled decay values within the required
reference precision. Width, sign, punctuation and errors remain circuit-owned.
Offline export can resume from checksummed source-fingerprinted checkpoints.
Legacy weight reuse requires the explicit graph-v1 checksum allowlist and its
compatible generator source fingerprints; a matching filename is insufficient.

`square_offset(a,b)` is an additional exact-integer structural plan for
`(a+b)**2-a**2`. It evaluates `b*(a+(a+b))` through shared add/mul weights, with
a compiled zero-b predicate. This avoids a double-width square. If b is nonzero,
any overflow in these intermediates implies overflow of the final result;
opposite-sign addition permits cancellation without first materializing 2a.
It handles both signs and zero; it is not tied to one numerical example.
The 89-digit trace case `(10**80+123456789)**2-10**160` can bind a=10**80 and
b=123456789 to this plan. A naive 128-digit square still overflows, so T5 must
recognize the exact structural identity before width validation. Do not apply
this rewrite to source languages whose intermediate overflow is observable.

Algorithm calls can select a smaller boot-loaded bucket from occupied BCD-slot
metadata, with conservative carry/product widths and the plan's checked-width
ceiling. This classifies storage shape without assembling a native integer or
calculating an answer. Missing smaller templates fall back to the plan bucket.
`--fixed-widths` in the probe is the token-exact comparison control. Digit sum
uses one parallel compiled reduction tree rather than one ALU call per digit.

## Whole-cell packets and exact formatting

`Builder.program(allocator, values, ops, prints)` binds a complete arithmetic
SSA program to the boot-loaded bank. It compiles no graph at request time.
The packet retains version1 and adds `prints`; a named plan may not override
its schedule or output. Generic operations may invoke the existing integer
algorithms, including multiple output slots. All operations run before output.
Set `checked=true` on a frontend's operations to turn an FFN error flag into
`GraphArithmetic` before any printing; the backend abstains and clears output.
Unchecked arithmetic retains its existing `error\n` convention.

`Print` selects a value reference, optional second reference, style, precision,
ending and optional opening parenthesis. Styles are `integer`, `fixed`, `comma`,
`integral_float`, `boolean`, `fraction` and `tuple`. Precision0..18 applies only
to `fixed`. Multi-field prints use `space` between arguments and `newline` at
line end. Other endings are `empty`, `comma_space`, `close`, `slash`,
`close_space`, `close_empty`, `close_comma_space`. Tuple fields accept newline,
space, empty or comma-space; they supply their own parentheses. Fraction fields
consume reduced numerator/positive-denominator outputs of the fraction algorithm.
The formatter elides `/1`; zero/negative denominators are refused. Output is
bounded to4096bytes and128Print descriptors per packet.

All ASCII, signs, decimal points, grouping commas, parentheses and separators
come from the two-input prepared `format-62`/`format-128` circuits. Native code
binds the formatting controls and follows compiled predicates; it never decodes
an operand into a host integer or renders an answer. Raw control BCD digits are:
precision at0/1, mode at2 (integer0/fixed1/comma2/integral-float3/bool4), ending
at3 (newline0,space1,empty2,comma-space3,close4,slash5,close-space6,
close-empty7,close-comma-space8), and prefix0/1 at4. Other digits must be zero. Boolean
values must be0/1. Integer/comma zero is unsigned; fixed/integral-float preserve
an explicitly encoded negative zero. Cursor segments stay within the existing
FP32-exact64/65 boundary, including128digit grouped outputs.

New integer algorithm families:

| Name | Inputs | Outputs | Contract |
|---|---:|---:|---|
| `fraction` |2|2|Reduced signed numerator and positive denominator; named result prints like Fraction.|
| `divexact` |2|1|Exact integral quotient, preserving numerator-XOR-denominator sign at zero; error if remainder is nonzero.|
| `fixed_div` |2|1|Coefficient for a rational rounded half-even at precision0..18; signed zero retained.|

`Builder.integer(.fraction, operands)` selects a named fraction plan.
`Builder.fixedDivision(allocator, lhs, rhs, places)` selects
`fixed_div-PLACES-WIDTH`. Generic operations carry the same `precision` metadata
for `fixed_div`; other operations require zero precision. New kernels are
`decimal_mul`, `round_div`, `scale10`, `float_exact`, `compare`, `min`, `max`.
`decimal_mul` preserves zero's XOR sign. `round_div` consumes unsigned quotient
and remainder plus original signed numerator/denominator. `float_exact` rejects
coefficients with more than15digits; it performs no binary floating arithmetic.

For a Python true-division bridge, use checked `divexact` then checked
`float_exact`, and style `integral_float`. Only exact integral quotients within
15digits are accepted; this bounded domain is exactly representable and uses
ordinary decimal `.0` spelling in CPython, including `0/-1 -> -0.0`. Fractional
or wider true division must abstain. This is not general IEEE float emulation.
Likewise Decimal multiplication is exact only when the frontend proves a finite
coefficient/scale domain within the selected Decimal context. The development
binder conservatively requires product coefficient bounds≤28digits and total
scale≤6, matching fixed-notation output in the unchanged default context. It
abstains on larger scales, general Decimal contexts, exponents or rounding. Literal integer `-0` is bound as0;
Decimal's signed zero is retained.

`Builder.sortedDistance(allocator, values)` selects an offline compare-exchange
plan `loop-sorted_distance-N-WIDTH` for0..16inputs. It computes
`sum(abs(a[i]-a[j]) for i<j)` using prepared signed min/max followed by a sorted
prefix reduction. The width selector uses only operand storage widths and the
bounded pair count. Every operation checks overflow. The existing sum, square,
factorial, Fibonacci and list-sum plans remain available. Width128 is the hard
coefficient limit; unsupported or overflowing intermediate values abstain when
checked, rather than invoking native bignum arithmetic.

`runner_graph_program.compile_program` is an offline development binder for
whole-cell qualification. It parses a bounded numeric AST into operand bindings,
checked operations and print descriptors without evaluating answers. Production
source admission remains owned by the native frontend. Bank capability, offline
source binding, frontend admission and live ownership are separate measurements.
