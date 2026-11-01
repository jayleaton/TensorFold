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
