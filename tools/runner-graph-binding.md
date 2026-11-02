# Direct integer graph binding

`runner_graph.directFrontend(&bank)` implements the runner Frontend contract
without importing or invoking the Python/RV32 frontend or a host compiler.
`runner_graph.bindCell(allocator, &bank, cell)` returns owned graph packet bytes,
or null for an unsupported cell. The bank must outlive the frontend.
T3 can apply `tools/runner_graph_bind_route.patch` to try this binder before
its existing routed frontend. The route file belongs to integration and is
absent from the exporter branch; the patch requires explicit application.

Binding parses source, converts integer literal notation, and selects existing
prepared kernels. It never evaluates an expression answer or compiles weights.
Every arithmetic operation is checked; all output bytes come from the compiled
formatters. The graph-v4 bank remains unchanged, with manifest SHA
`ec0f284687d499e812cc72a6c3d61611e32ad436df9048c8d5430ca26e69587c`.

The direct frontend accepts empty-context Python inline, REPL and fenced cells.
Inline math interprets caret as power and a separate `mod` word as remainder;
Python REPL/fences preserve Python syntax and refuse unsupported XOR. Inline
percent followed by a sign, such as `50%+2`, defers to the ordinary percentage
route; explicit `mod` remains unambiguous. Literal
radix conversion accepts decimal and hexadecimal integers up to 128 decimal
digits. Underscores, floats, booleans, true division and unknown names refuse.
The source limit is 4096 bytes, the AST limit 256 nodes, nesting 32, and the
emitted operation limit 1024. Conservative decimal-width bounds can refuse a
valid expression whose intermediate bound exceeds 128 digits.

Expressions support integer addition, subtraction, multiplication, floor
division, remainder, unary signs and literal-exponent powers through 999.
Computed ordinary exponents, including `2**3**2`, conservatively refuse;
computed exponents inside three-argument modular power remain graph values.
`pow(a,b,m)` and a power immediately under remainder select `powmod` before
materializing the power. Computed moduli such as `10**18` stay graph operations.
Unsupported or out-of-domain arithmetic never owns a result.

Fenced cells support straight-line assignments and numeric prints, including
multiple print arguments. Declared `import math` or `from math import gcd,
isqrt` enables those exact integer intrinsics. Assignments retain SSA bindings;
unused computations are still checked. Notebook state, builtin shadowing,
other imports, functions, loops, containers, formatting strings and effects
refuse and remain available to the ordinary frontend. Arithmetic failures
abstain for the entire cell, with no owned partial output; Python exception
and partial-stdout behavior is outside this successful-cell contract.

The source goldens include the three saved S2 statements and S1's modpow,
gcd and four-output integer-square-root cells. The proof corpus also includes
precedence, signs, radix and semicolon neighbors, context/shadowing refusals,
checked arithmetic failures, changed operands and incorrect historical Result
claims. Fresh isolated CPython supplies successful-cell stdout. The native
probe retains the bound packets and checks actual BYTE/HALT traces against
those values, with fresh/reused State and fixed/adaptive widths. Replaying the
packets on the accepted graph-v4 evaluator checks the same raw token digests.
These are admission/arithmetic proofs, not evidence that the live scanner fires
or that an image owns a same-line `Result: value` claim.

The latest T5 abstain audit needs no additional integer weights beyond v4:
Fraction/exact integral division, Decimal coefficients, digit reversal and
sorting already have kernels; remaining text/container/I/O and binary64 cases
need their frontend/environment contracts. No bank rebake is required here.
