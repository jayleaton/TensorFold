# Direct arithmetic RV32 firmware

`runner_fast_firmware.py` is an experimental firmware generator for the existing
Qwenputer P2 circuit. It specializes the arithmetic schedule to an expression,
removing the calculator's runtime parser and digit-register allocator. It does
not change the evaluator, instruction set, weights or training procedure.

The host parses and schedules operations; it never evaluates their arithmetic
results. Decimal operands are split into base-10,000 chunks in volatile RAM.
The generated firmware loads those operands and computes all expression results
with RV32 instructions inside P2. Volatile input loads prevent LLVM from folding
the whole expression into its answer. The source embeds the original expression
as metadata, with no checkout-specific path.

Supported input is decimal nonnegative integer literals, parentheses, unary `+`,
binary `+`, `-`, `*`, and `**` with a literal exponent from 0 through 999. Python
precedence applies. Subtraction must remain nonnegative. A negative intermediate
or an intermediate exceeding 62 decimal digits prints `error\n` and returns 1.
Zero, including `0**0 = 1`, is supported. Unary minus, computed exponents, names,
division, fractions, decimals, gcd and loops are outside this subset. The existing
ELF exporter remains the path for arbitrary short C programs, including loops.

Each value uses at most sixteen 32-bit limbs. Multiplication emits unrolled
convolution columns and normalizes once per column, rather than once per cell.
At most sixteen products of `9999 * 9999`, plus a carry below 160,000, fit within
32-bit unsigned arithmetic. Generated overflow columns and the top limb's
62-digit boundary are checked before a value is consumed by another operation.
Exponentiation uses a scheduled square-and-multiply chain, without computing
powers on the host. Inputs are limited to 1024 characters and 128 AST nodes;
generation also rejects C exceeding 600 lines. Large expressions may need splitting.

Run these commands on an authorized compute node, with the parent's resource
guards. `QWENPUTER` denotes the upstream checkout and `ZIG` the Zig executable.

```sh
python3 tools/runner_fast_firmware.py \
  --expr '123456789012345678901234567890*987654321098765432109876543210' \
  --output-c out/direct-multiply.c --output-elf out/direct-multiply.elf \
  --qwenputer "$QWENPUTER" --compiler "$ZIG" --compile
python3 tools/runner_fast_firmware.py \
  --expr '2**100' --output-c out/direct-power.c \
  --qwenputer "$QWENPUTER" --compiler "$ZIG" --compile
python3 tools/runner_export.py --qwenputer "$QWENPUTER" \
  --elf out/direct-multiply.elf --output out/direct-multiply-circuit
```

Without `--compile`, generation requires only Python's standard library and emits
C. Compilation uses Zig's `cc`, `generic_rv32+m+zba`, ILP32 soft ABI, `-O2` by
default (`--optimize O3` is available), no libc, sanitizers or stack protector,
and upstream `env/crt0.S` plus `env/rom.ld`. Its minimal byte-output and exit
runtime is included in the emitted C. P2 supplies the initial stack and loads
ELF RAM/DROM sections. No calc runtime or custom digit instructions are needed.
The printed command can be retained as part of a local receipt; it contains
the caller's filesystem paths and should be redacted before publication.

The JSON summary reports scheduled limb products and multiplication columns,
not instruction counts or circuit tokens. For example, a pair of 30-digit operands
has eight limbs each: 64 limb products and sixteen normalization columns,
compared with up to 900 decimal digit products. This arithmetic-work reduction
does not establish a 5–10x circuit-step reduction: loads, carry normalization,
branches, compiler spills, printing and multi-token values still cost tokens.
The baseline observations supplied for this experiment are 2280 steps for the
30-digit multiply, 3131 for `2**100`, and 599 for the short loop; retain their
exact workload identities before comparing. No candidate speedup is measured here.

Before accepting the candidate, compile on the node, inspect its actual RV32
instructions, and run its ELF through upstream `elf.program` and `ref.Machine`.
Compare output bytes with an independent reference for ordinary arithmetic,
zero, carry chains, borrow chains, exponent zero/one/999, subtraction underflow
and the 62/63-digit boundary. Capture both instructions and circuit tokens.
Export separate assets through `runner_export.py --elf`, then compare the
exported circuit's complete token trace and final output with the ELF reference.
Check volatile RAM operands and absence of an embedded answer in the artifact.
Changing the expression requires generating another firmware; this does not
replace the baseline calculator's expression-input interface.

Repository checks remain `zig fmt --check build.zig zig`, lean comparison by
finding identity with zero new findings, `zig build test test-kernel-contract
test-golden fuzz-cortheon`, plain `zig build`, and `git diff --check`, using the
pinned Zig version. This implementation helper intentionally runs no source
checks, tests, compiler or builds on the development machine. The parent owns
node validation and the task ledger.
