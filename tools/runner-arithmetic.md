# Compiled arithmetic experiment

`runner_arithmetic.py` builds an arithmetic circuit, rather than RV32 firmware.
It uses Qwenputer's pinned MIT-licensed IR, exact SwiGLU primitives, P2 token bus
and DeltaNet heads. The complete upstream license and credit remain embedded.
No training or host arithmetic-result execution is involved.

The host parses decimal operands and schedules unsigned `+`, `*`, and literal
`**` (exponent 0..999), with parentheses and unary plus. Each operand is stored
as decimal-digit bits in immutable DeltaNet rows. The weights load those bits,
multiply digits, normalize base-10 carries and add partial rows. Intermediate
results exceeding 62 digits produce `error\n`. Negative values, subtraction,
division, fractions, gcd and arbitrary C/ELFs are outside this experimental
frontend. Reject unsupported input and retain the existing calc/RV32 path.
The expression limit is 255 characters / 128 AST nodes. Large schedules can
still create large graphs; this is an explicit prototype, not an enabled default.

`--loop-bound N` separately unrolls the sum of squares for integers 0 through
N-1, for any N in 0..32. Each induction value is an operand snapshot and every
square and addition executes in FFNs. This is a bounded loop compiler example;
it is not an ELF translator. The original C loop ELF and an optimized RV32 ELF
provide independent output and firmware comparisons.

Operand heads read only at GO. The circuit latches computed digit bits and the
overflow flag in DeltaNet result rows, then a cursor selects each output byte.
Later positions feed zero operands to the arithmetic blocks; the existing
sparse CPU evaluator can skip their inactive units. It still visits the blocks,
so this is not a claim that GPU arithmetic layers have zero cost while printing.
Its generated stream is exactly
`BYTE(stdout[0]), ..., BYTE(stdout[-1]), HALT`, starting from input `GO`.
Thus a successful D-digit unsigned answer takes D+2 generated steps. Internal
RV32 trace tokens have deliberately been removed. This preserves output byte
tokens and HALT, not the old ISA trace. The compiler never embeds answer digits;
changing an operand's snapshot bit changes the computation without editing FFNs.

## Experimental container and entry point

Files use `.tfac`, magic `TFARITH1`, and JSON format
`tensorfold.runner.arithmetic`, version 1. The rest of the 64-byte header is
identical to TFIMR001: little-endian `<8sIIQII32s`, version 1, header size 64,
payload byte count, zero flags/reserved, then SHA-256 of the UTF-8 JSON payload.
JSON encoding is deterministic, compact and sorted by object key. All numeric
weights and snapshot values use the same exact f64 JSON representation as P2;
token files use little-endian u32. Source, constants, dimensions, features,
nodes, token families and output code retain the P2 schema and ordering.
`program.kind` is `compiled_arithmetic`; its expression or loop bound, 62-digit
limit and reduced trace contract replace ROM/DROM/ISA metadata. Operand rows
contain A_REF in columns 0/1 and A_BIT times each bit in columns 2 onward, at
key row 0; one head packs at most 31 decimal digits. Operand queries are gated by
GO, with tiny=2^-90 defining a zero read at other positions. Result heads write
only on GO and read row 0 at every position; columns 2..125 pack up to 31 digits
and column 126 retains the overflow flag. The cursor is a standard Qwenputer
decay head. No ISA RAM or program image is required.

The entry point is `State.init(snapshot)` followed by `GO`. Do not call the
existing `Machine.boot(calc|elf)`, patch an expression into RV32 RAM, or replace
an ELF in these graphs. Production loaders intentionally reject the experimental
magic. Integration needs an explicit compiled-arithmetic program kind and
frontend, while retaining the same FFN/DeltaNet evaluation primitives.
This prototype is not wired into the model or the existing calc asset.

For matched native measurements, `runner_steps_check.py` additionally writes
`.bench.tfpc` wrappers around the exact same graph and snapshots. Those wrappers
supply the original baseline's unused ROM/RAM metadata so the unmodified T2
loader can parse it. They carry an explicit `experimental.entry` warning and
must be run through `State.init` and `eval.step`, never `Machine.boot`.
They are disposable benchmark inputs, not deployable firmware circuits.

## Verification

Run only on an authorized node inside its session/QUIET guard, after repository
fmt, lean identity comparison, combined test/contract/golden/fuzz gates and build.
For an in-memory experiment followed by export and replay of the actual file:

```sh
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 python3 tools/runner_arithmetic_check.py \
  --qwenputer "$QWENPUTER" --expr '2**100' --export out/power.tfac
python3 tools/runner_arithmetic_check.py \
  --qwenputer "$QWENPUTER" --circuit out/power.tfac
```

The checker runs upstream's own reference machine, compares its oracle and
stdout, restores the serialized graph, and checks every next token with
upstream BF16 projections, SwiGLU gates and DeltaNet exactness checks enabled.
All prefix predictions matching the reference output stream prove free-run
causal equivalence by induction. An independent T2 f32 free-run benchmark adds
native evaluator evidence; its output, step count and full token SHA must match.
The suite also covers the supported portion of the first 20 seed-0 golden
cases, explicitly lists unsupported cases, exercises carry/zero/overflow, and
requires an operand-snapshot mutation to fail next-token replay.

The circuit's extra depth and width matter. Report actual matched native
latencies, graph sizes and steps separately. Step reduction alone does not
establish CUDA performance, packing into a stock Qwen layer count, or model
integration correctness. Carry lookahead and common-subexpression sharing
remain possible follow-ups. The original `calc.tfpc` / `calc.tfimr` is never replaced.

`runner-arithmetic-bench.zig` is the small native proof harness. Stage it beside
T2's unchanged `runner_circuit{,_model,_load,_state,_eval,_elf,_rom}.zig` and
`runner_api.zig`, name it `main.zig`, and compile with pinned Zig:
`zig build-exe main.zig -lc -O ReleaseFast -femit-bin=runner-native`.
`runner_native_check.py --help` lists the report directory and original baseline
assets it consumes. The harness runs each main comparison five times, records
load/run nanoseconds, and hashes every generated token plus GO as LEu32.
The driver requires every hash and step count to equal the upstream reference.
It also checks every one of the twenty routed sample cases and the separate
wider-limb/optimized-loop ELF circuits, with full internal RV32 traces there.

The `program.input_slots` list gives each operand head, row, digit offset,
digit count, first column, four bits per digit and LSD-first order. This makes
caching by operator schedule and operand widths possible: retarget the snapshot
without compiling or loading new FFN weights. The suite changes a 30-digit
operand bit, first requires the old reference comparison to fail, then retargets
the expression metadata and requires the new result to pass with identical
weights. Preparation cost can only be amortized after that template cache and
explicit input-slot loading are integrated; current tools build per request.
The new compiler schedules its single-assignment nodes by dependency level to
expose parallel FFNs. It preserves each node's reduction order and leaves the
original P2 circuit ordering untouched.
