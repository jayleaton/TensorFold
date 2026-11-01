# Runner reference goldens v1

`runner_trace.py` is a development reference exporter, using the supplied Qwenputer
checkout at `bec1a0c9654ac8151b6f36ed0154a19c53141384` (MIT). It imports upstream
`calc`, `ref`, `spec`, `elf`, and their assembler dependency at runtime; no upstream
implementation is copied into TensorFold. The export track retains the upstream
license separately. Python 3.11 or newer and the Python standard library suffice;
this command does not import the circuit compiler, Torch, or a GPU executor.

Large corpora should run on isolated CPU capacity under the project's resource
admission policy. The examples omit that surrounding admission wrapper.
Compilation and golden generation require no GPU or model checkpoint.

## Build the loop ELF

Set `QWENPUTER` to the pinned checkout and `GOLDENS` to a fresh output directory.
This freestanding build uses Zig 0.17's bundled compiler/linker, upstream
`crt0.S`, `rt.h` and `rom.ld`. It needs neither picolibc, `sys.c`, RISC-V GCC nor
a separate `ld.lld`. It produces a static ELF with the upstream P2 memory map.
The C source implements `exit` for `crt0.S` with `rv_ecall2(93, ...)` and formats
the computed unsigned integer as decimal bytes using `rv_putchar` directly.

```sh
QWENPUTER=/path/to/qwenputer
GOLDENS=/path/to/runner-goldens
RV_ENV="$QWENPUTER/experimental/riscv/chipformer/rv/env"
zig cc -target riscv32-freestanding-none -mcpu=generic_rv32+m+zba -mabi=ilp32 -O0 -fno-sanitize=all -fno-stack-protector \
  -nostdlib -static -Wl,-T,"$RV_ENV/rom.ld" \
  "$RV_ENV/crt0.S" tools/runner-loop.c -I"$RV_ENV" -o /path/to/runner-loop.elf
```

`runner-loop.c` sums `i*i` for `i = 0..9` using a real loop and prints exactly
`285\n` (hex `3238350a`). `-O0` and the volatile bound prevent replacing the loop
with a constant. Inspect its disassembly with an available RISC-V-capable
disassembler during parent verification, then generate and check the artifacts:

```sh
python3 tools/runner_trace.py --qwenputer "$QWENPUTER" --output "$GOLDENS" \
  --cases 10000 --seed 0 --checkpoint-interval 256 --loop-elf /path/to/runner-loop.elf
python3 tools/runner_trace_check.py --output "$GOLDENS"
```

The exporter always appends the requested 30-digit multiplication after the
`calc.cases(N, seed)` cases, without changing their upstream order or RNG stream:

```text
123456789012345678901234567890*987654321098765432109876543210
121932631137021795226185032733622923332237463801111263526900
```

The output product has a terminating newline. `--loop-elf` adds one more case;
without it the export contains `N+1` cases, with it `N+2`. Defaults are 10000,
seed 0 and checkpoint interval 256. `--cases 0` is useful for focused execution
of the multiplication and optional loop without drawing random cases.

## Files and case records

`manifest.json` identifies format `tensorfold-runner-goldens-v1`, exact upstream
revision, source/firmware SHA256 hashes, seed, requested random case count,
checkpoint interval, total case/instruction/token counts, machine-limit count, and each data file's
size and SHA256 over its stored bytes. It is written last, after all comparisons
pass. Consumers must require it and verify file checksums before loading rows.
The revision and tracked changes to imported upstream files and calc firmware
are checked before execution. No absolute source paths or timestamps are stored.

`cases.jsonl.gz` contains one compact JSON object per line, in execution order.
`tokens.leu32.gz` is one gzip stream containing all cases' traces concatenated
as unsigned 32-bit little-endian token IDs, without headers or padding. Gzip
headers have fixed timestamp zero and no filename. A supplied ELF is copied to
`loop.elf`, allowing the exact same program snapshot to be reconstructed.
An existing directory is allowed, but existing golden artifact names are refused.
All reference cases are staged before publication; a failed comparison publishes
no manifest. A process interruption during file publication may leave data files
without a manifest; use a fresh destination when retrying.

Every row contains:

- `id`: `calc-00000` onward, then `multiply-30digit`, then optional `loop-squares`.
- `kind`: `calc` or `elf`; `operation`: upstream operation label, `mul`, or `sum_squares`.
- `input`: for calc, `expression`, `ram_address` (256), and `bytes_hex` including
  the terminating NUL; for the loop, `elf` (`loop.elf`) and empty `stdin_hex`.
- `instruction_count`: actual `ref.Machine.steps`; refresh records do not count.
- `trace_token_count`: complete trace length including initial GO and final HALT.
- `generated_steps`: `trace_token_count - 1`, since GO is supplied as input.
- `refresh_count`: number of R tokens, which advance the trace without executing an instruction.
- `token_offset_bytes`: byte offset in the **decompressed** token blob; read
  exactly `4 * trace_token_count` bytes. This is not a compressed file offset.
- `trace_sha256`: SHA256 of exactly that case's LEu32 trace bytes, including GO.
- `expected_output`: the mathematical oracle's exact UTF-8 string, retaining the final newline.
- `final_output`: the reference machine's exact output; T2 must match these bytes.
- `outcome`: `ok` (output equals the oracle) or `machine_limit` (upstream's exact
  `error (too complex: split it)\n` register-exhaustion result). No cases are dropped.
  Required multiply and loop cases must be `ok`.
- `final_output_hex`: exact output bytes; `exit_code` is 0 for values/loop, 1 for
  expected calc errors or register exhaustion.
- `checkpoints`: token-prefix diagnostics described below.

Each calc machine is freshly constructed with `calc.machine(expression)`, which
loads the pinned `calc.elf` and patches the NUL-terminated expression into RAM.
T2 must use that same firmware snapshot and patch before feeding GO. Each loop
machine is freshly constructed with `ref.Machine.from_program(elf.program(elf))`:
ROM/DROM/RAM segments, zero-filled BSS, x2 initialized to SP_INIT, other registers
zero, cleared digit file, and entry anchor follow upstream semantics. No live
machine state is carried between cases.

## Trace and checkpoint semantics

The exporter calls `ref.Machine.record()` until halted and consumes every token
in each returned record. A record may contain several tokens; an R refresh
contains no executed instruction. The hard limit is 10,000,000 total tokens per
case, including GO, and a record crossing that limit fails the export. Faults,
unexpected exit codes and output mismatches also fail. Calc firmware returns
exit 1 with `error\n` or its exact register-exhaustion message; these expected
errors are retained, while successful values and the loop require exit 0.
Every calc output is compared
strictly to `calc.expected(expression)`, including `error\n` cases. Upstream's
separate register-exhaustion result is retained as `machine_limit`, with the
mathematical oracle recorded separately in `expected_output`; it is not an oracle
match. Any other discrepancy fails with the expression and actual/expected output.
`manifest.machine_limit_cases` records the total; limit cases remain in all token
and checkpoint comparisons.

The running SHA256 is updated **per trace token**, using `struct.pack("<I", id)`.
Checkpoint digests are saved at token count 1 (GO), every positive multiple of
the configured interval, and the final token count, deduplicated. Interval 256
therefore means counts 1, 256, 512, ... and the final count, not instruction
counts, record counts or generated-step multiples. A checkpoint may lie inside
a multi-token instruction record. It describes a trace prefix, not the Python
machine state (which `record()` already committed through the whole record).

Each checkpoint has `token_count` C, `generated_steps` C-1, `prefix_sha256` over
tokens `[0:C]`, and a trailing window of at most eight IDs in `window`, beginning
at zero-based `window_start`. There are no machine-state digests or per-token
digest arrays. Full tokens are retained, so any per-step prefix digest can be
recomputed without rerunning the machine. Digests reset independently per case.

For T2, decompress the blob once or stream cases in order, initialize each case's
snapshot, supply token zero (GO), and compare all subsequently generated tokens
through HALT. Compare the recorded checkpoints at exactly C consumed tokens and
the final trace hash/count/output. On mismatch, use the checkpoint window for
orientation and the full trace to locate the first differing token. Comparing
only output bytes does not establish circuit token exactness.

## Verification handoff

Run the actual ELF build/disassembly inspection, focused `--cases 0 --loop-elf ...`
export and full 10000-case seed-0 export. Verify output bytes, expected
case counts, manifest checksums, gzip decoding, per-case offsets/counts/hashes,
checkpoint prefixes/windows, generated steps and final HALT directly from the
produced artifacts. Repeat the focused export into another fresh destination
and compare manifests/data hashes for determinism.

`runner_trace_check.py --output DIR` uses only the standard library, without an
upstream checkout. It hashes the stored files before streaming both gzip files,
checks every byte offset/count, token ID, GO/HALT boundary, instruction/refresh
count, checkpoint position/hash/window and trace hash, and reconstructs stdout
from all BYTE payloads below 256 (including BPRD's BYTE tokens). Payloads 256..511
are not stdout. It compares reconstructed bytes to both final-output encodings
and the expected output. It requires the exact multiplication case and loop
case, verifies their inputs/results, and checks manifest totals and EOF for both
streams. Thus a calc-only export remains valid for the generator but fails this
full-acceptance verifier. This checker validates artifact consistency; the
generator's independent `calc.expected` comparison supplies the random-case
arithmetic oracle, and the checker does not execute the ELF or the circuit.

Repository gates remain `zig fmt --check build.zig zig`, lean checker comparison
by identity through `lean_check.py` and `lean_new.py` (zero new findings),
`zig build test test-kernel-contract test-golden fuzz-cortheon`, plain `zig build`,
and `git diff --check`, as recorded in the parent task ledger. These artifact
checks establish reference goldens only; they make no evaluator or speed claim.
