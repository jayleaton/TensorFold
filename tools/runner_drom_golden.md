# Supplemental DROM golden

The original 10,002-case corpus remains frozen. `runner_drom_golden.py` generates
one independent `drom-byte-string` ELF case from the pinned Qwenputer reference
machine, including every token, interval-256 prefix checkpoints and exact output.
It prints `circuit drom bytes\n` twice and exits zero. Both signed and unsigned
volatile byte loops read the string through its NUL terminator. A high-bit DROM
byte also checks LB sign extension and LBU zero extension.

The tool extracts the actual `runtime` and `linker_script` multiline literals from
the supplied served frontend compiler source. It compiles freestanding RV32IM+Zba
C at O0, with `-fno-builtin`, without the CLI's expression/IR translation route.
Explicit `--clang` / `--linker` override `TF_RUNNER_CLANG` / `TF_RUNNER_LLD`; the
defaults match the frontend. It requires DROM `.rodata`
containing both probes and the terminator, and both LB/LBU instruction encodings. `--zig /absolute/path/zig` selects Zig cc;
`--objdump` optionally adds full external disassembly. Without it, the artifact
lists the LB/LBU addresses decoded directly from the ELF instructions.
The generated source, linker script, disassembly, compiler command and checksums
remain in the supplement for inspection. A changed frontend literal format fails
closed. Generation refuses a nonempty destination.

Run only inside the parent's admitted node01/node02 CPU scope (nice 19, idle IO,
at most 2 GiB, no swap, at most 30 minutes, with QUIET and thermal/boot floors):

```sh
python3 tools/runner_drom_golden.py \
  --qwenputer "$QWENPUTER" \
  --compiler-source "$CIRCUIT_SOURCE/runner_frontend_program_compile.zig" \
  --clang "$RV_CLANG" --linker "$RV_LLD" --objdump "$RV_OBJDUMP" \
  --output "$GOLDENS-drom"
```

Use an absolute compiler source path; the compiler can come from the served image
when the host lacks lld. The reference recorder refuses more than 100,000 generated
tokens, including refreshes. A fault, nonzero exit or different output fails export.

The actual shard checker is `zig/src/core/runner_circuit_golden_test.zig`, supplied
by T2 and absent in this exporter checkout. `runner_native_check.py` is a separate
benchmark checker. Apply `runner_drom_golden.patch` to the mutable integrated source
overlay before its build, and run `zig fmt` there. Do not patch frozen source kits.
The patch targets the isolation-v2 checker and leaves its existing cases untouched.

The checker runs the original modulo selection first, then the entire supplement.
`TF_RUNNER_GOLDEN_LIMIT` only limits original calc cases. The default supplement
path is `${TF_RUNNER_GOLDENS}-drom`; `TF_RUNNER_DROM_GOLDENS` can select another
supplement, but cannot disable it. Missing files, invalid supplement metadata, an
unsupported ELF boot or a missing/extra supplemental case fail the test.
The ELF, rows and tokens are checked against the supplement manifest before use.
Every token, checkpoint, output byte and final HALT is compared by the existing
checker, with the supplemental run capped at 100,000 steps.

For shard `0/25`, the expected result is the existing 401 original cases plus one
supplemental case, 402 total, with no supplemental skips. Other shards each gain
one case independently of modulo selection. Parent verification must run the
accepted circuit and the m3 DROM-disabled mutant separately with this same shard:

```sh
TF_RUNNER_CIRCUIT="$CORRECT_CALC" TF_RUNNER_GOLDENS="$GOLDENS" \
  TF_RUNNER_GOLDEN_SHARD=0/25 zig build test-runner-host -j1
TF_RUNNER_CIRCUIT="$M3_CALC" TF_RUNNER_GOLDENS="$GOLDENS" \
  TF_RUNNER_GOLDEN_SHARD=0/25 zig build test-runner-host -j1
```

The first must pass 402 cases; the second must fail on the supplemental DROM case
within the cap. Keep the existing m1-m6 control proofs. The exact accepted circuit
identity and all original corpus hash pins remain requirements of the parent gate.

Task ledger: read `AGENTS.md`, `CONTRIBUTING.md`, `docs/atc/SKILL.md`,
`docs/recipes/SKILL.md`, `RUNBOOK.md`, `docs/recipes/README.md`, `.zig-version`,
`build.zig`, `build.zig.zon`, `.github/workflows/no-gpu.yml` and `zig-fmt.yml`.
Relevant checks: Zig 0.17.0 `zig fmt --check build.zig zig`,
`zig build test test-kernel-contract test-golden fuzz-cortheon`, plain `zig build`,
and lean identity comparison using `tools/zig/lean_check.py` / `lean_new.py`.
The parent also requires the 11 codec tests and seeded single-job gates.
The qualified `t1-artifacts/drom-v1` release retains the actual patched checker,
402-case positive receipt, m3 failure, project gates and source identities.
Apply the patch explicitly when integrating into the T2 checker branch.
