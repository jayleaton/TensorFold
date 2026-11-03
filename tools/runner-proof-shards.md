# Serial proof-host processes

`runner-proof-shards.patch` is the T1 integration patch for the proof-host registration owned by T3.
Apply it to the integration tree with `git apply --check` followed by `git apply`; the export branch has no such target.
The patch adds two Zig modules and the Zig SDK's MIT license. Keep the license with the adapted test runner.

The existing `test-runner-proof-host` target retains its root module, native ReleaseSafe settings and the literal
`runner proof:` compiler filter. That substring includes `compute runner proof:`. Compiler-discovered anonymous
and import tests remain included. No proof title is renamed or manually enumerated.

One compiler artifact supplies a mandatory catalog and 256 strictly serial singleton test-server processes.
Each occupied process exposes one original test name at protocol index zero; unused indices expose no tests.
The catalog rejects an empty population or more than 256 tests before execution. Increasing the population beyond
that capacity fails closed. Compilation is cached normally; every catalog and test process runs on every invocation.
The separate compute-proof target is unchanged.

This prevents process-lifetime bank ownership in one test from accumulating across the entire proof suite.
Each individual test and the compiler must still fit the same memory scope. It does not change graph evaluation,
model tokens, templates, bank loading or request behavior. Run only this target in its scope; unrelated build targets
are outside the serial dependency chain.

## Verify the population

Run in the project's guarded CPU environment, with the real circuit, graph-v7 bank and frontend compiler configured:

```sh
zig build test-runner-proof-list -j1 --summary all > catalog.log 2>&1
zig build test-runner-proof-host -j1 --summary all > shards.log 2>&1
python3 -B tools/runner_proof_shards_check.py catalog.log shards.log --original-catalog original.log
```

`original.log` must come from the original compiler-discovered population on the same source, filter, imports,
compiler and target. It accepts the original inventory tool's full names followed by `GATE_INVENTORY_COMPLETE N`,
or an independently obtained machine catalog. An old declared-test count is insufficient.
The verifier requires identical full-name sets/counts, one ordered start/result per occupied index, all unused-index
receipts, zero skips/errors/leaks and no missing or duplicate execution. Keep native Build exit status and the scope's
memory/OOM receipt too: neither a successful catalog nor a zero Build exit with skipped tests qualifies.

Records are JSON on stderr, prefixed `PROOF_SHARD `. Protocol stdout is unchanged. Preserve raw logs and source,
SDK, binary and bank checksums beside the reconciliation report. Repeat the unchanged build to prove execution
is fresh while compilation is reused.

## Protocol controls

The disposable fixture covers anonymous tests, the compute-prefix substring, JSON escaping, skip/failure/log/leak
flags and panic cleanup. Build the fixture without retaining the compiler during test execution:

```sh
zig test -Osafe -Mroot=tools/runner-proof-shards-fixture.zig -lc --test-filter 'runner proof:' --test-runner zig/runner_proof_shard_runner.zig --test-no-exec -femit-bin=proof-fixture
python3 -B tools/runner_proof_shards_selfcheck.py ./proof-fixture
```

Run the same helper through a disposable native Build root with the fixture as its root module. Setting
`PROOF_SHARD_FIXTURE_MODE` to `skip`, `fail`, `log`, `leak` or `panic` must prevent qualification. Non-skip failures
must also fail native Build. A skip may retain the SDK's successful Build exit, but the coverage verifier rejects it.
Do not set this fixture variable on production proof runs. Fuzz mode is unsupported and fails explicitly.
