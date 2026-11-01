# Native graph verification harness

This is a rerunnable qualification tool, not a qualification receipt. The author performed source inspection and edits only. The parent must build and execute it on its assigned isolated verification host; no formatter, compiler, tests, runtime probe, SSH, or Git mutations were run by this helper.

## Task ledger and standards

Scope is exactly `tools/runner-graph-probe.zig`, `tools/runner_graph_verify.py`, and this document, on the assigned existing branch. Runtime implementation, corpus generation, template packing, API changes, ATC coordination, and publication belong to the parent. No further helper was launched: this bounded verification slice shares one CLI/report contract and has exclusive ownership of three files.

Standards inspected: `AGENTS.md`, `CONTRIBUTING.md`, `docs/atc/SKILL.md`, `docs/recipes/SKILL.md`, `RUNBOOK.md`, `docs/recipes/README.md`, `.zig-version`, `build.zig`, `build.zig.zon`, `.github/workflows/no-gpu.yml`, and `.github/workflows/zig-fmt.yml`. No additional AGENTS.md or CLAUDE.md was found under tools. The assigned prove-it-works skill was read; executable proof is delegated to the parent under the explicit no-execution boundary.

Sources are limited to 600 lines each. Comments and docstrings are single lines of at most 120 columns. The tool uses the pinned Zig 0.17.0 interfaces and Linux libc clocks, matching the existing arithmetic benchmark. There is no separate Python formatter/linter configuration for these tools in the inspected project configuration.

Parent checks before any commit or PR:

```sh
zig fmt --check build.zig zig tools/runner-graph-probe.zig
python3 tools/zig/lean_check.py
python3 tools/zig/lean_new.py lean-base.txt lean-head.txt
zig build test test-kernel-contract test-golden fuzz-cortheon -j2 --summary all
zig build -j2
```

Use the same checker on base and head and compare all 427 inherited finding identities, with zero new findings. The normal lean checker scans `zig` and `tools/zig`; explicitly include these new source files in the parent's additional lean check. Format the new Zig file on the verification host before the format check if necessary. Also run the staged probe build, focused graph controls and backend lifecycle proof below. None of these checks has been run by this helper.

## Staging and invocation

Stage the probe as `main.zig` beside the parent's frozen nine-file T2 set and these runtime graph files:

- `runner_graph.zig`
- `runner_graph_bank.zig`
- `runner_graph_machine.zig`
- `runner_graph_packet.zig`
- `runner_graph_binary.zig`

Use the parent-staged `runner_api.zig` that includes `Program.kind.graph`. Do not substitute the changing circuit checkout for the frozen T2 sources. Build in that staging directory on the verification host:

```sh
zig build-exe main.zig -lc -O ReleaseFast -femit-bin=runner-graph-probe
```

This tool does not pack templates or generate expected answers. The parent supplies the full prepared `.tfgp` bank, named plans and JSONL corpus. The bank's paths resolve from the child process working directory, rather than the manifest directory.

A normal case line is:

```json
{"id":"example","packet":{"version":1,"plan":"add-4","values":["12","-3"]},"expected_output":"9\n"}
```

`expected_output` is the old golden row's `final_output`, including its terminating newline. Optional `expected_status` defaults to `halted`; other machine statuses are `budget`, `cancelled`, `fault`, or exact native error names prefixed with `error:`. A boot rejection must expect an empty output. The driver forbids status overrides and mismatch waivers on production rows.

Direct native invocation:

```sh
./runner-graph-probe bank.json cases.jsonl --budget 100000 --lifecycle > native.json
```

The probe loads the bank exactly once, creates a fresh Machine per selected case and emits one JSON object on stdout. It writes progress every 100 cases on stderr. Output or token mismatches are accumulated, rather than stopping at the first failed case. Setup failures emit a JSON object with `pass:false` and `fatal_error`. Exit status is nonzero for any unexpected case failure, requested lifecycle failure, or an empty selection.

`--start N` selects a zero-based case offset. `--count N` and its alias `--maxcases N` limit the selected count. These are diagnostic/sharding controls, not full-corpus evidence. Each selected packet or loaded plan is capped at 4096 operations; native packet/value/output limits still apply. The file input limit is 128 MiB. Raw token storage and output storage are freed after every case; case receipts remain in memory until the final summary.

Full driver invocation, from a working directory chosen by the parent:

```sh
python3 tools/runner_graph_verify.py \
  --probe stage/runner-graph-probe \
  --probe-source stage/main.zig \
  --runtime-source-root stage \
  --manifest assets/bank.json \
  --cases assets/native-cases.jsonl \
  --goldens reference/cases.jsonl.gz \
  --cwd . \
  --report graph-verification.json
```

The driver reads plain or gzip JSONL, requires unique IDs, verifies every expected output against the original `final_output` and optional `final_output_hex`, appends five controls, then starts the binary once. Native stderr is inherited so progress remains visible. Temporary combined JSONL is deleted automatically. The report retains the native summary, source/binary/manifest/corpus/golden/pack hashes, case counts, slice scope and process wall time. A failed native summary is preserved in the report.

The driver defaults to named production packets with empty or absent `ops` and a zero or absent `result`. `--allow-generic-ssa` explicitly permits the T5 SSA surface without claiming every production packet was named. `--start` and `--count` select production rows before controls are appended. Every selected shard must contain a successful named addition for the operand control and two successful cases whose first operation names differ for lifecycle checks. The default process timeout is 7200 seconds; `--timeout` changes it. `--budget` is per machine and defaults to 100000 evaluator steps. Do not interpret a sliced or generic run as the full named corpus gate.

## Exact token evidence

The native source sets `m.trace = &trace` before calling Machine.run. That ArrayList contains the actual return token from every evaluator step. There is no reconstructed actual trace from output bytes and no initial synthetic GO.

For a halted case, the audit requires precisely one canonical HALT for each non-bind ALU operation, followed by the actual canonical BYTE token IDs for output bytes, followed by one final canonical HALT. Every generated non-BYTE token must equal HALT. Pure bind copies consume no evaluator steps. The trace length must equal both `m.steps` and `ALU calls + UTF-8 output bytes + 1`. The BYTE payload stream must also equal Machine.run's independently collected output bytes. Finally, the output bytes must equal the frozen golden output.

Canonical P2 IDs are derived from the documented family layout: BYTE base 229380 and HALT 230321. Every prepared template must advertise the same HALT and BYTE family identity. The generated-token audit checks these numeric IDs directly, rather than trusting the emitter's rendered text alone.

Each receipt contains:

- `actual_canonical_sha256`: actual BYTE tokens plus the actual final HALT, encoded as LEu32.
- `expected_canonical_sha256`: golden UTF-8 BYTE IDs plus canonical HALT, encoded as LEu32.
- `raw_trace_sha256`: every actual generated token, including all internal ALU HALTs, encoded as LEu32.
- `steps`, `raw_tokens`, `alu_halts`, `trace_valid`, output equality, status, output and timings.

The summary's `actual_canonical_sha256` and `raw_audit_trace_sha256` hash concatenated case streams in selected corpus order, followed by appended controls. They exclude GO and add no case separators. Per-case hashes and token counts retain the boundaries. Rejected-before-execution controls contribute no generated tokens. The mutation control contributes its actual changed output, not the unchanged expected answer. The Python driver independently checks visible hashes and accounting, but cannot recompute raw trace hashes without the original trace arrays; raw audit hashes are native evidence.

The original golden corpus contains roughly 30 million ISA/refresh tokens. Those full-trace hashes cannot equal this changed graph's internal token sequence. The exact shared contract is the golden UTF-8 BYTE token stream plus final HALT, with internal graph ALU HALTs projected out. This does not claim unchanged ISA execution or equality of old and new internal step counts.

## Controls and lifecycle scope

All five controls are appended by the driver to its single child invocation:

- Change one decimal operand digit of a real successful named addition, keeping its golden expected output. The output/token comparison must detect a mismatch while the mutated execution still halts with a valid actual trace.
- Unknown named plan: require `error:GraphPlan` before any evaluator steps.
- A 63-digit input literal: require `error:GraphLiteral` before any evaluator steps.
- A generic operation with width 63: require `error:GraphTemplate` before any evaluator steps.
- A named request that also supplies operations: require `error:GraphNamedPlanOverride` before any evaluator steps.

The mutation row alone uses `expect_mismatch:true`. It passes only when execution halts, its actual trace is structurally valid, and its output differs from the unchanged golden. A crash, refusal, malformed trace or unexpected success cannot satisfy this control. It is an operand mutation, not an offline weight/pack mutation test.

`--lifecycle` executes the real graph backend interface with two worker slots and successful corpus seeds from two different first-operation families. Its nine counted checks are: Busy on a third admitted job; both concurrent outputs matching their solo corpus results; no second successful delivery; budget zero returning budget with zero steps/bytes; a fresh run after budget exhaustion; cancellation followed by successful stream reuse; release followed by successful stream reuse; clean reuse after releasing prior streams; and an already-set Machine cancellation flag returning cancelled with zero steps/bytes.

Polling and slot-recovery loops have a 300-second timeout and sleep between polls. Immediate cancel/release may race worker completion; these checks establish cleanup/reuse, not a deterministic mid-ALU cancellation interruption. Token evidence comes from the sequential fresh Machines; backend jobs intentionally do not enable tracing. The backend checks establish concurrent result/state isolation and lifecycle behavior separately. If `Program.kind.graph` has not been staged, lifecycle fails explicitly with `GraphKindUnavailable`; the driver does not accept a skip.

## Boot and compilation claims

The driver requires a fresh-process Bank.load wall time strictly below 1,000,000,000 ns. `boot_process_cpu_ns` is process CPU time for the same bank load. Per-case `machine_boot_ns` measures binding setup; `run_ns` and `cpu_ns` measure Machine.run. Summary run times cover the sequential case loop and receipt bookkeeping, excluding bank load, lifecycle checks and final JSON serialization. Whole-process wall time includes those phases and process launch.

This is cold process startup, not cold filesystem cache latency. Hashing deployment packs during the driver's structural audit may warm the filesystem cache. No cache purge, GPU speed claim or service throughput claim is implied.

`request_compiled_graphs:0` is a structural assertion: the probe calls Bank.load once, bank weights come only from prepared pack loading, and requests execute Machine bindings/calls. The driver checks the actual staged runtime entrypoint sources for direct circuit-loader/compiler calls and records their hashes alongside the binary hash. Frozen model modules can still contain unused loader methods; the audit is not dynamic compiler instrumentation and does not claim a complete call-graph proof or source-to-binary attestation. Named corpus packets also forbid request-provided operations. Offline pack construction and corpus plan construction are excluded from request timings.

The qualified receipts cover 6,356 routed goldens, all20 routed samples,236 extra native numeric/loop cases and246 upstream IR cases. The driver also executes three boot controls: altered payload checksum, stale schema fingerprint and a valid wrong-width prepared pack substitution. Each must fail its specific boot gate. See runner-graph-results.md for measurements, artifact locations and integration limits.
