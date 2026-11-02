# Optional 256-digit arithmetic bank

`runner_graph_extend.py` extends a qualified graph-v4 bank offline with five
optional templates: `add`, `sub`, `mul`, `emit`, and `format`, all width 256.
The output templates serve named plans and whole-cell output respectively.
The extension adds 1,003 plans: `add-256`, `sub-256`, `mul-256`, and
`pow-0-256` through `pow-999-256`. Wide division and other integer algorithms
are deferred. Old payloads and manifest entries remain unchanged.

```sh
python3 tools/runner_graph_extend.py \
  --bank graph-v4/bank --qwenputer upstream-qwenputer \
  --baseline baseline-template.tfimr --packer ./runner-graph-pack --output graph-v5/bank
```

The output must be empty. Resume with the same arguments and `--resume`;
compiler, baseline, packer, upstream or old-bank checksum changes refuse resume.
If all five packs completed before a publication interruption, `--publish-only`
validates their recorded checksums and publishes without invoking a builder or
packer. It permits only entrypoint-file drift, retaining the original weight
compiler identity and recording the new publisher separately; every numerical
compiler module and all other inputs must match. Incomplete packs refuse.
Qwenputer credit, pinned revision and MIT text remain in the bank.

New static arithmetic invocations are checked. Multiplication computes low
256 digits and flags every discarded high partial product, boundary tens digit
and compressor carry. Nonnegative partial products make these overflow flags
exact. Overflow refuses the entire cell before output. Host code binds operands;
it does not calculate answers. Power uses the existing multiplication schedule.

The runtime widens public width fields to u16. Reserved zero in the existing
prepared u8 width field represents 256; old width codes retain their meaning.
Nine groups carry the 256 decimal digits. Optional templates reuse temporary
registers after their last use while preserving stored node order and head
barriers. Only these templates use the compactor; old payloads are copied.
Zero-weight dependencies preserve overwrite boundaries. Wide output comparisons
clamp the lower tail before applying Qwenputer's bounded exact comparators.

Builders and packers run sequentially in separate processes with address-space
caps of 1,792 MiB; the parent cap is 192 MiB while children run and
512 MiB for final metadata serialization after every child exits. BLAS threads are one. Run under
an external aggregate 2 GiB guard. Allocation failures stop publication.
Address-space caps alone do not establish measured memory usage or correctness.

Qualification runs actual BYTE/HALT tokens against independent CPython outputs,
including 129-, 191- and 256-digit results, 3**400, signed arithmetic, three
independent multiply overflow mechanisms, input mutation and wrong-result controls.
Compare fresh and reused execution, cancellation, raw trace hashes and old-bank
goldens. Lazy boot must prepare zero templates and stay below one second;
request-time graph compilation remains zero. A valid numerical weight mutation
must be detected; a redundant scheduling dependency need not alter output.

`runner-graph-wide-proof.patch` adds integration tests named `runner proof:` so
`test-runner-proof-host` discovers them. It corrects the historical 3**400
expectation, exercises all three arithmetic families at each boundary, and
checks that overflow owns no output. Apply to the documented integration base;
run the real target with the circuit and bank environment variables set.

Run the project's formatter, lean finding-identity comparison, codec checks,
combined host/kernel/golden/fuzz gates, security gates and plain build before
publication. CPU qualification does not qualify the GPU evaluator.
