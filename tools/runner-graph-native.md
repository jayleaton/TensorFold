# Native graph runner contract

The graph backend owns no arithmetic implementation. It binds decimal literals to tensor rows, invokes offline prepared T2 circuits, and copies output rows directly into later inputs. T2 evaluates every arithmetic result and error flag.

## Request packet and frontend builder

UTF-8 JSON, at most 1 MiB:

```json
{"version":1,"values":["12","-3"],"ops":[{"op":"add","a":0,"b":1,"width":4}],"result":2}
```

References index literals first, then operation outputs in operation order. An operation has `op`, `width`, `a` (default 0), `b` (optional), optional `args` (overrides a/b), and `outputs` (default 1). `args` supports division stage calls with more than two logical inputs. Each output appends a reference. Forward references, unknown templates and widths outside 1,2,4,8,16,32,62 are refused. At most 4096 operations and 8192 total values are accepted. The runner automatically invokes `emit` at width 62 on `result`; a bank must include that template. Explicit emit calls in the packet are refused. Frontends may serialize the public `Packet`/`Op` structs with `Builder.encode`; building a packet performs no execution or compilation.

Each logical value is two 128-float rows holding 62 decimal digits, least significant digit first. A group holds 31 digits as four binary bits per digit in columns 2..125. Sign and circuit error occupy columns 126/127 of group zero. Columns 0/1 retain reference ballast. Literals populate digit bits, sign and fixed ballast. Raw output rows, including error flags, are copied without decoding or arithmetic.

## Offline packer

Build the main at `zig/src/core/runner_graph_pack.zig` with the same staged T2 sources and pinned Zig as the runtime. Invoke:

```sh
runner-graph-pack template.tfimr template.tfgp
```

The source is a TFIMR001 container accepted by the unchanged T2 loader. Packing calls that loader once offline, requires recognized graph input/output slots, strips unused ISA image arrays/RAM mappings, and serializes the prepared batches, sparse/shared forms and runtime fields from the frozen T2 model. The binary preserves exact float bits and stored ordering. Runtime loading never calls the JSON circuit loader or compiles FFN/shared forms. The frozen model has no Flow or split fields.

The prepared format uses `TFGRPH01`, a little-endian payload length and SHA-256 checksum, followed by a recursive field stream. Scalars are little-endian, slices have u64 lengths, optionals have boolean presence, tagged unions have u32 enum tags, and pointers store their target. The arena field is omitted and reconstructed. Packs are tied to the pinned Zig/T2 runtime schema and must be regenerated together when that schema changes. SHA-256 checks corruption, not authenticity; packs are trusted deployment artifacts.

## Boot bank

`Bank.load(allocator, manifest_path)` reads a descriptor-only JSON manifest:

```json
{"version":1,"templates":[{"name":"add","width":4,"path":"add-4.tfgp"},{"name":"emit","width":62,"path":"emit-62.tfgp"}]}
```

Relative pack paths resolve against the manifest parent directory; absolute paths remain absolute. A bank and its basename-named companion packs can move together and load from any server working directory. Slot mapping comes from prepared `Circuit.table_names`: `in0.0`, `in0.31`, `in1.0`, `out0.0`, etc. Offsets are decimal digit offsets, not group indices. The bank validates complete contiguous logical slots for the template width. An optional manifest `inputs`/`outputs` count can assert the inferred mapping. All weights live until Bank.deinit, after backend destruction. Jobs receive fresh T2 State instances and own input tables before writing row zero.

`runner_graph.backend(allocator, &bank, workers)` adapts the existing runner API; `destroy` cancels and joins outstanding workers. The parent adds Program.kind.graph and backend selection. Until then the module compares `@tagName(program.kind)` against `graph` without referencing a missing enum field.

## Named plans, constants and raw binding copies

A manifest may also contain `plans:[{name,inputs,literals,ops,result}]`. `inputs` counts request literals; `literals` is an optional array of signed decimal constants appended after them. References index request values, then plan constants, then call outputs. A named request is `{"version":1,"plan":"pow-17-62","values":["3"]}`. Its `ops` must be empty and `result` must be zero (both default); the runner uses the immutable boot-loaded plan. `Builder.named(allocator,name,values)` makes this request. A plan catalog shares prepared templates; an exponent or bounded-loop schedule never duplicates weights. Both named schedules and optional request SSA are data-only call plans, with zero FFN compilation at request time.

`bind` is a pure tensor-copy operation: `{"op":"bind","copies":[{"ref":0,"from":5,"to":0,"digits":1}],"flags":0}` creates one zero-initialized Value, copies the specified decimal digit bit ranges, and optionally copies sign/error from `flags`. Copies execute in array order; no flags are combined or computed. This supports division digit extraction and quotient assembly. `outputs` must be 1; `a`/`width` are unused for bind. Division calls use `args:[R,divisor,nextdigit],outputs:2` for divstep and `args:[Q,R,dividend,divisor],outputs:2` for divfinal. The result refs append in the documented R/Qdigit and Qsigned/Rsigned order. Generic calls support `select` with three inputs as well.

Before a call, every tensor digit bit above its template width must be zero; narrowing with nonzero bits is refused. Sign/error are always transferred. Destination input row columns 0/1 retain the template's reference constants; columns 2..127 copy from the logical Value. All intermediate rows remain binary32 tensors without text conversion.

The dependency is the parent's frozen, prior-native26-proven nine-file T2 source set, not the actively changing circuit checkout. The 80-byte prepared header is: eight-byte magic, u64 payload length, 32-byte structural schema hash, 32-byte payload SHA-256. Schema hashing includes field names, nested shapes, enum names/values, encoding version, exact Zig version and the exact frozen T2 model/state/evaluator source bytes. An ABI mismatch is refused before allocation. Regenerate packs with any T2 schema change.

`Machine.trace` may point to an allocator-compatible `std.ArrayList(u32)` before `run`. The machine appends exactly each `eval.step` return token, including every ALU call's HALT and the emitter's BYTE outputs/final HALT. It adds no synthetic GO or boundary tokens. `steps` counts those same evaluator calls globally across the plan. The caller owns and frees trace storage with the machine's allocator. Backend jobs leave tracing disabled.

Pinned Qwenputer row amplitudes are `A_BIT=128` for set digit/sign/error bits and `A_REF=16777216` in reference columns 0/1. Literal encoding uses those amplitudes; copying never rescales or decodes intermediate rows. Bank loading refuses a different row amplitude schema.

Scalar leaf slices use a u64 element count followed by an encoding byte: 0 means contiguous little-endian raw bits; 1 means repeated `(u32 run_length, scalar_bits)` until the element count is filled. The offline writer chooses RLE only when smaller. Boot bulk-copies raw slices and fills repeated runs; zero tables and long Sparse.starts runs avoid redundant serialized bytes. The packer refuses inputs lacking `in0.0`/`out0.0` graph slots before discarding unused ISA image fields; it is exclusively a graph-template packer.

Named plan arities/references are checked once at boot, and constant literals become immutable typed rows then. Requests select that prepared plan, parse their bound literals, and copy prepared constants into fresh job storage.

## Public frontend selectors and prepared identity

`Builder.binary(allocator, family, lhs, rhs)` selects a preloaded named plan for add/sub/mul/div/mod/pow and binds signed decimal literals. Powers require a literal control exponent in 0..999; the packet binds only the base. `Builder.loop(allocator, shape, bound, values)` selects sum/squares/factorial/fib/list_sum with bound0..128 and enforces the list arity. These helpers validate and serialize bindings only; their runtime cost contains no graph compiler.

The prepared payload root is `{ name: []const u8, width: u8, circuit: Circuit }`. Both operation and digit bucket are covered by the payload checksum and must equal the manifest descriptor before the circuit is admitted. The emitter's separate `output.cursor` state table is excluded from operand-slot inference.
