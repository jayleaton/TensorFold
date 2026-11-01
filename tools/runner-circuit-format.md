# TensorFold runner circuit format v1 (T1 contract)

Published 2026-11-01. Frozen v1 wire contract for T2/T4. Export tooling is
`tools/runner_export.py` on `dsv41/imr-export` (base 58bece1d3).
Source: Qwenputer https://github.com/divagr18/qwenputer at
`bec1a0c` (MIT, Copyright (c) 2026 Divyansh Agrawal). This is the standalone
RV32IM + Zba + decimal digit-file P2 IR, **not** the Qwen text bridge / P3.
No Qwen weights, embedding ballast, instruction interpreter or tokenizer is needed.
The artifact contains compiled sparse FFN/head weights and a program snapshot.

## Container and numeric representation

Exactly 64 header bytes, followed immediately by one UTF-8 JSON object, no padding/trailer.
All binary header integers are **unsigned little endian**, independent of host endian.

| Offset | Bytes | Type | Value |
| --- | --- | --- | --- |
| 0 | 8 | bytes | `TFIMR001` (ASCII) |
| 8 | 4 | u32 | version = 1 |
| 12 | 4 | u32 | header bytes = 64 |
| 16 | 8 | u64 | JSON payload byte length |
| 24 | 4 | u32 | flags = 0 |
| 28 | 4 | u32 | reserved = 0 |
| 32 | 32 | bytes | SHA-256 of exact payload bytes (raw digest) |

JSON is uncompressed, no BOM. Exporter writes deterministic compact JSON, sorted object keys,
no insignificant whitespace, no NaN/Infinity. Checksum is over the stored bytes, **not** reserialized JSON.
Reject unknown version/flags, nonzero reserved, truncation/trailing bytes, checksum mismatch,
invalid indices/counts, invalid tokens, nonfinite numbers and unsupported feature kinds.
A `.sha256` sidecar additionally identifies the complete container for transport.

Every index, dimension, level, token, count, address, ROM/RAM word and payload mask is **u32**
(except byte offsets/lengths in the header, u64); negative dimensions do not occur.
Every coefficient, tap, state value, gate scalar, scale and threshold is **IEEE-754 binary64**,
serialized as a JSON decimal that roundtrips exactly to binary64. Integer-looking coefficients
are still f64 at evaluation. JSON numbers have no byte endianness. Hex byte strings encode bytes
in increasing address order; multi-byte RV32 words are **little endian u32**.
The generic mathematical evaluator uses f64, no fast-math/reassociation. An f32 implementation
must establish token equality against the reference traces; reduced precision is not implied.

## Root object

Required keys (additional metadata may be ignored):

```
{
  "format":"tensorfold.runner.p2", "version":1,
  "source":{"repository":"https://github.com/divagr18/qwenputer",
            "commit":"<full upstream SHA>","license":"MIT","credit":"Divyansh Agrawal"},
  "stats":{"dims":D,"layers":L,"heads":H,"ffn_units":U,...},
  "constants":{"one_dim":0,"head_dim":128,"a_ref":16777216.0,"a_bit":128.0,
               "ref_cols":[0,1],"swiglu_k":128.0,"kappa":0.00000095367431640625},
  "dims":[{"name":"one","kind":"one","level":0},...],
  "features":[...], "nodes":[...],
  "tokens":{"vocab":V,"families":[...],"go":G,"halt":H,"byte_family":B},
  "output":{"family_bits":N,"payload_bits":16,"code_dims":[...],"families":[...]},
  "program":{"kind":"calc"|"elf", ...}
}
```

D = dims.len. Residual values live in **IR units**, no hidden RMSNorm or host model scale.
Each token step starts a fresh all-zero residual vector, sets dim 0 to 1 and fills token features.
Execute `nodes` in **stored array order** (the upstream IR evaluator order). `level` is original
sublayer placement metadata (odd = head, even = FFN); never reorder nodes using this value.
Per-stream persistent state consists only of head matrices and per-channel convolution history.
Initial matrices are `init` below; all channel histories start at zero. A new call resets to that
snapshot and supplies GO as its first input. After evaluating input token t, decode the next token
from `output`. Feed predictions back as the next input, through HALT. Do not evaluate beyond HALT.
The caller emits BYTE payloads 0..255 as stdout bytes. Payloads >=256 are not stdout; P3 absent.

### Token families and embedding features

`tokens.families` and `output.families` are the same ordered list of
`{"name":str,"base":u32,"size":u32}` for all P2 families and controls.
Controls have size 1. Token t belongs to the unique interval `[base,base+size)`;
its payload is `t-base`, and its family ID is its index in this list.
The output family code has N little-significance-first bits (bit 0 first).
P3's unrepresentable Q family is excluded; decoding an unused family code must fail.
After the full graph, bit[i] = (residual[code_dims[i]] > 0). The first N bits encode family ID,
the remaining 16 bits encode payload, both least-significant-bit first. Code values are
+KAPPA or -KAPPA, not integer 0/1; do not round them. A zero code coordinate is invalid.
Require payload < the selected family's size, including payload=0 for controls.

`features` is a list of small generic token-bus encodings, each writes one `dim`:

- `{"dim":d,"kind":"family","family":f}`: 1 iff input family ID is f, else 0.
- `{"dim":d,"kind":"payload_bit","family":f,"bit":b}`: payload bit b iff family f, else 0.
- `{"dim":d,"kind":"next_digit","digit":k,"qs_family":f,"qd_family":g}`:
  1 iff `(family==f && payload==k) || (family==g && (payload>>4)==k-1)`, else 0.
  k is 0..7; for k=0 the second condition is always false (do not unsigned-underflow).
- `{"dim":d,"kind":"code_bit","bit":b,"family_bits":N,"amplitude":128.0}`:
  form code = family_ID | (payload << N), then write amplitude * (2*((code>>b)&1)-1).

These features encode the input symbol only, not instruction semantics. All derived logic is
in the compiled nodes. The output code's family ordering follows upstream OUT_NAMES through SC
(excludes final Q). `output.family_bits` is upstream NFAM, **including** the spare code space.

### Sparse expressions and channels

An expression E is `[[dim,coefficient],...]`; empty = 0. No duplicate dimensions;
array order is upstream expression accumulation order. `eval(E)=sum(coef*x[dim])`.
Dim 0 is the constant ONE; there is no implicit bias.
A channel is `{"expr":E,"taps":[w0,w1,w2,w3]}`. Its result at step t is
`max(sum(wlag * projection[t-lag],lag=0..3),0)` with
`projection[t]=eval(E)` computed from the residual **at this node**, before its outputs.
Save the last 3 **pre-convolution projections**, separately for every head/q/k/v channel.
Missing q/k/v channels are zero. The max is the circuit's exact-range SiLU.

## FFN / exact-threshold SwiGLU weights

```
{"kind":"ffn","name":str,"level":u32,"out":dim,"into":bool,
 "terms":[[coefficient, up_expression, gate_expression], ...]}
```

Compute all terms before writing the output:
`y = sum(c * eval(up) * max(eval(gate),0))` in listed term order.
If into=true, `x[out] += y`; else `x[out] = y`.
These **are** sparse SwiGLU weights in IR units: emitted gate weights are K*gate,
up weights are up/K, down coefficient is c, K=128. Thus
`silu(K*b)*(a/K)*c = c*a*max(b,0)` on the compiler's exact activation range
(0, >=18, or <=-104 for the scaled SiLU input). Zero has output zero.
Do not use a generic approximate sigmoid/SiLU kernel at the threshold boundary.
Every term is an FFN unit; nodes can share an output dimension via `into`.

## DeltaNet heads, one-hot RAM, ROM and DROM

```
{"kind":"head","name":str,"level":u32,"n_cols":128,
 "levels":"bit"|"byte"|"decay","tiny":f64,
 "q":[[row,channel],...],"k":[[row,channel],...],
 "value_heads":[
   {"v":[[column,channel],...],"a":E,"beta":E|null,
    "z":[[column,E],...],"out":[[column,dim,weight],...],"into":bool,
    "init":[[row,[[column,value],...]],...]}, ...]}
```

Rows/columns are 0..127. One q/k pair can own multiple independent value-head matrices.
Empty init = zero matrix. Sparse init omits zero columns and zero rows; restore everything
else to 0, **not** an implicit ballast. Each value head's a/z/beta uses the current residual.
Project/convolve q and k once before any output writes. Then, for each value head in listed
order, project/convolve its v channels and evaluate its a/z/beta from the residual before that
value head's outputs. Apply its state update and outputs before proceeding to the next value head:

1. Decay the entire state: a=0 -> multiply by 1; a=110 -> multiply by 0.5;
   a=238 -> clear. Any other a is invalid. (Counters require half decay.)
2. beta null means 1, otherwise eval(beta); beta=1 enables write, integer beta<=0 disables it.
   Positive k has at most one nonzero row. If enabled and k active, **replace** that row with v.
   k magnitude is a selector, not an extra multiply. This is the one-hot gated delta update
   `S += k*(v-S^T*k)^T`; query/write keys are interpreted in normalized one-hot units.
3. Read **after** decay/write: sum every row with q[row]>0 (all positive q entries have equal
   magnitude). One-hot and multi-hot queries are supported; magnitude is discarded. Absent rows=0.
4. Let o be that sum. ref=o[0]^2+o[1]^2. If ref=0, normalized output is all 0 (nonzero data
   without ballast is invalid); else norm=o/sqrt(ref/128). This is the IR's exact ballast norm,
   **not** a full-data RMSNorm and has no epsilon. For each [column,dim,weight] in out:
   value=norm[column]*max(eval(z[column]),0)*weight; missing z is zero.
   into=true adds value to x[dim], else assigns. Perform outputs in listed order.

A_REF=2^24, A_BIT=2^7. Most mutable data is 0/A_BIT with ballast columns 0/1.
ROM/DROM `levels=byte` holds exact bytes 0..255 and uses paired row queries; preserve their
predecoded snapshot as provided. `levels=decay` preserves dyadic counter states.
`tiny` records the upstream exactness-check exception for very small counter output;
it does not authorize runtime clipping or threshold changes.
ROM/DROM are already encoded into named head init matrices; do not execute raw ROM instructions.
The generic core uses all heads, including pc/count/j/vcnt, registers, quotient, digit-file,
ROM, DROM and writable RAM. It is not sufficient to export only the RAM heads.

## Program image and invocation

```
{"kind":"calc"|"elf","elf_sha256":hex,"entry":u32,
 "rom_base":262144,"rom_words":65536,"rom_le_hex":hex,
 "drom_base":131072,"drom_hex":hex,"ram_base":0,"ram_bytes":32768,
 "ram_hex":hex,"input_address":256,"input_capacity":256,"expression":str|null}
```

`rom_le_hex` is the **full zero-padded original ISA ROM** (65536 LE u32 words), for provenance
and frontend program loading, not evaluation. Head init contains `ref.rom_word` predecoded bytes.
`drom_hex` is the actual read-only image (remaining address space reads 0); `ram_hex` is full 32KB.
Initial SP/register/anchor/counters/digit-file are all represented by head init; entry is metadata.
The generic `--elf` path accepts any static ELF32 LE RISC-V program fitting the upstream map,
compiled with `-march=rv32im_zba -mabi=ilp32`, upstream crt0/sys/rom.ld. No host execution of ELF.

Default calc snapshot has an empty NUL-terminated expression at RAM address 0x100. Frontend
replaces bytes there (including trailing NUL), <=256 bytes total, before GO. When editing the
snapshot in place, map RAM word wi to head h=(wi&15)|((wi>>12)<<4), value-head=(wi>>4)&1,
row=(wi>>5)&127 in head named `ram<h>`; row columns 0/1=A_REF and columns 2+i=A_BIT*((word>>i)&1).
A zero word may be absent; a nonzero word must carry ballast. Other zero RAM rows may remain absent.
`--expr` exports a ready-to-run calc snapshot with these bytes already encoded.
Loading a different ELF needs a new snapshot (export via --elf), not merely changing raw ROM metadata.

## Golden trace companion

T1 supplies deterministic gzip JSONL plus LE-u32 token blobs, generated by upstream
`ref.Machine` / `calc.cases(10000, seed=0)`. `tools/runner-goldens.md` specifies indices,
interval SHA-256 checkpoints and count semantics. Full token streams permit token-exact
comparison, not merely matching stdout. Initial input is GO, followed by all reference
predictions through HALT. stdout is exact bytes, including trailing newline.
