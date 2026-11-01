# P2 circuit export tooling

These development tools export the compiled Qwenputer RV32IM + Zba circuit for
TensorFold's optional in-model runner. They do not execute programs in the serving
engine, change model weights, or add a Python dependency to native inference.
The exported file contains sparse SwiGLU weights, generic DeltaNet head projections,
all initial state matrices and the program images. It includes no native ISA executor.

Qwenputer is by Divyansh Agrawal, MIT, pinned to
`bec1a0c9654ac8151b6f36ed0154a19c53141384`. Retain
[its license](../LICENSES/Qwenputer-MIT.txt) with redistributed circuit artifacts.
The [v1 wire specification](runner-circuit-format.md) defines exact units and evaluation order.
The [golden format](runner-goldens.md) defines reference token traces and checkpoints.

## Export and inspect

Use a clean pinned upstream checkout and Python 3.11+ with NumPy (verified with 1.26.4).
Only the development compiler and replay comparison require NumPy. The inspector
and reference golden exporter use the standard library. No Torch, Qwen checkpoint,
tokenizer, CUDA or GPU is required.

```sh
git clone https://github.com/divagr18/qwenputer /path/to/qwenputer
git -C /path/to/qwenputer checkout bec1a0c9654ac8151b6f36ed0154a19c53141384
python3 tools/runner_export.py --qwenputer /path/to/qwenputer --output /path/to/calc.tfpc
python3 tools/runner_circuit.py /path/to/calc.tfpc
```

Default output is calc firmware with an empty expression at RAM 0x100. Use `--expr`
for a ready-to-run snapshot. Expressions must be ASCII, NUL-free and at most 255
bytes (plus the terminating NUL). The frontend can patch the RAM heads as specified
in the wire contract. The 62-digit result limit is a firmware limit, not a file limit.

```sh
python3 tools/runner_export.py --qwenputer /path/to/qwenputer \
  --expr '123456789012345678901234567890*987654321098765432109876543210' \
  --output /path/to/multiply.tfpc
python3 tools/runner_export_check.py /path/to/multiply.tfpc --qwenputer /path/to/qwenputer
```

The checker reads the actual checksummed file, restores its graph and initial state,
and compares every cell of all initial state matrices and every residual dimension
against upstream's original IR on every
reference trace token. It also decodes the serialized output code and checks each
next token against the reference machine. This proves serialization fidelity;
CPU/GPU runtime qualification belongs to the native evaluator's separate gates.

For C, use a static ELF32 little-endian RISC-V executable built with rv32im_zba,
ILP32, and the upstream ROM/DROM/RAM linker map (see the golden instructions).

```sh
python3 tools/runner_export.py --qwenputer /path/to/qwenputer \
  --elf /path/to/program.elf --output /path/to/program.tfpc
python3 tools/runner_export_check.py /path/to/program.tfpc \
  --qwenputer /path/to/qwenputer --elf /path/to/program.elf
```

The exporter refuses dirty/revised upstream sources, malformed/overlapping ELF
segments, dynamic images and unsupported address maps. ROM words are predecoded
by the upstream compiler into head state, so merely replacing raw image metadata
in an existing circuit does not load a different program. Re-export its snapshot.
The JSON payload and complete-file SHA256 sidecar are deterministic; outputs are
written through a temporary file. Keep generated circuits and traces outside the
source checkout, and remove scratch output after verification.

The compiler and replay-check CLIs restart with `PYTHONHASHSEED=0` before importing
upstream modules. Upstream constructs some sparse expression orders from sets;
pinning the seed preserves its accumulation order and makes independent exports
byte-identical. This does not change instructions or weights.
