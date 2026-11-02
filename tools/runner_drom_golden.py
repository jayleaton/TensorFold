#!/usr/bin/env python3
"""Generate a separate, bounded DROM golden using the served C prelude and pinned Qwenputer."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

from runner_trace import UPSTREAM, file_hash, gzip_stream, json_line, load_upstream, record_case


CASE_ID = "drom-byte-string"
OUTPUT = "circuit drom bytes\ncircuit drom bytes\n"
BUDGET = 100_000


def zig_string(source, name):
    """Read the actual Zig multiline literal rather than maintaining a second compiler prelude."""
    marker = f"{name} =\n"
    body = source.split(marker, 1)[1].split("\n;", 1)[0]
    lines = body.splitlines()
    if not lines or any(not line.startswith(" " * 4 + chr(92) * 2) for line in lines):
        raise ValueError(f"unsupported {name} literal")
    return "\n".join(line[6:] for line in lines) + "\n"


def inspect_elf(path):
    """Require real DROM data and both signed and unsigned RV32 byte-load instructions."""
    data = path.read_bytes()
    if data[:6] != b"\x7fELF\x01\x01" or struct.unpack_from("<H", data, 18)[0] != 243:
        raise ValueError("expected little-endian RV32 ELF")
    offset = struct.unpack_from("<I", data, 32)[0]
    size, count, names_index = struct.unpack_from("<HHH", data, 46)
    sections = [struct.unpack_from("<10I", data, offset + i * size) for i in range(count)]
    names = sections[names_index]
    strings = data[names[4]:names[4] + names[5]]
    found = {}
    for section in sections:
        name = strings[section[0]:].split(b"\0", 1)[0].decode("ascii")
        found[name] = section
    rodata, text = found[".rodata"], found[".text"]
    payload = data[rodata[4]:rodata[4] + rodata[5]]
    if not (0x20000 <= rodata[3] and rodata[3] + rodata[5] <= 0x40000):
        raise ValueError("rodata is outside DROM")
    if b"circuit drom bytes\n\0" not in payload or b"\x81\0" not in payload:
        raise ValueError("missing DROM string/terminator/sign probe")
    code = data[text[4]:text[4] + text[5]]
    loads = {kind: [] for kind in (0, 4)}
    for i in range(0, len(code) - 3, 4):
        word = struct.unpack_from("<I", code, i)[0]
        kind = (word >> 12) & 7
        if word & 127 == 3 and kind in loads:
            loads[kind].append(text[3] + i)
    if not all(loads.values()):
        raise ValueError("ELF must contain both LB and LBU")
    return {"rodata_address": rodata[3], "rodata_bytes": rodata[5],
            "lb_addresses": loads[0], "lbu_addresses": loads[4]}


def export(args):
    """Stage an independent supplemental corpus without opening or rewriting the frozen goldens."""
    (_, ref, spec, elf), hashes = load_upstream(args.qwenputer.resolve())
    compiler = args.compiler_source.read_text()
    runtime = zig_string(compiler, "runtime")
    linker = zig_string(compiler, "linker_script")
    fixture = Path(__file__).with_suffix(".c")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("supplemental destination must be empty")
    with tempfile.TemporaryDirectory(prefix=".drom-", dir=args.output) as temporary:
        staging = Path(temporary)
        (staging / "program.c").write_text(runtime + "\n" + fixture.read_text())
        (staging / "rom.ld").write_text(linker)
        command = ([str(args.zig), "cc", "-target", "riscv32-freestanding", "-mcpu=generic_rv32+m+zba"]
                   if args.zig else [args.clang, "--target=riscv32-unknown-elf", "-march=rv32im_zba"])
        command += ["-mabi=ilp32",
                   "-O0", "-fno-sanitize=all", "-ffreestanding", "-fno-builtin", "-nostdinc", "-nostdlib",
                   "-fno-stack-protector", "-fno-pic", "-mno-relax", "-fno-unwind-tables",
                   "-fno-asynchronous-unwind-tables",
                   *([f"--ld-path={args.linker}"] if args.linker else ([] if args.zig else ["-fuse-ld=lld"])),
                   *([] if args.zig else ["-Wl,--no-relax", "-Wl,--build-id=none"]), "-Wl,-T,rom.ld",
                   "-x", "c", "program.c", "-o", "drom.elf"]
        subprocess.run(command, cwd=staging, check=True, timeout=30)
        inspection = inspect_elf(staging / "drom.elf")
        if args.objdump:
            disassembly = subprocess.check_output([args.objdump, "-d", "drom.elf"], cwd=staging,
                                                 text=True, timeout=30)
            if "\tlb\t" not in disassembly or "\tlbu\t" not in disassembly:
                raise ValueError("disassembler did not confirm LB and LBU")
        else:
            disassembly = "# Byte-load instruction addresses decoded from RV32 ELF .text\n"
            for kind, name in (("lb_addresses", "lb"), ("lbu_addresses", "lbu")):
                disassembly += "".join(f"{address:08x}: {name}\n" for address in inspection[kind])
        (staging / "disassembly.txt").write_text(disassembly)
        machine = ref.Machine.from_program(elf.program(staging / "drom.elf"))
        with ExitStack() as stack:
            tokens = gzip_stream(stack, staging / "tokens.leu32.gz")
            row = record_case(machine, spec, tokens, 256, CASE_ID, token_limit=BUDGET + 1)
        if row["final_output"] != OUTPUT or row["exit_code"] != 0:
            raise ValueError("DROM reference output or sign extension mismatch")
        row.update(id=CASE_ID, kind="elf", operation="drom_lb_lbu", expected_output=OUTPUT,
                   outcome="ok", token_offset_bytes=0, input={"elf": "drom.elf", "stdin_hex": ""})
        with ExitStack() as stack:
            gzip_stream(stack, staging / "cases.jsonl.gz").write(json_line(row))
        files = ["drom.elf", "program.c", "rom.ld", "disassembly.txt", "cases.jsonl.gz", "tokens.leu32.gz"]
        manifest = {"format": "tensorfold-runner-drom-goldens-v1", "upstream_commit": UPSTREAM,
                    "source": {"repository": "https://github.com/divagr18/qwenputer", "license": "MIT",
                               "credit": "Divyansh Agrawal",
                               "license_text": (args.qwenputer / "LICENSE").read_text(encoding="utf-8")},
                    "upstream_files_sha256": hashes, "compiler_source_sha256": file_hash(args.compiler_source),
                    "fixture_sha256": file_hash(fixture), "compile_command": command,
                    "case_count": 1, "checkpoint_interval": 256, "window_tokens": 8,
                    "step_budget": BUDGET, "token_encoding": "little-endian-u32", "inspection": inspection,
                    "files": {name: {"sha256": file_hash(staging / name),
                                     "bytes": (staging / name).stat().st_size} for name in files}}
        (staging / "manifest.json").write_bytes(json_line(manifest))
        for name in [*files, "manifest.json"]:
            os.replace(staging / name, args.output / name)
    print(json.dumps({"id": CASE_ID, "generated_steps": row["generated_steps"],
                      "trace_sha256": row["trace_sha256"], "final_output_hex": row["final_output_hex"]}))


def main():
    """Honor explicit compiler options before the same environment overrides used by the served frontend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwenputer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler-source", type=Path, required=True)
    parser.add_argument("--clang", default=os.environ.get("TF_RUNNER_CLANG", "/usr/bin/clang"))
    parser.add_argument("--linker", default=os.environ.get("TF_RUNNER_LLD"))
    parser.add_argument("--zig", type=Path, help="use Zig cc instead of clang")
    parser.add_argument("--objdump", help="optional full RV32 disassembly; byte-load decoding is always required")
    args = parser.parse_args()
    try:
        export(args)
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as error:
        parser.exit(1, f"runner_drom_golden: {error}\n")


if __name__ == "__main__":
    main()
