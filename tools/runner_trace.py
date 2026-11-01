#!/usr/bin/env python3
"""Export token-exact RV32 reference traces from a supplied, pinned Qwenputer checkout."""

import argparse
from collections import deque
from contextlib import ExitStack
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


UPSTREAM = "bec1a0c9654ac8151b6f36ed0154a19c53141384"
MULTIPLY = "123456789012345678901234567890*987654321098765432109876543210"
PRODUCT = "121932631137021795226185032733622923332237463801111263526900\n"
TOKEN_LIMIT = 10_000_000
WINDOW = 8
ARTIFACTS = ("cases.jsonl.gz", "tokens.leu32.gz", "loop.elf", "manifest.json")


def json_line(value):
    """Encode canonical compact JSON with one terminating newline."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def file_hash(path):
    """Hash a file without loading its entire contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_upstream(root):
    """Import only the CPU reference modules from the supplied checkout."""
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM:
        raise ValueError(f"Qwenputer revision must be {UPSTREAM}; got {revision}")
    relative = Path("experimental/riscv")
    sources = [relative / "chipformer" / "__init__.py"]
    sources += [relative / "chipformer" / "rv" / name for name in (
        "__init__.py", "asm.py", "calc.py", "ref.py", "spec.py", "elf.py")]
    sources.append(relative / "tests" / "data" / "rv" / "calc.elf")
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--",
                    *map(str, sources)], check=True, stdout=subprocess.DEVNULL)
    sys.path.insert(0, str(root / relative))
    sys.dont_write_bytecode = True
    modules = [importlib.import_module(f"chipformer.rv.{name}") for name in ("calc", "ref", "spec", "elf")]
    for module in modules:
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError("reference module was imported from outside the supplied checkout")
    return modules, {str(path): file_hash(root / path) for path in sources}


def gzip_stream(stack, path):
    """Open a deterministic gzip stream with no filename or timestamp header."""
    raw = stack.enter_context(path.open("wb"))
    return stack.enter_context(gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6))


def record_case(machine, spec, token_stream, interval, case_id):
    """Hash every token prefix and retain checkpoints at exact token counts."""
    digest = hashlib.sha256()
    window = deque(maxlen=WINDOW)
    checkpoints = []
    count = 0
    refreshes = 0

    def checkpoint():
        return {"token_count": count, "generated_steps": count - 1,
                "prefix_sha256": digest.hexdigest(), "window_start": count - len(window),
                "window": list(window)}

    def append(token):
        nonlocal count, refreshes
        packed = struct.pack("<I", token)
        token_stream.write(packed)
        digest.update(packed)
        window.append(token)
        count += 1
        refreshes += int(spec.family(token)[0] == "R")
        if count == 1 or count % interval == 0:
            checkpoints.append(checkpoint())

    append(spec.tok("GO"))
    while not machine.halted:
        record = machine.record()
        if not record or count + len(record) > TOKEN_LIMIT:
            raise ValueError(f"{case_id}: empty record or trace exceeds {TOKEN_LIMIT} tokens")
        for token in record:
            append(token)
    if checkpoints[-1]["token_count"] != count:
        checkpoints.append(checkpoint())
    if machine.fault is not None or machine.exit_code is None:
        raise ValueError(f"{case_id}: fault={machine.fault}, exit_code={machine.exit_code}")
    return {"instruction_count": machine.steps, "trace_token_count": count, "generated_steps": count - 1,
            "refresh_count": refreshes, "trace_sha256": digest.hexdigest(), "checkpoints": checkpoints,
            "final_output": bytes(machine.out).decode("utf-8"), "final_output_hex": bytes(machine.out).hex(),
            "exit_code": machine.exit_code}


def export(args):
    """Publish the manifest only after every reference run and output comparison succeeds."""
    (calc, ref, spec, elf), hashes = load_upstream(args.qwenputer.resolve())
    if calc.expected(MULTIPLY) != PRODUCT:
        raise ValueError("upstream oracle disagrees with the required exact product")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        if (args.output / name).exists():
            raise ValueError(f"output already contains {name}; choose an unused golden destination")
    total_tokens = total_instructions = case_count = machine_limits = 0
    with tempfile.TemporaryDirectory(prefix=".runner-goldens-", dir=args.output) as temporary:
        staging = Path(temporary)
        with ExitStack() as stack:
            rows = gzip_stream(stack, staging / "cases.jsonl.gz")
            tokens = gzip_stream(stack, staging / "tokens.leu32.gz")

            def emit(case_id, kind, operation, expression=None, program=None):
                nonlocal total_tokens, total_instructions, case_count, machine_limits
                if expression is not None:
                    machine = calc.machine(expression)
                    expected = calc.expected(expression)
                    input_data = {"expression": expression, "ram_address": calc.CALC_IN,
                                  "bytes_hex": (expression.encode("utf-8") + b"\0").hex()}
                else:
                    machine = ref.Machine.from_program(program)
                    expected = "285\n"
                    input_data = {"elf": "loop.elf", "stdin_hex": ""}
                row = record_case(machine, spec, tokens, args.checkpoint_interval, case_id)
                limited = kind == "calc" and row["final_output"] == "error (too complex: split it)\n"
                if limited and not case_id.startswith("calc-"):
                    raise ValueError(f"{case_id}: required proof hit a machine limit")
                if row["final_output"] != expected and not limited:
                    raise ValueError(f"{case_id}: {input_data}: got {row['final_output']!r}, expected {expected!r}")
                expected_exit = int(kind == "calc" and (limited or expected == "error\n"))
                if row["exit_code"] != expected_exit:
                    raise ValueError(f"{case_id}: exit {row['exit_code']}, expected {expected_exit}")
                row.update({"id": case_id, "kind": kind, "operation": operation, "input": input_data,
                            "expected_output": expected, "token_offset_bytes": total_tokens * 4,
                            "outcome": "machine_limit" if limited else "ok"})
                machine_limits += int(limited)
                rows.write(json_line(row))
                total_tokens += row["trace_token_count"]
                total_instructions += row["instruction_count"]
                case_count += 1

            for index, (operation, expression) in enumerate(calc.cases(args.cases, args.seed)):
                emit(f"calc-{index:05d}", "calc", operation, expression=expression)
                if (index + 1) % 100 == 0:
                    print(f"verified {index + 1}/{args.cases} calc cases", file=sys.stderr, flush=True)
            emit("multiply-30digit", "calc", "mul", expression=MULTIPLY)
            if args.loop_elf is not None:
                shutil.copyfile(args.loop_elf, staging / "loop.elf")
                emit("loop-squares", "elf", "sum_squares", program=elf.program(staging / "loop.elf"))
        filenames = ["cases.jsonl.gz", "tokens.leu32.gz"]
        if args.loop_elf is not None:
            filenames.append("loop.elf")
        manifest = {
            "format": "tensorfold-runner-goldens-v1", "upstream_commit": UPSTREAM,
            "source": {"repository": "https://github.com/divagr18/qwenputer", "license": "MIT",
                       "credit": "Divyansh Agrawal",
                       "license_text": (args.qwenputer / "LICENSE").read_text(encoding="utf-8")},
            "upstream_files_sha256": hashes, "seed": args.seed, "random_calc_cases": args.cases,
            "checkpoint_interval": args.checkpoint_interval, "window_tokens": WINDOW,
            "case_count": case_count, "trace_token_count": total_tokens,
            "machine_limit_cases": machine_limits,
            "instruction_count": total_instructions, "token_encoding": "little-endian-u32",
            "tokens_file": "tokens.leu32.gz", "cases_file": "cases.jsonl.gz",
            "files": {name: {"sha256": file_hash(staging / name), "bytes": (staging / name).stat().st_size}
                      for name in filenames},
        }
        (staging / "manifest.json").write_bytes(json_line(manifest))
        for name in [*filenames, "manifest.json"]:
            os.replace(staging / name, args.output / name)
    print(json.dumps({"cases": case_count, "trace_tokens": total_tokens, "instructions": total_instructions,
                      "machine_limit_cases": machine_limits}))


def main():
    """Parse the standalone golden-export CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwenputer", type=Path, required=True, help="upstream checkout at the pinned revision")
    parser.add_argument("--output", type=Path, required=True, help="directory without existing golden artifacts")
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=256)
    parser.add_argument("--loop-elf", type=Path, help="ROM-map RV32 ELF; must print 285 followed by newline")
    args = parser.parse_args()
    if args.cases < 0 or args.checkpoint_interval < 1:
        parser.error("--cases must be nonnegative and --checkpoint-interval must be positive")
    try:
        export(args)
    except (OSError, ValueError, AssertionError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"runner_trace: {error}\n")


if __name__ == "__main__":
    main()
