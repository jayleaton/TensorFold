#!/usr/bin/env python3
"""Append five optional 256-digit arithmetic/output templates to a qualified bank without recompiling old payloads."""

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys


WIDTH = 256
OPERATIONS = ("add", "sub", "mul", "emit", "format")
WORKER_MIB = 1792
PARENT_MIB = 192
SOURCES = ("runner_graph_extend.py", "runner_graph_ir.py", "runner_graph_registers.py", "runner_graph_format.py",
           "runner_graph_plans.py", "runner_graph_export.py", "runner_graph_algorithms.py",
           "runner_graph_whole.py", "runner_graph_div.py", "runner_graph_byte.py",
           "runner_export.py", "runner_trace.py", "runner_circuit.py", "runner_isolate.py")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def limited_child():
    limit = WORKER_MIB << 20
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def extension_plans():
    from runner_graph_plans import binary_plan

    plans = [binary_plan(op, 128) for op in ("add", "sub", "mul")]
    plans += [binary_plan("pow", 128, exponent) for exponent in range(1000)]
    for plan in plans:
        plan["name"] = plan["name"].removesuffix("-128") + f"-{WIDTH}"
        for call in plan["ops"]:
            call.update(width=WIDTH, checked=True)
    return plans


def append_manifest(raw, templates, plans):
    """Insert only new array elements, retaining every byte of every old entry."""
    decoder = json.JSONDecoder()
    document = json.loads(raw)
    additions = {"templates": templates, "plans": plans}
    edits, seen = [], set()
    position = raw.index("{") + 1
    while True:
        while raw[position].isspace() or raw[position] == ",":
            position += 1
        if raw[position] == "}":
            break
        key, position = decoder.raw_decode(raw, position)
        if key in seen:
            raise ValueError("duplicate manifest key")
        seen.add(key)
        while raw[position].isspace():
            position += 1
        if raw[position] != ":":
            raise ValueError("invalid manifest key separator")
        position += 1
        while raw[position].isspace():
            position += 1
        value, end = decoder.raw_decode(raw, position)
        if key in additions:
            if not isinstance(value, list):
                raise ValueError(f"manifest {key} must be an array")
            insertion = ("," if value else "") + ",".join(
                json.dumps(item, sort_keys=True, separators=(",", ":")) for item in additions[key])
            edits.append((end - 1, insertion))
        position = end
    if not additions.keys() <= seen:
        raise ValueError("manifest requires templates and plans")
    for position, insertion in sorted(edits, reverse=True):
        raw = raw[:position] + insertion + raw[position:]
    extended = json.loads(raw)
    for key in additions:
        if extended[key] != document[key] + additions[key]:
            raise ValueError("manifest append changed an existing entry")
    return raw


def streamed_circuit(doc, path):
    """Stream canonical JSON to avoid simultaneous document, JSON string and payload copies."""
    from runner_circuit import validate
    from runner_export import HEADER, MAGIC

    validate(doc)
    temporary = path.with_name(path.name + ".tmp")
    payload_hash, length = hashlib.sha256(), 0
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    with temporary.open("wb") as stream:
        stream.write(bytes(HEADER.size))
        for piece in encoder.iterencode(doc):
            block = piece.encode("utf-8")
            stream.write(block)
            payload_hash.update(block)
            length += len(block)
        stream.seek(0)
        stream.write(HEADER.pack(MAGIC, 1, HEADER.size, length, 0, 0, payload_hash.digest()))
    temporary.replace(path)
    info = {"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path),
            "payload_sha256": payload_hash.hexdigest(), "stats": doc["stats"]}
    return info


def build_one(args):
    limited_child()
    from runner_circuit import read_circuit
    from runner_graph_export import bridge
    from runner_graph_ir import build_template
    from runner_graph_registers import compact

    baseline = read_circuit(args.baseline)
    source, program = baseline["source"], baseline["program"]
    del baseline
    gc.collect()
    doc, slots = build_template(args.qwenputer, args.worker, WIDTH)
    gc.collect()
    registers = compact(doc)
    doc["register_allocation"] = registers
    if doc["source"] != source:
        raise ValueError("baseline and template upstream provenance differ")
    if WIDTH + (WIDTH - 1) // 3 + 8 >= 1 << doc["output"]["payload_bits"]:
        raise ValueError("wide control bounds exceed the token payload")
    info = streamed_circuit(bridge(doc, {"program": program}), args.output / f"{args.worker}-{WIDTH}.tfimr")
    atomic_json(args.output / f"{args.worker}-{WIDTH}.source.json",
                {"source": info, "slots": slots, "register_allocation": registers})


def bank_snapshot(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("input bank must contain regular files, not symlinks")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = digest(path)
    return files


def copy_verified(source, target, expected):
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        shutil.copyfile(source, temporary)
        if digest(temporary) != expected:
            raise ValueError(f"input changed during copy: {source.name}")
        temporary.replace(target)
    if digest(target) != expected:
        raise ValueError(f"resume payload checksum mismatch: {target.name}")


def input_identity(args):
    from runner_export import UPSTREAM

    revision = subprocess.check_output(["git", "-C", str(args.qwenputer), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(args.qwenputer), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if revision != UPSTREAM or dirty:
        raise ValueError("Qwenputer must be clean at the pinned upstream commit")
    names = subprocess.check_output(
        ["git", "-C", str(args.qwenputer), "ls-files", "-z"], text=True).split("\0")
    upstream = {name: digest(args.qwenputer / name) for name in names if name}
    return {"format": "tensorfold.runner.graph-extension-build", "version": 1,
            "width": WIDTH, "operations": list(OPERATIONS), "old_bank": bank_snapshot(args.bank),
            "compiler_sources": {name: digest(Path(__file__).with_name(name)) for name in SOURCES},
            "qwenputer_commit": revision, "qwenputer_sources": upstream,
            "baseline_sha256": digest(args.baseline), "packer_sha256": digest(args.packer)}


def validate_base(args, identity):
    raw = (args.bank / "bank.json").read_bytes().decode("utf-8")
    manifest = json.loads(raw)
    provenance = json.loads((args.bank / "provenance.json").read_text())
    if manifest["version"] != 1 or not manifest["templates"] or not manifest["plans"]:
        raise ValueError("expected an existing version-1 graph bank")
    if provenance["source"]["license"] != "MIT":
        raise ValueError("input bank must retain Qwenputer MIT provenance")
    if (args.bank / "Qwenputer-MIT.txt").read_bytes() != provenance["source"]["license_text"].encode():
        raise ValueError("bank MIT text differs from upstream provenance")
    receipts = {item["name"]: item for item in provenance["templates"]}
    paths = set()
    for template in manifest["templates"]:
        path = Path(template["path"])
        if path.is_absolute() or ".." in path.parts or path.as_posix() not in identity["old_bank"]:
            raise ValueError("unsafe or missing old template payload")
        if len(path.parts) != 1 or path.suffix != ".tfgp":
            raise ValueError("extension requires the qualified bank's flat packed-payload layout")
        if path.name in {f"{op}-{WIDTH}.tfgp" for op in OPERATIONS}:
            raise ValueError("old payload path collides with a new template")
        receipt = receipts[f"{template['name']}-{template['width']}"]
        if identity["old_bank"][path.as_posix()] != receipt["packed_sha256"]:
            raise ValueError("old payload does not match its qualified provenance")
        paths.add(path.as_posix())
    if paths != {name for name in identity["old_bank"] if name.endswith(".tfgp")}:
        raise ValueError("old bank contains unreferenced packed payloads")
    if any(item["width"] == WIDTH for item in manifest["templates"]):
        raise ValueError("input bank already contains optional-width templates")
    plans = extension_plans()
    if {item["name"] for item in plans} & {item["name"] for item in manifest["plans"]}:
        raise ValueError("new plan names collide with old plans")
    return raw, manifest, provenance, paths, plans


def export(args):
    resource.setrlimit(resource.RLIMIT_AS, (PARENT_MIB << 20, WORKER_MIB << 20))
    if args.output == args.bank or args.output.is_relative_to(args.bank) or args.bank.is_relative_to(args.output):
        raise ValueError("output and input bank directories must be separate")
    current_identity = input_identity(args)
    identity = current_identity
    raw, manifest, provenance, paths, plans = validate_base(args, identity)
    if (args.qwenputer / "LICENSE").read_bytes() != (args.bank / "Qwenputer-MIT.txt").read_bytes():
        raise ValueError("old bank and pinned checkout MIT texts differ")
    args.output.mkdir(parents=True, exist_ok=True)
    build_file = args.output / "extension-build.json"
    if args.resume or args.publish_only:
        if not build_file.exists():
            raise ValueError("resume compiler identity missing")
        previous = json.loads(build_file.read_text())
        if args.publish_only:
            before, after = copy.deepcopy(previous), copy.deepcopy(identity)
            for item in (before, after):
                del item["compiler_sources"][Path(__file__).name]
            if before != after:
                raise ValueError("publication numerical compiler/input identity changed")
            identity = previous
        elif previous != identity:
            raise ValueError("resume input/compiler identity changed")
    elif any(args.output.iterdir()):
        raise ValueError("output must be a new or empty directory")
    else:
        atomic_json(build_file, identity)
    for name, checksum in identity["old_bank"].items():
        if name not in paths:
            copy_verified(args.bank / name, args.output / "extension-base" / name, checksum)
    for name in sorted(paths | {"Qwenputer-MIT.txt"}):
        copy_verified(args.bank / name, args.output / name, identity["old_bank"][name])
    progress_file = args.output / "extension-progress.json"
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else []
    if [row["name"] for row in progress] != [f"{op}-{WIDTH}" for op in OPERATIONS[:len(progress)]]:
        raise ValueError("invalid extension progress order")
    if args.publish_only and len(progress) != len(OPERATIONS):
        raise ValueError("publication requires every template already packed")
    descriptors, receipts = [], []
    for index, op in enumerate(OPERATIONS):
        stem = f"{op}-{WIDTH}"
        packed = args.output / f"{stem}.tfgp"
        if index < len(progress):
            receipt = progress[index]
            if digest(packed) != receipt["packed_sha256"]:
                raise ValueError(f"resume checksum mismatch: {stem}")
        else:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", op,
                       "--qwenputer", str(args.qwenputer), "--baseline", str(args.baseline),
                       "--output", str(args.output)]
            subprocess.run(command, check=True, preexec_fn=limited_child)
            receipt = json.loads((args.output / f"{stem}.source.json").read_text())
            temporary = packed.with_name(packed.name + ".tmp")
            subprocess.run([str(args.packer), str(args.output / f"{stem}.tfimr"), str(temporary)],
                           check=True, preexec_fn=limited_child)
            temporary.replace(packed)
            receipt.update(name=stem, packed_bytes=packed.stat().st_size, packed_sha256=digest(packed))
            progress.append(receipt)
            atomic_json(progress_file, progress)
        slots = receipt["slots"]
        descriptors.append({"name": op, "width": WIDTH, "path": packed.name,
                            "inputs": slots["input_arity"],
                            "outputs": max(slot["index"] for slot in slots["output_slots"]) + 1})
        receipts.append(receipt)
        for suffix in (".tfimr", ".tfimr.tmp", ".source.json"):
            (args.output / f"{stem}{suffix}").unlink(missing_ok=True)
        print(json.dumps({"template": stem, "packed_bytes": receipt["packed_bytes"]}), flush=True)
    if input_identity(args) != current_identity:
        raise ValueError("inputs changed during extension")
    for name in paths:
        if digest(args.output / name) != identity["old_bank"][name]:
            raise ValueError("old output payload changed during extension")
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, WORKER_MIB << 20))
    extended = copy.deepcopy(provenance)
    extended["templates"] += receipts
    extended["plans"] = len(manifest["plans"]) + len(plans)
    extended["widths"] = [*provenance["widths"], WIDTH]
    extended["extension"] = {"identity": identity, "base_manifest": "extension-base/bank.json",
                             "new_templates": receipts, "new_plans": len(plans),
                             "request_graph_compilation": False, "measured_cost": None,
                             "publication_identity": current_identity,
                             "publication_only": args.publish_only}
    atomic_json(args.output / "provenance.json", extended)
    temporary = args.output / "bank.json.tmp"
    temporary.write_bytes(append_manifest(raw, descriptors, plans).encode("utf-8"))
    temporary.replace(args.output / "bank.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--qwenputer", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--packer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish-only", action="store_true",
                        help="Publish completed packs, retaining their original compiler provenance")
    parser.add_argument("--worker", choices=OPERATIONS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("bank", "qwenputer", "baseline", "packer", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.worker:
        build_one(args)
    elif args.bank is None or args.packer is None:
        parser.error("--bank and --packer are required")
    else:
        export(args)


if __name__ == "__main__":
    environment = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if any(os.environ.get(name) != value for name, value in environment.items()):
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)
    main()
