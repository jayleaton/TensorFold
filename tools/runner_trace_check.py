#!/usr/bin/env python3
"""Stream-verify complete runner reference goldens without importing Qwenputer."""

import argparse
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path
import struct


UPSTREAM = "bec1a0c9654ac8151b6f36ed0154a19c53141384"
MULTIPLY = "123456789012345678901234567890*987654321098765432109876543210"
PRODUCT = "121932631137021795226185032733622923332237463801111263526900\n"
OPS = ("add", "sub", "mul", "div", "mod", "gcd", "pow", "expr", "frac", "rat")
LAYOUT = (("V", 32768), ("F0", 32768), ("F1", 32768), ("F2", 4),
          ("J", 65536), ("R", 65536), ("BYTE", 512), ("FLT", 8), ("HEX", 16),
          ("QS", 16), ("QD", 128), ("BD", 256), ("NT", 1), ("ST0", 1),
          ("ST1", 1), ("SS0", 1), ("SS1", 1), ("HALT", 1), ("GO", 1), ("SC", 1))
FINAL = {"F0", "F1", "F2", "J", "BYTE", "NT", "ST0", "ST1", "SS0", "SS1", "HALT"}


def require(condition, message):
    """Reject an inconsistent artifact even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    """Require an integer count, excluding booleans and negative values."""
    require(type(value) is int and value >= minimum, f"invalid {name}: {value!r}")
    return value


def digest_file(path):
    """Compute a stored file's checksum with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def family(token):
    """Decode the documented P2 vocabulary layout, without instruction semantics."""
    base = 0
    for name, size in LAYOUT:
        if base <= token < base + size:
            return name, token - base
        base += size
    raise ValueError(f"invalid P2 token ID {token}")


def check_input(row, index, random_cases):
    """Validate ordered case identities and exact required proof inputs."""
    if index < random_cases:
        require(row["id"] == f"calc-{index:05d}", "random case ID/order mismatch")
        require(row["operation"] == OPS[index % len(OPS)], "random operation/order mismatch")
    elif index == random_cases:
        require(row["id"] == "multiply-30digit" and row["operation"] == "mul", "missing required multiply")
        require(row["input"]["expression"] == MULTIPLY, "required multiply input mismatch")
        require(row["expected_output"] == PRODUCT, "required multiply product mismatch")
    elif index == random_cases + 1:
        require(row["id"] == "loop-squares" and row["operation"] == "sum_squares", "missing required loop")
        require(row["kind"] == "elf", "loop must use an ELF snapshot")
        require(row["input"] == {"elf": "loop.elf", "stdin_hex": ""}, "loop input mismatch")
        require(row["expected_output"] == "285\n", "required loop result mismatch")
        return
    else:
        raise ValueError("unexpected extra case")
    require(row["kind"] == "calc", "arithmetic case must use calc snapshot")
    expression = row["input"]["expression"]
    require(isinstance(expression, str), "expression must be a string")
    require(row["input"]["ram_address"] == 256, "calc input RAM address mismatch")
    encoded = expression.encode("utf-8") + b"\0"
    require(len(encoded) <= 256, "calc input exceeds capacity")
    require(row["input"]["bytes_hex"] == encoded.hex(), "calc input bytes mismatch")


def check_trace(row, tokens, interval):
    """Verify a case's trace, exact prefix checkpoints and byte output in one pass."""
    count = integer(row["trace_token_count"], "trace_token_count", 2)
    require(count <= 10_000_000, "trace exceeds exporter token limit")
    require(integer(row["generated_steps"], "generated_steps") == count - 1, "generated step count mismatch")
    integer(row["instruction_count"], "instruction_count", 1)
    integer(row["refresh_count"], "refresh_count")
    require(type(row["exit_code"]) is int and row["exit_code"] in (0, 1), "invalid exit code")
    checkpoints = row["checkpoints"]
    require(isinstance(checkpoints, list), "checkpoints must be an array")
    digest = hashlib.sha256()
    window = deque(maxlen=8)
    checkpoint_index = instructions = refreshes = 0
    output = bytearray()
    for consumed in range(1, count + 1):
        packed = tokens.read(4)
        require(len(packed) == 4, "truncated token blob")
        token, = struct.unpack("<I", packed)
        name, payload = family(token)
        require((name == "GO") == (consumed == 1), "GO must occur only at the first token")
        require((name == "HALT") == (consumed == count), "HALT must occur only at the final token")
        require(name not in {"FLT", "HEX"}, "fault tokens in successful trace")
        instructions += int(name in FINAL)
        refreshes += int(name == "R")
        if name == "BYTE" and payload < 256:
            output.append(payload)
        digest.update(packed)
        window.append(token)
        if consumed == 1 or consumed % interval == 0 or consumed == count:
            require(checkpoint_index < len(checkpoints), "missing checkpoint")
            checkpoint = checkpoints[checkpoint_index]
            expected = {"token_count": consumed, "generated_steps": consumed - 1,
                        "prefix_sha256": digest.hexdigest(), "window_start": consumed - len(window),
                        "window": list(window)}
            require(checkpoint == expected, f"checkpoint mismatch at token count {consumed}")
            checkpoint_index += 1
    require(checkpoint_index == len(checkpoints), "unexpected extra checkpoints")
    require(instructions == row["instruction_count"], "instruction count mismatch")
    require(refreshes == row["refresh_count"], "refresh count mismatch")
    require(digest.hexdigest() == row["trace_sha256"], "trace SHA256 mismatch")
    require(output.hex() == row["final_output_hex"], "BYTE stdout differs from final_output_hex")
    require(output.decode("utf-8") == row["final_output"], "BYTE stdout differs from final_output")
    if row["outcome"] == "machine_limit":
        require(row["kind"] == "calc" and row["id"].startswith("calc-"), "required proof hit a machine limit")
        require(row["final_output"] == "error (too complex: split it)\n", "unknown machine limit")
    else:
        require(row["outcome"] == "ok", "unknown outcome")
        require(row["final_output"] == row["expected_output"], "output differs from oracle")
    expected_exit = int(row["kind"] == "calc" and row["final_output"] in (
        "error\n", "error (too complex: split it)\n"))
    require(row["exit_code"] == expected_exit, "exit code differs from firmware semantics")
    return count, instructions


def check(root):
    """Verify manifest transport checksums before inspecting streamed case payloads."""
    manifest = json.loads((root / "manifest.json").read_bytes())
    require(manifest["format"] == "tensorfold-runner-goldens-v1", "unsupported golden format")
    require(manifest["upstream_commit"] == UPSTREAM, "unexpected upstream revision")
    require(manifest["token_encoding"] == "little-endian-u32", "unsupported token encoding")
    require(manifest["tokens_file"] == "tokens.leu32.gz", "unexpected tokens filename")
    require(manifest["cases_file"] == "cases.jsonl.gz", "unexpected cases filename")
    require(manifest["window_tokens"] == 8, "unsupported checkpoint window")
    require(type(manifest["seed"]) is int, "invalid seed")
    random_cases = integer(manifest["random_calc_cases"], "random_calc_cases")
    interval = integer(manifest["checkpoint_interval"], "checkpoint_interval", 1)
    require(integer(manifest["case_count"], "case_count") == random_cases + 2, "multiply and loop required")
    for name in ("trace_token_count", "instruction_count"):
        integer(manifest[name], name)
    files = manifest["files"]
    require(set(files) == {"cases.jsonl.gz", "tokens.leu32.gz", "loop.elf"}, "required file set mismatch")
    for name, metadata in files.items():
        path = root / name
        size = integer(metadata["bytes"], f"{name} bytes")
        require(path.stat().st_size == size, f"{name}: stored size mismatch")
        require(digest_file(path) == metadata["sha256"], f"{name}: stored SHA256 mismatch")
    with (root / "loop.elf").open("rb") as stream:
        require(stream.read(6) == b"\x7fELF\x01\x01", "loop must be a little-endian ELF32")
    total_tokens = total_instructions = case_count = machine_limits = 0
    with gzip.open(root / "cases.jsonl.gz", "rb") as rows, gzip.open(root / "tokens.leu32.gz", "rb") as tokens:
        for line in rows:
            require(line.endswith(b"\n"), "unterminated JSONL row")
            row = json.loads(line)
            try:
                check_input(row, case_count, random_cases)
                offset = integer(row["token_offset_bytes"], "token_offset_bytes")
                require(offset == total_tokens * 4, "token offset mismatch")
                count, instructions = check_trace(row, tokens, interval)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"case {case_count}: {error}") from error
            total_tokens += count
            total_instructions += instructions
            machine_limits += int(row["outcome"] == "machine_limit")
            case_count += 1
        require(tokens.read(1) == b"", "trailing bytes in token blob")
    require(case_count == manifest["case_count"], "manifest case total mismatch")
    require(total_tokens == manifest["trace_token_count"], "manifest token total mismatch")
    require(total_instructions == manifest["instruction_count"], "manifest instruction total mismatch")
    require(machine_limits == manifest["machine_limit_cases"], "manifest machine-limit total mismatch")
    print(json.dumps({"verified_cases": case_count, "trace_tokens": total_tokens,
                      "instructions": total_instructions, "machine_limit_cases": machine_limits}, sort_keys=True))


def main():
    """Parse the standalone artifact-verification CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="complete golden directory to verify")
    args = parser.parse_args()
    try:
        check(args.output)
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        parser.exit(1, f"runner_trace_check: {error}\n")


if __name__ == "__main__":
    main()
