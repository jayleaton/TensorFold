#!/usr/bin/env python3
"""Profile pinned Qwenputer reference records without building or loading a circuit."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from runner_trace import MULTIPLY, TOKEN_LIMIT, UPSTREAM, file_hash, load_upstream


PHASES = ("boot", "parse", "arith", "print")
SUBRECORDS = {"BD", "SC", "QS", "QD"}
ARITH = {
    "apply", "add", "add_mag", "sub_mag", "cmp_mag", "mul", "divmod", "divide", "gcd",
    "power", "reduce", "frac_op", "shift", "copy_of", "trim", "is_zero",
}
PRINT = {"bn_print", "print_unsigned", "rv_putchar", "putchar", "puts"}


class Symbols:
    """Read sized executable ELF32 STT_FUNC symbols, retaining aliases and section bounds."""

    def __init__(self, path):
        data = path.read_bytes()
        if len(data) < 52 or data[:7] != b"\x7fELF\x01\x01\x01":
            raise ValueError("expected little-endian ELF32 version 1")
        if struct.unpack_from("<H", data, 18)[0] != 243:
            raise ValueError("expected RISC-V ELF")
        offset = struct.unpack_from("<I", data, 32)[0]
        size, count = struct.unpack_from("<HH", data, 46)
        if size < 40 or not count:
            raise ValueError("ELF needs ordinary section headers and function symbols")
        sections = [struct.unpack_from("<10I", data, offset + i * size) for i in range(count)]
        entries = defaultdict(list)
        for section in sections:
            if section[1] != 2:
                continue
            strings = sections[section[6]]
            names = data[strings[4]:strings[4] + strings[5]]
            if section[9] < 16 or section[5] % section[9]:
                raise ValueError("invalid ELF32 symbol table entry size")
            for pos in range(section[4], section[4] + section[5], section[9]):
                name, value, length, info, _, index = struct.unpack_from("<IIIBBH", data, pos)
                if info & 15 != 2 or not 0 < index < count or not sections[index][2] & 4:
                    continue
                end = names.find(b"\0", name)
                if end < 0:
                    raise ValueError("unterminated ELF symbol name")
                label = names[name:end].decode("utf-8", errors="replace")
                sec = sections[index]
                entries[value].append((label, length, sec[3] + sec[5]))
        if not entries:
            raise ValueError("no executable function symbols; supply an unstripped ELF")
        self.starts = sorted(entries)
        self.rows = []
        for i, start in enumerate(self.starts):
            aliases = sorted(entries[start])
            label, length, section_end = max(aliases, key=lambda row: (row[1], row[0]))
            next_start = self.starts[i + 1] if i + 1 < len(self.starts) else section_end
            end = min(start + length if length else next_start, section_end)
            self.rows.append({"name": label, "start": start, "end": end,
                              "aliases": [row[0] for row in aliases], "size_inferred": not bool(length)})

    def at(self, pc):
        """Resolve only PCs inside a function range; never extend a symbol through a gap."""
        index = bisect_right(self.starts, pc) - 1
        if index >= 0 and pc < self.rows[index]["end"]:
            return self.rows[index]["name"]
        return "<unmapped>"


def base_name(name):
    """Ignore compiler clone suffixes when applying the published attribution policy."""
    return name.split(".", 1)[0]


def bucket():
    """Create one additive record counter."""
    return {"generated_steps": 0, "instructions": 0, "records": 0, "refresh_R": 0,
            "token_families": Counter(), "opcodes": Counter()}


def add(target, families, opcode, instruction):
    """Count every generated token exactly once within this view."""
    target["generated_steps"] += sum(families.values())
    target["instructions"] += instruction
    target["records"] += 1
    target["refresh_R"] += families["R"]
    target["token_families"].update(families)
    if instruction:
        target["opcodes"][opcode] += 1


def rank(counters, key, limit=30):
    """Keep bounded hot-spot tables with deterministic tie ordering."""
    ordered = sorted(counters.items(), key=lambda item: (-item[1]["generated_steps"], str(item[0])))
    return [{key: name, **counts} for name, counts in ordered[:limit]]


def projections(total, phases, sub_tokens, sub_records, kept_sub_tokens, kept_sub_records):
    """Compute conditional token floors from measured records, never predicted throughput."""
    overhead = phases["boot"]["generated_steps"] + phases["parse"]["generated_steps"]
    floors = {
        "same_control_zero_cost_subrecords": total - sub_tokens,
        "same_control_one_step_per_subrecord": total - sub_tokens + sub_records,
        "ast_direct_zero_cost_boot_parse": total - overhead,
        "ast_direct_and_one_step_per_retained_subrecord": total - overhead - kept_sub_tokens + kept_sub_records,
    }
    return {"boot_parse_policy_tokens": overhead, "subrecord_tokens": sub_tokens,
            "records_with_subrecords": sub_records,
            "conditional_scenarios": {
                name: {"optimistic_generated_step_floor": floor,
                       "speedup_ceiling": total / floor if floor else None,
                       "could_reach_5x": floor * 5 <= total,
                       "could_reach_10x": floor * 10 <= total}
                for name, floor in floors.items()},
            "assumptions": "Unchanged tokens cost one step; replaced subrecords cost zero or one as named. "
                           "AST scenario removes policy boot/parse, including initialization and dispatch; "
                           "a replacement must restore required state. No circuit or timing claim."}


def profile(machine, symbols, ref, spec, kind):
    """Trace real records and partition their tokens by a documented caller-aware policy."""
    total = bucket()
    digest = hashlib.sha256(struct.pack("<I", spec.tok("GO")))
    phases = {name: bucket() for name in PHASES}
    functions, pcs, callers = defaultdict(bucket), defaultdict(bucket), defaultdict(bucket)
    sub_by_opcode = defaultdict(Counter)
    frames = []
    boot_done = printing = False
    sub_tokens = sub_records = kept_sub_tokens = kept_sub_records = 0
    bd_carry = bd_carry_nonzero = unmapped = 0
    boundaries = {}
    while not machine.halted:
        pc = machine.pc
        leaf = symbols.at(pc)
        refresh = machine.n == spec.N_MAX
        ins = None if refresh else ref.decode(machine.rom[(pc - spec.ROM_BASE) >> 2])
        opcode = "refresh_R" if refresh else ins.op
        chain = [frame[1] for frame in frames] + [leaf]
        names = [base_name(name) for name in chain]
        is_output = not refresh and (opcode == "BPRD" or opcode == "ECALL" and machine.x[17] == 1)
        if not boot_done and not refresh and (opcode == "BSCAN" or kind == "elf" and base_name(leaf) == "main"):
            boot_done = True
            boundaries["first_parse_or_main_pc"] = f"0x{pc:08x}"
            boundaries["boot_end_generated_steps"] = total["generated_steps"]
        if is_output and not printing:
            printing = True
            boundaries["first_output_pc"] = f"0x{pc:08x}"
            boundaries["first_output_generated_steps"] = total["generated_steps"]
        if printing or any(name in PRINT or name.startswith("bn_print") for name in names):
            phase = "print"
        elif not boot_done:
            phase = "boot"
        elif kind == "elf" or any(name in ARITH for name in names):
            phase = "arith"
        else:
            phase = "parse"
        before = machine.steps
        tokens = machine.record()
        digest.update(struct.pack(f"<{len(tokens)}I", *tokens))
        families = Counter(spec.family(token)[0] for token in tokens)
        instruction = machine.steps - before
        if not tokens or total["generated_steps"] + len(tokens) > TOKEN_LIMIT:
            raise ValueError("empty record or token limit exceeded")
        if instruction != int(not refresh) or (refresh and families != Counter({"R": 1})):
            raise ValueError("upstream instruction/refresh semantics changed")
        if families["GO"]:
            raise ValueError("record unexpectedly generated GO")
        for target in (total, phases[phase], functions[leaf], pcs[f"0x{pc:08x}"], callers[" > ".join(chain)]):
            add(target, families, opcode, instruction)
        unmapped += len(tokens) if leaf == "<unmapped>" else 0
        sub = sum(families[name] for name in SUBRECORDS)
        sub_tokens += sub
        sub_records += int(sub > 0)
        if phase not in ("boot", "parse"):
            kept_sub_tokens += sub
            kept_sub_records += int(sub > 0)
        sub_by_opcode[opcode].update({name: families[name] for name in SUBRECORDS if families[name]})
        for token in tokens:
            family, payload = spec.family(token)
            if family == "BD" and opcode in ("BMAC", "BMSB"):
                bd_carry += 1
                bd_carry_nonzero += int(payload >> 4 != 0)
        if refresh or opcode not in ("JAL", "JALR") or machine.halted:
            continue
        destination = machine.pc
        returns = [i for i, frame in enumerate(frames) if frame[0] == destination]
        if ins.rd not in (1, 5) and returns:
            frames = frames[:returns[-1]]
        elif ins.rd in (1, 5):
            frames.append((pc + 4, leaf))
            if len(frames) > 4096:
                raise ValueError("call stack exceeds safety limit")
    if machine.fault is not None or machine.exit_code is None:
        raise ValueError(f"reference fault={machine.fault}, exit={machine.exit_code}")
    if sum(row["generated_steps"] for row in phases.values()) != total["generated_steps"]:
        raise ValueError("phase partition does not sum to generated steps")
    if total["instructions"] != machine.steps:
        raise ValueError("instruction accounting disagrees with Machine.steps")
    return {**total, "trace_sha256": digest.hexdigest(), "machine_steps": machine.steps,
            "trace_tokens_including_GO": total["generated_steps"] + 1,
            "phases": phases, "boundaries": boundaries, "top_functions": rank(functions, "function"),
            "top_pcs": [{**row, "function": symbols.at(int(row["pc"], 16))} for row in rank(pcs, "pc")],
            "top_callstacks": rank(callers, "callstack"), "subrecords_by_opcode": dict(sub_by_opcode),
            "carry_diagnostics": {"BD_arithmetic_tokens_including_carry": bd_carry,
                                  "BD_arithmetic_nonzero_carry_or_borrow": bd_carry_nonzero,
                                  "separate_carry_steps": None},
            "unmapped_tokens": unmapped, "output": bytes(machine.out).decode("utf-8"),
            "exit_code": machine.exit_code,
            "optimization": projections(total["generated_steps"], phases, sub_tokens, sub_records,
                                        kept_sub_tokens, kept_sub_records)}


def sums(rows):
    """Aggregate complete measured cases without counting sample views twice."""
    total = bucket()
    phases = {name: bucket() for name in PHASES}
    for row in rows:
        for destination, source in [(total, row), *[(phases[name], row["phases"][name]) for name in PHASES]]:
            for key in ("generated_steps", "instructions", "records", "refresh_R"):
                destination[key] += source[key]
            for key in ("token_families", "opcodes"):
                destination[key].update(source[key])
    return {"cases": len(rows), **total, "phases": phases}


def run(args):
    """Profile requested expressions, deterministic sample cases, and an optional RV32 loop."""
    (calc, ref, spec, elf), hashes = load_upstream(args.qwenputer.resolve())
    symbols = Symbols(calc.ELF)
    rows = []
    expressions = args.expr if args.expr else [MULTIPLY, "2**100"]
    cases = [(f"expr-{i}", "requested", expression) for i, expression in enumerate(expressions)]
    cases += [(f"sample-{i:04d}", op, expression) for i, (op, expression) in enumerate(calc.cases(args.cases, 0))]
    for case_id, op, expression in cases:
        row = profile(calc.machine(expression), symbols, ref, spec, "calc")
        expected = calc.expected(expression)
        limited = row["output"] == "error (too complex: split it)\n"
        if row["output"] != expected and not limited:
            raise ValueError(f"{case_id}: output {row['output']!r} differs from oracle {expected!r}")
        if limited and op == "requested":
            raise ValueError(f"{case_id}: required expression hit register limit")
        if row["exit_code"] != int(limited or expected == "error\n"):
            raise ValueError(f"{case_id}: unexpected exit code")
        rows.append({"id": case_id, "operation": op, "expression": expression,
                     "expected_output": expected, "outcome": "machine_limit" if limited else "ok", **row})
    loop = None
    if args.loop_elf:
        loop_symbols = Symbols(args.loop_elf)
        loop = profile(ref.Machine.from_program(elf.program(args.loop_elf)), loop_symbols, ref, spec, "elf")
        if loop["output"] != "285\n" or loop["exit_code"] != 0:
            raise ValueError("loop ELF must print sum of squares 0..9 as 285 followed by newline, then exit 0")
        loop.update({"id": "loop-squares", "elf_sha256": file_hash(args.loop_elf),
                     "symbols": loop_symbols.rows})
    samples = [row for row in rows if row["id"].startswith("sample-")]
    result = {"format": "tensorfold-runner-profile-v1", "upstream_commit": UPSTREAM,
              "upstream_files_sha256": hashes, "seed": 0, "random_case_count": args.cases,
              "symbols": symbols.rows, "calc_cases": rows, "calc_sums": sums(rows),
              "ten_sample_case_ids": [row["id"] for row in samples[:10]], "sample_sums": sums(samples),
              "loop": loop, "loop_sums": sums([loop] if loop else []),
              "all_sums": sums(rows + ([loop] if loop else [])),
              "accounting": "generated_steps excludes GO, includes refresh R; instructions equals Machine.steps",
              "phase_policy": "Caller-aware arithmetic symbols; boot until first BSCAN (or ELF main); "
                              "print helpers and suffix from first output; parse is remaining calc work. "
                              "Inlining hides print setup and initialization boundaries; these are policy counts, "
                              "not exact source-line timings. See runner-profile.md."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")


def main():
    """Parse the CPU-only profiling CLI; never compile firmware or execute native code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwenputer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new JSON file")
    parser.add_argument("--loop-elf", type=Path, help="unstripped ROM-map RV32 sum-of-squares ELF")
    parser.add_argument("--cases", type=int, default=20, help="seed-0 sample count")
    parser.add_argument("--expr", action="append", help="repeatable; replaces the two default expressions")
    args = parser.parse_args()
    if args.cases < 0:
        parser.error("--cases must be nonnegative")
    try:
        run(args)
    except (OSError, ValueError, AssertionError, struct.error, IndexError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"runner_profile: {error}\n")


if __name__ == "__main__":
    main()
