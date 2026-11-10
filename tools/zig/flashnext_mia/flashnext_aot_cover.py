"""flashnext_aot.py's cover stage: W5 fixture launches each get a spec variant; the spec helpers it shares."""

from __future__ import annotations

import json
import struct
from pathlib import Path


def _reorder(k: dict) -> dict:
    """Signature, attrs and constexprs in parameter order, as the JIT builds them (the attrs' order is hashed)."""

    p = k["params"]
    k["signature"] = {n: k["signature"][n] for n in p}
    k["attrs"] = {n: k["attrs"].get(n, []) for n in p}
    k["constexprs"] = {n: k["constexprs"][n] for n in sorted(k["constexprs"], key=lambda x: p.index(x.split(".")[0]))}
    return k


def zig_const(v):
    if "bool" in v:
        return {"int": int(v["bool"])}
    if "int" in v:
        return {"int": v["int"]}
    if "fp64_bits" in v:
        return {"f32": struct.unpack("<I", struct.pack("<f", struct.unpack("<d", struct.pack("<Q", int(v["fp64_bits"], 16)))[0]))[0]}
    if "none" in v:
        return {"none": True}
    return {"str": json.dumps(v)}


# ------------------------------------------------------------------------------------------------------------ cover

def _fixture_matches(k: dict, launch: dict) -> bool:
    """zig/src/cuda/aot.zig's ``matches`` on a W5 fixture launch (pointers taken as 16-byte aligned)."""

    if k["name"] != launch["fn"]:
        return False
    zc = {n: zig_const(v) for n, v in k["constexprs"].items()}
    for n, v in launch["consts"].items():
        if not isinstance(v, dict) or not ({"int", "f32"} & set(v)):
            continue                                       # a None (or named) constexpr: Zig does not compare it
        got = zc.get(n)
        if got is None or any(got.get(t) != v.get(t) for t in ("int", "f32") if t in v):
            return False
    runtime = 0
    params = {n for n in k["params"] if k["signature"].get(n) != "constexpr"}
    for a in launch["args"]:
        n, ty = a["name"], a["type"]
        div = bool(k["attrs"].get(n))
        if ty.startswith("*"):
            if n not in params or k["signature"][n] != ty or not div:
                return False
        elif "int" in a:
            x = a["int"]
            if n not in params:
                if x != 1 or zc.get(n, {}).get("int") != 1:
                    return False
                continue
            if k["signature"][n] != ty or x == 1 or div != (x % 16 == 0):
                return False
        elif n not in params or k["signature"][n] != ty:
            return False
        runtime += 1
    return runtime == len(params)


def _from_fixture(t: dict, launch: dict, case: str) -> dict:
    """A spec entry for a fixture launch, on a template of the same kernel: its types, int forms, constexprs, warps."""

    out = json.loads(json.dumps(t))
    for drop in ("hash", "cubin_sha256", "tp1_hash"):
        out.pop(drop, None)
    for a in launch["args"]:
        n = a["name"]
        if "int" in a and a["int"] == 1:
            out["signature"][n], out["constexprs"][n], out["attrs"][n] = "constexpr", {"int": 1}, []
            continue
        out["signature"][n] = a["type"]
        out["constexprs"].pop(n, None)
        if a["type"].startswith("*"):
            out["attrs"][n] = [["tt.divisibility", 16]]
        elif "int" in a:
            out["attrs"][n] = [["tt.divisibility", 16]] if a["int"] % 16 == 0 else []
    for n, v in launch["consts"].items():
        old = t["constexprs"].get(n, {})
        if not isinstance(v, dict) or not ({"int", "f32"} & set(v)):
            continue
        if "f32" in v:
            if old and zig_const(old).get("f32") == v["f32"]:
                continue                                  # the template's own value (its fp64 bits)
            x = struct.unpack("<f", struct.pack("<I", v["f32"]))[0]     # an fp32 value, as a Python float
            out["constexprs"][n] = {"float": x, "fp64_bits": "0x%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]}
            continue
        out["constexprs"][n] = {"bool": bool(v["int"])} if "bool" in old else {"int": v["int"]}
    out["options"]["num_warps"] = launch["num_warps"]
    if launch.get("num_stages") is not None:
        out["options"]["num_stages"] = launch["num_stages"]
    out["tp"] = [2] if "/tp2" in case else [1] if "/tp1" in case else [1, 2]
    out["expanded_from"] = t.get("hash") or t.get("expanded_from")
    out["rule"] = f"W5 fixture {case}"
    return _reorder(out)


def cover(spec: Path, fixtures: Path, out: Path) -> int:
    """Every launch of W5's fixtures (zig/src/families/flashnext_mia/fixtures_cuda_triton.json) must find a variant."""

    data = json.loads(spec.read_text())
    kernels = data["kernels"]
    cases = json.loads(fixtures.read_text())["cases"]
    cases = list(cases.items()) if isinstance(cases, dict) else cases
    added, missing = [], 0
    for case, launches in cases:
        for launch in launches:
            if any(_fixture_matches(k, launch) for k in kernels + added):
                continue
            missing += 1
            same = [k for k in kernels if k["name"] == launch["fn"]
                    and {a["name"] for a in launch["args"]} <= set(k["params"])]
            if not same:
                print(f"NO TEMPLATE {case} {launch['fn']}")
                continue
            # the template closest in constexprs
            best = max(same, key=lambda k: (sum(zig_const(k["constexprs"].get(n, {"none": True})) == v
                                                for n, v in launch["consts"].items() if isinstance(v, dict)),
                                            sum(k["signature"].get(a["name"]) == a["type"] for a in launch["args"])))
            new = _from_fixture(best, launch, case)
            if not _fixture_matches(new, launch):
                raise SystemExit(f"{case}: the synthesized entry does not match its launch")
            added.append(new)
            print(f"added {case} {launch['fn']}: " + ", ".join(f"{a['name']} {a['type']}" for a in launch["args"]
                                                              if a["type"] != best["signature"].get(a["name"])))
    data["kernels"] = kernels + added
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"{sum(len(l) for _, l in cases)} fixture launches, {missing} without a variant, {len(added)} entries added "
          f"-> {out}")
    return 0
