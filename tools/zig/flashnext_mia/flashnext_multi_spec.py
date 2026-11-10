#!/usr/bin/env python3
"""The batched decode attention's (attn_multi.py) ``*_multi`` entries, cloned from the decode one-stream entries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODULE = "tensorfold.families.qwen4_exp.cuda.attn_multi"
SOURCE = "tensorfold/families/qwen4_exp/cuda/attn_multi.py"
DIV16 = [["tt.divisibility", 16]]


def _line(name: str) -> int | None:
    path = Path(__file__).resolve().parents[2] / "src" / SOURCE
    for i, text in enumerate(path.read_text().splitlines(), 1):
        if text.lstrip().startswith(f"def {name}("):
            return i - 1            # the decorator's line, as the captured entries record
    return None


def _c(k: dict) -> dict:
    return k["constexprs"]


def _entry(base: dict, name: str, sig: list[tuple[str, str]], consts: dict, warps: int, stages: int, n_form: str,
           why: str) -> dict:
    e = {"function": f"{MODULE}.{name}", "name": name, "source": {"file": SOURCE, "line": _line(name)}}
    params = [n for n, _ in sig] + list(consts)
    e["signature"] = {n: t for n, t in sig}
    e["signature"].update({n: "constexpr" for n in consts})
    e["params"] = params
    e["constexprs"] = consts
    attrs = {}
    for n, t in sig:
        if t.startswith("*"):
            attrs[n] = DIV16
        elif n == "N":
            attrs[n] = DIV16 if n_form == "div16" else []
        else:
            attrs[n] = []
    attrs.update({n: [] for n in consts})
    e["attrs"] = attrs
    e["options"] = dict(base["options"], num_warps=warps, num_stages=stages)
    e["tp"] = base.get("tp", [1])
    e["rule"] = f"attn_multi: {why} (N {n_form})"
    return e


def entries(kernels: list[dict]) -> list[dict]:
    out, seen = [], set()
    text = {"bool": False}
    kt = {"dtype": "bf16"}
    for k in kernels:
        if k.get("rule", "").startswith("attn_multi"):
            continue
        c, name = _c(k), k["name"]
        made = []
        if name == "_attn_prep" and c["MODE"]["int"] == 0 and c["BITS"]["int"] == 0:
            consts = {n: c[n] for n in ("PW", "NQ", "NKV", "HD", "NI", "IHD", "HALF", "BITS")}
            consts.update(KT=kt, VISION=text, S1=c["S1"], S2=c["S2"])
            sig = [("P", "*bf16"), ("POSR", "*i32"), ("SID", "*i32"), ("CP", "*i64"), ("VP", "*i64"), ("QW", "*fp32"),
                   ("KW", "*fp32"), ("IW", "*fp32"), ("INV", "*fp32"), ("Q", "*bf16"), ("IQ", "*bf16"),
                   ("eps", "fp32"), ("N", "i32")]
            made.append(("_prep_multi", sig, consts, 2, 3, "_attn_prep's rows by table"))
        elif name == "_pool" and "R" not in c and c["MODE"]["int"] == 0:
            consts = {n: c[n] for n in ("DI", "HALF", "RATIO")}
            consts.update(VISION=text, S1=c["S1"], S2=c["S2"])
            sig = [("CP", "*i64"), ("VP", "*i64"), ("P0", "*i32"), ("RS", "*i32"), ("W", "*fp32"), ("INV", "*fp32"),
                   ("eps", "fp32"), ("N", "i32")]
            made.append(("_pool_multi", sig, consts, 1, 3, "_pool's blocks a stream"))
        elif name == "_chunks" and c["BITS"]["int"] == 0:
            consts = {n: c[n] for n in ("H", "HK", "D", "G", "CH", "NCH", "SCALE", "IDW", "QSA", "BITS")}
            consts.update(KT=kt, RATIO={"int": 4}, TOP={"int": 512})
            sig = [("Q", "*bf16"), ("CP", "*i64"), ("POSR", "*i32"), ("SID", "*i32"), ("PO", "*fp32"), ("PM", "*fp32"),
                   ("PL", "*fp32"), ("IDS", "*i32"), ("NKR", "*i32"), ("N", "i32")]
            made.append(("_chunks_multi", sig, consts, 4, 1, "_chunks' rows by table"))
        elif name == "_merge" and c["BITS"]["int"] == 0:
            consts = {n: c[n] for n in ("H", "HK", "D", "G", "CH", "NCH", "QSA", "BITS")}
            consts.update(RATIO={"int": 4}, TOP={"int": 512})
            sig = [("PO", "*fp32"), ("PM", "*fp32"), ("PL", "*fp32"), ("POSR", "*i32"), ("OUT", "*bf16"),
                   ("NKR", "*i32")]
            made.append(("_merge_multi", sig, consts, 4, 3, "_merge's rows"))
        for new, sig, consts, warps, stages, why in made:
            forms = ("div16", "plain") if any(n == "N" for n, _ in sig) else ("plain",)
            for form in forms:
                e = _entry(k, new, sig, consts, warps, stages, form, why)
                key = json.dumps([e["function"], e["signature"], e["constexprs"], e["attrs"], e["options"]],
                                 sort_keys=True)
                if key in seen:
                    seen_tp = next(x for x in out if json.dumps([x["function"], x["signature"], x["constexprs"],
                                                                 x["attrs"], x["options"]], sort_keys=True) == key)
                    seen_tp["tp"] = sorted(set(seen_tp["tp"]) | set(e["tp"]))
                    continue
                seen.add(key)
                out.append(e)
    return out


def offset_entries(kernels: list[dict]) -> list[dict]:
    """A shared round's sparse stream's scratch pointers offset by row a0: adds the NKR / SPR / SC alignment forms."""

    out = []
    for k in kernels:
        if k.get("rule", "").startswith("attn_multi") or k["name"] not in ("_scores", "_select", "_select_tiles"):
            continue
        forms = [("SC",)] if k["name"] == "_scores" else [("NKR", "SPR"), ("SC", "NKR", "SPR")]
        for off in forms:
            if not all(n in k["attrs"] for n in off):
                continue
            e = json.loads(json.dumps(k))
            for drop in ("hash", "cubin_sha256", "tp1_hash", "expanded_from"):
                e.pop(drop, None)
            for n in off:
                e["attrs"][n] = []
            e["rule"] = f"attn_multi: {k['name']} on a sparse stream's rows ({', '.join(off)} at a row offset)"
            out.append(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = json.loads(Path(a.spec).read_text())
    keep = [k for k in data["kernels"] if not k.get("rule", "").startswith("attn_multi")]
    new = entries(keep) + offset_entries(keep)
    data["kernels"] = keep + new
    Path(a.out).write_text(json.dumps(data, indent=1) + "\n")
    for e in new:
        c = {n: next(iter(v.values())) for n, v in e["constexprs"].items()}
        if e["name"].endswith("_multi"):
            print(e["name"], e["rule"], e["tp"], {n: c[n] for n in ("NQ", "NKV", "H", "HK", "NCH", "QSA") if n in c})
    print(f"{len(new)} attn_multi entries, {len(data['kernels'])} in all -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
