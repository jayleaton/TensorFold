"""Writes (--write) or checks (default) zig/kernels/cuda/deepseek_v41: verbatim copies of the Python engine's
DeepSeek-V4.1 device code at git refs (SOURCES: dsv41-quant-e2, and the branches built on it that are not merged yet:
x3gm v2 and the R1 decode kernels; each copy names its source), torch host code cut, plus the explicit
instantiations the Python wrappers launch.

Same lines, same namespaces (so the same mangled symbols and the same SASS as the Python extensions), same nvcc flags
(zig/build/cuda.zig). Each copy's header names its source and the source's git blob hash; `--check` regenerates every
copy from the Python tree and fails on any difference.

usage: python3 sync.py [--write] [--base <repo>:<ref>]   (the base source; the others: SOURCES)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = "/home/sireggy/Documents/Projects/spark-work"
# (repo, ref) by source key. When a branch lands in dsv41-quant-e2, point its files back at "base".
SOURCES = {
    "base": (f"{WORK}/tensorfold-dsquant", "dsv41-quant-e2"),
    "x3gm2": (f"{WORK}/tensorfold-x3gm1", "dsv41-x3gm1"),      # x3gm v2 (gm2_kernel), dsq-x3gm-1
    "r1": (f"{WORK}/tensorfold-decode1", "78b703d"),   # R1 + R1c decode, final (dsv41-decode1)
}
FROM = {"x3gm.cu": "x3gm2", "attn_cuda.cu": "r1", "topk_cuda.cu": "r1", "mhc_cuda.cu": "r1", "exl3_experts.cu": "r1",
        "router_gemv.cu": "r1", "x3seg.cu": "r1", "dense3.cu": "r1"}   # dense3.cu includes x3seg.cu: one tree
EXL3 = "src/tensorfold/cuda/exl3"
FAM = "src/tensorfold/families/deepseek_v41/cuda"

TORCH_INCLUDE = re.compile(r'^#include <(ATen|c10|torch)/')


def lines_of(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def footer_x3ld() -> str:
    out = ["// The instantiations dsv41_x3ld_grouped_cuda dispatches: mul1, (nt, pd) x probe x the (lo, hi) ranges.\n"]
    out.append("#define DSV41_X3LD_INST(NT, PD, LO, HI, PROBE) template __global__ void dsv41_x3ld::ld_kernel<2, NT, PD, "
               "LO, HI, PROBE>( \\\n    const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, "
               "const int*, const int*, \\\n    const int*, float*, int, int, int, int, int, int);\n")
    for probe, cfgs in ((0, ((8, 1), (8, 2), (4, 2))), (3, ((8, 1), (8, 2)))):
        for nt, pd in cfgs:
            for lo, hi in ((8, 8), (2, 10), (2, 12)):
                out.append(f"DSV41_X3LD_INST({nt}, {pd}, {lo}, {hi}, {probe})\n")
    return "".join(out)


def footer_x3pf() -> str:
    out = ["// The instantiations dsv41_x3pf_grouped_cuda dispatches: mul1, (nt, mtl) x the (lo, hi) ranges.\n"]
    out.append("#define DSV41_X3PF_INST(NT, MTL, LO, HI) template __global__ void dsv41_x3pf::pf_kernel<2, NT, MTL, LO, "
               "HI>( \\\n    const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, "
               "const int*, const int*, \\\n    const int*, float*, int, int, int, int, int, int);\n")
    for nt, mtl in ((4, 4), (8, 2)):
        for lo, hi in ((8, 8), (2, 10), (2, 12)):
            out.append(f"DSV41_X3PF_INST({nt}, {mtl}, {lo}, {hi})\n")
    return "".join(out)


X3GM_K2S = (3, 4, 5, 6, 7, 8, 10)


def footer_x3gm() -> str:
    out = ["// The instantiations x3gm's bindings launch: gate/up (shared input XM 1, own inputs XM 2) at GM_GU0-5,\n"
           "// down at GM_DN0-4, every width K2 3-8, 10; the rotation (bf16 / fp16 input, one or two matrices); dequant.\n"]
    out.append("#define DSV41_X3GM_GU(XM, K2, CFG) template __global__ void dsv41_x3gm::gm_kernel<2, XM, K2, CFG>("
               "const dsv41_x3gm::Args);\n")
    out.append("#define DSV41_X3GM_DN(K2, CFG) template __global__ void dsv41_x3gm::gm_kernel<1, 1, K2, CFG>("
               "const dsv41_x3gm::Args);\n")
    out.append("// v2 (gm2_kernel: every width in one launch), the tiles launch2 takes: gate/up GM_GU0 / GM_GU1 at either\n"
               "// input mode, down GM_DN0\n")
    for xm in (1, 2):
        for c in (0, 1):
            out.append(f"template __global__ void dsv41_x3gm::gm2_kernel<2, {xm}, GM_GU{c}>(const dsv41_x3gm::Args);\n")
    out.append("template __global__ void dsv41_x3gm::gm2_kernel<1, 1, GM_DN0>(const dsv41_x3gm::Args);\n")
    for k2 in X3GM_K2S:
        for xm in (1, 2):
            for c in range(6):
                out.append(f"DSV41_X3GM_GU({xm}, {k2}, GM_GU{c})\n")
        for c in range(5):
            out.append(f"DSV41_X3GM_DN({k2}, GM_DN{c})\n")
    out.append("#define DSV41_X3GM_ROT(T, M) template __global__ void dsv41_x3gm::rot_kernel<T, M>(const T*, int, "
               "const int*, const half*, \\\n    const half*, half*, half*, int, int, int);\n")
    for t in ("__nv_bfloat16", "half"):
        for m in (1, 2):
            out.append(f"DSV41_X3GM_ROT({t}, {m})\n")
    for k2 in X3GM_K2S:
        out.append(f"template __global__ void dsv41_x3gm::dequant_kernel<{k2}>(const uint32_t*, half*, int);\n")
    # x3gm3.cu (v3) includes this copy for its device code, without these instances
    return "#ifndef DSV41_X3GM_NO_INSTANCES\n" + "".join(out) + "#endif  // DSV41_X3GM_NO_INSTANCES\n"


def footer_exl3_experts() -> str:
    out = ["// upstream grouped_kernel at mul1 (grouped_launch<2>: experts_cb2.cu) for every (nt, warps, pf) x TF_RANGES,\n"
           "// and dequant_kernel<2, K2> (tests).\n"]
    out.append("#define TF_EXL3X_INST(NT, W, PF, LO, HI) template __global__ void tf_exl3x::grouped_kernel<2, NT, W, PF, "
               "LO, HI>( \\\n    const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, "
               "const int*, const int*, \\\n    const int*, float*, int, int, int, int, int, int);\n")
    for nt, w, pf in ((8, 4, 1), (8, 4, 2), (4, 4, 2)):
        for lo, hi in ((8, 8), (2, 10), (2, 16)):
            out.append(f"TF_EXL3X_INST({nt}, {w}, {pf}, {lo}, {hi})\n")
    for k2 in range(2, 17):
        out.append(f"template __global__ void tf_exl3x::dequant_kernel<2, {k2}>(const uint32_t*, half*, int, int);\n")
    out.append("// down_combine_kernel at both outputs the binding takes (fp32, bf16: R1's TF_DSV41_MOE_BF16)\n")
    for t in ("float", "__nv_bfloat16"):
        out.append(f"template __global__ void tf_exl3_experts::down_combine_kernel<{t}>(const float*, const int*, "
                   f"const half*, float*, \\\n    const float*, {t}*, int, int, int, int, int);\n")
    out.append("// rot_in_kernel for both inputs the binding takes\n")
    for t in ("__nv_bfloat16", "half"):
        out.append(f"template __global__ void tf_exl3_experts::rot_in_kernel<{t}>(const {t}*, int, const int*, "
                   "const half*, const half*, half*, \\\n    half*, int, int, int);\n")
    return "".join(out)


def footer_linear() -> str:
    out = ["// The instantiations exl3_linear_cuda / exl3_unpack_cuda launch at mul1 (TF_EXL3_WIDTHS(X, 2) + K2 3, 5, 7):\n"
           "// linear_kernel<K2, 2, WK> at WK 2 / 4 / 8, unpack_kernel<K2, 2>.\n"]
    for k2 in LINEAR_K2S:
        for wk in (2, 4, 8):
            out.append(f"template __global__ void tf_exl3_linear::linear_kernel<{k2}, 2, {wk}>(const half*, const uint32_t*, "
                       "long long, long long, \\\n    const half*, const half*, void*, int, float*, int*, int, int, int, int, "
                       "const int*);\n")
        out.append(f"template __global__ void tf_exl3_linear::unpack_kernel<{k2}, 2>(const uint32_t*, half*, int, int64_t, "
                   "int64_t);\n")
    return "".join(out)


LINEAR_K2S = (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16)
X3SEG_K2S = (4, 6, 8, 10, 12, 16)
DENSE3_K2S = (4, 6, 8, 10, 12)


def footer_x3seg() -> str:
    out = ["// The instantiations dsv41_x3seg_linear_cuda launches (TF_SEG_WIDTH): mul1, (WK, MINB) (4, 3), (4, 2), (8, 1);\n"
           "// not when dense3.cu includes this file for its device code.\n#ifndef DSV41_DENSE3_NO_TORCH\n"]
    for k2 in X3SEG_K2S:
        for wk, mb in ((4, 3), (4, 2), (8, 1)):
            out.append(f"template __global__ void dsv41_x3seg::seg_linear_kernel<{k2}, 2, {wk}, {mb}>(const dsv41_x3seg::LinTable, "
                       "int, int);\n")
    out.append("#endif\n")
    return "".join(out)


def footer_dense3() -> str:
    out = ["// The instantiations dsv41_dense3_linear_cuda / _unpack_cuda launch (TF_D3_WIDTH, TF_D3_UNPACK): mul1.\n"]
    for k2 in DENSE3_K2S:
        for wk, mb in ((4, 3), (4, 4), (8, 1), (8, 2)):
            out.append(f"template __global__ void dsv41_dense3::lanes_linear_kernel<{k2}, 2, {wk}, {mb}>("
                       "const dsv41_x3seg::LinTable, int, int);\n")
        out.append(f"template __global__ void dsv41_dense3::lanes_unpack_kernel<{k2}, 2>(const uint32_t*, half*, int, int64_t, "
                   "int64_t);\n")
    return "".join(out)


def footer_attn() -> str:
    out = ["// The instantiations dispatch() launches: attn_kernel<KW 2 / 4, ROWS, HAS_HI>.\n"]
    for kw in (2, 4):
        for rows in ("false", "true"):
            for hi in ("false", "true"):
                out.append(f"template __global__ void dsv41_attn::attn_kernel<{kw}, {rows}, {hi}>(dsv41_attn::Args);\n")
    out.append("// R1: dispatch_split's split_kernel<KW, ROWS, HAS_HI, P> at P 1 and 4 (a cluster of 4)\n")
    for p in (1, 4):
        for kw in (2, 4):
            for rows in ("false", "true"):
                for hi in ("false", "true"):
                    out.append(f"template __global__ void dsv41_attn::split_kernel<{kw}, {rows}, {hi}, {p}>(dsv41_attn::Args);\n")
    return "".join(out)


MHC_INST = (("true", 2, "true", "false"), ("false", 1, "true", "false"), ("false", 2, "true", "false"),
            ("true", 2, "false", "true"))       # (POST, COLL, MIX, ALL4) of modes 0-3; FN32 both except mode 3


def footer_mhc() -> str:
    out = ["// The instantiations dispatch() launches: modes 0-2 at bf16 and fp32 fn, mode 3 (final) at bf16, each with\n"
           "// R1's parallel tail off and on (coef_kernel is not a template: always emitted).\n"]
    for i, (post, coll, mix, all4) in enumerate(MHC_INST):
        for fn32 in (("false", "true") if i < 3 else ("false",)):
            for tail in ("false", "true"):
                out.append(f"template __global__ void dsv41_mhc::boundary_kernel<{post}, {coll}, {mix}, {all4}, {fn32}, "
                           f"{tail}>(const dsv41_mhc::Args);\n")
    return "".join(out)


MHC_PF_INST = (("true", 2, "false", "false"), ("true", 2, "false", "true"), ("true", 2, "true", "false"),
               ("true", 2, "true", "true"), ("false", 1, "false", "false"), ("false", 1, "true", "false"),
               ("false", 2, "false", "false"), ("false", 2, "true", "false"))   # (POST, COLL, FN32, TAP)


def footer_mhc_pf() -> str:
    out = ["// The instantiations dispatch() launches: (mode, fn32, tap) as its switch lists them.\n"]
    for post, coll, fn32, tap in MHC_PF_INST:
        out.append(f"template __global__ void dsv41_mhc_pf::site_kernel<{post}, {coll}, {fn32}, {tap}>("
                   "const dsv41_mhc_pf::Args);\n")
    return "".join(out)


RG_RT = (1, 2, 4, 8, 16)
RG_WIDE = ((8, 1), (4, 1), (8, 2), (4, 2))
RG_NARROW = (1, 2, 4, 8)


def footer_rg() -> str:
    out = ["// The instantiations dispatch() launches: gemv_kernel<RT, NW, EW> and narrow_kernel<RT, NW>.\n"]
    for rt in RG_RT:
        for nw, ew in RG_WIDE:
            out.append(f"template __global__ void dsv41_rg::gemv_kernel<{rt}, {nw}, {ew}>(const dsv41_rg::Args);\n")
        for nw in RG_NARROW:
            out.append(f"template __global__ void dsv41_rg::narrow_kernel<{rt}, {nw}>(const dsv41_rg::Args);\n")
    return "".join(out)


# Files copied whole: their torch code (if any) is behind a NO_TORCH macro the Python compile tests define, which
# is defined first here (None: a file with no torch code at all, copied as it is).
PRELUDE = {"x3seg.cu": "DSV41_X3SEG_NO_TORCH", "dense3.cu": "DSV41_DENSE3_NO_TORCH", "attn_cuda.cu": "DSV41_ATTN_NO_TORCH",
           "topk_cuda.cu": "DSV41_ATTN_NO_TORCH", "mhc_cuda.cu": "DSV41_MHC_NO_TORCH",
           "mhc_pf.cu": "DSV41_MHC_PF_NO_TORCH", "router_gemv.cu": "DSV41_RG_NO_TORCH", "l2pace.cu": None,
           "l2pf.cu": None, "pfdense.cuh": None, "pfdense_args.h": None, "pfdense_k8.cu": None, "pfdense_k10.cu": None,
           "pfdense_k12.cu": None, "pfdense_k16.cu": None}

# name -> (source, kept 1-based inclusive line ranges, {line: replacement}, footer)
COPIES: dict[str, tuple[str, tuple[tuple[int, int], ...], dict[int, str], str]] = {
    "decode.cuh": (f"{EXL3}/decode.cuh", ((1, 162),), {}, ""),
    "experts_grouped.cuh": (f"{EXL3}/experts_grouped.cuh", ((1, 307), (356, 356)), {}, ""),
    "exl3_experts.cu": (f"{EXL3}/experts.cu", ((1, 244),), {11: "namespace tf_exl3_experts {\n",
                                                            244: "}  // namespace tf_exl3_experts\n"},
                        "footer_exl3_experts"),
    "x3ld.cu": (f"{FAM}/x3ld.cu", ((1, 251), (297, 297)), {}, "footer_x3ld"),
    "x3pf.cu": (f"{FAM}/x3pf.cu", ((1, 204), (240, 240)), {}, "footer_x3pf"),
    "x3gm.cu": (f"{FAM}/x3gm.cu", ((1, 39), (45, 628), (649, 666), (732, 733)), {}, "footer_x3gm"),
    "linear.cu": (f"{EXL3}/linear.cu", ((1, 283), (288, 288)), {14: "namespace tf_exl3_linear {\n",
                                                                  288: "}  // namespace tf_exl3_linear\n"},
                  "footer_linear"),
    "x3seg.cu": (f"{FAM}/x3seg.cu", ((1, 483),), {}, "footer_x3seg"),
    "dense3.cu": (f"{FAM}/dense3.cu", ((1, 402),), {}, "footer_dense3"),
    "attn_cuda.cu": (f"{FAM}/csa2/attn_cuda.cu", ((1, 1082),), {}, "footer_attn"),
    "topk_cuda.cu": (f"{FAM}/csa2/topk_cuda.cu", ((1, 304),), {}, ""),
    "mhc_cuda.cu": (f"{FAM}/mhc_cuda.cu", ((1, 635),), {}, "footer_mhc"),
    "mhc_pf.cu": (f"{FAM}/mhc_pf.cu", ((1, 341),), {}, "footer_mhc_pf"),
    "router_gemv.cu": (f"{FAM}/router_gemv.cu", ((1, 617),), {}, "footer_rg"),
    "engram_gate.cu": (f"{FAM}/engram_gate.cu", ((1, 60),), {15: "namespace dsv41_engram_gate {\n",
                                                            60: "}  // namespace dsv41_engram_gate\n"}, ""),
    "l2pace.cu": (f"{FAM}/l2pace.cu", ((1, 85),), {}, ""),
    "l2pf.cu": ("src/tensorfold/families/glm5_next/spark/l2pf.cu", ((1, 114),), {}, ""),
    "pfdense.cuh": (f"{FAM}/pfdense.cuh", ((1, 520),), {}, ""),
    "pfdense_args.h": (f"{FAM}/pfdense_args.h", ((1, 61),), {}, ""),
    "pfdense_k8.cu": (f"{FAM}/pfdense_k8.cu", ((1, 16),), {}, ""),
    "pfdense_k10.cu": (f"{FAM}/pfdense_k10.cu", ((1, 16),), {}, ""),
    "pfdense_k12.cu": (f"{FAM}/pfdense_k12.cu", ((1, 16),), {}, ""),
    "pfdense_k16.cu": (f"{FAM}/pfdense_k16.cu", ((1, 13),), {}, ""),
}

FOOTERS = {"footer_x3ld": footer_x3ld, "footer_x3pf": footer_x3pf, "footer_x3gm": footer_x3gm,
           "footer_exl3_experts": footer_exl3_experts, "footer_linear": footer_linear, "footer_x3seg": footer_x3seg,
           "footer_dense3": footer_dense3, "footer_attn": footer_attn, "footer_mhc": footer_mhc,
           "footer_mhc_pf": footer_mhc_pf, "footer_rg": footer_rg}


def blob_hash(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def read(src_root: Path, ref: str, rel: str) -> bytes:
    """A file's bytes at `ref` (never the working tree: the copies follow a branch, not whatever is checked out)."""
    return subprocess.run(["git", "-C", str(src_root), "show", f"{ref}:{rel}"], check=True, capture_output=True).stdout


def render(base: tuple[str, str], name: str) -> str:
    rel, ranges, replace, footer = COPIES[name]
    src_root, ref = base if FROM.get(name, "base") == "base" else SOURCES[FROM[name]]
    data = read(Path(src_root), ref, rel)
    src = lines_of(data.decode())
    spans = ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)
    whole = name in PRELUDE
    if whole and (len(ranges) != 1 or ranges[0] != (1, len(src))):
        raise SystemExit(f"{name}: {rel} has {len(src)} lines now; update the range")
    macro = PRELUDE.get(name)
    what = (f"the whole file, {macro} defined" if macro else "the whole file") if whole else \
        f"lines {spans}; torch includes and host code cut"
    out = [f"// Device code of {rel} (git blob {blob_hash(data)[:12]} at {ref}; {what}),\n",
           "// generated and checked by zig/kernels/cuda/deepseek_v41/sync.py. Do not edit: change the Python source.\n"]
    if macro:
        out.append(f"#define {macro}\n")
    shim = False
    for a, b in ranges:
        for n in range(a, b + 1):
            line = src[n - 1]
            if n in replace:
                line = replace[n]
            elif TORCH_INCLUDE.match(line) and not whole:
                if shim:
                    continue
                line, shim = "#include <cstdint>  // what the cut torch includes provided: the fixed-width integers\n", True
            out.append(line)
    if footer:
        out.append("\n")
        out.append(FOOTERS[footer]())
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--base", default=":".join(SOURCES["base"]))
    ap.add_argument("--sources", action="store_true", help="print 'copy key repo ref' a line (sass_check.sh reads it)")
    args = ap.parse_args()
    repo, _, ref = args.base.rpartition(":")
    base = (repo, ref)
    if args.sources:
        for name in COPIES:
            key = FROM.get(name, "base")
            r, f = base if key == "base" else SOURCES[key]
            print(name, key, r, f)
        return 0
    bad = 0
    for name in COPIES:
        text = render(base, name)
        path = HERE / name
        if args.write:
            path.write_text(text)
            print(f"wrote {name}")
        elif not path.exists() or path.read_text() != text:
            print(f"DIFFERS {name}")
            bad += 1
        else:
            print(f"ok {name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
