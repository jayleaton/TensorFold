"""Writes (--write) or checks (default) zig/kernels/cuda/deepseek_v41: verbatim copies of the Python engine's
DeepSeek-V4.1 device code (tensorfold-dsquant, branch dsv41-quant-e2), torch host code cut, plus the explicit
instantiations the Python wrappers launch.

Same lines, same namespaces (so the same mangled symbols and the same SASS as the Python extensions), same nvcc flags
(zig/build/cuda.zig). Each copy's header names its source and the source's git blob hash; `--check` regenerates every
copy from the Python tree and fails on any difference.

usage: python3 sync.py [--write] [--src <tensorfold-dsquant checkout>]   (default $TF_DSV41_PY_SRC)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = "/home/sireggy/Documents/Projects/spark-work/tensorfold-dsquant"
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
    return "".join(out)


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
    out = ["// The instantiations dsv41_x3seg_linear_cuda launches (TF_SEG_WIDTH): mul1, (WK, MINB) (4, 3), (4, 2), (8, 1).\n"]
    for k2 in X3SEG_K2S:
        for wk, mb in ((4, 3), (4, 2), (8, 1)):
            out.append(f"template __global__ void dsv41_x3seg::seg_linear_kernel<{k2}, 2, {wk}, {mb}>(const dsv41_x3seg::LinTable, "
                       "int, int);\n")
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


# Files whose torch code is already behind a NO_TORCH macro (the Python tests compile them that way): copied whole,
# the macro defined first.
PRELUDE = {"x3seg.cu": "DSV41_X3SEG_NO_TORCH", "dense3.cu": "DSV41_DENSE3_NO_TORCH"}

# name -> (source, kept 1-based inclusive line ranges, {line: replacement}, footer)
COPIES: dict[str, tuple[str, tuple[tuple[int, int], ...], dict[int, str], str]] = {
    "decode.cuh": (f"{EXL3}/decode.cuh", ((1, 162),), {}, ""),
    "experts_grouped.cuh": (f"{EXL3}/experts_grouped.cuh", ((1, 307), (356, 356)), {}, ""),
    "exl3_experts.cu": (f"{EXL3}/experts.cu", ((1, 239),), {11: "namespace tf_exl3_experts {\n",
                                                            239: "}  // namespace tf_exl3_experts\n"},
                        "footer_exl3_experts"),
    "x3ld.cu": (f"{FAM}/x3ld.cu", ((1, 251), (297, 297)), {}, "footer_x3ld"),
    "x3pf.cu": (f"{FAM}/x3pf.cu", ((1, 204), (240, 240)), {}, "footer_x3pf"),
    "x3gm.cu": (f"{FAM}/x3gm.cu", ((1, 39), (45, 353), (374, 391), (423, 424)), {}, "footer_x3gm"),
    "linear.cu": (f"{EXL3}/linear.cu", ((1, 283), (288, 288)), {14: "namespace tf_exl3_linear {\n",
                                                                  288: "}  // namespace tf_exl3_linear\n"},
                  "footer_linear"),
    "x3seg.cu": (f"{FAM}/x3seg.cu", ((1, 430),), {}, "footer_x3seg"),
    "dense3.cu": (f"{FAM}/dense3.cu", ((1, 402),), {}, "footer_dense3"),
}

FOOTERS = {"footer_x3ld": footer_x3ld, "footer_x3pf": footer_x3pf, "footer_x3gm": footer_x3gm,
           "footer_exl3_experts": footer_exl3_experts, "footer_linear": footer_linear, "footer_x3seg": footer_x3seg,
           "footer_dense3": footer_dense3}


def blob_hash(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def render(src_root: Path, name: str) -> str:
    rel, ranges, replace, footer = COPIES[name]
    data = (src_root / rel).read_bytes()
    src = lines_of(data.decode())
    spans = ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)
    whole = name in PRELUDE
    if whole and (len(ranges) != 1 or ranges[0] != (1, len(src))):
        raise SystemExit(f"{name}: {rel} has {len(src)} lines now; update the range")
    what = f"the whole file, {PRELUDE[name]} defined" if whole else f"lines {spans}; torch includes and host code cut"
    out = [f"// Device code of {rel} (git blob {blob_hash(data)[:12]}; {what}),\n",
           "// generated and checked by zig/kernels/cuda/deepseek_v41/sync.py. Do not edit: change the Python source.\n"]
    if whole:
        out.append(f"#define {PRELUDE[name]}\n")
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
    ap.add_argument("--src", default=os.environ.get("TF_DSV41_PY_SRC", DEFAULT_SRC))
    args = ap.parse_args()
    root = Path(args.src)
    bad = 0
    for name in COPIES:
        text = render(root, name)
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
