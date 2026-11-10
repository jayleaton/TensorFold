"""Writes (or with --check checks) fn_qmmf_ld.cu: fn_qmmf.cu with output row stride `ldo`, the same computed values."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "zig/kernels/cuda/fn_qmmf.cu"
OUT = ROOT / "zig/kernels/cuda/fn_qmmf_ld.cu"

FOOTER = """
// Generated from fn_qmmf.cu (Ash Hart) by flashnext_qmmf_ld.py, do not edit: bf16 out, row stride `ldo`, its instances.
#define TF_QMMF_LD(BM, CLUSTER, FUSE) template __global__ void tf_fn_qmmf_ld::qmmf_kernel<3, BM, 64, 1, 4, 4, false, \\
    CLUSTER, FUSE>(const __nv_bfloat16*, const unsigned char*, const uint8_t*, float, void*, float*, int, int, int, \\
    int, int, int, int, int);
TF_QMMF_LD(16, false, false)
TF_QMMF_LD(32, false, false)
TF_QMMF_LD(64, false, false)
TF_QMMF_LD(16, true, false)
TF_QMMF_LD(32, true, false)
TF_QMMF_LD(64, true, false)
TF_QMMF_LD(64, false, true)
"""


def edit(text: str) -> str:
    def once(s: str, old: str, new: str) -> str:
        if s.count(old) != 1:
            raise SystemExit(f"fn_qmmf.cu changed: {old!r} found {s.count(old)} times")
        return s.replace(old, new)

    body = text[: text.index("// Block FP8 (FP8G, an fp32 scale per 64 inputs")]
    body = once(body, "namespace tf_fn_qmmf {", "namespace tf_fn_qmmf_ld {")
    body = once(body, "} // namespace tf_fn_qmmf\n", "} // namespace tf_fn_qmmf_ld\n")
    body = once(body, "int npad, int ldx,\n        int group) {", "int npad, int ldx,\n        int group, int ldo) {")
    body = once(body, "reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;",
                "reinterpret_cast<float*>(out) + static_cast<size_t>(row) * ldo + col;")
    body = once(body, "reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;",
                "reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * ldo + col;")
    head = ""
    return head + body + FOOTER


def main() -> int:
    want = edit(SRC.read_text())
    if "--check" in sys.argv:
        ok = OUT.exists() and OUT.read_text() == want
        print(("OK " if ok else "STALE ") + str(OUT.relative_to(ROOT)))
        return 0 if ok else 1
    OUT.write_text(want)
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
