# SPDX-License-Identifier: MIT
"""Goldens for the Zig image front end (zig/src/families/deepseek_v41/vision): images in every PNG / JPEG flavour
Pillow opens, each decoded by Pillow (``decode`` + ``to_rgb``) and preprocessed by the prod engine's own
``vision_prep.preprocess`` (8474f31). Writes ``OUT/<name>`` (the file bytes) and ``OUT/golden.jsonl``:
{name, size, rgb_sha256, grid [n_llm_h, n_llm_w, vit_h, vit_w], patches_sha256, digest, vids_sha256, vids_head}.

Run in the e1e2 image (Pillow 12.3, torch): ``python -I gen_prep_golden.py --py-src <tree>/src --config
<release config.json> --out DIR [--small]`` (``--small``: only the small images, the committed fixture set).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import zlib
from pathlib import Path


def png_bytes(arr, color: int, depth: int, interlace: bool = False, palette=None, trns=None, filt: int | None = None):
    """A PNG of ``arr`` (h x w x channels, ints already in range) with any colour type / depth (Pillow cannot write
    16-bit RGB, LA16, 2 / 4-bit grey or Adam7)."""

    import numpy as np

    arr = np.asarray(arr)
    h, w = arr.shape[:2]
    ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    arr = arr.reshape(h, w, ch)

    def rows(sub):
        sh, sw = sub.shape[:2]
        out = bytearray()
        bpp = max(1, ch * depth // 8)
        prev = bytes((sw * ch * depth + 7) // 8)
        for y in range(sh):
            if depth == 16:
                raw = sub[y].astype(">u2").tobytes()
            elif depth == 8:
                raw = sub[y].astype("u1").tobytes()
            else:
                bits = "".join(format(int(v), f"0{depth}b") for v in sub[y].reshape(-1))
                bits += "0" * (-len(bits) % 8)
                raw = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
            f = (y % 5) if filt is None else filt
            enc = bytearray(raw)
            for i in range(len(raw)):
                a = raw[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                if f == 1:
                    enc[i] = (raw[i] - a) & 255
                elif f == 2:
                    enc[i] = (raw[i] - b) & 255
                elif f == 3:
                    enc[i] = (raw[i] - ((a + b) >> 1)) & 255
                elif f == 4:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                    enc[i] = (raw[i] - pr) & 255
            out += bytes([f]) + enc
            prev = raw
        return bytes(out)

    if interlace:
        data = b""
        for x0, y0, dx, dy in ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)):
            sub = arr[y0::dy, x0::dx]
            if sub.shape[0] and sub.shape[1]:
                data += rows(sub)
    else:
        data = rows(arr)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color, 0, 0, int(interlace)))
    if palette is not None:
        out += chunk(b"PLTE", bytes(palette))
    if trns is not None:
        out += chunk(b"tRNS", bytes(trns))
    z = zlib.compress(data, 6)
    out += chunk(b"IDAT", z[: len(z) // 2]) + chunk(b"IDAT", z[len(z) // 2:])   # two IDATs
    return out + chunk(b"IEND", b"")


def images(small: bool):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(4101)

    def field(h, w, c, hi=256):
        y, x = np.mgrid[0:h, 0:w]
        base = (x * 7 + y * 3)[..., None] + np.arange(c)[None, None, :] * 50
        return ((base + rng.integers(0, 40, (h, w, c))) % hi).astype(np.int64)

    def pil(img, fmt, **kw):
        b = io.BytesIO()
        img.save(b, fmt, **kw)
        return b.getvalue()

    out = []
    h, w = (23, 37)
    out.append(("rgb8.png", png_bytes(field(h, w, 3), 2, 8)))
    out.append(("rgb8-adam7.png", png_bytes(field(h, w, 3), 2, 8, interlace=True)))
    out.append(("rgb16.png", png_bytes(field(h, w, 3, 65536), 2, 16)))
    out.append(("rgba8.png", png_bytes(field(h, w, 4), 6, 8)))
    out.append(("rgba16-adam7.png", png_bytes(field(h, w, 4, 65536), 6, 16, interlace=True)))
    out.append(("la8.png", png_bytes(field(h, w, 2), 4, 8)))
    out.append(("la16.png", png_bytes(field(h, w, 2, 65536), 4, 16)))
    for d in (1, 2, 4, 8):
        out.append((f"l{d}.png", png_bytes(field(h, w, 1, 1 << d), 0, d, interlace=d == 2)))
    out.append(("l16.png", png_bytes(field(h, w, 1, 600), 0, 16)))
    pal = rng.integers(0, 256, 3 * 200).tolist()
    for d in (1, 4, 8):
        out.append((f"p{d}.png", png_bytes(field(h, w, 1, 1 << d), 3, d, palette=pal[: 3 * (1 << min(d, 7))])))
    out.append(("p8-trns.png", png_bytes(field(h, w, 1, 200), 3, 8, palette=pal, trns=rng.integers(0, 256, 150).tolist())))
    out.append(("p8-trns1.png", png_bytes(field(h, w, 1, 200), 3, 8, palette=pal, trns=[255] * 7 + [0] + [255] * 3)))
    out.append(("rgb8-trns.png", png_bytes(field(h, w, 3), 2, 8, trns=[0, 1, 0, 2, 0, 3])))
    rgb = Image.fromarray(field(61, 83, 3).astype("u1"), "RGB")
    for sub in (0, 1, 2):
        out.append((f"q{sub}.jpg", pil(rgb, "JPEG", quality=87, subsampling=sub)))
        out.append((f"q{sub}-prog.jpg", pil(rgb, "JPEG", quality=70, subsampling=sub, progressive=True)))
    out.append(("gray.jpg", pil(rgb.convert("L"), "JPEG", quality=90)))
    out.append(("gray-prog.jpg", pil(rgb.convert("L"), "JPEG", quality=90, progressive=True)))
    out.append(("cmyk.jpg", pil(rgb.convert("CMYK"), "JPEG", quality=90)))
    out.append(("q100.jpg", pil(rgb, "JPEG", quality=100, subsampling=0)))
    out.append(("tiny.jpg", pil(Image.fromarray(field(1, 2, 3).astype("u1"), "RGB"), "JPEG", subsampling=2)))
    out.append(("narrow.png", png_bytes(field(400, 3, 3), 2, 8)))
    if small:
        return out
    big = Image.fromarray(field(900, 1500, 3).astype("u1"), "RGB")
    out.append(("big.jpg", pil(big, "JPEG", quality=85)))
    out.append(("big-prog.jpg", pil(big, "JPEG", quality=85, progressive=True, subsampling=1)))
    out.append(("big.png", png_bytes(field(700, 1300, 4), 6, 8)))
    out.append(("tall.png", png_bytes(field(3000, 200, 3), 2, 8)))
    out.append(("wide.jpg", pil(Image.fromarray(field(120, 2600, 3).astype("u1"), "RGB"), "JPEG")))
    out.append(("square544.png", png_bytes(field(544, 544, 3), 2, 8)))
    out.append(("odd.jpg", pil(Image.fromarray(field(333, 517, 3).astype("u1"), "RGB"), "JPEG", subsampling=2, quality=60)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--bias-vl", action="store_true", help="digests with TF_DSV41_BIAS_VL set")
    a = ap.parse_args()
    sys.path.insert(0, a.py_src)
    import os

    from tensorfold.families.deepseek_v41.cuda import vision_prep as V

    env = dict(os.environ)
    env.pop("TF_DSV41_BIAS_VL", None)
    if a.bias_vl:
        env["TF_DSV41_BIAS_VL"] = "/x"
    cfg_dir = Path(a.config).parent
    s = V.Settings.read(cfg_dir, env)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = V.Registry()
    with open(out / "golden.jsonl", "w") as g:
        for name, raw in images(a.small):
            (out / name).write_bytes(raw)
            img = V.decode(raw, s)
            rgb = img.tobytes()
            p = V.preprocess(img, s)
            vids = reg.vids(p.digest, p.tokens)
            rec = {"name": name, "size": list(p.size), "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
                   "grid": [p.n_llm_h, p.n_llm_w, p.n_vit_h, p.n_vit_w],
                   "patches_sha256": hashlib.sha256(p.patches.view(__import__("torch").int16).numpy().tobytes()).hexdigest(),
                   "digest": p.digest.hex(), "vids_head": vids[:4],
                   "vids_sha256": hashlib.sha256(b"".join(v.to_bytes(4, "little") for v in vids)).hexdigest()}
            g.write(json.dumps(rec) + "\n")
    print(f"wrote {out}/golden.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
