#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The vision gate's Python reference (dsq-zig-vision-1): prod's engine (8474f31, /dsv41-tf/src) on one load at TP=2.

1. Images: ``IMAGES`` generated with Pillow (seeded), each through prod's ``vision_prep`` (decode, preprocess, virtual
   ids) into VDIR: ``<name>.patches.bin`` (bf16) and ``manifest.json`` (grids, digest, ids), the Zig side's
   TF_DSV41_VISION_HOLD format.
2. The tower (rank 0, ``vision.Tower`` from the pack's vision tensors on this GPU): each image's stages as
   ``vision.Tower.vit`` / ``align`` compute them (the same helpers, so the same bits; checked against ``Tower.span``)
   into ``<name>.<stage>.bin`` (embed, block0, block1, block15, block31, vit, gelu) and ``<name>.span.bin``.
3. ``PROMPTS``: text ids around the images' spans of virtual ids; rank 0 holds the images' rows in ``vision.STORE``,
   then dsv41_m2b_ref.py's own ``one`` runs each prompt from a fresh slot (state dump, greedy steps, digests) into
   OUT/p<i>/ (``--prompts 1,2,3``: prompt i is ``PROMPTS[i - 1]``), so tf-dsv41-m1 m2b's prefill gate reads them as
   any M2b reference; ``OUT/trace.jsonl`` holds every prompt and its tokens for ``tf-dsv41-m1 generate``.

Image rows route with the text bias (TF_DSV41_BIAS_VL unset: the Zig prefill refuses the bias for now).

    python -u -B vision_ref.py --vdir VDIR -- --pack PACK --out OUT --layers 0-24 --steps 32 --prompts 1,2,3
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # tools/zig: dsv41_m2b_ref and its helpers
import dsv41_m2b_ref as M2B  # noqa: E402

IMAGES = {  # name -> (width, height, format, mode)
    "photo": (640, 480, "JPEG", "RGB"),
    "wide": (1280, 400, "PNG", "RGB"),
    "small": (200, 150, "PNG", "RGBA"),
}
# each prompt: text lengths and image names in order (ints: that many random text ids)
PROMPTS = [[37, "photo", 21], [15, "wide", 9, "small", 30], ["small", 12]]
STAGES = ("embed", "block0", "block1", "block15", "block31", "vit", "gelu")


def make_image(name: str):
    import numpy as np
    from PIL import Image

    w, h, fmt, mode = IMAGES[name]
    rng = np.random.default_rng(sum(map(ord, name)))
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) * 3) % 256], -1)
    for _ in range(6):                          # a few flat boxes (screenshot-like edges)
        x0, y0 = rng.integers(0, w // 2), rng.integers(0, h // 2)
        base[y0:y0 + h // 4, x0:x0 + w // 4] = rng.integers(0, 256, 3)
    arr = (base + rng.integers(0, 24, base.shape)).clip(0, 255).astype("u1")
    img = Image.fromarray(arr, "RGB")
    if mode == "RGBA":
        img.putalpha(Image.fromarray((x * 255 // max(w - 1, 1)).astype("u1"), "L"))
    b = io.BytesIO()
    img.save(b, fmt, **({"quality": 88} if fmt == "JPEG" else {}))
    return b.getvalue()


def prepare(vdir: Path, pack: Path):
    """Step 1: every image's Prepared (prod's front end) and the manifest."""

    from tensorfold.families.deepseek_v41.cuda import vision_prep as VP

    env = {k: v for k, v in os.environ.items() if k != "TF_DSV41_BIAS_VL"}
    s = VP.Settings.read(pack, env)
    reg = VP.Registry()
    vdir.mkdir(parents=True, exist_ok=True)
    out, man = {}, []
    for name in IMAGES:
        raw = make_image(name)
        (vdir / f"{name}.img").write_bytes(raw)
        p = VP.preprocess(VP.decode(raw, s), s)
        p.vids = reg.vids(p.digest, p.tokens)
        (vdir / f"{name}.patches.bin").write_bytes(p.patches.contiguous().view(__import__("torch").int16).numpy().tobytes())
        man.append({"name": name, "vit_h": p.n_vit_h, "vit_w": p.n_vit_w, "llm_h": p.n_llm_h, "llm_w": p.n_llm_w,
                    "digest": p.digest.hex(), "vids": p.vids, "patches": f"{name}.patches.bin", "size": list(p.size)})
        out[name] = p
    (vdir / "manifest.json").write_text(json.dumps({"images": man}) + "\n")
    return out


def tower_dumps(vdir: Path, pack: Path, images: dict, dev):
    """Step 2 (rank 0): the stages and span rows of every image; returns the encoder (its tower) for step 3."""

    import torch
    import torch.nn.functional as F

    from tensorfold.families.deepseek_v41.cuda import vision as V

    enc = V.Encoder.load(pack, dev)
    t = enc.tower
    for name, img in images.items():
        dumps = {}
        with torch.inference_mode():
            tt = t.t
            x = t.lin(img.patches.to(t.device, torch.bfloat16).flatten(1), "vision.patch_embed.proj")
            dumps["embed"] = x.clone()
            n, dim = x.shape
            hd = dim // t.heads
            cos, sin = V._rope(img.n_vit_h, img.n_vit_w, hd // 2, t.theta, x.device)
            for i in range(t.layers):
                p = f"vision.blocks.{i}"
                h = V._rms(x, tt[f"{p}.norm1.weight"])
                q, k, v = (a.view(n, t.heads, hd) for a in t.lin(h, f"{p}.attn.wqkv").chunk(3, dim=-1))
                q, k = V._rotate(q, cos, sin), V._rotate(k, cos, sin)
                o = V._sdpa(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
                x = x + t.lin(o.transpose(0, 1).reshape(n, -1), f"{p}.attn.wo")
                h = V._rms(x, tt[f"{p}.norm2.weight"])
                g, u = t.lin(h, f"{p}.mlp.w1").chunk(2, dim=-1)
                x = x + t.lin(F.silu(g) * u, f"{p}.mlp.w2")
                if f"block{i}" in STAGES:
                    dumps[f"block{i}"] = x.clone()
            vit = V._rms(x, tt["vision.norm.weight"])
            dumps["vit"] = vit
            r = t.ratio
            y = vit.view(img.n_vit_h, img.n_vit_w, -1).permute(2, 0, 1)
            y = F.pad(y, (0, -img.n_vit_w % r, 0, -img.n_vit_h % r))
            y = F.unfold(y.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
            a1 = F.gelu(t.lin(y, "aligner.w1"))
            dumps["gelu"] = a1
            span = t.span(img)
            staged = t.lin(a1, "aligner.w2")
            if not torch.equal(span[1:-1][torch.tensor(img.types[1:-1]) == 1], staged.to(torch.bfloat16)):
                raise SystemExit(f"{name}: the staged tower differs from Tower.span (the reference is not prod's)")
        for k, v in dumps.items():
            (vdir / f"{name}.{k}.bin").write_bytes(v.contiguous().view(torch.int16).cpu().numpy().tobytes())
        (vdir / f"{name}.span.bin").write_bytes(span.contiguous().view(torch.int16).cpu().numpy().tobytes())
        print(f"vision ref {name}: {img.n_vit_h} x {img.n_vit_w} patches, span {span.shape[0]} rows, "
              f"sha256 {hashlib.sha256(span.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]}", flush=True)
    return enc


_ORIG_ONE = M2B.one
_STATE: dict = {}


def prompt_ids(spec, images, vocab, seed):
    import torch

    g = torch.Generator().manual_seed(seed)
    ids = []
    for part in spec:
        ids.extend(images[part].vids if isinstance(part, str) else torch.randint(3, vocab, (part,), generator=g).tolist())
    return ids


def one_image(rank, a, fw, cfg, greedy, Seg, digest, prompt, out, *rest):
    """dsv41_m2b_ref.one on prompt ``PROMPTS[prompt - 1]``: its ids in place of the random ones."""

    import torch

    from tensorfold.families.deepseek_v41.cuda import vision as V

    vdir, pack = Path(os.environ["DSV41_VISION_DIR"]), Path(a.pack)
    if "images" not in _STATE:
        _STATE["images"] = prepare(vdir, pack) if rank == 0 else None
        fw.comm.barrier() if hasattr(fw.comm, "barrier") else None
        if rank != 0:
            _STATE["images"] = _load_manifest(vdir)
        if rank == 0:
            enc = tower_dumps(vdir, pack, _STATE["images"], fw.device)
            imgs = list(_STATE["images"].values())
            V.STORE.hold(enc.table(imgs))
            del enc.tower.t                      # the rows are in the store; the tower's weights go
    ids = prompt_ids(PROMPTS[prompt - 1], _STATE["images"], cfg.vocab_size, a.seed + prompt)
    real = torch.randint

    def once(*x, **k):                                       # one() draws its prompt with torch.randint, first
        torch.randint = real
        return torch.tensor(ids)

    torch.randint = once
    try:
        return _ORIG_ONE(rank, a, fw, cfg, greedy, Seg, digest, len(ids), out, *rest)
    finally:
        torch.randint = real


class _Vids:
    def __init__(self, vids):
        self.vids = vids


def _load_manifest(vdir: Path) -> dict:
    import time

    for _ in range(600):                          # rank 0 writes it before its tower runs
        f = vdir / "manifest.json"
        if f.exists():
            return {e["name"]: _Vids(e["vids"]) for e in json.loads(f.read_text())["images"]}
        time.sleep(1)
    raise SystemExit("vision ref: no manifest from rank 0")


M2B.one = one_image


def main() -> int:
    argv = sys.argv[1:]
    if "--vdir" not in argv or "--" not in argv:
        raise SystemExit(__doc__)
    os.environ["DSV41_VISION_DIR"] = argv[argv.index("--vdir") + 1]
    os.environ.pop("TF_DSV41_BIAS_VL", None)
    sys.argv = [sys.argv[0]] + argv[argv.index("--") + 1:]
    rc = M2B.main()
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    lines = []
    for i in range(1, len(PROMPTS) + 1):
        ref = json.loads((out / f"p{i}" / "ref.json").read_text())
        lines.append(json.dumps({"prompt": ref["prompt_ids"], "tokens": ref["tokens"]}))
    (out / "trace.jsonl").write_text("\n".join(lines) + "\n")
    print(f"vision ref: {len(lines)} image prompts, trace {out / 'trace.jsonl'}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
