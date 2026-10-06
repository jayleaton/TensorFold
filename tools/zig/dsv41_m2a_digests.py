#!/usr/bin/env python3
"""DeepSeek-V4.1 M2a reference: the Python engine's whole backbone on ONE GPU at rank 0 of TP=2 (the peer's half of
every exchange zeros, Engram rows a function of the row index: dsv41_m1_capture.py's ZeroPeer / IndexRows), a prompt,
then decode windows of 1, 3 and 16 rows. Per window: each block's streams after it (``Forward.trace``) as a SHA-256,
and this rank's logits (fp32, raw). No launch is recorded: the digests are what tf-dsv41-m2a compares against our
forward, layer by layer.

    python dsv41_m2a_digests.py --pack PACK --out DIR [--prompt 1536] [--seed 4101]
Writes DIR/digests.json (meta, windows: ids, per-layer digests, logits file + digest) and DIR/logits-w<n>.bin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WINDOWS = (1, 3, 16)


def digest(t) -> str:
    import torch

    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=4101)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import torch
    from dsv41_m1_capture import IndexRows, ZeroPeer
    from tensorfold.families.deepseek_v41.cuda import weights as W
    from tensorfold.families.deepseek_v41.cuda.config import Config
    from tensorfold.families.deepseek_v41.cuda.csa2.backend import Triton
    from tensorfold.families.deepseek_v41.cuda.engram_host import NgramHasher, token_map
    from tensorfold.families.deepseek_v41.cuda.forward import Forward, Seg

    t0 = time.time()
    dev = torch.device("cuda:0")
    pack = Path(a.pack)
    cfg = Config.from_file(pack / "config.json")
    tmap, _ = token_map(pack / "tokenizer.json")
    src = W.Shards(pack)
    loader = W.Loader(cfg, src, 0, 2, dev)
    vh = cfg.vocab_size // 2
    embed = loader._dev(src.rows("embed.weight", 0, vh).to(torch.bfloat16))
    head, norm = loader.x3("head", "col"), loader.native("norm.weight")
    layers = [loader.layer(L) for L in range(cfg.num_hidden_layers)]
    tree = W.RankW(0, 2, 0, embed, layers, norm, head, None, {})
    load_s = time.time() - t0
    rows = {L: IndexRows(cfg.engram_head_dim) for L in cfg.engram_layer_ids}
    fw = Forward(cfg, tree, csa2=Triton(128, cfg.num_attention_heads // 2, dev), comm=ZeroPeer(), device=dev,
                 limit=a.limit, chunk=128, engram_rows=rows, hasher=NgramHasher(cfg, tmap), slots=1,
                 contiguous=True, pchunk=2048)
    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(3, vh, (a.prompt + sum(WINDOWS),), generator=g).tolist()
    fw.prompt(ids[:a.prompt])
    pos = a.prompt
    meta = {"pack": str(pack), "prompt": a.prompt, "limit": a.limit, "seed": a.seed, "load_s": round(load_s, 1),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "env": {k: v for k, v in os.environ.items() if k.startswith("TF_DSV41")}, "peer": "zeros",
            "engram_rows": "index function (dsv41_m1_capture.IndexRows)", "windows": []}
    for n in WINDOWS:
        fw.trace = {}
        start = fw.slot.pos
        logits = fw.run([Seg(0, start, tuple(ids[pos:pos + n]))])
        torch.cuda.synchronize()
        lb = out / f"logits-w{n}.bin"
        lb.write_bytes(logits.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        meta["windows"].append({"n": n, "start": start, "ids": ids[pos:pos + n],
                                "layers": {str(L): digest(t) for L, t in sorted(fw.trace.items())},
                                "logits": {"file": lb.name, "shape": list(logits.shape), "sha256": digest(logits)}})
        fw.trace = None
        fw.keep(0, n - 1)
        pos += n
    meta["run_s"] = round(time.time() - t0 - load_s, 1)
    (out / "digests.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"windows": len(meta["windows"]), "layers": len(meta["windows"][0]["layers"]),
                      "load_s": meta["load_s"], "run_s": meta["run_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
