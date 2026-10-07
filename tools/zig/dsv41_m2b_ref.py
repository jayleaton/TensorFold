#!/usr/bin/env python3
"""DeepSeek-V4.1 M2b reference: the Python engine at TP=2 on two GPUs of one host (NCCL, tensorfold.cuda.comm), the
backbone's layers 0..L and the head, Engram rows a function of the row index (dsv41_m1_capture.IndexRows, as M2a): a
prompt (Python's own prefill), each rank's slot state dumped under our forward's role names, then greedy decoding one
token a window. Recorded: every token (the first is the prompt's greedy pick), each rank's logits digest a step, each
rank's per-block stream digests of the first decode window, the run's Triton variants (rank 0, the decode windows) and
this host's rope tables.

    python dsv41_m2b_ref.py --pack PACK --out DIR [--layers 0-27] [--prompt 1536] [--steps 64] [--port 29611]
Writes DIR/ref.json, DIR/rank<r>/state/*, DIR/rope.bin, DIR/aot/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(t) -> str:
    import torch

    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def rank_main(rank: int, a) -> None:
    sys.path.insert(0, str(HERE))
    import torch

    dev_index = rank if a.rank < 0 else 0     # two hosts: each rank on its own GPU 0
    torch.cuda.set_device(dev_index)
    from dsv41_m1_capture import IndexRows, pack_aot, specialization
    from tensorfold.cuda.comm import open_comm
    from tensorfold.families.deepseek_v41.cuda import rope as RO
    from tensorfold.families.deepseek_v41.cuda import weights as W
    from tensorfold.families.deepseek_v41.cuda.config import Config
    from tensorfold.families.deepseek_v41.cuda.csa2.backend import Triton
    from tensorfold.families.deepseek_v41.cuda.engram_host import NgramHasher, token_map
    from tensorfold.families.deepseek_v41.cuda.forward import Forward, Seg, greedy

    t0 = time.time()
    out = Path(a.out)
    dev = torch.device(f"cuda:{dev_index}")
    comm = open_comm(rank, 2, a.master, a.port)
    pack = Path(a.pack)
    cfg = Config.from_file(pack / "config.json")
    tmap, _ = token_map(pack / "tokenizer.json")
    src = W.Shards(pack)
    loader = W.Loader(cfg, src, rank, 2, dev)
    vh = cfg.vocab_size // 2
    embed = loader._dev(src.rows("embed.weight", rank * vh, (rank + 1) * vh).to(torch.bfloat16))
    head, norm = loader.x3("head", "col"), loader.native("norm.weight")
    lo, hi = (int(v) for v in a.layers.split("-"))
    layers = []
    for L in range(lo, hi + 1):
        layers.append(loader.layer(L))
        if rank == 0:
            print(f"loaded layer {L} at {time.time() - t0:.0f} s", flush=True)
    tree = W.RankW(rank, 2, rank * vh, embed, layers, norm, head, None, {})
    rows = {L: IndexRows(cfg.engram_head_dim) for L in cfg.engram_layer_ids}
    fw = Forward(cfg, tree, csa2=Triton(128, cfg.num_attention_heads // 2, dev), comm=comm, device=dev,
                 limit=a.limit, chunk=128, engram_rows=rows, hasher=NgramHasher(cfg, tmap), slots=1,
                 contiguous=True, pchunk=2048)
    load_s = time.time() - t0
    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(3, cfg.vocab_size, (a.prompt,), generator=g).tolist()
    logits = fw.prompt(ids)
    tok = int(greedy(fw, logits[-1:])[0])
    torch.cuda.synchronize()
    if rank == 0:
        print(f"prompt of {a.prompt} at {time.time() - t0:.0f} s, first token {tok}", flush=True)
    # the slot after the prompt, by our forward's role names (block.zig: s.L<i>.swa / comp .v .s, .ik, .carry)
    sd = out / f"rank{rank}" / "state"
    sd.mkdir(parents=True, exist_ok=True)
    roles, s0 = {}, fw.slot

    def put(role, t):
        f = sd / f"{role}.bin"
        f.write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        roles[role] = f.name

    for L in range(lo, hi + 1):
        v, sc = s0.swa[L].kernel()
        put(f"s.L{L}.swa.v", v)
        put(f"s.L{L}.swa.s", sc)
        if L in s0.comp:
            v, sc = s0.comp[L].kernel()
            put(f"s.L{L}.comp.v", v)
            put(f"s.L{L}.comp.s", sc)
            put(f"s.L{L}.ik", s0.ikeys[L].kernel())
        if L in s0.carry:
            put(f"s.L{L}.carry", s0.carry[L])
    (sd / "state.json").write_text(json.dumps({"pos": s0.pos, "tail": list(s0.tail), "roles": roles}, indent=1) + "\n")
    rec = None
    if rank == 0 or a.rank >= 0:   # the AOT set and rope tables on each host of a two-host run
        import triton_aot_manifest as aot

        rec = aot.Recorder().install()
    tokens, digests, layers_d = [tok], [], {}
    for step in range(a.steps):
        fw.trace = {} if step == 0 else None
        lg = fw.run([Seg(0, fw.slot.pos, (tok,))])
        if step == 0:
            layers_d = {str(L): digest(t) for L, t in sorted(fw.trace.items())}
        digests.append(digest(lg))
        tok = int(greedy(fw, lg)[0])
        tokens.append(tok)
        fw.keep(0, 0)
    fw.trace = None
    # windows of 2.. rows after every recorded step (nothing recorded comes from them): their Triton variants join
    # the AOT set (a 1 specializes), for M3's lanes gates whose verify windows hold drafts
    for n in a.extra_rows:
        fw.run([Seg(0, fw.slot.pos, tuple([tok] * n))])
        fw.keep(0, n - 1)
    torch.cuda.synchronize()
    run_s = time.time() - t0 - load_s
    mine = {"rank": rank, "load_s": round(load_s, 1), "run_s": round(run_s, 1), "tokens": tokens,
            "logits_sha256": digests, "layers_first_window": layers_d, "prompt_ids": ids}
    (out / f"rank{rank}.json").write_text(json.dumps(mine) + "\n")
    if rank == 0 or a.rank >= 0:   # the AOT set and rope tables on each host of a two-host run
        tabs = RO.tables(cfg, a.limit + 2048)
        (out / "rope.bin").write_bytes(b"DSV41RP1" + struct.pack("<II", *tabs[0].cs.shape)
                                       + tabs[0].cs.contiguous().numpy().tobytes()
                                       + tabs[1].cs.contiguous().numpy().tobytes())
        rec.dump(out / "launches.json")
        cache = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))
        aot.build(cache, out / "launches.json", out / "manifest.json", None)
        pack_aot(out / "manifest.json", cache, out / "aot", specialization())
        print(f"rank 0: {len(tokens)} tokens, load {load_s:.0f} s, run {run_s:.0f} s", flush=True)
    comm.barrier()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-24")
    ap.add_argument("--prompt", type=int, default=1536)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--limit", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=4101)
    ap.add_argument("--port", type=int, default=29611)
    ap.add_argument("--rank", type=int, default=-1, help="two hosts: this host's rank (one process); -1: both ranks here")
    ap.add_argument("--extra-rows", type=lambda v: [int(x) for x in v.split(",") if x], default=[2, 3, 4, 5, 6, 16],
                    help="windows of these rows after the steps (their Triton variants in the AOT set)")
    ap.add_argument("--master", default="127.0.0.1", help="rank 0's address (two hosts: the head's CX7 address)")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.rank >= 0:                         # two hosts: this host's rank alone (the other host runs the other)
        rank_main(a.rank, a)
        mine = json.loads((Path(a.out) / f"rank{a.rank}.json").read_text())
        ranks = [None, None]
        ranks[a.rank] = {k: mine[k] for k in ("load_s", "run_s", "logits_sha256", "layers_first_window")}
        meta = {"layers": a.layers, "prompt": a.prompt, "steps": a.steps, "seed": a.seed, "limit": a.limit,
                "tokens": mine["tokens"], "prompt_ids": mine["prompt_ids"], "host_rank": a.rank,
                "env": {k: v for k, v in os.environ.items() if k.startswith("TF_DSV41")}, "ranks": ranks}
        (Path(a.out) / "ref.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(json.dumps({"rank": a.rank, "tokens": len(mine["tokens"]), "first": mine["tokens"][:8]}), flush=True)
        return 0
    import torch.multiprocessing as mp

    mp.spawn(rank_main, args=(a,), nprocs=2, join=True)
    r0 = json.loads((Path(a.out) / "rank0.json").read_text())
    r1 = json.loads((Path(a.out) / "rank1.json").read_text())
    if r0["tokens"] != r1["tokens"]:
        print("the ranks' greedy tokens differ", flush=True)
        return 1
    meta = {"layers": a.layers, "prompt": a.prompt, "steps": a.steps, "seed": a.seed, "limit": a.limit,
            "tokens": r0["tokens"], "prompt_ids": r0["prompt_ids"],
            "env": {k: v for k, v in os.environ.items() if k.startswith("TF_DSV41")},
            "ranks": [{k: r[k] for k in ("load_s", "run_s", "logits_sha256", "layers_first_window")} for r in (r0, r1)]}
    (Path(a.out) / "ref.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"tokens": len(meta["tokens"]), "first": meta["tokens"][:8]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
