#!/usr/bin/env python3
"""ExLlamaV3 reference rows of a MiniMax-M3.1-Flash EXL3 pack for `tf-cuda-test mm3-layer-parity`, and the comparison.

    mm3_layer_ref.py prompt  --tok TOKENIZER_DIR --out REF            ids.npy: a chat-template conversation
    mm3_layer_ref.py ref     --pack PACK --exllamav3 SRC --out REF --layers 0,3,18,31 [--head-rows 8]
    mm3_layer_ref.py compare --ref REF --tf TF --layers 0,3,18,31     TF/L<n>/*.npy from the harness, a table

ref runs the embedding and every layer up to the last chosen one, one row of bsz 1 (padded batches are wrong on
this pack), and writes REF/L<n>/{in,attn,mid,mlp|routed+shared+pick+wts+scores,out}.npy, the layer's partials as
TensorFold's forward names them (attn = o_proj output, mid = in + attn, out = mid + feed-forward). REF/head_in.npy
holds rows of the last chosen layer's output and REF/head.npy their logits through the final norm and the head.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TURNS = [
    ("system", "You are a careful senior engineer. Answer precisely and point out real bugs only."),
    ("user", """I wrote this thread-safe LRU cache for a web service and it sometimes returns stale entries
under load. Can you review it?

```python
import threading
import time
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int, ttl: float):
        self.capacity = capacity
        self.ttl = ttl
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        item = self._data.get(key)
        if item is None:
            return None
        value, stamp = item
        if time.time() - stamp > self.ttl:
            with self._lock:
                del self._data[key]
            return None
        with self._lock:
            self._data.move_to_end(key)
        return value

    def put(self, key, value):
        with self._lock:
            self._data[key] = (value, time.time())
            self._data.move_to_end(key)
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def invalidate(self, prefix: str):
        for key in list(self._data):
            if key.startswith(prefix):
                self._data.pop(key, None)
```

The service runs 32 worker threads; invalidate() is called from an admin endpoint whenever a product is
edited, and get() is on the hot path. We measured about 40k requests per second at peak."""),
    ("ai", """There are three real problems, all in the paths that touch `_data` without the lock.

1. `get()` reads `self._data.get(key)` outside the lock, then deletes or moves the key under a second,
separate acquisition. Between the read and the delete another thread can `put()` a fresh value for the same
key, and your `del` then removes the fresh entry; between the read and `move_to_end` an `invalidate()` can
remove the key, and `move_to_end` raises `KeyError`.

2. The expiry check uses `time.time()`, which jumps when the wall clock is adjusted. Use `time.monotonic()`
for ages.

3. `invalidate()` iterates and pops without the lock. `list(self._data)` can raise `RuntimeError` if the
dictionary changes size during the copy, and a `put()` that lands after the copy is never invalidated, which
is exactly the stale entry you see after a product edit.

Hold the lock for the whole of `get()` and `invalidate()`; at 40k requests per second an uncontended
`threading.Lock` costs well under a microsecond, so it will not be your bottleneck."""),
    ("user", """Thanks. If I hold the lock for the whole get(), is there still a way for an edit to be missed?
The admin endpoint writes the new product to the database first and then calls invalidate(), while a worker
may have read the old row from the database just before the write and call put() just after invalidate().
How would you close that window without a global lock around the database read?"""),
]


def cmd_prompt(a):
    """The conversation through the tokenizer's chat template, ending at a generation prompt."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    msgs = [{"role": "assistant" if r == "ai" else r, "content": c} for r, c in TURNS]
    ids = list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ids.npy", np.asarray(ids, dtype=np.int32))
    print(f"{len(ids)} tokens -> {out / 'ids.npy'}; starts {tok.decode(ids[:12])!r}")


def save(d: Path, name: str, t):
    np.save(d / f"{name}.npy", t.detach().float().cpu().numpy().reshape(-1, t.shape[-1]))


def block_partials(block, x, params, torch, d: Path):
    """block.forward(x) split at TensorFold's seams, each partial saved; returns the block's own output."""
    y = block.attn_norm.forward(x, dict(params), out_dtype=torch.half)
    attn = block.attn.forward(y, dict(params)).float()
    mid = x + attn
    y2 = block.mlp_norm.forward(mid, dict(params), out_dtype=torch.half)
    mlp = block.mlp
    if hasattr(mlp, "routing_fn"):
        got = {}
        fn = mlp.routing_fn

        def spy(bsz, cfg, z, prm):
            sel, w = fn(bsz, cfg, z, prm)
            got["sel"], got["w"] = sel.clone(), w.clone()
            return sel, w
        mlp.routing_fn = spy
        full = mlp.forward(y2, dict(params, residual=mid)).float()
        mlp.routing_fn = fn
        shared = mlp.shared_experts.forward(y2, dict(params)).float()
        cfg = mlp.routing_cfg
        logits = y2.view(-1, y2.shape[-1]).float() @ cfg.gate_tensor.float()
        scores = logits.sigmoid() + cfg.e_score_correction_bias.float().view(1, -1)
        np.save(d / "pick.npy", got["sel"].cpu().numpy().astype(np.int32))
        np.save(d / "wts.npy", got["w"].float().cpu().numpy())
        save(d, "router_logits", logits)
        save(d, "scores", scores)
        save(d, "routed", full - shared)
        save(d, "shared", shared)
    else:
        full = mlp.forward(y2, dict(params)).float()
        save(d, "mlp", full)
    out = mid + full
    save(d, "in", x)
    save(d, "attn", attn)
    save(d, "mid", mid)
    save(d, "out", out)
    whole = block.forward(x.clone(), dict(params)).float()
    save(d, "block_out", whole)
    diff = (whole - out).abs().max().item()
    print(f"    split vs block.forward: max-abs {diff:.3e} (x {x.dtype}, attn {attn.dtype}, mlp {full.dtype})")
    return whole


def cmd_ref(a):
    sys.path.insert(0, a.exllamav3)
    import torch
    from exllamav3 import Config, Model
    from exllamav3.util.memory import free_mem
    layers = sorted(int(v) for v in a.layers.split(","))
    out = Path(a.out)
    ids = torch.from_numpy(np.load(out / "ids.npy").astype(np.int64)).view(1, -1)
    dev = torch.device(a.device)
    c = Config.from_directory(a.pack)
    c.override_dynamic_seq_len(max(2048, ids.shape[1]))
    m = Model.from_config(c)
    params = {}
    x = m.prepare_inputs(ids, params)

    def run(idx, x, body):
        c.stc.begin_deferred_load()
        mod = m.modules[idx]
        mod.load(dev)
        c.stc.end_deferred_load()
        p = dict(params)
        y = body(mod, mod.prepare_for_device(x, p), p)
        mod.unload()
        c.stc.close()
        c.stc.release_arena()
        free_mem()
        return y

    with torch.inference_mode():
        x = run(0, x, lambda mod, v, p: mod.forward(v, p).float())
        save(out, "emb", x)
        for layer in range(layers[-1] + 1):
            idx = m.first_block_idx + layer
            if layer in layers:
                d = out / f"L{layer}"
                d.mkdir(parents=True, exist_ok=True)
                print(f" -- layer {layer} ({m.modules[idx].key}): partials -> {d}", flush=True)
                x = run(idx, x, lambda mod, v, p: block_partials(mod, v.float(), p, torch, d))
            else:
                x = run(idx, x, lambda mod, v, p: mod.forward(v, p).float())
        n = x.shape[1]
        rows = sorted(set(np.linspace(0, n - 1, a.head_rows).round().astype(int).tolist()))
        hx = x[:, rows, :].contiguous()
        np.save(out / "head_in.npy", hx.float().cpu().numpy().reshape(len(rows), -1))
        norm = run(m.logit_layer_idx - 1, hx, lambda mod, v, p: mod.forward(v, p))
        logits = run(m.logit_layer_idx, norm, lambda mod, v, p: mod.forward(v, p).float())
        save(out, "head", logits)
        (out / "head_rows.json").write_text(json.dumps(rows))
    print(f" -- {n} rows, layers {layers}, head rows {rows}")


def bf16(a):
    """`<u2` bf16 bits (the harness' bf16 buffers) as float32."""
    return (a.astype(np.uint32) << 16).view(np.float32) if a.dtype == np.uint16 else a.astype(np.float32)


def stats(tf, ref):
    tf, ref = bf16(tf).astype(np.float64), bf16(ref).astype(np.float64)
    d = np.abs(tf - ref)
    num = (tf * ref).sum(axis=1)
    den = np.linalg.norm(tf, axis=1) * np.linalg.norm(ref, axis=1) + 1e-30
    row_cos = num / den
    cos = (tf * ref).sum() / (np.linalg.norm(tf) * np.linalg.norm(ref) + 1e-30)
    rel = np.linalg.norm(tf - ref) / (np.linalg.norm(ref) + 1e-30)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "cos": float(cos), "worst_row_cos": float(row_cos.min()),
            "worst_row": int(row_cos.argmin()), "rel_l2": float(rel), "ref_rms": float(np.sqrt((ref ** 2).mean()))}


def router(tf_pick, ref_pick, scores, tf_w, ref_w):
    """Rows whose top-4 sets agree, the overlap, and each disagreement's reference margin (4th - 5th score)."""
    same, overlap, margins, wdiff = 0, 0, [], 0.0
    for r in range(ref_pick.shape[0]):
        a, b = set(tf_pick[r].tolist()), set(ref_pick[r].tolist())
        overlap += len(a & b)
        if a == b:
            same += 1
            ta = dict(zip(tf_pick[r].tolist(), tf_w[r].tolist()))
            wdiff = max(wdiff, max(abs(ta[e] - w) for e, w in zip(ref_pick[r].tolist(), ref_w[r].tolist())))
        else:
            s = np.sort(scores[r])[::-1]
            margins.append(float(s[3] - s[4]))
    n = ref_pick.shape[0]
    return {"rows": n, "same_set": same, "agree": same / n, "overlap": overlap / (4 * n), "w_max_abs_agreeing": wdiff,
            "disagree_margins": sorted(margins)[:16], "max_disagree_margin": max(margins) if margins else 0.0}


def gap(scores):
    """Each row's 4th minus 5th largest score."""
    s = -np.sort(-scores.astype(np.float64), axis=1)
    return s[:, 3] - s[:, 4]


def flips(tf_routed, ref_routed, tf_pick, ref_pick, ref_scores, tf_scores):
    """Routed cosine over the rows whose top-4 sets agree, and every flipped row with both sides' 4th-5th gap."""
    same = np.array([set(x.tolist()) == set(y.tolist()) for x, y in zip(tf_pick, ref_pick)])
    out = {"rows": int(same.size), "agreeing": int(same.sum()), "flips": int((~same).sum())}
    if same.any():
        out.update({f"agreeing_{k}": v for k, v in stats(tf_routed[same], ref_routed[same]).items()})
    g_ref, g_tf = gap(ref_scores), gap(tf_scores)
    out["flipped"] = [{"row": int(r), "ref_gap": float(g_ref[r]), "tf_gap": float(g_tf[r]),
                       "tf_only": sorted(set(tf_pick[r].tolist()) - set(ref_pick[r].tolist())),
                       "ref_only": sorted(set(ref_pick[r].tolist()) - set(tf_pick[r].tolist()))}
                      for r in np.flatnonzero(~same)]
    out["max_flip_gap"] = max((max(f["ref_gap"], f["tf_gap"]) for f in out["flipped"]), default=0.0)
    return out


def cmd_compare(a):
    ref, tf = Path(a.ref), Path(a.tf)
    report = {}
    for layer in (int(v) for v in a.layers.split(",")):
        r, t = ref / f"L{layer}", tf / f"L{layer}"
        load = lambda d, n: np.load(d / f"{n}.npy")
        rows = {}
        inp = load(r, "in")
        for name in ("attn", "mid", "mlp", "routed", "shared", "out"):
            if (r / f"{name}.npy").exists() and (t / f"{name}.npy").exists():
                rows[name] = stats(load(t, name), load(r, name))
        rows["update"] = stats(load(t, "out") - inp, load(r, "out") - inp)
        rows["split_vs_block"] = stats(load(r, "out"), load(r, "block_out"))
        if (t / "pick.npy").exists():
            rows["router"] = router(load(t, "pick"), load(r, "pick"), load(r, "scores"), load(t, "wts"), load(r, "wts"))
            rows["router_logits"] = stats(load(t, "router_logits"), load(r, "router_logits"))
            # TF's scores: its logits through the reference's sigmoid plus the bias the reference's scores carry
            ref_lg = bf16(load(r, "router_logits")).astype(np.float64)
            bias = (bf16(load(r, "scores")).astype(np.float64) - 1 / (1 + np.exp(-ref_lg)))[0]
            tf_scores = 1 / (1 + np.exp(-bf16(load(t, "router_logits")).astype(np.float64))) + bias
            rows["routed_flips"] = flips(bf16(load(t, "routed")), bf16(load(r, "routed")), load(t, "pick"),
                                         load(r, "pick"), bf16(load(r, "scores")), tf_scores)
        if (t / "emb.npy").exists():
            rows["emb"] = stats(load(t, "emb"), np.load(ref / "emb.npy"))
        report[f"L{layer}"] = rows
    heads = sorted(tf.glob("L*/head.npy"))
    if heads:
        h_tf, h_ref = np.load(heads[0]), np.load(ref / "head.npy")
        head = stats(h_tf, h_ref)
        head["top1_agree"] = float((h_tf.argmax(1) == h_ref.argmax(1)).mean())
        t5 = [len(set(np.argsort(-x)[:5]) & set(np.argsort(-y)[:5])) / 5 for x, y in zip(h_tf, h_ref)]
        head["top5_overlap"] = float(np.mean(t5))
        report["head"] = head
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    for k, rows in report.items():
        items = rows.items() if k != "head" else [("logits", rows)]
        for name, s in items:
            if "cos" in s:
                print(f"{k:5s} {name:15s} max-abs {s['max_abs']:.4e} mean-abs {s['mean_abs']:.4e} cos {s['cos']:.6f} "
                      f"worst-row {s['worst_row_cos']:.6f} (row {s['worst_row']}) rel-l2 {s['rel_l2']:.3e} ref-rms {s['ref_rms']:.3e}")
            else:
                print(f"{k:5s} {name:15s} {json.dumps(s)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompt")
    p.add_argument("--tok", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("ref")
    p.add_argument("--pack", required=True)
    p.add_argument("--exllamav3", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="0,3,18,31")
    p.add_argument("--head-rows", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("compare")
    p.add_argument("--ref", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--layers", default="0,3,18,31")
    p.add_argument("--out")
    a = ap.parse_args()
    {"prompt": cmd_prompt, "ref": cmd_ref, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
