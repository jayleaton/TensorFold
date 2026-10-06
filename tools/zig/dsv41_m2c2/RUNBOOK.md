# M2c-2: the Zig forward on the two DGX Sparks (whole model, TP=2, RoCE)

**What it proves.** Our Zig forward over all 40 backbone layers plus the head is gated against the Python engine at TP=2 on the same nodes, the same q28-v2 pack and the same state:
- every greedy token;
- every step's logits on each rank, bit for bit;
- every block's output of the first window.

The settings match prod's: expert top-p 0.85, M1's prod kernel knobs (`tools/zig/dsv41_m1/prod-knobs.env`), the exchanges over RoCE (small) and NCCL (large).

The Python side is prod's own engine tree, 8474f31, in prod's image and cache volume, with the R1 knobs left off. R1 is bit-identical by design, and the Zig emitter implements M1's prod launch sequence.

**Not in this run:**
- the DSpark drafter (greedy decoding alone; DSpark is M3);
- our own prefill: we start from the Python engine's state after its prompt;
- timing as a gate: our forward is eager, without graphs, so its per-token time is informational.

**Duration:** about 45–60 min for one seed; prod is down for all of it.

| step | where | what | time |
|---|---|---|---|
| 0 | dev box | `rsync -a <kit>/ dgx01@192.168.1.116:~/zig-m2c2-kit/` and the same path on dgx-02 (`root@…:/home/root/zig-m2c2-kit/` via dgx-01) | 2 min |
| 1 | both | `cd ~/zig-m2c2-kit && sha256sum -c SHA256SUMS` | — |
| 2 | dgx-01 | stop prod: `cd ~/ai/src/dsv41-tensorfold-spark && CONFIG=config/prod.env scripts/serve.sh stop` (stops both ranks; the watchdog stands down after a deliberate stop in this boot). Check `docker ps` shows no `dsv41` container on either node | 1 min |
| 3 | both, within ~1 min | dgx-01: `./run_host.sh 0 out`; dgx-02: `./run_host.sh 1 out`. Optional `BACKEND=nccl` (NCCL only), `STEPS=512` (default), `SEEDS="4101 4102"` (one more prompt, about +25 min), `LANES=1` (also M3's first gate, below; about +12 min) | 35–50 min |
| 4 | dgx-01 | restore prod: `cd ~/ai/src/dsv41-tensorfold-spark && scripts/v2-rollback.sh perf1`. It restarts both ranks on `config/prod-perf1.env`, checks 17 × 23 → 391, and leaves the timers as they are | ~65 s |
| 5 | dgx-01 | `scripts/serve.sh status`: both ranks up, model `DeepSeek-V4.1-Flash-TF` | — |
| 6 | dev box | collect `~/zig-m2c2-kit/out/` from both nodes into `dsv41-exl3-quant/results/M2C2-<date>/r0` and `r1` | 2 min |

**Pass:** both nodes' `out/run.log` ends with `PASS M2b rank N: tokens S/S, logits S/S, first window blocks 40/40`. Each seed's `zig.log` also has load and decode times.

**Optional M3 gate (`LANES=1`):** the same greedy decode once more, this time through the lanes engine (`draft/lanes.zig` over the GPU target, `target.zig`) on rank 0, with drafts off. Rank 1 follows rank 0's forward operations over the plan link (`forward.follow`). It passes when rank 0's `out/seed*/lanes.log` ends with `PASS M3 lanes rank 0: tokens S+1/S+1 equal`. It compares tokens only; the digests are the M2b phase's.

**Phases inside `run_host.sh`** (each is a container on the prod image; caches are dropped between them):
1. **Reference:** `dsv41_m2b_ref.py --rank R --master 192.168.100.10`. It loads layers 0–39 and the head from the pack, prefills a 1,536-token prompt, dumps this rank's slot state under our role names, then decodes STEPS greedy tokens. It also writes the decode windows' Triton variants and this host's rope tables. NCCL rendezvous on port 29811.
2. **Engram host tables:** the tokenizer's map and the hash multipliers.
3. **Ours:** `tf-dsv41-m1 m2b /model <ref> <zig>` with `TF_COMM_BACKEND=roce` and the mailbox / RoCE kernels (`tp_mailbox.fatbin`).
   - `TF_TP_ROCE_FALLBACK=error`, so a fallback to NCCL fails the run rather than hiding.
   - Bootstrap on port 29821 (CX7 ports 29710–29840 were the TP gate's, and prod is down).

**On failure:**
- **Exit 2 at the start:** the preflight found a running `dsv41` container, a kit checksum mismatch, or a missing tree or pack. Fix it and re-run step 3.
- **Reference rc ≠ 0:** see `out/seed*/ref.log`. The ranks wait up to 10 min for each other's NCCL rendezvous.
- **Ours FAIL:** `out/seed*/zig/m2b-rankR.jsonl` names the first differing token step and, for the first window, the first differing block.
- **Either way, restore prod (step 4) before anything else.** Last resort, as for every window: `~/ai/restore-prod.sh`.

**Memory (unified, about 119 GiB a node):**
- The Python reference holds this rank's weights, about 100 GB, and exits before ours starts.
- Ours holds the same weights plus about 3 GB of run-time forms (strips and lanes) and a 4,096-position slot.
- `Pack.read` lets go of the page cache of what it read (`POSIX_FADV_DONTNEED`), and the script drops caches between phases.
- If ours fails with an out-of-memory allocation, send `out/` back. The next lever is building the strips in place of the stored trellises: Python relayouts in place, and we keep both.
