# The Zig engine on the two DGX Sparks: M2c-2, then M4, then DSpark's acceptance

One prod window, three runs in order. Each run gates the next:

1. **M2c-2: the Zig forward is token-identical to the Python engine on the whole model.**
   - The setup: TP=2 over RoCE (small exchanges) and NCCL (large), prod's knobs including R1, expert top-p 0.85, every layer and the head.
   - The checks: every greedy token, each step's logits on each rank (bit for bit), and every block of the first window.
2. **M4: the Zig server end to end.**
   - `tensorfold-dsv41` serves on rank 0 (lanes over the GPU target and DSpark's GPU pass), and rank 1 follows.
   - The prompts in `prompts.txt` go through HTTP. The same prompts replayed through the same engine from the CLI must give the same token ids ("HTTP tokens == CLI").
3. **DSpark's acceptance on the real model.** The same replay reports drafted / accepted. On a PRO 6000 pod, the 0–24-layer prefix gives the drafter zero taps, so only the Sparks can measure this.

**Downtime: about 1 h 45 min, up to 2 h 15 min**, with prod down from step 2 to step 12:

| part | time |
|---|---|
| M2c-2, one seed | 45–60 min |
| DSpark's AOT set | 5 min |
| M4 server: load, then requests | 15–20 min |
| M4 CLI replay: load, then replay | 15–20 min |
| restore | 2 min |

**Rollback, at any point:** `cd ~/ai/src/dsv41-tensorfold-spark && scripts/v2-rollback.sh perf1` on dgx-01 (about 65 s; it restarts both ranks on `config/prod-perf1.env` and checks 17 × 23 → 391). Last resort: `~/ai/restore-prod.sh`.

**The kit:**
- `tf-dsv41-m1` and `tensorfold-dsv41`: aarch64, with dsq-zig-kern-1's sm_121 fatbins from 81f0477 embedded (26 dsv41 modules; 441 / 441 twin kernels SASS-equal to 8474f31);
- `tp_mailbox.fatbin`;
- `run_host.sh` (M2c-2), `run_m4.sh`, `requests.sh`, `prompts.txt`;
- `tools/`;
- `RUNBOOK-M4.md` (M4's detail);
- `SHA256SUMS`.

| step | where | what | time |
|---|---|---|---|
| 0 | dev box | `rsync -a <kit>/ dgx01@192.168.1.116:~/zig-m2c2-kit/`, and the same to dgx-02 (`/home/root/zig-m2c2-kit/` via dgx-01) | 2 min |
| 1 | both | `cd ~/zig-m2c2-kit && sha256sum -c SHA256SUMS` | — |
| 2 | dgx-01 | **stop prod:** `cd ~/ai/src/dsv41-tensorfold-spark && CONFIG=config/prod.env scripts/serve.sh stop`. Check that `docker ps` shows no `dsv41` container on either node | 1 min |
| 3 | both, within ~1 min | **M2c-2:** dgx-01 `./run_host.sh 0 out`, dgx-02 `./run_host.sh 1 out`. R1 is on by default (`R1=0`: prod.env's knobs alone) | 45–60 min |
| 4 | both | **Check:** `out/run.log` ends with `PASS M2b rank N: tokens 512/512, logits 512/512, first window blocks 40/40`. **If either rank fails, go to step 12** and send `out/` back | — |
| 5 | both | **M4's assets:** `mkdir -p out/m4-assets && cp -r out/seed4101/ref/{rope.bin,engram-host.bin,aot} out/m4-assets/` | — |
| 6 | both | **DSpark's Triton set:** `ASSETS=$PWD/out/m4-assets ./run_m4.sh R out dspark-aot` (R = 0 on dgx-01, 1 on dgx-02). It ends with `{"added": N, ...}` | 5 min |
| 7 | both, within ~1 min | **M4 server:** dgx-01 `ASSETS=$PWD/out/m4-assets ./run_m4.sh 0 out server`, dgx-02 `ASSETS=$PWD/out/m4-assets ./run_m4.sh 1 out follow` | 8 min load |
| 8 | dgx-01, second shell | when `out/server.log` says it is listening: `./requests.sh 8091 256`. One JSON line per reply, and `out/trace.jsonl` fills | 5–10 min |
| 9 | both | stop the server (Ctrl-C). Rank 1 ends on its own; if it is still waiting, `docker stop dsv41-zig-m4-r1` | — |
| 10 | both, within ~1 min | **CLI replay:** dgx-01 `ASSETS=$PWD/out/m4-assets ./run_m4.sh 0 out generate`, dgx-02 `ASSETS=$PWD/out/m4-assets ./run_m4.sh 1 out follow` | 15–20 min |
| 11 | dgx-01 | **Check** `out/generate.log`: `PASS M4 CLI == HTTP: 8/8 replies equal`, then `DSpark: D drafted, A accepted (rate)` (the acceptance measurement) | — |
| 12 | dgx-01 | **restore prod:** `scripts/v2-rollback.sh perf1`, then `scripts/serve.sh status` (both ranks up, model `DeepSeek-V4.1-Flash-TF`) | 2 min |
| 13 | dev box | collect `~/zig-m2c2-kit/out/` from both nodes into `dsv41-exl3-quant/results/SPARK-ZIG-<date>/r0` and `r1` | 2 min |

**Optional knobs:**
- M2c-2: `SEEDS="4101 4102"` adds about 25 min. `BACKEND=nccl` takes NCCL only.
- **`PREFILL=1` on step 3** (about **+10 min**): Zig's own prefill gate. After the decode gate, our ranks load again from an empty slot and prefill the reference's prompt in 2,048-row segments, with the exchanges over NCCL (20 MiB each, above the RoCE mailbox's 16 MB ceiling).
  - The checks: every state role (SWA rings, compressed pools, index keys, ratio-2 carries) bytes-equal to Python's dump, the same position and Engram tail, the same first token, then the decode gate from that state.
  - The pass line: `PASS Zig prefill rank N: ... state roles K/K equal ...`, then `PASS M2b rank N` in `out/seed*/prefill.log`.
  - `PROMPT=2600` makes the prompt cross a segment boundary (2,048 + 552).
  - **Passed on 2x PRO 6000** (layers 0–24, TP=2, prod's knobs without R1). From an empty slot, each prompt gave 65/65 state roles equal to Python's dump, the same position, Engram tail and first token, then 64/64 tokens and logits:
    - 2,600 rows (2,048 + 552, pod 23);
    - 2,060, 2,049 and 2,064 rows (2,048 + a 12-, 1- and 16-row segment) and a 9-token prompt (pod 25).
    - So the Zig prefill is Python's bits for every prompt length: segments of 16 rows or fewer take Python's short-segment kernels.
  - **R1:** those gates ran without R1. This step on the Sparks (R1 on by default) is the first prefill gate under R1. **If it fails while M2c-2 passed, run M4 (steps 7 and 10) with `OWN_PREFILL=0`** and send `out/` back.
  - **M4 (steps 7 and 10) prefills with the Zig prefill** (the default; `OWN_PREFILL=0` goes back to 16-row decode windows, the same model but not Python's prefill bits).
- M4: `DRAFTS=0` serves without DSpark (then step 6 is not needed). `TF_DSV41_GRAPHS=1` on both ranks enables decode graphs.

**What the numbers mean:**
- M2c-2 and M4 are pass / fail.
- The acceptance is the number to compare with prod's own draft statistics on similar prompts (the Python engine's `/metrics` or its draft log). The Zig engine serves one slot, and prod's replies come from other prompts and requests: compare acceptance, not tokens, against prod.

**On failure:**
- **Exit 2 at the start:** the preflight found a running `dsv41` container, a checksum mismatch, or a missing tree or pack.
- **M2c-2 reference rc ≠ 0:** see `out/seed*/ref.log`.
- **Ours FAIL:** `out/seed*/zig/m2b-rankR.jsonl` names the first differing step and block.
- **M4 load failure:** `out/server.log` / `out/follow.log`. An out-of-memory load means the runtime forms did not fit beside the weights.
- **Whatever happened, restore prod (step 12) first.**

**Memory (unified, about 119 GiB a node):**
- Python's reference exits before ours starts.
- Ours holds this rank's weights (about 100 GB), its run-time forms and one 4,096-position slot.
- `Pack.read` drops the page cache of what it read, and the scripts drop caches between phases.
