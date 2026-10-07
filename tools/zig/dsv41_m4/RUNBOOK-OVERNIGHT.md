# The Zig engine on the two DGX Sparks, overnight: everything gated, in order

One overnight window, the whole model at TP=2 over RoCE. It runs everything gated on the pods so far, in this order:
1. **Correctness:** M2c-2 and the prefill gate on the whole model, then the served path (HTTP == CLI).
2. **Speed against prod:** 1 slot, 4 slots, and 4 slots with drafts (prod's shape).
3. **Long context:** 1M served context with split KV.
4. **Zig as prod (optional):** the image, the fast boot, the wire, the fail-fast heal.

**Restore on failure, at every step:** `./restore.sh 1` on dgx-02, then `./restore.sh 0` on dgx-01. It stops every container this kit started and runs `scripts/v2-rollback.sh perf1`, about 65 s. Then `scripts/serve.sh status` must show both ranks up with model `DeepSeek-V4.1-Flash-TF`.
- A failure in part 1 (correctness) ends the night: restore, then collect `out/` (step 16).
- A failure in parts 2–3 skips that run only. Stop its containers (`./restore.sh` without the prod restart: `docker stop $(docker ps -q --filter name=dsv41-zig)` on both hosts) and go on with the next run.
- Part 4 replaces prod with Zig. Its own last step goes back to Python.

**Downtime** (prod down from step 4 to step 15):

| part | time | prod |
|---|---|---|
| 0. lease, kit, prod's own numbers | 30 min | up |
| 1. correctness: M2c-2 + prefill gate, assets, run A + its replay | 2 h–2 h 20 min | down |
| 2. speed: runs B and B2 | 45 min | down |
| 3. long context: run C | 30 min | down |
| restore | 2 min | back up |
| **total, parts 1–3** | **about 3 h 20 min (up to 4 h 15 min)** | |
| 4. Zig as prod (optional) | 55–75 min | Zig serves, then Python again |

**The kit** (`~/zig-window-kit/` on both nodes):
- `tf-dsv41-m1` and `tensorfold-dsv41`: aarch64, with every sm_121 fatbin embedded (the 441 / 441 SASS-equal base set, `dsv41_kvsplit`, `dsv41_sampling`, `dsv41_vision`);
- `tp_mailbox.fatbin`;
- `run_host.sh` (M2c-2), `run_m4.sh` (server, follower, CLI replay, `dspark-aot` / `rows-aot` / `long-aot`);
- `requests.sh`, `prompts.txt`, `bench_http.py`, `restore.sh`;
- `tools/` (including `tools/zig/dsv41_ops/WINDOW-STEPS.md` and `wire_probe.py` for part 4);
- `RUNBOOK.md` (this file), `COMMIT`, `SHA256SUMS`.

`R` = 0 on dgx-01, 1 on dgx-02. "Both, within ~1 min": start the two commands within a minute of each other. `A=$HOME/zig-window-kit/out/assets`.

| step | where | what | time |
|---|---|---|---|
| **0. Before the window (prod up)** | | | |
| 1 | dev box | **the lease:** `ssh dgx01@192.168.1.116 'test -f ~/ai/.test-window-lease && echo held'` must print nothing. If it prints `held`, Spark-tool-kit owns the Sparks: touch neither, wait. Both kit scripts refuse on rank 0 while it is held | — |
| 2 | dev box | `rsync -a <kit>/ dgx01@192.168.1.116:~/zig-window-kit/`, the same to dgx-02 (`/home/root/zig-window-kit/`, via dgx-01). On both: `cd ~/zig-window-kit && sha256sum -c --quiet SHA256SUMS` | 5 min |
| 3 | dgx-01 | **prod's own numbers:** `python3 bench_http.py --url http://127.0.0.1:8000 --label prod --out out/prod.json`, then `python3 bench_http.py --url http://127.0.0.1:8000 --label prod --long 32768,131072,524288 --out out/prod-long.json` | 25 min |
| **1. Correctness (prod down from here)** | | | |
| 4 | dgx-01 | **stop prod:** `cd ~/ai/src/dsv41-tensorfold-spark && CONFIG=config/prod.env scripts/serve.sh stop`. `docker ps` shows no `dsv41` container on either node | 1 min |
| 5 | both, within ~1 min | **M2c-2 + the prefill gate on the whole model:** `PREFILL=1 ./run_host.sh R out`. **Pass:** `out/run.log` ends with `PASS M2b rank R: tokens 512/512, logits 512/512, first window blocks 40/40` and `out/seed4101/prefill.log` holds `PASS Zig prefill rank R ... state roles K/K equal`, then `PASS M2b rank R`. **Fail: restore, end** | 55–70 min |
| 6 | both | **assets:** `mkdir -p out/assets && cp -r out/seed4101/ref/{rope.bin,engram-host.bin,aot} out/assets/` | — |
| 7 | both, within ~1 min | `ASSETS=$A ./run_m4.sh R out dspark-aot` (DSpark's set P), then `ASSETS=$A ./run_m4.sh R out rows-aot` (row-mode buckets), then `ASSETS=$A ./run_m4.sh R out long-aot` (`_stream` variants, a 1M + 2,048-row `rope.bin`). Each ends with `{"added": N, ...}`; long-aot also with `rope.bin: 1050624 rows`. **Fail: restore, end** | 25 min |
| 8 | both, within ~1 min | **run A, one slot, everything gated on:** dgx-01 `ASSETS=$A PHASES=1 ./run_m4.sh 0 out/a server`, dgx-02 `ASSETS=$A ./run_m4.sh 1 out/a follow` | 8 min load |
| 9 | dgx-01, second shell | when `out/a/server.log` says `serving pack at`: `python3 bench_http.py --url http://127.0.0.1:8091 --label zig-1slot --out out/zig-a.json`, then `./requests.sh 8091 256`. Stop the server (Ctrl-C); rank 1 ends on its own (else `docker stop dsv41-zig-m4-r1`) | 15 min |
| 10 | both, within ~1 min | **run A's CLI replay:** dgx-01 `ASSETS=$A ./run_m4.sh 0 out/a generate`, dgx-02 `ASSETS=$A ./run_m4.sh 1 out/a follow`. **Pass:** `out/a/generate.log` has `PASS M4 CLI == HTTP: N/N replies equal` (greedy and sampled, drafts and the speculative pass on), then `DSpark: D drafted, A accepted (rate)`. **Fail: restore, end** | 15 min |
| **2. Speed against prod** | | | |
| 11 | both, within ~1 min | **run B, 4 slots, drafts off:** dgx-01 `ASSETS=$A SLOTS=4 POOL_TOKENS=1400000 INDEX_KV=fp8 ./run_m4.sh 0 out/b server`, dgx-02 the same with `1 out/b follow`. dgx-01: `python3 bench_http.py --url http://127.0.0.1:8091 --label zig-4slots --out out/zig-b.json`. Stop both | 15 min |
| 12 | both, within ~1 min | **run B2, 4 slots with drafts (prod's shape):** dgx-01 `ASSETS=$A SLOTS=4 SLOT_DRAFTS=1 POOL_TOKENS=1400000 INDEX_KV=fp8 PHASES=1 ./run_m4.sh 0 out/b2 server`, dgx-02 the same with `1 out/b2 follow`. dgx-01: `python3 bench_http.py --url http://127.0.0.1:8091 --label zig-4slots-drafts --out out/zig-b2.json`, then `./requests.sh 8091 256`. Stop both. Then the replay at 4 streams: the same knobs plus `GEN_STREAMS=4` with `./run_m4.sh 0 out/b2 generate` / `./run_m4.sh 1 out/b2 follow`: `PASS M4 CLI == HTTP` | 30 min |
| **3. Long context** | | | |
| 13 | both, within ~1 min | **run C, 1M context, split KV:** dgx-01 `ASSETS=$A CONTEXT=1048576 POOL_TOKENS=3000000 KV_SPLIT=1 INDEX_KV=fp8 ./run_m4.sh 0 out/c server`, dgx-02 the same with `1 out/c follow`. Note the boot check (`grep -i "boot check\|kv:" out/c/server.log`); if it refuses the pool it names the bytes: lower `POOL_TOKENS` (2400000, then 1400000) on both and restart. dgx-01: `python3 bench_http.py --url http://127.0.0.1:8091 --label zig-split --long 32768,131072,524288 --out out/zig-long.json`. Stop both | 30 min |
| 14 | dgx-01 | **the tables:** `python3 bench_http.py --compare out/prod.json out/zig-a.json`, the same with `out/zig-b.json` and `out/zig-b2.json`, and `--compare out/prod-long.json out/zig-long.json` | — |
| 15 | dgx-02, then dgx-01 | **restore prod:** `./restore.sh 1`, then `./restore.sh 0` | 2 min |
| 16 | dev box | collect `~/zig-window-kit/out/` from both nodes into `dsv41-exl3-quant/results/SPARK-ZIG-<date>/r0` and `r1` | 2 min |
| **4. Zig as prod (optional, prod is back on Python before it starts)** | | | |
| 17 | dgx-01 | `tools/zig/dsv41_ops/WINDOW-STEPS.md` steps 1–10: build the Zig image on dgx-01 (prod mirror branch `zig-serve`), `scripts/v2-rollback.sh zig` (first boot writes the prepared files), a restart for the fast-boot time, `wire_probe.py` against Zig then Python, the request log, a disconnect, a fail-fast kill of rank 1 and the watchdog's heal, `scripts/v2-rollback.sh perf1`. Assets: copy `out/assets` to `HEAD_ASSETS` / `WORKER_ASSETS` of `config/prod-zig.env` | 55–75 min |

**What run A turns on** (`run_m4.sh`'s defaults; both ranks alike): R1, decode graphs, the Zig prefill, DSpark drafts with prod's speculative pass (`SPEC=1`), drafts priced by prod's stored calibration (`CALIB=real`: the cache volume read-only), keyed sampling (`SAMPLING=1`), R3 Engram from the node's shards (`ENGRAM=1`), and prod's RoCE / index-budget / 2,048-row prefill knobs. The features gated on the last pod (structured output, images with `bias_vl`, x3gm v2 prefill, the fused dense prefill, Engram read-ahead) are listed below with the knobs that turn them on. Each can be turned off: `ENGRAM=0`, `CALIB=`, `SAMPLING=0`, `SPEC=0`, `DRAFTS=0`, `GRAPHS=0`, `OWN_PREFILL=0`.

**Features behind a knob, off in this kit until their pod gate passes** (both ranks alike; turn one on by adding its knob to every `run_m4.sh` command of a run):

| feature | knob | pod gate (pod 30, 2026-10-07) |
|---|---|---|
| structured output (grammar) | `GRAMMAR=1` | **PASS**: 6/6 replies == Python, drafts on and serial |
| fused dense prefill | `PF_DENSE=fused` | **PASS**: 65/65 roles, 64/64 tokens + logits |
| multi-slot drafting (run B2) | `SLOT_DRAFTS=1` | **PASS** eager (4/4 batched == alone == Python); row graphs' boot panic fixed in `a36c4c2` (this kit has it) |
| images, with prod's image routing bias | `IMAGES=native BIAS_VL=1` | **FAIL**: the tower is not bit-exact (0/3 images). **Leave off** |
| x3gm v2 prefill | `GM_V2=1` | inconclusive (the check missed R1's arguments; Python's v2 == v1 bits). **Leave off** |
| Engram read-ahead | `PREFETCH_AHEAD=1` | not run (needs the shards). Leave off |
| several slots with sessions | none yet | not built (refused with `SLOTS` > 1) |

Long prompts past 4,096 keys (run C's `CONTEXT`) are host-gated, and their pod gate (`M2B_LONGPF`) is also on pod 30. Run C is where they first run on GPUs.

**What the numbers mean:**
- Step 5 and step 10 are pass / fail: Zig's forward == Python's on the whole model, and the served path's replies == the same requests replayed through the CLI.
- Step 14's speed tables are Zig's tok/s against prod's per cell (best of 2 reps; 4 streams: the aggregate while all four are live). B2 is the like-for-like row: prod also serves 4 slots with drafts.
- `T0 replies equal: N/M` compares Zig's greedy replies with prod's text for text. **Equal only for prompts of 128 tokens or fewer:** prod prefills with CED bounded replay (`TF_DSV41_PREFILL=replay`), which Zig has not ported. Zig runs full prefill, which is Python's `PREFILL=full`. The bench's code and prose prompts are short. Differences on longer prompts are expected and are not a Zig bug.
- Long context: prompt tok/s (prompt tokens over the time to the first token) and the 128-token answer's decode tok/s, against prod's. Replies differ for the same reason.
- `out/a/phases.json` and `out/b2/phases.json`: Zig's per-round phase times.

**R1 on the pods (pods 29 / 30):** a fault in `mhc_cuda.run` at 4-row R1 windows turned out to be the M2b test driver's 1-row buffer plan (fixed `a585336`); the served model plans every window size and is not affected. Last resort only: if a run with R1 dies with `CudaFailed` / an illegal address, restore that run's containers, rerun it with `R1=0` on both ranks, and record it.

**On failure (beyond restore):**
- **Exit 3 at a start:** the lease is held. Wait; touch nothing.
- **Exit 2 at a start:** a running `dsv41` container, a checksum mismatch, or a missing tree or pack.
- **Load failure** (`out/*/server.log`, `follow.log`): `EngramNotOpen` (wrong shards path: `ENGRAM_DIR=<path>` or `ENGRAM=0`), a calibration error (`CALIB=`), `MissingTritonVariant` (an asset step did not merge: send its log), out of memory with `POOL_TOKENS` (lower it).
- **A replay differs:** the line's `first_differing` names the token. Send `out/a/` (or `out/b2/`).

**Memory (unified, about 119 GiB a node):** each run holds this rank's weights (about 100 GB), the runtime forms and the pool. The boot check measures free memory before it allocates the pool, and refuses below prod's floor. Before each load, run `sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` on both nodes when `free -g` shows less than 104 GiB available.
