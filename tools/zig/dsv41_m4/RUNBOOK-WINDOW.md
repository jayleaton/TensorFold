# The Zig engine on the two DGX Sparks, the full window: everything on, speed against prod, long context

One window, the whole model at TP=2 over RoCE, with every finished part on:
- R1, decode graphs, the Zig prefill;
- R3 Engram (the node's packed shards, as prod reads them);
- keyed sampling;
- DSpark drafts priced by prod's stored calibration table;
- several live slots (row mode), with DSpark drafting per slot and a batched verify;
- the paged pool, and split KV for long context.

It answers three questions:
1. **Speed against prod.** Decode tok/s on code and prose, T0 and T0.7, 1 and 4 streams. Prod is measured first, while it still serves, with the same client (`bench_http.py`).
2. **The served path holds.** Every HTTP reply of run A (greedy and sampled, drafts on) replays through the CLI to the same ids: `PASS M4 CLI == HTTP`.
3. **Long context.** Split KV's measured boot check on the whole model. Prompt and decode tok/s at 32K / 128K / 512K tokens against prod.

**Downtime: about 2 h 30 min, up to 3 h 10 min (part 1)**, with prod down from step 3 to step 12. Steps 1–2 run before the window, while prod serves.

| part | time |
|---|---|
| prod baseline (prod up) | 20–25 min |
| row-mode AOT set | 8 min |
| long-context assets (`_stream` variants, 1M rope tables) | 10 min |
| run A: 1 slot, all on (load, bench, requests) | 25 min |
| run A's CLI replay | 15 min |
| run B: 4 slots, drafts off (load, bench) | 15 min |
| run B2: 4 slots with drafts (load, bench, replay) | 30 min |
| run C: long context, split KV (load, bench) | 25–30 min |
| restore | 2 min |

**Rollback, at any point:** `cd ~/ai/src/dsv41-tensorfold-spark && scripts/v2-rollback.sh perf1` on dgx-01 (about 65 s). Last resort: `~/ai/restore-prod.sh`.

**The kit** (copy to `~/zig-window-kit/` on both nodes; it does not replace `~/zig-m2c2-kit/`, whose `out/m4-assets` it reuses):
- `tf-dsv41-m1` and `tensorfold-dsv41` (aarch64, sm_121 fatbins embedded, including `dsv41_kvsplit` and `dsv41_sampling`);
- `tp_mailbox.fatbin`;
- `run_m4.sh` (the server, follower, CLI replay, `dspark-aot` and `rows-aot` phases), `requests.sh`, `prompts.txt`;
- `bench_http.py`;
- `tools/` (including `tools/zig/dsv41_slots/ref.py`);
- `RUNBOOK-WINDOW.md` (this file), `RUNBOOK-M4.md`, `SHA256SUMS`.

`R` below is 0 on dgx-01 and 1 on dgx-02. "Both, within ~1 min" means start the two commands within a minute of each other. `A=$HOME/zig-window-kit/out/assets`.

| step | where | what | time |
|---|---|---|---|
| 00 | dev box | **the lease:** `ssh dgx01@192.168.1.116 'test -f ~/ai/.test-window-lease && echo held'` must print nothing (held: Spark-tool-kit owns the Sparks; touch neither, wait). `run_m4.sh` / `run_host.sh` refuse on rank 0 while it is held | — |
| 0 | dev box | `rsync -a <kit>/ dgx01@192.168.1.116:~/zig-window-kit/`, and the same to dgx-02 (`/home/root/zig-window-kit/`, via dgx-01). Then on both: `cd ~/zig-window-kit && sha256sum -c SHA256SUMS` | 2 min |
| 1 | dgx-01, prod up | **prod speed:** `python3 bench_http.py --url http://127.0.0.1:8000 --label prod --out out/prod.json` (8 cells x 2 reps, 384 tokens, thinking off) | 10 min |
| 2 | dgx-01, prod up | **prod long context:** `python3 bench_http.py --url http://127.0.0.1:8000 --label prod --long 32768,131072,524288 --out out/prod-long.json` | 10–15 min |
| 3 | dgx-01 | **stop prod:** `cd ~/ai/src/dsv41-tensorfold-spark && CONFIG=config/prod.env scripts/serve.sh stop`. `docker ps` must show no `dsv41` container on either node | 1 min |
| 4 | both | **assets:** `mkdir -p out && cp -r ~/zig-m2c2-kit/out/m4-assets out/assets` (the last window's set: M2c-2's prefill variants plus DSpark's). If it is missing, run `ASSETS=$A ./run_m4.sh R out dspark-aot` after copying a seed's `ref/{rope.bin,engram-host.bin,aot}` | 1 min |
| 5 | both, within ~1 min | **row-mode AOT set:** `ASSETS=$A ./run_m4.sh R out rows-aot`. Each ends with `{"added": N, ...}` | 8 min |
| 5b | both, within ~1 min | **long-context assets:** `ASSETS=$A ./run_m4.sh R out long-aot`: Python prefills one 8,448-token prompt (the stream top-k's `_stream` variants, used past 4,096 visible keys) and `rope.bin` is rebuilt for 1,050,624 positions. Each ends with `{"added": N, ...}` and `rope.bin: 1050624 rows`. Without it, run C's long prompts fail with `MissingTritonVariant` or `RopeTooShort` | 10 min |
| 6 | both, within ~1 min | **run A, 1 slot, all on:** dgx-01 `ASSETS=$A PHASES=1 ./run_m4.sh 0 out/a server`, dgx-02 `ASSETS=$A ./run_m4.sh 1 out/a follow` | 8 min load |
| 7 | dgx-01, second shell | when `out/a/server.log` says it is listening: `python3 bench_http.py --url http://127.0.0.1:8091 --label zig-1slot --out out/zig-a.json`, then `./requests.sh 8091 256`. Then stop the server (Ctrl-C); rank 1 ends on its own (else `docker stop dsv41-zig-m4-r1`) | 15 min |
| 8 | both, within ~1 min | **run A's CLI replay:** dgx-01 `ASSETS=$A ./run_m4.sh 0 out/a generate`, dgx-02 `ASSETS=$A ./run_m4.sh 1 out/a follow`. **Check** `out/a/generate.log`: `PASS M4 CLI == HTTP: N/N replies equal` (the bench's sampled replies included), then `DSpark: D drafted, A accepted (rate)` | 15 min |
| 9 | both, within ~1 min | **run B, 4 slots:** dgx-01 `ASSETS=$A SLOTS=4 POOL_TOKENS=1400000 INDEX_KV=fp8 ./run_m4.sh 0 out/b server`, dgx-02 the same with `1 out/b follow`. Then dgx-01: `python3 bench_http.py --url http://127.0.0.1:8091 --label zig-4slots --out out/zig-b.json`. Stop both | 15 min |
| 9b | both, within ~1 min | **run B2, 4 slots with drafts (prod's shape):** dgx-01 `ASSETS=$A SLOTS=4 SLOT_DRAFTS=1 POOL_TOKENS=1400000 INDEX_KV=fp8 PHASES=1 ./run_m4.sh 0 out/b2 server`, dgx-02 the same with `1 out/b2 follow`. Then dgx-01: `python3 bench_http.py --url http://127.0.0.1:8091 --label zig-4slots-drafts --out out/zig-b2.json`, then `./requests.sh 8091 256`. Stop both. Then the replay at 4 streams: dgx-01 `ASSETS=$A SLOTS=4 SLOT_DRAFTS=1 POOL_TOKENS=1400000 INDEX_KV=fp8 GEN_STREAMS=4 ./run_m4.sh 0 out/b2 generate`, dgx-02 the same with `1 out/b2 follow`. **Check** `out/b2/generate.log`: `PASS M4 CLI == HTTP` and the `DSpark:` line | 30 min |
| 10 | both, within ~1 min | **run C, long context:** dgx-01 `ASSETS=$A CONTEXT=1048576 POOL_TOKENS=3000000 KV_SPLIT=1 INDEX_KV=fp8 ./run_m4.sh 0 out/c server`, dgx-02 the same with `1 out/c follow`. Note the boot check's lines in `out/c/server.log` (`grep -i "boot check\|pool" out/c/server.log`). Then dgx-01: `python3 bench_http.py --url http://127.0.0.1:8091 --label zig-split --long 32768,131072,524288 --out out/zig-long.json`. Stop both | 25–30 min |
| 11 | dgx-01 | the tables: `python3 bench_http.py --compare out/prod.json out/zig-a.json`, the same for `out/zig-b.json` and `out/zig-b2.json`, and `--compare out/prod-long.json out/zig-long.json` | — |
| 12 | dgx-01 | **restore prod:** `scripts/v2-rollback.sh perf1`, then `scripts/serve.sh status` (both ranks up, model `DeepSeek-V4.1-Flash-TF`) | 2 min |
| 13 | dev box | collect `~/zig-window-kit/out/` from both nodes into `dsv41-exl3-quant/results/SPARK-ZIG-<date>/r0` and `r1` | 2 min |

**Defaults of `run_m4.sh`** (both ranks must match): R1, graphs, Zig prefill, drafts with prod's speculative pass (`SPEC=1`), keyed sampling (`SAMPLING=1`), R3 Engram from the node's shards (`ENGRAM=1`: `/home/dgx01/dsv41-engram`, `/home/root/dsv41-engram`), drafts priced by prod's stored table (`CALIB=real`: the `dsv41-perf1-cache` volume read-only at `/cache`, `TF_DSV41_CALIB_DIR=/cache/dsv41-calib`), expert top-p 0.85, RoCE. Each can be turned off: `ENGRAM=0`, `CALIB=`, `SAMPLING=0`, `SPEC=0`, `DRAFTS=0`, `GRAPHS=0`, `OWN_PREFILL=0`.

**What each run is:**
- **A, the prod candidate's one slot:** drafts on, contiguous slot (4,096 positions). Its 4-stream cells queue on one slot: they measure the one-slot server under load, not batching.
- **B, batching:** 4 live slots in row mode (graphs per row bucket), keyed sampling per slot, the paged pool at prod's 1.4M tokens. Drafts are off: DSpark is one slot's for now. Compare B's 4-stream cells with prod's 4-stream cells (prod drafts with 4 slots), and A's 1-stream cells with prod's 1-stream cells.
- **B2, prod's shape:** 4 live slots with DSpark drafting in each and one batched verify a round, as prod's batcher runs `PARALLEL=4`. **This is the like-for-like run against prod: compare every cell of `out/zig-b2.json` with `out/prod.json`.** Its replay at 4 streams checks that batched, drafted replies are the same ids as the HTTP ones, and its `DSpark:` line gives the multi-slot acceptance to set against prod's.
- **C, long context:** split KV (the comp rows sharded over the two ranks) on a 3.0M-token pool, drafts on, prod's 1M served context (`CONTEXT=1048576`: prompts past 4,096 keys go through the stream top-k and split KV's row blocks). The measured boot check runs first. If it refuses the pool, it names the bytes it needs: lower `POOL_TOKENS` (2400000, then 1400000) on both ranks and restart.

**Optional knobs:**
- `TREE=3` on run A: DSpark draft trees as prod's `TF_DSV41_TREE=3` runs them. Compare drafted / accepted / rounds with prod's on the same prompts; tokens must stay equal (the CLI replay).
- `--long 1048576` on run C (and prod) adds a 1M-token prompt (about +10 min each side).
- `BACKEND=nccl`: NCCL only.

**What the numbers mean:**
- The step 11 tables end with `T0 replies equal: N/M`: Zig's greedy replies against prod's, text for text (the same prompts, knobs and model, so they must all be equal). A difference names a bug to report, not a speed result.
- The step 11 tables give Zig's tok/s against prod's per cell (best of 2 reps; 4 streams: the aggregate while all four are live).
- Long context gives prompt tok/s (prompt tokens over the time to the first token) and the 128-token answer's decode tok/s.
- `out/a/phases.json` holds Zig's per-round phase times: the host's share of a round, against Python's own phase report.
- Run A's generate line is pass / fail. The acceptance rate compares with prod's draft statistics.

**Part 2 (optional, its own window if part 1 ran long): Zig as prod, the serving and ops checks.** `tools/zig/dsv41_ops/WINDOW-STEPS.md`, about 55–75 min:
- build the Zig image natively on dgx-01 (prod mirror branch `zig-serve`: `scripts/zig-image.sh`, `config/prod-zig.env`);
- `scripts/v2-rollback.sh zig` (Zig serves as prod; the first boot writes the prepared files), then a restart for the fast-boot time against Python's 33–48 s;
- `wire_probe.py` against Zig and then Python, the request log, a client disconnect, a fail-fast kill of rank 1 with the watchdog's heal;
- `scripts/v2-rollback.sh perf1`.
Its assets are part 1's: copy `out/assets` to `HEAD_ASSETS` / `WORKER_ASSETS` of `config/prod-zig.env`. Prod's `PARALLEL=4` with NVMe sessions is not Zig's yet (sessions are refused with 4 slots), so part 2 runs Zig with the sessions tier closed at 4 slots.

**On failure:**
- **A run fails to load** (`out/*/server.log`, `follow.log`):
  - `EngramNotOpen`: the shards path is wrong. Run with `ENGRAM_DIR=<path>`, or `ENGRAM=0`.
  - A calibration error: use `CALIB=`.
  - A missing Triton variant with `SLOTS=4`: step 5's set lacks it. Send `out/rows-aot.log`. Run B is then skipped.
  - Out of memory with `POOL_TOKENS`: lower it.
- **Run C fails with drafts** (an error naming the pool or split): rerun C with `DRAFTS=0`. The long-context numbers still hold, without drafting.
- **Run B2 refuses to load with `SLOT_DRAFTS=1`**, or its replay fails: run B's numbers stand without drafting. Send `out/b2/`.
- **The CLI replay fails:** the line's `first_differing` names the token index. Send `out/a/`.
- **Whatever happened, restore prod (step 12) first.**

**Memory (unified, about 119 GiB a node):**
- Each run holds this rank's weights (about 100 GB), the runtime forms and the pool.
- The boot check measures free memory before it allocates the pool, and refuses below prod's floor.
- `run_m4.sh` does not drop caches. Before each load, run `sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` on both nodes when `free -g` shows less than 104 GiB available.
