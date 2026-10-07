# M4: the Zig server end to end on the two DGX Sparks

**What it proves.** The served path works on the real model at TP=2: the native server (`tensorfold-dsv41`, DeepSeek's text side from `zig/src/server/deepseek.zig`) runs on rank 0 over the CUDA engine (`serve_engine.zig`: lanes over the GPU target and DSpark's GPU pass), and rank 1 follows. The gate checks that every HTTP reply's token ids equal the same prompt replayed through the same engine from the CLI (`tf-dsv41-m1 generate`): "HTTP tokens == CLI". It is the served path's own consistency gate; token identity with Python is M2c-2's.

**Prerequisites:**
- The M2c-2 kit's binaries, rebuilt with `tensorfold-dsv41` (`build-kit.sh` installs every binary of the tree) and the current sm_121 fatbins (26 dsv41 modules).
- `ASSETS`: an M2c-2 seed's `out/seedN/ref/`, which holds `rope.bin`, `engram-host.bin` and `aot/`.
  - With drafts: also merge a set P capture's AOT set made on the Spark: `python dsv41_m1_capture.py --sets P`, then `tools/zig/dsv41_aot_merge.py ASSETS/aot CAPTURE/aot`.
  - Without it: run `DRAFTS=0`.

**Duration:** about 30 min with prod down: two loads of about 8 min each, plus the requests and the replay.

| step | where | what |
|---|---|---|
| 1 | dgx-01 | stop prod: `cd ~/ai/src/dsv41-tensorfold-spark && CONFIG=config/prod.env scripts/serve.sh stop` |
| 2 | both | dgx-01 `ASSETS=... ./run_host.sh 0 out server`; dgx-02 `ASSETS=... ./run_host.sh 1 out follow` |
| 3 | dgx-01 | when `server.log` says it is listening: a few chat requests at temperature 0, e.g. `curl -s localhost:8091/v1/chat/completions -d '{"model":"x","messages":[{"role":"user","content":"Name three primes."}],"temperature":0,"max_tokens":200}'`. Each finished reply lands in `out/trace.jsonl` |
| 4 | both | stop the server (Ctrl-C). Rank 1 ends when the server closes cleanly; if it is still waiting, stop it too (Ctrl-C, or `docker stop dsv41-zig-m4-r1`) |
| 5 | both | dgx-01 `./run_host.sh 0 out generate`; dgx-02 `./run_host.sh 1 out follow` |
| 6 | dgx-01 | restore prod: `scripts/v2-rollback.sh perf1` (~65 s), then `scripts/serve.sh status` |

**Pass:** `out/generate.log` ends with `PASS M4 CLI == HTTP: N/N replies equal`.

**Known limits:**
- One slot: the server takes one request at a time.
- Prefill runs as 16-row decode windows, because Zig's own prefill has its emitter and glue kernels but no forward yet.
- Graphs are off unless `TF_DSV41_GRAPHS=1` on both ranks.
