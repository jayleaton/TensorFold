# Serving and ops on the Sparks: the window steps (dsq-zig-ops-1)

Steps for the lead's Spark window kit. None of them was run. They need the whole model on both Sparks, so no pod
covers them. Prod's repo files are on the prod mirror's branch `zig-serve` (a worktree at
`~/Documents/Projects/spark-work/dsv41-tensorfold-spark-zig-serve`, no push). Copy them to dgx-01's
`~/ai/src/dsv41-tensorfold-spark` at the window's start, or check out the branch there.

Prerequisites from the rest of the window:
- the M4 assets on each node (`rope.bin`, `engram-host.bin`, `aot/` with DSpark's and the row-mode variants): the kit's
  `m4-assets`, copied to `HEAD_ASSETS` / `WORKER_ASSETS` of `config/prod-zig.env`;
- the sm_121 fatbins the pod gates passed (`zig-dsv41-scratch/zig-out-sm121/fatbin` + `tp_mailbox.fatbin`).

| # | Step | Command | Compares | Minutes |
|---|---|---|---|---|
| 1 | Stage the build context | dev box: `CONFIG=config/prod-zig.env ZIG_COMMIT=<hash> ZIG_FATBINS=<dir> scripts/zig-image.sh stage` | — | 2 |
| 2 | Build the image natively | dgx-01: `CONFIG=config/prod-zig.env ZIG_COMMIT=<hash> scripts/zig-image.sh build`, then `check` | both nodes' content keys equal; `docker run --rm --init IMAGE tensorfold-dsv41 --help` | 10–15 |
| 3 | Preflight (prod still up) | `CONFIG=config/prod-zig.env scripts/serve.sh preflight static` | assets, image label == ZIG_COMMIT, RoCE ports | 1 |
| 4 | Switch prod to Zig, first boot | `scripts/v2-rollback.sh zig` (stops Python prod, starts Zig, `17*23 -> 391`) | rank 0 log: `serving: 4 slot(s) x 1048576 tokens; wire spark`, `dsv41 load: weights ... prepared ...: 0 read, N built, N written`, the canary minus `json` | 15–20 |
| 5 | Fast boot | `scripts/serve.sh restart` | `dsv41 load: ... N read, 0 built`; record `ready after N s` (Python perf1: 33–48 s) | 3–10 |
| 6 | The wire against Python prod | `python3 tools/zig/dsv41_ops/wire_probe.py --base http://127.0.0.1:8000` on Zig, then after step 10 on Python, `diff` the two outputs | /v1/models, /health keys, /metrics names, error shapes, SSE framing, usage, `tensorfold` keys, tokenize; replies are model output, so only shapes are compared | 5 |
| 7 | Request log | `tail -3 $HEAD_STATE/requests.jsonl` after step 6 | Python's 37 keys in Python's order (`wire_test.zig` already checks every value on host) | 1 |
| 8 | Disconnect | `curl -sN -m 2 .../v1/chat/completions -d '{...,"stream":true,"max_tokens":4000}'`, then `/health` | `inflight` back to 0 within a round; the request-log line has `"finish":"cancelled"` | 2 |
| 9 | Fail-fast and the watchdog | `ssh worker docker kill -s KILL dsv41-tf-r1`; watch `scripts/serve.sh watch --once` each minute | rank 0 exits 70 about 1 s later; the tick says `(fail-fast exit 70)` and heals at once (WATCH_FF_MIN_HEAL); back to serving | 10–20 |
| 10 | Back to Python | `scripts/v2-rollback.sh perf1` | `17*23 -> 391` on 8474f31 | 3–5 |

Total: about 55–75 min, most of it the first boot and one heal.

Step 6's probe (`wire_probe.py`, stdlib only) prints each answer's shape: status, Content-Type, keys and value types,
SSE event runs and metric names. Byte-level parity with Python's server is already proven on host by `wire_test.zig`,
so the window run is a smoke check with the real model and the real engine's stats.

Knobs to watch in the logs:
- `TF_DSV41_SESSIONS` / `SESSION_DISK_GIB` / `SESSION_DISK_MIN`: the tier opens with budget 128 GiB and min 1024
  tokens (sessions_gpu.zig). With `TF_DSV41_SLOTS=4`, sessions are not opened yet (slots refuse them by name).
- `TF_DSV41_FLOOR_GIB`: `kv: memory: floor X GiB is under the 5 GiB target` (warning only, as Python's load check).
