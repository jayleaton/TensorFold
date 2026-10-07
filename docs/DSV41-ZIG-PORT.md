# DeepSeek-V4.1-Flash on the Zig engine: port plan

This is the plan for moving our DeepSeek-V4.1-Flash serving stack from TensorFold's Python engine to the Zig engine (branch `zig-flashnext`, `ZIG-PREVIEW.md`). The work happens on branch `dsv41-zig`.

The reference is the Python engine at `tensorfold-dsquant` branch `dsv41-quant-e2`. That is prod G19 (`7bd2d67`) plus our E1 and E2 changes, in `src/tensorfold/families/deepseek_v41/cuda/`. Below, `py/` stands for that directory.

The long-context features live on two sibling branches:
- `dsv41-kvsplit`: split KV.
- `dsv41-sessres`: session residency and streamed NVMe.

The quant repo's `docs/ENGINE-E1E2.md`, `LONG-CONTEXT.md` and `RESULTS.md` give the numbers to match.

## 0. Ownership and interfaces (source of truth)

Several agents work on this branch at once. Each one commits only its own paths (`git add <paths>`, never `-A`) and asks the port lead about any interface change. This section is updated whenever an interface changes.

| Owner | Paths | Interface it provides |
|---|---|---|
| Port lead (`dsq-zig-lead`) | `zig/src/families/deepseek_v41/*.zig` except `kernels*.zig`, `m1_*`, this document, `tools/zig/dsv41_*`, the root `build.zig` hook | `config.Config`, `pack.Pack`, `plan.Plan` (a rank's TP weight plan), `stage` (host images), `named` (the loader's tensors by name), `load` (device upload), the M1 gate `tf-dsv41-m1` |
| Kernels (`dsq-zig-kern-1`) | `zig/kernels/cuda/deepseek_v41/**`, `zig/src/families/deepseek_v41/kernels*.zig`, the DeepSeek parts of `zig/build/cuda.zig`, `zig/tests/cuda/deepseek_v41/**`, `docs/DSV41-ZIG-KERNELS.md` | `dsv41_kernels.Kernels` (`load`, `experts(stream)`, `linears(stream)`, ...); `dsv41_kernels.replay.call` (one captured extension call re-issued through its Ops) |
| TP (`dsq-zig-tp-1`) | `zig/src/fabric/tp/**`, `zig/build/tp.zig` and its `build.zig` hook, `docs/DSV41-ZIG-TP.md` | `tp.Session` (below): the bootstrap across two hosts, the communicator `Collective` (NCCL, or the hybrid route over the same-host mailbox or RoCE), the plan link and the fate channel |
| KV and sessions (`dsq-zig-kv-1`) | `zig/src/families/deepseek_v41/kv/**`, `zig/src/sessions/**`, `zig/build/kv.zig` and its `build.zig` hook | the paged pool (256-token pages, `comp.*` 584 B rows, `index_k` 132 B), split KV and its row exchange, the KV side of the priced floor, the session store and NVMe tier (below). The per-slot rings, carries and taps stay in the forward's `state.zig`; sessions see them as opaque bounded state |
| Server (`dsq-zig-srv-1`) | `zig/src/server/**`, `zig/src/core/tokenizer/**`, `zig/src/core/template/**`, sampling, tool parsing | `encoding` (DeepSeek's message encoder), DSML tool calls, reasoning effort, the CUDA engine opened by the server |
| Performance (`dsq-zig-perf-1`) | `zig/kernels/cuda/deepseek_v41/proto/**` | CUDA rewrites where they beat the port by a lot; each one lands behind the same launch interface and is proven bit-equal (or carries its own session tag) |
| Drafting (`dsq-zig-draft-1`) | `zig/src/families/deepseek_v41/draft/**`, `zig/src/core/lanes/trim.zig` and its hooks, the `test-dsv41-draft` step in `build.zig` | DSpark drafter, depth and trees; the `core/lanes` backend for DeepSeek (`draft.Lanes`) over `Target` and `Pass` (below) |

Rules for every owner:
- Interfaces first, so implementations can change behind them.
- Tests run the real implementation. A test that only checks its own fixture does not count.
- No nested per-element loops on hot paths, host or device.
- Where a CUDA rewrite gives a large speed-up, take the time to write it rather than port the old kernel as is.
- **GPU budget:** the RunPod account balance (about $78, no further top-ups) is the limit, and about $50 of it is for pods across all agents.
  - Batch GPU checks into as few pods as possible; the port lead's `zigm1` pod carries other agents' checks.
  - Use short hard maxima, and don't re-run pods to re-confirm what's already proven.
  - Launch with `DSQ_AGENT=<agent id> DSQ_PURPOSE="<what the pod checks>" bash runpod/session.sh ...` (or `hold.sh`). Since quant repo 0240dbf, both refuse to launch without `DSQ_AGENT` and write their own row in `dsv41-exl3-quant/results/zig-pod-ledger.md`. Do not add manual rows for pods launched this way.

**Shared types.**
- **Weights by name.** The loader names every tensor the way the M1 capture names the Python engine's tree: `L<layer>.attn.wq_a.trellis`, `L<layer>.attn.wo_a.<j>.suh`, `L<layer>.moe.w1.data` (ragged) or `.trellis` (stack), `L<layer>.moe.shared.0.w2.svh`, `L<layer>.engram.qk`, `embed`, `norm`, `head.trellis`.
- **Expert tables.** Per routed projection, an int64 `[E]` table of each expert's trellis device address and an int32 `[E]` K2 table. `suh`/`svh` are stacked `[E, K]`/`[E, N]`.
- **Drafting** (`draft/iface.zig`; the GPU forward implements both vtables, `draft/twin.zig` is the CPU twin the tests run):
  - `Target.prefill(slot, ids, sampling, draw) -> u32`: the prompt into the slot's state, the keyed choice after its last row.
  - `Target.window(segments, choices)`: one forward over every slot's `Segment{slot, start, tokens, parents, draws, sampling}`. Row 0 is the pending token at `start` (the slot's cache length); row r sits at `start` + its depth and attends the committed cache plus its ancestors (`parents[r]`, row 0: -1; null: a chain). `choices[i][r]` is row r's keyed choice at `draws[r]`. The rows stay in the state until `keep`; a chain window that is never kept keeps every row.
  - `Target.keep(slot, path)`: keep the listed rows of the last window (a tree path, row 0 first), drop the rest: pool truncation, SWA rings and ratio-2 carries roll back to the path.
  - `Target.taps(slot) -> Taps{address, rows, stride}`: the last window's (or prompt's) taps, bf16 `[rows, 3 x hidden]` (the streams' mean entering layers 37/38/39), row i = window row i.
  - `Pass.ingest(slot, start, taps, rows)`: taps rows `rows[i]` into the slot's context rings at positions `start + i` (the lanes backend sends only the kept path, last `window` rows). In a shared round the lane core drafts before `keep`, so ingest reads the path, never the pool.
  - `Pass.propose(asks, out)`: one pass over every `Ask{slot, anchor, start, params}` (anchor = the pending token at `start`, `params` = the chain kernel's keyed sampling); `out[i]` gets `block` drafts and confidences, and with trees the gathered candidates and base logits `[block, K]`. `Pass.reset(slot)` restarts a slot's context; `Pass.markov()` returns host copies of the Markov / confidence heads for trees (null: chains only).
  - Row cap: 64 rows a forward, 16 a slot (graph buckets 1-8, 12, 16, 24, 32, 48, 64). The depth prices rounds from the measured table (`draft.Costs`, read from Python's `calib-*.json`), never a clock.
- **KV and sessions** (`zig build test-kv`; family-neutral parts in `zig/src/sessions`, DeepSeek's in `kv/`):
  - `kv.Layout.fromConfig(cfg, .{ .kv = .fp8|.fp4, .index = .fp8|.bf16, .split })`: `comp.L` / `index_k.L` for every KV source layer (duck-typed on `Config`); `bytesPerToken(W)` is 1,790 replicated, 1,060 split over two ranks; `pool()` gives the `sessions.Layout`.
  - `sessions.Pool.init(gpa, layout, tokens, world, rank)`: page ids are lowest free first, one free set a residue mod W, so both ranks hold the same ids for the same calls. Logical page k always lands on a page of residue k % W, and rank `p % W` stores the split families' rows of page p. `Slot.ensure / truncate / adopt / push / reserve / releaseAll`; `takeDirty()` is the table range to upload; `tableAt(i)` / `localTableAt(i)` give the physical table and the split table (local page or the discard page). `share` / `drop` are holder counts. `available()` is O(W): pages any request can count on.
  - `sessions.PageStore` (vtable `read / write / copyPage / fence / waitFence / publish`): the bytes behind pages, by family-tensor page. `kv.DevicePool` implements it on the GPU, `sessions.HostPool` in host memory.
  - `kv.DevicePool.init(gpa, driver, ctx, pool, compute_stream, slots, max_pages)`: one zeroed device tensor a family (split families hold npages / W + the local null and discard pages) and the stacked `[slots, max_pages]` int32 tables. `syncTables()` uploads each slot's dirty range on the compute stream. `view(fam, slot)` is the attention's pool arguments (`ops.attn.Args`): `cv` = base, `csc` = base + 576, `cvs` = `css` = row bytes, `pt` = the slot's table (the split table for split families), `pts` = max_pages, `psh` = log2(256 / ratio).
  - `kv.split`: `Exchange.dense(DenseArgs, recv, stream)` is the graph-shaped gather + remap (entry (r, j) becomes token `owner x R x K + r x K + j`) followed by tp's `allGather`. `planUnion` + `Exchange.unionBlock` handle prefill segments (sorted unique rows, per-owner lists padded to M, halved while (W + 1) x M x row bytes passes the cap). The device kernels sit behind `split.Kernels`; `HostKernels` is the reference.
  - `sessions.Store.init(gpa, io, pool, page_store, disk, policy)`: `find(tag, prompt)`, `save(slot, bounded, ids, tag, kind)`, `restoreRam(id, slot)` (full pages shared, the partial page copied), `beginRestore(id, slot)` / `beginPark(id)` / `settle(job)` (the tier thread moves bytes; pages and the index change only at `settle`, the plan's point, on both ranks), `enforce()` (the residency budget: idle chats and pool bytes held by entries alone), `planSpill(need, keep)`, `reconcile()` after a restart. Bounded state is a `store.Bounded` vtable the forward implements.
  - `sessions.Disk.open(gpa, io, .{ root, compat, rank, world, budget, staging, chunk, lanes })` is the NVMe tier, format 2: per-chunk SHA-256 on lane threads, a host transient of the staging set only, and delta files, so a newer turn writes only its own pages.
  - `kv.budget`: `Boot.poolTokens` gives the largest pool that keeps the hard floor (split's union buffers and the NVMe staging priced). `prefillPrice`, `restorePrice`, and `admit(knobs, pool, ask, usable_by_rank)` gate an admission on pages by residue and on every rank's memory.
- **TP** (`zig/src/fabric/tp`, module `tp` from `tp_build.module(b, target, optimize, cuda)`; `docs/DSV41-ZIG-TP.md`):
  - `Session.open(s: *Session, driver, cfg: tp.Config, kernel_image)`: bootstrap, settings agreement, NCCL, the small-exchange transport (`TF_COMM_BACKEND`: `nccl` | `mailbox` | `roce`, a probe against NCCL, all ranks fall back to NCCL together), plan link, fate channel. `tp.Config.fromEnv()` reads `TF_TP_*` and Python's names. `s.close()` is the normal stop.
  - `s.comm() -> Collective`: `allReduce(src, dst, count, dtype, op, stream)` (rank-order sums), `allGather(src, dst, count, dtype, stream)` (recv `[world][count]`, rank 0 first), `exchange(src, dst, count, dtype, peer, stream)`, `send` / `recv`. All are graph-capturable on the caller's stream, with no host sync and no allocation. `barrier()`, `check()` (a recorded transport failure) and `abort()` are host calls.
  - `s.plan: tp.PlanLink`, its own TCP connection: rank 0 `send(ints: []const i64)`, the others `recvGrow(gpa, *std.ArrayList(i64)) -> []i64` (grows to the frame, up to 64M words, so admissions with 1M-token prompts fit; the buffer is reused) or `recv(out: []i64)` into a fixed buffer. Both block while rank 0 is idle and return `error.PeerClosed` once it is gone. Plan encode/decode, `follow()` and the stop plan are the port lead's (`rounds.zig`).
  - `s.fate: ?tp.Fate` (null with `TF_DSV41_FAILFAST=0`): `s.fatal(where, err)` fails this rank and every rank exits 70 about a second later; `fate.onFail(.{ctx, call})` adds a callback (rank 0 fails its requests); `fate.peerMem(r)` is rank r's last memory report; `fate.expectStop()` before a normal stop.
  - Tests without a GPU: `tp.host.run(world, ctx, body)` runs in-process ranks behind the same `Collective`.
- **Server** (`zig/src/families/deepseek_v41/serve/`, module `dsv41_serve`; host only, goldens against the Python engine):
  - `sampling.Picker.init(gpa, capacity) -> Picker`: one thread's scratch (capacity = the vocabulary, for whole rows); nothing allocates after.
  - `Picker.pick(row: Row, position: u64, s: ?Sampling) -> Pick` and `Picker.pickRows(rows, positions, s, out: []Pick)`: `batch._choose`. `Row{values: []f32, ids: []u32, stats: [][2]f64, count}` is a row's gathered candidates (any order; `Candidates.row` reads the `[W, R, 2k]` fp32 gather), `count` the merged cut (`candidateCount(s, vocab, nucleus_count)`), `stats` each rank's (max, sum exp(v/T - max/T)) for nucleus rows (`isNucleus`). `Pick` is `.token` or `.full_row` (gather the row's whole vocabulary and pick again without stats). Greedy (`s == null` or T <= 0) is the largest logit, ties to the lower id.
  - `sampling.accept(window: []u32, chosen: []u32) -> usize`: drafts kept from a chain window `[pending, d1, ...]`; the round emits `chosen[0..accepted + 1]`.
  - `tokenizer.Tokenizer.load(gpa, io, dir)`, `Encoder.encode`, `Detokenizer.push`; `template.render(a, messages, Options, *Problem)` (reasoning effort: `template.effortOf`, the Python app's names — medium is 75, not the release template's 62); `tools.parse` / `tools.Stream.feed`. The server reaches the engine only through `engine_api.Engine`.

## 0a. Parallel workstreams (2026-10-07, integration: `dsq-zig-arch-1`)

Five agents work on `dsv41-zig` at once toward one combined pod gate and one Spark window. Each owns its new files; hooks into shared files stay small and are reviewed by the integrator.

| Agent | Workstream | Owns (new files) | Hooks it may add to shared files |
|---|---|---|---|
| `dsq-zig-arch-1` (integrator) | the Spark window's fixes, M5 (paged pool, split / compact KV, sessions, boot check), the combined pod job, the Spark kit and runbook | `forward.zig`, `forward_prefill.zig`, `block*.zig`, `target.zig`, `model.zig`, `run.zig`, `kv_state.zig`, `sessions_gpu.zig`, `profile.zig`, `m5*.zig`, `m2b.zig`, `tools/zig/dsv41_m1/job.sh`, `tools/zig/dsv41_m2c2/**`, `tools/zig/dsv41_m4/**` | — |
| `dsq-zig-samp-1` | exact sampling (== Python's keyed sampling, top-p / top-k / temperature) | its own sampling files | `target.zig` (the `sampling` / `draw` path of prefill and window), `forward.zig` (a candidates call next to `greedy`) |
| `dsq-zig-slots-1` | several live slots and graph row buckets (row mode) | its own slot / row-mode files | `forward.zig` (slot state by index), `target.zig` (`Slots` refusals), `graphs.zig`, the emitters' row-mode arguments (`SL`, `PTS`, `ROWS`, `rslot`) |
| `dsq-zig-trees-1` | DSpark draft trees (`parents`), drafted == serial | its own tree files | `target.zig` (`TreeWindow` refusal), `keep(path)` |
| `dsq-zig-speed-1` | graphs on in M4, R3 Engram, host overhead, the per-round timing tool | its own speed files | `forward.zig` (the window path), `serve_engine.zig`, `graphs.zig`, `profile.zig` (extend, don't replace) |

**State the workstreams start from** (do not rebase away):
- `cf1cd20`, the Spark window's fixes:
  - `op_release`: every rank's slot resets on release;
  - prefill selects with `attn_cuda` top-k at up to 64 rows (`backend._dt`);
  - deterministic drafting: `op_ds_reset` reaches the followers, and lanes `Config.timed` is off in the served engine;
  - `TF_DSV41_PROFILE` / `profile.zig`, and `M2B_PROF_ROWS`;
  - kit M4 with graphs on;
  - `dsv41_m2b_ref.py` `--aot-prompts` / `--profile`;
  - `job.sh` `M2B_M4`.
- `7aefcc4`, M5:
  - `block.Pool` and the `s.kv.*` roles, `kv_state.zig`, `sessions_gpu.zig`;
  - `tf-dsv41-m1 m5-emit` / `m5`;
  - the session ops `op_sess_save` / `op_sess_restore`;
  - `Forward.sess`, and `Forward.kv`.

**Interface rules.**
- **Plan-link operation numbers** (`forward.zig`): 1–11 are taken (window, greedy, keep, stop, ds_ingest, ds_propose, prefill, sess_save, sess_restore, release, ds_reset). Reserve new ones by appending to this list in this section, in the same commit that adds them:
  - 12–15: `dsq-zig-samp-1` (12 taken: `op_sample`, `[op, row0, n, T, top_k, top_p, min_p]`);
  - 16–19: `dsq-zig-slots-1` (16 `op_select`, 17 `op_rows`, 18 `op_rows_keep`, 19 `op_rows_drop`: `slots.zig`);
  - 20–23: `dsq-zig-trees-1` (20: `op_tree_drop`, `Forward.drop`);
  - 24–27: `dsq-zig-speed-1` (none used: its hooks ride on `op_ds_propose`).
- **Defaults.** Every new behaviour is behind a knob or a new code path, and defaults to today's. With every new knob off, the gates `check cap24-*/cap15/cap18/cap22` and `dcheck cap16` still give 0 mismatches. Run `tf-dsv41-m1 m5-emit cap18` before committing a change to an emitter.
- **Pool vs contiguous.** A slot's KV is either the contiguous roles (`s.L<i>.comp.*`, `s.L<i>.ik`) or the pool (`block.Options.pool`, the `s.kv.*` roles). New code must work with both, or refuse the other by name.
- **Collectives.** Leader and followers make the same collective calls in the same order. Anything rank 0 decides (sampling draws, tree shapes, slot admission) travels on the plan link before the work that depends on it.
- **Commits.**
  - Use `git add <your paths>` only, then `git -c user.name="Jay Leaton" -c user.email="jayleaton@gmail.com" commit`, with the message ending in `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. No push.
  - Build with at most `-j4`.
  - Pod binaries are built from a clean `git archive` tree.
  - Hook commits to shared files: one small commit each, its message naming the hook, so the integrator can review it.
- **Pod steps.** No workstream launches a pod: the integrator runs one combined held 2x PRO 6000 pod (cap $5). Each workstream lists its pod step here, under "Pod steps", as follows:
  - the `job.sh` knob or the command;
  - what it compares, and against what;
  - the minutes it needs;
  - any assets it needs (captures, references, AOT prompts).

**Pod steps** (filled in by each workstream):
- `dsq-zig-arch-1`:
  - the window fixes;
  - prefill gates at 45 / 173 / 9 tokens (64-row top-k, odd key counts);
  - the R1 decode gate, eager (profiled) and graphed;
  - M4 on the pod (HTTP == CLI with drafts, drafted == serial);
  - M5 `paged` / `split` / `compact` / `park` (`M1_JOB=m5`);
  - the boot check (host goldens).
- `dsq-zig-samp-1` (keyed sampling on the GPU path):
  - **What it is.** `sampling_pipe.zig` runs Python's steps on every rank, so the collectives pair up and every rank decides alike:
    - each rank's top `count` (`pick.top`), all-gathered;
    - `vsample.merge`;
    - the nucleus statistics plus the whole-row fallback (`nucleus.finish`);
    - `serve/sampling.zig`'s `Picker` (= `Batcher._choose`), keyed at the row's position.
  - **Device side.** `sampling_dev.zig` (`topk_keys.cu` + the new `sampling.cu`: `row_stats`, the packed rows) behind `sampling_pipe.Device`. `HostDevice` is the reference the host tests run.
  - **The other files.** `sampling_gpu.zig` holds `GpuSampler`: model.zig makes one on every rank; rank 0 sends `op_sample` before its gathers, and the followers run it from `follow`. `sampling_kernels.zig` loads its own fatbin `dsv41_sampling`.
  - **Hooks:**
    - `forward.zig`: `Forward.sampler` and `op_sample`;
    - `target.zig`: prefill and window choices with T > 0 go to `sampler`; without one they are refused (`error.Sampling`);
    - `model.zig`: init / deinit;
    - `serve_cli.zig`: a trace line's `sampling`;
    - build: `zig/build/sampling.zig`. `cuda.zig` gets the `dsv41_sampling` kernel, `tf-dsv41-samp`, and `dsv41_serve` for `tf-dsv41-m1`, whose `core` is now `safetensors_root.zig`. `server.zig` gives `dsv41_serve` to the native engine.
  - **Host gate:** `zig build test-dsv41-sampling`. The committed serving goldens on two in-process ranks: 207/207 sampled rows equal Python's tokens, including 14 whole-row fallbacks. A window's rows pick as each row alone does, and as `exact_sampling.choose` over the whole row.
  - **Slots and trees.** A mixed window calls `GpuSampler.choose` once per slot's rows: `row0` / `n`, one sampling each, one `op_sample` each. Tree chains go through `window`, so they sample too.
  - **Pod step:** `tools/zig/dsv41_samp/samp_step.sh Z LP ASSETS HI RES LOCAL`, inside `M1_JOB=m2b` after the M2b reference (its assets: aot / rope / engram-host). About 1 min, plus a Python load if `SAMP_TRACE` is unset, plus 2 x (Zig load + 6 x 64 tokens). It runs:
    1. `tf-dsv41-samp` on GPU 0. Expected: `PASS tf-dsv41-samp: 0 checks failed`.
    2. `dsv41_samp_ref.py` at TP=2: 6 prompts (37, 50, 66, 83, 128 and 292 ids), 6 samplings (T 0.6–1.3; top_p 0.8–1.0; top_k 0 / 20; min_p 0 / 0.05; a seed above 2^63). To save the second Python load, call `dsv41_samp_ref.run_cases(fw, cfg, rank, out, prompts, steps)` at the end of `dsv41_m2b_ref.py` on both ranks and pass `SAMP_TRACE`.
    3. `tf-dsv41-m1 generate` on that trace, `TF_DSV41_DRAFTS=0`, then `1`. Expected in each: `PASS M4 CLI == HTTP: 6/6 replies equal` (here it means Zig == Python).
    - With `SAMP_OWN_PREFILL=1` (the default), the prompt lengths need their Triton variants: add `37,50,66,83,128,292` to `M2B_AOT_PROMPTS`, or set `SAMP_OWN_PREFILL=0`.
    - The kernel fatbins must include `dsv41_sampling`. With `-Dnvcc=` it is built. With `-Dfatbins=` (`build-pod.sh`), copy `zig-dsv41-scratch/samp/fatbin/dsv41_sampling.fatbin` into the fatbin dir first. It was built with nvcc 13.3, the build's flags, for sm_120 and sm_121.
- `dsq-zig-speed-1` (decode speed):
  - **Found: the served engine never read Engram's table.** Every Zig path took M2a's index rows (`engram_host.indexRows`), so the Spark window's M4 replies were not the model's, and its 46.7 tok/s paid no NVMe reads at all. M2c-2 passed because its reference used the same index rows.
  - **R3, built:**
    - `engram_rows.zig` reads the kit's packed shards (`engram-l<L>-r<r>of<W>.bin`, Python `ShardRows`). It uses O_DIRECT, with a buffered fallback that drops the read pages.
    - Values convert through one 65,536-entry table, whose SHA-256 equals the one torch builds (`dequant(...).to(bfloat16)`). Only the 514 NaN entries depend on torch's code path, and no packed table holds a NaN code.
    - Reads go out as io_uring batches of merged sector extents. Without io_uring (a seccomp profile) they run when the window needs them.
    - An LRU cache holds `TF_DSV41_ENGRAM_CACHE` records a layer (65,536). Prompt-sized batches bypass the cache.
    - `engram_host.Source` is the interface (doc 4b's `EngramSource`). The forward opens `TF_DSV41_ENGRAM_DIR` on first use on every rank. With it unset, the forward keeps today's index rows and logs a warning.
    - **Read-ahead:** the pending token's rows (the next window's row 0) are issued when rank 0 sends `op_ds_propose` and when a follower receives it, so they are read during the DSpark pass. `engram_host.lookbackAt` gives the lookback from the slot's tail plus the unkept window. A wrong guess costs a read, never a bit. `TF_DSV41_ENGRAM_PREFETCH=0` turns the read-ahead off.
    - **Not built:** Python's GPU-side wait (`engram_gate.cu`'s spin on a pinned word, then the copy, before layers 1 / 14). Without it, the drafted rows' reads still sit in `graph.stage`. It needs a kernel (dsq-zig-kern-1) and a pod.
  - **Graphs on in M4:** `model.zig` sets `Forward.graphs_default` (`graphs.Settings.fromEnvOr`) on every rank, so the ranks agree. `TF_DSV41_GRAPHS=0` turns graphs off. The gates' drivers (m2a / m2b) still default to off.
    - Slots: `Model.openRows` sets `graphs_read`, so one-slot graphs stay off with several slots. Keys are one slot's (rows, context bucket); row mode brings its own.
  - **Host overhead:**
    - `Forward.greedy` now stages through pinned memory and keeps its device pairs. Before, every window did a `cuMemAlloc` and a `cuMemFree` (which synchronizes), plus pageable copies.
    - `greedy.argmax` is vectorized and picks the same column as the scalar loop (ties, ±0, NaN): 61 → 9 µs a 64,640-wide row.
    - An eager window re-emits its whole program and scans the AOT set per call. `tf-dsv41-m1 plan` emits 6 whole-model windows plus 3 DSpark blocks in 20 ms on the dev box, about 2–3 ms a window before any issue cost. Graphed decode does neither, which is why graphs-on is the first lever.
    - Still eager every round: the DSpark pass (about 100 calls issued, ingest re-emitted, and `hostStep`'s pageable download of [5, 64,640] fp32 logits plus a host top-k). Graphing the pass, as Python's `draft_graphs` does, is the next host item.
  - **Glue fusion:** not built. It is kernel work in the shared kernel dir, it needs a GPU bits test, and KERNELS 3.2f estimates 2.4–2.8 %, below graphs, R3's GPU wait and the pass graph.
  - **Per-round timing:**
    - `phases.zig` (`TF_DSV41_PHASES=1|sync`, `TF_DSV41_PHASES_OUT`) uses Python `phases.py`'s names and report shape.
    - `phases_timed.zig` wraps the lanes' `Target` / `Pass`. The forward times `forward`, `graph.stage` / `graph.replay` / `graph.capture`, `engram.wait`, `prefetch` and `candidates`.
    - `tools/zig/dsv41_perf/gap.py` compares the two engines. `phases` sets Python m2bench's report against the Zig server's, in ms a round. `windows` sets the pod's R1 per-window wall times (Python `--profile`, Zig `M2B_ZIG_PROF_ROWS`, eager and graphed) side by side with GPU ms by kernel family.
  - **Hooks** (one commit each): `forward.zig` (fields, greedy, the Engram source and read-ahead, timers, `fromEnvOr`); `model.zig` (`graphs_default`); `serve_engine.zig` (the phase decorators); `deepseek_v41.zig` (the test list); `dsv41_m2b_ref.py` (`--engram-dir` / `--engram-make`).
  - **Host gate:** `zig build test-dsv41`, 29 pass and 1 skip. It covers:
    - the Engram rows against Python's own `engram_host.py` (`fixtures/engram-rows-ref.txt`, from `tools/zig/dsv41_perf/engram_fixture.py`): the table, three index sets read synchronously and through io_uring, read-ahead, bulk read-through and eviction;
    - the lookback;
    - `greedy.argmax` against the scalar loop (200 random rows plus edge cases);
    - the phase report.
    - `python3 tools/zig/dsv41_perf/gap.py selftest`.
  - **Pod step** (inside `M1_JOB=m2b`; no extra Python load, about 2 min in all):
    1. **`M2B_ENGRAM=1`** (to add to `job.sh`): put the real-row path under every existing gate.
       - The R0 and R1 references get `--engram-dir $LOCAL/engram --engram-make`. Each rank writes its own sparse shards, with records under the prompts' rows (`engram_shards.py`, a few MB written).
       - Export `TF_DSV41_ENGRAM_DIR=$LOCAL/engram` for every Zig rank step.
       - Expected: every gate's PASS lines unchanged (tokens / logits 64/64, blocks, `M2B_PREFILL_GATE=1`'s 65/65 roles, which read the written records), and `engram: rows from ...` in each rank's log.
    2. **The R1 gates with timing:** `M2B_R1=1 M2B_R1_GRAPHS=1 M2B_PROFILE_ROWS="1 2 4 6" M2B_ZIG_PROF_ROWS="1 2 4 6"`. The greedy rewrite is gated here: m2b-r1 and m2b-r1-graphs PASS on both ranks.
    3. **`M2B_M4=1`**, with three changes to `m4env`:
       - drop `TF_DSV41_GRAPHS=1`, so the new default is what runs: `server.log` must show `graphs: on`;
       - add `TF_DSV41_PHASES=1 TF_DSV41_PHASES_OUT=$m4dir/phases.json`, and `TF_DSV41_ENGRAM_DIR` with step 1;
       - run the `generate` replays with `TF_DSV41_ENGRAM_PREFETCH=0`.
       `PASS M4 CLI == HTTP` then also proves R3's read-ahead on == off. `drafted == serial` holds as before.
    4. **`bash $Z/tools/zig/dsv41_perf/speed_step.sh $Z $RES`** (about 10 s) writes `RES/speed-gap.md`: the per-window table (rows 1 / 2 / 4 / 6: Python R1, Zig eager, Zig graphed) and the GPU ms by family. It also greps the `graphs: on` and `engram:` lines.
  - **Spark window:** the kit sets `TF_DSV41_ENGRAM_DIR` to the node's shards (`HEAD_ENGRAM` / `WORKER_ENGRAM` of prod-perf1.env) on both ranks.
    - M2c-2's reference takes `--engram-dir` with the same path (the real-row gate on the whole model).
    - The served comparison: `bench_http.py` on Zig with `TF_DSV41_PHASES=1`, and Python's m2bench with `TF_DSV41_PHASES=1` on the same prompts, then `gap.py phases`.
- `dsq-zig-trees-1` (draft trees on the GPU target; design below):
  - **`M2B_LANES=4`** (`M1_JOB=m2b`, 2 GPUs, about 4 min after the shared load): lanes with first-position trees (2 siblings) and the tree oracle's drafts over the GPU target at TP=2. Rank 0's tokens must equal the reference's serial greedy tokens, with trees > 0 and 0 < sibling wins < trees. Expected lines: `PASS M3 lanes mode 4 rank 0: tokens 64/64 equal, drafted N, accepted M, trees T, sibling wins W, ...` and `PASS M3 lanes mode 4 rank 1`. `m3-4/m3-rank0.json` also records `chains` and `rerun_rows`. No new assets.
  - **`M2B_LANES=3` with `TF_DSV41_TREE=3`** (about 4 min): DSpark's own siblings (gathered `[block, 128]` candidates) through the same path. The gate is tokens equal and drafted > 0. On the 0–24 prefix the taps are zeros, so the tree counts are informational. `job.sh` runs one mode list with one environment, so this needs its own mode token. The suggested hook is `3t` → `M2B_LANES=3 TF_DSV41_TREE=3`, with the result dir `m3-3t`.
  - **Pool and split with trees:** the `M2B_M4=1` step again, with `TF_DSV41_TREE=3 TF_DSV41_POOL_TOKENS=<the M5 step's> TF_DSV41_KV_SPLIT=1` added to `m4env`. It checks HTTP == CLI with drafts, and drafts == serial. Trees use no private pages (below), so this checks `Forward.drop` against the pool's reserved pages and split's exchange.
  - **Spark window (whole model):** M4 with `TF_DSV41_TREE=3` against Python prod's engine with `TF_DSV41_TREE=3` on the same 8 prompts, at the same calibration table. Tokens must be equal. Drafted, accepted, rounds, trees and sibling wins should be close to Python's, not identical: the host `_chain` order is fp32 (as for chains).
  - Host: `zig build test-dsv41-draft` (`branches_test.zig`).

**Draft trees on the GPU target (`dsq-zig-trees-1`, `draft/branches.zig`).** Python's `tree.py` runs each sibling branch as its own chain window on a shadow slot. The shadow holds a copy of the slot's rings and carries, and gets private pages under the window. Ours runs the branches on the slot itself, lazily, as chain windows:
- the window first runs the chain of first children from row 0;
- when the target's choice after a row is a later child of it, that chain is dropped uncommitted (`Forward.drop`, op 20, on every rank), and the path so far plus the sibling's chain runs instead;
- the accepted path is then a prefix of the last chain, which `keep(path)` commits.

What follows from this:
- **Exactness:** every chain is an ordinary chain window, so each row's bits are the serial decode's. A re-run row must choose as before (`error.BranchUnstable`).
- **No shadow state:** the paged pool, split KV and sessions see plain windows. Python refuses trees with split KV.
- **Cost:** a losing sibling costs no row. A winning one costs a second window of its path (Python: the duplicated pending row plus the sibling's rows in the same window).
- **Pricing:** the plan prices trees as Python does (`TF_DSV41_TREE_DUP_MS` with the pending row counted), so the rounds' choices are Python's.

The hooks:
- `target.zig`: `GpuTarget.tree`, the `parents` path in `window`, `keep`, `taps().map`;
- `forward.zig`: `drop`;
- `serve_engine.zig`: the knobs;
- `m3.zig`: `shapeOf.candidates`, the gathered width (128 at TP=2: lanes' siblings read `[block, world × k]`, as Python's `last_tree`), and `generateTrees`;
- `m2b.zig`: mode 4.

`iface.Taps.map` maps the tree's rows to the last chain's taps rows for lanes' ingest. Parent-conditioned trees (`TF_DSV41_TREE_PC=1`) are refused by name.

### 0b. Several live slots and row buckets (`dsq-zig-slots-1`)

`TF_DSV41_SLOTS=N` (1 by default: today's one-slot path, nothing changes) gives the GPU path N live slots with batched decode windows, as Python's batcher runs them (row mode, G8: `rowtab.py`, `rowmode.py`, `rowgraphs.py`). It needs the paged pool (`TF_DSV41_POOL_TOKENS`); the contiguous slot is refused by name.

- **Files:**
  - `rowtab.zig`: the row table. One int64 column each for ids, pos, rslot, wslot, prev and src, then an int32 copy of rslot for split KV's exchange. Also the buckets (1–8, 12, 16, 24, 32, 48, 64), the padding (rows mirror the last real row and write nothing), the checks, and the packing of a mix into forwards under the row cap.
  - `rowmode.zig`: the row program, as a rewrite of `block.emit`'s window at R rows, its end at the context bucket's end. POS / SL / ROWS / PTS / PREV / OFF in `_rope`, `_kv_store`, `_pool_norm`, `_index_k` and `_scores`, and pos / sl / pts in the CUDA attention and top-k, all as `compress.py` / `index.py` / `attn_cuda.py` pass them in row mode. The stacked roles are sized for S slots. A Triton call with row knobs it does not know is refused (`UnknownRowsCall`); a union exchange is refused (`UnionInRows`). `block.zig` is unchanged, so the M1 / M5 emit gates are untouched.
  - `slots.zig`: `SlotSet`. The stacked roles are the SWA rings (`s.L<i>.swa.*`, S x ring rows), the ratio-2 carries (`s.L<i>.carry`, [S, 2D]) and the page tables (`s.kv.pt` / `s.kv.ct`, [S, pts]). `activate(s)` points the one-slot roles at slot s's views and loads its `Forward.Slot`, so prefill / release run on slot s unchanged; the leader sends `op_select` first.
  - `batch.zig`: `Batch`, the batched window. It keeps pending windows, maps every slot's pages, stages the row table and the Engram rows (pinned halves), then runs the row program eagerly or through the row graph cache keyed (bucket rows, context bucket). Each slot's window stays pending until `keep(slot, accepted)`, which rolls the carry from that slot's accepted row into its carry row and moves its tail and position. A round whose rows pass the cap runs as several forwards; the earlier forwards' projection rows are stashed for their keeps.
  - `slots_cli.zig`: `tf-dsv41-m1 slots PACK ASSETS TRACE`, the gate (below). `rows_test.zig`: `zig build test-dsv41-rows`.
- **Knobs:** `TF_DSV41_SLOTS` (1–16, every rank the same), `TF_DSV41_ROWS_CAP` (16: a bucket, the rows a forward takes; `block.emit`'s decode path is gated up to 16 rows, `mhc_cuda`'s MAXR), `TF_DSV41_GRAPHS` (row graphs, with the forward's default: on in M4; one-slot graphs stay off with several slots).
- **Hooks** (one commit each): `kv_state.zig` (`Options.slots`, `Kv.slots`, `select`); `forward.zig` (`Forward.ext`: a follower hands op numbers it does not know to the workstream); `target.zig` (`rows`: window / keep / taps / release go to `Batch`, prefill activates the slot); `model.zig` (`slots`, `openRows`, the row programs in the buffer plan); `serve_engine.zig` (lanes slots and LaneHost admission = `m.slots`); `m1_main.zig` (`slots`); `zig/build/cuda.zig` (`test-dsv41-rows`).
- **Not yet with several slots** (each refused by name): sessions (`sessions_gpu.zig` snapshots one slot's roles: not opened), draft trees and keyed sampling in row windows (`TreeWindow` / `Sampling`), DSpark drafting (`dspark_gpu.zig` is one slot: run `TF_DSV41_DRAFTS=0`). Engram rows come through the forward's source (`Host.windowRows`: R3's table or M2a's index rows), per segment with its slot's tail. Mixes over 16 rows run as several forwards; 24–64-row forwards need `block.emit`'s >16-row decode path (Triton mHC above `mhc_cuda`'s 16 rows).
- **Host gate:** `zig build test-dsv41-rows`, 13/13. It covers the table against Python's `plan_rows` example; checks, packing and layout; and the whole backbone window (paged and split pool, R1 on and off, R 1 / 4 / 12 / 16) through the real `block.emit` + `transform`. In that window every CSA2 step reads the table's columns, NK is the context bucket's, the attention / top-k get pos / rslot / pts, and the plan's stacked roles are S x one slot's while every other role is unchanged.
- **Pod step** (TWO GPUs, inside `M1_JOB=m2b` after the pack copy; `source tools/zig/dsv41_slots/job.sh; slots_gate`, with `SLOTS_PYSRC` = a Python tree with `rowgraphs` / `slots`, e.g. the decode1 twin's src with its `TORCH_EXTENSIONS_DIR`). About 14 min for the reference (one TP=2 load of layers 0–`$hi`, 4 prompts x 64 tokens, the row buckets warmed), then about 4 x 5 min:
  1. `slots-ref`: `ref.py` writes `trace.jsonl`, `ref.json`, `rope.bin` and `aot/`, including the row-mode Triton variants (`ROWS`, int64 POS, SL / PTS) for every bucket up to 16. `slots-engram-host` writes `engram-host.bin`.
  2. `slots-s4-eager`, `slots-s4-graphs`, `slots-s2-graphs`, `slots-s1`: `ranks.sh`, our two ranks serving the trace (`TF_DSV41_SLOTS` 4 / 4 / 2 / 1, `TF_DSV41_GRAPHS` 0 / 1 / 1 / 0, `TF_DSV41_DRAFTS=0`, `TF_DSV41_POOL_TOKENS=18432`, `TF_DSV41_LAYERS=0-$hi`). Expected in each `rank0.log`: `PASS slots batched == Python: 4/4`, `PASS slots alone == Python: 4/4`, `PASS slots batched == alone: 4/4`. s4-eager against s4-graphs is graphs off == on; s1 is today's path against the same trace.

## 1. How the Zig engine is built

| Part | Where | What it does |
|---|---|---|
| Build | `build.zig`, `zig/build/cuda.zig` | Linux: `.cu` files go to fatbins through `-Dnvcc=` with each Python extension's own nvcc flags, or prebuilt ones come in through `-Dfatbins=`. They are embedded in the binary (`zig/src/cuda/kernels.zig`). Without nvcc, the runtime builds host-only with empty images. `zig build test` runs the host unit tests on any OS. |
| CUDA runtime | `zig/src/cuda/` | `driver.zig` (dlopen of libcuda), `context`, `memory` (`DeviceBuffer`, `HostBuffer`), `stream` and `event`, `module` and `launch`, `graph.zig` (capture, or build node by node, then update in place), `cublaslt`, `nccl.zig` (dlopen of libnccl: `CommInitRank`/`InitAll`, `AllReduce`/`AllGather`/`Broadcast`), `triton.zig` and `aot.zig` (Triton cubins captured from the Python engine, launched as Triton 3.7's launcher does), `segments.zig` (staggered prompt segments). No torch anywhere. |
| Backend-neutral core | `zig/src/core/` | `safetensors.zig` and `checkpoint.zig` (header index, mmap), `tokenizer/` (HF `tokenizer.json`, parity with `tokenizers`), `template/` (Jinja2 as `apply_chat_template`), `copy_index.zig` (copies from the context), `engine_api.zig` (the engine a server drives), `lane_host.zig`. |
| Lane core | `zig/src/core/lanes/` | The host side of `lane_engine.py`: streams, windows, drafts and acceptance. A backend implements `backend.Backend` (`prefill`, `first`, `queue`, `read`, `verify`, `keep`, `draft`, `release`). The exactness rule is that drafted output is byte-identical to serial decode, and `--no-drafts` is the reference. A forward holds at most 32 rows. |
| Families | `zig/src/families/` | `nemotron/` is the model to copy. It has `cuda_weights` (loader into the Python layouts), `cuda_kernels` (`.cu` plus the captured Triton set), `cuda_forward` (kernel order as in Python), `cuda_state`, `cuda_engine`, `cuda_lanes` (lane backend), `cuda_mtp` (drafter), `cuda_sampler` and `cuda_dump` (per-block dumps for the oracle). `flashnext/` and `kimi_k3/` are Metal-only. |
| Server | `zig/src/server/` | OpenAI and Anthropic routes, SSE, tool parsing, grammar fields, metrics. Engines are opened through `zig/src/server/engines.zig`, which reads the backend module built in. Only `native/metal.zig` and `native/none.zig` exist today, so CUDA is not served yet. |
| CLI | `zig/src/cli/cuda_main.zig` | `tensorfold run MODEL --tokens ...`, `check-weights`, `teacher`, `prefill`, `rounds`, `lanes`. These are Nemotron's oracle checks. |
| Oracle runs | `zig/tests/cuda/box/run.sh`, `zig/tests/cuda/nemotron/capture.py`, `aot_pack.py` | Run the Python engine in `nvcr.io/nvidia/pytorch:26.07-py3` to capture Triton launches, weight digests, block dumps and tokens. Then build the Zig engine with the image's nvcc and compare. |

Rules from `ZIG-PREVIEW.md`:
- Output stays identical to `--no-drafts`.
- bf16 activations, fp32 accumulation, no precision trades.
- Files of 600 lines or fewer, with one-line comments.
- `zig build test` passes.

## 2. Where the port stands (this branch)

- **Toolchain.** Zig 0.17.0 runs from user space (`~/.local/opt/zig`). `zig build test` passes on the x86 dev box: 17/17 steps at the base commit (about 5 minutes cold), 22/22 with the family.
- **Cross-build for GB10.** `zig build -Dtarget=aarch64-linux-gnu` cross-compiles `tensorfold`, `tf-cuda-test` and `tf-dsv41-test` for GB10 in about 80 s without any CUDA toolkit.
  - Fatbins need nvcc. The kernel agent builds them in the local `tensorfold-dsv41` image (nvcc, `sm_120`/`sm_121`).
  - GPU tests need a GPU.
- **Family host side (`zig/src/families/deepseek_v41/`, step `zig build test-dsv41`):**

| File | Python twin | What it does |
|---|---|---|
| `config.zig` | `py/config.py` | Reads `config.json`'s `text_config`. Computes the CSA2 topology (`mode`: swa / full / reindex / reuse, `kvSource`, `indexSource`, candidates), Engram and DSpark. Refuses routing the kernels do not implement (`scoring_func` other than `sqrtsoftplus`, `topk_method` other than `noaux_tc`). |
| `exl3.zig` | `py/weights.py` `X3`, `X3Stack`, `X3Ragged`, `ragged()` | Group geometry (K2 = words / 8) and codebook markers. A rank's TP part as byte spans: rows are contiguous, columns are one run per K-tile row. The routed-expert `Layout` is a stack when there is one width and ragged (back-to-back offsets) when each expert has its own. |
| `pack.zig` | `py/weights.py` `Shards` | Indexes the pack from its headers with pread and no mmap (on GB10 the page cache is GPU memory). Follows the index's weight map, since real packs hold the same name in two shards. Reads spans and marker values. |
| `names.zig` | `py/weights.py` `Loader`, `py/drafter.py` | Every block's EXL3 groups, natives, splits and expected shapes, DSpark's heads, and the names the text engine knowingly leaves unread: vision, `image_*`, `gate.bias_vl`, and the bf16 `.weight` copies of EXL3 groups. |
| `stage.zig` | `py/weights.py` `Loader.x3`, `ragged()`, `torch.stack` | Reads a group's TP part into host images, and packs a routed projection's experts back to back with their int32 K2 table and byte offsets, the format the kernels take. Suh and svh are stacked `[E, K]`/`[E, N]`. The device loader only uploads these and turns offsets into addresses. |
| `plan.zig` | `py/weights.py` `Loader.load` | A rank's whole weight plan: parts, layouts, byte totals, gate/up width mismatches (x3gm cannot fuse those), and coverage, so a pack with an unplanned name is refused. |

- **Tests:**
  - the release `config.json`
  - a synthetic pack, named the way the real ones are, on both ranks
  - all 43 blocks of the q28 pack: routed plus shared trellis bytes per rank equal the Python loader's listing (`fixtures/q28-widths.txt`, generated by `tools/zig/dsv41_widths_fixture.py` from `q28-expert.json`)
  - real header dumps through `TF_DSV41_HEADERS`. Layers 2 and 20 of the v2 pack plan on both ranks; layer 2's down projection is ragged with K2 4/6/8 = 204/171/9.

### The M1 gate, as built

- **Capture** (`tools/zig/dsv41_m1_capture.py`, run in the e1e2 runtime on one sm_120 GPU).
  - The Python engine runs blocks 1–3 and 20–24 at rank 0's TP=2 shapes. The peer's half of every exchange is zero.
  - A 1,536-token prompt fills the KV state past the index top-512. Then decode windows of 1, 3 and 16 rows are recorded.
  - Every Triton launch and every extension call is logged with each buffer's bytes before and after. Output:
    - `ops.jsonl`
    - `blobs/` (content-addressed)
    - the weights by the loader's names, with sha256 (`weights.json`)
    - the launched cubins (`aot/aot.json`)
- **Replay** (`tf-dsv41-m1 replay PACK CAPTURE OUT`, files `m1_main.zig`, `m1_replay.zig`, `m1_triton.zig`, `named.zig`, `load.zig`).
  1. Our loader builds the captured blocks from the pack and checks every weight's digest against the capture.
  2. It re-issues every launch on its own buffers:
     - Triton cubins by hash, with global scratch where a kernel needs it.
     - Extension calls through `dsv41_kernels.replay.call`.
     - Weight arguments are bound to our loader's tensors by name, and pointer tables are relocated to our addresses.
  3. It compares every buffer:
     - **Alone:** each op starts from the captured inputs.
     - **Chained:** our own outputs feed the next op wherever nothing ran between them.
  - Exit 0 means every weight is equal and every op is equal alone. Ops without a binding are listed as skipped.
- **Pod job.** The quant repo's `runpod/session.sh zigm1 --gpu pro6000` (staged by `runpod/stage-zigm1.sh`) runs, in one pod:
  - the kernel agent's `zig/tests/cuda/deepseek_v41/job.sh`
  - the capture
  - the replay

  Launch it with `DSQ_AGENT=dsq-zig-arch-1 DSQ_PURPOSE="M1 run N ..."`. The script writes the ledger row (section 0).
- **Block forward** (`tf-dsv41-m1 check CAPTURE` on the host; `tf-dsv41-m1 both PACK CAPTURE OUT` on the GPU, which runs the replay and then the forward on one load; files `block.zig`, `block_moe.zig`, `calls.zig`, `m1_check.zig`, `m1_forward.zig`). See M1's second half in section 5.
- **Results on the prod path (pod 8, `M1_KNOBS=prod`, 41ce3ed):**
  - **Replay: PASS.**
    - 1,254/1,254 launches bit-exact, alone and chained (4,389 inputs from our own outputs).
    - 72 L2 prefetch launches skipped.
    - Weights 535/535.
  - **Block forward:**
    - 789 calls; all 717 launched calls are bit-equal on our own buffers. 0 differ, 0 failed; the 72 prefetches are skipped.
    - Glue re-seeds are exactly the predicted roles: `w.recv.attn` and `w.recv.moe` (18 each), and `s.L2.carry` (2).
    - Our run-time forms equal Python's: 103 of 103 we build. That covers upstream strips (Engram `wkv`, head), the mHC bf16 weights, and the expert `suh`/`svh` stacks and K2 tables.
    - The 78 dense3 groups were compared as plain strips, but dense3 reads the lanes layout.
    - `prepare.lanes` (03f31dd) builds the lanes form. It equals `tests/dsv41_dense3_emu.py`'s reference at K2 4/6/8/10/12/16. The GPU compare of it rides on the next pod that runs the M1 job.
  - **Kernel agent's job, in the same pod:**
    - bits 13/13;
    - kvsplit 3/3;
    - SASS 182/182 equal to the Python build;
    - symbols PASS.
    - The x3gm cases logged "v2 0 launches", so x3gm v2 itself was not exercised.
- **Pod 9** (bf2340b, `M1_KNOBS=prod`, 24 min, $0.84; ledger row by `session.sh`):
  - **Replay:** PASS, 1,254/1,254 bit-exact alone and chained.
  - **Block forward:** PASS, 717/717 launched calls bit-equal; glue re-seeds as before.
  - **Prepared forms:** 181/181 equal to Python's, including the 78 dense3 groups through `prepare.lanes`. The open item from pod 8 is closed.
  - **Kernel job:**
    - on the decode1 78b703d twin, prebuilt for sm_120 in the e1e2 image and shipped as `r1c.tar.gz`: bits 13/13, kvsplit 3/3, SASS equal;
    - x3gm v2 ran ("v2 4 launches" every x3gm case);
    - the twin's suites ran under pytest (367 passed, 1 twin-side failure), but their recorded calls did not reach the replay (no `capture.json`). This is with dsq-zig-kern-1.
- **Pod 10** (M2a's first run, 40519d5, 35 min, $1.22): no M2a verdict yet.
  - The Python reference over layers 0–24 did not finish loading plus prefill within its 22 minutes. Python loads about 40 s a block from the volume. M2a now runs layers 0–8 (6cef64a): SWA + Engram (layer 1), the ratio-2 KV and index sources 2 and 8 with their carries, and the reuse layers between them.
  - `glue.cu`'s first GPU run: every glue kernel equal to torch except `embed_send` (2 of 1,792 words). That is with dsq-zig-kern-1; the next M2a pod waits for the fix.
  - `session.sh` was replaced while its watcher ran (quant 57e710d), and bash re-read it mid-loop. The port lead fetched the results, deleted the pod and closed the ledger row by hand. Don't edit `session.sh` while a session runs.
- **Pod 11** (2c500a5, 24 min, $0.85): **M2a PASS.**
  - Our whole forward on one GPU at rank 0 of TP=2 with the zero peer, over layers 0–8 and the head: `block.zig`'s program run by `run.zig`, with `forward.zig`'s glue on `glue.cu` (`embed_send` fixed), and Triton by name through the AOT set.
  - It starts from the Python engine's own prefill state at 1,536 positions.
  - Every block's streams (9/9) and every logit are bit-equal to Python in windows of 1, 3 and 16 rows (64,640 / 193,920 / 1,034,240 logits).
  - Load: our loader 19.4 GB, plus 206 weight forms built, 284 s. Python's reference is 276 s to load and 474 s to run.
  - Kernel job: the twin suites' recording works (302 calls, 141 replayed bit-exact). Then `tf-dsv41-test` panicked on an empty tensor's negative offset in its replay (dsq-zig-kern-1), so the oracle cases, glue included, did not run this time.
- **The next pod's kernel job** (stage from ≥ ea3e600, rebuild `tf-dsv41-test`; fatbins and `job.sh` unchanged): `replay.zig` now nulls empty or negative-offset tensors. Its first GPU checks are:
  - all 302 recorded replays (pod 11 got 141 bit-exact before the panic: the first GPU evidence for the R1/R1c kernels through Zig);
  - the 14 oracle cases;
  - `embed_send`'s NaN edge.
- **M2 next:**
  - **M2b**, as built (a413ce3), is `tf-dsv41-m1 m2b`: one process per rank, run in parallel by `m2b_ranks.sh`.
    - **Setup:** each rank opens the `tp` runtime's NCCL session and loads its weights for layers 0–27 plus the head. Both start from the Python engine's state after its own 1,536-token prompt.
    - **Run:** 64 greedy tokens, one per window.
    - **Greedy,** as Python's `forward.greedy`: each rank's max with the lowest column, then the f64 (value, id) pairs all-gathered. The higher value wins; on ties, the lower rank.
    - **Compared:** every token, every step's logits digest per rank, and the first window's block streams per rank.
    - **Reference:** `tools/zig/dsv41_m2b_ref.py`, the Python engine at TP=2 over `tensorfold.cuda.comm` NCCL.
    - **Expert top-p is off** (`Options.expert_topp = null`). With no prune, the router's tail folds the kit rounding and grouping itself, as for DSpark.
    - **Why layers 0–27:** the whole backbone needs about 110 GB a rank, and a PRO 6000 has 96 GB. 0–27 is about 69 GB a rank and includes every layer kind: SWA, both Engram positions, ratio-2 and ratio-1 KV sources, the candidate source, Reindex, and reuse. The whole model is a Spark run (M2c).
    - **Pod job (`M1_JOB=m2b`, 2 GPUs):**
      - those shards copied to local NVMe in parallel (`Pack.open` now skips index shards the folder lacks);
      - the prod twin's extensions prebuilt for sm_120 (`e1e2x.tar.gz`; pods 10–11 JIT-built them each time);
      - the reference, both ranks, then the kernel job.
  - **M2b PASS** (pod 13, 449e857, 2x PRO 6000, 5 min, $0.40):
    - Our forward at real TP=2 matched the Python engine's TP=2 from its prompt state at 1,536: two processes over NCCL through the `tp` runtime, layers 0–24 plus the head, expert top-p off.
    - **Each rank:** 64/64 greedy tokens, 64/64 per-step logits digests (bit-equal), and 25/25 first-window block streams.
    - **Times:** our load 99 s a rank (57.7 GB from local NVMe); decode of 64 tokens 0.5 s. Python: load 56 s, run 17 s.
    - **Pack copy:** 115 GB of shards in 23 s. The volume's host cache was warm from pod 12, so a cold copy can take far longer.
    - **The kernel job in the same pod:** 316 cases, 0 failed. That is all 302 recorded twin-suite replays (R1/R1c kernels through Zig) and the oracle cases, `glue.cu` with `embed_send`'s NaN edge included.
  - **Pod 12** (the first M2b attempt, 60 min, $4.21) **produced nothing.**
    - The M2b mode's step limits summed past the hard max, and results reached the volume only at the end.
    - Fixes (449e857): the job streams `$RES` to `dsq/results/live/<sid>/` every 30 s and at exit; M2b's steps fit the budget (the copy gets 10 min, else loads read the volume); the copy logs its progress.
    - Pod 13 also showed both Python ranks waiting on one extension's build lock at start, harmless with the prebuilt dir. Pod 12's long stall is unexplained, since its logs are lost.

**M2c plan: prod top-p, then the Spark run.**
- **M2c-1 PASS** (pod 14, 60c0af7, 2x PRO 6000, 6 min, $0.44): the M2b gate with prod's expert top-p 0.85 (`M2B_TOPP=0.85`, Python `TF_DSV41_EXPERT_TOPP=0.85`).
  - Both ranks: tokens 64/64, logits 64/64, first-window blocks 25/25.
  - Kernel job: 316/0. The experts SASS line is clean (32/32: the build now ships the exact fatbin dir).
  - The x3gm SASS line differs (119 of 135, 5 not in Python) against the decode1 78b703d twin. That twin's `x3gm.cu` predates v2, which added `k2e` to `Args` and the `gm2` kernels.
    - dsq-zig-kern-1 checked ours against the v2 source (8474f31) offline: 135/135 SASS-equal for sm_120, the same for sm_121.
    - Pods 9–13 showed 130/130 only because they shipped stale pre-v2 fatbins.
    - With the 78b703d twin, read this line as expected. The next twin rebuilt for any reason comes from 8474f31, and then the line is clean.
- **The M2c-2 kit** is `~/Documents/Projects/spark-work/zig-m2c2-kit` (tree ffba58d; sm_121 fatbins from dsq-zig-kern-1's 07935cc, SASS-equal to Spark prod's 8474f31 build). It is for the user's morning run: `RUNBOOK.md`.
- **M2c-1, prod expert top-p on the same gate.** The prune path (`_prune`, `group`, the kit-weights glue) is already in `block.zig` and bit-checked by M1 on the prod captures.
  - The run: `--env M1_JOB=m2b --env M2B_TOPP=0.85` on 2x PRO 6000. It is one environment flag, passed to Python's reference and to our ranks.
  - Expected: about 6 min, about $0.45. The gate is the same (tokens, logits, blocks).
- **M2c-2, the whole model on the two Sparks** (needs a Spark window; prod runs there now):
  - **Build first, all offline:**
    - The aarch64 `tf-dsv41-m1` with sm_121 fatbins (dsq-zig-kern-1 builds them).
    - The reference: Python's prod engine at TP=2 on the Sparks dumping its state and greedy tokens (`dsv41_m2b_ref.py` over all 40 layers, `--layers 0-39`).
    - Our ranks over the hybrid collective (`TF_COMM_BACKEND=roce`). The Spark window measured RoCE's decode window at 767 µs against NCCL's 1,804 µs.
    - Prefill's all-gathers on the same path.
  - **In the window:**
    1. Greedy tokens identical to Python over long runs (thousands of tokens, several prompts), top-p 0.85.
    2. Per-window time against Python prod. Eager, with no graphs, is informational; R4 graphs come before the speed gate.
  - **Kit:** like `zig-tp-spark-kit` (binaries, `SHA256SUMS`, one script per host), with the Spark window's runbook row.
- **The M2c-2 kit, as built** (`tools/zig/dsv41_m2c2/`: `run_host.sh`, `build-kit.sh`, `RUNBOOK.md`):
  - **Per host:**
    - The Python reference: prod's engine tree 8474f31, in prod's image and cache volume, one rank per Spark (`dsv41_m2b_ref.py --rank / --master`), all 40 layers, top-p 0.85, M1's prod kernel knobs (R1 off; it is bit-identical by design). It records the tokens, logits digests and first-window blocks, plus each host's AOT set and rope tables.
    - Then our rank: `tf-dsv41-m1 m2b` over the hybrid collective (`TF_COMM_BACKEND=roce`, `TF_TP_KERNEL_IMAGE` = `tp_mailbox.fatbin` built by the e1e2 image's nvcc for sm_120 / sm_121, `TF_TP_ROCE_FALLBACK=error`).
  - **Prod on the Sparks:** stopped with `serve.sh stop` and restored with `scripts/v2-rollback.sh perf1`, about 45–60 min for one seed.
  - **Built for GB10:**
    - `Pack.read` drops the page cache of what it read (`POSIX_FADV_DONTNEED`: the page cache is GPU memory there);
    - the kit drops caches between phases;
    - the aarch64 build is cross-compiled on the dev box (1 min);
    - the sm_121 fatbins are dsq-zig-kern-1's.
  - **Optional M3 phase (`LANES=1`, about +12 min):** the same decode through lanes over the GPU target, below. It compares tokens only.
  - **Prefill over RoCE: design only.** Zig has no prefill yet; the run starts from Python's prompt state. When it lands, its all-gathers use the same `Collective` as decode: the hybrid sends exchanges up to `mailbox_max_bytes` (256 KiB by default; `TF_TP_MAILBOX_MAX_KB`, up to 16 MB) over RoCE and larger ones over NCCL. A 16-row prefill window's exchanges fit the mailbox. Python's 2,048-row prefill chunks would need the threshold raised or NCCL.
  - **DSpark in the kit: weights only.** The plan and loader carry the DSpark heads (`named.dspark`: `main_proj`, the norms, Markov `w1` / `w2`, `conf`) and blocks 40–42. Capture set P (layers 36–39 plus the drafter's phases i0 / d0 / i1 / d1) is ready in `dsv41_m1_capture.py`. The GPU pass is not written: it waits for a P capture on a pod, as `block.zig` waited for M1's captures. Drafts never change the emitted tokens (verification is exact), so the Spark run's token gate does not depend on it.
  - **Not in M2c-2:** the DSpark pass, our own prefill, graphs, timing as a gate.
- **Pod 15** (760bff9, 2x PRO 6000, 17 min, $1.22):
  - **The 2,048-row prefill capture replays bit-exact.** Set B (layers 20–24) on prod's knobs with a 2,048-row prefill segment (`M1_PREFILL_ROWS=2048`: prod's PREFILL_CHUNK): 713/713 launches equal alone and chained, the 320 prefill launches included. The capture's small files are kept at `dsv41-zig-ref/cap15`; they are the prefill emitter's reference. The step read FAIL only for 8 `dspark.*` names that the failed set P had registered.
  - **Set P failed in Python** (`KeyError: 20`): layers 36–39 read layer 20's KV pool and candidate blocks. Set P now loads (20, 36–39).
  - **The M2b regression and both lanes gates stopped at `StateRole`.** The launch omitted `M1_KNOBS=prod` (pod 14 had it), so Python's state used the code-default row formats (e.g. `INDEX_KV` not fp8) and our plan refused them. That is a launch error, not a code one; pod 16 reruns with the knob.
  - **x3gm in the capture is v1.** The pod's Python tree predates v2 (`TF_DSV41_GM_V2=1` was ignored); v2 is bit-identical, and dsq-zig-kern-1's oracle covers gm2 at these shapes (6bb4634).
- **Zig prefill, the exchange decision.** Prefill runs Python's chunks: 2,048 rows (`TF_DSV41_PREFILL_CHUNK`), the last chunk ragged. The state after the prompt must be Python's bit for bit, and Python's prefill kernels run per segment (row blocks of 128, `_gemm` BM 128), so the chunking is part of the contract. The adaptive 1,024 / 512 chunks are prod's batcher policy and belong to M4's scheduler, not the forward.
  - A chunk's exchange is n × hidden × 2 B a rank: 2,048 × 5,120 × 2 = 20 MiB. That is over the mailbox's 16 MB ceiling (`TF_TP_MAILBOX_MAX_KB` max 16,384), so prefill exchanges go through NCCL in the hybrid Collective.
  - Raising the mailbox to fit a smaller chunk (1,600 rows = 16 MB) is not worth it. At 16–20 MiB the transfer (about 1 ms at CX7 rates) dwarfs the mailbox's latency gain (8.9 µs vs 21 µs an exchange on the Sparks, under 2 %). The mailbox slots would also hold 16 MB of registered memory a direction, which on GB10 is GPU memory.
  - The default stays at 256 KiB: decode windows up to 25 rows (25 × 10 KiB) use RoCE. For shared lanes rounds up to 64 rows, set `TF_TP_MAILBOX_MAX_KB=640` so they stay on RoCE too (64 × 5,120 × 2 B = 640 KiB).
- **Pod 16** (6bb4634, 2x PRO 6000, 14 min, $1.02; prod knobs):
  - **Set P (with layer 20) replays bit-exact:** 176/176 launches alone and chained, weights 469/469, the DSpark heads included. The small files are in `dsv41-zig-ref/cap16`.
  - **M2b (M2c-1 regression):** PASS on both ranks, 64/64 tokens and logits, 25/25 blocks.
  - **M2b with `TF_DSV41_GRAPHS=1`:** PASS, but no graph ran. The traced step 0 cached a traced program that `graphs.audit` refused, so every window ran eager. dsq-zig-kern-1 has fixed this; it is not graphs evidence.
  - **Lanes mode 1 stopped at `NotAChain`.** Lanes' undrafted step (`queue`) sends the next one-row window without a keep. The iface contract says an unkept chain window keeps every row, and GpuTarget refused. Fixed in 288a28e.
  - **Lanes mode 2 stopped at a missing Triton variant** (`_prune` with R > 1). The reference had only run one-row windows, and Triton specializes a 1. The reference now adds 2–6-row windows after its recorded steps (2c61b24).
- **Pod 17** (fcd1ea5, 19 min, $1.34):
  - **Graphs evidence (R4):** M2b with `TF_DSV41_GRAPHS=1` passes on both ranks (64/64 tokens and logits) with 1 capture and 62 replays.
  - **The M1 step stopped at load** (`DuplicateWeight`): sets P and B both hold layer 20. The loader now loads a shared block once.
  - **Lanes modes 1 and 2 diverged at tokens 2 and 6.** `Forward.window` borrowed the caller's ids, and `keep()` reads them for Engram's tail. Lanes hands a stack row (queue) or a scratch arena that is reset before the keep (verify). The forward now owns a copy.
  - **Mode 3 stopped at `MissingWeight`.** A DSpark block stores its routed experts as one uniform stack (`L40.moe.w1.trellis`), and forms.zig only knew the ragged `.data` form. The expert tables now take the stack's base, size-checked against the layout.
- **Pod 18** (ff34389, 21 min, $1.49):
  - **M1 replay of P + B(2048) passes:** 693/693 weights, every launch equal. The GPU forward of the decode windows is 131/131 equal per window (14 L2-prefetch launches skipped).
  - **The prefill segment stopped at a structure check.** This capture gave Python's MoE exchange new storage each layer (cap15 had reused one). block_prefill now layer-scopes it as it already did the attention's; cap15 and cap18 both host-check 320/320.
  - **Lanes mode 1 PASS:** 65/65 tokens equal, drafts off.
  - **Lanes mode 2 PASS:** 65/65 tokens equal, 79 drafted, 47 accepted. Multi-row verify windows, partial keeps and rejects all ran through the GPU target and the leader/follower plan ops at TP=2.
  - **Mode 3 stopped at the pass's own guard (`NotAPrefix`).** Lanes trims the prompt to the window itself and hands the taps' row indices (1408..1535), so the pass now ingests a contiguous run from any first row.
- **Pod 19** (a0cf72d, 15 min, $1.10), all PASS:
  - **M1 on fresh captures P + B(2048):** the replay is 889/889 launches alone and chained, weights 693/693. The **GPU forward of our emitters is bit-equal to Python**: decode windows 131/131 each, the **2,048-row prefill segment 320/320**, the **DSpark ingest 15/15 + 15/15 and passes 71/71 + 71/71**, and prepared forms 182/182. The torch glue is re-seeded from the capture.
  - **M2b regression:** 64/64 tokens and logits on both ranks.
  - **Lanes mode 3 (the GPU DSpark pass at TP=2):** 65/65 tokens equal to the serial reference, 68 drafted. Acceptance is 0, as expected here: on the 0–24 prefix the taps are zeros, not layers 37–39's. Acceptance with real taps needs the whole model, i.e. the Sparks (M2c-2 with LANES) or a bigger GPU.
  - **The round's pods 15–19 cost $6.17** against an $8 cap.
- **R1 in block.zig** (a66008b, 8b72239): `Options.r1`, written from decode1 78b703d's source, since there was no R1 capture to write it against. The changes:
  - attention split 4 + q RoPE in the kernel (two trailing attn args; no q `_rope`);
  - mhc_cuda's tail spin + late flag, plus a `coef` launch after each mixing site;
  - the router's prune + kit in its tail (a prune list; no `_prune`, no kit_weights glue, no `group`);
  - down_combine to bf16 with a plain exchange;
  - q_norm folded into wq_b's / ix_wq_b's rot_in (norm list + eps), with kv_norm as `_rms` before the SWA store.
  The checks take R1 from a capture's env and M2b from the reference's. The pre-R1 output is unchanged: cap15 and cap18 0 mismatches, every pre-R1 GPU gate still passes.
- **Pod 20** (9253c7a, 18 min, $1.27): the batched pod.
  - **Regression all PASS:** M2b eager and with graphs (1 capture, 62 replays), lanes modes 1 / 2 / 3 (65/65 each).
  - **dsq-zig-kern-1's kernel job: 322/322 bit-exact**, including the **prefill glue kernels == torch** (top positions, candidate keys, visible counts, gm picks, width mask, fp64 casts, projection, ring copies) and the x3gm prefill cases.
  - **The R1 capture failed inside Python:** the r1c twin (78b703d) lacks 8474f31's FastLinear `norm` fix, which is Python-only; the R1 kernels are identical between the two. Set D also needed layer 8 (13–15's KV source). `dsv41-zig-ref/r1c-fix.tar.gz` is the twin with 8474f31's prefill_mm.py.
- **Pod 21** (81f0477, 15 min, $1.06): the R1 fix-up.
  - **The R1 capture ran out of the pod's disk:** four sets with 2,048-row prefill blobs beside the 135 GB local pack. `M1_PREFILL_ROWS=0` now captures decode windows only.
  - **The R1 reference ran on the patched twin. Our R1 decode at TP=2 differs from it from block 0 of the first window** (tokens 0/64). One cause found offline since: the prune list's `drop` must be `Et` (moe's `block.ex.count`), not the routed count (8b72239). The rest needs the R1 capture, call for call, as block.zig had M1's.
  - The pre-R1 M2b regression passes again.
- **M4 on the GPU** (81f0477, 45b1791), host-built:
  - `model.zig` loads one rank of the whole model for serving. It needs the pack + `TF_DSV41_ASSETS` (rope.bin, engram-host.bin, aot/).
  - `serve_engine.zig` is the native server's engine: lanes over GpuTarget + GpuPass in a LaneHost on rank 0, one slot. `TF_DSV41_TRACE_TOKENS` records each reply's prompt and ids.
  - The binary is `tensorfold-dsv41` (zig/build/cuda.zig, server.zig `withEngine`).
  - `tf-dsv41-m1 follow` runs the ranks > 0. `tf-dsv41-m1 generate` replays a trace through the same engine and requires the same ids ("HTTP tokens == CLI").
  - The Spark runbook is `tools/zig/dsv41_m4/` (run_host.sh, RUNBOOK.md). The whole model does not fit a PRO 6000, so its first run is a Spark window.
- **The night's pods:** runs 15–21 for M3 cost **$8.50**: 15–19 $6.17, 20–21 $2.33.
- **Pod 22, held** (b-hold 750vz54tgburg9, 42 min, $2.92): **R1 PASS.** The pod stayed up through the whole fix loop: capture, check locally, fix, re-check, push the binary, gate.
  - **The capture:** decode-only R1 (A, B, D with layer 8, P) on `r1c-fix.tar.gz` (the r1c twin with 8474f31's FastLinear `norm` fix; the R1 kernels are identical).
  - **The blind emitter had one mismatch class:** attn_cuda's ticket is 1,024 words under the split kernel (36 calls). The R1 drafter takes R1 only partly: the attention gets split 1 and rope 0 and keeps its own q `_rope` and both `_rms`; the mHC run gets the tail args with a `coef` after its first site; the final mHC gets the tail args; the router gets the prune list. Fixed in 219abeb.
  - **Host check on the R1 capture:** decode windows 0 mismatches, drafter (dcheck) 0 mismatches; cap15 / cap18 unchanged.
  - **The R1 M2b gate at TP=2 against Python R1:** both ranks 64/64 tokens, 64/64 logits, 25/25 first-window blocks, **graphs off and on** (1 capture, 62 replays).
  - **Held-pod notes:** an ssh session lacks the container's `LIBRARY_PATH` (Triton's launcher link fails). Load `/proc/1/environ` first, and call job.sh directly, since `run.sh` runs once a pod.
- **The Spark kit** (`~/Documents/Projects/spark-work/zig-m2c2-kit`, tree b4723d5) **with R1 on:**
  - aarch64 `tf-dsv41-m1` and `tensorfold-dsv41`, built against dsq-zig-kern-1's sm_121 fatbins from 81f0477 (26 modules; no kernel source changed since), from a clean archive of the tree;
  - `RUNBOOK.md` is the one window: M2c-2 (with R1) → DSpark's AOT set on each host → the M4 server and requests → the CLI replay (HTTP == CLI) and DSpark's acceptance → restore with `scripts/v2-rollback.sh perf1`. **About 1 h 45 min of downtime, up to 2 h 15 min.**
- **The night's pods:** runs 15–22 cost **$11.42** in all: 15–19 $6.17, 20–21 $2.33, 22 $2.92.
- **Zig prefill, wired** (1332aa6 by a helper agent; 2671c15, 982f12f), offline. It has not run on a GPU.
  - **The runner:** `forward.prefill(ids)` (forward_prefill.zig) runs block_prefill's segments, 2,048 rows each (Python's PREFILL_CHUNK, the last ragged).
    - The exchanges (20 MiB a rank) go through NCCL in the hybrid, as recorded above.
    - All 21 glue steps run on dsq-zig-kern-1's prefill glue kernels (gated == torch, pod 20) or as plain async copies / memsets.
    - **Each segment is committed as Python's `keep`:** the ratio-2 carry from its fp32 projection's last row, the Engram tail, the position. A segment of 16 rows or fewer is block_prefill's short segment (pod 25 below). The head runs on the last segment only; `f.last_rows` tells greedy how many rows it left.
    - **Leader / follower:** the prompt is sent in frames (`op_prefill`); the followers run the same segments.
    - **Tables filled on the first prefill:** the bf16 Hadamard, and the x3gm pointer tables (forms' `tp_*`). `m2a.widths` now fills `Widths.gm`.
  - **M4:** GpuTarget and the served model prefill through it (`TF_DSV41_OWN_PREFILL=0`: back to 16-row decode windows).
  - **The gate is `M2B_PREFILL=1`** on the M2b driver. The slot starts empty, our prefill runs the reference's prompt, then every state role must equal Python's dump byte for byte: SWA rings, compressed pools, index keys, ratio-2 carries. The position, Engram tail and first token must match too, and then the decode gate runs as usual.
    - **Pod job:** `M2B_PREFILL_GATE=1`, with `M2B_PROMPT=2600` to cross a segment boundary (prepared, not launched).
    - **Spark kit:** `PREFILL=1` on the M2c-2 step (about +10 min); the runbook falls M4 back to `OWN_PREFILL=0` unless it passed. Kit tree 982f12f.
  - **Known limits:**
    - Prompts whose full-mode layers see more than 4,096 keys need the stream top-k, which block_prefill does not port (`error.Unsupported`).
    - Greedy after a full 2,048-row segment downloads every row's logits; a last-row greedy is a later speed fix.
    - Segment shapes other than 2,048 and the SWA, Engram and ratio-2 layers' prefill have only been host-checked; the gate covers them.
- **Pod 23, held** (63f16d7 + the reference fix, 26 min, $1.81): **the Zig prefill gate passes for full and ragged segments.**
  - **A 2,600-row prompt** (one 2,048-row segment + one of 552) from an empty slot, at TP=2 on layers 0–24 with prod's knobs: **65/65 state roles equal** to Python's dump (SWA rings, compressed pools, index keys, ratio-2 carries), the same position, Engram tail and first token. Decode from it then gives 64/64 tokens and logits and 25/25 blocks on both ranks.
  - The prompt prefilled in 0.47–0.76 s.
  - **A 2,060-row prompt** (2,048 + a 12-row tail) **fails:** 4/65 roles equal, first token equal. A tail of 16 rows or fewer runs as decode windows, but Python's prefill path for such a segment is a mix: it takes some decode kernels (mhc_cuda, dtopk, the narrow router) and keeps its prefill projections. The fix is a ≤16-row prefill segment in block_prefill, written against a capture with `M1_PREFILL_ROWS=12`.
  - **The first run failed with `MissingTritonVariant`:** the M2b reference installed its AOT recorder after the prompt, so the prefill kernels were not in the set. It now records from before the prompt.
  - **Held-pod note:** the hold job marks the pod's SID as run, so `run.sh zigm1` needs its own `DSQ_SID`. And never `pkill -f` a pattern that appears in the ssh command line itself.
  - **The kit's M4 now prefills with Zig prefill by default.** Prompts with a 1–16-row last segment still use decode windows (the same model, not Python's bits for those rows).
- **Pod 24, held** (3a8a672, 28 min, $1.95, partial): **the captures for the ≤16-row segment, no fix or gate.** A host session restart took the pod's middle; the hard max ended it before the emitter could be written.
  - **The captures, in `dsv41-zig-ref/`:**
    - `cap24-12`: a 12-row prefill segment for sets A, B and D (`M1_PREFILL_ROWS=12`), 1,848 ops;
    - `cap24-1` and `cap24-16`: set B at 1 and 16 rows, 725 ops each.
    - Each includes decode windows of 1, 3 and 16 rows, plus weights, jit and aot.
  - **Python's ≤16-row prefill segment, as captured** (the same kernel mix at 1, 12 and 16 rows):
    - prefill side: `rot_in` / `unpack` / `_gemm` projections, x3gm experts, `_fused` attention at the segment's rows;
    - decode side: `mhc_cuda.run` at the segment's rows, `attn_cuda.topk`, `_rms2`, `router_gemv`, and the L2 prefetch (`l2pf.segments`, `l2pace.paced`).
  - **Not done on that pod:** the block_prefill path for these segments, and the gates (pod 25 below).
- **Short segments (6587de9), offline:** block_prefill takes segments of 1–16 rows as Python does.
  - **Kept from the prefill path:** the projections (`rot_in` / `unpack` / `_gemm`), the indexer's `_plain` / `_scores` with the fp32 head weights, the fused attention, x3gm and the shared expert's GEMMs.
  - **Taken from the decode window:**
    - mhc_cuda for every site, post-only and the final norm (no `_finish_k`; R1's tail and coefficient launch as in decode);
    - the L2 prefetch at the o, f and x sites;
    - attn_cuda's top-k job list (a Reindex layer scores over its candidate blocks with `GATHER`; no `_keys` or torch top-k);
    - the narrow router config `(rows_pad, 2, 0)`.
  - **Glue roles:** the exchanges land in `w.recv.*` (mhc_cuda's). At one row, the picks keep torch's `[1, topk]` view of the `[1, slots]` rows: `.contiguous()` does not copy a one-row tensor.
  - **R1:** the R1 twin's route always takes the prune list (empty in prefill). That now applies to every prefill segment, from the source; no R1 prefill capture exists.
  - **The runtime:** forward_prefill runs every segment through the program; the decode-window tail is gone.
  - **Host checks:** pod 24's captures, call by call: p12 for sets A / B / D 744/744, p1 290/290, p16 290/290. Regression: cap15 / cap18 p2048 320/320, cap22 (R1) clean, `dcheck` cap16 / 18 / 22 clean. cap6 / cap8 (pods 6 / 8, old knobs) fail as they did before.
  - **m1 check:** roles one of our glue steps touched are no longer flagged when their bytes equal what a freed storage held. x3gm's gate and down plans of a few picks often hold the same bytes.
- **Pod 25, held** (779a303, 16 min, $1.11): **the Zig prefill gate passes for every prompt length tested.**
  - **One reference load for five prompts** (`--prompts`, a fresh `Forward.new_slot` each; `job.sh` `M2B_PROMPTS`): 94 s in all. The 2,060- and 2,600-row first tokens equal pod 23's single-prompt references.
  - **The M2b decode gate** on 2,060's state: 64/64 tokens and logits, 25/25 blocks, both ranks.
  - **The prefill gate from an empty slot** (layers 0–24, TP=2, prod's knobs without R1). Each prompt gave 65/65 state roles equal, the same position, Engram tail and first token, then 64/64 tokens and logits, on both ranks:

    | prompt | segments | first token |
    |---|---|---|
    | 2,060 | 2,048 + 12 | 129275 |
    | 2,049 | 2,048 + 1 | 129275 |
    | 2,064 | 2,048 + 16 | 26039 |
    | 2,600 | 2,048 + 552 | 59756 |
    | 9 | 9 | 5744 |

  - Each gate took about 105 s, mostly the load. The prefill itself (a timing, not a benchmark) took 0.35–0.46 s on rank 0 and 2.5–2.6 s on rank 1 for the 2,048+ prompts; 0.65 / 0.08 s for the 9-token prompt. Rank 1's gap is not yet explained.
  - **The kit runbook's limit note is gone.** M4 prefills with the Zig prefill for every prompt length. The kit's `PREFILL=1` step runs under R1, the first prefill gate with it.
- **Next, in order:**
  1. **The Spark window** (the kit's RUNBOOK.md): M2c-2 with R1, M4, and DSpark's acceptance. Spark prod has run R1's knobs since 10-07 at -6.5 % a window with the same bits: attention split 4 + qrope, mHC spin / defer + coef, the router's prune, downCombine bf16, x3seg rot_in with the q_norm fold. The gate is a capture on prod-knobs.env plus r1c-knobs.env's R1 line on the 8474f31 twin. It goes ahead of new glue fusions, which are estimated under 3 %.
  2. **The prefill runtime:** the 15 torch-glue steps as kernels, then `Forward.prefill` over 2,048-row chunks (NCCL exchanges).
  3. **The M2c-2 kit rebuilt with the DSpark pass**, so the Spark run measures acceptance with real taps.
- **The DSpark GPU pass** (`dspark_emit.zig`, `dspark_gpu.zig`):
  - **Emitter:** ingest (main_proj with split K, main_norm, each block's wkv / kv_norm / ring store) and the pass. The pass is:
    - the block ids embedded;
    - the first site on mhc_cuda, then the Triton `_site` + `_finish_k` boundaries;
    - ring attention on attn_cuda, with Python's DraftInputs statics laid out byte for byte;
    - the 128-expert MoE (block_moe.moe);
    - the final norm with `dspark.norm`, then the head.
  - **Host gate (`tf-dsv41-m1 dcheck cap16`):** 172/172 calls equal (ingest 15 + 15, passes 71 + 71), 58 Triton variants 0 wrong, 0 miswired. The glue inputs are exactly the runtime's steps: statics, taps, the fp32 copy of q, embed, exchanges.
  - **Not emitted:** Python's `_keys` + torch top-k + `_chain`. The host does them instead: each rank's best 64 columns (`dspark.candidates`), the ranks' lists gathered over the Collective in rank order, then `dspark.chain` (chain_host's rule, golden-tested). Drafts never change a reply, so fp32 summation order is the only difference allowed against the `_chain` kernel.
  - **Runtime:** `GpuPass` implements `iface.Pass`. The leader sends `op_ds_ingest` / `op_ds_propose` and the followers run them (`Forward.drafter`), so the pass's exchanges pair up. A slot resumed from another engine's prompt state has no taps for that prompt, so its context restarts after it (Python's restore without DSpark rows). q's fp32 copy is done on the host until dsq-zig-kern-1's `widen` kernel lands.
  - **Gates on the next pod:**
    1. `M1_MODE=both` on a set P capture runs our emitted drafter launches on our buffers against Python's outputs, bit for bit.
    2. `M2B_LANES=3` runs lanes with the GPU pass at TP=2: drafted tokens must equal the serial reference, and the pass must draft. On the 0–24 prefix the taps are not the trained layers', so acceptance is not a measure there.
- **Zig prefill** (`block_prefill.zig`, by a helper agent; 5355e01):
  - **Emitter:** the fast-prefill launch sequence (mhc_pf, rot_in / unpack / `_gemm`, `_fused` attention, `_keys` / `_block_keys`, x3gm per K2 width, combine, the shared expert, the head over all rows).
  - **Host gate on cap15:** p2048 320/320 calls equal, 192 Triton variants 0 wrong, 0 miswired. Decode is unchanged (145/145 per window).
  - **Verified layer modes:** a full KV + candidate source (20), reuse (21–23), reindex (24).
  - **Emitted but not yet compared:** SWA layers 0–1, the Engram layers, ratio-2 sources 2 / 8 / 14, and row counts other than 2,048. A host smoke ran them, but there is no capture to compare against.
  - **Refused (`error.Unsupported`):** segments of 16 rows or fewer (they go through the decode kernels) and the streamed top-k for 4,096+ keys.
  - **What remains before Zig can prefill:** the forward's prefill glue, 15 steps that Python runs in torch between the launches: `topk`, `moe_sum`, `swiglu`, `counts`, `gm_picks`, `gm_plan`, `gm_ticket`, `stage_in` / `stage_out`, `ix_w`, `proj`, `proj_carry`, `cand_keys`, `block_pos`, `zeros`. Several are reductions whose bits depend on torch's order (top-k ties, sums), so they are kernel work for dsq-zig-kern-1, each gated against the capture's buffers. M1's GPU forward re-seeds them from the capture meanwhile, so the next pod gates every launch of the segment on the GPU (`M1_MODE=both` on a p2048 capture).
- **M3 status:**
  - **Built:**
    - `target.zig`: draft/iface.zig's `Target` over `forward.zig`. It covers one slot, chain verify windows with greedy choices (`forward.greedy`: Python's cross-rank rule), `keep` of a chain prefix, prefill as consecutive decode windows, and the taps.
    - `block.Options.taps`: the boundaries entering layers 37–39 write `w.taps` `[n, 3 × hidden]`, as `dspark.py`.
    - **Leader / follower** (`forward.zig`, commit ffba58d): with `Forward.link` set (the tp session's plan link), rank 0 sends each `window(ids)` (at most 128 ids), `greedy(n)` and `keep(k)` as a plan op before running it. The other ranks run `follow()`, which replays the ops until `stop`. The host engine (lanes) runs on rank 0 alone.
    - `m3.zig`: `generate()` runs `lanes.Engine` over `draft/lanes.zig` with the GPU target and a `NoDraft` pass (streams with `drafts = false`; the pass is never asked). `GpuTarget.resume_first` starts from a slot that already holds the prompt.
    - **The gate:** `M2B_LANES=1` on the M2b driver. Rank 0's lanes tokens must equal the reference's (prompt's pick + S steps); rank 1 follows. It is host-built only; the first run is the Spark kit's `LANES=1` or the next 2-GPU pod.
  - **Next:**
    1. **The DSpark pass on the GPU** (draft/iface.zig `Pass`):
       - `main_proj` + RMS norm of the taps;
       - each block's ring (`wkv_b` → `kv_norm` → rope → ratio-0 store, 128 positions);
       - the 3 blocks over [anchor, noise × (N−1)] with the non-causal attention metadata (the ring as the compressed list plus the block rows, `lo` = P, `hi` = P+N−1);
       - the head's candidates (`topk_keys`), gathered across ranks;
       - the Markov / confidence chain kernel (a `.cu` for dsq-zig-kern-1).
       Its gate is a capture of Python's drafting pass, drafts compared bit for bit.
    2. **Trees and sampling in the target** (`parents`, keyed `lanes.Sampling`).
    3. **Several slots** through `state.zig` and the session pool.
- **Ahead of M2c-2 (offline, then pods as needed):**
  - our own prefill (exact-tag kernels first), so a run does not start from Python's state dump;
  - CUDA graphs per row bucket (R4), with `run.zig`'s calls already fixed per window;
  - the RoCE / hybrid backend in `forward.zig`'s driver;
  - Engram rows through `EngramSource` (R3).
  - **The superseded "next pod" list:**

    - the GPU compare of `prepare.lanes` against Python's dense3 words (pod 8 only compared them as plain strips);
    - the kernel agent's x3gm v2 check (f97ffde). gm2, with the ticket on and off, is compared bit for bit against the fixture's `gm_kernel` outputs, so each x3gm case should log "v2 4 launches".
    - Stage from a tree at or after 6ff3472, with the fatbins rebuilt at 22:44 (R1c: SASS equal to decode1 78b703d).
    - The R1c router prune pass has no GPU check yet. The replay checks it only if the pod's Python twin is built from decode1 78b703d with `RG_PRUNE` on. The e1e2 image is not.
- **R1c and `block.zig`:**
  - `block.zig` follows the prod capture (prod.env), which has none of R1c.
  - When prod adopts R1c, a new prod capture becomes the spec, and the `block.zig` changes are:
    - attention with `split > 0` and `qrope` (un-rotated q; the q `_rope` launch goes);
    - the router's `prune` (`rg.pruneOf`; the `_prune` launch goes);
    - x3seg `rot_in` with `nw`/`eps` (the `q_norm` fold; bf16 rows).
  - `tf-dsv41-m1 check` on that capture gates the change before any pod.
- **`roce.cu` wait fix (bb0cff8, Python):** nothing to port. The Zig TP runtime's one flag wait (`fabric/tp/mailbox.cu` `wait_flag`, used by the mailbox and `tp_roce`) only times out when `now > t0`, so a `%globaltimer` read below the start can't end it early. The Zig port is what found the Python bug.


### Spark window, 2026-10-07: outcome

Prod was down 12:34-14:07 (93 min) and was restored on perf1 (`scripts/v2-rollback.sh perf1`: 17 x 23 -> 391,
canaries ok, a chat completion on :8000 answered). The kit was tree `3a8a672`, copied to both Sparks and checksum-verified.
M4 ran on two fixes rebuilt on that tree with the kit's sm_121 fatbins (81f0477). Logs are in the quant repo,
`results/zig-spark-20261007/` (`r0/`, `r1/`, `zig-m4-fixes.patch`).

| step | result |
| --- | --- |
| M2c-2, whole model, TP=2 over RoCE, R1, top-p 0.85, seed 4101 | **PASS** on both ranks: tokens 512/512, logits 512/512, first-window blocks 40/40 (reference 385 s, ours 517 s) |
| `PREFILL=1`: Zig prefill from an empty slot, 1,536 rows | **PASS** on both ranks: state roles 95/95 == Python, position, Engram tail, first token 223 == 223, then 512/512 tokens + logits |
| DSpark's Triton set (`dspark-aot`) | 4 variants added on each rank (dgx-01's merge by hand: the step's `rm -rf blobs` hit root-owned files) |
| M4 server, 8 prompts over HTTP | first run: every request failed (`CUDA_ERROR_INVALID_CONTEXT`); after fix 1, `MissingTritonVariant` `_keys` (Zig prefill); with `OWN_PREFILL=0`, request 1 ok, then `BadWindow`; after fix 2: **8/8 replies** (256 tokens each) |
| M4 CLI replay == HTTP | **FAIL, 6/8**: lines 3 (43-token prompt) and 6 (41) differ; 1, 2, 4, 5, 7 and 8 equal |
| DSpark acceptance, q28-v2, 8 chat prompts, thinking on | HTTP **1,398 / 2,103 = 0.665**, 3.18 tokens a round (644 rounds for 2,048 tokens); CLI replay 1,335 / 2,208 = 0.605 |

**Fixes (both general, not Spark-specific):**
1. **LaneHost runs the lanes on its own thread, and the CUDA context was never current there.** Every driver call on
   the served path failed. `LaneHost.on_thread` (core/lane_host.zig) is a hook that runs first on the engine thread;
   `Served` binds the model's context with it. The CLI `generate` goes through the same `Served`, so it had the same bug.
   M2b/M3 never saw it: they drive the forward from the main thread.
2. **`GpuTarget.release` reset the slot on rank 0 only.** The followers kept the finished reply's position and its
   last, never-kept window, so the next request's first window returned `BadWindow` on rank 1. `Forward.release()` now
   sends `op_release` on the plan link, and the follow loop resets its slot too. This is not on the branch: the
   M5 sessions work in progress rewrites `release` and uses op 8. It must carry the same broadcast; the patch is in
   `zig-m4-fixes.patch`.

**Open, from the window:**
- **The Zig prefill's `_keys` launches pass strides = NK**, while Python pads that buffer (every captured `_keys`
  variant has div16 strides). A prompt whose compressed-key count is not a multiple of 16 (every short chat prompt:
  NK 22 on a 45-row prompt) has no AOT variant. M2c-2's 1,536-row prompt avoids it. Pad like Python, then rerun M4
  with the Zig prefill. A one-off 45-row reference (`--prompt 45 --steps 4`) added 10 other variants (`_site` at R=45
  and more) to the AOT set. The kit's reference should capture an odd-length prompt as well.
- **CLI != HTTP on 2/8:** the draft counts differ between the two runs even where the tokens match (line 4: 207 vs 199
  drafted). So the verify windows differ in rows, and a near tie flips. DSpark's drafts must be a function of the
  request alone (reset its rings / proposer state per stream), or the gate should run `DRAFTS=0`. `generate` should
  print the first differing index.
- Speed, one slot, graphs off, prefill as 16-row windows: decode **46.7 tok/s** (2,048 tokens in 43.8 s), against
  prod's Python engine at 52.6 (prose T0) to 86.6 (code T0) tok/s (E1E2-SPARK-WINDOW section 9). Prefill: 375 prompt tokens in 2.17 s
  (173 tok/s on 41-60-token prompts). Not comparable to prod's 2,077 / 2,380 tok/s cold 8K / 32K. Graphs
  (`TF_DSV41_GRAPHS=1`) and the Zig prefill are the next measurements.

## 3. Python to Zig, component by component

Two things decide how each part is ported:

1. **Triton is the default in Python.** The Python engine's default hot path is mostly Triton. Its `.cu` files are opt-in twins, documented as bit-identical to the Triton kernels.
2. **Two ways to bring a kernel over:**
   - **AOT:** Triton-only kernels come over as captured cubins (`zig/src/cuda/aot.zig`, the Nemotron route), bit-identical by construction.
   - **CU:** where a `.cu` twin exists, it is copied verbatim and checked bit-equal to the Triton kernel by the kernel agent's fixture tool.

**Weights and model structure**

| Python (`py/`) | Zig (`zig/src/families/deepseek_v41/`) | State | Kernels |
|---|---|---|---|
| `config.py` | `config.zig` | done | — |
| `weights.py` (`Shards`, `Loader`, `X3*`) | `pack.zig`, `exl3.zig`, `names.zig`, `plan.zig` (host); `load.zig` (device upload: one trellis buffer per projection, suh/svh `[E,K]`/`[E,N]`, int64 address and int32 K2 tables per projection, the format the kernel agent's launches take) | host side and staging done; `load.zig` (uploads, address tables) is M1 | — |
| `fastboot.py` (prepared folder) | `prepared.zig`: one `data.bin` per rank, written from the planned device buffers, plus a manifest of offsets. No pickled tree: the plan is rebuilt from headers, which takes milliseconds. | M2 (boot time) | — |

**Forward pass**

| Python (`py/`) | Zig (`zig/src/families/deepseek_v41/`) | State | Kernels |
|---|---|---|---|
| `forward.py` (`_run`, `_layers`, `_finish`, `embed_rows`) | `forward.zig` (window), `block.zig` (one block's kernel order) | M1–M2 | the embedding row gather + `allsum` is a small `.cu` (new; torch `index_select`/`where`) |
| `mhc.py`, `mhc_dec.py` (Triton `_site`, `_finish_k`, `_site_dec`); `mhc_cuda.cu`, `mhc_pf.cu` | `mhc.zig` (boundary, site, `post_only`, final; coefficient scratch) | M1 | CU: `mhc_cuda.cu` (decode boundary), `mhc_pf.cu` (prefill site). Note: `exp` is the hardware approximation, so the bits match Triton only if the twin was proven equal. The kernel agent checks this first. |
| `blocks.attention`, `_out`, `_wo_b`, `fused_proj.py`, `rope_fused.py`, `rmsnorm.py` | `attention.zig` (projections, q/kv norms, RoPE, `wo_a`/`wo_b`) | M1 | CU: `x3seg.cu` (fused projection segments), upstream `exl3/linear.cu` + `decode.cuh`, `x3dn.cu`, `dense3.cu` (8-bit `ix_wk` fallback). AOT: `_rms`, `_rms2`, `_rms_row` (fp64 norms), `_rope`, `_plain` (`ix_wp`). The RoPE/YaRN tables are built on the host to Python's exact fp32 values; check them against a dump (risk R6). |
| `csa2/compress.py` (`_pool_norm`, `_kv_store`, `_index_k`), `csa2/fmt.py`, `stores.py` | `compress.zig` (ratio-2 carry, pooling, 584-byte records, index keys) | M1 | AOT: `_pool_norm`, `_kv_store`, `_index_k` |
| `csa2/index.py`, `dtopk.py`, `stream_topk.py`, `backend.select*`, `reindex`, `candidates` | `indexer.zig` (scores, top-512, candidate blocks at layer 20, reindex over 16,384 candidates) | M1 | AOT: `_scores`, `_keys`, `_block_keys`, `_dtopk`, `_stream`/`_offer`/`_compact`/`_kth`. CU: `topk_cuda.cu` (cluster radix select). The torch `topk`/`sort` fallback in `top_positions` is not used: the keys are unique int64, so any exact top-k gives the same set. |
| `csa2/attn.py` (`_chunks` + `_merge`), `attn_pf.py`; `attn_cuda.cu` | `attention.zig` (core: 512 selected rows plus the SWA window, sink, inverse RoPE) | M1 | CU: `attn_cuda.cu` (sm_90+, same chunk/merge order) or AOT `_chunks`/`_merge`. `attn_pf._fused` is fast-prefill only and has a different bit tag. |
| `engram*.py` (`engram_dec._fuse_dec`, `engram._dequant`), `engram_host.py` (hashing, shard rows), `engram_native.py`, `engram_gate.cu` | `engram.zig` (device fuse), `engram_host.zig` (n-gram hashing into 24 heads, 264-byte NVMe rows, shard reader; host code) | M1 (layers 1 and 14) | AOT: `_fuse_dec`, `_dequant`. `engram_gate.cu` is I/O orchestration: port it in M2 for overlap. |
| `router.py` (`_fused`, `_logits`, `_select`), `router_gemv.cu`, `moe_fused.cu`, `pick.kit_weights` | `moe.zig` (router, routed experts, shared expert, combine) | M1 | CU: `router_gemv.cu` or `moe_fused.cu` (router + group + rot_in + fp16 weights), once proven equal to `_fused`. Otherwise AOT `_fused`. Fast prefill uses `_logits`/`_select` (AOT). |
| `experts.py`, `exl3/experts.cu` + `experts_grouped.cuh`, `x3ld.cu`, `x3pf.cu`, `x3gm.cu`, `moe.py` | `moe.zig` | M1 (decode: grouped or `x3ld`); M2 (prefill: `x3gm`) | CU, all of it (mul1 codebook only, as in Python). `x3gm.plan` (argsort/bincount/searchsorted in torch) becomes a small `.cu` or host step; the kernel agent lists it. |
| `prefill_mm.py`, `exl3/prefill._gemm`, `pfdense.cuh`, `prefill_moe.py` | `prefill.zig` (2048-row segments, fast kernels) | M2 | AOT `_gemm` (unpack to fp16 + GEMM with Hadamard epilogue); CU `pfdense` (opt-in in Python, `PF_DENSE=fused` in prod) |
| `forward._finish`, `pick.py` (`kit_logits`, `top`), `slots.cand_gather`, `exact_sampling.choose` | `sampler.zig` (head, bf16 logits, per-rank top-k on int32 keys, candidate gather, keyed choice) | M2 | torch `topk` on keys: a `.cu` replacement in `zig/kernels/cuda/torch_ops` style (Nemotron already has `torch_topk.cu` and `torch_argmax.cu`; check that tie-breaking matches). |

**Drafting, rounds and memory**

| Python (`py/`) | Zig (`zig/src/families/deepseek_v41/`) | State | Kernels |
|---|---|---|---|
| `drafter.py`, `dspark.py` (`_chain`, keyed noise), `draft_head.py`, `spec.py`, `tree.py` | `drafter.zig` (DSpark: taps at layers 37–39, `main_proj`, context rings, 5-row blocks, Markov and confidence heads), wired as the lane core's draft hook | M3 | backbone kernels; AOT `_chain`, `keyed_uniform` |
| `lookup.py` (suffix index, l / d / s plan) | the lane core's own copy index (`core/copy_index.zig`) plus a DSpark proposer. Python's per-round planner policy becomes the proposer's choice. | M3 | — |
| `depth.py`, `calib.py` (E(k) − λ·verify[k]) | `core/draft_depth.zig` with DeepSeek's measured costs table (Nemotron uses `--costs`) | M3 | — |
| `pool.py`, `slots.py`, `state.py`, `rowmode.py`, `rowtab.py`, GLM `kvpool.py` | `state.zig` (paged pool: 256-token pages, `comp.*` 584 B rows, `index_k` fp8 132 B; per-slot SWA rings, ratio-2 carries, Engram lookback, DSpark taps), `pool.zig` (deterministic `take`, holders, `need_pages`) | M2 | `.cu` page copy (or `cuMemcpy`) |
| `rowgraphs.py`, `graphs.py` | `graphs.zig` (`cuda/graph.zig` capture keyed by row bucket × context bucket, per-row table `[6, rows]`, padding rows that write nothing, warm-up at buckets 1–8) | M2 (decode speed) | — |
| `batch.py`, `plan.py`, `rounds.py`, `planlink.py`, `failfast.py` | Mostly replaced by the lane core and `lane_host.zig` on rank 0. `tp.zig` holds the round plan (encode/decode, the same ordering of commits, finishes, admits, pieces and windows), the rank-1 follower loop, the TCP fate channel and the plan link. | M2 (follower), M4 (admission and fairness) | — |
| `memory.py` (priced floor, `Adapt`), `budget.py` (lossy, skip), `hostmem.py` | `memory.zig` (load check, floor against both ranks, prefill price, adaptive rows 2048/1024/512) | M4 | — |

**Server and long context**

| Python (`py/`) | Zig (`zig/src/families/deepseek_v41/`) | State | Kernels |
|---|---|---|---|
| `app.py`, GLM `server.py`, `encoding.py`, `dsml.py`, `structured.py` | Server: `zig/src/native/cuda.zig` (new: the CUDA engines a native server opens), with `engines.zig` picking it on Linux. `encoding.zig`: DeepSeek's `encode_messages`, native, since it is not Jinja. `dsml.zig`: the lenient DSML tool-call parser and streamer, plus `reasoning_effort` mapping and reasoning memory. | M4 | — |
| `sessions.py`, `sessdisk.py`, `residency.py` (sessres) | `sessions.zig` (prefix store on the 16-token grid, page sharing, LRU, spill), `sessdisk.zig` (NVMe format with 4 KiB-aligned segments and SHA-256 per segment, O_DIRECT, streamed through staging buffers, async writer and reader) | M5 | — |
| `kvsplit.py` (kvsplit) | `kvsplit.zig` (`comp.*` page k on rank k % 2, residue allocator, ctable plus discard page, the dense gather for decode and union exchange for prefill) | M5 | `.cu` gather/remap (from `blocks._attend_blocks`); NCCL all-gather |

**Python modules outside `py/` that the engine needs.** These live in the GLM family or the shared engine and must be ported as well: `glm5_next/spark/kvpool.py`, `server.py`, `roce.py` (one-shot RoCE all-gather and host mailbox), `tensorfold.cuda.comm` (`NCCL`, `fast_gather`), `tensorfold.engine.exact_sampling`.

## 4. TP=2 on two GB10s

**Processes.** One process per Spark, as today: `tensorfold serve /model --tp 2 --rank R --master HEAD_IP`, with rank 1 started first. The two nodes are linked by ConnectX-7 (192.168.100.10/11, `NCCL_SOCKET_IFNAME=enp1s0f1np1`).

**Bootstrap.** New in `tp.zig`:
1. Rank 0 calls `ncclGetUniqueId` and sends the 128-byte id over a TCP side channel on master port + 7 (Python's `planlink` TCP port).
2. Both ranks call `ncclCommInitRank(comm, 2, id, rank)` with `zig/src/cuda/nccl.zig`, which already exposes everything needed.
3. Both ranks all-gather their settings (context, chunk, token-map hash, pack fingerprint, knobs that change bytes) and refuse to start if they differ.
4. The same TCP socket stays open as the fate channel. Rank 1 reports its free memory every 0.5 s, and a failure on either rank ends both within about 1 s.

**Rank roles.**
- **Rank 0** owns the server, the lane core, admission, planning and sampling.
- **Rank 1** runs `follow()`: it receives each round's plan (an int stream over the plan link) and runs the same forward. It never decides anything.
- **Plan link.** It starts as a GPU all-gather over NCCL. The RoCE host mailbox (`roce.py`) comes later as a speed item.

**What each rank holds.** This is `py/weights.py`'s table, implemented in `names.zig` and `plan.zig`:

| Rank r of 2 holds | Tensors |
|---|---|
| Half the query heads | `wq_b` columns, the sinks, and `wo_a` groups 4r..4r+3 |
| Input-row half (partial sum) | `wo_b` |
| Intermediate half | routed and shared `w1`/`w3` columns; `w2` rows (partial sum) |
| Vocabulary half | the embedding rows and the head columns |
| Whole | `wq_a`, `wkv`, compressor, indexer (all 32 index heads), norms, `hc_*`, router, Engram `wkv`, DSpark heads |

- The single KV head means both ranks hold the whole KV cache. With split KV in M5, the `comp` pages are sharded instead.
- Every split falls on a 128-value boundary, so EXL3's Hadamard blocks stay whole.

**Exchanges.** There is no all-reduce. Every exchange is an all-gather of bf16 partials, and the receiver sums them in rank order inside the next mHC boundary kernel. That keeps both ranks' streams bit-identical.

Per 1-row window there are 81–83 exchanges:
- the embedding (once)
- Engram column concat (layers 1 and 14)
- two per block (after `wo_b`, after the MoE)
- the head's candidate gather (`[R, 2k]` fp32 per rank; greedy as max-then-lower-id)
- the DSpark passes

The all-gathers run on the compute stream so they can be captured in CUDA graphs, the way Python captures `fast_gather`. Decode alternates two persistent receive buffers.

**Exactness across ranks.** Both ranks run identical plans, page tables and kernels. M2 compares a hash of the stream state per layer between ranks on the first rounds.

### 4a. TP=N: one boot constant, TP=2 unchanged

**Requirement** (user, conditional): TP is N ranks instead of a hardcoded 2, *only if TP=2 performance is unchanged*.
- TP=2 output stays bit-identical to the Python engine (the M1/M2 gates).
- N>2 gets its own gates once hardware exists. A third Spark is only being considered.

**Rule: N is a boot constant.** `--tp N` / `TF_TP` fixes `Topology{ world, rank }` before load. Everything that depends on N is computed once from it: the plan, the run-time weight forms, the window's calls (`block.Options.world`), the KV page ownership, the collective's backend, and buffer sizes. The per-token path is the same launch sequence with the same integer arguments for a given N; it never branches on N. CUDA graphs capture that fixed sequence.

**Why TP=2 performance cannot change:**
1. **Shard math is load time.** `plan.build` (`exl3.part`, `names.blockGroups`, `named`) has taken `world` since the start. `block.emit` now takes `Options.world` (cf18035): at world 2 it emits the identical 789 calls of pod 6's prod capture (`tf-dsv41-m1 check`: 789/789). Kernels receive the same `N`, `K`, `H` ints, so the GPU does the same work instruction for instruction.
2. **Kernels are already world-generic where a rank count reaches them, and TP=2 runs that code today:**
   - The mHC boundary/prefill kernels (`mhc_cuda.cu`, `mhc_pf.cu`) take `world` (1–8) as an argument and add the gathered partials in rank order. TP=2 is this same loop; Python's twin is the same code.
   - The split-KV exchange (`kvsplit.cu`) takes `world` (page k on rank k % world). Its GPU test already covers world 3.
   - Nothing is re-instantiated or re-branched for TP=2.
3. **Collectives keep their TP=2 implementations.** `Collective` (fabric/tp/collective.zig) is one vtable call per exchange (about 80 a window, host side, captured into graphs), as today.
   - The 2-rank backends are untouched and remain TP=2's fast path: the mailbox and RoCE exchanges, and the hybrid. Their 2-rank specialisations are listed below.
   - N>2 is a separate backend behind the same vtable: NCCL first (any N), then an N-rank mailbox/RoCE all-gather where each rank writes its piece into the N-1 peers' receive slots with one doorbell each.
   - The 2-rank code is never generalised in place.
4. **Exchanges stay equal-sized.** Every DSV4.1 exchange is an all-gather of an equal count per rank:
   - full-width bf16 partials after `wo_b` and the MoE;
   - the embedding's partial rows;
   - the head's `[R, 2k]` candidates.
   Uneven shards at N>2 (below) therefore never need an uneven (v-sized) collective.
5. **Where a generalisation would cost TP=2, TP=2 stays hardwired.** A survey of fabric/tp, kv and sessions found the 2-rank hot-path code below, and every item stays as it is:
   - **Mailbox** (`mailbox.zig`, `mailbox.cu`): two inboxes, peer = 1 - rank, and a gather into `[2, n]`. The sum is rank0 + rank1, one fp32 add and one bf16 rounding. It refuses world ≠ 2 at init.
   - **RoCE** (`roce.zig`, `roce_verbs.zig`): one RC queue pair to a single peer, and a 2-rank setup vote. It refuses world ≠ 2.
   - **Hybrid** (`hybrid.zig`): its small path is the 2-rank backend. At N>2, `Session.comm()` already falls back to NCCL on every rank.
   - **N>2 is new code beside these:** an N-way gather kernel (N inboxes, rank r writes slot `r·n`, waits on N-1 flags) and an N-peer RoCE mesh, under new `Kind`s.
6. **The exactness rule at N>2: no float `all_reduce`.**
   - Every float sum stays gather-then-sum: all-gather the partials, then the consuming kernel adds ranks 0..N-1 in order in fp32 and rounds once. The mHC kernels already do this.
   - NCCL's ring all-reduce has no rank order and rounds at each hop, and the 2-rank mailbox/RoCE sum is order-free only because two operands commute. So model float sums never go through `all_reduce`.
   - `reference.reduce` (the host test oracle) rounds after each add. It matches the GPU only at N=2 and must round once before any N>2 test.
7. **Already generic, nothing to change:**
   - bootstrap (star, up to 64 ranks), NCCL, the plan link (rank 0 to N-1 followers), and the fate channel;
   - the split-KV ownership (`k % world`), its dense and union exchanges (one all-gather each, `recv [W, …]`), and its budget;
   - the session pool (one free set per residue, owned page = `local·W + r`) and the disk tier (per-rank directories, with `world` in the header; a file from another world is `Foreign`);
   - the head-candidate merge (`serve/sampling.zig`, rank order);
   - the replay bindings.
   - Limits: the session pool and the split planner stop at 8 ranks. `kv/split.zig` indexed `[8]` arrays by rank without a check; it now refuses `world > max_world` at plan time.
   - `kvsplit.cu` does a runtime `% world` per selected token. TP=2 runs exactly that code today. No `& 1` instance is planned: one modulo per 8-byte word of a copy kernel isn't measurable (dsq-zig-kern-1).
8. **Kernel limits that bound N** (dsq-zig-kern-1's check; no kernel hardwires 2 ranks):
   - **`mhc_cuda` / `mhc_pf`:** world 1–8 (the dispatch refuses more than 8; `mhc_pf` `MAXW` is 8). D = 5120 is compiled in, as in `router_gemv`, but D is the model width, never sharded.
   - **`attn_cuda` (incl. the split kernel):** H % 16 == 0 (16-head tiles; the dispatch refuses otherwise). N=4 fits; N=3 needs the 8-head instance.
   - **`topk_keys.cu` (head candidates):** at most 65,536 columns a rank (the column is packed into 16 bits, as in Python). Any N ≥ 2 fits (64,640 at N=2, 32,320 at N=4). TP=1's 129,280 doesn't, and Python falls back to a stable sort there too.
   - **Expert widths:**
     - x3ld / x3pf / grouped need N % (16 nt) == 0 and K % (16 · SK · 4) == 0;
     - x3gm / gm2 need width % 128 == 0 and (width / 16) % KS == 0;
     - the gm plan needs E < 512.
     Block-balanced 128-value shards (4 or 5 blocks a rank) meet all of these, and every check is per call, so uneven widths are fine.
   - So the effective bound is N ≤ 8 (mHC, the session pool, the split planner), with N=3 the only size below it that needs a new kernel instance.

**Shard sizes for DSV4.1 at N ranks.** Splits stay on 128-value blocks (EXL3 Hadamard blocks), so `width / N` is not always possible. A dimension that doesn't divide evenly gets block-balanced ranges: the first `blocks % N` ranks take one extra block. These are per-rank constants at load. TP=2 is the even case, with exactly today's shapes.

| Dimension | N = 2 (today) | N = 3 | N = 4 |
|---|---|---|---|
| query heads (attention runs 16-head tiles; heads follow `wo_a` groups of 8) | 32 / 32 | 24 / 24 / 16 (groups 3/3/2), **not runnable on 16-head tiles**: needs an 8-head-tile attention instance (a separate kernel instance, TP=2's untouched) | 16 each (groups 2) |
| routed + shared expert width 2304 (18 blocks) | 9 / 9 | 6 each | 5 / 5 / 4 / 4 |
| vocabulary 129,280 (1,010 blocks; embedding rows, head columns) | 505 / 505 | 337 / 337 / 336 | 253 / 253 / 252 / 252 |
| `wo_b`, `w2` rows (partial sums) | follow the heads / experts above | | |
| whole on every rank | `wq_a`, `wkv`, compressor, indexer, norms, `hc_*`, router, Engram `wkv`, DSpark | same | same |

What the uneven splits cost N>2 (not TP=2):
- The slowest rank sets the pace. At N=4 that is +11 % expert width on 2 of 4 ranks (5 blocks vs 4.5 ideal).
- At N=3 the attention is unbalanced (24 vs 21.3 ideal heads) and needs a new attention instance.
- Expert parallelism (whole experts per rank, 384/N) would balance the MoE but changes the sum order and adds an all-to-all; it is not planned.
- So N=4 is the natural next size for this model. N=3 Sparks are better used as TP=2 plus a service node (4c).

**What else changes with N (all load time):**
- **KV:** split-KV pages are owned k % N. `DevicePool` capacity is per rank, ⌈pages / N⌉, and `kv.budget` prices every rank.
- **Sessions:** park/restore files record `world`. A restore needs the same world (part of the compat key, as the pack fingerprint is).
- **Plan link and fate channel:** rank 0 sends a round's plan to N-1 followers. The fate channel is a star around rank 0, and any rank's failure ends all. Both already work this way.
- **DSpark candidates:** `draft/dspark.zig` caps `candidates` at 128 / world. It is a boot constant.

**Exactness gates for N>2** (later; no hardware):
- **Bit-exact:** an N-rank emulator on one GPU. It runs the N shards' forwards in sequence and gathers their partials in rank order, and its output is compared with the real N-rank run.
- **Agreement, not bits:** greedy token agreement against TP=2 over long runs. The sums differ from TP=2's, so bits cannot match.
- TP=2 keeps M1/M2 against the Python engine.

### 4b. Remote drafter and remote Engram (interfaces only)

The aim: a third Spark hosts services the model calls over the fabric, without the forward knowing.
- **DSpark drafter:** frees about 3.6 GiB a rank.
- **Engram table:** rows are prefetched over CX7 instead of read from local NVMe.

Both fit behind seams that already exist.

**Remote drafter = another `draft/iface.zig` `Pass`.** `Pass` is already the drafter's whole contract (`ingest`, `propose`, `reset`, `markov`), and the target never sees the drafter's state. `RemotePass` implements the vtable over a fabric connection from rank 0 only, since drafts are plan data and the plan link carries them to the followers.
- `ingest(slot, start, taps, rows)`: sends the committed taps rows to the drafter host.
  - Each row is bf16 `[3 × hidden]`, 30 KiB. The send is a device-to-device RDMA write from the `Taps` address (GPUDirect over CX7), so there is no host copy.
  - It is asynchronous; the next `propose` orders after it.
- `propose(asks, out)`: one request a round, for all slots' `Ask`s. The response is drafts and confidences, plus candidates and base logits for trees. The caller's `Proposal` buffers are unchanged.
- `reset(slot)` and `markov()`: requests. `markov()` can return null (no trees) until the Markov heads move too.
- **Latency:** one round trip (about 10 µs RDMA on CX7) plus the pass on the remote GB10. With several lanes, the lane core runs other slots' verify windows while a slot waits, so the round trip leaves the critical path once more than one slot is active.
- **Failure:** the fate channel marks the drafter down and the lane core drafts nothing (chains of 1). Output is unchanged because verification is exact, and only the speed drops.

**Remote Engram = an `EngramSource`** (`engram_host.zig`, the R3 seam). Both the local and remote implementations produce exactly the forward's existing glue input (`L.engram.rows`, bf16 `[R, 24 × 256]` for layers 1 and 14) and an event:
```zig
pub const EngramSource = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        /// Start fetching the rows of a window's tokens (hashes per Engram layer) into the slot's staging; returns at once.
        issue: *const fn (ptr: *anyopaque, window: u64, hashes: []const [2][24]u64, rows: u32) Error!Ticket,
        /// The rows' device view for layer `l` (1 or 14) and the event the window's graph waits on before that layer.
        ready: *const fn (ptr: *anyopaque, t: Ticket, l: u32) Error!struct { rows: u64, event: cuda.abi.Event },
        release: *const fn (ptr: *anyopaque, t: Ticket) void,
    };
};
```
- **Local (R3):** io_uring reads from the local NVMe shards into pinned staging, then an async H2D copy.
- **Remote:** the service holds the tables and RDMA-writes the rows straight into every rank's staging. Every rank needs them, because Engram `wkv` is whole.
  - The traffic is about 12 KiB a token a layer, 2 layers, multicast to N ranks.
  - It is issued at draft time, as R3 plans, so it overlaps the previous window.
- The rows are table bytes either way, so output is bit-identical whichever source serves them.

**Who changes what:**
- **dsq-zig-kern-1:** for TP=2, nothing. For N>2:
  - an 8-head-tile attention instance (N=3 only);
  - the uneven expert widths (5/4 blocks) through x3ld/x3seg's existing per-call N.
- **KV / sessions** (the port lead's now that dsq-zig-kv-1 is settled):
  - ⌈pages / N⌉ per-rank capacity;
  - `world` in the session compat key.
- **TP fabric** (the port lead's now that dsq-zig-tp-1 is settled):
  - an N-rank backend behind `Collective` (NCCL first);
  - a star plan link and fate channel.
  The 2-rank backends are unchanged.

## 5. Milestones

Effort is in engineering days for this agent team, assuming GPU windows are available when needed. GPU work is listed where it happens.

**M0. Toolchain and host side** (done, this branch).
- Zig in user space, `zig build test` green, aarch64 cross-build, the family's host side with 13 tests.

**M1. One block, bit-exact against its Python twin** (about 10–15 days).
- **Build:**
  - `load.zig`: device upload of a block from the plan, plus the expert tables.
  - The Triton AOT capture for DeepSeek's Triton-only kernels: a `tools/zig/dsv41_capture.py` in the PyTorch image against `dsv41-quant-e2`, packed with `zig/tests/cuda/nemotron/aot_pack.py`'s format.
  - `mhc.zig`, `attention.zig`, `compress.zig`, `indexer.zig`, `engram.zig`, `moe.zig`, `block.zig`.
  - A `tf-dsv41-test block LAYER FIXTURE_DIR` runner.
- **Blocks covered:**
  - layer 1 (SWA + Engram)
  - layer 2 (full: KV source, compressor gate, indexer)
  - layer 3 (reuse)
  - layer 20 (ratio 1, candidate source)
  - layer 24 (reindex)
  - layer 40 (DSpark block shape)
- **Inputs:** fixed rows at 1, 3 and 16 rows, with a prepared KV state (prompt rows written by the Python twin and loaded as the fixture's pool pages). Rank 0's slice is used, with the peer's partial taken from the fixture.
- **GPU:** one GPU of the same architecture as the Python run.
  - The bits only compare within one architecture, because the SASS differs between sm_120 and sm_121.
  - Either a GB10 window (the Sparks are owned by the Python-engine agent now) or a RunPod RTX PRO 6000 (sm_120) on the Python `tensorfold-dsv41:e1e2-…-sm120` image. RunPod needs approval: about 2 hours a run, roughly $4–5.

- **State (2026-10-06): the M1 replay gate PASSES on one RTX PRO 6000** (sm_120, against the e1e2 runtime's Python engine on the q28-v2 pack). Pods 3 and 4:
  - **Weights:** our loader's 480 tensors are bit-identical to the Python loader's.
  - **Ops:** 1,206 launches of blocks 1-3 and 20-24 are bit-exact, both alone and chained on our own outputs:
    - decode windows of 1, 3 and 16 rows, plus a 256-row prefill segment;
    - every Triton kernel kind (71 variants);
    - the EXL3 linears, x3seg, the routed-expert chain, x3gm, the fast-prefill GEMM and the split router.
  - **Kernel exactness job:** 13/13 cases pass, and the SASS of the ported kernels is identical to the Python extensions'.
  - Each M1 pod takes about 15-19 min and costs about $0.6.
  - **Pod 5 completes the replay gate:**
    - sets A, B and C: 9 blocks, including the DSpark block (layer 40: 128 experts, top-3);
    - weights 535/535;
    - 1,353/1,353 ops bit-exact alone and chained (4,533 inputs from our own outputs), 0 skipped.
- **M1, second half: our own block forward.** The capture's op log is the spec. A full-attention block's decode window is about 30-35 launches:
  1. Engram (layers 1 and 14): wkv linear, `_fuse_dec`.
  2. mHC: `_site_dec` + `_finish_k`.
  3. The q/kv projections through x3seg, then `_rms2`.
  4. `wq_b` (linear), `_rope`, `_kv_store`.
  5. Full layers: the compressor (`_pool_norm`, `_kv_store`), indexer keys (linear on `ix_wk`, `_index_k`), `_scores`, `_dtopk`.
  6. Attention: `_chunks` + `_merge`.
  7. `wo_a` (x3seg), `wo_b` (linear).
  8. mHC again, then the router `_fused`.
  9. Experts: `group`, `rot_in`, `grouped` gate/up, `gateup_epilogue`, `grouped` down, `down_combine`.
  - **`prepare.zig`** builds the derived weight forms Python's `Block` makes at load:
    - linear trellises re-laid out as int32 `[N/128, K/16, 8, 32]`;
    - x3seg segment tables;
    - the experts' int64 address and int32 K2 tables over 385 entries (384 routed plus the shared expert), with `suh`/`svh` stacked `[385, K]`;
    - dense3's lanes layout for the 8-bit `ix_wk`.
  - **Target: the prod kernel path** (prod.env's knobs, captured by `M1_KNOBS=prod`), not the engine's code defaults. A full block's prod decode window is:
    1. `mhc_cuda.run` (the boundary).
    2. `_plain` (`ix_wp`, on index layers).
    3. The x group: x3seg `rot_in`, then dense3 lanes launches by K2 (`wq_a` at K2 8; `wkv` + compressor `wkv`/`wgate` at K2 10).
    4. `_rms2`.
    5. On KV sources: `_pool_norm` + `_kv_store` (compressed row), the `ix_wk` upstream linear (K2 16), `_index_k`.
    6. On index sources: the `ix_wq_b` dense3, `_rope` (index q), `_scores`, `attn_cuda.topk`.
    7. The `wq_b` dense3, `l2pf.segments`, `_rope` (q), `_kv_store` (SWA ring), `attn_cuda.attn`.
    8. `wo_a` (dense3, 4 segments), `wo_b` (dense3), `l2pace`.
    9. `mhc_cuda.run`, `router_gemv.route`, `_prune` (top-p 0.85).
    10. Experts: `group`, `rot_in`, `x3ld` gate/up, `gateup_epilogue`, `x3ld` down, `down_combine`.
  - Every `.cu` op has a typed launcher in `dsv41_kernels` (`kernels_ops.zig`, `kernels_dense.zig`, `kernels_exl3.zig`). The Triton ops go through the M1 AOT set. Pod 6's capture, with every argument, is the per-op spec (`dsv41-zig-ref/prod-block-L2.txt`).
  - **As built (41ce3ed):**
    - `block.zig` + `block_moe.zig` emit a decode window (`block.emit(cfg, widths, options, layers, n, start, head)`) as `calls.Call`s:
      - each call is a Triton function and grid with named arguments, or an extension function with its arguments;
      - every tensor is named by role: `s.` persists (run-time weight forms, pools, scratch), `w.` is the window's, `L.` is the layer's.
    - `Widths` is the interface to the loader: the dense K2s by weight prefix, and each routed layer's expert K2 range.
      - A run fills it from the plan. The M1 check fills it from the capture's `weights.json` and `fixtures/q28v2-expert-k2.txt`.
    - `Options` holds the sizes Python fixes at load (pool positions, rope rows, expert scratch rows) and the prod L2 prefetch settings.
    - The same calls are M2's forward: M2 replaces `m1_forward.zig`'s seeding with `state.zig` and the glue below.
  - **Host gate, `tf-dsv41-m1 check CAPTURE`:** compares every captured decode window. On pod 6's prod capture (sets A/B/C at 1, 3 and 16 rows), 789/789 calls are equal:
    - kernel, grid, every scalar (floats by bits);
    - every tensor's dtype, shape, strides and offset, and weights by name;
    - each role is one captured storage within its scope, and two roles in one launch are two storages;
    - by content, no role's first read holds another role's last bytes. This caught `pos`: top-k and attention read the stores' `pos` tensor.
  - **The glue the forward runs**, i.e. what the capture shows changing between launches with no recorded op:
    - the two exchanges (`w.recv.attn` before the FFN site; `w.recv.moe` before the next attention site and the final norm);
    - `kit_weights`' fp16 rounding of the pruned expert weights (`L.wts16`);
    - positions (`w.pos` i32, `w.pos64` i64 per row, `w.lo` the SWA lower bounds);
    - Engram's hashed rows;
    - the streams' input (`w.x0`);
    - the ratio-2 compressor's carry between windows (`s.L2.carry`);
    - zero-initialised counters (dense3/linear `cnt`, `w.mhc.cnt`).
  - **GPU gate, `tf-dsv41-m1 forward|both PACK CAPTURE OUT`** (`m1_forward.zig`):
    - Runs the same calls on our own buffers, one per role, with weights bound by name to our loader's.
    - After each launch, compares every buffer the launch touches bit for bit.
    - Seeds come from the capture: a role's first touch in its scope, and the glue points above (re-seeded and counted).
    - Everything else a call reads is what our earlier calls left.
    - Our run-time forms are compared with the seeds Python built: `prepare.strips`, the mHC bf16 weights, the expert `suh`/`svh` stacks and K2 tables.
    - The expert address tables are relocated, not rebuilt (M2: `prepare.tables`).
    - The Triton variant is picked from the captured op's hash. M2 picks it from name, constexprs and specialization (`manifest.json`).
    - The L2 prefetchers are skipped (their tables are checked for shape only).

**M2. Whole model, TP=2, greedy token-identical on GB10** (about 15–20 days).
- **Build:**
  - `forward.zig` (window + prefill), `state.zig`/`pool.zig`, `sampler.zig`, `tp.zig` (bootstrap, follower, exchanges, fate channel).
  - `prefill.zig` with the exact-tag prefill first, then the fast kernels (`x3gm`, `_gemm`) under their own session tag as in Python.
  - `graphs.zig`, `prepared.zig`.
  - A `tensorfold run` path for DeepSeek: `zig/src/cli/dsv41_main.zig`, or a family switch in `cuda_main.zig`.
- **Plan, as M1 left it (2026-10-06).** M1 proved the spec and the seam. `block.emit` produces a window's launches on the prod path; `tf-dsv41-m1 check` gates them against a capture on the host. `m1_forward.zig` showed those calls reproduce Python's bits on our own buffers once the glue is supplied. M2 is that executor with every seed replaced by our own state and glue. In order:
  1. **`run.zig`, the call executor** (from `m1_forward.zig` minus seeding):
     - Roles become a buffer plan. One emit per row bucket at load sizes every role. `s.` roles are persistent allocations, `w.` and `L.` roles one arena reused across layers. No allocation per token.
     - Extension calls go through the typed launchers (`dsv41_kernels`), not `replay.call`'s argument decoding. Triton calls use the engine's `cuda/aot.zig` `Set.find`: name, constexprs, and the 16-byte specialization of our own arguments, the variant Triton itself would pick. `aot.Set` gains global scratch (it refuses it today; `m1_triton.zig` handles it).
     - **Triton variant choice is done offline.** `triton_call.zig` splits a call's named arguments into the engine's runtime args and constexprs over the function's variants. The capture's `pack_aot` now writes the engine's `aot.json` form: constexprs (floats as fp32 bits), `div16` from the compiled signature, `nospec` from `jit.json`. The engine's own `aot.zig` matcher (`specMatches`, now public with `parseSpecs`) runs on that. `tf-dsv41-m1 check` gates it: on pod 8's prod capture, 153/153 Triton calls pick exactly the variant Python launched, so M2 launches Triton by name, not by a captured hash. The job now returns the capture's AOT set whole (`aot.tar.gz`).
     - **Gate:** `tf-dsv41-m1 check` keeps gating the emitted calls. The executor reuses M1's GPU forward with seeding off for everything but window inputs.
  2. **AOT coverage set:** the Triton variants of every row bucket (1–16 decode, the prefill chunk sizes), recorded from a prod-knob Python run without blobs (a launch log is enough). This is a short pod step, and the kernel agent owns the set's build.
  3. **Glue, from Python's `forward.py` (each is one small kernel, written to torch's arithmetic):**
     - **Window start:**
       - `w.pos` = the first position (i32). `w.pos64` = `arange(start, start + n)` (i64). `w.lo` = zeros (i32, `Context.lo`).
       - The streams: this rank's embedding rows (`ids - vocab_lo`, masked to zero outside the rank's range, bf16 to fp32). Then the all-sum, the window's first exchange: fp32 to bf16, a gather into the pair of receive buffers, back to fp32, `parts[0] + parts[1] + …` in rank order, then to bf16. Then repeat ×4 into `w.x0`. A faithful sum matters: 0 + (-0) is +0.
     - **Exchanges:** each takes the next of the two alternating receive buffers. In decode the embedding takes one and the MoE partials share it (`w.recv.moe`); attention's partials take the other (`w.recv.attn`).
       - Attention's `wo_b` partial is already bf16 (`TF_DSV41_PARTIAL_BF16`): gather only.
       - The MoE's fp32 partial is cast to bf16 (RNE), then gathered.
     - **`kit_weights`:** the pruned expert weights fp32 to fp16 to fp32 (`L.wts16`), backbone layers only (DSpark's router folds it).
     - **Engram rows (layers 1, 14):** the host hashes the ids (with the slot's tail), reads this rank's columns of the table rows, then `gather_cols` (an all-gather, permuted to `[n, world · c]`) into `L.engram.rows`.
     - **Head:** `kit_logits` (fp32 to bf16 to fp32) on this rank's logits, then sampling (`serve/sampling.zig`).
     - **`keep(slot, accepted)`:** the ratio-2 carry ← fp32 of row `accepted` of the window's compressor projection. That projection is the role `w.L<i>.comp` for ratio-2 sources, so it outlives its layer as `ctx.proj` does. Positions and the Engram tail advance.
     - **Mid-stack Engram (layer 14):** Python runs the last exchange's post alone (Triton `_site`, `POST_ON`, no collapse or mix), then Engram, then a site2. `block.zig` emits that (1,083 launches a window). Capture set D (layers 13–15) checks it on the next pod.
     - positions (`w.pos`, `w.pos64`, the SWA lower bounds `w.lo`);
     - the stream input (the embedding gather of this rank's vocabulary rows, its exchange, the 4-stream expand);
     - the two exchanges into `w.recv.attn` / `w.recv.moe` (partial to bf16, then `Collective.allGather` into `[world, R, D]`);
     - `kit_weights`' fp16 rounding of the pruned weights;
     - the ratio-2 carry roll between windows;
     - Engram rows (hashing on the host, `EngramSource` local first);
     - zeroed counters at allocation.
     Each glue op is a small kernel next to `pointwise.cu`, written to torch's arithmetic.
  4. **`state.zig`:**
     - per slot: the SWA rings, compressed pools and index keys through the session pool (`sessions/pool.zig` residues, `kv/device.zig`), and the ratio-2 carries;
     - positions, and the rollback `draft/iface.zig` `keep` needs.
     The M1 forward's `s.L*.swa/comp/ik/carry` roles map one to one.
  5. **`forward.zig`:** a window is `block.emit` over every layer plus the head, run by `run.zig` with `state.zig`'s buffers bound. Prefill comes later with the exact-tag kernels first, as in Python.
  6. **Gates:**
     - **M2a, one GPU, rank 0, a peer of zeros:** the whole model's decode windows against Python on the same pack. Compared: per-layer stream digests and the head's logits, bit for bit, recorded by a capture mode that keeps digests only (no blobs, so all 43 blocks fit). A short PRO 6000 pod.
     - **M2b, two GPUs on one pod (NCCL):** TP=2 real exchanges against Python's TP=2 on the same pod. Logits bitwise, greedy tokens identical.
     - **M2c, the two Sparks:** the same on GB10, plus timing.
  - **Started offline:** `buffers.zig` sizes every role. `tf-dsv41-m1 plan` emits the whole model from the q28-v2 listings: `fixtures/q28v2-dense-k2.txt`, derived from `plans/q28-v2.json` and equal to all 109 trellises the capture holds. Results:
    - The backbone window is 1,081 launches at every row bucket from 1 to 16 (198 Triton; 883 extension, 119 of them L2 prefetch). Each DSpark block is 22.
    - Activations are 6.4 MiB (window) + 12.7 MiB (layer) at 16 rows.
    - Persistent is 3.1 GiB: run-time weight forms, scratch and one slot's pools. Python relayouts the dense trellises in place, so M2 must replace the stored trellis with its strips / lanes form, not hold both.
    - Not yet captured, so emitted from rules: layer 0, Engram on layer 14 (the order of its glue around the mHC site is unconfirmed), the KV sources 8/14, the index sources 28–36, DSpark 41–42. M2a's digests cover them.
  - **Ownership:** the port lead owns `run.zig`, `state.zig`, `forward.zig` and the glue. dsq-zig-kern-1 owns the glue kernels' `.cu`, `aot.Set` scratch, and the coverage set. The TP fabric and KV/sessions code is used as is.
- **Pack:** Mia's 2.9 bpw pack first (one width per layer, so the stack layout only), then ours (q28/v2: ragged).
- **GPU:** both Sparks for several sessions of about 2 hours. This needs the Python-engine agent to release them, or a planned window like the E1/E2 one.

- **Spark results that set M2's design** (quant repo `docs/E1E2-SPARK-WINDOW.md` §9, a7330a1; the Sparks now serve prod q28-v2 on the Python perf1 engine, 8474f31, and are not available to the port):
  - **The Zig two-Spark TP gate passed on real CX7:** RoCE suite 378/0, NCCL 189/0, fail-fast OK.
    - RoCE latency is 8.3 / 8.9 / 10.4 µs at 4 / 8 / 16 KiB, against NCCL's ~20.3 µs.
    - The 86-exchange decode window takes 767 µs on RoCE against 1,804 µs on NCCL.
    - So M2c's decode exchanges go through the hybrid (RoCE small, NCCL large) from the start; M2b's NCCL is only the pod's wire.
  - **Python's decode exchange wait is about 0.5 ms, fixed:** rank 1 waits at exchange 3 while rank 0 reads Engram rows from NVMe. That confirms R3 (Engram rows prefetched off the window, `EngramSource`).
    - Per-exchange jitter on both ranks adds to it. Fewer and faster exchanges help; rank balancing won't.
  - **A 32K prefill chunk is GPU-bound** (idle 3–5 %). It runs prefill attention, then 172 NCCL all-gathers a chunk, then the mHC site and the router / combine.
    - Prefill does not use RoCE in Python. In Zig, prefill's all-gathers go through the same RoCE / hybrid path as decode's from the start.
- **Performance designed in from the start** (`docs/DSV41-ZIG-PERF.md`, owned by the port lead):
  - **R3, Engram reads off the window** (+3-5 %, exact).
    - The Engram row reads for the next window are issued at draft time, through io_uring on the host, into pinned staging.
    - The window's graph waits on a GPU-side event only before layers 1 and 14, not at the window's start. Python's `gate_copy_kernel` spins about 1.65 ms waiting on the host.
    - `engram_host.zig` owns the hashing, the shard reads and the staging ring. `forward.zig` exposes an "Engram rows ready" event per window.
  - **R4, hidden graph submission** (+2-3 %, exact). The window graph's host launch takes about 1.27 ms.
    - Windows are split into pre-instantiated graph segments that are enqueued ahead of time: the next round's segments are submitted while the current round runs, with device-side ordering through events and the row table updated in place.
    - Fewer nodes come from R1's fusions (about 1,331 down to about 700 per window).
    - `graphs.zig` is designed for this from the first version: per row-bucket and context-bucket segment graphs, updates of node parameters instead of re-capture, and no host sync inside a round.
  - **Copies in the forward:** stream-ordered `cuMemcpyHtoDAsync` / `DtoHAsync` from pinned memory on the forward's stream only. A plain `cuMemcpyHtoD` from pageable memory can return before its DMA lands, and a non-blocking stream does not wait on the legacy stream; M1 pod 3 caught exactly that race.
  - Kernel-side gains (R1 decode fusions, R2 x3gm v2) land behind the kernel launch interfaces (`dsq-decode-r1-1`, `dsq-x3gm-onelaunch-1`).
  - **Not approved:** R5, FP8 prefill (it changes outputs; the user rejected it before).
  - **Blocked:** R6, 4K-row chunks (memory).
  - **No megakernel.**

**M3. Lanes and drafter** (about 10–15 days).
- `lanes.zig` (the lane core's backend):
  - `verify` is a window of at most 32 rows, against Python's 16 per slot and 64 in total.
  - `keep` commits the accepted rows: pool truncation, SWA rings and ratio-2 carries roll back to the kept rows.
  - `draft` is the DSpark pass.
- `drafter.zig`; copy drafts from the lane core's copy index; DeepSeek's depth costs.
- Keyed sampling at T > 0 with the same Gumbel keys (seed, absolute position, id) as Python.
- **GPU:** both Sparks.
- **State:** the host side is done (`draft/**`, `core/lanes/trim.zig`, commits `4d94c92..a49ea13`, `zig build test-dsv41-draft` 17/17, upstream lanes 37/37; interface in section 0).
- **Open items:**
  1. The speculative early pass (`TF_DSV41_SPEC_DRAFT`) and the nucleus candidate path (`TF_DSV41_SPEC_NUCLEUS`) are not ported.
  2. The first round drafts at the lane's prior depth, not a full DSpark block.
  3. The cost of verify windows over 32 rows is to be measured once M2's forward exists.
  4. Load the real prod calibration table (`calib-*.json` from the Sparks' `/cache`, through `Costs.parseJson`) once the Spark agent can copy it.
  5. M2 contract: a chain window's rows stay in the target's state until `keep()` is called, because in a shared round the lane core asks for drafts before it keeps. Ingest reads the kept path from the window's taps, not from the pool.

**M4. Server** (about 8–12 days).
- `zig/src/native/cuda.zig`, wiring into `engines.zig` for Linux builds.
- `encoding.zig`, `dsml.zig`, reasoning effort, `memory.zig` (the floor and adaptive prefill rows), admission and fairness for 4 slots, and `/metrics`.
- Grammar: the Zig server's structured-output fields and the engine's grammar enforcement. Python uses xgrammar; this needs a decision (R9).
- **GPU:** both Sparks, then a soak.

- **State:** the server side is done (`deepseek_v41/serve/**`, `zig/src/server/**`, commits `ad44e18..f66c019`).
  - Tokenizer, encoding, tool calls and sampling are byte-identical to Python on large golden sets.
  - 43/43 HTTP tests pass on a stub engine; the tokenizer is about 23x faster than HF's.
  - The real engine plugs in by implementing `engine_api.Engine`.
- **Open items:**
  1. Native image input (`TF_DSV41_IMAGES=native`) is not supported.
  2. Structured-output grammar (R9) waits for the engine.
  3. `tool_choice` required or named offers all tools, as Python does without grammar.
  4. Stop strings match only the answer, but Python's non-streamed no-tools path also matches the reasoning. **Decision: match Python exactly**, because reply text must equal prod's.

**M5. Long context** (about 15–20 days).
- `sessions.zig` and `sessdisk.zig` (streamed NVMe, residency budget, priced restores, as on `dsv41-sessres`).
- `kvsplit.zig` (as on `dsv41-kvsplit`), at the 1M context.
- FP4 KV rows (`dsv41-kvq`) after the gates pass.
- **GPU:** both Sparks; a context ladder up to 1M tokens.

**Total:** about 60–80 engineering days, or roughly 3 months calendar with GPU windows. The CUDA paths M1 and M2 carry most of the risk; M4 and M5 are mostly host code.

## 6. Exactness gates

| Milestone | Gate (equality unless stated) | Reference |
|---|---|---|
| M1 | Every op's output bits equal the Python twin's on the same GPU architecture, for each covered block, at 1, 3 and 16 rows. Then the block's output streams are equal. Also: a row's bits do not depend on how many rows share the window (one row alone against the same row in 2-, 3- and 16-row windows). Weight digests per tensor equal Python's `RankW` (a `check-weights`-style JSON). | `dsv41-quant-e2` in the PyTorch image, fixtures from the kernel agent's oracle |
| M2 | Greedy `--no-drafts` tokens are identical to the Python engine for 16 prompts × 256 tokens (code, prose, long prompt past one 2048-row chunk, tool call), with expert top-p off (`TF_DSV41_EXPERT_TOPP` unset) and then at the prod 0.85. Selected steps' bf16 logits are equal. Both ranks' per-layer stream hashes are equal. The prefill exact tag is equal to serial; the fast prefill tag is equal to Python's fast tag. Sanity only: teacher-forced top-1 0.9961 (Mia) / 0.9881 (q28). | Python prod configuration on the same pack |
| M3 | Drafted output is byte-identical to `--no-drafts` on the whole prompt set at T=0. At T > 0 it equals Python's keyed sampling for the same seed. Batched equals alone. Graph replay equals eager. | The Zig engine's own serial path, plus Python |
| M4 | HTTP replies have the same tokens as the CLI. `encoding.zig` output is token-identical to `encoding.encode` on the chat fixtures (tools, reasoning, images-as-text). `dsml.zig` produces the same parsed calls and stream fragments as `dsml.py` on recorded replies. | Python app on recorded fixtures |
| M5 | A resumed session's next tokens equal an uninterrupted run, from RAM and from NVMe. Split KV equals unsplit, bit-exact, at 1, 16 and 64 rows and across prefill. | The Zig engine's own unsplit, uninterrupted path |

Speed targets come from the Python engine's measurements:
- Single-stream decode (q28 pack): code about 83–85 tok/s, prose about 47 tok/s; serial about 28 tok/s; a 1-row window about 21.8 ms.
- Cold prefill at 32K: about 2,300 tok/s.
- Boot: about 40 s from a prepared folder.
- Pool: about 2.24M tokens at the 4 GiB floor (v2 pack).

The Zig engine is not shipped to prod until it matches or beats these.

## 7. Risks

1. **R1, Triton dependence.** Most of the default path is Triton. AOT cubins are tied to Triton's version, the constexprs and the specialization keys. The capture has to cover every shape the forward uses: row buckets 1–64, prefill 2048, context buckets.
   - Mitigation: capture from the exact Python image, and refuse to start on a missing variant (as `aot.zig` does).
   - Replace Triton kernels with proven `.cu` twins where they exist.
2. **R2, the 32-row cap.** The Zig engine holds at most 32 rows a forward. Python's row mode takes 64 (4 slots × 16). Up to M3 this limits concurrent drafting, not exactness. Raising it is upstream's open item 3.
3. **R3, no multi-host NCCL bootstrap in the Zig runtime.** `nccl.zig` has the calls but nothing exchanges the unique id across hosts, and no family has run NCCL yet. It is small to write, but it is new and sits on the critical path for M2. The RoCE fast all-gather is a later speed item; prod uses it today.
4. **R4, modules missing from the reference.** `kvpool`, `server`, `roce`, `comm` and `exact_sampling` live in GLM or shared code that the DeepSeek family imports. They need their own reading before M2 and M4.
5. **R5, GB10 memory headroom.** Weights are about 95 GiB per rank (v2), and the hard floor is 4 GiB.
   - The Zig engine must not hold more than Python does: no mmap of the pack (`pack.zig` already reads with pread), pinned staging sized like Python's, and scratch priced into the floor.
   - Boot-time load check as in `memory.load_check`.
6. **R6, host-computed tables.** RoPE/YaRN cos/sin tables, Engram hashing and the sampler's keyed uniforms must equal Python's bits.
   - Gate each with a dump.
   - Prefer computing them the way torch does: fp32 ops in the same order, or copy torch's kernels.
7. **R7, prod is not all-experts exact.** `TF_DSV41_EXPERT_TOPP=0.85` prunes experts lossily but stays rank- and batch-consistent. The `replay` prefill (CED) is approximate by design and has its own session tag.
   - The Zig twin has to implement the same pruning and tags to be token-identical to prod.
   - M2 gates both off first, then on.
8. **R8, GPU access.**
   - The Sparks belong to the Python-engine agent now.
   - M1 can run on a single same-architecture GPU (a RunPod RTX PRO 6000, after approval).
   - M2–M5 need both Sparks in planned windows.
9. **R9, structured output.** Python uses xgrammar 0.2.8 with DeepSeek's structural tag. The Zig server validates grammar fields but has no xgrammar. Choose either a native grammar matcher (the lane core already masks per row) or serving without grammar at first, behind `TF_DSV41_GRAMMAR=0`.
10. **R10, the vision tower.** It runs on rank 0 only in Python (`TF_DSV41_IMAGES=native`). It is out of scope for M1–M5; image requests are refused or served as placeholders.
11. **R11, upstream churn.** `zig-flashnext` moves fast. The family stays in its own directory, and touches of shared files are kept to one function each in `build.zig` and `zig/build/cuda.zig` (the latter is the kernel agent's).
