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
  - 28–31: `dsq-zig-grammar-1` (28 `op_grammar`: bind / step / release, `grammar_gpu.zig`);
  - 40–43: `dsq-zig-mdraft-1` (40 `op_ingest`, 41 `op_propose`, 42 `op_reset`: `dspark_rows.zig`; 28–31 are `dsq-zig-grammar-1`'s `op_grammar`).
  - 44–47: `dsq-zig-ced-1` (44 `op_ced_finish` `[op, end]`: `Forward.finishPrompt`; 45 `op_sess_prompt`: a replay-mode prompt snapshot, `sessions_gpu.savePrompt`).
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
  - **Knob: `TF_DSV41_SAMPLING=1`** (default off, the same on every rank).
    - Off is today's behaviour: requests with T > 0 are refused (`error.Sampling`), and no kernel is loaded or memory allocated. Greedy is unchanged either way, and no emitter is touched, so the `check` / `dcheck` gates see the same calls.
    - On: `sampling_gpu.zig`'s `GpuSampler` exists on every rank. Rank 0 sends `op_sample` before its gathers, and the followers run it from `follow`.
    - `sampling_kernels.zig` loads its own fatbin, `dsv41_sampling`. It is optional in the build: `-Dnvcc=` builds it; `-Dfatbins=` takes it when the dir holds `dsv41_sampling.fatbin`; otherwise it is empty and the sampler refuses to load.
  - **Hooks:**
    - `forward.zig`: `Forward.sampler` and `op_sample`;
    - `target.zig`: prefill and window choices with T > 0 go to `sampler`; without one they are refused (`error.Sampling`);
    - `model.zig`: init / deinit, behind the knob;
    - `serve_cli.zig`: a trace line's `sampling`;
    - build: `zig/build/sampling.zig`. `cuda.zig` gets the optional `dsv41_sampling` image, `tf-dsv41-samp`, and `dsv41_serve` for `tf-dsv41-m1`, whose `core` is now `safetensors_root.zig`. `server.zig` gives `dsv41_serve` to the native engine.
  - **Host gate:** `zig build test-dsv41-sampling`. The committed serving goldens on two in-process ranks: 207/207 sampled rows equal Python's tokens, including 14 whole-row fallbacks. A window's rows pick as each row alone does, and as `exact_sampling.choose` over the whole row.
  - **Slots (agreed with `dsq-zig-slots-1`; landed on the sampler side, the smaller one).**
    - Row ranges: `Batch.targetWindow` packs each forward's segments back to back in "w.logits" rows [0, n), in segment order, so segment i owns rows [a_i, a_i + len_i).
    - A sampled segment (T > 0) gets one `GpuSampler.choose(a_i, len_i, its sampling, its draws)`, so one `op_sample` per sampled segment, sent after the forward's `op_greedy`.
    - `op_greedy` is sent only when one of that forward's segments is greedy.
    - A slot's prefill choice goes through `target.prefill` (`pickOne` on its last row), as with one slot.
  - **Trees.** Tree chains go through `window`, so they sample too.
  - **Trace replay.** The HTTP recorder (`TF_DSV41_TRACE_TOKENS`) saves each request's `sampling`, seed included, so sampled HTTP replies replay through `generate` (M4 CLI == HTTP).
  - **Pod step: `M2B_SAMP=1`** (`M1_JOB=m2b`, 2 GPUs). With `M2B_M4=1` set too, it runs after M4.
    - **Reference.** `dsv41_m2b_ref.py --sampled` writes the sampled replies on the reference's own load: no second Python load, about 2 min of decoding. That is `dsv41_samp_ref.run_cases`, before the AOT dump, so its prompt lengths' Triton variants land in the set: nothing to add to `M2B_AOT_PROMPTS`. The 6 prompts are 37, 50, 66, 83, 128 and 292 ids; the 6 samplings cover T 0.6–1.3, top_p 0.8–1.0, top_k 0 / 20, min_p 0 / 0.05 and a seed above 2^63. The output is `$REF/samp/trace.jsonl`.
    - **Then `tools/zig/dsv41_samp/samp_step.sh`**, about 13 min (each step capped at `left()`, at most 12):
      1. `tf-dsv41-samp` on GPU 0 (under 1 min): the device steps against the host reference. Expected: `PASS tf-dsv41-samp: 0 checks failed`.
      2. `tf-dsv41-m1 generate` on the trace, `TF_DSV41_SAMPLING=1`, `TF_DSV41_DRAFTS=0` then `1` (about 6 min each: the Zig load, then 6 x 64 tokens). It compares Zig's sampled replies with Python's, token by token. Expected in each: `PASS M4 CLI == HTTP: 6/6 replies equal` (here it means Zig == Python).
    - **Assets:**
      - the R0 reference (`aot/` with the sampled lengths, `rope.bin`, `engram-host.bin`; no R1);
      - the reference's `TF_DSV41_EXPERT_TOPP` (`M2B_TOPP`), inherited by the Zig side;
      - **the fatbin copy:** before `build-pod.sh`, `cp zig-dsv41-scratch/samp/fatbin/dsv41_sampling.fatbin zig-dsv41-scratch/zig-out/fatbin/` (nvcc 13.3, the build's flags, sm_120 + sm_121). Without it the build still succeeds but `tf-dsv41-samp` and the sampler refuse to load (`SamplingKernelNotBuilt`).
    - **Standalone** (no `--sampled`): `samp_step.sh Z LP R0 HI RES LOCAL` runs `dsv41_samp_ref.py` itself (about 10 min more). With the Zig prefill it then needs `37,50,66,83,128,292` in `M2B_AOT_PROMPTS`, or `SAMP_OWN_PREFILL=0`.
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
  - **Graphs in M4:** the engine default stays off (the default-off rule above; 93367c6 reverted in 4893cdf). The kit's `run_host.sh` and `job.sh`'s `M2B_M4` set `TF_DSV41_GRAPHS=1`. `Forward.graphs_default` (`graphs.Settings.fromEnvOr`) is in place for the integrator to flip in `model.zig`: every rank loads through it, so the ranks agree.
    - With the pool: `reserve()` syncs the tables before the replay, and `kx_dense` is device-only with persistent owner lengths, so graphed windows work with the paged pool and split KV as built. Nothing here changes that.
    - Slots: `Model.openRows` sets `graphs_read`, so one-slot graphs stay off with several slots. Keys are one slot's (rows, context bucket); row mode brings its own.
  - **Defaults:** the Engram shards (`TF_DSV41_ENGRAM_DIR`), read-ahead (only with shards) and phases are off unless set. The greedy staging is always on but picks the same tokens by construction (gated by every M2b run). With all of it off: `check` cap24-1 / cap24-12 / cap24-16 / cap15 / cap18 / cap22 and `dcheck cap16` give 0 mismatches (HEAD 4893cdf).
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
  - **Hooks** (one commit each): `forward.zig` (c61dd88: fields, greedy, the Engram source and read-ahead, timers, `fromEnvOr`); `model.zig` (93367c6, reverted by 4893cdf); `serve_engine.zig` (c9e427e: the phase decorators); `deepseek_v41.zig` (dc8e3d9: the test list); `dsv41_m2b_ref.py` (25950ea: `--engram-dir` / `--engram-make`). `graphs.zig`'s hook (`Settings.fromEnvOr`, 11 lines) went in with bc55ef4 rather than its own commit. `profile.zig` is untouched: the phases sit beside it, and the timing table reads its output.
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
    3. **`M2B_M4=1`**, with two changes to `m4env` (keep `TF_DSV41_GRAPHS=1`; `server.log` shows `graphs: on`):
       - add `TF_DSV41_PHASES=1 TF_DSV41_PHASES_OUT=$m4dir/phases.json`, and `TF_DSV41_ENGRAM_DIR` with step 1;
       - run the `generate` replays with `TF_DSV41_ENGRAM_PREFETCH=0`.
       `PASS M4 CLI == HTTP` then also proves R3's read-ahead on == off. `drafted == serial` holds as before.
    4. **`bash $Z/tools/zig/dsv41_perf/speed_step.sh $Z $RES`** (about 10 s) writes `RES/speed-gap.md`: the per-window table (rows 1 / 2 / 4 / 6: Python R1, Zig eager, Zig graphed) and the GPU ms by family. It also greps the `graphs: on` and `engram:` lines.
    - **The combined pod's timing table vs Python R1** is `speed-gap.md`, made only from these existing `job.sh` steps:
      - `m2b-ref-r1` with `M2B_PROFILE_ROWS="1 2 4 6"` gives Python's `prof-rank*.json`;
      - `m2b-r1` (`M2B_PROF=1`, `M2B_ZIG_PROF_ROWS`) gives Zig's eager per-call profile and window times;
      - `m2b-r1-graphs` (`M2B_R1_GRAPHS=1`) gives Zig's graphed window times;
      - `M2B_M4` with `TF_DSV41_PHASES` gives Zig's per-round phases.
      The pod has no Python serving run, so a per-round phase comparison against Python waits for the Spark window.
    - **Assets:** none beyond the M2b reference's (the shards are generated in `$LOCAL/engram`, a few MB written, sparse).
  - **Spark window:** the kit sets `TF_DSV41_ENGRAM_DIR` to the node's shards (`HEAD_ENGRAM` / `WORKER_ENGRAM` of prod-perf1.env) on both ranks.
    - M2c-2's reference takes `--engram-dir` with the same path (the real-row gate on the whole model).
    - The served comparison: `bench_http.py` on Zig with `TF_DSV41_PHASES=1`, and Python's m2bench with `TF_DSV41_PHASES=1` on the same prompts, then `gap.py phases`.
- `dsq-zig-trees-1` (draft trees on the GPU target; design below):
  - **`M2B_LANES=4`** (`M1_JOB=m2b`, 2 GPUs, about 4 min after the shared load): lanes with first-position trees (2 siblings) and the tree oracle's drafts over the GPU target at TP=2. Rank 0's tokens must equal the reference's serial greedy tokens, with trees > 0 and 0 < sibling wins < trees. Expected lines: `PASS M3 lanes mode 4 rank 0: tokens 64/64 equal, drafted N, accepted M, trees T, sibling wins W, ...` and `PASS M3 lanes mode 4 rank 1`. `m3-4/m3-rank0.json` also records `chains` and `rerun_rows`. No new assets.
  - **`M2B_LANES=3` with `TF_DSV41_TREE=3`** (about 4 min): DSpark's own siblings (gathered `[block, 128]` candidates) through the same path. The gate is tokens equal and drafted > 0. On the 0–24 prefix the taps are zeros, so the tree counts are informational. Run it as the mode token `3t` in `M2B_LANES` (for example `M2B_LANES="1 2 3 3t 4"`): `job.sh` runs it as `M2B_LANES=3 TF_DSV41_TREE=3`, port 29733, results in `m3-3t`.
  - **Pool and split with trees:** the `M2B_M4=1` step again, with `M2B_M4_ENV="TF_DSV41_TREE=3 TF_DSV41_POOL_TOKENS=<the M5 step's> TF_DSV41_KV_SPLIT=1"`. It checks HTTP == CLI with drafts, and drafts == serial. Trees use no private pages (below), so this checks `Forward.drop` against the pool's reserved pages and split's exchange.
  - **Spark window (whole model):** M4 with `TF_DSV41_TREE=3` against Python prod's engine with `TF_DSV41_TREE=3` on the same 8 prompts, at the same calibration table. Tokens must be equal. Drafted, accepted, rounds, trees and sibling wins should be close to Python's, not identical: the host `_chain` order is fp32 (as for chains).
  - **Draft pricing from prod's calibration (`draft/calib_env.zig`):**
    - On the pod, in the `M2B_M4=1` step (no extra minutes), add `M2B_M4_ENV="TF_DSV41_CALIB_FILE=/workspace/<prod's calib-*.json>"`. The asset is prod's newest `/cache/dsv41-calib/calib-*.json`, copied read-only from the Spark cache volume `dsv41-tf-cache`, with world 2, DSpark on and 4 slots.
    - `m4/server.log` must show `dsv41 draft costs: cached <that file>, verify 1 / 16 rows <ms> / <ms> ms, pass <ms> ms`, with the file's values after the microsecond round trip. HTTP == CLI must still hold, with the same drafted counts, because the requests are priced from the same table.
    - Without the knob the line reads `default`, which is today's behaviour.
  - **Spark window, the pricing check:** run Zig's M4 with prod's own knobs, `TF_DSV41_CALIB=real TF_DSV41_CALIB_DIR=/cache/dsv41-calib`, and the cache volume mounted. It reads the table prod's last boot stored. Then compare drafted, accepted and rounds on the 8 prompts against prod with drafts on, at the same seed and knobs.
    - Tokens must be equal.
    - The counts should be close, not identical, because the host chain sums in a different fp32 order. They were not comparable before this, since Zig priced from `depth.default_costs`.
  - Host: `zig build test-dsv41-draft`:
    - `branches_test.zig`;
    - `calib_env_test.zig`, against `fixtures/calib-golden.json` from `tools/calib_golden.py` run on prod's 8474f31 `calib.py` / `depth.py`. It checks the same entry picked from a directory, the same shared table bit for bit, and 400 depth / tree choices equal.

- `dsq-zig-spec-1` (the speculative DSpark pass, `TF_DSV41_SPEC_DRAFT=1` as prod runs it, and `TF_DSV41_SPEC_NUCLEUS`; design below):
  - **Knob / command:** `M2B_SPEC=1` inside `M1_JOB=m2b` (job.sh's gate loop sources `tools/zig/dsv41_spec/job.sh`, `spec_gate`). No Python twin, no second Python load. It runs after the other gates and reuses them.
  - **Runs and what each compares** (each prints `PASS spec <run>: ...` / `FAIL spec <run>: ...` into `$RES/spec-<run>.log` and `steps.txt`):
    1. `spec-m3-3`: lanes mode 3 (DSpark's GPU pass: `begin` / `collect` on rank 0, the whole pass on rank 1) with the knob on. Tokens must equal the reference's serial greedy tokens (Python's). Drafted and accepted must equal the knob-off run's (`$RES/m3-3` when `M2B_LANES` held 3, else its own `spec-off-3` run). At least one speculation must be used (`dsv41 spec: speculative DSpark passes L launched, H used, ...` in rank 0's log).
    2. `spec-m3-3t` (DSpark's trees, `TF_DSV41_TREE=3`) and `spec-m3-4` (the tree oracle), knob on: their lanes PASS must still hold. Tree rounds are not speculated; the chain rounds around them are.
    3. `spec-m4` (with `M2B_M4=1`): `generate` replays the M4 server's trace (recorded knob off) with drafts and the knob on, in M4's `m4env`. `PASS M4 CLI == HTTP` then means tokens and drafted counts equal the knob-off server's; M4's own `m4-serial` is drafted == serial.
    4. `spec-samp` (opt-in, `SPEC_SAMP=1`, needs `M2B_SAMP`'s reference and the sampling fatbin): the sampled replies with drafts, `TF_DSV41_SPEC_DRAFT=1` and `TF_DSV41_SPEC_NUCLEUS=1` (top-p-without-top-k rows speculate) must equal Python's sampled replies.
  - **Minutes:** about 5 a lanes run (the Zig load plus 64 tokens), plus 5 for `spec-off-3` when mode 3 did not run. `spec-m4` takes about 6–8. The default (`SPEC_RUNS="3 3t 4"`, `spec-m4`) is about 20 min; `SPEC_RUNS="3 4"` fits 15. `SPEC_SAMP` adds about 8.
  - **Assets:** none new. It uses the M2b reference (`$R0` / `$REF`: `ref.json`, `aot/`, `rope.bin`, `engram-host.bin`), no new prompt lengths, plus M4's trace and env when `M2B_M4` ran.
  - **Python's acceptance and tokens a round:** the pod runs layers 0–`$hi`, without DSpark's target layers, so its acceptance means nothing against Python's. The comparison is a Spark-window item. Run Zig's M4 with prod's knobs (`TF_DSV41_SPEC_DRAFT=1`, prod's calibration) on the 8 prompts. Read drafted, accepted and rounds from the trace, and the `dsv41 spec:` line at the server's close. Compare them with prod's own run of the same prompts (m2bench's per-round counts and its `describe()` line). Tokens must be equal. Drafted and accepted should be close (the host chain sums in a different fp32 order) and unchanged by the knob, which the pod proves exactly.
  - **Host gate:** `zig build test-dsv41-draft` (`spec_test.zig`). It covers:
    - the decisions replayed against Python's `spec.py` (`fixtures/spec-golden.json`, from `draft/tools/spec_golden.py` on prod's 8474f31): 560 operations, with every rate, probe, hit and miss reason equal;
    - on the CPU twin, through `propose` and through `begin` / `collect`: speculated == unspeculated == serial (greedy, keyed, shared rounds over 3 slots, nucleus rows with and without the knob, trees), the same drafted / accepted / kept counts, and passes and ingests accounted exactly.
  - **Default off:** with the knob unset, nothing launches early and no emitter is touched. `check` cap24-1 / cap24-12 / cap24-16 / cap15 / cap18 / cap22 and `dcheck cap16` give 0 mismatches.

**The speculative DSpark pass (`draft/spec.zig`, `dsq-zig-spec-1`).** Python's spec.py starts the next round's pass right after a verify window, before the hosts sample, commit and plan. Here the window's choices are on the host when `Target.window` returns, so Lanes already knows each DSpark window's accepted count, bonus token and next start; the GPU still drafts while the lane core commits, emits and plans. The decisions are Python's, replayed bit for bit:
- **Launch (after a window):** an unused speculation is dropped first. A window is a candidate when its slot's last pass held drafts and the window carries no host drafts. Its slot must also be `wanted`: the decayed hit rate (0.8) at least 0.3, or a probe every 8 skips. Rounds with a tree window are not speculated, and nor are rounds with a top-p-without-top-k row unless `TF_DSV41_SPEC_NUCLEUS=1`. The kept rows (0..accepted) are ingested at once, then `Pass.begin` launches the pass at (bonus, start + accepted + 1).
- **Commit:** the ingest is skipped when the kept path is the speculated one (`ingested`). Otherwise the commit ingests as before.
- **Draft:** `take` hands the speculated proposals over when the round drafts exactly the speculated slots at the same (anchor, start, keyed params) with every commit matched. Otherwise the pass runs as before.
- **Exactness:** a used speculation is the pass the round would have run on the same rings. Drafts only propose.
- **The device contract** (`iface.Pass.begin` / `collect`, optional):
  - `GpuPass.begin` sends `op_ds_propose` (the followers run the whole pass, as before) and launches the pass up to its host step (`glue.ds_out`). `collect` runs the host step (its gather pairs with the followers') and the chain.
  - Between the two, only work without collectives may follow on the stream: the target's `keep` (carry kernels and `op_keep`). Lanes collects before any window, ingest, pass, prefill or release.
  - A pass without `begin` (the multi-slot pass until it adds one) is run through `propose` at launch, which is still exact, just without the overlap.
  - One pass serves every speculated slot, so `dsq-zig-mdraft-1`'s multi-slot pass gets the overlap by implementing `begin` / `collect` over several asks.
- **Hooks** (one commit each): `dspark_gpu.zig` (`begin` / `collect`), `serve_engine.zig` (the knobs, and the counts line at close), `phases.zig` + `phases_timed.zig` (`spec`: the launch; the collect counts as `draft.pass`), `core/lanes/engine.zig` (the first round, below).
- **The first round** (M3 open item 2): the lane core asked the prompt's first draft request at the lanes' prior depth (`Rule.depth`). Python's planner asks every DSpark round at the cap (`lookup.Planner.window`: `min(cap, room)`), the first included, and the depth then cuts from the pass's confidences. The core now asks a head with node probabilities its whole reach in the first round too (`headDepth`, as every later round), so the policy picks the first window's k from a full block. Only DeepSeek's lanes set `draft_probabilities`.

- `dsq-zig-mdraft-1` (DSpark drafts over several live slots, `TF_DSV41_SLOT_DRAFTS=1`; section 0c):
  - **Knob / command:** `M2B_MDRAFT=1` inside `M1_JOB=m2b` on the 2-GPU pod, after `M2B_SLOTS=1`. job.sh sources `tools/zig/dsv41_mdraft/job.sh` and runs `mdraft_gate`, with slots_gate's env: `Z`, `LP`, `LOCAL`, `RES`, `hi`, `step` / `left`, and `SLOTS_PYSRC` / `TORCH_EXTENSIONS_DIR`, which mdraft does not use.
  - **No Python load.** Drafted == serial holds by exact verification and keyed sampling, so Python's batched 4-slot output is the slots gate's reference:
    - `$LOCAL/slots-ref/trace.jsonl`: 4 greedy replies, prompts 45 / 173 / 9 / 640, Python's batched row mode;
    - with `M2B_SAMP`, `$REF/samp/trace.jsonl`: 6 keyed sampled replies.
  - **Runs:** `mdraft-s4-eager` (`TF_DSV41_GRAPHS=0`) and `mdraft-s4-graphs` (`=1`). Each is `ranks.sh` running `tf-dsv41-m1 mdraft` on rank 0 and `follow` on rank 1, with `TF_DSV41_SLOTS=4 TF_DSV41_SLOT_DRAFTS=1 TF_DSV41_DRAFTS=1 TF_DSV41_POOL_TOKENS=18432 TF_DSV41_LAYERS=0-$hi`, plus `TF_DSV41_SAMPLING=1` when the sampled trace is there. Every line is a request with drafts on: first all at once (batched), then each alone.
  - **Expected in each `$RES/mdraft/<run>/rank0.log`** (also in `$RES/mdraft-<run>.log`):
    - `PASS mdraft batched == Python: N/N`, `PASS mdraft alone == Python: N/N` and `PASS mdraft batched == alone: N/N`, where N is 4, or 10 with the sampled lines;
    - `PASS mdraft several slots a pass` (some batched pass held 2–3 slots);
    - `mdraft stats batched / alone`: drafted, accepted, acceptance, stream rounds, tokens a round, passes, slots a pass.
    - eager against graphs is graphs off == on.
  - **Acceptance vs Python:** on layers 0–24 the drafter's taps are zeros, so the pod's acceptance is not the model's. The pod proves the tokens. Acceptance and tokens a round against Python's batcher are the Spark window's run (section 0c).
  - **Minutes:** about 7 a run (the Zig load, then 10 requests batched and 10 alone at 64 tokens). Both runs come to about 14 (`MDRAFT_MIN`, default 7, caps each). `MDRAFT_RUNS=s4-graphs` alone is about 7.
  - **Assets** (nothing new to make): a copy of the R0 reference's assets (`${R0:-$REF}`: `aot/` with DSpark's set P from the capture merge and the sampled prompts' lengths, `rope.bin`, `engram-host.bin`), with `$LOCAL/slots-ref/aot` (the row-mode buckets) merged in by `dsv41_aot_merge.py`. The copy is `$LOCAL/mdraft-assets` and is removed afterwards.
    - The 2- and 3-slot passes need no Triton variant beyond set P's one-slot ones: `dspark_emit.Layout` keeps every launch's specialization (`test-dsv41-mdraft`). `MDRAFT_GROUP=1` (`TF_DSV41_SLOT_DRAFT_GROUP=1`) is the fallback, one pass a slot.
    - The sampled lines need the `dsv41_sampling` fatbin in the build (samp's note).
  - **Default off:** with `TF_DSV41_SLOT_DRAFTS` unset, drafts with several slots stay refused, as before. On HEAD 6e02ca2 plus this work, `check` cap24-1 / cap24-12 / cap24-16 / cap15 / cap18 / cap22, `dcheck cap16` and `m5-emit cap18` give 0 mismatches.

**Draft pricing (`draft/calib_env.zig`).** The served engine prices drafts as prod does, with Python's knob names. Followers price nothing: the lanes run on rank 0 only.
- **Modes (`TF_DSV41_CALIB`):**
  - `cached` (the default) reads Python's stored `calib-<key>.json` from `TF_DSV41_CALIB_DIR`, by default `~/.cache/tensorfold/dsv41-calib`.
  - `real` is prod's mode: it measures at every boot and stores the table. Zig does not measure, so `real` reads the newest stored table.
  - `default` uses `depth.default_costs`.
- **Picking the entry:** Python's key hashes its own image digest, which Zig cannot reproduce. So the entry is picked by the `meta.shape` stored with it:
  - the same world and DSpark setting;
  - at least as many slots as ours (prod's 4-slot table prices a 1-slot engine's rows the same);
  - then, in order: the same slots, the same context, the newest.
  - `TF_DSV41_CALIB_FILE` names one entry outright. It is Zig's own knob, with no Python twin.
- **The values:** the table goes through `Costs.encode` / `decode` (microsecond ints, half to even), as rank 0 shares it. The depth then prices with prod's exact bits.
- **Depth knobs:** read with Python's names: `TF_DSV41_DEPTH`, `_DRAFT_DEPTH`, `_DRAFT_SKIP`, `_DEPTH_JOINT`, `_DEPTH_FAIR`, `_DEPTH_HOST_MS`, `TF_DSV41_GRAPH_ROWS_MAX`.
- **No matching entry:** today's 64-row defaults are used.
- **Hook:** `serve_engine.zig`, one commit. The shape passes the served slots (`m.slots`).

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

- `dsq-zig-slots-1` (several live slots, row mode; section 0b):
  - **Knob / command:** inside `M1_JOB=m2b` on the 2-GPU pod, after the pack copy: `source "$Z/tools/zig/dsv41_slots/job.sh"; slots_gate`. Needs `SLOTS_PYSRC`, a Python tree with `rowgraphs` / `slots` (the decode1 twin's src and its `TORCH_EXTENSIONS_DIR`).
  - **Compares:**
    - Zig's served replies (`tf-dsv41-m1 slots`, through the lanes) at `TF_DSV41_SLOTS` 4 / 4 / 2 / 1 with graphs off / on / on / off, against the Python engine's batched row-mode output (`ref.py`: 4 slots, prompts 45 / 173 / 9 / 640, 64 greedy steps).
    - Each slot alone against batched.
    - Expected in each run's `rank0.log`: `PASS slots batched == Python: 4/4`, `PASS slots alone == Python: 4/4`, `PASS slots batched == alone: 4/4`.
  - **Minutes:** about 14 for the reference (one TP=2 load of layers 0–`$hi`), then about 4 x 5 for the Zig runs.
  - **Assets:** made by the step itself. `ref.py` writes `trace.jsonl`, `rope.bin` and `aot/`, including the row-mode Triton variants of every bucket up to 16; `dsv41_engram_host.py` writes `engram-host.bin`. No capture is needed.
  - **Default off:** `TF_DSV41_SLOTS` unset is today's path. On 8f40a20, `check` cap24-1 / cap24-12 / cap24-16 / cap15 / cap18 / cap22, `dcheck cap16` and `m5-emit cap18` give 0 mismatches.

- `dsq-zig-longpf-1` (long prompts up to the 1M limit; `longpf.zig`):
  - **What it ports** (tensorfold-decode1 8474f31 `csa2/stream_topk.py`, `backend._stream_select`; dsv41-kvsplit / midprofile `kvsplit.plan`, `blocks._attend_blocks`):
    - **Stream top-k.** From `TF_DSV41_INDEX_STREAM_MIN` (prod 4,096) visible keys, a full-mode layer without the candidates' scores (2 / 8 / 14) selects as Python does: per row block of `stream_topk.plan` (scratch under its own 256 MiB BUDGET), the Triton `_stream` (program = row x 16,384-key split) and the glue `stream_merge`. The merge gathers each split's first K keys (`kvGather` over every other K-slot row; one split: none) and runs `index.top_positions` over them. Keys are unique, so the set is Python's whatever the split order. No new kernel: `_stream` comes from the reference's AOT set, the rest are prefill glue kernels gated on pod 20.
    - **Split KV past the union cap.** `kv_state` keeps a union's row blocks (planUnion's halves) instead of `error.UnionSplit`. `longpf.window` runs a segment's calls and gives each `_fused` attention that reads `w.kx.recv` (the index layer's and its reuse layers') block by block: block j's rows exchanged, then the same launch over rows [a, b) (Q / TOK / CNT / LO / OUT rows moved, POS = start + a, every constexpr and OR unchanged: the same AOT variant).
    - **The served limit.** `TF_DSV41_CONTEXT` (Python's context; prod 1,048,576) sets the served model's slot limit (default 4,096 as before) and its rope rows (limit + 2,048; `rope.bin` must hold them: `dsv41_rope_tables.py --rows`, else `error.RopeTooShort`). The M2b driver takes the reference's `limit` from `ref.json`.
  - **Hooks** (one commit each): `block_prefill.zig` (`streamSelect`), `forward_prefill.zig` (`stream_merge`, `longpf.window`), `kv_state.zig` (`blocks`, `fetchBlock`), `model.zig` (`TF_DSV41_CONTEXT`), `m2b.zig` (the reference's limit), `zig/build/cuda.zig` (`test-dsv41-longpf`).
  - **Host gate:** `zig build test-dsv41-longpf`, 5/5 on the real emitter (release config, q28-v2 widths, 1M limit, contiguous / paged / split): a 2,048-row segment at the limit's end streams on every full-mode non-candidate layer in two row blocks (512K visible keys: 32 splits, 1,024 rows a launch), scratch within the budget, no `_keys` there; below the threshold nothing changes; a split segment's exchanged attentions split into row blocks moving only the row-indexed tensors.
  - **Default off:** short prompts emit exactly as before. `check` cap24-1 / cap24-12 / cap24-16 / cap15 / cap18 / cap22 and `dcheck cap16`: 0 mismatches; `m5-emit cap18`: 8 programs, 0 problems (on HEAD + these commits alone).
  - **Pod step** (`M2B_LONGPF=1`, TWO GPUs inside `M1_JOB=m2b`; `tools/zig/dsv41_longpf/job.sh`, `longpf_gate`):
    - **Python twin:** the midprofile twin (`midprofile.tar.gz` in `$Z` or `dsq/m5/`, unpacked to `/opt/midprofile` as `M1_JOB=m5` does; its prefill is 8474f31's, and split KV is its code). The optional `prod` run uses `/dsv41-tf/src`.
    - **Runs** (`LONGPF_RUNS`, default `paged split`): `longpf-ref-paged` / `longpf-ref-split` (`dsv41_m2b_ref.py --pool` / `--split --compact 0`, prompts `LONGPF_PROMPTS` = 8,192 / 32,768 / 131,072 in one load, `--limit` = the longest + 2,048, `LONGPF_STEPS` 16 steps, prod-knobs.env with expert top-p off, `TF_DSV41_KV_SPLIT_UNION_MIB=8` so the long prompts' unions split into row blocks on both sides), then `longpf-paged` / `longpf-split` (`tf-dsv41-m1 m5 paged|split` through `m5_ranks.sh`, `TF_DSV41_CONTEXT` = the reference's limit). Each compares the pool's rows in logical order (a split rank's owned rows), the SWA rings and carries, position, Engram tail and first token, then every step's token and logits digest. `prod` adds the contiguous slot on prod's twin through `M2B_PREFILL=1`, a prompt each.
    - **Verdicts:** `PASS longpf <run> ...` / `FAIL longpf <run> ...` in `$RES/longpf-gate/summary.log`, with the Zig exchanges' counts and Python's union log lines.
    - **Minutes** (estimates): a reference about 6, a Zig run about 4; the default runs about 20 min (`LONGPF_PROMPTS="8192 32768"`: about 11).
    - **Assets:** made by the step. The references write `rope.bin` (limit + 2,048 rows) and `aot/` (with `_stream`); `engram-host.bin` is the M2b reference's (`$LOCAL/m2b-ref/engram-host.bin`) or made by `dsv41_engram_host.py`, copied into each `p<N>/` under that exact name.
  - **Only the Sparks can check:** 512K-1M prompts on the whole model (a 1M reference does not fit the pod's time or a PRO 6000's 25-layer budget at both modes), prod's own `TF_DSV41_INDEX_BUDGET_MIB=64` (Zig keeps 256: the materialised selection's row blocks differ, its rows do not), and prefill memory at 1M (`L.ix.stream` is 256 MiB of the segment arena; `kv.budget`'s prefill price does not count it yet).
  - **Spark kit note:** the served model needs `TF_DSV41_CONTEXT` and a `rope.bin` of that many rows + 2,048, and DSpark's AOT set must include `_stream` (an AOT prompt over 8,192 tokens), or long prompts fail with `MissingTritonVariant`.

- `dsq-zig-ops-1` (serving and ops parity with prod's Python server; section 0d):
  - **No pod step.** Everything host-checkable is gated on host: `zig build test-server` with `TF_DSV41_MODEL` (the release tokenizer) runs `wire_test.zig`. That test checks 49/49 exchanges and 28/28 request-log lines equal to prod's own server, recorded at 8474f31 by `tools/zig/dsv41_ops/http_golden.py`. `test-dsv41` covers `prepared.zig`.
  - **Spark window steps:** `tools/zig/dsv41_ops/WINDOW-STEPS.md`, about 55–75 min, not run. In order:
    1. build the Zig image natively on dgx-01;
    2. `scripts/v2-rollback.sh zig` (the first boot writes the prepared files);
    3. a restart, for the fast-boot time;
    4. `wire_probe.py` against Zig, then against Python;
    5. the request log and disconnect;
    6. a fail-fast kill of rank 1 with the watchdog's heal;
    7. `scripts/v2-rollback.sh perf1`.
  - **Assets:** the kit's `m4-assets` on both nodes (`HEAD_ASSETS` / `WORKER_ASSETS` of `config/prod-zig.env`), and the gated sm_121 fatbins for the image's build context.

- `dsq-zig-knobs-1` (prod knob parity; section 0e): `tools/zig/dsv41_knobs/job.sh`, `knobs_gate` inside `M1_JOB=m2b` after the M2b reference (`M2B_KNOBS=1`, the integrator's hook). Six runs of the `M2B_PREFILL` gate with one prod knob each on the Zig ranks (fused dense prefill, 1,024- / 512-row segments, 64 MiB index budget, Engram read-ahead, all together): 65/65 roles and 64/64 tokens each, about 18 min, no Python load. Details in 0e.

### Pod 26, the combined gate (2026-10-07, 57 min, $3.97)

The tree was `798b4f5`, built from a clean archive. The fatbins were the sm_120 base set plus `dsv41_kvsplit` and `dsv41_sampling`. Logs are in the quant repo, `results/20261007-comb26-pod26`.

| gate | result |
|---|---|
| set P capture: `check`, `dcheck` | 0 mismatches |
| M2b decode (prompts 45, 173; prod's knobs, top-p 0.85, R3 Engram shards) | PASS |
| Zig prefill gate, 45 and 173 rows | PASS |
| lanes, mode 4 (chains) and mode 3t (trees) | PASS |
| M4, R3 Engram, graphs, phases: CLI replay == HTTP, drafts on | 8/8 (1,769 drafted; acceptance 0.003, as expected: the drafter's taps are zero on the 0–24 prefix) |
| M4, drafts off: the same HTTP trace | 8/8, so drafted == serial |
| keyed sampling: `generate` on Python's 6 sampled replies (T 0.6–1.3, top-p / top-k / min-p, seed above 2^63) | 6/6 == Python |
| M5 references (pool, split, compact, park) and `m5-agree` (split == replicated, park == uninterrupted, in Python) | PASS |
| M5 Zig gates: split, park, compact | not run: `FileNotFound` on `engram-host.bin` (fixed in `836dd47`), then out of budget |

- **Served phases (M4, 1,008 rounds):** 6.07 ms a window on the 25-layer prefix. The host's part (forward 0.108 ms + graph stage 0.041 ms + replay launch 0.060 ms) is about 0.21 ms of it.
- **The pod's own bugs, now fixed:**
  - staging did not ship `tools/zig/dsv41_perf` / `dsv41_slots` / `dsv41_samp` or `tf-dsv41-samp` (the quant repo's `stage-zigm1.sh`);
  - a failed reference ended job.sh on `R0` (`068f8c7`);
  - M4 waited on `serving model` while the server prints `serving pack at` (`b886a4b`).
- **Still to run on the fix-up pod (cap $5):**
  - M5 split / park / compact on pod 26's references (`M5_REFS_FROM=/workspace/dsq/m5-refs-26`);
  - the slots gate (`M2B_SLOTS=1`) and `samp-slots` (sampling at 4 slots, `798b4f5`);
  - R1 and the timing table against Python R1 (`M2B_R1=1 M2B_R1_GRAPHS=1 M2B_PROFILE_ROWS`);
  - M4 at prod's calibration (`M2B_M4_ENV=TF_DSV41_CALIB_FILE=...`; prod's tables, copied read-only, are in `dsv41-zig-ref/calib-prod/`);
  - multi-slot drafting (`M2B_MDRAFT=1`, `dsq-zig-mdraft-1`).
- **The gap-closing workstreams** (`dsq-zig-longpf-1`, `dsq-zig-spec-1`, `dsq-zig-grammar-1`, `dsq-zig-vision-1`) each ship `tools/zig/dsv41_<name>/job.sh` with a `<name>_gate`. The gate runs inside `M1_JOB=m2b` with slots_gate's environment, behind `M2B_LONGPF` / `M2B_SPEC` / `M2B_GRAMMAR` / `M2B_VISION` (`7618d35`). The ones not ready for the fix-up pod go on a second batched pod. The Spark window does not wait for them.

- `dsq-zig-grammar-1` (structured output, `TF_DSV41_GRAMMAR=1`; design in "Structured output" below):
  - **Pod step: `M2B_GRAMMAR=1`** (`M1_JOB=m2b`, 2 GPUs): `source "$Z/tools/zig/dsv41_grammar/job.sh"; grammar_gate`. Runs after `M2B_SAMP` (it reuses its AOT prompt lengths and the sampler's fatbin).
    1. **grammar-ref** (about 3 min on the reference's own load with `dsv41_m2b_ref.py --grammar`, not built yet: a 3-line hook like `--sampled` calling `grammar_ref.run_cases(fw, cfg, rank, out/"grammar", pack, 48)`; standalone about 9 min with its own TP=2 load). Prod's grammar (`structured.Dsv41Grammars` + GLM `Constraint` over xgrammar 0.2.8) masks serial decoding of 6 cases: json_schema greedy / sampled, tool_choice required greedy, a named tool sampled, strict tools greedy, json_object sampled. Prompts 37, 50, 66, 83, 128 and 292 random ids (the sampled gate's lengths), 48 tokens. Output: `trace.jsonl` with each line's `structure`.
    2. **grammar-drafts / grammar-serial** (about 5 min each): `tf-dsv41-m1 generate` on that trace with `TF_DSV41_GRAMMAR=1 TF_DSV41_SAMPLING=1`, drafts on then off. Expected in each: `PASS grammar drafts: Zig == Python: 6/6 replies equal` (generate's line, rewritten by job.sh).
    3. **grammar-http** (about 6 min): `tensorfold-dsv41 serve` with `TF_DSV41_GRAMMAR=1` answers `requests.sh`'s 6 chat requests (response_format json_schema / json_object, tool_choice required / named, strict tools; greedy and seeded), each checked against its grammar (`PASS grammar http: 6/6`), then `generate` replays the server's trace under the recorded grammars: `PASS grammar http: CLI == HTTP: 6/6`. The logs also show `structured output: on (xgrammar 0.2.8 core, ...)`.
    - **Assets:** the M2b R0 reference's (`aot/` with the six lengths, as `M2B_SAMP=1`'s `--sampled` writes them, `rope.bin`, `engram-host.bin`); the pack's `tokenizer.json` (in `LP`); the `dsv41_sampling.fatbin` copy as for `M2B_SAMP`. The Python side needs `xgrammar==0.2.8`: prod's E1E2 image has it, and the step installs it with `--no-deps` when it is missing.
    - **Minutes:** about 14 with `--grammar` on the reference (16 with the standalone reference). `GRAMMAR_MODES` trims it ("drafts http": about 9).
  - **Host gate:** `zig build test-grammar` (5/5). The goldens come from `zig/tests/grammar/gen_goldens.py`, which runs prod's own `grammar.py` / `structured.py` at 8474f31 over xgrammar 0.2.8's Python package; they are committed under `zig/tests/grammar/fixtures`. Both tokenizers pass: the synthetic one (always) with 589 masked rows, and DeepSeek-V4.1's (`TF_DSV41_MODEL`) with 667. Every row's bitmask equals xgrammar's (sha256 and allowed count), and so does every window's cut (keep, first constrained row, stop). The checks also cover the vocabulary views, the tokens never allowed, the think gate, the DSML structural tag JSON byte for byte, and the refusal messages. The masks on logits: `rows.maskRows` against the per-token rule on both ranks' halves.
  - **Knob off** (the default): no grammar object exists, nothing is loaded or sent, and requests with structured output are refused as before. `check` cap24-1 / cap24-12 / cap24-16 / cap15 / cap18 / cap22 and `dcheck cap16`: 0 mismatches.

- `dsq-zig-vision-1` (image input, `TF_DSV41_IMAGES=native`; prod's G15a-G16 path, public repo docs/IMAGES.md):
  - **What it is.** Prod's `vision_prep.py` + `vision.py` (8474f31) in Zig, no torch.
    - **Host side** (`deepseek_v41/vision/`, module `dsv41_vision`, re-exported as `serve.vision`): image parts (OpenAI `image_url` object / string, `input_image`, `image`, Anthropic `source`, as `part_url`), data: URLs (`b64decode(validate=False)` as binascii, percent-encoded), Pillow 12.3's decoding (PNG every mode / bit depth / Adam7 / tRNS; JPEG baseline, extended and progressive Huffman with libjpeg-turbo's islow IDCT, fancy upsampling and YCbCr / RGB / CMYK / YCCK as Pillow asks), `to_rgb` (alpha over white as `ImagingAlphaComposite`), `ImageOps.pad` with Pillow's bicubic `ImagingResample`, `plan_grid`, the bf16 patches, the digest (`Settings.key` as `json.dumps(sort_keys)`), GLM's `derive` / `Registry` virtual ids, the sentinels / `splice` / `expand`, and rank 0's prepared-image cache (`TF_DSV41_VISION_PREP_MB`, `_MAX_IMAGES`, `_MAX_TOKENS`, `_MAX_BYTES`, `_MAX_PIXELS`, `_PREP_SLOTS`).
    - **Server** (`deepseek.zig`): `TF_DSV41_IMAGES=native` renders each image part as a per-request sentinel, splices the placeholders in prompt order, prepares the images and expands each placeholder id to its span of virtual ids (`/tokenize`: the image token at every position). A typed `<｜deepseek_image｜>` stays text. The prepared images ride `api.Request.images` (`vision/held.zig`) and are held until the reply ends.
    - **Engine** (`vision_rows.zig`, `vision_tower.zig`, `vision_kernels.zig` + `zig/kernels/cuda/deepseek_v41_vision/vision.cu`, own optional fatbin `dsv41_vision`): rank 0 copies a request's images at submit and encodes each image once on the tower (BF16 from the pack's `native.safetensors`, 0.97 GB; span rows cached by digest, `TF_DSV41_VISION_CACHE_MB`), then writes the image positions' rows into its embedding partial after `embed_send` (zeros there on every rank), so rank 1 gets them through the all-sum (row + 0: exact). Every rank feeds Engram's `_fuse_dec` the keep vector (0 at image positions, `HAS_KEEP`) in prefill segments that hold one; the n-gram hashes already treat ids past the vocabulary as dead. The tower follows torch's op order: `_rms` as at::native's inner-dim reduce (32 lanes, 4 float4 accumulators, shuffle-down), rope / SiLU / GELU / residual as separate fp32 ops (`_rn` intrinsics; libdevice's `expf` / `erff` / `powf` / `cosf` / `sinf` under nvcc's defaults, as torch builds them), SDPA as FA2's forward (128 x 128 tiles, 4 warps, key blocks last first, `exp2f(s * scale - max * scale)` unfused as flash is built with `UNFUSE_FMA`, P in bf16, the row sums quad-reduced at the end). Linears take torch's two GEMM paths: with a bias (qkv) addmm's cuBLASLt `gemm_and_bias` (bias epilogue, the heuristic's first algorithm under torch's 1 MiB `CUBLASLT_WORKSPACE_SIZE` and the pointers' `MIN_ALIGNMENT_*`); without one (wo, w1, w2, the aligner) `mm`'s `cublasGemmEx` (`vision_blas.zig`: libcublas, `DEFAULT_MATH`, fp32 compute, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, the handle's stream then its 32 MiB workspace before each call, as `getCurrentCUDABlasHandle`).
  - **`TF_DSV41_BIAS_VL`** (prod sets it): the folder's `layers.<i>.ffn.gate.bias_vl` (fp32, a missing layer refused, as `attach_bias_vl`) are loaded on every rank as weights `L<i>.moe.bias_vl` (the router is replicated). A prefill segment with image positions then emits Python's `vision.moe`: glue `image_split`, the image rows' whole `prefill_moe.forward` at their own row count routed with `gate.bias_vl`, the text rows' one with `gate.bias` (its ticket `gm_ticket_again`), glue `image_merge` (the bf16 partials back to their rows: the bits of Python's fp32 scatter then `gather_partials`' cast). `block_prefill.moeSplit` / `moeRows` (the emitter's row count is the call's while it runs); `Options.image_rows` (< 0: the buffer plan, both calls at full rows). Host check `tf-dsv41-m1 vision-emit CAPTURE`: for every captured prefill segment and image rows 1, 5, 16, 17, n/3, n-1, n, each call equals, launch by launch, a plain segment's MoE of that many rows (which `check` holds to Python): cap15 / cap24-1 / cap24-12 / cap24-16, 30 programs, 0 mismatches.
  - **Formats and URLs:** GIF (the first frame, `GifImagePlugin` + `GifDecode.c`) and BMP (`BmpImagePlugin`: core / info headers, palettes, 5-5-5, the bitfield layouts, RLE8 / RLE4 with Pillow's own decoder) beside PNG and JPEG; https image URLs through `vision/fetch.zig`, prod's `image_fetch.py` policy (`check_url`, every address public, the connection pinned to a checked address with TLS for the host name, a declared JPEG / PNG / WebP / GIF, `_MAX_BYTES`, 3 checked redirects, `_FETCH_TIMEOUT` within `_FETCH_TOTAL_S` with a watchdog); a busy front end waits up to 60 s (`acquire(timeout=60)`) before its 503.
  - **Interfaces (hooks, one commit each):** `engine_api.Request.images`; `block.Options.image_rows`, `moeSplit` (`block_prefill.zig`), glue `image_split` / `image_merge` / `gm_ticket_again` (`forward_prefill.zig`), `Forward.image_bias`, `tf-dsv41-m1 vision-emit`; `family.Rendered.images` (`prompt.zig`, `chat.zig`, `tokens.zig`); `block.Options.image_keep`; glue `image_keep` (`forward_prefill.zig`); `Forward.images` / `.vision` and the `embed` glue's fill (`forward.zig`); `vision_rows.boot` / `close` (`model.zig`, `m2b.zig`); the engine wrap (`serve_engine.zig`); `tf-dsv41-m1 vision` (`m1_main.zig`); the `dsv41_vision` fatbin (`zig/build/cuda.zig`, `kernels.zig`); `dsv41_vision` in the serving module (`zig/build/server.zig`, `serve/root.zig`). No plan-link operation: the followers only need the ids (keep) and the embedding exchange they already do.
  - **Host gates:** `zig build test-dsv41-vision`: 54/54 committed images (PNG every Pillow mode, Adam7, tRNS, 16-bit; JPEG 4:4:4 / 4:2:2 / 4:2:0, progressive, grey, CMYK; GIF interlaced / flat / transparent / grey / offset frames; BMP Pillow-written and hand-built 1 / 4 / 8-bit, core header, 5-5-5, 5-6-5, 32-bit orders, RLE8 / RLE4, top-down) decoded, planned, patched, digested and given virtual ids exactly as prod's `vision_prep` (with and without `TF_DSV41_BIAS_VL` in the digest); `TF_DSV41_VISION_GOLDEN`: the 61-image set from `tools/zig/dsv41_vision/gen_prep_golden.py` (e1e2 image), 61/61; the fetcher's 649/649 `public_ip` / `check_url` decisions equal to `image_fetch.py` (`gen_fetch_golden.py`) and an in-process HTTP server (redirect, type, size cap, HTTP error). `test-server`: native mode renders a data: URL PNG to its span of virtual ids, held. With the knobs off: `check` cap24-1 / cap24-12 / cap24-16 / cap15 / cap18 / cap22 and `dcheck cap16` 0 mismatches, `m5-emit cap18` 0 problems.
  - **Pod step: `M2B_VISION=1`** (`M1_JOB=m2b`, 2 GPUs): `source "$Z/tools/zig/dsv41_vision/job.sh"; vision_gate`. Python twin `/dsv41-tf/src` (prod 8474f31, e1e2x extensions; `VISION_PYSRC` overrides). About 22 min:
    1. **vision-ref** (about 10 min, its own TP=2 load of layers 0-`$hi`): `vision_ref.py` makes 3 images with Pillow (seeded: a 640 x 480 JPEG, a 1280 x 400 PNG, a 200 x 150 RGBA PNG), runs prod's `vision_prep` on them, dumps the tower's stages (embed, block0, block1, block15, block31, vit, gelu) and span rows on rank 0, holds the rows in `vision.STORE`, then runs 3 image prompts (text around one or two images; one starts with an image) through `dsv41_m2b_ref.py`'s own `one`: `p1..p3` with the text bias, then `p4..p6`, the same prompts after `vision.attach_bias_vl(fw.blocks, $LP)` (prod's `TF_DSV41_BIAS_VL`: the pack's `native.safetensors` holds the 43 `gate.bias_vl`); state dumps, 32 greedy tokens, the AOT set with the image path's Triton variants (`_fuse_dec` with `HAS_KEEP`, the split calls' row counts); `trace.jsonl` / `trace-bias.jsonl`. Then `engram-host.bin`.
    2. **vision-vit** (about 1 min, GPU 0): `tf-dsv41-m1 vision $LP $LOCAL/vision-ref/img`. Expected `PASS vision ViT + aligner == Python: 3/3 images`; otherwise each image names its first differing stage and the count / largest difference. The stages: embed, block 0's every op (`b0.norm1`, `b0.qkv`, `b0.rope` (q / k / v), `b0.attn`, `b0.wo`, `b0.res1`, `b0.norm2`, `b0.w1`, `b0.silu`, `b0.w2`), block0 / 1 / 15 / 31, vit, gelu, span. On a FAIL, **vision-vit-ops** reruns it with `TF_DSV41_VISION_RESYNC=1`: each stage starts from Python's bits of the stage before, so every op's count is its own (one run names every differing op, not only the first).
    3. **vision-prefill-p1..p3** and **p4..p6** (about 2 min each): `m2b_ranks.sh` with `M2B_PREFILL=1 TF_DSV41_IMAGES=native TF_DSV41_VISION_HOLD=<img dir>` (p4..p6 also `TF_DSV41_BIAS_VL=$LP`): our prefill from an empty slot, every state role == Python's, then the tokens.
    4. **vision-generate** / **vision-generate-bias** (about 3 min each, budget `VISION_GEN_MIN` = 8): `gen_ranks.sh`: `tf-dsv41-m1 generate` (drafts off) on `trace.jsonl` / `trace-bias.jsonl` with the same environment: `PASS M4 CLI == HTTP: 3/3 replies equal` (here Zig == Python). If rank 0 stops early, rank 1 is killed 30 s later and the step fails with both logs' tails (pod 30's generate-bias hung 5 min in `wait` on a follower stuck in a collective; its cause is not in the logs on hand).
    - Each run writes `PASS vision ...` / `FAIL vision ...` to `steps.txt`. `VISION_RUNS` ("vit prefill generate prefill-bias generate-bias") trims, `VISION_STEPS` (32), `VISION_REF_MIN` (14).
    - **Assets:** none to stage. The tower's tensors and the 43 `gate.bias_vl` are in `native.safetensors`, which the m2b pack copy already takes (`TF_DSV41_BIAS_VL=$LP` reads them through its index); the images are generated in the step.
    - **The fatbin:** `-Dnvcc` builds `dsv41_vision` with the others (sm_120 + sm_121, the build's torch flags, `-O3`); with `-Dfatbins`, copy `dsv41_vision.fatbin` into that dir first. Without it the build succeeds and native images refuse at boot (`VisionKernelNotBuilt`). The prebuilt one (nvcc 13.3, sm_120 + sm_121) is `zig-dsv41-scratch/vision/fatbin/dsv41_vision.fatbin`, sha256 `6f2ab33a7c012d2969db39362ca9f7db19d31e1019594fcd29ede067b0bcd6fe`; pod 30's (`dsv41_vision.fatbin.pod30`, `--fmad=false`, the ftz `ex2` softmax) must not be reused.
  - **Pod 30 FAIL: diagnosis and fix (`b3c699b`, GPU rerun pending).** Pod 30 had the tower 1 bf16 ulp off from block 0 (0.35-0.6 % of elements), 0/3 images, and `vision-generate-bias` timed out. The raw pod 30 logs were not on the dev box (only the summary in the table below), so the fix comes from reading torch 2.13's own source in the e1e2 image (`/opt/pytorch/pytorch`: `CUDABlas.cpp`, `CublasHandlePool.cpp`, `Blas.cpp`, flash's `softmax.h` / `flash_fwd_kernel.h` / build flags, the unary / activation / reduce kernels) op by op against `vision.cu` and `vision_tower.zig`. Three differences, all fixed:
    1. **Attention's exp:** ours was `ex2.approx.ftz(fma(s, scale_log2, -max_scaled))`. Flash is built with `UNFUSE_FMA` and no fast math: `exp2f(__fmul_rn(s, scale) - max_scaled)`, and the rescale `exp2f((prev - cur) * scale)`. The fused FMA rounds once where torch rounds twice, so P differs by an ulp in a fraction of rows. That is the size of block 0's differences.
    2. **The unbiased linears** (wo, w1, w2, the aligner's two; only qkv has a bias) ran through cuBLASLt. torch runs `F.linear` without a bias as `mm` → `gemm_internal_cublas_bfloat16_helper` → `cublasGemmEx` (the preferred backend is cuBLAS), which picks its own kernel. Now `vision_blas.zig` makes that exact call.
    3. **cuBLASLt's workspace** for the biased qkv: ours offered 32 MiB to the heuristic; torch offers `CUBLASLT_WORKSPACE_SIZE` = 1 MiB (unified with the cuBLAS workspace) plus `MIN_ALIGNMENT_A/B/C/D` from the pointers. The heuristic's first algorithm can change with the limit. Now 1 MiB and the alignments.
    - Also: the vision fatbin builds with nvcc's default `--fmad=true` (torch's build) instead of `--fmad=false`. Our arithmetic is `_rn` intrinsics (never contracted), so this only changes libdevice's internals, now as torch's.
    - Checked equal and left as they were: `_rms` (float mean: vt0 4, input_vec 4; factor 1/D; `rsqrtf`), `_rope` (`theta ** e` as `powf`, `1.0 / x` as `reciprocal` IEEE div, `arange / dim` as a multiply by the exact 1/32), `_rotate`, SiLU (`x / (1 + exp(-x))` in fp32, bf16, then `* u`), GELU (`x * 0.5 * (1 + erf(x * M_SQRT1_2))`), the residual add, FA2's tile shape, column order and quad reductions, `scale_softmax_log2` (fp64 product cast to fp32).
    - **Host-checked:** the engine builds with every fatbin embedded (`zig build -Dfatbins=<pod 30's set + the new dsv41_vision.fatbin>`, ptxas: no errors); `zig test vision_blas.zig` (torch's `_getAlignment`). Not GPU-run: no pods without the user, and the Sparks are the overnight agent's.
    - **Rerun on a pod** (`M2B_VISION=1`, the pod-step block above, at or after `1de1ccf`): put the new `dsv41_vision.fatbin` (sha256 above) in the job's `-Dfatbins` dir. Pass: `PASS vision vit` (3/3, every stage), then the prefill / generate lines. On a vit FAIL, `vision-vit-ops.log` lists every op still off.
    - **Rerun on the Sparks:** see "FOR THE NIGHT RUN: images" below section 0a's pod 29 / 30 notes (`tools/zig/dsv41_vision/spark.sh`: the whole gate over both Sparks). `vision_ref.py --tower-only --vdir DIR --pack PACK` stays for a one-GPU tower check without a model load.
  - **Gaps (prod accepts, Zig refuses by name):** WebP (libwebp's VP8 / VP8L / alpha and the animation decoder Pillow uses, to its fancy upsampler: a port of its own) and TIFF (Pillow's libtiff paths: LZW / Deflate / PackBits / JPEG / CCITT, YCbCr, premultiplied alpha, tiles); an https image host with only IPv6 addresses (std's client pins an IPv4 literal) and an internationalized host name (no IDNA); arithmetic-coded and 12-bit JPEGs (libjpeg-turbo decodes them).

- `dsq-zig-gm2pf-1` (x3gm v2 on Zig's prefill, `TF_DSV41_GM_V2`; `gm2pf.zig`):
  - **What it is.** `block_prefill.zig`'s routed experts as x3gm.`_run_v2` (prod twin 8474f31) runs a ragged block, written call by call against its code (no v2 capture existed: every capture's twin predated v2, so their launches are v1 whatever meta.json's env says):
    - `1` (`one`, prod): the rotation, ONE `x3gm.plan` over every pick (glue `gm_plan` with k2 0: no width mask, bm 64), one `gateup2` over every width (ticket[0]) and one `down2` on the same plan tensors (ticket[7]), each reading the experts' widths from `s.L<i>.gm.k2g` / `k2d` (int32 [E + 1]: forms' K2s, then a 0 sentinel, as `Ragged.k2g`; filled once in `forward_prefill.ready`). Configs are `tuned2` without `TF_DSV41_GM_CFG`: gate/up 1, down 0. Two launches a MoE call instead of two a width.
    - `split`: v1's loop shape (a masked plan and a launch a width present) on gm2_kernel.
    - unset / `0`: today's v1 launches. Anything else is refused (`error.BadKnob`), as x3gm.config.
  - **Knob:** `prod_knobs.apply` reads it on every rank into `block.Options.gm_v2` before the buffer plan and logs `x3gm v2 one` in the "prod knobs:" line. The M1 tools (`check`, `forward` / `both`, `m5-emit`) take the mode from the capture's launches (`m1_check.gmMode`), not from its env.
  - **Hooks (one commit each):** `block.zig` (Options.gm_v2), `block_prefill.zig` (moeRows), `forward_prefill.zig` (the k2 tables, `gm_plan` k2 0), `prod_knobs.zig`, `m1_check.zig`, `m1_forward.zig`, `m5_emit.zig`, `zig/build/cuda.zig` (`test-dsv41-gm2pf`), `job.sh` (`gm2pf` in `ws_gates`).
  - **Host gates:** `zig build test-dsv41-gm2pf` 3/3 (the knob's modes; the capture's mode from its launches; the real emitter at 2,048 / 300 / 12 / 1 rows: every call that is not a routed launch or its plan unchanged in order, `one` = one plan + one `gateup2` + one `down2` a MoE call on shared plan tensors with tickets 0 / 7 and [E + 1] width tables, `split` = a launch a width, `one` needs no widths, widths outside x3gm.K2S refused). With the knob off: `check` cap24-1 / cap24-12 / cap24-16 / cap15 / cap18 / cap22 and `dcheck cap16` 0 mismatches, `m5-emit cap18` 0 problems; `test-dsv41`, `-knobs`, `-longpf`, `-rows` pass.
  - **Pod step: `M2B_GM2PF=1`** (`M1_JOB=m2b`, 2 GPUs, after the M2b reference): `source "$Z/tools/zig/dsv41_gm2pf/job.sh"; gm2pf_gate`. About 35 min:
    1. **gm2pf-cap-v2 / -v1** (GPU 0, about 8 min each): `dsv41_m1_capture.py` on the **midprofile twin** (`midprofile.tar.gz` in `$Z` or `dsq/m5/`, as `M1_JOB=m5`: dsv41-midprofile dba6a1f, whose x3gm.py / .cu / .cpp and prefill_moe.py equal prod's 8474f31) with prod-knobs.env and `TF_DSV41_GM_V2=1` / `0`, set B, a 2,048-row prefill segment. On v2: `tf-dsv41-m1 check` (expected `options: x3gm v2 one` and 0 mismatches), then `tf-dsv41-m1 both` on GPU 0 (every launch replayed bit for bit, gateup2 / down2 included; then our v2 calls on our buffers == Python v2 launch by launch). Then `same.py` on the two ops.jsonl: `PASS gm2pf-same` = every routed partial and phase end the same bits with v2 and v1 (Python's v2 == v1).
    2. **gm2pf-off-1 / gm2pf-v2-1** (both GPUs, about 6 min each): the `M2B_PREFILL` gate on the last reference (the longest prompt) with `TF_DSV41_GM_V2=0`, then `=1` on the Zig ranks: every state role == the reference, the first token, the decode gate; rank 0 logs `x3gm v2 one` on the second. **Timing:** "Zig prefill rank 0: N rows in S s" of each (cold first prefills; `GM2PF_REPEAT=2` alternates them twice).
    - Knobs: `GM2PF_STEPS` ("cap prefill"), `GM2PF_SETS` (B), `GM2PF_ROWS` (2048), `GM2PF_REF`, `GM2PF_REPEAT` (1), `GM2PF_CAP_MIN` (12), `GM2PF_M1_MIN` (12), `GM2PF_MIN` (6). Lines to `steps.txt`; small capture files to `$RES/gm2pf-cap-v*/`, the blobs deleted after each capture.
    - **Assets:** `midprofile.tar.gz` (built: `../dsv41-zig-ref/midprofile.tar.gz`, `tools/zig/dsv41_m5/stage_midprofile_twin.sh`); without it the capture half reports `gm2pf-cap skipped` and the prefill half still runs. Best with `M2B_PROMPTS` holding a prompt over 2,048 tokens (e.g. 2600) so a full 2,048-row chunk is timed.

### Structured output (`dsq-zig-grammar-1`, R9)

**Decision: xgrammar's own C++ core through a C ABI, not a native Zig grammar engine.** The masks must be bit-identical to prod's, and prod's masks are xgrammar 0.2.8's. They come out of its JSON-schema converter, Earley parser, adaptive token-mask cache, the DSML structural tag, and the token decoding of the tokenizer info. A native engine would have to reproduce all of it, and every xgrammar version moves it. Building the same sources gives the same masks by construction, and the goldens check it.
- `zig/third_party/xgrammar/`: the 0.2.8 sdist's `cpp/` (without the Python bindings), `include/`, picojson and dlpack, unmodified (sdist sha256 `8305646f…ccbb80`, Apache-2.0).
- The library is built by Zig's C++ toolchain (`zig/build/grammar.zig`, always ReleaseFast).
- `zig/src/core/grammar/xgr_c.{h,cc}`: a 20-call C ABI with Python's defaults; exceptions become a per-thread message.

**Module `grammar`** (`zig/src/core/grammar/`, family-neutral):
- `engine.zig`: the `Compiler` / `Compiled` / `Matcher` vtables; `xgr.zig` implements them.
- `vocab.zig`: Python's `_vocab`. Added tokens are blanked except a view's kept ones (DeepSeek's tools view keeps `｜DSML｜`).
- `grammars.zig`: GLM's `Grammars`. Two views; json_object is `{"type": "object"}`; at most 32 blanks; choice compiles to EBNF; tools go through `tags.zig`, xgrammar's `deepseek_v4_1` structural tag, written as pydantic writes it.
- `constraint.zig`: GLM's `Constraint` (`cut` / `fill` / `advance` / `finished`, the think gate).
- `pack.zig`: a spec for the followers.
- `rows.zig` + `mask_kernel.zig`: the bitmask on logits. The kernel is Zig compiled to sm_80 PTX by the NVPTX backend and JIT-loaded by the driver, so no nvcc is needed.

**Where the masks apply (as prod):**
- `grammar_gpu.GrammarTarget` (rank 0) decorates the lanes target:
  - at prefill it binds the slot's grammar (the think state is read from the prompt) and stages the reply's first token's mask on the prompt's last row;
  - before each window it follows the committed tokens (kept drafts, then the pending token), cuts the drafts the grammar rejects (only R changes; rows are row-invariant), and stages the kept rows' masks;
  - a cut row's choice is `maxInt(u32)`, so the lanes never accept past the cut.
- A draft that ends the grammar (its stop token) is not cut. Its row and the rows after it run unconstrained, as prod's later windows run them (Python cuts it and emits it as the bonus instead): the same tokens, other round counts.
- Trees with a grammar run as their main chain (Python builds no tree rows beside grammar masks).
- `Masks` (every rank) applies the staged rows' -inf columns in "w.logits" at the first pick after the window, through the hooks in `Forward.greedy` and `GpuSampler.run`, on every rank's vocabulary half. Greedy, sampled and nucleus picks and the whole-row fallback then see Python's masked logits.
- Fills run on `io.concurrent` tasks started before the forward and are awaited at the pick. The bits go to the device in one pinned copy, and one kernel launch covers all staged rows. Nothing runs for requests without a grammar.

**Followers:** op 28 (`op_grammar`) carries:
- bind: the packed spec and think state; the follower compiles the same grammar;
- step: per slot, the tokens to follow, the cut and the window's tokens;
- release.

Each rank's matcher walks the same tokens, so every rank fills the same masks for its own half. No bitmask crosses the plan link.

**Requests:**
- `grammar_gpu.GrammarEngine` decorates the served engine (`info.structures = true`). It compiles each request's structure on its HTTP thread; a grammar that cannot be enforced fails at once with xgrammar's message.
- It hands the compiled grammar to the target by prompt: admission is in submit order, and every request registers.
- The server (`zig/src/server/grammar.zig` `toolSpec`, GLM's `_tool_spec`, with `TF_DSV41_TOOL_GRAMMAR`) holds tool_choice required / named and strict tools to the DSML grammar. It does so only when the engine enforces structures; otherwise tools are served as before.
- `engine_api.Structure` gains `.tools`.
- The trace (`TF_DSV41_TRACE_TOKENS`) records each request's structure, and `generate` replays it.

**Open:**
- A compile failure reaches the client as a failed reply, not Python's HTTP 400: the engine API has no pre-submit check.
- A mid-reply grammar failure fails the whole round's requests, not only that slot's.
- `TF_DSV41_SPEC_DRAFT`'s early pass with masks is not covered (not ported).
- Fills overlap the forward but are awaited before the pick. At 4 slots × 16 rows the fill cost sits on the critical path only when it exceeds the forward. Measure it on the Spark window (`TF_DSV41_PHASES`).

### Pods 29 and 30: the fix-up and last combined gates (2026-10-07, for the overnight run)

Logs are in the quant repo: `results/20261007-fix29-pod29` (pod 29, tree 4d91532) and `results/20261007-comb30-pod30` (pod 30, tree 8278ede). Both ran on index Engram rows (no `M2B_ENGRAM`): pod 26 had already gated R3 on M4 and m2b.

| gate | result | note |
|---|---|---|
| M2b decode regression | PASS (29, 30) | |
| speculative pass (`TF_DSV41_SPEC_DRAFT`), lanes 3 and 4 | **PASS** (29): 65/65 tokens == Python's serial, 20/20 and 1/1 speculations used, 4.64 tokens a round (mode 4) | first GPU run |
| structured output (`TF_DSV41_GRAMMAR`), drafts on and serial | **PASS** (30): Zig == Python 6/6 replies each | first GPU run; the HTTP step did not run (script bug, fixed `1c863d2`) |
| fused dense prefill (`TF_DSV41_PF_DENSE=fused`) | **PASS** (30): 65/65 state roles, 64/64 tokens + logits, both ranks | knobs gate's first run; the rest below |
| M5 split / park / compact | **PASS** (30): both prompts (2,060 and 9 rows), 61/61 state roles, position, Engram tail, first token, 64/64 tokens + logits; park through a RAM and an NVMe round trip | pod 29's 9-row failure was the gate's stale ring rows, fixed `a4caf46` |
| images (`TF_DSV41_IMAGES=native`, `BIAS_VL`) | **FAIL** (30): the tower is not bit-exact, 0/3 images; block 0 already differs by one bf16 ulp in 0.35-0.6 % of elements, compounding over 32 blocks (final span up to ~0.1); every image prefill differs from block 0 (first token still equal) | torch SDPA / cuBLAS summation order; `dsq-zig-vision-1` reopened. **Keep `IMAGES` off overnight** |
| long prompts (`TF_DSV41_CONTEXT`, stream top-k, split row blocks) | not run: both Zig runs stopped at load with `MissingTritonVariant _prune` | gate bug: `m5.zig` turns the m2b job's `M2B_TOPP=0.85` into the expert top-p while the reference ran with it off; fixed `33be853`. **Run C overnight is the first GPU run of long prompts** |
| R1 decode gate (eager, 64 steps) | **PASS** (30): 64/64 tokens + logits, both ranks | |
| R1 timed windows (`M2B_PROF_ROWS`) | **FAIL** (29, 30): `CUDA_ERROR_ILLEGAL_ADDRESS` in `tf_dsv41_mhc_cuda_v1.run` (call 18 of the 720-call window, layer 0's mHC; named by `TF_DSV41_SYNC_EACH=1`) at 4-row windows; 1 and 2 rows ran. **Cause found and fixed (`a585336`): a test-driver bug, not R1 and not the served path.** The M2b driver planned its buffers for 1-row windows only (`f.plan(&.{1})` outside lanes mode), so the decode-only roles (`L.mhc.part` [n, 4, 40, 32] about 20 KB a row, ...) were one row deep and the timed 2- / 4-row windows ran past them; R1's mHC tail reads every row's partials from many CTAs and left mapped memory at 4 rows. The served model plans every window from 1 to `max(prefill rows, DSpark block + 1)` and refuses larger ones (`target.zig` `error.WindowRows`), so serving never had this. The multi-row "Zig eager" timings of pods 26 / 29 ran on overrun buffers: rerun the timing table (`M2B_PROF_ROWS`) after `a585336` | the overnight `R1=0` fallback should not be needed |
| slots, 4 slots, row graphs | **FAIL** at boot (30): `batch.zig` set the row graphs' optional to `undefined` before its init; ReleaseSafe's `.?` panicked on every rank | **fixed `a36c4c2`.** The overnight kit must be built from HEAD at or after `a36c4c2` for runs B / B2 (4 slots, graphs on by default) |
| multi-slot drafting (`TF_DSV41_SLOT_DRAFTS`), eager | **PASS** (30): batched == Python 4/4, alone == Python 4/4, batched == alone 4/4; passes over several slots (256 drafted, 130 passes over 256 slots) | first GPU run; the graphs run hit the boot panic above |
| prod knobs: `PREFILL_ROWS` 1,024 / 512 | not run: `MissingTritonVariant _kv_store` (the knobs gate has no Python load, and the reference ran 2,048-row segments) | gate-asset gap only; prod and the kit use 2,048 |
| prod knobs: `INDEX_BUDGET_MIB=64` | **PASS** (30) | |
| prod knobs: read-ahead, all together | skipped (they need `M2B_ENGRAM` shards) | |
| x3gm v2 prefill (`M2B_GM2PF`, captures only) | **host check PASS** after `eacab35` (offline, on pod 30's v2 capture): w1 / w3 / w16 145/145 and the 2,048-row prefill segment 290/290 calls equal, x3gm v2 included; `same.py` PASS (Python's v2 == v1, 0 differ). The pod's 83 "mismatches" were not v2: the midprofile twin's bindings always take R1's trailing arguments (mHC spin -1 / defer 0, attention split 0 / rope 0 + 1,024 ticket words, the route's empty prune list, `_prune`'s KIT false) even with R1 off, and the check emitted the older 24-argument calls. `block.Options.r1_sig`, read from the capture's launches, fixes check / both / m5-emit / dcheck | still to run on a GPU: `both` (the v2 capture replayed bit for bit) and the prefill timing (+7 % at 8K-32K expected); `GM_V2` stays off until then |

Pod-side lessons for anyone running pods: the shared `runpod/idle-guard.sh` deletes a pod whose container log is unchanged 60 min (pod 29 at 19:40): a job driven over ssh must write a heartbeat to `/proc/1/fd/1`. `vision_kernels.zig` did not compile with `dsv41_vision.fatbin` embedded (Zig 0.17: `field_names`), fixed `8278ede`.

**For the overnight run (`dsq-zig-night-1`):**
- **Rebuild the kit from HEAD at or after `a36c4c2`** (the row graphs' boot panic: runs B and B2 crash at boot on e19bb41's binaries). A kit already built from `77bafde` (with the fix; the same sm_121 fatbins) is at `~/Documents/Projects/spark-work/zig-window-kit-next`, checksums verified; its RUNBOOK.md carries pod 30's feature table and the `R1=0` fallback.
- R1's 4-row `mhc_cuda` fault was the M2b test driver's 1-row buffer plan (fixed `a585336`); the served path plans every window size, so no `R1=0` fallback is expected (keep it as a last resort).
- Kit knobs: keep `IMAGES` off (the tower is not bit-exact); `GRAMMAR=1`, `PF_DENSE=fused` and the 64 MiB index budget are gated (PASS here); `SLOT_DRAFTS=1` is gated eager.
- Long prompts (run C) have no GPU gate yet: run C is their first GPU run.
- Part 4 (Zig as prod): `config/prod-zig.env` on the prod mirror's `zig-serve` branch is at `57a1ea0`: it now turns on `TF_DSV41_SPEC_DRAFT=1`, `TF_DSV41_GRAMMAR=1` (the canary's json probe back on) and `TF_DSV41_PF_DENSE=fused` with prod's table, all gated on pods 29 / 30; `IMAGES` stays `placeholder`.
- **Images (`IMAGES=native` + `BIAS_VL`):** the tower fix's gate runs on the Sparks this night: "FOR THE NIGHT RUN: images" just below. Keep `IMAGES` off in the kit's other runs whatever it prints.

### FOR THE NIGHT RUN: images (`dsq-zig-vision-1`, TF_DSV41_IMAGES=native + BIAS_VL)

Pod 30 failed the vision tower (1 bf16 ulp off from block 0, 0/3 images). The fix is `b3c699b` + `ea8198d` (section 0a, "Pod 30 FAIL: diagnosis and fix"); it has not run on a GPU. This gates it on the Sparks: the tower, the image prefill with and without prod's image routing bias, and the served replies, Zig against prod's Python (8474f31) on the same load shape. About 45 min, prod down, both nodes. Nothing here changes the kit's other runs; `IMAGES` stays off in them.

**Build (dev box), before the window:**
1. Tree: HEAD of `dsv41-zig` at or after `4358658` (the fix; `tools/zig/dsv41_vision/spark.sh`; `dsv41_m2b_ref.py --prompts` with `--rank`; `build-kit.sh` ships both vision scripts).
2. FATBINS: the kit's sm_121 fatbin dir with **`dsv41_vision.fatbin` replaced** by `~/Documents/Projects/spark-work/zig-dsv41-scratch/vision/fatbin/dsv41_vision.fatbin` (sha256 `6f2ab33a7c012d2969db39362ca9f7db19d31e1019594fcd29ede067b0bcd6fe`, sm_120 + sm_121, nvcc 13.3, `-O3` without `--fmad=false`). Any older `dsv41_vision.fatbin` (pod 30's included) carries the bug.
3. `FATBINS=<that dir> TP_FATBIN=<tp_mailbox.fatbin, sm_121> bash tools/zig/dsv41_m2c2/build-kit.sh <kit>` from a clean archive of the tree, then the kit's usual rsync to both nodes and `sha256sum -c --quiet SHA256SUMS`. Check: `<kit>/tools/zig/dsv41_vision/spark.sh` and `vision_ref.py` exist.

**Assets:** none to stage. Each node already has what the steps read: the pack (`dsv41-q28-v2` at the kit's paths), prod's tree `~/ai/src/dsv41-prod/8474f3164ea60cb9c9f8e9bd5a4493fdf13eaa47`, the image `tensorfold-dsv41:e1e2-8474f31`, and prod's cache volume `dsv41-perf1-cache` (`/cache/dsv41-bias-vl`: the release's 43 `gate.bias_vl`, prod's `TF_DSV41_BIAS_VL`; its Triton / extension caches). The three images are generated (seeded) by the reference on each node; Engram rows are the index function on both sides (as M2b), so no Engram shards.

**Steps** (in `~/zig-window-kit` on each node, `R` = 0 on dgx-01, 1 on dgx-02; prod stopped and no other `dsv41` container running; "both" = start within about a minute of each other). Every step appends `PASS ...` / `FAIL ...` to `out/vision/vision-steps.txt`:

| step | where | command | pass | time |
|---|---|---|---|---|
| V1 | both | `./tools/zig/dsv41_vision/spark.sh R out/vision ref` | `PASS vision ref rank R: 6 prompts, both traces` on both | ~15 min |
| V2 | dgx-01 | `./tools/zig/dsv41_vision/spark.sh 0 out/vision vit` | `PASS vision vit` (the log: `PASS vision ViT + aligner == Python: 3/3 images`) | ~2 min |
| V3 | both | `./tools/zig/dsv41_vision/spark.sh R out/vision prefill` | `PASS vision prefill pN ...` for p1..p3 and `pN bias_vl` for p4..p6, on both | ~15 min |
| V4 | dgx-01 + dgx-02 | dgx-01 `./tools/zig/dsv41_vision/spark.sh 0 out/vision generate`, dgx-02 `./tools/zig/dsv41_vision/spark.sh 1 out/vision follow` | `PASS vision generate (replies == Python's)` | ~5 min |
| V5 | dgx-01 + dgx-02 | the same with a last argument `bias` on both (`... generate bias`, `... follow bias`) | `PASS vision generate-bias (replies == Python's)` | ~5 min |

- V1 runs prod's Python at TP=2 over the two nodes on layers 0-24 (as the pod gate): prod's `vision_prep` on the 3 images, the tower's stage dumps on rank 0 (`out/vision/img`), 6 image prompts x 32 greedy tokens (p4..p6 after `vision.attach_bias_vl` from `/cache/dsv41-bias-vl`), state dumps, each node's AOT set, `trace.jsonl` / `trace-bias.jsonl`, then `engram-host.bin`.
- V2: the Zig tower against those dumps, stage by stage (embed, block 0's every op, blocks 0 / 1 / 15 / 31, vit, gelu, span). On a FAIL it reruns with `TF_DSV41_VISION_RESYNC=1` (`vision-vit-ops.log`: every op still off, apart) and stops. **On a V2 FAIL skip V3-V5** (they would differ from the tower's bits) and report `vision-vit.log` + `vision-vit-ops.log`.
- V3: `tf-dsv41-m1 m2b` with `M2B_PREFILL=1` a prompt (the Zig prefill from an empty slot, the image rows into rank 0's embedding, Engram's keep, p4..p6 the image rows' own MoE call with `gate.bias_vl`): every state role == Python's, then the tokens. 20 min cap a prompt (`timeout`). `PROMPTS="1 4"` trims.
- V4 / V5: the served engine (`tf-dsv41-m1 generate`, drafts off, `TF_DSV41_LAYERS=0-24`, p1's assets) replays the traces; 15 min cap. If rank 0 stops, `docker stop dsv41-zig-vgen-r1` on dgx-02.
- **On failure:** a step's log is `out/vision/vision-<step>.log` on each node; keep them and `vision-steps.txt` from both nodes, go on with the night's other runs (no restore needed: these steps leave prod stopped as they found it).
- **Afterwards (both nodes):** `docker run --rm -v $PWD/out/vision:/out --entrypoint rm tensorfold-dsv41:e1e2-8474f31 -rf /out/ref /out/img /out/zig-p1 /out/zig-p2 /out/zig-p3 /out/zig-p4 /out/zig-p5 /out/zig-p6` (state dumps and AOT copies, several GB; root-owned), keeping the logs.
- All five PASS = images can go on in the kit (`IMAGES=native BIAS_VL=1`) from the next window; record it in the pod 29 / 30 table's images row.

### FOR THE NIGHT RUN: R1's timed windows (`dsq-zig-arch-1`; fix `a585336`, kit knob `3ea37ef`)

**What was wrong.** On pods 29 and 30, Zig's R1 run died with `CUDA_ERROR_ILLEGAL_ADDRESS` in `tf_dsv41_mhc_cuda_v1.run` at 4-row timed windows. The cause is the M2b gate driver's buffer plan, not R1: it planned only 1-row windows, so the timed 2- and 4-row windows overran the decode-only roles (`L.mhc.part`, ...). The served model plans every window size (1 to max(prefill rows, DSpark block + 1)) and was never affected: `run_m4.sh`'s runs need nothing. Only the M2c-2 gate with timed rows exercises the fix.

**Rebuild** (dev box, once; the same rebuild serves the images section above):
1. Tree: a clean `git archive` of `dsv41-zig` at or after `3ea37ef` (it holds `a585336`, `eacab35` and the `PROF_ROWS` knob).
2. FATBINS: `~/Documents/Projects/spark-work/dsv41-zig-ref/fatbin-kit` (the sm_121 set the kits use: base 81f0477 + `dsv41_kvsplit` + `dsv41_sampling` + `dsv41_vision`). If the images fix is also being rerun, first replace its `dsv41_vision.fatbin` as the images section says.
3. `KIT_COMMIT=<hash> FATBINS=<dir> TP_FATBIN=~/Documents/Projects/spark-work/zig-m2c2-kit/tp_mailbox.fatbin bash tools/zig/dsv41_m2c2/build-kit.sh <new kit dir>` (it runs under `heavy.sh`), then rsync to both nodes and `sha256sum -c` as the runbook's step 2.

**Rerun on the Sparks** (prod stopped, lease free, no other `dsv41` container; about 40 min: one whole-model Python load, then ours):

| step | where | command | pass |
|---|---|---|---|
| R1a | both, within ~1 min | `PROF_ROWS="1 2 4" STEPS=64 ./run_host.sh R out/r1prof` (R1 is on by default; `PREFILL` off) | `out/r1prof/run.log`: `PASS M2b rank R: tokens 64/64, logits 64/64, first window blocks 40/40`; `out/r1prof/seed4101/zig.log` has `rank R: decode steps: tokens 64/64, logits 64/64`, then `rank R: 1-row windows ... ms`, `2-row ...`, `4-row ...` with **no** `CudaFailed` / illegal address |
| R1b | dgx-01 | record the R1 timing table: Zig's `N-row windows <mean> ms` lines (`seed4101/zig.log`, both ranks) and Python's per-row timings (`seed4101/ref.log`, `--profile`) | the table, Zig vs Python at 1 / 2 / 4 rows (whole model, R1, RoCE) |

**If R1a still faults at 4 rows:** rerun once with `TF_DSV41_SYNC_EACH=1` added to the Zig docker run (it names the faulting call), save `out/r1prof/`, and note it in the window report: it would then be a different bug from `a585336`'s. Either way the served runs (A / B / B2 / C) are unaffected.

**Not needed tonight:** the x3gm v2 check fix (`eacab35`) is host-only (pod 30's capture: 0 mismatches); `GM_V2` stays off until a GPU replay of a v2 capture.

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
- **Not yet with several slots** (each refused by name): sessions (`sessions_gpu.zig` snapshots one slot's roles: not opened), draft trees and keyed sampling in row windows (`TreeWindow` / `Sampling`), DSpark drafting unless `TF_DSV41_SLOT_DRAFTS=1` (section 0c; else `dspark_gpu.zig` is one slot: run `TF_DSV41_DRAFTS=0`). Engram rows come through the forward's source (`Host.windowRows`: R3's table or M2a's index rows), per segment with its slot's tail. Mixes over 16 rows run as several forwards; 24–64-row forwards need `block.emit`'s >16-row decode path (Triton mHC above `mhc_cuda`'s 16 rows).
- **Host gate:** `zig build test-dsv41-rows`, 13/13. It covers the table against Python's `plan_rows` example; checks, packing and layout; and the whole backbone window (paged and split pool, R1 on and off, R 1 / 4 / 12 / 16) through the real `block.emit` + `transform`. In that window every CSA2 step reads the table's columns, NK is the context bucket's, the attention / top-k get pos / rslot / pts, and the plan's stacked roles are S x one slot's while every other role is unchanged.
- **Pod step** (TWO GPUs, inside `M1_JOB=m2b` after the pack copy; `source tools/zig/dsv41_slots/job.sh; slots_gate`, with `SLOTS_PYSRC` = a Python tree with `rowgraphs` / `slots`, e.g. the decode1 twin's src with its `TORCH_EXTENSIONS_DIR`). About 14 min for the reference (one TP=2 load of layers 0–`$hi`, 4 prompts x 64 tokens, the row buckets warmed), then about 4 x 5 min:
  1. `slots-ref`: `ref.py` writes `trace.jsonl`, `ref.json`, `rope.bin` and `aot/`, including the row-mode Triton variants (`ROWS`, int64 POS, SL / PTS) for every bucket up to 16. `slots-engram-host` writes `engram-host.bin`.
  2. `slots-s4-eager`, `slots-s4-graphs`, `slots-s2-graphs`, `slots-s1`: `ranks.sh`, our two ranks serving the trace (`TF_DSV41_SLOTS` 4 / 4 / 2 / 1, `TF_DSV41_GRAPHS` 0 / 1 / 1 / 0, `TF_DSV41_DRAFTS=0`, `TF_DSV41_POOL_TOKENS=18432`, `TF_DSV41_LAYERS=0-$hi`). Expected in each `rank0.log`: `PASS slots batched == Python: 4/4`, `PASS slots alone == Python: 4/4`, `PASS slots batched == alone: 4/4`. s4-eager against s4-graphs is graphs off == on; s1 is today's path against the same trace.

### 0c. DSpark drafts over several live slots (`dsq-zig-mdraft-1`)

`TF_DSV41_SLOT_DRAFTS=1` (with `TF_DSV41_SLOTS` > 1 and drafts on; the same on every rank) drafts every live slot as Python's batcher does. When it is unset, drafts with several slots are refused as before. The verify side is section 0b's row windows: one forward over every slot's chain, per-slot acceptance in the lanes, per-slot `keep` on every rank (`op_rows_keep`), and sampling a slot's rows. This section adds the pass.

- **State a slot** (Python's `Drafter` in row mode, G8):
  - every DSpark block's ring holds S rings back to back (`s.L<b>.ring.*`, S × ring rows, Python's `stacked`);
  - `valid` a slot.
  The Markov / confidence heads and weights are the one-slot pass's.
- **Ingest** (`op_ingest`): a slot's committed taps rows go into its own ring, through `emitIngestAt` with a view at slot × ring rows. The taps come from:
  - the row window's rows of "w.taps";
  - `batch.zig`'s taps stash (`s.rows.taps`), when a round ran as several forwards and a later one overwrote "w.taps" (before, `tapsOf` gave nothing there);
  - the prompt's rows, from the one-slot prefill.
  Rows not on the device restart the slot's context, as with one slot.
- **Propose** (`op_propose`):
  - The asks are sorted by slot (Python's order) and split into balanced passes of at most `TF_DSV41_SLOT_DRAFT_GROUP` slots (default 3: the most whose rows stay below 16).
  - Each pass is one launch set over its slots' rows (`emitPassSlots`). Ring stores and attention run in row mode: each row's ring slot comes from the statics, and the context list holds rows of the stacked ring.
  - Then the host's candidates, the ranks' gather, and each slot's chain over its own rows.
  - Python runs every slot in one pass, padded to a power of two (20 rows at 4 slots). Ours stays below 16 rows because `mhc_cuda`'s site1 / final take at most 16, and the >16-row Triton mHC path is not emitted. Every launch is row-invariant, so grouping changes no draft.
- **No new Triton variants.** `dspark_emit.Layout` places a k-slot pass's statics at the one-slot pass's address residues. Every Triton launch of a 1–3-slot pass is then in the one-slot pass's specialization, the same kernels, which `test-dsv41-mdraft` checks. Set P's one-slot variants cover it.
- **Pool, split KV, graphs, sampling:** the pass touches only its rings and window buffers. The verify windows are section 0b's: paged pool, split KV, row graphs, a sampler a slot. The pass itself is eager, as with one slot.
- **Not yet:**
  - the speculative pass (`iface.Pass.begin` / `collect`, `dsq-zig-spec-1`): several-slot passes go through `propose` (exact, no overlap);
  - trees with several slots (refused by `batch.zig`);
  - a pass over more than 15 rows;
  - Engram read-ahead for row windows.
- **Files:**
  - `dspark_rows.zig`: groups, statics, messages, taps placement, the gather's order;
  - `dspark_slots.zig`: `SlotPass`, `iface.Pass` on every rank;
  - `mdraft_cli.zig`: `tf-dsv41-m1 mdraft`;
  - `mdraft_test.zig`;
  - `tools/zig/dsv41_mdraft/`.
- **Hooks** (one commit each):
  - `dspark_emit.zig`: `Layout`, `Slots`, `emitIngestAt`, `emitPassSlots`; the default emission is unchanged;
  - `batch.zig`: the taps stash;
  - `model.zig`: the knob, plan, init and `draftPass`;
  - `serve_engine.zig`: the served pass is `m.draftPass()`;
  - `m1_main.zig`: `mdraft`;
  - build: `test-dsv41-mdraft`.
- **Host gate:** `zig build test-dsv41-mdraft`, 7/7, on the release config with q28-v2 widths. It covers:
  - the 1- to 3-slot passes against the one-slot pass: the same launches, every Triton argument's specialization equal, rings sized S × ring;
  - `Layout.of(1)` equal to the one-slot statics;
  - the default emission identical;
  - an ingest's slot view and taps role;
  - Python's row-mode `DraftInputs.stage` contents;
  - groups, messages, taps placement and the gather.
- **Spark window run (for the kit's runbook, run B2):** prod's 4-slot table and knobs (`TF_DSV41_CALIB=real`, `TF_DSV41_CALIB_DIR`), `run_m4.sh` with `SLOTS=4 SLOT_DRAFTS=1` (`TF_DSV41_SLOTS=4 TF_DSV41_SLOT_DRAFTS=1`, pool and graphs as run B), and `bench_http.py` at 4 streams on the 8 prompts, against prod's Python engine at 4 streams, same prompts, seed and knobs.
  - Tokens must equal prod's.
  - From Zig's `slot drafts` / lanes counters and prod's per-request `drafting` stats, compare acceptance (accepted / drafted) and tokens a stream round. They should be close, not identical, because the host chain sums in a different fp32 order.
  - Assets: `rows-aot` plus DSpark's set P, nothing more.
  - About 15 min after run B's load.

### 0d. Serving and ops parity with prod (`dsq-zig-ops-1`)

The Zig server answers as prod's Python server does. Prod's DeepSeek app (`deepseek_v41/cuda/app.py` at 8474f31) runs on the GLM Spark server base (`glm5_next/spark/server.py`, `health.py`, `reqlog.py`), not on upstream TensorFold's `server/app.py`, so the Zig server, which was upstream's, gained that surface as `Config.wire = .spark` (`zig/src/server/spark.zig`).
- **Selection:** `main` picks it for DeepSeek-V4.1, and `TENSORFOLD_WIRE=spark|tensorfold` overrides. The upstream wire is unchanged.
- **The gate:** `wire_test.zig` replays `fixtures/serve/wire.jsonl` against the Zig server with the same scripted engine. `tools/zig/dsv41_ops/http_golden.py` recorded the file from prod's own handler over a socket. It holds 49 exchanges plus the request log. It compares status, Content-Type, the body (JSON, or the whole SSE stream), the engine's inputs (prompt ids, max_tokens, keyed sampling and seed, drafts, EOS, background) and every request-log record, with ids and clocks masked. **49/49 and 28/28 equal.**

| Surface | Prod (Python) | Zig before | Zig now |
|---|---|---|---|
| Reply ids, `model` | `chatcmpl-` + 24 hex; the served name | 32 hex; the asked alias | = prod |
| `usage` | no `completion_tokens_details` | had it | = prod |
| Engine stats | `tensorfold`: the engine's job stats | `tensorfold` runtime + `speculative` + `exact_mode` | `tensorfold`: prompt, cached, ttft_s, finish, completion, prefill_s, decode_s, rounds, drafted, accepted (+ `token_ids`, `images_omitted`) |
| SSE end | usage only with `include_usage`, in the last chunk; `[DONE]` | usage always, or a separate chunk | = prod |
| Streaming reasoning / content / tool calls | `dsml.Stream` deltas | same (srv-1) | = prod (byte-identical in the test) |
| Stop strings, max_tokens, finish_reason, `ignore_eos` | answer-only stops, holdback, `length` / `stop` / `tool_calls` | same | = prod |
| `reasoning_effort` / thinking / kwargs errors | app.py's messages | same | = prod |
| A call's prose (`content`) | kept beside `tool_calls` | null | = prod |
| `n`, numbers, `stop`, token-ID prompts | `App.check` messages | upstream's | = prod |
| Context limit | `context_problem` words, `context_length_exceeded`, param | upstream's words | = prod |
| Errors | `{"error": {message, type, code?, param?}}`; 404 `{"error": "not found"}` | upstream's | = prod |
| `logprobs`, modalities | ignored | refused (400) | ignored, as prod |
| `/v1/models` | `created`, `root`, `max_model_len`, `context_length`, aliases | id / created / owned_by | = prod |
| `/health` | `health.py` body; strict mode 503 | upstream's | = prod (`TF_DSV41_HEALTH` / `GLM53_TF_HEALTH`, `STALL_S`) |
| `/metrics` | `tensorfold_*` rows | upstream's histograms | = prod |
| `/tokenize` | `count`, `max_model_len`, `tokens`, `token_strs: null` | no `token_strs` | = prod |
| Request log | `reqlog.py` (blake2b-64, prefix fields, rotation) | upstream's body log | = prod (`TF_DSV41_REQUEST_LOG`, `GLM53_TF_REQUEST_LOG_*`) |
| Disconnect | `TF_DSV41_DISCONNECT` (socket check every round) | always on | = prod (`0`: a failed write only) |
| Startup | greedy warning (public PR #14); `serving: N slot(s)` | neither | both |
| Concurrency, admission | batcher, 4 slots | LaneHost; `TF_DSV41_SLOTS` (slots-1) | engine-side, not the server's |
| Grammar (`response_format` enforced, `tool_choice` required / named) | xgrammar | none | `dsq-zig-grammar-1` |
| Native images | `TF_DSV41_IMAGES=native` | refused | `dsq-zig-vision-1` |

Where the Zig numbers are more faithful than prod's:
- Prod's batch engine reports no top-level `drafted` / `accepted`, so prod's `/health` `drafted_total` / `accepted_total` stay 0. Zig fills them from the lanes' counters.
- Prod's `rounds` is the batcher's global round counter, so `rounds_total` sums global counters. Zig's is the request's own.

**Engine knobs from `prod.env`:**
- **Sessions:** `TF_DSV41_SESSIONS` (0: no store), `SESSION_DISK_GIB` and `SESSION_DISK_MIN` go to `sessions.Disk.open` (`sessions_gpu.zig`).
- **Memory floor:** `TF_DSV41_FLOOR_GIB` is validated against the hard floor (`0 < hard <= target`). The measured boot check warns when the worst serving point falls under it, as `memory.py`'s load check does. This hunk landed inside `81c7187`.
- **Prepared folders:** `TF_DSV41_PREPARED` / `_PREPARED_WRITE` (`prepared.zig`).
  - The Spark window's first Zig server took **505 s** to load, against prod's 33–48 s with Python's prepared folders.
  - Each layer's named host images are now written once and read back on later boots. The folder is keyed by the builders' source, the pack's shards, the rank, the plan and DSpark.
  - `dsv41 load:` logs where a boot went (images vs upload + digest). The window measures the gain.
  - The SHA-256 digest of every tensor at upload (`load.zig`) is still paid on every boot. It is the next cut if the window shows it matters.

**Ops** (prod mirror, branch `zig-serve`, worktree `dsv41-tensorfold-spark-zig-serve`, local only):
- `docker/zig/Dockerfile`: a native aarch64 / sm_121 build on dgx-01, from the gated fatbins or nvcc.
- `scripts/zig-image.sh stage|build|check`.
- `scripts/serve.sh ENGINE=zig`:
  - `--init` on every container;
  - `build` / `start` / `stop` / `restart` / `status` / `logs` / `canary` / `watch`;
  - the assets and the prepared folder checked by preflight;
  - the Python configs' args byte-identical to `main`'s.
- `config/prod-zig.env`.
- `scripts/v2-rollback.sh zig` (`perf1` returns to Python).
- `canary.py --skip`.
- `docs/ZIG-SERVE.md`.
- The watchdog tick (`/health` + exit 70) and the systemd units need no change.

### 0e. Prod knob parity: the engine knobs (`dsq-zig-knobs-1`)

Every key of `dsv41-tensorfold-spark/config/prod.env` (PERF1, engine 8474f31), checked against the Zig port and the Python reference (`tensorfold-decode1` at 8474f31). Classes:
- **(a)** changes the bits, and the Zig emitters already match it: the captures were taken with prod's knobs (`M1_KNOBS=prod`, `tools/zig/dsv41_m1/prod-knobs.env`; R1 through `r1c-knobs.env`);
- **(b)** changes the bits, and Zig does not match it;
- **(c)** speed only;
- **(d)** memory or admission;
- **(e)** serving or ops (`dsq-zig-ops-1`, section 0d);
- **(f)** Python-only, with no Zig meaning.

**(a) Matched bits.** The evidence is the M1 replays and the block forward on prod's captures (pods 8 / 9: 1,254 / 1,254 launches, 717 / 717 calls), the 2,048-row prefill capture (pod 15, cap15), the short segments (cap24-1 / 12 / 16), the R1 capture (pod 22, cap22: decode and DSpark 0 mismatches, the R1 M2b gate graphs off and on), M2c-1 (pod 14: top-p 0.85) and the prefill gates (pods 23 / 25).

| Knobs | How Zig matches them |
|---|---|
| `TF_DSV41_PREFILL_KERNELS=fast`, `FAST_EXPERTS=gm`, `PF_COPIES=1`, `MHC_PF=1`, `PREFILL_GATHER=bf16`, `PREFILL_ATTN=fused`, `_ATTN_BMQ=32`, `_ATTN_WARPS=8`, `PREFILL_ROUTER=gemv`, `INDEX_STREAM_MIN=4096`, `PLAIN_TRITON=1`, `ROPE_INPLACE=1`, `MHC_BM=64`, `MHC_WARPS=8`, `EXPERT_PREFILL=0` | the prefill emitter (`block_prefill.zig`) is prod's path; cap15 / cap18 / cap24-* check 0 mismatches |
| `ROUTER=gemv`, `DENSE=0`, `MHC_DEC=1`, `MHC_FIN2=1`, `RMS_PAIR=1`, `PARTIAL_BF16=1`, `RG_NARROW=2`, `RG_TAIL=1`, `INDEX_DTOPK=1`, `ENGRAM_FUSE_DEC=1`, `MHC_FN=bf16`, `MHC_CUDA=1`, `ATTN_CUDA=1`, `DENSE_V3=1`, `PDL=0`, `INDEX_KV=fp8`, `EXPERT_LOADS=1`, `KV_DTYPE=fp8` (serve.sh's `--kv-dtype`) | the decode emitter (`block.zig`, `block_moe.zig`) is prod's path; M1 pods 8 / 9, `check` on every capture |
| `EXPERT_TOPP=0.85`, `EXPERT_MIN_K=3`, `EXPERT_RENORM=kept` | `TF_DSV41_EXPERT_TOPP` (same name), min-k 3 and kept renorm in the emitter; M2c-1 PASS |
| `ATTN_SPLIT=4`, `ATTN_ROPE=1`, `MHC_TAIL=1`, `MHC_DEFER=1`, `MOE_BF16=1`, `PRUNE_KIT=1`, `RG_PRUNE=1`, `RMS_FOLD=1` (R1; the same bits) | `block.Options.r1`; cap22 0 mismatches, R1 M2b PASS. **Ported here:** the served model also turns R1 on from these eight Python names at prod's values (`prod_knobs.r1`; it only read `TF_DSV41_R1=1` before, which still wins). A partial set is logged and runs without R1 |
| `TF_DSV41_ENGRAM_DIR` | R3 (`engram_rows.zig`, `dsq-zig-speed-1`): the table's own rows; without it the served engine used synthetic rows |
| `BRANCHES_PRIO=side`, `L2PF*` | their launches are in the captures (L2 prefetch replayed for timing); see (c) for `BRANCHES` |

**(b) Bits not matched.**

| Knob | State |
|---|---|
| `TF_DSV41_PREFILL=replay` | **Ported, default off, Spark gate pending** (`dsq-zig-ced-1`, `ced.zig`; section "CED replay prefill" below): the encoder pass, the stash ring, the decoder replay over the prompt's tail with windows from R0, replay-mode prompt snapshots. Host-gated; the GPU proof is the M2c-2 kit with `PREFILL_MODE=replay`. Until that passes, Zig's replies equal Python's with `TF_DSV41_PREFILL=full` only (prompts over 128 tokens). |

**(c) Speed only.**

| Knob | State |
|---|---|
| `TF_DSV41_PF_DENSE=fused`, `PF_DENSE_TABLE=/cache/pfdense-table.json` | **Ported** (`prod_knobs.zig`, `block_prefill.Seg.fused`). Upstream's `rot_in`, then one `tf_dsv41_pfdense_v1.gemm` a projection over the whole layer, with the words the unpack read and the outputs the `_gemm` column blocks wrote, and no W_q workspace (`s.pf.w`, 64 MiB, leaves the plan). The configuration is `pfdense.pick`'s: `_CFG` override, then the JSON table (its `small` at ≤ `small_rows`), then the heuristic on 48 SMs. Ineligible widths (lanes 16, K2 < 8) keep the old path, as in Python. The same bits by pfdense.py's proof. Off by default. |
| `TF_DSV41_PREFETCH_AHEAD=1` | **Ported** (`forward_prefill.readAhead`). Before a prompt segment runs, the next segment's Engram reads are issued with the lookback they will see. `engram_rows` keeps an unread bulk batch through one read of its table (dropped after a second: a wrong guess costs a read, never a bit). Needs `TF_DSV41_ENGRAM_DIR`; `TF_DSV41_ENGRAM_PREFETCH=0` turns it off with the rest of R3's read-ahead. Off by default. |
| `TF_DSV41_PLAN_LINK=rdma`, `PLAN_PIN=auto` | **Ported, as an equivalent** (`tp/config.zig`, `tp/planlink.zig`). Python sends the plan through the RoCE host mailbox and rank 1 spins on a flag. Zig keeps its TCP plan link; with `rdma` the follower polls the socket for `TF_DSV41_PLAN_SPIN_US` (5,000) before it blocks, so a plan is read without waking a sleeping thread. `PLAN_PIN=N|auto` pins the sending (rank 0) and receiving threads at their first plan; `auto` picks the highest `cpu_capacity` other than the RoCE proxy's CPU, and stays unpinned where the capacities are all equal. `tcp` / `nccl` / unset: as before. Only the Sparks measure it: rank 1's window-entry lag (Python G14: +130 → −9 µs). |
| `GLM53_TF_COMM_BACKEND=roce`, `GLM53_TF_ROCE_MAX_KB=1024`, (`GLM53_TF_ROCE_FALLBACK`) | **Ported:** `tp.Config` reads prod's names where `TF_COMM_BACKEND` / `TF_TP_MAILBOX_MAX_KB` / `TF_TP_ROCE_FALLBACK` are unset (ours win). 1,024 KiB keeps 4-slot x 16-row windows (640 KiB) and 64-row lanes rounds on RoCE; at Zig's default 256 KiB those go to NCCL. |
| `TF_TP_ROCE_FALLBACK` (prod: unset) | **Settled: `nccl`, by leaving it unset.** Prod sets no `GLM53_TF_ROCE_FALLBACK`, so `roce.py`'s default holds: a failed RoCE setup or load-time probe on either rank puts both ranks on NCCL, and the server comes up slower rather than not at all. Zig's default is the same `nccl`, agreed by both ranks in the settings digest. `error` belongs only to gates and kits, where a silent fall back to NCCL would hide a RoCE failure (the M2c-2 kit sets it). |
| `GLM53_TF_ROCE_FAST=1` | **Partly matched.** Python's `gather_fast_kernel` makes seven changes to one exchange's critical path. Zig's `tp_roce` (`mailbox.cu`) already has one: the local shard is copied while the peer's travels. The others are not ported: a one-round-trip prologue, lean release/acquire order without `fence.sc.sys`, per-block waits, staggered pollers, no system-scope tail read, and two loads before stores. It is kernel work for `dsq-zig-tp-1` and only a two-Spark run can measure it. Zig's exchange is 8.9 µs against NCCL's 21 µs today. |
| `TF_DSV41_ENGRAM_GATE=1` | **Not ported.** In Python the GPU, not the host, waits for decode's Engram rows (`engram_gate.cu`: it spins on a host-mapped word, then copies, before layers 1 / 14; C2 / C4 +3.5 / +2.6 %). The kernel and its launcher are in Zig (`dsv41_engram_gate`, `kernels_ops`). Missing: the emitter's `gate_copy` calls at layers 1 / 14 under a knob, the forward's host-mapped rows and flags, and the reads completing into them while the graph runs. It needs a GPU bits gate (on == off). Owner suggestion: `dsq-zig-speed-1`, next to R3's read-ahead. |
| `TF_DSV41_GM_V2=1` | **Ported, default off, pod step pending (`dsq-zig-gm2pf-1`, `gm2pf.zig`; 0a "Pod steps").** x3gm v2 is bit-identical to v1 (Python's gate-eq on both packs; dsq-zig-kern-1's oracle covers gm2 at prod's shapes; the fatbins are built from the v2 source, 135 / 135 SASS-equal). With the knob set, Zig's prefill emitter runs x3gm.`_run_v2`: one plan and one `gateup2` / `down2` a MoE call (`split`: a launch a width), for v2's prefill speed (q28-v2 cold prefill +7.1 % at 8K, +6.7 % at 32K over v1). Unset, it launches v1 as before. Written against Python's code: the captures so far are v1 (their twins predated v2), so the pod step captures v2 on the midprofile twin and checks it call by call. |
| `TF_DSV41_BRANCHES=1` | **Not matched:** Zig issues the indexer / compressor on the main stream (the same bits), while Python forks them to a side stream (1-row window −0.53 ms). It needs fork / join events in `run.zig` and graph capture across two streams. |
| `TF_DSV41_FULL_CONE=1` | **Inert in prod.** It only acts with `TF_DSV41_PREFILL=full` (`cone.py`); prod runs replay. It shares replay's encoder / decoder split, so it comes with the CED port. |
| `TF_DSV41_ENGRAM_NATIVE=1`, `NATIVE_DIR` | Matched by an equivalent: `engram_rows.zig` is native (O_DIRECT, io_uring) for prefill and decode alike. |
| `TF_DSV41_GRAPH_MODE=rows` | Equivalent: row graphs with several slots (`TF_DSV41_SLOTS`, `TF_DSV41_GRAPHS=1`; section 0b). |
| `TF_DSV41_SPEC_DRAFT=1` | `dsq-zig-spec-1`. |
| `TF_DSV41_CALIB=real`, `CALIB_DIR`, `DEPTH_JOINT=0` | Draft pricing policy: `draft/calib_env.zig` reads them by these names. |

**(d) Memory and admission.**

| Knob | State |
|---|---|
| `TF_DSV41_PREFILL_CHUNK=2048`, `PREFILL_ROWS=2048` | **Ported:** a segment's rows = the smaller of the two (multiples of 16 in 32–2,048), read at boot before the buffer plan, on the served model and the M2b driver. Prefill is segmentation-invariant (memory.py, G19), so the bits do not change. |
| `TF_DSV41_PREFILL_ADAPT` (default 1), `PREFILL_ADAPT_GIB=4.5` | **Zig equivalent: static.** Python steps a round's rows down to 1,024 / 512 because torch allocates the chunk's transient each round. Zig has no per-round transient: the prefill program's buffers are in the boot buffer plan (`planPrefill`), behind the measured boot check. The memory a smaller segment would save is chosen at boot with `TF_DSV41_PREFILL_CHUNK`. |
| `TF_DSV41_INDEX_BUDGET_MIB=64` | **Ported** (`block.Options.index_budget`, set by `prod_knobs.apply` on the served model and the M2b driver): the prefill indexer's scores / sort keys in row blocks under prod's 64 MiB, not 256. Only the row batching changes, not the values (csa2 backend), and the scratch shrinks by 192 MiB a rank. The default stays 256, which the captures used. The kv budget's prefill price reads the same knob (`kv_state.bootCheck`). `longpf`'s stream top-k keeps its own 256 MiB BUDGET (`stream_topk.plan`), as Python does: next row. |
| Long-prompt stream scratch (`L.ix.stream`) | **Counted.** Python allocates it per launch, so the measured terms (`kv/budget.zig` Serve, taken on Python) hold none of it. Zig plans it at boot and keeps it resident: 256 MiB at prod's 1M context with 2,048-row segments (a ratio-2 layer's 512K keys: 32 splits, 1,024 rows a launch), 0 under 4,096 compressed keys. `prod_knobs.streamScratch` sizes it (host test: equal to the emitter's own role at the limit), and the measured boot check adds it to the boot growth (`kv.Options.resident`; logged as `kv: boot growth + ... GiB`). Admission's prefill price stays Python's, because nothing transient remains to price. |
| `TF_DSV41_POOL_TOKENS=1400000`, `FLOOR_HARD_GIB=4` | M5 (`kv_state.zig`, `kv/budget.zig`): the pool and its measured boot check. |
| `TF_DSV41_FLOOR_GIB=5` | Validated against the hard floor; the measured boot check warns under it (section 0d, `dsq-zig-ops-1`). |
| `MALLOC_ARENA_MAX=4`, `MIMALLOC_ALLOW_THP=0` | (f) for the engine (torch's allocators). Zig's host allocations are its own; glibc's arenas still apply to libc calls, so serve.sh may keep passing them. |

**(e) Serving and ops** (`dsq-zig-ops-1`): `WORKER_SSH`, `HEAD_IP`, `HEAD_SSH`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `MASTER_PORT` (Zig: `TF_TP_PORT`; the plan link at +7 as Python), `TF_REPO`, `TF_COMMIT`, `HEAD_TF_ROOT`, `WORKER_TF_ROOT`, `BASE_IMAGE`, `IMAGE`, `CACHE_VOL`, `PYLIB`, `HEAD_MODEL`, `WORKER_MODEL`, `HEAD_ENGRAM`, `WORKER_ENGRAM` (→ `TF_DSV41_ENGRAM_DIR`), `HEAD_STATE`, `WORKER_STATE`, `NAME`, `HOST`, `PORT`, `SERVED_NAME`, `ALIASES`, `CONTEXT` (→ `TF_DSV41_CONTEXT`, longpf), `PARALLEL` (→ `TF_DSV41_SLOTS`), `MAX_TOKENS`, `TF_DSV41_SESSIONS`, `SESSION_RAM_MIB`, `SESSION_DISK`, `SESSION_DISK_GIB`, `SESSION_DISK_MIN` (read by `sessions_gpu.zig`), `TF_DSV41_GRAMMAR`, `TOOL_GRAMMAR` (`dsq-zig-grammar-1`), `TF_DSV41_THINKING`, `DEFAULT_EFFORT`, `SCHEMA_PROMPT` (the server reads them), `TF_DSV41_REQUEST_LOG`, `DISCONNECT`, `TF_DSV41_IMAGES`, `BIAS_VL`, `VISION_PREP_MB` (`dsq-zig-vision-1`), `DROP_CACHES`, `MEM_GATE_GIB`, `MEM_GATE_TIMEOUT`, `CANARY`, `READY_TIMEOUT`, `PREFLIGHT`, `WATCH_HEAL`, `WATCH_FAILS`.

`TF_DSV41_PREPARED`, `PREPARED_WRITE` (boot time) are ported by `prepared.zig` (section 0d).

**For `prod-zig.env` (`dsq-zig-ops-1`).** Its comment "Left out ON PURPOSE ... the Zig engine reads none of them" is out of date for these, which Zig now reads under prod's names:
- add now (the same bits, no gate needed): `GLM53_TF_ROCE_MAX_KB=1024` (or `TF_TP_MAILBOX_MAX_KB=1024`), `TF_DSV41_INDEX_BUDGET_MIB=64`, `TF_DSV41_PREFILL_CHUNK=2048`, `TF_DSV41_PREFILL_ROWS=2048` (the default, written out);
- add after the knobs pod step passes: `TF_DSV41_PF_DENSE=fused`, `TF_DSV41_PF_DENSE_TABLE=/cache/pfdense-table.json`, `TF_DSV41_PREFETCH_AHEAD=1`;
- add after the Spark window measures them: `TF_DSV41_PLAN_LINK=rdma`, `TF_DSV41_PLAN_PIN=auto`;
- add after the `M2B_GM2PF` pod step passes: `TF_DSV41_GM_V2=1` (prod's value; the same bits);
- leave out, since Zig has no effect for them: `GLM53_TF_ROCE_FAST`, `TF_DSV41_ENGRAM_GATE`, `TF_DSV41_BRANCHES*`, `TF_DSV41_FULL_CONE`, `TF_DSV41_PREFILL_ADAPT*`;
- add after the CED replay gate passes (section "CED replay prefill"): `TF_DSV41_PREFILL=replay` (prod's value; Python's server default);
- `TF_TP_ROCE_FALLBACK` / `GLM53_TF_ROCE_FALLBACK`: unset (`nccl`), as prod.

**(f) Python-only:** `TF_DSV41_NATIVE_DIR` (the C reader's build dir), `PYLIB`, `MIMALLOC_ALLOW_THP`.

**Defaults and gates.** Every port is off unless its prod knob is set; R1 by its Python names is the one exception (the same bits, already gated). With the knobs unset, the emitters are unchanged: `check` cap24-1 / cap24-12 / cap24-16 / cap15 / cap18 / cap22, `dcheck cap16` and `m5-emit cap18` give 0 mismatches. Host: `zig build test-dsv41-knobs` (the readers; the fused dense prefill on the real emitter at 2,048 / 300 / 12 rows: every other call unchanged and in order, each fused GEMM fed by `rot_in`'s workspace, whole-layer words and outputs, Python's pick), `test-dsv41` (`engram_rows`: a bulk batch issued ahead survives one unrelated read and serves the next with 0 misses; a wrong guess is dropped after two), `test-tp` (prod's names, the spin with a plan arriving late, the pin); `test-dsv41-knobs` also checks the stream scratch's size against the emitter at the 1M limit (256 MiB; 128 MiB with 512-row segments).

**Pod step** (`tools/zig/dsv41_knobs/job.sh`, `knobs_gate`; sourced inside `M1_JOB=m2b` like `dsv41_slots`; the integrator adds `M2B_KNOBS=1`):
- **Needs:** the M2b reference (`refs`) with a prompt over 2,048 rows (`M2B_PROMPTS="2600"` or more, with `M2B_PREFILL_GATE=1`). `ahead` and `prod` need `M2B_ENGRAM=1`. Optional: prod's `/cache/pfdense-table.json`, copied read-only from the Spark cache volume `dsv41-perf1-cache`, as `KNOBS_PFD_TABLE`.
- **Runs** (`KNOBS_RUNS`, default `pfd rows1024 rows512 budget64 ahead prod`): each one is the `M2B_PREFILL` gate (`tf-dsv41-m1 m2b`: Zig's prefill from an empty slot against the reference's dump), with one setting on the Zig ranks:
  - `TF_DSV41_PF_DENSE=fused`;
  - `TF_DSV41_PREFILL_CHUNK=1024`, then `512`;
  - `TF_DSV41_INDEX_BUDGET_MIB=64`;
  - `TF_DSV41_PREFETCH_AHEAD=1`;
  - all of prod's together.
- **Compares:** bits on == off, since the reference is the knobs-off truth. Each `knobs/<run>/rank*.log` must give `PASS Zig prefill rank r: ... state roles 65/65 equal ...` and the decode gate's PASS (64/64 tokens and logits). Rank 0 logs `prod knobs: ...` with the run's settings.
- **Speed:** each run's prefill seconds against the `m2b-prefill` run on the same prompt. `pfd` gives the fused GEMM's gain.
- **Minutes:** about 3 a run (no Python load), about 18 for all six. The `rows512` run also proves that the reference's AOT set covers 512- and 1,024-row segments; on `MissingTritonVariant`, add `--aot-prompts` with those lengths.
- **Spark only** (not a pod): the plan link spin and pin (rank 1's window-entry lag with `TF_DSV41_PHASES`), prod's `GLM53_*` names on the hybrid, and the fused dense prefill's speed on GB10 with prod's table.

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
    - Prompts whose full-mode layers see more than 4,096 keys need the stream top-k, which block_prefill does not port (`error.Unsupported`); ported since by `dsq-zig-longpf-1` (§0a, Pod steps).
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
  - **Refused (`error.Unsupported`):** segments of 16 rows or fewer (they go through the decode kernels) and the streamed top-k for 4,096+ keys (both since ported: short segments in 6587de9, the stream top-k by `dsq-zig-longpf-1`).
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


### Spark overnight window, 2026-10-07/08 (`dsq-zig-night-1`): outcome

Both Sparks. **Prod (Python perf1) down 15:16:36 -> 22:42:12 UTC (7 h 25 min)**, restored with `scripts/v2-rollback.sh perf1`
(17 x 23 -> 391, `serve.sh status` both ranks on 8474f31, a chat completion answered, watchdog + rearm timers on, lease
removed; `config/prod.env` byte-identical to before the window, the zig-serve script copies replaced by the originals). Logs: the quant repo's
`results/zig-night-20261007/` (`r0/`, `r1/`, `NOTES.md`). Kits were rebuilt eight times as fixes landed (clean `git archive`,
aarch64, sm_121 fatbins = `dsv41-zig-ref/fatbin-kit` with the vision fix's `dsv41_vision.fatbin` 6f2ab33a); the last is
tree `acae981`. Prod's own numbers were measured first (perf1, before the stop).

| step | result |
| --- | --- |
| M2c-2, whole model, TP=2 over RoCE, R1, top-p 0.85 (kit 4144278) | **PASS** both ranks: tokens 512/512, logits 512/512, first-window blocks 40/40 |
| `PREFILL=1`, 1,536 rows | **PASS** both ranks: state roles 95/95, first token 223 == 223, then 512/512 |
| R1 timed windows (`PROF_ROWS="1 2 4" STEPS=64`, a585336) | **PASS** both ranks: 64/64 tokens + logits, 40/40 blocks, no fault. Zig R1 1 / 2 / 4-row windows 30.39 / 31.26 / 32.30 ms mean (both ranks alike) |
| run A, 1 slot, drafts + spec pass + sampling + R3 Engram + graphs | **PASS** after 3 fixes: 16/16 bench cells, 8/8 requests; CLI replay **49/49 equal**, DSpark 9,682 / 14,460 = 0.670, speculative passes 800 / 800 used |
| run B, 4 slots, drafts off | **PASS** after the pool-aot phase: 16/16 cells |
| run B2, 4 slots + per-slot drafts (prod's shape) | **PASS** after 1 fix: 16/16 cells, 8/8 requests; CLI replay at 4 streams **49/49 equal**, DSpark 9,482 / 14,231 = 0.666 |
| run C, 1M context (replicated 1.4M pool, graphs off) | 32K and 131K **PASS** (128-token answers); 619,688-token prefill completed (no error) but the bench client's 900 s timeout left before the first token. Split KV not run (no Python tree with a split pool on the Sparks) |
| Zig as prod (part 4, image `tensorfold-dsv41-zig:acae9818440c`, zig-serve + `2d80119`) | first boot ready after 810 s (44 prepared layer files written; `4 slot(s) x 4096` - `CONTEXT` never reached the engine, fixed in serve.sh `2d80119`); canaries chat / thinking / tool / json / tokenize ok; **fast boot 375 s** (44 read, 0 built; weights 367 s, 261 s of it upload + digest; Python: 33-48 s); wire probe: every shape == Python's except `/health` `prefill_seconds_total` (Python int 0 before any prefill); request log 41 keys a line; disconnect -> `finish: cancelled`, inflight 0; **fail-fast**: rank 1 killed 22:34:09, rank 0 exit 70 ~1 s later, the first watch tick healed, serving + 17 x 23 -> 391 at 22:40:52. **At `CONTEXT=1048576` with 4 slots rank 1 refused `UnknownRowsCall` at boot** (row mode does not rewrite the long-context selections; Python runs `_dtopk` / `_stream` with `ROWS=1`), then segfaulted on exit |
| Zig prod vs Python prod, the same 8 greedy requests (`requests.sh`, thinking on, 41-60-token prompts) | **5/8 replies equal** (256 tokens each, same prompt hashes); 2, 5, 7 differ inside the reasoning (Zig stops at 240 / 217 tokens where Python runs to 256) |
| images (V1-V5, `tools/zig/dsv41_vision/spark.sh`) | V1 PASS both ranks; **V2 FAIL 0/3**: first differing stage b0.attn, 3 / 12 / 8 elements (photo / wide / small) 1 bf16 ulp (pod 30: 0.35-0.6 % of elements); resync keeps b0.attn's counts. V3-V5 skipped. Keep `IMAGES` off |

**Speed, Zig against prod (perf1), tok/s, best of 2, 384 tokens, thinking off:**

| workload | T | streams | prod | run A (1 slot, 8914a48) | run B (4 slots, no drafts) | run B2 (4 slots + drafts, 2d92469) |
|---|---|---|---:|---:|---:|---:|
| code | 0 | 1 | 84.7 | 44.6 | 40.6 | 60.4 |
| code | 0 | 4 | 143.6 | 45.5 | 93.6 | 82.2 |
| code | 0.7 | 1 | 87.3 | 42.2 | 40.6 | 58.1 |
| code | 0.7 | 4 | 141.8 | 45.8 | 90.4 | 90.8 |
| prose | 0 | 1 | 51.6 | 22.3 | 40.2 | 34.3 |
| prose | 0 | 4 | 96.4 | 23.6 | 94.2 | 57.4 |
| prose | 0.7 | 1 | 47.4 | 22.5 | 40.2 | 35.2 |
| prose | 0.7 | 4 | 95.5 | 23.7 | 91.2 | 55.9 |

Run A predates `9bc88af` (the DSpark pass 31.5 -> 7.0 ms a round); B2 has it. **T0 replies equal to prod: 0/20** in every run
(the bench keeps hashes only; see open items).

**Long context (run C, eager decode):**

| prompt tokens | prod TTFT s (tok/s) | Zig TTFT s (tok/s) | prod decode | Zig decode |
|---:|---:|---:|---:|---:|
| 37,974 | 15.6 (2,430) | 67.4 (563) | 50.5 | 36.4 |
| 154,183 | 68.1 (2,263) | 263.4 (585) | 54.0 | 38.9 |
| 619,688 | 371.2 (1,669) | ~1,319 (~470; prefill done, client gone) | 43.2 | not measured |

**Fixes made in the window** (each general; own-hunk commits):
- `5df2977` slots `ref.py`: Pool's `split=` / `rank=` only where the tree takes them (prod's 8474f31 Pool has neither: rows-aot failed).
- `d44c167` `model.zig`: `rope.bin` read without the 256 MiB cap (the 1M limit's tables are 538 MB: every rank's load failed `StreamTooLong`).
- `817ff07` kit `taps-aot` + `dsv41_m2b_ref.py --aot-taps`: a drafting server's prefill launches the tap layers' `_site` with `TAP_ON`; no reference recorded one.
- `8914a48` `dspark_gpu.zig` / `dspark_slots.zig`: DSpark's host uploads on the runner's stream. A pageable `cuMemcpyHtoD` returns before its DMA lands and the non-blocking stream does not wait, so the candidates' all-gather could send stale bf16 partials as token ids (rank 0 panicked in `dspark.bias` on id 0x4002C00B = two bf16s). Gathered candidates are checked on every rank (`error.BadCandidates`); `2d92469` limits the slots' check to the pass's rows.
- `9bc88af` `draft/dspark.zig`: candidates by a bounded selection under the same (value desc, id asc) order instead of sorting the rank's whole vocabulary a row (40.9 ms a pass; the same first k, tested against the full sort with ties).
- `1ce0d7c`, `73418f9`, `fe661fd`, `607a561`, `943eb82` kit `pool-aot` (+ `dsv41_m5_kv.py` on prod's Pool): prompts and decode on the paged pool at the served limit - `_fused` / `_scores` / `_block_keys` / `_stream` / `_dtopk` variants the contiguous references never launch. Every one was a gap in the AOT set, not in Zig's arguments: Python passes the same values (`nsplit` is a runtime int: Triton folds 1, so a set with only one-split streams made Zig treat it as a constexpr).
- `e4fce04` `aot.zig`: a missing variant's report lists the launch's constexprs.
- `115b46e`, `acae981` `block.zig`: past attn_cuda's 8 x 512 x 31 entries a selection runs Python's Triton `_dtopk` (`dtopk.select`, then the candidate source's blocks as their own CUDA job, or `dtopk.blocks` past 126,976 blocks) as `backend.select` does when `attn_cuda.plan` is None (decode at 154K: `error.Shape` on both ranks; the 1M plan refused at boot).

**Open, from the window:**
- **Zig prod != Python prod on 3/8 short greedy requests** (and the bench's T0 cells 0/20, thinking off, streamed) while Zig's served path == its own CLI replay (49/49) and the M2c-2 / prefill gates are bit-exact. Prompts render alike (same head hashes and counts). **Cause found offline: the prompt's last token** (next section): fixed by `TF_DSV41_PROMPT_TAIL=verify` (the default), proof steps under NEXT SPARK WINDOW.
- **4 slots x 1M on Zig:** row mode needs the long-context selections (`_dtopk` / `_stream` with `ROWS=1`) before prod can switch; one slot at 1M works (run C). **Ported offline (`9bb3227`):** dtopk with `ROWS` and `backend._blocked` under the index budget (Python never streams a `RowWin`); proof steps under NEXT SPARK WINDOW: 4 slots x 1M and the AOT fill.
- **Fast boot is upload-bound:** 261 s of the 367 s is "upload + digest" with the prepared images already read.
- **`/cache/roce-failed`** exists in prod's cache volume (preflight: "both ranks start on NCCL"); delete it to retry RoCE, if that is intended.
- **Long prefill ~4x slower than prod** (563-585 tok/s vs 2,263-2,430): the stream top-k path and 2,048-row segments; profile with `TF_DSV41_PHASES`.
- **Graphs with one slot on the paged pool** pad NK to the 2,048 context bucket: no reference decodes through Python's graph buckets (`_scores` paged NK 2048 missing); run C ran eager.
- **Speed at short prompts:** B2 code T0 1-stream 60.4 vs 84.7; 4 streams 82-91 vs 142-144. The window's choice reads wait on the GPU forward; next is the per-round phase split (`out/b2/phases.json`).
- **The AOT set is sampled, not complete:** a served shape no reference launched fails `MissingTritonVariant`. A fill step that compiles the emitter's specs straight from Python's Triton source (the same cubins its JIT builds) would close the class. **Built offline:** `tf-dsv41-m1 aot-needs` + `tools/zig/triton_fill.py` (build-kit `TRITON_FILL=1`); same section.
- Images: V2 still 1 ulp at b0.attn (owner `dsq-zig-vision-1`).

### Zig prod vs Python prod: the prompt's last token (`dsq-zig-diverge-1`, offline)

**What the night recorded.** `requests.sh` keeps usage and the content's first 80 characters, so Python's reasoning
and ids are not in the logs, and neither server reports logprobs. No first differing index or margin can be read from
them. They give bounds: Zig closes its reasoning (`</think>`) at reply token 240 / 105 / 198 (replies 2 / 5 / 7, from
`out/a/trace.jsonl`), while Python is still reasoning at 256 (2, 5) or starts its content about 9 tokens before 256
(7). So the first differing token is inside the reasoning, before 240 / 105 / 198.

**Window shape is ruled out on both sides.** Zig's T0 bench hashes are identical in run A (1 slot, drafts), run B
(4 slots, no drafts) and run B2 (4 slots, drafts), and Zig prod's 8 replies (part 4) equal run A's. Python prod's
hashes are identical between 1 and 4 streams in both reps. Each engine is invariant to drafting, verify rows and row
mode, so the difference is a fixed one.

**The cause: where the prompt's last token runs.** Python's server never prefills a prompt's last token:
`batch.py` `_pieces` prefills `[0, n - 1)`, `finals` makes `prompt[-1]` the slot's pending token, and `rounds.py`
`finish_prompt(slot, n - 1, ...)` runs before the first verify window, whose row 0 is that token. That row is a decode
row: decode kernels, decode routing with expert top-p 0.85 / min-k 3 (`PR.phase(False)`), and R1. Every Python
reference the gates use (`dsv41_m2b_ref.py`, `dsv41_samp_ref.py`, `grammar_ref.py`, the slots `ref.py`,
`dsv41_m1_capture.py`) calls `Forward.prompt`, which prefills all n rows: prefill kernels, every routed expert. Zig's
own prefill mirrors `Forward.prompt` (the gates pass), and the served `GpuTarget.prefill` took the first choice from
the last prefill row. So Zig's served path had the gates' shape, not the server's. One row's KV, ratio-2 carry, index
key and first choice then differ (a whole top-p cut, not an ulp), and the replies drift from there. That fits 5/8
equal on the thinking requests and 0/8 on the bench's distinct T0 prompts (384 tokens, more room to drift).

**Ranked below it** (each with its evidence; none needs a change for these prompts):
1. CED replay (`TF_DSV41_PREFILL=replay`, not ported). Python's `replay.py` and `test_dsv41_replay.py` show it equals
   full prefill for prompts of 128 tokens or fewer (R0 = 0); every prompt here is 40-60 tokens. It still matters past
   128 tokens (section 0e).
2. `TF_DSV41_GM_V2=1`, `PF_DENSE=fused`, `MOE_BF16` and the other prefill knobs: the same bits by Python's own gates
   (gate-eq, pfdense.py's proof). Zig part 4 had `PF_DENSE=fused` and gave the same replies as run A without it.
3. RoCE vs NCCL (prod ran on NCCL because of the stale `/cache/roce-failed`): the exchanges carry bf16 values both
   ways, a copy.
4. Calibration, depth and the speculative pass: these choose drafts only, ruled out above.
5. Engram: R3 reads the same shards and was gated against Python (pod 26, prompts 45 / 173). `ENGRAM_GATE` and
   `ENGRAM_NATIVE` are exact by Python's design.
6. Greedy ties: both take the lower id after the ranks' fp32 maxima, so only an exact fp32 tie could differ.

**The fix (default on).**
- `prod_knobs.PromptTail` / `TF_DSV41_PROMPT_TAIL` (rank 0's choice; the followers replay its operations):
  - `verify` (default): the served target prefills `[0, n - 1)`. The last token then runs as the target's own decode
    window (one slot, or row mode's `Batch.targetWindow`), with its greedy or keyed choice at the request's draw. The
    window is left pending, as any chain window is: the next window keeps it, and `release` keeps it before a session
    save.
  - `prefill`: the old shape.
- `promptSplit` is the one rule both shapes share.
- `m2b.zig`'s `M2B_PREFILL` gate follows the reference's `ref.json` `prompt_tail`. `dsv41_m2b_ref.py
  --prompt-tail verify` runs Python's server shape, and the M2c-2 kit passes it as `PROMPT_TAIL`.
- The pod steps that replay `Forward.prompt` references through `generate` (samp, grammar, vision) pin
  `TF_DSV41_PROMPT_TAIL=prefill`.
- Checks: `test-dsv41-knobs` 5/5 (the reader and the split against `batch.py`'s rule). `check` cap15 / cap18 / cap22 /
  cap24-1 / cap24-12 / cap24-16 and `dcheck cap16` give 0 mismatches; no emitter changed.
- Speed note: the drafter's first ingest after a `verify` prompt sees one taps row (the window's). Its context restarts
  there, as for a resumed session's prompt. Python's drafter sees the prompt's taps. The first rounds' acceptance may
  dip; ingesting the prefill's taps before the tail window is the follow-up, if the numbers show it.

#### NEXT SPARK WINDOW: proving the prompt tail (not run)

The kit is built from the fix's commit or later. It also carries `tools/zig/dsv41_diverge/diverge.py` and the M4
`prompts.txt`. Before any step, check the lease on dgx-01. `OUT=~/zig-window-kit/out/diverge` on dgx-01.

0. **No downtime (prod up, about 4 min).** Get Python prod's exact ids (`return_token_ids`) for the 8 requests and the
   bench's 8 T0 prompts, drafts on and off, then compare them with the night's recorded Zig ids:
   ```
   K=~/zig-window-kit; D="python3 $K/tools/zig/dsv41_diverge/diverge.py"; TOK=/home/dgx01/ai/models/dsv41-q28-v2/tokenizer.json
   $D ask --url http://127.0.0.1:8000 --prompts $K/prompts.txt --bench --out $OUT/py.jsonl
   $D ask --url http://127.0.0.1:8000 --prompts $K/prompts.txt --bench --no-draft --out $OUT/py-nd.jsonl
   $D compare $OUT/py.jsonl $OUT/py-nd.jsonl --tokenizer $TOK       # expect 16/16: Python drafted == serial
   $D fromtrace <night r0>/a/trace.jsonl --lines 42-49 --prompts $K/prompts.txt --out $OUT/zig-night.jsonl
   $D compare <(head -8 $OUT/py.jsonl) $OUT/zig-night.jsonl --tokenizer $TOK   # 5/8; prints each first differing index
   ```
   Record the `first_differing` index and its context for 2 / 5 / 7. If `py != py-nd`, Python's served bits depend on
   drafting. Stop there and report: Zig must then copy Python's window choices, which this fix does not do.
1. **Zig prod with the fix** (the usual Zig-as-prod switch, `prod-zig.env`, a new image; fast boot about 375 s):
   ```
   $D ask --url http://127.0.0.1:8000 --prompts $K/prompts.txt --bench --out $OUT/zig.jsonl
   $D ask --url http://127.0.0.1:8000 --prompts $K/prompts.txt --bench --no-draft --out $OUT/zig-nd.jsonl
   $D compare $OUT/py.jsonl $OUT/zig.jsonl --tokenizer $TOK         # PASS = 16/16
   $D compare $OUT/zig.jsonl $OUT/zig-nd.jsonl --tokenizer $TOK     # 16/16
   ```
2. **The A/B that names the cause.** Restart Zig prod with `TF_DSV41_PROMPT_TAIL=prefill` added to `prod-zig.env`
   (remove it afterwards). Then `ask ... --out $OUT/zig-prefill.jsonl` and `compare $OUT/py.jsonl
   $OUT/zig-prefill.jsonl`: expect the night's result again (the 8 requests 5/8, bench prompts differing). If step 1
   is 16/16 and step 2 is not, the prompt tail is the cause, confirmed.
3. **If step 1 is not 16/16**, the next difference is past the prompt. Name Zig's first token off Python's reply:
   `$D trace $OUT/py.jsonl <zig server trace.jsonl of the same requests> --tokenizer $TOK --out $OUT/py-trace.jsonl`,
   then `tf-dsv41-m1 generate /model /assets /out/py-trace.jsonl` through the M4 kit (`run_host.sh 0 OUT generate`,
   rank 1 `follow`), `DRAFTS=1` then `DRAFTS=0`. Each line's `first_differing` is then Zig's first token off
   Python's. The M4 server needs `TF_DSV41_TRACE_TOKENS` for the prompt ids (`run_host.sh` sets it).
4. **The logits-level gate (optional, about 30 min, prod down):** the M2c-2 kit with `PROMPT_TAIL=verify PREFILL=1`
   (`run_host.sh`). The reference runs Python's server shape, and Zig's `M2B_PREFILL` follows `ref.json`. PASS =
   65/65 / 95/95 roles, the first token, and then 512/512 tokens and logits. This is the first gate whose reference is
   the served shape.
5. Prod back on perf1 (`scripts/v2-rollback.sh perf1`), as in every window. Keep `py*.jsonl`, `zig*.jsonl` and the
   compare output in the quant repo's results.

### CED replay prefill (`dsq-zig-ced-1`, offline)

Prod serves with `TF_DSV41_PREFILL=replay` (prod.env; it is also Python's server default, `stack.py`
`DEFAULT_PREFILL = "replay"`). Zig ran full prefill, so for any prompt over 128 tokens Zig-as-prod left another
decoder state than prod (the night's 384-token bench prompts: 0/8 equal) and spent about twice prod's time on long
prefill. Commits `557dc713` (engine), `22044785` (aot-needs hook), `d4216b61` (reference + kit knob), `df3e30ee`
(fill env). Default off; `prod-zig.env` on zig-serve carries the line commented until the gate below passes.

**Python's semantics, and where each lives in Zig** (`replay.py`, `slots.py`, `rounds.py`, `batch.py`, `sessions.py` at 8474f31):

| Python | Zig |
|---|---|
| `decoder_start`: the first ratio-1 layer, a kv source (20) | `ced.decoderStart` (also refuses an Engram layer at or past it: `replay.finish` runs the decoder without Engram rows) |
| `encode`: each segment `run(upto=20)`: layers 0-19, layer 20's attention site, `compress_rows` (its comp rows and index keys for every row); `keep(n - 1)` as a full segment | `block_prefill.emitReplay(.encoder)`: the whole segment's calls through layer 20's site, then `compProj` + `compress`, then glue `ced_stash`; `forward_prefill.segment` commits as before (carries 2 / 8 / 14, Engram tail, position); no head |
| `Stash`: streams [m, 4D], normed input [m, D], layer 20's site pre / post / comb of the last 127 positions; a gap restarts it | a 128-row ring `s.ced.x / .out / .pre / .post / .comb` (row = position % 128) + `Slot.ced` (`ced.Stash`: from, hi, ids) |
| `finish(slot, n - 1, tail = prompt[max(0, n-128) : n-1])` before the first verify window; decoder from the stash, `ctx.lo = R0`, `comp_done`, Streams without `alt`, `_finish(logits=False)` | `Forward.finishPrompt` (op 44, every rank) -> `forward_prefill.finish`: `emitReplay(.decoder)` over layers 20-39, glue `ced_load` (ring -> `w.x0`, `w.out`, `w.c0.*`), `positions` with `w.lo` = R0, layer 20's compressor skipped, every boundary in place (`block_moe.mhcInto`, `_site` without SPLIT), no final norm or head; nothing committed |
| the last token: the first verify row, a full 40-layer decode row | unchanged: `TF_DSV41_PROMPT_TAIL=verify` (`a390113`) |
| n <= 128: R0 = 0, replay == full bit for bit | the same rule (`ced.tailOf`); the host test checks the call-level identity |
| sessions: no turn snapshots in replay mode; a prompt snapshot at S = (n-1)//16*16 when S > cached and S >= 64, with the stash; own tag | `target.prefill` splits the prefill at S and calls `Sessions.savePrompt` (op 45: an entry, the slot keeps its pages); `release` saves nothing in replay mode; the "slot" blob carries the stash's from / hi / ids, the ring rides as pieces; entries tagged `ced.tag` (16) |

**Interactions.**
- Chunked segments and 1M prompts: the encoder segments are the full segments' launches up to layer 20 (any segment
  size, the stream top-k, split-KV row blocks: `longpf.window` runs them); the decoder pass is 127 rows (or n - 1)
  once a prompt, never staged. The ring keeps rows across segments; the decoder pass emits at every tail shape to
  the 1M limit (host test).
- Split KV: the decoder's index layers exchange their selections as any segment does (`kx_*` glue); host test covers
  pooled and split programs.
- 4-slot row mode: a prompt's encoder segments and its decoder run inside one `target.prefill` on the activated slot,
  so one ring a forward suffices. Row mode opens no sessions (as before).
- Images: the decoder's tail ids come from the ring, so image rows route with `bias_vl` (`replay._image`).
- **Refused** (boot error, by name): replay without `TF_DSV41_OWN_PREFILL=1` or with `TF_DSV41_PROMPT_TAIL=prefill`
  (`Forward.prompt` never replays: Python runs replay only in the server's piece / finals shape); a forward whose
  layers do not reach layer 20; a config without a decoder layer.
- Not copied: Python feeds the replayed rows' DSpark taps to the drafter (`replayed`). Zig's drafter starts from the
  verify window's row, as after `a390113`; this changes draft acceptance, not bits.
- `rounds.zig` already encodes `Mode.replay` in the plan; the served path does not use rounds plans, the knob is
  read on every rank (the follower checks op 44 against its own mode and position).

**Proven offline** (HEAD + these files in a clean tree; other agents' WIP left out):
- `test-dsv41-knobs` 12/12, with `ced_test.zig`: encoder pass ++ decoder pass == the whole segment's calls with layer
  20's compressor moved to the encoder's end, at 12 / 30 / 64 / 100 / 127 rows, contiguous / pooled / split KV,
  R0 = 0 and 1000; the decoder's windows from R0, its boundaries in place, the stash roles; the decoder pass emits at
  every tail length from 0 and 127 rows up to the 1M limit; `ced.zig`'s rules against replay.py / sessions.py /
  batch.py.
- `check` cap15 / cap18 / cap22 / cap24-1 / cap24-12 / cap24-16 and `dcheck cap16`: 0 mismatches; `m5-emit cap18`:
  0 problems; `test-dsv41` 32/33 (1 skipped as before), `-longpf` 7/7, `-rows` 18/18, `-mdraft` 7/7, `-gm2pf` 5/5,
  `-draft` 32/32.
- Not offline: the values. No Python replay capture exists (it needs a GPU), so the bits are proven only by the
  M2c-2 gate below. `aot-needs` with replay was not run end to end (the dev box's Triton is 3.8; the fill needs
  3.7.1): the hook's programs are the ones the host test emits.

**Found on the way.**
- Python's server defaults to replay; any "Python full" comparison must set `TF_DSV41_PREFILL=full` explicitly.
- Python prod's fast prefill tag already forbids turn snapshots in full mode too (`rounds._finish`: `not tag >> 8`).
  Zig's one-slot sessions save turn ends in full mode, so a resumed multi-turn chat in Zig full mode reads
  decode-written rows where Python re-prefills them. Irrelevant for prod's row mode (no sessions) and for replay
  mode (now prompt snapshots only); a full-mode follow-up if one-slot Zig serves sessions with full prefill.
- The 4 x 1M window's step 5 (long T0 replies Zig vs prod) compares against prod's replay: it needs
  `TF_DSV41_PREFILL=replay` on the Zig server to be equal past 128 tokens.

#### NEXT SPARK WINDOW: CED replay (not run; right after the prompt-tail proof)

Check the lease on dgx-01 first. Kit and image from `df3e30ee` or later. `K` = the kit dir on both hosts.

| step | what | time | PASS |
|---|---|---|---|
| C0 dev box | the kit as in "4 slots x 1M" step 0 (`TRITON_FILL=1`, `TRITON_FILL_ENV=tools/zig/dsv41_m4/fill-4x1m.env`, which now has `TF_DSV41_PREFILL=replay`), `KIT_COMMIT=<df3e30ee or later>`; the zig-serve image from the same commit | before | `needs.log` has `aot-needs: CED decoder, ...`; `compile.json` `failed: []` |
| C1 gate, prod down | both hosts within a minute: dgx-01 `cd $K && PREFILL=1 PREFILL_MODE=replay PROMPT=1536 ./run_host.sh 0 $K/out/ced-1536`, dgx-02 the same with rank `1` (`PREFILL_MODE=replay` sets `PROMPT_TAIL=verify`) | ~40 min | `run.log`: `ours ... PASS` (decode from Python's replay state), `prefill gate ... PASS Zig prefill rank r: ... state roles N/N equal ..., first token a vs a` and `PASS M2b rank r: tokens 512/512, logits 512/512` on both ranks |
| C2 gate, two segments | the same with `PROMPT=2600 AOT_PROMPTS=""` (R0 = 2472 in the second segment, the ring across segments) | ~30 min | as C1 |
| C3 bits vs prod | Zig-as-prod (`prod-zig.env` with `TF_DSV41_PREFILL=replay` uncommented, the image from C0): `$D ask --url http://127.0.0.1:8000 --prompts $K/prompts.txt --bench --out $OUT/zig-ced.jsonl`, `$D compare $OUT/py.jsonl $OUT/zig-ced.jsonl --tokenizer $TOK` (`py.jsonl` from the prompt-tail step 0) | ~15 min incl. boot | **16/16** (the night: bench prompts 0/8); boot log `prod knobs: ... prefill replay` |
| C4 prefill speed | the same server: `python3 $K/bench_http.py --url http://127.0.0.1:8000 --label zig-ced --long 32768,131072 --out $K/out/zig-ced-long.json`; A / B one cell with the line commented again | ~25 min | 32K **> 1,200 tok/s** (full: 571 at the night, ~690 expected after the Engram fix; prod 2,430); the 32K T0 reply equal to prod's |
| C5 restore | prod back on perf1 if the window ends here | 2 min | as every window |

If C1 fails, `first_differing_role` names the part: `s.L2x/3x.swa.*` = the decoder pass (lo, in-place boundaries,
the ring), `s.L20.comp.* / .ik` = the encoder's compressor, `s.L<20.*` = the encoder pass. Keep `TF_DSV41_PREFILL`
commented in prod-zig.env until C1 and C3 pass.

### Prod parity: prefill (`dsq-zig-pfparity-1`, offline)

Every prefill knob of prod-perf1 (`config/prod.env`, engine 8474f31), checked against Python's source at 8474f31 and
the Zig tree at `1f5e75c7`. CED replay itself is `dsq-zig-ced-1`'s (section above). Prod's speed baseline, from
Python's own G17 study (`dsv41-tensorfold-spark/docs/G17-PREFILL.md` section 4, prod's knobs at 2,048 rows, x3gm v1):
32K full **1,109** tok/s, 32K replay **2,285**; with x3gm v2 (+6.7 %) that is the served **2,430**. So the night's Zig
571 tok/s (full) sat at 0.51x Python's *full* prefill: replay explains 2x of prod's lead, the rest is below.

**Knobs** (status: *built in* = the emitter's constant at prod's value, checked launch by launch by the captures, which
were taken with prod's knobs; *ported* = read under prod's name):

| knob (prod) | Python (8474f31) | Zig status | evidence (Zig, HEAD `1f5e75c7`) | commit |
|---|---|---|---|---|
| `PREFILL_CHUNK=2048`, `PREFILL_ROWS=2048` | `forward.pchunk`, `batch.py` round budget | ported | `prod_knobs.zig:129-133` (a segment = min of the two) | 0e (`dsq-zig-knobs-1`) |
| `PREFILL_ADAPT_GIB=4.5` (adaptive rows) | `memory.py`: a round steps down to 1,024 / 512 rows when the tighter rank is short | equivalent, static | `forward_prefill.plan` / `model.zig:251` (`planPrefill`): the segment's buffers in the boot plan, behind the measured boot check; adaptive only ever slows a round | - |
| `PF_COPIES=1` | `csa2/attn_pf.py` group-major copies | built in | `block_prefill.zig:14`, `:347` (`_segment with PF_COPIES`) | M1 caps |
| `MHC_PF=1`, `MHC_BM=64`, `MHC_WARPS=8` | `mhc_pf.py`, `mhc.py` | built in | `block_prefill.zig:62-63` (`pf_rows` 256, `mhc_bm` 64), `:253` (`tf_dsv41_mhc_pf_v1.run`); warps are the AOT variant's | M1 caps |
| `FULL_CONE=1` | `rounds.py:217`: only when `mode == "full"`; prod runs replay | inert in prod | - (would only matter if Zig served `full`) | - |
| `PREFILL_KERNELS=fast` | `prefill_mm.py` | built in | `block_prefill.zig:1-8` (rot_in, unpack, `_gemm` column blocks under `ws_limit` `:60`) | M1 caps |
| `FAST_EXPERTS=gm` + `GM_V2=1` | `x3gm.py` `_run_ragged` / `_run_v2` | built in (v1); **v2 ported, now ON in prod-zig.env** | `block_prefill.zig:899-939` (x3gm rot / gateup / down), `:907` + `gm2pf.zig:37` (`TF_DSV41_GM_V2`) | `550a371..46b15c6`; zig-serve `8bc8318` |
| `PF_DENSE=fused` + `PF_DENSE_TABLE` | `pfdense.py` | ported, ON in prod-zig.env | `prod_knobs.zig:102`, `block_prefill.zig:118`, `:166-168` (`Seg.fused`) | 0e; zig-serve ed716a5 |
| `ENGRAM_NATIVE=1`, `NATIVE_DIR` | `engram_native.py` (pthread pool: 32 threads prefill, 64 decode) | equivalent | `engram_rows.zig:16` (`TF_DSV41_ENGRAM_QD`, 64 threads without io_uring), dd33a75 / d6c5e02; `NATIVE_DIR` is Python's C build dir | dd33a75, d6c5e02 |
| `PREFETCH_AHEAD=1` | `prefetch.py` (one segment ahead; the next round keeps a bulk read in flight) | ported, ON; **its layer-14 batch was dropped before use: fixed** | `forward_prefill.zig:171`, `:181` (`readAhead`); `engram_rows.zig` `settle` | 19be37a7; **this pass** |
| `INDEX_STREAM_MIN=4096` | `csa2/backend.py` | built in | `block_prefill.zig:65` (`stream_min`), `longpf.zig` | longpf |
| `PLAIN_TRITON=1` | `plain_fp64.py` (weights_proj in one Triton launch) | built in | `block_prefill.zig:721-723` (`_plain`) | M1 caps |
| `PREFILL_ATTN=fused`, `_ATTN_BMQ=32`, `_ATTN_WARPS=8` | `csa2/attn_pf.py` | built in | `block_prefill.zig:66` (`bmq`), `:772` (`_fused`, grid H / bmq); warps are the AOT variant's | M1 caps |
| `PREFILL_GATHER=bf16` | `forward.py` / `mhc.py` (the exchange's bf16 gather) | built in | `block_prefill.zig:225` (`gathered`), the `exchange` glue (bf16 all-gather) | M1 caps |
| `PREFILL_ROUTER=gemv` | `router_gemv.py` route over every row | built in | `block_prefill.zig:68` (`router_chunk` 2,048), `:855-861` (`tf_dsv41_router_gemv_v2.route`) | M1 caps |
| `ROPE_INPLACE=1` | `rope.py` | built in | `block_prefill.zig:353-356` (`_rope` with `OUT` = `X`) | M1 caps |
| `EXPERT_PREFILL=0` | `expert_prefill.py` (x3pf off) | built in (no x3pf launch) | `block_prefill.zig:888-939`: x3gm only | M1 caps |

Nothing on the list is MISSING. Two were OFF in every night run (GM_V2, PREFETCH_AHEAD) and PF_DENSE in all but part
4; all three are on in `prod-zig.env` now. One was ported but not working (read-ahead, below).

**The segment loop, Python against Zig.**

| | Python (`rounds.py`, `slots.py`, `forward.run`, `prefetch.py`) | Zig (`forward_prefill.prompt` / `segment`, `target.prefill`) |
|---|---|---|
| segments | the batcher's rounds: one 2,048-row piece a round, a plan per round to rank 1 | one `prefill` call a prompt: rank 1 gets the ids once (op), every rank loops over 2,048-row segments itself; no plan exchange between segments |
| between segments | `keep`, then the next round's plan; the GPU drains at the plan exchange (G5's nsys: the only idle time is layer 1's Engram wait) | `commit` (carries copied on the stream), `emitPrefill` (1.9-3.8 ms host) while the GPU runs the previous segment's tail; the `embed` glue syncs before rewriting its pinned ids |
| host syncs in a segment | `rows_src.rows(...).to(device)`: layer 1 and, lazily (`decode.DecodeForward`), layer 14 | `engramRows` syncs before layers 1 and 14 (the staging's reuse): the same two points |
| Engram read-ahead | the round issues this segment's reads and the next one's (`ahead`); a bulk read in flight is kept | `readAhead` issues segment k + 1's reads before segment k runs; **before this pass `settle` dropped the k + 1 layer-14 batch at k + 1's layer-1 read**, so every layer-14 read was a synchronous miss with the GPU drained (~70-75 ms a segment with the reader pool, ~250 ms serial at the night). Fixed: an unread bulk batch is only aged by reads of its own table |
| rank-1 exchange | the plan (RDMA mailbox) each round, then the forward's all-gathers | the segment's all-gathers only (> 1 MiB: NCCL, as Python above `ROCE_MAX_KB`) |

**Changes in this pass** (engine `dsv41-zig` `51acf4ea`, `144a97a5`; HEAD + these files in a clean `git archive` tree):
1. `engram_rows.zig` `settle`: a settle on another table never drops an unread bulk batch; it is still dropped at the
   second read of its own table (a wrong guess costs one read, never a bit). `engram_rows_test.zig`: segment k + 1's
   batches on two tables survive segment k's layer-1 then layer-14 reads and segment k + 1's layer-1 read, 0 misses on
   both, value by value, in sync / pool / io_uring modes. The test fails on the old rule (verified) and passes on the
   new one. Same bits: the rows are the file's whatever the path.
2. Prefill attribution, the tool the Spark window needs (TF_DSV41_PROFILE did not cover prefill): `profile.zig` keeps
   prefill segments in their own totals (JSON `"prefill"`: segments, wall, GPU, calls by name), `longpf.window` times
   each call of a segment (and of the CED decoder pass) with an event pair, the glue included, so an Engram wait shows
   as `glue.engram_rows` time; `dsv41_m2b_ref.py --profile-prefill N` times Python's prefill of the gate's prompt
   shape (warm, timed, torch.profiler); `gap.py prefill` puts them side by side by family and by kernel; the M2c-2 kit
   takes `PROF_PREFILL=1` and `LIMIT` (prompts past 4,096). Off unless set: nothing changes in the served engine.
3. zig-serve `prod-zig.env` (`8bc8318`): `TF_DSV41_GM_V2=1`; the "left out" note now names the knobs built into the emitter.

**Expected at 32K** (one stream, `bench_http.py --long 32768`; each row multiplies the one above; the basis column is
the measured share or Python's own number):

| step | tok/s | basis |
|---|---:|---|
| night (full, serial Engram, no read-ahead, PF_DENSE / GM_V2 off) | 571 | run C |
| + reader pool (d6c5e02) and read-ahead that hits on both layers (this pass) | ~690 | 0.64 s of a 3.58 s segment were `engram.wait` |
| + PF_DENSE fused with prod's table | ~720 | Python dense-t +4.1 % at 32K full |
| + GM_V2 | ~770 | Python q28-v2 +6.7 % at 32K |
| + CED replay (`dsq-zig-ced-1`, its gate first) | ~1,600 | Python replay / full 2.06x at 32K (2,285 / 1,109) |
| prod | 2,430 | prod-long-bench.log |

The last ~1.5x (Zig full ~770 against Python full ~1,180 with the same knobs) is not attributed offline: the launches
are Python's (the captures), the exchanges take the same NCCL path, the host emits a segment in 2-4 ms, and the
sessions commit per segment only appends ids. The P2 profile below names it by kernel family; the candidates are the
Zig glue kernels that replace torch ops (`top_positions`, one CTA a row with shared-memory atomics against
torch.topk's radix select; `ring_copy`, `widen_cat`, `cand_keys`) and GPU idle at the two Engram syncs.

#### NEXT SPARK WINDOW: prefill parity (not run; after the CED gate C1, before C4)

Check the lease on dgx-01 first. Kit (M2c-2) and zig-serve image from the engine commit with this section or later.
`K` = the kit dir on both hosts, both hosts within a minute, prod down for P1-P2.

| step | what | time | PASS / record |
|---|---|---|---|
| P1 GM_V2 gate | dgx-01 `cd $K && PREFILL=1 GM_V2=1 PROMPT=2600 AOT_PROMPTS="" ./run_host.sh 0 $K/out/pf-gm2`, dgx-02 the same with rank `1` (two segments: the read-ahead's path) | ~35 min | `prefill.log`: `prod knobs: ... x3gm v2 one`, **`PASS Zig prefill rank r: ... state roles N/N equal`, first token equal, `PASS M2b ... 512/512`** on both ranks. FAIL: comment `TF_DSV41_GM_V2=1` out of prod-zig.env |
| P2 prefill profile at 32K | `PREFILL=1 GM_V2=1 PROF_PREFILL=1 PROMPT=32768 LIMIT=36864 STEPS=16 AOT_PROMPTS="" ./run_host.sh 0 $K/out/pf-prof` (rank 1 likewise); then on dgx-01 `python3 $K/tools/zig/dsv41_perf/gap.py prefill --py $K/out/pf-prof/seed4101/ref/prof-rank0.json --zig-profile $K/out/pf-prof/seed4101/zig-prefill/prof-rank0.json --top 25 > $K/out/pf-prof/gap-prefill.md` (add `PREFILL_MODE=replay` for the replay shape once C1 passes) | ~40 min | the prefill gate PASS as P1; **record** `gap-prefill.md`: wall and GPU a prompt on both sides, the family with the largest `Zig - Python`, `glue.engram_rows`' ms (the GPU waiting on Engram) |
| P3 served 32K | Zig-as-prod (prod-zig.env as committed, `TF_DSV41_PHASES=1 TF_DSV41_PHASES_OUT=/state/phases.json` added for the window): `python3 $K/bench_http.py --url http://127.0.0.1:8000 --label zig-pf --long 32768,131072 --out $K/out/zig-pf-long.json` | ~25 min | 32K **> 700 tok/s full** (> 1,500 with replay on); `phases.json` `engram.wait` **mean < 15 ms** (read-ahead hits; was 249) |
| P4 A / B | the same server, one 32K cell each with `TF_DSV41_PREFETCH_AHEAD=0`, then `TF_DSV41_GM_V2=0`, then `TF_DSV41_PF_DENSE=off` | ~30 min | each knob's share of P3 (expected: ahead 3-8 %, GM_V2 5-8 %, PF_DENSE 3-5 %) |
| P5 restore | prod back on perf1 (`scripts/v2-rollback.sh perf1`) if the window ends here | 2 min | as every window |

### Prod parity: decode runtime (`dsq-zig-rt-1`, offline)

Every decode-round knob of prod-perf1 (`config/prod.env`, engine 8474f31), checked against Python's source at 8474f31
and the Zig tree, plus the round loop itself (Python `rounds.py` / `batch.py` / `decode.py` against Zig `lanes` /
`draft/lanes.zig` / `target.zig` / `forward.zig`). The M1 captures were taken under `tools/zig/dsv41_m1/prod-knobs.env`
(+ `r1c-knobs.env`): every R1 word, `L2PF=1` / `_MB=12` / `_PACE_GBPS=150`, `BRANCHES=1` / `_PRIO=side`; not
`ENGRAM_GATE`, `GRAPH_MODE`, `PLAN_*`, `GLM53_TF_ROCE_FAST`. So *built in* below means the emitter's launch, checked
launch by launch against those captures; it says nothing about streams, overlap or host work, which the captures do not
record.

**The "~20 ms a round outside every phase" (run A) is not in the data.** It came from dividing code T0's round (94.9 ms:
384 tokens at 40.47 tok/s in 100 rounds, `server.log`) by phase averages taken over *all* cells (`window` 43.7, `draft.pass`
31.5 ms), and prose rounds have about half code's rows (prose 1.2 drafted a round, code 3.5). Over the whole run
(`r0/a/phases.json`): 648.1 s wall, 585.6 s in top-level phases, so at most **8.4 ms a round** is untimed, and that
includes the server's idle time from `t0` (model open) to the first request and between bench cells. `a6080cf`
adds a `host` phase (the gaps between top-level phases inside a round; gaps of 50 ms or more are idle) so the next
window measures it per cell instead of inferring it. Where else the round's time goes, found in the code:
- **Engram reads serialised per layer and per slot.** The graphed window staged layer 1's rows (read and wait), then
  layer 14's (read and wait); row mode did that per slot as well: 2 layers x 4 slots = 8 read latencies a window.
  Python's gate worker reads every row of the round, all slots and all layers, in one batch. Fixed (`Forward.issueEngram`,
  below): every layer's and slot's reads in flight before the first wait, so one read latency a window.
- **Greedy picks on the host.** `Forward.greedy` downloads the window's whole logits ([n, 64,640] fp32, ~1.3 MB at 5 rows),
  argmaxes on the CPU, uploads the pairs, all-gathers, downloads, and syncs twice. Python's `cand_gather` takes the top
  candidates on the GPU and copies [R, C] back once. Estimated 0.2-0.5 ms a round; not changed here (it needs a GPU
  argmax kernel and a GPU to prove it).
- **The speculative pass starts after the host's pick** (DSpark area, `dsq-zig-dspark-1`). Python's `decode.window`
  queues the candidates' copy, launches the next pass on the GPU (`spec.launch`), and only then waits. Zig's
  `Lanes.verify` runs `target.window` (greedy's two syncs included) and then `speculate`, so the GPU idles for the
  host's pick and the lanes' bookkeeping. Run A's speculative pass ran in only 800 of 7,493 passes; prod hits 99 %.
- **`candidates` 40.9 ms a call** in run A is the pass's logits download (before `9bc88af`), not the window.

**Knobs** (status: *built in* = the emitter's constant at prod's value, checked by the captures; *ported* = a Zig knob,
on in `prod-zig.env` unless said; *missing* = not in the Zig engine):

| knob (prod) | Python at 8474f31 | Zig | evidence | commit |
|---|---|---|---|---|
| `L2PF=1`, `_MB=12`, `_PACE_GBPS=150` | `l2pf.py`: the next dense group's first 12 MiB prefetched into L2 on a side stream at sites o / x / f (x, f paced at 150 GB/s on 2 CTAs; windows <= 16 rows), joined at the window's end. `l2pace.cu`: off costs 1.7 ms a 1-row window | **ported, on.** Before: the launches were emitted with empty tables and **skipped at issue** (`run.zig` `issue`: `tf_dsv41_l2p*` returned) | `l2pf.zig` (`install`: tables from `Emitter.l2pfSources`, l2pf.site_tensors' order and `take`'s budget, every span inside a live allocation; knobs), `run.zig` (`Side`: fork at the site, join at the window's end, `rows` > 16 skip), `block.zig` (the sizes now come from the sources: `check` 0 mismatches) | engine `1f5e75c`, zig-serve `1ba798c` |
| `GLM53_TF_ROCE_MAX_KB=1024` | the RoCE mailbox's message cap (4-slot x 16-row windows fit) | **ported** (`tp/config.zig` alias), but **it never reached the Zig ranks**: serve.sh passed only `TF_DSV41_*` into the Zig containers, so the cap stayed 256 KiB and 4-slot windows' exchanges went to NCCL (21 vs 8.9 us an exchange) | `scripts/serve.sh` `run_args_zig` now passes `GLM53_TF_*` (checked: the generated `docker run` holds `-e GLM53_TF_ROCE_MAX_KB=1024`) | zig-serve `a45d79f` |
| `PLAN_LINK=rdma`, `PLAN_PIN=auto` | `planlink.py`: rank 1 spins 5 ms on a RoCE mailbox flag, then naps; plan threads pinned (G14: rank 1's window-entry lag +130 -> -9 us) | **ported (equivalent), on**: TCP link, rank 1 spins on the socket 5 ms (`TF_DSV41_PLAN_SPIN_US`), then blocks; pin as Python's `pin_cpu`. Zig sends 3-5 plan messages a round (window, greedy, keep, DSpark ops), Python one | `tp/planlink.zig` `spin` / `pinOnce`, `tp/config.zig` | zig-serve `a45d79f` |
| `DEPTH_JOINT=0` | `joint.py`: the branch default is 1; prod 0 (G9: joint failed its gate) | **ported, on**: Zig read it but defaulted to 1 and prod-zig.env did not set it | `draft/calib_env.zig` `depthSettings`, `draft/depth.zig` `joint: u2 = 1` | zig-serve `1ba798c` |
| `ENGRAM_GATE=1` | `engram_gate.py`: the graph launches at once and a kernel before layers 1 / 14 spins on a host-mapped flag, then copies the rows a worker read (all slots and layers in one batch); C2 / C4 +3.5 / +2.6 % | **missing** (the kernel and launcher exist: `kernels_ops.engramGate`; no emitter call, no host-mapped staging, no worker). **Partly covered:** `Forward.issueEngram` puts every layer's and slot's reads in flight before the first wait (the batch half of the gate; the host still waits before the launch) | `forward.zig` `issueEngram`, `graphedWindow`, `window`; `batch.zig` `stageEngram` | engine `8774969` |
| `BRANCHES=1`, `BRANCHES_PRIO=side` | `branches.py`: the indexer / compressor chains on a high-priority side stream under wq_b's weight stream, joined before the attention core (1-row window -0.53 ms over 3 boots); e4's folded copies | **e4 built in** (the launch set: `_plain` bf16, `index` before wq_b, split q group; `block.zig`), **the side stream missing** (one stream). A port needs per-call stream tags + fork / join at `plan()`'s points, a second ext scratch (the runner's 64 MiB `scratch` is shared by every ext launch), and a GPU bits gate | `block.zig` attention, `run.zig` | - |
| `MHC_DEFER=1` (R1) | `mhc_cuda.py`: the coefficient kernel on a side stream, awaited by the next reader | bits built in (`block_moe.zig` `mhc_cuda_v1.coef`), **serial** (no side stream); ~0.2 ms a window | `block_moe.zig` | - |
| `RMS_FOLD=1` (R1) | `rmsnorm.py`: q norm folded into rot_in; kv_norm on the side stream | q fold built in; kv_norm **on the main stream** (~0.1 ms a window) | `block.zig` `fusedNorm`, `rms` | - |
| `GLM53_TF_ROCE_FAST=1` | `roce.py` `gather_fast_kernel`: seven critical-path changes (1-row -0.16, 16-row -0.51 ms) | **1 of 7** (`tp/mailbox.cu`: local copy while the peer's travels); `dsq-zig-tp-1`'s kernel work | `fabric/tp/mailbox.cu` | - |
| `COMM_BACKEND=roce` | `engine.py` | ported (`TF_COMM_BACKEND`, serve.sh) | `tp/config.zig` | - |
| `GRAPH_MODE=rows` | `rowgraphs.py`: slot-agnostic row graphs | equivalent: `TF_DSV41_GRAPHS=1` + row mode graphs (`graphs.zig`, `batch.zig`) | - | - |
| `EXPERT_LOADS=1` | `expert_loads.py`: x3ld `ld_kernel` (nt 8, pd 1) | built in | `block_moe.zig` `x3ld_v1.grouped` | - |
| `EXPERT_TOPP=0.85`, `MIN_K=3`, `RENORM=kept` | `prune.py` | TOPP ported (same name, prod-zig.env; **off when unset** on the served model); MIN_K / RENORM built in at prod's values (their env vars are not read) | `model.zig`, `block_moe.zig` | - |
| `MHC_CUDA`, `MHC_DEC`, `MHC_FIN2`, `MHC_FN=bf16`, `MHC_TAIL` | `mhc_cuda.py`, `mhc_dec.py`, `mhc.py` | built in (`MHC_TAIL` under `r1`) | `block_moe.zig` | - |
| `RMS_PAIR`, `RG_NARROW=2`, `RG_TAIL`, `RG_PRUNE`, `PRUNE_KIT` | `rmsnorm.py`, `router_gemv.py`, `prune.py` | built in (`RG_PRUNE` / `PRUNE_KIT` under `r1`) | `block.zig` `_rms2`, `block_moe.zig` `route` | - |
| `PARTIAL_BF16`, `MOE_BF16`, `DENSE_V3`, `ATTN_CUDA`, `ATTN_ROPE`, `ATTN_SPLIT=4` | `forward.py`, `moe.py`, `dense3.py`, `attn_cuda.py` | built in (`MOE_BF16`, `ATTN_*` under `r1`) | `block.zig`, `block_moe.zig` | - |
| `ROUTER=gemv`, `INDEX_DTOPK`, `INDEX_KV=fp8`, `PDL=0`, `ENGRAM_FUSE_DEC` | `router.py`, `dtopk.py`, `backend.py`, `pdl.py`, `engram_dec.py` | built in | `block_moe.zig`, `block.zig` | - |
| `CALIB=real` | `calib.py`: the engine's own window / pass costs measured at boot, stored | read only: Zig prices drafts from Python's last stored table, i.e. Python's costs, not Zig's (Zig's 1-row round 24.7 ms against the table's 21.5; pass 7.0 against 3.4-3.7), so the depth policy optimises for the wrong engine. `TF_DSV41_DRAFT_TIMED=1` prices from Zig's own measured rounds (off in prod-zig.env: HTTP == CLI replay); no boot measurer | `draft/calib_env.zig`, `serve_engine.zig` `cfg.timed` | - |

Open item for verification (not settled offline): Python's `MHC_CUDA` / `MHC_DEC` have `MAX_ROWS = 16` and fall back to
Triton above; Zig's row graphs reach 64 rows and the emitter has no row-count branch there. The captures' windows are
all <= 16 rows, so `check` does not cover a > 16-row decode window.

**Gates on the engine commits** (each on a clean `git archive` of HEAD plus the commit's files): `tf-dsv41-m1 check`
cap15 / cap18 / cap22 / cap24-1 / cap24-12 / cap24-16 and `dcheck cap16` **0 mismatches**; `zig build test-dsv41`
(incl. `l2pf.take` against l2pf.py's span rules), `test-dsv41-knobs`, `test-dsv41-rows`, `test-dsv41-draft`, `test-tp`
pass. No GPU: the prefetch launches themselves were replayed bit-exact in the M1 pods (they write nothing), but the
side stream, the tables and the batched Engram issue are proven only on the Sparks (below).

#### NEXT SPARK WINDOW: decode runtime (not run; before S4)

Check the lease on dgx-01 first. Image from engine **8774969 or later** and zig-serve
**1ba798c or later**. `K` = the kit dir. Zig-as-prod: `scripts/v2-rollback.sh zig` (prod-zig.env as committed) with
`TF_DSV41_PHASES=1 TF_DSV41_PHASES_OUT=/state/phases.json` added for the window. Copy `phases.json` after every cell
(it is cumulative): `ssh dgx01 cp ~/ai/dsv41-tf/phases.json $K/out/rt-<cell>-phases.json`.

| step | what | time | PASS / record |
|---|---|---|---|
| R0 boot | `scripts/serve.sh start` (config/prod-zig.env); `docker logs dsv41-tf-r0 2>&1 \| grep -E 'l2pf\|plan link\|tp:\|engram:'` and the same on dgx-02's `dsv41-tf-r1` | ~10 min (fast boot) | **`l2pf: bulk, 12 MiB a site, N tables (o / f / x ... MiB)`** on both ranks (N: about three a backbone layer, o / f / x), **no `l2pf: OFF`**; `plan link rank 1 pinned to CPU n`; `docker inspect dsv41-tf-r1 \| grep ROCE_MAX_KB` shows 1024 |
| R1 bits | `python3 $K/bench_http.py --url http://127.0.0.1:8000 --label zig-rt --work code,prose --temps 0 --streams 1 --reps 1 --out $K/out/rt-bits-on.json`; then `TF_DSV41_L2PF=0` in prod-zig.env, `serve.sh stop && serve.sh start`, the same command `--out $K/out/rt-bits-off.json`; restore `TF_DSV41_L2PF=1` | ~25 min | **every reply's `sha` equal on == off** (the prefetch writes nothing); FAIL: `TF_DSV41_L2PF=0` in prod-zig.env, record the first differing cell |
| R2 1-stream | L2PF on: `python3 $K/bench_http.py --url http://127.0.0.1:8000 --label zig-rt --work code,prose --temps 0 --streams 1 --reps 2 --out $K/out/rt-s1.json`, save `phases.json`; then `python3 $K/bench_http.py --compare $K/out/prod.json $K/out/rt-s1.json` (prod.json: the night's Python prod run, quant repo `results/zig-night-20261007/r0/prod.json`, copied to `$K/out/`) | ~20 min | code T0 1-stream **> 65 tok/s** (night 60.4; prod 84.7); `host` ms a round **< 2** (if larger: the host time is the lanes / token events, record it for `dsq-zig-dspark-1`); `engram.wait` a window **< 1 ms** (one batched read) |
| R3 4-stream | the same with `--streams 4` (`--out $K/out/rt-s4.json`), save `phases.json` | ~15 min | code T0 4-stream **> 95** (night 82.2; prod 143.6); `engram.wait` a window **< 1.5 ms** (was 8 serial reads) |
| R4 A / B | one `--work code --temps 0 --streams 1 --reps 2` cell each with, in turn: `TF_DSV41_L2PF=0`; `TF_DSV41_PLAN_LINK=tcp`; `TF_DSV41_ENGRAM_PREFETCH=0` (the old per-layer reads); `TF_DSV41_DRAFT_TIMED=1` (drafts priced from Zig's own rounds, not Python's table); restart between | ~45 min | each knob's ms a round (expected: L2PF 1-1.7 ms at 1-2 rows, less at 5 rows; plan link 0.1-0.4 ms; Engram batch 0.3-1 ms at 1 stream, more at 4; DRAFT_TIMED: tokens a round and tok/s, either way); a knob that measures slower: off in prod-zig.env, recorded here |
| R5 restore | `scripts/v2-rollback.sh perf1` if the window ends here | 2 min | as every window |

### Speed against prod after the night (`dsq-zig-speed-2`, offline)

Every number below is from the night's logs (quant repo `results/zig-night-20261007/r0/`: `*/server.log`,
`*/phases.json`, `*/bench.log`, `r1prof/`), prod's own cost table (`calib-*.json`, the one the served Zig read) or a
dev-box measurement where named. No Spark or pod was used.

**What each run actually had** (each server's `prod knobs:` / `engram:` / `graphs:` boot lines):

| run | graphs | Engram rows | io_uring | PF_DENSE | GM_V2 | PREFETCH_AHEAD | prefill rows | drafts | speculative pass |
|---|---|---|---|---|---|---|---|---|---|
| A (1 slot) | on | `/engram` (R3) | **off** (PermissionDenied) | off | off | off | 2,048 | on | 800 of 7,493 passes |
| B (4 slots) | row graphs | `/engram` | **off** | off | off | off | 2,048 | off | - |
| B2 (4 slots) | row graphs | `/engram` | **off** | off | off | off | 2,048 | slot drafts, passes of <= 3 slots | 189 of 3,530 passes |
| C (1M) | **off** (eager) | `/engram` | **off** | off | off | off | 2,048 | on | 6 |
| part 4 (Zig prod) | row graphs | `/engram` | **off** | fused | off | off | 2,048 | slot drafts | - |

So every run had the real Engram shards (bc55ef4's `TF_DSV41_ENGRAM_DIR`), but none had GM_V2 or read-ahead, only
part 4 had PF_DENSE, run C was eager, and **every run read Engram rows one `pread` at a time on the engine thread**:
Docker's default seccomp profile refuses `io_uring_setup`, and R3's fallback was serial. Python's reader says the same
in its docstring (`engram_native.py`: "Docker's default seccomp profile refuses io_uring") and reads with a pthread
pool (32 threads for prefill, a persistent pool of 64 for decode).

**Gap -> cause -> fix:**

| gap (Zig vs prod) | cause, with the evidence | kind | fix |
|---|---|---|---|
| all: Engram reads | io_uring refused in every run; serial 264-byte reads: run C 1,022 `engram.wait`, 249 ms mean (~0.64 s of each ~3.6 s 2,048-row segment, 12 lookups a row a layer a rank); run A 3.8 ms `engram.wait` + 2.6 ms `prefetch` a window; row mode (B / B2) has no read-ahead, so every row is read at its window | code | **dd33a75 + d6c5e02**: a persistent pool of reader threads (`TF_DSV41_ENGRAM_QD`, Python's name and default 64); a window's misses go ahead of queued read-ahead. Dev box O_DIRECT, a segment's 24,576 rows: 174-271 ms serial -> 71-74 ms. Same bits (engram_rows_test: serial / pool / io_uring) |
| prefill: read-ahead | `prefetch ahead off` in every run; with serial reads it could not overlap anyway | config | prod-zig.env `TF_DSV41_PREFETCH_AHEAD=1` (zig-serve **a62b7a1**) |
| decode 1 stream (code 60.4 vs 84.7, prose 34.3 vs 51.6) | not the forward: Zig's eager R1 windows 30.4 / 31.3 / 32.3 ms (1 / 2 / 4 rows) vs Python's eager reference 33.3 / 37.5 / 38.3 on the same windows (`r1prof/.../prof-rank0.json`); run B's 1-stream round 24.7 ms vs prod's served 1-row verify 21.5-22.1 ms. With drafts the round is ~65 ms (B2 code: 384 tokens, 98 rounds) against prod's ~43 (its 2026-10-06 table: verify 4 / 5 rows 36.4 / 41.1 ms, pass 3.66 ms) | code | Engram above (~4-6 ms a round); the rest needs hardware, below |
| decode: the DSpark pass | 7.0 ms (B2) vs prod's 3.66: eager launches, a synchronous download of the block's whole logits ([5 x 64,640] fp32, pageable), the candidates on the host, two stream syncs | code | Spark: graph the pass and keep the candidates on the GPU (not done offline: needs the GPU to prove) |
| decode: unattributed host time | run A code T0: 94.9 ms a round = window 43.7 + pass 31.5 (before 9bc88af) + ~20 ms outside every phase (phases time the target / pass calls only, summed over all cells) | unknown | Spark step S4: phases per cell |
| decode 4 streams (82-94 vs 143.6) | 4 live slots need two DSpark passes a round (passes <= 3 slots: `mhc_cuda` site1 / final take <= 16 rows, the AOT set has no 20-row pass; prod: one 20-row pass); the speculative pass ran 189 of 3,530 passes (prod: 99 % hits); serial Engram reads of 16-row windows (~380 a window) | code | Engram above; 20-row slot pass and the speculative pass over slot passes need kernels + AOT on hardware |
| long prefill (32K 571, 131K 585 vs 2,430 / 2,263 tok/s) | (1) prod runs **CED replay** (`TF_DSV41_PREFILL=replay`): layers 20-39 only over a prompt's last 127 rows, ~2x by Python's own numbers (DEPLOY.md 950 vs 480; ENGINE-PLAN 2,200 vs 1,500); Zig runs `full` (not ported, section 0e). So Zig's full prefill is ~2.1x behind Python's full, not 4.2x. (2) Engram serial reads ~18 % of a segment. (3) GM_V2, PF_DENSE off. Not host emission: `emitPrefill` of a 40-layer 2,048-row segment is 1.9-3.8 ms (dev box, ReleaseFast) | code + config | Engram fix + read-ahead; GM_V2 after its gate (S5); CED replay is the largest item and a port of its own (below) |
| long-context decode (35.6-38.9 vs 50.5-54.0) | run C was eager (graphs on the paged pool need `_scores paged NK 2048`, never recorded): `forward` 31.3 ms of host launches a window (graphed run A: 3.55); `prefetch` 7.08 ms a window (serial Engram) | config + AOT | Engram fix; graphs once the AOT fill covers the context buckets (`dsq-zig-rows1m-1`'s aot-needs / fill), S3 |
| fast boot 375 s (prod 33-48 s), first boot 810 s | 261 s of the 367 s weights load is "upload + digest": a SHA-256 of every tensor on the way up (95 GiB a rank, one thread). The image is built for generic `aarch64-linux-gnu`, where std's SHA-256 has no `sha2` instructions (checked: `builtin.cpu.has(.aarch64, .sha2)` is false there, true with `-mcpu=baseline+sha2+aes`); only M1's weights check reads the digests. Prepared reads ~106 s (~0.9 GB/s, one thread) | code + build | **a5ce61a**: the served model takes no digests; the next prepared file is read ahead (`POSIX_FADV_WILLNEED`) while a layer uploads. zig-serve **a62b7a1**: `ZIG_CPU=baseline+sha2+aes` (the first boot's file digests and `PREPARED_VERIFY` in hardware). The kit's `build-kit.sh` (owner `dsq-zig-rows1m-1`) can take the same `-Dcpu` |

Checks on d6c5e02 (these commits on a390113, other agents' uncommitted work left out): `check` cap15 / cap18 / cap22 / cap24-1 /
cap24-12 / cap24-16 and `dcheck cap16` 0 mismatches, `test-dsv41-knobs` PASS, engram_rows_test 8/8 (Debug, and
ReleaseSafe five times). No kernel, emitter or value changed.

**Expected after these fixes** (estimates; the basis is the measured share each fix removes):

| workload | night | expected | basis |
|---|---:|---:|---|
| code T0, 1 stream | 60.4 | ~63-65 | Engram ~4-6 ms of a ~65 ms round |
| prose T0, 1 stream | 34.3 | ~36-37 | the same on a ~48 ms round |
| code T0, 4 streams | 82.2 | ~90-95 | ~380 serial Engram reads a 16-row window (run A: 3.8 ms a window of ~4.6 rows) |
| 32K prefill | 571 tok/s | ~690; ~730 with GM_V2 | 0.64 s of 3.58 s a segment; GM_V2 +7 % on q28-v2 in Python |
| 32K decode | 36.4 | ~40 eager; up to ~50 graphed | `prefetch` 7.08 ms of a ~72 ms round; graphs: the launch share, unknown until S3 |
| fast boot | 375 s | ~120-160 s | the digest out of 261 s, read-ahead of the 106 s read |
| first boot | 810 s | ~400-500 s | the upload digest out, the file digests in hardware |

**Beyond these (code, needs hardware to prove; ranked by expected gain):**
1. **CED replay prefill** (`TF_DSV41_PREFILL=replay`): ~2x long prefill and the bits prod serves for prompts over 128
   tokens (section 0e). Python `replay.py` at 8474f31: encoder pass (layers 0-19 + layer 20's site and compressor:
   the decoder's global KV for every position), a stash of the rows entering layer 20 for the last 127 positions,
   then layers 20-39 over [R0, n - 1) with every SWA window starting at R0; the last token is the reply's first
   verify row (Zig's `PROMPT_TAIL=verify` already does that part). Gate: a `dsv41_m2b_ref.py` reference with
   `TF_DSV41_PREFILL=replay` (M2c-2 kit, `PREFILL=1`). A workstream of its own.
2. **The DSpark pass**: graph its launches and keep its candidates on the GPU (pinned staging at least): 7.0 -> ~3.7
   ms a round, ~+5 % at 1 stream.
3. **One pass over 4 slots** (20 rows: `mhc_cuda` site1 / final past 16 rows and the AOT variants) and the
   speculative pass over slot passes: the 4-stream drafted rounds.
4. **Row-mode Engram read-ahead** (batch.zig, owner `dsq-zig-rows1m-1`): the one-slot path reads the next window's
   row 0 during the pass; row mode reads everything at the window (now in parallel, still on the round's path).
5. Engram GPU gate (Python `ENGRAM_GATE`, +2.6-3.5 % at C2 / C4), `BRANCHES` side stream (-0.53 ms a 1-row
   window), `ROCE_FAST` (-0.16 to -0.51 ms), `PLAN_LINK=rdma`.

#### NEXT SPARK WINDOW: speed (not run; after the prompt-tail steps, ordered by expected gain)

Kit and image from engine **d6c5e02 or later** (plus `dsq-zig-rows1m-1`'s AOT fill when it lands); the image from
zig-serve **a62b7a1** (`ZIG_CPU`). Check the lease on dgx-01 first. Both nodes' `dsv41-zig-prepared` were deleted
after the night, so the first Zig boot rebuilds them. Each step's PASS criterion is in bold; a step that fails is
recorded and the window moves on.

| step | what | time | expect |
|---|---|---|---|
| S1 boot | Zig-as-prod switch (`v2-rollback.sh zig`, prod-zig.env), then `serve.sh stop` / `start` again. If 4 slots x 1M still refuses at boot (9bb3227 is not proven on the Sparks), run S1-S2 with `PARALLEL=1 TF_DSV41_SLOTS=1` as run C did | ~25 min | rank 0's `dsv41 load:` line: first boot **< 600 s** ready (was 810), then fast boot **< 200 s** (was 375); `upload + digest` far below 261 s |
| S2 Engram | the same server with `TF_DSV41_PHASES=1 TF_DSV41_PHASES_OUT=/state/phases.json` added to prod-zig.env for the window (removed after): boot log **`engram: no io_uring (...); rows read by 64 threads`**; `bench_http.py --long 32768`: **`engram.wait` mean < 30 ms** (was 249), TTFT **< 58 s** (was 66.5). A / B one cell with `TF_DSV41_ENGRAM_QD=0` (the serial path) | ~20 min | 32K ~690 tok/s |
| S3 long decode | after the AOT fill: run C's shape (`CONTEXT=1048576`, one slot) with `GRAPHS=1`, `--long 32768,131072` | ~25 min | **no MissingTritonVariant**; decode **> 45 tok/s** at 32K (was 36.4); else record the variant, `GRAPHS=0` |
| S4 short decode, per cell | B2's shape (4 slots, slot drafts, prod-zig.env, S2's phases knobs), one `bench_http.py --work code --temps 0 --streams 1 --reps 2` run, save `phases.json`; then `--streams 4`; then `--work prose --streams 1` (copy `phases.json` between runs: it is cumulative) | ~30 min | code 1-stream **> 62** (was 60.4), 4-stream **> 88** (82.2); per-cell phases name the ~20 ms a round outside every phase |
| S5 GM_V2 | M2c-2 kit `PREFILL=1 GM_V2=1 ./run_host.sh` (both hosts; `prefill.log`'s `prod knobs:` line must say `x3gm v2 one`), then S2's 32K cell with `TF_DSV41_GM_V2=1` | ~35 min | **prefill gate PASS** (95/95 roles, first token, 512/512); 32K **+5-10 %**. PASS: `TF_DSV41_GM_V2=1` into prod-zig.env |
| S6 restore | prod back on perf1 (`scripts/v2-rollback.sh perf1`) | ~2 min | as every window |

### 4 slots x 1M and the AOT fill (`dsq-zig-rows1m-1`, offline)

**4 slots x 1M (`9bb3227`).** The night's rank 1 `UnknownRowsCall` came from the row programs (`Batch.plan`, every bucket
at the 1M limit's end): `115b46e` / `acae981` added `_dtopk` for one slot and the row rewriter did not know it. Python's
row path (`backend.select` / `select_cand` with a `RowWin`, 8474f31) has two long-context selections, and Zig had one:
- past attn_cuda's plan (more than 126,976 keys): `dtopk.select` / `dtopk.blocks` with `rows=True` (`_dtopk` reads
  `POS[r]`). Row mode now rewrites `_dtopk` and `_block_keys` onto the table's `pos` with `ROWS` on.
- past the index budget (prod's `TF_DSV41_INDEX_BUDGET_MIB=64`: 6+ rows at 1M keys, 11+ at ratio 2, and one slot's
  16-row verify windows past about 350K positions): `backend._blocked` (`_row_blocks`, each block `_scores` + `_keys` +
  `index.top_positions`, the candidate source's `_block_keys` + `top_positions`, then `visible_counts`). The decode
  emitter had no budget rule, so these windows took the whole-window dtopk / CUDA path and were not Python's bits, in
  row mode and for one slot. `block.zig` now applies `backend.select`'s rule. A `RowWin.sub` block keeps the bucket's
  cap (`block.Options.rows`), and row mode never streams (`not win.rows`). A row block reads rows [a, a + n) of the
  table (`block_pos` carries a and n). The glue `topk` / `counts` / `block_pos` moved to the shared decode glue.
- Host checks (`test-dsv41-rows`, 17/17): the real emitter at `CONTEXT=1048576`, 4 slots, prod's knobs, every bucket
  up to the cap at six context buckets and `Batch.plan`'s windows. Every selection launch is counted against an
  independent table of `backend.py` / `attn_cuda.py`'s rules, and every row-mode argument is checked on its rows of the
  table. Buckets 24-64 run as 16-row blocks. One slot runs at 1M (1-16 rows at the limit and at 400K). The row table
  of a 4-slot mix sits at the limit. The plan at 16 rows adds `L.ix.keys` 128 MiB, `L.scores` 64 MiB and
  `L.ix.bkeys` 16 MiB a rank, the bytes Python allocates per window. `check`: cap8 / 15 / 18 / 22 / 24-1 / 24-12 /
  24-16 / 30-gm2v2 0 mismatches; cap6 22, the same at 9c7e415 (its 256-row prefill head).

**The AOT fill (`tools/zig/triton_fill.py`, `tf-dsv41-m1 aot-needs`).** The served set held only what reference runs
happened to launch; the night patched five phases in. The fill compiles every variant the served config can launch,
from Python's own Triton source, ahead of time:
1. `triton_fill.py sigs --py <8474f31 src>`: each kernel's parameters, constexprs, `do_not_specialize` and
   `do_not_specialize_on_alignment`, from the JITFunctions of the family's and `tensorfold/cuda`'s modules.
2. `tf-dsv41-m1 aot-needs SIGS OUT`, with the served env (`tools/zig/dsv41_m4/fill-4x1m.env`): the engine's own
   programs, each Triton launch reduced to Triton's specialization (constexpr values; per runtime argument the dtype
   and 16-byte alignment, the int width, == 1 folded, % 16, fp32). The programs:
   - one-slot decode windows of 1-16 rows, eager at every end position and graphed at every context bucket;
   - row mode, every bucket up to the cap at every context bucket;
   - prefill segments of 1-2,048 rows at each regime's first start;
   - DSpark's pass, the slots' passes and ingests.
   The position scan relies on two facts: a window's launches change only where a threshold crosses, and between
   thresholds every int's class repeats every 256 positions. So it emits every position of the first 512, then every
   256th, then the 512 around any sample whose variant set changed, then the last 256.
3. `triton_fill.py compile --needs ... --options tools/zig/dsv41_triton_options.json --arch 121`: each variant through
   the JIT's own steps without a GPU. Those steps are the function's binder (`create_function_from_signature` for the
   target) over stand-ins of the specialization, `_pack_args`, `ASTSource` and `triton.compile`. So the signature,
   constexprs, attrs and options are the ones Python's launch makes. The output is aot.json in `pack_aot`'s form
   (`"filled": true`). Variants an existing set holds (`--have`, by specialization) are not compiled.
   - Options are a call site's, so `options` derives one rule per function from the Sparks' 18 night manifests. All
     agree. `mhc_dec`'s maxnreg rule is mirrored, and `_dtopk` / `_prune`, which no capture launched, come from their
     8474f31 call sites. `compile` refuses a function without a rule.
4. Byte identity needs the image's toolchain and files: Triton 3.7.1, ptxas V13.3.73 (PyPI `nvidia-cuda-nvcc==13.3.73`,
   the image's `/usr/local/cuda/bin/ptxas`), and the line info's paths (`/dsv41-tf/src`, the image's Triton dir;
   `get_jit_fn_file_line` is mapped). ptxas also writes each source file's mtime and size into the DWARF line table,
   from the file at that path. `--stamps` (`stamps --cubins <a set of the image>`) compiles in a bubblewrap root where
   those paths hold the image's stamps. Without stamps, the SASS is the same and only those DWARF bytes differ.
5. `build-kit.sh` with `TRITON_FILL=1` runs 1-3 into `$KIT/triton-fill/aot` and ships `triton_fill.py` + the options.
   `triton_fill.py merge` adds the fill to a served set by specialization (the fill's hashes are another host's
   Triton key). `sass` compares the fill against a served set's captured cubins.

**Checks (dev box, no GPU):**

| check | result |
|---|---|
| pod 24 sm_120 captures (cap24-16 / 24-1 / 24-12, tree 67f3850^): specialization from the binder == captured | 103/103 |
| the same, SASS (`nvdisasm -c`) == captured cubin | 103/103 |
| the same with stamps read from the capture's own cubins: cubin bytes == captured (sha256 == the manifest's) | 29/29 (cap24-16) |
| the Sparks' sm_121 night manifests (long / rows / taps-aot, capture-p, seed4101 ref; 8474f31): specialization == captured | 313/313 (cubins stay on the Sparks: SASS is window step 2) |
| options rules vs every Spark variant | 17 functions, 0 disagree (+ 2 from call sites) |
| `aot-needs` at 4 x 1M (fill-4x1m.env) | 149,317 programs, 134 variants (decode 39, + row mode 18, + prefill 73, + DSpark 4; 14 prefill regimes); 10.5 min (5.9 min CPU, ReleaseFast) |
| `compile` of those needs, `--have` the night's 7 local sets | **119 already held** (the Zig reduction and Python's specialization name the same variant), **15 compiled**, 0 failed: `_dtopk` / `_block_keys` with `ROWS` (row mode, never captured), `_dtopk` mode 2 in 4 classes (one slot past 1,015,808 positions, `acae981`), `_stream` with `nsplit % 16 == 0` (the night's gap 5 class), `_site` with `R % 16 == 0` x 4, one `_scores`; 1.5 MB |
| `merge` twice into a copy of a set, then `sass` against it | 15 added then 0; 15/15 SASS and bytes |

**Limits.** Prefill coverage is every row count at each regime's first start: starts that are not multiples of 2,048
(a resumed session's prompt) are sampled, not proven. The images path (`IMAGES` off) and `TF_DSV41_PF_DENSE=fused`'s
replacement of `_gemm` are not modelled; `_gemm`'s variants are listed anyway (a superset).

#### NEXT SPARK WINDOW: 4 slots x 1M and the AOT fill (not run)

The kit is built from this section's commit or later, with the fill. Check the lease on dgx-01 before any step.
`K=~/zig-window-kit-rows1m`, `ASSETS=~/ai/dsv41-zig-assets` on both hosts, `IMAGE=tensorfold-dsv41:e1e2-8474f31`.

0. **Dev box, before the window:** the kit with the fill (`TRITON_FILL_HAVE` empty, so step 2 can compare every
   variant the served set also holds):
   ```
   git -C ~/Documents/Projects/spark-work/tensorfold-decode1 archive 8474f31 src | tar -x -C /tmp/py8474
   TRITON_FILL=1 TRITON_FILL_PY=/tmp/py8474/src TRITON_FILL_PYTHON=~/.cache/dsq-triton-fill/venv/bin/python \
   TRITON_FILL_PTXAS=~/.cache/dsq-triton-fill/cuda133/nvidia/cu13/bin/ptxas \
   TRITON_FILL_ENV=tools/zig/dsv41_m4/fill-4x1m.env FATBINS=<fatbin-kit> TP_FATBIN=<tp_mailbox.fatbin> \
   KIT_COMMIT=<commit> bash tools/zig/dsv41_m2c2/build-kit.sh $K
   ```
   Expect `compile.json` `failed: []`. Copy `$K` to both Sparks and check `SHA256SUMS`.
1. **No downtime (prod up):** Python prod's long T0 replies, for step 5:
   `python3 $K/bench_http.py --url http://127.0.0.1:8000 --label prod --long 400000,900000 --out $K/out/prod-long2.json`
   (one stream each; 900K takes about 10 min of prefill).
2. **The fill's SASS against the served set's captured sm_121 cubins** (both hosts, no GPU work):
   ```
   docker run --rm -v $K:/kit -v $ASSETS:/assets -e TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm $IMAGE \
     python3 /kit/tools/zig/triton_fill.py sass --fill /kit/triton-fill/aot --served /assets/aot
   ```
   PASS = `differ: []`, with `both` > 0 (every variant both sets hold has the same SASS). Then the image's stamps,
   for byte-identical fills from the next kit on:
   `docker run --rm -v $K:/kit $IMAGE python3 /kit/tools/zig/triton_fill.py stamps --root /dsv41-tf/src
   /usr/local/lib/python3.12/dist-packages/triton --out /kit/triton-fill/stamps.json`. Copy it back to the dev box
   (`TRITON_FILL_STAMPS`). With stamps, a later `sass` must show `bytes_equal == both`.
3. **Merge** (both hosts): `cp -r $ASSETS/aot $ASSETS/aot.pre-fill && python3 $K/tools/zig/triton_fill.py merge
   $ASSETS/aot $K/triton-fill/aot`. Keep `aot.pre-fill` for the rollback.
4. **Boot 4 x 1M** (prod down; part 4's Zig-as-prod shape with `CONTEXT=1048576`, `SLOTS=4`, `SLOT_DRAFTS=1`,
   `POOL_TOKENS=1400000`, prod's knobs and the merged `ASSETS`; `run_m4.sh 0 OUT server` / `run_m4.sh 1 OUT follow`,
   `GRAPHS=1`). PASS = rank 0 logs `serving: 4 slot(s)` and `row graphs: on`, no `UnknownRowsCall`, no
   `MissingTritonVariant`, the canaries answer. Record both ranks' free memory after boot (the plan's new `L.ix.keys`
   / `L.scores` / `L.ix.bkeys`: about 208 MiB a rank).
5. **Long context on 4 slots** (the row path's blocked selections and dtopk at 400K / 900K):
   `python3 $K/bench_http.py --url http://127.0.0.1:8091 --label zig --long 400000,900000 --out $K/out/zig-long2.json`
   while a second shell runs `python3 $K/bench_http.py --url http://127.0.0.1:8091 --streams 4 --reps 1 --out
   $K/out/zig-mix.json` (short slots beside the long one). PASS = no error on either rank, and
   `bench_http.py --compare $K/out/prod-long2.json $K/out/zig-long2.json` equal T0 replies (with the prompt-tail fix,
   `a390113`, in the kit). If they differ, report the first differing token per length; that is a new divergence,
   not this change.
6. **No missing variant across the night's workloads:** rerun run B2's bench (`--streams 1,4`) and run C's 32K / 131K
   cells on the 4 x 1M server, with no new AOT phase. PASS = 0 `MissingTritonVariant` in both ranks' logs.
7. Prod back on perf1 (`scripts/v2-rollback.sh perf1`). Keep `out/`, `triton-fill/`, the `sass` / `merge` output and
   both ranks' logs in the quant repo's results.

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
  1. Done: the speculative early pass (`TF_DSV41_SPEC_DRAFT`) and its nucleus rounds (`TF_DSV41_SPEC_NUCLEUS`) (`draft/spec.zig`, section 0a). Its pod gate is `M2B_SPEC=1`. Python's acceptance on the same prompts is compared in the Spark window.
  2. Done: the first round drafts a full DSpark block, and the depth cuts it, as in Python (section 0a).
  3. The cost of verify windows over 32 rows is to be measured once M2's forward exists.
  4. Load the real prod calibration table (`calib-*.json` from the Sparks' `/cache`, through `Costs.parseJson`) once the Spark agent can copy it.
  5. M2 contract: a chain window's rows stay in the target's state until `keep()` is called, because in a shared round the lane core asks for drafts before it keeps. Ingest reads the kept path from the window's taps, not from the pool.

**M4. Server** (about 8–12 days).
- `zig/src/native/cuda.zig`, wiring into `engines.zig` for Linux builds.
- `encoding.zig`, `dsml.zig`, reasoning effort, `memory.zig` (the floor and adaptive prefill rows), admission and fairness for 4 slots, and `/metrics`.
- Grammar: the Zig server's structured-output fields and the engine's grammar enforcement: xgrammar 0.2.8's C++ core (R9, section 0a "Structured output").
- **GPU:** both Sparks, then a soak.

- **State:** the server side is done (`deepseek_v41/serve/**`, `zig/src/server/**`, commits `ad44e18..f66c019`).
  - Tokenizer, encoding, tool calls and sampling are byte-identical to Python on large golden sets.
  - 43/43 HTTP tests pass on a stub engine; the tokenizer is about 23x faster than HF's.
  - The real engine plugs in by implementing `engine_api.Engine`.
- **Open items:**
  1. Native image input (`TF_DSV41_IMAGES=native`, `TF_DSV41_BIAS_VL`): built (section 0a, `dsq-zig-vision-1`); its pod step is `M2B_VISION=1`. WebP and TIFF are refused.
  2. Structured-output grammar (R9): built behind `TF_DSV41_GRAMMAR=1` (section 0a "Structured output"); its pod step is `M2B_GRAMMAR=1`.
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
9. **R9, structured output.** Decided: xgrammar 0.2.8's own C++ core, built by Zig behind a C ABI (section 0a "Structured output"): masks bit-identical to prod's by construction, checked by `test-grammar`. `TF_DSV41_GRAMMAR` defaults to off in the Zig engine (prod's Python default is on): the kit must set it.
10. **R10, the vision tower.** It runs on rank 0 only (`TF_DSV41_IMAGES=native`), in Zig as in Python (section 0a, `dsq-zig-vision-1`). Open: the tower's bits against torch's SDPA and cuBLAS (the pod step names the first differing stage), WebP / TIFF input.
11. **R11, upstream churn.** `zig-flashnext` moves fast. The family stays in its own directory, and touches of shared files are kept to one function each in `build.zig` and `zig/build/cuda.zig` (the latter is the kernel agent's).
