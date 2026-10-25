# MiniMax-M3.1-Flash: interface decisions (M1 to M14)

`zig/src/families/minimax_m3/iface.zig` fixes the signatures the engine, forward, prefill, backend, follower, weights and
kernels build against. The 13 mismatches the GLM-reuse study found are resolved here; each decision is also a constant,
a type or a test in the code, so a change to one fails a test instead of drifting. Nothing below ran on a GPU: this tree
was built and tested on a machine without one.

| # | Mismatch | Decision | Where it lives |
| --- | --- | --- | --- |
| M1 | Lane 0 sat outside the pool; `kv_pool.plan` counted only `lanes - 1` streams against it | Every stream pages, the engine's own too. `kv_pool.Layout.all_paged` makes `plan` and `holds` count all `lanes` streams, each with its table row and a page; `Plan.paged` is the number that take pages (`lanes - 1` for every other family, `lanes` here); `Loaded.free_streams` is 0 and `stream_bytes` 0; `attach(ctx, paged, pages)` receives all of them | `core/kv_pool.zig`, `native/cuda_kv_pool.zig`, `pool.Dims.layout`, `iface.decisions` |
| M2 | Only families with `snap_save` got `Info.prompt_cache` and a `pc.Store`; MiniMax's cache is a radix store over pool pages | A family may declare `owns_prompt_cache`. The host then builds no `pc.Store`, plans no copied bytes (the cached pages are pool pages), reports `Info.prompt_cache = true` when the plan leaves the cache room, and tells the family its limit through `cache_limit(ctx, pages, snaps)`: half the pool at most, bounded by `--prompt-cache-gib` (0 turns it off), snapshot records by `--keep` or 64. The startup line says "a prompt cache of up to N GiB of the pool's pages and M snapshot records" | `native/cuda_kv_pool.zig` (`owns`, `cachePages`, `kept`, `text`), `native/cuda.zig`, `native.cache_limit` |
| M3 | Nothing made the device copy of `Tables`; `Paged.row_tab` wants entry offsets | The engine owns one int32 device table of `slots x max_pages` entries and installs `pool.Tables.upload` to push rows on its stream (`Tables.write` rolls a failed push back). A row of a window reads at `slot * max_pages` (`iface.tableOffset`, `iface.rowTabs` fills the per-row array of a shared batch from its `Seg`s); a one-stream prompt chunk passes `row_tab = 0` with `tables` at that slot's row | `iface.zig` |
| M4 | Is the layout's page the kernels' page, and is `fixed_bytes` charged right | Yes, and tested: `Dims.layout(max_pages).pageBytes() == Dims.pageBytes()` (76,800 bytes a token at two ranks); `fixed_bytes` is one table row, charged once per stream, and the table is allocated for exactly the planned streams, so no lowering is needed | `iface` test M4 |
| M5 | Router bias, the missing fp32 GEMV, runtime `Switches` | The T0 `Switches` are frozen at their compile-time defaults (`decisions.switches_runtime = false`); none may become environment-selectable without entering `ranks.Fingerprint`. `routePlan` keeps refusing `router_weights_use_bias`. The fp32 router GEMV is kernel work (branch mm3-t8k), not an interface; its buffers are the `[rows, 128]` fp32 logits the T4 route kernel already reads | `iface.decisions`, `ffn.zig` |
| M6 | `exl3_experts.Layer` additions and the mixed layer 17 | `weights.uniformK2` is nonzero only when all three projections of a layer are one width for every expert; layer 17 (w1/w3 4-bit, w2 3-bit) therefore gets 0 and the kernel reads the per-expert widths. The GPU gate that layer 17 runs through `x3gm` stays open | `weights.zig` (tested on the mini pack's mixed layers) |
| M7 | `shared_input` needs a data comparison T2's plan does not do | Left false (correct, slower). A device-side comparison of the experts' gate and up `suh` can set it later; nothing depends on it | `weights.moeLayer`, `iface.decisions.shared_input` |
| M8 | Think markers were test-only | `Server.init` switches `markers` to `mm_think_markers` and records the two token ids when the tokenizer has `<mm:think>` and `</mm:think>`; the forced think close uses `</mm:think>`. Each reply is split by `Server.markersFor(prompt)`: a prompt ending in the opener (enabled mode) leaves only the close (`mm_think_prompt_markers`); adaptive and disabled prompts keep the full pair. The template's `thinking_mode` is now written as `enabled` / `disabled` for a template that has an `adaptive` branch (it was `thinking` / `chat`, which MiniMax's template read as adaptive) | `server/server.zig`, `server/chat.zig`, `server/hf_text.zig` |
| M9 | `tool_grammar.style` knew four markups; MiniMax fell back to the JSON envelope | `Style.minimax` is detected from the probe turn (`]<]minimax[>[<invoke name="tfprobe_fn">`). xgrammar cannot be shown to express it: each parameter is a tag named after the parameter (`NS<city>Paris NS</city>`), and no xgrammar source is in this tree to check a structural-tag style against. So `tool_choice` "required" or a named function is refused with `minimax_refusal`; "auto" and the streaming parser work as before | `server/tool_grammar.zig` |
| M10 | Two budgets: `budget.zig` and the host's memory plan | One source of truth for the reserve and the limit: the host passes its room before the load as `Options.grant` (net of `TENSORFOLD_MEMORY_RESERVE_GIB` and `TENSORFOLD_CUDA_MEMORY_LIMIT_GB`); `budget.hosted()` adds no reserve of its own, only the runtime and workspace terms the host cannot see. `open` runs `budget.check` with the plan's exact `slotBytes` and refuses with its words; the host's `kv_pool.plan` still decides the pages and has the last word, and `open` logs the budget's pool for comparison | `native.admission`, `budget.hosted`, `native/cuda.zig` |
| M11 | `TF_ROCE` and `TF_ROCE_MAX_KB` were in no fingerprint | Both are in `ranks.Settings` (`roce_kb`, 0 when `TF_ROCE=0`) and in the fingerprint (`roce`, the old pad word), with a refusal text. The byte-gather closure for `ranks.agree` wraps `glm/cuda_ranks.zig`'s `gather` in the engine (T8) | `ranks.zig` |
| M12 | Window widths | `prefill_rows` 2048, `window_rows` 16, `shared_rows` 64, `max_shared` 16, nested by comptime checks. UNVERIFIED on a GPU: widths 1, 2, 3, 16, 17, 64 and 128 must give the bits of one-token steps | `iface.zig` |
| M13 | Vocabulary spans | The head is cut in 128-row units (`shard.unit`); logits cross ranks in `tp.LogitsGather` slots padded to the largest share and greedy picks in `tp.mergePicks`. A sampler over MiniMax's spans takes `Share.vocab` per rank, not GLM's 64-unit `cuda_shard` | `iface.decisions.head_unit`, `tp.zig` |
| M14 | The dspark draft reads the target's hidden states and needs its own state per stream | The forward taps the draft's five layers into a taps buffer (`iface.tapRows`: a prompt chunk at rows 0..2047, a stream's last window at `tapWindowRow(slot)`); the draft keeps a K/V ring per slot (`engine_ops.Draft`, `dspark_device.zig`) driven by three follower kinds (`absorb`, `draft`, `snap`); the lanes backend answers `prepare_features`, `features` and `prefill_features` for the core's `Drafted`; the prompt cache saves a ring with each kept mark (`state.RingHook`); the ranks' fingerprint carries the draft's digest | `iface.Kind`, `engine.zig` `absorb`/`draftPass`/`draftSnap`, `backend.zig`, `dspark_drafter.zig`, `dspark_engine_test.zig` |

## What the host does for a family like this

`native/cuda.zig` reads, from the family namespace: `model_type`, `formats`, `default_context`, `max_segments`,
`prompt_rows`, `window_rows`, `Options` (with `context`, `drafts`, `segments`, `slide`, `lanes`, `lanes_fixed`, `room`,
`grant`), `open`, `explain` with `explains_open`, `weightShare`, `attach`, `fits`, `cache_limit` and `owns_prompt_cache`;
`iface.conformsNative` lists them and fails the build when one is missing.

## `open` today (T8b)

`native.open` does the host half as before (rank place and `NCCL_COMM_ID`, `Settings`, `config.json`, shard headers, this
rank's plan, the weights digest, the budget: each refusal is precise and comes first), then joins the ranks: `tp_watch.start`,
the NCCL defaults GLM sets (`NCCL_RUNTIME_CONNECT`, `NCCL_CUMEM_HOST_ENABLE`, `NCCL_CUMEM_ENABLE` = 0 unless set),
`ncclCommInitRank`, `warmPeers`, `Engine.init` (config, quantization and executable hashes plus the header digest go into
the `Fingerprint`, all-gathered as bytes through `ranks.agree`; a difference is `RankMismatch` on every rank), the weights,
the RoCE small gather, the control link, the rooms gather, `tp_watch.ready`. Rank 1 then follows and exits; rank 0 returns
`Loaded` (`stream_bytes` 0, `free_streams` 0, `layout` the pool's, `peer_room`). Until forward.zig and prefill.zig are merged
`engine_ops.available()` is false and `open` stops after the fingerprint with `error.ForwardNotBuilt` (explained) before any
weight is read. Nothing below ran on a GPU: `engine_dev.zig`, `draw.zig` and `native.openRank` are compile-checked only.

### The engine loop (T8b): files and protocol

`engine.zig` (slots, chunked prompts, windows, draws; logic over `engine_ops` seams, host-tested), `engine_dev.zig` (NCCL
`Comm`, device arena and block table, `NcclWire`, `Engine.init`), `draw.zig` (rank 0 draws on its host from the forward's
gathered fp32 logits with the lane core's keyed rule; the other ranks draw nothing), `follow.zig` (`Leader`, `follow`, `Link` over a `Wire`), `backend.zig` (`Lanes`).
`engine_fake.zig`, `engine_test.zig`, `backend_test.zig` are the recording fakes and tests (`zig build test-minimax-engine`).

The control link is `ncclBroadcast` of the 64-byte `iface.Header` then `n` ids (never a TCP mirror). Uses of the contract's
kinds, with the fields it left free:

| Kind | Header | ids | Rank 1 does |
| --- | --- | --- | --- |
| begin | slot, `flags` 1: sampled, seed, temperature, top_p, min_p, top_k | the stream's block-table row (lease page ids) | writes the row, starts the sequence |
| chunk | `a` rows resumed, `depth` start, `top_k` prompt length | the chunk's ids (a chunk ends at a page multiple, a mark or the prompt) | one prompt forward, gate first |
| prefill | as chunk, ids are the whole prompt | all ids | reserved: rank 0 sends chunks, never this |
| verify | slot | the window | one forward, sends its logits span |
| shared | `depth` parts | per part slot, rows; then every part's ids | one shared forward |
| keep | `a` rows (flags 0) | none | commits `a` rows |
| absorb | slot, `depth` start position, `a` low 32 bits rows and high 32 bits the first tap row | none | the draft's fc, hidden_norm and K/V ring writes for those tap rows |
| draft | slot, `depth` the pending token's position, `a` the pending token | none | one block pass of the draft |
| snap | slot, `flags` 1 save, 2 restore, 3 drop, 4 reset the slot's ring, `depth` snapshot id | none | the ring copy or reset |
| keep, flags 1 | `depth` first table entry | the new page ids | writes them (the radix store swapped pages at a mark) |
| release | slot | none | clears the slot's row |
| idle, flags 0 | none | none | sleeps on its liveness line (`tp_watch.sleep`) |
| idle, flags 1 | `depth` streams, `a` pages | none | makes the arena and tables (rank 0's `attach`) |
| end | none | none | leaves |

Deviations from the contract text, all in code and tests: (1) `flags` of verify and shared are 0, because only rank 0 draws,
so a grammar's masks (host words, `bits` is their address) stay on rank 0; (2) draws run on rank 0's host over the gathered
fp32 rows `[rank][row][slot]` the head leaves on every rank (no fp32 draw kernel and no per-rank pick kernel exist): argmax
takes the lowest id on ties, a sampled row uses `lanes.gpu_rule`'s rule through a bounded candidate heap (tested equal to
`gpu_rule.sample` on random rows, ties and masked tokens), costing one device-to-host copy of the rows per step (a device
kernel is a later saving, `mergePicks` stays for it); (3) a prompt chunk is `part = ids[start..end]` without the next id (no draft
head), and marks cut chunks, so every rank sees the same chunk grid; (4) a mark's page swap goes out before the chunk
step that follows it; (5) there is no `first_masked`: a grammar that constrains the first token is `StructuresUnsupported`
(later tokens are masked on rank 0); (6) `T6 state.Cache.admit` writes only the pages it takes into the table, so the
backend writes the whole lease row after it (adopted pages included), on rank 0's device and, through `begin`, on rank 1.

### T8b needs (from forward.zig / prefill.zig, branch mm3-t8a)

The engine calls them through `engine_ops.zig`; nothing else of theirs is used.

1. `engine_ops.Ops` over their buffers, returned by one function: `forward.open(args: engine_ops.OpenArgs) !engine_ops.Ops`.
   `OpenArgs` carries the driver and context, the engine's stream (every forward is enqueued on it), the config, the rank's
   `Share`, `*const weights.Weights` (already loaded by `weights_device.loadWeights`), `Settings`, `Place`, `capacity`, the
   NCCL library and communicator, and `group: *const ?*const tp_group.Group` (read at each exchange; set after open).
2. `Ops.attach(PoolView)`: the page arena (`arena + id * page_bytes`) and the int32 block table (`table`, `max_pages` entries
   a slot, -1 past a stream's pages; `iface.tableOffset`). Called at `Engine.attach`, again if the pool is remade.
3. `Ops.run(Forward{kind, ids, segs, starts})`: ids of all segments in order, segment `i` is slot `segs[i].slot` with rows
   `row0 .. row0 + rows` whose first key position is `starts[i]`. KV rows are written at their positions. Kinds: `prompt`
   (one segment, 128-multiple rows unless it ends the prompt; `head` false for a chunk that does not end it, and only its
   LAST row's logits are kept), `window` (one segment, all rows' logits), `shared` (up to 16 segments, 64 rows, all rows'
   logits in segment order). Both ranks get identical `ids`, `segs`, `starts`, so their collectives match call for call.
4. `Ops.logits(kind)`: device address of the fp32 logits of the last pass of that kind as every rank holds them after the
   head's gather, `[rank][row][slot]` (slot: the largest vocabulary share; world 1: `[row][slot]`), valid until the next
   `run` of that kind. `Ops.sync()` waits for every enqueued forward; `Ops.deinit()` frees their buffers.
5. Merge: T8a's actual API is `Forward.run(Call) -> Out` over `Modules`, `Buffers`, `Pages{cache, layer_bytes, mirror}` and
   `Link{group, lib, comm}`; it is not `Ops`. `patches/tasks/0011b-minimax-m3-engine-glue.patch` (on top of 0010 and 0011)
   is the adapter `forward_ops.zig` (about 110 lines), the three-line `engine_ops.zig` switch (`available()` true, `openOps`
   calls it) and the family `cuda.zig` lines; it was built and tested on a scratch merge of both branches
   (test-minimax-engine, -engine-host, -forward, check-native-cuda-host, compile-cuda-tests -Dsm=121 all pass there).
   Without it `open` stops at `ForwardNotBuilt` after the fingerprint. The kernels T8a lists as missing (fp16 index
   projection, `NoIndexProjKernel`) still stop the first forward on a GPU.

## NFS and the page cache

Rank 1 reads the checkpoint over a read-only NFS mount. The streamer reads with `O_DIRECT` and no mmap; a refused direct
read falls back to the page cache, which on a unified-memory Spark competes with the KV pool. `core/direct_io.File.dropCache`
(`posix_fadvise(DONTNEED)` on Linux, a no-op elsewhere) is called by `weights_device` after the last run of each shard,
through `load.Streamer.ended`. The digest (`weights_device.digest`) is a sha256 over each tensor's name, shard, dtype, shape
and byte range, which the shard headers already gave the catalog: no data byte is read for it. The marker check of
`Streamer.verifyMarkers` still does about 22,000 small direct reads; measure it over NFS before batching it.

## Credits

The contract and the host half follow TensorFold v1.1.2's own families; no code was taken from outside it. The native-server
registry and the CUDA serving path are Jürgen Schmied's (jschmied, PR #443), pooled KV pages are jschmied's and
cwschroeder's (#935), and the GLM family's two-rank engine, follower, idle sleep and structured output that
`iface.zig`'s signatures and `Header`/`Kind` are taken from are Bizuayeu's (TP over hosts #804, idle sleep #1053,
structured output #1017, tool calls #1018) with BobClawblaw (rank watch #855, #976) and BHCC2025 (the 128-unit vocabulary
cut #593). The backend-owned radix prompt cache the decision M2 builds on is BlivionIaG's and ThinkOffApp's HIP work
(#625, #626) with jschmied and BobClawblaw. The EXL3 operators `weights.zig` describes records for are vcruz305's EXL3
module (#42) inside TensorFold, with justincasey and BHCC2025, and turboderp's exllamav3 format. Ash Hart (ashhart)
and the TensorFold contributors wrote the rest of the tree this builds on. The forward and prompt driver of T8a take the
shape of v1.1.2's `glm/cuda_forward.zig` and `Engine.prefillChunk` (Bizuayeu, BobClawblaw and the contributors named above)
and call the EXL3 linear and grouped-expert wrappers (vcruz305's EXL3 module #42 inside TensorFold) as that file does.

## T8a requests

`iface.zig` is unchanged. The forward (`forward.zig`, `forward_plan.zig`, `forward_exec.zig`) and the prompt driver
(`prefill.zig`) are written against it. What T8a needs from other tasks, and what it chose where the plan says UNVERIFIED:

- Index projection and pick kernels (T8c, `minimax_m3_proj.cu`, `cuda/minimax_m3_proj.zig`). `index_proj` runs
  `Proj.project`: fp16 `index_q_proj` [256, 6144] and `index_k_proj` [128, 6144] times the bf16 `a`, fp32 fma sums in the
  router's fixed order (independent of rows), one rounding to fp16 (`opts.flags.index_fp16`) or bf16, into `iq_raw`, `ik_raw`.
  `local_pick` runs `Proj.pick` over `hlog` (own_vocab of slot logits, ids from vocab_lo) into `pairs` ([row] logit bits,
  id); `pick_gather` all-gathers them into `pairs_all` ([rank][row]). Order: larger logit, ties to the lower id, NaN last
  (`tp.betterPick` now ranks NaN last too). The padded logits gather (`.gather`, `tp.LogitsGather` layout `[rank][row][slot]`)
  is the other head mode; the engine (T8b) draws from `Forward.address(out)` with `Sampler.draw`, or merges picks.
- For T8b (engine, follower, backend): the engine owns the `Pages` (pool base, `rk.Mirror`, `rk.Cache`), the `Modules`,
  the `Buffers` and the `Link` (group, NCCL library and communicator); `Forward.run(Call)` queues one step on the stream
  and returns the head's layout. A follower runs the same `Call` on its rank: both ranks plan the identical op sequence
  (tested), which is what keeps the NCCL calls in step. `prefill.zig` is generic over a `host` with `capacity`, `gate`,
  `chunk`, `first` and `advance`; Engine.prefill and prefillChunk are a line each over it. Add `forward.zig` and
  `prefill.zig` to `cuda.zig`'s re-exports and test block (not done here: T8b edits that file).
- Chunking choice (not in the plan): a prompt pass splits chunks at rank 0's marks as well as at 2,048 rows, so every
  chunk starts on a page and a mark is always a chunk start. `prefillChunk` takes `part` as exactly the chunk's rows, or
  2,049 ids (a full chunk plus the next id, GLM's follower form); a non-final chunk must be whole pages.

### UNVERIFIED semantics chosen

- R1, residual stream: fp32 (`res`), the T0 default (`bf16_residual` false). Each block's output is the fp32 rank-order sum;
  `norm_add` adds it into `res` and norms into bf16 `a` in one launch (`Norms.addRms`, folded `1 + w` weights). A bf16
  residual would round at every add; if upstream does, the stream and the `norm_add` kernel change together.
- R2, routed plus shared: `Experts.run` writes the routed fp32 rows, the shared down projection writes its own, and
  `tf_rank_sum` with world 2 and me 1 adds them (`routed + shared`, one rounded fp32 add, no fma) into the partial that
  the one FFN exchange carries. The routed scale 2.0 is inside the router kernel (`ffn.routePlan` default); the planner
  assumes `post_scale` 1 (the frozen default switches).
- R3, prompt experts: chunks run the decode grouped kernel in calls of 256 rows (`plan.expert_rows`, ~125 MB of scratch);
  the x3gm prompt path is not wired. Prompt dense layers run `Dense.prompt` (128-row blocks of the decode kernel).
- R4, the head computes the final norm for every row of a chunk although only the last row is read.
- R5, index heads: `index_proj` is planned before `qk_rope` (which consumes it); T0 computes the index projection after
  the main norm and rope, which reads only `a`, so the order of the two does not change a value.

### Later step (speed), deliberately not done

Tensor-core prompt attention and scoring (speed-borrow-glm C1), RoCE small gathers and L2 prefetch beside the exchanges
(C2; `Group.allGather` already uses RoCE when the engine opened it), overlapped or row-split exchanges and expert chunk
sizing (C4), the shared expert on a side stream and launch order (C6), mul1 prompt kernels and x3seg decode kernels from
DeepSeek (candidates 1 and 2), CUDA graphs and keyed round graphs (candidate 6). Each needs a GPU gate first; the
planner's op list is the place to fuse or reorder them, and its tests must keep passing.
