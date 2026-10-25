# MiniMax-M3.1 on TensorFold main: proposed upstream pull requests

The `minimax_m3` family (MiaAI-Lab) and our MiniMax-M3.1 engine work, rebased from v1.1.2 onto `main` at
`81b476df1` (2026-10-25), as one commit group per future pull request. Branch `mm3/main-stack`; each group's last
commit is also branch `mm3-up/NN-*`. This file is a working plan, not part of any pull request.

## Base

`main` at `81b476df1` builds and passes its gates on our test machine (the two timing tests noted below aside), so the
stack sits on `main`, not on `v1.1.4`. Upstream has no `minimax_m3` family yet (only `docs/recipes/minimax-h3-*`).

## Proposed sequence

| PR | Branch (last commit) | Title | Commits | Depends on |
| --- | --- | --- | --- | --- |
| 1 | `mm3-up/01-family` | minimax_m3: the MiniMax-M3.1 family | 3 | main |
| 2 | `mm3-up/02-loader` | minimax_m3: load the published EXL3 pack (out_scales auto, 7/8-bit dense, f16/bf16 router) | 4 | 1 |
| 3 | `mm3-up/03-gpu-contracts` | minimax_m3: index kernel and pack-shape GPU contracts | 1 | 2 |
| 4 | `mm3-up/04-layer-parity` | minimax_m3: single-layer parity harness against ExLlamaV3 | 2 | 2 |
| 5 | `mm3-up/05-robust` | cuda: opt-in coherent TP failure; minimax_m3 memory floor | 2 | 1 |
| 6 | `mm3-up/06-speed-graphs` | minimax_m3: row CUDA graphs and adaptive prompt rows | 4 | 5 (fingerprint layout), 4 (harness fix) |
| 7 | `mm3-up/07-prompt-kernels` | cuda exl3: prompt-sized x3gm experts and pfdense rows; minimax_m3 prompt chunks use them | 5 | 5 (admission), 3 (test usage text) |
| 8 | `mm3-up/08-api-tools` | server: MiniMax-M3.1 tool calls, thinking modes and Anthropic stream fixes | 6 | 1 |

PR 8 touches only the server and can go in parallel with PRs 2 to 7. PRs 3 and 4 are independent of each other. The
stack order here is the order the branches merged on the old base, so each later group's conflict resolutions are
already in it.

### PR 1: the family (Mia's AI Lab, ported by us)

- `e4faa7660`-equivalent `minimax_m3: the MiniMax-M3.1 family (MiaAI-Lab's 0001-minimax-m3-family.patch)`, author
  Mia's AI Lab: the patch at `fd1d379` applied to `main`, textual conflicts only (each side kept; listed in the message).
- `minimax_m3: one-line comments and 600-line files for main's lean rule` (ours): the patch added 200 lean findings
  against `main`; comment-only rewrites, plus two moves so upstream files stay at 600 lines (`tool_parse.zig`:
  `tool_minimax.appendBlocks`; `contract_tests.zig`: `contract_minimax_register.zig`).
- `minimax_m3: build on main's reply_text and Zig 0.17 fmt` (ours): MiniMax's opener in main's new `reply_text.leads`
  table takes no payload lead (v1.1.2 behaviour); `@backingInt` and one literal for `zig fmt`; DeepSeek's nvcc 13.0
  launch records for the 28 x3gm/x3ld_epi launches the patch's activation branch recompiles (see below).

What it changes for other families (all opt-in or additive):
- `core/kv_pool.zig`: `Layout.all_paged` (default false) beside main's `run_bytes` and `Budget.engine_paged`; `Plan.paged`
  (equals `lanes - 1` for every existing family).
- `native/cuda_kv_pool.zig`, `native/cuda.zig`: `owns_prompt_cache`, `cache_limit` and `Options.grant` hooks, used only
  by a family that declares them; registry gains `minimax.native`.
- Shared EXL3 x3gm kernels (`exl3_block/x3gm*.cu`, `x3ld_epi.cu`, `exl3_experts.cu`, `exl3_prompt.cuh`) and their Zig
  drivers: swigluoai activation modes 2 and 3 (`alpha`, `up_bias`) beside modes 0 and 1, whose arithmetic is unchanged.
  GLM and DeepSeek run these kernels, so the kernel-contract and launch-limit receipts are required here; two recorded
  register counts change (`tf_exl3_gateup_epilogue`, `tf_exl3_prompt_gu`). The runtime `act_mode` branch also
  recompiles DeepSeek's `dsv41_x3gm` (27 launches, -5 to +3 registers, all at or under 128) and one `dsv41_x3ld_epi`
  launch (168 to 165); spills are unchanged and every launch still fits sm_80 to sm_121, but a DeepSeek decode/prefill
  receipt (or making the mode a template parameter so DeepSeek's code is untouched) should come with this PR.
- Server: per-prompt think markers (`Generation.markers`, `Server.markersFor`) feeding main's token-level think split;
  MiniMax tool-call style in `tool_parse`, `tool_stream`, `tool_grammar`. Other families get their old markers.

### PR 2 to 8

| PR | Shared code touched | Effect on other families |
| --- | --- | --- |
| 2 | none (family loader, config, tests) | none |
| 3 | `tests/cuda/exl3_dense_tests.zig` (exports), `tests/cuda/main.zig` usage | none |
| 4 | `tools/mm3_layer_ref.py`, `tests/cuda/main.zig` | none |
| 5 | `cuda/tp_group.zig` (`Group.coherent`, default false), `tp_watch.zig` (`exitCoherently`, exit 70), `roce.zig`, new `roce_mark.zig` | none unless a group sets `coherent`; GLM, Nemotron, DeepSeek keep exit 1 and their RoCE behaviour (host tests pin both) |
| 6 | `tests/cuda/main.zig` | none (graphs, row plan and fingerprint are family-local) |
| 7 | `cuda/exl3.zig` (`Dense.usePromptTiles`), new images `exl3_gm`, `exl3_pfd`, `x3gm.cu`'s `X3GM_FLUSH_K` (defined only by `exl3_gm.cu`), 88 registered launches (65 `exl3_gm` records re-measured: they were already stale on the old base; the same source compiles to the same registers on both bases) | DeepSeek's x3gm build is unchanged (the flush macro is undefined there); `usePromptTiles` is opt-in |
| 8 | `server/tool_parse_test.zig` (main's linear-time check timed at 8,000/64,000 openers; it failed at 1,000/8,000 under load on main too), `server/anthropic.zig`, `messages.zig`, `prompt.zig`, `chat.zig`, `reply_text.zig`, `tool_*`, `engine_api.zig` (`Info.first_mask`, default true) | Anthropic: text held behind an open `tool_use` block is now copied (all families), a restarting server's stream error is 503; an engine that reports `first_mask = false` gets grammars after a forced first token; the rest is MiniMax-gated |

## Upstream changes our work had to meet

- TP/comm `Group` (`cuda/tp_group.zig`, `tp_watch.zig`): unchanged on main since v1.1.2; PR 5 applies as written.
  `cuda/root.zig` gained `roce_nap.zig` beside our `roce_mark.zig`.
- EXL3: `cuda/exl3.zig` gained 7 lines on main; `exl3_gm`/`exl3_pfd` and the x3gm block files are unchanged on main,
  so PR 7's kernels apply as written. The contract image list moved from `contract_tests.zig` to `contract_limits.zig`,
  and `landed_cuda` gained `dory` and `glm_f32w` records; our images and records were added there.
- Server: main's tool-call opener table gained LFM2 and a per-opener `leads` table; think splitting now finds the
  close token in the reply's tokens (`splitThinkingAt`, `splitThinkingTokens`, `Server.close_id`). MiniMax's per-prompt
  markers feed those, and `close_id` is read after the MiniMax markers are chosen. PR 8's "lone close" early return sits
  before main's token-close lookup. `anthropic.zig`, `messages.zig`, `tool_grammar.zig`, `tool_stream.zig` are unchanged
  on main.
- KV pool: main added `engine_paged` and `run_bytes` (qwen3_5's paged engine window and recurrent runs). Mia's
  `all_paged` (no stream outside the pool) is a separate switch; both are kept and the plan arithmetic honours both.
- Graphs: no upstream change touches the family's row graphs; they build on `cuda` graph capture as before.
- Integration resolutions from `mm3/integrate` (old base), carried unchanged: `ranks.zig` floor + speed fingerprint
  words (layout 3), `engine.zig`/`engine_dev.zig` parts, `tests/cuda/main.zig` usage order, `promptTerms` pricing the
  428 MB x3gm scratch (its own commit in PR 7), and the parity harness's scratch-size fix (its own commit in PR 6).
  The family tree equals `mm3/integrate`'s byte for byte apart from the lean-rule comment edits.

## Evidence (one GB10, sm_121, driver 580.178.04, nvcc 13.0.88, Zig 0.17.0, SM clock cap 2100 MHz)

Upstream's no-GPU CI list plus `test-cuda-host`, on `main` and on the stack, same machine:

| Gate | main `81b476df1` | stack | group 01 alone |
| --- | --- | --- | --- |
| `zig build test` | fail: 2 wall-clock linear-time tests under load (cortheon fuzz, tool_parse namespaced openers) | pass | pass |
| `test-kernel-contract` | pass | pass | pass |
| `test-golden` | pass | pass | pass |
| `test-server-security test-pull-security` | pass | pass | pass |
| `fuzz-cortheon` | pass | pass | pass |
| `test-cuda-host` | fail: cortheon linear-time test under load | pass | pass |
| Flash Next serve unittest, build-options unittest | pass | pass | pass |
| plain `zig build` | pass | pass | pass |
| `zig fmt --check build.zig zig` | 3 files (`qk_rows.zig`, `dory/ref_fixture.zig`, `l2_ref.zig`) | the same 3 | the same 3 |
| `lean_check` / `lean_new` | 435 problems | 434, 0 new | 0 new |
| NFKD tables (Python 3.14) | not run: the test machine has 3.12; no table or generator changed | | |

The two `main` failures are timing assertions that failed while four other builds shared the machine. The cortheon
one is untouched and passed on the stack's runs; the tool_parse one is the check PR 8's last commit re-times at 8,000
and 64,000 openers (its 1,000-opener size measured build noise, not growth). The host gates above are from the final
27-commit tree; the GPU checks below ran on the tree before the two late commits and the launch-record refreshes,
which touch no kernel, forward or loader code.

GPU (`tf-cuda-test`, ReleaseSafe, under the GPU lock):

| Check | Result |
| --- | --- |
| `mm3-contract` | 37 kernels, 37 pass gates 1-4, 0 fail, 0 known (old base: 37/37) |
| `mm3-proj-contract`, `mm3-dspark-contract` | 4/4 and 14/14 pass gates 1-4 |
| `mm3-pack`, `mm3-prompt-contract`, `mm3-prompt-dense` | pass |
| `mm3-graph-eq` (1; 2 moe=1; 1 kv8) | pass, bitwise |
| `mm3-rows-eq` (1; 2 moe=1; 2 kv8 moe=1) | pass, bitwise |
| `mm3-synth-smoke` (1; 2 moe=1; 1 kv8; 2 kv8 moe=1; 4) | pass |
| `kernel-contract` (all runners) | aborts on main and on the stack alike: upstream's wide-branch dim512 GQA16 mode2 shapes case hits `CUDA_ERROR_MISALIGNED_ADDRESS` and loses the context before the MiniMax kernels run; the MiniMax runners above pass alone |
| `kernel-contract-limits` sm_80/86/89/120/121 | every launch fits; 171 stale records, the same 171 as on main, 0 new (4,069 current) |

Layer parity on the real MiniMax-M3.1-Flash EXL3 pack against ExLlamaV3 1.6.4 (852-token prompt, cosine over all rows):

| Layer | attn | routed | shared | out | old base (`mm3/integrate`) |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.000000 | | (dense mlp 1.000000) | 1.000000 | identical |
| 3 | 0.999999 | 0.999861 | 0.999999 | 1.000000 | identical |
| 18 | 0.999993 | 0.999975 | 0.999995 | 0.999999 | identical |
| 31 | 0.999991 | 0.999978 | 0.999990 | 0.999999 | identical |
| head | cos 0.999998, top-1 8/8, top-5 overlap 1.0 | | | | identical |

Every partial (`.npy`) the harness writes is byte-identical to the old-base integrate build's on the same inputs.
Against the layer-parity branch alone (before prompt kernels), routed cosine moves L3 0.999929 to 0.999861 and L31
0.999959 to 0.999978: prompt chunks now run x3gm/pfdense with a different summation order (PR 7's receipt); the head is
unchanged. All parts stay at or above 0.9999.

PR 8's `tool_choice` required/named and structured output need the optional `libtfgrammar` runtime library at serve
time (`zig build grammar -Dxgrammar=DIR`, xgrammar v0.2.8, which main already pins); without it the server refuses
those requests with 400. Deployments build it beside the binary.

Not run in this rebase: two-rank (TP2) serving, served decode and cold-prefill benchmarks, drafted-equals-plain and
concurrent-equals-solo on the real model. Those need both machines and a model load, and are the release gates the
family PR will be asked for.

## Before any pull request

- Upstream asks for an issue before a new model family, to agree its gate, and an ATC claim; none exists yet. The
  family PR should open with that issue, the gate proposal (the parity table, contracts, TP2 exactness), and
  `atc overlap --json` checked against `docs/recipes/minimax-h3-*` work.
- Commit messages carry old-base measurements (lean counts, test counts); this file's numbers are the ones on `main`.
- Receipts under `docs/receipts/mm3-*` name no hosts or paths; recheck the full publication tree before upload.
- Mia's commit keeps her published identity (`Mia's AI Lab <mia.ailab1@gmail.com>`); upstream prefers a GitHub
  noreply address, and MiaAI-Lab is an organization, so there is no personal `Co-authored-by` identity to add.

## Recommendation

Send PR 1 first, then PR 2: the family alone cannot load the published pack without the loader fixes, and together
they are the reviewable unit "MiniMax-M3.1 loads and runs". PR 8 can follow in parallel because it is server-only.
PRs 3 and 4 are test-only and small; send them with or right after PR 2 as the evidence for the family gate. PR 5
touches the shared TP group (opt-in) and should go before the speed work. PRs 6 and 7 last: PR 7 changes shared EXL3
kernels and needs the full kernel-contract and launch-limit receipts.

On who opens PR 1: it is almost entirely Mia's AI Lab's code, so the cleanest path is for Mia to open it from her own
account (with our lean and port commits offered to her branch), which keeps authorship and the conversation with the
maintainers in her hands. If Mia prefers not to, we open it with her as author of the family commit and credit in the
description; upstream's rule is to keep every contributor's authorship through a port, which this stack already does.
JJ decides.

## Replaying the in-progress speed branches

`mm3/s8-loadpipe`, `mm3/s9-prefetch`, `mm3/s11-sampling` and `mm3/s12-prefill-ttft` are on the old base. When one is
ready: `git rebase --onto mm3/main-stack f6505e0d3 <branch>` (rerere is on and holds this stack's resolutions), fix
lean findings in that branch, then rerun the gates above plus that branch's own receipt, and add it as its own group
after PR 7.
