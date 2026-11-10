# Native environment validation

Malformed numeric settings are refused at startup with the variable, supplied value and accepted forms.
The same parser is used by direct CUDA entry points. An unset variable keeps its existing default.
A valid value keeps its existing meaning, including documented caps and clamping.
Diagnostics quote and escape the supplied text so it cannot create another terminal line.

| Variable | Accepted explicit value |
| --- | --- |
| `TENSORFOLD_VIDEO_TOKENS` | Nonnegative whole video-token budget; default 16384; stored as an unsigned 64-bit count |
| `TENSORFOLD_MAX_PDF_PAGES` | Whole count 1 through 4294967295, capped by `--vision-max-images`; default min(16, image limit) |
| `TENSORFOLD_PRESSURE_CACHE_PERCENT` | Whole percentage 0 through 100; default 50 |
| `TENSORFOLD_PREFIX_WAIT_MS` | Whole milliseconds 0 through 4294967295; zero disables waiting; default 60000 |
| `TENSORFOLD_RESTORE_PREFETCH` | Whole unsigned 64-bit count, capped at 65536; zero disables reading ahead; default 2 |
| `TENSORFOLD_RESTORE_PREFETCH_GIB` | Whole unsigned 64-bit GiB count, capped at 16777216; default 16 |
| `TENSORFOLD_RESTORE_PREFETCH_MS` | Whole unsigned 64-bit milliseconds, capped at 1099511627776; default 30000 |
| `TENSORFOLD_TOOL_ID_KEY_FILE` | The path of the file that holds the key for tool-call ids ([API](api.md#messages-and-tools)). The file must be a regular file, not a link, that only its owner can read or write (`chmod 600`), with 32 to 4096 bytes in it, and the path must not contain control characters. When this is unset, each server process makes its own random key. If the file is wrong, the server does not start; the message names this setting but never shows the path or what the file holds |
| `TF_LANES_BUDGET_LOG` | Exactly `0` or `1`; `1` writes a `[lanes-budget]` stderr line each shared or lone head round of the lane engine, and a `[lanes-costs]` line every 32 of them or when its shared-cost table changes; default off |
| `TF_CUDA_DEVICE` | Whole ordinal 0 through 4294967295, or empty; default 0 |
| `TF_CUDA_SEGMENTS` | Whole count 1 through 4, or empty; default 1 |
| `TF_CUDA_CARVEOUT` | Exactly `0` or `1`; default off |
| `TF_CUDA_CARVEOUT_MIB` | Positive whole MiB count whose byte size fits the platform, or empty for 1792 MiB |
| `TF_Q27_COPY_ROWS` | Whole u32 count; zero disables copied rows; default 48 |
| `TF_Q27_DRAFT_SLOTS` | Whole u32 count, clamped to 1 through the lane stream limit; default 8. DFlash2 can draft for at most the smaller of this count and the admitted streams, and none without a drafter. |
| `TF_MTP_DRAFTS` | Whole u32 count, clamped to 1 through the drafter's window less one; existing depth when unset |
| `TF_DRAFT_NODES` | Whole native-size nonnegative count, capped by the drafter's window; zero disables trees |
| `TF_QMMF_HALF_BELOW` | Whole native-size nonnegative block count; zero disables half tiles; default 96 |
| `TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS` | Exactly `0` or `1`, checked when the CUDA backend starts; default off ([profiles](profiles.md)) |
| `TF_HEAD_GEOMETRY` | Whole native-size nonnegative geometry count in the head benchmark |
| `TF_ROCE_GID_INDEX` | Whole index 0 through 2147483647; unset searches for a RoCE v2 IPv4 GID |
| `TF_FLASHNEXT_ENGINE` | Exactly `upstream` or `mia`; `mia` serves qwen4_exp checkpoints with MiaAI-Lab's Flash Next engine; default `upstream` |
| `TF_GEMMA4_CHUNK_SOLO` | Gemma 4 on CUDA: a multiple of 512 from 512 through 4096; unset, empty or 512 is off. Checked when the model loads, with the family's own message |
| `TF_DRAMA_PREFILL_ROWS` | Drama 3 on CUDA: whole prompt rows a chunk, 1 through 1024; default 1. Checked when the checkpoint opens, with its own message |
| `TF_DRAMA_REFERENCE_CACHE` | Drama 3 on CUDA: kept reference states, 0 (off) through 16; default 0. Checked when the checkpoint opens, with its own message |
| `TF_PROMPT_ROWS` | Native serving: a multiple of 16 from 128 through 2048; unset keeps the chip/family default |
| `TF_ANE_COLUMNS` | Whole u32 count; zero disables the experimental share; default zero |
| `XPU_A8_BITS` | Exactly `4` or `8`, for the XPU quantization study; unset keeps the study's existing setting |
| `LFM3_FUSE` | Whole u32 fusion mask; default 7 |
| `TF_SLIDE_POCKET_KEY` | Exactly `auto`, `reply`, `subject` or `both`; default `auto` |
| `TF_SLIDE_FIT_GIB` | Whole native-size nonnegative GiB count; default 4 |
| `TF_SLIDE_FORGET_FAIL` | Whole u32 fact index, zero off; only checked in a slide-test-hooks build |

The video-token reader has no fixed model-independent token ceiling. Its unsigned storage width is not a useful token budget.
The request geometry and model context decide whether a parsed budget fits; zero remains accepted as before.

u32 means 0 through 4294967295. Native-size counts use the platform's unsigned integer size.
Existing numeric spellings accepted by each reader, such as leading zeroes, remain valid.
Pressure, prefix-wait and restore-prefetch values retain trimming of spaces and tabs.
Prefix-wait and restore-prefetch still require the first remaining character to be a digit.
CUDA device/segments and the slide test hook retain their space trimming.
Other numeric readers do not gain whitespace aliases. Empty or whitespace-only text means unset for every variable, as it did before.
`TF_GEMMA4_CHUNK_SOLO` is the exception to "as before": its old reader refused blank text, and now reads it as off.
The table checks only that it is a whole number; the lane core's `solo_chunk.parse` judges it against the family's grid.
The two `TF_DRAMA_*` settings are another: their old reader refused text of only tabs or line breaks, and now reads it as unset.

An explicit `--device` or `--segments` wins over its environment value, even if the ignored value is malformed.
`--carveout` likewise enables the carveout regardless of its enable variable.
The carveout size is checked only when the carveout is enabled, and still rounds to framebuffer rows.
`TF_NO_TREES` retains precedence over `TF_DRAFT_NODES`.
Backend-specific settings are checked when that backend starts; slide settings require learning to be enabled.
Common server settings are checked before checkpoint resolution, including when no request would use them yet.
Direct family readers also refuse malformed values when called without the server.

## MiaAI-Lab's Flash Next engine

These settings are read only with `TF_FLASHNEXT_ENGINE=mia`. Each has a `flashnext_mia` row in `platform/environment.zig`.
Every row is checked when her engine opens, before it loads anything; a malformed value is refused with the usual message.
An accepted value keeps her engine's meaning: her readers still interpret the text, so the table only refuses text they would misread.
`TF_FLASHNEXT_CONFIDENCE` and `TF_FLASHNEXT_KEEP_LATE` are also read by the upstream Flash Next engine, with its own meaning.

| Variable | Accepted explicit value | Meaning (default) |
| --- | --- | --- |
| `TF_FLASHNEXT_DEPTH` | Whole count | Drafts a window, 1 through the lane core's most (15: 16-row windows) |
| `TF_FLASHNEXT_YARN` | Factor above 1, or `0`, `off`, `no`, `false`, `none`, `default` (any case) | YaRN factor; `4` gives a 1,048,576 window (the checkpoint's rope scaling) |
| `TF_FLASHNEXT_YARN_RAMP_POSITIONS` | Whole count | YaRN ramp positions (0) |
| `TF_FLASHNEXT_ROCE` | `0` or `1` | `0` turns off the one-shot RoCE all-gather between two ranks (on) |
| `TF_FLASHNEXT_L2PF` | `0` or `1` | `0` turns off the L2 weight prefetch (on) |
| `TF_FLASHNEXT_ROCE_MAX` | Whole byte count | Largest all-gather sent over RoCE (524288) |
| `TF_FLASHNEXT_L2PF_BYTES` | Whole byte count | Bytes of a weight to prefetch (8388608) |
| `TF_FLASHNEXT_FILL` | `0` or `1` | `1` fills prompts in layer slices between rounds (off) |
| `TF_FLASHNEXT_FILL_LAYERS` | Whole count | Layers a fill step, at least 1 (8) |
| `TF_FLASHNEXT_OVERCOMMIT` | `0` or `1` | `1` admits a stream against memory free now instead of its window's reserved caches (off) |
| `TF_FLASHNEXT_CONFIDENCE` | Finite number | Draft stop rule from -1 through 1: c > 0 per draft, c < 0 running product (-0.4); others keep the default |
| `TF_FLASHNEXT_PRODUCT_STREAMS` | Whole count | Live streams up to which the running-product rule applies (2) |
| `TF_FLASHNEXT_KEEP_LATE` | `0` or `1` | `0` turns off pending DeltaNet rows for partly kept windows (on) |
| `TF_FLASHNEXT_MTP_TOPK` | Whole count | MTP layer's routed experts a row; only 10 is used besides the checkpoint's (0: the checkpoint's) |
| `TF_FLASHNEXT_MTP_Q4` | `0`, `1`, `all`, or a comma list of `fc`, `attn`, `hc`, `mixer` | MTP head matrices quantized to 4 bits for drafts (off) |
| `TF_FLASHNEXT_MTP_Q4_MSE` | `0` or `1` | `1` picks the 4-bit scales by least squared error (off) |
| `TF_FLASHNEXT_PREFILL_ROWS` | Whole count, 16 through 16384 | Prompt chunk rows (4096 on two ranks, 2048 on one) |
| `TF_FLASHNEXT_PREFILL_TAIL` | Whole count, a multiple of 16 | Rows a last chunk may take instead of its own chunk (512) |
| `TF_FLASHNEXT_INT4AR_FAST` | `0`, `off`, `1` or `on` | `1` loads INT4-AutoRound's lossy `fast-fp8/` overlay (off) |
| `TF_FLASHNEXT_PROMPT_MM` | `0`, `off`, `1` or `on` | Prompt matmuls without split-K partials when the kernel set has them (on) |
| `TF_FLASHNEXT_DRAW_ONCE` | `0`, `off`, `1` or `on` | Two ranks draw a round's tokens once from gathered candidates when they fit (on) |
| `TF_FLASHNEXT_SHARED_SIDE` | `0`, `off`, `1` or `on` | Shared expert on a second stream beside the routed experts (on) |
| `TF_FLASHNEXT_FP8_LD` | `0`, `off`, `1` or `on` | INT4-AutoRound block-FP8 columns stored in place (on) |
| `TF_FLASHNEXT_FP4_SERIAL` | `0`, `off`, `1` or `on` | Shared expert's single-slice gate/up on the K-serial kernel (on) |
| `TF_FLASHNEXT_PROMPT_EXPERTS` | `0`, `off`, `1` or `on` | Prompt experts on 128-pair items (on) |
| `TF_FLASHNEXT_GLUE_SPLIT` | `0`, `off`, `1` or `on` | Two ranks split the prompt glue by rows (on) |
| `TF_FLASHNEXT_DROP_PAGES` | `0`, `off`, `1` or `on` | Drop the checkpoint's pages from the page cache once loaded (on) |
| `TF_FLASHNEXT_GLUE_FUSE` | `0`, `off`, `1` or `on` | Fused up-mix glue when the kernel set has it (on) |
| `TF_FLASHNEXT_WB_NORM` | `0`, `off`, `1` or `on` | Fused write-back norm when the kernel set has it (on) |
| `TF_FLASHNEXT_QSA_FAST` | `0`, `off`, `1` or `on` | Fast indexer score kernel (on) |
| `TF_FLASHNEXT_QSA_RT` | Whole count | Indexer score row tile (0: the default path) |
| `TF_FLASHNEXT_QSA_TILES_FROM` | Whole count | Key blocks from which indexer scores run in tiles (8192) |
| `TF_FLASHNEXT_QSA_SPLIT_KEYS` | Whole count | Keys a split of the indexer scores (65536) |
| `TF_FLASHNEXT_QSA_TILE` | Whole count | Indexer score tile choice, above 3 read as 3 (0) |
| `TF_FLASHNEXT_OVERLAP` | Whole number 0 through 255 | With split glue, its exchanges on their own stream in this mode (0: in line) |
| `TF_FLASHNEXT_COMM_PRIORITY` | Whole number | That exchange stream's CUDA priority (0) |
| `TF_FLASHNEXT_EXPERT_SMS_FREE` | Whole count | SMs the persistent expert kernels leave free for NCCL with overlap (0) |
| `TF_FLASHNEXT_EXPERT_SHAPE` | Whole count | Expert units up to which gate/up takes the NT shape (230) |
| `TF_FLASHNEXT_INT4_TILE` | Whole count | INT4 expert tile (the planned tile) |
| `TF_FLASHNEXT_INT4_PROMPT_DOWN` | `0` or `1` | `0` turns off the INT4 prompt down-projection kernel (on) |
| `TF_FLASHNEXT_PROMPT_EXPERTS_OCC` | `0` or `1` | `1` runs gate/up on the occupancy prompt-expert kernels; `0` also takes the down projection off them (gate/up off, down on) |
| `TF_FLASHNEXT_PROMPT_EXPERTS_GU2` | `0` or `1` | `1` splits the prompt experts' gate and up (off) |
| `TF_FLASHNEXT_PROMPT_EXPERTS_CFG` | Comma-separated whole numbers | Prompt-expert tile, gate/up and down threads (32,2,2) |
| `TF_FLASHNEXT_PROMPT_EXPERTS_ROWS` | Comma-separated whole numbers | Prompt-expert gate/up and down rows (4096,2048) |
| `TF_FLASHNEXT_MULTI_GRAPHS` | Whole count | Shared-round widths captured as graphs (4) |
| `TF_FLASHNEXT_ATTN_MULTI` | Whole count | Streams from which attention runs in one launch (graph widths + 1; 0: never) |
| `TF_FLASHNEXT_MEDIA_GRAPHS` | `0` or `1` | `0` runs media prompts without graphs (on) |
| `TF_FLASHNEXT_COUNT_EXPERTS` | `0` or `1` | `1` counts routed experts while profiling (off) |
| `TF_FLASHNEXT_PROFILE` | `0` or `1` | `1` profiles the forward (off) |
| `TF_FLASHNEXT_GRAPH_FAIL` | Whole count | Test hook: a graph launch fails every this many (0: never) |
| `TF_FLASHNEXT_GRAPH_LOG`, `TF_FLASHNEXT_PREFILL_DIGEST`, `TF_FLASHNEXT_POISON_PADS`, `TF_FLASHNEXT_PASS_TIMES`, `TF_FLASHNEXT_BURST_LOG` | `0` or `1` | Diagnostics, on whenever set, `0` included (off) |
| `TF_TP_TIMEOUT_S` | Whole seconds | How long a rank waits for the other (300) |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | Finite GiB amount of at least 0 | GiB her engine keeps free, as in the server's admission |

## Existing lenient switches

These settings retain their existing boolean policies rather than gaining new aliases or refusals:

- `TF_GRAPH_CACHE`: exact `0` or case-insensitive `off` disables graphs; every other value leaves the family's settings unchanged.
- `TF_QWEN27_SLOT_RESUME`: exact `0` disables resume; every other value keeps it enabled.
- Kolibri pages: exact `TENSORFOLD_KOLIBRI_KV_PAGES=0` disables requested pages; other text keeps the option unchanged.
- Dory pages: unset or exact `TENSORFOLD_DORY_KV_PAGES=0` uses windows; every other present value selects pages.
- `TF_SLIDE_EDIT`, `TF_SLIDE_PROBE` and `TF_SLIDE_ROUTED` select modes by presence, even empty or `0`; probe wins over edit.
- `TF_COOP_SPLIT` retains its first-character `0` disabling rule, `TF_NEMOTRON_TRACE` its first-character `1` trace rule,
  and `TF_SLIDE_FIT_REDUCE` its exact `tree` reduction selection. These developer switches keep their existing semantics.
- The existing tool-grammar, rows-sync, log-time, memory-pressure and XPU-MTP boolean policies are unchanged.

The profile registry reports selected arithmetic/experimental overrides; it does not change their parsing or defaults.
See [profiles](profiles.md) for classes, scopes and grouped slide-learning names.
`capabilities --json` no longer advertises `TENSORFOLD_NO_UPDATE_CHECK` or `HF_HUB_OFFLINE`: the native binary reads neither.
The command-line `--no-update-check` remains an accepted no-op.
