# TP=2 runtime on the Zig engine (`zig/src/fabric/tp`)

This is the multi-process tensor-parallel runtime for DeepSeek-V4.1 on the Zig engine. It closes risk R3 of `DSV41-ZIG-PORT.md`: the engine had NCCL calls but no way to bootstrap NCCL across two hosts. The target is two DGX Sparks with one GPU each, linked by CX7. The Python engine's equivalents are `tensorfold.cuda.comm`, `deepseek_v41/cuda/failfast.py`, `planlink.py` and GLM's `spark/roce.*`.

## Interface

`collective.zig` defines `Collective`, a vtable. The model sees only this:

| call | contract |
| --- | --- |
| `allReduce(src, dst, count, dtype, op, stream)` | sum / max / min, added in rank order, so every rank holds the same bits |
| `allGather(src, dst, count, dtype, stream)` | rank 0's shard first (NCCL's layout) |
| `exchange(src, dst, count, dtype, peer, stream)` | two ranks trade buffers in one group (Python's `swap`) |
| `send` / `recv` | point to point |
| `barrier()`, `check()` | host calls. `check` reports a failure the transport recorded. |
| `abort()` | any thread. It ends pending and later work; kernels waiting on a dead peer return. |

Every data call only enqueues on `stream`. There is no host sync and no allocation, so every call can be captured in a CUDA graph.

| Backend | File | Use |
| --- | --- | --- |
| NCCL | `nccl.zig` | Default. `ncclSend`/`Recv`/`CommAbort`/`CommGetAsyncError` were added to `zig/src/cuda/nccl.zig`. |
| Mailbox | `mailbox.zig`, `mailbox.cu` | One-kernel exchanges for two ranks on one host (below). |
| Hybrid | `hybrid.zig` | Mailbox for shards up to `TF_TP_MAILBOX_MAX_KB`, NCCL above that. The route depends only on the call, dtype and size, so both ranks always pick the same transport. |
| In-process | `host.zig` | Ranks as threads over host memory, for unit tests. Same contract; a desynchronised call is refused instead of hanging. |

## Bring-up (`tp.Session.open`)

1. **Bootstrap** (`bootstrap.zig`). Rank 0 listens on the plan link's port, `TF_DSV41_PLAN_PORT` or else `TF_TP_PORT` + 7. The other ranks connect with retries until `TF_DSV41_PLAN_TIMEOUT` (300 s). The topology is a star; small host blobs are all-gathered.
2. **Agreement.** The ranks must agree on the settings digest (world, backend, mailbox size, fail-fast) and on the NCCL version. If either differs, both ranks refuse to start.
3. **Unique id.** Rank 0's `ncclGetUniqueId` is broadcast, then every rank calls `ncclCommInitRank`.
4. **Mailbox** (`TF_COMM_BACKEND=mailbox`). It is enabled only if every rank has the kernel and every rank reports the same boot id. Otherwise all ranks stay on NCCL and a log line says so.
5. **Fate channel.** The bootstrap sockets are kept open for it.

**Environment:**

| Variable | Meaning |
| --- | --- |
| `TF_TP_WORLD` | number of ranks |
| `TF_TP_RANK` | this rank |
| `TF_TP_MASTER` | rank 0's address |
| `TF_TP_PORT` | base port |
| `TF_TP_DEVICE` | GPU index |
| `TF_COMM_BACKEND` | `nccl` or `mailbox` |
| `TF_TP_MAILBOX_MAX_KB` | largest mailbox shard (256) |
| `TF_TP_MAILBOX_TIMEOUT_S` | mailbox wait limit (120) |
| `TF_NCCL_LIB` | path to the NCCL library |
| `TF_DSV41_FAILFAST`, `TF_DSV41_FAILFAST_GRACE_S`, `TF_DSV41_PEER_MEM_S` | fail-fast knobs, same names as in Python |

## Fail-fast (`fate.zig`)

This is the Python G16 design. Frames are `<IIq>` with magic `0x46463431`:
- Rank k sends `MEM` frames (MemAvailable) every 0.5 s.
- A failing rank sends `FAIL` with its reason.
- A hang-up without `FAIL` (crash, SIGKILL) is the same signal.

A rank that learns of a failure does the following in order:
1. Aborts its collectives on a detached thread. For NCCL this is `ncclCommAbort`; for the mailbox it sets the shared abort word, which ends both ranks' spinning kernels.
2. Runs the `onFail` hooks (rank 0 fails its requests).
3. Hangs up.
4. Waits the grace period, then `_exit(70)`.

`Session.fatal(where, err)` is the call for a failure inside a round. `fate.expectStop()` marks a normal stop.

## Mailbox protocol (`mailbox.cu`)

This is the Python RoCE all-gather's protocol, ported:
- a device epoch, so graph replays continue the sequence;
- `seq = epoch + 1` with two parity slots;
- a flag after the payload;
- system-scope acquire polling;
- a `%globaltimer` timeout that poisons the transport;
- the last block publishes the epoch only if no wait failed.

The ranks share a host, so there is no doorbell, proxy or NIC. Each block writes its share straight into the peer's inbox and raises its own flag, and the peer's block with the same index waits for that flag alone. Sums are added during the copy-out, using the same expression on both ranks.

The shared region is a `/dev/shm` file. Rank 0 creates and sizes it before broadcasting the name, both ranks map it and register it with `cuMemHostRegister(PORTABLE|DEVICEMAP)`, and then the file is unlinked.

**Not ported yet:** the CX7 transport for the Sparks. That is the same kernel protocol with a doorbell and an RDMA proxy thread (`spark/roce.cpp`; `fabric/verbs*.zig` already has ibverbs). Until it exists, `TF_COMM_BACKEND=mailbox` across two hosts falls back to NCCL.

## Results: RunPod, 2× RTX PRO 6000 Blackwell in one pod

The test was `run_pair.sh` on pod bk2tai4i1tsk73: PCIe with no NVLink, NCCL 2.30.7, sm_120, 7 minutes, $0.49.

- **Bootstrap plus NCCL init:** 0.64–0.69 s.
- **Bit-exact:** 378 of 378 checks pass on each rank, for NCCL and for the mailbox route:
  - all-reduce sum for f32, bf16, f16 and i32, out of place and in place at a skewed offset;
  - all-reduce max and min (f32);
  - all-gather (out of place and in place) and exchange for u8, bf16 and f32;
  - send/recv;
  - sizes from 6 B to 4 MiB, including 256 KiB + 16 B, which crosses the hybrid's routing threshold.
- **Graph replays:** a captured chain (sum → gather → sum → exchange) replayed with fresh inputs, with eager calls between replays.
- **NCCL over TCP sockets:** with P2P and SHM disabled, it was also exact (189 of 189). The pod has no RDMA device, so this is sockets, not RoCE.
- **Fail-fast:** 8 of 8 pass (NCCL or mailbox × own failure or SIGKILL × victim rank 0 or 1). The survivor exits 70 between 1.13 and 1.26 s after the injection; a killed rank's peer exits after 1.14–1.23 s.

Latency in µs per call, graph-replayed (100 calls per graph), bf16:

| bytes a rank | all-reduce NCCL | all-reduce mailbox | all-gather NCCL | all-gather mailbox |
| ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 6.17 | **5.04** | 6.17 | **5.27** |
| 8 KiB | 6.76 | **5.32** | 6.20 | **5.52** |
| 16 KiB | 8.22 | **6.68** | 7.18 | **6.98** |
| 64 KiB | 9.24 | **8.69** | 10.23 | **8.71** |
| 256 KiB | 18.46 | **15.92** | 24.18 | **16.12** |
| 1 MiB | 44.3 | (NCCL) | 47.1 | (NCCL) |
| 4 MiB | 131.3 (32 GB/s busbw) | (NCCL) | 147.5 (28 GB/s) | (NCCL) |

**Decode window:** 86 all-gathers of one row's bf16 partials (8 KiB) in one graph. NCCL takes 561 µs (6.5 µs per exchange); the mailbox takes **475 µs (5.5 µs per exchange)**. NCCL over sockets takes 2.9 ms (33.7 µs per exchange).

## Comparison with the Python engine on the Sparks

These numbers are not apples to apples: the Sparks cross CX7 RoCE, while the pod uses one host's PCIe. The Sparks were not used.

| | Python on 2 Sparks | Zig on the pod |
| --- | --- | --- |
| NCCL all-gather in a graph, small | about 27 µs alpha (COMM-ANALYSIS) | 6.2 µs (P2P), 19–34 µs (sockets) |
| custom exchange, one row | RoCE, G14a: late rank's kernel 6.5 / 7.5 / 17.6 µs (p10 / med / p90); one-way 5.1–5.7 µs | mailbox: 5.0–5.5 µs per call |
| one 1-row window's exchanges | 0.62–0.83 ms critical path (84 exchanges) | 0.47 ms (86, mailbox) |
| fail-fast exit after a failure | about 1 s grace (G16) | 1.13–1.26 s |

## Build and run

```
zig build test-tp                                # host tests: bootstrap, fate over TCP, in-process ranks
zig build tf-tp-test -Dtarget=x86_64-linux-gnu.2.35 -Dtp-nvcc=nvcc        # or -Dtp-fatbin=/abs/tp_mailbox.fatbin
zig/src/fabric/tp/run_pair.sh zig-out/bin/tf-tp-test OUT                  # on a two-GPU host
```
