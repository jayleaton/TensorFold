# TP=2 runtime on the Zig engine (`zig/src/fabric/tp`)

This is the multi-process tensor-parallel runtime for DeepSeek-V4.1 on the Zig engine. It closes risk R3 of `DSV41-ZIG-PORT.md`: the engine had NCCL calls but no way to bootstrap NCCL across two hosts. The target is two DGX Sparks with one GPU each, linked by CX7. The Python engine's equivalents are `tensorfold.cuda.comm`, `deepseek_v41/cuda/failfast.py`, `planlink.py` and GLM's `spark/roce.*`.

## Interface

`collective.zig` defines `Collective`, a vtable. The model sees only this:

| call | contract |
| --- | --- |
| `allReduce(src, dst, count, dtype, op, stream)` | sum / max / min, added in rank order, so every rank holds the same bits |
| `allGather(src, dst, count, dtype, stream)` | rank 0's shard first (NCCL's layout) |
| `allGatherV(src, dst, stride, lens, dtype, stream)` | `allGather` of `stride` elements where only rank r's first `lens[r]` bytes matter (`lens`: device int32 `[world]`, the same on every rank). Bytes past a length are unspecified. See "Variable lengths" below. |
| `varFits(bytes)`, `varAgreed()` | whether this rank's transport moves only the lengths for `bytes`-byte strides; whether every rank's does (agreed once a communicator) |
| `exchange(src, dst, count, dtype, peer, stream)` | two ranks trade buffers in one group (Python's `swap`) |
| `send` / `recv` | point to point |
| `barrier()`, `check()` | host calls. `check` reports a failure the transport recorded. |
| `abort()` | any thread. It ends pending and later work; kernels waiting on a dead peer return. |

Every data call only enqueues on `stream`. There is no host sync and no allocation, so every call can be captured in a CUDA graph.

| Backend | File | Use |
| --- | --- | --- |
| NCCL | `nccl.zig` | Default. `ncclSend`/`Recv`/`CommAbort`/`CommGetAsyncError` were added to `zig/src/cuda/nccl.zig`. |
| Mailbox | `mailbox.zig`, `mailbox.cu` (`tp_mailbox`) | One-kernel exchanges for two ranks on one host (below). |
| RoCE | `roce.zig`, `roce_verbs.zig`, `mailbox.cu` (`tp_roce`) | Two hosts: the kernel stages the shard and rings a doorbell, and a proxy thread RDMA-writes it to the peer over the CX7 (below). |
| Hybrid | `hybrid.zig` | The mailbox (kind `hybrid`) or RoCE (kind `roce`) for shards up to `TF_TP_MAILBOX_MAX_KB`, NCCL above that. The route depends only on the call, dtype and size, so both ranks always pick the same transport. |
| In-process | `host.zig` | Ranks as threads over host memory, for unit tests. Same contract; a desynchronised call is refused instead of hanging. |

## Bring-up (`tp.Session.open`)

1. **Bootstrap** (`bootstrap.zig`). Rank 0 listens on the plan link's port, `TF_DSV41_PLAN_PORT` or else `TF_TP_PORT` + 7. The other ranks connect with retries until `TF_DSV41_PLAN_TIMEOUT` (300 s). The topology is a star; small host blobs are all-gathered.
2. **Agreement.** The ranks must agree on the settings digest (world, backend, mailbox size, fail-fast) and on the NCCL version. If either differs, both ranks refuse to start.
3. **Unique id.** Rank 0's `ncclGetUniqueId` is broadcast, then every rank calls `ncclCommInitRank`.
4. **Small exchanges.**
   - `TF_COMM_BACKEND=mailbox` is enabled only if every rank has the kernel and every rank reports the same boot id. Otherwise all ranks stay on NCCL and a log line says so.
   - `TF_COMM_BACKEND=roce` sets up collectively: every rank opens its HCAs, the ranks trade connection records, then the device side comes up. A failure on any rank, or a load-time probe that differs from NCCL's bytes (a 4 KiB gather and a bf16 sum), puts every rank on NCCL. With `TF_TP_ROCE_FALLBACK=error` they all stop instead.
5. **Plan link.** Every other rank opens a second TCP connection to rank 0 at bootstrap, so `Session.plan` has its own socket. It is separate from the fate channel's, so the fate thread never consumes plan bytes.
6. **Fate channel.** The bootstrap's control sockets are kept open for it.

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
| `TF_TP_ROCE_HCA` | `name[:gid],...` RDMA devices (default: every active Ethernet port with an IPv4 RoCE v2 GID, by name, at most `TF_TP_ROCE_HCAS` = 2) |
| `TF_TP_ROCE_TC` | traffic class (default `NCCL_IB_TC`, else 0) |
| `TF_TP_ROCE_CPU` | pin the proxy thread, which busy-spins one core |
| `TF_TP_ROCE_FALLBACK` | `nccl` (default) or `error` |
| `TF_TP_ROCE_WIRE` | `verbs` (default) or `shm`, the RoCE protocol on one host without RDMA (tests) |
| `TF_DSV41_FAILFAST`, `TF_DSV41_FAILFAST_GRACE_S`, `TF_DSV41_PEER_MEM_S` | fail-fast knobs, same names as in Python |

## Variable lengths (`allGatherV`)

This is the Python engine's `comm.var_gather` / `var_fits` (cf28d37) for split KV's compact exchange (`TF_DSV41_KV_SPLIT_COMPACT`, `kv/split.zig`'s `densePacked`). The receive layout is `allGather`'s, `[world][stride]`: rank r's `lens[r]` bytes land at `dst + r x stride`. The lengths are device data, so a captured graph replays with new lengths and nothing syncs with the host.

| Transport | What moves |
| --- | --- |
| RoCE (`tp_roce`, a stride the small path takes) | Rank r stages and posts only `lens[r]` bytes: the doorbell carries `lens[r]` padded to 16, at least 16 (the proxy posts no empty op). The peer copies `lens[src]` bytes out. With a 16-aligned stride both copies round up to 16 bytes and stay on the 16-byte path. The proxy and the wire are unchanged: they already post the doorbell's count, and every HCA gets its flag even for an empty stripe, so the waits do not depend on the length. |
| Mailbox, NCCL, RoCE above `TF_TP_MAILBOX_MAX_KB` | The whole strides (`allGather`): the same bytes where `lens` says. |
| In-process (`host.zig`) | Only the lengths; the rest of `dst` is left as it was, so a reader past a length sees stale bytes. It refuses lengths that differ between ranks. `runWith(.{ .var_refuse = mask })` makes those ranks' transports refuse device lengths. |

`varFits(bytes)` is a rank-local answer. It is true only for RoCE with its device side, for a stride the small path takes. `varAgreed()` is every rank's `varFits(16)`. `Session.open` computes it once with an all-gather over the bootstrap and keeps it on the route (`Hybrid.var_agreed`); NCCL alone answers false. The in-process ranks compute it with one int all-gather the first time and keep it on the `Rank`. The answer lives on the communicator object, never in a cache keyed by its address: in Python, a reused `id()` once gave one rank a stale answer and the collectives fell out of step (dba6a1f).

Optional vtable entries (`all_gather_v`, `var_fits`, `var_agreed`; null by default) keep other `Collective` implementations building unchanged. A null `all_gather_v` falls back to `all_gather` of the whole strides.

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

## RoCE (`roce.zig`, `roce_verbs.zig`, `tp_roce` in `mailbox.cu`)

This is the Python engine's `spark/roce.cpp` and `roce.cu`, ported. It keeps the same protocol and the region layout's roles.

**Region.** Each rank has one pinned host region: ctrl (4 KiB), flags `[2 slots][2 HCAs]` at a 128 B stride, send `[2][slot]` and recv `[2][slot]`. It is registered with `cuMemHostRegister`, and with `ibv_reg_mr` on every HCA.

**Kernel.** Every block stages its share of the shard into `send[seq & 1]`. The last block to finish publishes the slot's padded byte count, then `seq` with a system-scope release (the doorbell). Each block then waits for every used HCA's flag and copies out of `recv[seq & 1]`. Sums are added in rank order during the copy-out. The epoch, timeout and poison rules are the mailbox's.

**Proxy.** This is `roce.cpp`'s `main_loop`:
- It spins on the doorbell and posts every new sequence; at most two can be pending, since a peer is never more than one op ahead.
- Per HCA it posts one RC queue pair work request for its stripe, chained to an inline 4-byte `seq` flag write, so the flag cannot land before the payload.
- It reaps completions after each op and while idle, and naps 20 µs only after 20M idle polls.
- A post or completion error sets the proxy failure word and this rank's abort word, so waits end at once.

**HCAs and GIDs.**
- The GID is the lowest RoCE v2 index (from sysfs, as `roce.py` reads it) that is IPv4-mapped. GID indices are never hard-coded.
- HCAs are paired across ranks by IPv4 /24, so the two CX7 functions match even when their names or order differ.
- Queue pairs use `roce.cpp`'s settings: MTU the lower of the two ports, `timeout 14`, `retry_cnt 7`, `rnr_retry 7`.

**Not ported.** These are `roce.cpp`'s later knobs, all off by default in Python:
- `STRIPE_KB`: one HCA below a size;
- `INLINE`: inline payloads;
- `LAZY_CQ`;
- `TRACE`.

They are speed items for after the first Spark numbers.

**A bug this port found: `%globaltimer` is not monotonic.**
- On the RTX PRO 6000, the RoCE kernel's waits sometimes ended in under a second with a 120 s timeout. The flag never arrived in time and the abort word was 0.
- That fits a `%globaltimer` read below the wait's start, which makes the unsigned `now - t0` wrap past the timeout. I didn't capture a failure record proving it; the evidence is the before/after below.
- The mailbox never showed it, because its flags come from the peer GPU within microseconds, so its polls are short.
- Fix: both kernels count time only forward (`now > t0 && now - t0 > timeout`). The failure record now also stores why the wait ended and for how long.
- Before the fix, every RoCE run failed (5 runs). After it, 3 of 3 passed, and so did the full gate.
- Python's `roce.cu` has the same expression (`globaltimer() - t0 > timeout_ns`) and is worth the same guard.

## Plan link (`planlink.zig`)

This is Python's `planlink.TcpLink`:
- One frame per plan: a 4-byte byte length, then little-endian int64s. The length goes out with `MSG_MORE`, so a plan leaves as one segment.
- Rank 0 sends to every other rank. The others block in `recv` for as long as rank 0 is idle.
- `recv(out)` uses a fixed buffer; a frame bigger than it is read and dropped (`error.TooLarge`), so the link stays on frame boundaries.
- `recvGrow(gpa, *std.ArrayList(i64))` grows the buffer to the frame, up to `max_frame_words` (64M words, 512 MiB). It covers admissions that carry a prompt (about 1M words for a 1M-token prompt). The buffer is kept, so window plans (20–60 words) allocate nothing. A 1.2M-word frame is in the host tests.
- A closed rank 0 returns `error.PeerClosed`, and the fate channel ends the process right after.

The RDMA host-mailbox variant (Python's `rdma` mode) will come behind the same API.

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

### Second pod: the RoCE kernel on real GPUs, one host

Pod 06p80w7rr3yg4k: 2× RTX PRO 6000, 21 min, $1.43. There is no RDMA in a pod, so the proxy ran over the shm wire (`TF_TP_ROCE_WIRE=shm`): it copies the send slot into the peer's receive slot and stores the flag. Everything else is real: the `tp_roce` kernel, doorbell, slots, flags, proxy loop, probe and fallback.

- **Bit-exact:** 378 of 378 checks per rank on the RoCE route, and again on the mailbox route (rebuilt with the timeout fix).
- **Fail-fast:** 8 of 8 on RoCE and mailbox; the survivor exits 70 between 1.14 and 1.32 s after the injection.
- **Latency, graph-replayed:**

| | 4 KiB | 8 KiB | 64 KiB | 256 KiB |
| --- | --- | --- | --- | --- |
| all-reduce, RoCE-over-shm | 6.44 µs | 7.60 µs | 19.2 µs | 64.2 µs |
| all-reduce, mailbox | 4.67 µs | 4.90 µs | 7.64 µs | 14.5 µs |
| all-reduce, NCCL | 6.17 µs | 6.49 µs | 8.71 µs | 18.4 µs |

The proxy round trip (doorbell → CPU → flag) costs about 1.5 µs over the direct mailbox at small sizes. The shm wire's copy is a CPU `memcpy`, so larger sizes say nothing about CX7; on the Sparks the NIC moves the bytes.

### Soft-RoCE (rxe) in a container: the verbs wire

The setup:
- `rdma_rxe` over a veth pair, with one end in its own network namespace, so the two ranks are two hosts as far as IP and RoCE are concerned.
- Rank 0 uses `rxa` (10.77.0.1), rank 1 uses `rxb` (10.77.0.2), over RoCE v2 with GID 1 (IPv4-mapped).
- The script is in the dev cache (`rxe-up.sh`); it needs a privileged container and the host's `/lib/modules`.

Results:
- **Host tests:** `zig build test-tp`'s RoCE tests over the verbs wire passed: 2,000 gathers of every size up to the slot, bit for bit. These are the tests behind `TF_TP_ROCE_TEST_HCA=rxa,rxb`.
- **`tf-tp-test roce-host`, two processes in two network namespaces:** bootstrap over TCP, then 1,000 gathers from 1 B to 256 KiB, all bit-exact.
- **Software-RoCE times** (a CPU-bound emulation, not CX7): median 19.5 µs at 16 B, 40 µs at 4 KiB.

## Comparison with the Python engine on the Sparks

These numbers are not apples to apples: the Sparks cross CX7 RoCE, while the pod uses one host's PCIe. The Sparks were not used.

| | Python on 2 Sparks | Zig on the pod |
| --- | --- | --- |
| NCCL all-gather in a graph, small | about 27 µs alpha (COMM-ANALYSIS) | 6.2 µs (P2P), 19–34 µs (sockets) |
| custom exchange, one row | RoCE, G14a: late rank's kernel 6.5 / 7.5 / 17.6 µs (p10 / med / p90); one-way 5.1–5.7 µs | mailbox: 5.0–5.5 µs per call |
| one 1-row window's exchanges | 0.62–0.83 ms critical path (84 exchanges) | 0.47 ms (86, mailbox) |
| fail-fast exit after a failure | about 1 s grace (G16) | 1.13–1.26 s |

## Runbook: the two-Spark gate (not yet run)

The kit is `~/Documents/Projects/spark-work/zig-tp-spark-kit/`:
- `tf-tp-test`: aarch64, glibc 2.35, with the mailbox/RoCE fatbin for sm_121;
- `run_hosts.sh`;
- `SHA256SUMS`.

It is built from this branch with `zig build tf-tp-test -Dtarget=aarch64-linux-gnu.2.35 -Dtp-fatbin=…`. It needs only the driver, `libnccl.so.2` and `libibverbs.so.1` on the hosts, with no Python and no container. If the Sparks' NCCL is not on the loader path, set `TF_NCCL_LIB=/path/to/libnccl.so.2`.

1. Copy the kit to the same path on both Sparks. Stop anything using the GPUs or the CX7 ports first: the gate needs both GPUs idle and ports 29710–29840 free.
2. Start both ranks within about a minute of each other (the bootstrap waits up to 300 s):
   - spark-1 (head, 192.168.100.10): `./run_hosts.sh 0 192.168.100.10 ./tf-tp-test out`
   - spark-2: `./run_hosts.sh 1 192.168.100.10 ./tf-tp-test out`
   - Optional: `TF_TP_ROCE_HCA=rocep1s0f1,roceP2p1s0f1` (else auto), `TF_TP_ROCE_CPU=<an X925 core>`, `NCCL_IB_TC` as prod sets it. `NCCL_SOCKET_IFNAME` defaults to `enp1s0f1np1`.
3. Expected, in order, on both hosts:
   - `roce-host`: `PASS roce-host: 1000 gathers ... bit-exact over verbs`, `rc 0`, and RESULT lines (rank 0) with the host-to-host time by size. For comparison, Python's G14 one-way is 5.1–5.7 µs.
   - `suite-roce`: the INFO line says `backend roce`, not `nccl`; a fallback is a failure here, since `TF_TP_ROCE_FALLBACK=error`. Then `SUMMARY rank R: 378 passed, 0 failed`, `rc 0`, and RESULT lines for NCCL and RoCE from 4 KiB to 4 MiB, plus a decode window of 86 exchanges.
   - `suite-nccl`: `189 passed, 0 failed`, `rc 0`.
   - `ff-*`: each victim's log ends `rc 70` (fatal) or `rc 137` (kill). The survivor's ends `rc 70`, its EXIT time about 1.2 s after the victim's INJECT time (the hosts' clocks are NTP-synced).
4. Bring back `out/` from both hosts. The numbers to report: RoCE vs NCCL graph latency at 4–16 KiB, and the decode window per exchange. The Python references are about 27 µs for NCCL in a graph and 6.5 / 7.5 / 17.6 µs (p10 / med / p90) for the late rank's RoCE kernel.

If `suite-roce` fails at setup:
- `ibv_reg_mr` failing means the memlock limit (`ulimit -l unlimited`).
- No HCA means `ibv_devices` is empty or the ports are down.
- A probe mismatch is a real bug.

The log says which rank and which step.

## Build and run

```
zig build test-tp                                # host tests: bootstrap, fate over TCP, in-process ranks
zig build tf-tp-test -Dtarget=x86_64-linux-gnu.2.35 -Dtp-nvcc=nvcc        # or -Dtp-fatbin=/abs/tp_mailbox.fatbin
zig/src/fabric/tp/run_pair.sh zig-out/bin/tf-tp-test OUT                  # on a two-GPU host
```
