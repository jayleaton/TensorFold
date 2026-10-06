// One-shot two-rank exchange through pinned host memory both GPUs map (the TP mailbox; mailbox.zig is its host side).
//
// The protocol is the Python engine's RoCE all-gather (glm5_next/spark/roce.cu, adapted from b12x's RoCEnante,
// Apache-2.0): a device-resident epoch so a replayed CUDA graph continues the sequence, seq = epoch + 1, two slots
// by seq parity, a flag word carrying seq after the payload, system-scope acquire loads, a %globaltimer timeout that
// poisons the transport (fail-stop), and the epoch published by the last block only when no wait failed. What
// differs: the ranks share a host, so each block writes its share of the shard straight into the peer's inbox and
// raises its own flag (no doorbell, no proxy thread, no NIC); the receiving block of the same index waits for that
// flag alone. A reduction adds the two shards in the copy-out (fp32 adds; bf16 through fp32, rounded once), the same
// expression on both ranks, so both hold the same bits.
//
// Shared region (mailbox.zig): ctrl words, then per rank an inbox: flags [2 slots][kMaxBlocks] at a 128-byte stride,
// then data [2 slots][slot_bytes].

#include <cuda_bf16.h>
#include <stdint.h>

namespace tftp {

constexpr int kMaxBlocks = 16;
constexpr int kFlagStride = 32;  // words: one 128-byte line per flag
// ctrl words: [0] abort (any rank, any host thread); per rank r at 16 + 8 r: failed, seq, seen, block
constexpr int kCtrlAbort = 0;
constexpr int kCtrlRank = 16;

enum Mode : uint32_t { kGather = 0, kSumF32 = 1, kSumBf16 = 2, kExchange = 3 };

struct MailArgs {
    const uint8_t *in;          // this rank's shard (device)
    uint8_t *out;               // gather [2 n], exchange / sums [n]
    uint8_t *peer_slots;        // the peer's inbox data, as this GPU maps it
    uint32_t *peer_flags;       // the peer's inbox flags
    const uint8_t *my_slots;    // this rank's inbox data
    const uint32_t *my_flags;   // this rank's inbox flags
    uint32_t *state;            // device: [0] epoch, [1] tail counter, [2] poison
    uint32_t *ctrl;             // shared control words
    uint64_t nbytes;            // bytes a rank
    uint64_t slot_bytes;
    uint64_t timeout_ns;
    uint32_t rank;
    uint32_t mode;
};

__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint4 ld_relaxed_sys_v4(const void *p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t ld_relaxed_sys_u8(const uint8_t *p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u8 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_relaxed_sys(uint32_t *p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_release_sys(uint32_t *p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_release_gpu(uint32_t *p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t atom_add_acq_rel_gpu(uint32_t *p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ uint32_t atom_cas_gpu(uint32_t *p, uint32_t cmp, uint32_t v) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.cas.b32 %0, [%1], %2, %3;" : "=r"(old) : "l"(p), "r"(cmp), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// dst[i] <- src[i] for i in [0, n), grid-stride; 16-byte vectors when both pointers and n allow. SysSrc: the
// source is the shared host inbox, read with system-scope loads.
template <bool SysSrc>
__device__ __forceinline__ void copy_bytes(uint8_t *dst, const uint8_t *src, uint64_t n, uint64_t index,
                                           uint64_t stride) {
    const uint64_t a = reinterpret_cast<uint64_t>(dst) | reinterpret_cast<uint64_t>(src) | n;
    if ((a & 15u) == 0u) {
        const uint64_t packs = n >> 4;
        for (uint64_t i = index; i < packs; i += stride) {
            const uint4 v = SysSrc ? ld_relaxed_sys_v4(src + (i << 4)) : reinterpret_cast<const uint4 *>(src)[i];
            reinterpret_cast<uint4 *>(dst)[i] = v;
        }
    } else {
        for (uint64_t i = index; i < n; i += stride) {
            dst[i] = static_cast<uint8_t>(SysSrc ? ld_relaxed_sys_u8(src + i) : src[i]);
        }
    }
}

__device__ __forceinline__ uint32_t add_bf16x2(uint32_t x, uint32_t y) {
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&y));
    const __nv_bfloat162 r = __floats2bfloat162_rn(a.x + b.x, a.y + b.y);
    return *reinterpret_cast<const uint32_t *>(&r);
}

__device__ __forceinline__ uint32_t add_f32(uint32_t x, uint32_t y) {
    return __float_as_uint(__uint_as_float(x) + __uint_as_float(y));
}

__device__ __forceinline__ uint32_t ld_relaxed_sys_u16(const uint8_t *p) {
    uint16_t v;
    asm volatile("ld.relaxed.sys.global.u16 %0, [%1];" : "=h"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint16_t add_bf16(uint16_t x, uint16_t y) {
    const __nv_bfloat16 r = __float2bfloat16_rn(__bfloat162float(__ushort_as_bfloat16(x)) +
                                                __bfloat162float(__ushort_as_bfloat16(y)));
    return __bfloat16_as_ushort(r);
}

// out <- mine + peer's, rank 0's first; IEEE addition commutes, so both ranks get the same bits. 16-byte vectors
// when the pointers and n allow, else one element at a time (the routing never depends on alignment, so both ranks
// always take the mailbox together).
template <bool Bf16>
__device__ __forceinline__ void sum_bytes(uint8_t *out, const uint8_t *mine, const uint8_t *peer, uint64_t n,
                                          uint32_t rank, uint64_t index, uint64_t stride) {
    const uint64_t al = reinterpret_cast<uint64_t>(out) | reinterpret_cast<uint64_t>(mine) |
                        reinterpret_cast<uint64_t>(peer) | n;
    if ((al & 15u) == 0u) {
        const uint64_t packs = n >> 4;
        for (uint64_t i = index; i < packs; i += stride) {
            const uint4 m = reinterpret_cast<const uint4 *>(mine)[i];
            const uint4 p = ld_relaxed_sys_v4(peer + (i << 4));
            const uint4 a = rank == 0 ? m : p;
            const uint4 b = rank == 0 ? p : m;
            uint4 r;
            r.x = Bf16 ? add_bf16x2(a.x, b.x) : add_f32(a.x, b.x);
            r.y = Bf16 ? add_bf16x2(a.y, b.y) : add_f32(a.y, b.y);
            r.z = Bf16 ? add_bf16x2(a.z, b.z) : add_f32(a.z, b.z);
            r.w = Bf16 ? add_bf16x2(a.w, b.w) : add_f32(a.w, b.w);
            reinterpret_cast<uint4 *>(out)[i] = r;
        }
    } else if (Bf16) {
        for (uint64_t i = index; i < (n >> 1); i += stride) {
            const uint16_t m = reinterpret_cast<const uint16_t *>(mine)[i];
            const uint16_t p = static_cast<uint16_t>(ld_relaxed_sys_u16(peer + (i << 1)));
            reinterpret_cast<uint16_t *>(out)[i] = rank == 0 ? add_bf16(m, p) : add_bf16(p, m);
        }
    } else {
        for (uint64_t i = index; i < (n >> 2); i += stride) {
            const uint32_t m = reinterpret_cast<const uint32_t *>(mine)[i];
            const uint32_t p = ld_relaxed_sys(reinterpret_cast<const uint32_t *>(peer) + i);
            reinterpret_cast<uint32_t *>(out)[i] = rank == 0 ? add_f32(m, p) : add_f32(p, m);
        }
    }
}

// Thread 0 of a block: spin until the peer block's flag holds seq; false on abort or timeout.
__device__ __noinline__ bool wait_flag(const uint32_t *flag, uint32_t seq, const uint32_t *ctrl, uint64_t timeout_ns,
                                       uint32_t *seen) {
    const uint64_t t0 = globaltimer();
    uint32_t polls = 0;
    while (true) {
        const uint32_t v = ld_acquire_sys(flag);
        if (v == seq) {
            *seen = v;
            return true;
        }
        if ((++polls & 63u) == 0u) {
            if (ld_relaxed_sys(ctrl + kCtrlAbort) != 0u || globaltimer() - t0 > timeout_ns) {
                *seen = ld_acquire_sys(flag);
                return *seen == seq;
            }
        }
    }
}

}  // namespace tftp

using namespace tftp;

extern "C" __global__ void __launch_bounds__(512) tp_mailbox(MailArgs a) {
    __shared__ uint32_t s_head[2];  // poison, seq
    __shared__ uint32_t s_bad;
    const uint32_t tid = threadIdx.x;
    const uint32_t gdim = gridDim.x;
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + tid;
    const uint64_t stride = static_cast<uint64_t>(gdim) * blockDim.x;

    // a poisoned transport does nothing (the host raises on its next check); poison and epoch in one round trip
    if (tid == 0) {
        s_head[0] = ld_relaxed_gpu(a.state + 2);
        s_head[1] = ld_relaxed_gpu(a.state) + 1u;
        s_bad = 0u;
    }
    __syncthreads();
    if (s_head[0] != 0u) {
        return;
    }
    const uint32_t seq = s_head[1];
    const uint64_t slot = seq & 1u;
    const uint64_t flag_at = (slot * kMaxBlocks + blockIdx.x) * kFlagStride;

    // 1. this block's share of the shard into the peer's inbox, then its flag after it (system-scope release)
    copy_bytes<false>(a.peer_slots + slot * a.slot_bytes, a.in, a.nbytes, index, stride);
    __syncthreads();
    if (tid == 0) {
        fence_sc_sys();
        st_release_sys(a.peer_flags + flag_at, seq);
    }
    // the local shard while the peer's writes travel
    if (a.mode == kGather) {
        copy_bytes<false>(a.out + static_cast<uint64_t>(a.rank) * a.nbytes, a.in, a.nbytes, index, stride);
    }

    // 2. the peer's block of the same index (it wrote exactly the bytes this block reads)
    if (tid == 0) {
        uint32_t seen = 0u;
        if (!wait_flag(a.my_flags + flag_at, seq, a.ctrl, a.timeout_ns, &seen)) {
            s_bad = 1u;
            if (atom_cas_gpu(a.state + 2, 0u, 1u) == 0u) {  // one failure record a rank
                uint32_t *rec = a.ctrl + kCtrlRank + 8u * a.rank;
                st_relaxed_sys(rec + 1, seq);
                st_relaxed_sys(rec + 2, seen);
                st_relaxed_sys(rec + 3, blockIdx.x);
                fence_sc_sys();
                st_relaxed_sys(rec, 1u);
                fence_sc_sys();
            }
        }
    }
    __syncthreads();

    // 3. copy out, gated by this block's own wait
    if (s_bad == 0u) {
        const uint8_t *inbox = a.my_slots + slot * a.slot_bytes;
        switch (a.mode) {
            case kGather:
                copy_bytes<true>(a.out + static_cast<uint64_t>(1u - a.rank) * a.nbytes, inbox, a.nbytes, index,
                                 stride);
                break;
            case kExchange:
                copy_bytes<true>(a.out, inbox, a.nbytes, index, stride);
                break;
            case kSumF32:
                sum_bytes<false>(a.out, a.in, inbox, a.nbytes, a.rank, index, stride);
                break;
            default:
                sum_bytes<true>(a.out, a.in, inbox, a.nbytes, a.rank, index, stride);
                break;
        }
    }

    // 4. the last block publishes the epoch, unless a wait of this launch failed (later launches stay no-ops)
    __syncthreads();
    if (tid == 0) {
        // the counter is back at 0 for the next launch (stream order), whatever its grid
        const bool last = gdim == 1u || atom_add_acq_rel_gpu(a.state + 1, 1u) == gdim - 1u;
        const bool ok = gdim == 1u ? s_bad == 0u : ld_relaxed_gpu(a.state + 2) == 0u;
        if (last && gdim > 1u) {
            st_release_gpu(a.state + 1, 0u);
        }
        if (last && ok) {
            st_release_gpu(a.state, seq);
        }
    }
}
