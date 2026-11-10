// GPU half of the two-Spark RoCE all-gather: b12x's RoCEnante protocol (b12x contributors, Apache-2.0), new CUDA code.
#include <stdint.h>

namespace tf_fn_roce {

constexpr int SLOTS = 2, PORTS_MAX = 2, FLAG_WORDS = 32;  // 128-byte flags

struct State {
  unsigned epoch, stage, tail, pad;
};

__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys(unsigned* p, unsigned v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint4 ld_sys(const uint4* p) {
  uint4 r;
  // no "memory" clobber: one copy loop's loads may overlap (the flag's acquire and block barrier order them)
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}
__device__ __forceinline__ unsigned ld_sys(const unsigned* p) {
  unsigned r;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// T: uint4 (16-byte units, 16-aligned) or unsigned (4-byte); `bytes` the shard rounded up to 16; regions mapped host.
__device__ __forceinline__ void prefetch_lines(const char* pf, unsigned long long bytes, unsigned long long first,
                                               unsigned long long step) {
  for (unsigned long long off = first * 128; off < bytes; off += step * 128)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(pf + off));
}

__global__ void __launch_bounds__(512) prefetch(const char* pf, unsigned long long pf_bytes) {
  prefetch_lines(pf, pf_bytes, (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x,
                 (unsigned long long)gridDim.x * blockDim.x);
}

template <typename T>
__global__ void __launch_bounds__(512) gather(const T* in, T* out, int n, unsigned bytes,
                                              T* send, const T* recv, const unsigned* flags, unsigned* ctrl,
                                              unsigned long long slot_bytes, State* st, int rank, int ports,
                                              unsigned long long timeout_ns, int gblocks, const char* pf,
                                              unsigned long long pf_bytes) {
  if ((int)blockIdx.x >= gblocks) {
    prefetch_lines(pf, pf_bytes, (unsigned long long)(blockIdx.x - gblocks) * blockDim.x + threadIdx.x,
                   (unsigned long long)(gridDim.x - gblocks) * blockDim.x);
    return;
  }
  const int peer = 1 - rank;
  const unsigned seq = *reinterpret_cast<volatile unsigned*>(&st->epoch) + 1u;
  const unsigned slot = seq % SLOTS;
  const size_t slot_units = slot_bytes / sizeof(T);
  const int index = blockIdx.x * blockDim.x + threadIdx.x, stride = gblocks * blockDim.x;
  // 1. this rank's shard into its send slot
  T* mine = send + slot * slot_units;
  for (int i = index; i < n; i += stride) mine[i] = in[i];
  __syncthreads();
  // 2. the last block to finish staging rings the doorbell (a system fence per block orders its stores first)
  if (threadIdx.x == 0) {
    __threadfence_system();
    if (atomicAdd(&st->stage, 1u) == (unsigned)gblocks - 1) {
      atomicExch(&st->stage, 0u);
      st_relaxed_sys(ctrl + 4 + slot, bytes);
      __threadfence_system();
      st_relaxed_sys(ctrl, seq);
    }
  }
  // 3. the peer's shard: its flag on every port (each written after that port's stripe on the same queue pair)
  if ((int)threadIdx.x < ports) {
    const unsigned* f = flags + ((peer * SLOTS + slot) * PORTS_MAX + threadIdx.x) * FLAG_WORDS;
    const unsigned long long t0 = globaltimer();
    while (ld_acquire_sys(f) != seq) {
      if (globaltimer() - t0 > timeout_ns) {
        // peer gone or its proxy failed: record which, then trap rather than read a stale slot (context fatal)
        st_relaxed_sys(ctrl + 3, (unsigned)peer);
        st_relaxed_sys(ctrl + 2, seq);
        __threadfence_system();
        __trap();
      }
    }
  }
  __syncthreads();
  // 4. both shards into place, rank order
  const T* theirs = recv + (size_t)(peer * SLOTS + slot) * slot_units;
  T* out_mine = out + (size_t)rank * n;
  T* out_theirs = out + (size_t)peer * n;
  // four loads from the receive slot in flight a thread, then their stores
  for (int i0 = index; i0 < n; i0 += 4 * stride) {
    T v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u)
      if (i0 + u * stride < n) v[u] = ld_sys(theirs + i0 + u * stride);
#pragma unroll
    for (int u = 0; u < 4; ++u)
      if (i0 + u * stride < n) {
        out_theirs[i0 + u * stride] = v[u];
        out_mine[i0 + u * stride] = in[i0 + u * stride];
      }
  }
  // 5. the last block to finish publishes the sequence (every block read the old epoch before it got here)
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(&st->tail, 1u) == (unsigned)gblocks - 1) {
    atomicExch(&st->tail, 0u);
    __threadfence();
    *reinterpret_cast<volatile unsigned*>(&st->epoch) = seq;
  }
}

}  // namespace tf_fn_roce

template __global__ void tf_fn_roce::gather<uint4>(const uint4*, uint4*, int, unsigned, uint4*, const uint4*,
                                                   const unsigned*, unsigned*, unsigned long long,
                                                   tf_fn_roce::State*, int, int, unsigned long long, int,
                                                   const char*, unsigned long long);
template __global__ void tf_fn_roce::gather<unsigned>(const unsigned*, unsigned*, int, unsigned, unsigned*,
                                                      const unsigned*, const unsigned*, unsigned*,
                                                      unsigned long long, tf_fn_roce::State*, int, int,
                                                      unsigned long long, int, const char*, unsigned long long);
